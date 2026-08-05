/*
 * context.c — see context.h.
 */
#include "context.h"
#include "tensor.h"

#include <string.h>
#include <time.h>

/* Big enough for the largest program any emitter produces, with room. A
 * program that outgrew this would be silently truncated, so the launch checks
 * rather than trusting. */
/* Room for every program that can be outstanding at once: a queued launch's
 * code must survive until the GPU has run it, so one shared code buffer would
 * be overwritten by the next enqueue. */
#define PROGRAM_BYTES 8192
#define CODE_BYTES (PROGRAM_BYTES * HELIOS_RING_SLOTS)
#define LAUNCH_TIMEOUT_NS 5000000000ull
/* Dwords a queued launch may occupy, generously. Used only to decide when to
 * drain before the pushbuffer wraps. */
#define LAUNCH_DWORD_BUDGET 512

#define FAIL(stage)                                                            \
  do {                                                                         \
    ctx->failStage = (stage);                                                  \
    helios_context_close(ctx);                                                 \
    ctx->failStage = (stage);                                                  \
    return -1;                                                                 \
  } while (0)

/* Allocate, map to the GPU, and map to the host. Every buffer here is read or
 * written by both sides. */
static int alloc_shared(helios_context *ctx, gaia_buffer *b, NvU64 size) {
  if (gaia_alloc(&ctx->device, b, size, GAIA_SYSMEM) != 0) return -1;
  if (gaia_map_gpu(&ctx->device, b) != 0) return -1;
  if (gaia_map_host(&ctx->device, b) != 0) return -1;
  return 0;
}

/*
 * Bind the compute class and set up the SM's address windows.
 *
 * ONCE, before any launch, and in its own submission so it is consumed first.
 * This was missing from the first version of the context, and the symptom is
 * worth recording: every kernel that only reads and writes global memory ran
 * correctly, and the first one to touch SHARED memory faulted with
 * GR_EXCEPTION. Without this the SM has no shared-memory window, so STS and
 * BAR.SYNC address nothing -- and a stack whose simple kernels all pass looks
 * like a stack that works.
 */
static int init_engine(helios_context *ctx) {
  hermes_compute_config cfg;
  memset(&cfg, 0, sizeof cfg);
  cfg.classId = HERMES_COMPUTE_CLASS;
  cfg.spaVersion = HERMES_SPA_VERSION_SM86;
  cfg.sharedWindow = HERMES_SHARED_WINDOW_DEFAULT;
  cfg.localWindow = HERMES_LOCAL_WINDOW_DEFAULT;
  cfg.localMem = ctx->lmem.gpuAddr;
  cfg.localMemSize = ctx->lmem.size;
  cfg.smCount = HERMES_SM_COUNT_SM86;

  hermes_begin(&ctx->channel);
  hermes_compute_init(&ctx->channel, 1, &cfg);
  if (hermes_submit(&ctx->device, &ctx->channel) != 0) return -1;
  hermes_ring(&ctx->channel, (volatile NvU32 *)ctx->channel.userd.hostPtr,
              ctx->channel.doorbell, ctx->channel.token);
  return 0;
}

int helios_context_open(helios_context *ctx, int index) {
  memset(ctx, 0, sizeof *ctx);

  if (aether_device_open(&ctx->device, index) != 0) {
    ctx->failStage = ctx->device.failStage ? ctx->device.failStage : "no device";
    return -1;
  }
  ctx->open = 1;

  if (hermes_channel_open(&ctx->device, &ctx->channel) != 0)
    FAIL(ctx->channel.failStage ? ctx->channel.failStage : "channel");

  if (alloc_shared(ctx, &ctx->scratch,
                   (NvU64)HERMES_CBUF0_BYTES * HELIOS_RING_SLOTS) != 0)
    FAIL("scratch");
  if (alloc_shared(ctx, &ctx->code, CODE_BYTES) != 0) FAIL("code");
  if (alloc_shared(ctx, &ctx->qmd, (NvU64)HERMES_QMD_BYTES * HELIOS_RING_SLOTS) != 0)
    FAIL("qmd");
  if (alloc_shared(ctx, &ctx->fence, 4096) != 0) FAIL("fence");

  /* Local memory lives in video memory and is never touched by the host. */
  if (gaia_alloc(&ctx->device, &ctx->lmem, 1024 * 1024, GAIA_VIDMEM) != 0)
    FAIL("lmem");
  if (gaia_map_gpu(&ctx->device, &ctx->lmem) != 0) FAIL("lmem map");

  if (init_engine(ctx) != 0) FAIL("compute init");

  memset(ctx->scratch.hostPtr, 0, (size_t)HERMES_CBUF0_BYTES * HELIOS_RING_SLOTS);
  memset(ctx->fence.hostPtr, 0, 4096);
  return 0;
}

void helios_context_close(helios_context *ctx) {
  if (!ctx->open) return;
  gaia_free(&ctx->device, &ctx->lmem);
  gaia_free(&ctx->device, &ctx->fence);
  gaia_free(&ctx->device, &ctx->qmd);
  gaia_free(&ctx->device, &ctx->code);
  gaia_free(&ctx->device, &ctx->scratch);
  hermes_channel_close(&ctx->device, &ctx->channel);
  aether_device_close(&ctx->device);
  memset(ctx, 0, sizeof *ctx);
}

static NvU64 now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (NvU64)ts.tv_sec * 1000000000ull + (NvU64)ts.tv_nsec;
}


int helios_enqueue(helios_context *ctx, const hp_word *program, unsigned count,
                   NvU32 gridX, NvU32 gridY, NvU32 blockX, NvU32 sharedBytes,
                   const NvU64 *buffers, unsigned nbuffers,
                   const NvU32 *scalars, unsigned nscalars) {
  if (count * sizeof(hp_word) > PROGRAM_BYTES) return -1;
  /*
   * A BLOCK CANNOT HAVE MORE THAN 1024 THREADS, and asking for more is not a
   * clean failure — it is an invalid launch, which raises GR_EXCEPTION on the
   * channel and is reported at whatever flushes next.
   *
   * Most kernels here take their block width from a model dimension: one thread
   * per output column, per class, per feature. That is fine up to the limit and
   * silently catastrophic past it, and a 105M-parameter model goes past it in
   * several places at once (vocab 12,288, qkv 1,920, FFN 1,728). matmul now
   * walks its columns in chunks; the rest do not yet, and until they do this is
   * what tells you which one you hit instead of "layerNorm failed" three
   * operations later.
   */
  if (blockX > HELIOS_MAX_BLOCK_THREADS) {
    ctx->failStage = "launch: block width over 1024 threads";
    return -1;
  }
  if (nbuffers > HERMES_CBUF0_PARAM_COUNT) return -1;
  if (nscalars > HERMES_CBUF0_SCALAR_COUNT) return -1;

  /* A full ring must drain before its oldest slot can be reused. */
  if (ctx->pending >= HELIOS_RING_SLOTS && helios_flush(ctx) != 0) return -1;

  /*
   * And so must a pushbuffer that is about to WRAP.
   *
   * hermes_begin resets pushOffset to zero when a segment would not fit at the
   * end, which is safe when the GPU has consumed everything -- and it has, in
   * the synchronous design, because the host just waited. Batching breaks that:
   * a wrap in the middle of a batch overwrites segments that are queued and not
   * yet fetched, and the channel stops making progress with no error. The
   * two-launch test never saw it because two launches never reach the end.
   *
   * So drain before the wrap rather than after it. The budget is generous: a
   * launch is a barrier plus a PCAS pair, well under this.
   */
  const NvU32 capacityDwords = (NvU32)(ctx->channel.pushbuffer.size / 4);
  if (ctx->pending > 0 &&
      ctx->channel.pushOffset + LAUNCH_DWORD_BUDGET > capacityDwords &&
      helios_flush(ctx) != 0)
    return -1;

  const unsigned slot = ctx->ringNext;
  volatile NvU8 *cb =
      (volatile NvU8 *)ctx->scratch.hostPtr + (size_t)slot * HERMES_CBUF0_BYTES;
  const NvU64 cbAddr = ctx->scratch.gpuAddr + (NvU64)slot * HERMES_CBUF0_BYTES;

  *(volatile NvU32 *)(cb + HERMES_CBUF0_NTID_X) = blockX;
  for (unsigned i = 0; i < nbuffers; i++)
    *(volatile NvU64 *)(cb + HERMES_CBUF0_PARAM_N(i)) = buffers[i];
  for (unsigned i = 0; i < nscalars; i++)
    *(volatile NvU32 *)(cb + HERMES_CBUF0_SCALAR_N(i)) = scalars[i];

  const NvU64 codeAddr = ctx->code.gpuAddr + (NvU64)slot * PROGRAM_BYTES;
  hp_word *slotCode =
      (hp_word *)((NvU8 *)ctx->code.hostPtr + (size_t)slot * PROGRAM_BYTES);
  const hp_word pad = hp_exit(hp_ctrl_safe());
  for (unsigned i = count; i < PROGRAM_BYTES / sizeof(hp_word); i++)
    slotCode[i] = pad;
  memcpy(slotCode, program, count * sizeof(hp_word));

  NvU32 qmd[HERMES_QMD_DWORDS];
  hermes_qmd_build(qmd, codeAddr, cbAddr, gridX, gridY ? gridY : 1, 1, blockX, 1, 1,
                   sharedBytes,
                   count * (NvU32)sizeof(hp_word));
  memcpy((NvU8 *)ctx->qmd.hostPtr + (size_t)slot * HERMES_QMD_BYTES, qmd,
         HERMES_QMD_BYTES);
  __asm__ __volatile__("sfence" ::: "memory");

  /*
   * SUBMIT here, ring and wait in flush.
   *
   * The pushbuffer is segmented and hermes_submit pushes one GPFIFO entry for
   * the segment just written -- so queuing many launches and submitting once
   * submits only the LAST of them, and every earlier kernel silently never
   * runs. The expensive parts are the doorbell and the fence wait, and those
   * are what batching removes; writing a GPFIFO entry per launch costs a few
   * stores.
   */
  hermes_begin(&ctx->channel);
  /* Drain before this launch if anything is already queued: the kernels in a
   * batch routinely consume each other's output, and dispatches pipeline. */
  if (ctx->pending > 0) hermes_barrier(&ctx->channel);
  hermes_launch(&ctx->channel, ctx->qmd.gpuAddr + (NvU64)slot * HERMES_QMD_BYTES);
  if (hermes_submit(&ctx->device, &ctx->channel) != 0) return -1;
  ctx->ringNext = (slot + 1) % HELIOS_RING_SLOTS;
  ctx->pending++;
  ctx->statEnqueued++;
  return 0;
}

/*
 * A FLUSH NO LONGER RETIRES. The step boundary does.
 *
 * It used to: a flush drained the queue, so nothing could still be reading a
 * freed buffer and recycling was safe. Safe against the GPU, and not against
 * the graph -- the tape releases tensors it turns out to still reference, and
 * recycling on the next flush handed that memory straight to somebody else.
 *
 * Retiring at the step boundary instead keeps a released buffer valid and
 * untouched for the rest of the step, which is what makes the tape's release
 * callback usable at all. helios_end_step is where it happens, and a harness
 * that never calls it simply never recycles -- which is exactly the behaviour
 * that shipped before, so nothing regresses by not adopting it.
 */
int helios_flush(helios_context *ctx) {
  if (ctx->pending == 0) return 0;
  ctx->statFlushed++;

  /* ONE semaphore for the whole batch: the channel runs its pushbuffer in
   * order, so the last kernel retiring means all of them have. */
  const NvU32 want = ++ctx->fenceValue;
  volatile NvU32 *fence = (volatile NvU32 *)ctx->fence.hostPtr;
  *fence = 0;
  __asm__ __volatile__("sfence" ::: "memory");

  hermes_begin(&ctx->channel);
  hermes_semaphore_release(&ctx->channel, ctx->fence.gpuAddr, want);
  if (hermes_submit(&ctx->device, &ctx->channel) != 0) return -1;
  hermes_ring(&ctx->channel, (volatile NvU32 *)ctx->channel.userd.hostPtr,
              ctx->channel.doorbell, ctx->channel.token);

  const NvU64 spinStart = now_ns();
  const NvU64 deadline = spinStart + LAUNCH_TIMEOUT_NS;
  while (*fence != want) {
    if (now_ns() > deadline) {
      ctx->lastError = ((volatile NvU32 *)ctx->channel.errnotif.hostPtr)[2];
      ctx->pending = 0;
      ctx->statSpinNs += now_ns() - spinStart;
      return -1;
    }
  }
  ctx->statSpinNs += now_ns() - spinStart;
  ctx->lastError = 0;
  ctx->pending = 0;
  return 0;
}

/*
 * The end of a step: drain, then recycle everything released during it.
 *
 * The drain must come FIRST and it is not a formality -- a kernel that was
 * enqueued and has not run may still be reading a buffer that was released, and
 * handing that memory to the next allocation lets the host overwrite the
 * kernel's input. That produces a finite, plausible, wrong number from an
 * operation that looks unrelated.
 */
int helios_end_step(helios_context *ctx) {
  if (helios_flush(ctx) != 0) return -1;
  helios_tensor_retire();
  return 0;
}

/* The synchronous form, kept for callers that want one launch and its result:
 * queue it and drain immediately. */
int helios_launch(helios_context *ctx, const hp_word *program, unsigned count,
                  NvU32 gridX, NvU32 gridY, NvU32 blockX, NvU32 sharedBytes,
                  const NvU64 *buffers, unsigned nbuffers, const NvU32 *scalars,
                  unsigned nscalars) {
  if (helios_enqueue(ctx, program, count, gridX, gridY, blockX, sharedBytes,
                     buffers, nbuffers, scalars, nscalars) != 0)
    return -1;
  return helios_flush(ctx);
}
