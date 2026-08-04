/*
 * context.c — see context.h.
 */
#include "context.h"

#include <string.h>
#include <time.h>

/* Big enough for the largest program any emitter produces, with room. A
 * program that outgrew this would be silently truncated, so the launch checks
 * rather than trusting. */
#define CODE_BYTES 65536
#define LAUNCH_TIMEOUT_NS 5000000000ull

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

  if (alloc_shared(ctx, &ctx->scratch, HERMES_CBUF0_BYTES) != 0) FAIL("scratch");
  if (alloc_shared(ctx, &ctx->code, CODE_BYTES) != 0) FAIL("code");
  if (alloc_shared(ctx, &ctx->qmd, HERMES_QMD_BYTES) != 0) FAIL("qmd");
  if (alloc_shared(ctx, &ctx->fence, 4096) != 0) FAIL("fence");

  /* Local memory lives in video memory and is never touched by the host. */
  if (gaia_alloc(&ctx->device, &ctx->lmem, 1024 * 1024, GAIA_VIDMEM) != 0)
    FAIL("lmem");
  if (gaia_map_gpu(&ctx->device, &ctx->lmem) != 0) FAIL("lmem map");

  if (init_engine(ctx) != 0) FAIL("compute init");

  memset(ctx->scratch.hostPtr, 0, HERMES_CBUF0_BYTES);
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

/*
 * Fill the code buffer.
 *
 * The whole buffer is filled with EXIT first, then the program copied over it.
 * The padding matters: instruction fetch runs ahead of execution, so it will
 * read past a program's last instruction, and what it finds there had better be
 * a legal encoding. Leaving the tail as whatever the previous program wrote is
 * usually harmless and occasionally a fault in a kernel that looks unrelated.
 */
static void write_code(gaia_buffer *code, const hp_word *prog, unsigned count) {
  const hp_word pad = hp_exit(hp_ctrl_safe());
  hp_word *slot = (hp_word *)code->hostPtr;
  for (NvU64 i = 0; i < code->size / sizeof(hp_word); i++) slot[i] = pad;
  memcpy(code->hostPtr, prog, count * sizeof(hp_word));
}

int helios_launch(helios_context *ctx, const hp_word *program, unsigned count,
                  NvU32 gridX, NvU32 blockX, NvU32 sharedBytes,
                  const NvU64 *buffers, unsigned nbuffers, const NvU32 *scalars,
                  unsigned nscalars) {
  if (count * sizeof(hp_word) > ctx->code.size) return -1;
  if (nbuffers > HERMES_CBUF0_PARAM_COUNT) return -1;
  if (nscalars > HERMES_CBUF0_SCALAR_COUNT) return -1;

  /* Constant bank 0, in CUDA's layout: the block dimension, then the buffer
   * pointers, then the scalars. */
  volatile NvU8 *cb = (volatile NvU8 *)ctx->scratch.hostPtr;
  *(volatile NvU32 *)(cb + HERMES_CBUF0_NTID_X) = blockX;
  for (unsigned i = 0; i < nbuffers; i++)
    *(volatile NvU64 *)(cb + HERMES_CBUF0_PARAM_N(i)) = buffers[i];
  for (unsigned i = 0; i < nscalars; i++)
    *(volatile NvU32 *)(cb + HERMES_CBUF0_SCALAR_N(i)) = scalars[i];

  write_code(&ctx->code, program, count);

  NvU32 qmd[HERMES_QMD_DWORDS];
  hermes_qmd_build(qmd, ctx->code.gpuAddr, ctx->scratch.gpuAddr, gridX, 1, 1,
                   blockX, 1, 1, sharedBytes, count * (NvU32)sizeof(hp_word));
  memcpy(ctx->qmd.hostPtr, qmd, HERMES_QMD_BYTES);

  /* The stores above are to write-combined memory; the fence makes them visible
   * before the doorbell tells the GPU to go looking. */
  __asm__ __volatile__("sfence" ::: "memory");

  /*
   * A fresh fence value per launch.
   *
   * Reusing one value means a wait can be satisfied by the PREVIOUS launch's
   * release, which returns immediately and reports success while the kernel is
   * still running. The bug surfaces as a race that only appears under load,
   * which is the worst kind to find later.
   */
  const NvU32 want = ++ctx->fenceValue;
  volatile NvU32 *fence = (volatile NvU32 *)ctx->fence.hostPtr;
  *fence = 0;

  hermes_begin(&ctx->channel);
  hermes_launch(&ctx->channel, ctx->qmd.gpuAddr);
  hermes_semaphore_release(&ctx->channel, ctx->fence.gpuAddr, want);
  hermes_submit(&ctx->device, &ctx->channel);
  hermes_ring(&ctx->channel, (volatile NvU32 *)ctx->channel.userd.hostPtr,
              ctx->channel.doorbell, ctx->channel.token);

  const NvU64 deadline = now_ns() + LAUNCH_TIMEOUT_NS;
  while (*fence != want) {
    if (now_ns() > deadline) {
      ctx->lastError = ((volatile NvU32 *)ctx->channel.errnotif.hostPtr)[2];
      return -1;
    }
  }
  ctx->lastError = 0;
  return 0;
}
