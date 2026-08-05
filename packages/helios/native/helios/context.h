/*
 * context.h — the one long-lived thing the whole backend hangs off.
 *
 * WHAT: an open device, one channel, the constant-bank scratch every launch
 * reads its parameters from, and the fence the host waits on. Created once and
 * kept for the process's life.
 *
 * WHY ONE OF EVERYTHING: bringing up a channel costs several dozen ioctls and
 * a page of RM object allocations. The test harness did it per suite, which was
 * fine at forty kernels a run; a training step issues thousands of launches and
 * cannot pay that each time. One channel also means launches are ORDERED by
 * construction -- the GPU executes a channel's pushbuffer in sequence -- so a
 * kernel that consumes another's output needs no explicit dependency, which is
 * the property the whole dispatcher is built on.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: no multi-channel, no streams, no
 * concurrency between kernels. Every launch runs after the one before it. That
 * leaves overlap on the table and it makes the execution model something that
 * can be reasoned about in one sentence, which is worth more while correctness
 * is still being established.
 */
#ifndef HELIOS_CONTEXT_H
#define HELIOS_CONTEXT_H

#include "../gaia/memory.h"
#include "../hermes/channel.h"
#include "../hermes/pushbuffer.h"
#include "../hermes/qmd.h"
#include "../hephaestus/sm86.h"

/* How many launches may be outstanding before a flush is forced. */
#define HELIOS_RING_SLOTS 64

/* Distinct buffers tracked between barriers. Beyond this the barrier is taken
 * unconditionally, which is the safe direction. */
#define HELIOS_TOUCH_SLOTS 48

/*
 * The scratch buffer holds constant bank 0: the block dimension, the buffer
 * pointers, and the scalars. Every launch rewrites it, so it is per-context
 * rather than per-launch -- and because launches are ordered and the host
 * writes it before ringing the doorbell, one buffer is enough. It would not be
 * if two launches could be in flight.
 */
typedef struct {
  aether_device device;
  hermes_channel channel;
  /*
   * A RING of constant banks and QMDs, not one of each.
   *
   * One bank means one launch can be in flight: the host must wait for the GPU
   * to read it before writing the next launch's parameters. That made a
   * 150-operation step into 150 round trips, and measured 88 tokens/second
   * against Vulkan's 601 -- the gap was almost entirely this, not the kernels.
   *
   * With a ring, N launches queue into one submission and the host waits once.
   * The ring size bounds how many can be outstanding; when it fills, the next
   * enqueue flushes first, which is correctness rather than policy.
   */
  gaia_buffer scratch; /* HELIOS_RING_SLOTS constant banks, back to back */
  gaia_buffer fence;   /* the semaphore the host polls */
  gaia_buffer code;    /* the program currently loaded */
  gaia_buffer qmd;     /* HELIOS_RING_SLOTS launch descriptors */
  unsigned ringNext;   /* the slot the next enqueue will use */
  unsigned pending;    /* launches queued and not yet waited on */
  /*
   * Backing store for per-thread local memory.
   *
   * Nothing here spills today -- every kernel fits in its registers -- but the
   * engine is configured with a local window regardless, because the window is
   * part of the SM's address setup rather than an opt-in. Configuring it wrong
   * or not at all is what makes the first kernel to touch shared memory fault.
   */
  gaia_buffer lmem;
  NvU32 fenceValue;    /* incremented per launch, so waits cannot alias */
  /* Why the last launch failed: the channel's error code if the GPU faulted,
   * zero if it simply never signalled. A timeout and a fault need different
   * investigations and look identical without this. */
  NvU32 lastError;
  /* Counters, not state: the ratio of launches queued to queue drains is the
   * batching factor, and it is the only way to tell a batching design that is
   * working from one that is silently flushing per operation. */
  /*
   * THE BUFFERS TOUCHED SINCE THE LAST BARRIER, so the next launch can decide
   * whether it needs one.
   *
   * A dispatch does not wait for its predecessor — the channel executes its
   * PUSHBUFFER in order but the dispatches within it pipeline — so a kernel
   * reading what the previous one wrote needs an explicit WAIT_FOR_IDLE. That
   * barrier used to be unconditional, which made it the floor on every small
   * operation: an elementwise kernel over 1,024 elements measures 5.7 us where
   * the arithmetic and the memory together are under a microsecond, and it is
   * flat at 5.7 all the way to 65,536 elements. A step runs ~1,474 launches.
   *
   * Worse than the floor, nothing ever OVERLAPS: a kernel that could hide
   * behind a larger one never does.
   *
   * So the barrier is now conditional on a real dependency. The rule is
   * deliberately conservative — every buffer a launch names counts as both read
   * and written, so a barrier is skipped only when two launches share NO buffer
   * at all. That is sound because a kernel reaches memory ONLY through the
   * pointers in its constant bank, which are exactly these, and because a
   * tensor owns its whole size class so a launch cannot write past its own
   * buffer into another's.
   *
   * The set is small and overflow is safe: a full table forces the barrier.
   */
  NvU64 touched[HELIOS_TOUCH_SLOTS];
  unsigned touchedCount;
  NvU32 statBarriers;
  NvU32 statEnqueued;
  NvU32 statFlushed;
  /*
   * How long the host SPUN on the fence, in nanoseconds.
   *
   * The ratio above says how well launches batch. It cannot say what the batching
   * is worth, because a flush is a full drain -- submit, then spin until the GPU
   * signals -- so a step is host-enqueue plus GPU-execute with no overlap
   * whatever. This counter splits those two: spin time is GPU, and everything
   * else in the step is host.
   *
   * Without it the split has to be inferred from per-method wall times, and that
   * inference is wrong: a flush forced from inside helios_enqueue is charged to
   * whichever operation happened to be the one that filled the pushbuffer, which
   * is how `sub` came to look like it cost 2,843 us a call.
   */
  NvU64 statSpinNs;
  int open;
  const char *failStage;
} helios_context;

/* Open the device at `index` and bring up everything above. Returns 0, or -1
 * with failStage naming the step. */
int helios_context_open(helios_context *ctx, int index);

void helios_context_close(helios_context *ctx);

/* The hardware's threads-per-block limit. Exceeding it is an invalid launch,
 * not a rejected one, so helios_enqueue checks rather than letting the channel
 * fault asynchronously. */
#define HELIOS_MAX_BLOCK_THREADS 1024u

/*
 * Queue a launch. Does NOT wait.
 *
 * The channel executes its pushbuffer in order, so a kernel queued after
 * another sees its output -- no dependency tracking is needed, and that is what
 * makes batching safe here. What is NOT safe is the host reading a result
 * before flushing, which is why every path that touches device memory from the
 * host calls helios_flush first.
 */
int helios_enqueue(helios_context *ctx, const hp_word *program, unsigned count,
                   NvU32 gridX, NvU32 gridY, NvU32 blockX, NvU32 sharedBytes,
                   const NvU64 *buffers, unsigned nbuffers,
                   const NvU32 *scalars, unsigned nscalars);

/*
 * The same, with a THIRD grid dimension.
 *
 * Every kernel in this stack until the tensor-core GEMM was one- or
 * two-dimensional: blocks over rows, and a second axis for the batch plane. A
 * GEMM that tiles the OUTPUT needs both output axes, which leaves nowhere for
 * the batch — so it takes the depth the QMD has always had (CTA_RASTER_DEPTH)
 * and which nothing had asked for. helios_enqueue is this with gridZ = 1, so no
 * existing caller changes.
 */
int helios_enqueue3(helios_context *ctx, const hp_word *program, unsigned count,
                    NvU32 gridX, NvU32 gridY, NvU32 gridZ, NvU32 blockX,
                    NvU32 sharedBytes, const NvU64 *buffers, unsigned nbuffers,
                    const NvU32 *scalars, unsigned nscalars);

/* Submit everything queued and wait for it. A no-op when nothing is pending. */
int helios_flush(helios_context *ctx);

/*
 * End of a training step: drain, then return every buffer released during the
 * step to the pool.
 *
 * Reclamation is HERE and not in helios_flush because a released buffer must
 * stay valid for the rest of the step. The tape releases tensors it turns out
 * to still reference, and recycling at the next flush handed that memory to the
 * next allocation -- see the long comment on helios_tensor_free. A caller that
 * never calls this never recycles, which is the behaviour that shipped before.
 */
int helios_end_step(helios_context *ctx);

/*
 * Launch `program` over the given grid, with up to four buffers and six
 * scalars in the constant bank, and wait for it to retire.
 *
 * Synchronous on purpose: the host writes the constant bank in place, so it
 * cannot start building the next launch until this one has read it. Making
 * launches asynchronous means giving each its own bank, which is a change worth
 * making once there is a measurement saying it matters.
 */
int helios_launch(helios_context *ctx, const hp_word *program, unsigned count,
                  NvU32 gridX, NvU32 gridY, NvU32 blockX, NvU32 sharedBytes,
                  const NvU64 *buffers, unsigned nbuffers, const NvU32 *scalars,
                  unsigned nscalars);

#endif /* HELIOS_CONTEXT_H */
