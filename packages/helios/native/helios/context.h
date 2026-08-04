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
  gaia_buffer scratch; /* constant bank 0 */
  gaia_buffer fence;   /* the semaphore the host polls */
  gaia_buffer code;    /* the program currently loaded */
  gaia_buffer qmd;     /* the launch descriptor */
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
  int open;
  const char *failStage;
} helios_context;

/* Open the device at `index` and bring up everything above. Returns 0, or -1
 * with failStage naming the step. */
int helios_context_open(helios_context *ctx, int index);

void helios_context_close(helios_context *ctx);

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
                  NvU32 gridX, NvU32 blockX, NvU32 sharedBytes,
                  const NvU64 *buffers, unsigned nbuffers, const NvU32 *scalars,
                  unsigned nscalars);

#endif /* HELIOS_CONTEXT_H */
