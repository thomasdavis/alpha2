/*
 * helios_batch_test.c — launches that share a submission.
 *
 * Split from helios_test.c because that file passed 300 lines, and this is a
 * coherent piece of it: everything here is about what happens when more than
 * one launch is queued before the host waits.
 *
 * That turned out to be the least obvious part of the stack. A channel executes
 * its PUSHBUFFER in order, which is easy to confirm and easy to over-read: the
 * DISPATCHES it contains pipeline and do not serialise. The synchronous design
 * hid it for the whole project, because a fence wait after every launch was
 * doing the work of a barrier without anyone naming it.
 */
#include "harness.h"

#include "../helios/context.h"
#include "../helios/program.h"
#include "../prometheus/elementwise.h"

#include <string.h>

/*
 * TWO launches in one submission, one fence.
 *
 * The minimal form of the batching that hangs. Isolating it here rather than
 * through the TypeScript stack means a failure is the submission path and
 * nothing else -- no pool, no dispatcher, no tape.
 *
 * The second kernel consumes the first's output, so a pass proves ordering as
 * well as delivery: the channel runs its pushbuffer in sequence, which is the
 * property the whole batching design rests on.
 */
static void test_batched_launch(void) {
  HT_CASE("two launches in one submission, one fence");

  helios_context ctx;
  if (helios_context_open(&ctx, 0) != 0) {
    printf("skip (no NVIDIA driver)\n");
    ht_case_failed = 0;
    return;
  }
  helios_program_reset();

  gaia_buffer a, b;
  memset(&a, 0, sizeof a);
  memset(&b, 0, sizeof b);
  int ok = gaia_alloc(&ctx.device, &a, 4096, GAIA_SYSMEM) == 0 &&
           gaia_map_gpu(&ctx.device, &a) == 0 &&
           gaia_map_host(&ctx.device, &a) == 0 &&
           gaia_alloc(&ctx.device, &b, 4096, GAIA_SYSMEM) == 0 &&
           gaia_map_gpu(&ctx.device, &b) == 0 &&
           gaia_map_host(&ctx.device, &b) == 0;
  if (!ok) {
    HT_FAIL("buffers");
    helios_context_close(&ctx);
    HT_END();
    return;
  }

  volatile NvU32 *ha = (volatile NvU32 *)a.hostPtr;
  for (unsigned i = 0; i < 64; i++) {
    const float v = (float)(i + 1);
    memcpy((void *)&ha[i], &v, 4);
  }
  memset((void *)b.hostPtr, 0, 4096);

  /* copy a -> b, then double b in place. Chained on purpose. */
  const helios_key ck = {HL_ELEMENTWISE, PR_EW_COPY, 64, 0};
  const helios_key dk = {HL_ELEMENTWISE, PR_EW_ADD, 64, 0};
  const helios_program *cp = helios_program_get(ck);
  const helios_program *dp = helios_program_get(dk);
  HT_TRUE(cp != NULL && dp != NULL);

  const NvU64 c1[3] = {b.gpuAddr, a.gpuAddr, a.gpuAddr};
  const NvU64 c2[3] = {b.gpuAddr, b.gpuAddr, b.gpuAddr};
  HT_TRUE(helios_enqueue(&ctx, cp->code, cp->count, cp->gridX, cp->blockX,
                         cp->sharedBytes, c1, 3, NULL, 0) == 0);
  HT_TRUE(helios_enqueue(&ctx, dp->code, dp->count, dp->gridX, dp->blockX,
                         dp->sharedBytes, c2, 3, NULL, 0) == 0);
  if (helios_flush(&ctx) != 0)
    HT_FAIL("flush failed, channel error 0x%x", ctx.lastError);

  /* b should be 2*(i+1): copied, then added to itself. */
  const volatile NvU32 *hb = (const volatile NvU32 *)b.hostPtr;
  for (unsigned i = 0; i < 64; i++) {
    float got;
    memcpy(&got, (const void *)&hb[i], 4);
    if (got != 2.0f * (float)(i + 1)) {
      HT_FAIL("batched: b[%u]=%g want %g", i, (double)got,
              2.0 * (double)(i + 1));
      break;
    }
  }

  gaia_free(&ctx.device, &b);
  gaia_free(&ctx.device, &a);
  helios_context_close(&ctx);
  HT_END();
}

/*
 * How many launches can share a submission before it breaks.
 *
 * Two work. A full training step does not. This walks the sizes between them
 * and reports the largest that is CORRECT, which turns "it hangs at scale" into
 * a number -- and a number is something the next change can be measured
 * against.
 *
 * Each batch is a chain: every kernel doubles what the one before it wrote, so
 * the answer after n launches is 2^n times the input. That makes the check
 * sensitive to ORDER as well as to delivery -- a batch whose kernels ran out of
 * sequence, or one where a launch was dropped, gives the wrong power of two
 * rather than merely a wrong number.
 */
static void test_batch_depth(void) {
  HT_CASE("how deep a batch can go and stay correct");

  helios_context ctx;
  if (helios_context_open(&ctx, 0) != 0) {
    printf("skip (no NVIDIA driver)\n");
    ht_case_failed = 0;
    return;
  }
  helios_program_reset();

  gaia_buffer buf;
  memset(&buf, 0, sizeof buf);
  if (gaia_alloc(&ctx.device, &buf, 4096, GAIA_SYSMEM) != 0 ||
      gaia_map_gpu(&ctx.device, &buf) != 0 ||
      gaia_map_host(&ctx.device, &buf) != 0) {
    HT_FAIL("buffer");
    helios_context_close(&ctx);
    HT_END();
    return;
  }

  const helios_key k = {HL_ELEMENTWISE, PR_EW_ADD, 64, 0};
  const helios_program *p = helios_program_get(k);
  HT_TRUE(p != NULL);

  unsigned deepest = 0;
  for (unsigned depth = 1; depth <= 32; depth *= 2) {
    volatile NvU32 *h = (volatile NvU32 *)buf.hostPtr;
    for (unsigned i = 0; i < 64; i++) {
      const float one = 1.0f;
      memcpy((void *)&h[i], &one, 4);
    }

    const NvU64 bufs[3] = {buf.gpuAddr, buf.gpuAddr, buf.gpuAddr};
    int queued = 1;
    for (unsigned i = 0; i < depth && queued; i++)
      queued = helios_enqueue(&ctx, p->code, p->count, p->gridX, p->blockX,
                              p->sharedBytes, bufs, 3, NULL, 0) == 0;
    if (!queued || helios_flush(&ctx) != 0) {
      printf("\n      depth %u: flush failed, channel error 0x%x", depth,
             ctx.lastError);
      break;
    }

    float got;
    memcpy(&got, (const void *)&h[0], 4);
    float want = 1.0f;
    for (unsigned i = 0; i < depth; i++) want *= 2.0f;
    if (got != want) {
      printf("\n      depth %u: got %g want %g", depth, (double)got,
             (double)want);
      break;
    }
    deepest = depth;
  }

  printf("\n      deepest correct batch: %u  ", deepest);
  /* Two is the floor: below that batching buys nothing and the earlier test
   * already covers it. Anything more is progress toward the step. */
  HT_TRUE(deepest >= 2);

  gaia_free(&ctx.device, &buf);
  helios_context_close(&ctx);
  HT_END();
}

void hl_batch_tests(void) {
  test_batched_launch();
  test_batch_depth();
}
