/*
 * helios_test.c — the facade: a persistent context, a program cache, and a
 * reduction over a tensor larger than one block.
 *
 * WHAT THE HARDWARE PART PROVES that the layers below did not: that a context
 * survives many launches, that the cache hands back the same program for the
 * same shape rather than regenerating it, and that a whole-tensor sum -- the
 * one thing block-local reductions could not do -- comes out right.
 *
 * The two-level sum is the important case. It is how the gradient norm and the
 * loss get computed, both of which are compared against thresholds, so an
 * answer that is wrong by a few percent changes training behaviour without ever
 * looking like a failure.
 */
#include "harness.h"

void hl_batch_tests(void);

#include "../helios/context.h"
#include "../helios/dispatch.h"
#include "../prometheus/elementwise.h"
#include "../helios/program.h"
#include "../helios/tensor.h"
#include "../prometheus/reduction.h"

#include <string.h>

/*
 * Deliberately not a multiple of the block size.
 *
 * 1000 over blocks of 64 leaves a short final block, which is where a two-level
 * reduction goes wrong: averaging per-block averages, or reducing a partial
 * array whose tail was never written, both produce an answer that is close and
 * wrong. A round count would hide it.
 */
#define TEST_ELEMENTS 1000u
#define TEST_BLOCK 64u

static float host_value(unsigned i) { return (float)(i % 13u) - 6.0f; }

static void test_cache_identity(void) {
  HT_CASE("the cache returns one program per shape, and distinguishes shapes");
  helios_program_reset();

  const helios_key k = {HL_MATMUL, 8, 16, 32};
  const helios_program *a = helios_program_get(k);
  HT_TRUE(a != NULL);
  HT_EQ_U64(helios_program_count(), 1);

  /* The same key must not generate a second time. */
  HT_TRUE(helios_program_get(k) == a);
  HT_EQ_U64(helios_program_count(), 1);

  /*
   * The same arguments in a different ORDER must be a different program.
   *
   * This is the collision a hash that summed or xor'd its fields would produce,
   * and its symptom is a matmul running code built for a transposed shape --
   * which executes, reads memory it should not, and returns plausible numbers.
   */
  const helios_key swapped = {HL_MATMUL, 32, 16, 8};
  const helios_program *b = helios_program_get(swapped);
  HT_TRUE(b != NULL && b != a);
  HT_EQ_U64(helios_program_count(), 2);

  /* And the launch shape travels WITH the code, so it matches the key. */
  HT_EQ_U64(a->gridX, 8);
  HT_EQ_U64(a->blockX, 16);
  HT_EQ_U64(b->gridX, 32);
  HT_EQ_U64(b->blockX, 16);
  HT_END();
}

/* Sum `count` floats from `in` into `out`, in two passes. Returns 0 on
 * success. This is the dispatcher's algorithm, written here first so the test
 * exercises it before anything depends on it. */
static int two_level_sum(helios_context *ctx, gaia_buffer *in,
                         gaia_buffer *partial, gaia_buffer *out,
                         unsigned count) {
  const unsigned blocks = (count + TEST_BLOCK - 1) / TEST_BLOCK;

  /* Pass one: one partial per block. */
  const helios_key pk = {HL_REDUCE_PARTIAL, PR_COMBINE_ADD, TEST_BLOCK, blocks};
  const helios_program *pp = helios_program_get(pk);
  if (!pp) return -1;
  const NvU64 bufs1[2] = {partial->gpuAddr, in->gpuAddr};
  if (helios_launch(ctx, pp->code, pp->count, pp->gridX, pp->blockX,
                    pp->sharedBytes, bufs1, 2, NULL, 0) != 0)
    return -1;

  /*
   * Pass two: reduce the partials.
   *
   * The tree needs a power of two, and `blocks` is not one, so the partial
   * buffer is padded with zeroes beyond `blocks` and the second pass reduces
   * the padded width. Zero is the identity for a sum, so the padding cannot
   * change the answer -- which is exactly why this trick does NOT generalise to
   * a maximum, where the identity is negative infinity and a zero-padded buffer
   * would report at least zero for a tensor that is entirely negative.
   */
  unsigned width = 1;
  while (width < blocks) width *= 2;
  const helios_key fk = {HL_REDUCE, PR_RED_SUM, width, 0};
  const helios_program *fp = helios_program_get(fk);
  if (!fp) return -1;
  const NvU64 bufs2[2] = {out->gpuAddr, partial->gpuAddr};
  return helios_launch(ctx, fp->code, fp->count, fp->gridX, fp->blockX,
                       fp->sharedBytes, bufs2, 2, NULL, 0);
}

static void test_context_and_reduction(void) {
  HT_CASE("a context runs a whole-tensor sum across many blocks");
  helios_program_reset();

  helios_context ctx;
  if (helios_context_open(&ctx, 0) != 0) {
    if (ctx.failStage == NULL || strstr(ctx.failStage, "no ") != NULL) {
      printf("skip (no NVIDIA driver)\n");
      ht_case_failed = 0;
      return;
    }
    HT_FAIL("context open failed at %s", ctx.failStage);
    HT_END();
    return;
  }

  gaia_buffer in, partial, out;
  memset(&in, 0, sizeof in);
  memset(&partial, 0, sizeof partial);
  memset(&out, 0, sizeof out);
  int ok = gaia_alloc(&ctx.device, &in, 65536, GAIA_SYSMEM) == 0 &&
           gaia_map_gpu(&ctx.device, &in) == 0 &&
           gaia_map_host(&ctx.device, &in) == 0 &&
           gaia_alloc(&ctx.device, &partial, 4096, GAIA_SYSMEM) == 0 &&
           gaia_map_gpu(&ctx.device, &partial) == 0 &&
           gaia_map_host(&ctx.device, &partial) == 0 &&
           gaia_alloc(&ctx.device, &out, 4096, GAIA_SYSMEM) == 0 &&
           gaia_map_gpu(&ctx.device, &out) == 0 &&
           gaia_map_host(&ctx.device, &out) == 0;
  if (!ok) {
    HT_FAIL("buffer allocation failed");
    helios_context_close(&ctx);
    HT_END();
    return;
  }

  volatile NvU32 *host_in = (volatile NvU32 *)in.hostPtr;
  double want = 0;
  for (unsigned i = 0; i < TEST_ELEMENTS; i++) {
    float v = host_value(i);
    memcpy((void *)&host_in[i], &v, 4);
    want += (double)v;
  }
  /* The tail of the last block, past the real data: zero, the identity. */
  for (unsigned i = TEST_ELEMENTS; i < 16384; i++) host_in[i] = 0;
  memset((void *)partial.hostPtr, 0, 4096);
  memset((void *)out.hostPtr, 0, 4096);

  if (two_level_sum(&ctx, &in, &partial, &out, TEST_ELEMENTS) != 0)
    HT_FAIL("two-level sum failed, channel error 0x%x, %u programs",
            ctx.lastError, helios_program_count());

  float got;
  memcpy(&got, (const void *)out.hostPtr, 4);
  /* Exact would be wrong to demand: the GPU sums pairwise by the tree and the
   * host sums sequentially in double, so they differ in the last bits. The
   * bound is tight enough that a dropped block fails it by orders of
   * magnitude. */
  const double err = (double)got - want;
  HT_TRUE(err > -1e-3 && err < 1e-3);

  /* Two programs generated, two shapes -- and every launch after the first of
   * each reused them. */
  HT_EQ_U64(helios_program_count(), 2);

  gaia_free(&ctx.device, &out);
  gaia_free(&ctx.device, &partial);
  gaia_free(&ctx.device, &in);
  helios_context_close(&ctx);
  HT_END();
}

/*
 * The pool, and the handles that index it.
 *
 * The generation check is the part worth testing directly: a freed handle must
 * be REJECTED, not silently address whatever now occupies its slot. Without
 * that, a use-after-free corrupts an unrelated tensor with no fault and no way
 * to trace the damage back.
 */
static void test_tensor_pool(void) {
  HT_CASE("the pool reuses buffers and rejects stale handles");

  helios_context ctx;
  if (helios_context_open(&ctx, 0) != 0) {
    printf("skip (no NVIDIA driver)\n");
    ht_case_failed = 0;
    return;
  }
  helios_tensor_release_all(&ctx);

  const helios_tensor a = helios_tensor_alloc(&ctx, 4096);
  HT_TRUE(a != HELIOS_TENSOR_NONE);
  HT_TRUE(helios_tensor_addr(a) != 0);
  HT_TRUE(helios_tensor_host(a) != NULL);
  HT_EQ_U64(helios_tensor_bytes(a), 4096);
  HT_EQ_U64(helios_tensor_get_stats().allocations, 1);

  /* Freed and re-requested at the same size: served from the pool, no second
   * trip to the driver. This is the whole reason the pool exists. */
  helios_tensor_free(a);
  const helios_tensor b = helios_tensor_alloc(&ctx, 4096);
  HT_TRUE(b != HELIOS_TENSOR_NONE);
  HT_EQ_U64(helios_tensor_get_stats().allocations, 1);

  /* Same slot, and therefore the same memory -- but NOT the same handle, and
   * the old one is dead. */
  HT_TRUE(b != a);
  HT_EQ_U64(helios_tensor_addr(a), 0);
  HT_TRUE(helios_tensor_host(a) == NULL);
  HT_TRUE(helios_tensor_addr(b) != 0);

  helios_tensor_free(b);
  helios_tensor_release_all(&ctx);
  helios_context_close(&ctx);
  HT_END();
}

/*
 * A short chain of real operations through the dispatcher.
 *
 * Chained on purpose: each reads what the one before it wrote, so it also tests
 * that launches are ordered. If they were not, the second would read whatever
 * the buffer held before the first ran -- which, on a buffer the host just
 * wrote, is a plausible number.
 */
static void test_dispatch_chain(void) {
  HT_CASE("operations chain through the dispatcher in order");

  helios_context ctx;
  if (helios_context_open(&ctx, 0) != 0) {
    printf("skip (no NVIDIA driver)\n");
    ht_case_failed = 0;
    return;
  }
  helios_tensor_release_all(&ctx);
  helios_program_reset();

  const unsigned N = 256;
  const helios_tensor x = helios_tensor_alloc(&ctx, N * 4);
  const helios_tensor y = helios_tensor_alloc(&ctx, N * 4);
  const helios_tensor sum = helios_tensor_alloc(&ctx, 4096);
  const helios_tensor scratch = helios_tensor_alloc(&ctx, 4096);
  HT_TRUE(x && y && sum && scratch);

  float *hx = (float *)helios_tensor_host(x);
  double want_sum = 0;
  for (unsigned i = 0; i < N; i++) {
    hx[i] = (float)(i % 7u) - 3.0f;
    want_sum += 2.0 * (double)hx[i]; /* after the doubling below */
  }
  memset(helios_tensor_host(scratch), 0, 4096);
  memset(helios_tensor_host(sum), 0, 4096);

  /* y = x + x, then sum(y). The second reads the first's output. */
  HT_TRUE(hl_elementwise(&ctx, PR_EW_ADD, y, x, x, N, NULL, 0) == 0);
  HT_TRUE(hl_reduce(&ctx, 0, sum, y, scratch, N) == 0);

  const float got = *(const float *)helios_tensor_host(sum);
  const double err = (double)got - want_sum;
  HT_TRUE(err > -1e-3 && err < 1e-3);

  helios_tensor_release_all(&ctx);
  helios_context_close(&ctx);
  HT_END();
}

void ht_run(void) {
  hl_batch_tests();
  test_tensor_pool();
  test_dispatch_chain();
  test_cache_identity();
  test_context_and_reduction();
}
