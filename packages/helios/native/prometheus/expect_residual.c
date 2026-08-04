/*
 * expect_residual.c — the expected fused results.
 *
 * Both are written as the composition they FUSE: add, then normalise; drop,
 * then add. The kernels interleave those steps to avoid a round trip through
 * memory, and the whole point of fusing is that it should not change the
 * answer, so the oracle is the unfused version.
 */
/*
 * NOTE ON THE COMPARISON'S SHAPE: written as !(err <= tol), never err > tol.
 *
 * They differ on NaN, and only on NaN. Any comparison involving NaN is false,
 * so "err > tol" does not fire and the value is ACCEPTED; "!(err <= tol)" fires
 * and rejects it. Three checkers here were the first form, and a kernel that
 * produced NaN would have passed all three -- which is the exact output a
 * broken normalisation gives, from a zero-divided-by-zero or an
 * infinity-minus-infinity.
 *
 * Found by the runner's mutation pass rather than by reading: it flips an
 * exponent bit in each checked slot and requires the checker to object, and any
 * output in [1,2) becomes a NaN under that flip.
 */
#include "expect.h"

const char *chk_residual_rms(const volatile NvU32 *o) {
  float sumsq = 0;
  for (unsigned i = 0; i < PR_N; i++) {
    const float h = pr_res_x(i) + pr_res_r(i);
    sumsq += h * h;
  }
  const float inv = 1.0f / sqrtf(sumsq / (float)PR_N + PR_RES_EPS);
  for (unsigned i = 0; i < PR_N; i++) {
    const float want = pr_res_w(i) * (pr_res_x(i) + pr_res_r(i)) * inv;
    const float got = pr_u2f(o[i]);
    /* One MUFU and a tree-ordered sum, so a tolerance rather than equality. */
    if (!(fabsf(got - want) / (fabsf(want) + 1e-6f) <= 1e-4f)) {
      snprintf(pr_msg(), PR_MSG_SIZE, "resRms: o[%u]=%g want %g", i,
               (double)got, (double)want);
      return pr_msg();
    }
  }
  return NULL;
}

const char *chk_residual_dropout(const volatile NvU32 *o) {
  for (unsigned i = 0; i < PR_N; i++) {
    const float want =
        pr_res_r(i) + pr_res_x(i) * pr_res_mask(i) * PR_DROP_SCALE;
    const float got = pr_u2f(o[i]);
    /* Exact: no transcendental anywhere in this one. */
    if (got != want) {
      snprintf(pr_msg(), PR_MSG_SIZE, "resDrop: o[%u]=%g want %g", i,
               (double)got, (double)want);
      return pr_msg();
    }
  }
  return NULL;
}
