/*
 * expect_crossentropy.c — the expected loss.
 *
 * Computed the way the DEFINITION reads: form the softmax, take the target
 * probability, take its negative log. The kernel does none of that -- it never
 * forms a softmax at all, and evaluates log(sum exp(z-m)) + m - z_target
 * instead. The two are equal algebraically, and writing the oracle as the
 * definition rather than as the kernel's rearrangement is what makes the
 * rearrangement something this test can actually check.
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

const char *chk_cross_entropy(const volatile NvU32 *o) {
  for (unsigned r = 0; r < PR_CE_ROWS; r++) {
    float sum = 0;
    for (unsigned c = 0; c < PR_CE_CLASSES; c++)
      sum += expf(pr_ce_logit(r, c));
    const float prob = expf(pr_ce_logit(r, PR_CE_TARGET(r))) / sum;
    const float want = -logf(prob);
    const float got = pr_u2f(o[r]);
    /* Two MUFU operations and a tree-ordered sum, so a tolerance. Tight enough
     * that a missing max shift or a forgotten ln(2) fails by a wide margin. */
    if (!(fabsf(got - want) / (fabsf(want) + 1e-6f) <= 1e-4f)) {
      snprintf(pr_msg(), PR_MSG_SIZE, "crossEnt: o[%u]=%g want %g", r,
               (double)got, (double)want);
      return pr_msg();
    }
  }
  return NULL;
}
