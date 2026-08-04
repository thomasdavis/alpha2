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
    if (fabsf(got - want) / (fabsf(want) + 1e-6f) > 1e-4f) {
      snprintf(pr_msg(), PR_MSG_SIZE, "crossEnt: o[%u]=%g want %g", r,
               (double)got, (double)want);
      return pr_msg();
    }
  }
  return NULL;
}
