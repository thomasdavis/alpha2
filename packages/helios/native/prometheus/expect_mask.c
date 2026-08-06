/*
 * expect_mask.c — the expected masking.
 *
 * The causal check tests the DIAGONAL explicitly as well as the two triangles,
 * because ">" and ">=" differ only there and getting it wrong is silent: a model
 * with a mask that hides the current token still trains, just worse, and nothing
 * in the numbers looks out of place.
 */
#include "expect.h"

#include <math.h>

const char *chk_causal(const volatile NvU32 *o) {
  for (unsigned r = 0; r < PR_MASK_N; r++)
    for (unsigned c = 0; c < PR_MASK_N; c++) {
      const float got = pr_u2f(o[r * PR_MASK_N + c]);
      if (c > r) {
        /* Masked: true negative infinity, not merely very negative. Checking
         * isinf rather than a threshold is the point -- a large finite number
         * would pass a threshold and then behave differently under softmax. */
        if (!(isinf(got) && got < 0)) {
          snprintf(pr_msg(), PR_MSG_SIZE, "causal: o[%u][%u]=%g want -inf", r, c,
                   (double)got);
          return pr_msg();
        }
      } else {
        /* Kept, INCLUDING the diagonal where c == r. */
        const float want = (float)(r * PR_MASK_N + c + 1);
        if (got != want) {
          snprintf(pr_msg(), PR_MSG_SIZE, "causal: o[%u][%u]=%g want %g", r, c,
                   (double)got, (double)want);
          return pr_msg();
        }
      }
    }
  return NULL;
}

const char *chk_masked_fill(const volatile NvU32 *o) {
  for (unsigned i = 0; i < PR_N; i++) {
    const float want = pr_mask_set(i) ? PR_MASK_FILL : (float)(i + 1);
    const float got = pr_u2f(o[i]);
    if (got != want) {
      snprintf(pr_msg(), PR_MSG_SIZE, "maskedFill: o[%u]=%g want %g", i,
               (double)got, (double)want);
      return pr_msg();
    }
  }
  return NULL;
}
