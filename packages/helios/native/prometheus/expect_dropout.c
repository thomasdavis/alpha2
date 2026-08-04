/*
 * expect_dropout.c — the expected mask.
 *
 * Checked POSITION by position, not by counting how many were dropped. At
 * p = 0.5 a kernel with the comparison inverted drops the right NUMBER of
 * elements and the wrong ones, and a count would call that correct.
 *
 * It also asserts that both outcomes actually occur. A hash that returned a
 * constant would produce an all-kept or all-dropped mask, which agrees with a
 * per-position check only if the oracle's hash is broken in the same way -- and
 * since the oracle's hash IS the same algorithm, that is a real possibility
 * worth ruling out separately.
 */
#include "expect.h"

const char *chk_dropout(const volatile NvU32 *o) {
  unsigned kept = 0, dropped = 0;
  for (unsigned i = 0; i < PR_N; i++) {
    const int drop = pr_drop_hash(i) < PR_DROP_THRESHOLD;
    const float want = drop ? 0.0f : PR_DROP_SCALE;
    const float got = pr_u2f(o[i]);
    if (got != want) {
      snprintf(pr_msg(), PR_MSG_SIZE, "dropout: o[%u]=%g want %g", i,
               (double)got, (double)want);
      return pr_msg();
    }
    if (drop) dropped++; else kept++;
  }
  if (!kept || !dropped) {
    snprintf(pr_msg(), PR_MSG_SIZE, "dropout: degenerate mask, %u kept %u dropped",
             kept, dropped);
    return pr_msg();
  }
  return NULL;
}
