/*
 * expect_cast.c — the expected conversions.
 *
 * Both compare against the host implementation of binary16, which is written
 * from the format definition rather than from what the kernel does. The
 * comparison is EXACT: a conversion is a bit operation, not an approximation,
 * and the only correct answer is the one the format specifies. Allowing a
 * tolerance here would accept a wrong rounding mode, which is the single most
 * likely thing to be wrong.
 */
#include "expect.h"

const char *chk_cast_to_f16(const volatile NvU32 *o) {
  for (unsigned i = 0; i < PR_N / 2u; i++) {
    const NvU32 want = pr_f32_to_f16_bits(pr_cast_in(2 * i)) |
                       (pr_f32_to_f16_bits(pr_cast_in(2 * i + 1)) << 16);
    if (o[i] != want) {
      snprintf(pr_msg(), PR_MSG_SIZE, "castF16: o[%u]=%08x want %08x", i, o[i],
               want);
      return pr_msg();
    }
  }
  return NULL;
}

const char *chk_cast_to_f32(const volatile NvU32 *o) {
  for (unsigned i = 0; i < PR_N; i++) {
    /* The input was packed FROM these values, so widening returns whatever
     * survived the narrowing -- the round trip, not the original. */
    const float want = pr_half_round_trip(pr_cast_in(i));
    const float got = pr_u2f(o[i]);
    if (got != want) {
      snprintf(pr_msg(), PR_MSG_SIZE, "castF32: o[%u]=%g want %g", i,
               (double)got, (double)want);
      return pr_msg();
    }
  }
  return NULL;
}
