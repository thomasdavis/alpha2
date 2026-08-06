/*
 * expect_elementwise.c — the expected answer for every per-element kernel.
 *
 * Each of these is the operation written the way the MATHEMATICS states it,
 * not the way the kernel computes it. gelu is the clearest case: the kernel
 * evaluates x*(1 - 1/(e^2y + 1)) because that is one reciprocal instead of a
 * tanh, and the expectation below is 0.5*x*(1 + tanh(y)) because that is the
 * definition. An oracle that mirrored the kernel would be checking the
 * simplification against itself and would agree with it even if the algebra
 * were wrong.
 */
#include "oracle.h"


ICHECK(copy, i + 1)
ICHECK(addidx, (i + 1) + i)
ICHECK(addconst, (i + 1) + 0x1234u)
ICHECK(index, i)

FCHECK(fadd, pr_in_pos, x + x)
FCHECK(fmul, pr_in_pos, x *x)
FCHECK(ffma, pr_in_pos, x *x + x)
FCHECK(fneg, pr_in_signed, -x)
FCHECK(relu, pr_in_signed, x > 0.0f ? x : 0.0f)

BCHECK(add, x + y)
BCHECK(sub, x - y)
BCHECK(mul, x *y)
BUCHECK(div, x / y)

/* scale multiplies by the scalar the kernel reads from the constant bank. */
FCHECK(scale, pr_in_pos, x *PR_SCALE_BY)
/* fills write a constant everywhere and read nothing. */
const char *chk_fill(const volatile NvU32 *o) {
  for (unsigned i = 0; i < PR_N; i++)
    if (pr_u2f(o[i]) != PR_FILL_VALUE) {
      snprintf(pr_msg(), PR_MSG_SIZE, "fill: o[%u]=%g", i, (double)pr_u2f(o[i]));
      return pr_msg();
    }
  return NULL;
}
const char *chk_zeros(const volatile NvU32 *o) {
  for (unsigned i = 0; i < PR_N; i++)
    if (pr_u2f(o[i]) != 0.0f) return "zeros: not zero";
  return NULL;
}
const char *chk_ones(const volatile NvU32 *o) {
  for (unsigned i = 0; i < PR_N; i++)
    if (pr_u2f(o[i]) != 1.0f) return "ones: not one";
  return NULL;
}

/* clamp is tested on signed input spanning both bounds, so both the max and the
 * min actually fire. Bounds chosen to bite in the middle of the range. */
FCHECK(clamp, pr_in_signed, x < PR_CLAMP_LO ? PR_CLAMP_LO : (x > PR_CLAMP_HI ? PR_CLAMP_HI : x))

/* add-inplace reads the output it is about to write. The runner seeds the
 * output with a distinct sequence so "did it accumulate" is answerable. */
void pr_seed_addinp(volatile NvU32 *o) {
  for (unsigned i = 0; i < PR_N; i++) o[i] = pr_f2u((float)(100 + i));
}
const char *chk_addinp(const volatile NvU32 *o) {
  for (unsigned i = 0; i < PR_N; i++) {
    const float want = (float)(i + 1) + (float)(100 + i);
    if (pr_u2f(o[i]) != want) {
      snprintf(pr_msg(), PR_MSG_SIZE, "addinp: o[%u]=%g want %g", i,
               (double)pr_u2f(o[i]), (double)want);
      return pr_msg();
    }
  }
  return NULL;
}

UCHECK(silu, pr_in_signed, x / (1.0f + expf(-x)))

/* Reference gelu and softCap, written the way the maths is stated rather than
 * the way the kernel computes it -- an oracle that mirrors the kernel's own
 * simplification would validate the simplification against itself. */
UCHECK(gelu, pr_in_signed,
       0.5f * x * (1.0f + tanhf(PR_GELU_K0 * (x + PR_GELU_K1 * x * x * x))))
UCHECK(softcap, pr_in_signed, PR_SOFTCAP_C *tanhf(x / PR_SOFTCAP_C))
UCHECK(exp, pr_in_pos, expf(x))
UCHECK(logn, pr_in_pos, logf(x))
UCHECK(sqrt, pr_in_pos, sqrtf(x))
UCHECK(exp2, pr_in_pos, exp2f(x))
UCHECK(log2, pr_in_pos, log2f(x))
UCHECK(rcp, pr_in_pos, 1.0f / x)
UCHECK(rsq, pr_in_pos, 1.0f / sqrtf(x))

/* ---- builders ----------------------------------------------------------- */


/* The loop probe: in[i] added to itself PR_LOOP_TRIPS times. Exact -- these are
 * small integers and the additions are exact in float. */
FCHECK(loop_scale, pr_in_pos, x *(float)PR_LOOP_TRIPS)

/* The zero-distance branch must leave the copy untouched. */
FCHECK(branch_nop, pr_in_pos, x)
