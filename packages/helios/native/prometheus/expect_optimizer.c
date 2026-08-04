/*
 * expect_optimizer.c — the expected AdamW step.
 *
 * Written as the ALGORITHM states it: b1*m + (1-b1)*g, not the
 * m + (1-b1)*(g-m) the kernel evaluates. They are equal in exact arithmetic and
 * they are not the same expression, which is the point -- an oracle that
 * mirrored the kernel's rearrangement would agree with it even if the
 * rearrangement were wrong.
 *
 * Only the PARAMETER is checked, and that is a real limitation worth stating:
 * the kernel also writes m and v back, the test harness reads only the output
 * buffer, and so a kernel that computed the right parameter while corrupting
 * the moments would pass. The parameter depends on both moments, so a
 * corruption that changes their VALUES is caught; one that computes them
 * correctly and stores them to the wrong place is not.
 */
#include "expect.h"

const char *chk_adamw(const volatile NvU32 *o) {
  for (unsigned i = 0; i < PR_N; i++) {
    const float g = pr_adam_grad(i);
    const float m = PR_ADAM_B1 * pr_adam_m(i) + (1.0f - PR_ADAM_B1) * g;
    const float v =
        PR_ADAM_B2 * pr_adam_v(i) + (1.0f - PR_ADAM_B2) * (g * g);
    const float p0 = pr_adam_param(i);
    const float want =
        p0 - PR_ADAM_LR * (m / (sqrtf(v) + PR_ADAM_EPS) + PR_ADAM_WD * p0);
    const float got = pr_u2f(o[i]);
    /* MUFU is approximate and there are two of them in the chain, so this is a
     * tolerance rather than an equality. It is tight enough that a dropped term
     * -- the weight decay, say, or the epsilon -- fails it by orders of
     * magnitude. */
    const float err = fabsf(got - want) / (fabsf(want) + 1e-6f);
    if (!(err <= 1e-4f)) {
      snprintf(pr_msg(), PR_MSG_SIZE, "adamw: o[%u]=%g want %g", i, (double)got,
               (double)want);
      return pr_msg();
    }
  }
  return NULL;
}
