/*
 * expect_optimizer.c — the expected AdamW step.
 *
 * Written as the ALGORITHM states it: b1*m + (1-b1)*g, not the
 * m + (1-b1)*(g-m) the kernel evaluates. They are equal in exact arithmetic and
 * they are not the same expression, which is the point -- an oracle that
 * mirrored the kernel's rearrangement would agree with it even if the
 * rearrangement were wrong.
 *
 * BOTH moments are checked as well as the parameter, in chk_adamw_moments.
 * Checking only the parameter was the earlier state and it left a real hole:
 * the parameter depends on both moments, so wrong VALUES were caught, but a
 * kernel that computed them correctly and stored them to the wrong place was
 * not -- and on the second step that becomes a wrong value too, in a run where
 * nothing points back at the optimizer.
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

/*
 * The updated moments, in the buffers the kernel wrote them back to.
 *
 * Same expressions as above, evaluated a second time rather than shared with
 * the parameter check. Sharing them would mean one expression proving both the
 * moment and the parameter that depends on it, and a single wrong constant
 * would then move both sides together and cancel out.
 */
const char *chk_adamw_moments(const volatile NvU32 *m_out,
                              const volatile NvU32 *v_out) {
  for (unsigned i = 0; i < PR_N; i++) {
    const float g = pr_adam_grad(i);
    const float m = PR_ADAM_B1 * pr_adam_m(i) + (1.0f - PR_ADAM_B1) * g;
    const float v = PR_ADAM_B2 * pr_adam_v(i) + (1.0f - PR_ADAM_B2) * (g * g);
    /* Exact: these are adds and multiplies, with no MUFU anywhere. */
    if (pr_u2f(m_out[i]) != m) {
      snprintf(pr_msg(), PR_MSG_SIZE, "adamw m: [%u]=%g want %g", i,
               (double)pr_u2f(m_out[i]), (double)m);
      return pr_msg();
    }
    if (pr_u2f(v_out[i]) != v) {
      snprintf(pr_msg(), PR_MSG_SIZE, "adamw v: [%u]=%g want %g", i,
               (double)pr_u2f(v_out[i]), (double)v);
      return pr_msg();
    }
  }
  return NULL;
}
