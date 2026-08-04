/*
 * oracle.h — what the answer should be, and how close counts as right.
 *
 * WHAT: the inputs every kernel is fed, the constants it is fed alongside them,
 * and the comparison macros that turn "an expression in C" into a checker.
 *
 * WHY IT IS A SEPARATE LAYER FROM THE ANSWERS: the expected values are algebra
 * -- exp2(x), max(x,0), x*x + x -- evaluated on the host, which is a different
 * machine doing different arithmetic from the GPU. That is as close to an
 * independent oracle as this stack gets, and it only stays independent if the
 * policy for judging agreement lives apart from the thing being judged.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: it does not compute any expected value. It
 * says how to compare and what to compare against, and nothing about what any
 * particular kernel ought to produce.
 */
#ifndef PROMETHEUS_ORACLE_H
#define PROMETHEUS_ORACLE_H

#include "kernel.h"
#include "shapes.h"

#include <math.h>
#include <stdio.h>

/* ---- inputs ------------------------------------------------------------- */

void pr_fill_ints(volatile NvU32 *a, volatile NvU32 *b);
void pr_fill_pos(volatile NvU32 *a, volatile NvU32 *b);
void pr_fill_signed(volatile NvU32 *a, volatile NvU32 *b);
void pr_fill_pair(volatile NvU32 *a, volatile NvU32 *b);
void pr_fill_embedding(volatile NvU32 *table, volatile NvU32 *ids);
NvU32 pr_emb_id(unsigned i);
void pr_fill_mask(volatile NvU32 *a, volatile NvU32 *mask);
void pr_fill_adam(volatile NvU32 *grad, volatile NvU32 *m);
void pr_fill_adam_v(volatile NvU32 *v);
void pr_fill_residual(volatile NvU32 *x, volatile NvU32 *res);
void pr_fill_res_weight(volatile NvU32 *w);
void pr_fill_res_mask(volatile NvU32 *m);
void pr_fill_cast(volatile NvU32 *a, volatile NvU32 *b);
void pr_fill_packed(volatile NvU32 *a, volatile NvU32 *b);
float pr_cast_in(unsigned i);
float pr_res_x(unsigned i);
float pr_res_r(unsigned i);
float pr_res_w(unsigned i);
float pr_res_mask(unsigned i);
void pr_seed_adam(volatile NvU32 *param);
float pr_adam_param(unsigned i);
float pr_adam_grad(unsigned i);
float pr_adam_m(unsigned i);
float pr_adam_v(unsigned i);
int pr_mask_set(unsigned i);

float pr_in_a(unsigned i);
float pr_in_b(unsigned i);
float pr_in_pos(unsigned i);
float pr_in_signed(unsigned i);

/* One shared message buffer. A checker returns NULL for pass and a pointer to
 * this for fail, so only one failure is ever in flight. */
#define PR_MSG_SIZE 96
char *pr_msg(void);

/* The constants each kernel is fed live in shapes.h, included above. */

/* ---- comparison policy --------------------------------------------------- */
/* Exact integer comparison. */
#define ICHECK(tag, expr)                                                      \
  const char *chk_##tag(const volatile NvU32 *o) {                      \
    for (unsigned i = 0; i < PR_N; i++)                                        \
      if (o[i] != (NvU32)(expr)) {                                             \
        snprintf(pr_msg(), PR_MSG_SIZE, #tag ": o[%u]=%u want %u", i, o[i],      \
                 (NvU32)(expr));                                               \
        return pr_msg();                                                          \
      }                                                                        \
    return NULL;                                                               \
  }

/* Exact float comparison — for operations the hardware computes exactly. */
#define FCHECK(tag, in_fn, expr)                                               \
  const char *chk_##tag(const volatile NvU32 *o) {                      \
    for (unsigned i = 0; i < PR_N; i++) {                                      \
      const float x = in_fn(i);                                                \
      (void)x;                                                                 \
      if (pr_u2f(o[i]) != (expr)) {                                            \
        snprintf(pr_msg(), PR_MSG_SIZE, #tag ": o[%u]=%g want %g", i,            \
                 (double)pr_u2f(o[i]), (double)(expr));                        \
        return pr_msg();                                                          \
      }                                                                        \
    }                                                                          \
    return NULL;                                                               \
  }

/*
 * Approximate float comparison, for MUFU.
 *
 * The bound is relative, generous enough for the unit's documented precision
 * and tight enough that a wrong function selector -- EX2 where LG2 was meant --
 * fails it by a mile.
 *
 * It also accepts a small ABSOLUTE error, because relative error is the wrong
 * measure near zero and gelu genuinely lands there. Its simplified form,
 * x*(1 - 1/(e+1)), subtracts two nearly equal numbers when x is a few units
 * negative: at x = -4 the true answer is -7.02e-5 and the kernel returns
 * -7.01e-5, an absolute error of 1.2e-8 and a RELATIVE error of 1.7e-3. That is
 * a real property of the algebra, not a bug, and it is the cost of turning a
 * tanh into one reciprocal. Recording it rather than quietly widening the
 * relative bound, which would also excuse errors that are not near zero.
 */
#define PR_MUFU_REL_TOL 1e-5f
#define PR_MUFU_ABS_TOL 1e-6f
#define UCHECK(tag, in_fn, expr)                                               \
  const char *chk_##tag(const volatile NvU32 *o) {                      \
    for (unsigned i = 0; i < PR_N; i++) {                                      \
      const float x = in_fn(i);                                                \
      const float want = (expr), got = pr_u2f(o[i]);                           \
      const float abs_err = fabsf(got - want);                                 \
      const float err = abs_err / (fabsf(want) + 1e-30f);                      \
      if (!(err <= PR_MUFU_REL_TOL || abs_err <= PR_MUFU_ABS_TOL)) {     \
        snprintf(pr_msg(), PR_MSG_SIZE, #tag ": o[%u]=%g want %g (rel %g)", i,   \
                 (double)got, (double)want, (double)err);                      \
        return pr_msg();                                                          \
      }                                                                        \
    }                                                                          \
    return NULL;                                                               \
  }

/* Binary comparison: x is a[i] and y is b[i]. */
#define BCHECK(tag, expr)                                                      \
  const char *chk_##tag(const volatile NvU32 *o) {                      \
    for (unsigned i = 0; i < PR_N; i++) {                                      \
      const float x = pr_in_a(i), y = pr_in_b(i);                                    \
      (void)x; (void)y;                                                        \
      if (pr_u2f(o[i]) != (expr)) {                                            \
        snprintf(pr_msg(), PR_MSG_SIZE, #tag ": o[%u]=%g want %g", i,            \
                 (double)pr_u2f(o[i]), (double)(expr));                        \
        return pr_msg();                                                          \
      }                                                                        \
    }                                                                          \
    return NULL;                                                               \
  }

/* Same, with tolerance, for anything routed through MUFU. */
#define BUCHECK(tag, expr)                                                     \
  const char *chk_##tag(const volatile NvU32 *o) {                      \
    for (unsigned i = 0; i < PR_N; i++) {                                      \
      const float x = pr_in_a(i), y = pr_in_b(i);                                    \
      const float want = (expr), got = pr_u2f(o[i]);                           \
      const float err = fabsf(got - want) / (fabsf(want) + 1e-30f);            \
      if (!(err <= PR_MUFU_REL_TOL)) {                                      \
        snprintf(pr_msg(), PR_MSG_SIZE, #tag ": o[%u]=%g want %g", i,            \
                 (double)got, (double)want);                                   \
        return pr_msg();                                                          \
      }                                                                        \
    }                                                                          \
    return NULL;                                                               \
  }

#endif /* PROMETHEUS_ORACLE_H */
