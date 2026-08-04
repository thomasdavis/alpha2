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
int pr_mask_set(unsigned i);

float pr_in_a(unsigned i);
float pr_in_b(unsigned i);
float pr_in_pos(unsigned i);
float pr_in_signed(unsigned i);

/* One shared message buffer. A checker returns NULL for pass and a pointer to
 * this for fail, so only one failure is ever in flight. */
#define PR_MSG_SIZE 96
char *pr_msg(void);

/* ---- the constants the kernels are fed ---------------------------------- */
/* These are shared deliberately: the table passes them to the GPU through the
 * constant bank and the checker uses the same symbol. A kernel scaled by one
 * number and checked against another would pass only by coincidence. */
#define PR_SCALE_BY 0.25f
#define PR_FILL_VALUE 3.5f
#define PR_CLAMP_LO (-8.0f)
#define PR_CLAMP_HI 20.0f
#define PR_RMS_EPS 1e-5f
#define PR_GELU_K0 0.7978845608028654f
#define PR_GELU_K1 0.044715f
#define PR_SOFTCAP_C 4.0f
#define PR_LOG2_E 1.4426950408889634f
#define PR_LN_2 0.6931471805599453f

/*
 * The matmul test shape: 8x8 times 8x8, which is 64 outputs and therefore
 * exactly PR_N. Square and small on purpose -- a rectangular shape would let a
 * transposed index pass, so the SQUARE case is checked here and the rectangular
 * one separately, where M, N and K are all different and no two can be
 * confused.
 */
/* How many times the loop probe goes round. Not a power of two and not equal
 * to any block or grid dimension, so a trip count confused with a thread index
 * gives a visibly wrong answer rather than an accidentally right one. */
#define PR_LOOP_TRIPS 5

/*
 * The indexing shapes. Rectangular on purpose, and rows != cols, so a transpose
 * that returned its input unchanged -- or that swapped the wrong pair of
 * dimensions -- cannot pass. A square shape would let both through.
 */
/* The causal mask is square -- it has to be, it is a token-by-token relation. */
#define PR_MASK_N 8

/* What masked_fill writes where the mask is set. Distinctive and not a value
 * any input takes, so a fill that never fired is visible. */
#define PR_MASK_FILL (-42.0f)

#define PR_TR_ROWS 4
#define PR_TR_COLS 16

/* Embedding: 8 tokens of 8 features, from a table with more rows than tokens so
 * a lookup that ignored the id and used the position would read the wrong row. */
#define PR_EMB_TOKENS 8
#define PR_EMB_DIM 8
/* The table has PR_N entries, so PR_N / PR_EMB_DIM rows. */
#define PR_EMB_ROWS (PR_N / PR_EMB_DIM)

#define PR_MM_M 8
#define PR_MM_N 8
#define PR_MM_K 8

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
