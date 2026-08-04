/*
 * registry.c — every kernel the stack can run, and what "correct" means for it.
 *
 * WHY the expected answers are written as expressions rather than tables: an
 * expected value computed by the same code that drives the GPU is a comparison
 * against a second implementation, and two implementations agreeing proves
 * nothing when both are wrong. These are algebra — exp2(x), max(x,0), x*x + x —
 * evaluated in C on the host, which is a different machine doing different
 * arithmetic. That is as close to an independent oracle as this layer gets.
 *
 * WHY the transcendentals get a tolerance and the rest do not: MUFU is the
 * multi-function unit and it is APPROXIMATE by design, roughly 22 bits for
 * exp2/log2 and 23 for reciprocal. Demanding exact equality there would be
 * demanding the hardware be something it is not. Everything else — adds,
 * multiplies, fused multiply-add, min/max, integer work — is exact, and is
 * checked exactly. A blanket tolerance would hide real bugs in the exact ops.
 */
#include "elementwise.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

NvU32 pr_f2u(float f) { NvU32 u; memcpy(&u, &f, 4); return u; }
float pr_u2f(NvU32 u) { float f; memcpy(&f, &u, 4); return f; }

/* ---- inputs ------------------------------------------------------------- */

static void fill_ints(volatile NvU32 *a, volatile NvU32 *b) {
  (void)b;
  for (unsigned i = 0; i < PR_N; i++) a[i] = i + 1;
}
/* Strictly positive, so log2 and rsqrt are defined everywhere. */
static void fill_pos(volatile NvU32 *a, volatile NvU32 *b) {
  (void)b;
  for (unsigned i = 0; i < PR_N; i++) a[i] = pr_f2u((float)(i + 1));
}
/* Alternating sign, so relu and negation have something to do. A relu tested
 * only on positive input tests nothing. */
static void fill_signed(volatile NvU32 *a, volatile NvU32 *b) {
  (void)b;
  for (unsigned i = 0; i < PR_N; i++)
    a[i] = pr_f2u((i & 1) ? -(float)(i + 1) : (float)(i + 1));
}

/* Two operands for the binary kernels: a[i] = i+1, b[i] = 2i+3. Distinct, both
 * non-zero so division is defined, and different enough that a kernel which
 * confuses its inputs fails rather than coincidentally agreeing. */
static void fill_pair(volatile NvU32 *a, volatile NvU32 *b) {
  for (unsigned i = 0; i < PR_N; i++) {
    a[i] = pr_f2u((float)(i + 1));
    b[i] = pr_f2u((float)(2 * i + 3));
  }
}
static float in_a(unsigned i) { return (float)(i + 1); }
static float in_b(unsigned i) { return (float)(2 * i + 3); }

static float in_pos(unsigned i) { return (float)(i + 1); }
static float in_signed(unsigned i) {
  return (i & 1) ? -(float)(i + 1) : (float)(i + 1);
}

/* ---- checkers ----------------------------------------------------------- */

static char g_msg[96];

/* Exact integer comparison. */
#define ICHECK(tag, expr)                                                      \
  static const char *chk_##tag(const volatile NvU32 *o) {                      \
    for (unsigned i = 0; i < PR_N; i++)                                        \
      if (o[i] != (NvU32)(expr)) {                                             \
        snprintf(g_msg, sizeof g_msg, #tag ": o[%u]=%u want %u", i, o[i],      \
                 (NvU32)(expr));                                               \
        return g_msg;                                                          \
      }                                                                        \
    return NULL;                                                               \
  }

/* Exact float comparison — for operations the hardware computes exactly. */
#define FCHECK(tag, in_fn, expr)                                               \
  static const char *chk_##tag(const volatile NvU32 *o) {                      \
    for (unsigned i = 0; i < PR_N; i++) {                                      \
      const float x = in_fn(i);                                                \
      (void)x;                                                                 \
      if (pr_u2f(o[i]) != (expr)) {                                            \
        snprintf(g_msg, sizeof g_msg, #tag ": o[%u]=%g want %g", i,            \
                 (double)pr_u2f(o[i]), (double)(expr));                        \
        return g_msg;                                                          \
      }                                                                        \
    }                                                                          \
    return NULL;                                                               \
  }

/* Approximate float comparison, for MUFU. The bound is relative and generous
 * enough for the unit's documented precision, and tight enough that a wrong
 * function selector — EX2 where LG2 was meant — fails it by a mile. */
#define MUFU_REL_TOLERANCE 1e-5f
#define UCHECK(tag, in_fn, expr)                                               \
  static const char *chk_##tag(const volatile NvU32 *o) {                      \
    for (unsigned i = 0; i < PR_N; i++) {                                      \
      const float x = in_fn(i);                                                \
      const float want = (expr), got = pr_u2f(o[i]);                           \
      const float err = fabsf(got - want) / (fabsf(want) + 1e-30f);            \
      if (!(err <= MUFU_REL_TOLERANCE)) {                                      \
        snprintf(g_msg, sizeof g_msg, #tag ": o[%u]=%g want %g (rel %g)", i,   \
                 (double)got, (double)want, (double)err);                      \
        return g_msg;                                                          \
      }                                                                        \
    }                                                                          \
    return NULL;                                                               \
  }

ICHECK(copy, i + 1)
ICHECK(addidx, (i + 1) + i)
ICHECK(addconst, (i + 1) + 0x1234u)
ICHECK(index, i)

FCHECK(fadd, in_pos, x + x)
FCHECK(fmul, in_pos, x *x)
FCHECK(ffma, in_pos, x *x + x)
FCHECK(fneg, in_signed, -x)
FCHECK(relu, in_signed, x > 0.0f ? x : 0.0f)

/* Binary comparison: x is a[i] and y is b[i]. */
#define BCHECK(tag, expr)                                                      \
  static const char *chk_##tag(const volatile NvU32 *o) {                      \
    for (unsigned i = 0; i < PR_N; i++) {                                      \
      const float x = in_a(i), y = in_b(i);                                    \
      (void)x; (void)y;                                                        \
      if (pr_u2f(o[i]) != (expr)) {                                            \
        snprintf(g_msg, sizeof g_msg, #tag ": o[%u]=%g want %g", i,            \
                 (double)pr_u2f(o[i]), (double)(expr));                        \
        return g_msg;                                                          \
      }                                                                        \
    }                                                                          \
    return NULL;                                                               \
  }

/* Same, with tolerance, for anything routed through MUFU. */
#define BUCHECK(tag, expr)                                                     \
  static const char *chk_##tag(const volatile NvU32 *o) {                      \
    for (unsigned i = 0; i < PR_N; i++) {                                      \
      const float x = in_a(i), y = in_b(i);                                    \
      const float want = (expr), got = pr_u2f(o[i]);                           \
      const float err = fabsf(got - want) / (fabsf(want) + 1e-30f);            \
      if (!(err <= MUFU_REL_TOLERANCE)) {                                      \
        snprintf(g_msg, sizeof g_msg, #tag ": o[%u]=%g want %g", i,            \
                 (double)got, (double)want);                                   \
        return g_msg;                                                          \
      }                                                                        \
    }                                                                          \
    return NULL;                                                               \
  }

BCHECK(add, x + y)
BCHECK(sub, x - y)
BCHECK(mul, x *y)
BUCHECK(div, x / y)

/* scale multiplies by the scalar the kernel reads from the constant bank. */
#define SCALE_BY 0.25f
FCHECK(scale, in_pos, x *SCALE_BY)

UCHECK(exp, in_pos, expf(x))
UCHECK(logn, in_pos, logf(x))
UCHECK(sqrt, in_pos, sqrtf(x))
UCHECK(exp2, in_pos, exp2f(x))
UCHECK(log2, in_pos, log2f(x))
UCHECK(rcp, in_pos, 1.0f / x)
UCHECK(rsq, in_pos, 1.0f / sqrtf(x))

/* ---- builders ----------------------------------------------------------- */

/* One thunk per operation. They exist only because a table needs function
 * pointers and pr_emit_elementwise needs its op; there is no logic here. */
#define EW(tag, opv)                                                           \
  static unsigned bld_##tag(hp_word *p, NvU64 out, NvU64 in) {                 \
    (void)out;                                                                 \
    (void)in;                                                                  \
    return pr_emit_elementwise(p, opv);                                        \
  }

EW(copy, PR_EW_COPY)
EW(addidx, PR_EW_ADD_INDEX)
EW(addconst, PR_EW_ADD_CONST)
EW(index, PR_EW_INDEX)
EW(fadd, PR_EW_FADD)
EW(fmul, PR_EW_FMUL)
EW(ffma, PR_EW_FFMA)
EW(fneg, PR_EW_FNEG)
EW(relu, PR_EW_RELU)
EW(exp2, PR_EW_EXP2)
EW(log2, PR_EW_LOG2)
EW(rcp, PR_EW_RCP)
EW(rsq, PR_EW_RSQ)
EW(add, PR_EW_ADD)
EW(sub, PR_EW_SUB)
EW(mul, PR_EW_MUL)
EW(div, PR_EW_DIV)
EW(scale, PR_EW_SCALE)
EW(exp, PR_EW_EXP)
EW(logn, PR_EW_LOG)
EW(sqrt, PR_EW_SQRT)

/* ---- the table ---------------------------------------------------------- */

/* Two blocks of 32 rather than one of 64, so every kernel exercises both the
 * block index and the thread index. A grid of one would let a broken ctaid pass
 * everything here. */
#define BLOCK 32
#define GRID (PR_N / BLOCK)

static const pr_kernel KERNELS[] = {
    /* name                       build      block  grid   fill         check       scalar */
    {"elementwise copy",          bld_copy,  BLOCK, GRID, fill_ints,   chk_copy,     0.0f},
    {"elementwise add index",     bld_addidx,BLOCK, GRID, fill_ints,   chk_addidx,   0.0f},
    {"elementwise add constant",  bld_addconst,BLOCK,GRID,fill_ints,   chk_addconst, 0.0f},
    {"elementwise index",         bld_index, BLOCK, GRID, NULL,        chk_index,    0.0f},

    {"elementwise fadd",          bld_fadd,  BLOCK, GRID, fill_pos,    chk_fadd,     0.0f},
    {"elementwise fmul",          bld_fmul,  BLOCK, GRID, fill_pos,    chk_fmul,     0.0f},
    {"elementwise ffma",          bld_ffma,  BLOCK, GRID, fill_pos,    chk_ffma,     0.0f},
    {"elementwise negate",        bld_fneg,  BLOCK, GRID, fill_signed, chk_fneg,     0.0f},
    {"elementwise relu",          bld_relu,  BLOCK, GRID, fill_signed, chk_relu,     0.0f},

    {"elementwise exp2",          bld_exp2,  BLOCK, GRID, fill_pos,    chk_exp2,     0.0f},
    {"elementwise log2",          bld_log2,  BLOCK, GRID, fill_pos,    chk_log2,     0.0f},
    {"elementwise reciprocal",    bld_rcp,   BLOCK, GRID, fill_pos,    chk_rcp,      0.0f},
    {"elementwise rsqrt",         bld_rsq,   BLOCK, GRID, fill_pos,    chk_rsq,      0.0f},

    /* Binary. */
    {"elementwise add (a+b)",     bld_add,   BLOCK, GRID, fill_pair,   chk_add,      0.0f},
    {"elementwise sub (a-b)",     bld_sub,   BLOCK, GRID, fill_pair,   chk_sub,      0.0f},
    {"elementwise mul (a*b)",     bld_mul,   BLOCK, GRID, fill_pair,   chk_mul,      0.0f},
    {"elementwise div (a/b)",     bld_div,   BLOCK, GRID, fill_pair,   chk_div,      0.0f},

    /* Scalar operand, and the composed unaries the hardware has no opcode for.
     * The constants are the base conversions: log2(e) for exp, ln(2) for log. */
    {"elementwise scale",         bld_scale, BLOCK, GRID, fill_pos,    chk_scale, SCALE_BY},
    {"elementwise exp",           bld_exp,   BLOCK, GRID, fill_pos,    chk_exp,
     1.4426950408889634f},
    {"elementwise log",           bld_logn,  BLOCK, GRID, fill_pos,    chk_logn,
     0.6931471805599453f},
    {"elementwise sqrt",          bld_sqrt,  BLOCK, GRID, fill_pos,    chk_sqrt,     0.0f},
};


const pr_kernel *pr_kernels(unsigned *count) {
  *count = sizeof KERNELS / sizeof KERNELS[0];
  return KERNELS;
}
