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

static void fill_ints(volatile NvU32 *in) {
  for (unsigned i = 0; i < PR_N; i++) in[i] = i + 1;
}
/* Strictly positive, so log2 and rsqrt are defined everywhere. */
static void fill_pos(volatile NvU32 *in) {
  for (unsigned i = 0; i < PR_N; i++) in[i] = pr_f2u((float)(i + 1));
}
/* Alternating sign, so relu and negation have something to do. A relu tested
 * only on positive input tests nothing. */
static void fill_signed(volatile NvU32 *in) {
  for (unsigned i = 0; i < PR_N; i++)
    in[i] = pr_f2u((i & 1) ? -(float)(i + 1) : (float)(i + 1));
}

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

/* ---- the table ---------------------------------------------------------- */

/* Two blocks of 32 rather than one of 64, so every kernel exercises both the
 * block index and the thread index. A grid of one would let a broken ctaid pass
 * everything here. */
#define BLOCK 32
#define GRID (PR_N / BLOCK)

static const pr_kernel KERNELS[] = {
    {"elementwise copy", bld_copy, BLOCK, GRID, fill_ints, chk_copy},
    {"elementwise add index", bld_addidx, BLOCK, GRID, fill_ints, chk_addidx},
    {"elementwise add constant", bld_addconst, BLOCK, GRID, fill_ints, chk_addconst},
    {"elementwise index", bld_index, BLOCK, GRID, NULL, chk_index},
    {"elementwise fadd", bld_fadd, BLOCK, GRID, fill_pos, chk_fadd},
    {"elementwise fmul", bld_fmul, BLOCK, GRID, fill_pos, chk_fmul},
    {"elementwise ffma", bld_ffma, BLOCK, GRID, fill_pos, chk_ffma},
    {"elementwise negate", bld_fneg, BLOCK, GRID, fill_signed, chk_fneg},
    {"elementwise relu", bld_relu, BLOCK, GRID, fill_signed, chk_relu},
    {"elementwise exp2", bld_exp2, BLOCK, GRID, fill_pos, chk_exp2},
    {"elementwise log2", bld_log2, BLOCK, GRID, fill_pos, chk_log2},
    {"elementwise reciprocal", bld_rcp, BLOCK, GRID, fill_pos, chk_rcp},
    {"elementwise rsqrt", bld_rsq, BLOCK, GRID, fill_pos, chk_rsq},
};

const pr_kernel *pr_kernels(unsigned *count) {
  *count = sizeof KERNELS / sizeof KERNELS[0];
  return KERNELS;
}
