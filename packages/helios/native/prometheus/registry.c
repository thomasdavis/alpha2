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
#include "reduction.h"

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

/* fills write a constant everywhere and read nothing. */
#define FILL_VALUE 3.5f
static const char *chk_fill(const volatile NvU32 *o) {
  for (unsigned i = 0; i < PR_N; i++)
    if (pr_u2f(o[i]) != FILL_VALUE) {
      snprintf(g_msg, sizeof g_msg, "fill: o[%u]=%g", i, (double)pr_u2f(o[i]));
      return g_msg;
    }
  return NULL;
}
static const char *chk_zeros(const volatile NvU32 *o) {
  for (unsigned i = 0; i < PR_N; i++)
    if (pr_u2f(o[i]) != 0.0f) return "zeros: not zero";
  return NULL;
}
static const char *chk_ones(const volatile NvU32 *o) {
  for (unsigned i = 0; i < PR_N; i++)
    if (pr_u2f(o[i]) != 1.0f) return "ones: not one";
  return NULL;
}

/* clamp is tested on signed input spanning both bounds, so both the max and the
 * min actually fire. Bounds chosen to bite in the middle of the range. */
#define CLAMP_LO (-8.0f)
#define CLAMP_HI 20.0f
FCHECK(clamp, in_signed, x < CLAMP_LO ? CLAMP_LO : (x > CLAMP_HI ? CLAMP_HI : x))

/* add-inplace reads the output it is about to write. The runner seeds the
 * output with a distinct sequence so "did it accumulate" is answerable. */
static void seed_addinp(volatile NvU32 *o) {
  for (unsigned i = 0; i < PR_N; i++) o[i] = pr_f2u((float)(100 + i));
}
static const char *chk_addinp(const volatile NvU32 *o) {
  for (unsigned i = 0; i < PR_N; i++) {
    const float want = (float)(i + 1) + (float)(100 + i);
    if (pr_u2f(o[i]) != want) {
      snprintf(g_msg, sizeof g_msg, "addinp: o[%u]=%g want %g", i,
               (double)pr_u2f(o[i]), (double)want);
      return g_msg;
    }
  }
  return NULL;
}

/*
 * Reductions write ONE value, so only o[0] is checked.
 *
 * The tolerance is relative and small: the additions themselves are exact, but
 * a tree sums in a different ORDER than a sequential loop, and float addition
 * is not associative. Demanding bit-equality with a host loop would be
 * demanding the GPU reduce in an order it has no reason to use.
 */
static const char *chk_sum(const volatile NvU32 *o) {
  float want = 0;
  for (unsigned i = 0; i < PR_N; i++) want += (float)(i + 1);
  const float got = pr_u2f(o[0]);
  if (fabsf(got - want) / want <= 1e-6f) return NULL;
  snprintf(g_msg, sizeof g_msg, "sum: %g want %g", (double)got, (double)want);
  return g_msg;
}
static const char *chk_mean(const volatile NvU32 *o) {
  float want = 0;
  for (unsigned i = 0; i < PR_N; i++) want += (float)(i + 1);
  want /= (float)PR_N;
  const float got = pr_u2f(o[0]);
  if (fabsf(got - want) / want <= 1e-6f) return NULL;
  snprintf(g_msg, sizeof g_msg, "mean: %g want %g", (double)got, (double)want);
  return g_msg;
}

UCHECK(silu, in_signed, x / (1.0f + expf(-x)))
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
EW(fill, PR_EW_FILL)
EW(clamp, PR_EW_CLAMP)
EW(addinp, PR_EW_ADD_INPLACE)
EW(silu, PR_EW_SILU)

static unsigned bld_sum(hp_word *p, NvU64 out, NvU64 in) {
  (void)out; (void)in;
  return pr_emit_reduction(p, PR_RED_SUM, PR_N);
}
static unsigned bld_mean(hp_word *p, NvU64 out, NvU64 in) {
  (void)out; (void)in;
  return pr_emit_reduction(p, PR_RED_MEAN, PR_N);
}

/* ---- the table ---------------------------------------------------------- */

#define BLOCK 32
#define GRID (PR_N / BLOCK)

/* Base conversions, named because 1.4426950408889634 in a table row tells the
 * reader nothing. */
#define LOG2_E 1.4426950408889634f
#define LN_2 0.6931471805599453f

/*
 * The table.
 *
 * DESIGNATED INITIALISERS on purpose. A positional table has to be edited in
 * every row whenever a field is added, and gets silently wrong if two fields of
 * the same type are ever transposed -- which already happened once here, when a
 * scalar landed in the checker's slot and the compiler caught it only because
 * the types differed. Named fields make each row say what it means and make
 * adding a kernel a local change.
 *
 * Two blocks of 32 rather than one of 64, so every kernel exercises the block
 * index as well as the thread index; a grid of one would let a broken ctaid
 * pass everything here.
 */
#define K(...) {__VA_ARGS__}

static const pr_kernel KERNELS[] = {
    K(.name = "elementwise copy", .build = bld_copy, .fill = fill_ints,
      .check = chk_copy),
    K(.name = "elementwise add index", .build = bld_addidx, .fill = fill_ints,
      .check = chk_addidx),
    K(.name = "elementwise add constant", .build = bld_addconst,
      .fill = fill_ints, .check = chk_addconst),
    K(.name = "elementwise index", .build = bld_index, .check = chk_index),

    K(.name = "elementwise fadd", .build = bld_fadd, .fill = fill_pos,
      .check = chk_fadd),
    K(.name = "elementwise fmul", .build = bld_fmul, .fill = fill_pos,
      .check = chk_fmul),
    K(.name = "elementwise ffma", .build = bld_ffma, .fill = fill_pos,
      .check = chk_ffma),
    K(.name = "elementwise negate", .build = bld_fneg, .fill = fill_signed,
      .check = chk_fneg),
    K(.name = "elementwise relu", .build = bld_relu, .fill = fill_signed,
      .check = chk_relu),

    K(.name = "elementwise exp2", .build = bld_exp2, .fill = fill_pos,
      .check = chk_exp2),
    K(.name = "elementwise log2", .build = bld_log2, .fill = fill_pos,
      .check = chk_log2),
    K(.name = "elementwise reciprocal", .build = bld_rcp, .fill = fill_pos,
      .check = chk_rcp),
    K(.name = "elementwise rsqrt", .build = bld_rsq, .fill = fill_pos,
      .check = chk_rsq),

    K(.name = "elementwise add (a+b)", .build = bld_add, .fill = fill_pair,
      .check = chk_add),
    K(.name = "elementwise sub (a-b)", .build = bld_sub, .fill = fill_pair,
      .check = chk_sub),
    K(.name = "elementwise mul (a*b)", .build = bld_mul, .fill = fill_pair,
      .check = chk_mul),
    K(.name = "elementwise div (a/b)", .build = bld_div, .fill = fill_pair,
      .check = chk_div),

    K(.name = "elementwise scale", .build = bld_scale, .fill = fill_pos,
      .check = chk_scale, .scalar = SCALE_BY),
    K(.name = "elementwise exp", .build = bld_exp, .fill = fill_pos,
      .check = chk_exp, .scalar = LOG2_E),
    K(.name = "elementwise log", .build = bld_logn, .fill = fill_pos,
      .check = chk_logn, .scalar = LN_2),
    K(.name = "elementwise sqrt", .build = bld_sqrt, .fill = fill_pos,
      .check = chk_sqrt),

    /* zeros, ones and full are one kernel with a different scalar, which is
     * what they are in the existing stack too. */
    K(.name = "fill (full)", .build = bld_fill, .check = chk_fill,
      .scalar = FILL_VALUE),
    K(.name = "fill (zeros)", .build = bld_fill, .check = chk_zeros),
    K(.name = "fill (ones)", .build = bld_fill, .check = chk_ones,
      .scalar = 1.0f),

    K(.name = "elementwise clamp", .build = bld_clamp, .fill = fill_signed,
      .check = chk_clamp, .scalar = CLAMP_LO, .scalar2 = CLAMP_HI),
    K(.name = "elementwise add in place", .build = bld_addinp, .fill = fill_pos,
      .check = chk_addinp, .seed = seed_addinp),
    K(.name = "elementwise silu", .build = bld_silu, .fill = fill_signed,
      .check = chk_silu, .scalar = LOG2_E, .scalar2 = 1.0f),

    /*
     * Reductions. One block covering every element, because a tree reduction
     * is within a block by construction -- crossing blocks needs a second pass
     * or atomics, which is a separate problem.
     */
    K(.name = "reduce sum", .build = bld_sum, .fill = fill_pos,
      .check = chk_sum, .blockX = PR_N, .gridX = 1,
      .sharedBytes = PR_N * 4),
    K(.name = "reduce mean", .build = bld_mean, .fill = fill_pos,
      .check = chk_mean, .blockX = PR_N, .gridX = 1,
      .sharedBytes = PR_N * 4, .scalar = 1.0f / (float)PR_N),
};



/* Fill in the default geometry once rather than in every row. */
static pr_kernel g_resolved[sizeof KERNELS / sizeof KERNELS[0]];

const pr_kernel *pr_kernels(unsigned *count) {
  const unsigned n = sizeof KERNELS / sizeof KERNELS[0];
  for (unsigned i = 0; i < n; i++) {
    g_resolved[i] = KERNELS[i];
    if (!g_resolved[i].blockX) g_resolved[i].blockX = BLOCK;
    if (!g_resolved[i].gridX) g_resolved[i].gridX = GRID;
  }
  *count = n;
  return g_resolved;
}
