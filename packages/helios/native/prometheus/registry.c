/*
 * registry.c — every kernel the stack can run.
 *
 * WHAT: one table row per kernel, naming the code generator that emits it, the
 * input it is fed, the constants it reads from the constant bank, and the
 * oracle that judges its output.
 *
 * WHY IT IS ONLY A TABLE: the three things a kernel needs -- how to build it,
 * what to feed it, and what the answer is -- come from three different files on
 * purpose. When they lived together it was possible to change a constant in the
 * checker and have the kernel silently keep the old one, because both were one
 * edit away from each other. Now the constant has exactly one definition, in
 * oracle.h, and both sides read it.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: it computes nothing and expects nothing. A
 * row is a claim about which pieces belong together, and no more.
 */
#include "elementwise.h"
#include "expect.h"
#include "indexing.h"
#include "loop.h"
#include "matmul.h"
#include "normalize.h"
#include "oracle.h"
#include "reduction.h"

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
EW(gelu, PR_EW_GELU)
EW(softcap, PR_EW_SOFTCAP)

static unsigned bld_branch_nop(hp_word *p, NvU64 out, NvU64 in) {
  (void)out;
  (void)in;
  return pr_emit_branch_nop(p, PR_BRANCH_PLAIN);
}

static unsigned bld_branch_nop_pred(hp_word *p, NvU64 out, NvU64 in) {
  (void)out;
  (void)in;
  return pr_emit_branch_nop(p, PR_BRANCH_PREDICATED);
}

static unsigned bld_branch_skip(hp_word *p, NvU64 out, NvU64 in) {
  (void)out;
  (void)in;
  return pr_emit_branch_nop(p, PR_BRANCH_SKIP);
}

static unsigned bld_loop_scale(hp_word *p, NvU64 out, NvU64 in) {
  (void)out;
  (void)in;
  return pr_emit_loop_scale(p, PR_LOOP_TRIPS);
}

static unsigned bld_transpose(hp_word *p, NvU64 out, NvU64 in) {
  (void)out;
  (void)in;
  return pr_emit_transpose(p, PR_TR_ROWS, PR_TR_COLS);
}

static unsigned bld_embedding(hp_word *p, NvU64 out, NvU64 in) {
  (void)out;
  (void)in;
  return pr_emit_embedding(p, PR_EMB_DIM);
}

static unsigned bld_matmul(hp_word *p, NvU64 out, NvU64 in) {
  (void)out;
  (void)in;
  return pr_emit_matmul(p, PR_MM_M, PR_MM_N, PR_MM_K);
}

static unsigned bld_sum(hp_word *p, NvU64 out, NvU64 in) {
  (void)out; (void)in;
  return pr_emit_reduction(p, PR_RED_SUM, PR_N);
}
static unsigned bld_rms(hp_word *p, NvU64 out, NvU64 in) {
  (void)out; (void)in;
  return pr_emit_normalize(p, PR_NORM_RMS, PR_N);
}
static unsigned bld_softmax(hp_word *p, NvU64 out, NvU64 in) {
  (void)out; (void)in;
  return pr_emit_normalize(p, PR_NORM_SOFTMAX, PR_N);
}
static unsigned bld_layer(hp_word *p, NvU64 out, NvU64 in) {
  (void)out; (void)in;
  return pr_emit_normalize(p, PR_NORM_LAYER, PR_N);
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
    K(.name = "elementwise copy", .build = bld_copy, .fill = pr_fill_ints,
      .check = chk_copy),
    K(.name = "elementwise add index", .build = bld_addidx, .fill = pr_fill_ints,
      .check = chk_addidx),
    K(.name = "elementwise add constant", .build = bld_addconst,
      .fill = pr_fill_ints, .check = chk_addconst),
    K(.name = "elementwise index", .build = bld_index, .check = chk_index),

    K(.name = "elementwise fadd", .build = bld_fadd, .fill = pr_fill_pos,
      .check = chk_fadd),
    K(.name = "elementwise fmul", .build = bld_fmul, .fill = pr_fill_pos,
      .check = chk_fmul),
    K(.name = "elementwise ffma", .build = bld_ffma, .fill = pr_fill_pos,
      .check = chk_ffma),
    K(.name = "elementwise negate", .build = bld_fneg, .fill = pr_fill_signed,
      .check = chk_fneg),
    K(.name = "elementwise relu", .build = bld_relu, .fill = pr_fill_signed,
      .check = chk_relu),

    K(.name = "elementwise exp2", .build = bld_exp2, .fill = pr_fill_pos,
      .check = chk_exp2),
    K(.name = "elementwise log2", .build = bld_log2, .fill = pr_fill_pos,
      .check = chk_log2),
    K(.name = "elementwise reciprocal", .build = bld_rcp, .fill = pr_fill_pos,
      .check = chk_rcp),
    K(.name = "elementwise rsqrt", .build = bld_rsq, .fill = pr_fill_pos,
      .check = chk_rsq),

    K(.name = "elementwise add (a+b)", .build = bld_add, .fill = pr_fill_pair,
      .check = chk_add),
    K(.name = "elementwise sub (a-b)", .build = bld_sub, .fill = pr_fill_pair,
      .check = chk_sub),
    K(.name = "elementwise mul (a*b)", .build = bld_mul, .fill = pr_fill_pair,
      .check = chk_mul),
    K(.name = "elementwise div (a/b)", .build = bld_div, .fill = pr_fill_pair,
      .check = chk_div),

    K(.name = "elementwise scale", .build = bld_scale, .fill = pr_fill_pos,
      .check = chk_scale, .scalar = PR_SCALE_BY),
    K(.name = "elementwise exp", .build = bld_exp, .fill = pr_fill_pos,
      .check = chk_exp, .scalar = PR_LOG2_E),
    K(.name = "elementwise log", .build = bld_logn, .fill = pr_fill_pos,
      .check = chk_logn, .scalar = PR_LN_2),
    K(.name = "elementwise sqrt", .build = bld_sqrt, .fill = pr_fill_pos,
      .check = chk_sqrt),

    /* zeros, ones and full are one kernel with a different scalar, which is
     * what they are in the existing stack too. */
    K(.name = "fill (full)", .build = bld_fill, .check = chk_fill,
      .scalar = PR_FILL_VALUE),
    K(.name = "fill (zeros)", .build = bld_fill, .check = chk_zeros),
    K(.name = "fill (ones)", .build = bld_fill, .check = chk_ones,
      .scalar = 1.0f),

    K(.name = "elementwise clamp", .build = bld_clamp, .fill = pr_fill_signed,
      .check = chk_clamp, .scalar = PR_CLAMP_LO, .scalar2 = PR_CLAMP_HI),
    K(.name = "elementwise add in place", .build = bld_addinp, .fill = pr_fill_pos,
      .check = chk_addinp, .seed = pr_seed_addinp),
    K(.name = "elementwise silu", .build = bld_silu, .fill = pr_fill_signed,
      .check = chk_silu, .scalar = PR_LOG2_E, .scalar2 = 1.0f),
    K(.name = "elementwise gelu", .build = bld_gelu, .fill = pr_fill_signed,
      .check = chk_gelu, .scalar = PR_GELU_K1,
      .scalar2 = 2.0f * PR_GELU_K0 * PR_LOG2_E, .scalar3 = 1.0f),
    K(.name = "elementwise softCap", .build = bld_softcap, .fill = pr_fill_signed,
      .check = chk_softcap, .scalar = 2.0f * PR_LOG2_E / PR_SOFTCAP_C,
      .scalar2 = 1.0f, .scalar3 = PR_SOFTCAP_C, .scalar4 = 2.0f),

    /*
     * Reductions. One block covering every element, because a tree reduction
     * is within a block by construction -- crossing blocks needs a second pass
     * or atomics, which is a separate problem.
     */
    K(.name = "branch to next instruction", .build = bld_branch_nop,
      .fill = pr_fill_pos, .check = chk_branch_nop),
    K(.name = "predicated branch to next", .build = bld_branch_nop_pred,
      .fill = pr_fill_pos, .check = chk_branch_nop),
    K(.name = "branch over an instruction", .build = bld_branch_skip,
      .fill = pr_fill_pos, .check = chk_branch_nop),

    /* Ordered BEFORE matmul on purpose: it is the smaller of the two kernels
     * with a branch, so when both fail it says which suspect to look at. */
    K(.name = "loop scale", .build = bld_loop_scale, .fill = pr_fill_pos,
      .check = chk_loop_scale),

    K(.name = "transpose 4x16", .build = bld_transpose, .blockX = PR_TR_COLS,
      .gridX = PR_TR_ROWS, .fill = pr_fill_pos, .check = chk_transpose),
    K(.name = "embedding lookup", .build = bld_embedding, .blockX = PR_EMB_DIM,
      .gridX = PR_EMB_TOKENS, .fill = pr_fill_embedding, .check = chk_embedding),

    /* One block per row, one thread per column -- NOT the default launch, and
     * the registry is where that is said, because the kernel cannot see it. */
    K(.name = "matmul 8x8x8", .build = bld_matmul, .blockX = PR_MM_N,
      .gridX = PR_MM_M, .fill = pr_fill_pair, .check = chk_matmul),

    K(.name = "reduce sum", .build = bld_sum, .fill = pr_fill_pos,
      .check = chk_sum, .blockX = PR_N, .gridX = 1,
      .sharedBytes = PR_N * 4),
    K(.name = "reduce mean", .build = bld_mean, .fill = pr_fill_pos,
      .check = chk_mean, .blockX = PR_N, .gridX = 1,
      .sharedBytes = PR_N * 4, .scalar = 1.0f / (float)PR_N),

    /* Normalisation: a reduction whose result every thread reads back. */
    K(.name = "rmsNorm", .build = bld_rms, .fill = pr_fill_signed,
      .check = chk_rms, .blockX = PR_N, .gridX = 1, .sharedBytes = PR_N * 4,
      .scalar = 1.0f / (float)PR_N, .scalar2 = PR_RMS_EPS),
    K(.name = "softmax", .build = bld_softmax, .fill = pr_fill_signed,
      .check = chk_softmax, .blockX = PR_N, .gridX = 1,
      .sharedBytes = PR_N * 4, .scalar = PR_LOG2_E),
    K(.name = "layerNorm", .build = bld_layer, .fill = pr_fill_signed,
      .check = chk_layer, .blockX = PR_N, .gridX = 1, .sharedBytes = PR_N * 4,
      .scalar = 1.0f / (float)PR_N, .scalar2 = PR_RMS_EPS),
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
