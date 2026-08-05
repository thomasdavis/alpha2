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
#include "builders.h"
#include "expect.h"
#include "oracle.h"

/* One thunk per operation. They exist only because a table needs function
 * pointers and pr_emit_elementwise needs its op; there is no logic here. */
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

    /* The parameter is BOTH input and output, so it is seeded into the output
     * buffer rather than filled into an input one. Slot 1 is the gradient,
     * slot 2 is m -- which fill writes as a pair -- and slot 3 is v. */
    /* One block of PR_N threads: the RMS reduction is block-local, so the row
     * has to fit in a block. A longer row needs a two-level reduction, which
     * is a different kernel and not a parameter of this one. */
    K(.name = "residual + rmsNorm", .build = bld_residual_rms, .blockX = PR_N,
      .gridX = 1, .fill = pr_fill_residual, .fillC = pr_fill_res_weight,
      .check = chk_residual_rms, .scalar = 1.0f / (float)PR_N,
      .scalar2 = PR_RES_EPS, .sharedBytes = PR_N * 4),
    K(.name = "residual + dropout add", .build = bld_residual_dropout,
      .blockX = PR_N, .gridX = 1, .fill = pr_fill_residual,
      .fillC = pr_fill_res_mask, .check = chk_residual_dropout,
      .scalar = PR_DROP_SCALE),

    /* Half the threads, because each handles a pair. The launch is where that
     * is said; the kernel cannot see how many elements exist. */
    K(.name = "cast f32 to f16", .build = bld_cast_to_f16,
      .blockX = PR_BLOCK / 2, .gridX = PR_GRID, .fill = pr_fill_cast,
      .check = chk_cast_to_f16, .elementsPerThread = 2,
      .checkedElements = PR_N / 2),
    K(.name = "cast f16 to f32", .build = bld_cast_to_f32,
      .blockX = PR_BLOCK / 2, .gridX = PR_GRID, .fill = pr_fill_packed,
      .check = chk_cast_to_f32, .elementsPerThread = 2),

    /* Three raw integer scalars and one float. The mask is derived from the
     * element index alone, so this kernel reads no input at all. */
    K(.name = "dropout mask", .build = bld_dropout, .check = chk_dropout,
      .scalar4 = PR_DROP_SCALE,
      .rawScalar = {PR_DROP_SEED, PR_DROP_COUNTER, PR_DROP_THRESHOLD}),

    /* One block per row. Only PR_CE_ROWS of the output are written, so the
     * launch covers the LOGITS and the check reads only the rows. */
    K(.name = "cross entropy", .build = bld_cross_entropy,
      .blockX = PR_CE_CLASSES, .gridX = PR_CE_ROWS, .fill = pr_fill_ce,
      .check = chk_cross_entropy, .checkedElements = PR_CE_ROWS, .scalar = PR_LOG2_E, .scalar2 = PR_LN_2,
      .sharedBytes = PR_CE_CLASSES * 4),

    K(.name = "adamw step", .build = bld_adamw, .fill = pr_fill_adam,
      .fillC = pr_fill_adam_v, .seed = pr_seed_adam, .check = chk_adamw,
      .checkAux = chk_adamw_moments,
      .scalar = 1.0f - PR_ADAM_B1, .scalar2 = 1.0f - PR_ADAM_B2,
      .scalar3 = PR_ADAM_LR, .scalar4 = PR_ADAM_EPS, .scalar5 = PR_ADAM_WD),

    K(.name = "causal mask 8x8", .build = bld_causal, .blockX = PR_MASK_N,
      .gridX = PR_MASK_N, .fill = pr_fill_pos, .check = chk_causal),
    K(.name = "masked fill", .build = bld_masked_fill, .fill = pr_fill_mask,
      .check = chk_masked_fill, .scalar = PR_MASK_FILL),

    K(.name = "slice", .build = bld_slice, .blockX = PR_SLICE_COUNT,
      .gridX = 1, .fill = pr_fill_pos, .check = chk_slice,
      .workElements = PR_SLICE_COUNT,
      .rawScalar = {PR_SLICE_OFFSET, PR_SLICE_STRIDE}),

    K(.name = "transpose 4x16", .build = bld_transpose, .blockX = PR_TR_COLS,
      .gridX = PR_TR_ROWS, .fill = pr_fill_pos, .check = chk_transpose),
    K(.name = "embedding lookup", .build = bld_embedding, .blockX = PR_EMB_DIM,
      .gridX = PR_EMB_TOKENS, .fill = pr_fill_embedding, .check = chk_embedding),

    /* One block per row, one thread per column -- NOT the default launch, and
     * the registry is where that is said, because the kernel cannot see it. */
    /* Shared memory comes from the emitter, not from a number written here: the
     * matmul stages A's row when the shape allows and a QMD that declared zero
     * would fault the launch. Asking the emitter is the only way the two cannot
     * disagree. */
    /* 64 because the tiled emitter's four-row register map reaches R53, and a
     * declaration with no slack above the highest register used raises
     * GR_EXCEPTION (see qmd.c). The device path declares the same via
     * helios_program.regs. */
    K(.name = "matmul 8x8x8", .build = bld_matmul, .blockX = PR_MM_N, .regs = 64,
      .sharedBytes = PR_MM_K * 4, .gridX = PR_MM_M, .fill = pr_fill_pair, .check = chk_matmul),

    K(.name = "reduce sum", .build = bld_sum, .checkedElements = 1, .fill = pr_fill_pos,
      .check = chk_sum, .blockX = PR_N, .gridX = 1,
      .sharedBytes = PR_N * 4),
    K(.name = "reduce mean", .build = bld_mean, .checkedElements = 1, .fill = pr_fill_pos,
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
    if (!g_resolved[i].blockX) g_resolved[i].blockX = PR_BLOCK;
    if (!g_resolved[i].gridX) g_resolved[i].gridX = PR_GRID;
  }
  *count = n;
  return g_resolved;
}
