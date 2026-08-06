/*
 * builders.c — the one-line adapters between a kernel's emitter and the table.
 *
 * WHAT: each emitter takes the arguments its own kernel needs -- a shape, a trip
 * count, an operation selector -- while the table needs them all to look the
 * same. These supply the shape and discard the two addresses the common
 * signature carries for the kernels that want them.
 *
 * WHY SEPARATE FROM THE TABLE: the table is a claim about which pieces belong
 * together and nothing else. Adapters are code. Keeping code out of the table is
 * what lets it be read as data.
 */
#include "builders.h"

#define EW(tag, opv)                                                           \
  unsigned bld_##tag(hp_word *p, NvU64 out, NvU64 in) {                 \
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

unsigned bld_cpasync4(hp_word *p, NvU64 out, NvU64 in) {
  (void)out; (void)in;
  return pr_emit_cpasync_copy(p, 4);
}
unsigned bld_cpasync16(hp_word *p, NvU64 out, NvU64 in) {
  (void)out; (void)in;
  return pr_emit_cpasync_copy(p, 16);
}

unsigned bld_branch_nop(hp_word *p, NvU64 out, NvU64 in) {
  (void)out;
  (void)in;
  return pr_emit_branch_nop(p, PR_BRANCH_PLAIN);
}

unsigned bld_branch_nop_pred(hp_word *p, NvU64 out, NvU64 in) {
  (void)out;
  (void)in;
  return pr_emit_branch_nop(p, PR_BRANCH_PREDICATED);
}

unsigned bld_branch_skip(hp_word *p, NvU64 out, NvU64 in) {
  (void)out;
  (void)in;
  return pr_emit_branch_nop(p, PR_BRANCH_SKIP);
}

unsigned bld_loop_scale(hp_word *p, NvU64 out, NvU64 in) {
  (void)out;
  (void)in;
  return pr_emit_loop_scale(p, PR_LOOP_TRIPS);
}

unsigned bld_adamw(hp_word *p, NvU64 out, NvU64 in) {
  (void)out;
  (void)in;
  return pr_emit_adamw(p, 0);
}

unsigned bld_causal(hp_word *p, NvU64 out, NvU64 in) {
  (void)out;
  (void)in;
  return pr_emit_causal_mask(p, PR_MASK_N);
}

unsigned bld_masked_fill(hp_word *p, NvU64 out, NvU64 in) {
  (void)out;
  (void)in;
  return pr_emit_masked_fill(p);
}

unsigned bld_cpasync_gemm(hp_word *p, NvU64 out, NvU64 in) {
  (void)out; (void)in;
  return pr_emit_hmma_cpasync(p, 128, 64, 64, 0);
}

unsigned bld_transpose(hp_word *p, NvU64 out, NvU64 in) {
  (void)out;
  (void)in;
  return pr_emit_transpose(p, PR_TR_ROWS, PR_TR_COLS);
}

unsigned bld_embedding(hp_word *p, NvU64 out, NvU64 in) {
  (void)out;
  (void)in;
  return pr_emit_embedding(p, PR_EMB_DIM);
}

unsigned bld_matmul(hp_word *p, NvU64 out, NvU64 in) {
  (void)out;
  (void)in;
  return pr_emit_matmul(p, PR_MM_M, PR_MM_N, PR_MM_K);
}

unsigned bld_imma(hp_word *p, NvU64 out, NvU64 in) {
  (void)out;
  (void)in;
  return pr_emit_imma_tile(p);
}

unsigned bld_imma_gemm(hp_word *p, NvU64 out, NvU64 in) {
  (void)out;
  (void)in;
  return pr_emit_imma_gemm_16x8(p, IMMA_GEMM_K);
}

unsigned bld_sum(hp_word *p, NvU64 out, NvU64 in) {
  (void)out; (void)in;
  return pr_emit_reduction(p, PR_RED_SUM, PR_N);
}
unsigned bld_rms(hp_word *p, NvU64 out, NvU64 in) {
  (void)out; (void)in;
  return pr_emit_normalize(p, PR_NORM_RMS, PR_N);
}
unsigned bld_softmax(hp_word *p, NvU64 out, NvU64 in) {
  (void)out; (void)in;
  return pr_emit_normalize(p, PR_NORM_SOFTMAX, PR_N);
}
unsigned bld_layer(hp_word *p, NvU64 out, NvU64 in) {
  (void)out; (void)in;
  return pr_emit_normalize(p, PR_NORM_LAYER, PR_N);
}
unsigned bld_mean(hp_word *p, NvU64 out, NvU64 in) {
  (void)out; (void)in;
  return pr_emit_reduction(p, PR_RED_MEAN, PR_N);
}


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

unsigned bld_residual_rms(hp_word *p, NvU64 out, NvU64 in) {
  (void)out;
  (void)in;
  return pr_emit_residual_rms(p, PR_N);
}

unsigned bld_residual_dropout(hp_word *p, NvU64 out, NvU64 in) {
  (void)out;
  (void)in;
  return pr_emit_residual_dropout(p);
}

unsigned bld_cast_to_f16(hp_word *p, NvU64 out, NvU64 in) {
  (void)out;
  (void)in;
  return pr_emit_cast_f32_to_f16(p);
}

unsigned bld_cast_to_f32(hp_word *p, NvU64 out, NvU64 in) {
  (void)out;
  (void)in;
  return pr_emit_cast_f16_to_f32(p);
}

unsigned bld_dropout(hp_word *p, NvU64 out, NvU64 in) {
  (void)out;
  (void)in;
  return pr_emit_dropout_mask(p);
}

unsigned bld_cross_entropy(hp_word *p, NvU64 out, NvU64 in) {
  (void)out;
  (void)in;
  return pr_emit_cross_entropy(p, PR_CE_CLASSES);
}

unsigned bld_slice(hp_word *p, NvU64 out, NvU64 in) {
  (void)out;
  (void)in;
  return pr_emit_slice(p);
}
