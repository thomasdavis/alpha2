/*
 * residual.c — the two fused kernels a transformer block is built from.
 *
 * WHAT: residual-add-then-RMS-normalise, and residual-add-with-dropout.
 *
 * WHY FUSED: both are memory-bound. Written as separate kernels, the sum would
 * be written to memory and read straight back by the normalise, doubling the
 * traffic to save nothing -- the intermediate is used exactly once, by the very
 * next instruction, and never needs to exist in memory at all. Fusion here is
 * not a micro-optimisation; it halves the bandwidth of the most frequently
 * executed pair in the model.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: it does not write the summed residual back.
 * A real block needs that sum as the input to the NEXT residual connection, and
 * emitting it would need a fifth buffer slot. The layout has four. This is a
 * genuine gap rather than a decision, and it is recorded here rather than
 * silently worked around, because a caller that assumes the residual stream is
 * being maintained would be wrong.
 *
 * THE HARDWARE FACT: the reduction goes through shared memory and every thread
 * needs the same total, so the tree's barriers are what make the result defined
 * at all -- see reduction.c. A block's worth of elements is one kernel's worth
 * of work here; a row longer than a block would need a second level.
 */
#include "residual.h"

#include "reduction.h"

enum {
  R_TID = 0,
  R_ADDR = 2, /* R2:R3 */
  R_X = 4,    /* the summed residual, live across the whole kernel */
  R_ESIZE = 5,
  R_LHS = 6,
  R_RHS = 7,
  R_ACC = 8,
  R_RED = 9,
  R_OUT = 10, /* R10:R11 */
  R_S0 = 12,
  R_S1 = 13,
  R_TMP = 14,
  R_OTHER = 15,
};

#define BAR_TID 0
#define BAR_LOAD 1
#define BAR_LDS 2
#define BAR_MUFU 3

/* Load element `tid` of the tensor in parameter slot `slot` into `dst`. */
static unsigned load_slot(hp_word *p, unsigned dst, unsigned slot,
                          hp_control c) {
  unsigned n = 0;
  p[n++] = hp_imad_wide_const(R_ADDR, R_TID, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(slot), hp_ctrl_safe());
  p[n++] = hp_ldg(dst, R_ADDR, 0, c);
  return n;
}

/* tid into R_TID, and 4 into R_ESIZE. Every kernel here starts this way. */
static unsigned emit_preamble(hp_word *p) {
  unsigned n = 0;
  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(BAR_TID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_wait(BAR_TID));
  return n;
}

/*
 * out = w * (x + residual) / sqrt(mean((x + residual)^2) + eps)
 *
 * The sum is formed first and then normalised, which is the only order that
 * makes sense -- normalising x and then adding the residual would leave the
 * residual stream unnormalised and is a different function entirely. Stating
 * that because the two are one line apart in source and produce plausible
 * numbers either way.
 */
unsigned pr_emit_residual_rms(hp_word *p, unsigned elements) {
  unsigned n = emit_preamble(p);

  n += load_slot(&p[n], R_X, 1, hp_ctrl_setbar(BAR_LOAD));
  n += load_slot(&p[n], R_OTHER, 2, hp_ctrl_setbar(BAR_LOAD));
  p[n++] = hp_fadd(R_X, R_X, R_OTHER, hp_ctrl_wait(BAR_LOAD));

  /* The sum of squares, through the shared-memory tree. */
  p[n++] = hp_fmul(R_ACC, R_X, R_X, hp_ctrl_safe());
  p[n++] = hp_sts(R_TID, R_ACC, 0, hp_ctrl_safe());
  p[n++] = hp_bar_sync(hp_ctrl_safe());
  n += pr_emit_tree(&p[n], elements, PR_COMBINE_ADD, R_TID, R_LHS, R_RHS);
  p[n++] = hp_lds(R_RED, HP_RZ, 0, hp_ctrl_setbar(BAR_LDS));

  /* mean, then + eps, then the reciprocal square root -- which is the primitive
   * the hardware actually has, so normalising is a multiply not a divide. */
  p[n++] = hp_mov_const(R_S0, 0, HERMES_CBUF0_SCALAR_N(0), hp_ctrl_safe());
  p[n++] = hp_mov_const(R_S1, 0, HERMES_CBUF0_SCALAR_N(1), hp_ctrl_safe());
  p[n++] = hp_fmul(R_TMP, R_RED, R_S0, hp_ctrl_wait(BAR_LDS));
  p[n++] = hp_fadd(R_TMP, R_TMP, R_S1, hp_ctrl_safe());
  p[n++] = hp_mufu(R_TMP, R_TMP, HP_MUFU_RSQ, hp_ctrl_setbar(BAR_MUFU));

  /* The per-feature weight, from the fourth slot. */
  n += load_slot(&p[n], R_OTHER, 3, hp_ctrl_setbar(BAR_LOAD));
  p[n++] = hp_fmul(R_X, R_X, R_TMP, hp_ctrl_wait(BAR_MUFU));
  p[n++] = hp_fmul(R_X, R_X, R_OTHER, hp_ctrl_wait(BAR_LOAD));

  p[n++] = hp_imad_wide_const(R_OUT, R_TID, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(0), hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_X, 0, hp_ctrl_safe());
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}

/*
 * out = residual + x * mask * scale
 *
 * The mask arrives as a tensor rather than being generated here, and the scale
 * arrives as a scalar. That split is deliberate: the mask has to be the SAME on
 * the forward and backward pass, so whoever owns the random seed owns the mask,
 * and a kernel that generated its own would produce a different one each time
 * it ran. The scale is 1/(1-p), applied at training time so that inference
 * needs no rescaling at all.
 *
 * The multiply by the mask is unconditional rather than predicated. A mask of
 * zero multiplies to zero, which is what dropping means, and doing it
 * arithmetically costs one FMUL against a predicate plus a branch or a select.
 */
unsigned pr_emit_residual_dropout(hp_word *p) {
  unsigned n = emit_preamble(p);

  n += load_slot(&p[n], R_X, 1, hp_ctrl_setbar(BAR_LOAD));
  n += load_slot(&p[n], R_OTHER, 3, hp_ctrl_setbar(BAR_LOAD));
  p[n++] = hp_mov_const(R_S0, 0, HERMES_CBUF0_SCALAR_N(0), hp_ctrl_safe());
  p[n++] = hp_fmul(R_X, R_X, R_OTHER, hp_ctrl_wait(BAR_LOAD));
  p[n++] = hp_fmul(R_X, R_X, R_S0, hp_ctrl_safe());

  n += load_slot(&p[n], R_OTHER, 2, hp_ctrl_setbar(BAR_LOAD));
  p[n++] = hp_fadd(R_X, R_X, R_OTHER, hp_ctrl_wait(BAR_LOAD));

  p[n++] = hp_imad_wide_const(R_OUT, R_TID, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(0), hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_X, 0, hp_ctrl_safe());
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}
