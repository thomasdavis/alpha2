/*
 * mask.c — kernels that choose between a value and a constant, per element.
 *
 * WHAT: the causal mask that attention needs, and the general masked fill.
 *
 * HOW THE CHOICE IS MADE: a predicated MOV over the loaded value, rather than
 * a select instruction. The value is loaded unconditionally and then
 * overwritten where the mask says so, which is one instruction and no branch.
 *
 * WHY NOT A BRANCH: the condition here is per-THREAD, not per-warp -- adjacent
 * columns fall on opposite sides of the diagonal. A branch on that diverges the
 * warp and both sides execute anyway, so it costs the same as predication and
 * adds a reconvergence problem. Predication is not an optimisation here, it is
 * the correct structure for a condition that varies within a warp.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: it does not skip the load on masked
 * elements. Reading a value that is about to be discarded looks wasteful and is
 * not -- the load is issued for the whole warp regardless, and predicating it
 * would leave a register undefined for the masked lanes, which is worse than
 * loading a number nobody reads.
 */
#include "mask.h"

enum {
  R_ROW = 0,
  R_COL = 1,
  R_INDEX = 2,
  R_ESIZE = 5,
  R_ADDR = 6,  /* R6:R7 */
  R_MASK_ADDR = 8, /* R8:R9 */
  R_VALUE = 10,
  R_MASK = 11,
  R_OUT = 14, /* R14:R15 */
};

#define BAR_ID 0
#define BAR_LOAD 1
#define P_MASKED 0

/*
 * Negative infinity, as a bit pattern.
 *
 * This is what a causal mask writes, and it has to be the true infinity rather
 * than a merely large negative number: the masked positions go straight into a
 * softmax, where exp(-inf) is exactly zero and exp(-1e30) is also zero but only
 * because it underflows. The difference shows up when the whole row is masked --
 * with true infinities the row sums to zero and produces NaN, which is honest,
 * and with large finite numbers it produces a uniform distribution over
 * positions that were supposed to be forbidden.
 */
#define NEG_INF_BITS 0xff800000u

unsigned pr_emit_causal_mask(hp_word *p, unsigned cols) {
  unsigned n = 0;
  p[n++] = hp_s2r(R_ROW, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_COL, HP_SR_TID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_INDEX, R_ROW, cols, R_COL, hp_ctrl_wait(BAR_ID));

  p[n++] = hp_imad_wide_const(R_ADDR, R_INDEX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
  p[n++] = hp_ldg(R_VALUE, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));

  /* Mask where the column is strictly beyond the row: a token may attend to
   * itself and to everything before it, and to nothing after. Getting this to
   * ">=" instead of ">" hides the diagonal, which is the single most common way
   * to get a causal mask wrong and is silent -- the model still trains, just
   * without ever seeing the current token. */
  p[n++] = hp_isetp_reg(P_MASKED, R_COL, R_ROW, HP_CMP_GT, 0, hp_ctrl_safe());
  p[n] = hp_predicated(hp_mov_imm(R_VALUE, NEG_INF_BITS, hp_ctrl_wait(BAR_LOAD)),
                       P_MASKED, 0);
  n++;

  p[n++] = hp_imad_wide_const(R_OUT, R_INDEX, R_ESIZE, 0, HERMES_CBUF0_PARAM_N(0),
                              hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_VALUE, 0, hp_ctrl_wait(BAR_LOAD));
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}

/*
 * masked fill: out[i] = mask[i] ? fill : in[i], mask in the second input.
 *
 * The mask is compared against zero rather than against one, so any non-zero
 * value counts as set. A mask produced by a comparison elsewhere may be 1, or
 * 0xffffffff, or anything the producer chose, and testing for equality with a
 * particular truth value would work until the producer changed.
 */
unsigned pr_emit_masked_fill(hp_word *p) {
  unsigned n = 0;
  p[n++] = hp_s2r(R_ROW, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_COL, HP_SR_TID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_safe());
  p[n++] = hp_imad_const(R_INDEX, R_ROW, 0, HERMES_CBUF0_NTID_X, R_COL,
                         hp_ctrl_wait(BAR_ID));

  p[n++] = hp_imad_wide_const(R_ADDR, R_INDEX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
  p[n++] = hp_ldg(R_VALUE, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));
  p[n++] = hp_imad_wide_const(R_MASK_ADDR, R_INDEX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(2), hp_ctrl_safe());
  p[n++] = hp_ldg(R_MASK, R_MASK_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));

  p[n++] = hp_isetp_gt_imm(P_MASKED, R_MASK, 0, hp_ctrl_wait(BAR_LOAD));
  p[n] = hp_predicated(
      hp_mov_const(R_VALUE, 0, HERMES_CBUF0_SCALAR, hp_ctrl_safe()), P_MASKED,
      0);
  n++;

  p[n++] = hp_imad_wide_const(R_OUT, R_INDEX, R_ESIZE, 0, HERMES_CBUF0_PARAM_N(0),
                              hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_VALUE, 0, hp_ctrl_safe());
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}
