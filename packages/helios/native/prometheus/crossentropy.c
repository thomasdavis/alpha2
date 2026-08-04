/*
 * crossentropy.c — the loss, computed without ever forming a softmax.
 *
 * WHAT: out[r] = -log(softmax(logits[r])[target[r]]), one block per row.
 *
 * WHY NO SOFTMAX: the definition expands to
 *
 *     -log( exp(z_t - m) / sum_j exp(z_j - m) )
 *   =  -(z_t - m) + log( sum_j exp(z_j - m) )
 *
 * so the exponentials cancel for the target term and the whole loss is a max, a
 * sum of exponentials, and one logarithm. Computing the softmax first would
 * exponentiate every class, divide every class, then take the log of one of
 * them -- throwing away all but one of the divisions and inverting the
 * exponential it just applied. It is also numerically worse: for a confident
 * wrong prediction the softmax probability underflows to zero and its log is
 * negative infinity, where this form gives a large finite loss, which is the
 * correct answer and the one that lets training recover.
 *
 * WHAT THE MAX SHIFT IS FOR: it is not an optimisation. exp of a large logit
 * overflows to infinity and the sum becomes infinity for every class at once.
 * Subtracting the row maximum makes the largest term exactly 1 and every other
 * term smaller, so the sum is between 1 and the number of classes -- always
 * representable. Adding m back at the end makes it exact rather than
 * approximate: log(sum exp(z-m)) + m is log(sum exp z), identically.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: no label smoothing, no ignore-index, no
 * class weights, and no reduction across rows. Each is a policy the caller
 * should choose, and each would be a different function wearing the same name.
 *
 * THE SHAPE CONSTRAINT: one block per row, so a row must fit in a block --
 * the two reductions are block-local. A vocabulary larger than a block needs a
 * two-level reduction, which is a different kernel.
 */
#include "crossentropy.h"

#include "reduction.h"

enum {
  R_TID = 0,
  R_ROW = 1,
  R_ADDR = 2, /* R2:R3 */
  R_Z = 4,    /* this thread's logit, live across both reductions */
  R_ESIZE = 5,
  R_LHS = 6,
  R_RHS = 7,
  R_ACC = 8,
  R_RED = 9,
  R_OUT = 10, /* R10:R11 */
  R_MAX = 12,
  R_TMP = 13,
  R_TARGET = 14,
  R_IDX = 15,
};

#define BAR_TID 0
#define BAR_LOAD 1
#define BAR_LDS 2
#define BAR_MUFU 3
#define P_NOT_TARGET 0

unsigned pr_emit_cross_entropy(hp_word *p, unsigned classes) {
  unsigned n = 0;
  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(BAR_TID));
  p[n++] = hp_s2r(R_ROW, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_TID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_wait(BAR_TID));

  /* logits[row][tid] */
  p[n++] = hp_imad_imm(R_IDX, R_ROW, classes, R_TID, hp_ctrl_safe());
  p[n++] = hp_imad_wide_const(R_ADDR, R_IDX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
  p[n++] = hp_ldg(R_Z, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));

  /* Pass one: the row maximum. */
  p[n++] = hp_iadd3_imm(R_ACC, R_Z, 0, hp_ctrl_wait(BAR_LOAD));
  p[n++] = hp_sts(R_TID, R_ACC, 0, hp_ctrl_safe());
  p[n++] = hp_bar_sync(hp_ctrl_safe());
  n += pr_emit_tree(&p[n], classes, PR_COMBINE_MAX, R_TID, R_LHS, R_RHS);
  p[n++] = hp_lds(R_MAX, HP_RZ, 0, hp_ctrl_setbar(BAR_LDS));

  /* exp(z - m), via exp2 and the base conversion in the constant bank. */
  p[n++] = hp_mov_const(R_TMP, 0, HERMES_CBUF0_SCALAR_N(0), hp_ctrl_safe());
  p[n++] = hp_fneg(R_ACC, R_MAX, hp_ctrl_wait(BAR_LDS));
  p[n++] = hp_fadd(R_ACC, R_Z, R_ACC, hp_ctrl_safe());
  p[n++] = hp_fmul(R_ACC, R_ACC, R_TMP, hp_ctrl_safe());
  p[n++] = hp_mufu(R_ACC, R_ACC, HP_MUFU_EX2, hp_ctrl_setbar(BAR_MUFU));

  /* Shared memory still holds the maximum pass, so barrier before reusing it. */
  p[n++] = hp_bar_sync(hp_ctrl_wait(BAR_MUFU));

  /* Pass two: the sum of those exponentials. */
  p[n++] = hp_sts(R_TID, R_ACC, 0, hp_ctrl_safe());
  p[n++] = hp_bar_sync(hp_ctrl_safe());
  n += pr_emit_tree(&p[n], classes, PR_COMBINE_ADD, R_TID, R_LHS, R_RHS);
  p[n++] = hp_lds(R_RED, HP_RZ, 0, hp_ctrl_setbar(BAR_LDS));

  /* log(sum) = log2(sum) * ln(2), since the hardware provides log base two. */
  p[n++] = hp_mov_const(R_TMP, 0, HERMES_CBUF0_SCALAR_N(1), hp_ctrl_safe());
  p[n++] = hp_mufu(R_RED, R_RED, HP_MUFU_LG2,
                   hp_ctrl_wait_setbar(BAR_LDS, BAR_MUFU));
  p[n++] = hp_fmul(R_RED, R_RED, R_TMP, hp_ctrl_wait(BAR_MUFU));

  /*
   * loss = log(sum exp(z - m)) + m - z_target.
   *
   * Only the thread holding the target class knows z_target, and rather than
   * broadcast it through shared memory the whole row computes the loss and only
   * that one thread stores it. One predicated store against an extra barrier
   * and a third trip through shared memory.
   */
  p[n++] = hp_fadd(R_RED, R_RED, R_MAX, hp_ctrl_safe());
  p[n++] = hp_fneg(R_TMP, R_Z, hp_ctrl_safe());
  p[n++] = hp_fadd(R_RED, R_RED, R_TMP, hp_ctrl_safe());

  /* targets[row] */
  p[n++] = hp_imad_wide_const(R_ADDR, R_ROW, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(2), hp_ctrl_safe());
  p[n++] = hp_ldg(R_TARGET, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));

  /* Store from the one thread whose class is the target. The predicate is the
   * inequality because it is used negated, as everywhere else here. */
  p[n++] = hp_isetp_reg(P_NOT_TARGET, R_TID, R_TARGET, HP_CMP_NE, 0,
                        hp_ctrl_wait(BAR_LOAD));
  p[n++] = hp_imad_wide_const(R_OUT, R_ROW, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(0), hp_ctrl_safe());
  p[n] = hp_predicated(hp_stg(R_OUT, R_RED, 0, hp_ctrl_safe()), P_NOT_TARGET, 1);
  n++;
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}
