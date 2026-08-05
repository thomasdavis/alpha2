/*
 * reduction.c — see reduction.h.
 */
#include "reduction.h"

enum {
  R_TID = 0,
  R_ADDR = 2, /* R2:R3 */
  R_VALUE = 4,
  R_ESIZE = 5,
  R_LHS = 6,
  R_RHS = 7,
  R_SCALAR = 8,
  R_OUT = 10,  /* R10:R11 */
  R_BLOCK = 12,
  R_GLOBAL = 13,
};

#define BAR_TID 0
#define BAR_LOAD 1
#define BAR_LDS 2
#define P_INACTIVE 0 /* set when this thread is PAST the active range */

/*
 * One step of the tree.
 *
 * Threads below `stride` add the element `stride` away to their own, and the
 * rest sit out. The predicate is expressed as "tid > stride-1" because the
 * comparison available is GT, and it is used NEGATED -- @!P0 is "tid < stride",
 * the active half. Writing it as the inactive condition and inverting is one
 * instruction cheaper than materialising the active one.
 *
 * The barrier at the end is not optional and not a formality: without it a
 * thread can read a neighbour's slot before that neighbour has written it, and
 * the result is a sum that is wrong by an amount that varies per run.
 */
static unsigned emit_step(hp_word *p, unsigned stride, unsigned active,
                          pr_combine how, unsigned tid, unsigned lhs,
                          unsigned rhs) {
  unsigned n = 0;
  p[n++] = hp_isetp_gt_imm(P_INACTIVE, tid, active - 1, hp_ctrl_safe());
  p[n++] = hp_predicated(hp_lds(lhs, tid, 0, hp_ctrl_setbar(BAR_LDS)),
                         P_INACTIVE, 1);
  p[n++] = hp_predicated(hp_lds(rhs, tid, stride * 4, hp_ctrl_setbar(BAR_LDS)),
                         P_INACTIVE, 1);
  p[n++] = hp_predicated(
      how == PR_COMBINE_MAX
          ? hp_fmnmx(lhs, lhs, rhs, 1, hp_ctrl_wait(BAR_LDS))
          : hp_fadd(lhs, lhs, rhs, hp_ctrl_wait(BAR_LDS)),
      P_INACTIVE, 1);
  p[n++] = hp_predicated(hp_sts(tid, lhs, 0, hp_ctrl_safe()), P_INACTIVE, 1);
  p[n++] = hp_bar_sync(hp_ctrl_safe());
  return n;
}

/*
 * THE TREE HAD TO BE MADE CORRECT AT WIDTHS THAT ARE NOT A POWER OF TWO, and
 * the model's own width was one of them.
 *
 * The loop used to run `stride = elements/2` and halve, which is exact only
 * while the live count stays even. At 640 it goes 320, 160, 80, 40, 20, 10 —
 * all exact — and then 5, where the count is odd. The stride-2 step after it
 * reduces slots 0..3 and NEVER TOUCHES SLOT 4, so a 640-wide layerNorm
 * normalised over 512 of its 640 features. Measured, not reasoned: the output
 * row mean came back 0.80 where a layer norm's is 0 by construction, at widths
 * 20, 40 and 640, while 8, 16, 64, 512 and 1024 were exact to 3.6e-7. Every
 * width this stack had ever been arbitrated at was a power of two — the 2L/64d
 * parity benchmark, the attention softmax at T=64 — so nothing caught it.
 *
 * This is the SECOND instance of the same defect. `pr_cross_entropy_block`
 * returned an unrounded class count into this same tree, and every vocabulary
 * that was not a power of two reduced over part of each row. Fixing it there
 * left the general fault here, one caller away.
 *
 * The fix is one FOLD before the tree: threads below `elements - pot` add the
 * tail onto the head, leaving exactly `pot` live slots for the halving loop,
 * where `pot` is the largest power of two at or below `elements`. It costs one
 * step, needs no extra shared memory, and wastes no threads.
 *
 * The fold's active count is `elements - pot`, NOT its stride. Predicating on
 * the stride the way every other step does would make threads 128..511 read
 * slots 640..1023 of a 640-slot allocation — past the end of this block's
 * shared memory, which does not fault here, it returns other rows' data. That
 * is why `active` is now a parameter separate from `stride` rather than the
 * same number spelt twice.
 *
 * For a power-of-two width `pot == elements`, the fold is skipped and every
 * step has active == stride, so the emitted program is byte-identical to the
 * old one. Widths that already worked cannot regress.
 */
unsigned pr_emit_tree(hp_word *p, unsigned elements, pr_combine how,
                      unsigned tid, unsigned lhs, unsigned rhs) {
  unsigned n = 0;
  unsigned pot = 1;
  while (pot * 2 <= elements) pot *= 2;
  if (pot != elements)
    n += emit_step(&p[n], pot, elements - pot, how, tid, lhs, rhs);
  for (unsigned stride = pot / 2; stride >= 1; stride >>= 1)
    n += emit_step(&p[n], stride, stride, how, tid, lhs, rhs);
  return n;
}

/*
 * A PARTIAL reduction: each block reduces its own slice and writes one value.
 *
 * WHY THIS EXISTS: the reduction above handles one block, because the tree runs
 * through shared memory and shared memory is per-block. That covers a row of a
 * normalisation, where the row is the block. It does not cover a whole-tensor
 * sum -- the gradient norm, the loss -- where the tensor is millions of
 * elements and no block is that large. Those need two passes, and this is the
 * first: N elements in, one value per block out, which the caller then reduces
 * again.
 *
 * IT DOES NOT SCALE. A mean over the whole tensor divides by the TOTAL count,
 * once, at the very end. Dividing per block and averaging the averages is only
 * correct when every block holds the same number of elements, and it is wrong
 * the moment the last block is short -- silently, by a few percent, in a number
 * whose job is to be compared against a threshold. So the scaling belongs to
 * the final pass and this one only ever combines.
 *
 * THE TWO DIFFERENCES from the single-block form are both in the indexing: the
 * LOAD uses the global thread index rather than the thread id, and the STORE
 * uses the block index rather than zero. Everything between is the same tree.
 */
unsigned pr_emit_reduction_partial(hp_word *p, pr_combine how,
                                   unsigned elements) {
  unsigned n = 0;
  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(BAR_TID));
  p[n++] = hp_s2r(R_BLOCK, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_TID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_safe());

  /* The global element this thread owns. */
  p[n++] = hp_imad_const(R_GLOBAL, R_BLOCK, 0, HERMES_CBUF0_NTID_X, R_TID,
                         hp_ctrl_wait(BAR_TID));
  p[n++] = hp_imad_wide_const(R_ADDR, R_GLOBAL, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
  p[n++] = hp_ldg(R_VALUE, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));
  p[n++] = hp_sts(R_TID, R_VALUE, 0, hp_ctrl_wait(BAR_LOAD));
  p[n++] = hp_bar_sync(hp_ctrl_safe());

  n += pr_emit_tree(&p[n], elements, how, R_TID, R_LHS, R_RHS);

  /* Thread 0 writes this block's answer to out[blockIdx]. */
  p[n++] = hp_isetp_gt_imm(P_INACTIVE, R_TID, 0, hp_ctrl_safe());
  p[n++] = hp_predicated(hp_lds(R_LHS, R_TID, 0, hp_ctrl_setbar(BAR_LDS)),
                         P_INACTIVE, 1);
  p[n++] = hp_imad_wide_const(R_OUT, R_BLOCK, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(0), hp_ctrl_safe());
  p[n++] = hp_predicated(hp_stg(R_OUT, R_LHS, 0, hp_ctrl_wait(BAR_LDS)),
                         P_INACTIVE, 1);
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}

unsigned pr_emit_reduction(hp_word *p, pr_red_op op, unsigned elements) {
  unsigned n = 0;

  /* Each thread loads one element and parks it in shared memory. The address
   * register for LDS/STS is an ELEMENT index because of the .X4 mode, so the
   * thread id serves directly. */
  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(BAR_TID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_safe());
  p[n++] = hp_imad_wide_const(R_ADDR, R_TID, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(1), hp_ctrl_wait(BAR_TID));
  p[n++] = hp_ldg(R_VALUE, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));
  p[n++] = hp_sts(R_TID, R_VALUE, 0, hp_ctrl_wait(BAR_LOAD));
  p[n++] = hp_bar_sync(hp_ctrl_safe());

  /* The tree, unrolled. Halving from elements/2 down to 1. */
  n += pr_emit_tree(&p[n], elements, PR_COMBINE_ADD, R_TID, R_LHS, R_RHS);

  /*
   * Thread 0 alone writes the answer.
   *
   * Every thread reaches here, so the store is predicated on "not (tid > 0)".
   * Letting all of them store would be a race writing the same address, which
   * happens to produce the right value and would hide a genuinely wrong
   * predicate -- so the guard earns its place even where it looks redundant.
   */
  p[n++] = hp_isetp_gt_imm(P_INACTIVE, R_TID, 0, hp_ctrl_safe());
  p[n++] = hp_predicated(hp_lds(R_LHS, R_TID, 0, hp_ctrl_setbar(BAR_LDS)),
                         P_INACTIVE, 1);

  if (op == PR_RED_MEAN) {
    p[n++] = hp_mov_const(R_SCALAR, 0, HERMES_CBUF0_SCALAR, hp_ctrl_safe());
    p[n++] = hp_predicated(
        hp_fmul(R_LHS, R_LHS, R_SCALAR, hp_ctrl_wait(BAR_LDS)), P_INACTIVE, 1);
  }

  p[n++] = hp_imad_wide_const(R_OUT, R_TID, R_ESIZE, 0, HERMES_CBUF0_PARAM_N(0),
                              hp_ctrl_safe());
  p[n++] = hp_predicated(
      hp_stg(R_OUT, R_LHS, 0,
             op == PR_RED_MEAN ? hp_ctrl_safe() : hp_ctrl_wait(BAR_LDS)),
      P_INACTIVE, 1);
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}
