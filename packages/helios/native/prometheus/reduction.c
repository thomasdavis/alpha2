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
  R_OUT = 10, /* R10:R11 */
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
static unsigned emit_step(hp_word *p, unsigned stride, pr_combine how,
                          unsigned tid, unsigned lhs, unsigned rhs) {
  unsigned n = 0;
  p[n++] = hp_isetp_gt_imm(P_INACTIVE, tid, stride - 1, hp_ctrl_safe());
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

unsigned pr_emit_tree(hp_word *p, unsigned elements, pr_combine how,
                      unsigned tid, unsigned lhs, unsigned rhs) {
  unsigned n = 0;
  for (unsigned stride = elements / 2; stride >= 1; stride >>= 1)
    n += emit_step(&p[n], stride, how, tid, lhs, rhs);
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
                              HERMES_CBUF0_PARAM0 + 8, hp_ctrl_wait(BAR_TID));
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

  p[n++] = hp_imad_wide_const(R_OUT, R_TID, R_ESIZE, 0, HERMES_CBUF0_PARAM0,
                              hp_ctrl_safe());
  p[n++] = hp_predicated(
      hp_stg(R_OUT, R_LHS, 0,
             op == PR_RED_MEAN ? hp_ctrl_safe() : hp_ctrl_wait(BAR_LDS)),
      P_INACTIVE, 1);
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}
