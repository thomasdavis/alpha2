/*
 * loop.c — the smallest kernel that contains a backward branch.
 *
 * WHAT: out[i] = in[i] * TRIPS, computed by adding in[i] to an accumulator
 * TRIPS times. Arithmetically pointless -- a single FMUL does it -- and that is
 * the point: the answer is trivially checkable, so anything wrong is wrong in
 * the CONTROL FLOW and nowhere else.
 *
 * WHY IT EXISTS: matmul faulted the channel on its first hardware run, and its
 * disassembly was correct instruction for instruction. That leaves two
 * suspects, the loop and the two-operand addressing, and no way to tell them
 * apart from one failing kernel. This one has a loop and the same single-input
 * addressing every passing kernel already uses, so it separates them: if it
 * passes, branches work and matmul's problem is in its indexing.
 *
 * A probe that isolates one variable is worth more than a theory about two.
 */
#include "loop.h"

enum {
  R_INDEX = 0,
  R_TID = 1,
  R_K = 2,
  R_ESIZE = 5,
  R_ADDR = 6, /* R6:R7 */
  R_VALUE = 10,
  R_ACC = 12,
  R_OUT = 14, /* R14:R15 */
};

#define BAR_ID 0
#define BAR_LOAD 1
#define P_DONE 0
#define INSTR_BYTES 16

/*
 * The same kernel with the loop replaced by a single forward branch to the very
 * next instruction: out[i] = in[i], with a BRA that must be a no-op.
 *
 * This splits the remaining suspects one more time. The loop probe faults with
 * MMU_ERR_FLT, which means the program counter is going somewhere it should not
 * -- but "the BRA instruction is malformed" and "the backward offset is wrong"
 * both produce exactly that. A forward branch of zero distance is the same
 * instruction with an offset that cannot be wrong: if this faults, the encoding
 * is the problem; if it passes, the distance is.
 */
unsigned pr_emit_branch_nop(hp_word *p, pr_branch_mode mode) {
  unsigned n = 0;
  p[n++] = hp_s2r(R_INDEX, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_safe());
  p[n++] = hp_imad_const(R_INDEX, R_INDEX, 0, HERMES_CBUF0_NTID_X, R_TID,
                         hp_ctrl_wait(BAR_ID));
  p[n++] = hp_imad_wide_const(R_ADDR, R_INDEX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM0 + 8, hp_ctrl_safe());
  p[n++] = hp_ldg(R_VALUE, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));
  /*
   * Predicated or not, the branch goes to the instruction after it, so the
   * kernel's answer is the same either way and any difference in OUTCOME is
   * caused by the predication alone.
   *
   * This exists because the first version of this probe was unpredicated and
   * forward, while both failing kernels were predicated and backward. It passed
   * and I read that as "the encoding is fine, the distance is wrong" -- which
   * varied two things and concluded about one. P0 is set false first so the
   * branch is genuinely taken rather than skipped.
   */
  if (mode == PR_BRANCH_SKIP) {
    /*
     * A forward branch that SKIPS an instruction, which the zero-distance probe
     * does not do. That one passed, and reading it as "branches work" was too
     * generous: a jump to the very next instruction is indistinguishable from
     * falling through, so the hardware may never compute a target at all. This
     * one has to.
     *
     * The skipped instruction doubles the value. So the outcome separates three
     * cases rather than two: the right answer means the branch was taken, twice
     * the answer means it was not, and a fault means the jump itself is the
     * problem.
     */
    p[n++] = hp_mov_imm(R_K, 0, hp_ctrl_safe());
    p[n++] = hp_isetp_gt_imm(P_DONE, R_K, 7, hp_ctrl_safe()); /* false */
    p[n] = hp_predicated(hp_bra(INSTR_BYTES, hp_ctrl_branch()), P_DONE, 1);
    n++;
    p[n++] = hp_fadd(R_VALUE, R_VALUE, R_VALUE, hp_ctrl_wait(BAR_LOAD));
  } else if (mode == PR_BRANCH_PREDICATED) {
    p[n++] = hp_mov_imm(R_K, 0, hp_ctrl_safe());
    p[n++] = hp_isetp_gt_imm(P_DONE, R_K, 7, hp_ctrl_safe()); /* 0 > 7: false */
    p[n] = hp_predicated(hp_bra(0, hp_ctrl_branch()), P_DONE, 1); /* @!P0 */
    n++;
  } else {
    p[n++] = hp_bra(0, hp_ctrl_branch());
  }
  p[n++] = hp_imad_wide_const(R_OUT, R_INDEX, R_ESIZE, 0, HERMES_CBUF0_PARAM0,
                              hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_VALUE, 0, hp_ctrl_wait(BAR_LOAD));
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}

unsigned pr_emit_loop_scale(hp_word *p, unsigned trips) {
  unsigned n = 0;

  p[n++] = hp_s2r(R_INDEX, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_safe());
  p[n++] = hp_mov_imm(R_ACC, 0, hp_ctrl_safe());
  p[n++] = hp_mov_imm(R_K, 0, hp_ctrl_safe());

  /* The global index, exactly as every element-wise kernel computes it. */
  p[n++] = hp_imad_const(R_INDEX, R_INDEX, 0, HERMES_CBUF0_NTID_X, R_TID,
                         hp_ctrl_wait(BAR_ID));
  p[n++] = hp_imad_wide_const(R_ADDR, R_INDEX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM0 + 8, hp_ctrl_safe());
  p[n++] = hp_ldg(R_VALUE, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));

  /* The load is hoisted out of the loop deliberately: this kernel is about the
   * branch, and a load inside would put a second suspect back in. */
  const unsigned loop_top = n;
  p[n++] = hp_fadd(R_ACC, R_ACC, R_VALUE, hp_ctrl_wait(BAR_LOAD));
  p[n++] = hp_iadd3_imm(R_K, R_K, 1, hp_ctrl_safe());
  p[n++] = hp_isetp_gt_imm(P_DONE, R_K, trips - 1, hp_ctrl_safe());
  const int back = -(int)((n + 1 - loop_top) * INSTR_BYTES);
  p[n] = hp_predicated(hp_bra(back, hp_ctrl_branch()), P_DONE, 1);
  n++;

  p[n++] = hp_imad_wide_const(R_OUT, R_INDEX, R_ESIZE, 0, HERMES_CBUF0_PARAM0,
                              hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_ACC, 0, hp_ctrl_safe());
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}
