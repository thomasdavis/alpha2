/*
 * elementwise.c — see elementwise.h.
 */
#include "elementwise.h"

/* Registers, named so the sequence below reads as intent rather than numbers. */
enum {
  R_INDEX = 0,
  R_IN_ADDR = 2,  /* R2:R3 */
  R_TID = 3,      /* consumed by the index IMAD, then reused as address high */
  R_VALUE = 4,
  R_ESIZE = 5,
  R_OUT_ADDR = 6, /* R6:R7 */
  R_RESULT = 8,
};

/* Bytes per element. Everything here is 32-bit. */
#define ELEMENT_BYTES 4

/* Scoreboard barriers. Both S2Rs share barrier 0 on purpose: the scoreboard
 * counts outstanding writes, so one barrier tracking two producers is exactly
 * what it is for, and ptxas does the same. Splitting them across two barriers
 * with a combined wait was tried and faults. */
#define BAR_INDEX 0
#define BAR_LOAD 1
#define BAR_MUFU 2

/* Does the operation read in[i]? PR_EW_INDEX does not, and emitting a load it
 * never uses would make it a worse probe. */
static int reads_input(pr_ew_op op) { return op != PR_EW_INDEX; }

/* The operation itself: R_RESULT = f(R_VALUE, R_INDEX). Every case here waits
 * on the load barrier when it consumes the loaded value. */
static unsigned emit_op(hp_word *p, pr_ew_op op) {
  const hp_control after_load = hp_ctrl_wait(BAR_LOAD);
  switch (op) {
    case PR_EW_COPY:
      p[0] = hp_iadd3_imm(R_RESULT, R_VALUE, 0, after_load);
      return 1;
    case PR_EW_ADD_INDEX:
      p[0] = hp_iadd3_reg(R_RESULT, R_VALUE, R_INDEX, after_load);
      return 1;
    case PR_EW_ADD_CONST:
      p[0] = hp_iadd3_imm(R_RESULT, R_VALUE, 0x1234, after_load);
      return 1;
    case PR_EW_FADD:
      p[0] = hp_fadd(R_RESULT, R_VALUE, R_VALUE, after_load);
      return 1;
    case PR_EW_FMUL:
      p[0] = hp_fmul(R_RESULT, R_VALUE, R_VALUE, after_load);
      return 1;
    case PR_EW_FFMA:
      p[0] = hp_ffma(R_RESULT, R_VALUE, R_VALUE, R_VALUE, after_load);
      return 1;
    case PR_EW_FNEG:
      p[0] = hp_fneg(R_RESULT, R_VALUE, after_load);
      return 1;
    /*
     * MUFU both waits and sets: it consumes the loaded value, so it must wait
     * for the load, and its own result is variable-latency, so the store must
     * wait for it. Setting a barrier without waiting on the load reads a stale
     * register -- and MUFU.EX2 of a stale zero returns 1.0, which looks like a
     * plausible answer rather than a bug.
     */
    case PR_EW_EXP2:
      p[0] = hp_mufu(R_RESULT, R_VALUE, HP_MUFU_EX2,
                       hp_ctrl_wait_setbar(BAR_LOAD, BAR_MUFU));
      return 1;
    case PR_EW_LOG2:
      p[0] = hp_mufu(R_RESULT, R_VALUE, HP_MUFU_LG2,
                       hp_ctrl_wait_setbar(BAR_LOAD, BAR_MUFU));
      return 1;
    case PR_EW_RCP:
      p[0] = hp_mufu(R_RESULT, R_VALUE, HP_MUFU_RCP,
                       hp_ctrl_wait_setbar(BAR_LOAD, BAR_MUFU));
      return 1;
    case PR_EW_RSQ:
      p[0] = hp_mufu(R_RESULT, R_VALUE, HP_MUFU_RSQ,
                       hp_ctrl_wait_setbar(BAR_LOAD, BAR_MUFU));
      return 1;
    /* RZ reads as +0.0 in a float operand, so max(x, RZ) is relu with no
     * constant to materialise. */
    case PR_EW_RELU:
      p[0] = hp_fmnmx(R_RESULT, R_VALUE, HP_RZ, 1, after_load);
      return 1;
    case PR_EW_INDEX:
      p[0] = hp_iadd3_imm(R_RESULT, R_INDEX, 0, hp_ctrl_safe());
      return 1;
    case PR_EW_COUNT:
      break;
  }
  return 0;
}

/* Does the operation leave its result under a scoreboard barrier? */
static int op_sets_barrier(pr_ew_op op) {
  return op == PR_EW_EXP2 || op == PR_EW_LOG2 || op == PR_EW_RCP ||
         op == PR_EW_RSQ;
}

unsigned pr_emit_elementwise(hp_word *p, pr_ew_op op) {
  unsigned n = 0;

  /* index = ctaid.x * ntid.x + tid.x, with ntid read from the constant bank in
   * CUDA's layout. */
  p[n++] = hp_s2r(R_INDEX, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_INDEX));
  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(BAR_INDEX));
  p[n++] = hp_mov_imm(R_ESIZE, ELEMENT_BYTES, hp_ctrl_safe());
  p[n++] = hp_imad_const(R_INDEX, R_INDEX, 0, HERMES_CBUF0_NTID_X, R_TID,
                         hp_ctrl_wait(BAR_INDEX));

  /* Addresses: base + index * 4, widened to 64 bits. */
  p[n++] = hp_imad_wide_const(R_OUT_ADDR, R_INDEX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM0, hp_ctrl_safe());

  if (reads_input(op)) {
    p[n++] = hp_imad_wide_const(R_IN_ADDR, R_INDEX, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM0 + 8, hp_ctrl_safe());
    p[n++] = hp_ldg(R_VALUE, R_IN_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));
  }

  n += emit_op(&p[n], op);

  p[n++] = hp_stg(R_OUT_ADDR, R_RESULT, 0,
                  op_sets_barrier(op) ? hp_ctrl_wait(BAR_MUFU) : hp_ctrl_safe());
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}
