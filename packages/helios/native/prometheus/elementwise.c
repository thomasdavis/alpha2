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
  R_B_ADDR = 10,  /* R10:R11 — the second input, for binary operations */
  R_B_VALUE = 12,
  R_SCALAR = 13,
  R_TEMP = 14,
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
#define BAR_LOAD_B 3

/* Does the operation read in[i]? PR_EW_INDEX does not, and emitting a load it
 * never uses would make it a worse probe. */
static int reads_input(pr_ew_op op) { return op != PR_EW_INDEX; }

/* Does the operation read a SECOND input array? */
static int reads_b(pr_ew_op op) {
  return op == PR_EW_ADD || op == PR_EW_SUB || op == PR_EW_MUL ||
         op == PR_EW_DIV;
}

/* Does it read the scalar from the constant bank? */
static int reads_scalar(pr_ew_op op) {
  return op == PR_EW_SCALE || op == PR_EW_EXP || op == PR_EW_LOG;
}

/* The operation itself: R_RESULT = f(R_VALUE, R_INDEX). Every case here waits
 * on the load barrier when it consumes the loaded value. */
static unsigned emit_op(hp_word *p, pr_ew_op op) {
  const hp_control after_load = hp_ctrl_wait(BAR_LOAD);
  const hp_control both_loads =
      hp_ctrl_waitmask((1u << BAR_LOAD) | (1u << BAR_LOAD_B));
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

    /* Binary. Both loads are outstanding, so wait on both barriers. */
    case PR_EW_ADD:
      p[0] = hp_fadd(R_RESULT, R_VALUE, R_B_VALUE, both_loads);
      return 1;
    case PR_EW_MUL:
      p[0] = hp_fmul(R_RESULT, R_VALUE, R_B_VALUE, both_loads);
      return 1;
    /* Subtraction has no opcode: negate then add. Two instructions is the
     * honest cost, and cheaper than pretending FADD has a negate modifier we
     * have not verified. */
    case PR_EW_SUB:
      p[0] = hp_fneg(R_TEMP, R_B_VALUE, both_loads);
      p[1] = hp_fadd(R_RESULT, R_VALUE, R_TEMP, hp_ctrl_safe());
      return 2;
    /* Division likewise: the hardware offers a reciprocal, not a divide. */
    case PR_EW_DIV:
      p[0] = hp_mufu(R_TEMP, R_B_VALUE, HP_MUFU_RCP,
                     hp_ctrl_wait_setbar(BAR_LOAD_B, BAR_MUFU));
      p[1] = hp_fmul(R_RESULT, R_VALUE, R_TEMP, hp_ctrl_wait(BAR_MUFU));
      return 2;

    case PR_EW_SCALE:
      p[0] = hp_fmul(R_RESULT, R_VALUE, R_SCALAR, after_load);
      return 1;

    /* exp(x) = exp2(x * log2 e), with the constant coming from the bank rather
     * than being materialised as an immediate — which is how a real kernel
     * receives a scalar, and avoids an FMUL-immediate encoding we have not
     * verified. */
    case PR_EW_EXP:
      p[0] = hp_fmul(R_TEMP, R_VALUE, R_SCALAR, after_load);
      p[1] = hp_mufu(R_RESULT, R_TEMP, HP_MUFU_EX2, hp_ctrl_setbar(BAR_MUFU));
      return 2;
    case PR_EW_LOG:
      p[0] = hp_mufu(R_TEMP, R_VALUE, HP_MUFU_LG2,
                     hp_ctrl_wait_setbar(BAR_LOAD, BAR_MUFU));
      p[1] = hp_fmul(R_RESULT, R_TEMP, R_SCALAR, hp_ctrl_wait(BAR_MUFU));
      return 2;
    /* sqrt(x) = 1 / rsqrt(x). Two dependent MUFUs, so the second waits on the
     * first and the store waits on the second. */
    case PR_EW_SQRT:
      p[0] = hp_mufu(R_TEMP, R_VALUE, HP_MUFU_RSQ,
                     hp_ctrl_wait_setbar(BAR_LOAD, BAR_MUFU));
      p[1] = hp_mufu(R_RESULT, R_TEMP, HP_MUFU_RCP,
                     hp_ctrl_wait_setbar(BAR_MUFU, BAR_MUFU));
      return 2;
    case PR_EW_COUNT:
      break;
  }
  return 0;
}

/* Does the operation leave its result under a scoreboard barrier? */
/* Does the result land under a scoreboard barrier the store must wait on? True
 * whenever the LAST instruction of the operation is a MUFU. */
static int op_sets_barrier(pr_ew_op op) {
  return op == PR_EW_EXP2 || op == PR_EW_LOG2 || op == PR_EW_RCP ||
         op == PR_EW_RSQ || op == PR_EW_EXP || op == PR_EW_SQRT;
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

  if (reads_b(op)) {
    p[n++] = hp_imad_wide_const(R_B_ADDR, R_INDEX, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM0 + 16, hp_ctrl_safe());
    p[n++] = hp_ldg(R_B_VALUE, R_B_ADDR, 0, hp_ctrl_setbar(BAR_LOAD_B));
  }

  if (reads_scalar(op))
    p[n++] = hp_mov_const(R_SCALAR, 0, HERMES_CBUF0_SCALAR, hp_ctrl_safe());

  n += emit_op(&p[n], op);

  p[n++] = hp_stg(R_OUT_ADDR, R_RESULT, 0,
                  op_sets_barrier(op) ? hp_ctrl_wait(BAR_MUFU) : hp_ctrl_safe());
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}
