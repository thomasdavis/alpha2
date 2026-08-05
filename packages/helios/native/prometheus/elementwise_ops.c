/*
 * elementwise_ops.c — what each element-wise operation computes.
 *
 * WHAT: one case per operation, each producing R_RESULT from R_VALUE (this
 * thread's input), R_INDEX (its position) and up to four constants. The
 * surrounding kernel -- addresses, loads, store, exit -- is elementwise.c's job
 * and is identical for every operation here.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: no addressing, no loads, no stores. A case
 * that needed one of those would be an operation that is not element-wise.
 *
 * THE HARDWARE FACT: MUFU is variable-latency, so every case that uses it sets
 * a scoreboard barrier and the consumer waits on it. Setting one without
 * waiting reads the register before the unit has written it, and MUFU.EX2 of a
 * stale zero returns 1.0 -- a plausible number, which is the worst kind.
 */
#include "ew_layout.h"

const unsigned SCALAR_REG[HERMES_CBUF0_SCALAR_COUNT] = {
    R_SCALAR, R_SCALAR2, R_SCALAR3, R_SCALAR4, R_SCALAR5, R_SCALAR6};

/*
 * How many constants from the constant bank the operation reads.
 *
 * ONE number per operation, and that is a correctness property rather than
 * tidiness. This was three membership tests -- reads_scalar, reads_scalar2,
 * reads_scalar34 -- and gelu was in the second and third and absent from the
 * first, so its leading constant was never loaded and the register held
 * whatever the previous kernel had left there. A wrong answer, no fault, no
 * obvious cause. A count cannot disagree with itself.
 *
 * Every operation is listed and there is deliberately NO default: a new one
 * fails to compile under -Werror=switch rather than silently reading nothing,
 * which is the same failure in a new coat.
 */
unsigned pr_ew_scalars_read(pr_ew_op op) {
  switch (op) {
    case PR_EW_GELU_GRAD:
      return 5;
    case PR_EW_GELU:
    case PR_EW_SOFTCAP:
      return 4;
    case PR_EW_CLAMP:
    case PR_EW_SILU:
      return 2;
    case PR_EW_SCALE:
    case PR_EW_EXP:
    case PR_EW_LOG:
    case PR_EW_FILL:
      return 1;
    case PR_EW_COPY:
    case PR_EW_ADD_INDEX:
    case PR_EW_ADD_CONST:
    case PR_EW_FADD:
    case PR_EW_FMUL:
    case PR_EW_FFMA:
    case PR_EW_FNEG:
    case PR_EW_EXP2:
    case PR_EW_LOG2:
    case PR_EW_RCP:
    case PR_EW_RSQ:
    case PR_EW_RELU:
    case PR_EW_INDEX:
    case PR_EW_ADD:
    case PR_EW_MUL:
    case PR_EW_SUB:
    case PR_EW_DIV:
    case PR_EW_ADD_INPLACE:
    case PR_EW_SQRT:
    case PR_EW_COUNT:
      return 0;
  }
  return 0;
}

/* The operation itself: R_RESULT = f(R_VALUE, R_INDEX). Every case here waits
 * on the load barrier when it consumes the loaded value. */
unsigned pr_ew_emit_op(hp_word *p, pr_ew_op op) {
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

    case PR_EW_FILL:
      p[0] = hp_iadd3_imm(R_RESULT, R_SCALAR, 0, hp_ctrl_safe());
      return 1;

    /* clamp(a, lo, hi) = min(max(a, lo), hi). Two FMNMX, opposite senses. */
    case PR_EW_CLAMP:
      p[0] = hp_fmnmx(R_TEMP, R_VALUE, R_SCALAR, 1, after_load);
      p[1] = hp_fmnmx(R_RESULT, R_TEMP, R_SCALAR2, 0, hp_ctrl_safe());
      return 2;

    case PR_EW_ADD_INPLACE:
      p[0] = hp_fadd(R_RESULT, R_VALUE, R_B_VALUE, both_loads);
      return 1;

    /*
     * silu(x) = x / (1 + exp(-x)), built from what the hardware has:
     *   t  = -x * log2(e)      negate, then scale into exp2's base
     *   t  = exp2(t)           == exp(-x)
     *   t  = t + 1
     *   t  = 1/t               == sigmoid(x)
     *   r  = x * t
     * Five instructions for one activation, which is what it costs when the
     * only transcendental is exp2 and the only divide is a reciprocal.
     */
    /*
     * gelu(x) = x * (1 - 1/(exp2(s1*(x + s0*x^3)) + 1))
     *
     * FFMA does the cubic term in one instruction: s0*x^3 + x. The rest is the
     * same shape as silu, which is not a coincidence -- both are x times a
     * sigmoid of something, and both cost a reciprocal.
     */
    case PR_EW_GELU:
      p[0] = hp_fmul(R_TEMP, R_VALUE, R_VALUE, after_load);
      p[1] = hp_fmul(R_TEMP, R_TEMP, R_VALUE, hp_ctrl_safe());
      p[2] = hp_ffma(R_TEMP, R_TEMP, R_SCALAR, R_VALUE, hp_ctrl_safe());
      p[3] = hp_fmul(R_TEMP, R_TEMP, R_SCALAR2, hp_ctrl_safe());
      p[4] = hp_mufu(R_TEMP, R_TEMP, HP_MUFU_EX2, hp_ctrl_setbar(BAR_MUFU));
      p[5] = hp_fadd(R_TEMP, R_TEMP, R_SCALAR3, hp_ctrl_wait(BAR_MUFU));
      p[6] = hp_mufu(R_TEMP2, R_TEMP, HP_MUFU_RCP, hp_ctrl_setbar(BAR_MUFU));
      p[7] = hp_fneg(R_TEMP3, R_TEMP2, hp_ctrl_wait(BAR_MUFU));
      p[8] = hp_fadd(R_TEMP3, R_TEMP3, R_SCALAR3, hp_ctrl_safe());
      p[9] = hp_fmul(R_RESULT, R_VALUE, R_TEMP3, hp_ctrl_safe());
      return 10;

    /*
     * geluBackward: g * d/dx [x * sigma(2u)],  u = K0(x + K1 x^3)
     *
     * The forward leaves sigma(2u) in a register on its way to x*sigma(2u), and
     * the derivative reuses exactly that:
     *
     *   d/dx = s + x * s(1-s) * 2K0 * (1 + 3K1 x^2)
     *
     * BINARY, because it needs both the pre-activation x and the incoming
     * gradient g -- which is why it is here rather than as a unary with a
     * scalar. It replaces a JavaScript loop over the whole tensor behind a
     * drain, worth ~33 ms a step at batch 128.
     *
     * Five scalars: K1, the folded exp2 argument, 1, 3K1, 2K0. Four were mapped
     * before this; see ew_layout.h.
     */
    case PR_EW_GELU_GRAD:
      p[0] = hp_fmul(R_TEMP, R_VALUE, R_VALUE, both_loads);
      p[1] = hp_fmul(R_TEMP2, R_TEMP, R_VALUE, hp_ctrl_safe());
      p[2] = hp_ffma(R_TEMP2, R_TEMP2, R_SCALAR, R_VALUE, hp_ctrl_safe());
      p[3] = hp_fmul(R_TEMP2, R_TEMP2, R_SCALAR2, hp_ctrl_safe());
      p[4] = hp_mufu(R_TEMP2, R_TEMP2, HP_MUFU_EX2, hp_ctrl_setbar(BAR_MUFU));
      p[5] = hp_fadd(R_TEMP2, R_TEMP2, R_SCALAR3, hp_ctrl_wait(BAR_MUFU));
      p[6] = hp_mufu(R_TEMP3, R_TEMP2, HP_MUFU_RCP, hp_ctrl_setbar(BAR_MUFU));
      p[7] = hp_fneg(R_TEMP3, R_TEMP3, hp_ctrl_wait(BAR_MUFU));
      p[8] = hp_fadd(R_TEMP3, R_TEMP3, R_SCALAR3, hp_ctrl_safe());
      /* s(1-s), reusing TEMP2 now the exponential is spent */
      p[9] = hp_fneg(R_TEMP2, R_TEMP3, hp_ctrl_safe());
      p[10] = hp_fadd(R_TEMP2, R_TEMP2, R_SCALAR3, hp_ctrl_safe());
      p[11] = hp_fmul(R_TEMP2, R_TEMP2, R_TEMP3, hp_ctrl_safe());
      /* TEMP still holds x^2 from p[0] */
      p[12] = hp_ffma(R_TEMP, R_TEMP, R_SCALAR4, R_SCALAR3, hp_ctrl_safe());
      p[13] = hp_fmul(R_TEMP, R_TEMP, R_TEMP2, hp_ctrl_safe());
      p[14] = hp_fmul(R_TEMP, R_TEMP, R_VALUE, hp_ctrl_safe());
      p[15] = hp_ffma(R_TEMP, R_TEMP, R_SCALAR5, R_TEMP3, hp_ctrl_safe());
      p[16] = hp_fmul(R_RESULT, R_TEMP, R_B_VALUE, hp_ctrl_safe());
      return 17;

    /*
     * softCap(x) = c * tanh(x/c) = c * (1 - 2/(exp2(s0*x) + 1))
     * with s0 = 2*log2(e)/c, s1 = 1, s2 = c, s3 = 2.
     */
    case PR_EW_SOFTCAP:
      p[0] = hp_fmul(R_TEMP, R_VALUE, R_SCALAR, after_load);
      p[1] = hp_mufu(R_TEMP, R_TEMP, HP_MUFU_EX2, hp_ctrl_setbar(BAR_MUFU));
      p[2] = hp_fadd(R_TEMP, R_TEMP, R_SCALAR2, hp_ctrl_wait(BAR_MUFU));
      p[3] = hp_mufu(R_TEMP2, R_TEMP, HP_MUFU_RCP, hp_ctrl_setbar(BAR_MUFU));
      p[4] = hp_fmul(R_TEMP2, R_TEMP2, R_SCALAR4, hp_ctrl_wait(BAR_MUFU));
      p[5] = hp_fneg(R_TEMP3, R_TEMP2, hp_ctrl_safe());
      p[6] = hp_fadd(R_TEMP3, R_TEMP3, R_SCALAR2, hp_ctrl_safe());
      p[7] = hp_fmul(R_RESULT, R_TEMP3, R_SCALAR3, hp_ctrl_safe());
      return 8;

    case PR_EW_SILU:
      p[0] = hp_fneg(R_TEMP, R_VALUE, after_load);
      p[1] = hp_fmul(R_TEMP, R_TEMP, R_SCALAR, hp_ctrl_safe());
      p[2] = hp_mufu(R_TEMP, R_TEMP, HP_MUFU_EX2, hp_ctrl_setbar(BAR_MUFU));
      p[3] = hp_fadd(R_TEMP, R_TEMP, R_SCALAR2, hp_ctrl_wait(BAR_MUFU));
      p[4] = hp_mufu(R_TEMP2, R_TEMP, HP_MUFU_RCP, hp_ctrl_setbar(BAR_MUFU));
      p[5] = hp_fmul(R_RESULT, R_VALUE, R_TEMP2, hp_ctrl_wait(BAR_MUFU));
      return 6;

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
int pr_ew_sets_barrier(pr_ew_op op) {
  return op == PR_EW_EXP2 || op == PR_EW_LOG2 || op == PR_EW_RCP ||
         op == PR_EW_RSQ || op == PR_EW_EXP || op == PR_EW_SQRT;
}

