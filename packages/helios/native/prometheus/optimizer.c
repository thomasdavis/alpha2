/*
 * optimizer.c — the AdamW parameter update.
 *
 * WHAT: one element of the update, given the parameter, its gradient, and the
 * two moment estimates. Four tensors, three of which are read-modify-write.
 *
 * WHY IT IS ONE KERNEL AND NOT FIVE: the update is memory-bound -- it touches
 * four tensors and does about a dozen flops per element -- so splitting it into
 * "update m", "update v", "compute the step", "apply weight decay" would read
 * and write the same memory four times over to save nothing. Fusing it is not
 * an optimisation here so much as the only sensible shape.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: it does not maintain the step counter or
 * compute the bias correction. Those are the same for every element in the
 * tensor, so computing them per-thread would be the same arithmetic repeated
 * millions of times to reach a number the host already knows. They arrive as
 * scalars in the constant bank, already corrected.
 *
 * WHAT THE HOST HAS ALREADY FOLDED, and why it is worth saying: the update is
 *
 *     m <- b1*m + (1-b1)*g
 *     v <- b2*v + (1-b2)*g^2
 *     p <- p - lr * ( mhat / (sqrt(vhat) + eps) + wd*p )
 *
 * with mhat = m/(1-b1^t) and vhat = v/(1-b2^t). The bias corrections are
 * powers of the step count, identical across the tensor, so the host folds
 * them into the learning rate and the epsilon before the launch. What arrives
 * is four numbers: b1, b2, an effective learning rate, and an effective
 * epsilon. That is the [[feedback]] rule about folding at the host rather than
 * emitting arithmetic the GPU repeats per element, applied to the one kernel
 * where it matters most.
 */
#include "optimizer.h"

enum {
  R_INDEX = 0,
  R_TID = 1,
  R_ESIZE = 5,
  R_ADDR = 6,  /* R6:R7, reused for each tensor in turn */
  R_PARAM = 8,
  R_GRAD = 9,
  R_M = 10,
  R_V = 11,
  R_B1 = 12,
  R_B2 = 13,
  R_LR = 14,
  R_EPS = 15,
  R_TMP = 16,
  R_DENOM = 17,
  R_OUT = 18, /* R18:R19 */
  /* The f16 weight shadow: pack this element and its neighbour, even lane stores. */
  R_NEIGHBOR = 20,
  R_PACKED = 21,
  R_HALF = 22,
  R_ONE = 23,
  R_LOWBIT = 24,
};

#define BAR_ID 0
#define BAR_LOAD 1
#define BAR_MUFU 2
#define P_SHADOW 1 /* set on odd lanes — guards the packed shadow store */

/* Load one tensor's element into `dst` from parameter slot `slot`. */
static unsigned load_at(hp_word *p, unsigned dst, unsigned slot) {
  unsigned n = 0;
  p[n++] = hp_imad_wide_const(R_ADDR, R_INDEX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(slot), hp_ctrl_safe());
  p[n++] = hp_ldg(dst, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));
  return n;
}

unsigned pr_emit_adamw(hp_word *p, int with_shadow) {
  unsigned n = 0;
  p[n++] = hp_s2r(R_INDEX, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_safe());
  p[n++] = hp_imad_const(R_INDEX, R_INDEX, 0, HERMES_CBUF0_NTID_X, R_TID,
                         hp_ctrl_wait(BAR_ID));

  /* Slot 0 is the parameter, which this kernel both reads and writes -- the
   * update is in place, as it has to be for a tensor of any size. */
  n += load_at(&p[n], R_PARAM, 0);
  n += load_at(&p[n], R_GRAD, 1);
  n += load_at(&p[n], R_M, 2);
  n += load_at(&p[n], R_V, 3);

  p[n++] = hp_mov_const(R_B1, 0, HERMES_CBUF0_SCALAR_N(0), hp_ctrl_safe());
  p[n++] = hp_mov_const(R_B2, 0, HERMES_CBUF0_SCALAR_N(1), hp_ctrl_safe());
  p[n++] = hp_mov_const(R_LR, 0, HERMES_CBUF0_SCALAR_N(2), hp_ctrl_safe());
  p[n++] = hp_mov_const(R_EPS, 0, HERMES_CBUF0_SCALAR_N(3), hp_ctrl_safe());

  /*
   * m <- m + (1-b1)*(g - m), which is b1*m + (1-b1)*g rearranged.
   *
   * The rearrangement is not cosmetic. Written directly it is two multiplies
   * and an add and needs (1-b1) as a separate constant; written this way it is
   * a subtract and one fused multiply-add, and the same b1 serves both moments
   * after one subtraction from 1. Fewer constants crossing the bank and fewer
   * instructions per element, for identical arithmetic.
   *
   * The single wait covers all four loads: the scoreboard barrier counts
   * outstanding writes, so one barrier tracking four producers clears when the
   * last of them retires.
   */
  p[n++] = hp_fneg(R_TMP, R_M, hp_ctrl_wait(BAR_LOAD));
  p[n++] = hp_fadd(R_TMP, R_GRAD, R_TMP, hp_ctrl_safe()); /* g - m */
  p[n++] = hp_ffma(R_M, R_TMP, R_B1, R_M, hp_ctrl_safe());

  /* v <- v + (1-b2)*(g*g - v), the same rearrangement. */
  p[n++] = hp_fmul(R_TMP, R_GRAD, R_GRAD, hp_ctrl_safe());
  p[n++] = hp_fneg(R_DENOM, R_V, hp_ctrl_safe());
  p[n++] = hp_fadd(R_TMP, R_TMP, R_DENOM, hp_ctrl_safe()); /* g^2 - v */
  p[n++] = hp_ffma(R_V, R_TMP, R_B2, R_V, hp_ctrl_safe());

  /*
   * The step: p <- p - lr * m / (sqrt(v) + eps).
   *
   * sqrt then reciprocal, two MUFU operations, rather than one rsqrt. rsqrt
   * would compute 1/sqrt(v), and the epsilon has to be added to sqrt(v) BEFORE
   * the reciprocal -- that is what stops the denominator reaching zero, which
   * is the entire reason epsilon is in the formula. Folding it as
   * 1/sqrt(v+eps) is a different and weaker guard: it bounds the step by
   * 1/sqrt(eps) instead of by 1/eps, and it is not what the algorithm says.
   */
  p[n++] = hp_mufu(R_TMP, R_V, HP_MUFU_SQRT, hp_ctrl_setbar(BAR_MUFU));
  p[n++] = hp_fadd(R_TMP, R_TMP, R_EPS, hp_ctrl_wait(BAR_MUFU));
  p[n++] = hp_mufu(R_DENOM, R_TMP, HP_MUFU_RCP, hp_ctrl_setbar(BAR_MUFU));
  p[n++] = hp_fmul(R_TMP, R_M, R_DENOM, hp_ctrl_wait(BAR_MUFU));

  /* Decoupled weight decay: applied to the parameter, NOT folded into the
   * gradient. Folding it into g would put it through the moment estimates and
   * make the decay depend on the gradient history, which is precisely the
   * behaviour AdamW exists to remove. */
  p[n++] = hp_mov_const(R_B1, 0, HERMES_CBUF0_SCALAR_N(4), hp_ctrl_safe());
  p[n++] = hp_ffma(R_TMP, R_PARAM, R_B1, R_TMP, hp_ctrl_safe());
  p[n++] = hp_fneg(R_TMP, R_TMP, hp_ctrl_safe());
  p[n++] = hp_ffma(R_PARAM, R_TMP, R_LR, R_PARAM, hp_ctrl_safe());

  /* All three updated tensors are written back. */
  p[n++] = hp_imad_wide_const(R_OUT, R_INDEX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(2), hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_M, 0, hp_ctrl_safe());
  p[n++] = hp_imad_wide_const(R_OUT, R_INDEX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(3), hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_V, 0, hp_ctrl_safe());
  p[n++] = hp_imad_wide_const(R_OUT, R_INDEX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(0), hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_PARAM, 0, hp_ctrl_safe());

  /*
   * A FREE f16 shadow of the updated weight (slot 4), so a forward GEMM can
   * stage the weight straight through cp.async without a separate cast pass —
   * the cast is the byproduct of a store the step was going to do anyway.
   *
   * Each even lane packs its element and its neighbour's into one 32-bit word
   * of two f16 (F2FP, in the low-then-high order cast.c measured) and stores it
   * at half the index. The neighbour arrives by a down-shuffle every lane must
   * execute; the odd lane's own store is predicated off. Weights are even-sized,
   * so every pair is covered.
   */
  if (with_shadow) {
    /* SHFL has latency and sets a write barrier the pack must wait on — reading
     * R_NEIGHBOR straight away gets a stale register (reduction.c does the same
     * dance; the vendor's own SHFL control decodes to write-barrier 0). */
    p[n++] = hp_shfl(HP_SHFL_DOWN, R_NEIGHBOR, R_PARAM, 1, hp_shfl_segment(32),
                     hp_ctrl_setbar(BAR_MUFU));
    p[n++] = hp_f2fp_pack(R_PACKED, R_NEIGHBOR, R_PARAM, hp_ctrl_wait(BAR_MUFU));
    p[n++] = hp_shr_imm(R_HALF, R_INDEX, 1, hp_ctrl_safe());
    p[n++] = hp_mov_imm(R_ONE, 1, hp_ctrl_safe());
    p[n++] = hp_lop3(R_LOWBIT, R_INDEX, R_ONE, 0xC0u, hp_ctrl_safe()); /* index & 1 */
    p[n++] = hp_isetp_gt_imm(P_SHADOW, R_LOWBIT, 0, hp_ctrl_safe());   /* odd lane */
    p[n++] = hp_imad_wide_const(R_OUT, R_HALF, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM_N(4), hp_ctrl_safe());
    p[n] = hp_predicated(hp_stg(R_OUT, R_PACKED, 0, hp_ctrl_safe()), P_SHADOW, 1);
    n++;
  }

  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}
