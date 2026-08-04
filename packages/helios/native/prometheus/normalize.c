/*
 * normalize.c — see normalize.h.
 */
#include "normalize.h"
#include "reduction.h"

enum {
  R_TID = 0,
  R_ADDR = 2, /* R2:R3 */
  R_X = 4,    /* this thread's element, live across the whole kernel */
  R_ESIZE = 5,
  R_LHS = 6,
  R_RHS = 7,
  R_ACC = 8,  /* what this thread contributes to the reduction */
  R_RED = 9,  /* the reduced value, once every thread reads it back */
  R_OUT = 10, /* R10:R11 */
  R_S0 = 12,
  R_S1 = 13,
  R_TMP = 14,
};

#define BAR_TID 0
#define BAR_LOAD 1
#define BAR_LDS 2
#define BAR_MUFU 3

/* Load this thread's element and leave it in R_X. */
static unsigned emit_load(hp_word *p) {
  unsigned n = 0;
  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(BAR_TID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_safe());
  p[n++] = hp_imad_wide_const(R_ADDR, R_TID, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM0 + 8, hp_ctrl_wait(BAR_TID));
  p[n++] = hp_ldg(R_X, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));
  return n;
}

/* Put R_ACC into shared memory, barrier, reduce, and read the result into
 * R_RED. Every thread ends up holding the same reduced value. */
static unsigned emit_reduce(hp_word *p, unsigned elements, pr_combine how) {
  unsigned n = 0;
  p[n++] = hp_sts(R_TID, R_ACC, 0, hp_ctrl_safe());
  p[n++] = hp_bar_sync(hp_ctrl_safe());
  n += pr_emit_tree(&p[n], elements, how, R_TID, R_LHS, R_RHS);
  /* Slot 0 holds the answer, and the tree's final barrier has already run, so
   * every thread may read it. */
  p[n++] = hp_lds(R_RED, HP_RZ, 0, hp_ctrl_setbar(BAR_LDS));
  return n;
}

/* Store R_X to out[tid]. */
static unsigned emit_store(hp_word *p, hp_control c) {
  unsigned n = 0;
  p[n++] = hp_imad_wide_const(R_OUT, R_TID, R_ESIZE, 0, HERMES_CBUF0_PARAM0,
                              hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_X, 0, c);
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}

/*
 * rmsNorm: x / sqrt(mean(x^2) + eps).
 *
 * The reciprocal square root is the natural primitive here -- the hardware has
 * rsqrt directly, so normalising is a multiply rather than a divide, which is
 * why the formula is written with a reciprocal in the first place.
 */
static unsigned emit_rms(hp_word *p, unsigned elements) {
  unsigned n = emit_load(p);
  p[n++] = hp_fmul(R_ACC, R_X, R_X, hp_ctrl_wait(BAR_LOAD));
  n += emit_reduce(&p[n], elements, PR_COMBINE_ADD);

  /* mean = sum * (1/N), then + eps, then rsqrt. */
  p[n++] = hp_mov_const(R_S0, 0, HERMES_CBUF0_SCALAR, hp_ctrl_safe());
  p[n++] = hp_mov_const(R_S1, 0, HERMES_CBUF0_SCALAR2, hp_ctrl_safe());
  p[n++] = hp_fmul(R_TMP, R_RED, R_S0, hp_ctrl_wait(BAR_LDS));
  p[n++] = hp_fadd(R_TMP, R_TMP, R_S1, hp_ctrl_safe());
  p[n++] = hp_mufu(R_TMP, R_TMP, HP_MUFU_RSQ, hp_ctrl_setbar(BAR_MUFU));
  p[n++] = hp_fmul(R_X, R_X, R_TMP, hp_ctrl_wait(BAR_MUFU));
  n += emit_store(&p[n], hp_ctrl_safe());
  return n;
}

/*
 * softmax: exp(x - max) / sum(exp(x - max)).
 *
 * TWO reductions, and the subtraction between them is not an optimisation. exp
 * of a large positive value overflows to infinity, and infinity divided by
 * infinity is NaN -- so the max shift is what makes the kernel correct rather
 * than merely faster. The tree is reused with a different combiner for the
 * first pass, which is the whole reason it takes one.
 */
static unsigned emit_softmax(hp_word *p, unsigned elements) {
  unsigned n = emit_load(p);

  /* Pass one: the maximum. */
  p[n++] = hp_iadd3_imm(R_ACC, R_X, 0, hp_ctrl_wait(BAR_LOAD));
  n += emit_reduce(&p[n], elements, PR_COMBINE_MAX);

  /* x = exp(x - max), via exp2 and a base conversion from the constant bank. */
  p[n++] = hp_mov_const(R_S0, 0, HERMES_CBUF0_SCALAR, hp_ctrl_safe());
  p[n++] = hp_fneg(R_TMP, R_RED, hp_ctrl_wait(BAR_LDS));
  p[n++] = hp_fadd(R_X, R_X, R_TMP, hp_ctrl_safe());
  p[n++] = hp_fmul(R_X, R_X, R_S0, hp_ctrl_safe());
  p[n++] = hp_mufu(R_X, R_X, HP_MUFU_EX2, hp_ctrl_setbar(BAR_MUFU));

  /* A barrier before reusing shared memory: the tree's slots still hold the
   * maximum pass, and a thread racing ahead would overwrite a slot another
   * thread has not finished reading. */
  p[n++] = hp_bar_sync(hp_ctrl_wait(BAR_MUFU));

  /* Pass two: the sum of those exponentials. */
  p[n++] = hp_iadd3_imm(R_ACC, R_X, 0, hp_ctrl_safe());
  n += emit_reduce(&p[n], elements, PR_COMBINE_ADD);

  p[n++] = hp_mufu(R_TMP, R_RED, HP_MUFU_RCP, hp_ctrl_wait_setbar(BAR_LDS, BAR_MUFU));
  p[n++] = hp_fmul(R_X, R_X, R_TMP, hp_ctrl_wait(BAR_MUFU));
  n += emit_store(&p[n], hp_ctrl_safe());
  return n;
}

unsigned pr_emit_normalize(hp_word *p, pr_norm_op op, unsigned elements) {
  return op == PR_NORM_RMS ? emit_rms(p, elements)
                           : emit_softmax(p, elements);
}
