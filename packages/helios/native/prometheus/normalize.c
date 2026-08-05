/*
 * normalize.c — see normalize.h.
 */
#include "normalize.h"
#include "reduction.h"

enum {
  R_TID = 0,
  R_ROW = 1,   /* blockIdx.x -- which row of the batch this block owns */
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
  R_MEAN = 15,
  /* R16, not R3: R2:R3 is the address PAIR, and putting the index in R3 quietly
   * overwrote the high half of every address the moment the row offset was
   * added. The kernels still ran and read from somewhere near enough to look
   * like data. */
  R_IDX = 16,

  /* The backward pass only. REGISTER_COUNT_V is 32, so these are within
   * budget -- and under-requesting raises GR_EXCEPTION rather than corrupting,
   * which is the one failure mode here that is loud. */
  R_G = 17,    /* the incoming gradient, then dxhat = g*w */
  R_W = 18,    /* this feature's weight */
  R_RSTD = 19, /* 1/sqrt(var + eps), live from reduction two to the end */
  R_M1 = 20,   /* mean(dxhat) */
  R_M2 = 21,   /* mean(dxhat * xhat) */

  /*
   * A SEPARATE ADDRESS PAIR PER MEMORY OPERATION, all even-aligned.
   *
   * The obvious way to write three loads is to compute an address into R_ADDR,
   * load, and compute the next one into R_ADDR again. That is a
   * write-after-read hazard and this stack has no interlock for it: a global
   * load holds its address registers until the memory pipe accepts them, so
   * overwriting R_ADDR can change where a load already issued reads from. It is
   * wrong only under pipe pressure, which is why it produced correct results at
   * 64 blocks and garbage at 256, differed between two runs of the same inputs,
   * and left 20 of 256 rows accidentally right.
   *
   * It could be fixed with read barriers -- hp_ctrl_setread exists for exactly
   * this -- but there are only six barriers and four are already spoken for
   * here. Registers are the cheaper resource: 32 per thread, and this kernel
   * used 22. Giving every load and store its own pair removes the hazard by
   * construction, which is a stronger guarantee than ordering it correctly.
   *
   * NOTE: residual.c's load_slot has the same pattern -- two loads through one
   * R_ADDR. It is not on the model's hot path and is not exercised at these
   * shapes, so it is flagged here rather than changed blind.
   */
  R_ADDR_G = 22,  /* R22:R23 */
  R_ADDR_W = 24,  /* R24:R25 */
  R_OUT_XH = 26,  /* R26:R27 */
};

#define BAR_TID 0
#define BAR_LOAD 1
#define BAR_LDS 2
#define BAR_MUFU 3
/* No barrier for the address-register hazard: it is removed by construction
 * instead, with a register pair per memory operation. See the enum above. */

/*
 * Load this thread's element and leave it in R_X.
 *
 * ONE BLOCK PER ROW, and the row index has to reach the address: the first
 * version indexed by thread id alone and hardcoded a grid of one, so on an
 * eight-row tensor exactly one row was normalised and the other seven kept
 * whatever the output buffer held. Every value was finite and plausible and the
 * loss was wrong by two percent -- which is small enough to look like a
 * tolerance problem and is not.
 *
 * The reduction stays block-local, which is what makes this correct: each block
 * has its own shared memory and its own barriers, so rows cannot see each
 * other's partial sums no matter how many run at once.
 */
static unsigned emit_load(hp_word *p, unsigned width) {
  unsigned n = 0;
  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(BAR_TID));
  p[n++] = hp_s2r(R_ROW, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_TID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_IDX, R_ROW, width, R_TID, hp_ctrl_wait(BAR_TID));
  p[n++] = hp_imad_wide_const(R_ADDR, R_IDX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
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
  p[n++] = hp_imad_wide_const(R_OUT, R_IDX, R_ESIZE, 0, HERMES_CBUF0_PARAM_N(0),
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
  unsigned n = emit_load(p, elements);
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
  unsigned n = emit_load(p, elements);

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

/*
 * layerNorm: (x - mean) / sqrt(var + eps).
 *
 * Two reductions, and deliberately the two-pass formulation rather than the
 * algebraic shortcut var = mean(x^2) - mean(x)^2. That identity is exact in
 * real arithmetic and catastrophic in floating point when the mean is large
 * relative to the spread: it subtracts two nearly equal numbers and keeps the
 * rounding error. Computing the deviations first costs one more pass over the
 * data and is the reason this kernel is trustworthy at all.
 */
static unsigned emit_layer(hp_word *p, unsigned elements) {
  unsigned n = emit_load(p, elements);

  /* Pass one: the mean. */
  p[n++] = hp_iadd3_imm(R_ACC, R_X, 0, hp_ctrl_wait(BAR_LOAD));
  n += emit_reduce(&p[n], elements, PR_COMBINE_ADD);
  p[n++] = hp_mov_const(R_S0, 0, HERMES_CBUF0_SCALAR, hp_ctrl_safe());
  p[n++] = hp_fmul(R_MEAN, R_RED, R_S0, hp_ctrl_wait(BAR_LDS));

  /* x becomes its deviation from the mean, which is what both the variance and
   * the final result need. */
  p[n++] = hp_fneg(R_TMP, R_MEAN, hp_ctrl_safe());
  p[n++] = hp_fadd(R_X, R_X, R_TMP, hp_ctrl_safe());

  /* Shared memory still holds pass one, so barrier before reusing it. */
  p[n++] = hp_bar_sync(hp_ctrl_safe());

  /* Pass two: the mean of the squared deviations. */
  p[n++] = hp_fmul(R_ACC, R_X, R_X, hp_ctrl_safe());
  n += emit_reduce(&p[n], elements, PR_COMBINE_ADD);
  p[n++] = hp_mov_const(R_S1, 0, HERMES_CBUF0_SCALAR2, hp_ctrl_safe());
  p[n++] = hp_fmul(R_TMP, R_RED, R_S0, hp_ctrl_wait(BAR_LDS));
  p[n++] = hp_fadd(R_TMP, R_TMP, R_S1, hp_ctrl_safe());
  p[n++] = hp_mufu(R_TMP, R_TMP, HP_MUFU_RSQ, hp_ctrl_setbar(BAR_MUFU));
  p[n++] = hp_fmul(R_X, R_X, R_TMP, hp_ctrl_wait(BAR_MUFU));
  n += emit_store(&p[n], hp_ctrl_safe());
  return n;
}

/*
 * layerNorm's backward: dx and xhat, in ONE launch.
 *
 * WHY IT IS A KERNEL AND NOT A COMPOSITION. Every arithmetic step below already
 * exists as a device operation, and building the backward out of them was tried
 * and measured and LOST -- at batch 1, 16 and 128, one backward at a time and
 * all of them together. The record is in nativeBackend.ts. The reason is that a
 * composition costs about twenty launches at 20-50 us each, and the JavaScript
 * fallback it replaces became cheap once tensors were mapped cached. What that
 * refutation leaves standing is exactly this: ONE launch instead of twenty.
 *
 * It is worth the trouble because it is the single largest item in a training
 * step -- five of these per step at batch 128, 106.88 ms of a 280 ms step, 38%,
 * all of it a host round-trip and a JavaScript loop over 262,144 elements.
 *
 * THE ARITHMETIC, per row. With xhat = (x - mean)/sigma and dxhat = g*w:
 *
 *   dx = (1/sigma) * (dxhat - mean(dxhat) - xhat*mean(dxhat*xhat))
 *
 * Four reductions -- mean(x), mean(xc^2), mean(dxhat), mean(dxhat*xhat) -- and
 * the two-pass variance for the same reason emit_layer uses it: the shortcut
 * var = mean(x^2) - mean(x)^2 subtracts two nearly equal numbers and keeps the
 * rounding error.
 *
 * WHY xhat IS AN OUTPUT. dw = sum_rows(g*xhat) needs it, and recomputing it
 * outside this kernel would cost the first two reductions over again. dw and db
 * reduce down the OTHER axis -- across rows, not within one -- which is a
 * different kernel shape entirely and stays the caller's job.
 *
 * A BARRIER BEFORE EVERY REUSE OF SHARED MEMORY. The tree's slots still hold
 * the previous pass, and a thread racing ahead would overwrite a slot another
 * thread has not finished reading. Three reuses here, so three barriers; the
 * value each protects has already been consumed into a register by the wait on
 * BAR_LDS that precedes it.
 */
static unsigned emit_layer_backward(hp_word *p, unsigned elements) {
  unsigned n = 0;

  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(BAR_TID));
  p[n++] = hp_s2r(R_ROW, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_TID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_IDX, R_ROW, elements, R_TID, hp_ctrl_wait(BAR_TID));

  /*
   * Three loads, one barrier, one wait. A scoreboard barrier counts outstanding
   * operations, so waiting once drains all three rather than only the last --
   * but each load gets its OWN address pair, for the reason in the register
   * enum above.
   *
   * The WEIGHT is indexed by R_TID and the other two by R_IDX, and that is the
   * whole difference between a per-feature parameter and a per-element tensor.
   * Indexing w by R_IDX would read past its end on every row but the first and
   * still return finite numbers.
   */
  p[n++] = hp_imad_wide_const(R_ADDR, R_IDX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
  p[n++] = hp_imad_wide_const(R_ADDR_G, R_IDX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(2), hp_ctrl_safe());
  p[n++] = hp_imad_wide_const(R_ADDR_W, R_TID, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(3), hp_ctrl_safe());
  p[n++] = hp_ldg(R_X, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));
  p[n++] = hp_ldg(R_G, R_ADDR_G, 0, hp_ctrl_setbar(BAR_LOAD));
  p[n++] = hp_ldg(R_W, R_ADDR_W, 0, hp_ctrl_setbar(BAR_LOAD));

  p[n++] = hp_mov_const(R_S0, 0, HERMES_CBUF0_SCALAR_N(0), hp_ctrl_safe());
  p[n++] = hp_mov_const(R_S1, 0, HERMES_CBUF0_SCALAR_N(1), hp_ctrl_safe());

  /* One: the mean. R_X becomes its deviation from it. */
  p[n++] = hp_iadd3_imm(R_ACC, R_X, 0, hp_ctrl_wait(BAR_LOAD));
  n += emit_reduce(&p[n], elements, PR_COMBINE_ADD);
  p[n++] = hp_fmul(R_MEAN, R_RED, R_S0, hp_ctrl_wait(BAR_LDS));
  p[n++] = hp_fneg(R_TMP, R_MEAN, hp_ctrl_safe());
  p[n++] = hp_fadd(R_X, R_X, R_TMP, hp_ctrl_safe());

  /* Two: the variance, and the reciprocal square root the hardware has. */
  p[n++] = hp_bar_sync(hp_ctrl_safe());
  p[n++] = hp_fmul(R_ACC, R_X, R_X, hp_ctrl_safe());
  n += emit_reduce(&p[n], elements, PR_COMBINE_ADD);
  p[n++] = hp_fmul(R_TMP, R_RED, R_S0, hp_ctrl_wait(BAR_LDS));
  p[n++] = hp_fadd(R_TMP, R_TMP, R_S1, hp_ctrl_safe());
  p[n++] = hp_mufu(R_RSTD, R_TMP, HP_MUFU_RSQ, hp_ctrl_setbar(BAR_MUFU));

  /* R_X becomes xhat, and is stored: the caller needs it for dw. Its own
   * address pair, so the dx store at the end cannot disturb it. */
  p[n++] = hp_fmul(R_X, R_X, R_RSTD, hp_ctrl_wait(BAR_MUFU));
  p[n++] = hp_imad_wide_const(R_OUT_XH, R_IDX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(4), hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT_XH, R_X, 0, hp_ctrl_safe());

  /* R_G becomes dxhat. */
  p[n++] = hp_fmul(R_G, R_G, R_W, hp_ctrl_safe());

  /* Three: mean(dxhat). */
  p[n++] = hp_bar_sync(hp_ctrl_safe());
  p[n++] = hp_iadd3_imm(R_ACC, R_G, 0, hp_ctrl_safe());
  n += emit_reduce(&p[n], elements, PR_COMBINE_ADD);
  p[n++] = hp_fmul(R_M1, R_RED, R_S0, hp_ctrl_wait(BAR_LDS));

  /* Four: mean(dxhat * xhat). */
  p[n++] = hp_bar_sync(hp_ctrl_safe());
  p[n++] = hp_fmul(R_ACC, R_G, R_X, hp_ctrl_safe());
  n += emit_reduce(&p[n], elements, PR_COMBINE_ADD);
  p[n++] = hp_fmul(R_M2, R_RED, R_S0, hp_ctrl_wait(BAR_LDS));

  /* dx = rstd * (dxhat - m1 - xhat*m2). */
  p[n++] = hp_fneg(R_TMP, R_M1, hp_ctrl_safe());
  p[n++] = hp_fadd(R_G, R_G, R_TMP, hp_ctrl_safe());
  p[n++] = hp_fmul(R_TMP, R_X, R_M2, hp_ctrl_safe());
  p[n++] = hp_fneg(R_TMP, R_TMP, hp_ctrl_safe());
  p[n++] = hp_fadd(R_G, R_G, R_TMP, hp_ctrl_safe());
  p[n++] = hp_fmul(R_G, R_G, R_RSTD, hp_ctrl_safe());

  p[n++] = hp_imad_wide_const(R_OUT, R_IDX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(0), hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_G, 0, hp_ctrl_safe());
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}

unsigned pr_emit_normalize(hp_word *p, pr_norm_op op, unsigned elements) {
  switch (op) {
    case PR_NORM_RMS: return emit_rms(p, elements);
    case PR_NORM_SOFTMAX: return emit_softmax(p, elements);
    case PR_NORM_LAYER: return emit_layer(p, elements);
    case PR_NORM_LAYER_BACKWARD: return emit_layer_backward(p, elements);
  }
  return 0;
}
