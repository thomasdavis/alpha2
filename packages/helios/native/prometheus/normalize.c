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
  /*
   * The AFFINE FORWARD only: this feature's weight and bias, and an address
   * pair each. A pair per memory operation and even-aligned, like everything
   * else here.
   *
   * They sit at 17-23, which the backward pass and the chunked softmax also
   * use — safely, because no kernel is ever both. What they must NOT do is run
   * past 31: REGISTER_COUNT_V is 32 and under-requesting raises GR_EXCEPTION.
   * The first draft put the bias address at R32:R33 and would have.
   */
  R_AFF_W = 17,
  R_AFF_B = 18,
  /* The masked softmax only. Shares the affine forward's window, which is safe
   * because no kernel is ever both. */
  R_MASK = 17,
  R_MIDX = 18,
  R_WRAPC = 19,
  R_MASK_ADDR = 20, /* R20:R21 — aliases R_ADDR_WA, same reason */
  R_CAPT = 22,      /* the soft cap's running temporary */
  R_CONST = 23,     /* an immediate, materialised */
  R_ADDR_WA = 20, /* R20:R21 */
  R_ADDR_BA = 22, /* R22:R23 */
  /* The chunked softmax only: a row wider than a block. */
  R_ROWBASE = 24, /* row * elements — this row's first value */
  R_CHUNK = 25,   /* which chunk of BW columns this pass is on */
  R_SUM = 26,     /* the running sum of exponentials, across chunks */
  R_MAXV = 27,    /* the row maximum, live from pass one to pass three */

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

/*
 * A block is at most 1024 threads. A feature vector fits; a VOCABULARY does not.
 *
 * Rows at or below this keep the one-element-per-thread kernel, which holds the
 * value in a register across both reductions and touches memory once. Rows past
 * it take emit_softmax_chunked.
 */
#define NORM_MAX_THREADS 1024u
#define NORM_INSTR_BYTES 16
#define P_OOR 2   /* set when this thread's column is past the row */
#define P_MASKED 4 /* the fused softmax: this position is forbidden */
/* -1e9, as bits. See emit_softmax_masked for why the value cannot matter. */
#define SOFTMAX_MASK_FILL 0xce6e6b28u
/* log2(e) and the two small constants the soft cap needs, as f32 bit patterns.
 * Immediates rather than constant-bank slots because both slots carry values
 * that VARY (the folded scale and the cap) and these never do. */
#define F32_LOG2E 0x3fb8aa3bu
#define F32_ONE   0x3f800000u
#define F32_TWO   0x40000000u
#define P_CHUNK 3 /* set when a chunk loop has run its course */

/* Threads a normalize block runs. Only softmax can cover a row wider than the
 * block; the others hold one element per thread by construction. */
unsigned pr_normalize_block(pr_norm_op op, unsigned elements) {
  if ((op == PR_NORM_SOFTMAX || op == PR_NORM_SOFTMAX_BACKWARD) &&
      elements > NORM_MAX_THREADS)
    return NORM_MAX_THREADS;
  /*
   * THE REDUCTION TREE NEEDS A POWER OF TWO, and the softmax backward is the
   * first op here to be handed a row width that is not one.
   *
   * emit_reduce halves a block repeatedly; at a width of 7 or 129 or 640 the
   * halving skips elements and the row sum is wrong — which is a plausible
   * number, not a crash. The diff test caught exactly those three widths and
   * passed 8, 64 and 1024, which is the signature.
   *
   * Rounding UP is safe because the extra threads are predicated out of both
   * the accumulate and the store, and zero is the identity for a sum. The
   * existing ops keep their behaviour: they are only ever given widths that are
   * already powers of two, and changing what they launch is not this change.
   */
  if (op == PR_NORM_SOFTMAX_BACKWARD) {
    unsigned w = 1;
    while (w < elements && w < NORM_MAX_THREADS) w *= 2;
    return w;
  }
  return elements;
}

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

  /*
   * THE WARP PATH NEVER PUTS THE CONTRIBUTION IN SHARED MEMORY AT ALL.
   *
   * The halving tree needs it there because a thread combines with a
   * NEIGHBOUR'S value; a butterfly gets the neighbour's value out of the
   * neighbour's register. So the store goes, the round trip through slot 0
   * goes, and what is left of the shared traffic is one word per warp.
   *
   * THE BARRIER STAYS, and it is not the tree's — it is the one the STS used to
   * carry. A kernel runs several reductions over the same slots (a layer norm
   * runs two, its backward four), so the cross-warp step of reduction N+1 must
   * not overwrite a slot reduction N is still reading. Dropping this because
   * "the warp path does not use shared" would be right about the contribution
   * and wrong about the partials.
   */
  if (pr_reduce_wants_warp(elements)) {
    p[n++] = hp_bar_sync(hp_ctrl_safe());
    n += pr_emit_tree_warp_reg(&p[n], elements, how, R_TID, R_ACC, R_LHS);
    /*
     * R_ACC holds the total in every thread; the callers read R_RED. An integer
     * add of zero is a bit copy, which is exactly what a move of a float
     * through an integer unit should be — and there is no register-to-register
     * MOV in this encoder.
     *
     * The callers then wait on BAR_LDS, which this path leaves clear. A wait on
     * a barrier nothing set is a no-op, so they need no change.
     */
    p[n++] = hp_iadd3_imm(R_RED, R_ACC, 0, hp_ctrl_safe());
    return n;
  }

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
 * Load this feature's weight, and its bias when there is one.
 *
 * Issued immediately after the input load and BEFORE the reductions, so two
 * dependent tree reductions stand between the load and its use. The affine is
 * then free of latency at the point it is applied, which is the whole reason
 * this is a fold rather than two more kernels.
 */
static unsigned emit_affine_load(hp_word *p, int withBias) {
  unsigned n = 0;
  p[n++] = hp_imad_wide_const(R_ADDR_WA, R_TID, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(2), hp_ctrl_safe());
  p[n++] = hp_ldg(R_AFF_W, R_ADDR_WA, 0, hp_ctrl_setbar(BAR_LOAD));
  if (withBias) {
    p[n++] = hp_imad_wide_const(R_ADDR_BA, R_TID, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM_N(3), hp_ctrl_safe());
    p[n++] = hp_ldg(R_AFF_B, R_ADDR_BA, 0, hp_ctrl_setbar(BAR_LOAD));
  }
  return n;
}

/* x = x * w (+ b). BAR_LOAD covers both loads; the normalisation that precedes
 * this has already waited on it for the input. */
static unsigned emit_affine_apply(hp_word *p, int withBias) {
  unsigned n = 0;
  p[n++] = hp_fmul(R_X, R_X, R_AFF_W, hp_ctrl_wait(BAR_LOAD));
  if (withBias) p[n++] = hp_fadd(R_X, R_X, R_AFF_B, hp_ctrl_safe());
  return n;
}

/*
 * rmsNorm: x / sqrt(mean(x^2) + eps).
 *
 * The reciprocal square root is the natural primitive here -- the hardware has
 * rsqrt directly, so normalising is a multiply rather than a divide, which is
 * why the formula is written with a reciprocal in the first place.
 */
static unsigned emit_rms(hp_word *p, unsigned elements, int affine) {
  unsigned n = emit_load(p, elements);
  if (affine) n += emit_affine_load(&p[n], 0);
  p[n++] = hp_fmul(R_ACC, R_X, R_X, hp_ctrl_wait(BAR_LOAD));
  n += emit_reduce(&p[n], elements, PR_COMBINE_ADD);

  /* mean = sum * (1/N), then + eps, then rsqrt. */
  p[n++] = hp_mov_const(R_S0, 0, HERMES_CBUF0_SCALAR, hp_ctrl_safe());
  p[n++] = hp_mov_const(R_S1, 0, HERMES_CBUF0_SCALAR2, hp_ctrl_safe());
  p[n++] = hp_fmul(R_TMP, R_RED, R_S0, hp_ctrl_wait(BAR_LDS));
  p[n++] = hp_fadd(R_TMP, R_TMP, R_S1, hp_ctrl_safe());
  p[n++] = hp_mufu(R_TMP, R_TMP, HP_MUFU_RSQ, hp_ctrl_setbar(BAR_MUFU));
  p[n++] = hp_fmul(R_X, R_X, R_TMP, hp_ctrl_wait(BAR_MUFU));
  if (affine) n += emit_affine_apply(&p[n], 0);
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
/*
 * A ROW WIDER THAN A BLOCK, which a vocabulary is.
 *
 * The kernel above puts one element in one thread's register and keeps it live
 * across both reductions. That is the right shape for a feature vector — 640
 * wide, one pass over memory — and it cannot express a softmax over 12,288
 * classes: the launch would need 12,288 threads, which is invalid, and 48 KB of
 * shared memory, which is the entire per-block budget.
 *
 * The backward of cross-entropy is exactly that softmax, so a 105M-parameter
 * model hits it on every step.
 *
 * This version keeps nothing live. Each thread walks its columns three times —
 * once for the maximum, once for the sum, once to write — and only the
 * per-thread partials go through the tree, so shared memory is BW floats
 * whatever the row is. Three passes over memory instead of one is the price,
 * and it is only paid by rows that could not run at all before.
 */
static unsigned emit_softmax_chunked(hp_word *p, unsigned elements) {
  unsigned n = 0;
  const unsigned BW = NORM_MAX_THREADS;
  const unsigned chunks = (elements + BW - 1u) / BW;
  const int guard = chunks * BW > elements;

  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(BAR_TID));
  p[n++] = hp_s2r(R_ROW, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_TID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_wait(BAR_TID));
  p[n++] = hp_imad_imm(R_ROWBASE, R_ROW, elements, HP_RZ, hp_ctrl_safe());

  /* Pass one: this thread's maximum. Chunk zero is always in range and seeds
   * the accumulator, so no negative-infinity identity is needed. */
  p[n++] = hp_iadd3_reg(R_IDX, R_ROWBASE, R_TID, hp_ctrl_safe());
  p[n++] = hp_imad_wide_const(R_ADDR, R_IDX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
  p[n++] = hp_ldg(R_ACC, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));
  p[n++] = hp_iadd3_imm(R_X, R_ACC, 0, hp_ctrl_wait(BAR_LOAD));
  if (chunks > 1) {
    p[n++] = hp_mov_imm(R_CHUNK, 1, hp_ctrl_safe());
    const unsigned top = n;
    p[n++] = hp_imad_imm(R_IDX, R_CHUNK, BW, R_TID, hp_ctrl_safe());
    if (guard)
      p[n++] = hp_isetp_gt_imm(P_OOR, R_IDX, elements - 1, hp_ctrl_safe());
    p[n++] = hp_iadd3_reg(R_IDX, R_IDX, R_ROWBASE, hp_ctrl_safe());
    p[n++] = hp_imad_wide_const(R_ADDR, R_IDX, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
    {
      /* Off, R_X keeps the previous chunk's value — a real element of this row,
       * so the maximum is unaffected. */
      hp_word ld = hp_ldg(R_X, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = guard ? hp_predicated(ld, P_OOR, 1) : ld;
    }
    p[n++] = hp_fmnmx(R_ACC, R_ACC, R_X, 1, hp_ctrl_wait(BAR_LOAD));
    p[n++] = hp_iadd3_imm(R_CHUNK, R_CHUNK, 1, hp_ctrl_safe());
    p[n++] = hp_isetp_gt_imm(P_CHUNK, R_CHUNK, chunks - 1, hp_ctrl_safe());
    const int back = -(int)((n + 1 - top) * NORM_INSTR_BYTES);
    p[n] = hp_predicated(hp_bra(back, hp_ctrl_branch()), P_CHUNK, 1);
    n++;
  }
  n += emit_reduce(&p[n], BW, PR_COMBINE_MAX);
  p[n++] = hp_iadd3_imm(R_MAXV, R_RED, 0, hp_ctrl_wait(BAR_LDS));
  p[n++] = hp_bar_sync(hp_ctrl_safe());

  /* Pass two: this thread's sum of exp(x - max). Zero is the identity, so the
   * overhang is predicated out of the accumulate. */
  p[n++] = hp_mov_const(R_S0, 0, HERMES_CBUF0_SCALAR, hp_ctrl_safe());
  p[n++] = hp_mov_imm(R_SUM, 0, hp_ctrl_safe());
  p[n++] = hp_mov_imm(R_CHUNK, 0, hp_ctrl_safe());
  {
    const unsigned top = n;
    p[n++] = hp_imad_imm(R_IDX, R_CHUNK, BW, R_TID, hp_ctrl_safe());
    if (guard)
      p[n++] = hp_isetp_gt_imm(P_OOR, R_IDX, elements - 1, hp_ctrl_safe());
    p[n++] = hp_iadd3_reg(R_IDX, R_IDX, R_ROWBASE, hp_ctrl_safe());
    p[n++] = hp_imad_wide_const(R_ADDR, R_IDX, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
    {
      hp_word ld = hp_ldg(R_X, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = guard ? hp_predicated(ld, P_OOR, 1) : ld;
    }
    p[n++] = hp_fneg(R_TMP, R_MAXV, hp_ctrl_wait(BAR_LOAD));
    p[n++] = hp_fadd(R_TMP, R_X, R_TMP, hp_ctrl_safe());
    p[n++] = hp_fmul(R_TMP, R_TMP, R_S0, hp_ctrl_safe());
    p[n++] = hp_mufu(R_TMP, R_TMP, HP_MUFU_EX2, hp_ctrl_setbar(BAR_MUFU));
    {
      hp_word acc = hp_fadd(R_SUM, R_SUM, R_TMP, hp_ctrl_wait(BAR_MUFU));
      p[n++] = guard ? hp_predicated(acc, P_OOR, 1) : acc;
    }
    p[n++] = hp_iadd3_imm(R_CHUNK, R_CHUNK, 1, hp_ctrl_safe());
    p[n++] = hp_isetp_gt_imm(P_CHUNK, R_CHUNK, chunks - 1, hp_ctrl_safe());
    const int back = -(int)((n + 1 - top) * NORM_INSTR_BYTES);
    p[n] = hp_predicated(hp_bra(back, hp_ctrl_branch()), P_CHUNK, 1);
    n++;
  }
  p[n++] = hp_iadd3_imm(R_ACC, R_SUM, 0, hp_ctrl_safe());
  n += emit_reduce(&p[n], BW, PR_COMBINE_ADD);
  p[n++] = hp_mufu(R_S1, R_RED, HP_MUFU_RCP,
                   hp_ctrl_wait_setbar(BAR_LDS, BAR_MUFU));

  /* Pass three: write exp(x - max) * (1/sum) back over the row. */
  p[n++] = hp_mov_imm(R_CHUNK, 0, hp_ctrl_wait(BAR_MUFU));
  {
    const unsigned top = n;
    p[n++] = hp_imad_imm(R_IDX, R_CHUNK, BW, R_TID, hp_ctrl_safe());
    if (guard)
      p[n++] = hp_isetp_gt_imm(P_OOR, R_IDX, elements - 1, hp_ctrl_safe());
    p[n++] = hp_iadd3_reg(R_IDX, R_IDX, R_ROWBASE, hp_ctrl_safe());
    p[n++] = hp_imad_wide_const(R_ADDR, R_IDX, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
    {
      hp_word ld = hp_ldg(R_X, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = guard ? hp_predicated(ld, P_OOR, 1) : ld;
    }
    p[n++] = hp_fneg(R_TMP, R_MAXV, hp_ctrl_wait(BAR_LOAD));
    p[n++] = hp_fadd(R_TMP, R_X, R_TMP, hp_ctrl_safe());
    p[n++] = hp_fmul(R_TMP, R_TMP, R_S0, hp_ctrl_safe());
    p[n++] = hp_mufu(R_TMP, R_TMP, HP_MUFU_EX2, hp_ctrl_setbar(BAR_MUFU));
    p[n++] = hp_fmul(R_TMP, R_TMP, R_S1, hp_ctrl_wait(BAR_MUFU));
    p[n++] = hp_imad_wide_const(R_OUT, R_IDX, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM_N(0), hp_ctrl_safe());
    {
      hp_word st = hp_stg(R_OUT, R_TMP, 0, hp_ctrl_safe());
      p[n++] = guard ? hp_predicated(st, P_OOR, 1) : st;
    }
    p[n++] = hp_iadd3_imm(R_CHUNK, R_CHUNK, 1, hp_ctrl_safe());
    p[n++] = hp_isetp_gt_imm(P_CHUNK, R_CHUNK, chunks - 1, hp_ctrl_safe());
    const int back = -(int)((n + 1 - top) * NORM_INSTR_BYTES);
    p[n] = hp_predicated(hp_bra(back, hp_ctrl_branch()), P_CHUNK, 1);
    n++;
  }
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}

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
 * softmax with attention's scale and causal mask folded in.
 *
 * The three used to be separate kernels over the same [B,H,T,T] tensor. Fusing
 * them is worth more than the arithmetic suggests because the arithmetic was
 * never the cost: each pass read 3.93 MB and wrote 3.93 MB to apply one
 * multiply, or one predicated move, per element.
 *
 * ORDER MATTERS AND IS PRESERVED: scale, then mask, then the reduction. The
 * composed path scaled before masking, so the fill lands in SCALED units, and
 * doing it the other way round would change which value the maximum picks.
 *
 * THE FILL IS A COMPILE-TIME CONSTANT and that costs nothing, because its exact
 * value cannot reach the answer. It has to be below every real score so the
 * maximum ignores it, and far enough below that exp2 of the difference
 * underflows to exactly zero — which -1e9 is, by an enormous margin, at any
 * score this model produces. Both scalar slots are spoken for (log2(e) and the
 * score scale), and inventing a third to carry a number that does not affect
 * the result would be worse than naming it here.
 */
static unsigned emit_softmax_masked(hp_word *p, unsigned elements, int cap) {
  unsigned n = emit_load(p, elements);

  if (cap) {
    /*
     * softCap with the score scale folded into its exponent constant:
     *   c*tanh(s*x/c) = c*(1 - 2/(exp2(s0*x) + 1)),  s0 = 2*log2(e)*s/c
     * The caller computes s0; here it is simply scalar 1, and c is scalar 2.
     */
    p[n++] = hp_mov_const(R_S0, 0, HERMES_CBUF0_SCALAR, hp_ctrl_safe());
    p[n++] = hp_mov_const(R_S1, 0, HERMES_CBUF0_SCALAR2, hp_ctrl_safe());
    p[n++] = hp_fmul(R_CAPT, R_X, R_S0, hp_ctrl_wait(BAR_LOAD));
    p[n++] = hp_mufu(R_CAPT, R_CAPT, HP_MUFU_EX2, hp_ctrl_setbar(BAR_MUFU));
    p[n++] = hp_mov_imm(R_CONST, F32_ONE, hp_ctrl_safe());
    p[n++] = hp_fadd(R_CAPT, R_CAPT, R_CONST, hp_ctrl_wait(BAR_MUFU));
    p[n++] = hp_mufu(R_CAPT, R_CAPT, HP_MUFU_RCP, hp_ctrl_setbar(BAR_MUFU));
    p[n++] = hp_mov_imm(R_TMP, F32_TWO, hp_ctrl_safe());
    p[n++] = hp_fmul(R_CAPT, R_CAPT, R_TMP, hp_ctrl_wait(BAR_MUFU));
    p[n++] = hp_fneg(R_CAPT, R_CAPT, hp_ctrl_safe());
    p[n++] = hp_fadd(R_CAPT, R_CAPT, R_CONST, hp_ctrl_safe());
    p[n++] = hp_fmul(R_X, R_CAPT, R_S1, hp_ctrl_safe());
  } else {
    /* Scale only, matching the composed order. */
    p[n++] = hp_mov_const(R_S1, 0, HERMES_CBUF0_SCALAR2, hp_ctrl_safe());
    p[n++] = hp_fmul(R_X, R_X, R_S1, hp_ctrl_wait(BAR_LOAD));
  }

  /*
   * The mask is [T,T] and the scores are [B,H,T,T], so the index wraps at T*T.
   * The row width IS T here — softmax runs over the last axis — so the wrap is
   * elements*elements, a power of two whenever T is, which lets it be an AND
   * rather than a division.
   */
  p[n++] = hp_mov_imm(R_WRAPC, elements * elements - 1u, hp_ctrl_safe());
  p[n++] = hp_lop3(R_MIDX, R_IDX, R_WRAPC, HP_LUT_AND, hp_ctrl_safe());
  p[n++] = hp_imad_wide_const(R_MASK_ADDR, R_MIDX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(2), hp_ctrl_safe());
  p[n++] = hp_ldg(R_MASK, R_MASK_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));
  p[n++] = hp_isetp_gt_imm(P_MASKED, R_MASK, 0, hp_ctrl_wait(BAR_LOAD));
  p[n] = hp_predicated(hp_mov_imm(R_X, SOFTMAX_MASK_FILL, hp_ctrl_safe()),
                       P_MASKED, 0);
  n++;

  /* Pass one: the maximum. */
  p[n++] = hp_iadd3_imm(R_ACC, R_X, 0, hp_ctrl_safe());
  n += emit_reduce(&p[n], elements, PR_COMBINE_MAX);

  /* log2(e) as an IMMEDIATE: both constant slots carry values that vary and
   * this one never does. */
  p[n++] = hp_mov_imm(R_S0, F32_LOG2E, hp_ctrl_safe());
  p[n++] = hp_fneg(R_TMP, R_RED, hp_ctrl_wait(BAR_LDS));
  p[n++] = hp_fadd(R_X, R_X, R_TMP, hp_ctrl_safe());
  p[n++] = hp_fmul(R_X, R_X, R_S0, hp_ctrl_safe());
  p[n++] = hp_mufu(R_X, R_X, HP_MUFU_EX2, hp_ctrl_setbar(BAR_MUFU));

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
static unsigned emit_layer(hp_word *p, unsigned elements, int affine) {
  unsigned n = emit_load(p, elements);
  if (affine) n += emit_affine_load(&p[n], 1);

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
  if (affine) n += emit_affine_apply(&p[n], 1);
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

/*
 * out[i] = s[i] * (g[i] - sum_j g[j]*s[j]) — softmax's backward in one pass.
 *
 * Structure copied from emit_softmax_chunked rather than invented: one block
 * per row, a chunk loop so a row wider than a block still works, emit_reduce
 * for the row sum, and a second chunk loop that writes. Copying a proven shape
 * is deliberate — the silent-wrong-answer bugs this file records were all in
 * the interaction between a chunk loop, a reduction and a store, and none of
 * them were visible in the output's shape or magnitude.
 *
 * The DOT PRODUCT has to be complete before any element is written, which is
 * what makes this a reduction rather than an elementwise op. Zero is the
 * identity for a sum, so the overhang is predicated out of the accumulate and
 * needs no special value.
 *
 * Param 1 is the softmax OUTPUT and param 2 the incoming gradient; param 0 is
 * the result. R_MAXV and R_S1 are unused by this op and carry g and the negated
 * dot product.
 */
static unsigned emit_softmax_backward(hp_word *p, unsigned elements) {
  unsigned n = 0;
  const unsigned BW = pr_normalize_block(PR_NORM_SOFTMAX_BACKWARD, elements);
  const unsigned chunks = (elements + BW - 1u) / BW;
  const int guard = chunks * BW > elements;

  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(BAR_TID));
  p[n++] = hp_s2r(R_ROW, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_TID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_wait(BAR_TID));
  p[n++] = hp_imad_imm(R_ROWBASE, R_ROW, elements, HP_RZ, hp_ctrl_safe());

  /* Pass one: this thread's share of sum_j g[j]*s[j]. */
  p[n++] = hp_mov_imm(R_SUM, 0, hp_ctrl_safe());
  p[n++] = hp_mov_imm(R_CHUNK, 0, hp_ctrl_safe());
  {
    const unsigned top = n;
    p[n++] = hp_imad_imm(R_IDX, R_CHUNK, BW, R_TID, hp_ctrl_safe());
    if (guard)
      p[n++] = hp_isetp_gt_imm(P_OOR, R_IDX, elements - 1, hp_ctrl_safe());
    p[n++] = hp_iadd3_reg(R_IDX, R_IDX, R_ROWBASE, hp_ctrl_safe());
    p[n++] = hp_imad_wide_const(R_ADDR, R_IDX, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
    p[n++] = hp_imad_wide_const(R_OUT, R_IDX, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM_N(2), hp_ctrl_safe());
    {
      hp_word ls = hp_ldg(R_X, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));
      hp_word lg = hp_ldg(R_MAXV, R_OUT, 0, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = guard ? hp_predicated(ls, P_OOR, 1) : ls;
      p[n++] = guard ? hp_predicated(lg, P_OOR, 1) : lg;
    }
    p[n++] = hp_fmul(R_TMP, R_X, R_MAXV, hp_ctrl_wait(BAR_LOAD));
    {
      hp_word acc = hp_fadd(R_SUM, R_SUM, R_TMP, hp_ctrl_safe());
      p[n++] = guard ? hp_predicated(acc, P_OOR, 1) : acc;
    }
    p[n++] = hp_iadd3_imm(R_CHUNK, R_CHUNK, 1, hp_ctrl_safe());
    p[n++] = hp_isetp_gt_imm(P_CHUNK, R_CHUNK, chunks - 1, hp_ctrl_safe());
    const int back = -(int)((n + 1 - top) * NORM_INSTR_BYTES);
    p[n] = hp_predicated(hp_bra(back, hp_ctrl_branch()), P_CHUNK, 1);
    n++;
  }
  p[n++] = hp_iadd3_imm(R_ACC, R_SUM, 0, hp_ctrl_safe());
  n += emit_reduce(&p[n], BW, PR_COMBINE_ADD);
  /* Negated once here rather than per element below. */
  p[n++] = hp_fneg(R_S1, R_RED, hp_ctrl_wait(BAR_LDS));
  p[n++] = hp_bar_sync(hp_ctrl_safe());

  /* Pass two: s * (g - dot), over the row. */
  p[n++] = hp_mov_imm(R_CHUNK, 0, hp_ctrl_safe());
  {
    const unsigned top = n;
    p[n++] = hp_imad_imm(R_IDX, R_CHUNK, BW, R_TID, hp_ctrl_safe());
    if (guard)
      p[n++] = hp_isetp_gt_imm(P_OOR, R_IDX, elements - 1, hp_ctrl_safe());
    p[n++] = hp_iadd3_reg(R_IDX, R_IDX, R_ROWBASE, hp_ctrl_safe());
    p[n++] = hp_imad_wide_const(R_ADDR, R_IDX, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
    p[n++] = hp_imad_wide_const(R_OUT, R_IDX, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM_N(2), hp_ctrl_safe());
    {
      hp_word ls = hp_ldg(R_X, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));
      hp_word lg = hp_ldg(R_MAXV, R_OUT, 0, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = guard ? hp_predicated(ls, P_OOR, 1) : ls;
      p[n++] = guard ? hp_predicated(lg, P_OOR, 1) : lg;
    }
    p[n++] = hp_fadd(R_TMP, R_MAXV, R_S1, hp_ctrl_wait(BAR_LOAD));
    p[n++] = hp_fmul(R_TMP, R_TMP, R_X, hp_ctrl_safe());
    p[n++] = hp_imad_wide_const(R_OUT, R_IDX, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM_N(0), hp_ctrl_safe());
    {
      hp_word st = hp_stg(R_OUT, R_TMP, 0, hp_ctrl_safe());
      p[n++] = guard ? hp_predicated(st, P_OOR, 1) : st;
    }
    p[n++] = hp_iadd3_imm(R_CHUNK, R_CHUNK, 1, hp_ctrl_safe());
    p[n++] = hp_isetp_gt_imm(P_CHUNK, R_CHUNK, chunks - 1, hp_ctrl_safe());
    const int back = -(int)((n + 1 - top) * NORM_INSTR_BYTES);
    p[n] = hp_predicated(hp_bra(back, hp_ctrl_branch()), P_CHUNK, 1);
    n++;
  }
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}

unsigned pr_emit_normalize(hp_word *p, pr_norm_op op, unsigned elements) {
  switch (op) {
    case PR_NORM_RMS: return emit_rms(p, elements, 0);
    case PR_NORM_RMS_AFFINE: return emit_rms(p, elements, 1);
    case PR_NORM_SOFTMAX_MASKED: return emit_softmax_masked(p, elements, 0);
    case PR_NORM_SOFTMAX_MASKED_CAP: return emit_softmax_masked(p, elements, 1);
    case PR_NORM_SOFTMAX:
      return elements > NORM_MAX_THREADS ? emit_softmax_chunked(p, elements)
                                         : emit_softmax(p, elements);
    case PR_NORM_LAYER: return emit_layer(p, elements, 0);
    case PR_NORM_LAYER_AFFINE: return emit_layer(p, elements, 1);
    case PR_NORM_LAYER_BACKWARD: return emit_layer_backward(p, elements);
    case PR_NORM_SOFTMAX_BACKWARD: return emit_softmax_backward(p, elements);
  }
  return 0;
}
