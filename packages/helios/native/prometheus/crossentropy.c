/*
 * crossentropy.c — the loss, computed without ever forming a softmax.
 *
 * WHAT: out[r] = -log(softmax(logits[r])[target[r]]), one block per row.
 *
 * WHY NO SOFTMAX: the definition expands to
 *
 *     -log( exp(z_t - m) / sum_j exp(z_j - m) )
 *   =  -(z_t - m) + log( sum_j exp(z_j - m) )
 *
 * so the exponentials cancel for the target term and the whole loss is a max, a
 * sum of exponentials, and one logarithm. Computing the softmax first would
 * exponentiate every class, divide every class, then take the log of one of
 * them -- throwing away all but one of the divisions and inverting the
 * exponential it just applied. It is also numerically worse: for a confident
 * wrong prediction the softmax probability underflows to zero and its log is
 * negative infinity, where this form gives a large finite loss, which is the
 * correct answer and the one that lets training recover.
 *
 * WHAT THE MAX SHIFT IS FOR: it is not an optimisation. exp of a large logit
 * overflows to infinity and the sum becomes infinity for every class at once.
 * Subtracting the row maximum makes the largest term exactly 1 and every other
 * term smaller, so the sum is between 1 and the number of classes -- always
 * representable. Adding m back at the end makes it exact rather than
 * approximate: log(sum exp(z-m)) + m is log(sum exp z), identically.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: no label smoothing, no ignore-index, no
 * class weights, and no reduction across rows. Each is a policy the caller
 * should choose, and each would be a different function wearing the same name.
 *
 * THE SHAPE CONSTRAINT: one block per row, so a row must fit in a block --
 * the two reductions are block-local. A vocabulary larger than a block needs a
 * two-level reduction, which is a different kernel.
 */
#include "crossentropy.h"

#include "reduction.h"

enum {
  R_TID = 0,
  R_ROW = 1,
  R_ADDR = 2, /* R2:R3 */
  R_Z = 4,    /* this thread's logit, live across both reductions */
  R_ESIZE = 5,
  R_LHS = 6,
  R_RHS = 7,
  R_ACC = 8,
  R_RED = 9,
  R_OUT = 10, /* R10:R11 */
  R_MAX = 12,
  R_TMP = 13,
  R_TARGET = 14,
  R_IDX = 15,
  /* The class-chunk loop, so the vocabulary is not capped at 1024. */
  R_ROWBASE = 16, /* row * classes — this row's first logit */
  R_CHUNK = 17,   /* which chunk of BW classes this pass is on */
  R_SUM = 18,     /* the running sum of exponentials, across chunks */
};

#define BAR_TID 0
#define BAR_LOAD 1
#define BAR_LDS 2
#define BAR_MUFU 3
#define P_NOT_TARGET 0
#define P_OOR 1    /* set when this thread's class is past the vocabulary */
#define P_CHUNK 2  /* set when the chunk loop has run its course */
#define INSTR_BYTES 16

/*
 * A block is at most 1024 threads, and `classes` is the VOCABULARY.
 *
 * One thread per class is the natural way to write a row softmax and it capped
 * the vocabulary at 1024. A 105M-parameter model has 12,288 classes, which is
 * an invalid launch — and it would also have asked for 48 KB of shared memory,
 * the entire per-block budget, to hold a row it only ever reduces.
 *
 * Threads now cover the row in chunks of the block width: each accumulates its
 * own maximum and its own sum over the classes it owns, and only those
 * per-thread values go through the shared-memory tree. Shared memory drops to
 * BW floats — 4 KB — whatever the vocabulary is.
 */
#define CE_MAX_THREADS 1024u

unsigned pr_cross_entropy_block(unsigned classes) {
  return classes < CE_MAX_THREADS ? classes : CE_MAX_THREADS;
}

unsigned pr_emit_cross_entropy(hp_word *p, unsigned classes) {
  unsigned n = 0;
  const unsigned BW = pr_cross_entropy_block(classes);
  const unsigned chunks = BW ? (classes + BW - 1u) / BW : 1u;
  const int guard = chunks * BW > classes;

  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(BAR_TID));
  p[n++] = hp_s2r(R_ROW, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_TID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_wait(BAR_TID));
  p[n++] = hp_imad_imm(R_ROWBASE, R_ROW, classes, HP_RZ, hp_ctrl_safe());

  /*
   * PASS ONE: this thread's maximum over the classes it owns.
   *
   * Chunk zero always exists and is always in range -- BW <= classes -- so it
   * seeds the accumulator and the loop below only has to handle the rest. That
   * removes the need for a negative-infinity identity, which the encoder has no
   * clean way to materialise.
   */
  p[n++] = hp_iadd3_reg(R_IDX, R_ROWBASE, R_TID, hp_ctrl_safe());
  p[n++] = hp_imad_wide_const(R_ADDR, R_IDX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
  p[n++] = hp_ldg(R_ACC, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));
  p[n++] = hp_iadd3_imm(R_Z, R_ACC, 0, hp_ctrl_wait(BAR_LOAD));

  if (chunks > 1) {
    p[n++] = hp_mov_imm(R_CHUNK, 1, hp_ctrl_safe());
    const unsigned top = n;
    p[n++] = hp_imad_imm(R_IDX, R_CHUNK, BW, R_TID, hp_ctrl_safe());
    if (guard)
      p[n++] = hp_isetp_gt_imm(P_OOR, R_IDX, classes - 1, hp_ctrl_safe());
    p[n++] = hp_iadd3_reg(R_IDX, R_IDX, R_ROWBASE, hp_ctrl_safe());
    p[n++] = hp_imad_wide_const(R_ADDR, R_IDX, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
    {
      /* Predicated off, R_Z keeps the previous chunk's value -- a real logit
       * from this row, so the maximum is unchanged. That is why the load needs
       * no identity and the sum below does. */
      hp_word ld = hp_ldg(R_Z, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = guard ? hp_predicated(ld, P_OOR, 1) : ld;
    }
    p[n++] = hp_fmnmx(R_ACC, R_ACC, R_Z, 1, hp_ctrl_wait(BAR_LOAD));
    p[n++] = hp_iadd3_imm(R_CHUNK, R_CHUNK, 1, hp_ctrl_safe());
    p[n++] = hp_isetp_gt_imm(P_CHUNK, R_CHUNK, chunks - 1, hp_ctrl_safe());
    const int back = -(int)((n + 1 - top) * INSTR_BYTES);
    p[n] = hp_predicated(hp_bra(back, hp_ctrl_branch()), P_CHUNK, 1);
    n++;
  }

  /* Only the per-thread maxima go through shared memory: BW floats, not the
   * whole row. */
  p[n++] = hp_sts(R_TID, R_ACC, 0, hp_ctrl_safe());
  p[n++] = hp_bar_sync(hp_ctrl_safe());
  n += pr_emit_tree(&p[n], BW, PR_COMBINE_MAX, R_TID, R_LHS, R_RHS);
  p[n++] = hp_lds(R_MAX, HP_RZ, 0, hp_ctrl_setbar(BAR_LDS));
  p[n++] = hp_bar_sync(hp_ctrl_wait(BAR_LDS));

  /* PASS TWO: this thread's sum of exp(z - m) over the same classes. Zero IS
   * the identity here, so the out-of-range threads are predicated out of the
   * accumulate rather than left holding a stale value. */
  p[n++] = hp_mov_imm(R_SUM, 0, hp_ctrl_safe());
  p[n++] = hp_mov_const(R_TMP, 0, HERMES_CBUF0_SCALAR_N(0), hp_ctrl_safe());
  p[n++] = hp_mov_imm(R_CHUNK, 0, hp_ctrl_safe());
  {
    const unsigned top = n;
    p[n++] = hp_imad_imm(R_IDX, R_CHUNK, BW, R_TID, hp_ctrl_safe());
    if (guard)
      p[n++] = hp_isetp_gt_imm(P_OOR, R_IDX, classes - 1, hp_ctrl_safe());
    p[n++] = hp_iadd3_reg(R_IDX, R_IDX, R_ROWBASE, hp_ctrl_safe());
    p[n++] = hp_imad_wide_const(R_ADDR, R_IDX, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
    {
      hp_word ld = hp_ldg(R_Z, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = guard ? hp_predicated(ld, P_OOR, 1) : ld;
    }
    p[n++] = hp_fneg(R_RED, R_MAX, hp_ctrl_wait(BAR_LOAD));
    p[n++] = hp_fadd(R_RED, R_Z, R_RED, hp_ctrl_safe());
    p[n++] = hp_fmul(R_RED, R_RED, R_TMP, hp_ctrl_safe());
    p[n++] = hp_mufu(R_RED, R_RED, HP_MUFU_EX2, hp_ctrl_setbar(BAR_MUFU));
    {
      hp_word acc = hp_fadd(R_SUM, R_SUM, R_RED, hp_ctrl_wait(BAR_MUFU));
      p[n++] = guard ? hp_predicated(acc, P_OOR, 1) : acc;
    }
    if (chunks > 1) {
      p[n++] = hp_iadd3_imm(R_CHUNK, R_CHUNK, 1, hp_ctrl_safe());
      p[n++] = hp_isetp_gt_imm(P_CHUNK, R_CHUNK, chunks - 1, hp_ctrl_safe());
      const int back = -(int)((n + 1 - top) * INSTR_BYTES);
      p[n] = hp_predicated(hp_bra(back, hp_ctrl_branch()), P_CHUNK, 1);
      n++;
    }
  }

  p[n++] = hp_sts(R_TID, R_SUM, 0, hp_ctrl_safe());
  p[n++] = hp_bar_sync(hp_ctrl_safe());
  n += pr_emit_tree(&p[n], BW, PR_COMBINE_ADD, R_TID, R_LHS, R_RHS);
  p[n++] = hp_lds(R_RED, HP_RZ, 0, hp_ctrl_setbar(BAR_LDS));

  /* log(sum) = log2(sum) * ln(2), since the hardware provides log base two. */
  p[n++] = hp_mov_const(R_TMP, 0, HERMES_CBUF0_SCALAR_N(1), hp_ctrl_safe());
  p[n++] = hp_mufu(R_RED, R_RED, HP_MUFU_LG2,
                   hp_ctrl_wait_setbar(BAR_LDS, BAR_MUFU));
  p[n++] = hp_fmul(R_RED, R_RED, R_TMP, hp_ctrl_wait(BAR_MUFU));
  p[n++] = hp_fadd(R_RED, R_RED, R_MAX, hp_ctrl_safe());

  /*
   * z_target is FETCHED rather than held.
   *
   * It used to be that the one thread whose class was the target already had it
   * in a register, and the whole row computed the loss so that thread could
   * store it. With chunking that thread is (target % BW) on chunk (target / BW)
   * and its logit is long overwritten, so the row loads targets[row] and then
   * logits[row][target] directly. One extra dependent load, and the predicate
   * becomes "am I thread zero" instead of an index comparison.
   */
  p[n++] = hp_imad_wide_const(R_ADDR, R_ROW, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(2), hp_ctrl_safe());
  p[n++] = hp_ldg(R_TARGET, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));
  p[n++] = hp_iadd3_reg(R_IDX, R_ROWBASE, R_TARGET, hp_ctrl_wait(BAR_LOAD));
  p[n++] = hp_imad_wide_const(R_ADDR, R_IDX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
  p[n++] = hp_ldg(R_Z, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));
  p[n++] = hp_fneg(R_TMP, R_Z, hp_ctrl_wait(BAR_LOAD));
  p[n++] = hp_fadd(R_RED, R_RED, R_TMP, hp_ctrl_safe());

  p[n++] = hp_isetp_gt_imm(P_NOT_TARGET, R_TID, 0, hp_ctrl_safe());
  p[n++] = hp_imad_wide_const(R_OUT, R_ROW, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(0), hp_ctrl_safe());
  p[n] = hp_predicated(hp_stg(R_OUT, R_RED, 0, hp_ctrl_safe()), P_NOT_TARGET, 1);
  n++;
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}
