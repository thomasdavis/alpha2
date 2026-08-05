/*
 * colsum.c — see colsum.h.
 */
#include "colsum.h"
#include "reduction.h"

enum {
  R_TID = 0,   /* thread id within the block */
  R_BLOCK = 1, /* blockIdx.x — which group of 32 columns */
  R_COL = 2,   /* this thread's global column */
  R_LANE = 3,  /* this thread's row-lane, tid >> 5 */
  R_ESIZE = 4,
  R_INDEX = 5,
  R_ACC = 6, /* the running sum, live across the whole row walk */
  R_VALUE = 7,
  R_ROW = 8,
  R_LHS = 9,
  R_RHS = 10,
  R_SLOT = 11, /* this thread's shared-memory slot */
  /*
   * A SEPARATE ADDRESS PAIR PER MEMORY OPERATION, all even-aligned — the rule
   * normalize.c states, and it is load-bearing here.
   *
   * A global load holds its address registers until the memory pipe accepts
   * them, not until it issued, and there is no write-after-read interlock. The
   * plain form has one load and waits on it before recomputing the address, so
   * a single pair would be safe; the PRODUCT form has two outstanding at once
   * and a shared pair would be the hazard this stack has hit six times. Giving
   * each its own removes it by construction rather than by an argument about
   * ordering that stops being true the next time the loop grows.
   */
  R_ADDR = 12,  /* R12:R13 */
  /* The product form only: the second operand and its own address pair. */
  R_ADDR2 = 14, /* R14:R15 */
  R_VALUE2 = 17,
  R_STOREIDX = 16, /* lane, or 1 when the column does not exist */
  R_OUT = 18,   /* R18:R19 */
};

#define BAR_ID 0
#define BAR_LOAD 1
#define BAR_LDS 3
#define P_OOR 0   /* this thread's column is past the tensor */
#define P_DONE 1  /* the row walk has run its course */
#define P_STORE 2 /* the one lane that writes the answer */

#define INSTR_BYTES 16

unsigned pr_colsum_grid(unsigned cols) {
  return (cols + PR_COLSUM_COLS - 1u) / PR_COLSUM_COLS;
}

unsigned pr_colsum_shared(void) { return PR_COLSUM_BLOCK * 4u; }

unsigned pr_emit_column_sum(hp_word *p, unsigned rows, unsigned cols,
                            int product) {
  unsigned n = 0;

  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_BLOCK, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_safe());

  /*
   * The row-lane is `tid >> 5` and the column within the block is the
   * remainder, subtracted back out with a negative multiplier — the same idiom
   * hmma.c uses to split a thread id into warp and lane, because there is no
   * mask-with-immediate here and IMAD sign-extends.
   *
   * PR_COLSUM_COLS is a warp's width on purpose. The 32 lanes of a warp then
   * hold 32 ADJACENT columns and every global read below is one coalesced
   * 128-byte transaction. A layout where a warp spanned rows instead would
   * issue 32 separate transactions for the same 32 values — the difference
   * between reading at the card's bandwidth and at a thirty-second of it. It
   * also makes every warp's row-lane UNIFORM, which is what lets the row walk
   * below be a real branch rather than predication.
   */
  p[n++] = hp_shr_imm(R_LANE, R_TID, 5, hp_ctrl_wait(BAR_ID));
  p[n++] = hp_imad_imm(R_COL, R_LANE, (uint32_t)-(int)PR_COLSUM_COLS, R_TID,
                       hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_COL, R_BLOCK, PR_COLSUM_COLS, R_COL, hp_ctrl_safe());

  /*
   * OUT-OF-RANGE COLUMNS ARE CLAMPED, NOT PREDICATED — and the choice is about
   * the loop, not about the load.
   *
   * The grid rounds up, so the last block runs threads whose column does not
   * exist. Predicating their loads off is the obvious move and it is the wrong
   * one: the loop's exit test would then have to combine "past the last row"
   * with "column does not exist", the two differ WITHIN a warp, and a divergent
   * backward branch is a reconvergence problem this emitter has no construct
   * for.
   *
   * Clamping to the last real column keeps every address in bounds and keeps
   * the branch warp-uniform, at the cost of some duplicated arithmetic in the
   * final block. The garbage it accumulates is harmless because a column's
   * shared-memory slots are only ever combined with the SAME column's other
   * lanes, and the store is suppressed below. Reading past the end would not
   * have faulted here — it would have returned whatever the pool put next to
   * the tensor, which is another live tensor.
   */
  p[n++] = hp_isetp_gt_imm(P_OOR, R_COL, cols - 1u, hp_ctrl_safe());
  p[n] = hp_predicated(hp_mov_imm(R_COL, cols - 1u, hp_ctrl_safe()), P_OOR, 0);
  n++;

  p[n++] = hp_mov_imm(R_ACC, 0, hp_ctrl_safe());
  p[n++] = hp_iadd3_imm(R_ROW, R_LANE, 0, hp_ctrl_safe());

  /*
   * ---- the row walk ----
   *
   * Lane l accumulates rows l, l+32, l+64, ... A stride equal to the lane count
   * keeps every iteration's read coalesced across the warp, because all 32
   * lanes of a warp share one row-lane and differ only in column.
   *
   * A real branch rather than an unroll: at the model's shape this runs 48
   * times, and 48 unrolled load-add pairs would be most of the 512-instruction
   * budget for a kernel that has to work at other shapes too. The guard is at
   * the TOP as well as the bottom, so a row count below the lane count — where
   * some lanes own no row at all — does no work instead of reading row `lane`.
   */
  p[n++] = hp_isetp_gt_imm(P_DONE, R_ROW, rows - 1u, hp_ctrl_safe());
  {
    /* Skip the whole loop when this lane owns no rows. Forward over the body,
     * computed once the body's length is known. */
    const unsigned skip_at = n;
    n++; /* placeholder, filled in below */
    const unsigned loop_top = n;

    p[n++] = hp_imad_imm(R_INDEX, R_ROW, cols, R_COL, hp_ctrl_safe());
    p[n++] = hp_imad_wide_const(R_ADDR, R_INDEX, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
    p[n++] = hp_ldg(R_VALUE, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));
    if (product) {
      /* Its OWN address pair. Both loads are outstanding at once and a global
       * load holds its address registers until the memory pipe accepts them,
       * not until it issued — sharing the pair is the write-after-read hazard
       * this stack has hit six times now. One barrier still covers both: the
       * scoreboard counts outstanding operations, so the wait below drains the
       * pair rather than only the last. */
      p[n++] = hp_imad_wide_const(R_ADDR2, R_INDEX, R_ESIZE, 0,
                                  HERMES_CBUF0_PARAM_N(2), hp_ctrl_safe());
      p[n++] = hp_ldg(R_VALUE2, R_ADDR2, 0, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = hp_fmul(R_VALUE, R_VALUE, R_VALUE2, hp_ctrl_wait(BAR_LOAD));
      p[n++] = hp_fadd(R_ACC, R_ACC, R_VALUE, hp_ctrl_safe());
    } else {
      p[n++] = hp_fadd(R_ACC, R_ACC, R_VALUE, hp_ctrl_wait(BAR_LOAD));
    }

    p[n++] = hp_iadd3_imm(R_ROW, R_ROW, PR_COLSUM_LANES, hp_ctrl_safe());
    p[n++] = hp_isetp_gt_imm(P_DONE, R_ROW, rows - 1u, hp_ctrl_safe());
    {
      const int back = -(int)((n + 1u - loop_top) * INSTR_BYTES);
      p[n] = hp_predicated(hp_bra(back, hp_ctrl_branch()), P_DONE, 1);
      n++;
    }
    {
      const int fwd = (int)((n - skip_at - 1u) * INSTR_BYTES);
      p[skip_at] = hp_predicated(hp_bra(fwd, hp_ctrl_branch()), P_DONE, 0);
    }
  }

  /*
   * ---- the 32 partials per column meet in shared memory ----
   *
   * Slot layout is `lane * 32 + column-within-block`, which is the thread id
   * itself, so the store needs no arithmetic. The tree then walks the LANE
   * axis, a stride of 32 slots rather than 1 — so this cannot use
   * pr_emit_tree, whose every step assumes neighbouring slots.
   *
   * Five halvings for 32 lanes. The active predicate is on the LANE and not on
   * the thread id: all 32 columns of a surviving lane stay live at every step,
   * which is the whole point of reducing 32 columns at once.
   */
  p[n++] = hp_iadd3_imm(R_SLOT, R_TID, 0, hp_ctrl_safe());
  p[n++] = hp_sts(R_SLOT, R_ACC, 0, hp_ctrl_safe());
  p[n++] = hp_bar_sync(hp_ctrl_safe());

  for (unsigned stride = PR_COLSUM_LANES / 2u; stride >= 1u; stride >>= 1) {
    p[n++] = hp_isetp_gt_imm(P_DONE, R_LANE, stride - 1u, hp_ctrl_safe());
    p[n] = hp_predicated(hp_lds(R_LHS, R_SLOT, 0, hp_ctrl_setbar(BAR_LDS)),
                         P_DONE, 1);
    n++;
    p[n] = hp_predicated(hp_lds(R_RHS, R_SLOT, stride * PR_COLSUM_COLS * 4u,
                                hp_ctrl_setbar(BAR_LDS)),
                         P_DONE, 1);
    n++;
    p[n] = hp_predicated(hp_fadd(R_LHS, R_LHS, R_RHS, hp_ctrl_wait(BAR_LDS)),
                         P_DONE, 1);
    n++;
    p[n] = hp_predicated(hp_sts(R_SLOT, R_LHS, 0, hp_ctrl_safe()), P_DONE, 1);
    n++;
    p[n++] = hp_bar_sync(hp_ctrl_safe());
  }

  /*
   * ONE PREDICATE FOR TWO CONDITIONS: lane zero, and a column that exists.
   *
   * There is no AND of two predicate registers to hand and applying
   * hp_predicated twice does not compose — the second call overwrites the first
   * one's guard rather than intersecting with it, which would silently let
   * every out-of-range thread store. So the conditions are merged in INTEGER
   * form first: a thread whose column does not exist has its lane index forced
   * to one, and the single test "lane > 0" then suppresses both.
   */
  p[n++] = hp_iadd3_imm(R_STOREIDX, R_LANE, 0, hp_ctrl_safe());
  p[n] = hp_predicated(hp_mov_imm(R_STOREIDX, 1, hp_ctrl_safe()), P_OOR, 0);
  n++;
  p[n++] = hp_isetp_gt_imm(P_STORE, R_STOREIDX, 0, hp_ctrl_safe());

  p[n] = hp_predicated(hp_lds(R_LHS, R_SLOT, 0, hp_ctrl_setbar(BAR_LDS)),
                       P_STORE, 1);
  n++;
  p[n++] = hp_imad_wide_const(R_OUT, R_COL, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(0), hp_ctrl_safe());
  p[n] = hp_predicated(hp_stg(R_OUT, R_LHS, 0, hp_ctrl_wait(BAR_LDS)),
                       P_STORE, 1);
  n++;

  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}
