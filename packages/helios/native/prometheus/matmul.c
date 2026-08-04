/*
 * matmul.c — C[M,N] = A[M,K] * B[K,N], one thread per output element.
 *
 * WHAT: block x is the row, thread x is the column, and each thread walks the
 * K dimension accumulating into one register. The first kernel in this stack
 * with a real loop rather than a straight line.
 *
 * WHY A LOOP AND NOT AN UNROLL: unrolling is tempting because the shape is
 * known when the code is generated, and for K = 8 it would be shorter. It stops
 * being tempting at K = 768, where the body is six thousand instructions and no
 * longer fits in an instruction cache, let alone the program buffer. The loop
 * costs four instructions of overhead per iteration and works at every K, so it
 * is what gets written first; unrolling it by a factor is an optimisation that
 * can come later and be measured.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: no shared-memory tiling, no register
 * blocking, no vectorised loads. Every thread reads both operands straight from
 * global memory, which means A's row is re-read by all N threads in the block
 * and B's column by all M blocks. That is the naive algorithm and it is slow on
 * purpose -- it is the version whose correctness can be argued from the
 * definition of matrix multiplication, and tiling is a transformation that has
 * to be checked against something.
 *
 * THE HARDWARE FACT: a branch offset is relative to the instruction AFTER the
 * branch, so the distance back to the top of the loop is measured from there.
 * The emitter computes it from recorded positions rather than from a count
 * maintained by hand, because a hand-maintained count is correct until someone
 * adds an instruction to the body.
 */
#include "matmul.h"

enum {
  R_ROW = 0,   /* blockIdx.x  -- which row of A and of C */
  R_COL = 1,   /* threadIdx.x -- which column of B and of C */
  R_K = 2,     /* the loop counter */
  R_AIDX = 3,  /* element index into A, walks by 1 */
  R_BIDX = 4,  /* element index into B, walks by N */
  R_ESIZE = 5, /* 4, the size of a float, for the address IMADs */
  R_AADDR = 6, /* R6:R7 */
  R_BADDR = 8, /* R8:R9 */
  R_AVAL = 10,
  R_BVAL = 11,
  R_ACC = 12,
  R_OIDX = 13,
  R_OUT = 14, /* R14:R15 */
};

#define BAR_ID 0   /* the two S2Rs */
#define BAR_LOAD 1 /* BOTH loads -- see the accumulate below */
#define P_DONE 0   /* set when the loop has run its course */
#define INSTR_BYTES 16

unsigned pr_emit_matmul(hp_word *p, unsigned M, unsigned N, unsigned K) {
  unsigned n = 0;
  (void)M; /* the grid supplies it; no instruction here needs the value */

  p[n++] = hp_s2r(R_ROW, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_COL, HP_SR_TID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_safe());
  p[n++] = hp_mov_imm(R_ACC, 0, hp_ctrl_safe());
  p[n++] = hp_mov_imm(R_K, 0, hp_ctrl_safe());

  /* A[row][0] is element row*K, and B[0][col] is element col. Both indices then
   * only ever need adding to, which is why they are set up outside the loop:
   * the multiply is loop-invariant and doing it inside would be paying for it K
   * times to get the same answer. */
  p[n++] = hp_imad_imm(R_AIDX, R_ROW, K, HP_RZ, hp_ctrl_wait(BAR_ID));
  p[n++] = hp_iadd3_imm(R_BIDX, R_COL, 0, hp_ctrl_safe());

  const unsigned loop_top = n;

  /* A[row][k] */
  p[n++] = hp_imad_wide_const(R_AADDR, R_AIDX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM0 + 8, hp_ctrl_safe());
  p[n++] = hp_ldg(R_AVAL, R_AADDR, 0, hp_ctrl_setbar(BAR_LOAD));

  /* B[k][col] */
  p[n++] = hp_imad_wide_const(R_BADDR, R_BIDX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM0 + 16, hp_ctrl_safe());
  p[n++] = hp_ldg(R_BVAL, R_BADDR, 0, hp_ctrl_setbar(BAR_LOAD));

  /*
   * The accumulate waits on ONE barrier covering both loads.
   *
   * The reasoning that says two is wrong, and it is seductive: a single barrier
   * is "set twice", so surely waiting on it only proves the second load landed.
   * It does not work that way. A scoreboard barrier is a COUNTER of outstanding
   * writes, not a flag -- two loads increment it twice and the wait clears when
   * both have retired. Two separate barriers with a combined wait faults the
   * channel, which is the same thing ew_layout.h already records for the pair
   * of S2Rs, and which this kernel rediscovered by ignoring it.
   */
  p[n++] = hp_ffma(R_ACC, R_AVAL, R_BVAL, R_ACC, hp_ctrl_wait(BAR_LOAD));

  /* Walk both indices and the counter. A moves along its row one element at a
   * time; B moves down its column, which is N elements per step. */
  p[n++] = hp_iadd3_imm(R_AIDX, R_AIDX, 1, hp_ctrl_safe());
  p[n++] = hp_iadd3_imm(R_BIDX, R_BIDX, N, hp_ctrl_safe());
  p[n++] = hp_iadd3_imm(R_K, R_K, 1, hp_ctrl_safe());

  /* "k > K-1" rather than "k >= K" because the comparison the hardware offers
   * is GT, and the predicate is used negated -- @!P0 is "keep going". */
  p[n++] = hp_isetp_gt_imm(P_DONE, R_K, K - 1, hp_ctrl_safe());

  /* The branch is the last instruction of the body, so the instruction after it
   * is at n+1 and the distance back is measured from there. */
  const int back = -(int)((n + 1 - loop_top) * INSTR_BYTES);
  p[n] = hp_predicated(hp_bra(back, hp_ctrl_branch()), P_DONE, 1);
  n++;

  /* C[row][col] is element row*N + col. */
  p[n++] = hp_imad_imm(R_OIDX, R_ROW, N, R_COL, hp_ctrl_safe());
  p[n++] = hp_imad_wide_const(R_OUT, R_OIDX, R_ESIZE, 0, HERMES_CBUF0_PARAM0,
                              hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_ACC, 0, hp_ctrl_safe());
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}
