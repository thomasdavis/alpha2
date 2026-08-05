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
  R_BATCH = 16,
  R_TMP = 17,
  R_LOAD_IDX = 18,
};

#define BAR_ID 0   /* the two S2Rs */
#define BAR_LOAD 1 /* BOTH loads -- see the accumulate below */
#define P_DONE 0   /* set when the loop has run its course */
#define INSTR_BYTES 16

/*
 * Whether the row of A can be staged in shared memory.
 *
 * Every thread in a block computes a different COLUMN of the same output row,
 * so all of them read the same row of A -- and in the untiled kernel each reads
 * it separately, which is N redundant global loads of every element. Staging it
 * once cuts A's traffic from M*N*K to M*K.
 *
 * Only when K divides evenly by the block width, and only when the row fits the
 * shared-memory budget. The even division is what lets the cooperative load be
 * unrolled with no predication -- K and N are both known when the code is
 * generated -- and an uneven one would need a guard per load for a case that
 * does not arise in the shapes the model uses.
 */
#define MATMUL_TILE_MAX_K 1024u
static int can_tile(unsigned N, unsigned K) {
  return N > 0 && K % N == 0 && K <= MATMUL_TILE_MAX_K &&
         (K / N) * 6u + 40u < PR_MAX_INSTRUCTIONS;
}

unsigned pr_matmul_shared_bytes(unsigned N, unsigned K) {
  return can_tile(N, K) ? K * 4u : 0u;
}

unsigned pr_emit_matmul(hp_word *p, unsigned M, unsigned N, unsigned K) {
  unsigned n = 0;

  p[n++] = hp_s2r(R_ROW, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_COL, HP_SR_TID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_BATCH, HP_SR_CTAID_Y, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_safe());
  p[n++] = hp_mov_imm(R_ACC, 0, hp_ctrl_safe());
  p[n++] = hp_mov_imm(R_K, 0, hp_ctrl_safe());

  /* A[row][0] is element row*K, and B[0][col] is element col. Both indices then
   * only ever need adding to, which is why they are set up outside the loop:
   * the multiply is loop-invariant and doing it inside would be paying for it K
   * times to get the same answer. */
  p[n++] = hp_imad_imm(R_AIDX, R_ROW, K, HP_RZ, hp_ctrl_wait(BAR_ID));
  p[n++] = hp_iadd3_imm(R_BIDX, R_COL, 0, hp_ctrl_safe());

  /*
   * BATCHED, by taking the plane from the block's Y index.
   *
   * Every operand gets its own plane stride because they differ: A is M x K,
   * B is K x N, and C is M x N. Launched with gridY = 1 the offsets are all
   * zero and this is exactly the single-matrix kernel, which is why one
   * emitter serves both and there is no second program to keep in step.
   *
   * It replaces a host loop that copied each plane in, launched, DRAINED the
   * queue and copied out -- four round trips for four attention heads, and
   * after batching went in those drains were most of what remained.
   */
  p[n++] = hp_imad_imm(R_TMP, R_BATCH, M * K, HP_RZ, hp_ctrl_safe());
  p[n++] = hp_iadd3_reg(R_AIDX, R_AIDX, R_TMP, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_TMP, R_BATCH, K * N, HP_RZ, hp_ctrl_safe());
  p[n++] = hp_iadd3_reg(R_BIDX, R_BIDX, R_TMP, hp_ctrl_safe());

  /*
   * Stage A's row in shared memory, cooperatively.
   *
   * Thread `col` loads elements col, col+N, col+2N ... which is coalesced --
   * adjacent threads touch adjacent addresses -- and unrolled, because K and N
   * are both known here. Then one barrier, and the K loop below reads A from
   * shared memory instead of re-reading it from global once per column.
   */
  const int tiled = can_tile(N, K);
  if (tiled) {
    for (unsigned t = 0; t < K / N; t++) {
      p[n++] = hp_iadd3_imm(R_TMP, R_COL, t * N, hp_ctrl_safe());
      p[n++] = hp_iadd3_reg(R_LOAD_IDX, R_AIDX, R_TMP, hp_ctrl_safe());
      p[n++] = hp_imad_wide_const(R_AADDR, R_LOAD_IDX, R_ESIZE, 0,
                                  HERMES_CBUF0_PARAM_N(1),
                                  hp_ctrl_safe());
      p[n++] = hp_ldg(R_AVAL, R_AADDR, 0, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = hp_sts(R_TMP, R_AVAL, 0, hp_ctrl_wait(BAR_LOAD));
    }
    p[n++] = hp_bar_sync(hp_ctrl_safe());
  }

  const unsigned loop_top = n;

  /* A[row][k] -- from shared memory when staged, from global when not. */
  if (tiled) {
    p[n++] = hp_lds(R_AVAL, R_K, 0, hp_ctrl_setbar(BAR_LOAD));
  } else {
    p[n++] = hp_imad_wide_const(R_AADDR, R_AIDX, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
    p[n++] = hp_ldg(R_AVAL, R_AADDR, 0, hp_ctrl_setbar(BAR_LOAD));
  }

  /* B[k][col] */
  p[n++] = hp_imad_wide_const(R_BADDR, R_BIDX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(2), hp_ctrl_safe());
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
  p[n++] = hp_imad_imm(R_TMP, R_BATCH, M * N, HP_RZ, hp_ctrl_safe());
  p[n++] = hp_iadd3_reg(R_OIDX, R_OIDX, R_TMP, hp_ctrl_safe());
  p[n++] = hp_imad_wide_const(R_OUT, R_OIDX, R_ESIZE, 0, HERMES_CBUF0_PARAM_N(0),
                              hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_ACC, 0, hp_ctrl_safe());
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}
