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

  /*
   * TWO SETS OF STAGING REGISTERS, ALTERNATED BY ROUND.
   *
   * The cooperative stage is unrolled, and every round used to compute its
   * index into R_TMP, its address into R_AADDR and its value into R_AVAL. Round
   * t+1 therefore overwrote registers round t's STS may not have read yet: a
   * shared-memory store holds its operands until the memory pipe accepts them,
   * and this stack has no interlock for write-after-read.
   *
   * It survived because only one shape in the model ever staged more than once
   * -- the MLP down-projection, K=256 over N=64, four rounds -- and system
   * memory's latencies happened to hide it. Video memory's do not:
   *
   *     [1024,256]x[256,64]   sysmem  6.20e-6 against cpu_ref   ok
   *     [1024,256]x[256,64]   vidmem  1.98e+1                   garbage
   *
   * while the one-round shapes (qkv K=64 N=192, MLP up K=64 N=256) are correct
   * under both, which is exactly the split a multi-round hazard predicts.
   *
   * Alternating two sets means round t+1 touches nothing round t is still
   * using, and round t+2 is separated from round t by the intervening round's
   * issue. Removing the hazard by construction rather than ordering it with
   * barriers is the same choice normalize.c made for the same reason, and the
   * address registers are even-aligned because they are pairs.
   */
  R_STG_IDX_A = 19,
  R_STG_ADDR_A = 20, /* R20:R21 */
  R_STG_VAL_A = 22,
  /* The chunk loop, added when N stopped being bounded by the block width. */
  R_TID = 27,   /* threadIdx.x, kept because R_COL now moves */
  R_CHUNK = 28, /* which chunk of BW columns this pass is computing */
  R_AROW = 29,  /* A's row start incl. batch plane — invariant across chunks */
  R_BPLANE = 30, /* B's batch plane offset — likewise */
  /*
   * The value the STORE reads, copied out of the accumulator first.
   *
   * Chunk t ends by storing and chunk t+1 begins by zeroing the accumulator,
   * and nothing interlocks those: the store holds its operands until the pipe
   * takes them, so the reset can land first and the store writes ZERO. Copying
   * to a register the next chunk does not touch until its OWN store — a whole K
   * loop later — puts the reuse out of reach.
   */
  R_STOREVAL = 31,
  R_STG_IDX_B = 23,
  R_STG_ADDR_B = 24, /* R24:R25 */
  R_STG_VAL_B = 26,
};

#define BAR_ID 0   /* the two S2Rs */
#define BAR_LOAD 1 /* BOTH loads -- see the accumulate below */
#define P_DONE 0   /* set when the loop has run its course */
#define P_TILE 1   /* set when a thread is past the end of a short stage round */
#define P_COL 2    /* clear when this thread's column is inside N */
#define P_CHUNK 3  /* set when the chunk loop has run its course */
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

/* How many cooperative rounds it takes N threads to stage K elements. The last
 * one is short whenever N does not divide K, and it is PREDICATED. */
/*
 * THE BLOCK IS AT MOST 1024 THREADS, whatever N is.
 *
 * One thread per output column was a clean design and it made N un-runnable
 * past the hardware's limit. A block now runs min(N, 1024) threads and each
 * thread walks its column in strides of that width, so the launch geometry is
 * the same for N = 64 and N = 12288 and the only thing that grows is how many
 * times the inner loop runs.
 */
#define MATMUL_MAX_THREADS 1024u

unsigned pr_matmul_block(unsigned N) {
  return N < MATMUL_MAX_THREADS ? N : MATMUL_MAX_THREADS;
}

/* How many passes of BW columns cover N. */
static unsigned col_chunks(unsigned N) {
  const unsigned bw = pr_matmul_block(N);
  return bw ? (N + bw - 1u) / bw : 1u;
}

/* How many cooperative rounds it takes BW threads to stage K elements. The last
 * one is short whenever BW does not divide K, and it is PREDICATED. */
static unsigned tile_rounds(unsigned N, unsigned K) {
  const unsigned bw = pr_matmul_block(N);
  return (K + bw - 1u) / bw;
}

/*
 * K % N == 0 WAS THE CONDITION. Lifting it is CORRECT and it is NOT FASTER.
 *
 * The even division let the cooperative load be unrolled with no predication,
 * and the original note here said an uneven one "would need a guard per load
 * for a case that does not arise in the shapes the model uses". It does arise,
 * in the two biggest: the qkv projection is K=64 N=192 and the MLP
 * up-projection is K=64 N=256, so neither ever staged anything, while the MLP
 * down-projection at K=256 N=64 did -- and that one is twice as fast per flop.
 *
 * The obvious reading is that staging is the difference, because unstaged every
 * one of the N threads in a block re-reads the same row of A and the step is
 * bandwidth-bound (an elementwise add measures 18.4 GB/s against system
 * memory's 19.7 GB/s ceiling). Measured, that reading is wrong:
 *
 *     [4096,64]x[64,256]   unstaged  1.42 ms   94 Gflop/s
 *     [4096,64]x[64,256]   staged    1.38 ms   98 Gflop/s
 *
 * -- inside the spread, and the whole step is unchanged at 19.3k tok/s. A's row
 * is 256 bytes and every thread in the block wants it at the same moment, so L1
 * was already serving those reads; staging moves them to shared memory, which
 * is not meaningfully closer. Whatever makes the two shapes differ, it is not
 * A's traffic.
 *
 * The generalisation stays because it is correct, it is verified branch by
 * branch against the definition (packages/tests/diff-matmul-tiled.mjs), and it
 * removes a restriction that was arbitrary from the caller's side. It is not
 * kept because it made anything faster. A guard costs one ISETP on the last
 * round only.
 */
static int can_tile(unsigned N, unsigned K) {
  return N > 0 && K > 0 && K <= MATMUL_TILE_MAX_K &&
         tile_rounds(N, K) * 7u + 40u < PR_MAX_INSTRUCTIONS;
}

unsigned pr_matmul_shared_bytes(unsigned N, unsigned K) {
  return can_tile(N, K) ? K * 4u : 0u;
}

unsigned pr_emit_matmul(hp_word *p, unsigned M, unsigned N, unsigned K) {
  return pr_emit_matmul_kind(p, M, N, K, 0);
}

/*
 * C = A @ B, or C = A @ B-TRANSPOSE, from one emitter.
 *
 * The only difference is how B is addressed: untransposed it is [K,N] and
 * B[k][col] is `k*N + col`, walking DOWN a column N elements a step;
 * transposed it is [N,K] and the same element is `col*K + k`, walking ALONG a
 * row one element a step.
 *
 * Without a fused form autograd transposes the weight and calls matmul, which
 * costs a launch AND a tensor nobody frees — 225 MB a step across three call
 * sites at 18 layers, the largest single entry in the allocation census. Every
 * weight here is stored [out, in] and used transposed, so this is the common
 * case rather than the special one.
 */
unsigned pr_emit_matmul_kind(hp_word *p, unsigned M, unsigned N, unsigned K,
                             int transposedB) {
  unsigned n = 0;

  const unsigned BW = pr_matmul_block(N);
  const unsigned chunks = col_chunks(N);
  /* Only when the chunks overhang N: for 12288 over 1024 they do not, and the
   * guard is not emitted at all. */
  const int guard_col = chunks * BW > N;

  p[n++] = hp_s2r(R_ROW, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_BATCH, HP_SR_CTAID_Y, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_safe());
  /* R_COL is the staging index until the chunk loop opens, and the column
   * after. Staging reads it as "which element of A's row am I fetching", which
   * is the thread id, so they agree there. */
  p[n++] = hp_iadd3_imm(R_COL, R_TID, 0, hp_ctrl_wait(BAR_ID));

  /* A[row][0] is element row*K, and B[0][col] is element col. Both indices then
   * only ever need adding to, which is why they are set up outside the loop:
   * the multiply is loop-invariant and doing it inside would be paying for it K
   * times to get the same answer. */
  /* A's row start, including the batch plane. It does not depend on the column,
   * so it is computed once and copied into R_AIDX at the top of every chunk. */
  p[n++] = hp_imad_imm(R_AROW, R_ROW, K, HP_RZ, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_TMP, R_BATCH, M * K, HP_RZ, hp_ctrl_safe());
  p[n++] = hp_iadd3_reg(R_AROW, R_AROW, R_TMP, hp_ctrl_safe());
  /* B's batch plane, likewise. */
  p[n++] = hp_imad_imm(R_BPLANE, R_BATCH, K * N, HP_RZ, hp_ctrl_safe());

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
    const unsigned rounds = tile_rounds(N, K);
    for (unsigned t = 0; t < rounds; t++) {
      const unsigned base = t * N;
      /*
       * The last round is short unless N divides K, and the threads past the
       * end must not load: A[row][K] is the next row's first element, so an
       * unguarded read would stage a neighbouring row's data and produce a
       * finite, plausible, wrong dot product.
       *
       * The predicate is set when the thread is PAST the range and used
       * NEGATED, which is the convention reduction.c established -- the
       * comparison the hardware offers is GT, so "col > (K-base)-1" is the
       * inactive half and @!P is the active one.
       */
      const int partial = base + N > K;
      if (partial)
        p[n++] = hp_isetp_gt_imm(P_TILE, R_COL, (K - base) - 1u, hp_ctrl_safe());

      /* Alternate the register set so this round cannot disturb the last one's
       * store while it is still reading. See the enum. */
      const unsigned rIdx = (t & 1u) ? R_STG_IDX_B : R_STG_IDX_A;
      const unsigned rAddr = (t & 1u) ? R_STG_ADDR_B : R_STG_ADDR_A;
      const unsigned rVal = (t & 1u) ? R_STG_VAL_B : R_STG_VAL_A;

      p[n++] = hp_iadd3_imm(rIdx, R_COL, base, hp_ctrl_safe());
      /* R_AROW, not R_AIDX: the walking index is reset per chunk and the chunk
       * loop has not opened yet. Reading R_AIDX here staged from whatever it
       * held — zero for row 0, which is why only row 1 reported a wrong dot
       * product and row 0 looked fine. */
      p[n++] = hp_iadd3_reg(R_LOAD_IDX, R_AROW, rIdx, hp_ctrl_safe());
      p[n++] = hp_imad_wide_const(rAddr, R_LOAD_IDX, R_ESIZE, 0,
                                  HERMES_CBUF0_PARAM_N(1),
                                  hp_ctrl_safe());
      hp_word load = hp_ldg(rVal, rAddr, 0, hp_ctrl_setbar(BAR_LOAD));
      hp_word store = hp_sts(rIdx, rVal, 0, hp_ctrl_wait(BAR_LOAD));
      if (partial) {
        load = hp_predicated(load, P_TILE, 1);
        store = hp_predicated(store, P_TILE, 1);
      }
      p[n++] = load;
      p[n++] = store;
    }
    p[n++] = hp_bar_sync(hp_ctrl_safe());
  }

  /*
   * THE CHUNK LOOP: this thread's columns are tid, tid+BW, tid+2BW, ...
   *
   * Everything below it was the whole kernel when a thread owned exactly one
   * column. It is re-entered per chunk with the accumulator, the counter and
   * both indices reset, which is why those moves live here and not in the
   * prologue. A's row is staged once, outside, because it does not depend on
   * the column.
   */
  p[n++] = hp_mov_imm(R_CHUNK, 0, hp_ctrl_safe());
  const unsigned chunk_top = n;
  p[n++] = hp_imad_imm(R_COL, R_CHUNK, BW, R_TID, hp_ctrl_safe());
  if (guard_col)
    p[n++] = hp_isetp_gt_imm(P_COL, R_COL, N - 1, hp_ctrl_safe());
  p[n++] = hp_mov_imm(R_ACC, 0, hp_ctrl_safe());
  p[n++] = hp_mov_imm(R_K, 0, hp_ctrl_safe());
  p[n++] = hp_iadd3_imm(R_AIDX, R_AROW, 0, hp_ctrl_safe());
  /* B[0][col] is element `col` when B is [K,N] and `col*K` when it is [N,K]. */
  if (transposedB)
    p[n++] = hp_imad_imm(R_BIDX, R_COL, K, R_BPLANE, hp_ctrl_safe());
  else
    p[n++] = hp_iadd3_reg(R_BIDX, R_COL, R_BPLANE, hp_ctrl_safe());

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
  {
    /* A thread whose column overhangs N must not LOAD either: B[k][col] past
     * the last column is the next row's data, and on the final k it is past the
     * tensor entirely. Predicated off, no address is formed. */
    hp_word bl = hp_ldg(R_BVAL, R_BADDR, 0, hp_ctrl_setbar(BAR_LOAD));
    p[n++] = guard_col ? hp_predicated(bl, P_COL, 1) : bl;
  }

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
  /* Down a column is N elements; along a row is one. */
  p[n++] = hp_iadd3_imm(R_BIDX, R_BIDX, transposedB ? 1 : (int)N, hp_ctrl_safe());
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
  /* Off the accumulator before the store — see R_STOREVAL. */
  p[n++] = hp_iadd3_imm(R_STOREVAL, R_ACC, 0, hp_ctrl_safe());
  {
    /* The overhanging threads computed a dot product from data they were not
     * allowed to read; the store is where that has to stop. */
    hp_word st = hp_stg(R_OUT, R_STOREVAL, 0, hp_ctrl_safe());
    p[n++] = guard_col ? hp_predicated(st, P_COL, 1) : st;
  }

  /* Next chunk of columns, if there is one. Emitted only when there is: for
   * every shape at or below the block width this is one pass and the branch
   * would be a loop that always exits. */
  if (chunks > 1) {
    p[n++] = hp_iadd3_imm(R_CHUNK, R_CHUNK, 1, hp_ctrl_safe());
    p[n++] = hp_isetp_gt_imm(P_CHUNK, R_CHUNK, chunks - 1, hp_ctrl_safe());
    const int cback = -(int)((n + 1 - chunk_top) * INSTR_BYTES);
    p[n] = hp_predicated(hp_bra(cback, hp_ctrl_branch()), P_CHUNK, 1);
    n++;
  }

  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}
