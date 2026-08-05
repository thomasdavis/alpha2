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
  /*
   * The tiled kernel: ROWS accumulators and ROWS staged A values.
   *
   * A block owning ONE output row re-reads all of B from global for every row
   * of the output — at N=1920, K=640 that is 4.9 MB read 64 times for a single
   * matmul, and it is why the GPU half of a 105M step sustains 0.54% of peak.
   * Giving a block several rows lets one B load feed several FFMAs.
   */
  /*
   * ONE SET PER ROW, not two alternating.
   *
   * Alternating between two sets is enough for two rows and not for four: row 2
   * reuses row 0's registers about seven instructions after row 0's store
   * issued, which is inside the window where the store has not yet taken its
   * operands. It presented as N=1023 wrong at ROWS=4 while every shape had been
   * right at ROWS=2 — the same hazard, at a shorter distance.
   */
  /* The K-unrolled loop reuses these when the tiled path is off; the two are
   * mutually exclusive by construction. */
  R_UA0 = 32, R_UA1 = 33, R_UA2 = 34, R_UA3 = 35,
  R_UB0 = 36, R_UB1 = 37, R_UB2 = 38, R_UB3 = 39,
  /* Column-blocked kernel: four accumulators, four B values, four store values
   * and four output address pairs — 24 registers, but on a 256-thread block
   * rather than a 1024-thread one, which is the whole point. */
  R_CACC0 = 32, R_CACC1 = 33, R_CACC2 = 34, R_CACC3 = 35,
  R_CB0 = 36, R_CB1 = 37, R_CB2 = 38, R_CB3 = 39,
  R_CSV0 = 40, R_CSV1 = 41, R_CSV2 = 42, R_CSV3 = 43,
  R_CO0 = 44, R_CO1 = 46, R_CO2 = 48, R_CO3 = 50, /* pairs */
  R_ACC0 = 32, R_ACC1 = 33, R_ACC2 = 34, R_ACC3 = 35,
  R_AV0 = 36, R_AV1 = 37, R_AV2 = 38, R_AV3 = 39,
  /* Numbered so that the ROWS actually used stay low: at two rows the highest
   * touched is R45, which leaves slack under a 48-register declaration. A
   * declaration with no slack above the highest register used raises
   * GR_EXCEPTION — that is the third time this file has learned it. */
  R_SV0 = 40, R_SV1 = 41, R_SV2 = 50, R_SV3 = 51,
  R_O0 = 42, R_O1 = 44, R_O2 = 46, R_O3 = 48, /* pairs */
  /*
   * A store value and an output address PER ROW.
   *
   * The rows store one after another, and row 1 computing its value clobbers
   * the register row 0's store is still reading — the sixth instance of this
   * hazard in this stack, and the second inside this file. It presents as
   * elements that are zero, on some shapes and not others, because whether the
   * store has taken its operand is a timing question: N=1025 failed while
   * N=1920 with the same chunk count and the same guard passed.
   */

  R_STG_IDX_B = 23,
  R_STG_ADDR_B = 24, /* R24:R25 */
  R_STG_VAL_B = 26,
};

#define BAR_ID 0   /* the two S2Rs */
#define BAR_LOAD 1 /* BOTH loads -- see the accumulate below */
#define P_DONE 0   /* set when the loop has run its course */
#define P_TILE 1   /* set when a thread is past the end of a short stage round */
#define P_COL 2    /* clear when this thread's column is inside N */
/*
 * ONE PREDICATE PER COLUMN in the column-blocked kernel.
 *
 * Its four columns are strided by the thread count, so col_0 can be inside N
 * while col_3 is past it — a single guard covers the first and lets the other
 * three write out of range. It showed up as exactly the shapes where N is not a
 * multiple of the chunk span (1920, 1728, 1025 wrong; 12288 and everything
 * below the span right), which is the signature of a guard that is computed but
 * not per lane.
 */
#define P_COL0 2
#define P_COL1 4
#define P_COL2 5
#define P_COL3 6
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

/*
 * OFF, and the reason is the one that explains the other two refutations.
 *
 *     105M seq 64, GPU ms/step (drained per op, 4 runs)
 *       untiled, unrolled by 4    264  286  267  368    matmul 452 us/call
 *       column-blocked, 256 thr   315  416  341  307    matmul 558 us/call
 *
 * It has strictly better arithmetic — 1.25 loads per FFMA against 2.0, and 3.5
 * instructions per FFMA against 4.25 — and it is slower, because at this shape
 * M IS 64. A matmul launches gridX = M blocks, so 64 blocks of 1024 threads is
 * 65,536 of this card's ~70,656 thread slots, and 64 blocks of 256 is 16,384 —
 * 23% of the machine.
 *
 * THAT IS WHY EVERY STRUCTURE THAT REDUCED THREAD COUNT LOST. Row tiling cut
 * the block count, column blocking cut the block width, and both were measured
 * against a shape that cannot fill the GPU either way. Unrolling won because it
 * adds work per thread without touching parallelism.
 *
 * So the GEMM is PARALLELISM-STARVED at batch 1, seq 64, and that is a property
 * of the shape rather than of the kernel: M = batch * seq. The way to a
 * measurable GEMM is a larger M, which needs the step's memory bounded so batch
 * and sequence can grow — the leak is not only about fitting, it gates having
 * enough work to optimise against.
 *
 * Kept and unwired, correct at every model shape. It should be re-measured, not
 * rewritten, the first time a step runs at a batch that fills the card.
 */
#define COLBLOCK_MATMUL 0

/*
 * OFF, and REFUTED — the commit that introduced it claimed 12% and that was one
 * favourable sample, not a result. A/B on one build, three runs each:
 *
 *     105M seq 64, GPU ms/step
 *       two rows a block   355  361  372
 *       one row  a block   342  360  355
 *
 * The same, within the spread. Four rows is worse still (345 ms and 56 declared
 * registers, which is one block an SM).
 *
 * WHY THE REASONING FAILED, which is the part worth keeping: the case for
 * tiling was that a block owning one output row re-reads all of B once per row
 * — 4.9 MB read 64 times for one matmul. That counts re-reads as DRAM traffic,
 * and they are not. B is 4.9 MB against this card's 4 MB L2, and 64 blocks
 * reading the same matrix concurrently hit in L2 for almost all of it. The
 * reuse tiling would provide was already there.
 *
 * So the GPU half sustaining 0.54% of peak is NOT explained by B's traffic, and
 * the next attempt should measure where that time actually goes before
 * proposing a structure to fix it. Kept and unwired because the kernel is
 * correct at every model shape and is the structure HMMA would slot into.
 */
/*
 * OFF, and refuted a SECOND time, for a different reason than the first.
 *
 * First attempt: tiling as a MEMORY optimisation — B re-read once per output
 * row. Refuted because B fits L2 and the reuse was already there.
 *
 * Second attempt, after the K-unroll made the GEMM issue-bound: tiling as an
 * ISSUE optimisation, since a B value loaded once feeds ROWS FFMAs. The
 * arithmetic is right — 1.5 loads per FFMA against 2 — and it is still slower:
 *
 *     105M seq 64, GPU ms/step (drained per op, 4 runs)
 *       untiled, unrolled by 4   264  286  267  368     matmul 452 us/call
 *       2 rows,  unrolled by 2   380  383  379  318     matmul 720 us/call
 *
 * Because it does not fit 48 declared registers and 56 costs more occupancy
 * than the saved instructions are worth. REGISTER PRESSURE is the binding
 * constraint on this kernel at 1024 threads a block, not instruction count —
 * which is the same wall four rows hit, and is worth knowing before designing
 * anything else around register blocking.
 *
 * The way past it is fewer threads a block with more work each, so a row costs
 * registers that are not multiplied by 1024 lanes. That is a different kernel,
 * and it is the one HMMA wants anyway.
 */
#define TILED_MATMUL 0

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

/*
 * Output rows one block computes.
 *
 * TWO first, deliberately. The register and shared-memory budgets both allow
 * four, and four was tried first and failed a layer suite for reasons that took
 * two eliminated suspects to not find. Two is the smallest step that still
 * halves B's traffic, and it is the one that can be checked against cpu_ref
 * shape by shape before anything larger is attempted.
 */
/*
 * TWO, and four is measured slower.
 *
 * Four halves B's traffic again and costs the registers that hide latency:
 * ACC, staged A, store value and output address per row is 56 declared, which
 * is one block an SM and not enough warps in flight.
 *
 *     105M seq 64, GPU time per step
 *       one row  354 ms      two rows  301 ms      four rows  345 ms
 *
 * So the win is not monotone in the tile, and the ceiling here is occupancy
 * rather than bandwidth. Four becomes right when a row costs fewer registers —
 * which is what a proper register-blocked GEMM, or HMMA's accumulator layout,
 * changes.
 */
#define MATMUL_TILE_ROWS 2u

int pr_matmul_tiled(unsigned M, unsigned N, unsigned K) {
  return TILED_MATMUL && M % MATMUL_TILE_ROWS == 0 && M >= MATMUL_TILE_ROWS &&
         (NvU64)MATMUL_TILE_ROWS * K * 4u <= 48u * 1024u &&
         can_tile(N, K) &&
         tile_rounds(N, K) * MATMUL_TILE_ROWS * 8u + 90u < PR_MAX_INSTRUCTIONS;
}

unsigned pr_matmul_rows(unsigned M, unsigned N, unsigned K) {
  return pr_matmul_tiled(M, N, K) ? MATMUL_TILE_ROWS : 1u;
}

unsigned pr_matmul_tiled_shared(unsigned M, unsigned N, unsigned K) {
  return pr_matmul_tiled(M, N, K) ? MATMUL_TILE_ROWS * K * 4u : 0u;
}

/*
 * C = A @ B (or A @ B^T) with a block computing MATMUL_TILE_ROWS output rows.
 *
 * The point is B REUSE: one global load of B[k][col] feeds ROWS FFMAs, so B
 * crosses the bus ROWS times less often. A's rows live in shared memory, ROWS
 * of them, addressed by `R_K*4 + r*K*4` — the immediate is a BYTE offset, which
 * tools/shared_offset_probe.c established rather than assumed.
 */
static unsigned emit_matmul_tiled(hp_word *p, unsigned M, unsigned N, unsigned K,
                                  int transposedB) {
  unsigned n = 0;
  const unsigned ROWS = MATMUL_TILE_ROWS;
  const unsigned BW = pr_matmul_block(N);
  const unsigned chunks = (N + BW - 1u) / BW;
  const int guard_col = chunks * BW > N;
  const unsigned ACC[4] = {R_ACC0, R_ACC1, R_ACC2, R_ACC3};
  /* ROWS * UNROLL staged A values, then UNROLL loaded B values. */
  const unsigned AV[4] = {R_AV0, R_AV1, R_AV2, R_AV3};
  /* R_UB1 aliases R_AV1 — the unrolled and tiled register maps were written
   * separately and overlap. R46 is clear of both AV (36..39) and the per-row
   * store registers used at two rows (40,41 and 42..45). */
  const unsigned BV[2] = {R_BVAL, 46};
  const unsigned SV[4] = {R_SV0, R_SV1, R_SV2, R_SV3};
  const unsigned OA[4] = {R_O0, R_O1, R_O2, R_O3};

  p[n++] = hp_s2r(R_ROW, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_BATCH, HP_SR_CTAID_Y, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_safe());
  /* This block's FIRST output row. */
  p[n++] = hp_imad_imm(R_ROW, R_ROW, ROWS, HP_RZ, hp_ctrl_wait(BAR_ID));

  p[n++] = hp_imad_imm(R_AROW, R_ROW, K, HP_RZ, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_TMP, R_BATCH, M * K, HP_RZ, hp_ctrl_safe());
  p[n++] = hp_iadd3_reg(R_AROW, R_AROW, R_TMP, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_BPLANE, R_BATCH, K * N, HP_RZ, hp_ctrl_safe());

  /* Stage ROWS rows of A. Round t covers elements [t*BW, t*BW+BW) of a row. */
  const unsigned rounds = tile_rounds(N, K);
  for (unsigned r = 0; r < ROWS; r++) {
    for (unsigned t = 0; t < rounds; t++) {
      const unsigned base = t * BW;
      const int partial = base + BW > K;
      if (partial)
        p[n++] = hp_isetp_gt_imm(P_TILE, R_TID, (K - base) - 1u, hp_ctrl_safe());
      const unsigned rIdx = (t & 1u) ? R_STG_IDX_B : R_STG_IDX_A;
      const unsigned rAddr = (t & 1u) ? R_STG_ADDR_B : R_STG_ADDR_A;
      const unsigned rVal = (t & 1u) ? R_STG_VAL_B : R_STG_VAL_A;
      /* Element within the row; the row itself is the STS byte offset. */
      p[n++] = hp_iadd3_imm(rIdx, R_TID, base, hp_ctrl_safe());
      p[n++] = hp_iadd3_reg(R_LOAD_IDX, R_AROW, rIdx, hp_ctrl_safe());
      p[n++] = hp_iadd3_imm(R_LOAD_IDX, R_LOAD_IDX, (int)(r * K), hp_ctrl_safe());
      p[n++] = hp_imad_wide_const(rAddr, R_LOAD_IDX, R_ESIZE, 0,
                                  HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
      hp_word load = hp_ldg(rVal, rAddr, 0, hp_ctrl_setbar(BAR_LOAD));
      hp_word store = hp_sts(rIdx, rVal, r * K * 4u, hp_ctrl_wait(BAR_LOAD));
      if (partial) { load = hp_predicated(load, P_TILE, 1);
                     store = hp_predicated(store, P_TILE, 1); }
      p[n++] = load;
      p[n++] = store;
    }
  }
  p[n++] = hp_bar_sync(hp_ctrl_safe());

  p[n++] = hp_mov_imm(R_CHUNK, 0, hp_ctrl_safe());
  const unsigned chunk_top = n;
  p[n++] = hp_imad_imm(R_COL, R_CHUNK, BW, R_TID, hp_ctrl_safe());
  if (guard_col)
    p[n++] = hp_isetp_gt_imm(P_COL, R_COL, N - 1, hp_ctrl_safe());
  for (unsigned r = 0; r < ROWS; r++)
    p[n++] = hp_mov_imm(ACC[r], 0, hp_ctrl_safe());
  p[n++] = hp_mov_imm(R_K, 0, hp_ctrl_safe());
  if (transposedB)
    p[n++] = hp_imad_imm(R_BIDX, R_COL, K, R_BPLANE, hp_ctrl_safe());
  else
    p[n++] = hp_iadd3_reg(R_BIDX, R_COL, R_BPLANE, hp_ctrl_safe());

  /*
   * UNROLLED BY TWO, on top of the two rows.
   *
   * Tiling was refuted as a MEMORY optimisation — B fits L2 and the reuse was
   * already there — and it is not a memory optimisation any more. Once the K
   * loop was unrolled the GEMM became issue-bound at two loads per FFMA, and
   * that ratio is the one thing tiling does change: a B value loaded once feeds
   * ROWS FFMAs.
   *
   *     untiled, unrolled by 4    17 instructions / 4 FFMA   = 4.25
   *     2 rows,  unrolled by 2    14 instructions / 4 FFMA   = 3.50
   *
   * Two and two rather than four and four because the register budget is what
   * made four rows lose before: 56 declared is one block an SM. This fits the
   * same 48 the unrolled kernel already uses.
   */
  const unsigned UNROLL = (K % 2u == 0 && K >= 2u) ? 2u : 1u;
  const unsigned loop_top = n;
  p[n++] = hp_imad_wide_const(R_BADDR, R_BIDX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(2), hp_ctrl_safe());
  for (unsigned j = 0; j < UNROLL; j++) {
    const unsigned off = transposedB ? j * 4u : j * N * 4u;
    hp_word bl = hp_ldg(BV[j], R_BADDR, off, hp_ctrl_setbar(BAR_LOAD));
    p[n++] = guard_col ? hp_predicated(bl, P_COL, 1) : bl;
  }
  for (unsigned j = 0; j < UNROLL; j++)
    for (unsigned r = 0; r < ROWS; r++)
      p[n++] = hp_lds(AV[j * ROWS + r], R_K, r * K * 4u + j * 4u,
                      hp_ctrl_setbar(BAR_LOAD));
  /* One wait covering every load — the barrier counts outstanding writes, which
   * is the reasoning the untiled kernel records. */
  {
    int first = 1;
    for (unsigned j = 0; j < UNROLL; j++)
      for (unsigned r = 0; r < ROWS; r++) {
        p[n++] = hp_ffma(ACC[r], AV[j * ROWS + r], BV[j], ACC[r],
                         first ? hp_ctrl_wait(BAR_LOAD) : hp_ctrl_safe());
        first = 0;
      }
  }

  p[n++] = hp_iadd3_imm(R_BIDX, R_BIDX,
                        transposedB ? (int)UNROLL : (int)(UNROLL * N),
                        hp_ctrl_safe());
  p[n++] = hp_iadd3_imm(R_K, R_K, (int)UNROLL, hp_ctrl_safe());
  p[n++] = hp_isetp_gt_imm(P_DONE, R_K, K - 1, hp_ctrl_safe());
  const int back = -(int)((n + 1 - loop_top) * INSTR_BYTES);
  p[n] = hp_predicated(hp_bra(back, hp_ctrl_branch()), P_DONE, 1);
  n++;

  for (unsigned r = 0; r < ROWS; r++) {
    /* Per row, because the stores are consecutive and nothing interlocks the
     * next one's operand setup against this one's read. See R_STOREVAL1. */
    const unsigned sv = SV[r];
    const unsigned outAddr = OA[r];
    p[n++] = hp_imad_imm(R_OIDX, R_ROW, N, R_COL, hp_ctrl_safe());
    p[n++] = hp_iadd3_imm(R_OIDX, R_OIDX, (int)(r * N), hp_ctrl_safe());
    p[n++] = hp_imad_imm(R_TMP, R_BATCH, M * N, HP_RZ, hp_ctrl_safe());
    p[n++] = hp_iadd3_reg(R_OIDX, R_OIDX, R_TMP, hp_ctrl_safe());
    p[n++] = hp_imad_wide_const(outAddr, R_OIDX, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM_N(0), hp_ctrl_safe());
    p[n++] = hp_iadd3_imm(sv, ACC[r], 0, hp_ctrl_safe());
    hp_word st = hp_stg(outAddr, sv, 0, hp_ctrl_safe());
    p[n++] = guard_col ? hp_predicated(st, P_COL, 1) : st;
  }

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


/*
 * COLUMN BLOCKING on a NARROW block.
 *
 * The K-unroll left the GEMM issue-bound at two loads per FFMA, and the only
 * way past that ratio is for a loaded value to feed more than one FFMA. Row
 * tiling does that and was refuted twice, most recently because it needs more
 * registers and every register is multiplied by 1024 lanes — the block size,
 * not the instruction count, is what priced it out.
 *
 * So shrink the block. 256 threads each computing FOUR columns covers the same
 * 1024 columns a chunk did before, and:
 *
 *   - one LDS of A[k] now feeds four FFMAs instead of one (1.25 loads per FFMA
 *     against 2.0), because the four columns share a row of A;
 *   - the four B loads share one computed address and differ by an immediate,
 *     which tools/shared_offset_probe.c established is a byte offset;
 *   - and 24 extra registers cost 24*256 rather than 24*1024, so occupancy goes
 *     UP rather than down: ~56 registers on 256 threads is 14,336 a block
 *     against 49,152 for 48 on 1024.
 *
 * Columns are STRIDED by the thread count, not blocked per thread, so adjacent
 * threads still read adjacent B addresses and the loads stay coalesced.
 */
#define MATMUL_COL_THREADS 256u
#define MATMUL_COL_PER_THREAD 4u

int pr_matmul_colblocked(unsigned M, unsigned N, unsigned K) {
  const unsigned span = MATMUL_COL_THREADS * MATMUL_COL_PER_THREAD;
  return COLBLOCK_MATMUL && N >= span && K <= MATMUL_TILE_MAX_K && K >= 1 &&
         ((NvU64)K * 4u <= 48u * 1024u) &&
         ((K + MATMUL_COL_THREADS - 1u) / MATMUL_COL_THREADS) * 7u + 90u
             < PR_MAX_INSTRUCTIONS;
}

static unsigned emit_matmul_cols(hp_word *p, unsigned M, unsigned N, unsigned K,
                                 int transposedB) {
  unsigned n = 0;
  const unsigned T = MATMUL_COL_THREADS, CPT = MATMUL_COL_PER_THREAD;
  const unsigned span = T * CPT;
  const unsigned chunks = (N + span - 1u) / span;
  const int guard = chunks * span > N;
  const unsigned ACC[4] = {R_CACC0, R_CACC1, R_CACC2, R_CACC3};
  const unsigned BV[4] = {R_CB0, R_CB1, R_CB2, R_CB3};
  const unsigned SV[4] = {R_CSV0, R_CSV1, R_CSV2, R_CSV3};
  const unsigned OA[4] = {R_CO0, R_CO1, R_CO2, R_CO3};
  const unsigned PC[4] = {P_COL0, P_COL1, P_COL2, P_COL3};

  p[n++] = hp_s2r(R_ROW, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_BATCH, HP_SR_CTAID_Y, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_wait(BAR_ID));
  p[n++] = hp_imad_imm(R_AROW, R_ROW, K, HP_RZ, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_TMP, R_BATCH, M * K, HP_RZ, hp_ctrl_safe());
  p[n++] = hp_iadd3_reg(R_AROW, R_AROW, R_TMP, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_BPLANE, R_BATCH, K * N, HP_RZ, hp_ctrl_safe());

  /* Stage A's row: T threads, ceil(K/T) rounds. */
  const unsigned rounds = (K + T - 1u) / T;
  for (unsigned t = 0; t < rounds; t++) {
    const unsigned base = t * T;
    const int partial = base + T > K;
    if (partial)
      p[n++] = hp_isetp_gt_imm(P_TILE, R_TID, (K - base) - 1u, hp_ctrl_safe());
    const unsigned rIdx = (t & 1u) ? R_STG_IDX_B : R_STG_IDX_A;
    const unsigned rAddr = (t & 1u) ? R_STG_ADDR_B : R_STG_ADDR_A;
    const unsigned rVal = (t & 1u) ? R_STG_VAL_B : R_STG_VAL_A;
    p[n++] = hp_iadd3_imm(rIdx, R_TID, base, hp_ctrl_safe());
    p[n++] = hp_iadd3_reg(R_LOAD_IDX, R_AROW, rIdx, hp_ctrl_safe());
    p[n++] = hp_imad_wide_const(rAddr, R_LOAD_IDX, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
    hp_word load = hp_ldg(rVal, rAddr, 0, hp_ctrl_setbar(BAR_LOAD));
    hp_word store = hp_sts(rIdx, rVal, 0, hp_ctrl_wait(BAR_LOAD));
    if (partial) { load = hp_predicated(load, P_TILE, 1);
                   store = hp_predicated(store, P_TILE, 1); }
    p[n++] = load;
    p[n++] = store;
  }
  p[n++] = hp_bar_sync(hp_ctrl_safe());

  p[n++] = hp_mov_imm(R_CHUNK, 0, hp_ctrl_safe());
  const unsigned chunk_top = n;
  /* col_0 = chunk*span + tid; col_j = col_0 + j*T. */
  p[n++] = hp_imad_imm(R_COL, R_CHUNK, span, R_TID, hp_ctrl_safe());
  if (guard)
    for (unsigned j = 0; j < CPT; j++) {
      p[n++] = hp_iadd3_imm(R_TMP, R_COL, (int)(j * T), hp_ctrl_safe());
      p[n++] = hp_isetp_gt_imm(PC[j], R_TMP, N - 1, hp_ctrl_safe());
    }
  for (unsigned j = 0; j < CPT; j++)
    p[n++] = hp_mov_imm(ACC[j], 0, hp_ctrl_safe());
  p[n++] = hp_mov_imm(R_K, 0, hp_ctrl_safe());
  if (transposedB)
    p[n++] = hp_imad_imm(R_BIDX, R_COL, K, R_BPLANE, hp_ctrl_safe());
  else
    p[n++] = hp_iadd3_reg(R_BIDX, R_COL, R_BPLANE, hp_ctrl_safe());

  const unsigned loop_top = n;
  /* ONE A value, four FFMAs — the reuse this kernel exists for. */
  p[n++] = hp_lds(R_AVAL, R_K, 0, hp_ctrl_setbar(BAR_LOAD));
  p[n++] = hp_imad_wide_const(R_BADDR, R_BIDX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(2), hp_ctrl_safe());
  for (unsigned j = 0; j < CPT; j++) {
    const unsigned off = transposedB ? j * T * K * 4u : j * T * 4u;
    hp_word bl = hp_ldg(BV[j], R_BADDR, off, hp_ctrl_setbar(BAR_LOAD));
    p[n++] = guard ? hp_predicated(bl, PC[j], 1) : bl;
  }
  for (unsigned j = 0; j < CPT; j++)
    p[n++] = hp_ffma(ACC[j], R_AVAL, BV[j], ACC[j],
                     j == 0 ? hp_ctrl_wait(BAR_LOAD) : hp_ctrl_safe());

  p[n++] = hp_iadd3_imm(R_BIDX, R_BIDX, transposedB ? 1 : (int)N, hp_ctrl_safe());
  p[n++] = hp_iadd3_imm(R_K, R_K, 1, hp_ctrl_safe());
  p[n++] = hp_isetp_gt_imm(P_DONE, R_K, K - 1, hp_ctrl_safe());
  const int back = -(int)((n + 1 - loop_top) * INSTR_BYTES);
  p[n] = hp_predicated(hp_bra(back, hp_ctrl_branch()), P_DONE, 1);
  n++;

  for (unsigned j = 0; j < CPT; j++) {
    p[n++] = hp_imad_imm(R_OIDX, R_ROW, N, R_COL, hp_ctrl_safe());
    p[n++] = hp_iadd3_imm(R_OIDX, R_OIDX, (int)(j * T), hp_ctrl_safe());
    p[n++] = hp_imad_imm(R_TMP, R_BATCH, M * N, HP_RZ, hp_ctrl_safe());
    p[n++] = hp_iadd3_reg(R_OIDX, R_OIDX, R_TMP, hp_ctrl_safe());
    p[n++] = hp_imad_wide_const(OA[j], R_OIDX, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM_N(0), hp_ctrl_safe());
    p[n++] = hp_iadd3_imm(SV[j], ACC[j], 0, hp_ctrl_safe());
    hp_word st = hp_stg(OA[j], SV[j], 0, hp_ctrl_safe());
    p[n++] = guard ? hp_predicated(st, PC[j], 1) : st;
  }

  if (chunks > 1) {
    p[n++] = hp_iadd3_imm(R_CHUNK, R_CHUNK, 1, hp_ctrl_safe());
    p[n++] = hp_isetp_gt_imm(P_CHUNK, R_CHUNK, chunks - 1, hp_ctrl_safe());
    const int cb = -(int)((n + 1 - chunk_top) * INSTR_BYTES);
    p[n] = hp_predicated(hp_bra(cb, hp_ctrl_branch()), P_CHUNK, 1);
    n++;
  }
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}

unsigned pr_emit_matmul_kind(hp_word *p, unsigned M, unsigned N, unsigned K,
                             int transposedB) {
  if (pr_matmul_colblocked(M, N, K))
    return emit_matmul_cols(p, M, N, K, transposedB);
  if (pr_matmul_tiled(M, N, K))
    return emit_matmul_tiled(p, M, N, K, transposedB);
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

  /*
   * UNROLLED BY FOUR when K allows, because the loop was nine instructions per
   * FFMA — about 11% of issued instructions doing arithmetic, which is the
   * shape of a GEMM sustaining 0.54% of peak.
   *
   * Both immediates are BYTE offsets (tools/shared_offset_probe.c measured LDS
   * and LDG), so four loads share one computed address and the four index
   * updates collapse to two. That is 4 FFMAs per 17 instructions against 1 per
   * 9 — issue efficiency roughly doubles without touching the memory pattern.
   */
  const int unroll = (K % 4u == 0) && (K >= 4u);
  if (unroll) {
    const unsigned UA[4] = {R_UA0, R_UA1, R_UA2, R_UA3};
    const unsigned UB[4] = {R_UB0, R_UB1, R_UB2, R_UB3};
    const unsigned loop_top4 = n;

    if (tiled) {
      for (unsigned j = 0; j < 4; j++)
        p[n++] = hp_lds(UA[j], R_K, j * 4u, hp_ctrl_setbar(BAR_LOAD));
    } else {
      p[n++] = hp_imad_wide_const(R_AADDR, R_AIDX, R_ESIZE, 0,
                                  HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
      for (unsigned j = 0; j < 4; j++)
        p[n++] = hp_ldg(UA[j], R_AADDR, j * 4u, hp_ctrl_setbar(BAR_LOAD));
    }
    p[n++] = hp_imad_wide_const(R_BADDR, R_BIDX, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM_N(2), hp_ctrl_safe());
    for (unsigned j = 0; j < 4; j++) {
      /* B walks DOWN a column, N elements a step — or along a row when B is
       * transposed, one element a step. */
      const unsigned off = transposedB ? j * 4u : j * N * 4u;
      p[n++] = hp_ldg(UB[j], R_BADDR, off, hp_ctrl_setbar(BAR_LOAD));
    }
    for (unsigned j = 0; j < 4; j++)
      p[n++] = hp_ffma(R_ACC, UA[j], UB[j], R_ACC,
                       j == 0 ? hp_ctrl_wait(BAR_LOAD) : hp_ctrl_safe());

    p[n++] = hp_iadd3_imm(R_AIDX, R_AIDX, 4, hp_ctrl_safe());
    p[n++] = hp_iadd3_imm(R_BIDX, R_BIDX, transposedB ? 4 : (int)(4u * N),
                          hp_ctrl_safe());
    p[n++] = hp_iadd3_imm(R_K, R_K, 4, hp_ctrl_safe());
    p[n++] = hp_isetp_gt_imm(P_DONE, R_K, K - 1, hp_ctrl_safe());
    const int back4 = -(int)((n + 1 - loop_top4) * INSTR_BYTES);
    p[n] = hp_predicated(hp_bra(back4, hp_ctrl_branch()), P_DONE, 1);
    n++;
  } else {

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
  }

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
