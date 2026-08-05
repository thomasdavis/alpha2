/*
 * hmma.c — the GEMM that runs on the TENSOR CORES.
 *
 * WHAT: C[M,N] = A[M,K] * B[K,N] (or B^T) built out of
 * `HMMA.16816.F32` — the warp-level 16x8x16 matrix multiply-accumulate, f16
 * operands, f32 accumulator. matmul.c does the same job with scalar FFMA and
 * sustains ~1.2 TFLOP/s, which is 6% of this card's FP32 peak. This exists
 * because that number cannot be fixed by scheduling it better.
 *
 * WHY, with the arithmetic, so the target is not a wish. Measured on the 3070
 * this stack runs on (tools/hmma_ceiling.cu, and these are this GPU rather than
 * a table):
 *
 *     mma.sync m16n8k16, f32 accumulate, from registers   45.5 TFLOP/s
 *     mma.sync m16n8k16, f16 accumulate                   90.3
 *     cuBLAS f16/f32 at this model's shapes             24-32
 *     scalar FFMA, this stack, today                       1.2
 *
 * 30,000 tok/s at 104.4M parameters is 17.4 TFLOP/s sustained — 38% of the
 * instruction ceiling and 55-70% of what the vendor library gets at the same
 * shapes. Out of reach in FP32 by a factor of four; in reach here.
 *
 * THE PART THAT IS DANGEROUS. `m16n8k16` is a WARP instruction: its 32 lanes
 * cooperatively hold one 16x8 output tile and each lane owns a specific,
 * non-obvious quarter of A, B and C. Getting the ENCODING wrong faults; getting
 * the FRAGMENT LAYOUT wrong returns a finite, plausible, wrong matrix that
 * passes any test looking for NaNs. So the layout below is not read from a
 * manual — it was measured on this GPU with a known-answer CUDA probe (worst
 * absolute error 0 across all 128 outputs) and is recorded in
 * hephaestus/isa/hmma-sm86.md. With g = lane>>2 and l = lane&3:
 *
 *     A  a0: row g,   k 2l, 2l+1      a1: row g+8, k 2l,   2l+1
 *        a2: row g,   k 2l+8, 2l+9    a3: row g+8, k 2l+8, 2l+9
 *     B  b0: col g,   k 2l, 2l+1      b1: col g,   k 2l+8, 2l+9
 *     C  d0,d1: row g, cols 2l,2l+1   d2,d3: row g+8, cols 2l,2l+1
 *
 * A is read row-major and B COLUMN-major — the instruction is `row.col`, so it
 * WANTS B transposed. matmul.c refutes the fused transposed form twice because
 * a transposed B makes the scalar kernel's loads uncoalesced; that refutation
 * does not carry here, and the transposed case is the CHEAPER one below.
 *
 * WHAT THIS DELIBERATELY DOES NOT DO, yet: no shared-memory staging and no
 * asynchronous copy. Operands come straight from global into registers and are
 * converted to f16 on the way in. That leaves the kernel reading 4 bytes per
 * flop-pair from L1 where a staged version would read from shared memory, and
 * it is the next thing to measure rather than the next thing to assume — the
 * register blocking below is what removes most of that traffic, and it is worth
 * knowing how much is left before adding a barrier to every k-step.
 */
#include "hmma.h"

/*
 * The output a WARP owns: TM fragments down, TN across — so 16*TM rows by
 * 8*TN columns, held entirely in registers.
 *
 * This is the whole reason the kernel is not memory-bound. One A fragment feeds
 * TN multiply-accumulates and one B fragment feeds TM of them, so the loads per
 * HMMA fall from 12 at TM=TN=1 to (8*TM + 4*TN)/(TM*TN) — four at 2x4, three at
 * 4x4. The accumulator is TM*TN*4 registers and never leaves them until the
 * epilogue.
 */
/* Overridable at build time (-DHMMA_TM=...) so the tile can be SWEPT rather
 * than argued about: the loads-per-HMMA arithmetic says bigger is better and
 * says nothing about where register pressure and occupancy turn it round. */
#ifndef HMMA_TM
#define HMMA_TM 2u
#endif
#ifndef HMMA_TN
#define HMMA_TN 4u
#endif
/*
 * The warps of a block, laid out in TWO dimensions.
 *
 * They were all along N, which makes the block tile 32 rows by 128 columns and
 * decides how often each operand is re-read: A once per column block, B once
 * per row block. At the shapes this model runs that is the wrong way round —
 * B is the larger matrix and it was being re-read sixteen times while A was
 * re-read fifteen. Going 2 x 4 over eight warps makes the tile 64 x 128 and
 * cuts the implied traffic of a qkv projection from 98 MB to 59.
 *
 * WARPS_N must be a power of two: the warp's two coordinates are recovered with
 * a shift and a multiply-subtract, because sm_86 has no integer divide.
 */
#ifndef HMMA_WARPS_M
#define HMMA_WARPS_M 2u
#endif
#ifndef HMMA_WARPS_N
#define HMMA_WARPS_N 2u
#endif
#define HMMA_WARPS (HMMA_WARPS_M * HMMA_WARPS_N)

/* One HMMA covers 16 rows, 8 columns and 16 of K. Not tunable: they are the
 * instruction's shape. */
#define MMA_M 16u
#define MMA_N 8u
#define MMA_K 16u

unsigned pr_hmma_block_rows(void) { return MMA_M * HMMA_TM * HMMA_WARPS_M; }
unsigned pr_hmma_block_cols(void) { return MMA_N * HMMA_TN * HMMA_WARPS_N; }
unsigned pr_hmma_threads(void) { return 32u * HMMA_WARPS; }

/*
 * Whether this shape can use the tensor path at all.
 *
 * The tile is not padded and there is no bounds predicate, so the shape has to
 * divide. Every matmul a 105M GPT runs does: M is batch*seq (512), N is 640,
 * 1920, 2560 or 12288, K is 640, 2560 or 64. A shape that does not divide falls
 * back to matmul.c, which is correct at every shape and is why this one is
 * allowed to require things.
 *
 * Requiring rather than padding is deliberate. A masked tail costs a predicate
 * on every load in the inner loop — the loop this kernel exists to make fast —
 * to serve shapes that the model does not have.
 */
int pr_hmma_applies(unsigned M, unsigned N, unsigned K) {
  if (!PR_HMMA_ENABLED) return 0;
  if (M == 0 || N == 0 || K == 0) return 0;
  if (K % MMA_K != 0) return 0;
  if (M % pr_hmma_block_rows() != 0) return 0;
  if (N % pr_hmma_block_cols() != 0) return 0;
  /*
   * The loads reach their second row and their k+8 pair through the LOAD's
   * IMMEDIATE OFFSET, which is 24 bits signed. That is what lets a fragment own
   * one address pair instead of two. A shape past it would silently wrap, so it
   * falls back instead — no model dimension comes close (K=2560 needs 82 KB of
   * offset against 8 MB available), and a bound that is never hit is still the
   * difference between a fallback and a wrong answer.
   */
  if (32u * K + 36u > 0x7fffffu) return 0;
  if (36u * N > 0x7fffffu) return 0;
  /* The transposed-A layout reaches along M with the same immediates. Checked
   * unconditionally because pr_hmma_applies does not take the layout — a shape
   * this kernel can run must be runnable in every layout the caller may pick,
   * and no model dimension comes near the bound either way. */
  if (36u * M > 0x7fffffu) return 0;
  /* The instruction budget: the k-step body is emitted once, the epilogue once
   * per accumulator fragment. Both are bounded by the tile, which is a compile
   * -time constant, so this can only fail if the tile is raised carelessly. */
  return 1;
}

/*
 * THE REGISTER MAP, computed from the tile rather than written out.
 *
 * Every quad an HMMA reads must start on a multiple of four and every pair on a
 * multiple of two; the hardware does not check, and a misaligned operand reads
 * neighbouring registers. Laying the map out by hand for one tile and then
 * changing the tile is how that goes wrong, so the offsets are derived and the
 * alignment is asserted at the top of the emitter.
 */
enum {
  R_CTA_M = 0,  /* ctaid.x — which block of rows */
  R_CTA_N = 1,  /* ctaid.y — which block of columns */
  R_BATCH = 2,  /* ctaid.z — which plane */
  R_TID = 3,
  R_WARP = 4,
  R_LANE = 5,
  R_G = 6,      /* lane >> 2 */
  R_L = 7,      /* lane & 3 */
  R_ESIZE = 8,  /* 4 */
  R_K = 9,      /* the k loop counter, stepping by 16 */
  R_TMP = 10,
  R_TMP2 = 11,
  R_KN = 12,    /* (k + 2l) * N, for the untransposed B only */
  R_KM = 14,    /* (k + 2l) * M, for the transposed A only */
  R_WARPN = 15, /* warp % WARPS_N; R_WARP holds warp / WARPS_N */
  R_OBASE = 13, /* the output index of this lane's first element */
  R_ADDR_BASE = 16,
};

/*
 * ONE ADDRESS PAIR PER FRAGMENT, not one reused pair — and this is THE bug the
 * first version had, so it is written down rather than left as a convention.
 *
 * A global load holds its address registers until the memory pipe accepts them,
 * which is not when it issued, and this stack has no write-after-read interlock
 * (control.h). The first version computed one pair, issued a fragment's loads,
 * and immediately recomputed the same pair for the next fragment — so every
 * fragment after the first raced its predecessor's loads.
 *
 * It passed at a 1x1 tile and failed at every larger one, which is the tell: at
 * TM=TN=1 there is exactly one fragment and the pair is never reused. Larger
 * tiles returned a finite, plausible, WRONG matrix — output row 0 held the
 * answer for column 4, rows with an even lane group were right and odd ones
 * wrong — because which load is still holding its operands depends on queue
 * depth. matmul.c records four earlier instances of this same hazard; this is
 * the fifth, and the first on the LOAD side.
 *
 * Barriers would order it; separate registers remove it, and there are only six
 * barriers against TM+TN fragments. So each fragment owns a pair for the whole
 * k-step, and the two rows of an A fragment share one pair by putting the
 * 8-row stride in the load's IMMEDIATE offset — which is what makes this cost
 * TM+TN pairs rather than 2*TM+TN.
 */
#define ADDR_PAIRS (HMMA_TM + HMMA_TN)
#define ADDR_A(tm) (R_ADDR_BASE + 2u * (tm))
#define ADDR_B(tn) (R_ADDR_BASE + 2u * (HMMA_TM + (tn)))

/* The row/column bases, TM of them then TN. */
#define R_IDX (R_ADDR_BASE + 2u * ADDR_PAIRS)
#define R_TEMP_BASE (R_IDX + HMMA_TM + HMMA_TN)
/* The k-step loads every operand before converting any of them: one wait for
 * the whole step instead of one per fragment, which is the difference between
 * paying global latency once and paying it six times. */
#define R_TEMP_COUNT (8u * HMMA_TM + 4u * HMMA_TN)
#define ALIGN4(x) (((x) + 3u) & ~3u)
#define R_ACC ALIGN4(R_TEMP_BASE + R_TEMP_COUNT)
#define R_AFRAG ALIGN4(R_ACC + 4u * HMMA_TM * HMMA_TN)
#define R_BFRAG (((R_AFRAG + 4u * HMMA_TM) + 1u) & ~1u)
#define R_HIGHEST (R_BFRAG + 2u * HMMA_TN - 1u)

unsigned pr_hmma_regs(void) {
  /* Slack above the highest register used, rounded to eight — qmd.c: "a
   * declaration with no slack above the highest register used is what raises
   * GR_EXCEPTION", one register was not enough there either, and the hardware
   * allocates the register file in groups rather than singly. */
  return ((R_HIGHEST + 8u) + 7u) & ~7u;
}

/*
 * The alignment the instruction requires, checked by the compiler.
 *
 * An HMMA quad must start on a multiple of four and a pair on a multiple of
 * two. The hardware does not check: a misaligned operand silently reads
 * neighbouring registers and returns a plausible wrong matrix. Deriving the map
 * from the tile makes that a build error instead of a debugging session.
 */
_Static_assert(R_ACC % 4u == 0, "HMMA accumulator quad must be 4-aligned");
_Static_assert(R_AFRAG % 4u == 0, "HMMA A fragment quad must be 4-aligned");
_Static_assert(R_BFRAG % 2u == 0, "HMMA B fragment pair must be 2-aligned");
_Static_assert(R_HIGHEST < 250u, "HMMA tile exceeds the register file");

#define BAR_ID 0    /* the special registers */
#define BAR_LOAD 1  /* every global load in a k-step */
#define BAR_MMA 2   /* the accumulator, for the epilogue's first read */
/* The four rotating store address pairs. BAR_ID and BAR_LOAD are finished by
 * the epilogue — the loop they served has exited — so two of them are reused
 * rather than asking the hardware for barriers it does not have (six). */
#define BAR_ST0 3
#define BAR_ST1 4
#define BAR_ST2 5
#define BAR_ST3 BAR_ID
#define P_DONE 0
#define INSTR_BYTES 16

/*
 * The epilogue's address pairs live in the k-loop's STAGING registers, which
 * are dead once the loop has exited. Even-aligned because they are 64-bit
 * pairs, and taken from the temps because the alternative is four more
 * registers on every thread for the sake of the last thirty instructions.
 */
#define EPI_BASE (((R_TEMP_BASE) + 1u) & ~1u)
#define EPI_ADDR(i) (EPI_BASE + 2u * (i))

/* Where lane-owned pieces live within the tile. */
#define ACC_OF(tm, tn) (R_ACC + 4u * ((tm) * HMMA_TN + (tn)))
#define AFRAG_OF(tm) (R_AFRAG + 4u * (tm))
#define BFRAG_OF(tn) (R_BFRAG + 2u * (tn))
#define TEMP_A(tm, i) (R_TEMP_BASE + 8u * (tm) + (i))
#define TEMP_B(tn, i) (R_TEMP_BASE + 8u * HMMA_TM + 4u * (tn) + (i))

unsigned pr_emit_hmma(hp_word *p, unsigned M, unsigned N, unsigned K,
                      pr_mm_kind kind) {
  const int transposedB = (kind == PR_MM_NT);
  const int transposedA = (kind == PR_MM_TA);
  unsigned n = 0;
  const unsigned BM = pr_hmma_block_rows();
  const unsigned BN = pr_hmma_block_cols();
  /* Columns one warp owns, which is how the block's N extent is divided. */
  const unsigned WN = MMA_N * HMMA_TN;

  p[n++] = hp_s2r(R_CTA_M, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_CTA_N, HP_SR_CTAID_Y, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_BATCH, HP_SR_CTAID_Z, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_safe());

  /*
   * lane and warp WITHOUT a mask instruction.
   *
   * warp = tid >> 5 and lane = tid - 32*warp; g = lane >> 2 and l = lane -
   * 4*g. IMAD with a negative immediate is a subtract, and the two shifts are
   * the only other instructions needed — which avoids introducing an AND-
   * immediate form that no other kernel in this stack has had to encode.
   */
  p[n++] = hp_shr_imm(R_WARP, R_TID, 5, hp_ctrl_wait(BAR_ID));
  p[n++] = hp_imad_imm(R_LANE, R_WARP, (uint32_t)-32, R_TID, hp_ctrl_safe());
  p[n++] = hp_shr_imm(R_G, R_LANE, 2, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_L, R_G, (uint32_t)-4, R_LANE, hp_ctrl_safe());
  /* The warp's two coordinates, by the same shift-and-subtract: WARPS_N is a
   * power of two and this hardware has no integer divide. R_WARP becomes the
   * ROW coordinate in place. */
  {
    unsigned lgN = 0;
    while ((1u << lgN) < HMMA_WARPS_N) lgN++;
    p[n++] = hp_shr_imm(R_TMP, R_WARP, lgN, hp_ctrl_safe());          /* warpM */
    p[n++] = hp_imad_imm(R_WARPN, R_TMP, (uint32_t)-(int)HMMA_WARPS_N,
                         R_WARP, hp_ctrl_safe());                     /* warpN */
    p[n++] = hp_iadd3_imm(R_WARP, R_TMP, 0, hp_ctrl_safe());
  }

  /*
   * A's row bases, one per row fragment, WITHOUT k.
   *
   * idxA[tm] = batch*M*K + (ctaid.x*BM + 16*tm + g)*K + 2l, so the loop adds
   * only k. The second row of a fragment is eight rows further on, which is a
   * constant 8*K elements and is folded into the load's immediate offset when
   * it fits and into a second address when it does not.
   */
  p[n++] = hp_imad_imm(R_TMP, R_CTA_M, BM, R_G, hp_ctrl_safe());  /* row0 */
  p[n++] = hp_imad_imm(R_TMP, R_WARP, MMA_M * HMMA_TM, R_TMP, hp_ctrl_safe());
  for (unsigned tm = 0; tm < HMMA_TM; tm++) {
    p[n++] = hp_iadd3_imm(R_TMP2, R_TMP, (int)(MMA_M * tm), hp_ctrl_safe());
    if (transposedA) {
      /* A is [K,M]: element (row, k) is k*M + row, so the base holds only the
       * ROW and the k loop adds (k + 2l)*M — the same shape as an
       * untransposed B, and addressed by the same immediate offsets. */
      p[n++] = hp_iadd3_imm(R_IDX + tm, R_TMP2, 0, hp_ctrl_safe());
    } else {
      p[n++] = hp_imad_imm(R_IDX + tm, R_TMP2, K, HP_RZ, hp_ctrl_safe());
      p[n++] = hp_imad_imm(R_TMP2, R_L, 2, R_IDX + tm, hp_ctrl_safe());
      p[n++] = hp_iadd3_imm(R_IDX + tm, R_TMP2, 0, hp_ctrl_safe());
    }
  }
  /* The batch plane, added once to every A base. */
  p[n++] = hp_imad_imm(R_TMP, R_BATCH, M * K, HP_RZ, hp_ctrl_safe());
  for (unsigned tm = 0; tm < HMMA_TM; tm++)
    p[n++] = hp_iadd3_reg(R_IDX + tm, R_IDX + tm, R_TMP, hp_ctrl_safe());

  /*
   * B's column bases. Transposed B is [N,K] and its element (k, col) is
   * col*K + k — the same shape as A's, so the k loop adds k to it directly.
   * Untransposed B is [K,N] and the element is k*N + col, so the base holds
   * only the column and the loop adds (k + 2l)*N.
   */
  const unsigned RB = R_IDX + HMMA_TM;
  p[n++] = hp_imad_imm(R_TMP, R_CTA_N, BN, R_G, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_TMP, R_WARPN, WN, R_TMP, hp_ctrl_safe());
  for (unsigned tn = 0; tn < HMMA_TN; tn++) {
    p[n++] = hp_iadd3_imm(RB + tn, R_TMP, (int)(MMA_N * tn), hp_ctrl_safe());
    if (transposedB) {
      p[n++] = hp_imad_imm(RB + tn, RB + tn, K, HP_RZ, hp_ctrl_safe());
      p[n++] = hp_imad_imm(RB + tn, R_L, 2, RB + tn, hp_ctrl_safe());
    }
  }
  {
    const unsigned plane = transposedB ? N * K : K * N;
    p[n++] = hp_imad_imm(R_TMP, R_BATCH, plane, HP_RZ, hp_ctrl_safe());
    for (unsigned tn = 0; tn < HMMA_TN; tn++)
      p[n++] = hp_iadd3_reg(RB + tn, RB + tn, R_TMP, hp_ctrl_safe());
  }

  /* Clear the accumulator. */
  for (unsigned i = 0; i < 4u * HMMA_TM * HMMA_TN; i++)
    p[n++] = hp_mov_imm(R_ACC + i, 0, hp_ctrl_safe());

  p[n++] = hp_mov_imm(R_K, 0, hp_ctrl_safe());
  const unsigned loop_top = n;

  /*
   * ---- one k-step: 16 of K, for the whole tile ----
   *
   * Every load first, then one wait, then every conversion, then every HMMA.
   * The ordering is the point: a fragment-at-a-time version pays the global
   * latency once per fragment, and there are TM+TN of them.
   */
  if (transposedA) {
    /* (k + 2l) * M, shared by every row fragment — the mirror of R_KN. */
    p[n++] = hp_imad_imm(R_TMP, R_L, 2, R_K, hp_ctrl_safe());
    p[n++] = hp_imad_imm(R_KM, R_TMP, M, HP_RZ, hp_ctrl_safe());
  }
  for (unsigned tm = 0; tm < HMMA_TM; tm++) {
    /*
     * One pair for both rows of the fragment: row g+8 is a constant 8*K
     * elements on, which goes in the load's immediate offset rather than in a
     * second address. That is what keeps this at TM+TN pairs.
     */
    const unsigned a = ADDR_A(tm);
    if (transposedA) {
      /* (k + 2l)*M + row. The k pair is M elements apart and the +8 pair 8M
       * on, so the four elements of a fragment register pair are immediate
       * offsets exactly as the untransposed B is. The second ROW of the
       * fragment is 8 further along the row axis, which is +8 elements. */
      p[n++] = hp_iadd3_reg(R_TMP, R_IDX + tm, R_KM, hp_ctrl_safe());
      p[n++] = hp_imad_wide_const(a, R_TMP, R_ESIZE, 0,
                                  HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
      const unsigned m1 = M * 4u, m8 = 8u * M * 4u, m9 = 9u * M * 4u;
      p[n++] = hp_ldg(TEMP_A(tm, 0), a, 0, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = hp_ldg(TEMP_A(tm, 1), a, m1, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = hp_ldg(TEMP_A(tm, 2), a, m8, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = hp_ldg(TEMP_A(tm, 3), a, m9, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = hp_ldg(TEMP_A(tm, 4), a, 32, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = hp_ldg(TEMP_A(tm, 5), a, m1 + 32, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = hp_ldg(TEMP_A(tm, 6), a, m8 + 32, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = hp_ldg(TEMP_A(tm, 7), a, m9 + 32, hp_ctrl_setbar(BAR_LOAD));
    } else {
      const unsigned r1 = 32u * K;
      p[n++] = hp_iadd3_reg(R_TMP, R_IDX + tm, R_K, hp_ctrl_safe());
      p[n++] = hp_imad_wide_const(a, R_TMP, R_ESIZE, 0,
                                  HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
      p[n++] = hp_ldg(TEMP_A(tm, 0), a, 0, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = hp_ldg(TEMP_A(tm, 1), a, 4, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = hp_ldg(TEMP_A(tm, 2), a, 32, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = hp_ldg(TEMP_A(tm, 3), a, 36, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = hp_ldg(TEMP_A(tm, 4), a, r1 + 0, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = hp_ldg(TEMP_A(tm, 5), a, r1 + 4, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = hp_ldg(TEMP_A(tm, 6), a, r1 + 32, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = hp_ldg(TEMP_A(tm, 7), a, r1 + 36, hp_ctrl_setbar(BAR_LOAD));
    }
  }

  if (!transposedB) {
    /* (k + 2l) * N, shared by every column fragment. */
    p[n++] = hp_imad_imm(R_TMP, R_L, 2, R_K, hp_ctrl_safe());
    p[n++] = hp_imad_imm(R_KN, R_TMP, N, HP_RZ, hp_ctrl_safe());
  }
  for (unsigned tn = 0; tn < HMMA_TN; tn++) {
    const unsigned b = ADDR_B(tn);
    if (transposedB) {
      p[n++] = hp_iadd3_reg(R_TMP, RB + tn, R_K, hp_ctrl_safe());
      p[n++] = hp_imad_wide_const(b, R_TMP, R_ESIZE, 0,
                                  HERMES_CBUF0_PARAM_N(2), hp_ctrl_safe());
      p[n++] = hp_ldg(TEMP_B(tn, 0), b, 0, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = hp_ldg(TEMP_B(tn, 1), b, 4, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = hp_ldg(TEMP_B(tn, 2), b, 32, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = hp_ldg(TEMP_B(tn, 3), b, 36, hp_ctrl_setbar(BAR_LOAD));
    } else {
      p[n++] = hp_iadd3_reg(R_TMP, RB + tn, R_KN, hp_ctrl_safe());
      p[n++] = hp_imad_wide_const(b, R_TMP, R_ESIZE, 0,
                                  HERMES_CBUF0_PARAM_N(2), hp_ctrl_safe());
      p[n++] = hp_ldg(TEMP_B(tn, 0), b, 0, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = hp_ldg(TEMP_B(tn, 1), b, N * 4u, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = hp_ldg(TEMP_B(tn, 2), b, 8u * N * 4u, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = hp_ldg(TEMP_B(tn, 3), b, 9u * N * 4u, hp_ctrl_setbar(BAR_LOAD));
    }
  }

  /*
   * f32 -> f16, in the layout the instruction wants.
   *
   * F2FP packs two floats into one register, low half first, and the fragment
   * registers are exactly pairs of adjacent k. The first conversion carries the
   * wait for every load issued above; the rest follow it in order.
   */
  {
    int first = 1;
    for (unsigned tm = 0; tm < HMMA_TM; tm++) {
      const unsigned a = AFRAG_OF(tm);
      const hp_control c0 = first ? hp_ctrl_wait(BAR_LOAD) : hp_ctrl_safe();
      first = 0;
      p[n++] = hp_f2fp_pack(a + 0, TEMP_A(tm, 0), TEMP_A(tm, 1), c0);
      p[n++] = hp_f2fp_pack(a + 1, TEMP_A(tm, 4), TEMP_A(tm, 5), hp_ctrl_safe());
      p[n++] = hp_f2fp_pack(a + 2, TEMP_A(tm, 2), TEMP_A(tm, 3), hp_ctrl_safe());
      p[n++] = hp_f2fp_pack(a + 3, TEMP_A(tm, 6), TEMP_A(tm, 7), hp_ctrl_safe());
    }
    for (unsigned tn = 0; tn < HMMA_TN; tn++) {
      const unsigned b = BFRAG_OF(tn);
      const hp_control c0 = first ? hp_ctrl_wait(BAR_LOAD) : hp_ctrl_safe();
      first = 0;
      p[n++] = hp_f2fp_pack(b + 0, TEMP_B(tn, 0), TEMP_B(tn, 1), c0);
      p[n++] = hp_f2fp_pack(b + 1, TEMP_B(tn, 2), TEMP_B(tn, 3), hp_ctrl_safe());
    }
  }

  for (unsigned tm = 0; tm < HMMA_TM; tm++)
    for (unsigned tn = 0; tn < HMMA_TN; tn++) {
      const unsigned acc = ACC_OF(tm, tn);
      p[n++] = hp_hmma(acc, AFRAG_OF(tm), BFRAG_OF(tn), acc,
                       hp_ctrl_setbar(BAR_MMA));
    }

  p[n++] = hp_iadd3_imm(R_K, R_K, (int)MMA_K, hp_ctrl_safe());
  p[n++] = hp_isetp_gt_imm(P_DONE, R_K, K - 1u, hp_ctrl_safe());
  {
    const int back = -(int)((n + 1u - loop_top) * INSTR_BYTES);
    p[n] = hp_predicated(hp_bra(back, hp_ctrl_branch()), P_DONE, 1);
    n++;
  }

  /*
   * ---- the epilogue ----
   *
   * d0,d1 are row g at columns 2l and 2l+1; d2,d3 are row g+8 at the same two.
   * So each accumulator fragment is two pairs of adjacent floats at two rows
   * eight apart, which is two addresses and four stores.
   *
   * The address pairs ROTATE. A global store holds its address registers until
   * the memory pipe accepts them, which is not when it issued — recomputing a
   * pair the previous store is still reading corrupts that store, and it does
   * so as a function of how much traffic is queued, which is the worst way for
   * a fault to present. Two pairs with a read barrier each keeps the stores
   * pipelined and the reuse behind a wait.
   */
  p[n++] = hp_imad_imm(R_TMP, R_CTA_M, BM, R_G, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_TMP, R_WARP, MMA_M * HMMA_TM, R_TMP, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_OBASE, R_TMP, N, HP_RZ, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_TMP, R_CTA_N, BN, R_OBASE, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_TMP, R_WARPN, WN, R_TMP, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_OBASE, R_L, 2, R_TMP, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_TMP, R_BATCH, M * N, HP_RZ, hp_ctrl_safe());
  p[n++] = hp_iadd3_reg(R_OBASE, R_OBASE, R_TMP, hp_ctrl_safe());

  {
    /*
     * FOUR ADDRESS PAIRS, ROTATED, WITH A READ BARRIER EACH — and the reason is
     * measured, not precautionary.
     *
     * The first version of this loop used ONE pair per accumulator fragment and
     * recomputed it between the fragment's two rows. That is the write-after-
     * read hazard matmul.c records four times: a global store holds its address
     * registers until the memory pipe accepts them, which is not when it
     * issued, so recomputing the pair corrupts the store still reading it.
     *
     * It PASSED at a 1x1 tile and failed at 2x4, which is the signature: with
     * two stores in flight the pipe accepts them immediately, and with thirty-
     * two queued it does not. The corruption was not uniform either — output
     * rows with an even lane group were right and odd ones wrong — because
     * which store is still holding its operands depends on queue depth, and
     * that is the worst way for a fault to present. It is also exactly what
     * matmul.c warned would happen: "N=1025 failed while N=1920 with the same
     * chunk count and the same guard passed."
     *
     * A pair is now reused only after four intervening row-stores AND behind a
     * wait on the barrier the last of them set. The pairs live in the k-loop's
     * staging registers, which are dead by here.
     */
    const unsigned pairs[4] = {EPI_ADDR(0), EPI_ADDR(1), EPI_ADDR(2), EPI_ADDR(3)};
    const unsigned bars[4] = {BAR_ST0, BAR_ST1, BAR_ST2, BAR_ST3};
    unsigned unit = 0;
    for (unsigned tm = 0; tm < HMMA_TM; tm++)
      for (unsigned tn = 0; tn < HMMA_TN; tn++) {
        const unsigned acc = ACC_OF(tm, tn);
        const int base = (int)(MMA_M * tm * N + MMA_N * tn);
        /* Row g, then row g+8 — two independent units so neither shares an
         * address register with the other. */
        for (unsigned half = 0; half < 2; half++) {
          const unsigned slot = unit & 3u;
          const unsigned addr = pairs[slot], bar = bars[slot];
          hp_control c;
          if (unit == 0) c = hp_ctrl_wait(BAR_MMA); /* the last HMMA's result */
          else if (unit >= 4) c = hp_ctrl_wait(bar); /* this pair's last store */
          else c = hp_ctrl_safe();
          p[n++] = hp_iadd3_imm(R_TMP, R_OBASE,
                                base + (int)(half ? 8u * N : 0u), c);
          p[n++] = hp_imad_wide_const(addr, R_TMP, R_ESIZE, 0,
                                      HERMES_CBUF0_PARAM_N(0), hp_ctrl_safe());
          p[n++] = hp_stg(addr, acc + 2u * half + 0u, 0, hp_ctrl_safe());
          p[n++] = hp_stg(addr, acc + 2u * half + 1u, 4, hp_ctrl_setread(bar));
          unit++;
        }
      }
  }

  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}
