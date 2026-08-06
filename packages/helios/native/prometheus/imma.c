/*
 * imma.c — a single 16x8x32 INT8 tensor-core tile, to prove the IMMA encoding
 * AND fragment layout compute correctly on the GPU (sec5: a wrong encoding
 * faults, a wrong LAYOUT returns a plausible-wrong matrix, so this is a
 * known-answer test against a CPU reference, not a NaN check).
 *
 * D[16x8] s32 = A[16x32] s8 (row-major) * B[8x32] s8 (col-major, i.e. B[n][k]).
 * One warp, 32 threads. With g = lane>>2, l = lane&3, the m16n8k32 layout is
 * HMMA's m16n8k16 with FOUR s8 per register instead of two f16 and k doubled:
 *
 *   A  a0: row g,   k 4l..4l+3      a1: row g+8, k 4l..4l+3
 *      a2: row g,   k 4l+16..+19    a3: row g+8, k 4l+16..+19
 *   B  b0: col g,   k 4l..4l+3      b1: col g,   k 4l+16..+19
 *   C  d0,d1: row g,   cols 2l,2l+1   d2,d3: row g+8, cols 2l,2l+1
 *
 * Every fragment word is 4 contiguous s8 (one 32-bit load); the params are
 * slot 0 = D (s32), slot 1 = A (s8), slot 2 = B (s8, n-major).
 */
#include "imma.h"

#define BAR_ID 0
#define BAR_LOAD 1

enum {
  R_TID = 0,
  R_G = 1,      /* lane >> 2 */
  R_L = 2,      /* lane & 3 */
  R_ESIZE = 3,
  R_AIDX = 4,   /* g*32 + 4l — the A/B byte offset of this thread's k=0 group */
  R_TMP = 5,
  R_ADDR = 6,   /* R6:R7 */
  R_CIDX = 14,
  /* Fragment bases MUST be 4-register-aligned for the tensor core (an
   * unaligned base faults, errnotif 0x0d). A:8, B:12, C:16 all divide 4. */
  R_A0 = 8, R_A1 = 9, R_A2 = 10, R_A3 = 11,   /* A fragment, consecutive */
  R_B0 = 12, R_B1 = 13,                       /* B fragment, consecutive */
  R_C0 = 16, R_C1 = 17, R_C2 = 18, R_C3 = 19, /* C/D fragment, consecutive */
  R_OUT = 20,   /* R20:R21 */
};

unsigned pr_emit_imma_tile(hp_word *p) {
  unsigned n = 0;
  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_mov_imm(R_ESIZE, 1, hp_ctrl_wait(BAR_ID)); /* byte addressing */

  /* g = tid>>2, l = tid - g*4. */
  p[n++] = hp_shr_imm(R_G, R_TID, 2, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_L, R_G, (uint32_t)-4, R_TID, hp_ctrl_safe());

  /* A/B k=0 byte offset for this thread: g*32 + 4l. */
  p[n++] = hp_imad_imm(R_AIDX, R_G, 32, HP_RZ, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_AIDX, R_L, 4, R_AIDX, hp_ctrl_safe());

  /* A fragment from slot 1: rows g / g+8 (256 bytes apart), k 0..3 / 16..19. */
  p[n++] = hp_imad_wide_const(R_ADDR, R_AIDX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
  p[n++] = hp_ldg(R_A0, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));
  p[n++] = hp_ldg(R_A1, R_ADDR, 256, hp_ctrl_setbar(BAR_LOAD));
  p[n++] = hp_ldg(R_A2, R_ADDR, 16, hp_ctrl_setbar(BAR_LOAD));
  p[n++] = hp_ldg(R_A3, R_ADDR, 272, hp_ctrl_setbar(BAR_LOAD));

  /* B fragment from slot 2 (n-major): col g. Its own address register (R_OUT,
   * free until the store) so the in-flight A loads keep reading R_ADDR=slot 1. */
  p[n++] = hp_imad_wide_const(R_OUT, R_AIDX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(2), hp_ctrl_safe());
  p[n++] = hp_ldg(R_B0, R_OUT, 0, hp_ctrl_setbar(BAR_LOAD));
  p[n++] = hp_ldg(R_B1, R_OUT, 16, hp_ctrl_setbar(BAR_LOAD));

  /* C starts at zero. */
  p[n++] = hp_mov_imm(R_C0, 0, hp_ctrl_wait(BAR_LOAD));
  p[n++] = hp_mov_imm(R_C1, 0, hp_ctrl_safe());
  p[n++] = hp_mov_imm(R_C2, 0, hp_ctrl_safe());
  p[n++] = hp_mov_imm(R_C3, 0, hp_ctrl_safe());

  p[n++] = hp_imma_acc(R_C0, R_A0, R_B0, R_C0, hp_ctrl_safe());

  /* Store D (slot 0, s32): d0,d1 -> row g cols 2l,2l+1; d2,d3 -> row g+8. */
  p[n++] = hp_imad_imm(R_CIDX, R_G, 8, HP_RZ, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_CIDX, R_L, 2, R_CIDX, hp_ctrl_safe());
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_safe()); /* s32 store */
  p[n++] = hp_imad_wide_const(R_OUT, R_CIDX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(0), hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_C0, 0, hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_C1, 4, hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_C2, 256, hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_C3, 260, hp_ctrl_safe());
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}

#define BAR_MMA 2

/* Extra registers the multi-tile grid GEMM needs beyond the single-tile enum:
 * the block id and this block's tile origin (m0,n0), all persistent to the
 * store. R_ADDR(6:7)=A base, R_OUT(20:21)=B base then store address. */
enum {
  R_BID = 15, /* linearized block id (ctaid.x) */
  R_M0 = 22,  /* tile row origin = bx*16 */
  R_N0 = 23,  /* tile col origin = by*8 */
};

/*
 * pr_emit_imma_gemm — the verified tile grown into a real GEMM over both K and
 * the output grid. One warp per 16x8 output tile; the (M/16 x N/8) grid is
 * LINEARIZED into ctaid.x (there is no ctaid.y in this harness) and decoded as
 * by = bid >> log2(M/16), bx = bid - by*(M/16). A is [M][K] row-major s8, B is
 * [N][K] n-major s8, D is [M][N] s32.
 *
 * The addressing generalises the single tile: A's per-thread base is
 * (m0+g)*K + 4l, B's is (n0+g)*K + 4l, and the row-g+8 offset that was a fixed
 * +256 in the standalone [16][8] tile is 8*N elements in the [M][N] output.
 *
 * SERIALIZED ON PURPOSE (correctness first): every k-step fully completes
 * before the next. The IMMA reads and writes C (a RAW across steps) and the
 * next step's loads overwrite the A/B fragment registers the current IMMA still
 * reads (a WAR); BAR_MMA between steps covers both. The staged version hides
 * this latency by double-buffering — this one measures nothing but correct.
 */
unsigned pr_emit_imma_gemm(hp_word *p, unsigned M, unsigned N, unsigned K) {
  unsigned n = 0;
  const unsigned steps = K / 32u;
  const unsigned rowg8 = 8u * K;         /* byte offset of A row g+8 */
  const unsigned drowg8 = 8u * N * 4u;   /* byte offset of D row g+8 */
  const unsigned mtiles = M / 16u;
  unsigned lgM = 0;
  while ((1u << lgM) < mtiles) lgM++;

  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_BID, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_mov_imm(R_ESIZE, 1, hp_ctrl_wait(BAR_ID)); /* byte addressing */
  p[n++] = hp_shr_imm(R_G, R_TID, 2, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_L, R_G, (uint32_t)-4, R_TID, hp_ctrl_safe());

  /* Decode the tile: by = bid>>lgM, bx = bid - by*mtiles; m0 = bx*16, n0 = by*8.
   * R_TMP carries by, R_CIDX carries bx (both dead by the store, which rebuilds
   * R_CIDX). */
  p[n++] = hp_shr_imm(R_TMP, R_BID, lgM, hp_ctrl_safe());               /* by */
  p[n++] = hp_imad_imm(R_CIDX, R_TMP, (uint32_t)-(int)mtiles, R_BID,
                       hp_ctrl_safe());                                  /* bx */
  p[n++] = hp_imad_imm(R_M0, R_CIDX, 16, HP_RZ, hp_ctrl_safe());        /* m0 */
  p[n++] = hp_imad_imm(R_N0, R_TMP, 8, HP_RZ, hp_ctrl_safe());          /* n0 */

  /* A base byte offset = (m0+g)*K + 4l. */
  p[n++] = hp_iadd3_reg(R_AIDX, R_M0, R_G, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_AIDX, R_AIDX, K, HP_RZ, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_AIDX, R_L, 4, R_AIDX, hp_ctrl_safe());
  p[n++] = hp_imad_wide_const(R_ADDR, R_AIDX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
  /* B base byte offset = (n0+g)*K + 4l, into R_OUT (free until the store). */
  p[n++] = hp_iadd3_reg(R_TMP, R_N0, R_G, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_TMP, R_TMP, K, HP_RZ, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_TMP, R_L, 4, R_TMP, hp_ctrl_safe());
  p[n++] = hp_imad_wide_const(R_OUT, R_TMP, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(2), hp_ctrl_safe());

  /* C accumulator starts at zero (fixed-latency movs, retire during the loads). */
  p[n++] = hp_mov_imm(R_C0, 0, hp_ctrl_safe());
  p[n++] = hp_mov_imm(R_C1, 0, hp_ctrl_safe());
  p[n++] = hp_mov_imm(R_C2, 0, hp_ctrl_safe());
  p[n++] = hp_mov_imm(R_C3, 0, hp_ctrl_safe());

  for (unsigned ks = 0; ks < steps; ks++) {
    const unsigned kso = ks * 32u;
    /* First load waits the previous step's IMMA (BAR_MMA): it may only clobber
     * the A/B registers once that IMMA has consumed them, and C is written. */
    const hp_control aload0 =
        ks == 0 ? hp_ctrl_setbar(BAR_LOAD)
                : hp_ctrl_wait_setbar(BAR_MMA, BAR_LOAD);
    p[n++] = hp_ldg(R_A0, R_ADDR, kso + 0, aload0);
    p[n++] = hp_ldg(R_A1, R_ADDR, kso + rowg8, hp_ctrl_setbar(BAR_LOAD));
    p[n++] = hp_ldg(R_A2, R_ADDR, kso + 16, hp_ctrl_setbar(BAR_LOAD));
    p[n++] = hp_ldg(R_A3, R_ADDR, kso + rowg8 + 16, hp_ctrl_setbar(BAR_LOAD));
    p[n++] = hp_ldg(R_B0, R_OUT, kso + 0, hp_ctrl_setbar(BAR_LOAD));
    p[n++] = hp_ldg(R_B1, R_OUT, kso + 16, hp_ctrl_setbar(BAR_LOAD));
    /* IMMA: wait the loads (and, past step 0, the previous IMMA's C write),
     * then set BAR_MMA for the next step / the epilogue. */
    const hp_control immac =
        ks == 0 ? hp_ctrl_wait_setbar(BAR_LOAD, BAR_MMA)
                : (hp_control){15, 0, BAR_MMA, HP_NO_BARRIER,
                               (1u << BAR_LOAD) | (1u << BAR_MMA), 0};
    p[n++] = hp_imma_acc(R_C0, R_A0, R_B0, R_C0, immac);
  }

  /* Store D (slot 0, s32): element (m0+g)*N + n0 + 2l; row g+8 is 8*N on. Wait
   * the last IMMA. */
  p[n++] = hp_iadd3_reg(R_CIDX, R_M0, R_G, hp_ctrl_wait(BAR_MMA));
  p[n++] = hp_imad_imm(R_CIDX, R_CIDX, N, R_N0, hp_ctrl_safe()); /* (m0+g)*N+n0 */
  p[n++] = hp_imad_imm(R_CIDX, R_L, 2, R_CIDX, hp_ctrl_safe());  /* + 2l */
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_safe());
  p[n++] = hp_imad_wide_const(R_OUT, R_CIDX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(0), hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_C0, 0, hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_C1, 4, hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_C2, drowg8, hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_C3, drowg8 + 4, hp_ctrl_safe());
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}

unsigned pr_emit_imma_gemm_16x8(hp_word *p, unsigned K) {
  return pr_emit_imma_gemm(p, 16, 8, K);
}

/*
 * pr_emit_imma_shared_tile — the ONE mechanic the staged fast kernel adds that
 * the grid kernel never exercised: int8 fragments read from SHARED via LDS,
 * not from global. One warp, one 16x8 output tile over K.
 *
 * Each k-step: 32 threads cooperatively copy A[16][32] and B[8][32] from global
 * into a shared tile (TS=8 words/row = MMA_K/4, no pad), barrier, then LDS the
 * m16n8k32 fragments from shared (the SAME layout verified from global — a0/a2
 * row g, a1/a3 row g+8, k-halves a word apart) and IMMA-accumulate. LDS/STS use
 * .X4: the address register is a WORD index (scaled x4) and the offset is BYTES.
 *
 * Correctness-first: the staging is serialized (LDG setbar / STS wait) so an
 * in-flight LDG never has its address register (R_ADDR) recomputed under it —
 * the same clobber the tile hit. The staged kernel double-buffers this away.
 */
#define IMMA_TS 8u             /* shared tile stride, words per row (MMA_K/4) */
#define IMMA_AW (16u * IMMA_TS) /* A tile size in shared words */

enum {
  R_SHA = 22,   /* shared STORE word for the current staged word */
  R_SHF = 23,   /* A fragment base word (row g, word l) */
  R_LDW = 24,   /* the word in flight from global to shared */
  R_ROW = 25,   /* tile row being staged */
  R_COL = 26,   /* this thread's word column, tid&7 */
  R_SHFB = 27,  /* B fragment base word */
  R_TROW = 28,  /* tid>>3, this thread's row offset within a 4-row pass */
};

unsigned pr_emit_imma_shared_tile(hp_word *p, unsigned K) {
  unsigned n = 0;
  const unsigned steps = K / 32u;

  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_mov_imm(R_ESIZE, 1, hp_ctrl_wait(BAR_ID)); /* byte LDG */
  p[n++] = hp_shr_imm(R_G, R_TID, 2, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_L, R_G, (uint32_t)-4, R_TID, hp_ctrl_safe());
  p[n++] = hp_shr_imm(R_TROW, R_TID, 3, hp_ctrl_safe());              /* tid>>3 */
  p[n++] = hp_imad_imm(R_COL, R_TROW, (uint32_t)-8, R_TID, hp_ctrl_safe()); /* tid&7 */

  /* Fragment base WORDS: A row g word l; B col g word l, past A's tile. */
  p[n++] = hp_imad_imm(R_SHF, R_G, IMMA_TS, R_L, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_SHFB, R_G, IMMA_TS, R_L, hp_ctrl_safe());
  p[n++] = hp_iadd3_imm(R_SHFB, R_SHFB, IMMA_AW, hp_ctrl_safe());

  p[n++] = hp_mov_imm(R_C0, 0, hp_ctrl_safe());
  p[n++] = hp_mov_imm(R_C1, 0, hp_ctrl_safe());
  p[n++] = hp_mov_imm(R_C2, 0, hp_ctrl_safe());
  p[n++] = hp_mov_imm(R_C3, 0, hp_ctrl_safe());

  for (unsigned ks = 0; ks < steps; ks++) {
    const unsigned kso = ks * 32u;
    /* Stage A (16 rows, 4 rows per pass) then B (8 rows, 2 passes). Each word:
     * global byte = row*K + kso + col*4; shared word = [AW +] row*TS + col. */
    for (unsigned pass = 0; pass < 6; pass++) {
      const int isB = pass >= 4;
      const unsigned base4 = (isB ? pass - 4u : pass) * 4u;
      const unsigned param = isB ? 2u : 1u;
      const unsigned shbase = isB ? IMMA_AW : 0u;
      p[n++] = hp_iadd3_imm(R_ROW, R_TROW, (int)base4, hp_ctrl_safe());
      p[n++] = hp_imad_imm(R_TMP, R_ROW, K, HP_RZ, hp_ctrl_safe());   /* row*K */
      p[n++] = hp_iadd3_imm(R_TMP, R_TMP, (int)kso, hp_ctrl_safe());  /* + kso */
      p[n++] = hp_imad_imm(R_TMP, R_COL, 4, R_TMP, hp_ctrl_safe());   /* + col*4 */
      p[n++] = hp_imad_wide_const(R_ADDR, R_TMP, R_ESIZE, 0,
                                  HERMES_CBUF0_PARAM_N(param), hp_ctrl_safe());
      p[n++] = hp_ldg(R_LDW, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));
      /* shared store word = shbase + row*TS + col. */
      p[n++] = hp_imad_imm(R_SHA, R_ROW, IMMA_TS, R_COL, hp_ctrl_safe());
      if (shbase)
        p[n++] = hp_iadd3_imm(R_SHA, R_SHA, (int)shbase, hp_ctrl_safe());
      p[n++] = hp_sts(R_SHA, R_LDW, 0, hp_ctrl_wait(BAR_LOAD));
    }
    p[n++] = hp_bar_sync(hp_ctrl_safe());

    /* Fragments from shared (.X4: R_SHF/R_SHFB in words, offsets in bytes). Row
     * g+8 is 8*TS*4 bytes on; the k-half is (MMA_K/4/2)=4 words = 16 bytes on. */
    p[n++] = hp_lds(R_A0, R_SHF, 0, hp_ctrl_setbar(BAR_LOAD));
    p[n++] = hp_lds(R_A1, R_SHF, 8u * IMMA_TS * 4u, hp_ctrl_setbar(BAR_LOAD));
    p[n++] = hp_lds(R_A2, R_SHF, 16u, hp_ctrl_setbar(BAR_LOAD));
    p[n++] = hp_lds(R_A3, R_SHF, 8u * IMMA_TS * 4u + 16u, hp_ctrl_setbar(BAR_LOAD));
    p[n++] = hp_lds(R_B0, R_SHFB, 0, hp_ctrl_setbar(BAR_LOAD));
    p[n++] = hp_lds(R_B1, R_SHFB, 16u, hp_ctrl_setbar(BAR_LOAD));

    /* Every IMMA waits its fragment LDS and sets BAR_MMA; the C RAW across
     * k-steps is ordered by the post-multiply bar_sync (which waits BAR_MMA),
     * so no per-IMMA wait on the previous C write is needed. */
    p[n++] = hp_imma_acc(R_C0, R_A0, R_B0, R_C0,
                         hp_ctrl_wait_setbar(BAR_LOAD, BAR_MMA));
    /* The next k-step's staging STS must not overwrite the tile until this
     * IMMA has read its fragments — a full barrier after the multiply. */
    p[n++] = hp_bar_sync(hp_ctrl_wait(BAR_MMA));
  }

  /* Store D[16][8] (slot 0): same fixed offsets as the tile. */
  p[n++] = hp_imad_imm(R_CIDX, R_G, 8, HP_RZ, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_CIDX, R_L, 2, R_CIDX, hp_ctrl_safe());
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_safe());
  p[n++] = hp_imad_wide_const(R_OUT, R_CIDX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(0), hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_C0, 0, hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_C1, 4, hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_C2, 256, hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_C3, 260, hp_ctrl_safe());
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}
