/*
 * indexing.c — kernels whose whole content is where the data is, not what is
 * done to it.
 *
 * WHAT: transpose and embedding lookup. Neither performs any arithmetic on the
 * values it moves; both are entirely about computing two different addresses
 * for the same element.
 *
 * WHY THEY ARE TOGETHER: a bug in either is an INDEXING bug, and indexing bugs
 * have a signature the element-wise kernels do not -- they produce output that
 * is the right shape, the right magnitude, and drawn from the right set of
 * values, just in the wrong places. Nothing about the result looks wrong except
 * the arrangement. Their oracles are written to be sensitive to exactly that,
 * with inputs that have no symmetry to hide behind.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: no shared-memory staging for coalescing.
 * The transpose here reads along rows and writes down columns, which is the
 * uncoalesced arrangement and is slow. That is the version whose correctness
 * follows directly from the definition; a tiled one is a transformation that
 * needs this one to be checked against.
 */
#include "indexing.h"

enum {
  R_PLANE_ID = 16,
  R_H = 17,
  R_T = 18,
  R_B = 19,
  R_TMP2 = 20,
  R_MASK = 21,
  R_SRC2 = 22,
  R_DST2 = 23,
  /* The column-chunk loop, so a transpose is not capped at 1024 columns. */
  R_TID = 24,
  R_CHUNK = 25,
  /*
   * TWO value registers, alternated between chunks.
   *
   * A chunk loads into a register and stores from it, and the next chunk loads
   * into the same one. There is no write-after-read interlock on this hardware:
   * the store holds its operand until the pipe accepts it, and the next load can
   * overwrite the register first. The result is a transpose that disagrees with
   * ITSELF run to run — which is how it was found, by checksumming every
   * operation across two identical forward passes.
   *
   * The same hazard has now appeared three times: twice in normalize.c and once
   * in the staged matmul. Alternating removes it by construction rather than by
   * ordering, which is why the chunk loops below are UNROLLED — a runtime loop
   * cannot alternate a register.
   */
  R_VALUE_B = 26,
  /*
   * ...and the ADDRESS pairs too, which is the half that was missed first.
   *
   * Alternating only the value left the store's ADDRESS exposed: chunk t+1
   * recomputes R_OUT while chunk t's store has issued but not yet read its
   * operands, so the write lands somewhere else. It shows as 7% of elements
   * wrong and 89% of them varying run to run — a store that mostly goes to the
   * right place. Every register a store still needs has to alternate, not just
   * the obvious one.
   */
  R_ADDR_B = 28, /* R28:R29 */
  R_OUT_B = 30,  /* R30:R31 */
  R_ROW = 0,
  R_COL = 1,
  R_SRC_IDX = 2,
  R_DST_IDX = 3,
  R_ESIZE = 5,
  R_ADDR = 6,  /* R6:R7 */
  R_VALUE = 10,
  R_TABLE_ROW = 11,
  R_BATCH = 12,
  R_PLANE = 13,
  R_OUT = 14, /* R14:R15 */
};

#define BAR_ID 0
#define BAR_LOAD 1
#define P_COL 2   /* clear when this thread's column is inside `cols` */
#define P_CHUNK 3 /* set when the column-chunk loop has run its course */
#define INSTR_BYTES 16

/*
 * A block is at most 1024 threads, and `cols` is a MODEL DIMENSION.
 *
 * One thread per column made a transpose of anything wider than 1024 an invalid
 * launch — GR_EXCEPTION on the channel, asynchronously, so it surfaced at
 * whatever flushed next. A 105M-parameter model transposes weights that are
 * 1,728, 1,920 and 12,288 wide. Threads walk their columns in chunks instead,
 * exactly as matmul does, and the launch geometry stops depending on the shape.
 */
#define TRANSPOSE_MAX_THREADS 1024u

/* Threads a row-copy block runs — min(W, 1024). slice, cat and broadcast all put
 * one thread on each column, so all three stop at the same wall; a 105M model
 * concatenates 1,280-wide rows. */
unsigned pr_row_block(unsigned W) {
  return W < TRANSPOSE_MAX_THREADS ? W : TRANSPOSE_MAX_THREADS;
}

unsigned pr_transpose_block(unsigned cols) {
  return cols < TRANSPOSE_MAX_THREADS ? cols : TRANSPOSE_MAX_THREADS;
}

/*
 * transpose: out[c][r] = in[r][c], for an M x N input.
 *
 * Block x is the row and thread x the column, so neighbouring threads READ
 * neighbouring elements and WRITE elements N apart. The reverse assignment
 * would make the writes contiguous and the reads scattered. One of the two has
 * to be scattered -- that is what a transpose is -- and which one is a
 * performance question this kernel does not try to answer.
 */
unsigned pr_emit_transpose(hp_word *p, unsigned rows, unsigned cols) {
  unsigned n = 0;
  const unsigned BW = pr_transpose_block(cols);
  const unsigned chunks = BW ? (cols + BW - 1u) / BW : 1u;
  const int guard_col = chunks * BW > cols;

  p[n++] = hp_s2r(R_ROW, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_BATCH, HP_SR_CTAID_Y, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_PLANE, R_BATCH, rows * cols, HP_RZ,
                       hp_ctrl_wait(BAR_ID));

  /*
   * UNROLLED over the column chunks, alternating the value register.
   *
   * One thread per column capped a transpose at 1024 wide; threads cover the
   * row in chunks instead. The unrolling is not for speed — it is what lets
   * consecutive chunks use DIFFERENT value registers, which is the only thing
   * standing between chunk t+1's load and chunk t's store. See R_VALUE_B.
   */
  for (unsigned t = 0; t < chunks; t++) {
    const unsigned val = (t & 1u) ? R_VALUE_B : R_VALUE;
    const unsigned addr = (t & 1u) ? R_ADDR_B : R_ADDR;
    const unsigned out = (t & 1u) ? R_OUT_B : R_OUT;
    p[n++] = hp_iadd3_imm(R_COL, R_TID, t * BW, hp_ctrl_safe());
    if (guard_col)
      p[n++] = hp_isetp_gt_imm(P_COL, R_COL, cols - 1, hp_ctrl_safe());

    p[n++] = hp_imad_imm(R_SRC_IDX, R_ROW, cols, R_COL, hp_ctrl_safe());
    p[n++] = hp_iadd3_reg(R_SRC_IDX, R_SRC_IDX, R_PLANE, hp_ctrl_safe());
    p[n++] = hp_imad_imm(R_DST_IDX, R_COL, rows, R_ROW, hp_ctrl_safe());
    p[n++] = hp_iadd3_reg(R_DST_IDX, R_DST_IDX, R_PLANE, hp_ctrl_safe());

    p[n++] = hp_imad_wide_const(addr, R_SRC_IDX, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
    {
      /* An overhanging thread must touch memory at neither end: the load would
       * read the next row and the store would land in a live column. */
      hp_word ld = hp_ldg(val, addr, 0, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = guard_col ? hp_predicated(ld, P_COL, 1) : ld;
    }
    p[n++] = hp_imad_wide_const(out, R_DST_IDX, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM_N(0), hp_ctrl_safe());
    {
      hp_word st = hp_stg(out, val, 0, hp_ctrl_wait(BAR_LOAD));
      p[n++] = guard_col ? hp_predicated(st, P_COL, 1) : st;
    }
  }
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}

/*
 * slice: out[i] = in[offset + i*stride].
 *
 * One dimension, because that is what every slice reduces to once the shape is
 * flattened -- a start and a step through the source. A multi-dimensional slice
 * is this with an offset and stride the HOST computed from the shapes, which is
 * where that arithmetic belongs: it depends only on the shapes, so doing it per
 * thread would repeat one calculation across the whole tensor.
 *
 * Both arrive as raw integers in the constant bank rather than as immediates,
 * so one generated kernel serves every slice of a given rank. Baking them in
 * would mean regenerating and reassembling for every new offset, and unlike a
 * matrix dimension -- which changes rarely and is worth specialising for -- a
 * slice offset can change on every call.
 */
unsigned pr_emit_slice(hp_word *p) {
  unsigned n = 0;
  p[n++] = hp_s2r(R_ROW, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_COL, HP_SR_TID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_safe());
  p[n++] = hp_imad_const(R_DST_IDX, R_ROW, 0, HERMES_CBUF0_NTID_X, R_COL,
                         hp_ctrl_wait(BAR_ID));

  /* src = offset + dst*stride, which is one multiply-add with both operands
   * from the bank. */
  p[n++] = hp_mov_const(R_TABLE_ROW, 0, HERMES_CBUF0_SCALAR_N(0),
                        hp_ctrl_safe());
  p[n++] = hp_mov_const(R_VALUE, 0, HERMES_CBUF0_SCALAR_N(1), hp_ctrl_safe());
  p[n++] = hp_imad_const(R_SRC_IDX, R_DST_IDX, 0, HERMES_CBUF0_SCALAR_N(1),
                         R_TABLE_ROW, hp_ctrl_safe());

  p[n++] = hp_imad_wide_const(R_ADDR, R_SRC_IDX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
  p[n++] = hp_ldg(R_VALUE, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));
  p[n++] = hp_imad_wide_const(R_OUT, R_DST_IDX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(0), hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_VALUE, 0, hp_ctrl_wait(BAR_LOAD));
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}

/*
 * embedding: out[i][d] = table[ids[i]][d].
 *
 * Block x is the token position i, thread x the feature d. The token id is
 * loaded from the second input buffer and used to index the first, so this is
 * the first kernel here whose ADDRESS depends on memory rather than only on the
 * thread's coordinates.
 *
 * That dependency is why the row load and the value load cannot share a
 * barrier the way matmul's two operands do: the second address cannot be
 * computed until the first load has landed. The wait is not conservatism, it is
 * the data dependency.
 *
 * The ids are read as raw 32-bit words and used directly as an integer, which
 * is what they are -- the test writes them as integers rather than as floats
 * that happen to be whole numbers, because a float bit pattern used as an index
 * would address somewhere absurd and the failure would be a fault rather than
 * a wrong answer.
 */
unsigned pr_emit_embedding(hp_word *p, unsigned dim) {
  unsigned n = 0;
  p[n++] = hp_s2r(R_ROW, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_COL, HP_SR_TID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_safe());

  /* ids[i], from the second input. */
  p[n++] = hp_imad_wide_const(R_ADDR, R_ROW, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(2), hp_ctrl_wait(BAR_ID));
  p[n++] = hp_ldg(R_TABLE_ROW, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));

  /* table[id][d] -- the address that depends on the load above. */
  p[n++] = hp_imad_imm(R_SRC_IDX, R_TABLE_ROW, dim, R_COL,
                       hp_ctrl_wait(BAR_LOAD));
  p[n++] = hp_imad_wide_const(R_ADDR, R_SRC_IDX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
  p[n++] = hp_ldg(R_VALUE, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));

  /* out[i][d] -- contiguous, unlike the gather that fed it. */
  p[n++] = hp_imad_imm(R_DST_IDX, R_ROW, dim, R_COL, hp_ctrl_safe());
  p[n++] = hp_imad_wide_const(R_OUT, R_DST_IDX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(0), hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_VALUE, 0, hp_ctrl_wait(BAR_LOAD));
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}

/*
 * permute: out[b][h][t][d] = in[b][t][h][d].
 *
 * WHY A KERNEL AT ALL, when transpose already has one: that kernel swaps the
 * LAST TWO axes, and attention swaps the middle two. The host did it instead —
 * and a host permute must READ device memory, so it drains the queue as well as
 * costing the copy. At batch 128 that was 75 ms a step, a quarter of the model,
 * for an operation that performs no arithmetic. It is also one of the reads
 * that stops tensors moving to video memory, where the GPU reads at ~448 GB/s
 * instead of the 19.7 measured across PCIe.
 *
 * THE CONSTRAINT THAT MAKES IT CHEAP: T, H and D are powers of two in every
 * shape this model produces, so decomposing the plane index is shifts and masks
 * rather than division — of which sm_86 has no integer form at all. The caller
 * checks that and keeps the host path for anything else; a kernel that quietly
 * did the wrong thing on an odd shape would be worse than a slow one.
 *
 * One block per (b,t,h) plane over the block's Y index, one thread per feature.
 * Neighbouring threads read and write neighbouring elements — both sides
 * coalesced, which is the property the two-transpose decomposition could not
 * have, since one of its halves is always strided.
 */
unsigned pr_emit_permute(hp_word *p, unsigned T, unsigned H, unsigned D) {
  unsigned n = 0;
  unsigned lgH = 0, lgT = 0;
  while ((1u << lgH) < H) lgH++;
  while ((1u << lgT) < T) lgT++;

  p[n++] = hp_s2r(R_PLANE_ID, HP_SR_CTAID_Y, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_COL, HP_SR_TID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_safe());

  /* h = plane & (H-1) */
  p[n++] = hp_mov_imm(R_MASK, H - 1u, hp_ctrl_safe());
  p[n++] = hp_lop3(R_H, R_PLANE_ID, R_MASK, 0xc0, hp_ctrl_wait(BAR_ID));
  /* t = (plane >> lgH) & (T-1) */
  p[n++] = hp_shr_imm(R_TMP2, R_PLANE_ID, lgH, hp_ctrl_safe());
  p[n++] = hp_mov_imm(R_MASK, T - 1u, hp_ctrl_safe());
  p[n++] = hp_lop3(R_T, R_TMP2, R_MASK, 0xc0, hp_ctrl_safe());
  /* b = plane >> (lgH + lgT) */
  p[n++] = hp_shr_imm(R_B, R_PLANE_ID, lgH + lgT, hp_ctrl_safe());

  /* src = (((b*T) + t)*H + h)*D + d */
  p[n++] = hp_imad_imm(R_SRC2, R_B, T, R_T, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_SRC2, R_SRC2, H, R_H, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_SRC2, R_SRC2, D, R_COL, hp_ctrl_safe());
  /* dst = (((b*H) + h)*T + t)*D + d */
  p[n++] = hp_imad_imm(R_DST2, R_B, H, R_H, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_DST2, R_DST2, T, R_T, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_DST2, R_DST2, D, R_COL, hp_ctrl_safe());

  p[n++] = hp_imad_wide_const(R_ADDR, R_SRC2, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
  p[n++] = hp_ldg(R_VALUE, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));
  p[n++] = hp_imad_wide_const(R_OUT, R_DST2, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(0), hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_VALUE, 0, hp_ctrl_wait(BAR_LOAD));
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}

/*
 * slice along the LAST axis: out[r][c] = in[r][start + c].
 *
 * The one-dimensional slice kernel above cannot express this — its source index
 * is affine in the destination index, and here the stride between output rows
 * (W) differs from the stride between source rows (srcW), so the mapping is not
 * affine in a flat index. A two-dimensional launch supplies the row instead of
 * computing it, and then both indices are one multiply-add.
 *
 * `start` comes from the constant bank rather than being baked in, because a
 * slice offset changes per call — qkv takes three different ones from the same
 * tensor — and baking it would regenerate the program for each.
 *
 * No power-of-two requirement: nothing is decomposed, so nothing is divided.
 */
unsigned pr_emit_slice_rows(hp_word *p, unsigned W, unsigned srcW) {
  unsigned n = 0;
  const unsigned BW = pr_row_block(W);
  const unsigned chunks = BW ? (W + BW - 1u) / BW : 1u;
  const int guard_col = chunks * BW > W;

  p[n++] = hp_s2r(R_PLANE_ID, HP_SR_CTAID_Y, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_wait(BAR_ID));

  /* Unrolled over the column chunks, alternating the value AND both address
   * pairs — see R_VALUE_B. A runtime loop cannot alternate a register, and
   * without alternating them the next chunk clobbers a store still reading. */
  for (unsigned t = 0; t < chunks; t++) {
    const unsigned val = (t & 1u) ? R_VALUE_B : R_VALUE;
    const unsigned addr = (t & 1u) ? R_ADDR_B : R_ADDR;
    const unsigned out = (t & 1u) ? R_OUT_B : R_OUT;
    p[n++] = hp_iadd3_imm(R_COL, R_TID, t * BW, hp_ctrl_safe());
    if (guard_col)
      p[n++] = hp_isetp_gt_imm(P_COL, R_COL, W - 1, hp_ctrl_safe());

    /* dst = row*W + c ; src = row*srcW + start + c */
    p[n++] = hp_imad_imm(R_DST2, R_PLANE_ID, W, R_COL, hp_ctrl_safe());
    p[n++] = hp_mov_const(R_MASK, 0, HERMES_CBUF0_SCALAR_N(0), hp_ctrl_safe());
    p[n++] = hp_iadd3_reg(R_TMP2, R_COL, R_MASK, hp_ctrl_safe());
    p[n++] = hp_imad_imm(R_SRC2, R_PLANE_ID, srcW, R_TMP2, hp_ctrl_safe());

    p[n++] = hp_imad_wide_const(addr, R_SRC2, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
    {
      hp_word ld = hp_ldg(val, addr, 0, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = guard_col ? hp_predicated(ld, P_COL, 1) : ld;
    }
    p[n++] = hp_imad_wide_const(out, R_DST2, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM_N(0), hp_ctrl_safe());
    {
      hp_word st = hp_stg(out, val, 0, hp_ctrl_wait(BAR_LOAD));
      p[n++] = guard_col ? hp_predicated(st, P_COL, 1) : st;
    }
  }
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}

/*
 * broadcast, in the two forms the model actually asks for.
 *
 * mode 0, TILE: a vector repeated down the rows — [C] to [B,T,C], which is
 *   every bias and every norm weight. The source index is the COLUMN alone.
 * mode 1, ROW: one value per row spread across it — [B,T,1] to [B,T,C], which
 *   is every mean and reciprocal-deviation in a normalisation. The source index
 *   is the ROW alone.
 *
 * Both are trivial once the launch supplies the row, which is the whole reason
 * this is two kernels of ten instructions rather than one that decomposes a
 * flat index against a stride array. A general broadcast would need that, and
 * would need division; these two cover the model and cost nothing.
 */
unsigned pr_emit_broadcast(hp_word *p, unsigned mode, unsigned W) {
  unsigned n = 0;
  const unsigned BW = pr_row_block(W);
  const unsigned chunks = BW ? (W + BW - 1u) / BW : 1u;
  const int guard_col = chunks * BW > W;

  p[n++] = hp_s2r(R_PLANE_ID, HP_SR_CTAID_Y, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_wait(BAR_ID));

  /* Unrolled over the column chunks, alternating the value AND both address
   * pairs — see R_VALUE_B. A runtime loop cannot alternate a register, and
   * without alternating them the next chunk clobbers a store still reading. */
  for (unsigned t = 0; t < chunks; t++) {
    const unsigned val = (t & 1u) ? R_VALUE_B : R_VALUE;
    const unsigned addr = (t & 1u) ? R_ADDR_B : R_ADDR;
    const unsigned out = (t & 1u) ? R_OUT_B : R_OUT;
    p[n++] = hp_iadd3_imm(R_COL, R_TID, t * BW, hp_ctrl_safe());
    if (guard_col)
      p[n++] = hp_isetp_gt_imm(P_COL, R_COL, W - 1, hp_ctrl_safe());

    p[n++] = hp_imad_imm(R_DST2, R_PLANE_ID, W, R_COL, hp_ctrl_safe());
    /* The source is one coordinate or the other; that IS the broadcast. */
    p[n++] = hp_imad_imm(R_SRC2, mode ? R_PLANE_ID : R_COL, 1, HP_RZ,
                         hp_ctrl_safe());

    p[n++] = hp_imad_wide_const(addr, R_SRC2, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
    {
      hp_word ld = hp_ldg(val, addr, 0, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = guard_col ? hp_predicated(ld, P_COL, 1) : ld;
    }
    p[n++] = hp_imad_wide_const(out, R_DST2, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM_N(0), hp_ctrl_safe());
    {
      hp_word st = hp_stg(out, val, 0, hp_ctrl_wait(BAR_LOAD));
      p[n++] = guard_col ? hp_predicated(st, P_COL, 1) : st;
    }
  }
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}

/*
 * cat along the LAST axis: out[r][start + c] = in[r][c].
 *
 * The mirror of pr_emit_slice_rows, and it exists for the same reason: the
 * output row stride differs from the input's, so the mapping is not affine in a
 * flat index and a one-dimensional kernel cannot express it. Concatenating N
 * tensors is N launches, one per source, each writing its own column range —
 * which is cheaper than it sounds, because the alternative was a host copy that
 * drained the queue and cost 3.4 ms a call at batch 128.
 *
 * `start` is the destination offset and arrives in the constant bank, so one
 * program serves every piece of every concatenation.
 */
unsigned pr_emit_cat_rows(hp_word *p, unsigned W, unsigned dstW) {
  unsigned n = 0;
  const unsigned BW = pr_row_block(W);
  const unsigned chunks = BW ? (W + BW - 1u) / BW : 1u;
  const int guard_col = chunks * BW > W;

  p[n++] = hp_s2r(R_PLANE_ID, HP_SR_CTAID_Y, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_wait(BAR_ID));

  /* Unrolled over the column chunks, alternating the value AND both address
   * pairs — see R_VALUE_B. A runtime loop cannot alternate a register, and
   * without alternating them the next chunk clobbers a store still reading. */
  for (unsigned t = 0; t < chunks; t++) {
    const unsigned val = (t & 1u) ? R_VALUE_B : R_VALUE;
    const unsigned addr = (t & 1u) ? R_ADDR_B : R_ADDR;
    const unsigned out = (t & 1u) ? R_OUT_B : R_OUT;
    p[n++] = hp_iadd3_imm(R_COL, R_TID, t * BW, hp_ctrl_safe());
    if (guard_col)
      p[n++] = hp_isetp_gt_imm(P_COL, R_COL, W - 1, hp_ctrl_safe());

    /* src = row*W + c ; dst = row*dstW + start + c */
    p[n++] = hp_imad_imm(R_SRC2, R_PLANE_ID, W, R_COL, hp_ctrl_safe());
    p[n++] = hp_mov_const(R_MASK, 0, HERMES_CBUF0_SCALAR_N(0), hp_ctrl_safe());
    p[n++] = hp_iadd3_reg(R_TMP2, R_COL, R_MASK, hp_ctrl_safe());
    p[n++] = hp_imad_imm(R_DST2, R_PLANE_ID, dstW, R_TMP2, hp_ctrl_safe());

    p[n++] = hp_imad_wide_const(addr, R_SRC2, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
    {
      hp_word ld = hp_ldg(val, addr, 0, hp_ctrl_setbar(BAR_LOAD));
      p[n++] = guard_col ? hp_predicated(ld, P_COL, 1) : ld;
    }
    p[n++] = hp_imad_wide_const(out, R_DST2, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM_N(0), hp_ctrl_safe());
    {
      hp_word st = hp_stg(out, val, 0, hp_ctrl_wait(BAR_LOAD));
      p[n++] = guard_col ? hp_predicated(st, P_COL, 1) : st;
    }
  }
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}
