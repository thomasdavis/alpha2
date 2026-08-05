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
/* The scatter issues its two loads together and needs to wait on them
 * separately: the id decides an ADDRESS, the gradient is the DATA. */
#define BAR_VALUE 4
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
 * pr_emit_embedding_scatter — the embedding gradient, straight.
 *
 *     dW[ids[i]][d] += g[i][d]
 *
 * It is the forward kernel with the load and the store exchanged: the gather
 * reads table[ids[row]][col] and writes a contiguous row, this reads a
 * contiguous row and writes to table[ids[row]][col]. The one difference that
 * matters is that the write must be an ATOMIC add, because the gather's
 * many-to-one mapping runs backwards as one-to-many: two tokens sharing a
 * vocabulary id write the same address, and on this hardware the loser of that
 * race does not fault, it silently wins.
 *
 * WHAT IT REPLACES: the same gradient computed as onehot^T @ g, where the
 * one-hot is built out of arithmetic through seven full-size elementwise passes
 * over a [tokens, vocab] tensor — 75 MB apiece at this model's batch — and then
 * multiplied out at 24 GFLOP to recover a table that is zero in all but
 * `tokens` of its rows. That form was chosen over a HOST loop, which it
 * comfortably beats; it was never measured against a device scatter, which
 * reads about 2 MB and needed only an atomic that the encoder did not have.
 *
 * The caller must ZERO dW first. This kernel only adds, and it visits exactly
 * the rows that appear in ids — every other row of the output is never
 * addressed, so whatever is in it survives.
 */
unsigned pr_emit_embedding_scatter(hp_word *p, unsigned dim) {
  unsigned n = 0;
  p[n++] = hp_s2r(R_ROW, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_COL, HP_SR_TID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_safe());

  /* ids[i] and this token's gradient row are INDEPENDENT loads, so they issue
   * together on separate barriers rather than one waiting for the other. Only
   * the destination address needs the id. */
  p[n++] = hp_imad_wide_const(R_ADDR, R_ROW, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(2), hp_ctrl_wait(BAR_ID));
  p[n++] = hp_ldg(R_TABLE_ROW, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));
  p[n++] = hp_imad_imm(R_DST_IDX, R_ROW, dim, R_COL, hp_ctrl_safe());
  p[n++] = hp_imad_wide_const(R_OUT, R_DST_IDX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
  p[n++] = hp_ldg(R_VALUE, R_OUT, 0, hp_ctrl_setbar(BAR_VALUE));

  p[n++] = hp_imad_imm(R_SRC_IDX, R_TABLE_ROW, dim, R_COL,
                       hp_ctrl_wait(BAR_LOAD));
  p[n++] = hp_imad_wide_const(R_ADDR, R_SRC_IDX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(0), hp_ctrl_safe());
  p[n++] = hp_red_add_f32(R_ADDR, R_VALUE, 0, hp_ctrl_wait(BAR_VALUE));
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
/*
 * TAKE h FROM THE GRID instead of decoding it, and H stops needing to be a
 * power of two.
 *
 * This decomposed one plane index into (b, t, h) with shifts and masks, so it
 * required H, T and D all to be powers of two -- and the caller's guard said
 * so. Every shape this model produces is, went the reasoning. A 10-HEAD model
 * is not: 105M is 640 wide over 10 heads, so H = 10, the guard failed, and
 * EVERY attention permute took the host fallback. That path must READ device
 * memory, which drains the queue, so the cost is not only the copy -- it is a
 * synchronisation in the middle of a step, times four per layer, times 18
 * layers.
 *
 * Nothing here ever divided by D; D is only ever a multiplier, so its
 * power-of-two guard was never needed either. That left H and T. Launching one
 * block per (t-major plane, h) pair -- h on the grid's X index, b*T+t on Y --
 * means h arrives already decoded and only T is divided. And the source index
 * needs no decode at all, because the Y index IS b*T+t:
 *
 *     src = (plane*H + h)*D + d
 *     dst = ((b*H + h)*T + t)*D + d
 *
 * So the requirement drops from "H, T and D all powers of two" to "T a power
 * of two" -- satisfied by any sequence length this model uses -- and the head
 * count becomes free.
 */
unsigned pr_permute_rows(unsigned T, unsigned D) {
  if (D == 0 || D > PR_MAX_BLOCK) return 1u;
  /*
   * D MUST BE A POWER OF TWO, because the kernel splits the thread index into
   * (row, feature) with a shift and sm_86 has no integer divide. Two of the ten
   * diff shapes have D = 5 and D = 3, and without this they came back wrong —
   * a reminder that a helper computing a launch parameter has to answer for
   * every shape the KERNEL will see, not only the ones the model uses.
   */
  if ((D & (D - 1u)) != 0u) return 1u;
  unsigned r = 1u;
  while (r * 2u * D <= PR_MAX_BLOCK && T % (r * 2u) == 0u) r *= 2u;
  return r;
}

unsigned pr_emit_permute(hp_word *p, unsigned T, unsigned H, unsigned D) {
  unsigned n = 0;
  const unsigned R = pr_permute_rows(T, D);
  unsigned lgD = 0;
  while ((1u << lgD) < D) lgD++;

  /*
   * ALL THREE INDICES COME FROM THE GRID — h on X, t on Y, b on Z — so nothing
   * is decoded and NOTHING NEEDS TO BE A POWER OF TWO.
   *
   * This used to pack (b, t) into the Y index and recover t with a mask and b
   * with a shift, which needs T to be a power of two. That guard sent the
   * REVERSE attention permute — [B,H,T,D] back to [B,T,H,D], where the axis in
   * the T position is the HEAD COUNT — down the host fallback, because 105M is
   * 640 wide over 10 heads. The fallback reads device memory, so it drains the
   * queue: measured at 105M seq 64 batch 4, 180 drains a step and 36.3 ms, the
   * largest single entry in the host profile and a third of the whole step.
   *
   * The same guard had already been narrowed once for the same reason (h moved
   * to the grid when H=10 broke it). Moving the last decoded index out too is
   * what makes the kernel shape-agnostic instead of shape-agnostic-so-far, and
   * it is possible only because the launch path grew a third grid dimension for
   * the tensor-core GEMM. The two axes are symmetric — permuting [B,X,Y,D] to
   * [B,Y,X,D] is this kernel with T=X and H=Y either way round — so one program
   * now serves both directions.
   */
  p[n++] = hp_s2r(R_H, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_T, HP_SR_CTAID_Y, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_B, HP_SR_CTAID_Z, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_COL, HP_SR_TID_X, hp_ctrl_setbar(BAR_ID));
  /* One wait covers all four S2Rs: the barrier counts outstanding writes, it
   * is not a flag. */
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_wait(BAR_ID));
  if (R > 1u) {
    /* The block covers R t-values: thread index splits into the row within the
     * block and the feature. D is a power of two here — pr_permute_rows only
     * returns more than one when it is — so this is a shift and a subtract, and
     * sm_86 has no integer divide. */
    p[n++] = hp_shr_imm(R_TMP2, R_COL, lgD, hp_ctrl_safe());
    p[n++] = hp_imad_imm(R_COL, R_TMP2, (uint32_t)-(int)D, R_COL, hp_ctrl_safe());
    p[n++] = hp_imad_imm(R_T, R_T, R, R_TMP2, hp_ctrl_safe());
  }

  /* src = ((b*T + t)*H + h)*D + d */
  p[n++] = hp_imad_imm(R_SRC2, R_B, T, R_T, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_SRC2, R_SRC2, H, R_H, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_SRC2, R_SRC2, D, R_COL, hp_ctrl_safe());
  /* dst = ((b*H + h)*T + t)*D + d */
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
