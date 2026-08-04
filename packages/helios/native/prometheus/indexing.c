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
  R_ROW = 0,
  R_COL = 1,
  R_SRC_IDX = 2,
  R_DST_IDX = 3,
  R_ESIZE = 5,
  R_ADDR = 6,  /* R6:R7 */
  R_VALUE = 10,
  R_TABLE_ROW = 11,
  R_OUT = 14, /* R14:R15 */
};

#define BAR_ID 0
#define BAR_LOAD 1

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
  p[n++] = hp_s2r(R_ROW, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_COL, HP_SR_TID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_safe());

  /* Source is row-major over cols; destination is row-major over rows, which
   * is the same thing with the two dimensions exchanged. */
  p[n++] = hp_imad_imm(R_SRC_IDX, R_ROW, cols, R_COL, hp_ctrl_wait(BAR_ID));
  p[n++] = hp_imad_imm(R_DST_IDX, R_COL, rows, R_ROW, hp_ctrl_safe());

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
