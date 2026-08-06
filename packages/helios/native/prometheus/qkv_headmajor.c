/*
 * qkv_headmajor.c — slice a grouped token-major QKV projection straight into
 * the head-major layout attention consumes, in ONE pass.
 *
 * WHAT: the attention block projects x to a grouped qkvFlat [B*T, 3*H*D], then
 * needs each of q, k, v as head-major [B, H, T, D] for the batched QK^T and AV.
 * The composed route is sliceQkv (three slices) followed by three permutes —
 * six launches and three [B*T,H*D] intermediates a layer, and their gradients
 * in the backward. This kernel reads qkvFlat directly and writes head-major, so
 * one launch per plane replaces the slice AND the permute for that plane, with
 * no intermediate. (This is the RoPE-free path — the gate model uses learned
 * position embeddings; qkvHeadMajorRope in ops.ts is the RoPE variant.)
 *
 * WHY IT IS A SIBLING OF pr_emit_permute: it is the same indexing kernel — no
 * arithmetic on the values, only two addresses for the same element. The only
 * differences from permute are that the source row stride is 3H (the grouped
 * width) not H, the source head index is plane*H + h, and the backward swaps
 * which address loads and which stores. Each plane writes disjoint columns of
 * qkvFlat, so the three backward launches cover the gradient with no overlap
 * and the caller need not zero it.
 *
 * An indexing bug here has the permute signature: right shape, right values,
 * wrong PLACEMENT. The oracle (expect_qkv_headmajor.c) is built to see exactly
 * that, with a source whose every element is distinct.
 */
#include "indexing.h" /* kernel.h + pr_permute_rows */
#include "qkv_headmajor.h"

/* Local barriers, as every prometheus kernel keeps its own — not ew_layout.h,
 * whose register enum would collide with the indexing convention below. */
#define BAR_ID 0   /* the four S2Rs */
#define BAR_LOAD 1 /* the global load */

enum {
  R_H = 17,   /* head index h, straight off grid X */
  R_T = 18,   /* time index t, off grid Y (times R when a block covers R rows) */
  R_B = 19,   /* batch index b, off grid Z */
  R_TMP2 = 20,
  R_HSRC = 21, /* plane*H + h — the head index within the grouped qkvFlat row */
  R_QKV = 22,  /* linear index into qkvFlat [B*T, 3*H*D] */
  R_HM = 23,   /* linear index into head-major [B, H, T, D] */
  R_COL = 1,   /* feature d within the head (thread index, low lgD bits) */
  R_ESIZE = 5, /* 4 — the element size, the wide-address multiplier */
  R_ADDR = 6,  /* R6:R7 — load address */
  R_VALUE = 10,
  R_OUT = 14,  /* R14:R15 — store address */
};

/*
 * plane selects q (0), k (1) or v (2); backward!=0 scatters a head-major
 * gradient back to the grouped qkvFlat gradient instead of gathering forward.
 * Grid is (H, T/R, batch) with R*D threads a block, exactly as permute — every
 * index is a grid index, so nothing needs to be a power of two except D, which
 * the row-splitting shift requires and pr_permute_rows already guards.
 */
unsigned pr_emit_qkv_headmajor(hp_word *p, unsigned T, unsigned H, unsigned D,
                               unsigned plane, int backward) {
  unsigned n = 0;
  const unsigned R = pr_permute_rows(T, D);
  const unsigned H3 = 3u * H;
  unsigned lgD = 0;
  while ((1u << lgD) < D) lgD++;

  p[n++] = hp_s2r(R_H, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_T, HP_SR_CTAID_Y, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_B, HP_SR_CTAID_Z, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_COL, HP_SR_TID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_wait(BAR_ID));

  if (R > 1u) {
    /* The block covers R t-values: the thread index splits into the row within
     * the block (high bits) and the feature (low lgD bits). D is a power of two
     * here — pr_permute_rows only returns more than one when it is. */
    p[n++] = hp_shr_imm(R_TMP2, R_COL, lgD, hp_ctrl_safe());
    p[n++] = hp_imad_imm(R_COL, R_TMP2, (uint32_t)-(int)D, R_COL, hp_ctrl_safe());
    p[n++] = hp_imad_imm(R_T, R_T, R, R_TMP2, hp_ctrl_safe());
  }

  /* Head index within the grouped row. plane 0 is just h, so no add. */
  unsigned hsrc = R_H;
  if (plane != 0u) {
    p[n++] = hp_iadd3_imm(R_HSRC, R_H, plane * H, hp_ctrl_safe());
    hsrc = R_HSRC;
  }

  /* qkvFlat index = ((b*T + t)*3H + (plane*H + h))*D + d. */
  p[n++] = hp_imad_imm(R_QKV, R_B, T, R_T, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_QKV, R_QKV, H3, hsrc, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_QKV, R_QKV, D, R_COL, hp_ctrl_safe());
  /* head-major index = ((b*H + h)*T + t)*D + d. */
  p[n++] = hp_imad_imm(R_HM, R_B, H, R_H, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_HM, R_HM, T, R_T, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_HM, R_HM, D, R_COL, hp_ctrl_safe());

  /* forward: gather qkvFlat (param 1) -> head-major (param 0).
   * backward: gather head-major grad (param 1) -> qkvFlat grad (param 0). */
  const unsigned loadIdx = backward ? R_HM : R_QKV;
  const unsigned storeIdx = backward ? R_QKV : R_HM;
  p[n++] = hp_imad_wide_const(R_ADDR, loadIdx, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
  p[n++] = hp_ldg(R_VALUE, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));
  p[n++] = hp_imad_wide_const(R_OUT, storeIdx, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(0), hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_VALUE, 0, hp_ctrl_wait(BAR_LOAD));
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}
