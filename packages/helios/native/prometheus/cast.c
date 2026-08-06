/*
 * cast.c — conversion between 32-bit and 16-bit floats.
 *
 * WHAT: the two directions the backend actually asks for, f32 to f16 and back.
 *
 * WHY TWO ELEMENTS PER THREAD: an f16 tensor is two bytes per element, and a
 * kernel that handled one element per thread would need 16-bit loads and
 * stores. Handling a PAIR keeps every memory access 32-bit and matches the
 * shape of the instructions the hardware provides -- F2FP packs two floats into
 * one register by construction, and the reverse reads one register and selects
 * a half. The pairing is not a trick to avoid narrow accesses; it is the form
 * the ISA is written in.
 *
 * THE CONSEQUENCE, stated because it constrains callers: the element count must
 * be even, and the f16 side is indexed in PAIRS. A tensor with an odd number of
 * elements would need its last element handled separately, which this does not
 * do and does not pretend to.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: no denormal or saturation handling beyond
 * whatever the hardware does. f32 values outside f16's range become infinities,
 * which is the documented behaviour of the instruction and the same thing every
 * other implementation does; the kernel does not clamp them into range, because
 * silently returning the largest finite half would hide an overflow that the
 * caller needs to know about.
 */
#include "cast.h"

enum {
  R_INDEX = 0,
  R_TID = 1,
  R_PAIR = 2, /* the element index of the first of this thread's two */
  R_ESIZE = 5,
  R_ADDR = 6, /* R6:R7 */
  R_A = 10,
  R_B = 11,
  R_PACKED = 12,
  R_OUT = 14, /* R14:R15 */
};

#define BAR_ID 0
#define BAR_LOAD 1

/* The global thread index, as every element-wise kernel computes it. */
static unsigned emit_index(hp_word *p) {
  unsigned n = 0;
  p[n++] = hp_s2r(R_INDEX, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_safe());
  p[n++] = hp_imad_const(R_INDEX, R_INDEX, 0, HERMES_CBUF0_NTID_X, R_TID,
                         hp_ctrl_wait(BAR_ID));
  /* Two elements per thread, so the f32 side is indexed at twice the stride. */
  p[n++] = hp_imad_imm(R_PAIR, R_INDEX, 2, HP_RZ, hp_ctrl_safe());
  return n;
}

/*
 * f32 -> f16: read in[2i] and in[2i+1], pack, write one word to out[i].
 *
 * The two loads share a barrier. That is correct for the same reason it is
 * correct in matmul -- the scoreboard counts outstanding writes rather than
 * flagging one -- and it is worth repeating here because getting it wrong reads
 * the second element before it arrives and packs a stale value into the high
 * half, which produces a tensor that is right in every other position.
 */
unsigned pr_emit_cast_f32_to_f16(hp_word *p) {
  unsigned n = emit_index(p);

  p[n++] = hp_imad_wide_const(R_ADDR, R_PAIR, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
  p[n++] = hp_ldg(R_A, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));
  p[n++] = hp_ldg(R_B, R_ADDR, 4, hp_ctrl_setbar(BAR_LOAD));

  /*
   * The operands are the other way round from the obvious reading: PACK_AB puts
   * srcA in the HIGH half and srcB in the LOW one, so the FIRST element in
   * memory goes in as srcB.
   *
   * Measured, not assumed. The first version passed them in the natural order
   * and produced c500c3b7 where c3b7c500 was wanted -- every pair transposed,
   * every value correct, which is the exact failure this file's header warns
   * about and which a test with two equal elements per pair would have missed
   * entirely.
   */
  p[n++] = hp_f2fp_pack(R_PACKED, R_B, R_A, hp_ctrl_wait(BAR_LOAD));

  p[n++] = hp_imad_wide_const(R_OUT, R_INDEX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(0), hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_PACKED, 0, hp_ctrl_safe());
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}

/* f16 -> f32: read one packed word, widen both halves, write two floats. */
unsigned pr_emit_cast_f16_to_f32(hp_word *p) {
  unsigned n = emit_index(p);

  p[n++] = hp_imad_wide_const(R_ADDR, R_INDEX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
  p[n++] = hp_ldg(R_PACKED, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));

  p[n++] = hp_half_to_float(R_A, R_PACKED, HP_HALF_LO, hp_ctrl_wait(BAR_LOAD));
  p[n++] = hp_half_to_float(R_B, R_PACKED, HP_HALF_HI, hp_ctrl_safe());

  p[n++] = hp_imad_wide_const(R_OUT, R_PAIR, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(0), hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_A, 0, hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_B, 4, hp_ctrl_safe());
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}
