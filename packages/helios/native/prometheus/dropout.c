/*
 * dropout.c — the dropout mask, generated rather than stored.
 *
 * WHAT: mask[i] = hash(seed, counter, i) < threshold ? 0 : scale, using the
 * murmur3 finalizer the existing backend uses. Same constants, same order, so
 * a model's masks do not change because its GPU stack did.
 *
 * WHY GENERATED AND NOT STORED: the mask has to be identical on the forward and
 * backward pass, and it is the size of the activation it applies to. Storing it
 * doubles the memory for that tensor and adds a read; deriving it from the
 * element index costs a dozen integer instructions and no memory at all. That
 * only works because the hash is a pure function of (seed, counter, index),
 * which is what makes it reproducible in the first place.
 *
 * THE COMPARISON IS ON INTEGERS, and that is a deliberate departure from the
 * SPIR-V version, which converts the hash to a float in [0,1) and compares
 * against p. Converting a 32-bit integer to a float has 24 bits of mantissa, so
 * the low eight bits of the hash are thrown away before the comparison ever
 * happens. Comparing the raw hash against a threshold of p * 2^32 -- computed
 * once on the host -- is the same test with all thirty-two bits, and it removes
 * an I2F and a float compare from the inner loop. The distribution is the same;
 * the resolution is better.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: it does not claim the hash is
 * cryptographic, or that it passes any statistical battery. It is the
 * finalizer of murmur3, which is a good AVALANCHE function and nothing more.
 * That is the right standard for dropout -- what matters is that flipping one
 * bit of the index changes about half the bits of the result -- and it would be
 * the wrong standard for anything else.
 */
#include "dropout.h"

enum {
  R_INDEX = 0,
  R_TID = 1,
  R_H = 2,      /* the hash, mutated through the finalizer */
  R_T = 3,      /* the shifted copy at each xor-shift step */
  R_ESIZE = 5,
  R_SEED = 6,
  R_COUNTER = 7,
  R_THRESH = 8,
  R_SCALE = 9,
  R_OUT = 14, /* R14:R15 */
};

#define BAR_ID 0
#define P_DROP 0

/* murmur3's finalizer constants, matching the SPIR-V kernel exactly. */
#define MIX_COUNTER 0x9E3779B1u
#define MIX_INDEX 0x85EBCA77u
#define MIX_1 0x85EBCA6Bu
#define MIX_2 0xC2B2AE35u

/* One xor-shift-multiply round: h ^= h >> shift; h *= mult. */
static unsigned emit_mix(hp_word *p, unsigned shift, NvU32 mult) {
  unsigned n = 0;
  p[n++] = hp_shr_imm(R_T, R_H, shift, hp_ctrl_safe());
  p[n++] = hp_lop3(R_H, R_H, R_T, HP_LUT_XOR, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_H, R_H, mult, HP_RZ, hp_ctrl_safe());
  return n;
}

unsigned pr_emit_dropout_mask(hp_word *p) {
  unsigned n = 0;
  p[n++] = hp_s2r(R_INDEX, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(BAR_ID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_safe());
  p[n++] = hp_imad_const(R_INDEX, R_INDEX, 0, HERMES_CBUF0_NTID_X, R_TID,
                         hp_ctrl_wait(BAR_ID));

  p[n++] = hp_mov_const(R_SEED, 0, HERMES_CBUF0_SCALAR_N(0), hp_ctrl_safe());
  p[n++] = hp_mov_const(R_COUNTER, 0, HERMES_CBUF0_SCALAR_N(1), hp_ctrl_safe());
  p[n++] = hp_mov_const(R_THRESH, 0, HERMES_CBUF0_SCALAR_N(2), hp_ctrl_safe());
  p[n++] = hp_mov_const(R_SCALE, 0, HERMES_CBUF0_SCALAR_N(3), hp_ctrl_safe());

  /* h = seed + counter*MIX_COUNTER + index*MIX_INDEX */
  p[n++] = hp_imad_imm(R_H, R_COUNTER, MIX_COUNTER, R_SEED, hp_ctrl_safe());
  p[n++] = hp_imad_imm(R_T, R_INDEX, MIX_INDEX, R_H, hp_ctrl_safe());
  p[n++] = hp_iadd3_imm(R_H, R_T, 0, hp_ctrl_safe());

  n += emit_mix(&p[n], 16, MIX_1);
  n += emit_mix(&p[n], 13, MIX_2);

  /* The last step is a bare xor-shift with no multiply, which is what makes the
   * top bits depend on the bottom ones. Dropping it leaves the high bits barely
   * mixed, and the high bits are exactly what an unsigned threshold compares
   * first. */
  p[n++] = hp_shr_imm(R_T, R_H, 16, hp_ctrl_safe());
  p[n++] = hp_lop3(R_H, R_H, R_T, HP_LUT_XOR, hp_ctrl_safe());

  /*
   * Keep unless the hash is below the threshold. Written as "drop if h <= t-1"
   * because the comparison available is GT and it is used negated -- the same
   * shape as everywhere else here. The threshold is p * 2^32 rounded on the
   * host, so a p of zero drops nothing and the predicate is never true.
   */
  p[n++] = hp_isetp_reg(P_DROP, R_H, R_THRESH, HP_CMP_LT, 0, hp_ctrl_safe());
  p[n] = hp_predicated(hp_mov_imm(R_SCALE, 0, hp_ctrl_safe()), P_DROP, 0);
  n++;

  p[n++] = hp_imad_wide_const(R_OUT, R_INDEX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(0), hp_ctrl_safe());
  p[n++] = hp_stg(R_OUT, R_SCALE, 0, hp_ctrl_safe());
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}
