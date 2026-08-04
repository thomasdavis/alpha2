/*
 * sm86_float.c — floating-point encoders.
 *
 * WHAT: FADD, FMUL, FFMA, FMNMX, and MUFU, the multi-function unit that serves
 * exp2, log2, reciprocal and reciprocal-square-root.
 *
 * THE HARDWARE FACT THAT SHAPES THIS FILE: MUFU is variable-latency and its
 * operand lives in srcB, at bit 32, not srcA. Three of the four references
 * consulted while writing it read R0 for both, which proves nothing about which
 * field is which -- it took a probe with distinct registers to tell them apart.
 */
#include "encode.h"

hp_word hp_fadd(unsigned dst, unsigned srcA, unsigned srcB, hp_control c) {
  hp_word w = hp_base(HP_OP_FADD, c);
  hp_put(&w, HP_F_DST, 8, dst);
  hp_put(&w, HP_F_SRCA, 8, srcA);
  hp_put(&w, HP_F_SRCB, 8, srcB);
  return w;
}

hp_word hp_fmul(unsigned dst, unsigned srcA, unsigned srcB, hp_control c) {
  hp_word w = hp_base(HP_OP_FMUL, c);
  hp_put(&w, HP_F_DST, 8, dst);
  hp_put(&w, HP_F_SRCA, 8, srcA);
  hp_put(&w, HP_F_SRCB, 8, srcB);
  hp_put(&w, 64, 32, 0x00400000);
  return w;
}

hp_word hp_ffma(unsigned dst, unsigned srcA, unsigned srcB, unsigned srcC,
                hp_control c) {
  hp_word w = hp_base(HP_OP_FFMA, c);
  hp_put(&w, HP_F_DST, 8, dst);
  hp_put(&w, HP_F_SRCA, 8, srcA);
  hp_put(&w, HP_F_SRCB, 8, srcB);
  hp_put(&w, 64, 8, srcC);
  return w;
}

/* MUFU's operand sits in the srcB slot at bit 32, not srcA at bit 24. The
 * reference MUFU.LG2 R8, R6 carries 0x06 at bits 32..39, and every other
 * captured MUFU reads R0, which is zero in both slots and so proves nothing --
 * a case where three of four references were silently compatible with the wrong
 * answer. Caught by disassembling our own output: MUFU.LG2 R8, R6 came back as
 * MUFU.LG2 R8, R0. */
hp_word hp_mufu(unsigned dst, unsigned src, unsigned fn, hp_control c) {
  hp_word w = hp_base(HP_OP_MUFU, c);
  hp_put(&w, HP_F_DST, 8, dst);
  hp_put(&w, HP_F_SRCB, 8, src);
  hp_put(&w, 72, 8, fn);
  return w;
}

hp_word hp_fmnmx(unsigned dst, unsigned srcA, unsigned srcB, int wantMax,
                 hp_control c) {
  hp_word w = hp_base(HP_OP_FMNMX, c);
  hp_put(&w, HP_F_DST, 8, dst);
  hp_put(&w, HP_F_SRCA, 8, srcA);
  hp_put(&w, HP_F_SRCB, 8, srcB);
  /* 0x03800000 is the predicate operand PT; bit 90 negates it to !PT, which is
   * what turns a minimum into a maximum. */
  hp_put(&w, 64, 32, wantMax ? 0x07800000u : 0x03800000u);
  return w;
}

hp_word hp_fneg(unsigned dst, unsigned srcA, hp_control c) {
  hp_word w = hp_base(HP_OP_FADD, c);
  hp_put(&w, HP_F_DST, 8, dst);
  hp_put(&w, HP_F_SRCA, 8, srcA);
  hp_put(&w, HP_F_SRCB, 8, HP_RZ);
  hp_put(&w, 63, 1, 1);  /* negate srcA */
  hp_put(&w, 72, 8, 0x01); /* negate srcB (RZ), so the sum is just -Ra */
  return w;
}
