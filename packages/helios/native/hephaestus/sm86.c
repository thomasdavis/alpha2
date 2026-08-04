/*
 * sm86.c — see sm86.h.
 *
 * Each encoder builds the word field by field. The shared prologue (opcode,
 * predicate, control) is factored into base(); everything else is per-shape.
 */
#include "sm86.h"

/* Opcode + always-true predicate + control. Every instruction starts here. */
static hp_word base(unsigned opcode, hp_control c) {
  hp_word w = {0, 0};
  hp_put(&w, HP_F_OPCODE, 12, opcode);
  hp_put(&w, HP_F_PRED, 3, HP_PT);
  hp_put(&w, HP_F_CONTROL, 23, hp_control_pack(c));
  return w;
}

hp_word hp_mov_const(unsigned dst, unsigned bank, unsigned offset, hp_control c) {
  hp_word w = base(HP_OP_MOV, c);
  hp_put(&w, HP_F_DST, 8, dst);
  /* srcA stays CLEAR, not RZ. Filling it with RZ the way IADD3 does produced
   * 0xa00ff017a02 against a captured 0xa0000017a02 -- MOV-from-const simply
   * does not use the field. Caught by the ptxas comparison, which is the whole
   * argument for testing against captured encodings rather than plausibility. */
  /* Const-bank operands: byte offset shifted left 6, bank above it. */
  hp_put(&w, HP_F_SRCB + 6, 16, offset);
  hp_put(&w, HP_F_SRCB + 22, 5, bank);
  /* MOV carries a lane mask in its high operand slot; the vendor compiler
   * always emits 0xf here for a full 32-bit move. */
  hp_put(&w, HP_F_SRCC, 8, 0x0);
  hp_put(&w, 72, 8, 0x0f);
  return w;
}

hp_word hp_mov_imm(unsigned dst, uint32_t imm, hp_control c) {
  hp_word w = base(HP_OP_MOV, c);
  hp_put(&w, HP_F_DST, 8, dst);
  /* As above: no srcA. */
  hp_put(&w, HP_F_SRCB, 32, imm);
  hp_put(&w, 72, 8, 0x0f);
  return w;
}

hp_word hp_s2r(unsigned dst, unsigned sreg, hp_control c) {
  hp_word w = base(HP_OP_S2R, c);
  hp_put(&w, HP_F_DST, 8, dst);
  hp_put(&w, HP_F_SREG, 8, sreg);
  return w;
}

hp_word hp_iadd3_imm(unsigned dst, unsigned srcA, uint32_t imm, hp_control c) {
  hp_word w = base(HP_OP_IADD3, c);
  hp_put(&w, HP_F_DST, 8, dst);
  hp_put(&w, HP_F_SRCA, 8, srcA);
  hp_put(&w, HP_F_SRCB, 32, imm);
  hp_put(&w, HP_F_SRCC, 8, HP_RZ);
  /* Two predicate outputs IADD3 can produce (carry). Unused => PT, which the
   * vendor encodes as 0x7 in each of the two 3-bit slots above srcC. */
  hp_put(&w, 81, 3, HP_PT);
  hp_put(&w, 84, 3, HP_PT);
  return w;
}

hp_word hp_iadd3_reg(unsigned dst, unsigned srcA, unsigned srcB, hp_control c) {
  hp_word w = base(HP_OP_IADD3, c);
  hp_put(&w, HP_F_DST, 8, dst);
  hp_put(&w, HP_F_SRCA, 8, srcA);
  hp_put(&w, HP_F_SRCB, 8, srcB);
  hp_put(&w, HP_F_SRCC, 8, HP_RZ);
  hp_put(&w, 81, 3, HP_PT);
  hp_put(&w, 84, 3, HP_PT);
  return w;
}

hp_word hp_stg(unsigned addrReg, unsigned dataReg, uint32_t offset, hp_control c) {
  hp_word w = base(HP_OP_STG, c);
  hp_put(&w, HP_F_SRCA, 8, addrReg);
  hp_put(&w, HP_F_SRCB, 8, dataReg);
  hp_put(&w, HP_F_SRCB + 8, 24, offset);
  return w;
}

hp_word hp_ldg(unsigned dst, unsigned addrReg, uint32_t offset, hp_control c) {
  hp_word w = base(HP_OP_LDG, c);
  hp_put(&w, HP_F_DST, 8, dst);
  hp_put(&w, HP_F_SRCA, 8, addrReg);
  /* Every captured LDG carries 0x04 in bits 32-39, unchanged across
   * destination register (R0/R4/R5) and access width (.E and .E.128). STG has
   * a data register in the same slot and no such constant, so this is specific
   * to loads.
   *
   * HONEST LIMIT: we have not isolated what this field means -- most likely
   * part of the address-width encoding that prints as ".64". It is reproduced
   * because the hardware was observed to accept it, not because it is
   * understood. If a future LDG form disagrees, this is the first thing to
   * re-derive. */
  hp_put(&w, HP_F_SRCB, 8, 0x04);
  hp_put(&w, HP_F_SRCB + 8, 24, offset);
  return w;
}

hp_word hp_exit(hp_control c) { return base(HP_OP_EXIT, c); }
hp_word hp_nop(hp_control c) { return base(HP_OP_NOP, c); }
hp_word hp_bar_sync(hp_control c) { return base(HP_OP_BAR, c); }
