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

/*
 * A 32-bit immediate does NOT go through MOV.
 *
 * The first version reused MOV's opcode and put the immediate in the const-bank
 * operand slot, which assembles to a perfectly valid instruction that reads a
 * constant bank: nvdisasm decoded `hp_mov_imm(0, 0x00060000)` as
 * `MOV R0, c[0x0][0x1800]`. It loads whatever happens to be at that constant
 * offset instead of the value asked for -- wrong silently, with no bad bit
 * anywhere for a bit-comparison test to catch.
 *
 * ptxas materialises immediates as IMAD.MOV.U32 Rd, RZ, RZ, imm. Reference
 * encoding for `IMAD.MOV.U32 R5, RZ, RZ, 0xcafef00d`, from cuobjdump:
 *
 *     0xcafef00dff057424   0x000fe200078e00ff
 *
 * so: opcode 0x424, dst at 16, srcA = RZ (0xff) at 24, the immediate at 32,
 * srcC = RZ (0xff) at 64, and a fixed 0x078e00 at 72.
 */
hp_word hp_mov_imm(unsigned dst, uint32_t imm, hp_control c) {
  hp_word w = base(HP_OP_IMAD, c);
  hp_put(&w, HP_F_DST, 8, dst);
  hp_put(&w, HP_F_SRCA, 8, HP_RZ);
  hp_put(&w, HP_F_SRCB, 32, imm);
  hp_put(&w, HP_F_SRCC, 8, HP_RZ);
  /* The .MOV.U32 selector. Reproduced from the reference rather than derived --
   * it is stable across destination register and immediate value. */
  hp_put(&w, 72, 24, 0x078e00);
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

/*
 * STG carries a memory descriptor in bits 64..95, and omitting it is not a
 * subtle error: nvdisasm rejects the instruction outright with "unrecognized
 * operation for functional unit uC". The first version set only the two
 * registers and the offset, leaving those bits zero.
 *
 * Reference encoding for `STG.E [R2.64], R5`, from cuobjdump:
 *
 *     0x0000000502007986   0x000fe2000c101904
 *
 * giving 0x0c101904 across bits 64..95. That word carries the .E 64-bit address
 * mode and the cache/scope descriptor together; it is reproduced because the
 * hardware and the disassembler both accept it, not because every sub-field in
 * it has been isolated. The same honest limit as the 0x04 in LDG below.
 */
hp_word hp_stg(unsigned addrReg, unsigned dataReg, uint32_t offset, hp_control c) {
  hp_word w = base(HP_OP_STG, c);
  hp_put(&w, HP_F_SRCA, 8, addrReg);
  hp_put(&w, HP_F_SRCB, 8, dataReg);
  hp_put(&w, HP_F_SRCB + 8, 24, offset);
  hp_put(&w, 64, 32, 0x0c101904);
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
  /* The memory descriptor, bits 64..95, exactly as STG has one. Reference
   * `LDG.E R2, [R2.64]` is 0x0000000402027981 / 0x000ea2000c1e1900. Leaving it
   * zero makes the instruction undecodable -- nvdisasm rejects it with the same
   * "unrecognized operation for functional unit uC" that STG produced. */
  hp_put(&w, 64, 32, 0x0c1e1900);
  return w;
}

/*
 * EXIT needs 0x03800000 in bits 64..95, and without it the instruction is not
 * merely odd -- it is CONDITIONAL. nvdisasm decodes a bare-opcode EXIT as
 * `EXIT P0`, predicated on a register whose value the kernel never sets. A
 * kernel that reaches it and does not take it runs off the end of its own code.
 *
 * Reference, from cuobjdump: `EXIT` is 0x000000000000794d / 0x000fea0003800000.
 *
 * This one is worth dwelling on. Every earlier check of EXIT compared the bits
 * we emit against a captured value and passed, because the captured value had
 * been transcribed with the same omission. Disassembling our own output is what
 * caught it -- the round-trip through nvdisasm that standard 7 asks for, which
 * had been treated as a nicety rather than the actual test.
 */
hp_word hp_exit(hp_control c) {
  hp_word w = base(HP_OP_EXIT, c);
  hp_put(&w, 64, 32, 0x03800000);
  return w;
}
hp_word hp_nop(hp_control c) { return base(HP_OP_NOP, c); }
hp_word hp_bar_sync(hp_control c) { return base(HP_OP_BAR, c); }
