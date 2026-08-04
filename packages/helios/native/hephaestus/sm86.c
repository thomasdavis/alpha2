/*
 * sm86.c — see sm86.h.
 *
 * Each encoder builds the word field by field. The shared prologue (opcode,
 * predicate, control) is factored into hp_base(); everything else is per-shape.
 */
#include "encode.h"

/* Opcode + always-true predicate + control. Every instruction starts here. */

hp_word hp_base(unsigned opcode, hp_control c) {
  hp_word w = {0, 0};
  hp_put(&w, HP_F_OPCODE, 12, opcode);
  hp_put(&w, HP_F_PRED, 3, HP_PT);
  hp_put(&w, HP_F_CONTROL, 23, hp_control_pack(c));
  return w;
}

hp_word hp_mov_const(unsigned dst, unsigned bank, unsigned offset, hp_control c) {
  hp_word w = hp_base(HP_OP_MOV, c);
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
  hp_word w = hp_base(HP_OP_IMAD, c);
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
  hp_word w = hp_base(HP_OP_S2R, c);
  hp_put(&w, HP_F_DST, 8, dst);
  hp_put(&w, HP_F_SREG, 8, sreg);
  return w;
}

hp_word hp_iadd3_imm(unsigned dst, unsigned srcA, uint32_t imm, hp_control c) {
  hp_word w = hp_base(HP_OP_IADD3, c);
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

/* The register-register add used the IMMEDIATE opcode (0x810) with a register
 * in the srcB slot. Those are different instructions, not one instruction with
 * two operand kinds -- reg-reg is 0x210. Reference:
 *   IADD3 R7, R0, R3, RZ   0x0000000300077210 0x004fca0007ffe0ff  */



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
/*
 * Reference encodings, from cuobjdump on a kernel doing
 * `o[blockIdx.x*blockDim.x+threadIdx.x] = a[i] + i`:
 *
 *   IADD3 R7, R0, R3, RZ                    0x0000000300077210 0x004fca0007ffe0ff
 *   IMAD R0, R0, c[0x0][0x0], R3            0x0000000000007a24 0x001fc800078e0203
 *   IMAD.WIDE.U32 R2, R0, R5, c[0x0][0x168] 0x00005a0000027625 0x000fcc00078e0005
 *
 * Note where the operands sit, because it is not uniform: IMAD.WIDE puts its
 * CONSTANT in the srcB slot at bit 32 and its register srcB at bit 64, the
 * opposite of what the printed operand order suggests.
 */
hp_word hp_iadd3_reg(unsigned dst, unsigned srcA, unsigned srcB, hp_control c) {
  hp_word w = hp_base(HP_OP_IADD3_R, c);
  hp_put(&w, HP_F_DST, 8, dst);
  hp_put(&w, HP_F_SRCA, 8, srcA);
  hp_put(&w, HP_F_SRCB, 8, srcB);
  hp_put(&w, 64, 32, 0x07ffe0ff); /* srcC = RZ, plus the fixed selector */
  return w;
}

hp_word hp_imad_imm(unsigned dst, unsigned srcA, uint32_t imm, unsigned srcC,
                    hp_control c) {
  hp_word w = hp_base(HP_OP_IMAD_IMM, c);
  hp_put(&w, HP_F_DST, 8, dst);
  hp_put(&w, HP_F_SRCA, 8, srcA);
  hp_put(&w, 32, 32, imm);
  /* The addend is a register at bit 64, and 0x078e02 above it is the same
   * fixed selector the const-bank form carries. */
  hp_put(&w, 64, 8, srcC);
  hp_put(&w, 72, 24, 0x078e02);
  return w;
}

hp_word hp_imad_const(unsigned dst, unsigned srcA, unsigned bank,
                      unsigned offset, unsigned srcC, hp_control c) {
  hp_word w = hp_base(HP_OP_IMAD_C, c);
  hp_put(&w, HP_F_DST, 8, dst);
  hp_put(&w, HP_F_SRCA, 8, srcA);
  hp_put(&w, HP_F_SRCB + 6, 16, offset);
  hp_put(&w, HP_F_SRCB + 22, 5, bank);
  hp_put(&w, 64, 8, srcC);
  hp_put(&w, 72, 24, 0x078e02);
  return w;
}

hp_word hp_imad_wide_const(unsigned dst, unsigned srcA, unsigned srcB,
                           unsigned bank, unsigned offset, hp_control c) {
  hp_word w = hp_base(HP_OP_IMAD_WIDE_C, c);
  hp_put(&w, HP_F_DST, 8, dst);
  hp_put(&w, HP_F_SRCA, 8, srcA);
  hp_put(&w, HP_F_SRCB + 6, 16, offset);
  hp_put(&w, HP_F_SRCB + 22, 5, bank);
  hp_put(&w, 64, 8, srcB);
  hp_put(&w, 72, 24, 0x078e00);
  return w;
}











