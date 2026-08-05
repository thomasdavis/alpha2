/*
 * sm86_mem.c — memory encoders.
 *
 * WHAT: global load and store, shared load and store.
 *
 * THE HARDWARE FACT THAT SHAPES THIS FILE: bits 64..95 are a memory descriptor
 * -- address size, cache policy, scope -- and they are NOT optional. Leaving
 * them zero produces a word that assembles fine, disassembles to nothing
 * nvdisasm recognises, and faults. The constants are what ptxas emits for a
 * plain 32-bit access.
 *
 * The shared-memory forms use scaled addressing, so their address register
 * holds an ELEMENT index rather than a byte offset. A thread id therefore
 * indexes shared memory directly, which is why the reduction tree never
 * multiplies by four.
 */
#include "encode.h"

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
  hp_word w = hp_base(HP_OP_STG, c);
  hp_put(&w, HP_F_SRCA, 8, addrReg);
  hp_put(&w, HP_F_SRCB, 8, dataReg);
  hp_put(&w, HP_F_SRCB + 8, 24, offset);
  hp_put(&w, 64, 32, 0x0c101904);
  return w;
}

hp_word hp_ldg(unsigned dst, unsigned addrReg, uint32_t offset, hp_control c) {
  hp_word w = hp_base(HP_OP_LDG, c);
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
 * Captured (tools/ldg64_capture.cu), against the 32-bit LDG already above:
 *
 *   LDG.E     R2, [R2.64]        ... / 0x000ea2000c1e1900
 *   LDG.E.64  R2, [R2.64]        0x0000000402027981 / 0x000ea2000c1e1b00
 *   LDG.E.64  R2, [R2.64+0x80]   0x0000800402027981 / 0x000ea2000c1e1b00
 *   LDG.E.128 R4, [R2.64]        0x0000000402047981 / 0x000ea2000c1e1d00
 *
 * The low word does not move at all — same opcode, same operand slots, the same
 * 0x04 constant, and the offset in the same place, which the +0x80 capture
 * pins. The WIDTH is entirely in the descriptor: 0x19, 0x1b, 0x1d in bits 8-15
 * of it, stepping by two per doubling. That is why this is a separate function
 * rather than a width argument threaded through hp_ldg's descriptor constant.
 */
hp_word hp_ldg_wide(unsigned dst, unsigned addrReg, uint32_t offset,
                    unsigned words, hp_control c) {
  hp_word w = hp_base(HP_OP_LDG, c);
  hp_put(&w, HP_F_DST, 8, dst);
  hp_put(&w, HP_F_SRCA, 8, addrReg);
  hp_put(&w, HP_F_SRCB, 8, 0x04);
  hp_put(&w, HP_F_SRCB + 8, 24, offset);
  hp_put(&w, 64, 32, 0x0c1e1900u | (words == 4u ? 0x400u : words == 2u ? 0x200u : 0u));
  return w;
}

hp_word hp_lds(unsigned dst, unsigned addrReg, uint32_t offset, hp_control c) {
  hp_word w = hp_base(HP_OP_LDS, c);
  hp_put(&w, HP_F_DST, 8, dst);
  hp_put(&w, HP_F_SRCA, 8, addrReg);
  hp_put(&w, HP_F_SRCB + 8, 24, offset);
  /* 0x48 at bits 72..79 selects the .X4 scaled addressing mode. */
  hp_put(&w, 72, 8, 0x48);
  return w;
}

/*
 * Captured on the box, one kernel per variant so the SASS could not interleave
 * them (tools/ldsm_capture.cu):
 *
 *   LDSM.16.M88.4  R4, [R0]   0x000000000004783b / 0x000e240000000200
 *   LDSM.16.M88.2  R4, [R0]   0x000000000004783b / 0x000e240000000100
 *   LDSM.16.M88    R5, [R0]   0x000000000005783b / 0x000e280000000000
 *   LDSM.16.MT88.4 R4, [R0]   0x000000000004783b / 0x000e240000004200
 *
 * Two fields fall straight out of the four: a two-bit COUNT at high bits 9:8
 * taking 0, 1, 2 for x1, x2, x4 — note it is an index, not the count — and
 * TRANS as a single bit at high 14. The low word is the ordinary shape every
 * other instruction has, opcode with PT in the predicate slot, destination at
 * 16 and the address register at 24, which is why the x1 capture differs from
 * the x4 only in its destination.
 *
 * Unlike LDG and LDS there is no memory descriptor and no 0x48 addressing mode:
 * the high word is zero apart from these two fields and the scheduling control.
 */
hp_word hp_ldsm(unsigned dst, unsigned addrReg, uint32_t offset,
                unsigned count, int trans, hp_control c) {
  hp_word w = hp_base(HP_OP_LDSM, c);
  hp_put(&w, HP_F_DST, 8, dst);
  hp_put(&w, HP_F_SRCA, 8, addrReg);
  /* Byte offset, in the slot LDS uses for the same thing. Captured as
   * `LDSM.16.M88.4 R4, [R5+0x800]` = 0x000800000504783b, which puts 0x800 at
   * bit 40 exactly. Note the address register is in BYTES here, not the scaled
   * words LDS takes -- LDSM carries none of LDS's 0x48 addressing mode. */
  hp_put(&w, HP_F_SRCB + 8, 24, offset);
  hp_put(&w, 64 + 8, 2, count == 4u ? 2u : count == 2u ? 1u : 0u);
  if (trans) hp_put(&w, 64 + 14, 1, 1);
  return w;
}

/*
 * Captured (tools/atom_capture.cu):
 *
 *   RED.E.ADD.F32.FTZ.RN.STRONG.GPU [R2.64], R7   0x000000070200798e
 *   RED.E.ADD.F32.FTZ.RN.STRONG.GPU [R2.64], R5   0x000000050200798e
 *
 * Two captures differing only in the data register prove that slot is SRCB,
 * where STG also puts it; the address is at SRCA and the immediate offset in
 * STG's slot. The 0x0c10e784 descriptor is reproduced for the same reason STG's
 * is: the hardware was observed to accept it. It differs from STG's 0x0c101904
 * in the bits that presumably select the reduction operator and the rounding
 * mode the mnemonic spells FTZ.RN, but that has not been isolated, and a
 * different reduction — a max, an integer add — must be captured, not derived.
 */
hp_word hp_red_add_f32(unsigned addrReg, unsigned dataReg, uint32_t offset,
                       hp_control c) {
  hp_word w = hp_base(HP_OP_RED, c);
  hp_put(&w, HP_F_SRCA, 8, addrReg);
  hp_put(&w, HP_F_SRCB, 8, dataReg);
  hp_put(&w, HP_F_SRCB + 8, 24, offset);
  hp_put(&w, 64, 32, 0x0c10e784);
  return w;
}

hp_word hp_sts(unsigned addrReg, unsigned dataReg, uint32_t offset,
               hp_control c) {
  hp_word w = hp_base(HP_OP_STS, c);
  hp_put(&w, HP_F_SRCA, 8, addrReg);
  hp_put(&w, HP_F_SRCB, 8, dataReg);
  hp_put(&w, HP_F_SRCB + 8, 24, offset);
  hp_put(&w, 72, 8, 0x48);
  return w;
}

/*
 * SHFL — warp shuffle, the both-immediate form. Captured in
 * tools/shfl_capture.cu; see HP_OP_SHFL_II in isa.h for why the opcode depends
 * on which operands are immediate.
 *
 * WHY IT IS IN THIS FILE: SHFL is a data exchange through the same crossbar
 * LDS uses, and it shares LDS's dangerous property — it is VARIABLE LATENCY and
 * there is no interlock. The vendor compiler's control field on every capture
 * decodes to stall 2, yield 1, WRITE BARRIER 0, which says plainly that a
 * consumer must wait on a barrier rather than on a stall count. Emitting
 * `hp_shfl(..., hp_ctrl_safe())` and reading the result on the next instruction
 * assembles, runs, and returns whatever the register held before — the exact
 * failure mode control.h opens by warning about. Use hp_ctrl_setbar() here and
 * hp_ctrl_wait() on the consumer, the way the reduction's LDS pairs already do.
 *
 * FIELD DERIVATION, each from two captures that differ in exactly one thing:
 *   bits 16-23  destination      R5 vs R9 with everything else fixed
 *   bits 24-31  source           R0 vs R12 vs R6
 *   bits 40-52  the `c` operand  0x1f vs 0x181f — THIRTEEN bits, not five: the
 *                                width-8 form puts 0x18 above the 0x1f, and a
 *                                five-bit read would silently drop it
 *   bits 53-57  lane immediate   0x1 -> 0x20, 0x2 -> 0x40, 0x10 -> 0x200
 *   bits 58-59  mode             IDX 0, UP 1, DOWN 2, BFLY 3
 *   bits 81-83  dest predicate   0xe at bits 80-87 on EVERY capture = PT
 *
 * The `c` operand packs two things: its low five bits are the clamp/max lane
 * and bits 8-12 are the segment mask. hp_shfl_segment() builds it.
 */
hp_word hp_shfl(unsigned mode, unsigned dst, unsigned src, unsigned laneImm,
                unsigned cImm, hp_control c) {
  hp_word w = hp_base(HP_OP_SHFL_II, c);
  hp_put(&w, HP_F_DST, 8, dst);
  hp_put(&w, HP_F_SRCA, 8, src);
  /* SRCB stays CLEAR rather than RZ. With an immediate lane the register slot
   * is unused and every capture shows 0x00 there; the register-lane form is
   * what puts a register number in it. Filling it with RZ is the mistake
   * hp_mov_const's comment records having made once already. */
  hp_put(&w, 40, 13, cImm);
  hp_put(&w, 53, 5, laneImm);
  hp_put(&w, 58, 2, mode);
  /* The predicate DESTINATION — SHFL reports whether the source lane was in
   * range. PT discards it, which is what every capture does. This is not the
   * guard predicate at bits 12-14; both being "PT" is a coincidence of naming
   * and writing one into the other's field costs an afternoon. */
  hp_put(&w, 81, 3, HP_PT);
  return w;
}
