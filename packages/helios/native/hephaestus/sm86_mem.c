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

hp_word hp_lds(unsigned dst, unsigned addrReg, uint32_t offset, hp_control c) {
  hp_word w = hp_base(HP_OP_LDS, c);
  hp_put(&w, HP_F_DST, 8, dst);
  hp_put(&w, HP_F_SRCA, 8, addrReg);
  hp_put(&w, HP_F_SRCB + 8, 24, offset);
  /* 0x48 at bits 72..79 selects the .X4 scaled addressing mode. */
  hp_put(&w, 72, 8, 0x48);
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
