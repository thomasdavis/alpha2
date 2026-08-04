/*
 * sm86.h — instruction encoders for the sm_86 subset Alpha needs.
 *
 * WHAT: one function per instruction shape, each returning a fully formed
 * 128-bit word.
 *
 * WHY: this is the subset that appears when the vendor compiler is asked to
 * build the kernels Alpha actually runs — measured, not guessed. Of the 31
 * distinct mnemonics observed across the probe corpus, these are the ones the
 * P0 spike and the elementwise kernel family require. The rest arrive as their
 * kernels are ported.
 *
 * WHAT THIS DELIBERATELY DOES NOT DO: no register allocation, no scheduling, no
 * validation that the program makes sense. Each function encodes exactly what it
 * is told. Choosing registers is regalloc.c's job; choosing control bits is
 * control.c's job.
 *
 * Every encoder here is tested by asserting bit-exact equality against an
 * encoding captured from ptxas -- see test/hephaestus_test.c. That is the only
 * standard that means anything for an undocumented ISA: not "looks plausible"
 * but "identical to what the vendor emits".
 */
#ifndef HELIOS_HEPHAESTUS_SM86_H
#define HELIOS_HEPHAESTUS_SM86_H

#include "control.h"
#include "word.h"

/* Opcodes, bits 0-11. Observed values, each seen in the captured corpus. */
#define HP_OP_MOV 0xa02
#define HP_OP_IMAD 0x424 /* IMAD.MOV.U32 is how ptxas materialises immediates */
#define HP_OP_S2R 0x919
#define HP_OP_IADD3 0x810      /* IADD3 Rd, Ra, imm, RZ */
#define HP_OP_IADD3_R 0x210    /* IADD3 Rd, Ra, Rb, RZ — a DIFFERENT opcode */
#define HP_OP_IMAD_C 0xa24     /* IMAD Rd, Ra, c[bank][off], Rc */
#define HP_OP_IMAD_WIDE_C 0x625 /* IMAD.WIDE.U32 Rd, Ra, Rb, c[bank][off] */
#define HP_OP_FADD 0x221
#define HP_OP_FMUL 0x220
#define HP_OP_FFMA 0x223
#define HP_OP_MUFU 0x308     /* transcendentals: EX2, LG2, RCP, RSQ */
#define HP_OP_FMNMX 0x209    /* FMNMX Rd, Ra, Rb, {PT|!PT} — min or max */

/* MUFU function selector, bits 72..79. Values captured from cuobjdump. */
#define HP_MUFU_EX2 0x08
#define HP_MUFU_LG2 0x0c
#define HP_MUFU_RCP 0x10
#define HP_MUFU_RSQ 0x14
#define HP_OP_STG 0x986
#define HP_OP_LDG 0x981
#define HP_OP_EXIT 0x94d
#define HP_OP_BRA 0x947
#define HP_OP_NOP 0x918
#define HP_OP_BAR 0xb1d

/* Special registers, bits 72-79. SR_TID.X and SR_CTAID.X are confirmed from
 * captured encodings; the .Y/.Z variants follow the observed +1 stride and are
 * marked as such because they have not been individually captured. */
#define HP_SR_TID_X 0x21  /* captured */
#define HP_SR_TID_Y 0x22  /* inferred from stride — verify before relying on it */
#define HP_SR_TID_Z 0x23  /* inferred */
#define HP_SR_CTAID_X 0x25 /* captured */
#define HP_SR_CTAID_Y 0x26 /* inferred */
#define HP_SR_CTAID_Z 0x27 /* inferred */

/* MOV Rd, c[bank][offset] — how every kernel parameter is read.
 * The offset is encoded as (byte_offset << 6); see word.h. */
hp_word hp_mov_const(unsigned dst, unsigned bank, unsigned offset, hp_control c);

/* MOV Rd, imm32 */
hp_word hp_mov_imm(unsigned dst, uint32_t imm, hp_control c);

/* S2R Rd, SR_* — reads a special register (thread/block index). */
hp_word hp_s2r(unsigned dst, unsigned sreg, hp_control c);

/* IADD3 Rd, Ra, imm, RZ — the workhorse integer add. The third operand is
 * genuinely a third addend; we pass RZ when only two are wanted, which is what
 * the vendor compiler does. */
hp_word hp_iadd3_imm(unsigned dst, unsigned srcA, uint32_t imm, hp_control c);

/* IADD3 Rd, Ra, Rb, RZ */
hp_word hp_iadd3_reg(unsigned dst, unsigned srcA, unsigned srcB, hp_control c);

/* Single-precision arithmetic. References from cuobjdump:
 *   FADD R9,  R0, R9       0x0000000900097221 0x004fca0000000000
 *   FMUL R11, R0, R11      0x0000000b000b7220 0x004fca0000400000
 *   FFMA R13, R0, R13, R15 0x0000000d000d7223 0x004fca000000000f
 * FMUL carries 0x00400000 in bits 64..95 where FADD carries nothing, so the
 * two are not the same shape with a different opcode. */
hp_word hp_fadd(unsigned dst, unsigned srcA, unsigned srcB, hp_control c);
hp_word hp_fmul(unsigned dst, unsigned srcA, unsigned srcB, hp_control c);
hp_word hp_ffma(unsigned dst, unsigned srcA, unsigned srcB, unsigned srcC,
                hp_control c);

/* MUFU Rd, Ra — the multi-function unit. `fn` is one of HP_MUFU_*.
 * References: MUFU.EX2 R7,R0  0x0000000000077308 0x000e240000000800
 *             MUFU.LG2 R8,R6  0x0000000600087308 0x000e640000000c00 */
hp_word hp_mufu(unsigned dst, unsigned src, unsigned fn, hp_control c);

/* FMNMX Rd, Ra, Rb — minimum when `wantMax` is 0, maximum when non-zero.
 * The choice is a predicate operand: PT selects min, !PT selects max, and the
 * negation is one bit. Reference (max): FMNMX R9, RZ, R0, !PT
 *   0x00000000ff097209 0x004fca0007800000  */
hp_word hp_fmnmx(unsigned dst, unsigned srcA, unsigned srcB, int wantMax,
                 hp_control c);

/* FADD Rd, -Ra, -RZ — negation, which has no opcode of its own.
 * Reference: FADD R13, -R0, -RZ  0x800000ff000d7221 0x004fca0000000100 */
hp_word hp_fneg(unsigned dst, unsigned srcA, hp_control c);

/* STG.E [Ra.64 + offset], Rb — store to global memory through a 64-bit address
 * held in Ra:Ra+1. */
hp_word hp_stg(unsigned addrReg, unsigned dataReg, uint32_t offset, hp_control c);

/* IADD3 Rd, Ra, Rb, RZ — the register-register add.
 *
 * NOTE the opcode differs from the immediate form (0x210 vs 0x810). They are
 * not the same instruction with a different operand; assuming otherwise
 * produces something that decodes as an unrelated operation. */
hp_word hp_iadd3_reg(unsigned dst, unsigned srcA, unsigned srcB, hp_control c);

/* Single-precision arithmetic. References from cuobjdump:
 *   FADD R9,  R0, R9       0x0000000900097221 0x004fca0000000000
 *   FMUL R11, R0, R11      0x0000000b000b7220 0x004fca0000400000
 *   FFMA R13, R0, R13, R15 0x0000000d000d7223 0x004fca000000000f
 * FMUL carries 0x00400000 in bits 64..95 where FADD carries nothing, so the
 * two are not the same shape with a different opcode. */
hp_word hp_fadd(unsigned dst, unsigned srcA, unsigned srcB, hp_control c);
hp_word hp_fmul(unsigned dst, unsigned srcA, unsigned srcB, hp_control c);
hp_word hp_ffma(unsigned dst, unsigned srcA, unsigned srcB, unsigned srcC,
                hp_control c);

/* MUFU Rd, Ra — the multi-function unit. `fn` is one of HP_MUFU_*.
 * References: MUFU.EX2 R7,R0  0x0000000000077308 0x000e240000000800
 *             MUFU.LG2 R8,R6  0x0000000600087308 0x000e640000000c00 */
hp_word hp_mufu(unsigned dst, unsigned src, unsigned fn, hp_control c);

/* FMNMX Rd, Ra, Rb — minimum when `wantMax` is 0, maximum when non-zero.
 * The choice is a predicate operand: PT selects min, !PT selects max, and the
 * negation is one bit. Reference (max): FMNMX R9, RZ, R0, !PT
 *   0x00000000ff097209 0x004fca0007800000  */
hp_word hp_fmnmx(unsigned dst, unsigned srcA, unsigned srcB, int wantMax,
                 hp_control c);

/* FADD Rd, -Ra, -RZ — negation, which has no opcode of its own.
 * Reference: FADD R13, -R0, -RZ  0x800000ff000d7221 0x004fca0000000100 */
hp_word hp_fneg(unsigned dst, unsigned srcA, hp_control c);

/* IMAD Rd, Ra, c[bank][offset], Rc — multiply a register by a constant-bank
 * value and add a register. This is how a global thread index is built:
 * ctaid.x * ntid.x + tid.x, with ntid.x living at c[0x0][0x0]. */
hp_word hp_imad_const(unsigned dst, unsigned srcA, unsigned bank,
                      unsigned offset, unsigned srcC, hp_control c);

/* IMAD.WIDE.U32 Rd, Ra, Rb, c[bank][offset] — a 32x32 multiply widened to 64
 * bits and added to a 64-bit constant-bank value, landing in Rd:Rd+1. This is
 * how an element index becomes an address: index * elementSize + base. */
hp_word hp_imad_wide_const(unsigned dst, unsigned srcA, unsigned srcB,
                           unsigned bank, unsigned offset, hp_control c);

/* LDG.E Rd, [Ra.64 + offset] */
hp_word hp_ldg(unsigned dst, unsigned addrReg, uint32_t offset, hp_control c);

/* EXIT — ends the thread. */
hp_word hp_exit(hp_control c);

/* NOP */
hp_word hp_nop(hp_control c);

/* BAR.SYNC — workgroup barrier. */
hp_word hp_bar_sync(hp_control c);

#endif /* HELIOS_HEPHAESTUS_SM86_H */
