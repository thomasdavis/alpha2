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
#define HP_OP_IADD3 0x810
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

/* STG.E [Ra.64 + offset], Rb — store to global memory through a 64-bit address
 * held in Ra:Ra+1. */
hp_word hp_stg(unsigned addrReg, unsigned dataReg, uint32_t offset, hp_control c);

/* LDG.E Rd, [Ra.64 + offset] */
hp_word hp_ldg(unsigned dst, unsigned addrReg, uint32_t offset, hp_control c);

/* EXIT — ends the thread. */
hp_word hp_exit(hp_control c);

/* NOP */
hp_word hp_nop(hp_control c);

/* BAR.SYNC — workgroup barrier. */
hp_word hp_bar_sync(hp_control c);

#endif /* HELIOS_HEPHAESTUS_SM86_H */
