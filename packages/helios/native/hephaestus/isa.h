/*
 * isa.h — instruction encoders for the sm_86 subset Alpha needs.
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
#ifndef HELIOS_HEPHAESTUS_ISA_H
#define HELIOS_HEPHAESTUS_ISA_H

#include "control.h"
#include "word.h"

/* Opcodes, bits 0-11. Observed values, each seen in the captured corpus. */
#define HP_OP_MOV 0xa02
#define HP_OP_IMAD 0x424 /* IMAD.MOV.U32 is how ptxas materialises immediates */
#define HP_OP_S2R 0x919
#define HP_OP_IADD3 0x810      /* IADD3 Rd, Ra, imm, RZ */
#define HP_OP_IADD3_R 0x210    /* IADD3 Rd, Ra, Rb, RZ — a DIFFERENT opcode */
#define HP_OP_IMAD_C 0xa24
/* IMAD Rd, Ra, imm, Rc -- a 32-bit multiply-add with an immediate multiplier
 * and a register addend. This is the form a generated kernel wants when a
 * matrix dimension is fixed at codegen time: row*K + k in one instruction, with
 * K baked in and no register spent holding it. (ptxas: mad.lo.s32 %r, %r, 12,
 * %r) */
#define HP_OP_IMAD_IMM 0x824     /* IMAD Rd, Ra, c[bank][off], Rc */
#define HP_OP_IMAD_WIDE_C 0x625 /* IMAD.WIDE.U32 Rd, Ra, Rb, c[bank][off] */
#define HP_OP_FADD 0x221
#define HP_OP_FMUL 0x220
#define HP_OP_FFMA 0x223
#define HP_OP_MUFU 0x308     /* transcendentals: EX2, LG2, RCP, RSQ */
#define HP_OP_FMNMX 0x209    /* FMNMX Rd, Ra, Rb, {PT|!PT} — min or max */
#define HP_OP_LDS 0x984      /* LDS Rd, [Ra.X4+off] — shared memory load  */
#define HP_OP_STS 0x388      /* STS [Ra.X4+off], Rb — shared memory store */
#define HP_OP_ISETP_IMM 0x80c
/* Register form. The immediate/register opcode pairs differ by 0x600 -- as with
 * IADD3, where 0x810 takes an immediate and 0x210 a register -- but the pattern
 * was CONFIRMED against ptxas rather than extrapolated. Assuming it once
 * produced an IADD3 that assembled, disassembled to something else, and gave a
 * wrong answer with no fault. */
#define HP_OP_ISETP_REG 0x20c
#define HP_OP_BRA 0x947 /* ISETP.<cmp>.U32.AND Pd, PT, Ra, imm, PT  */

/* MUFU function selector, bits 72..79. Values captured from cuobjdump. */
/*
 * ISETP comparison codes, at bits 76..79.
 *
 * GT, NE and GE are read directly off ptxas output. LT, EQ and LE are NOT --
 * ptxas rewrites those by swapping the operands, so it never emits them, and
 * their values here follow the obvious pattern of the three that are known.
 * They are proven a different way: nvdisasm decodes them independently, so the
 * round-trip test fails if the pattern does not hold. An unverified constant
 * with a test that would catch it being wrong is acceptable; an unverified
 * constant with no such test is not.
 */
#define HP_CMP_LT 1 /* inferred, proven by round-trip */
#define HP_CMP_EQ 2 /* inferred, proven by round-trip */
#define HP_CMP_LE 3 /* inferred, proven by round-trip */
#define HP_CMP_GT 4 /* observed: ptxas setp.gt.s32 */
#define HP_CMP_NE 5 /* observed: ptxas setp.ne.s32 */
#define HP_CMP_GE 6 /* observed: ptxas setp.ge.s32 */

#define HP_MUFU_EX2 0x08
#define HP_MUFU_LG2 0x0c
#define HP_MUFU_RCP 0x10
#define HP_MUFU_RSQ 0x14
/*
 * SQRT and TANH, read off ptxas (sqrt.approx.f32 -> 0x00002000 in bits 64..95,
 * tanh.approx.f32 -> 0x00002400; these constants are that field over 256).
 *
 * The values follow the obvious progression from the four above, and that is
 * exactly why they were CHECKED rather than assumed -- the progression also
 * predicted an IADD3 register opcode that turned out to be wrong, assembled
 * anyway, and gave a wrong answer with no fault.
 *
 * SQRT matters beyond saving an instruction: the existing pair, RSQ then RCP,
 * is correct at zero only by accident of infinity arithmetic (1/inf is 0), and
 * relying on that in an optimizer whose second moment starts at exactly zero is
 * not something to leave to accident.
 */
#define HP_MUFU_SQRT 0x20
#define HP_MUFU_TANH 0x24
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


#endif /* HEPHAESTUS_ISA_H */
