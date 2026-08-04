/*
 * sm86.h — the encoders, one per instruction form.
 *
 * WHAT: the functions that turn operands into a 128-bit word. The opcodes,
 * field positions and selector constants they are built from live in isa.h,
 * because those are FACTS ABOUT THE HARDWARE and these are choices about what
 * to expose. The two get read for different reasons: isa.h when checking a
 * constant against a capture, this when writing a kernel.
 *
 * Implementations are split by instruction class across sm86.c, sm86_float.c,
 * sm86_mem.c and sm86_flow.c.
 */
#ifndef HEPHAESTUS_SM86_H
#define HEPHAESTUS_SM86_H

#include "isa.h"

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

/*
 * Shared memory. The address register is scaled by four (the .X4 mode a
 * compiler emits for float arrays), so `addrReg` holds an ELEMENT index rather
 * than a byte offset, while `offset` is in bytes.
 *
 * References: STS [R7.X4], R2       0x0000000207007388 0x004fe80000004800
 *             LDS R5, [R7.X4+0x80]  0x0000800007058984 0x000e240000004800
 */
hp_word hp_lds(unsigned dst, unsigned addrReg, uint32_t offset, hp_control c);
hp_word hp_sts(unsigned addrReg, unsigned dataReg, uint32_t offset,
               hp_control c);

/*
 * ISETP.GT.U32.AND Pd, PT, Ra, imm, PT — set a predicate from a comparison.
 * Reference: ISETP.GT.U32.AND P0, PT, R7, 0x1f, PT
 *   0x0000001f0700780c 0x040fe40003f04070
 */
hp_word hp_isetp_gt_imm(unsigned destPred, unsigned srcA, uint32_t imm,
                        hp_control c);

/* destPred = (srcA <cmp> srcB), both registers. `isSigned` selects S32 over
 * U32, which matters for exactly the comparisons where one operand could be
 * negative -- a loop counter compared against a runtime bound, for instance. */
hp_word hp_isetp_reg(unsigned destPred, unsigned srcA, unsigned srcB,
                     unsigned cmp, int isSigned, hp_control c);

/*
 * Branch to `byteOffset` from the address of the FOLLOWING instruction.
 *
 * Callers work in instructions and this takes bytes, deliberately: the caller
 * has to have thought about the 16-byte instruction size to use it, and a
 * caller who has not thought about it gets an obviously wrong distance rather
 * than a plausible one. Combine with hp_predicated for a conditional branch.
 */
hp_word hp_bra(int32_t byteOffset, hp_control c);

/*
 * Predicate an already-encoded instruction: @P<pred> or @!P<pred>.
 *
 * A separate step rather than a parameter on every encoder, because predication
 * is orthogonal to what an instruction does and threading it through twenty
 * signatures would obscure both. PT (7) is "always", which is what base()
 * writes, so an unpredicated instruction is the default.
 */
hp_word hp_predicated(hp_word w, unsigned pred, int negate);

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

/*
 * Shared memory. The address register is scaled by four (the .X4 mode a
 * compiler emits for float arrays), so `addrReg` holds an ELEMENT index rather
 * than a byte offset, while `offset` is in bytes.
 *
 * References: STS [R7.X4], R2       0x0000000207007388 0x004fe80000004800
 *             LDS R5, [R7.X4+0x80]  0x0000800007058984 0x000e240000004800
 */
hp_word hp_lds(unsigned dst, unsigned addrReg, uint32_t offset, hp_control c);
hp_word hp_sts(unsigned addrReg, unsigned dataReg, uint32_t offset,
               hp_control c);

/*
 * ISETP.GT.U32.AND Pd, PT, Ra, imm, PT — set a predicate from a comparison.
 * Reference: ISETP.GT.U32.AND P0, PT, R7, 0x1f, PT
 *   0x0000001f0700780c 0x040fe40003f04070
 */
hp_word hp_isetp_gt_imm(unsigned destPred, unsigned srcA, uint32_t imm,
                        hp_control c);

/* destPred = (srcA <cmp> srcB), both registers. `isSigned` selects S32 over
 * U32, which matters for exactly the comparisons where one operand could be
 * negative -- a loop counter compared against a runtime bound, for instance. */
hp_word hp_isetp_reg(unsigned destPred, unsigned srcA, unsigned srcB,
                     unsigned cmp, int isSigned, hp_control c);

/*
 * Branch to `byteOffset` from the address of the FOLLOWING instruction.
 *
 * Callers work in instructions and this takes bytes, deliberately: the caller
 * has to have thought about the 16-byte instruction size to use it, and a
 * caller who has not thought about it gets an obviously wrong distance rather
 * than a plausible one. Combine with hp_predicated for a conditional branch.
 */
hp_word hp_bra(int32_t byteOffset, hp_control c);

/*
 * Predicate an already-encoded instruction: @P<pred> or @!P<pred>.
 *
 * A separate step rather than a parameter on every encoder, because predication
 * is orthogonal to what an instruction does and threading it through twenty
 * signatures would obscure both. PT (7) is "always", which is what base()
 * writes, so an unpredicated instruction is the default.
 */
hp_word hp_predicated(hp_word w, unsigned pred, int negate);

/* IMAD Rd, Ra, c[bank][offset], Rc — multiply a register by a constant-bank
 * value and add a register. This is how a global thread index is built:
 * ctaid.x * ntid.x + tid.x, with ntid.x living at c[0x0][0x0]. */
hp_word hp_imad_imm(unsigned dst, unsigned srcA, uint32_t imm, unsigned srcC,
                    hp_control c);
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
/*
 * Pack two floats into one register as two halves: dst.H0 = srcA, dst.H1 = srcB.
 * Rounds to nearest even, which is ptxas's cvt.rn and the only mode encoded.
 */
hp_word hp_f2fp_pack(unsigned dst, unsigned srcA, unsigned srcB, hp_control c);

/* Widen one half of `src` back to a float. `half` is HP_HALF_LO or _HI. */
hp_word hp_half_to_float(unsigned dst, unsigned src, unsigned half,
                         hp_control c);

hp_word hp_exit(hp_control c);

/* NOP */
hp_word hp_nop(hp_control c);

/* BAR.SYNC — workgroup barrier. */
hp_word hp_bar_sync(hp_control c);

#endif /* HELIOS_HEPHAESTUS_SM86_H */
