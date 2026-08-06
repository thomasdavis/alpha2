/*
 * sm86_flow.c — predication, comparison and control flow.
 *
 * WHAT: ISETP in both immediate and register forms, the predicate applicator,
 * BRA, EXIT, NOP and BAR.SYNC.
 *
 * THE HARDWARE FACT THAT SHAPES THIS FILE: a branch offset is relative to the
 * address of the FOLLOWING instruction, not to the branch itself. Verified
 * against ptxas: a BRA at 0x1f0 carrying -304 targets 0xd0, and 0x200 - 304 is
 * 0xd0. Getting this off by one instruction produces a loop that runs a
 * different number of times rather than a crash.
 */
#include "encode.h"

/*
 * A PREDICATE COSTS FAR MORE THAN A REGISTER, and the floor is enforced here
 * rather than at the call sites.
 *
 * tools/stall_probe.c sweeps the stall count on a producer whose consumer reads
 * its result, and asks the hardware where the answer stops being right:
 *
 *     IADD3 / IMAD / FFMA / SHF+LOP3     4
 *     MOV c[]                            5
 *     IMAD.WIDE, HADD2                   0
 *     ISETP -> @P                       13
 *
 * Thirteen, against four for everything else. That gap is why the general
 * default could be lowered from 15 and this could not follow it: every ISETP in
 * the tree is written `hp_ctrl_safe()` -- in matmul's loop, in all three
 * reductions, in the causal mask, in dropout, in cross-entropy -- and each one
 * would have been handed a stall of 7. A stale predicate does not fault. It
 * masks the wrong elements, or takes the loop a different number of times, and
 * returns a plausible number.
 *
 * So the emitter clamps rather than trusting what it is passed. 15 is 13 plus
 * the same margin the ALU default carries, and it happens to be the maximum,
 * which is the correct place for a value this close to the ceiling to sit.
 */
#define HP_ISETP_MIN_STALL 15

static hp_control isetp_ctrl(hp_control c) {
  if (c.stall < HP_ISETP_MIN_STALL) c.stall = HP_ISETP_MIN_STALL;
  return c;
}

hp_word hp_isetp_gt_imm(unsigned destPred, unsigned srcA, uint32_t imm,
                        hp_control c) {
  hp_word w = hp_base(HP_OP_ISETP_IMM, isetp_ctrl(c));
  hp_put(&w, HP_F_SRCA, 8, srcA);
  hp_put(&w, HP_F_SRCB, 32, imm);
  /* 0x03f04070 carries the comparison (GT), the type (U32), the combining
   * operation (AND) and the second predicate source (PT). The destination
   * predicate index sits at bits 81..83 within it. */
  hp_put(&w, 64, 32, 0x03f04070);
  hp_put(&w, 81, 3, destPred);
  return w;
}

hp_word hp_predicated(hp_word w, unsigned pred, int negate) {
  /*
   * The field is CLEARED before it is written, because hp_put ORs. Every other
   * caller starts from a zeroed word so that has never mattered; here the field
   * already holds PT from hp_base(), and OR-ing 0 into 7 leaves 7. The symptom is
   * quiet and specific: `@!P0` assembles as `@!PT`, which is "never" rather
   * than "when P0 is false" -- an instruction that silently does nothing.
   */
  w.lo &= ~((uint64_t)0xf << HP_F_PRED);
  w.lo |= (uint64_t)((pred & 7u) | (negate ? 8u : 0u)) << HP_F_PRED;
  return w;
}

hp_word hp_exit(hp_control c) {
  hp_word w = hp_base(HP_OP_EXIT, c);
  hp_put(&w, 64, 32, 0x03800000);
  return w;
}

hp_word hp_nop(hp_control c) { return hp_base(HP_OP_NOP, c); }

hp_word hp_bar_sync(hp_control c) { return hp_base(HP_OP_BAR, c); }

hp_word hp_isetp_reg(unsigned destPred, unsigned srcA, unsigned srcB,
                     unsigned cmp, int isSigned, hp_control c) {
  hp_word w = hp_base(HP_OP_ISETP_REG, isetp_ctrl(c));
  hp_put(&w, HP_F_SRCA, 8, srcA);
  hp_put(&w, HP_F_SRCB, 8, srcB);
  /* 0x03f00070 is the immediate form's constant with the comparison nibble
   * cleared: it carries the combining operation (AND) and the second predicate
   * source (PT). Signedness is bit 73; the comparison is bits 76..79. */
  hp_put(&w, 64, 32, 0x03f00070);
  if (isSigned) hp_put(&w, 73, 1, 1);
  hp_put(&w, 76, 4, cmp);
  hp_put(&w, 81, 3, destPred);
  return w;
}

/*
 * The branch offset is FIFTY bits, spanning 32..81, and it is sign-extended.
 *
 * It looked like thirty-two. Every forward branch ptxas emits carries
 * 0x03800000 in bits 64..95 and every backward one carries 0x0383ffff -- the
 * extra 0x3ffff is bits 64..81, the top eighteen bits of the offset, holding
 * the sign. Writing only the low thirty-two bits produced a word that
 * disassembled to the right target and jumped roughly four gigabytes forward on
 * hardware, faulting instruction fetch with MMU_ERR_FLT.
 *
 * WHY THE ROUND-TRIP TEST DID NOT CATCH IT: nvdisasm in raw mode prints targets
 * against a base of 0x100000000, and a truncated negative offset lands back in
 * plausible-looking range once that base is added. The disassembler agreed with
 * the intent while the hardware did not. A round-trip is a strong check and it
 * is not a complete one -- what settled this was reading the two ptxas
 * encodings side by side and noticing the high words differed by direction,
 * which is a comparison the round-trip never makes.
 *
 * Fifty bits is +/- 512 KiB of program, which is far more than any kernel here
 * will need and is not checked, because a caller cannot construct an offset
 * that large without a program that large.
 */
#define BRA_SIGN_BITS 0x0003ffffu /* bits 64..81, set when the offset is < 0 */
#define BRA_FIXED 0x03800000u     /* default divergence handling, both ways */

hp_word hp_bra(int32_t byteOffset, hp_control c) {
  hp_word w = hp_base(HP_OP_BRA, c);
  hp_put(&w, 32, 32, (uint32_t)byteOffset);
  hp_put(&w, 64, 32, BRA_FIXED | (byteOffset < 0 ? BRA_SIGN_BITS : 0u));
  return w;
}
