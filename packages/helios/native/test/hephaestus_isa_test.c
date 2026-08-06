/*
 * hephaestus_isa_test.c — encodings added after the first instruction set.
 *
 * WHAT: control flow, the immediate-multiplier IMAD, the MUFU selectors, half
 * conversion, and the bitwise pair. Split from hephaestus_test.c only because
 * that file passed 300 lines; the standard is identical and stated there --
 * bit-exact equality against an encoding captured from ptxas, with the
 * scheduling bits masked off because those are ours to choose.
 *
 * These are the encodings that carry the most information in their HIGH word,
 * which is the half a low-word-only comparison cannot see: the branch's sign
 * bits, the MUFU function selector, the LOP3 truth table. Every assertion here
 * that checks hi_of() exists because something in that half was once wrong and
 * the low word looked perfect.
 */
#include "harness.h"
#include "../hephaestus/sm86.h"

#include <stdint.h>

static uint64_t lo_of(hp_word w) { return w.lo; }
#define HI_ENCODING_MASK ((1ULL << 41) - 1)
static uint64_t hi_of(hp_word w) { return w.hi & HI_ENCODING_MASK; }

#define REF_ISETP_GT_HI 0x03f04270ULL /* ISETP.GT.AND P0, PT, R0, R3, PT */
#define REF_ISETP_GE_HI 0x03f06270ULL /* ISETP.GE.AND P0, ... */
#define REF_ISETP_NE_P1_HI 0x03f25270ULL /* ISETP.NE.AND P1, ... */
#define REF_ISETP_REG_LO 0x000000030000720cULL 


/* ptxas ground truth, from a compiled loop and a compiled comparison chain.
 * The comparisons are register-register and signed, which is what a loop
 * counter against a runtime bound assembles to. */
#define REF_BRA_NP0_LO 0x000002e000008947ULL /* @!P0 BRA +0x2e0 */
#define REF_BRA_SELF_LO 0xfffffff000007947ULL /* BRA -0x10 (self) */
#define REF_BRA_HI 0x03800000ULL
/* ptxas, backward: @!P1 BRA -0x130. The high word differs from the forward case
 * by 0x3ffff -- the sign of a fifty-bit offset. */
#define REF_BRA_BACK_LO 0xfffffed000009947ULL
#define REF_BRA_BACK_HI 0x0383ffffULL
/* ptxas: mad.lo.s32 %r5, %r4, 12, %r2  ->  IMAD R0, R0, 0xc, R3 */
#define REF_IMAD_IMM_LO 0x0000000c00007824ULL
#define REF_IMAD_IMM_HI 0x078e0203ULL
/* ptxas: shr.u32 -> SHF.R.U32.HI R0, RZ, 0x10, R3
 *        xor.b32 -> LOP3.LUT R0, R3, R0, RZ, 0x3c, !PT */
#define REF_SHF_LO 0x00000010ff007819ULL
#define REF_SHF_HI 0x00011603ULL
#define REF_LOP3_LO 0x0000000003007212ULL
#define REF_LOP3_HI 0x078e3cffULL
#define REF_MUFU_SQRT_LO 0x0000000000047308ULL
#define REF_MUFU_SQRT_HI 0x00002000ULL
#define REF_MUFU_TANH_HI 0x00002400ULL
#define REF_F2FP_LO 0x000000030000723eULL
#define REF_F2FP_HI 0x000000ffULL
#define REF_H2F_LO_LO 0x20000000ff047230ULL
#define REF_H2F_HI_LO 0x30000000ff057230ULL
#define REF_H2F_HI 0x00004100ULL


/* R0, R3 */

static void test_control_flow(void) {
  HT_CASE("BRA and register-form ISETP match ptxas");

  /* A branch offset is relative to the FOLLOWING instruction. Both directions
   * are checked because a sign error shows up in only one of them. */
  HT_EQ_U64(lo_of(hp_predicated(hp_bra(0x2e0, hp_ctrl_safe()), 0, 1)),
            REF_BRA_NP0_LO);
  HT_EQ_U64(lo_of(hp_bra(-16, hp_ctrl_safe())), REF_BRA_SELF_LO);
  /*
   * A NEGATIVE offset must sign-extend into bits 64..81. Checking the high word
   * is the whole point: the low word of a backward branch looks correct with or
   * without it, the disassembler prints the right target either way, and the
   * hardware jumps four gigabytes forward without it.
   */
  HT_EQ_U64(lo_of(hp_predicated(hp_bra(-0x130, hp_ctrl_safe()), 1, 1)),
            REF_BRA_BACK_LO);
  HT_EQ_U64(hi_of(hp_predicated(hp_bra(-0x130, hp_ctrl_safe()), 1, 1)),
            REF_BRA_BACK_HI);
  HT_EQ_U64(hi_of(hp_bra(-16, hp_ctrl_safe())), REF_BRA_BACK_HI);

  /* And a forward one must NOT: the sign bits are the only difference between
   * the two directions, so asserting both ways pins it from both sides. */
  HT_EQ_U64(hi_of(hp_bra(0x2e0, hp_ctrl_safe())), REF_BRA_HI);
  HT_EQ_U64(hi_of(hp_bra(0, hp_ctrl_safe())), REF_BRA_HI);

  HT_EQ_U64(lo_of(hp_isetp_reg(0, 0, 3, HP_CMP_GT, 1, hp_ctrl_safe())),
            REF_ISETP_REG_LO);
  HT_EQ_U64(hi_of(hp_isetp_reg(0, 0, 3, HP_CMP_GT, 1, hp_ctrl_safe())),
            REF_ISETP_GT_HI);
  HT_EQ_U64(hi_of(hp_isetp_reg(0, 0, 3, HP_CMP_GE, 1, hp_ctrl_safe())),
            REF_ISETP_GE_HI);
  HT_EQ_U64(hi_of(hp_isetp_reg(1, 0, 3, HP_CMP_NE, 1, hp_ctrl_safe())),
            REF_ISETP_NE_P1_HI);

  /*
   * LDSM — ldmatrix, against all four captures in tools/ldsm_capture.cu.
   *
   * The scheduling control is masked off rather than matched: the captures were
   * taken from four different kernels and ptxas gave them different write
   * barriers (0x000e24 against 0x000e28), which says nothing about the
   * instruction. Everything below the control is asserted exactly.
   */
  {
    const uint64_t ctl = 0xffffffffULL; /* the encoding, without the schedule */
    HT_EQ_U64(lo_of(hp_ldsm(4, 0, 0, 4, 0, hp_ctrl_safe())), 0x000000000004783bULL);
    HT_EQ_U64(hi_of(hp_ldsm(4, 0, 0, 4, 0, hp_ctrl_safe())) & ctl, 0x00000200ULL);
    HT_EQ_U64(hi_of(hp_ldsm(4, 0, 0, 2, 0, hp_ctrl_safe())) & ctl, 0x00000100ULL);
    HT_EQ_U64(lo_of(hp_ldsm(5, 0, 0, 1, 0, hp_ctrl_safe())), 0x000000000005783bULL);
    HT_EQ_U64(hi_of(hp_ldsm(5, 0, 0, 1, 0, hp_ctrl_safe())) & ctl, 0x00000000ULL);
    HT_EQ_U64(hi_of(hp_ldsm(4, 0, 0, 4, 1, hp_ctrl_safe())) & ctl, 0x00004200ULL);

    /* The count field must not reach the destination and the destination must
     * not reach the count -- the same both-directions check the predicate
     * fields get above, because a two-bit field in a word this sparse is
     * exactly where an off-by-one shift hides. */
    HT_EQ_U64(lo_of(hp_ldsm(9, 3, 0, 4, 0, hp_ctrl_safe())),
              lo_of(hp_ldsm(9, 3, 0, 1, 0, hp_ctrl_safe())));
    HT_EQ_U64(hi_of(hp_ldsm(9, 3, 0, 4, 0, hp_ctrl_safe())) & ctl,
              hi_of(hp_ldsm(4, 0, 0, 4, 0, hp_ctrl_safe())) & ctl);
    /* And the address register is where every other instruction puts srcA. */
    HT_EQ_U64(lo_of(hp_ldsm(4, 7, 0, 4, 0, hp_ctrl_safe())), 0x000000000704783bULL);
    /* HMMA's accumulate width, from tools/hmma_f16_capture.cu. The low words
     * must be identical and only bits 72-79 may differ — asserting both is what
     * shows the width shares no field with the operands. */
    HT_EQ_U64(lo_of(hp_hmma_acc(8, 8, 6, 255, 0, hp_ctrl_safe())),
              lo_of(hp_hmma_acc(8, 8, 6, 255, 1, hp_ctrl_safe())));
    HT_EQ_U64(lo_of(hp_hmma_acc(8, 8, 6, 255, 1, hp_ctrl_safe())), 0x000000060808723cULL);
    HT_EQ_U64(hi_of(hp_hmma_acc(8, 8, 6, 255, 0, hp_ctrl_safe())) & 0xffffULL, 0x18ffULL);
    HT_EQ_U64(hi_of(hp_hmma_acc(8, 8, 6, 255, 1, hp_ctrl_safe())) & 0xffffULL, 0x08ffULL);

    /* LDG.E.64 and .128, from tools/ldg64_capture.cu. The low word must be
     * IDENTICAL to the 32-bit form at the same operands — the width is only in
     * the descriptor — and asserting that is what proves the two do not share a
     * field. */
    HT_EQ_U64(lo_of(hp_ldg_wide(2, 2, 0, 2, hp_ctrl_safe())), 0x0000000402027981ULL);
    HT_EQ_U64(lo_of(hp_ldg_wide(2, 2, 0x80, 2, hp_ctrl_safe())), 0x0000800402027981ULL);
    HT_EQ_U64(lo_of(hp_ldg_wide(4, 2, 0, 4, hp_ctrl_safe())), 0x0000000402047981ULL);
    HT_EQ_U64(lo_of(hp_ldg_wide(2, 2, 0, 2, hp_ctrl_safe())),
              lo_of(hp_ldg(2, 2, 0, hp_ctrl_safe())));
    HT_EQ_U64(hi_of(hp_ldg_wide(2, 2, 0, 2, hp_ctrl_safe())) & 0xffffffffULL, 0x0c1e1b00ULL);
    HT_EQ_U64(hi_of(hp_ldg_wide(4, 2, 0, 4, hp_ctrl_safe())) & 0xffffffffULL, 0x0c1e1d00ULL);
    HT_EQ_U64(hi_of(hp_ldg(2, 2, 0, hp_ctrl_safe())) & 0xffffffffULL, 0x0c1e1900ULL);

    /* RED — the float atomic add, from tools/atom_capture.cu. Both captures,
     * which differ only in the data register and so pin that slot. */
    HT_EQ_U64(lo_of(hp_red_add_f32(2, 7, 0, hp_ctrl_safe())), 0x000000070200798eULL);
    HT_EQ_U64(lo_of(hp_red_add_f32(2, 5, 0, hp_ctrl_safe())), 0x000000050200798eULL);
    HT_EQ_U64(hi_of(hp_red_add_f32(2, 7, 0, hp_ctrl_safe())) & 0xffffffffULL,
              0x0c10e784ULL);
    /* The offset sits where STG puts it, one byte above the data register. */
    HT_EQ_U64(lo_of(hp_red_add_f32(2, 7, 0x40, hp_ctrl_safe())),
              0x000000070200798eULL | (0x40ULL << 40));

    /* The offset, from the fifth capture: LDSM.16.M88.4 R4, [R5+0x800]. */
    HT_EQ_U64(lo_of(hp_ldsm(4, 5, 0x800, 4, 0, hp_ctrl_safe())),
              0x000800000504783bULL);
  }

  HT_END();
}

/*
 * SHFL gets its own case rather than riding along inside the control-flow one.
 *
 * Not cosmetic: HT_CASE is what a failure reports, and this stack has already
 * lost time to a suite that passed while exercising nothing. A run that prints
 * "SHFL matches ptxas ok" is evidence the encoder was checked; twenty-four
 * assertions accumulating under someone else's heading is not.
 */
static void test_shfl(void) {
  HT_CASE("SHFL matches ptxas, and no field reaches its neighbour");

  /*
   * SHFL, against every capture in tools/shfl_capture.cu.
   *
   * The captures were taken from eleven different kernels and ptxas gave them
   * different write barriers and stalls, so the control field is masked off and
   * everything below it is asserted exactly — the same treatment LDSM gets
   * above, for the same reason.
   *
   * WHAT THE GROUPS PROVE, in order: that each field is where it is claimed to
   * be, and then — the part a single capture can never show — that no field
   * reaches into its neighbour.
   */
  {
    const uint64_t ctl = 0xffffffffULL; /* the encoding, without the schedule */

    /* THE LANE IMMEDIATE. Five captures at 1, 2, 4, 8 and 16 with every other
     * operand fixed, so the only thing that may move is bits 53-57. */
    HT_EQ_U64(lo_of(hp_shfl(HP_SHFL_BFLY, 5, 0, 1, 0x1f, hp_ctrl_safe())),
              0x0c201f0000057f89ULL);
    HT_EQ_U64(lo_of(hp_shfl(HP_SHFL_BFLY, 5, 0, 16, 0x1f, hp_ctrl_safe())),
              0x0e001f0000057f89ULL);
    HT_EQ_U64(lo_of(hp_shfl(HP_SHFL_BFLY, 4, 7, 2, 0x1f, hp_ctrl_safe())),
              0x0c401f0007047f89ULL);
    HT_EQ_U64(lo_of(hp_shfl(HP_SHFL_BFLY, 7, 2, 4, 0x1f, hp_ctrl_safe())),
              0x0c801f0002077f89ULL);
    HT_EQ_U64(lo_of(hp_shfl(HP_SHFL_BFLY, 2, 5, 8, 0x1f, hp_ctrl_safe())),
              0x0d001f0005027f89ULL);

    /* THE OPERAND REGISTERS. Same instruction, different dst and src. */
    HT_EQ_U64(lo_of(hp_shfl(HP_SHFL_BFLY, 9, 6, 8, 0x1f, hp_ctrl_safe())),
              0x0d001f0006097f89ULL);
    HT_EQ_U64(lo_of(hp_shfl(HP_SHFL_BFLY, 5, 12, 16, 0x1f, hp_ctrl_safe())),
              0x0e001f000c057f89ULL);

    /* THE MODE, bits 58-59. DOWN and IDX against BFLY at the same operands. */
    HT_EQ_U64(lo_of(hp_shfl(HP_SHFL_DOWN, 5, 0, 2, 0x1f, hp_ctrl_safe())),
              0x08401f0000057f89ULL);
    HT_EQ_U64(lo_of(hp_shfl(HP_SHFL_IDX, 5, 0, 7, 0x1f, hp_ctrl_safe())),
              0x00e01f0000057f89ULL);

    /* THE `c` OPERAND IS THIRTEEN BITS, not five. The width-8 capture is the
     * only evidence for that, and without it a five-bit field would encode
     * every full-warp shuffle correctly and every segmented one wrongly. */
    HT_EQ_U64(hp_shfl_segment(32), 0x1fULL);
    HT_EQ_U64(hp_shfl_segment(8), 0x181fULL);
    HT_EQ_U64(lo_of(hp_shfl(HP_SHFL_BFLY, 5, 0, 1, hp_shfl_segment(8),
                            hp_ctrl_safe())),
              0x0c381f0000057f89ULL);

    /* THE DESTINATION PREDICATE, bits 81-83, which is 0x0e0000 in the high word
     * and is present on every capture. It is a different field from the guard
     * predicate at bits 12-14 and asserting the high word is what shows it. */
    HT_EQ_U64(hi_of(hp_shfl(HP_SHFL_BFLY, 5, 0, 1, 0x1f, hp_ctrl_safe())) & ctl,
              0x000e0000ULL);
    HT_EQ_U64(hi_of(hp_shfl(HP_SHFL_IDX, 9, 3, 0, 0x1f, hp_ctrl_safe())) & ctl,
              0x000e0000ULL);

    /*
     * NO FIELD REACHES ITS NEIGHBOUR — the check a capture cannot make, because
     * a capture only ever shows one combination. Each pair below differs in
     * exactly one operand, so the words must differ; and each field is driven
     * to its maximum against a neighbour held at zero, which is where an
     * off-by-one shift shows up.
     */
    HT_TRUE(lo_of(hp_shfl(HP_SHFL_BFLY, 5, 0, 1, 0x1f, hp_ctrl_safe())) !=
            lo_of(hp_shfl(HP_SHFL_BFLY, 6, 0, 1, 0x1f, hp_ctrl_safe())));
    HT_TRUE(lo_of(hp_shfl(HP_SHFL_BFLY, 5, 0, 1, 0x1f, hp_ctrl_safe())) !=
            lo_of(hp_shfl(HP_SHFL_BFLY, 5, 1, 1, 0x1f, hp_ctrl_safe())));
    /* lane 31 with c 0 against lane 0 with c 0x1fff: the two fields are
     * adjacent, so a one-bit error would make one of these collide with a
     * shifted version of the other. */
    HT_EQ_U64(lo_of(hp_shfl(HP_SHFL_IDX, 0, 0, 31, 0, hp_ctrl_safe())),
              0x03e0000000007f89ULL);
    HT_EQ_U64(lo_of(hp_shfl(HP_SHFL_IDX, 0, 0, 0, 0x1fff, hp_ctrl_safe())),
              0x001fff0000007f89ULL);
    /* And the mode must not reach the lane: BFLY at lane 0 is exactly the two
     * mode bits and nothing else. */
    HT_EQ_U64(lo_of(hp_shfl(HP_SHFL_BFLY, 0, 0, 0, 0, hp_ctrl_safe())),
              0x0c00000000007f89ULL);
    HT_EQ_U64(lo_of(hp_shfl(HP_SHFL_IDX, 0, 0, 0, 0, hp_ctrl_safe())),
              0x0000000000007f89ULL);
  }
  HT_END();
}

static void test_isetp_fields(void) {
  HT_CASE("ISETP's predicate, comparison and signedness are separate fields");
  /* The destination predicate must not leak into the comparison, and the
   * comparison must not leak into the destination. Changing one and asserting
   * the other is unmoved is the only way to tell overlapping fields apart. */
  const uint64_t gt_p0 = hi_of(hp_isetp_reg(0, 0, 3, HP_CMP_GT, 1, hp_ctrl_safe()));
  const uint64_t gt_p2 = hi_of(hp_isetp_reg(2, 0, 3, HP_CMP_GT, 1, hp_ctrl_safe()));
  HT_EQ_U64(gt_p2 ^ gt_p0, 2ULL << 17); /* only bits 81..83 moved */

  /* Signedness is a bit of its own, not folded into the comparison. */
  HT_EQ_U64(hi_of(hp_isetp_reg(0, 0, 3, HP_CMP_GT, 0, hp_ctrl_safe())) ^
                REF_ISETP_GT_HI,
            1ULL << 9); /* bit 73 */
  HT_END();
}

static void test_imad_immediate(void) {
  HT_CASE("IMAD with an immediate multiplier matches ptxas");
  HT_EQ_U64(lo_of(hp_imad_imm(0, 0, 12, 3, hp_ctrl_safe())), REF_IMAD_IMM_LO);
  HT_EQ_U64(hi_of(hp_imad_imm(0, 0, 12, 3, hp_ctrl_safe())), REF_IMAD_IMM_HI);

  /* The multiplier is a full 32 bits, not the 16- or 24-bit field it would be
   * easy to assume from the const-bank form sitting next to it. */
  HT_EQ_U64(lo_of(hp_imad_imm(0, 0, 0x12345678, 3, hp_ctrl_safe())) >> 32,
            0x12345678ULL);
  HT_END();
}

static void test_mufu_functions(void) {
  HT_CASE("MUFU function selectors match ptxas");
  /* The selector lives in the HIGH word, so a low-word check cannot tell SQRT
   * from RCP -- which is the whole class of bug this catches. */
  HT_EQ_U64(lo_of(hp_mufu(4, 0, HP_MUFU_SQRT, hp_ctrl_safe())),
            REF_MUFU_SQRT_LO);
  HT_EQ_U64(hi_of(hp_mufu(4, 0, HP_MUFU_SQRT, hp_ctrl_safe())),
            REF_MUFU_SQRT_HI);
  HT_EQ_U64(hi_of(hp_mufu(5, 4, HP_MUFU_TANH, hp_ctrl_safe())),
            REF_MUFU_TANH_HI);
  HT_END();
}

static void test_half_conversion(void) {
  HT_CASE("f16 pack and unpack match ptxas");
  HT_EQ_U64(lo_of(hp_f2fp_pack(0, 0, 3, hp_ctrl_safe())), REF_F2FP_LO);
  HT_EQ_U64(hi_of(hp_f2fp_pack(0, 0, 3, hp_ctrl_safe())), REF_F2FP_HI);

  /* Both halves, because the selector is one bit and a kernel that read the
   * wrong one returns the neighbouring element -- a plausible number in the
   * wrong place, which is the failure this cannot afford to miss. */
  HT_EQ_U64(lo_of(hp_half_to_float(4, 0, HP_HALF_LO, hp_ctrl_safe())),
            REF_H2F_LO_LO);
  HT_EQ_U64(lo_of(hp_half_to_float(5, 0, HP_HALF_HI, hp_ctrl_safe())),
            REF_H2F_HI_LO);
  HT_EQ_U64(hi_of(hp_half_to_float(4, 0, HP_HALF_LO, hp_ctrl_safe())),
            REF_H2F_HI);
  HT_END();
}

static void test_bitwise(void) {
  HT_CASE("SHF and LOP3 match ptxas");
  HT_EQ_U64(lo_of(hp_shr_imm(0, 3, 16, hp_ctrl_safe())), REF_SHF_LO);
  HT_EQ_U64(hi_of(hp_shr_imm(0, 3, 16, hp_ctrl_safe())), REF_SHF_HI);
  HT_EQ_U64(lo_of(hp_lop3(0, 3, 0, HP_LUT_XOR, hp_ctrl_safe())), REF_LOP3_LO);
  HT_EQ_U64(hi_of(hp_lop3(0, 3, 0, HP_LUT_XOR, hp_ctrl_safe())), REF_LOP3_HI);

  /* The truth table is a field, not part of the opcode. Changing it must move
   * only its own eight bits -- if it bled into the neighbouring selector, XOR
   * would work and every other function would silently not. */
  HT_EQ_U64(hi_of(hp_lop3(0, 3, 0, 0xf0, hp_ctrl_safe())) ^ REF_LOP3_HI,
            (0xf0ULL ^ 0x3cULL) << 8);
  HT_END();
}

static void test_cpasync(void) {
  HT_CASE("LDGSTS / LDGDEPBAR / DEPBAR match ptxas");
  const uint64_t ctl = 0xffffffffULL; /* the encoding, without the schedule */

  /* THE OPERAND REGISTERS. The shared destination is at DST and the global
   * address PAIR at srcA — two captures differing only in the destination, and
   * two differing only in the source. */
  HT_EQ_U64(lo_of(hp_ldgsts(7, 2, 0, 16, hp_ctrl_safe())), 0x0000000002077faeULL);
  HT_EQ_U64(lo_of(hp_ldgsts(11, 2, 0, 16, hp_ctrl_safe())), 0x00000000020b7faeULL);
  HT_EQ_U64(lo_of(hp_ldgsts(11, 6, 0x1000, 16, hp_ctrl_safe())),
            0x01000000060b7faeULL);
  HT_EQ_U64(lo_of(hp_ldgsts(11, 8, 0x2000, 16, hp_ctrl_safe())),
            0x02000000080b7faeULL);

  /*
   * THE WIDTH AND THE CACHE HINT, which share the memory descriptor and are
   * therefore the pair most easily confused for one field. Three widths at
   * identical operands: only bits 73-74 may move.
   */
  HT_EQ_U64(hi_of(hp_ldgsts(7, 2, 0, 4, hp_ctrl_safe())) & ctl, 0x0b921844ULL);
  HT_EQ_U64(hi_of(hp_ldgsts(7, 2, 0, 8, hp_ctrl_safe())) & ctl, 0x0b921a44ULL);
  HT_EQ_U64(hi_of(hp_ldgsts(7, 2, 0, 16, hp_ctrl_safe())) & ctl, 0x0b901c44ULL);
  /* 4 and 8 bytes differ ONLY in the width field; 16 additionally clears the
   * L1 bit, because a 16-byte cp.async does not go through L1 on this
   * architecture and every 128-bit capture is BYPASS. Asserting the XOR is what
   * shows the two are separate bits rather than one selector. */
  HT_EQ_U64((hi_of(hp_ldgsts(7, 2, 0, 4, hp_ctrl_safe())) & ctl) ^
                (hi_of(hp_ldgsts(7, 2, 0, 8, hp_ctrl_safe())) & ctl),
            1ULL << 9);
  HT_EQ_U64((hi_of(hp_ldgsts(7, 2, 0, 8, hp_ctrl_safe())) & ctl) ^
                (hi_of(hp_ldgsts(7, 2, 0, 16, hp_ctrl_safe())) & ctl),
            (3ULL << 9) | (1ULL << 17));

  /* The offset must not reach the operands and the operands must not reach the
   * offset — the check a capture cannot make. */
  HT_TRUE(lo_of(hp_ldgsts(7, 2, 0, 16, hp_ctrl_safe())) !=
          lo_of(hp_ldgsts(7, 2, 0x1000, 16, hp_ctrl_safe())));
  HT_EQ_U64(lo_of(hp_ldgsts(0, 0, 0x1000, 16, hp_ctrl_safe())),
            0x0100000000007faeULL);

  /* LDGDEPBAR takes no operands at all — the group is implicit state. */
  HT_EQ_U64(lo_of(hp_ldgdepbar(hp_ctrl_safe())), 0x00000000000079afULL);

  /*
   * DEPBAR's COUNT, captured at 0, 1 and 2. Capturing only wait_group 0 would
   * have left the count indistinguishable from a constant — and 0 is the one
   * value that makes the instruction useless, since it drains everything.
   */
  HT_EQ_U64(lo_of(hp_depbar(0, hp_ctrl_safe())), 0x000080000000791aULL);
  HT_EQ_U64(lo_of(hp_depbar(1, hp_ctrl_safe())), 0x000080400000791aULL);
  HT_EQ_U64(lo_of(hp_depbar(2, hp_ctrl_safe())), 0x000080800000791aULL);
  HT_END();
}

void hp_isa_tests(void) {
  test_control_flow();
  test_shfl();
  test_cpasync();
  test_isetp_fields();
  test_imad_immediate();
  test_mufu_functions();
  test_half_conversion();
  test_bitwise();
}
