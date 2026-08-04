/*
 * hephaestus_test.c — the assembler measured against the vendor compiler.
 *
 * WHAT: asserts our encoders reproduce, bit for bit, encodings that ptxas
 * actually emitted for sm_86.
 *
 * WHY this and not a semantic test: SASS is undocumented, so there is no
 * specification to check against. The only meaningful standard is "identical to
 * what the vendor produces", because that is the only bit pattern we know the
 * hardware executes as intended. "Looks plausible" is worth nothing here — a
 * wrong field silently executes a different instruction rather than faulting.
 *
 * Every REF_ constant below was captured by scripts/sass-catalogue.mjs from a
 * real ptxas invocation and is quoted with the disassembly it came from. None
 * of them were produced by this code and pasted back, which would prove only
 * that the encoder is deterministic (standard 5).
 *
 * The control field is masked out of most comparisons and checked separately.
 * It encodes the *schedule*, which depends on an instruction's neighbours, so a
 * standalone encoder cannot be expected to reproduce the vendor's choice — only
 * the instruction proper.
 */
#include "../hephaestus/sm86.h"
#include "harness.h"


/* Captured from ptxas -arch=sm_86, via nvdisasm -c -hex. */
#define REF_EXIT_LO 0x000000000000794dULL /* EXIT */
#define REF_NOP_LO 0x0000000000007918ULL  /* NOP */
#define REF_BAR_LO 0x0000000000007b1dULL  /* BAR.SYNC.DEFER_BLOCKING 0x0 */
#define REF_S2R_R5_TID_LO 0x0000000000057919ULL   /* S2R R5, SR_TID.X */
#define REF_S2R_R0_TID_LO 0x0000000000007919ULL   /* S2R R0, SR_TID.X */
#define REF_S2R_R5_CTAID_HI 0x000e220000002500ULL /* S2R R5, SR_CTAID.X */
#define REF_S2R_R5_TID_HI 0x000e220000002100ULL   /* S2R R5, SR_TID.X */
#define REF_MOV_R1_C28_LO 0x00000a0000017a02ULL   /* MOV R1, c[0x0][0x28] */
#define REF_MOV_R2_C160_LO 0x0000580000027a02ULL  /* MOV R2, c[0x0][0x160] */
#define REF_MOV_R3_C164_LO 0x0000590000037a02ULL  /* MOV R3, c[0x0][0x164] */
#define REF_IADD3_LO 0x0000000705057810ULL        /* IADD3 R5, R5, 0x7, RZ */
#define REF_STG_LO 0x0000000502007986ULL          /* STG.E [R2.64], R5 */
#define REF_STG_OFF4_LO 0x0000040502007986ULL     /* STG.E [R2.64+0x4], R5 */
#define REF_LDG_R0_LO 0x0000000402007981ULL       /* LDG.E R0, [R2.64] */
#define REF_LDG_R5_LO 0x0000000402057981ULL       /* LDG.E R5, [R2.64] */

/* The low word carries opcode, predicate and all operand fields; the control
 * field lives entirely in the high word above bit 105. Comparing low words
 * therefore compares the instruction without its schedule. */
static uint64_t lo_of(hp_word w) { return w.lo; }

/*
 * The high word WITHOUT the scheduling control field.
 *
 * Control occupies bits 105..127, which is the top 23 bits of the high word,
 * and it is ours to choose -- ptxas picks different stall counts than we do for
 * the same instruction, so comparing the raw high word against ptxas output
 * would fail on a difference that is not a difference. Everything BELOW that is
 * encoding, and for ISETP it is where the comparison lives: a low-word check
 * alone cannot tell GT from GE, which is exactly the kind of wrong answer that
 * arrives without a fault.
 */
#define HI_ENCODING_MASK ((1ULL << 41) - 1)
static uint64_t hi_of(hp_word w) { return w.hi & HI_ENCODING_MASK; }

static void test_control_roundtrip(void) {
  HT_CASE("control field packs and unpacks losslessly");
  hp_control c = {13, 1, 3, 5, 0x2a, 0x9};
  hp_control r = hp_control_unpack(hp_control_pack(c));
  HT_EQ_U64(r.stall, 13);
  HT_EQ_U64(r.yield, 1);
  HT_EQ_U64(r.writeBarrier, 3);
  HT_EQ_U64(r.readBarrier, 5);
  HT_EQ_U64(r.waitMask, 0x2a);
  HT_EQ_U64(r.reuse, 0x9);

  /* The safe default must genuinely be maximally conservative: full stall, no
   * barriers set. A default that silently allowed a race would undermine every
   * kernel written before the scheduler exists. */
  hp_control s = hp_ctrl_safe();
  HT_EQ_U64(s.stall, 15);
  HT_EQ_U64(s.writeBarrier, HP_NO_BARRIER);
  HT_EQ_U64(s.readBarrier, HP_NO_BARRIER);
  HT_EQ_U64(s.waitMask, 0);
  HT_END();
}

static void test_control_lives_above_bit_105(void) {
  HT_CASE("control field occupies bits 105-127 and nothing else");
  /* Two instructions identical but for their control fields must differ only
   * in the high word above bit 105. If the field were misplaced it would
   * corrupt an operand instead, which is exactly the silent-wrong-instruction
   * failure this whole file exists to prevent. */
  hp_word a = hp_exit(hp_ctrl_safe());
  hp_word b = hp_exit(hp_ctrl_setbar(2));
  HT_EQ_U64(a.lo, b.lo);
  uint64_t diff = a.hi ^ b.hi;
  /* bits 105..127 of the word == bits 41..63 of the high word */
  HT_EQ_U64(diff & ((1ULL << 41) - 1ULL), 0);
  HT_END();
}

static void test_zero_operand_instructions(void) {
  HT_CASE("EXIT / NOP / BAR match ptxas bit for bit");
  HT_EQ_U64(lo_of(hp_exit(hp_ctrl_safe())), REF_EXIT_LO);
  HT_EQ_U64(lo_of(hp_nop(hp_ctrl_safe())), REF_NOP_LO);
  HT_EQ_U64(lo_of(hp_bar_sync(hp_ctrl_safe())), REF_BAR_LO);
  HT_END();
}

static void test_s2r(void) {
  HT_CASE("S2R matches ptxas, and the register field is at bit 16");
  HT_EQ_U64(lo_of(hp_s2r(5, HP_SR_TID_X, hp_ctrl_safe())), REF_S2R_R5_TID_LO);
  HT_EQ_U64(lo_of(hp_s2r(0, HP_SR_TID_X, hp_ctrl_safe())), REF_S2R_R0_TID_LO);

  /* The special-register index lives in the HIGH word at bit 72, so it is the
   * one operand not visible in lo. Checking it against both captured values is
   * what pins the field: TID.X is 0x21 and CTAID.X is 0x25, four apart, which
   * would look identical if the field were only one bit wide. */
  hp_word tid = hp_s2r(5, HP_SR_TID_X, hp_ctrl_safe());
  hp_word ctaid = hp_s2r(5, HP_SR_CTAID_X, hp_ctrl_safe());
  HT_EQ_U64(hp_get(&tid, HP_F_SREG, 8), 0x21);
  HT_EQ_U64(hp_get(&ctaid, HP_F_SREG, 8), 0x25);
  HT_EQ_U64(tid.hi & 0xffffULL, REF_S2R_R5_TID_HI & 0xffffULL);
  HT_EQ_U64(ctaid.hi & 0xffffULL, REF_S2R_R5_CTAID_HI & 0xffffULL);
  HT_END();
}

static void test_mov_const(void) {
  HT_CASE("MOV from const bank matches ptxas (kernel parameters)");
  /* This is how every kernel argument is read on NVIDIA, so it is the single
   * most load-bearing encoding in the set. The offset is byte_offset << 6:
   * 0x28 << 6 == 0xa00, which is visible in the captured word. */
  HT_EQ_U64(lo_of(hp_mov_const(1, 0, 0x28, hp_ctrl_safe())), REF_MOV_R1_C28_LO);
  HT_EQ_U64(lo_of(hp_mov_const(2, 0, 0x160, hp_ctrl_safe())), REF_MOV_R2_C160_LO);
  HT_EQ_U64(lo_of(hp_mov_const(3, 0, 0x164, hp_ctrl_safe())), REF_MOV_R3_C164_LO);
  HT_END();
}

static void test_iadd3(void) {
  HT_CASE("IADD3 with immediate matches ptxas");
  HT_EQ_U64(lo_of(hp_iadd3_imm(5, 5, 0x7, hp_ctrl_safe())), REF_IADD3_LO);
  HT_END();
}

static void test_memory(void) {
  HT_CASE("STG / LDG match ptxas, including the offset field");
  HT_EQ_U64(lo_of(hp_stg(2, 5, 0, hp_ctrl_safe())), REF_STG_LO);
  HT_EQ_U64(lo_of(hp_stg(2, 5, 4, hp_ctrl_safe())), REF_STG_OFF4_LO);
  HT_EQ_U64(lo_of(hp_ldg(0, 2, 0, hp_ctrl_safe())), REF_LDG_R0_LO);
  HT_EQ_U64(lo_of(hp_ldg(5, 2, 0, hp_ctrl_safe())), REF_LDG_R5_LO);
  HT_END();
}

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
#define REF_ISETP_GT_HI 0x03f04270ULL /* ISETP.GT.AND P0, PT, R0, R3, PT */
#define REF_ISETP_GE_HI 0x03f06270ULL /* ISETP.GE.AND P0, ... */
#define REF_ISETP_NE_P1_HI 0x03f25270ULL /* ISETP.NE.AND P1, ... */
#define REF_ISETP_REG_LO 0x000000030000720cULL /* R0, R3 */

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

/* ptxas: mad.lo.s32 %r5, %r4, 12, %r2  ->  IMAD R0, R0, 0xc, R3 */
#define REF_IMAD_IMM_LO 0x0000000c00007824ULL
#define REF_IMAD_IMM_HI 0x078e0203ULL

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

/* ptxas: sqrt.approx.f32 / tanh.approx.f32 */
#define REF_MUFU_SQRT_LO 0x0000000000047308ULL
#define REF_MUFU_SQRT_HI 0x00002000ULL
#define REF_MUFU_TANH_HI 0x00002400ULL

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

/* ptxas: cvt.rn.f16x2.f32 -> F2FP.PACK_AB R0, R0, R3
 *        cvt.f32.f16      -> HADD2.F32 R4, -RZ, R0.H0_H0 / R5, -RZ, R0.H1_H1 */
#define REF_F2FP_LO 0x000000030000723eULL
#define REF_F2FP_HI 0x000000ffULL
#define REF_H2F_LO_LO 0x20000000ff047230ULL
#define REF_H2F_HI_LO 0x30000000ff057230ULL
#define REF_H2F_HI 0x00004100ULL

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

static void test_field_placement_primitives(void) {
  HT_CASE("hp_put places fields correctly, including across the 64-bit seam");
  hp_word w = {0, 0};
  hp_put(&w, 0, 12, 0xabc);
  HT_EQ_U64(w.lo, 0xabc);

  /* A field starting below bit 64 and extending past it must land in both
   * words. Nothing in the current encoders straddles the seam, but the control
   * field sits right above it and a silent truncation here would be invisible. */
  hp_word s = {0, 0};
  hp_put(&s, 60, 8, 0xff);
  HT_EQ_U64(s.lo >> 60, 0xf);
  HT_EQ_U64(s.hi & 0xf, 0xf);
  HT_EQ_U64(hp_get(&s, 60, 8), 0xff);

  /* Round-trip at the top of the word. */
  hp_word t = {0, 0};
  hp_put(&t, HP_F_CONTROL, 23, 0x7fffff);
  HT_EQ_U64(hp_get(&t, HP_F_CONTROL, 23), 0x7fffff);
  HT_END();
}

void ht_run(void) {
  printf("\nhephaestus — sm_86 encoder vs ptxas\n");
  test_control_flow();
  test_imad_immediate();
  test_mufu_functions();
  test_half_conversion();
  test_field_placement_primitives();
  test_control_roundtrip();
  test_control_lives_above_bit_105();
  test_zero_operand_instructions();
  test_s2r();
  test_mov_const();
  test_iadd3();
  test_memory();
}
