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
#include "../aether/ioctl.h"
#include "../hermes/pushbuffer.h"
#include "../hermes/qmd.h"

#include <stdlib.h>
#include <time.h>

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

/*
 * WHERE THIS TEST LIVES, and why it is not in the hermes suite.
 *
 * It was written there first and would not link: hephaestus sits ABOVE hermes,
 * so a hermes test binary cannot reach the assembler. That is standard 4 doing
 * its job -- the layering is checked by the link graph rather than by review.
 */
static NvU64 now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (NvU64)ts.tv_sec * 1000000000ull + (NvU64)ts.tv_nsec;
}

/*
 * FIVE KERNELS, each proving something the previous one did not.
 *
 * A kernel is described by a builder that assembles it and a checker that says
 * whether the GPU did what it was asked. Everything else -- channel, engine
 * init, descriptor, submission, fence -- is shared, so a failure points at the
 * kernel rather than at the plumbing.
 *
 * Every instruction used here has been round-tripped through nvdisasm (see
 * tools/dump_prog.c). That is not a formality: three encoders were wrong in
 * ways that produced well-formed instructions meaning something else, and no
 * bit-comparison test caught any of them.
 */
#define MAGIC 0xcafef00du

typedef struct {
  const char *name;
  /* Assemble into `prog`, returning the instruction count. `out` and `in` are
   * GPU addresses; kernels that take parameters from the constant bank ignore
   * both and read them at run time instead. */
  unsigned (*build)(hp_word *prog, NvU64 out, NvU64 in);
  NvU32 blockX, gridX;
  /* Prepare the input buffer. NULL means the kernel reads no input. */
  void (*fill)(volatile NvU32 *in);
  /* Inspect the output. Returns NULL on success or a reason. */
  const char *(*check)(const volatile NvU32 *o);
} kernel_case;

/* 1. Store an immediate. The smallest thing that can prove execution. */
static unsigned k_store(hp_word *p, NvU64 out, NvU64 in) {
  (void)in;
  p[0] = hp_mov_imm(0, (uint32_t)(out & 0xffffffffu), hp_ctrl_safe());
  p[1] = hp_mov_imm(1, (uint32_t)(out >> 32), hp_ctrl_safe());
  p[2] = hp_mov_imm(2, MAGIC, hp_ctrl_safe());
  p[3] = hp_stg(0, 2, 0, hp_ctrl_safe());
  p[4] = hp_exit(hp_ctrl_safe());
  return 5;
}
static const char *c_store(const volatile NvU32 *o) {
  return o[0] == MAGIC ? NULL : "out[0] != MAGIC";
}

/* 2. Store at four different offsets -- exercises STG's offset field, which is
 *    encoded separately from the address register and was never proven. */
static unsigned k_offsets(hp_word *p, NvU64 out, NvU64 in) {
  (void)in;
  unsigned n = 0;
  p[n++] = hp_mov_imm(0, (uint32_t)(out & 0xffffffffu), hp_ctrl_safe());
  p[n++] = hp_mov_imm(1, (uint32_t)(out >> 32), hp_ctrl_safe());
  for (unsigned i = 0; i < 4; i++) {
    p[n++] = hp_mov_imm(2, 0x1000u + i, hp_ctrl_safe());
    p[n++] = hp_stg(0, 2, i * 4, hp_ctrl_safe());
  }
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}
static const char *c_offsets(const volatile NvU32 *o) {
  for (unsigned i = 0; i < 4; i++)
    if (o[i] != 0x1000u + i) return "offset slot wrong";
  return NULL;
}

/* 3. Load then store -- the first kernel that READS memory. */
static unsigned k_copy(hp_word *p, NvU64 out, NvU64 in) {
  unsigned n = 0;
  p[n++] = hp_mov_imm(4, (uint32_t)(in & 0xffffffffu), hp_ctrl_safe());
  p[n++] = hp_mov_imm(5, (uint32_t)(in >> 32), hp_ctrl_safe());
  p[n++] = hp_ldg(2, 4, 0, hp_ctrl_setbar(0));
  p[n++] = hp_mov_imm(0, (uint32_t)(out & 0xffffffffu), hp_ctrl_safe());
  p[n++] = hp_mov_imm(1, (uint32_t)(out >> 32), hp_ctrl_wait(0));
  p[n++] = hp_stg(0, 2, 0, hp_ctrl_safe());
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}
static const char *c_copy(const volatile NvU32 *o) {
  return o[0] == 0xf00dbabeu ? NULL : "copy did not carry the input value";
}

/* 4. Load, add an immediate, store -- arithmetic, and the dependency barrier
 *    that makes a load's result safe to consume. */
static unsigned k_addim(hp_word *p, NvU64 out, NvU64 in) {
  unsigned n = 0;
  p[n++] = hp_mov_imm(4, (uint32_t)(in & 0xffffffffu), hp_ctrl_safe());
  p[n++] = hp_mov_imm(5, (uint32_t)(in >> 32), hp_ctrl_safe());
  p[n++] = hp_ldg(2, 4, 0, hp_ctrl_setbar(0));
  p[n++] = hp_mov_imm(0, (uint32_t)(out & 0xffffffffu), hp_ctrl_safe());
  p[n++] = hp_mov_imm(1, (uint32_t)(out >> 32), hp_ctrl_wait(0));
  p[n++] = hp_iadd3_imm(3, 2, 0x1234, hp_ctrl_safe());
  p[n++] = hp_stg(0, 3, 0, hp_ctrl_safe());
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}
static const char *c_addim(const volatile NvU32 *o) {
  return o[0] == 0xf00dbabeu + 0x1234u ? NULL : "sum wrong";
}

/* 5. Thirty-two threads, a special register and a barrier.
 *
 *    Every thread reads its own lane id, adds a base, synchronises, and stores
 *    to the SAME address -- so the write races on purpose and the result is
 *    whichever thread landed last. The assertion is therefore a RANGE: the
 *    value must be base + [0,32). That is a weaker claim than an exact answer
 *    and it is the honest one, because proving which thread won would require
 *    per-thread addressing this instruction set does not yet have. What it does
 *    prove is that a 32-thread block launched, that S2R produced a real lane
 *    id, and that BAR.SYNC executed without faulting. */
static unsigned k_threads(hp_word *p, NvU64 out, NvU64 in) {
  (void)in;
  unsigned n = 0;
  p[n++] = hp_s2r(2, HP_SR_TID_X, hp_ctrl_setbar(0));
  p[n++] = hp_mov_imm(0, (uint32_t)(out & 0xffffffffu), hp_ctrl_safe());
  p[n++] = hp_mov_imm(1, (uint32_t)(out >> 32), hp_ctrl_wait(0));
  p[n++] = hp_iadd3_imm(3, 2, 0x1000, hp_ctrl_safe());
  p[n++] = hp_bar_sync(hp_ctrl_safe());
  p[n++] = hp_stg(0, 3, 0, hp_ctrl_safe());
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}
static const char *c_threads(const volatile NvU32 *o) {
  return (o[0] >= 0x1000u && o[0] < 0x1020u) ? NULL : "lane id out of range";
}

/*
 * THE ELEMENTWISE SKELETON.
 *
 * Every elementwise kernel is the same eleven instructions with one changed in
 * the middle: compute a global index, turn it into two addresses, load, apply
 * an operation, store. Writing them one at a time would be writing the same
 * prologue forty-six times, so the prologue is a function and the operation is
 * a parameter. This is Prometheus in embryo -- the codegen layer the plan calls
 * for -- arrived at from the bottom rather than designed up front.
 *
 * Parameters come from constant bank 0 in CUDA's layout, because the reference
 * encodings bake those offsets in: ntid.x at c[0x0][0x0], the output pointer at
 * c[0x0][0x160], the input pointer at c[0x0][0x168].
 *
 * Registers: R0 index, R2:R3 input address, R4 loaded value, R5 element size,
 * R6:R7 output address, R8 result. Chosen by hand and deliberately wasteful --
 * a register allocator is a separate problem and pretending otherwise here
 * would couple two hard things together.
 */
typedef enum {
  EW_COPY,       /* out[i] = in[i] */
  EW_ADD_INDEX,  /* out[i] = in[i] + i */
  EW_ADD_CONST,  /* out[i] = in[i] + 0x1234 */
  EW_FADD_SELF,  /* out[i] = in[i] + in[i] */
  EW_FMUL_SELF,  /* out[i] = in[i] * in[i] */
  EW_FFMA_SELF,  /* out[i] = in[i]*in[i] + in[i] */
  EW_INDEX,      /* out[i] = i -- a probe: shows what index each thread built */
} ew_op;

static unsigned emit_elementwise(hp_word *p, ew_op op) {
  unsigned n = 0;
  /* Both S2R results are consumed by the same IMAD, so one scoreboard barrier
   * covers them: it does not clear until every producer that set it retires. */
  p[n++] = hp_s2r(0, HP_SR_TID_X, hp_ctrl_setbar(0));
  p[n++] = hp_s2r(3, HP_SR_TID_X, hp_ctrl_setbar(0));
  p[n++] = hp_mov_imm(5, 4, hp_ctrl_safe()); /* bytes per element */
  p[n++] = hp_imad_const(0, 0, 0, HERMES_CBUF0_NTID_X, 3, hp_ctrl_wait(0));
  p[n++] = hp_imad_wide_const(6, 0, 5, 0, HERMES_CBUF0_PARAM0, hp_ctrl_safe());
  /* EW_INDEX needs no input, so it skips the load entirely -- which also makes
   * it a clean probe for whether LDG is what faults. */
  if (op != EW_INDEX) {
    p[n++] = hp_imad_wide_const(2, 0, 5, 0, HERMES_CBUF0_PARAM0 + 8, hp_ctrl_safe());
    p[n++] = hp_ldg(4, 2, 0, hp_ctrl_setbar(2));
  }

  switch (op) {
    case EW_COPY:      p[n++] = hp_iadd3_imm(8, 4, 0, hp_ctrl_wait(2)); break;
    case EW_ADD_INDEX: p[n++] = hp_iadd3_reg(8, 4, 0, hp_ctrl_wait(2)); break;
    case EW_ADD_CONST: p[n++] = hp_iadd3_imm(8, 4, 0x1234, hp_ctrl_wait(2)); break;
    case EW_FADD_SELF: p[n++] = hp_fadd(8, 4, 4, hp_ctrl_wait(2)); break;
    case EW_FMUL_SELF: p[n++] = hp_fmul(8, 4, 4, hp_ctrl_wait(2)); break;
    case EW_FFMA_SELF: p[n++] = hp_ffma(8, 4, 4, 4, hp_ctrl_wait(2)); break;
    case EW_INDEX:     p[n++] = hp_iadd3_imm(8, 0, 0, hp_ctrl_safe()); break;
  }

  p[n++] = hp_stg(6, 8, 0, hp_ctrl_safe());
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}

/*
 * Constant-bank read, isolated.
 *
 * The elementwise kernels are the first to read c[0x0][...], and they are also
 * the first to fault -- two changes at once. This one changes only the bank:
 * addresses stay immediates, and the single constant read is stored where it
 * can be compared against a value the host already knows.
 */
static unsigned k_cbank(hp_word *p, NvU64 out, NvU64 in) {
  (void)in;
  unsigned n = 0;
  p[n++] = hp_mov_imm(0, (uint32_t)(out & 0xffffffffu), hp_ctrl_safe());
  p[n++] = hp_mov_imm(1, (uint32_t)(out >> 32), hp_ctrl_safe());
  p[n++] = hp_mov_const(2, 0, HERMES_CBUF0_NTID_X, hp_ctrl_safe());
  p[n++] = hp_stg(0, 2, 0, hp_ctrl_safe());
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}
static NvU32 g_expect_cbank;
static const char *c_cbank(const volatile NvU32 *o) {
  return o[0] == g_expect_cbank ? NULL : "constant bank read wrong";
}

/*
 * S2R CTAID + scoreboard wait + a grid wider than one, isolated.
 *
 * The elementwise kernels introduce four things at once: SR_CTAID.X, a
 * dependency barrier, two new IMAD forms, and gridX > 1. This probe takes only
 * the first three, keeping addresses as immediates, so a fault here means the
 * index machinery and a fault there means the addressing built on top of it.
 */
static unsigned k_ctaid(hp_word *p, NvU64 out, NvU64 in) {
  (void)in;
  unsigned n = 0;
  p[n++] = hp_s2r(0, HP_SR_CTAID_X, hp_ctrl_setbar(0));
  p[n++] = hp_mov_imm(2, (uint32_t)(out & 0xffffffffu), hp_ctrl_safe());
  p[n++] = hp_mov_imm(3, (uint32_t)(out >> 32), hp_ctrl_safe());
  p[n++] = hp_iadd3_imm(4, 0, 0x100, hp_ctrl_wait(0));
  p[n++] = hp_stg(2, 4, 0, hp_ctrl_safe());
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}
static const char *c_ctaid(const volatile NvU32 *o) {
  return (o[0] >= 0x100u && o[0] < 0x103u) ? NULL : "ctaid out of range";
}

/*
 * The computed address, stored where it can be compared.
 *
 * One block, one thread, so the index is 0 and IMAD.WIDE.U32 should reproduce
 * the output pointer exactly. The store goes through an IMMEDIATE address, so
 * a wrong computation shows up as a wrong value rather than as a fault.
 */
static unsigned k_addr(hp_word *p, NvU64 out, NvU64 in) {
  (void)in;
  unsigned n = 0;
  p[n++] = hp_mov_imm(5, 4, hp_ctrl_safe());
  p[n++] = hp_s2r(0, HP_SR_CTAID_X, hp_ctrl_setbar(0));
  p[n++] = hp_s2r(3, HP_SR_TID_X, hp_ctrl_setbar(1));
  p[n++] = hp_imad_const(0, 0, 0, HERMES_CBUF0_NTID_X, 3, hp_ctrl_waitmask(0x3));
  p[n++] = hp_imad_wide_const(6, 0, 5, 0, HERMES_CBUF0_PARAM0, hp_ctrl_safe());
  p[n++] = hp_mov_imm(2, (uint32_t)(out & 0xffffffffu), hp_ctrl_safe());
  p[n++] = hp_mov_imm(3, (uint32_t)(out >> 32), hp_ctrl_safe());
  p[n++] = hp_stg(2, 6, 0, hp_ctrl_safe());  /* low half of the address  */
  p[n++] = hp_stg(2, 7, 4, hp_ctrl_safe());  /* high half                */
  p[n++] = hp_stg(2, 0, 8, hp_ctrl_safe());  /* and the index it used    */
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}
static NvU64 g_expect_addr;
static const char *c_addr(const volatile NvU32 *o) {
  static char msg[96];
  const NvU64 got = ((NvU64)o[1] << 32) | o[0];
  if (got == g_expect_addr && o[2] == 0) return NULL;
  snprintf(msg, sizeof msg, "addr=0x%llx want 0x%llx index=%u",
           (unsigned long long)got, (unsigned long long)g_expect_addr, o[2]);
  return msg;
}

/* out[tid] = tid, with the index taken straight from S2R -- no IMAD against a
 * constant bank in between. Isolates per-thread ADDRESSING from index
 * ARITHMETIC. */
static unsigned k_tidstore(hp_word *p, NvU64 out, NvU64 in) {
  (void)out; (void)in;
  unsigned n = 0;
  p[n++] = hp_mov_imm(5, 4, hp_ctrl_safe());
  p[n++] = hp_s2r(3, HP_SR_TID_X, hp_ctrl_setbar(0));
  p[n++] = hp_imad_wide_const(6, 3, 5, 0, HERMES_CBUF0_PARAM0, hp_ctrl_wait(0));
  p[n++] = hp_iadd3_imm(8, 3, 0, hp_ctrl_safe());
  p[n++] = hp_stg(6, 8, 0, hp_ctrl_safe());
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}
static const char *c_tidstore(const volatile NvU32 *o) {
  static char msg[80];
  for (unsigned i = 0; i < 64; i++)
    if (o[i] != i) {
      snprintf(msg, sizeof msg, "wrote %u of 64 slots (o[%u]=%u)", i, i, o[i]);
      return msg;
    }
  return NULL;
}

#define EW_KERNEL(tag, opv)                                                    \
  static unsigned k_##tag(hp_word *p, NvU64 out, NvU64 in) {                   \
    (void)out; (void)in;                                                       \
    return emit_elementwise(p, opv);                                           \
  }
EW_KERNEL(ew_copy, EW_COPY)
EW_KERNEL(ew_addidx, EW_ADD_INDEX)
EW_KERNEL(ew_addconst, EW_ADD_CONST)
EW_KERNEL(ew_fadd, EW_FADD_SELF)
EW_KERNEL(ew_fmul, EW_FMUL_SELF)
EW_KERNEL(ew_ffma, EW_FFMA_SELF)
EW_KERNEL(ew_index, EW_INDEX)

/* The elementwise tests run over EW_N elements with one element per thread, so
 * every checker sees the whole output rather than a single slot. */
#define EW_N 64

/* Byte offset of the completion fence within the output buffer, chosen to sit
 * clear of anything a kernel writes. */
#define FENCE_OFFSET 1024

static NvU32 f2u(float f) { NvU32 u; memcpy(&u, &f, 4); return u; }
static float u2f(NvU32 u) { float f; memcpy(&f, &u, 4); return f; }

/* Input is in[i] = i + 1, both as an integer and (for the float kernels) as a
 * float, so each checker knows exactly what to expect. */
static const char *c_ew_copy(const volatile NvU32 *o) {
  for (unsigned i = 0; i < EW_N; i++) if (o[i] != i + 1) return "copy mismatch";
  return NULL;
}
static const char *c_ew_addidx(const volatile NvU32 *o) {
  for (unsigned i = 0; i < EW_N; i++) if (o[i] != (i + 1) + i) return "add-index mismatch";
  return NULL;
}
static const char *c_ew_addconst(const volatile NvU32 *o) {
  for (unsigned i = 0; i < EW_N; i++) if (o[i] != (i + 1) + 0x1234u) return "add-const mismatch";
  return NULL;
}
static const char *c_ew_fadd(const volatile NvU32 *o) {
  for (unsigned i = 0; i < EW_N; i++) {
    const float want = (float)(i + 1) + (float)(i + 1);
    if (u2f(o[i]) != want) return "fadd mismatch";
  }
  return NULL;
}
static const char *c_ew_fmul(const volatile NvU32 *o) {
  for (unsigned i = 0; i < EW_N; i++) {
    const float v = (float)(i + 1);
    if (u2f(o[i]) != v * v) return "fmul mismatch";
  }
  return NULL;
}
static const char *c_ew_index(const volatile NvU32 *o) {
  for (unsigned i = 0; i < EW_N; i++) if (o[i] != i) return "index mismatch";
  return NULL;
}
static const char *c_ew_ffma(const volatile NvU32 *o) {
  for (unsigned i = 0; i < EW_N; i++) {
    const float v = (float)(i + 1);
    if (u2f(o[i]) != v * v + v) return "ffma mismatch";
  }
  return NULL;
}

static void fill_one(volatile NvU32 *in) { in[0] = 0xf00dbabeu; }
static void fill_ints(volatile NvU32 *in) {
  for (unsigned i = 0; i < EW_N; i++) in[i] = i + 1;
}
static void fill_floats(volatile NvU32 *in) {
  for (unsigned i = 0; i < EW_N; i++) in[i] = f2u((float)(i + 1));
}

static const kernel_case KERNELS[] = {
#ifdef HELIOS_REPEAT_PROBE
    {"store #1", k_store, 1, 1, NULL, c_store},
    {"store #2", k_store, 1, 1, NULL, c_store},
    {"store #3", k_store, 1, 1, NULL, c_store},
    {"store #4", k_store, 1, 1, NULL, c_store},
    {"store #5", k_store, 1, 1, NULL, c_store},
    {"store #6", k_store, 1, 1, NULL, c_store},
    {"store #7", k_store, 1, 1, NULL, c_store},
    {"store #8", k_store, 1, 1, NULL, c_store},
#endif
    {"read a constant bank", k_cbank, 1, 1, NULL, c_cbank},
    {"store an immediate", k_store, 1, 1, NULL, c_store},
    {"store at four offsets", k_offsets, 1, 1, NULL, c_offsets},
    {"load and store", k_copy, 1, 1, fill_one, c_copy},
    {"load, add, store", k_addim, 1, 1, fill_one, c_addim},
    {"32 threads, S2R and a barrier", k_threads, 32, 1, NULL, c_threads},
    /* Indexed elementwise: a real grid, real per-thread addressing, and
     * parameters read from a constant bank. */
    {"S2R ctaid with a 3-wide grid", k_ctaid, 1, 3, NULL, c_ctaid},
    {"computed address", k_addr, 1, 1, NULL, c_addr},
    {"out[tid] = tid, 64 threads", k_tidstore, 64, 1, NULL, c_tidstore},
    {"index probe: block 64, grid 1", k_ew_index, 64, 1, fill_ints, c_ew_index},
    {"index probe: block 1, grid 64", k_ew_index, 1, 64, fill_ints, c_ew_index},
    {"elementwise copy", k_ew_copy, 32, EW_N / 32, fill_ints, c_ew_copy},
    {"elementwise add index", k_ew_addidx, 32, EW_N / 32, fill_ints, c_ew_addidx},
    {"elementwise add constant", k_ew_addconst, 32, EW_N / 32, fill_ints, c_ew_addconst},
    {"elementwise fadd", k_ew_fadd, 32, EW_N / 32, fill_floats, c_ew_fadd},
    {"elementwise fmul", k_ew_fmul, 32, EW_N / 32, fill_floats, c_ew_fmul},
    {"elementwise ffma", k_ew_ffma, 32, EW_N / 32, fill_floats, c_ew_ffma},
};

static void test_gpu_runs_our_machine_code(void) {
  HT_CASE("GPU runs five kernels Hephaestus assembled");

  aether_device d;
  if (aether_device_open(&d, 0) != 0) {
    if (d.failStage == NULL) { printf("skip (no NVIDIA driver)\n"); ht_case_failed = 0; return; }
    HT_FAIL("device open failed at %s", d.failStage);
    HT_END(); return;
  }

  hermes_channel c;
  int rc = hermes_channel_open(&d, &c);
  if (rc != 0) {
    HT_FAIL("channel bring-up failed at %s: %s", c.failStage,
            aether_status_name((unsigned)c.failStatus));
    aether_device_close(&d); HT_END(); return;
  }

  gaia_buffer code, out, qmdbuf, lmem, scratch, in;
  memset(&code, 0, sizeof code); memset(&out, 0, sizeof out);
  memset(&qmdbuf, 0, sizeof qmdbuf); memset(&lmem, 0, sizeof lmem);
  memset(&scratch, 0, sizeof scratch); memset(&in, 0, sizeof in);

  if ((rc = gaia_alloc(&d, &out, 4096, GAIA_SYSMEM)) != 0 ||
      (rc = gaia_map_gpu(&d, &out)) != 0 ||
      (rc = gaia_map_host(&d, &out)) != 0 ||
      (rc = gaia_alloc(&d, &in, 4096, GAIA_SYSMEM)) != 0 ||
      (rc = gaia_map_gpu(&d, &in)) != 0 ||
      (rc = gaia_map_host(&d, &in)) != 0 ||
      (rc = gaia_alloc(&d, &code, 4096, GAIA_VIDMEM)) != 0 ||
      (rc = gaia_map_gpu(&d, &code)) != 0 ||
      (rc = gaia_map_host(&d, &code)) != 0 ||
      (rc = gaia_alloc(&d, &qmdbuf, 4096, GAIA_VIDMEM)) != 0 ||
      (rc = gaia_map_gpu(&d, &qmdbuf)) != 0 ||
      (rc = gaia_map_host(&d, &qmdbuf)) != 0 ||
      (rc = gaia_alloc(&d, &scratch, HERMES_QMD_SCRATCH_BYTES, GAIA_VIDMEM)) != 0 ||
      (rc = gaia_map_gpu(&d, &scratch)) != 0 ||
      (rc = gaia_map_host(&d, &scratch)) != 0 ||
      (rc = gaia_alloc(&d, &lmem, 1024 * 1024, GAIA_VIDMEM)) != 0 ||
      (rc = gaia_map_gpu(&d, &lmem)) != 0) {
    HT_FAIL("buffers: %s", aether_status_name((unsigned)rc)); goto done;
  }

  /* Engine init once, in its own submission, consumed before any launch. */
  {
    hermes_compute_config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.classId = 0xc7c0u;
    cfg.spaVersion = HERMES_SPA_VERSION_SM86;
    cfg.sharedWindow = HERMES_SHARED_WINDOW_DEFAULT;
    cfg.localWindow = HERMES_LOCAL_WINDOW_DEFAULT;
    cfg.localMem = lmem.gpuAddr;
    cfg.localMemSize = lmem.size;
    cfg.smCount = 46;
    hermes_begin(&c);
    hermes_compute_init(&c, 1, &cfg);
    hermes_submit(&d, &c);
    hermes_ring(&c, (volatile NvU32 *)c.userd.hostPtr, c.doorbell, c.token);
  }

  const unsigned n = sizeof KERNELS / sizeof KERNELS[0];
  unsigned passed = 0;
  for (unsigned k = 0; k < n; k++) {
    const kernel_case *kc = &KERNELS[k];
    volatile NvU32 *o = (volatile NvU32 *)out.hostPtr;
    for (unsigned i = 0; i < FENCE_OFFSET / 4 + 4; i++) o[i] = 0;
    if (kc->fill) kc->fill((volatile NvU32 *)in.hostPtr);

    /*
     * Kernel parameters, written into constant bank 0 in CUDA's layout. The
     * bank is just memory we own; the layout matters only because the reference
     * encodings bake these offsets into the instructions that read them.
     */
    {
      volatile NvU8 *cb = (volatile NvU8 *)scratch.hostPtr;
      *(volatile NvU32 *)(cb + HERMES_CBUF0_NTID_X) = kc->blockX;
      *(volatile NvU64 *)(cb + HERMES_CBUF0_PARAM0) = out.gpuAddr;
      *(volatile NvU64 *)(cb + HERMES_CBUF0_PARAM0 + 8) = in.gpuAddr;
      g_expect_cbank = kc->blockX; /* what we wrote at c[0x0][0x0] */
      g_expect_addr = out.gpuAddr;
    }

    /* Pad with EXIT: all-zero SASS is illegal and the SM prefetches past the
     * end of a kernel. */
    {
      hp_word pad = hp_exit(hp_ctrl_safe());
      hp_word *slot = (hp_word *)code.hostPtr;
      for (unsigned i = 0; i < 4096 / sizeof(hp_word); i++) slot[i] = pad;
    }
    hp_word prog[32];
    const unsigned count = kc->build(prog, out.gpuAddr, in.gpuAddr);
    memcpy(code.hostPtr, prog, count * sizeof(hp_word));

    NvU32 qmd[HERMES_QMD_DWORDS];
    static int shown = 0;
    hermes_qmd_build(qmd, code.gpuAddr, scratch.gpuAddr, kc->gridX, 1, 1,
                     kc->blockX, 1, 1);
    if (!shown) {
      shown = 1;
      printf("\n      cbuf0: scratch=0x%llx qmd[20]=%08x qmd[32]=%08x qmd[33]=%08x",
             (unsigned long long)scratch.gpuAddr, qmd[20], qmd[32], qmd[33]);
    }
    memcpy(qmdbuf.hostPtr, qmd, HERMES_QMD_BYTES);
    __asm__ __volatile__("sfence" ::: "memory");

    hermes_begin(&c);
    hermes_launch(&c, qmdbuf.gpuAddr);
    /*
     * The fence lives PAST the output, not inside it.
     *
     * It was at out+64, which is element 16 -- so a 64-element kernel wrote
     * every slot correctly and the harness then overwrote one of them with
     * 0x5EEEEEED. The checker reported "wrote 16 of 64 slots", which reads
     * exactly like a kernel that only launched one warp. Two hours of that
     * hypothesis were bought by a test that damaged the thing it measured.
     */
    hermes_semaphore_release(&c, out.gpuAddr + FENCE_OFFSET, 0x5eeeeeedu);
    hermes_submit(&d, &c);
    hermes_ring(&c, (volatile NvU32 *)c.userd.hostPtr, c.doorbell, c.token);

    /*
     * Wait on the KERNEL'S OWN EFFECT, not only on the fence.
     *
     * A fence proves the channel reached the end of the pushbuffer; the check
     * proves the kernel did its job. Polling the check directly means a broken
     * fence cannot mask a working kernel -- which it did: kernel 0 wrote its
     * magic value correctly while the run was reported as a failure because the
     * semaphore behind it had faulted.
     */
    const NvU64 deadline = now_ns() + 2000000000ull;
    while (now_ns() < deadline && kc->check(o) != NULL) {}

    const char *why = kc->check(o);
    if (why) {
      HT_FAIL("kernel %u (%s): %s [out=%08x %08x %08x %08x errnotif=%08x]", k,
              kc->name, why, o[0], o[1], o[2], o[3],
              ((NvU32 *)c.errnotif.hostPtr)[2]);
    } else {
      passed++;
      printf("\n        %u. %-32s ok", k + 1, kc->name);
    }
  }
  printf("\n      %u/%u kernels ", passed, n);
  HT_EQ_U64(passed, n);

done:
  gaia_free(&d, &in);
  gaia_free(&d, &scratch);
  gaia_free(&d, &lmem);
  gaia_free(&d, &qmdbuf);
  gaia_free(&d, &code);
  gaia_free(&d, &out);
  hermes_channel_close(&d, &c);
  aether_device_close(&d);
  HT_END();
}

void ht_run(void) {
  printf("\nhephaestus — sm_86 encoder vs ptxas\n");
  test_field_placement_primitives();
  test_control_roundtrip();
  test_control_lives_above_bit_105();
  test_zero_operand_instructions();
  test_s2r();
  test_mov_const();
  test_iadd3();
  test_memory();
  test_gpu_runs_our_machine_code();
}
