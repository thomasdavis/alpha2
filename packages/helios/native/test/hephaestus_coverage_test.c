/*
 * hephaestus_coverage_test.c — the ISA coverage register, kept honest.
 *
 * WHAT A TEST CAN AND CANNOT DO HERE. It cannot check that the table is
 * COMPLETE against the hardware — nothing short of NVIDIA can. What it can do
 * is make the table impossible to let rot in the three ways a table like this
 * actually rots:
 *
 *   1. an entry loses its reason, so the row survives as a name with no content
 *      and nobody can tell whether closing it would be worth anything;
 *   2. a row says HP_ISA_ENCODED while no encoder emits it, which is the
 *      dangerous direction — a caller reads the table, believes it, and calls
 *      something that does not exist;
 *   3. someone adds an encoder and forgets the row, so the register quietly
 *      stops describing the assembler.
 *
 * (3) is only half-checkable in C — an encoder that exists and has no row is
 * invisible from here. packages/tests/audit-isa-coverage.mjs closes it from the
 * other side, by diffing the table against the 728-instruction ptxas catalogue
 * in isa/sm86-catalogue.json.
 */
#include "../hephaestus/coverage.h"
#include "../hephaestus/sm86.h"
#include "harness.h"
#include <string.h>

void hp_coverage_tests(void);

/* The opcode field, so an "encoded" claim can be checked by emitting. */
static uint64_t opcode_of(hp_word w) { return w.lo & 0xfffULL; }

static const hp_isa_entry *find(const char *m) {
  unsigned n = 0;
  const hp_isa_entry *t = hp_isa_coverage(&n);
  for (unsigned i = 0; i < n; i++)
    if (strcmp(t[i].mnemonic, m) == 0) return &t[i];
  return 0;
}

void hp_coverage_tests(void) {
  unsigned n = 0;
  const hp_isa_entry *t = hp_isa_coverage(&n);

  HT_CASE("the ISA coverage table is well formed and has no empty reasons");
  /* A count that has to be updated deliberately. Not a style rule: a row added
   * without a thought about what it BLOCKS is the row that makes the register
   * decorative. */
  HT_EQ_U64(n, 41);
  for (unsigned i = 0; i < n; i++) {
    HT_TRUE(t[i].mnemonic && t[i].mnemonic[0]);
    /* Long enough to be a reason rather than a restatement of the mnemonic.
     * "float add" is fine for FADD; "TODO" is not. */
    HT_TRUE(t[i].blocks && strlen(t[i].blocks) >= 7);
    HT_TRUE(t[i].state == HP_ISA_ENCODED || t[i].state == HP_ISA_CAPTURED ||
            t[i].state == HP_ISA_MISSING);
    /* No duplicates: two rows for one mnemonic means hp_isa_have answers from
     * whichever comes first, which is not a property anyone reasons about. */
    for (unsigned j = i + 1; j < n; j++)
      HT_TRUE(strcmp(t[i].mnemonic, t[j].mnemonic) != 0);
  }
  HT_END();

  HT_CASE("every mnemonic the table calls ENCODED really is");
  /*
   * The dangerous direction, checked by EMITTING. A row is trusted by callers
   * through hp_isa_have, so a wrong claim here is worse than no table at all.
   *
   * The opcodes are spelt out rather than taken from isa.h so that a change to
   * a constant cannot make this test agree with itself. Where a mnemonic has
   * several opcodes (IADD3 register and immediate forms are DIFFERENT opcodes),
   * one representative is enough — the point is that something emits it.
   */
  HT_EQ_U64(opcode_of(hp_shfl(HP_SHFL_BFLY, 5, 0, 1, 0x1f, hp_ctrl_safe())), 0xf89);
  HT_EQ_U64(opcode_of(hp_ldsm(4, 0, 0, 4, 0, hp_ctrl_safe())), 0x83b);
  HT_EQ_U64(opcode_of(hp_red_add_f32(2, 7, 0, hp_ctrl_safe())), 0x98e);
  HT_EQ_U64(opcode_of(hp_hmma(8, 8, 6, 255, hp_ctrl_safe())), 0x23c);
  HT_EQ_U64(opcode_of(hp_lds(4, 0, 0, hp_ctrl_safe())), 0x984);
  HT_EQ_U64(opcode_of(hp_sts(0, 4, 0, hp_ctrl_safe())), 0x388);
  HT_EQ_U64(opcode_of(hp_ldg(4, 2, 0, hp_ctrl_safe())), 0x981);
  HT_EQ_U64(opcode_of(hp_stg(2, 4, 0, hp_ctrl_safe())), 0x986);
  HT_EQ_U64(opcode_of(hp_fadd(1, 2, 3, hp_ctrl_safe())), 0x221);
  HT_EQ_U64(opcode_of(hp_fmul(1, 2, 3, hp_ctrl_safe())), 0x220);
  HT_EQ_U64(opcode_of(hp_ffma(1, 2, 3, 4, hp_ctrl_safe())), 0x223);
  HT_EQ_U64(opcode_of(hp_fmnmx(1, 2, 3, 1, hp_ctrl_safe())), 0x209);
  HT_EQ_U64(opcode_of(hp_mufu(1, 2, HP_MUFU_RSQ, hp_ctrl_safe())), 0x308);
  HT_EQ_U64(opcode_of(hp_iadd3_reg(1, 2, 3, hp_ctrl_safe())), 0x210);
  HT_EQ_U64(opcode_of(hp_lop3(1, 2, 3, HP_LUT_AND, hp_ctrl_safe())), 0x212);
  HT_EQ_U64(opcode_of(hp_shr_imm(1, 2, 5, hp_ctrl_safe())), 0x819);
  HT_EQ_U64(opcode_of(hp_mov_imm(1, 7, hp_ctrl_safe())), 0x424);
  HT_EQ_U64(opcode_of(hp_s2r(0, HP_SR_TID_X, hp_ctrl_safe())), 0x919);
  HT_EQ_U64(opcode_of(hp_bar_sync(hp_ctrl_safe())), 0xb1d);
  HT_EQ_U64(opcode_of(hp_bra(0, hp_ctrl_safe())), 0x947);
  HT_EQ_U64(opcode_of(hp_nop(hp_ctrl_safe())), 0x918);
  HT_EQ_U64(opcode_of(hp_exit(hp_ctrl_safe())), 0x94d);
  HT_EQ_U64(opcode_of(hp_f2fp_pack(1, 2, 3, hp_ctrl_safe())), 0x23e);
  HT_EQ_U64(opcode_of(hp_isetp_gt_imm(0, 1, 3, hp_ctrl_safe())), 0x80c);
  HT_END();

  HT_CASE("hp_isa_have answers from the table, and only for ENCODED rows");
  HT_TRUE(hp_isa_have("SHFL"));
  HT_TRUE(hp_isa_have("HMMA"));
  HT_TRUE(hp_isa_have("LDSM"));
  /* CAPTURED is NOT available: the bits are known and no encoder emits them,
   * which is precisely the state a caller must not treat as usable. */
  HT_TRUE(!hp_isa_have("LDG.E.128"));
  HT_TRUE(!hp_isa_have("SHFL.reg"));
  HT_TRUE(!hp_isa_have("LDGSTS"));
  HT_TRUE(!hp_isa_have("HFMA2"));
  HT_TRUE(!hp_isa_have("FSETP"));
  /* And a name that is not in the table is not available either — the answer
   * for "never heard of it" and "not implemented" must be the same, or a typo
   * in a capability check reads as a capability. */
  HT_TRUE(!hp_isa_have("MADEUPINSTRUCTION"));
  HT_TRUE(!hp_isa_have(""));
  HT_END();

  HT_CASE("the gaps that are currently paying for themselves are recorded");
  /*
   * Named individually rather than counted, because these four are the ones a
   * measurement has already been charged to. If one is deleted from the table,
   * the finding that motivated it goes with it.
   */
  {
    const hp_isa_entry *e;
    /* Batched transposed-B is 2-3x slow for want of a 128-byte staging
     * request; the instruction that would give it one is encoded and unused. */
    e = find("LDG.E.128"); HT_TRUE(e && e->state == HP_ISA_CAPTURED);
    /* f16 in memory is the only remaining lever with a factor behind it, and
     * packed f16 arithmetic is its precondition. */
    e = find("HFMA2"); HT_TRUE(e && e->state == HP_ISA_MISSING);
    /* cp.async is what "double buffering" would actually mean here; the 3-5%
     * that declined it measured barriers, not the register round trip. */
    e = find("LDGSTS"); HT_TRUE(e && e->state == HP_ISA_MISSING);
    /* SHFL is closed — and the row must say so, because the reduction now
     * depends on it. */
    e = find("SHFL"); HT_TRUE(e && e->state == HP_ISA_ENCODED);
  }
  HT_END();
}
