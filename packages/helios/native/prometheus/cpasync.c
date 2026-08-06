/*
 * cpasync.c — a hardware validation of the cp.async family, correctly WIRED.
 *
 * The first version of this returned zeroes: the shared read ran before the
 * copy landed. The cause was decoded from a real ptxas pipeline
 * (tools/cpasync_pipeline.cu) and it is one control field — LDGDEPBAR must SET
 * WRITE BARRIER 0 to arm the async scoreboard SB0 that DEPBAR.LE SB0 waits on.
 * With hp_ctrl_safe on the commit, SB0 was never armed and DEPBAR returned at
 * once. That is the whole of the fix, and this kernel exists to prove it on
 * silicon before cp.async is threaded into the GEMM's staging — the change the
 * SASS says deletes 28 of the GEMM k-loop's 42 instructions.
 *
 * It is an identity copy global -> shared -> global where the global->shared leg
 * is performed only by cp.async, so it reuses pr_fill_ints and chk_copy: every
 * output must equal its input, and a wrong wait reads shared the copy has not
 * filled and chk_copy rejects it.
 */
#include "cpasync.h"
#include "../hermes/qmd.h"

enum {
  R_TID = 0,
  R_GADDR = 2,   /* R2:R3 — the global SOURCE address pair */
  R_ESIZE = 4,   /* 4, the element size */
  R_SBYTE = 5,   /* shared destination, a BYTE address (not the .X4 index) */
  R_VAL = 6,
  R_OADDR = 8,   /* R8:R9 — the global DESTINATION address pair */
};

#define BAR_LDS 1
/* The async-copy scoreboard. LDGDEPBAR arms it, DEPBAR waits on it. Barrier 0
 * matches what ptxas uses (control 0x000e2200 on LDGDEPBAR). */
#define SB_ASYNC 0

unsigned pr_emit_cpasync_copy(hp_word *p, unsigned bytes) {
  unsigned n = 0;
  const unsigned elems = bytes / 4u; /* floats a thread copies */

  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(3));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_safe());

  /* Global source in[tid*elems]; the width folds into `elems`. */
  p[n++] = hp_imad_imm(R_GADDR, R_TID, elems, HP_RZ, hp_ctrl_wait(3));
  p[n++] = hp_imad_wide_const(R_GADDR, R_GADDR, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());

  /* Shared destination is a BYTE address — cp.async has no .X4 scaling. */
  p[n++] = hp_imad_imm(R_SBYTE, R_TID, bytes, HP_RZ, hp_ctrl_safe());

  /*
   * THE THREE-INSTRUCTION DANCE, wired. LDGSTS is fire-and-forget; LDGDEPBAR
   * closes the group AND arms scoreboard SB0 via its WRITE BARRIER — this is the
   * bit the first probe lacked; DEPBAR 0 then waits until the group has landed.
   */
  p[n++] = hp_ldgsts(R_SBYTE, R_GADDR, 0, bytes, hp_ctrl_safe());
  p[n++] = hp_ldgdepbar(hp_ctrl_setbar(SB_ASYNC));
  p[n++] = hp_depbar(0, hp_ctrl_safe());
  /* A block barrier before any thread reads. DEPBAR covers THIS thread's copy,
   * which is all a self-read needs, but the GEMM reads slots other warps filled
   * and a probe weaker than the use it clears is not clearing it. */
  p[n++] = hp_bar_sync(hp_ctrl_safe());

  /* Read each float back from shared (LDS .X4 element index) and store it. */
  const unsigned base = R_TID;
  p[n++] = hp_imad_imm(base, R_TID, elems, HP_RZ, hp_ctrl_safe());
  for (unsigned i = 0; i < elems; i++) {
    p[n++] = hp_lds(R_VAL, base, i * 4u, hp_ctrl_setbar(BAR_LDS));
    p[n++] = hp_iadd3_imm(R_OADDR, base, i, hp_ctrl_safe());
    p[n++] = hp_imad_wide_const(R_OADDR, R_OADDR, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM_N(0), hp_ctrl_safe());
    p[n++] = hp_stg(R_OADDR, R_VAL, 0, hp_ctrl_wait(BAR_LDS));
  }
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}
