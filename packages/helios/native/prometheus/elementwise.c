/*
 * elementwise.c — see elementwise.h.
 */
#include "ew_layout.h"

/* Registers, named so the sequence below reads as intent rather than numbers. */


/* Does the operation read in[i]? PR_EW_INDEX does not, and emitting a load it
 * never uses would make it a worse probe. */
static int reads_input(pr_ew_op op) {
  return op != PR_EW_INDEX && op != PR_EW_FILL;
}

/* Does the operation read a SECOND input array? */
static int reads_b(pr_ew_op op) {
  /* GELU_GRAD is binary for a different reason from the arithmetic four: its
   * second operand is the incoming GRADIENT, not another term. Leaving it out
   * of this list did not fail to compile and did not fault — the kernel simply
   * multiplied by whatever R_B_VALUE happened to hold, so the forward loss
   * matched cpu_ref exactly and every gradient was wrong. Caught by
   * train-model-native, which compares parameter gradients rather than only the
   * loss; nothing that watches the loss can see a broken backward. */
  return op == PR_EW_ADD || op == PR_EW_SUB || op == PR_EW_MUL ||
         op == PR_EW_DIV || op == PR_EW_GELU_GRAD;
}

/* ADD_INPLACE reads the OUTPUT array as well as writing it, which is the whole
 * point of it -- an accumulate rather than an assign. */
static int reads_output(pr_ew_op op) { return op == PR_EW_ADD_INPLACE; }

unsigned pr_emit_elementwise(hp_word *p, pr_ew_op op) {
  unsigned n = 0;

  /* index = ctaid.x * ntid.x + tid.x, with ntid read from the constant bank in
   * CUDA's layout. */
  p[n++] = hp_s2r(R_INDEX, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_INDEX));
  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(BAR_INDEX));
  p[n++] = hp_mov_imm(R_ESIZE, ELEMENT_BYTES, hp_ctrl_safe());
  p[n++] = hp_imad_const(R_INDEX, R_INDEX, 0, HERMES_CBUF0_NTID_X, R_TID,
                         hp_ctrl_wait(BAR_INDEX));

  /* Addresses: base + index * 4, widened to 64 bits. */
  p[n++] = hp_imad_wide_const(R_OUT_ADDR, R_INDEX, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(0), hp_ctrl_safe());

  if (reads_input(op)) {
    p[n++] = hp_imad_wide_const(R_IN_ADDR, R_INDEX, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
    p[n++] = hp_ldg(R_VALUE, R_IN_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));
  }

  /* ADD_INPLACE's second operand is the output array, not a third pointer. */
  if (reads_output(op)) {
    p[n++] = hp_ldg(R_B_VALUE, R_OUT_ADDR, 0, hp_ctrl_setbar(BAR_LOAD_B));
  }

  if (reads_b(op)) {
    p[n++] = hp_imad_wide_const(R_B_ADDR, R_INDEX, R_ESIZE, 0,
                                HERMES_CBUF0_PARAM_N(2), hp_ctrl_safe());
    p[n++] = hp_ldg(R_B_VALUE, R_B_ADDR, 0, hp_ctrl_setbar(BAR_LOAD_B));
  }

  /* Slot i of the bank always lands in SCALAR_REG[i], so the kernel body can
   * name them positionally and the loader never has to know which op it is. */
  for (unsigned s = 0; s < pr_ew_scalars_read(op); s++)
    p[n++] =
        hp_mov_const(SCALAR_REG[s], 0, HERMES_CBUF0_SCALAR_N(s), hp_ctrl_safe());

  n += pr_ew_emit_op(&p[n], op);

  p[n++] = hp_stg(R_OUT_ADDR, R_RESULT, 0,
                  pr_ew_sets_barrier(op) ? hp_ctrl_wait(BAR_MUFU) : hp_ctrl_safe());
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}
