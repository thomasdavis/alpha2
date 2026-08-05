/*
 * reduction.c — see reduction.h.
 */
#include "reduction.h"

enum {
  R_TID = 0,
  R_ADDR = 2, /* R2:R3 */
  R_VALUE = 4,
  R_ESIZE = 5,
  R_LHS = 6,
  R_RHS = 7,
  R_SCALAR = 8,
  R_OUT = 10,  /* R10:R11 */
  R_BLOCK = 12,
  R_GLOBAL = 13,
};

#define BAR_TID 0
#define BAR_LOAD 1
#define BAR_LDS 2
/* The shuffle's own write barrier. 3 is normalize.c's BAR_MUFU and this file is
 * emitted INTO those kernels, so it cannot be reused; 4 and 5 are free in both.
 * SHFL is variable-latency with no interlock, exactly like LDS — see sm86.h. */
#define BAR_SHFL 4
#define BAR_SHFL2 5
#define P_INACTIVE 0 /* set when this thread is PAST the active range */
#define P_LANE 1     /* set when this thread is NOT lane 0 of its warp */

/*
 * One step of the tree.
 *
 * Threads below `stride` add the element `stride` away to their own, and the
 * rest sit out. The predicate is expressed as "tid > stride-1" because the
 * comparison available is GT, and it is used NEGATED -- @!P0 is "tid < stride",
 * the active half. Writing it as the inactive condition and inverting is one
 * instruction cheaper than materialising the active one.
 *
 * The barrier at the end is not optional and not a formality: without it a
 * thread can read a neighbour's slot before that neighbour has written it, and
 * the result is a sum that is wrong by an amount that varies per run.
 */
static unsigned emit_step(hp_word *p, unsigned stride, unsigned active,
                          pr_combine how, unsigned tid, unsigned lhs,
                          unsigned rhs) {
  unsigned n = 0;
  p[n++] = hp_isetp_gt_imm(P_INACTIVE, tid, active - 1, hp_ctrl_safe());
  p[n++] = hp_predicated(hp_lds(lhs, tid, 0, hp_ctrl_setbar(BAR_LDS)),
                         P_INACTIVE, 1);
  p[n++] = hp_predicated(hp_lds(rhs, tid, stride * 4, hp_ctrl_setbar(BAR_LDS)),
                         P_INACTIVE, 1);
  p[n++] = hp_predicated(
      how == PR_COMBINE_MAX
          ? hp_fmnmx(lhs, lhs, rhs, 1, hp_ctrl_wait(BAR_LDS))
          : hp_fadd(lhs, lhs, rhs, hp_ctrl_wait(BAR_LDS)),
      P_INACTIVE, 1);
  p[n++] = hp_predicated(hp_sts(tid, lhs, 0, hp_ctrl_safe()), P_INACTIVE, 1);
  p[n++] = hp_bar_sync(hp_ctrl_safe());
  return n;
}

/*
 * THE TREE HAD TO BE MADE CORRECT AT WIDTHS THAT ARE NOT A POWER OF TWO, and
 * the model's own width was one of them.
 *
 * The loop used to run `stride = elements/2` and halve, which is exact only
 * while the live count stays even. At 640 it goes 320, 160, 80, 40, 20, 10 —
 * all exact — and then 5, where the count is odd. The stride-2 step after it
 * reduces slots 0..3 and NEVER TOUCHES SLOT 4, so a 640-wide layerNorm
 * normalised over 512 of its 640 features. Measured, not reasoned: the output
 * row mean came back 0.80 where a layer norm's is 0 by construction, at widths
 * 20, 40 and 640, while 8, 16, 64, 512 and 1024 were exact to 3.6e-7. Every
 * width this stack had ever been arbitrated at was a power of two — the 2L/64d
 * parity benchmark, the attention softmax at T=64 — so nothing caught it.
 *
 * This is the SECOND instance of the same defect. `pr_cross_entropy_block`
 * returned an unrounded class count into this same tree, and every vocabulary
 * that was not a power of two reduced over part of each row. Fixing it there
 * left the general fault here, one caller away.
 *
 * The fix is one FOLD before the tree: threads below `elements - pot` add the
 * tail onto the head, leaving exactly `pot` live slots for the halving loop,
 * where `pot` is the largest power of two at or below `elements`. It costs one
 * step, needs no extra shared memory, and wastes no threads.
 *
 * The fold's active count is `elements - pot`, NOT its stride. Predicating on
 * the stride the way every other step does would make threads 128..511 read
 * slots 640..1023 of a 640-slot allocation — past the end of this block's
 * shared memory, which does not fault here, it returns other rows' data. That
 * is why `active` is now a parameter separate from `stride` rather than the
 * same number spelt twice.
 *
 * For a power-of-two width `pot == elements`, the fold is skipped and every
 * step has active == stride, so the emitted program is byte-identical to the
 * old one. Widths that already worked cannot regress.
 */
unsigned pr_emit_tree(hp_word *p, unsigned elements, pr_combine how,
                      unsigned tid, unsigned lhs, unsigned rhs) {
  unsigned n = 0;
  unsigned pot = 1;

  /*
   * THE WARP PATH, wrapped back into this function's slot-0 contract.
   *
   * Callers of pr_emit_tree want "the answer is in shared slot 0 and a barrier
   * has run", which is what a whole-tensor reduction needs. The warp reduction
   * leaves the answer in a register in every thread instead, so it is followed
   * by thread 0 publishing it and one barrier. That is two barriers total
   * against the halving tree's ten at 640 wide, and the callers do not change.
   *
   * `lhs` is used as the accumulator and must therefore be loaded first: this
   * function's contract is that the contribution is in SHARED memory, not in a
   * register, which is the one difference from pr_emit_tree_warp_reg.
   */
  if (pr_reduce_wants_warp(elements)) {
    p[n++] = hp_lds(lhs, tid, 0, hp_ctrl_setbar(BAR_LDS));
    /* The accumulator's load must land before the first shuffle reads it, and
     * a shuffle takes its source at issue like any other instruction. */
    p[n++] = hp_nop(hp_ctrl_wait(BAR_LDS));
    n += pr_emit_tree_warp_reg(&p[n], elements, how, tid, lhs, rhs);
    p[n++] = hp_isetp_gt_imm(P_INACTIVE, tid, 0, hp_ctrl_safe());
    p[n++] = hp_predicated(hp_sts(tid, lhs, 0, hp_ctrl_safe()), P_INACTIVE, 1);
    p[n++] = hp_bar_sync(hp_ctrl_safe());
    return n;
  }

  while (pot * 2 <= elements) pot *= 2;
  if (pot != elements)
    n += emit_step(&p[n], pot, elements - pot, how, tid, lhs, rhs);
  for (unsigned stride = pot / 2; stride >= 1; stride >>= 1)
    n += emit_step(&p[n], stride, stride, how, tid, lhs, rhs);
  return n;
}

/*
 * ============================ THE WARP REDUCTION ============================
 *
 * The same reduction with the warp doing its own share in REGISTERS.
 *
 * WHAT THE TREE ABOVE COSTS, measured rather than reasoned
 * (packages/tests/micro-norm-bandwidth.mjs, [1536,640] on a 3070):
 *
 *     add (control)          34.3 us    344 GB/s   100%
 *     rmsNorm                37.2 us    212 GB/s    61%   ONE reduction
 *     layerNorm              65.1 us    121 GB/s    35%   TWO reductions
 *     layerNormBackward     174.2 us     68 GB/s    20%
 *
 * rmsNorm and layerNorm run the same kernel over the same bytes and differ by
 * one reduction, so the difference PRICES a 640-wide tree at ~28 us across
 * 1536 rows. There are 37 of each in a step, so the pair holds ~6.6 ms of an
 * 80.4 ms step and four fifths of that is not memory traffic.
 *
 * WHY IT COSTS THAT. Two compounding reasons, and the second is the one that
 * makes the first expensive:
 *   1. Every step ends in a BLOCK-WIDE BAR.SYNC. A 640-wide row is a fold plus
 *      nine halving steps, so twenty barriers per layer norm.
 *   2. One element per thread makes the block 640 threads, and an SM holds
 *      1536, so only TWO blocks are resident. There is nothing to run while a
 *      block sits at a barrier.
 *
 * A warp needs neither. SHFL.BFLY exchanges registers across the warp's
 * crossbar, so five steps reduce 32 lanes with no barrier, no shared memory and
 * no traffic, and — because BFLY is symmetric — leave the answer in EVERY lane
 * rather than in lane 0, which is what a normalisation wants and what saves the
 * broadcast the old tree needed afterwards.
 *
 * WHAT IS LEFT ABOVE THE WARP is one store, one barrier and a walk over the
 * per-warp partials: 20 slots at 640 wide. So twenty barriers become one.
 *
 * ⚠️ THE SUMMATION ORDER CHANGES, and therefore so do the last bits of the
 * loss. A halving tree adds (0+1)+(2+3)+..., a butterfly adds lane-by-lane
 * within a warp and then walks the warps in order. Both are exact to fp
 * rounding and neither is more correct; "the loss is bit-identical" cannot be
 * the acceptance test for this change, and the known-answer tests use a
 * tolerance sized to the width rather than to zero.
 *
 * WIDTHS NOT DIVISIBLE BY 32 KEEP THE OLD TREE — see pr_emit_tree_warp_reg's
 * guard and pr_reduce_wants_warp. A partial last warp would have inactive lanes
 * inside a SHFL, whose results are undefined; the tail could be folded first,
 * but the model's widths are 640 and 64 and an untested path is worth more than
 * a few percent on widths the model never runs.
 */
int pr_reduce_wants_warp(unsigned elements) {
  return elements >= 32u && (elements % 32u) == 0u;
}

/*
 * `acc` holds this thread's contribution on entry and the TOTAL on exit, in
 * every thread. Shared memory is used only for the cross-warp step, and the
 * caller's shared array is assumed to be at least `elements` floats — which it
 * is, because the caller has already written one value per thread into it.
 *
 * `t0` and `t1` are scratch and are clobbered. `t0` in particular is the
 * shuffle destination and must not alias `acc`: SHFL is variable-latency, so
 * writing into the register the next step still has to read is the
 * write-after-read hazard this stack has now hit six times.
 */
unsigned pr_emit_tree_warp_reg(hp_word *p, unsigned elements, pr_combine how,
                               unsigned tid, unsigned acc, unsigned t0) {
  unsigned n = 0;
  const unsigned nWarps = elements / 32u;

  /*
   * THE BUTTERFLY. Five steps, lane ^= 16, 8, 4, 2, 1.
   *
   * The shuffle SETS a barrier and the combine WAITS on it. A stall count
   * cannot substitute: control.h's opening paragraph is about exactly this, and
   * the vendor compiler's own control field on every SHFL capture decodes to
   * write-barrier 0, which is it saying the same thing.
   *
   * The barrier alternates between two indices so a step's shuffle can issue
   * while the previous step's combine is still retiring. With one index the
   * shuffle would have to wait for its own barrier to be free.
   */
  for (unsigned mask = 16u, i = 0; mask; mask >>= 1, i++) {
    const unsigned bar = (i & 1u) ? BAR_SHFL2 : BAR_SHFL;
    p[n++] = hp_shfl(HP_SHFL_BFLY, t0, acc, mask, hp_shfl_segment(32),
                     hp_ctrl_setbar(bar));
    p[n++] = how == PR_COMBINE_MAX
                 ? hp_fmnmx(acc, acc, t0, 1, hp_ctrl_wait(bar))
                 : hp_fadd(acc, acc, t0, hp_ctrl_wait(bar));
  }

  /* One warp and we are done — every lane already holds the total. */
  if (nWarps == 1u) return n;

  /*
   * ACROSS THE WARPS. Lane 0 of each warp publishes its total to shared[warp].
   *
   * The guard is "lane != 0" expressed as a comparison because that is the
   * comparison this encoder has: tid & 31, then GT 0, used negated. Letting
   * every lane store instead would be a race that happens to write the same
   * value — the pattern pr_emit_reduction's comment already refuses, because it
   * cannot tell a right predicate from a wrong one.
   */
  /* ONE scratch register, not two: the butterfly is finished, so t0 — its
   * shuffle destination — is free, and LOP3 may name it as both a source and
   * the destination because an ALU op reads its operands at issue. Needing only
   * `acc` and `t0` is what lets pr_emit_tree route into this without a third
   * register its own callers do not have: normalize.c holds the live input in
   * R4 and would have had it clobbered. */
  p[n++] = hp_mov_imm(t0, 31u, hp_ctrl_safe());
  p[n++] = hp_lop3(t0, tid, t0, HP_LUT_AND, hp_ctrl_safe());
  p[n++] = hp_isetp_gt_imm(P_LANE, t0, 0, hp_ctrl_safe());
  p[n++] = hp_shr_imm(t0, tid, 5, hp_ctrl_safe());
  p[n++] = hp_predicated(hp_sts(t0, acc, 0, hp_ctrl_safe()), P_LANE, 1);
  p[n++] = hp_bar_sync(hp_ctrl_safe());

  /*
   * Every thread then walks the `nWarps` partials. Unrolled, and with TWO loads
   * in flight on alternating barriers, because a shared load is tens of cycles
   * and a strictly serial chain of twenty of them would put back a fraction of
   * what the barriers cost.
   *
   * Every thread doing the whole walk — rather than one warp doing it and
   * broadcasting — is deliberate: the loads are a BROADCAST (every lane reads
   * the same address, one transaction) and the alternative needs another store
   * and another barrier to get the answer back out.
   */
  p[n++] = hp_lds(acc, HP_RZ, 0, hp_ctrl_setbar(BAR_SHFL));
  for (unsigned w = 1; w < nWarps; w++) {
    const unsigned bar = (w & 1u) ? BAR_SHFL2 : BAR_SHFL;
    const unsigned prev = (w & 1u) ? BAR_SHFL : BAR_SHFL2;
    p[n++] = hp_lds(t0, HP_RZ, w * 4u, hp_ctrl_setbar(bar));
    /* Wait on THIS load's barrier. The first combine also needs the initial
     * LDS, which used BAR_SHFL — and w == 1 waits on BAR_SHFL2, so the
     * accumulator's own load is covered by the previous iteration's wait for
     * every w > 1 and by this special case for w == 1. */
    p[n++] = how == PR_COMBINE_MAX
                 ? hp_fmnmx(acc, acc, t0, 1,
                            w == 1u ? hp_ctrl_waitmask((1u << bar) | (1u << prev))
                                    : hp_ctrl_wait(bar))
                 : hp_fadd(acc, acc, t0,
                           w == 1u ? hp_ctrl_waitmask((1u << bar) | (1u << prev))
                                   : hp_ctrl_wait(bar));
  }
  return n;
}

/*
 * A PARTIAL reduction: each block reduces its own slice and writes one value.
 *
 * WHY THIS EXISTS: the reduction above handles one block, because the tree runs
 * through shared memory and shared memory is per-block. That covers a row of a
 * normalisation, where the row is the block. It does not cover a whole-tensor
 * sum -- the gradient norm, the loss -- where the tensor is millions of
 * elements and no block is that large. Those need two passes, and this is the
 * first: N elements in, one value per block out, which the caller then reduces
 * again.
 *
 * IT DOES NOT SCALE. A mean over the whole tensor divides by the TOTAL count,
 * once, at the very end. Dividing per block and averaging the averages is only
 * correct when every block holds the same number of elements, and it is wrong
 * the moment the last block is short -- silently, by a few percent, in a number
 * whose job is to be compared against a threshold. So the scaling belongs to
 * the final pass and this one only ever combines.
 *
 * THE TWO DIFFERENCES from the single-block form are both in the indexing: the
 * LOAD uses the global thread index rather than the thread id, and the STORE
 * uses the block index rather than zero. Everything between is the same tree.
 */
unsigned pr_emit_reduction_partial(hp_word *p, pr_combine how,
                                   unsigned elements) {
  unsigned n = 0;
  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(BAR_TID));
  p[n++] = hp_s2r(R_BLOCK, HP_SR_CTAID_X, hp_ctrl_setbar(BAR_TID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_safe());

  /* The global element this thread owns. */
  p[n++] = hp_imad_const(R_GLOBAL, R_BLOCK, 0, HERMES_CBUF0_NTID_X, R_TID,
                         hp_ctrl_wait(BAR_TID));
  p[n++] = hp_imad_wide_const(R_ADDR, R_GLOBAL, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(1), hp_ctrl_safe());
  p[n++] = hp_ldg(R_VALUE, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));
  p[n++] = hp_sts(R_TID, R_VALUE, 0, hp_ctrl_wait(BAR_LOAD));
  p[n++] = hp_bar_sync(hp_ctrl_safe());

  n += pr_emit_tree(&p[n], elements, how, R_TID, R_LHS, R_RHS);

  /* Thread 0 writes this block's answer to out[blockIdx]. */
  p[n++] = hp_isetp_gt_imm(P_INACTIVE, R_TID, 0, hp_ctrl_safe());
  p[n++] = hp_predicated(hp_lds(R_LHS, R_TID, 0, hp_ctrl_setbar(BAR_LDS)),
                         P_INACTIVE, 1);
  p[n++] = hp_imad_wide_const(R_OUT, R_BLOCK, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(0), hp_ctrl_safe());
  p[n++] = hp_predicated(hp_stg(R_OUT, R_LHS, 0, hp_ctrl_wait(BAR_LDS)),
                         P_INACTIVE, 1);
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}

unsigned pr_emit_reduction(hp_word *p, pr_red_op op, unsigned elements) {
  unsigned n = 0;

  /* Each thread loads one element and parks it in shared memory. The address
   * register for LDS/STS is an ELEMENT index because of the .X4 mode, so the
   * thread id serves directly. */
  p[n++] = hp_s2r(R_TID, HP_SR_TID_X, hp_ctrl_setbar(BAR_TID));
  p[n++] = hp_mov_imm(R_ESIZE, 4, hp_ctrl_safe());
  p[n++] = hp_imad_wide_const(R_ADDR, R_TID, R_ESIZE, 0,
                              HERMES_CBUF0_PARAM_N(1), hp_ctrl_wait(BAR_TID));
  p[n++] = hp_ldg(R_VALUE, R_ADDR, 0, hp_ctrl_setbar(BAR_LOAD));
  p[n++] = hp_sts(R_TID, R_VALUE, 0, hp_ctrl_wait(BAR_LOAD));
  p[n++] = hp_bar_sync(hp_ctrl_safe());

  /* The tree, unrolled. Halving from elements/2 down to 1. */
  n += pr_emit_tree(&p[n], elements, PR_COMBINE_ADD, R_TID, R_LHS, R_RHS);

  /*
   * Thread 0 alone writes the answer.
   *
   * Every thread reaches here, so the store is predicated on "not (tid > 0)".
   * Letting all of them store would be a race writing the same address, which
   * happens to produce the right value and would hide a genuinely wrong
   * predicate -- so the guard earns its place even where it looks redundant.
   */
  p[n++] = hp_isetp_gt_imm(P_INACTIVE, R_TID, 0, hp_ctrl_safe());
  p[n++] = hp_predicated(hp_lds(R_LHS, R_TID, 0, hp_ctrl_setbar(BAR_LDS)),
                         P_INACTIVE, 1);

  if (op == PR_RED_MEAN) {
    p[n++] = hp_mov_const(R_SCALAR, 0, HERMES_CBUF0_SCALAR, hp_ctrl_safe());
    p[n++] = hp_predicated(
        hp_fmul(R_LHS, R_LHS, R_SCALAR, hp_ctrl_wait(BAR_LDS)), P_INACTIVE, 1);
  }

  p[n++] = hp_imad_wide_const(R_OUT, R_TID, R_ESIZE, 0, HERMES_CBUF0_PARAM_N(0),
                              hp_ctrl_safe());
  p[n++] = hp_predicated(
      hp_stg(R_OUT, R_LHS, 0,
             op == PR_RED_MEAN ? hp_ctrl_safe() : hp_ctrl_wait(BAR_LDS)),
      P_INACTIVE, 1);
  p[n++] = hp_exit(hp_ctrl_safe());
  return n;
}
