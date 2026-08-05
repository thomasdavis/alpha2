/*
 * reduction.h — reducing an array to a single value.
 *
 * WHAT: emits a tree reduction across one block, using shared memory.
 *
 * WHY a tree and not a loop: a tree over N elements is log2(N) steps, and at
 * these sizes that unrolls to six. Unrolled means no branches, which means the
 * whole family is reachable with the instructions already verified —
 * predication, shared memory and a barrier — rather than waiting on a loop
 * construct. It is also what a real reduction kernel does.
 *
 * WHY one block: a multi-block reduction needs either a second pass or atomics,
 * and both are separable problems. One block reducing PR_N elements is the
 * kernel the rest are built from, and getting it right first is the point.
 *
 * WHAT THIS DELIBERATELY DOES NOT DO: no multi-block reduction and no
 * warp-shuffle fast path. Each is a real optimisation and each would be
 * premature against a reduction that is not yet known-correct.
 *
 * ARBITRARY LENGTHS ARE NOW SUPPORTED, and they were not optional: the halving
 * loop was exact only while the live count stayed even, so a 640-wide layerNorm
 * — the model's own width — reduced over 512 of its 640 features and said
 * nothing. See pr_emit_tree in reduction.c for the fold that fixes it and for
 * why the same defect had already been found once, in cross-entropy.
 */
#ifndef HELIOS_PROMETHEUS_REDUCTION_H
#define HELIOS_PROMETHEUS_REDUCTION_H

#include "kernel.h"

typedef enum {
  PR_RED_SUM,  /* out[0] = sum(a)                     */
  PR_RED_MEAN, /* out[0] = sum(a) * s, with s = 1/N   */
} pr_red_op;

/* Emit the reduction. `elements` equals the block width: every thread
 * contributes exactly one element. Any width is correct; a power of two emits
 * the same program it always did. */
unsigned pr_emit_reduction(hp_word *prog, pr_red_op op, unsigned elements);

/*
 * One value per BLOCK, for tensors larger than a block. Never scales -- a mean
 * divides by the total count once, at the end, and averaging per-block averages
 * is wrong as soon as one block is short.
 *
 * Declared after pr_combine below, which it takes; the declaration is at the
 * bottom of this header for that reason.
 */

/* How a tree step combines two values. */
typedef enum {
  PR_COMBINE_ADD,
  PR_COMBINE_MAX,
} pr_combine;

/*
 * The tree itself, exposed so normalisation can reuse it.
 *
 * Assumes every thread has already written its contribution to shared memory at
 * its own index and that a barrier has followed. Leaves the result in slot 0,
 * with a barrier after the final step, so every thread may then read it.
 *
 * `tid` is the register holding the thread index; the caller owns register
 * assignment and passes in three scratch registers, because a normalisation
 * kernel has live values a reduction does not and cannot have them clobbered.
 */
unsigned pr_emit_tree(hp_word *prog, unsigned elements, pr_combine how,
                      unsigned tid, unsigned lhs, unsigned rhs);

unsigned pr_emit_reduction_partial(hp_word *p, pr_combine how,
                                  unsigned elements);

#endif /* HELIOS_PROMETHEUS_REDUCTION_H */
