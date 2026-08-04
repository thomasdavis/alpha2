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
 * WHAT THIS DELIBERATELY DOES NOT DO: no multi-block reduction, no arbitrary
 * lengths, no warp-shuffle fast path. Each of those is a real optimisation and
 * each would be premature against a reduction that is not yet known-correct.
 */
#ifndef HELIOS_PROMETHEUS_REDUCTION_H
#define HELIOS_PROMETHEUS_REDUCTION_H

#include "kernel.h"

typedef enum {
  PR_RED_SUM,  /* out[0] = sum(a)                     */
  PR_RED_MEAN, /* out[0] = sum(a) * s, with s = 1/N   */
} pr_red_op;

/* Emit the reduction. `elements` must be a power of two and equal to the block
 * width: every thread contributes exactly one element. */
unsigned pr_emit_reduction(hp_word *prog, pr_red_op op, unsigned elements);

#endif /* HELIOS_PROMETHEUS_REDUCTION_H */
