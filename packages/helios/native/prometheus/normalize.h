/*
 * normalize.h — reduce, then map the result back over every element.
 *
 * WHAT: the shape shared by rmsNorm and softmax — every thread contributes to a
 * reduction, every thread then reads the single reduced value and rescales its
 * own element by it.
 *
 * WHY it is its own generator rather than a reduction plus an elementwise pass:
 * the value being normalised is still live in a register when the reduction
 * finishes. Splitting it into two kernels would mean writing the input back to
 * memory and reading it again, which is both slower and a different program.
 *
 * WHAT THIS DELIBERATELY DOES NOT DO: no per-row normalisation over a matrix,
 * no learnable scale or bias. One row of PR_N elements in one block, which is
 * the kernel the batched versions are built from.
 */
#ifndef HELIOS_PROMETHEUS_NORMALIZE_H
#define HELIOS_PROMETHEUS_NORMALIZE_H

#include "kernel.h"

typedef enum {
  /* out[i] = a[i] / sqrt(mean(a^2) + eps) */
  PR_NORM_RMS,
  /* out[i] = exp(a[i] - max(a)) / sum(exp(a - max)) */
  PR_NORM_SOFTMAX,
  /* out[i] = (a[i] - mean(a)) / sqrt(var(a) + eps) */
  PR_NORM_LAYER,
  /*
   * layerNorm's backward, for dx and xhat.
   *
   * FOUR reductions and THREE inputs, which is why it does not fit the shape
   * the other three share: slot 1 x, slot 2 the incoming gradient, slot 3 the
   * weight (indexed by feature, not by element); slot 0 dx, slot 4 xhat.
   * Scalars 1/N and eps.
   *
   * xhat is an output rather than an intermediate because dw = sum_rows(g*xhat)
   * needs it, and recomputing it outside would cost the same two reductions
   * again. dw and db reduce over the OTHER axis and are the caller's job.
   */
  PR_NORM_LAYER_BACKWARD,
} pr_norm_op;

unsigned pr_emit_normalize(hp_word *prog, pr_norm_op op, unsigned elements);

/*
 * Threads a normalize block runs.
 *
 * Softmax over a row wider than a block walks it in chunks and reduces only the
 * per-thread partials, so it returns 1024 and shared memory stays 4 KB. The
 * others hold one element per thread by construction and return `elements`,
 * which the launch guard will reject if a caller ever asks for a feature vector
 * wider than a block.
 */
unsigned pr_normalize_block(pr_norm_op op, unsigned elements);

#endif /* HELIOS_PROMETHEUS_NORMALIZE_H */
