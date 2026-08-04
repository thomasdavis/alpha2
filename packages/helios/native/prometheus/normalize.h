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
} pr_norm_op;

unsigned pr_emit_normalize(hp_word *prog, pr_norm_op op, unsigned elements);

#endif /* HELIOS_PROMETHEUS_NORMALIZE_H */
