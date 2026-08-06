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
  /*
   * Attention's whole score chain in ONE pass: scale, causal mask, softmax.
   *
   * The composed form is three kernels, and each of them reads a [B,H,T,T]
   * tensor and writes another — 23.6 MB of traffic per layer to do arithmetic
   * that is one multiply and one predicated move per element. Fused it is 7.9,
   * and it removes two launches per layer in each direction at a point where
   * 72% of launches already wait for the queue to drain.
   *
   * Slot 2 is the mask, TILED: it is [T,T] while the scores are [B,H,T,T], and
   * the kernel wraps the index at T*T rather than making the caller materialise
   * a copy. Scalar 1 is log2(e) as usual and scalar 2 the score scale.
   */
  PR_NORM_SOFTMAX_MASKED,
  /*
   * The same, with attention's SOFT CAP folded in as well — and this is the
   * variant the model actually runs, because softCap defaults to 30 whenever
   * RoPE is off.
   *
   * The chain composed is FOUR kernels over the same [B,H,T,T] tensor: scale,
   * softCap, causal mask, softmax. Fused it is one.
   *
   * The score scale does not appear as a multiply at all: softCap evaluates
   * c*(1 - 2/(exp2(s0*x) + 1)), and c*tanh(scale*x/c) is the same expression
   * with s0 = 2*log2(e)*scale/c. So the scale rides in a constant the kernel
   * was already going to load. That also keeps the BACKWARD cheap — softCap's
   * gradient then takes the raw scores, which the tape already holds, instead
   * of a scaled intermediate this kernel no longer materialises.
   *
   * Scalar 1 is s0 and scalar 2 is c. log2(e) for the softmax's own exp2 is an
   * immediate, because both constant slots are spoken for and it never varies.
   */
  PR_NORM_SOFTMAX_MASKED_CAP,
  /* out[i] = (a[i] - mean(a)) / sqrt(var(a) + eps) */
  PR_NORM_LAYER,
  /*
   * The same two, WITH the affine folded in — the form the model actually uses.
   *
   *   PR_NORM_RMS_AFFINE     out = a/rms(a) * w[c]
   *   PR_NORM_LAYER_AFFINE   out = (a-mean)/std * w[c] + b[c]
   *
   * A layer norm was three launches: normalise, then a broadcast multiply, then
   * a broadcast add. The multiply and the add each read a full [1536,640]
   * tensor and write another — 7.9 MB of traffic apiece to apply one number per
   * COLUMN, which every thread of a row-per-block kernel already has in hand.
   * There are 37 layer norms in a step and 74 of those extra passes.
   *
   * Slot 2 is the weight and slot 3 the bias, both indexed by feature. They are
   * read at the very start, beside the input, so their latency hides under two
   * reductions instead of under nothing.
   */
  PR_NORM_RMS_AFFINE,
  PR_NORM_LAYER_AFFINE,
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
  /*
   * layerNorm forward that ALSO saves the per-row mean and rstd for the
   * backward to reuse. Same slots 0-3 as PR_NORM_LAYER_AFFINE, plus slot 4 =
   * mean-out and slot 5 = rstd-out (one value per row, written by thread 0).
   */
  PR_NORM_LAYER_AFFINE_STATS,
  /*
   * layerNorm backward given the forward's saved stats. Slots 0-4 as
   * PR_NORM_LAYER_BACKWARD, plus slot 5 = mean-in and slot 6 = rstd-in. Loads
   * them instead of RECOMPUTING mean+variance — two of four reductions gone,
   * bit-identical (same f32 stats). Cuts layerNormBackward from 31% to ~60% of
   * roofline.
   */
  PR_NORM_LAYER_BACKWARD_STATS,
  /*
   * softmax's backward, fused: out = s * (g - sum_j g_j s_j).
   *
   * ops.ts composes it from mul, sum, broadcast, sub and mul — five passes over
   * a [B,H,T,T] tensor for one row-wise reduction and one write. The
   * elementwise half of a step already runs at the card's bandwidth ceiling, so
   * the only way to make it cheaper is to do less of it.
   *
   * Two inputs: param 1 is the softmax OUTPUT and param 2 the incoming
   * gradient. One block per row, as every reduction here is.
   */
  PR_NORM_SOFTMAX_BACKWARD,
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
