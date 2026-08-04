/*
 * expect_reduce.c — the expected answer for the reductions and normalisations.
 *
 * These carry their own tolerances rather than the shared MUFU one, and the
 * reason is the reduction tree: a sum of 64 floats accumulates rounding in a
 * different ORDER on the GPU (pairwise, by the tree) than in the loop below
 * (sequential). Both are correct; they are not bit-identical, and the gap grows
 * with the number of elements. Demanding the exact bound used for a single
 * multiply would be demanding an accident.
 */
/*
 * NOTE ON THE COMPARISON'S SHAPE: written as !(err <= tol), never err > tol.
 *
 * They differ on NaN, and only on NaN. Any comparison involving NaN is false,
 * so "err > tol" does not fire and the value is ACCEPTED; "!(err <= tol)" fires
 * and rejects it. Three checkers here were the first form, and a kernel that
 * produced NaN would have passed all three -- which is the exact output a
 * broken normalisation gives, from a zero-divided-by-zero or an
 * infinity-minus-infinity.
 *
 * Found by the runner's mutation pass rather than by reading: it flips an
 * exponent bit in each checked slot and requires the checker to object, and any
 * output in [1,2) becomes a NaN under that flip.
 */
#include "oracle.h"

/*
 * Reductions write ONE value, so only o[0] is checked.
 *
 * The tolerance is relative and small: the additions themselves are exact, but
 * a tree sums in a different ORDER than a sequential loop, and float addition
 * is not associative. Demanding bit-equality with a host loop would be
 * demanding an accident.
 */
const char *chk_sum(const volatile NvU32 *o) {
  float want = 0;
  for (unsigned i = 0; i < PR_N; i++) want += (float)(i + 1);
  const float got = pr_u2f(o[0]);
  if (fabsf(got - want) / want <= 1e-6f) return NULL;
  snprintf(pr_msg(), PR_MSG_SIZE, "sum: %g want %g", (double)got, (double)want);
  return pr_msg();
}
const char *chk_mean(const volatile NvU32 *o) {
  float want = 0;
  for (unsigned i = 0; i < PR_N; i++) want += (float)(i + 1);
  want /= (float)PR_N;
  const float got = pr_u2f(o[0]);
  if (fabsf(got - want) / want <= 1e-6f) return NULL;
  snprintf(pr_msg(), PR_MSG_SIZE, "mean: %g want %g", (double)got, (double)want);
  return pr_msg();
}

/* rmsNorm and softmax both go through MUFU, so both get a tolerance. Softmax
 * additionally sums in tree order, so its denominator differs from a host loop
 * in the last bits. */
const char *chk_rms(const volatile NvU32 *o) {
  float ss = 0;
  for (unsigned i = 0; i < PR_N; i++) ss += pr_in_signed(i) * pr_in_signed(i);
  const float inv = 1.0f / sqrtf(ss / (float)PR_N + PR_RMS_EPS);
  for (unsigned i = 0; i < PR_N; i++) {
    const float want = pr_in_signed(i) * inv, got = pr_u2f(o[i]);
    if (!(fabsf(got - want) / (fabsf(want) + 1e-30f) <= 1e-4f)) {
      snprintf(pr_msg(), PR_MSG_SIZE, "rms: o[%u]=%g want %g", i, (double)got,
               (double)want);
      return pr_msg();
    }
  }
  return NULL;
}

const char *chk_layer(const volatile NvU32 *o) {
  float mean = 0;
  for (unsigned i = 0; i < PR_N; i++) mean += pr_in_signed(i);
  mean /= (float)PR_N;
  float var = 0;
  for (unsigned i = 0; i < PR_N; i++) {
    const float d = pr_in_signed(i) - mean;
    var += d * d;
  }
  var /= (float)PR_N;
  const float inv = 1.0f / sqrtf(var + PR_RMS_EPS);
  for (unsigned i = 0; i < PR_N; i++) {
    const float want = (pr_in_signed(i) - mean) * inv, got = pr_u2f(o[i]);
    if (!(fabsf(got - want) / (fabsf(want) + 1e-30f) <= 1e-3f)) {
      snprintf(pr_msg(), PR_MSG_SIZE, "layerNorm: o[%u]=%g want %g", i,
               (double)got, (double)want);
      return pr_msg();
    }
  }
  return NULL;
}

const char *chk_softmax(const volatile NvU32 *o) {
  float mx = pr_in_signed(0), sum = 0, total = 0;
  for (unsigned i = 0; i < PR_N; i++)
    if (pr_in_signed(i) > mx) mx = pr_in_signed(i);
  for (unsigned i = 0; i < PR_N; i++) sum += expf(pr_in_signed(i) - mx);
  for (unsigned i = 0; i < PR_N; i++) {
    const float want = expf(pr_in_signed(i) - mx) / sum, got = pr_u2f(o[i]);
    total += got;
    if (!(fabsf(got - want) / (fabsf(want) + 1e-30f) <= 1e-4f)) {
      snprintf(pr_msg(), PR_MSG_SIZE, "softmax: o[%u]=%g want %g", i, (double)got,
               (double)want);
      return pr_msg();
    }
  }
  /* And it must be a distribution. A per-element check can pass while the
   * whole thing is scaled wrong only if every element is wrong by the same
   * factor, which this catches. */
  if (!(fabsf(total - 1.0f) <= 1e-4f)) {
    snprintf(pr_msg(), PR_MSG_SIZE, "softmax: sums to %g", (double)total);
    return pr_msg();
  }
  return NULL;
}
