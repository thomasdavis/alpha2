/*
 * expect_matmul.c — the expected product, computed from the definition.
 *
 * C[r][c] = sum over k of A[r][k] * B[k][c], written exactly that way. The
 * kernel walks the same sum in the same order, which means this is NOT an
 * independent implementation of the algorithm and cannot catch an error in the
 * algorithm itself. What it does catch is every way the INDEXING can be wrong,
 * which is where matrix multiplication actually goes wrong: a transposed
 * operand, a row stride used as a column stride, an off-by-one trip count. The
 * inputs are chosen so those are visible rather than masked -- see below.
 */
#include "expect.h"

/*
 * Inputs with no symmetry.
 *
 * A[i] = i+1 and B[i] = 2i+3 are both strictly increasing and unequal, so
 * A[r][k]*B[k][c] differs from A[k][r]*B[c][k] for almost every index pair. A
 * matrix of ones, or of row-constant values, would multiply correctly under a
 * transposed index and prove nothing.
 */
static float mm_a(unsigned r, unsigned k) {
  return (float)(r * PR_MM_K + k + 1);
}
static float mm_b(unsigned k, unsigned c) {
  return (float)(2 * (k * PR_MM_N + c) + 3);
}

const char *chk_matmul(const volatile NvU32 *o) {
  for (unsigned r = 0; r < PR_MM_M; r++)
    for (unsigned c = 0; c < PR_MM_N; c++) {
      float want = 0;
      for (unsigned k = 0; k < PR_MM_K; k++) want += mm_a(r, k) * mm_b(k, c);
      const float got = pr_u2f(o[r * PR_MM_N + c]);
      /* Exact: every term and every partial sum here is a small integer, well
       * inside what a float represents exactly, so any difference at all is a
       * real difference and not accumulated rounding. */
      if (got != want) {
        snprintf(pr_msg(), PR_MSG_SIZE, "matmul: c[%u][%u]=%g want %g", r, c,
                 (double)got, (double)want);
        return pr_msg();
      }
    }
  return NULL;
}
