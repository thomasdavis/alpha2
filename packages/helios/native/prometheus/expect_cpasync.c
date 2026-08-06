/*
 * expect_cpasync.c — the reference for the cp.async f16 GEMM, and the f16
 * operands it reads.
 *
 * The kernel computes C[M,N] = A[M,K] @ B[N,K]^T with A and B held as f16 in
 * global memory. This provides both halves the harness needs: a fill that writes
 * those f16 operands, and a checker that computes the same product on the host
 * from the SAME values, so a wrong staging, a wrong fragment address or a wrong
 * accumulator shows up as a wrong C at a known (row, col).
 *
 * The values are small non-negative integers, chosen so every operand and every
 * partial sum is exact in f16 (integers below 2048 are). That makes the host
 * reference an EXACT equality rather than a tolerance: an f16 GEMM over
 * f16-exact inputs with f32 accumulation has no rounding to hide behind, so any
 * mismatch is a real defect and not a precision argument.
 */
#include "oracle.h"
#include <string.h>

/* The test shape: 128x64x64 = two row-blocks and four k-steps, so the grid
 * (R_CTA_M advancing block-to-block), the k-loop (base + k addressing and the
 * done predicate) and the staging→LDSM→HMMA→epilogue chain are all exercised in
 * one kernel. N stays one block because the C harness launches gridY=1. */
#define CPA_M 128u  /* two 64-row blocks — exercises the grid (gridX=2) */
#define CPA_N 64u
#define CPA_K 64u   /* four k-steps — exercises the loop */

/* The operand values, shared by the fill and the check so they cannot disagree.
 * Distinct per (row, k) and (col, k), and small — the products reach 12 and the
 * K-sum reaches a few hundred, all exact in f16. */
static float cpa_a(unsigned r, unsigned k) { return (float)((r + 2u * k) % 5u); }
static float cpa_b(unsigned c, unsigned k) { return (float)((c + k) % 4u); }

/*
 * f32 -> f16 (round to nearest even), enough for the exact small integers here.
 * Not a general converter: subnormals underflow to zero and overflow saturates
 * to infinity, neither of which these values reach. Written out rather than
 * pulled from cast.c because this is the HOST side — the point is to feed the
 * kernel operands it must round-trip, computed independently of the device.
 */
static uint16_t cpa_f2h(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  const int32_t exp = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
  const uint32_t mant = x & 0x7fffffu;
  if (f == 0.0f) return (uint16_t)sign;
  if (exp <= 0) return (uint16_t)sign; /* underflow — unreached here */
  if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
  uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
  if (mant & 0x1000u) h++; /* round; the exact integers never trip it */
  return h;
}

/* Write A[M,K] into `a` and B[N,K] into `b`, both packed two f16 per word. */
void pr_fill_cpasync(volatile NvU32 *a, volatile NvU32 *b) {
  for (unsigned r = 0; r < CPA_M; r++)
    for (unsigned k = 0; k < CPA_K; k += 2) {
      const unsigned w = (r * CPA_K + k) / 2u;
      a[w] = (uint32_t)cpa_f2h(cpa_a(r, k)) |
             ((uint32_t)cpa_f2h(cpa_a(r, k + 1)) << 16);
    }
  for (unsigned c = 0; c < CPA_N; c++)
    for (unsigned k = 0; k < CPA_K; k += 2) {
      const unsigned w = (c * CPA_K + k) / 2u;
      b[w] = (uint32_t)cpa_f2h(cpa_b(c, k)) |
             ((uint32_t)cpa_f2h(cpa_b(c, k + 1)) << 16);
    }
}

const char *chk_cpasync(const volatile NvU32 *o) {
  for (unsigned r = 0; r < CPA_M; r++)
    for (unsigned c = 0; c < CPA_N; c++) {
      float want = 0.0f;
      for (unsigned k = 0; k < CPA_K; k++) want += cpa_a(r, k) * cpa_b(c, k);
      const float got = pr_u2f(o[r * CPA_N + c]);
      if (got != want) {
        snprintf(pr_msg(), PR_MSG_SIZE, "cpasync: C[%u,%u]=%g want %g", r, c,
                 (double)got, (double)want);
        return pr_msg();
      }
    }
  return NULL;
}
