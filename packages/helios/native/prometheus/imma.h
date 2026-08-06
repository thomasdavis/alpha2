/* imma.h — see imma.c. A single 16x8x32 int8 tensor-core tile, to prove the
 * IMMA encoding + fragment layout compute correctly against a CPU reference. */
#ifndef PROMETHEUS_IMMA_H
#define PROMETHEUS_IMMA_H
#include "kernel.h"

/* K used by the single-warp INT8 GEMM known-answer test (16x8xK, one warp). */
#define IMMA_GEMM_K 64u

/* The multi-tile grid GEMM test shape (M, N powers-of-2 multiples of 16/8). */
#define IMMA_GEMM2_M 64u
#define IMMA_GEMM2_N 64u
#define IMMA_GEMM2_K 256u

unsigned pr_emit_imma_tile(hp_word *p);
/* A multi-tile INT8 GEMM: D[MxN] s32 = A[MxK] s8 (row-major) * B[NxK] s8
 * (n-major), one warp per 16x8 output tile over a linearized (M/16 x N/8) grid,
 * K a multiple of 32. Fully serialized (no staging) — correctness first; the
 * staged multi-warp cp.async version is the throughput follow-on. */
unsigned pr_emit_imma_gemm(hp_word *p, unsigned M, unsigned N, unsigned K);
/* One-tile convenience (M=16, N=8): the original single-warp test. */
unsigned pr_emit_imma_gemm_16x8(hp_word *p, unsigned K);
/* Single-warp shared-staged 16x8 tile — verifies int8 fragments read from
 * SHARED via LDS (the staged kernel's core new mechanic). */
unsigned pr_emit_imma_shared_tile(hp_word *p, unsigned K);
#endif
