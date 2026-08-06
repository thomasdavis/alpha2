/* imma.h — see imma.c. A single 16x8x32 int8 tensor-core tile, to prove the
 * IMMA encoding + fragment layout compute correctly against a CPU reference. */
#ifndef PROMETHEUS_IMMA_H
#define PROMETHEUS_IMMA_H
#include "kernel.h"

/* K used by the single-warp INT8 GEMM known-answer test (16x8xK, one warp). */
#define IMMA_GEMM_K 64u

unsigned pr_emit_imma_tile(hp_word *p);
/* A single-warp INT8 GEMM: D[16x8] s32 = A[16xK] s8 * B[8xK] s8, K a multiple
 * of 32. Fully serialized (no staging) — correctness first; the staged,
 * multi-warp, cp.async version is the throughput follow-on. */
unsigned pr_emit_imma_gemm_16x8(hp_word *p, unsigned K);
#endif
