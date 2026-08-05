/*
 * hmma.h — the tensor-core GEMM. See hmma.c.
 */
#ifndef PROMETHEUS_HMMA_H
#define PROMETHEUS_HMMA_H

#include "kernel.h"

/*
 * The master switch.
 *
 * OFF ships the scalar matmul, which is correct at every shape and is the
 * arbiter this path is checked against. Kept as a compile-time constant rather
 * than an environment variable so a build cannot half-take it: a kernel that is
 * selected by the dispatcher but not by the launch-geometry helper is an
 * invalid launch, and those fault asynchronously.
 */
#define PR_HMMA_ENABLED 1

/* Whether this shape can use the tensor path. Requires K % 16, M % (16*TM) and
 * N % (8*TN*WARPS); anything else falls back to matmul.c. */
int pr_hmma_applies(unsigned M, unsigned N, unsigned K);

/* Launch geometry. gridX = M / rows, gridY = N / cols, gridZ = batch. */
unsigned pr_hmma_block_rows(void);
unsigned pr_hmma_block_cols(void);
unsigned pr_hmma_threads(void);

/* Per-thread registers this kernel needs declared. The accumulator alone is
 * TM*TN*4 of them, so it is well above the 48 the rest of the stack uses and
 * has to be declared per program. */
unsigned pr_hmma_regs(void);

/* Emit C[M,N] = A[M,K] * B[K,N] (transposedB 0) or A * B^T with B stored [N,K]
 * (transposedB 1). Returns the instruction count. */
unsigned pr_emit_hmma(hp_word *p, unsigned M, unsigned N, unsigned K,
                      int transposedB);

#endif /* PROMETHEUS_HMMA_H */
