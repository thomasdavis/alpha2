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

/* Shared memory the staged kernel needs, in bytes; zero when it stages nothing.
 * The launch must declare exactly this — a kernel that touches shared memory it
 * did not ask for faults. */
unsigned pr_hmma_shared(void);

/* Launch geometry. gridX = M / rows, gridY = N / cols, gridZ = batch. */
unsigned pr_hmma_block_rows(void);
unsigned pr_hmma_block_cols(void);
unsigned pr_hmma_threads(void);

/* Per-thread registers this kernel needs declared. The accumulator alone is
 * TM*TN*4 of them, so it is well above the 48 the rest of the stack uses and
 * has to be declared per program. */
unsigned pr_hmma_regs(void);

/*
 * Which operand is stored transposed.
 *
 * NN and NT were the whole world while autograd could reach a weight gradient
 * by materialising a transpose. It cannot afford to: `dL/dB = G^T @ A` runs
 * once per weight per step, 54 times at 18 layers, and the transpose it
 * allocates is up to 24 MiB for the LM head. TA computes it directly.
 *
 * TA costs NOTHING extra in this kernel, which is why it is worth having here
 * rather than in the scalar fallback. A fragment register already takes two
 * loads and a pack — the operands are f32 in memory and f16 in the fragment —
 * so a strided A is the same instruction count as a contiguous one, only a
 * different immediate offset. The B-untransposed branch already addresses
 * exactly this way.
 */
typedef enum {
  PR_MM_NN = 0, /* A[M,K] @ B[K,N] */
  PR_MM_NT = 1, /* A[M,K] @ B[N,K]^T */
  PR_MM_TA = 2, /* A[K,M]^T @ B[K,N] */
} pr_mm_kind;

/* Emit C[M,N] for the given operand layout, assigning or ACCUMULATING into C.
 * Returns the instruction count. */
unsigned pr_emit_hmma(hp_word *p, unsigned M, unsigned N, unsigned K,
                      pr_mm_kind kind, int accumulate);

#endif /* PROMETHEUS_HMMA_H */
