/*
 * matmul.h — see matmul.c.
 */
#ifndef PROMETHEUS_MATMUL_H
#define PROMETHEUS_MATMUL_H

#include "kernel.h"

/*
 * Emit C[M,N] = A[M,K] * B[K,N] into `p`, returning the instruction count.
 *
 * The shape is a CODEGEN parameter, not a kernel argument: K becomes an
 * immediate in the loop bound and N an immediate in the address arithmetic.
 * That is the same specialisation the SPIR-V generators did -- a kernel is
 * built for a shape and cached against it -- and it costs one regeneration per
 * new shape in exchange for no register holding a dimension and no load to read
 * one.
 *
 * The launch must be M blocks of N threads; nothing here checks that, because
 * nothing here can see it.
 */
unsigned pr_emit_matmul(hp_word *p, unsigned M, unsigned N, unsigned K);

/* Shared memory the emitted matmul needs, which is K floats when the row of A
 * can be staged and zero when it cannot. The launch must match. */
unsigned pr_matmul_shared_bytes(unsigned N, unsigned K);

/*
 * How many threads a matmul block runs — min(N, 1024).
 *
 * One thread per output column was the design, and it put a hard ceiling on N
 * at the hardware's threads-per-block limit: a 105M-parameter model needs
 * N = 1728, 1920 and 12288 for its FFN, qkv and LM head, and every one of them
 * faulted the channel with a compute exception (0xd). The launch is invalid, so
 * the failure is not a wrong answer, but it arrives asynchronously and is
 * reported at whatever flushes next.
 *
 * Threads now walk COLUMNS IN CHUNKS instead, so N is unbounded and the launch
 * geometry never changes. See pr_emit_matmul.
 */
unsigned pr_matmul_block(unsigned N);

#endif /* PROMETHEUS_MATMUL_H */
