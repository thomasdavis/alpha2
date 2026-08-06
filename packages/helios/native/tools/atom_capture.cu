/*
 * The float atomic add, captured the way every other instruction here was.
 *
 * WHY: the embedding gradient is dW[indices[i]] += g[i] — a scatter-add over
 * about 2 MB of reads. It is currently obtained by building a
 * [tokens, vocab] one-hot through seven full-size elementwise passes and
 * running a 24 GFLOP matmul to pull a mostly-zero table back out of it. The
 * scatter needs an atomic because two tokens can share a vocabulary id, and
 * that is the only reason the cheap form was not written.
 *
 * Two variants, because they encode differently and only one is needed: RED is
 * the fire-and-forget reduction (no return value, which is exactly this case)
 * and ATOM returns the old value.
 *
 * Build: nvcc -arch=sm_86 -cubin -o atom.cubin atom_capture.cu
 *        cuobjdump -sass atom.cubin
 */
extern "C" __global__ void k_red(float *dst, const float *src) {
  /* No use of the result, so ptxas should emit RED rather than ATOM. */
  atomicAdd(&dst[threadIdx.x], src[threadIdx.x]);
}

extern "C" __global__ void k_atom(float *dst, const float *src, float *out) {
  out[threadIdx.x] = atomicAdd(&dst[threadIdx.x], src[threadIdx.x]);
}

/* A second register assignment, to prove field POSITIONS rather than guess. */
extern "C" __global__ void k_red_alt(float *dst, const float *src, int off) {
  atomicAdd(&dst[threadIdx.x + off], src[threadIdx.x] * 2.0f);
}
