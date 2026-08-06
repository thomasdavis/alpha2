/*
 * The wide global load, captured.
 *
 * WHY: the GEMM's staging fetches each operand PAIR — two adjacent k for one
 * row — as two separate 32-bit loads, and a pair is exactly eight contiguous
 * bytes. Per k-step a warp issues sixteen such loads to feed eight tensor
 * instructions, and the measured 42% of the instruction ceiling is about what
 * that mix predicts. One 64-bit load in place of two 32-bit ones halves them,
 * and changes no arithmetic whatever.
 *
 * Build: nvcc -arch=sm_86 -cubin -o ldg64.cubin ldg64_capture.cu
 *        cuobjdump -sass ldg64.cubin
 */
extern "C" __global__ void k64(const float2 *src, float *out) {
  float2 v = src[threadIdx.x];
  out[threadIdx.x] = v.x + v.y;
}

extern "C" __global__ void k128(const float4 *src, float *out) {
  float4 v = src[threadIdx.x];
  out[threadIdx.x] = v.x + v.y + v.z + v.w;
}

/* With an immediate offset, to pin that field the way LDSM's was pinned. */
extern "C" __global__ void k64_off(const float2 *src, float *out) {
  float2 v = src[threadIdx.x + 16];
  out[threadIdx.x] = v.x - v.y;
}
