/*
 * What is this card's L2 read bandwidth, and is the GEMM sitting on it?
 *
 * The GEMM's operand traffic, counted as bytes fetched per block times blocks,
 * comes out at 1,700-1,800 GB/s on every shape that runs well — 1713 on qkv,
 * 1696 on the mlp projection, 1793 on the lm head. That is four times DRAM, so
 * it is being served by L2, and the tightness of the cluster is what makes it
 * worth checking: a kernel pinned against a roof looks exactly like that.
 *
 * IT DECIDES A REWRITE. If the GEMM is L2-bandwidth-bound, then keeping the
 * operands as f16 in memory HALVES that traffic and is worth close to a factor,
 * not the ~5% its instruction savings measured (HMMA_NOPACK: the F2FP packs
 * cost 1.3-3.7%). If instead L2 delivers well above 1.8 TB/s, the kernel is
 * bound by something else and the rewrite stays a five-percent change.
 *
 * Method: read a working set that FITS IN L2 (the 3070 has 4 MB) over and over,
 * so every access after the first pass is an L2 hit and DRAM is out of the
 * picture. float4 loads, enough blocks to fill the card, sum so nothing is
 * optimised away. A second size well beyond L2 gives the DRAM figure as a
 * control — if that one does not come out near 448 GB/s, the harness is wrong
 * and neither number should be believed.
 *
 * Build: nvcc -arch=sm_86 -O3 -o l2bw l2_bandwidth.cu
 */
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)

__global__ void readbw(const float4 *__restrict__ src, float *sink,
                       size_t vec4, int reps) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  float acc = 0.f;
  for (int r = 0; r < reps; r++)
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < vec4; i += stride) {
      float4 v = src[i];
      acc += v.x + v.y + v.z + v.w;
    }
  if (acc == 1234.5f) sink[0] = acc;
}

int main() {
  cudaDeviceProp p; CHECK(cudaGetDeviceProperties(&p, 0));
  printf("%s — L2 %.1f MB, %d SMs\n\n", p.name, p.l2CacheSize / 1048576.0,
         p.multiProcessorCount);
  printf("  working set        GB/s     where it lives\n");

  float *sink; CHECK(cudaMalloc(&sink, 4));
  const size_t sizes[] = {512u << 10, 2u << 20, 3u << 20, 8u << 20, 256u << 20};
  const char *where[] = {"L2 (fits)", "L2 (fits)", "L2 (fits)", "L2+DRAM", "DRAM"};
  for (int s = 0; s < 5; s++) {
    const size_t bytes = sizes[s], vec4 = bytes / sizeof(float4);
    float4 *src; CHECK(cudaMalloc(&src, bytes)); CHECK(cudaMemset(src, 0, bytes));
    /* Enough total work that the timing is not launch overhead, and enough
     * blocks that every SM is busy. */
    const int reps = (int)(2048u * 1048576u / bytes);
    const dim3 grid(p.multiProcessorCount * 8), block(256);
    readbw<<<grid, block>>>(src, sink, vec4, 4);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
    CHECK(cudaEventRecord(t0));
    readbw<<<grid, block>>>(src, sink, vec4, reps);
    CHECK(cudaEventRecord(t1)); CHECK(cudaEventSynchronize(t1));
    float ms; CHECK(cudaEventElapsedTime(&ms, t0, t1));
    printf("  %6zu KB        %7.0f     %s\n", bytes >> 10,
           (double)bytes * reps / (ms * 1e-3) / 1e9, where[s]);
    CHECK(cudaFree(src));
  }
  return 0;
}
