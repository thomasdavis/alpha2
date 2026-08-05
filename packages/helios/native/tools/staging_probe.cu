/* Why does the untransposed-B GEMM run at a quarter of the transposed one?
 *
 * Measured on the real kernels at identical shapes, m512 n1920 k640:
 *
 *     A @ B    (B stored [K,N])     3.31 TFLOP/s    380 us
 *     A @ B^T  (B stored [N,K])    12.98            97
 *
 * The gap survives turning shared-memory staging off, survives padding the
 * shared tiles against bank conflicts, and does not move when the grid axes are
 * swapped so that concurrent blocks share A instead of B. And it is INVERTED
 * with respect to coalescing: the untransposed pattern reads 128 CONTIGUOUS
 * bytes per warp per load — one cache line, one transaction — while the
 * transposed one reads four separate 64-byte half-lines. The better access
 * pattern is the slower one.
 *
 * So this strips everything else away. No tensor cores, no shared memory, no
 * conversion — just the two GLOBAL READ PATTERNS, in the same grid, over the
 * same operand, summing what they load so nothing is optimised out. If the 4x
 * is here, it is the memory system and the fix is a different tile mapping. If
 * the two come out equal, the 4x is somewhere else in the emitted kernel and
 * this rules the reads out.
 *
 * Build: nvcc -arch=sm_86 -O3 -o staging_probe staging_probe.cu
 */
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

/* The real kernel's geometry: 64x64 block tile, 128 threads, k-steps of 16. */
#define BM 64
#define BN 64
#define KSTEP 16
#define THREADS 128
#define WPR (KSTEP / 2)      /* words per tile row: a word is an f16 pair */
#define ITERS ((BN * WPR) / THREADS)

/*
 * UNTRANSPOSED B, stored [K,N]. A thread owns one column and a k-pair; the
 * threads are laid along N so consecutive threads read consecutive addresses.
 * Its pair is N elements apart and its next iteration 4N further.
 */
__global__ void stage_nn(const float *__restrict__ B, float *sink, int N, int K) {
  const int n0 = blockIdx.x * BN;
  const int t = threadIdx.x;
  const int col = t & (BN - 1), kp = t >> 6;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += KSTEP) {
    const float *p = B + (long)(k0 + 2 * kp) * N + n0 + col;
#pragma unroll
    for (int i = 0; i < ITERS; i++) {
      acc += p[(long)4 * i * N];
      acc += p[(long)4 * i * N + N];
    }
  }
  if (acc == 1234.5f) sink[0] = acc;
}

/*
 * TRANSPOSED B, stored [N,K]. A thread owns one column of the tile and a k-pair
 * WITHIN that column, so its two elements are adjacent and its next iteration
 * is 16 columns on.
 */
__global__ void stage_nt(const float *__restrict__ B, float *sink, int N, int K) {
  const int n0 = blockIdx.x * BN;
  const int t = threadIdx.x;
  const int row = t >> 3, kp = t & (WPR - 1);
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += KSTEP) {
    const float *p = B + (long)(n0 + row) * K + k0 + 2 * kp;
#pragma unroll
    for (int i = 0; i < ITERS; i++) {
      acc += p[(long)(THREADS / WPR) * i * K];
      acc += p[(long)(THREADS / WPR) * i * K + 1];
    }
  }
  if (acc == 1234.5f) sink[0] = acc;
}

template <typename K>
static void run(const char *name, K kern, const float *B, float *sink,
                int M, int N, int Kd) {
  /* The real GEMM launches one block per (row block, column block); the reads
   * here depend only on the column block, so the row blocks are replicated
   * exactly as they are in the kernel — they are what makes B re-read. */
  const dim3 grid(N / BN, M / BM);
  for (int i = 0; i < 20; i++) kern<<<grid, THREADS>>>(B, sink, N, Kd);
  CHECK(cudaDeviceSynchronize());

  cudaEvent_t t0, t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
  double best = 1e30;
  for (int s = 0; s < 5; s++) {
    CHECK(cudaEventRecord(t0));
    for (int i = 0; i < 20; i++) kern<<<grid, THREADS>>>(B, sink, N, Kd);
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    float ms; CHECK(cudaEventElapsedTime(&ms, t0, t1));
    if (ms / 20.0 < best) best = ms / 20.0;
  }
  /* Every block reads BN*K floats of B, and there are (M/BM)*(N/BN) blocks. */
  const double bytes = (double)(M / BM) * (N / BN) * BN * Kd * 4.0;
  printf("  %-10s m%-5d n%-6d k%-5d  %7.1f us  %7.0f GB/s\n", name, M, N, Kd,
         best * 1e3, bytes / (best * 1e-3) / 1e9);
}

int main() {
  cudaDeviceProp p; CHECK(cudaGetDeviceProperties(&p, 0));
  printf("%s — the two staging read patterns alone, no mma, no shared memory\n",
         p.name);
  printf("(each block reads BN*K floats of B; 448 GB/s is the card)\n\n");

  const int shapes[][3] = {
    {512, 1920, 640},   /* qkv:      NN 3.31 TFLOP/s vs B^T 12.98 in the GEMM */
    {512, 640, 640},    /* attn proj NN 3.13 vs 12.9 */
    {512, 2560, 640},   /* mlp fc    NN 4.01 vs 13.21 */
    {512, 640, 2560},   /* mlp proj  NN 13.72 — the fast untransposed one */
    {512, 12288, 640},  /* lm head   NN 9.75 vs 14.82 */
  };
  float *B, *sink;
  CHECK(cudaMalloc(&sink, 4));
  for (auto &s : shapes) {
    const int M = s[0], N = s[1], K = s[2];
    CHECK(cudaMalloc(&B, (size_t)N * K * 4));
    CHECK(cudaMemset(B, 0, (size_t)N * K * 4));
    run("A@B", stage_nn, B, sink, M, N, K);
    run("A@B^T", stage_nt, B, sink, M, N, K);
    printf("\n");
    CHECK(cudaFree(B));
  }
  return 0;
}
