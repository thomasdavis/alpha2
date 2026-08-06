/* What can the tensor cores on THIS card actually sustain?
 *
 * The native backend's GEMM runs scalar FP32 and reaches ~1.2 TFLOP/s, ~6% of
 * the 3070's 20.3 TFLOP/s FP32 peak. The plan to reach 30,000 tok/s at 105M
 * parameters rests on HMMA, and the published tensor peak (40.6 TFLOP/s dense,
 * FP16 in / FP32 out) is a number from a table, not from this GPU in this
 * container. Before writing a tensor-core GEMM in a from-scratch SASS emitter,
 * measure three things that decide whether the target is reachable and in what
 * precision:
 *
 *   1. mma.sync m16n8k16 f32-accumulate, from registers, no memory traffic
 *      -- the issue-rate ceiling the emitter could ever approach.
 *   2. the same with f16 accumulate -- GeForce Ampere is documented to run
 *      FP32-accumulate tensor ops at half the FP16-accumulate rate, and if that
 *      holds here it doubles the ceiling at a numerical cost worth pricing.
 *   3. cuBLAS at the SHAPES THIS MODEL ACTUALLY USES -- the practical ceiling,
 *      by a vendor library, including memory traffic. A hand-written kernel
 *      that beats this would be a surprise; the fraction of it we reach is the
 *      honest planning number.
 *
 * Build:  nvcc -arch=sm_86 -O3 -o hmma_ceiling hmma_ceiling.cu -lcublas
 *
 * Warmup is by TIME. This card idles at 210 MHz against 2100 max and cannot be
 * clock-locked inside a RunPod container, so a cold measurement understates by
 * up to 4.9x (X61).
 */
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

/* ---------------------------------------------------------------- instruction rate
 *
 * ACC independent accumulators so the measurement is issue-limited rather than
 * dependency-limited: mma.sync has a multi-cycle latency and a chain of them
 * against a single accumulator measures that latency instead of the throughput.
 * The operands are held in registers and never reloaded, so nothing here is
 * memory. `sink` exists only so the compiler cannot delete the loop.
 */
#define ACC 4

__global__ void mma_f32(float *sink, int iters) {
  unsigned a0 = 0x3c003c00u, a1 = 0x3c003c00u, a2 = 0x3c003c00u, a3 = 0x3c003c00u;
  unsigned b0 = 0x3c003c00u, b1 = 0x3c003c00u;
  float d[ACC][4];
#pragma unroll
  for (int i = 0; i < ACC; i++) { d[i][0] = 0.f; d[i][1] = 0.f; d[i][2] = 0.f; d[i][3] = 0.f; }

  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int i = 0; i < ACC; i++) {
      asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[i][0]), "+f"(d[i][1]), "+f"(d[i][2]), "+f"(d[i][3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
  }
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < ACC; i++) s += d[i][0] + d[i][1] + d[i][2] + d[i][3];
  if (s < 0.f) sink[0] = s;      /* never true: the inputs are all +1 */
}

__global__ void mma_f16(float *sink, int iters) {
  unsigned a0 = 0x3c003c00u, a1 = 0x3c003c00u, a2 = 0x3c003c00u, a3 = 0x3c003c00u;
  unsigned b0 = 0x3c003c00u, b1 = 0x3c003c00u;
  unsigned d[ACC][2];
#pragma unroll
  for (int i = 0; i < ACC; i++) { d[i][0] = 0u; d[i][1] = 0u; }

  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int i = 0; i < ACC; i++) {
      asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(d[i][0]), "+r"(d[i][1])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
  }
  unsigned s = 0u;
#pragma unroll
  for (int i = 0; i < ACC; i++) s ^= d[i][0] ^ d[i][1];
  if (s == 0xdeadbeefu) sink[0] = 1.f;
}

/* One mma.sync m16n8k16 is 16*8*16 multiply-adds = 4096 flop, per WARP. */
static double warp_flops() { return 16.0 * 8.0 * 16.0 * 2.0; }

template <typename K>
static void rate(const char *name, K kernel, int blocks, int threads, double peak) {
  float *sink; CHECK(cudaMalloc(&sink, 4));
  const int iters = 4096;

  /* Ramp the clock: this card starts at a tenth of its speed. */
  cudaEvent_t t0, t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
  for (int i = 0; i < 40; i++) kernel<<<blocks, threads>>>(sink, iters);
  CHECK(cudaDeviceSynchronize());

  double best = 0;
  for (int s = 0; s < 5; s++) {
    CHECK(cudaEventRecord(t0));
    for (int i = 0; i < 20; i++) kernel<<<blocks, threads>>>(sink, iters);
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    float ms; CHECK(cudaEventElapsedTime(&ms, t0, t1));
    double warps = (double)blocks * (threads / 32.0);
    double flops = warps * iters * ACC * warp_flops() * 20.0;
    double tflops = flops / (ms * 1e-3) / 1e12;
    if (tflops > best) best = tflops;
  }
  printf("  %-26s %8.2f TFLOP/s   %5.1f%% of %.1f\n", name, best, 100.0 * best / peak, peak);
  CHECK(cudaFree(sink));
}

/* ---------------------------------------------------------------- cuBLAS at real shapes */

static void gemm(cublasHandle_t h, const char *what, int m, int n, int k) {
  half *A, *B; float *C;
  CHECK(cudaMalloc(&A, (size_t)m * k * 2));
  CHECK(cudaMalloc(&B, (size_t)k * n * 2));
  CHECK(cudaMalloc(&C, (size_t)m * n * 4));
  CHECK(cudaMemset(A, 0x3c, (size_t)m * k * 2));
  CHECK(cudaMemset(B, 0x3c, (size_t)k * n * 2));
  const float alpha = 1.f, beta = 0.f;

  /* Column-major cuBLAS computing C = A*B for row-major A,B by swapping. */
  auto once = [&]() {
    cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                 B, CUDA_R_16F, n, A, CUDA_R_16F, k, &beta,
                 C, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  };
  for (int i = 0; i < 50; i++) once();
  CHECK(cudaDeviceSynchronize());

  cudaEvent_t t0, t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
  double best = 0;
  for (int s = 0; s < 5; s++) {
    CHECK(cudaEventRecord(t0));
    for (int i = 0; i < 100; i++) once();
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    float ms; CHECK(cudaEventElapsedTime(&ms, t0, t1));
    double tflops = 2.0 * m * n * k * 100.0 / (ms * 1e-3) / 1e12;
    if (tflops > best) best = tflops;
  }
  printf("  %-26s m%-6d n%-6d k%-6d  %7.2f TFLOP/s\n", what, m, n, k, best);
  CHECK(cudaFree(A)); CHECK(cudaFree(B)); CHECK(cudaFree(C));
}

int main() {
  cudaDeviceProp p; CHECK(cudaGetDeviceProperties(&p, 0));
  double fp32_peak = 2.0 * p.multiProcessorCount * 128 * (p.clockRate * 1e3) / 1e12;
  printf("%s  sm_%d%d  %d SMs  %.0f MHz  -> FP32 peak %.1f TFLOP/s, tensor table peak %.1f\n\n",
         p.name, p.major, p.minor, p.multiProcessorCount, p.clockRate / 1e3, fp32_peak, 2 * fp32_peak);

  /* Fill the card: 4 warps a block, enough blocks for several waves. */
  const int blocks = p.multiProcessorCount * 8, threads = 128;
  printf("instruction rate, from registers (%d blocks x %d threads):\n", blocks, threads);
  rate("mma m16n8k16 f32 acc", mma_f32, blocks, threads, 2 * fp32_peak);
  rate("mma m16n8k16 f16 acc", mma_f16, blocks, threads, 2 * fp32_peak);

  cublasHandle_t h; cublasCreate(&h);
  printf("\ncuBLAS f16 in / f32 acc, at this model's shapes (batch 8 seq 64 = 512 rows):\n");
  gemm(h, "wqkv          fwd", 512, 1920, 640);
  gemm(h, "attn proj     fwd", 512, 640, 640);
  gemm(h, "mlp fc        fwd", 512, 2560, 640);
  gemm(h, "mlp proj      fwd", 512, 640, 2560);
  gemm(h, "lm head       fwd", 512, 12288, 640);
  printf("\n  ... and at 4x the batch, to price giving the GEMM more rows:\n");
  gemm(h, "wqkv          b32", 2048, 1920, 640);
  gemm(h, "mlp fc        b32", 2048, 2560, 640);
  gemm(h, "lm head       b32", 2048, 12288, 640);
  printf("\n  ... and square, to see the shape-independent ceiling:\n");
  gemm(h, "square 4096", 4096, 4096, 4096);
  cublasDestroy(h);
  return 0;
}
