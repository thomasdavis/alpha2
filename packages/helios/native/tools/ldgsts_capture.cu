/*
 * LDGSTS — cp.async — captured the way every other instruction here was: write
 * it in CUDA, compile for sm_86, read the bits out of cuobjdump -sass.
 *
 * WHY THIS ONE, AND WHY NOW. The arithmetic of the goal decides it. A step is
 * 79.0 ms of which the GEMM is ~52.8 (67%) and everything else ~22. Thirty
 * thousand tokens a second is a 51.2 ms step, so even a FREE non-GEMM half
 * leaves 57 ms. The GEMM has to get faster; nothing else reaches the target.
 * It sustains 17.1 TFLOP/s over the step against cuBLAS's 24-32 at the same
 * shapes, and every structural lever anyone has tried on it — tile geometry,
 * register tile, warp count, occupancy, tile barriers, ldmatrix, f16
 * accumulate, split-K, L2, issue spacing, bank conflicts — has been measured
 * and is spent, each worth about 1%.
 *
 * What has NOT been tried is the staging path itself. Today each operand
 * element makes this journey:
 *
 *     global -> REGISTER -> F2FP pack -> shared -> LDSM -> tensor fragment
 *
 * cp.async deletes the middle three: it moves global straight to shared with no
 * register destination, so the load-use latency never lands on a register the
 * next instruction has to wait for, and the staging registers (R40-R57 here)
 * stop existing. "Double buffering" was measured at 3-5% and declined — but
 * that measurement removed BARRIERS (HMMA_NOBAR), which is a different thing
 * from removing the register round trip.
 *
 * ⚠️ AND IT IS COUPLED TO f16-IN-MEMORY, which is the honest reason to capture
 * it now rather than later. cp.async COPIES BYTES; it cannot convert f32 to
 * f16 on the way. So the staging can only use it once the operands are f16 in
 * memory — at which point the packs go too, and the operand traffic halves.
 * Those were priced separately as ~1.3-3.7% and ~5-9%, and separately they do
 * not justify the work. Together with this they are a different staging
 * structure rather than a percentage, and this instruction is the gate.
 *
 * WHAT ELSE THE FAMILY NEEDS. cp.async is asynchronous and its completion is
 * tracked by GROUPS, not by the scoreboard barriers everything else here uses:
 *
 *     LDGSTS      issue the copy
 *     LDGDEPBAR   close the current group (cp.async.commit_group)
 *     DEPBAR      wait until at most N groups are outstanding (wait_group N)
 *
 * All three are captured below. Capturing only the first would produce a kernel
 * that starts its copies and reads the shared memory before they land, which on
 * this hardware does not fault — it reads whatever was there.
 *
 * The .128 form additionally requires 16-byte alignment on BOTH addresses, and
 * a misaligned wide access here has already been observed to return the wrong
 * words rather than trap (see hp_ldg_wide's note in sm86.h).
 *
 * Build (nvcc is not on PATH on the pod):
 *   /usr/local/cuda-12.8/bin/nvcc -arch=sm_86 -cubin -o ldgsts.cubin ldgsts_capture.cu
 *   /usr/local/cuda-12.8/bin/cuobjdump -sass ldgsts.cubin
 */
#include <cuda_fp16.h>

/* ---- the three widths, each twice with different registers ---------------- */

extern "C" __global__ void k_cp16(const float *src, float *out) {
  __shared__ float s[1024];
  unsigned dst = __cvta_generic_to_shared(&s[threadIdx.x * 4]);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
               :: "r"(dst), "l"(src + threadIdx.x * 4));
  asm volatile("cp.async.commit_group;\n" ::);
  asm volatile("cp.async.wait_group 0;\n" ::);
  __syncthreads();
  out[threadIdx.x] = s[threadIdx.x * 4];
}

/* A second register assignment for the same form. Two captures of one
 * instruction with different operands is how a field's POSITION is proven
 * rather than guessed — the whole method of this directory. */
extern "C" __global__ void k_cp16_alt(const float *src, float *out, unsigned pad) {
  __shared__ float s[2048];
  unsigned dst = __cvta_generic_to_shared(&s[threadIdx.x * 4 + pad]);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
               :: "r"(dst), "l"(src + threadIdx.x * 4 + pad));
  asm volatile("cp.async.commit_group;\n" ::);
  asm volatile("cp.async.wait_group 0;\n" ::);
  __syncthreads();
  out[threadIdx.x] = s[threadIdx.x];
}

extern "C" __global__ void k_cp8(const float *src, float *out) {
  __shared__ float s[1024];
  unsigned dst = __cvta_generic_to_shared(&s[threadIdx.x * 2]);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"
               :: "r"(dst), "l"(src + threadIdx.x * 2));
  asm volatile("cp.async.commit_group;\n" ::);
  asm volatile("cp.async.wait_group 0;\n" ::);
  __syncthreads();
  out[threadIdx.x] = s[threadIdx.x * 2];
}

extern "C" __global__ void k_cp4(const float *src, float *out) {
  __shared__ float s[1024];
  unsigned dst = __cvta_generic_to_shared(&s[threadIdx.x]);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
               :: "r"(dst), "l"(src + threadIdx.x));
  asm volatile("cp.async.commit_group;\n" ::);
  asm volatile("cp.async.wait_group 0;\n" ::);
  __syncthreads();
  out[threadIdx.x] = s[threadIdx.x];
}

/* ---- the CACHE HINT, which is a whole bit and a real decision ------------- */
/*
 * `.cg` bypasses L1 and goes to L2 only; `.ca` keeps it in L1. A GEMM's staged
 * tile is read once by the block that copies it, so L1 residency buys nothing
 * and evicting other people's lines costs something. Which one the emitter
 * should use is a measurement, but it cannot be measured until both encode, and
 * they differ in one field that only a paired capture reveals.
 */
extern "C" __global__ void k_cp16_cg(const float *src, float *out) {
  __shared__ float s[1024];
  unsigned dst = __cvta_generic_to_shared(&s[threadIdx.x * 4]);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :: "r"(dst), "l"(src + threadIdx.x * 4));
  asm volatile("cp.async.commit_group;\n" ::);
  asm volatile("cp.async.wait_group 0;\n" ::);
  __syncthreads();
  out[threadIdx.x] = s[threadIdx.x * 4];
}

/* ---- wait_group with a NON-ZERO count, which is the whole point ----------- */
/*
 * `wait_group 0` drains everything and is no better than a synchronous load.
 * Pipelining means issuing stage N+1's copies, committing, and waiting until
 * only ONE group is outstanding — so the count field is what makes this
 * instruction worth having, and it must be captured at more than one value or
 * it cannot be told from a constant.
 */
extern "C" __global__ void k_cp_pipeline(const float *src, float *out, int n) {
  __shared__ float s[4096];
  const unsigned lane = threadIdx.x;
  unsigned d0 = __cvta_generic_to_shared(&s[lane * 4]);
  unsigned d1 = __cvta_generic_to_shared(&s[1024 + lane * 4]);

  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :: "r"(d0), "l"(src + lane * 4));
  asm volatile("cp.async.commit_group;\n" ::);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :: "r"(d1), "l"(src + 4096 + lane * 4));
  asm volatile("cp.async.commit_group;\n" ::);
  /* One group still in flight while the first is consumed — the shape a staged
   * GEMM wants, and the reason DEPBAR takes a count at all. */
  asm volatile("cp.async.wait_group 1;\n" ::);
  __syncthreads();
  float acc = s[lane * 4];
  asm volatile("cp.async.wait_group 0;\n" ::);
  __syncthreads();
  acc += s[1024 + lane * 4];
  out[lane] = acc * n;
}

/* wait_group 2, so the count field is proven by what MOVES between three
 * captures rather than by where it looks like it should be. */
extern "C" __global__ void k_cp_wait2(const float *src, float *out) {
  __shared__ float s[4096];
  const unsigned lane = threadIdx.x;
  for (int i = 0; i < 3; i++) {
    unsigned d = __cvta_generic_to_shared(&s[i * 1024 + lane * 4]);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(d), "l"(src + i * 4096 + lane * 4));
    asm volatile("cp.async.commit_group;\n" ::);
  }
  asm volatile("cp.async.wait_group 2;\n" ::);
  __syncthreads();
  out[lane] = s[lane * 4];
}

/* ---- what the staged GEMM would actually emit ----------------------------- */
/*
 * One k-step of the A tile, copied straight to shared with no register in the
 * path, at the f16 element size the operands would have. Having nvcc's version
 * of the exact loop beside the emitter's is the cheapest way to see a
 * difference in instruction count, ordering, or the group discipline.
 */
extern "C" __global__ void k_stage_tile(const __half *a, float *out) {
  __shared__ __half tile[64 * 32];
  const unsigned t = threadIdx.x;             /* 128 threads */
  /* Each thread copies 16 bytes = 8 halves; 128 threads cover 1024 halves,
   * which is a 64x16 tile. */
  unsigned dst = __cvta_generic_to_shared(&tile[t * 8]);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :: "r"(dst), "l"(a + t * 8));
  asm volatile("cp.async.commit_group;\n" ::);
  asm volatile("cp.async.wait_group 0;\n" ::);
  __syncthreads();
  out[t] = __half2float(tile[t * 8]);
}
