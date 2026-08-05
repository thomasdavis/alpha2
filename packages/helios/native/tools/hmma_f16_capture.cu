/*
 * mma.m16n8k16 with an f16 ACCUMULATOR, captured.
 *
 * The f32-accumulate form this kernel uses tops out at 45.51 TFLOP/s measured
 * on this card; the f16-accumulate form measures 90.28. It is the only lever
 * left with a factor rather than a percent behind it — issue mix, L2 bandwidth,
 * occupancy, barriers and tile geometry have each been measured and refuted.
 *
 * It also halves the accumulator: two registers per fragment instead of four,
 * which is 16 registers back per warp at this tile and relieves exactly the
 * pressure that sank three register-tile sweeps.
 *
 * WHAT IT COSTS: the k-loop accumulates in f16, eleven bits of mantissa, over
 * K/16 steps — forty of them at this model's 640. That is the first change in
 * this effort that moves the loss, and it must be reported as a number, not
 * assumed small.
 *
 * Build: nvcc -arch=sm_86 -cubin -o f16.cubin hmma_f16_capture.cu
 *        cuobjdump -sass f16.cubin
 */
#include <cuda_fp16.h>
extern "C" __global__ void k(const half *A, const half *B, half *C) {
  unsigned d[2] = {0, 0};
  unsigned const *a = reinterpret_cast<unsigned const *>(A);
  unsigned const *b = reinterpret_cast<unsigned const *>(B);
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
    "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
    : "+r"(d[0]), "+r"(d[1])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  reinterpret_cast<unsigned *>(C)[threadIdx.x] = d[0] + d[1];
}
