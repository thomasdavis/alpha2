/*
 * WHICH result does a lane hold, when mma.m16n8k16 accumulates in f16?
 *
 * The f32 form gives four registers and the layout is known: lane (g = lane>>2,
 * l = lane&3) holds (row g, col 2l), (row g, col 2l+1), (row g+8, col 2l),
 * (row g+8, col 2l+1). The f16 form gives TWO registers, each packing two
 * results, and which two is one bit of information that the emitter guessed
 * wrong — it assumed a register pairs the two COLUMNS of one row, and the
 * failures are equally consistent with pairing the two ROWS of one column.
 *
 * Guessing the other arrangement is not the fix. Two wrong layouts can agree on
 * a symmetric case, which is how this kernel's first tile bug survived a 1x1
 * test. So this makes every element of D DISTINCT and has each lane report what
 * it is holding, which answers the question outright.
 *
 * D[r][c] = r*8 + c + 1, so 1..128 with no repeats and every value exact in
 * f16. Built from two rank-one terms: A[r][0] = r with B[0][c] = 8, plus
 * A[r][1] = 1 with B[1][c] = c+1.
 *
 * Build: nvcc -arch=sm_86 -o f16layout hmma_f16_layout.cu && ./f16layout
 */
#include <cstdio>
#include <cuda_fp16.h>

__global__ void probe(float *out) {
  __shared__ half A[16 * 16], B[8 * 16];
  const int t = threadIdx.x;
  for (int i = t; i < 256; i += 32) {
    const int r = i / 16, k = i % 16;
    A[i] = __float2half(k == 0 ? (float)r : k == 1 ? 1.0f : 0.0f);
  }
  for (int i = t; i < 128; i += 32) {
    const int c = i / 16, k = i % 16;   /* B is [n][k] — the col operand */
    B[i] = __float2half(k == 0 ? 8.0f : k == 1 ? (float)(c + 1) : 0.0f);
  }
  __syncthreads();

  /* The canonical A/B fragment gather, so the operands are certainly right and
   * only the RESULT layout is in question. */
  const int g = t >> 2, l = t & 3;
  unsigned a[4], b[2];
  for (int i = 0; i < 4; i++) {
    const int row = g + 8 * (i & 1), k = 2 * l + 8 * (i >> 1);
    a[i] = (unsigned)__half_as_ushort(A[row * 16 + k])
         | ((unsigned)__half_as_ushort(A[row * 16 + k + 1]) << 16);
  }
  for (int i = 0; i < 2; i++) {
    const int col = g, k = 2 * l + 8 * i;
    b[i] = (unsigned)__half_as_ushort(B[col * 16 + k])
         | ((unsigned)__half_as_ushort(B[col * 16 + k + 1]) << 16);
  }
  unsigned d[2] = {0, 0};
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
    "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
    : "+r"(d[0]), "+r"(d[1])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));

  out[t * 4 + 0] = __half2float(__ushort_as_half((unsigned short)(d[0] & 0xffff)));
  out[t * 4 + 1] = __half2float(__ushort_as_half((unsigned short)(d[0] >> 16)));
  out[t * 4 + 2] = __half2float(__ushort_as_half((unsigned short)(d[1] & 0xffff)));
  out[t * 4 + 3] = __half2float(__ushort_as_half((unsigned short)(d[1] >> 16)));
}

int main() {
  float *d, h[128];
  cudaMalloc(&d, 128 * 4);
  probe<<<1, 32>>>(d);
  cudaMemcpy(h, d, 128 * 4, cudaMemcpyDeviceToHost);
  if (cudaDeviceSynchronize() != cudaSuccess) { printf("launch failed\n"); return 1; }
  printf("D[r][c] = r*8 + c + 1.  reg0.lo reg0.hi reg1.lo reg1.hi, decoded as (row,col)\n\n");
  const char *hdr = "lane  g  l |      reg0.lo      reg0.hi      reg1.lo      reg1.hi\n";
  printf("%s", hdr);
  for (int t = 0; t < 32; t++) {
    printf("%4d %2d %2d |", t, t >> 2, t & 3);
    for (int i = 0; i < 4; i++) {
      const int v = (int)h[t * 4 + i];
      if (v < 1 || v > 128) { printf("   %6.1f??  ", h[t * 4 + i]); continue; }
      printf("  %3d=(%2d,%d)", v, (v - 1) / 8, (v - 1) % 8);
    }
    printf("\n");
    if (t == 7) printf("  ...\n"), t = 27;   /* eight lanes is the whole pattern */
  }
  return 0;
}
