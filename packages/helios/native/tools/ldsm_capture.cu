/*
 * ldmatrix, captured the way every other instruction in this encoder was: write
 * it in CUDA, compile for sm_86, read the bits out of cuobjdump -sass.
 *
 * WHY: the staged GEMM issues SIXTEEN shared loads per k-step to feed EIGHT
 * tensor instructions. Each m16n8k16 A fragment is four registers and takes
 * four LDS; each B fragment is two and takes two. ldmatrix loads a whole
 * fragment — four registers, and the 8x8 transpose the layout needs — in ONE
 * instruction, and it is the difference between this kernel's 14-18 TFLOP/s and
 * cuBLAS's 24-32 at the same shapes.
 *
 * Each variant is in its own kernel so the SASS cannot interleave them.
 *
 * Build: nvcc -arch=sm_86 -cubin -o ldsm.cubin ldsm_capture.cu
 *        cuobjdump -sass ldsm.cubin
 */
extern "C" __global__ void k_x4(unsigned *out) {
  __shared__ unsigned s[1024];
  unsigned a, b, c, d;
  unsigned addr = __cvta_generic_to_shared(&s[threadIdx.x * 2]);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr));
  out[threadIdx.x] = a + b + c + d;
}

extern "C" __global__ void k_x2(unsigned *out) {
  __shared__ unsigned s[1024];
  unsigned a, b;
  unsigned addr = __cvta_generic_to_shared(&s[threadIdx.x * 2]);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(a), "=r"(b) : "r"(addr));
  out[threadIdx.x] = a + b;
}

extern "C" __global__ void k_x1(unsigned *out) {
  __shared__ unsigned s[1024];
  unsigned a;
  unsigned addr = __cvta_generic_to_shared(&s[threadIdx.x * 2]);
  asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
               : "=r"(a) : "r"(addr));
  out[threadIdx.x] = a;
}

/* The transposing form, which is what a row-major tile needs to become the
 * column-major B fragment m16n8k16 wants. */
extern "C" __global__ void k_x4t(unsigned *out) {
  __shared__ unsigned s[1024];
  unsigned a, b, c, d;
  unsigned addr = __cvta_generic_to_shared(&s[threadIdx.x * 2]);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr));
  out[threadIdx.x] = a + b + c + d;
}

/* A second register assignment for the same variant: two captures of one
 * instruction with different operands is how a field's POSITION is proven
 * rather than guessed. */
extern "C" __global__ void k_x4_alt(unsigned *out, unsigned pad) {
  __shared__ unsigned s[1024];
  unsigned a, b, c, d;
  unsigned addr = __cvta_generic_to_shared(&s[threadIdx.x * 2 + (pad & 7)]);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr));
  out[threadIdx.x + 1] = a * b + c * d;
}
