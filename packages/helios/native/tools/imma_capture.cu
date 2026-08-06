/* imma_capture.cu — capture the sm_86 integer tensor-core MMA (IMMA) encoding.
 * mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 — s8xs8 -> s32, the int8
 * tensor core. Two variants with different registers/immediates so a field can
 * be told from a constant that sits in it (the sec5 recipe). Build:
 *   export PATH=/usr/local/cuda-12.8/bin:$PATH
 *   nvcc -arch=sm_86 -cubin -o imma.cubin imma_capture.cu && cuobjdump -sass imma.cubin
 */
#include <cstdint>

// Variant A: accumulator in c0..c3, a in a0..a3, b in b0..b1.
__global__ void imma_A(int *C, const int *A, const int *B) {
  int a0=A[0],a1=A[1],a2=A[2],a3=A[3], b0=B[0],b1=B[1];
  int c0=0,c1=0,c2=0,c3=0;
  asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3)
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
  C[0]=c0;C[1]=c1;C[2]=c2;C[3]=c3;
}

// Variant B: different accumulator source (a separate c) to expose the C-source field.
__global__ void imma_B(int *C, const int *A, const int *B, const int *Cin) {
  int a0=A[0],a1=A[1],a2=A[2],a3=A[3], b0=B[0],b1=B[1];
  int d0,d1,d2,d3, e0=Cin[0],e1=Cin[1],e2=Cin[2],e3=Cin[3];
  asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3)
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
      "r"(e0),"r"(e1),"r"(e2),"r"(e3));
  C[0]=d0;C[1]=d1;C[2]=d2;C[3]=d3;
}
