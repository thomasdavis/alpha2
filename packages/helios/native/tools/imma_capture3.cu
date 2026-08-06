/* imma_capture3.cu — force dst != srcA (use srcA after the IMMA so nvcc cannot
 * reuse its registers for the destination) to split the dst and srcA fields. */
#include <cstdint>
__global__ void v_dstNeA(int *O, const int *A, const int *B, const int *C) {
  int a0=A[0],a1=A[1],a2=A[2],a3=A[3], b0=B[0],b1=B[1];
  int d0,d1,d2,d3, e0=C[0],e1=C[1],e2=C[2],e3=C[3];
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
    : "=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3)
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(e0),"r"(e1),"r"(e2),"r"(e3));
  /* srcA (a0..a3) live AFTER the IMMA -> forces distinct registers from dst. */
  O[0]=d0+a0; O[1]=d1+a1; O[2]=d2+a2; O[3]=d3+a3;
}
