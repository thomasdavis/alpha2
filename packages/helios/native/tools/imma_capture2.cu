/* imma_capture2.cu — more IMMA variants to pin every field (sec5). Vary dst,
 * srcA, srcB, srcC registers independently. Full 128-bit via nvdisasm. */
#include <cstdint>
#define IMMA(D0,D1,D2,D3, A0,A1,A2,A3, B0,B1, C0,C1,C2,C3) \
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 " \
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n" \
    : "=r"(D0),"=r"(D1),"=r"(D2),"=r"(D3) \
    : "r"(A0),"r"(A1),"r"(A2),"r"(A3),"r"(B0),"r"(B1),"r"(C0),"r"(C1),"r"(C2),"r"(C3))
// dst R4..R7, srcA R16..R19, srcB R20..R21, srcC R24..R27 — all distinct from variant A
__global__ void v_regs(int*O,const int*A,const int*B,const int*C){
  int a0=A[0],a1=A[1],a2=A[2],a3=A[3],b0=B[0],b1=B[1],c0=C[0],c1=C[1],c2=C[2],c3=C[3];
  int d0,d1,d2,d3; IMMA(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1, c0,c1,c2,c3);
  O[0]=d0;O[1]=d1;O[2]=d2;O[3]=d3;
}
