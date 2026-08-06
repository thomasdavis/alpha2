/*
 * cpasync_pipeline.cu — how does ptxas WIRE a cp.async pipeline?
 *
 * The three instructions LDGSTS / LDGDEPBAR / DEPBAR encode bit-for-bit, and a
 * validation kernel that used them still read ZEROES: the shared read ran
 * before the copy landed. So the encoding is not the whole contract. cp.async
 * is tracked by an asynchronous SCOREBOARD, and what makes the consumer wait is
 * not only the DEPBAR instruction — it is the CONTROL FIELDS ptxas puts on the
 * instructions around it.
 *
 * This compiles a realistic double-buffered pipeline the way the GEMM will want
 * it: issue a stage's copies, commit the group, and — crucially — CONSUME the
 * previous stage's shared memory while the next is in flight. Reading the SASS
 * of that shows exactly which scoreboard the copies increment, how DEPBAR names
 * it, and what wait the shared LOAD after the barrier carries. That is the
 * missing piece, and it is read, not guessed.
 *
 * The pattern below mirrors the GEMM's k-loop:
 *   prologue:  copy tile 0, commit
 *   loop:      copy tile n+1, commit; wait until 1 group left; __syncthreads;
 *              read tile n from shared and accumulate
 *   epilogue:  wait 0; read the last tile
 *
 * Build:
 *   /usr/local/cuda-12.8/bin/nvcc -arch=sm_86 -cubin -o cpp.cubin cpasync_pipeline.cu
 *   /usr/local/cuda-12.8/bin/cuobjdump -sass cpp.cubin
 *
 * Read for: LDGSTS's control field (which SBn it sets), DEPBAR.LE SBn, and the
 * LDS after __syncthreads (does it carry a wait mask, and on which barrier).
 */
#include <cuda_fp16.h>

#define TILE 1024   /* halves a stage copies */
#define STAGES 2

extern "C" __global__ void k_pipeline(const __half *g, float *out, int nTiles) {
  __shared__ __half sh[STAGES][TILE];
  const unsigned t = threadIdx.x; /* 128 threads, each copies 8 halves = 16 B */

  /* Prologue: launch stage 0. */
  {
    unsigned dst = __cvta_generic_to_shared(&sh[0][t * 8]);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(dst), "l"(g + t * 8));
    asm volatile("cp.async.commit_group;\n" ::);
  }

  float acc = 0.f;
  for (int n = 0; n < nTiles; n++) {
    const int cur = n & 1;
    const int nxt = (n + 1) & 1;

    /* Launch the NEXT stage while this one is consumed — the whole point. */
    if (n + 1 < nTiles) {
      unsigned dst = __cvta_generic_to_shared(&sh[nxt][t * 8]);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                   :: "r"(dst), "l"(g + (n + 1) * TILE + t * 8));
      asm volatile("cp.async.commit_group;\n" ::);
    }

    /* Wait until at most one group is still in flight, then a block barrier so
     * every thread sees the current stage complete. */
    asm volatile("cp.async.wait_group 1;\n" ::);
    __syncthreads();

    /* Consume the current stage from shared. */
    #pragma unroll
    for (int i = 0; i < 8; i++) acc += __half2float(sh[cur][t * 8 + i]);

    __syncthreads(); /* before the buffer is refilled */
  }
  out[t] = acc;
}

/* A single-stage variant, to see the degenerate wiring the failed probe used:
 * copy, commit, wait 0, read. If THIS one's LDS carries a wait the probe did
 * not, that difference IS the bug. */
extern "C" __global__ void k_single(const __half *g, float *out) {
  __shared__ __half sh[TILE];
  const unsigned t = threadIdx.x;
  unsigned dst = __cvta_generic_to_shared(&sh[t * 8]);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :: "r"(dst), "l"(g + t * 8));
  asm volatile("cp.async.commit_group;\n" ::);
  asm volatile("cp.async.wait_group 0;\n" ::);
  __syncthreads();
  float acc = 0.f;
  #pragma unroll
  for (int i = 0; i < 8; i++) acc += __half2float(sh[t * 8 + i]);
  out[t] = acc;
}
