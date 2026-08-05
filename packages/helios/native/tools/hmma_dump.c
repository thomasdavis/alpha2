/*
 * hmma_dump.c — emit the tensor-core GEMM for two operand layouts and write
 * both, so nvdisasm can say what actually differs.
 *
 * WHY: at identical shapes the untransposed-B GEMM runs at a quarter of the
 * transposed one (3.31 against 12.98 TFLOP/s on m512 n1920 k640). A CUDA
 * replica of the two staging READ patterns alone measures them equal to within
 * 0.4%, which rules the memory system out and says the difference is in the
 * instruction stream. The two emitters differ by about fifteen instructions;
 * this is how to see which.
 *
 * Usage on the box:
 *   node native/build-stack.mjs
 *   ./native/.build/hmma_dump
 *   nvdisasm -b SM86 /tmp/hmma_nn.bin > /tmp/nn.s
 *   nvdisasm -b SM86 /tmp/hmma_nt.bin > /tmp/nt.s
 *   diff /tmp/nn.s /tmp/nt.s
 */
#include "../prometheus/hmma.h"
#include <stdio.h>

static void dump(const char *path, pr_mm_kind kind, unsigned M, unsigned N,
                 unsigned K) {
  static hp_word prog[PR_MAX_INSTRUCTIONS];
  const unsigned n = pr_emit_hmma(prog, M, N, K, kind, 0);
  FILE *f = fopen(path, "wb");
  if (!f) { perror(path); return; }
  fwrite(prog, sizeof(hp_word), n, f);
  fclose(f);
  printf("%s: %u instructions\n", path, n);
}

int main(void) {
  /* The qkv projection, where the gap WAS thought to be widest — it was the
   * probe, not the kernel, and both layouts measure 19-21 TFLOP/s here now. */
  dump("/tmp/hmma_nn.bin", PR_MM_NN, 512, 1920, 640);
  dump("/tmp/hmma_nt.bin", PR_MM_NT, 512, 1920, 640);

  /*
   * THE BATCHED ATTENTION SHAPE, where a real 2.2x gap survives isolation.
   *
   * probe-gemm-rate, one case per process, L2 evicted, pool pre-filled:
   *
   *     attn qk  nn  b240 m64 n64 k64   3.38 TFLOP/s   37 us
   *     attn qk  ta  b240 m64 n64 k64   3.45 TFLOP/s   36 us
   *     attn qk  nt  b240 m64 n64 k64   1.54 TFLOP/s   82 us
   *
   * Same FLOPs, same 240 blocks, same bytes. The roofline for these is 4.9
   * TFLOP/s (11 FLOP per byte at 448 GB/s), so nn and ta are at 70% of it and
   * nt is at 31% — the gap is not the roofline and it is not the profiler.
   */
  dump("/tmp/attn_nn.bin", PR_MM_NN, 64, 64, 64);
  dump("/tmp/attn_nt.bin", PR_MM_NT, 64, 64, 64);
  dump("/tmp/attn_ta.bin", PR_MM_TA, 64, 64, 64);
  return 0;
}
