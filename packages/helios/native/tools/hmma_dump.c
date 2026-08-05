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
  /* The qkv projection, where the gap is widest. */
  dump("/tmp/hmma_nn.bin", PR_MM_NN, 512, 1920, 640);
  dump("/tmp/hmma_nt.bin", PR_MM_NT, 512, 1920, 640);
  return 0;
}
