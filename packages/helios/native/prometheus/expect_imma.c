/*
 * expect_imma.c — the known answer for the int8 tensor-core tile. A wrong IMMA
 * encoding faults; a wrong FRAGMENT LAYOUT returns a plausible-wrong matrix, so
 * this recomputes D[16x8] = A[16x32] * B[8x32]^col from the same deterministic
 * fill and compares every one of the 128 s32 outputs.
 */
#include "expect.h"

static int a_val(int i, int k) { return ((i * 3 + k) % 11) - 5; }
static int b_val(int j, int k) { return ((j * 5 + k * 2) % 9) - 4; }

void pr_fill_imma(volatile NvU32 *a, volatile NvU32 *b) {
  volatile signed char *A = (volatile signed char *)a; /* [16][32] row-major */
  volatile signed char *B = (volatile signed char *)b; /* [8][32] n-major */
  for (int i = 0; i < 16; i++)
    for (int k = 0; k < 32; k++) A[i * 32 + k] = (signed char)a_val(i, k);
  for (int j = 0; j < 8; j++)
    for (int k = 0; k < 32; k++) B[j * 32 + k] = (signed char)b_val(j, k);
}

const char *chk_imma(const volatile NvU32 *o) {
  const volatile int *D = (const volatile int *)o; /* [16][8] row-major */
  int nbad = 0, fi = -1, fj = -1, fg = 0, fw = 0;
  for (int i = 0; i < 16; i++)
    for (int j = 0; j < 8; j++) {
      int want = 0;
      for (int k = 0; k < 32; k++) want += a_val(i, k) * b_val(j, k);
      const int got = D[i * 8 + j];
      if (got != want) {
        if (nbad == 0) { fi = i; fj = j; fg = got; fw = want; }
        nbad++;
      }
    }
  if (nbad) {
    snprintf(pr_msg(), PR_MSG_SIZE,
             "imma: %d/128 wrong; first D[%d][%d]=%d want %d", nbad, fi, fj, fg, fw);
    return pr_msg();
  }
  return NULL;
}
