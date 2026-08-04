/*
 * expect_indexing.c — the expected arrangement.
 *
 * Both checkers verify PLACEMENT, which is the only thing these kernels decide.
 * They are written against the definition -- out[c][r] is in[r][c], out[i][d] is
 * table[ids[i]][d] -- and read the same id function the input generator used, so
 * an id scheme that changes cannot leave the two disagreeing.
 */
#include "expect.h"

const char *chk_transpose(const volatile NvU32 *o) {
  for (unsigned r = 0; r < PR_TR_ROWS; r++)
    for (unsigned c = 0; c < PR_TR_COLS; c++) {
      /* pr_fill_pos writes a[i] = i+1, so in[r][c] is r*COLS + c + 1. */
      const float want = (float)(r * PR_TR_COLS + c + 1);
      const float got = pr_u2f(o[c * PR_TR_ROWS + r]);
      if (got != want) {
        snprintf(pr_msg(), PR_MSG_SIZE, "transpose: o[%u][%u]=%g want %g", c, r,
                 (double)got, (double)want);
        return pr_msg();
      }
    }
  return NULL;
}

const char *chk_embedding(const volatile NvU32 *o) {
  for (unsigned i = 0; i < PR_EMB_TOKENS; i++)
    for (unsigned d = 0; d < PR_EMB_DIM; d++) {
      const float want = (float)(pr_emb_id(i) * PR_EMB_DIM + d + 1);
      const float got = pr_u2f(o[i * PR_EMB_DIM + d]);
      if (got != want) {
        snprintf(pr_msg(), PR_MSG_SIZE, "embedding: o[%u][%u]=%g want %g", i, d,
                 (double)got, (double)want);
        return pr_msg();
      }
    }
  return NULL;
}
