/*
 * oracle.c — see oracle.h.
 */
#include "oracle.h"

#include <string.h>

NvU32 pr_f2u(float f) { NvU32 u; memcpy(&u, &f, 4); return u; }
float pr_u2f(NvU32 u) { float f; memcpy(&f, &u, 4); return f; }

/* ---- inputs ------------------------------------------------------------- */

void pr_fill_ints(volatile NvU32 *a, volatile NvU32 *b) {
  (void)b;
  for (unsigned i = 0; i < PR_N; i++) a[i] = i + 1;
}
/* Strictly positive, so log2 and rsqrt are defined everywhere. */
void pr_fill_pos(volatile NvU32 *a, volatile NvU32 *b) {
  (void)b;
  for (unsigned i = 0; i < PR_N; i++) a[i] = pr_f2u((float)(i + 1));
}
/* Alternating sign, so relu and negation have something to do. A relu tested
 * only on positive input tests nothing. */
void pr_fill_signed(volatile NvU32 *a, volatile NvU32 *b) {
  (void)b;
  for (unsigned i = 0; i < PR_N; i++)
    a[i] = pr_f2u((i & 1) ? -(float)(i + 1) : (float)(i + 1));
}

/* Two operands for the binary kernels: a[i] = i+1, b[i] = 2i+3. Distinct, both
 * non-zero so division is defined, and different enough that a kernel which
 * confuses its inputs fails rather than coincidentally agreeing. */
void pr_fill_pair(volatile NvU32 *a, volatile NvU32 *b) {
  for (unsigned i = 0; i < PR_N; i++) {
    a[i] = pr_f2u((float)(i + 1));
    b[i] = pr_f2u((float)(2 * i + 3));
  }
}
float pr_in_a(unsigned i) { return (float)(i + 1); }
float pr_in_b(unsigned i) { return (float)(2 * i + 3); }

float pr_in_pos(unsigned i) { return (float)(i + 1); }
float pr_in_signed(unsigned i) {
  return (i & 1) ? -(float)(i + 1) : (float)(i + 1);
}


static char g_msg[PR_MSG_SIZE];
char *pr_msg(void) { return g_msg; }
