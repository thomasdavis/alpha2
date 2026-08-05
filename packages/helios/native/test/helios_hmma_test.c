/*
 * helios_hmma_test.c — the tensor-core GEMM, against answers from algebra.
 *
 * WHY THIS FILE EXISTS SEPARATELY FROM EVERY OTHER MATMUL CHECK. `HMMA.16816`
 * is a WARP instruction: its 32 lanes cooperatively hold one 16x8 output tile
 * and each lane owns a specific quarter of A, B and C. A wrong ENCODING faults
 * and announces itself. A wrong FRAGMENT LAYOUT does not — it returns a finite,
 * plausible, wrong matrix, which is the same failure mode that let X58's halved
 * gradient norm and X60's broken softmax survive a full parity suite.
 *
 * So the expectations here are CLOSED FORMS, not the output of a second matrix
 * multiply. Two structures, because they fail differently:
 *
 *   scale     A[i][k] = i+1, B[k][j] = j+1  ->  C[i][j] = K*(i+1)*(j+1)
 *             Every output element is distinct, so any row or column of the
 *             fragment landing in the wrong place changes the answer. It is
 *             blind to k, deliberately: it is the map of the OUTPUT.
 *
 *   k-align   A[i][k] = [k == i mod K],  B[k][j] = 8*(k+1) + (j mod 8)
 *                                  ->  C[i][j] = 8*((i mod K)+1) + (j mod 8)
 *             A selects ONE k and B's value depends on which. If A's fragment
 *             and B's fragment are indexed by different k -- the exact error a
 *             transposed or rotated layout produces, and the one `scale` cannot
 *             see because a consistent permutation of k leaves a sum alone --
 *             this returns the wrong k's value.
 *
 * BOTH MUST STAY INSIDE f16's EXACT RANGE, and the first version of this file
 * did not. `kalign` used B[k][j] = (k+1)*(j+1), which reaches 16,384 at the
 * largest shape here; f16 represents integers exactly only to 2048 and steps by
 * two above it, so 17*121 = 2057 came back as 2056 and the KERNEL was blamed
 * for a defect in the EXPECTATION. Hence the j mod 8: the column mapping is
 * `scale`'s job, and this case only has to be sensitive to k.
 *
 * With that, every input is an integer under 2048 -- exact in f16 -- and the
 * accumulation is f32, whose exact integers reach 2^24. The comparison can
 * therefore be tight rather than tolerant, which is the point: a loose
 * tolerance on a matrix multiply hides exactly the errors worth finding.
 */
#include "harness.h"

#include "../helios/context.h"
#include "../helios/dispatch.h"
#include "../helios/program.h"
#include "../prometheus/hmma.h"

#include <string.h>

typedef enum { CASE_SCALE, CASE_KALIGN } hmma_case;

static float a_of(hmma_case c, unsigned i, unsigned k, unsigned K) {
  return c == CASE_SCALE ? (float)(i + 1u) : (k == i % K ? 1.0f : 0.0f);
}
static float b_of(hmma_case c, unsigned k, unsigned j) {
  return c == CASE_SCALE ? (float)(j + 1u) : (float)(8u * (k + 1u) + j % 8u);
}
static float want_of(hmma_case c, unsigned i, unsigned j, unsigned K) {
  return c == CASE_SCALE ? (float)K * (float)(i + 1u) * (float)(j + 1u)
                         : (float)(8u * ((i % K) + 1u) + j % 8u);
}

/*
 * One shape, one case, one orientation.
 *
 * `transposed` stores B as [N,K] and calls the transposed entry point. The two
 * orientations exercise DIFFERENT addressing in the emitter -- the transposed
 * one walks a column contiguously and is the orientation the instruction
 * natively wants -- so a pass on one says nothing about the other.
 */
static void run_case(hmma_case cs, unsigned M, unsigned N, unsigned K,
                     unsigned batch, int layout) {
  const int transposed = (layout == 1), transposedA = (layout == 2);
  char name[96];
  snprintf(name, sizeof name, "hmma %s [%u,%u]x[%u,%u]%s b%u",
           cs == CASE_SCALE ? "scale " : "kalign", M, K, K, N,
           transposed ? " B^T" : transposedA ? " A^T" : "", batch);
  HT_CASE(name);

  helios_context ctx;
  if (helios_context_open(&ctx, 0) != 0) {
    HT_FAIL("no device");
    HT_END();
    return;
  }
  if (!pr_hmma_applies(M, N, K)) {
    HT_FAIL("shape does not select the tensor path — the test proves nothing");
    helios_context_close(&ctx);
    HT_END();
    return;
  }

  const helios_tensor ta = helios_tensor_alloc_host(&ctx, (size_t)batch * M * K * 4);
  const helios_tensor tb = helios_tensor_alloc_host(&ctx, (size_t)batch * K * N * 4);
  const helios_tensor tc = helios_tensor_alloc_host(&ctx, (size_t)batch * M * N * 4);
  HT_TRUE(ta && tb && tc);
  if (!ta || !tb || !tc) { helios_context_close(&ctx); HT_END(); return; }

  float *ha = (float *)helios_tensor_host(ta);
  float *hb = (float *)helios_tensor_host(tb);
  float *hc = (float *)helios_tensor_host(tc);

  /*
   * The BATCH PLANE is offset by the plane index, so a kernel that ignores
   * ctaid.z -- or reads a special register that is not ctaid.z, which HP_SR
   * marks as inferred rather than captured -- writes plane 0's answer into
   * every plane and is caught.
   */
  /* A is [M,K] normally and [K,M] for the transposed-A layout — the SAME
   * matrix, stored the other way round, so the expected answer is unchanged
   * and any confusion between the two shows up as a wrong one. */
  for (unsigned p = 0; p < batch; p++)
    for (unsigned i = 0; i < M; i++)
      for (unsigned k = 0; k < K; k++) {
        const float v = a_of(cs, i, k, K) + (float)p;
        if (transposedA) ha[(size_t)p * M * K + (size_t)k * M + i] = v;
        else ha[(size_t)p * M * K + (size_t)i * K + k] = v;
      }

  for (unsigned p = 0; p < batch; p++)
    for (unsigned k = 0; k < K; k++)
      for (unsigned j = 0; j < N; j++) {
        const float v = b_of(cs, k, j);
        if (transposed) hb[(size_t)p * K * N + (size_t)j * K + k] = v;
        else hb[(size_t)p * K * N + (size_t)k * N + j] = v;
      }
  memset(hc, 0, (size_t)batch * M * N * 4);

  const int rc = transposed
      ? hl_matmul_transposed(&ctx, tc, ta, tb, M, N, K, batch)
      : transposedA
      ? hl_matmul_transposed_a(&ctx, tc, ta, tb, M, N, K, batch)
      : hl_matmul(&ctx, tc, ta, tb, M, N, K, batch);
  HT_TRUE(rc == 0);
  HT_TRUE(helios_flush(&ctx) == 0);

  /* The plane offset adds p to every element of A, so it adds p * sum_k B[k][j]
   * to the answer -- computed here from the same closed form, not from a loop
   * over the device's inputs. */
  unsigned bad = 0;
  float firstGot = 0, firstWant = 0;
  unsigned firstI = 0, firstJ = 0, firstP = 0;
  for (unsigned p = 0; p < batch && bad < 1; p++)
    for (unsigned i = 0; i < M; i++)
      for (unsigned j = 0; j < N; j++) {
        float bsum = 0;
        for (unsigned k = 0; k < K; k++) bsum += b_of(cs, k, j);
        const float want = want_of(cs, i, j, K) + (float)p * bsum;
        const float got = hc[(size_t)p * M * N + (size_t)i * N + j];
        const float err = got - want;
        if (err > 1e-2f || err < -1e-2f) {
          if (!bad) { firstGot = got; firstWant = want; firstI = i; firstJ = j; firstP = p; }
          bad++;
          break;
        }
      }
  if (bad)
    HT_FAIL("plane %u [%u,%u]: got %.4f want %.4f", firstP, firstI, firstJ,
            (double)firstGot, (double)firstWant);
  else
    ht_checks++;

  /*
   * The MAP, when asked for. A wrong fragment layout is a permutation, and a
   * permutation is unreadable one element at a time -- the first failing index
   * says only that something moved, never what. Printing got/want over the
   * corner of the matrix shows the shift itself.
   */
  if (bad && getenv("HELIOS_HMMA_DUMP")) {
    const unsigned rowsShown = M < 4 ? M : 4, colsShown = N < 24 ? N : 24;
    printf("\n      got  (rows 0-%u, cols 0-%u)\n", rowsShown - 1, colsShown - 1);
    for (unsigned i = 0; i < rowsShown; i++) {
      printf("       ");
      for (unsigned j = 0; j < colsShown; j++) printf("%7.0f", (double)hc[(size_t)i * N + j]);
      printf("\n");
    }
    printf("      want\n");
    for (unsigned i = 0; i < rowsShown; i++) {
      printf("       ");
      for (unsigned j = 0; j < colsShown; j++)
        printf("%7.0f", (double)want_of(cs, i, j, K));
      printf("\n");
    }
  }

  helios_tensor_release_all(&ctx);
  helios_context_close(&ctx);
  HT_END();
}

void hl_hmma_tests(void) {
  /*
   * Shapes chosen to be the SMALLEST the tile allows and then to grow one axis
   * at a time: 32 rows is one block of rows, 128 columns is one block of
   * columns, and 16 of K is one k-step. A failure at the smallest shape is a
   * defect in code that runs for every shape, which is a far shorter search
   * than a failure that only appears at a model dimension.
   */
  const unsigned rows = pr_hmma_block_rows(), cols = pr_hmma_block_cols();
  /* layout 0 = A@B, 1 = A@B^T, 2 = A^T@B. The third is the weight gradient and
   * it addresses A along a different axis, so a pass on the first two says
   * nothing about it. */
  for (int t = 0; t < 3; t++) {
    run_case(CASE_SCALE, rows, cols, 16, 1, t);
    run_case(CASE_KALIGN, rows, cols, 16, 1, t);
    run_case(CASE_SCALE, rows * 2, cols, 16, 1, t);   /* two row blocks */
    run_case(CASE_SCALE, rows, cols * 2, 16, 1, t);   /* two column blocks */
    run_case(CASE_KALIGN, rows, cols, 64, 1, t);      /* four k-steps */
    run_case(CASE_SCALE, rows * 2, cols * 2, 64, 1, t);
    run_case(CASE_KALIGN, rows, cols, 32, 3, t);      /* three batch planes */
  }
}
