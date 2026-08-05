/*
 * dispatch.c — see dispatch.h.
 */
#include "dispatch.h"

#include <string.h>

/* Reinterpret a float as the word the constant bank actually stores. The bank
 * is untyped; this is only a spelling. */
static NvU32 bits(float f) {
  NvU32 u;
  memcpy(&u, &f, 4);
  return u;
}

/*
 * The common tail of every dispatch: look the program up, resolve the handles,
 * launch.
 *
 * Handles are resolved HERE rather than by each caller so that a stale one is
 * caught in one place. helios_tensor_addr returns zero for a dead handle, and a
 * launch with a null buffer address would fault the channel -- so the check is
 * explicit and the operation simply fails instead.
 */
static int run(helios_context *ctx, helios_key key, const helios_tensor *ts,
               unsigned nts, const NvU32 *scalars, unsigned nscalars) {
  const helios_program *p = helios_program_get(key);
  if (!p) return -1;

  NvU64 addrs[HERMES_CBUF0_PARAM_COUNT];
  if (nts > HERMES_CBUF0_PARAM_COUNT) return -1;
  for (unsigned i = 0; i < nts; i++) {
    addrs[i] = helios_tensor_addr(ts[i]);
    if (addrs[i] == 0) return -1;
  }
  /*
   * BATCHED: enqueue and let the caller flush.
   *
   * Launches queue into one submission and the host waits once, instead of
   * spinning on a fence per launch. A WAIT_FOR_IDLE between them is what makes
   * that safe -- dispatches on one channel pipeline and do not serialise, which
   * the synchronous design hid because its fence wait was the barrier.
   *
   * The caller flushes before reading device memory; in TypeScript that barrier
   * sits on the tensor's  property rather than at call sites, because one
   * missed read site is a bug with no symptom where it was caused.
   */
  /*
   * SYNCHRONOUS. Switch this to helios_enqueue to turn batching on.
   *
   * It is off because at SCALE it still hangs, while the two-launch case now
   * passes -- so what remains is something that only appears with many launches
   * queued, and the pushbuffer wrap was one such thing and not the only one.
   * Shipping a stack that deadlocks would be worse than shipping a slow one.
   */
  /*
   * SYNCHRONOUS. Flip to helios_enqueue to turn batching on.
   *
   * Batching is CORRECT where it has been measured -- 32 launches deep, and a
   * 20-operation chain that enqueues 20 times and drains once -- and it does
   * not complete inside gptForward. So the remaining fault is specific to a
   * shape or a path the model uses and not to the submission machinery, which
   * is the opposite of what "it hangs at scale" suggested.
   *
   * The enqueued/flushes counters in stats() are what will localise it: a
   * healthy step shows many enqueues per flush, and the operation after the
   * last flush is the one that did not complete.
   */
  return helios_enqueue(ctx, p->code, p->count, p->gridX, p->blockX,
                        p->sharedBytes, addrs, nts, scalars, nscalars);
}

int hl_elementwise(helios_context *ctx, unsigned op, helios_tensor out,
                   helios_tensor a, helios_tensor b, unsigned n,
                   const float *scalars, unsigned nscalars) {
  const helios_key k = {HL_ELEMENTWISE, op, n, 0};
  NvU32 s[HERMES_CBUF0_SCALAR_COUNT];
  if (nscalars > HERMES_CBUF0_SCALAR_COUNT) return -1;
  for (unsigned i = 0; i < nscalars; i++) s[i] = bits(scalars[i]);

  /* A unary operation still passes three slots: the emitter reads the second
   * input only for the binary cases, and passing a live handle it will not read
   * costs nothing, while passing NONE would fail the resolve above. */
  const helios_tensor ts[3] = {out, a, b != HELIOS_TENSOR_NONE ? b : a};
  return run(ctx, k, ts, 3, s, nscalars);
}

int hl_reduce(helios_context *ctx, int mean, helios_tensor out, helios_tensor a,
              helios_tensor scratch, unsigned n) {
  /*
   * One pass if it fits in a block, two if it does not.
   *
   * The tree needs a power of two, so the second pass reduces the partials
   * ROUNDED UP, relying on the caller having zeroed the scratch beyond the
   * block count. Zero is the identity for a sum. It is not the identity for a
   * maximum, which is why this function only does sums and means and a
   * whole-tensor max would need its own padding value.
   */
  unsigned width = 1;
  while (width < n && width < PR_MAX_BLOCK) width *= 2;

  if (n <= width && width <= PR_MAX_BLOCK && n <= PR_MAX_BLOCK) {
    const helios_key k = {HL_REDUCE, mean ? PR_RED_MEAN : PR_RED_SUM, width, 0};
    const helios_tensor ts[2] = {out, a};
    const NvU32 s[1] = {bits(1.0f / (float)n)};
    return run(ctx, k, ts, 2, s, mean ? 1 : 0);
  }

  const unsigned blocks = (n + PR_MAX_BLOCK - 1) / PR_MAX_BLOCK;
  const helios_key pk = {HL_REDUCE_PARTIAL, PR_COMBINE_ADD, PR_MAX_BLOCK,
                         blocks};
  const helios_tensor pts[2] = {scratch, a};
  if (run(ctx, pk, pts, 2, NULL, 0) != 0) return -1;

  unsigned second = 1;
  while (second < blocks) second *= 2;
  const helios_key fk = {HL_REDUCE, mean ? PR_RED_MEAN : PR_RED_SUM, second, 0};
  const helios_tensor fts[2] = {out, scratch};
  /* The mean divides by the TOTAL count, not by the number of partials. */
  const NvU32 s[1] = {bits(1.0f / (float)n)};
  return run(ctx, fk, fts, 2, s, mean ? 1 : 0);
}

int hl_reduce_rows(helios_context *ctx, helios_tensor out, helios_tensor a,
                   unsigned width, unsigned rows) {
  /* The partial reduction writes one value per block, so one block per row IS a
   * row-wise reduction -- the kernel written for the first pass of a
   * whole-tensor sum, used for what it already was. */
  const helios_key k = {HL_REDUCE_PARTIAL, PR_COMBINE_ADD, width, rows};
  const helios_tensor ts[2] = {out, a};
  return run(ctx, k, ts, 2, NULL, 0);
}

int hl_normalize(helios_context *ctx, unsigned op, helios_tensor out,
                 helios_tensor a, unsigned width, unsigned rows, float eps) {
  /* The row count is part of the KEY, not just the launch: a program built for
   * one row and launched over eight would have every block writing row zero. */
  const helios_key k = {HL_NORMALIZE, op, width, rows};
  const helios_tensor ts[2] = {out, a};

  /*
   * The scalars differ BY OPERATION, and getting that wrong is subtle enough to
   * be worth spelling out.
   *
   * rmsNorm and layerNorm read 1/N and epsilon. Softmax reads neither -- it
   * needs log2(e), because the hardware provides exp2 and the kernel converts
   * the base. Passing 1/N where log2(e) belongs was the first version of this
   * function, and the result still SUMMED TO ONE: softmax divides by its own
   * total, so a wrong exponent base rescales every term and the distribution
   * renormalises itself into something plausible and entirely wrong.
   *
   * That is why the parity check compares element by element against the
   * definition and not only against the "is it a distribution" property. The
   * property held throughout.
   */
  if (op == PR_NORM_SOFTMAX) {
    const NvU32 s[1] = {bits(1.4426950408889634f)}; /* log2(e) */
    return run(ctx, k, ts, 2, s, 1);
  }
  const NvU32 s[2] = {bits(1.0f / (float)width), bits(eps)};
  return run(ctx, k, ts, 2, s, 2);
}

int hl_matmul(helios_context *ctx, helios_tensor out, helios_tensor a,
              helios_tensor b, unsigned M, unsigned N, unsigned K) {
  const helios_key k = {HL_MATMUL, M, N, K};
  const helios_tensor ts[3] = {out, a, b};
  return run(ctx, k, ts, 3, NULL, 0);
}

int hl_transpose(helios_context *ctx, helios_tensor out, helios_tensor a,
                 unsigned rows, unsigned cols) {
  const helios_key k = {HL_TRANSPOSE, rows, cols, 0};
  const helios_tensor ts[2] = {out, a};
  return run(ctx, k, ts, 2, NULL, 0);
}

int hl_embedding(helios_context *ctx, helios_tensor out, helios_tensor table,
                 helios_tensor ids, unsigned tokens, unsigned dim) {
  const helios_key k = {HL_EMBEDDING, dim, tokens, 0};
  const helios_tensor ts[3] = {out, table, ids};
  return run(ctx, k, ts, 3, NULL, 0);
}

int hl_slice(helios_context *ctx, helios_tensor out, helios_tensor a,
             unsigned count, unsigned offset, unsigned stride) {
  const helios_key k = {HL_SLICE, count, 0, 0};
  const helios_tensor ts[2] = {out, a};
  const NvU32 s[2] = {offset, stride}; /* read as integers, not floats */
  return run(ctx, k, ts, 2, s, 2);
}

int hl_causal_mask(helios_context *ctx, helios_tensor out, helios_tensor a,
                   unsigned rows, unsigned cols) {
  const helios_key k = {HL_CAUSAL_MASK, cols, rows, 0};
  const helios_tensor ts[2] = {out, a};
  return run(ctx, k, ts, 2, NULL, 0);
}

int hl_masked_fill(helios_context *ctx, helios_tensor out, helios_tensor a,
                   helios_tensor mask, unsigned n, float value) {
  const helios_key k = {HL_MASKED_FILL, 0, n, 0};
  const helios_tensor ts[3] = {out, a, mask};
  const NvU32 s[1] = {bits(value)};
  return run(ctx, k, ts, 3, s, 1);
}

int hl_cast(helios_context *ctx, int toF16, helios_tensor out, helios_tensor a,
            unsigned n) {
  const helios_key k = {toF16 ? HL_CAST_TO_F16 : HL_CAST_TO_F32, 0, n, 0};
  const helios_tensor ts[2] = {out, a};
  return run(ctx, k, ts, 2, NULL, 0);
}

int hl_dropout_mask(helios_context *ctx, helios_tensor out, unsigned n,
                    NvU32 seed, NvU32 counter, float p) {
  const helios_key k = {HL_DROPOUT, 0, n, 0};
  const helios_tensor ts[1] = {out};
  /* The threshold is p * 2^32 so the kernel compares integers -- see dropout.c
   * for why that is better than converting the hash to a float. */
  const NvU32 s[4] = {seed, counter, (NvU32)((double)p * 4294967296.0),
                      bits(p < 1.0f ? 1.0f / (1.0f - p) : 0.0f)};
  return run(ctx, k, ts, 1, s, 4);
}

int hl_cross_entropy(helios_context *ctx, helios_tensor out,
                     helios_tensor logits, helios_tensor targets, unsigned rows,
                     unsigned classes) {
  const helios_key k = {HL_CROSS_ENTROPY, classes, rows, 0};
  const helios_tensor ts[3] = {out, logits, targets};
  const NvU32 s[2] = {bits(1.4426950408889634f), bits(0.6931471805599453f)};
  return run(ctx, k, ts, 3, s, 2);
}

int hl_residual_rms(helios_context *ctx, helios_tensor out, helios_tensor x,
                    helios_tensor residual, helios_tensor weight,
                    unsigned width, unsigned rows, float eps) {
  const helios_key k = {HL_RESIDUAL_RMS, width, rows, 0};
  const helios_tensor ts[4] = {out, x, residual, weight};
  const NvU32 s[2] = {bits(1.0f / (float)width), bits(eps)};
  return run(ctx, k, ts, 4, s, 2);
}

int hl_residual_dropout(helios_context *ctx, helios_tensor out, helios_tensor x,
                        helios_tensor residual, helios_tensor mask,
                        unsigned width, unsigned rows, float scale) {
  const helios_key k = {HL_RESIDUAL_DROPOUT, width, rows, 0};
  const helios_tensor ts[4] = {out, x, residual, mask};
  const NvU32 s[1] = {bits(scale)};
  return run(ctx, k, ts, 4, s, 1);
}

int hl_adamw(helios_context *ctx, helios_tensor param, helios_tensor grad,
             helios_tensor m, helios_tensor v, unsigned n, float b1, float b2,
             float lr, float eps, float wd) {
  const helios_key k = {HL_ADAMW, 0, n, 0};
  const helios_tensor ts[4] = {param, grad, m, v};
  /* The kernel wants 1-b, not b: it evaluates m + (1-b1)*(g-m), which is the
   * same arithmetic in fewer instructions. The subtraction belongs here, where
   * it happens once per step rather than once per element. */
  const NvU32 s[5] = {bits(1.0f - b1), bits(1.0f - b2), bits(lr), bits(eps),
                      bits(wd)};
  return run(ctx, k, ts, 4, s, 5);
}
