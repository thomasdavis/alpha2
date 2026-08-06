/*
 * dispatch.h — one entry point per Backend operation.
 *
 * WHAT: the layer that turns "multiply these two tensors" into a program-cache
 * lookup and a launch. Everything above this speaks in tensors and shapes;
 * everything below speaks in addresses and grids.
 *
 * WHY IT IS A FLAT LIST OF FUNCTIONS rather than a tagged union of operations:
 * the operations genuinely differ. A matmul takes three dimensions, a dropout
 * takes a seed and a probability, a slice takes an offset and a stride. Forcing
 * them through one struct means a struct that is mostly unused fields, and a
 * caller that can populate the wrong ones without the compiler objecting. One
 * function each means the signature IS the contract.
 *
 * ORDERING IS IMPLICIT. Every call launches on the same channel and waits for
 * it to retire, so operations happen in the order they are called. That is the
 * simplest execution model there is and it leaves overlap on the table; making
 * it asynchronous is a change to make with a measurement in hand.
 *
 * WHAT IT DELIBERATELY DOES NOT DO: no shape inference, no broadcasting, no
 * dtype promotion. The caller has already decided what shape the result is --
 * that is the tensor library's job, and duplicating it here would mean two
 * places that must agree about what a shape means.
 */
#ifndef HELIOS_DISPATCH_H
#define HELIOS_DISPATCH_H

#include "../prometheus/normalize.h"
#include "../prometheus/reduction.h"
#include "program.h"
#include "tensor.h"

/* Every function returns 0 on success, -1 on any failure. */

/* out = f(a) for the unary operations, and f(a, b) for the binary ones; `op` is
 * a pr_ew_op and decides which. `n` is the element count. */
int hl_elementwise(helios_context *ctx, unsigned op, helios_tensor out,
                   helios_tensor a, helios_tensor b, unsigned n,
                   const float *scalars, unsigned nscalars);

/* out[0] = sum or mean of `n` elements of `a`, using `scratch` for the
 * per-block partials. Two passes when n exceeds a block. */
int hl_reduce(helios_context *ctx, int mean, helios_tensor out,
              helios_tensor a, helios_tensor scratch, unsigned n);

/* out[r] = sum of row r, one block per row. The mean scales afterwards. */
int hl_reduce_rows(helios_context *ctx, helios_tensor out, helios_tensor a,
                   unsigned width, unsigned rows);

/*
 * out[c] = sum over rows of a[r][c] — the bias-shaped gradient.
 *
 * Its own entry point rather than a mode of hl_reduce_rows, which collapses the
 * COLUMN axis and keeps rows. This one is the transpose of that question and a
 * completely different access pattern: coalesced along the axis it keeps.
 */
int hl_column_sum(helios_context *ctx, helios_tensor out, helios_tensor a,
                  helios_tensor b, unsigned rows, unsigned cols);

/*
 * The affine forms: normalise, then scale by a per-feature weight and (for a
 * layer norm) add a per-feature bias, all in the one launch.
 *
 * `bias` is ignored for rmsNorm and may be zero. Separate from hl_normalize
 * because that one passes exactly two tensors.
 */
int hl_normalize_affine(helios_context *ctx, unsigned op, helios_tensor out,
                        helios_tensor a, helios_tensor weight,
                        helios_tensor bias, unsigned width, unsigned rows,
                        float eps);

/*
 * Attention's score chain in one launch: scale, causal mask, softmax.
 *
 * The mask is [width, width] and TILED across the rows; the kernel wraps at
 * width*width, which needs `width` to be a power of two. Callers whose T is not
 * one keep the composed path.
 */
int hl_softmax_masked(helios_context *ctx, helios_tensor out, helios_tensor a,
                      helios_tensor mask, unsigned width, unsigned rows,
                      float scale, float cap);

/* out = normalise(a) over rows of `width`, one block per row. */
int hl_normalize(helios_context *ctx, unsigned op, helios_tensor out,
                 helios_tensor a, unsigned width, unsigned rows, float eps);

/*
 * layerNorm's backward: dx and xhat from x, the incoming gradient and the
 * weight, one block per row.
 *
 * Separate from hl_normalize because that one passes exactly two tensors and
 * this one passes five. dw and db are NOT computed here -- they reduce across
 * rows rather than within one, which is a different kernel shape; the caller
 * gets xhat back precisely so it can form them without redoing the reductions.
 */
int hl_layer_norm_backward(helios_context *ctx, helios_tensor dx,
                           helios_tensor xhat, helios_tensor x, helios_tensor g,
                           helios_tensor w, unsigned width, unsigned rows,
                           float eps);

/* out[M,N] = a[M,K] * b[K,N]. */
/* out[b,M,N] = a[b,M,K] * b[b,K,N], the plane from the block's Y index. A batch
 * of 1 is the plain single-matrix case, same code. */
int hl_matmul(helios_context *ctx, helios_tensor out, helios_tensor a,
              helios_tensor b, unsigned M, unsigned N, unsigned K,
              unsigned batch);

/* Transposes `batch` planes of rows x cols in ONE launch, the plane taken from
 * the block's Y index. */
int hl_matmul_transposed(helios_context *ctx, helios_tensor out, helios_tensor a,
                         helios_tensor b, unsigned M, unsigned N, unsigned K,
                         unsigned batch);

/* C = A @ B^T with BOTH operands stored f16 (a, b are f16 buffers the caller
 * produced with the cast kernel), staged with cp.async. C is f32. Returns -1 if
 * the shape does not divide the tile. See emit_hmma_cpasync_f16. */
int hl_matmul_cpasync(helios_context *ctx, helios_tensor out, helios_tensor a,
                      helios_tensor b, unsigned M, unsigned N, unsigned K,
                      unsigned batch);

/* out = s * (g - sum_j g_j s_j) over each row: softmax's backward, fused. */
int hl_softmax_backward(helios_context *ctx, helios_tensor out, helios_tensor s,
                        helios_tensor g, unsigned width, unsigned rows);

/* C[M,N] += A @ B. `layout` is 0 for A@B, 1 for A@B^T, 2 for A^T@B. Returns -1
 * when the shape cannot use the tensor-core path. */
int hl_matmul_accumulate(helios_context *ctx, helios_tensor out, helios_tensor a,
                         helios_tensor b, unsigned M, unsigned N, unsigned K,
                         unsigned batch, unsigned layout);

/* C[M,N] = A[K,M]^T @ B[K,N]. Returns -1 when the shape cannot use the
 * tensor-core path, which is the caller's cue to transpose and multiply. */
int hl_matmul_transposed_a(helios_context *ctx, helios_tensor out,
                           helios_tensor a, helios_tensor b, unsigned M,
                           unsigned N, unsigned K, unsigned batch);

int hl_transpose(helios_context *ctx, helios_tensor out, helios_tensor a,
                 unsigned rows, unsigned cols, unsigned batch);

/* out[b][h][t][d] = in[b][t][h][d]; `planes` is b*t*h. T, H and D must be
 * powers of two — see pr_emit_permute. */
/* out[r][c] = in[r][start + c], over `rows` rows. */
int hl_slice_rows(helios_context *ctx, helios_tensor out, helios_tensor a,
                  unsigned W, unsigned srcW, unsigned start, unsigned rows);

/* mode 0 tiles a [W] vector down `rows`; mode 1 spreads one value per row. */
int hl_broadcast(helios_context *ctx, helios_tensor out, helios_tensor a,
                 unsigned mode, unsigned W, unsigned rows);

/* out[r][start + c] = in[r][c], over `rows` rows. */
int hl_cat_rows(helios_context *ctx, helios_tensor out, helios_tensor a,
                unsigned W, unsigned dstW, unsigned start, unsigned rows);

int hl_permute(helios_context *ctx, helios_tensor out, helios_tensor a,
               unsigned T, unsigned H, unsigned D, unsigned planes);

int hl_embedding(helios_context *ctx, helios_tensor out, helios_tensor table,
                 helios_tensor ids, unsigned tokens, unsigned dim);

int hl_slice(helios_context *ctx, helios_tensor out, helios_tensor a,
             unsigned count, unsigned offset, unsigned stride);

int hl_causal_mask(helios_context *ctx, helios_tensor out, helios_tensor a,
                   unsigned rows, unsigned cols);

int hl_masked_fill(helios_context *ctx, helios_tensor out, helios_tensor a,
                   helios_tensor mask, unsigned n, float value,
                   unsigned maskWrap);

int hl_cast(helios_context *ctx, int toF16, helios_tensor out, helios_tensor a,
            unsigned n);

int hl_dropout_mask(helios_context *ctx, helios_tensor out, unsigned n,
                    NvU32 seed, NvU32 counter, float p);

int hl_cross_entropy(helios_context *ctx, helios_tensor out,
                     helios_tensor logits, helios_tensor targets, unsigned rows,
                     unsigned classes);

int hl_embedding_scatter(helios_context *ctx, helios_tensor grad_table,
                         helios_tensor grad_out, helios_tensor ids,
                         unsigned tokens, unsigned dim);

int hl_cross_entropy_backward(helios_context *ctx, helios_tensor out,
                              helios_tensor logits, helios_tensor targets,
                              unsigned rows, unsigned classes, float scale);

int hl_residual_rms(helios_context *ctx, helios_tensor out, helios_tensor x,
                    helios_tensor residual, helios_tensor weight,
                    unsigned width, unsigned rows, float eps);

int hl_residual_dropout(helios_context *ctx, helios_tensor out, helios_tensor x,
                        helios_tensor residual, helios_tensor mask,
                        unsigned width, unsigned rows, float scale);

int hl_adamw(helios_context *ctx, helios_tensor param, helios_tensor grad,
             helios_tensor m, helios_tensor v, unsigned n, float b1, float b2,
             float lr, float eps, float wd);

#endif /* HELIOS_DISPATCH_H */
