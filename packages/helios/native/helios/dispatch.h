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

/* out = normalise(a) over rows of `width`, one block per row. */
int hl_normalize(helios_context *ctx, unsigned op, helios_tensor out,
                 helios_tensor a, unsigned width, unsigned rows, float eps);

/* out[M,N] = a[M,K] * b[K,N]. */
int hl_matmul(helios_context *ctx, helios_tensor out, helios_tensor a,
              helios_tensor b, unsigned M, unsigned N, unsigned K);

int hl_transpose(helios_context *ctx, helios_tensor out, helios_tensor a,
                 unsigned rows, unsigned cols);

int hl_embedding(helios_context *ctx, helios_tensor out, helios_tensor table,
                 helios_tensor ids, unsigned tokens, unsigned dim);

int hl_slice(helios_context *ctx, helios_tensor out, helios_tensor a,
             unsigned count, unsigned offset, unsigned stride);

int hl_causal_mask(helios_context *ctx, helios_tensor out, helios_tensor a,
                   unsigned rows, unsigned cols);

int hl_masked_fill(helios_context *ctx, helios_tensor out, helios_tensor a,
                   helios_tensor mask, unsigned n, float value);

int hl_cast(helios_context *ctx, int toF16, helios_tensor out, helios_tensor a,
            unsigned n);

int hl_dropout_mask(helios_context *ctx, helios_tensor out, unsigned n,
                    NvU32 seed, NvU32 counter, float p);

int hl_cross_entropy(helios_context *ctx, helios_tensor out,
                     helios_tensor logits, helios_tensor targets, unsigned rows,
                     unsigned classes);

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
