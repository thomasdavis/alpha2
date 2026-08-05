/*
 * napi_ops.c — one JavaScript binding per operation.
 *
 * WHAT: unwrap the arguments, call the dispatcher, return a boolean.
 *
 * WHY EACH ONE IS WRITTEN OUT rather than generated from a table: the argument
 * lists genuinely differ, and a table would have to describe them in some
 * encoding that is itself a small language nobody else reads. Written out, each
 * binding's argument order is visible next to the dispatcher signature it
 * feeds, which is where a mismatch would otherwise hide -- swapping two tensor
 * handles of the same type compiles, runs, and produces a wrong answer.
 *
 * EVERY BINDING RETURNS A BOOLEAN, never throws. A failed launch here means a
 * shape the emitters cannot serve or a stale handle, and the TypeScript side
 * turns that into an error with the context to explain it. Throwing from inside
 * a hot loop would also mean unwinding through the dispatcher, which owns a
 * channel mid-submission.
 */
#include <node_api.h>

#include "dispatch.h"
#include "../prometheus/elementwise.h"
#include "../prometheus/normalize.h"
#include "../prometheus/colsum.h"
#include "../prometheus/shapes.h"

/* Declared in napi.c, which owns the context and the argument helpers. */
NvU32 hl_arg_u32(napi_env env, napi_callback_info info, size_t i);
double hl_arg_double(napi_env env, napi_callback_info info, size_t i);
helios_context *hl_context(void);
napi_value hl_result(napi_env env, int rc);
void hl_export(napi_env env, napi_value exports, const char *name,
               napi_callback fn);

#define CTX                                                                    \
  helios_context *ctx = hl_context();                                          \
  if (!ctx) return hl_result(env, -1)

#define U32(i) hl_arg_u32(env, info, (i))
#define F32(i) ((float)hl_arg_double(env, info, (i)))

/* (op, out, a, b, n, s0, s1, s2, s3, s4, s5, nscalars) */
static napi_value js_elementwise(napi_env env, napi_callback_info info) {
  CTX;
  const float s[6] = {F32(5), F32(6), F32(7), F32(8), F32(9), F32(10)};
  return hl_result(env, hl_elementwise(ctx, U32(0), U32(1), U32(2), U32(3),
                                       U32(4), s, U32(11)));
}

/* (mean, out, a, scratch, n) */
static napi_value js_reduce(napi_env env, napi_callback_info info) {
  CTX;
  return hl_result(
      env, hl_reduce(ctx, (int)U32(0), U32(1), U32(2), U32(3), U32(4)));
}

/* (out, a, width, rows) */
static napi_value js_reduce_rows(napi_env env, napi_callback_info info) {
  CTX;
  return hl_result(env, hl_reduce_rows(ctx, U32(0), U32(1), U32(2), U32(3)));
}

/* (out, a, b, rows, cols) — b = 0 for the plain sum. */
static napi_value js_column_sum(napi_env env, napi_callback_info info) {
  CTX;
  return hl_result(env, hl_column_sum(ctx, U32(0), U32(1), U32(2), U32(3), U32(4)));
}

/* (out, a, mask, width, rows, scale, cap) — cap 0 means no soft cap. */
static napi_value js_softmax_masked(napi_env env, napi_callback_info info) {
  CTX;
  return hl_result(env, hl_softmax_masked(ctx, U32(0), U32(1), U32(2), U32(3),
                                          U32(4), F32(5), F32(6)));
}

/* (op, out, a, weight, bias, width, rows, eps) */
static napi_value js_normalize_affine(napi_env env, napi_callback_info info) {
  CTX;
  return hl_result(env, hl_normalize_affine(ctx, U32(0), U32(1), U32(2), U32(3),
                                            U32(4), U32(5), U32(6), F32(7)));
}

/* (op, out, a, width, rows, eps) */
static napi_value js_normalize(napi_env env, napi_callback_info info) {
  CTX;
  return hl_result(env, hl_normalize(ctx, U32(0), U32(1), U32(2), U32(3),
                                     U32(4), F32(5)));
}

/* (dx, xhat, x, g, w, width, rows, eps) */
static napi_value js_layer_norm_backward(napi_env env,
                                         napi_callback_info info) {
  CTX;
  return hl_result(env, hl_layer_norm_backward(ctx, U32(0), U32(1), U32(2),
                                               U32(3), U32(4), U32(5), U32(6),
                                               F32(7)));
}

/* (out, a, b, M, N, K, batch) */
static napi_value js_matmul(napi_env env, napi_callback_info info) {
  CTX;
  return hl_result(
      env, hl_matmul(ctx, U32(0), U32(1), U32(2), U32(3), U32(4), U32(5),
                     U32(6)));
}

static napi_value js_softmax_backward(napi_env env, napi_callback_info info) {
  CTX;
  return hl_result(env, hl_softmax_backward(ctx, U32(0), U32(1), U32(2), U32(3),
                                            U32(4)));
}

static napi_value js_matmul_accumulate(napi_env env, napi_callback_info info) {
  CTX;
  return hl_result(
      env, hl_matmul_accumulate(ctx, U32(0), U32(1), U32(2), U32(3), U32(4),
                                U32(5), U32(6), U32(7)));
}

static napi_value js_matmul_transposed_a(napi_env env, napi_callback_info info) {
  CTX;
  return hl_result(
      env, hl_matmul_transposed_a(ctx, U32(0), U32(1), U32(2), U32(3), U32(4),
                                  U32(5), U32(6)));
}

static napi_value js_matmul_transposed(napi_env env, napi_callback_info info) {
  CTX;
  return hl_result(
      env, hl_matmul_transposed(ctx, U32(0), U32(1), U32(2), U32(3), U32(4), U32(5),
                     U32(6)));
}

/* (out, a, rows, cols, batch) */
static napi_value js_transpose(napi_env env, napi_callback_info info) {
  CTX;
  return hl_result(
      env, hl_transpose(ctx, U32(0), U32(1), U32(2), U32(3), U32(4)));
}

/* (out, a, W, srcW, start, rows) */
static napi_value js_slice_rows(napi_env env, napi_callback_info info) {
  CTX;
  return hl_result(
      env, hl_slice_rows(ctx, U32(0), U32(1), U32(2), U32(3), U32(4), U32(5)));
}

/* (out, a, mode, W, rows) */
static napi_value js_broadcast(napi_env env, napi_callback_info info) {
  CTX;
  return hl_result(
      env, hl_broadcast(ctx, U32(0), U32(1), U32(2), U32(3), U32(4)));
}

/* (out, a, W, dstW, start, rows) */
static napi_value js_cat_rows(napi_env env, napi_callback_info info) {
  CTX;
  return hl_result(
      env, hl_cat_rows(ctx, U32(0), U32(1), U32(2), U32(3), U32(4), U32(5)));
}

/* (out, a, T, H, D, planes) — swaps the middle two axes of [B,T,H,D] */
static napi_value js_permute(napi_env env, napi_callback_info info) {
  CTX;
  return hl_result(
      env, hl_permute(ctx, U32(0), U32(1), U32(2), U32(3), U32(4), U32(5)));
}

/* (out, table, ids, tokens, dim) */
static napi_value js_embedding(napi_env env, napi_callback_info info) {
  CTX;
  return hl_result(
      env, hl_embedding(ctx, U32(0), U32(1), U32(2), U32(3), U32(4)));
}

/* (out, a, count, offset, stride) */
static napi_value js_slice(napi_env env, napi_callback_info info) {
  CTX;
  return hl_result(env,
                   hl_slice(ctx, U32(0), U32(1), U32(2), U32(3), U32(4)));
}

/* (out, a, rows, cols) */
static napi_value js_causal_mask(napi_env env, napi_callback_info info) {
  CTX;
  return hl_result(env, hl_causal_mask(ctx, U32(0), U32(1), U32(2), U32(3)));
}

/* (out, a, mask, n, value) */
static napi_value js_masked_fill(napi_env env, napi_callback_info info) {
  CTX;
  return hl_result(
      env, hl_masked_fill(ctx, U32(0), U32(1), U32(2), U32(3), F32(4), U32(5)));
}

/* (toF16, out, a, n) */
static napi_value js_cast(napi_env env, napi_callback_info info) {
  CTX;
  return hl_result(env, hl_cast(ctx, (int)U32(0), U32(1), U32(2), U32(3)));
}

/* (out, n, seed, counter, p) */
static napi_value js_dropout_mask(napi_env env, napi_callback_info info) {
  CTX;
  return hl_result(
      env, hl_dropout_mask(ctx, U32(0), U32(1), U32(2), U32(3), F32(4)));
}

/* (out, logits, targets, rows, classes) */
static napi_value js_cross_entropy(napi_env env, napi_callback_info info) {
  CTX;
  return hl_result(
      env, hl_cross_entropy(ctx, U32(0), U32(1), U32(2), U32(3), U32(4)));
}

/* (gradTable, gradOut, ids, tokens, dim) */
static napi_value js_embedding_scatter(napi_env env, napi_callback_info info) {
  CTX;
  return hl_result(env, hl_embedding_scatter(ctx, U32(0), U32(1), U32(2),
                                             U32(3), U32(4)));
}

/* (out, logits, targets, rows, classes, scale) */
static napi_value js_cross_entropy_backward(napi_env env,
                                            napi_callback_info info) {
  CTX;
  return hl_result(env, hl_cross_entropy_backward(ctx, U32(0), U32(1), U32(2),
                                                  U32(3), U32(4), F32(5)));
}

/* (out, x, residual, weight, width, rows, eps) */
static napi_value js_residual_rms(napi_env env, napi_callback_info info) {
  CTX;
  return hl_result(env, hl_residual_rms(ctx, U32(0), U32(1), U32(2), U32(3),
                                        U32(4), U32(5), F32(6)));
}

/* (out, x, residual, mask, width, rows, scale) */
static napi_value js_residual_dropout(napi_env env, napi_callback_info info) {
  CTX;
  return hl_result(env, hl_residual_dropout(ctx, U32(0), U32(1), U32(2), U32(3),
                                            U32(4), U32(5), F32(6)));
}

/* (param, grad, m, v, n, b1, b2, lr, eps, wd) */
static napi_value js_adamw(napi_env env, napi_callback_info info) {
  CTX;
  return hl_result(env, hl_adamw(ctx, U32(0), U32(1), U32(2), U32(3), U32(4),
                                 F32(5), F32(6), F32(7), F32(8), F32(9)));
}

napi_value hl_napi_register_ops(napi_env env, napi_value exports) {
  hl_export(env, exports, "elementwise", js_elementwise);
  hl_export(env, exports, "reduce", js_reduce);
  hl_export(env, exports, "reduceRows", js_reduce_rows);
  hl_export(env, exports, "columnSum", js_column_sum);
  hl_export(env, exports, "softmaxMasked", js_softmax_masked);
  hl_export(env, exports, "normalizeAffine", js_normalize_affine);
  hl_export(env, exports, "normalize", js_normalize);
  hl_export(env, exports, "layerNormBackward", js_layer_norm_backward);
  hl_export(env, exports, "matmul", js_matmul);
  hl_export(env, exports, "matmulTransposed", js_matmul_transposed);
  hl_export(env, exports, "matmulTransposedA", js_matmul_transposed_a);
  hl_export(env, exports, "matmulAccumulate", js_matmul_accumulate);
  hl_export(env, exports, "softmaxBackward", js_softmax_backward);
  hl_export(env, exports, "transpose", js_transpose);
  hl_export(env, exports, "permute", js_permute);
  hl_export(env, exports, "sliceRows", js_slice_rows);
  hl_export(env, exports, "catRows", js_cat_rows);
  hl_export(env, exports, "broadcastRows", js_broadcast);
  hl_export(env, exports, "embedding", js_embedding);
  hl_export(env, exports, "slice", js_slice);
  hl_export(env, exports, "causalMask", js_causal_mask);
  hl_export(env, exports, "maskedFill", js_masked_fill);
  hl_export(env, exports, "cast", js_cast);
  hl_export(env, exports, "dropoutMask", js_dropout_mask);
  hl_export(env, exports, "crossEntropy", js_cross_entropy);
  hl_export(env, exports, "crossEntropyBackward", js_cross_entropy_backward);
  hl_export(env, exports, "embeddingScatter", js_embedding_scatter);
  hl_export(env, exports, "residualRms", js_residual_rms);
  hl_export(env, exports, "residualDropout", js_residual_dropout);
  hl_export(env, exports, "adamw", js_adamw);

  /* The operation enums, so the TypeScript side names them rather than
   * hardcoding integers that would drift the moment one is inserted. */
  napi_value ops;
  napi_create_object(env, &ops);
  napi_value v;
#define OP(name, value)                                                        \
  napi_create_uint32(env, (value), &v);                                        \
  napi_set_named_property(env, ops, name, v)
  OP("add", PR_EW_ADD);
  OP("sub", PR_EW_SUB);
  OP("mul", PR_EW_MUL);
  OP("div", PR_EW_DIV);
  OP("neg", PR_EW_FNEG);
  OP("relu", PR_EW_RELU);
  OP("gelu", PR_EW_GELU);
  OP("geluGrad", PR_EW_GELU_GRAD);
  OP("clampGrad", PR_EW_CLAMP_GRAD);
  OP("silu", PR_EW_SILU);
  OP("exp", PR_EW_EXP);
  OP("log", PR_EW_LOG);
  OP("sqrt", PR_EW_SQRT);
  OP("scale", PR_EW_SCALE);
  OP("clamp", PR_EW_CLAMP);
  OP("fill", PR_EW_FILL);
  OP("copy", PR_EW_COPY);
  OP("addInPlace", PR_EW_ADD_INPLACE);
  OP("softCap", PR_EW_SOFTCAP);
  OP("softCapGrad", PR_EW_SOFTCAP_GRAD);
  OP("rmsNorm", PR_NORM_RMS);
  OP("softmax", PR_NORM_SOFTMAX);
  OP("layerNorm", PR_NORM_LAYER);
  OP("rmsNormAffine", PR_NORM_RMS_AFFINE);
  OP("layerNormAffine", PR_NORM_LAYER_AFFINE);
#undef OP
  napi_set_named_property(env, exports, "op", ops);

  /*
   * The FOLDED constants each kernel expects, exported by name.
   *
   * These were written out a second time in TypeScript and drifted: gelu's two
   * scalars ended up swapped and silu's was negated, so both computed something
   * plausible and wrong. The kernels do not evaluate the textbook formula --
   * they evaluate an algebraically equal one with fewer instructions, and which
   * constant goes in which slot is a property of THAT rearrangement. It belongs
   * next to the rearrangement, exported, not restated by every caller.
   */
  napi_value sc;
  napi_create_object(env, &sc);
  napi_value d;
#define SCALAR(name, value)                                                    \
  napi_create_double(env, (value), &d);                                        \
  napi_set_named_property(env, sc, name, d)
  SCALAR("log2e", PR_LOG2_E);
  SCALAR("ln2", PR_LN_2);
  SCALAR("geluK1", PR_GELU_K1);
  SCALAR("geluFolded", 2.0 * PR_GELU_K0 * PR_LOG2_E);
  SCALAR("gelu3K1", 3.0 * PR_GELU_K1);
  SCALAR("gelu2K0", 2.0 * PR_GELU_K0);
  SCALAR("softCapFolded", 2.0 * PR_LOG2_E / PR_SOFTCAP_C);
  SCALAR("softCapC", PR_SOFTCAP_C);
#undef SCALAR
  napi_set_named_property(env, exports, "scalar", sc);
  return exports;
}
