/**
 * kernels/index.ts — Kernel registry and cache.
 *
 * Imports all kernel generators from submodules and exposes getKernelSpirv().
 */

import { Op, GLSLstd450 } from "./helpers.js";

// Elementwise kernels
export {
  kernelAdd, kernelSub, kernelMul, kernelDiv, kernelNeg, kernelScale,
  kernelExp, kernelLog, kernelSqrt, kernelRelu, kernelGelu,
  kernelClampMin, kernelClamp,
  kernelGeluBackward, kernelReluBackward, kernelClampBackward,
  kernelSoftCapForward, kernelSoftCapBackward,
  kernelSoftCapForwardVec4, kernelSoftCapBackwardVec4,
  kernelAddVec4, kernelSubVec4, kernelMulVec4, kernelDivVec4,
  kernelNegVec4, kernelScaleVec4, kernelExpVec4, kernelLogVec4,
  kernelSqrtVec4, kernelReluVec4, kernelClampMinVec4, kernelClampVec4,
  kernelGeluVec4,
} from "./elementwise.js";

import {
  kernelAdd, kernelSub, kernelMul, kernelDiv, kernelNeg, kernelScale,
  kernelExp, kernelLog, kernelSqrt, kernelRelu, kernelGelu,
  kernelClampMin, kernelClamp,
  kernelGeluBackward, kernelReluBackward, kernelClampBackward,
  kernelSoftCapForward, kernelSoftCapBackward,
  kernelSoftCapForwardVec4, kernelSoftCapBackwardVec4,
  kernelAddVec4, kernelSubVec4, kernelMulVec4, kernelDivVec4,
  kernelNegVec4, kernelScaleVec4, kernelExpVec4, kernelLogVec4,
  kernelSqrtVec4, kernelReluVec4, kernelClampMinVec4, kernelClampVec4,
  kernelGeluVec4,
} from "./elementwise.js";

// Reduction kernels
export {
  kernelSumReduce, kernelSumOfSquares, kernelMaxReduce, kernelColumnSum, kernelSumAxis,
} from "./reduction.js";

import {
  kernelSumReduce, kernelSumOfSquares, kernelMaxReduce, kernelColumnSum, kernelSumAxis,
} from "./reduction.js";

// NN kernels (includes silu, silu_vec4, mulAdd, residualDropoutAdd, dropoutMask)
export {
  kernelSoftmax, kernelLayerNorm, kernelLayerNormBackward,
  kernelBroadcast, kernelMaskedFill,
  kernelCrossEntropyForwardFused, kernelCrossEntropyForwardPick,
  kernelCrossEntropyBackward,
  kernelEmbeddingForward, kernelEmbeddingBackward,
  kernelSilu, kernelSiluVec4,
  kernelMulAdd, kernelResidualDropoutAdd, kernelResidualDropoutAddVec4,
  kernelDropoutMask,
} from "./nn.js";

import {
  kernelSoftmax, kernelLayerNorm, kernelLayerNormBackward,
  kernelBroadcast, kernelMaskedFill,
  kernelCrossEntropyForwardFused, kernelCrossEntropyForwardPick,
  kernelCrossEntropyBackward,
  kernelEmbeddingForward, kernelEmbeddingBackward,
  kernelSilu, kernelSiluVec4,
  kernelMulAdd, kernelResidualDropoutAdd, kernelResidualDropoutAddVec4,
  kernelDropoutMask,
} from "./nn.js";

// Matmul kernels
export {
  kernelMatmul, kernelMatmulBatched,
  kernelMatmulTransposed, kernelMatmulTransposedBatched,
} from "./matmul.js";

import {
  kernelMatmul, kernelMatmulBatched,
  kernelMatmulTransposed, kernelMatmulTransposedBatched,
} from "./matmul.js";

// Optimizer / utility kernels
export {
  kernelAdamW, kernelTranspose,
  kernelAddInplace, kernelAddInplaceVec4,
} from "./optimizer.js";

import {
  kernelAdamW, kernelTranspose,
  kernelAddInplace, kernelAddInplaceVec4,
} from "./optimizer.js";

// Attention kernels (Flash Attention forward + backward)
export {
  kernelFlashAttentionForward,
  kernelFlashAttentionBackwardDQ,
  kernelFlashAttentionBackwardDKV,
} from "./attention.js";

import {
  kernelFlashAttentionForward,
  kernelFlashAttentionBackwardDQ,
  kernelFlashAttentionBackwardDKV,
} from "./attention.js";

// Copy / slice kernels
export { kernelSlice2D, kernelScatterSlice2D } from "./copy.js";

import { kernelSlice2D, kernelScatterSlice2D } from "./copy.js";

// F16 storage variant kernels + cast kernels
export { kernelBinaryOpF16, kernelUnaryOpF16, kernelCastF32ToF16, kernelCastF16ToF32 } from "./f16.js";

import { kernelBinaryOpF16, kernelUnaryOpF16, kernelCastF32ToF16, kernelCastF16ToF32 } from "./f16.js";

// ── Kernel cache ────────────────────────────────────────────────────────────

const spirvCache = new Map<string, Uint32Array>();

/** Get a cached SPIR-V binary, generating it on first use. */
export function getKernelSpirv(name: string, wgSize = 256): Uint32Array {
  const key = `${name}:${wgSize}`;
  let spirv = spirvCache.get(key);
  if (spirv) return spirv;

  switch (name) {
    case "add":   spirv = kernelAdd(wgSize); break;
    case "sub":   spirv = kernelSub(wgSize); break;
    case "mul":   spirv = kernelMul(wgSize); break;
    case "div":   spirv = kernelDiv(wgSize); break;
    case "neg":   spirv = kernelNeg(wgSize); break;
    case "scale": spirv = kernelScale(wgSize); break;
    case "exp":   spirv = kernelExp(wgSize); break;
    case "log":   spirv = kernelLog(wgSize); break;
    case "sqrt":      spirv = kernelSqrt(wgSize); break;
    case "relu":      spirv = kernelRelu(wgSize); break;
    case "clamp_min": spirv = kernelClampMin(wgSize); break;
    case "clamp":     spirv = kernelClamp(wgSize); break;
    case "gelu":      spirv = kernelGelu(wgSize); break;
    case "add_vec4":  spirv = kernelAddVec4(wgSize); break;
    case "sub_vec4":  spirv = kernelSubVec4(wgSize); break;
    case "mul_vec4":  spirv = kernelMulVec4(wgSize); break;
    case "div_vec4":  spirv = kernelDivVec4(wgSize); break;
    case "neg_vec4":  spirv = kernelNegVec4(wgSize); break;
    case "scale_vec4": spirv = kernelScaleVec4(wgSize); break;
    case "exp_vec4":  spirv = kernelExpVec4(wgSize); break;
    case "log_vec4":  spirv = kernelLogVec4(wgSize); break;
    case "sqrt_vec4": spirv = kernelSqrtVec4(wgSize); break;
    case "relu_vec4": spirv = kernelReluVec4(wgSize); break;
    case "clamp_min_vec4": spirv = kernelClampMinVec4(wgSize); break;
    case "clamp_vec4":     spirv = kernelClampVec4(wgSize); break;
    case "gelu_vec4": spirv = kernelGeluVec4(wgSize); break;
    case "sum_reduce": spirv = kernelSumReduce(wgSize); break;
    case "sum_sq_reduce": spirv = kernelSumOfSquares(wgSize); break;
    case "max_reduce": spirv = kernelMaxReduce(wgSize); break;
    case "softmax":   spirv = kernelSoftmax(wgSize); break;
    case "layernorm": spirv = kernelLayerNorm(wgSize); break;
    case "silu":      spirv = kernelSilu(wgSize); break;
    case "silu_vec4": spirv = kernelSiluVec4(wgSize); break;
    case "mul_add":   spirv = kernelMulAdd(wgSize); break;
    case "residual_dropout_add": spirv = kernelResidualDropoutAdd(wgSize); break;
    case "residual_dropout_add_vec4": spirv = kernelResidualDropoutAddVec4(wgSize); break;
    case "matmul":    spirv = kernelMatmul(); break;
    case "matmul_batched": spirv = kernelMatmulBatched(); break;
    case "matmul_transposed": spirv = kernelMatmulTransposed(); break;
    case "matmul_transposed_batched": spirv = kernelMatmulTransposedBatched(); break;
    // Tile-size variants (tile=32 for large matrices)
    case "matmul_T32": spirv = kernelMatmul(32 * 32, 32); break;
    case "matmul_batched_T32": spirv = kernelMatmulBatched(32 * 32, 32); break;
    case "matmul_transposed_T32": spirv = kernelMatmulTransposed(32 * 32, 32); break;
    case "matmul_transposed_batched_T32": spirv = kernelMatmulTransposedBatched(32 * 32, 32); break;
    case "add_inplace": spirv = kernelAddInplace(wgSize); break;
    case "add_inplace_vec4": spirv = kernelAddInplaceVec4(wgSize); break;
    case "gelu_backward": spirv = kernelGeluBackward(wgSize); break;
    case "relu_backward": spirv = kernelReluBackward(wgSize); break;
    case "clamp_backward": spirv = kernelClampBackward(wgSize); break;
    case "layernorm_backward": spirv = kernelLayerNormBackward(wgSize); break;
    case "column_sum": spirv = kernelColumnSum(wgSize); break;
    case "adamw_step": spirv = kernelAdamW(wgSize); break;
    case "transpose":  spirv = kernelTranspose(wgSize); break;
    case "sum_axis":   spirv = kernelSumAxis(wgSize); break;
    case "broadcast":  spirv = kernelBroadcast(wgSize); break;
    case "masked_fill": spirv = kernelMaskedFill(wgSize); break;
    case "ce_fwd_fused": spirv = kernelCrossEntropyForwardFused(wgSize); break;
    case "ce_fwd_pick": spirv = kernelCrossEntropyForwardPick(wgSize); break;
    case "cross_entropy_backward": spirv = kernelCrossEntropyBackward(wgSize); break;
    case "embedding_backward": spirv = kernelEmbeddingBackward(wgSize); break;
    case "embedding_forward": spirv = kernelEmbeddingForward(wgSize); break;
    case "slice_2d": spirv = kernelSlice2D(wgSize); break;
    case "scatter_slice_2d": spirv = kernelScatterSlice2D(wgSize); break;
    case "dropout_mask": spirv = kernelDropoutMask(wgSize); break;
    case "softcap_forward": spirv = kernelSoftCapForward(wgSize); break;
    case "softcap_backward": spirv = kernelSoftCapBackward(wgSize); break;
    case "softcap_forward_vec4": spirv = kernelSoftCapForwardVec4(wgSize); break;
    case "softcap_backward_vec4": spirv = kernelSoftCapBackwardVec4(wgSize); break;
    // f16 storage variants (compute in f32, load/store f16)
    case "add_f16":   spirv = kernelBinaryOpF16(Op.FAdd, wgSize); break;
    case "sub_f16":   spirv = kernelBinaryOpF16(Op.FSub, wgSize); break;
    case "mul_f16":   spirv = kernelBinaryOpF16(Op.FMul, wgSize); break;
    case "div_f16":   spirv = kernelBinaryOpF16(Op.FDiv, wgSize); break;
    case "neg_f16":   spirv = kernelUnaryOpF16(null, wgSize); break;
    case "exp_f16":   spirv = kernelUnaryOpF16(GLSLstd450.Exp, wgSize); break;
    case "log_f16":   spirv = kernelUnaryOpF16(GLSLstd450.Log, wgSize); break;
    case "sqrt_f16":  spirv = kernelUnaryOpF16(GLSLstd450.Sqrt, wgSize); break;
    // f16 ↔ f32 cast kernels
    case "cast_f32_to_f16": spirv = kernelCastF32ToF16(wgSize); break;
    case "cast_f16_to_f32": spirv = kernelCastF16ToF32(wgSize); break;
    default: {
      // Flash attention kernels — name encodes params: flash_attn_{variant}_{Br}_{Bc}_{D}
      const flashMatch = name.match(/^flash_attn_(fwd|bwd_dq|bwd_dkv)_(\d+)_(\d+)_(\d+)$/);
      if (flashMatch) {
        const [, variant, brS, bcS, dS] = flashMatch;
        const Br = parseInt(brS), Bc = parseInt(bcS), D = parseInt(dS);
        switch (variant) {
          case "fwd":     spirv = kernelFlashAttentionForward(Br, Bc, D); break;
          case "bwd_dq":  spirv = kernelFlashAttentionBackwardDQ(Br, Bc, D); break;
          case "bwd_dkv": spirv = kernelFlashAttentionBackwardDKV(Br, Bc, D); break;
        }
      }
      if (!spirv) throw new Error(`Helios: unknown kernel "${name}"`);
    }
  }

  spirvCache.set(key, spirv);
  return spirv;
}
