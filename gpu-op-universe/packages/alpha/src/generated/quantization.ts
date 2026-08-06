/* AUTO-GENERATED. Do not hand-edit; edit operation-registry.json. */
import { defineStub } from "../../../common/src/types";
import type { QuantizationRequest } from "../../../common/src/types";

/**
 * alpha.quantization.asymmetric-quantize
 * Asymmetric quantize operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationAsymmetricQuantize = defineStub<QuantizationRequest>("alpha.quantization.asymmetric-quantize");

/**
 * alpha.quantization.binary-code-quantize
 * Binary code quantize operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationBinaryCodeQuantize = defineStub<QuantizationRequest>("alpha.quantization.binary-code-quantize");

/**
 * alpha.quantization.block-scale-quantize
 * Block scale quantize operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationBlockScaleQuantize = defineStub<QuantizationRequest>("alpha.quantization.block-scale-quantize");

/**
 * alpha.quantization.codebook-quantize
 * Codebook quantize operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationCodebookQuantize = defineStub<QuantizationRequest>("alpha.quantization.codebook-quantize");

/**
 * alpha.quantization.compute-amax
 * Compute amax operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationComputeAmax = defineStub<QuantizationRequest>("alpha.quantization.compute-amax");

/**
 * alpha.quantization.compute-scale
 * Compute scale operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationComputeScale = defineStub<QuantizationRequest>("alpha.quantization.compute-scale");

/**
 * alpha.quantization.compute-zero-point
 * Compute zero point operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationComputeZeroPoint = defineStub<QuantizationRequest>("alpha.quantization.compute-zero-point");

/**
 * alpha.quantization.dequantize
 * Dequantize operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationDequantize = defineStub<QuantizationRequest>("alpha.quantization.dequantize");

/**
 * alpha.quantization.dynamic-quantize
 * Dynamic quantize operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationDynamicQuantize = defineStub<QuantizationRequest>("alpha.quantization.dynamic-quantize");

/**
 * alpha.quantization.error-feedback-quantize
 * Error feedback quantize operation in the quantization family.
 * Status: research; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationErrorFeedbackQuantize = defineStub<QuantizationRequest>("alpha.quantization.error-feedback-quantize");

/**
 * alpha.quantization.fake-quantize
 * Fake quantize operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationFakeQuantize = defineStub<QuantizationRequest>("alpha.quantization.fake-quantize");

/**
 * alpha.quantization.hadamard-quantize
 * Hadamard quantize operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationHadamardQuantize = defineStub<QuantizationRequest>("alpha.quantization.hadamard-quantize");

/**
 * alpha.quantization.microscale-quantize
 * Microscale quantize operation in the quantization family.
 * Status: standard; target: architecture-agnostic, future-or-emulated; differentiability: straight-through-or-none.
 */
export const quantizationMicroscaleQuantize = defineStub<QuantizationRequest>("alpha.quantization.microscale-quantize");

/**
 * alpha.quantization.outlier-restore
 * Outlier restore operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationOutlierRestore = defineStub<QuantizationRequest>("alpha.quantization.outlier-restore");

/**
 * alpha.quantization.outlier-split
 * Outlier split operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationOutlierSplit = defineStub<QuantizationRequest>("alpha.quantization.outlier-split");

/**
 * alpha.quantization.pack-binary
 * Pack binary operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationPackBinary = defineStub<QuantizationRequest>("alpha.quantization.pack-binary");

/**
 * alpha.quantization.pack-fp8
 * Pack fp8 operation in the quantization family.
 * Status: standard; target: architecture-agnostic, future-or-emulated; differentiability: straight-through-or-none.
 */
export const quantizationPackFp8 = defineStub<QuantizationRequest>("alpha.quantization.pack-fp8");

/**
 * alpha.quantization.pack-int2
 * Pack int2 operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationPackInt2 = defineStub<QuantizationRequest>("alpha.quantization.pack-int2");

/**
 * alpha.quantization.pack-int4
 * Pack int4 operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationPackInt4 = defineStub<QuantizationRequest>("alpha.quantization.pack-int4");

/**
 * alpha.quantization.pack-int8
 * Pack int8 operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationPackInt8 = defineStub<QuantizationRequest>("alpha.quantization.pack-int8");

/**
 * alpha.quantization.pack-nf4
 * Pack nf4 operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationPackNf4 = defineStub<QuantizationRequest>("alpha.quantization.pack-nf4");

/**
 * alpha.quantization.pack-ternary
 * Pack ternary operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationPackTernary = defineStub<QuantizationRequest>("alpha.quantization.pack-ternary");

/**
 * alpha.quantization.per-channel-quantize
 * Per channel quantize operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationPerChannelQuantize = defineStub<QuantizationRequest>("alpha.quantization.per-channel-quantize");

/**
 * alpha.quantization.per-column-quantize
 * Per column quantize operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationPerColumnQuantize = defineStub<QuantizationRequest>("alpha.quantization.per-column-quantize");

/**
 * alpha.quantization.per-group-quantize
 * Per group quantize operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationPerGroupQuantize = defineStub<QuantizationRequest>("alpha.quantization.per-group-quantize");

/**
 * alpha.quantization.per-row-quantize
 * Per row quantize operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationPerRowQuantize = defineStub<QuantizationRequest>("alpha.quantization.per-row-quantize");

/**
 * alpha.quantization.per-tensor-quantize
 * Per tensor quantize operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationPerTensorQuantize = defineStub<QuantizationRequest>("alpha.quantization.per-tensor-quantize");

/**
 * alpha.quantization.per-token-quantize
 * Per token quantize operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationPerTokenQuantize = defineStub<QuantizationRequest>("alpha.quantization.per-token-quantize");

/**
 * alpha.quantization.product-quantize
 * Product quantize operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationProductQuantize = defineStub<QuantizationRequest>("alpha.quantization.product-quantize");

/**
 * alpha.quantization.quantization-calibration
 * Quantization calibration operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationQuantizationCalibration = defineStub<QuantizationRequest>("alpha.quantization.quantization-calibration");

/**
 * alpha.quantization.quantization-error-metric
 * Quantization error metric operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationQuantizationErrorMetric = defineStub<QuantizationRequest>("alpha.quantization.quantization-error-metric");

/**
 * alpha.quantization.quantize
 * Quantize operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationQuantize = defineStub<QuantizationRequest>("alpha.quantization.quantize");

/**
 * alpha.quantization.requantize
 * Requantize operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationRequantize = defineStub<QuantizationRequest>("alpha.quantization.requantize");

/**
 * alpha.quantization.residual-quantize
 * Residual quantize operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationResidualQuantize = defineStub<QuantizationRequest>("alpha.quantization.residual-quantize");

/**
 * alpha.quantization.rotation-quantize
 * Rotation quantize operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationRotationQuantize = defineStub<QuantizationRequest>("alpha.quantization.rotation-quantize");

/**
 * alpha.quantization.round-to-nearest-even
 * Round to nearest even operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationRoundToNearestEven = defineStub<QuantizationRequest>("alpha.quantization.round-to-nearest-even");

/**
 * alpha.quantization.round-toward-negative
 * Round toward negative operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationRoundTowardNegative = defineStub<QuantizationRequest>("alpha.quantization.round-toward-negative");

/**
 * alpha.quantization.round-toward-positive
 * Round toward positive operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationRoundTowardPositive = defineStub<QuantizationRequest>("alpha.quantization.round-toward-positive");

/**
 * alpha.quantization.round-toward-zero
 * Round toward zero operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationRoundTowardZero = defineStub<QuantizationRequest>("alpha.quantization.round-toward-zero");

/**
 * alpha.quantization.sigma-delta-quantize
 * Sigma delta quantize operation in the quantization family.
 * Status: research; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationSigmaDeltaQuantize = defineStub<QuantizationRequest>("alpha.quantization.sigma-delta-quantize");

/**
 * alpha.quantization.smooth-quant
 * Smooth quant operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationSmoothQuant = defineStub<QuantizationRequest>("alpha.quantization.smooth-quant");

/**
 * alpha.quantization.static-quantize
 * Static quantize operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationStaticQuantize = defineStub<QuantizationRequest>("alpha.quantization.static-quantize");

/**
 * alpha.quantization.stochastic-round
 * Stochastic round operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationStochasticRound = defineStub<QuantizationRequest>("alpha.quantization.stochastic-round");

/**
 * alpha.quantization.symmetric-quantize
 * Symmetric quantize operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationSymmetricQuantize = defineStub<QuantizationRequest>("alpha.quantization.symmetric-quantize");

/**
 * alpha.quantization.unpack-binary
 * Unpack binary operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationUnpackBinary = defineStub<QuantizationRequest>("alpha.quantization.unpack-binary");

/**
 * alpha.quantization.unpack-fp8
 * Unpack fp8 operation in the quantization family.
 * Status: standard; target: architecture-agnostic, future-or-emulated; differentiability: straight-through-or-none.
 */
export const quantizationUnpackFp8 = defineStub<QuantizationRequest>("alpha.quantization.unpack-fp8");

/**
 * alpha.quantization.unpack-int2
 * Unpack int2 operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationUnpackInt2 = defineStub<QuantizationRequest>("alpha.quantization.unpack-int2");

/**
 * alpha.quantization.unpack-int4
 * Unpack int4 operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationUnpackInt4 = defineStub<QuantizationRequest>("alpha.quantization.unpack-int4");

/**
 * alpha.quantization.unpack-int8
 * Unpack int8 operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationUnpackInt8 = defineStub<QuantizationRequest>("alpha.quantization.unpack-int8");

/**
 * alpha.quantization.unpack-nf4
 * Unpack nf4 operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationUnpackNf4 = defineStub<QuantizationRequest>("alpha.quantization.unpack-nf4");

/**
 * alpha.quantization.unpack-ternary
 * Unpack ternary operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationUnpackTernary = defineStub<QuantizationRequest>("alpha.quantization.unpack-ternary");

/**
 * alpha.quantization.vector-quantize
 * Vector quantize operation in the quantization family.
 * Status: standard; target: architecture-agnostic; differentiability: straight-through-or-none.
 */
export const quantizationVectorQuantize = defineStub<QuantizationRequest>("alpha.quantization.vector-quantize");
