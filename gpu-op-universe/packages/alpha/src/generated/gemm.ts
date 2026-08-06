/* AUTO-GENERATED. Do not hand-edit; edit operation-registry.json. */
import { defineStub } from "../../../common/src/types";
import type { MatmulRequest } from "../../../common/src/types";

/**
 * alpha.gemm.gemm
 * Gemm operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemm = defineStub<MatmulRequest>("alpha.gemm.gemm");

/**
 * alpha.gemm.gemm-amax
 * Gemm amax operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmAmax = defineStub<MatmulRequest>("alpha.gemm.gemm-amax");

/**
 * alpha.gemm.gemm-anytime-bitplane
 * Progressively refines a matrix product from high-significance bitplanes and may stop once an error bound is satisfied.
 * Status: research; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmAnytimeBitplane = defineStub<MatmulRequest>("alpha.gemm.gemm-anytime-bitplane");

/**
 * alpha.gemm.gemm-automatic-precision
 * Gemm automatic precision operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmAutomaticPrecision = defineStub<MatmulRequest>("alpha.gemm.gemm-automatic-precision");

/**
 * alpha.gemm.gemm-banded
 * Gemm banded operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmBanded = defineStub<MatmulRequest>("alpha.gemm.gemm-banded");

/**
 * alpha.gemm.gemm-bf16
 * Gemm bf16 operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmBf16 = defineStub<MatmulRequest>("alpha.gemm.gemm-bf16");

/**
 * alpha.gemm.gemm-bias
 * Gemm bias operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmBias = defineStub<MatmulRequest>("alpha.gemm.gemm-bias");

/**
 * alpha.gemm.gemm-bias-gelu
 * Gemm bias gelu operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmBiasGelu = defineStub<MatmulRequest>("alpha.gemm.gemm-bias-gelu");

/**
 * alpha.gemm.gemm-bias-relu
 * Gemm bias relu operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmBiasRelu = defineStub<MatmulRequest>("alpha.gemm.gemm-bias-relu");

/**
 * alpha.gemm.gemm-bias-silu
 * Gemm bias silu operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmBiasSilu = defineStub<MatmulRequest>("alpha.gemm.gemm-bias-silu");

/**
 * alpha.gemm.gemm-bias-swi-glu
 * Gemm bias swi glu operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmBiasSwiGLU = defineStub<MatmulRequest>("alpha.gemm.gemm-bias-swi-glu");

/**
 * alpha.gemm.gemm-binary
 * Gemm binary operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmBinary = defineStub<MatmulRequest>("alpha.gemm.gemm-binary");

/**
 * alpha.gemm.gemm-block-diagonal
 * Gemm block diagonal operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmBlockDiagonal = defineStub<MatmulRequest>("alpha.gemm.gemm-block-diagonal");

/**
 * alpha.gemm.gemm-block-sparse
 * Gemm block sparse operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmBlockSparse = defineStub<MatmulRequest>("alpha.gemm.gemm-block-sparse");

/**
 * alpha.gemm.gemm-boolean-or-and
 * Gemm boolean or and operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmBooleanOrAnd = defineStub<MatmulRequest>("alpha.gemm.gemm-boolean-or-and");

/**
 * alpha.gemm.gemm-boolean-xor-and
 * Gemm boolean xor and operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmBooleanXorAnd = defineStub<MatmulRequest>("alpha.gemm.gemm-boolean-xor-and");

/**
 * alpha.gemm.gemm-broadcast-batched
 * Gemm broadcast batched operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmBroadcastBatched = defineStub<MatmulRequest>("alpha.gemm.gemm-broadcast-batched");

/**
 * alpha.gemm.gemm-butterfly
 * Gemm butterfly operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmButterfly = defineStub<MatmulRequest>("alpha.gemm.gemm-butterfly");

/**
 * alpha.gemm.gemm-checksum-verified
 * Gemm checksum verified operation in the gemm family.
 * Status: research; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmChecksumVerified = defineStub<MatmulRequest>("alpha.gemm.gemm-checksum-verified");

/**
 * alpha.gemm.gemm-circulant
 * Gemm circulant operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmCirculant = defineStub<MatmulRequest>("alpha.gemm.gemm-circulant");

/**
 * alpha.gemm.gemm-coded-microbatch
 * Gemm coded microbatch operation in the gemm family.
 * Status: research; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmCodedMicrobatch = defineStub<MatmulRequest>("alpha.gemm.gemm-coded-microbatch");

/**
 * alpha.gemm.gemm-complex
 * Gemm complex operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmComplex = defineStub<MatmulRequest>("alpha.gemm.gemm-complex");

/**
 * alpha.gemm.gemm-conserved-moment
 * Gemm conserved moment operation in the gemm family.
 * Status: research; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmConservedMoment = defineStub<MatmulRequest>("alpha.gemm.gemm-conserved-moment");

/**
 * alpha.gemm.gemm-cooperative
 * Gemm cooperative operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmCooperative = defineStub<MatmulRequest>("alpha.gemm.gemm-cooperative");

/**
 * alpha.gemm.gemm-countercurrent
 * Schedules forward and backward tile streams against a shared stationary weight tile.
 * Status: research; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmCountercurrent = defineStub<MatmulRequest>("alpha.gemm.gemm-countercurrent");

/**
 * alpha.gemm.gemm-count-sketch
 * Gemm count sketch operation in the gemm family.
 * Status: research; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmCountSketch = defineStub<MatmulRequest>("alpha.gemm.gemm-count-sketch");

/**
 * alpha.gemm.gemm-covariance-accumulating
 * Gemm covariance accumulating operation in the gemm family.
 * Status: research; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmCovarianceAccumulating = defineStub<MatmulRequest>("alpha.gemm.gemm-covariance-accumulating");

/**
 * alpha.gemm.gemm-deferred-weight-gradient
 * Banks activation/adjoint factors and forms one larger weight-gradient product after accumulation.
 * Status: research; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmDeferredWeightGradient = defineStub<MatmulRequest>("alpha.gemm.gemm-deferred-weight-gradient");

/**
 * alpha.gemm.gemm-dequantize
 * Gemm dequantize operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmDequantize = defineStub<MatmulRequest>("alpha.gemm.gemm-dequantize");

/**
 * alpha.gemm.gemm-diagonal-plus-low-rank
 * Gemm diagonal plus low rank operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmDiagonalPlusLowRank = defineStub<MatmulRequest>("alpha.gemm.gemm-diagonal-plus-low-rank");

/**
 * alpha.gemm.gemm-dropout
 * Gemm dropout operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmDropout = defineStub<MatmulRequest>("alpha.gemm.gemm-dropout");

/**
 * alpha.gemm.gemm-dual-number
 * Gemm dual number operation in the gemm family.
 * Status: speculative; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmDualNumber = defineStub<MatmulRequest>("alpha.gemm.gemm-dual-number");

/**
 * alpha.gemm.gemm-error-feedback
 * Gemm error feedback operation in the gemm family.
 * Status: research; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmErrorFeedback = defineStub<MatmulRequest>("alpha.gemm.gemm-error-feedback");

/**
 * alpha.gemm.gemm-ex
 * Gemm ex operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmEx = defineStub<MatmulRequest>("alpha.gemm.gemm-ex");

/**
 * alpha.gemm.gemm-finite-field
 * Gemm finite field operation in the gemm family.
 * Status: speculative; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmFiniteField = defineStub<MatmulRequest>("alpha.gemm.gemm-finite-field");

/**
 * alpha.gemm.gemm-fp16
 * Gemm fp16 operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmFp16 = defineStub<MatmulRequest>("alpha.gemm.gemm-fp16");

/**
 * alpha.gemm.gemm-fused-epilogue
 * Gemm fused epilogue operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmFusedEpilogue = defineStub<MatmulRequest>("alpha.gemm.gemm-fused-epilogue");

/**
 * alpha.gemm.gemm-fused-prologue
 * Gemm fused prologue operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmFusedPrologue = defineStub<MatmulRequest>("alpha.gemm.gemm-fused-prologue");

/**
 * alpha.gemm.gemm-fused-prologue-epilogue
 * Gemm fused prologue epilogue operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmFusedPrologueEpilogue = defineStub<MatmulRequest>("alpha.gemm.gemm-fused-prologue-epilogue");

/**
 * alpha.gemm.gemm-gather-a
 * Gemm gather a operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmGatherA = defineStub<MatmulRequest>("alpha.gemm.gemm-gather-a");

/**
 * alpha.gemm.gemm-gather-b
 * Gemm gather b operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmGatherB = defineStub<MatmulRequest>("alpha.gemm.gemm-gather-b");

/**
 * alpha.gemm.gemm-gradient-sketch
 * Gemm gradient sketch operation in the gemm family.
 * Status: research; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmGradientSketch = defineStub<MatmulRequest>("alpha.gemm.gemm-gradient-sketch");

/**
 * alpha.gemm.gemm-grouped
 * Gemm grouped operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmGrouped = defineStub<MatmulRequest>("alpha.gemm.gemm-grouped");

/**
 * alpha.gemm.gemm-hadamard-factored
 * Gemm hadamard factored operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmHadamardFactored = defineStub<MatmulRequest>("alpha.gemm.gemm-hadamard-factored");

/**
 * alpha.gemm.gemm-hermitian
 * Gemm hermitian operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmHermitian = defineStub<MatmulRequest>("alpha.gemm.gemm-hermitian");

/**
 * alpha.gemm.gemm-innovation-only
 * Gemm innovation only operation in the gemm family.
 * Status: research; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmInnovationOnly = defineStub<MatmulRequest>("alpha.gemm.gemm-innovation-only");

/**
 * alpha.gemm.gemm-int4
 * Gemm int4 operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmInt4 = defineStub<MatmulRequest>("alpha.gemm.gemm-int4");

/**
 * alpha.gemm.gemm-int8
 * Gemm int8 operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmInt8 = defineStub<MatmulRequest>("alpha.gemm.gemm-int8");

/**
 * alpha.gemm.gemm-interval
 * Gemm interval operation in the gemm family.
 * Status: speculative; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmInterval = defineStub<MatmulRequest>("alpha.gemm.gemm-interval");

/**
 * alpha.gemm.gemm-khatri-rao
 * Gemm khatri rao operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmKhatriRao = defineStub<MatmulRequest>("alpha.gemm.gemm-khatri-rao");

/**
 * alpha.gemm.gemm-kronecker-factored
 * Gemm kronecker factored operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmKroneckerFactored = defineStub<MatmulRequest>("alpha.gemm.gemm-kronecker-factored");

/**
 * alpha.gemm.gemm-log-sum-exp-plus
 * Gemm log sum exp plus operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmLogSumExpPlus = defineStub<MatmulRequest>("alpha.gemm.gemm-log-sum-exp-plus");

/**
 * alpha.gemm.gemm-low-rank
 * Gemm low rank operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmLowRank = defineStub<MatmulRequest>("alpha.gemm.gemm-low-rank");

/**
 * alpha.gemm.gemm-masked
 * Gemm masked operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmMasked = defineStub<MatmulRequest>("alpha.gemm.gemm-masked");

/**
 * alpha.gemm.gemm-max-plus
 * Gemm max plus operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmMaxPlus = defineStub<MatmulRequest>("alpha.gemm.gemm-max-plus");

/**
 * alpha.gemm.gemm-max-times
 * Gemm max times operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmMaxTimes = defineStub<MatmulRequest>("alpha.gemm.gemm-max-times");

/**
 * alpha.gemm.gemm-min-max
 * Gemm min max operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmMinMax = defineStub<MatmulRequest>("alpha.gemm.gemm-min-max");

/**
 * alpha.gemm.gemm-min-plus
 * Gemm min plus operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmMinPlus = defineStub<MatmulRequest>("alpha.gemm.gemm-min-plus");

/**
 * alpha.gemm.gemm-mixed-accumulator
 * Gemm mixed accumulator operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmMixedAccumulator = defineStub<MatmulRequest>("alpha.gemm.gemm-mixed-accumulator");

/**
 * alpha.gemm.gemm-mixed-input
 * Gemm mixed input operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmMixedInput = defineStub<MatmulRequest>("alpha.gemm.gemm-mixed-input");

/**
 * alpha.gemm.gemm-muon-update
 * Gemm muon update operation in the gemm family.
 * Status: research; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmMuonUpdate = defineStub<MatmulRequest>("alpha.gemm.gemm-muon-update");

/**
 * alpha.gemm.gemm-optimizer-consumed
 * Produces the optimizer-transformed update or sufficient statistics directly instead of materializing a conventional gradient.
 * Status: research; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmOptimizerConsumed = defineStub<MatmulRequest>("alpha.gemm.gemm-optimizer-consumed");

/**
 * alpha.gemm.gemm-path-counting
 * Gemm path counting operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmPathCounting = defineStub<MatmulRequest>("alpha.gemm.gemm-path-counting");

/**
 * alpha.gemm.gemm-persistent
 * Gemm persistent operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmPersistent = defineStub<MatmulRequest>("alpha.gemm.gemm-persistent");

/**
 * alpha.gemm.gemm-pointer-array-batched
 * Gemm pointer array batched operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmPointerArrayBatched = defineStub<MatmulRequest>("alpha.gemm.gemm-pointer-array-batched");

/**
 * alpha.gemm.gemm-polar-epilogue
 * Gemm polar epilogue operation in the gemm family.
 * Status: research; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmPolarEpilogue = defineStub<MatmulRequest>("alpha.gemm.gemm-polar-epilogue");

/**
 * alpha.gemm.gemm-quantize
 * Gemm quantize operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmQuantize = defineStub<MatmulRequest>("alpha.gemm.gemm-quantize");

/**
 * alpha.gemm.gemm-quaternion
 * Gemm quaternion operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmQuaternion = defineStub<MatmulRequest>("alpha.gemm.gemm-quaternion");

/**
 * alpha.gemm.gemm-random-projection
 * Gemm random projection operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmRandomProjection = defineStub<MatmulRequest>("alpha.gemm.gemm-random-projection");

/**
 * alpha.gemm.gemm-reduce-batch
 * Gemm reduce batch operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmReduceBatch = defineStub<MatmulRequest>("alpha.gemm.gemm-reduce-batch");

/**
 * alpha.gemm.gemm-requantize
 * Gemm requantize operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmRequantize = defineStub<MatmulRequest>("alpha.gemm.gemm-requantize");

/**
 * alpha.gemm.gemm-residual
 * Gemm residual operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmResidual = defineStub<MatmulRequest>("alpha.gemm.gemm-residual");

/**
 * alpha.gemm.gemm-residual-norm
 * Gemm residual norm operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmResidualNorm = defineStub<MatmulRequest>("alpha.gemm.gemm-residual-norm");

/**
 * alpha.gemm.gemm-residue-corrected
 * Computes a cheap bulk product and an exact or high-precision correction in selected subspaces.
 * Status: research; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmResidueCorrected = defineStub<MatmulRequest>("alpha.gemm.gemm-residue-corrected");

/**
 * alpha.gemm.gemm-sampled-output
 * Gemm sampled output operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmSampledOutput = defineStub<MatmulRequest>("alpha.gemm.gemm-sampled-output");

/**
 * alpha.gemm.gemm-scatter-d
 * Gemm scatter d operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmScatterD = defineStub<MatmulRequest>("alpha.gemm.gemm-scatter-d");

/**
 * alpha.gemm.gemm-semiring
 * Generalized matrix multiplication with independently supplied combine and reduction operators.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmSemiring = defineStub<MatmulRequest>("alpha.gemm.gemm-semiring");

/**
 * alpha.gemm.gemm-sparse2of4
 * Gemm sparse2of4 operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmSparse2of4 = defineStub<MatmulRequest>("alpha.gemm.gemm-sparse2of4");

/**
 * alpha.gemm.gemm-sparse-nm
 * Gemm sparse nm operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmSparseNm = defineStub<MatmulRequest>("alpha.gemm.gemm-sparse-nm");

/**
 * alpha.gemm.gemm-split-kparallel
 * Gemm split kparallel operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmSplitKParallel = defineStub<MatmulRequest>("alpha.gemm.gemm-split-kparallel");

/**
 * alpha.gemm.gemm-split-kserial
 * Gemm split kserial operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmSplitKSerial = defineStub<MatmulRequest>("alpha.gemm.gemm-split-kserial");

/**
 * alpha.gemm.gemm-stream-k
 * Gemm stream k operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmStreamK = defineStub<MatmulRequest>("alpha.gemm.gemm-stream-k");

/**
 * alpha.gemm.gemm-strided-batched
 * Gemm strided batched operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmStridedBatched = defineStub<MatmulRequest>("alpha.gemm.gemm-strided-batched");

/**
 * alpha.gemm.gemm-symmetric
 * Gemm symmetric operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmSymmetric = defineStub<MatmulRequest>("alpha.gemm.gemm-symmetric");

/**
 * alpha.gemm.gemm-tensor-sketch
 * Gemm tensor sketch operation in the gemm family.
 * Status: research; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmTensorSketch = defineStub<MatmulRequest>("alpha.gemm.gemm-tensor-sketch");

/**
 * alpha.gemm.gemm-ternary
 * Gemm ternary operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmTernary = defineStub<MatmulRequest>("alpha.gemm.gemm-ternary");

/**
 * alpha.gemm.gemm-tf32
 * Gemm tf32 operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmTf32 = defineStub<MatmulRequest>("alpha.gemm.gemm-tf32");

/**
 * alpha.gemm.gemm-tile-stream
 * Gemm tile stream operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmTileStream = defineStub<MatmulRequest>("alpha.gemm.gemm-tile-stream");

/**
 * alpha.gemm.gemm-toeplitz
 * Gemm toeplitz operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmToeplitz = defineStub<MatmulRequest>("alpha.gemm.gemm-toeplitz");

/**
 * alpha.gemm.gemm-triangular
 * Gemm triangular operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmTriangular = defineStub<MatmulRequest>("alpha.gemm.gemm-triangular");

/**
 * alpha.gemm.gemm-variable-batched
 * Gemm variable batched operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmVariableBatched = defineStub<MatmulRequest>("alpha.gemm.gemm-variable-batched");

/**
 * alpha.gemm.gemm-viterbi
 * Gemm viterbi operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmViterbi = defineStub<MatmulRequest>("alpha.gemm.gemm-viterbi");

/**
 * alpha.gemm.gemm-warp-specialized
 * Gemm warp specialized operation in the gemm family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const gemmGemmWarpSpecialized = defineStub<MatmulRequest>("alpha.gemm.gemm-warp-specialized");
