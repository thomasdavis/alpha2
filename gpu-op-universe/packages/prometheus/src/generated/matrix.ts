/* AUTO-GENERATED. Do not hand-edit; edit operation-registry.json. */
import { defineStub } from "../../../common/src/types";
import type { CompilerOpRequest } from "../../../common/src/types";

/**
 * prometheus.matrix.anytime-bitplane-matmul
 * Anytime bitplane matmul operation in the matrix family.
 * Status: research; target: compiler, sm86; differentiability: not-applicable.
 */
export const matrixAnytimeBitplaneMatmul = defineStub<CompilerOpRequest>("prometheus.matrix.anytime-bitplane-matmul");

/**
 * prometheus.matrix.block-matmul
 * Block matmul operation in the matrix family.
 * Status: standard; target: compiler, sm86; differentiability: not-applicable.
 */
export const matrixBlockMatmul = defineStub<CompilerOpRequest>("prometheus.matrix.block-matmul");

/**
 * prometheus.matrix.block-sparse-matmul
 * Block sparse matmul operation in the matrix family.
 * Status: standard; target: compiler, sm86; differentiability: not-applicable.
 */
export const matrixBlockSparseMatmul = defineStub<CompilerOpRequest>("prometheus.matrix.block-sparse-matmul");

/**
 * prometheus.matrix.fused-epilogue-matmul
 * Fused epilogue matmul operation in the matrix family.
 * Status: standard; target: compiler, sm86; differentiability: not-applicable.
 */
export const matrixFusedEpilogueMatmul = defineStub<CompilerOpRequest>("prometheus.matrix.fused-epilogue-matmul");

/**
 * prometheus.matrix.grid-matmul
 * Grid matmul operation in the matrix family.
 * Status: standard; target: compiler, sm86; differentiability: not-applicable.
 */
export const matrixGridMatmul = defineStub<CompilerOpRequest>("prometheus.matrix.grid-matmul");

/**
 * prometheus.matrix.grouped-matmul
 * Grouped matmul operation in the matrix family.
 * Status: standard; target: compiler, sm86; differentiability: not-applicable.
 */
export const matrixGroupedMatmul = defineStub<CompilerOpRequest>("prometheus.matrix.grouped-matmul");

/**
 * prometheus.matrix.masked-matmul
 * Masked matmul operation in the matrix family.
 * Status: standard; target: compiler, sm86; differentiability: not-applicable.
 */
export const matrixMaskedMatmul = defineStub<CompilerOpRequest>("prometheus.matrix.masked-matmul");

/**
 * prometheus.matrix.mixed-input-matmul
 * Mixed input matmul operation in the matrix family.
 * Status: standard; target: compiler, future-or-emulated, sm86; differentiability: not-applicable.
 */
export const matrixMixedInputMatmul = defineStub<CompilerOpRequest>("prometheus.matrix.mixed-input-matmul");

/**
 * prometheus.matrix.mma
 * Mma operation in the matrix family.
 * Status: standard; target: compiler, sm86; differentiability: not-applicable.
 */
export const matrixMma = defineStub<CompilerOpRequest>("prometheus.matrix.mma");

/**
 * prometheus.matrix.mma-bf16
 * Mma bf16 operation in the matrix family.
 * Status: standard; target: compiler, sm86; differentiability: not-applicable.
 */
export const matrixMmaBf16 = defineStub<CompilerOpRequest>("prometheus.matrix.mma-bf16");

/**
 * prometheus.matrix.mma-binary
 * Mma binary operation in the matrix family.
 * Status: standard; target: compiler, sm86; differentiability: not-applicable.
 */
export const matrixMmaBinary = defineStub<CompilerOpRequest>("prometheus.matrix.mma-binary");

/**
 * prometheus.matrix.mma-fp16
 * Mma fp16 operation in the matrix family.
 * Status: standard; target: compiler, sm86; differentiability: not-applicable.
 */
export const matrixMmaFp16 = defineStub<CompilerOpRequest>("prometheus.matrix.mma-fp16");

/**
 * prometheus.matrix.mma-int4
 * Mma int4 operation in the matrix family.
 * Status: standard; target: compiler, sm86; differentiability: not-applicable.
 */
export const matrixMmaInt4 = defineStub<CompilerOpRequest>("prometheus.matrix.mma-int4");

/**
 * prometheus.matrix.mma-int8
 * Mma int8 operation in the matrix family.
 * Status: standard; target: compiler, sm86; differentiability: not-applicable.
 */
export const matrixMmaInt8 = defineStub<CompilerOpRequest>("prometheus.matrix.mma-int8");

/**
 * prometheus.matrix.mma-sparse
 * Mma sparse operation in the matrix family.
 * Status: standard; target: compiler, sm86; differentiability: not-applicable.
 */
export const matrixMmaSparse = defineStub<CompilerOpRequest>("prometheus.matrix.mma-sparse");

/**
 * prometheus.matrix.mma-tf32
 * Mma tf32 operation in the matrix family.
 * Status: standard; target: compiler, sm86; differentiability: not-applicable.
 */
export const matrixMmaTf32 = defineStub<CompilerOpRequest>("prometheus.matrix.mma-tf32");

/**
 * prometheus.matrix.optimizer-consumed-matmul
 * Optimizer consumed matmul operation in the matrix family.
 * Status: research; target: compiler, sm86; differentiability: not-applicable.
 */
export const matrixOptimizerConsumedMatmul = defineStub<CompilerOpRequest>("prometheus.matrix.optimizer-consumed-matmul");

/**
 * prometheus.matrix.persistent-matmul
 * Persistent matmul operation in the matrix family.
 * Status: standard; target: compiler, future-or-emulated, sm86; differentiability: not-applicable.
 */
export const matrixPersistentMatmul = defineStub<CompilerOpRequest>("prometheus.matrix.persistent-matmul");

/**
 * prometheus.matrix.quantized-matmul
 * Quantized matmul operation in the matrix family.
 * Status: standard; target: compiler, sm86; differentiability: not-applicable.
 */
export const matrixQuantizedMatmul = defineStub<CompilerOpRequest>("prometheus.matrix.quantized-matmul");

/**
 * prometheus.matrix.residue-corrected-matmul
 * Residue corrected matmul operation in the matrix family.
 * Status: research; target: compiler, sm86; differentiability: not-applicable.
 */
export const matrixResidueCorrectedMatmul = defineStub<CompilerOpRequest>("prometheus.matrix.residue-corrected-matmul");

/**
 * prometheus.matrix.semiring-matmul
 * Semiring matmul operation in the matrix family.
 * Status: standard; target: compiler, sm86; differentiability: not-applicable.
 */
export const matrixSemiringMatmul = defineStub<CompilerOpRequest>("prometheus.matrix.semiring-matmul");

/**
 * prometheus.matrix.simt-matmul
 * Simt matmul operation in the matrix family.
 * Status: standard; target: compiler, future-or-emulated, sm86; differentiability: not-applicable.
 */
export const matrixSimtMatmul = defineStub<CompilerOpRequest>("prometheus.matrix.simt-matmul");

/**
 * prometheus.matrix.split-kmatmul
 * Split kmatmul operation in the matrix family.
 * Status: standard; target: compiler, sm86; differentiability: not-applicable.
 */
export const matrixSplitKMatmul = defineStub<CompilerOpRequest>("prometheus.matrix.split-kmatmul");

/**
 * prometheus.matrix.stream-kmatmul
 * Stream kmatmul operation in the matrix family.
 * Status: standard; target: compiler, sm86; differentiability: not-applicable.
 */
export const matrixStreamKMatmul = defineStub<CompilerOpRequest>("prometheus.matrix.stream-kmatmul");

/**
 * prometheus.matrix.warp-matmul
 * Warp matmul operation in the matrix family.
 * Status: standard; target: compiler, sm86; differentiability: not-applicable.
 */
export const matrixWarpMatmul = defineStub<CompilerOpRequest>("prometheus.matrix.warp-matmul");
