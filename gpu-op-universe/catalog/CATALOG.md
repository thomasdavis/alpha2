# Alpha GPU Operation Universe

This is a generated, searchable catalog of canonical operations and empty TypeScript implementation stubs for the complete Alpha stack. It is intentionally broader than CUDA, Vulkan, BLAS, or a conventional tensor library.

> No finite list can contain every mathematically conceivable operator. The registry therefore combines a broad canonical vocabulary with parameterized dtype, layout, algebra, precision, scope, and determinism dimensions. Add a new canonical operation only when its semantics or lowering are materially different.

## Counts

| Layer | Operations |
|---|---:|
| `aether` | 159 |
| `alpha` | 1,165 |
| `chronos` | 146 |
| `gaia` | 162 |
| `helios` | 188 |
| `hephaestus` | 288 |
| `hermes` | 141 |
| `prometheus` | 395 |
| **Total** | **2,644** |

| Status | Count |
|---|---:|
| `research` | 103 |
| `speculative` | 9 |
| `standard` | 2,532 |

## Registry fields

- `status=standard`: well-established semantics; implementation may still be absent.
- `status=research`: supported by related work or a coherent derivation, but requires dedicated validation.
- `status=speculative`: retained as a design prompt; Codex must not implement it without an approved experiment specification.
- `target`: distinguishes architecture-agnostic, host, compiler, native `sm_86`, and future/emulated ideas.
- `sourceTags`: points to the primary specifications or papers that motivated the canonical operation.

## Catalog


# Alpha


## `attention` (52)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `attentionAttentionBackward` | `alpha.attention.attention-backward` | `standard` | Attention backward operation in the attention family. |
| `attentionAttentionChecksum` | `alpha.attention.attention-checksum` | `research` | Attention checksum operation in the attention family. |
| `attentionAttentionKvAppend` | `alpha.attention.attention-kv-append` | `standard` | Attention kv append operation in the attention family. |
| `attentionAttentionKvCompact` | `alpha.attention.attention-kv-compact` | `standard` | Attention kv compact operation in the attention family. |
| `attentionAttentionKvEvict` | `alpha.attention.attention-kv-evict` | `standard` | Attention kv evict operation in the attention family. |
| `attentionAttentionKvGather` | `alpha.attention.attention-kv-gather` | `standard` | Attention kv gather operation in the attention family. |
| `attentionAttentionKvQuantize` | `alpha.attention.attention-kv-quantize` | `standard` | Attention kv quantize operation in the attention family. |
| `attentionAttentionPrefixShare` | `alpha.attention.attention-prefix-share` | `standard` | Attention prefix share operation in the attention family. |
| `attentionAttentionScoreOnly` | `alpha.attention.attention-score-only` | `standard` | Attention score only operation in the attention family. |
| `attentionAttentionTreeShare` | `alpha.attention.attention-tree-share` | `standard` | Attention tree share operation in the attention family. |
| `attentionAttentionValueOnly` | `alpha.attention.attention-value-only` | `standard` | Attention value only operation in the attention family. |
| `attentionAttentionWithRetrieval` | `alpha.attention.attention-with-retrieval` | `standard` | Attention with retrieval operation in the attention family. |
| `attentionBidirectionalAttention` | `alpha.attention.bidirectional-attention` | `standard` | Bidirectional attention operation in the attention family. |
| `attentionBlockSparseAttention` | `alpha.attention.block-sparse-attention` | `standard` | Block sparse attention operation in the attention family. |
| `attentionCausalAttention` | `alpha.attention.causal-attention` | `standard` | Causal attention operation in the attention family. |
| `attentionChunkedAttention` | `alpha.attention.chunked-attention` | `standard` | Chunked attention operation in the attention family. |
| `attentionCosineAttention` | `alpha.attention.cosine-attention` | `standard` | Cosine attention operation in the attention family. |
| `attentionCrossAttention` | `alpha.attention.cross-attention` | `standard` | Cross attention operation in the attention family. |
| `attentionDeltaNetAttention` | `alpha.attention.delta-net-attention` | `standard` | Delta net attention operation in the attention family. |
| `attentionDilatedAttention` | `alpha.attention.dilated-attention` | `standard` | Dilated attention operation in the attention family. |
| `attentionEntmaxAttention` | `alpha.attention.entmax-attention` | `standard` | Entmax attention operation in the attention family. |
| `attentionFlashAttention` | `alpha.attention.flash-attention` | `standard` | Flash attention operation in the attention family. |
| `attentionFlashAttentionBackward` | `alpha.attention.flash-attention-backward` | `standard` | Flash attention backward operation in the attention family. |
| `attentionFlashDecoding` | `alpha.attention.flash-decoding` | `standard` | Flash decoding operation in the attention family. |
| `attentionGatedDeltaNet` | `alpha.attention.gated-delta-net` | `standard` | Gated delta net operation in the attention family. |
| `attentionGatedRetention` | `alpha.attention.gated-retention` | `standard` | Gated retention operation in the attention family. |
| `attentionGlobalLocalAttention` | `alpha.attention.global-local-attention` | `standard` | Global local attention operation in the attention family. |
| `attentionGroupedQueryAttention` | `alpha.attention.grouped-query-attention` | `standard` | Grouped query attention operation in the attention family. |
| `attentionHardAttention` | `alpha.attention.hard-attention` | `standard` | Hard attention operation in the attention family. |
| `attentionHashAttention` | `alpha.attention.hash-attention` | `standard` | Hash attention operation in the attention family. |
| `attentionKernelLinearAttention` | `alpha.attention.kernel-linear-attention` | `standard` | Kernel linear attention operation in the attention family. |
| `attentionLocalAttention` | `alpha.attention.local-attention` | `standard` | Local attention operation in the attention family. |
| `attentionMemoryCompressedAttention` | `alpha.attention.memory-compressed-attention` | `standard` | Memory compressed attention operation in the attention family. |
| `attentionMonotonicAttention` | `alpha.attention.monotonic-attention` | `standard` | Monotonic attention operation in the attention family. |
| `attentionMultiHeadAttention` | `alpha.attention.multi-head-attention` | `standard` | Multi head attention operation in the attention family. |
| `attentionMultiHeadLatentAttention` | `alpha.attention.multi-head-latent-attention` | `standard` | Multi head latent attention operation in the attention family. |
| `attentionMultiQueryAttention` | `alpha.attention.multi-query-attention` | `standard` | Multi query attention operation in the attention family. |
| `attentionOnlineSoftmaxAttention` | `alpha.attention.online-softmax-attention` | `standard` | Online softmax attention operation in the attention family. |
| `attentionPagedAttention` | `alpha.attention.paged-attention` | `standard` | Paged attention operation in the attention family. |
| `attentionPerformerAttention` | `alpha.attention.performer-attention` | `standard` | Performer attention operation in the attention family. |
| `attentionPrefixAttention` | `alpha.attention.prefix-attention` | `standard` | Prefix attention operation in the attention family. |
| `attentionRecurrentAttention` | `alpha.attention.recurrent-attention` | `standard` | Recurrent attention operation in the attention family. |
| `attentionRetention` | `alpha.attention.retention` | `standard` | Retention operation in the attention family. |
| `attentionRingAttention` | `alpha.attention.ring-attention` | `standard` | Ring attention operation in the attention family. |
| `attentionRoutingAttention` | `alpha.attention.routing-attention` | `standard` | Routing attention operation in the attention family. |
| `attentionScaledDotProductAttention` | `alpha.attention.scaled-dot-product-attention` | `standard` | Scaled dot product attention operation in the attention family. |
| `attentionSinkhornAttention` | `alpha.attention.sinkhorn-attention` | `standard` | Sinkhorn attention operation in the attention family. |
| `attentionSlidingWindowAttention` | `alpha.attention.sliding-window-attention` | `standard` | Sliding window attention operation in the attention family. |
| `attentionSparsemaxAttention` | `alpha.attention.sparsemax-attention` | `standard` | Sparsemax attention operation in the attention family. |
| `attentionStreamingAttention` | `alpha.attention.streaming-attention` | `standard` | Streaming attention operation in the attention family. |
| `attentionStridedAttention` | `alpha.attention.strided-attention` | `standard` | Strided attention operation in the attention family. |
| `attentionTopKAttention` | `alpha.attention.top-kattention` | `standard` | Top kattention operation in the attention family. |

## `autodiff` (38)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `autodiffCheckpoint` | `alpha.autodiff.checkpoint` | `standard` | Checkpoint operation in the autodiff family. |
| `autodiffComplexStepDerivative` | `alpha.autodiff.complex-step-derivative` | `standard` | Complex step derivative operation in the autodiff family. |
| `autodiffCustomJvp` | `alpha.autodiff.custom-jvp` | `standard` | Custom jvp operation in the autodiff family. |
| `autodiffCustomVjp` | `alpha.autodiff.custom-vjp` | `standard` | Custom vjp operation in the autodiff family. |
| `autodiffEquilibriumAdjoint` | `alpha.autodiff.equilibrium-adjoint` | `standard` | Equilibrium adjoint operation in the autodiff family. |
| `autodiffFiniteDifference` | `alpha.autodiff.finite-difference` | `standard` | Finite difference operation in the autodiff family. |
| `autodiffForwardMode` | `alpha.autodiff.forward-mode` | `standard` | Forward mode operation in the autodiff family. |
| `autodiffForwardOverReverse` | `alpha.autodiff.forward-over-reverse` | `standard` | Forward over reverse operation in the autodiff family. |
| `autodiffGradientAccumulate` | `alpha.autodiff.gradient-accumulate` | `standard` | Gradient accumulate operation in the autodiff family. |
| `autodiffGradientAllReduce` | `alpha.autodiff.gradient-all-reduce` | `standard` | Gradient all reduce operation in the autodiff family. |
| `autodiffGradientCentralize` | `alpha.autodiff.gradient-centralize` | `standard` | Gradient centralize operation in the autodiff family. |
| `autodiffGradientClipNorm` | `alpha.autodiff.gradient-clip-norm` | `standard` | Gradient clip norm operation in the autodiff family. |
| `autodiffGradientClipValue` | `alpha.autodiff.gradient-clip-value` | `standard` | Gradient clip value operation in the autodiff family. |
| `autodiffGradientCompress` | `alpha.autodiff.gradient-compress` | `standard` | Gradient compress operation in the autodiff family. |
| `autodiffGradientErrorFeedback` | `alpha.autodiff.gradient-error-feedback` | `research` | Gradient error feedback operation in the autodiff family. |
| `autodiffGradientNoiseInject` | `alpha.autodiff.gradient-noise-inject` | `standard` | Gradient noise inject operation in the autodiff family. |
| `autodiffGradientReduceScatter` | `alpha.autodiff.gradient-reduce-scatter` | `standard` | Gradient reduce scatter operation in the autodiff family. |
| `autodiffGradientScale` | `alpha.autodiff.gradient-scale` | `standard` | Gradient scale operation in the autodiff family. |
| `autodiffGradientSketch` | `alpha.autodiff.gradient-sketch` | `research` | Gradient sketch operation in the autodiff family. |
| `autodiffGradientUnscale` | `alpha.autodiff.gradient-unscale` | `standard` | Gradient unscale operation in the autodiff family. |
| `autodiffHessian` | `alpha.autodiff.hessian` | `standard` | Hessian operation in the autodiff family. |
| `autodiffHessianVectorProduct` | `alpha.autodiff.hessian-vector-product` | `standard` | Hessian vector product operation in the autodiff family. |
| `autodiffImplicitDifferentiate` | `alpha.autodiff.implicit-differentiate` | `standard` | Implicit differentiate operation in the autodiff family. |
| `autodiffJacobian` | `alpha.autodiff.jacobian` | `standard` | Jacobian operation in the autodiff family. |
| `autodiffJvp` | `alpha.autodiff.jvp` | `standard` | Jvp operation in the autodiff family. |
| `autodiffQuantizedAdjoint` | `alpha.autodiff.quantized-adjoint` | `standard` | Quantized adjoint operation in the autodiff family. |
| `autodiffRecompute` | `alpha.autodiff.recompute` | `standard` | Recompute operation in the autodiff family. |
| `autodiffRematerialize` | `alpha.autodiff.rematerialize` | `standard` | Rematerialize operation in the autodiff family. |
| `autodiffReparameterizationEstimator` | `alpha.autodiff.reparameterization-estimator` | `standard` | Reparameterization estimator operation in the autodiff family. |
| `autodiffReverseMode` | `alpha.autodiff.reverse-mode` | `standard` | Reverse mode operation in the autodiff family. |
| `autodiffReverseOverForward` | `alpha.autodiff.reverse-over-forward` | `standard` | Reverse over forward operation in the autodiff family. |
| `autodiffScanAdjoint` | `alpha.autodiff.scan-adjoint` | `standard` | Scan adjoint operation in the autodiff family. |
| `autodiffScoreFunctionEstimator` | `alpha.autodiff.score-function-estimator` | `standard` | Score function estimator operation in the autodiff family. |
| `autodiffSparseAdjoint` | `alpha.autodiff.sparse-adjoint` | `standard` | Sparse adjoint operation in the autodiff family. |
| `autodiffStopGradient` | `alpha.autodiff.stop-gradient` | `standard` | Stop gradient operation in the autodiff family. |
| `autodiffStraightThroughEstimator` | `alpha.autodiff.straight-through-estimator` | `standard` | Straight through estimator operation in the autodiff family. |
| `autodiffVectorHessianProduct` | `alpha.autodiff.vector-hessian-product` | `standard` | Vector hessian product operation in the autodiff family. |
| `autodiffVjp` | `alpha.autodiff.vjp` | `standard` | Vjp operation in the autodiff family. |

## `binary_math` (51)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `binaryMathAdd` | `alpha.binary_math.add` | `standard` | Add operation in the binary math family. |
| `binaryMathAtan2` | `alpha.binary_math.atan2` | `standard` | Atan2 operation in the binary math family. |
| `binaryMathAverageRounded` | `alpha.binary_math.average-rounded` | `standard` | Average rounded operation in the binary math family. |
| `binaryMathBitwiseAnd` | `alpha.binary_math.bitwise-and` | `standard` | Bitwise and operation in the binary math family. |
| `binaryMathBitwiseOr` | `alpha.binary_math.bitwise-or` | `standard` | Bitwise or operation in the binary math family. |
| `binaryMathBitwiseXor` | `alpha.binary_math.bitwise-xor` | `standard` | Bitwise xor operation in the binary math family. |
| `binaryMathComplexDiv` | `alpha.binary_math.complex-div` | `standard` | Complex div operation in the binary math family. |
| `binaryMathComplexMul` | `alpha.binary_math.complex-mul` | `standard` | Complex mul operation in the binary math family. |
| `binaryMathCopySign` | `alpha.binary_math.copy-sign` | `standard` | Copy sign operation in the binary math family. |
| `binaryMathDiv` | `alpha.binary_math.div` | `standard` | Div operation in the binary math family. |
| `binaryMathDualNumberMul` | `alpha.binary_math.dual-number-mul` | `speculative` | Dual number mul operation in the binary math family. |
| `binaryMathEqual` | `alpha.binary_math.equal` | `standard` | Equal operation in the binary math family. |
| `binaryMathFloorDiv` | `alpha.binary_math.floor-div` | `standard` | Floor div operation in the binary math family. |
| `binaryMathFmax` | `alpha.binary_math.fmax` | `standard` | Fmax operation in the binary math family. |
| `binaryMathFmin` | `alpha.binary_math.fmin` | `standard` | Fmin operation in the binary math family. |
| `binaryMathFmod` | `alpha.binary_math.fmod` | `standard` | Fmod operation in the binary math family. |
| `binaryMathGcd` | `alpha.binary_math.gcd` | `standard` | Gcd operation in the binary math family. |
| `binaryMathGreater` | `alpha.binary_math.greater` | `standard` | Greater operation in the binary math family. |
| `binaryMathGreaterEqual` | `alpha.binary_math.greater-equal` | `standard` | Greater equal operation in the binary math family. |
| `binaryMathHypot` | `alpha.binary_math.hypot` | `standard` | Hypot operation in the binary math family. |
| `binaryMathIntervalAdd` | `alpha.binary_math.interval-add` | `speculative` | Interval add operation in the binary math family. |
| `binaryMathIntervalMul` | `alpha.binary_math.interval-mul` | `speculative` | Interval mul operation in the binary math family. |
| `binaryMathLcm` | `alpha.binary_math.lcm` | `standard` | Lcm operation in the binary math family. |
| `binaryMathLess` | `alpha.binary_math.less` | `standard` | Less operation in the binary math family. |
| `binaryMathLessEqual` | `alpha.binary_math.less-equal` | `standard` | Less equal operation in the binary math family. |
| `binaryMathLogAddExp` | `alpha.binary_math.log-add-exp` | `standard` | Log add exp operation in the binary math family. |
| `binaryMathLogAddExp2` | `alpha.binary_math.log-add-exp2` | `standard` | Log add exp2 operation in the binary math family. |
| `binaryMathLogicalAnd` | `alpha.binary_math.logical-and` | `standard` | Logical and operation in the binary math family. |
| `binaryMathLogicalOr` | `alpha.binary_math.logical-or` | `standard` | Logical or operation in the binary math family. |
| `binaryMathLogicalXor` | `alpha.binary_math.logical-xor` | `standard` | Logical xor operation in the binary math family. |
| `binaryMathMaximum` | `alpha.binary_math.maximum` | `standard` | Maximum operation in the binary math family. |
| `binaryMathMinimum` | `alpha.binary_math.minimum` | `standard` | Minimum operation in the binary math family. |
| `binaryMathMul` | `alpha.binary_math.mul` | `standard` | Mul operation in the binary math family. |
| `binaryMathNextAfter` | `alpha.binary_math.next-after` | `standard` | Next after operation in the binary math family. |
| `binaryMathNotEqual` | `alpha.binary_math.not-equal` | `standard` | Not equal operation in the binary math family. |
| `binaryMathPow` | `alpha.binary_math.pow` | `standard` | Pow operation in the binary math family. |
| `binaryMathRemainder` | `alpha.binary_math.remainder` | `standard` | Remainder operation in the binary math family. |
| `binaryMathRotateLeft` | `alpha.binary_math.rotate-left` | `standard` | Rotate left operation in the binary math family. |
| `binaryMathRotateRight` | `alpha.binary_math.rotate-right` | `standard` | Rotate right operation in the binary math family. |
| `binaryMathSaturatingAdd` | `alpha.binary_math.saturating-add` | `standard` | Saturating add operation in the binary math family. |
| `binaryMathSaturatingSub` | `alpha.binary_math.saturating-sub` | `standard` | Saturating sub operation in the binary math family. |
| `binaryMathShiftLeft` | `alpha.binary_math.shift-left` | `standard` | Shift left operation in the binary math family. |
| `binaryMathShiftRightArithmetic` | `alpha.binary_math.shift-right-arithmetic` | `standard` | Shift right arithmetic operation in the binary math family. |
| `binaryMathShiftRightLogical` | `alpha.binary_math.shift-right-logical` | `standard` | Shift right logical operation in the binary math family. |
| `binaryMathSquaredDifference` | `alpha.binary_math.squared-difference` | `standard` | Squared difference operation in the binary math family. |
| `binaryMathSub` | `alpha.binary_math.sub` | `standard` | Sub operation in the binary math family. |
| `binaryMathTrueDiv` | `alpha.binary_math.true-div` | `standard` | True div operation in the binary math family. |
| `binaryMathTruncDiv` | `alpha.binary_math.trunc-div` | `standard` | Trunc div operation in the binary math family. |
| `binaryMathUnorderedCompare` | `alpha.binary_math.unordered-compare` | `standard` | Unordered compare operation in the binary math family. |
| `binaryMathXlog1py` | `alpha.binary_math.xlog1py` | `standard` | Xlog1py operation in the binary math family. |
| `binaryMathXlogy` | `alpha.binary_math.xlogy` | `standard` | Xlogy operation in the binary math family. |

## `blas1` (17)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `blas1Asum` | `alpha.blas1.asum` | `standard` | Asum operation in the blas1 family. |
| `blas1Axpby` | `alpha.blas1.axpby` | `standard` | Axpby operation in the blas1 family. |
| `blas1Axpy` | `alpha.blas1.axpy` | `standard` | Axpy operation in the blas1 family. |
| `blas1Dot` | `alpha.blas1.dot` | `standard` | Dot operation in the blas1 family. |
| `blas1DotConjugate` | `alpha.blas1.dot-conjugate` | `standard` | Dot conjugate operation in the blas1 family. |
| `blas1GenerateGivens` | `alpha.blas1.generate-givens` | `standard` | Generate givens operation in the blas1 family. |
| `blas1GenerateModifiedGivens` | `alpha.blas1.generate-modified-givens` | `standard` | Generate modified givens operation in the blas1 family. |
| `blas1Iamax` | `alpha.blas1.iamax` | `standard` | Iamax operation in the blas1 family. |
| `blas1Iamin` | `alpha.blas1.iamin` | `standard` | Iamin operation in the blas1 family. |
| `blas1ModifiedGivens` | `alpha.blas1.modified-givens` | `standard` | Modified givens operation in the blas1 family. |
| `blas1Norm1` | `alpha.blas1.norm1` | `standard` | Norm1 operation in the blas1 family. |
| `blas1Norm2` | `alpha.blas1.norm2` | `standard` | Norm2 operation in the blas1 family. |
| `blas1NormInf` | `alpha.blas1.norm-inf` | `standard` | Norm inf operation in the blas1 family. |
| `blas1PlaneRotation` | `alpha.blas1.plane-rotation` | `standard` | Plane rotation operation in the blas1 family. |
| `blas1VectorCopy` | `alpha.blas1.vector-copy` | `standard` | Vector copy operation in the blas1 family. |
| `blas1VectorScale` | `alpha.blas1.vector-scale` | `standard` | Vector scale operation in the blas1 family. |
| `blas1VectorSwap` | `alpha.blas1.vector-swap` | `standard` | Vector swap operation in the blas1 family. |

## `blas2` (27)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `blas2Gbmv` | `alpha.blas2.gbmv` | `standard` | Gbmv operation in the blas2 family. |
| `blas2Gemv` | `alpha.blas2.gemv` | `standard` | Gemv operation in the blas2 family. |
| `blas2Ger` | `alpha.blas2.ger` | `standard` | Ger operation in the blas2 family. |
| `blas2GerConjugate` | `alpha.blas2.ger-conjugate` | `standard` | Ger conjugate operation in the blas2 family. |
| `blas2Hbmv` | `alpha.blas2.hbmv` | `standard` | Hbmv operation in the blas2 family. |
| `blas2Hemv` | `alpha.blas2.hemv` | `standard` | Hemv operation in the blas2 family. |
| `blas2Her` | `alpha.blas2.her` | `standard` | Her operation in the blas2 family. |
| `blas2Her2` | `alpha.blas2.her2` | `standard` | Her2 operation in the blas2 family. |
| `blas2HpmvPacked` | `alpha.blas2.hpmv-packed` | `standard` | Hpmv packed operation in the blas2 family. |
| `blas2Hpr` | `alpha.blas2.hpr` | `standard` | Hpr operation in the blas2 family. |
| `blas2Hpr2` | `alpha.blas2.hpr2` | `standard` | Hpr2 operation in the blas2 family. |
| `blas2OuterProduct` | `alpha.blas2.outer-product` | `standard` | Outer product operation in the blas2 family. |
| `blas2RankOneUpdate` | `alpha.blas2.rank-one-update` | `standard` | Rank one update operation in the blas2 family. |
| `blas2RankTwoUpdate` | `alpha.blas2.rank-two-update` | `standard` | Rank two update operation in the blas2 family. |
| `blas2Sbmv` | `alpha.blas2.sbmv` | `standard` | Sbmv operation in the blas2 family. |
| `blas2SpmvPacked` | `alpha.blas2.spmv-packed` | `standard` | Spmv packed operation in the blas2 family. |
| `blas2Spr` | `alpha.blas2.spr` | `standard` | Spr operation in the blas2 family. |
| `blas2Spr2` | `alpha.blas2.spr2` | `standard` | Spr2 operation in the blas2 family. |
| `blas2Symv` | `alpha.blas2.symv` | `standard` | Symv operation in the blas2 family. |
| `blas2Syr` | `alpha.blas2.syr` | `standard` | Syr operation in the blas2 family. |
| `blas2Syr2` | `alpha.blas2.syr2` | `standard` | Syr2 operation in the blas2 family. |
| `blas2Tbmv` | `alpha.blas2.tbmv` | `standard` | Tbmv operation in the blas2 family. |
| `blas2Tbsv` | `alpha.blas2.tbsv` | `standard` | Tbsv operation in the blas2 family. |
| `blas2Tpmv` | `alpha.blas2.tpmv` | `standard` | Tpmv operation in the blas2 family. |
| `blas2Tpsv` | `alpha.blas2.tpsv` | `standard` | Tpsv operation in the blas2 family. |
| `blas2Trmv` | `alpha.blas2.trmv` | `standard` | Trmv operation in the blas2 family. |
| `blas2Trsv` | `alpha.blas2.trsv` | `standard` | Trsv operation in the blas2 family. |

## `blas3` (11)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `blas3BatchedTrsm` | `alpha.blas3.batched-trsm` | `standard` | Batched trsm operation in the blas3 family. |
| `blas3Hemm` | `alpha.blas3.hemm` | `standard` | Hemm operation in the blas3 family. |
| `blas3Her2k` | `alpha.blas3.her2k` | `standard` | Her2k operation in the blas3 family. |
| `blas3Herk` | `alpha.blas3.herk` | `standard` | Herk operation in the blas3 family. |
| `blas3MatrixRankKUpdate` | `alpha.blas3.matrix-rank-kupdate` | `standard` | Matrix rank kupdate operation in the blas3 family. |
| `blas3Symm` | `alpha.blas3.symm` | `standard` | Symm operation in the blas3 family. |
| `blas3Syr2k` | `alpha.blas3.syr2k` | `standard` | Syr2k operation in the blas3 family. |
| `blas3Syrk` | `alpha.blas3.syrk` | `standard` | Syrk operation in the blas3 family. |
| `blas3TriangularSolveMatrix` | `alpha.blas3.triangular-solve-matrix` | `standard` | Triangular solve matrix operation in the blas3 family. |
| `blas3Trmm` | `alpha.blas3.trmm` | `standard` | Trmm operation in the blas3 family. |
| `blas3Trsm` | `alpha.blas3.trsm` | `standard` | Trsm operation in the blas3 family. |

## `convolution` (26)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `convolutionBlockSparseConv` | `alpha.convolution.block-sparse-conv` | `standard` | Block sparse conv operation in the convolution family. |
| `convolutionCausalConv` | `alpha.convolution.causal-conv` | `standard` | Causal conv operation in the convolution family. |
| `convolutionConv1d` | `alpha.convolution.conv1d` | `standard` | Conv1d operation in the convolution family. |
| `convolutionConv2d` | `alpha.convolution.conv2d` | `standard` | Conv2d operation in the convolution family. |
| `convolutionConv3d` | `alpha.convolution.conv3d` | `standard` | Conv3d operation in the convolution family. |
| `convolutionConvBackwardBias` | `alpha.convolution.conv-backward-bias` | `standard` | Conv backward bias operation in the convolution family. |
| `convolutionConvBackwardInput` | `alpha.convolution.conv-backward-input` | `standard` | Conv backward input operation in the convolution family. |
| `convolutionConvBackwardWeight` | `alpha.convolution.conv-backward-weight` | `standard` | Conv backward weight operation in the convolution family. |
| `convolutionConvTranspose1d` | `alpha.convolution.conv-transpose1d` | `standard` | Conv transpose1d operation in the convolution family. |
| `convolutionConvTranspose2d` | `alpha.convolution.conv-transpose2d` | `standard` | Conv transpose2d operation in the convolution family. |
| `convolutionConvTranspose3d` | `alpha.convolution.conv-transpose3d` | `standard` | Conv transpose3d operation in the convolution family. |
| `convolutionDeformableConv` | `alpha.convolution.deformable-conv` | `standard` | Deformable conv operation in the convolution family. |
| `convolutionDepthwiseConv` | `alpha.convolution.depthwise-conv` | `standard` | Depthwise conv operation in the convolution family. |
| `convolutionDilatedConv` | `alpha.convolution.dilated-conv` | `standard` | Dilated conv operation in the convolution family. |
| `convolutionDirectConv` | `alpha.convolution.direct-conv` | `standard` | Direct conv operation in the convolution family. |
| `convolutionDynamicConv` | `alpha.convolution.dynamic-conv` | `standard` | Dynamic conv operation in the convolution family. |
| `convolutionFftConv` | `alpha.convolution.fft-conv` | `standard` | Fft conv operation in the convolution family. |
| `convolutionGroupedConv` | `alpha.convolution.grouped-conv` | `standard` | Grouped conv operation in the convolution family. |
| `convolutionHyenaLongConv` | `alpha.convolution.hyena-long-conv` | `standard` | Hyena long conv operation in the convolution family. |
| `convolutionImplicitGemmConv` | `alpha.convolution.implicit-gemm-conv` | `standard` | Implicit gemm conv operation in the convolution family. |
| `convolutionMaskedConv` | `alpha.convolution.masked-conv` | `standard` | Masked conv operation in the convolution family. |
| `convolutionPointwiseConv` | `alpha.convolution.pointwise-conv` | `standard` | Pointwise conv operation in the convolution family. |
| `convolutionSeparableConv` | `alpha.convolution.separable-conv` | `standard` | Separable conv operation in the convolution family. |
| `convolutionStateSpaceConv` | `alpha.convolution.state-space-conv` | `standard` | State space conv operation in the convolution family. |
| `convolutionToeplitzConv` | `alpha.convolution.toeplitz-conv` | `standard` | Toeplitz conv operation in the convolution family. |
| `convolutionWinogradConv` | `alpha.convolution.winograd-conv` | `standard` | Winograd conv operation in the convolution family. |

## `distributed_collective` (29)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `distributedCollectiveAllGather` | `alpha.distributed_collective.all-gather` | `standard` | All gather operation in the distributed collective family. |
| `distributedCollectiveAllReduce` | `alpha.distributed_collective.all-reduce` | `standard` | All reduce operation in the distributed collective family. |
| `distributedCollectiveAllToAll` | `alpha.distributed_collective.all-to-all` | `standard` | All to all operation in the distributed collective family. |
| `distributedCollectiveAllToAllV` | `alpha.distributed_collective.all-to-all-v` | `standard` | All to all v operation in the distributed collective family. |
| `distributedCollectiveBarrier` | `alpha.distributed_collective.barrier` | `standard` | Barrier operation in the distributed collective family. |
| `distributedCollectiveBroadcast` | `alpha.distributed_collective.broadcast` | `standard` | Broadcast operation in the distributed collective family. |
| `distributedCollectiveExpertParallelCombine` | `alpha.distributed_collective.expert-parallel-combine` | `standard` | Expert parallel combine operation in the distributed collective family. |
| `distributedCollectiveExpertParallelDispatch` | `alpha.distributed_collective.expert-parallel-dispatch` | `standard` | Expert parallel dispatch operation in the distributed collective family. |
| `distributedCollectiveGather` | `alpha.distributed_collective.gather` | `standard` | Gather operation in the distributed collective family. |
| `distributedCollectiveGossipAverage` | `alpha.distributed_collective.gossip-average` | `standard` | Gossip average operation in the distributed collective family. |
| `distributedCollectiveHierarchicalAllReduce` | `alpha.distributed_collective.hierarchical-all-reduce` | `standard` | Hierarchical all reduce operation in the distributed collective family. |
| `distributedCollectiveNeighborAllGather` | `alpha.distributed_collective.neighbor-all-gather` | `standard` | Neighbor all gather operation in the distributed collective family. |
| `distributedCollectiveNeighborAllToAll` | `alpha.distributed_collective.neighbor-all-to-all` | `standard` | Neighbor all to all operation in the distributed collective family. |
| `distributedCollectiveParameterServerPull` | `alpha.distributed_collective.parameter-server-pull` | `standard` | Parameter server pull operation in the distributed collective family. |
| `distributedCollectiveParameterServerPush` | `alpha.distributed_collective.parameter-server-push` | `standard` | Parameter server push operation in the distributed collective family. |
| `distributedCollectivePipelineRecv` | `alpha.distributed_collective.pipeline-recv` | `standard` | Pipeline recv operation in the distributed collective family. |
| `distributedCollectivePipelineSend` | `alpha.distributed_collective.pipeline-send` | `standard` | Pipeline send operation in the distributed collective family. |
| `distributedCollectiveRecv` | `alpha.distributed_collective.recv` | `standard` | Recv operation in the distributed collective family. |
| `distributedCollectiveReduce` | `alpha.distributed_collective.reduce` | `standard` | Reduce operation in the distributed collective family. |
| `distributedCollectiveReduceScatter` | `alpha.distributed_collective.reduce-scatter` | `standard` | Reduce scatter operation in the distributed collective family. |
| `distributedCollectiveRemoteDistrictCall` | `alpha.distributed_collective.remote-district-call` | `research` | Remote district call operation in the distributed collective family. |
| `distributedCollectiveRemoteDistrictReturn` | `alpha.distributed_collective.remote-district-return` | `research` | Remote district return operation in the distributed collective family. |
| `distributedCollectiveRingAllReduce` | `alpha.distributed_collective.ring-all-reduce` | `standard` | Ring all reduce operation in the distributed collective family. |
| `distributedCollectiveScatter` | `alpha.distributed_collective.scatter` | `standard` | Scatter operation in the distributed collective family. |
| `distributedCollectiveSend` | `alpha.distributed_collective.send` | `standard` | Send operation in the distributed collective family. |
| `distributedCollectiveSendRecv` | `alpha.distributed_collective.send-recv` | `standard` | Send recv operation in the distributed collective family. |
| `distributedCollectiveSequenceParallelShard` | `alpha.distributed_collective.sequence-parallel-shard` | `standard` | Sequence parallel shard operation in the distributed collective family. |
| `distributedCollectiveTensorParallelShard` | `alpha.distributed_collective.tensor-parallel-shard` | `standard` | Tensor parallel shard operation in the distributed collective family. |
| `distributedCollectiveTreeAllReduce` | `alpha.distributed_collective.tree-all-reduce` | `standard` | Tree all reduce operation in the distributed collective family. |

## `gemm` (93)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `gemmGemm` | `alpha.gemm.gemm` | `standard` | Gemm operation in the gemm family. |
| `gemmGemmAmax` | `alpha.gemm.gemm-amax` | `standard` | Gemm amax operation in the gemm family. |
| `gemmGemmAnytimeBitplane` | `alpha.gemm.gemm-anytime-bitplane` | `research` | Progressively refines a matrix product from high-significance bitplanes and may stop once an error bound is satisfied. |
| `gemmGemmAutomaticPrecision` | `alpha.gemm.gemm-automatic-precision` | `standard` | Gemm automatic precision operation in the gemm family. |
| `gemmGemmBanded` | `alpha.gemm.gemm-banded` | `standard` | Gemm banded operation in the gemm family. |
| `gemmGemmBf16` | `alpha.gemm.gemm-bf16` | `standard` | Gemm bf16 operation in the gemm family. |
| `gemmGemmBias` | `alpha.gemm.gemm-bias` | `standard` | Gemm bias operation in the gemm family. |
| `gemmGemmBiasGelu` | `alpha.gemm.gemm-bias-gelu` | `standard` | Gemm bias gelu operation in the gemm family. |
| `gemmGemmBiasRelu` | `alpha.gemm.gemm-bias-relu` | `standard` | Gemm bias relu operation in the gemm family. |
| `gemmGemmBiasSilu` | `alpha.gemm.gemm-bias-silu` | `standard` | Gemm bias silu operation in the gemm family. |
| `gemmGemmBiasSwiGLU` | `alpha.gemm.gemm-bias-swi-glu` | `standard` | Gemm bias swi glu operation in the gemm family. |
| `gemmGemmBinary` | `alpha.gemm.gemm-binary` | `standard` | Gemm binary operation in the gemm family. |
| `gemmGemmBlockDiagonal` | `alpha.gemm.gemm-block-diagonal` | `standard` | Gemm block diagonal operation in the gemm family. |
| `gemmGemmBlockSparse` | `alpha.gemm.gemm-block-sparse` | `standard` | Gemm block sparse operation in the gemm family. |
| `gemmGemmBooleanOrAnd` | `alpha.gemm.gemm-boolean-or-and` | `standard` | Gemm boolean or and operation in the gemm family. |
| `gemmGemmBooleanXorAnd` | `alpha.gemm.gemm-boolean-xor-and` | `standard` | Gemm boolean xor and operation in the gemm family. |
| `gemmGemmBroadcastBatched` | `alpha.gemm.gemm-broadcast-batched` | `standard` | Gemm broadcast batched operation in the gemm family. |
| `gemmGemmButterfly` | `alpha.gemm.gemm-butterfly` | `standard` | Gemm butterfly operation in the gemm family. |
| `gemmGemmChecksumVerified` | `alpha.gemm.gemm-checksum-verified` | `research` | Gemm checksum verified operation in the gemm family. |
| `gemmGemmCirculant` | `alpha.gemm.gemm-circulant` | `standard` | Gemm circulant operation in the gemm family. |
| `gemmGemmCodedMicrobatch` | `alpha.gemm.gemm-coded-microbatch` | `research` | Gemm coded microbatch operation in the gemm family. |
| `gemmGemmComplex` | `alpha.gemm.gemm-complex` | `standard` | Gemm complex operation in the gemm family. |
| `gemmGemmConservedMoment` | `alpha.gemm.gemm-conserved-moment` | `research` | Gemm conserved moment operation in the gemm family. |
| `gemmGemmCooperative` | `alpha.gemm.gemm-cooperative` | `standard` | Gemm cooperative operation in the gemm family. |
| `gemmGemmCountercurrent` | `alpha.gemm.gemm-countercurrent` | `research` | Schedules forward and backward tile streams against a shared stationary weight tile. |
| `gemmGemmCountSketch` | `alpha.gemm.gemm-count-sketch` | `research` | Gemm count sketch operation in the gemm family. |
| `gemmGemmCovarianceAccumulating` | `alpha.gemm.gemm-covariance-accumulating` | `research` | Gemm covariance accumulating operation in the gemm family. |
| `gemmGemmDeferredWeightGradient` | `alpha.gemm.gemm-deferred-weight-gradient` | `research` | Banks activation/adjoint factors and forms one larger weight-gradient product after accumulation. |
| `gemmGemmDequantize` | `alpha.gemm.gemm-dequantize` | `standard` | Gemm dequantize operation in the gemm family. |
| `gemmGemmDiagonalPlusLowRank` | `alpha.gemm.gemm-diagonal-plus-low-rank` | `standard` | Gemm diagonal plus low rank operation in the gemm family. |
| `gemmGemmDropout` | `alpha.gemm.gemm-dropout` | `standard` | Gemm dropout operation in the gemm family. |
| `gemmGemmDualNumber` | `alpha.gemm.gemm-dual-number` | `speculative` | Gemm dual number operation in the gemm family. |
| `gemmGemmErrorFeedback` | `alpha.gemm.gemm-error-feedback` | `research` | Gemm error feedback operation in the gemm family. |
| `gemmGemmEx` | `alpha.gemm.gemm-ex` | `standard` | Gemm ex operation in the gemm family. |
| `gemmGemmFiniteField` | `alpha.gemm.gemm-finite-field` | `speculative` | Gemm finite field operation in the gemm family. |
| `gemmGemmFp16` | `alpha.gemm.gemm-fp16` | `standard` | Gemm fp16 operation in the gemm family. |
| `gemmGemmFusedEpilogue` | `alpha.gemm.gemm-fused-epilogue` | `standard` | Gemm fused epilogue operation in the gemm family. |
| `gemmGemmFusedPrologue` | `alpha.gemm.gemm-fused-prologue` | `standard` | Gemm fused prologue operation in the gemm family. |
| `gemmGemmFusedPrologueEpilogue` | `alpha.gemm.gemm-fused-prologue-epilogue` | `standard` | Gemm fused prologue epilogue operation in the gemm family. |
| `gemmGemmGatherA` | `alpha.gemm.gemm-gather-a` | `standard` | Gemm gather a operation in the gemm family. |
| `gemmGemmGatherB` | `alpha.gemm.gemm-gather-b` | `standard` | Gemm gather b operation in the gemm family. |
| `gemmGemmGradientSketch` | `alpha.gemm.gemm-gradient-sketch` | `research` | Gemm gradient sketch operation in the gemm family. |
| `gemmGemmGrouped` | `alpha.gemm.gemm-grouped` | `standard` | Gemm grouped operation in the gemm family. |
| `gemmGemmHadamardFactored` | `alpha.gemm.gemm-hadamard-factored` | `standard` | Gemm hadamard factored operation in the gemm family. |
| `gemmGemmHermitian` | `alpha.gemm.gemm-hermitian` | `standard` | Gemm hermitian operation in the gemm family. |
| `gemmGemmInnovationOnly` | `alpha.gemm.gemm-innovation-only` | `research` | Gemm innovation only operation in the gemm family. |
| `gemmGemmInt4` | `alpha.gemm.gemm-int4` | `standard` | Gemm int4 operation in the gemm family. |
| `gemmGemmInt8` | `alpha.gemm.gemm-int8` | `standard` | Gemm int8 operation in the gemm family. |
| `gemmGemmInterval` | `alpha.gemm.gemm-interval` | `speculative` | Gemm interval operation in the gemm family. |
| `gemmGemmKhatriRao` | `alpha.gemm.gemm-khatri-rao` | `standard` | Gemm khatri rao operation in the gemm family. |
| `gemmGemmKroneckerFactored` | `alpha.gemm.gemm-kronecker-factored` | `standard` | Gemm kronecker factored operation in the gemm family. |
| `gemmGemmLogSumExpPlus` | `alpha.gemm.gemm-log-sum-exp-plus` | `standard` | Gemm log sum exp plus operation in the gemm family. |
| `gemmGemmLowRank` | `alpha.gemm.gemm-low-rank` | `standard` | Gemm low rank operation in the gemm family. |
| `gemmGemmMasked` | `alpha.gemm.gemm-masked` | `standard` | Gemm masked operation in the gemm family. |
| `gemmGemmMaxPlus` | `alpha.gemm.gemm-max-plus` | `standard` | Gemm max plus operation in the gemm family. |
| `gemmGemmMaxTimes` | `alpha.gemm.gemm-max-times` | `standard` | Gemm max times operation in the gemm family. |
| `gemmGemmMinMax` | `alpha.gemm.gemm-min-max` | `standard` | Gemm min max operation in the gemm family. |
| `gemmGemmMinPlus` | `alpha.gemm.gemm-min-plus` | `standard` | Gemm min plus operation in the gemm family. |
| `gemmGemmMixedAccumulator` | `alpha.gemm.gemm-mixed-accumulator` | `standard` | Gemm mixed accumulator operation in the gemm family. |
| `gemmGemmMixedInput` | `alpha.gemm.gemm-mixed-input` | `standard` | Gemm mixed input operation in the gemm family. |
| `gemmGemmMuonUpdate` | `alpha.gemm.gemm-muon-update` | `research` | Gemm muon update operation in the gemm family. |
| `gemmGemmOptimizerConsumed` | `alpha.gemm.gemm-optimizer-consumed` | `research` | Produces the optimizer-transformed update or sufficient statistics directly instead of materializing a conventional gradient. |
| `gemmGemmPathCounting` | `alpha.gemm.gemm-path-counting` | `standard` | Gemm path counting operation in the gemm family. |
| `gemmGemmPersistent` | `alpha.gemm.gemm-persistent` | `standard` | Gemm persistent operation in the gemm family. |
| `gemmGemmPointerArrayBatched` | `alpha.gemm.gemm-pointer-array-batched` | `standard` | Gemm pointer array batched operation in the gemm family. |
| `gemmGemmPolarEpilogue` | `alpha.gemm.gemm-polar-epilogue` | `research` | Gemm polar epilogue operation in the gemm family. |
| `gemmGemmQuantize` | `alpha.gemm.gemm-quantize` | `standard` | Gemm quantize operation in the gemm family. |
| `gemmGemmQuaternion` | `alpha.gemm.gemm-quaternion` | `standard` | Gemm quaternion operation in the gemm family. |
| `gemmGemmRandomProjection` | `alpha.gemm.gemm-random-projection` | `standard` | Gemm random projection operation in the gemm family. |
| `gemmGemmReduceBatch` | `alpha.gemm.gemm-reduce-batch` | `standard` | Gemm reduce batch operation in the gemm family. |
| `gemmGemmRequantize` | `alpha.gemm.gemm-requantize` | `standard` | Gemm requantize operation in the gemm family. |
| `gemmGemmResidual` | `alpha.gemm.gemm-residual` | `standard` | Gemm residual operation in the gemm family. |
| `gemmGemmResidualNorm` | `alpha.gemm.gemm-residual-norm` | `standard` | Gemm residual norm operation in the gemm family. |
| `gemmGemmResidueCorrected` | `alpha.gemm.gemm-residue-corrected` | `research` | Computes a cheap bulk product and an exact or high-precision correction in selected subspaces. |
| `gemmGemmSampledOutput` | `alpha.gemm.gemm-sampled-output` | `standard` | Gemm sampled output operation in the gemm family. |
| `gemmGemmScatterD` | `alpha.gemm.gemm-scatter-d` | `standard` | Gemm scatter d operation in the gemm family. |
| `gemmGemmSemiring` | `alpha.gemm.gemm-semiring` | `standard` | Generalized matrix multiplication with independently supplied combine and reduction operators. |
| `gemmGemmSparse2of4` | `alpha.gemm.gemm-sparse2of4` | `standard` | Gemm sparse2of4 operation in the gemm family. |
| `gemmGemmSparseNm` | `alpha.gemm.gemm-sparse-nm` | `standard` | Gemm sparse nm operation in the gemm family. |
| `gemmGemmSplitKParallel` | `alpha.gemm.gemm-split-kparallel` | `standard` | Gemm split kparallel operation in the gemm family. |
| `gemmGemmSplitKSerial` | `alpha.gemm.gemm-split-kserial` | `standard` | Gemm split kserial operation in the gemm family. |
| `gemmGemmStreamK` | `alpha.gemm.gemm-stream-k` | `standard` | Gemm stream k operation in the gemm family. |
| `gemmGemmStridedBatched` | `alpha.gemm.gemm-strided-batched` | `standard` | Gemm strided batched operation in the gemm family. |
| `gemmGemmSymmetric` | `alpha.gemm.gemm-symmetric` | `standard` | Gemm symmetric operation in the gemm family. |
| `gemmGemmTensorSketch` | `alpha.gemm.gemm-tensor-sketch` | `research` | Gemm tensor sketch operation in the gemm family. |
| `gemmGemmTernary` | `alpha.gemm.gemm-ternary` | `standard` | Gemm ternary operation in the gemm family. |
| `gemmGemmTf32` | `alpha.gemm.gemm-tf32` | `standard` | Gemm tf32 operation in the gemm family. |
| `gemmGemmTileStream` | `alpha.gemm.gemm-tile-stream` | `standard` | Gemm tile stream operation in the gemm family. |
| `gemmGemmToeplitz` | `alpha.gemm.gemm-toeplitz` | `standard` | Gemm toeplitz operation in the gemm family. |
| `gemmGemmTriangular` | `alpha.gemm.gemm-triangular` | `standard` | Gemm triangular operation in the gemm family. |
| `gemmGemmVariableBatched` | `alpha.gemm.gemm-variable-batched` | `standard` | Gemm variable batched operation in the gemm family. |
| `gemmGemmViterbi` | `alpha.gemm.gemm-viterbi` | `standard` | Gemm viterbi operation in the gemm family. |
| `gemmGemmWarpSpecialized` | `alpha.gemm.gemm-warp-specialized` | `standard` | Gemm warp specialized operation in the gemm family. |

## `indexing` (49)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `indexingArgsort` | `alpha.indexing.argsort` | `standard` | Argsort operation in the indexing family. |
| `indexingArgwhere` | `alpha.indexing.argwhere` | `standard` | Argwhere operation in the indexing family. |
| `indexingBitonicSort` | `alpha.indexing.bitonic-sort` | `standard` | Bitonic sort operation in the indexing family. |
| `indexingBottomK` | `alpha.indexing.bottom-k` | `standard` | Bottom k operation in the indexing family. |
| `indexingBucketize` | `alpha.indexing.bucketize` | `standard` | Bucketize operation in the indexing family. |
| `indexingDynamicSlice` | `alpha.indexing.dynamic-slice` | `standard` | Dynamic slice operation in the indexing family. |
| `indexingDynamicUpdateSlice` | `alpha.indexing.dynamic-update-slice` | `standard` | Dynamic update slice operation in the indexing family. |
| `indexingEmbeddingBag` | `alpha.indexing.embedding-bag` | `standard` | Embedding bag operation in the indexing family. |
| `indexingEmbeddingLookup` | `alpha.indexing.embedding-lookup` | `standard` | Embedding lookup operation in the indexing family. |
| `indexingGather` | `alpha.indexing.gather` | `standard` | Gather operation in the indexing family. |
| `indexingGatherNd` | `alpha.indexing.gather-nd` | `standard` | Gather nd operation in the indexing family. |
| `indexingIndexAdd` | `alpha.indexing.index-add` | `standard` | Index add operation in the indexing family. |
| `indexingIndexPut` | `alpha.indexing.index-put` | `standard` | Index put operation in the indexing family. |
| `indexingIndexSelect` | `alpha.indexing.index-select` | `standard` | Index select operation in the indexing family. |
| `indexingKthValue` | `alpha.indexing.kth-value` | `standard` | Kth value operation in the indexing family. |
| `indexingLexSort` | `alpha.indexing.lex-sort` | `standard` | Lex sort operation in the indexing family. |
| `indexingMaskedFill` | `alpha.indexing.masked-fill` | `standard` | Masked fill operation in the indexing family. |
| `indexingMaskedScatter` | `alpha.indexing.masked-scatter` | `standard` | Masked scatter operation in the indexing family. |
| `indexingMaskedSelect` | `alpha.indexing.masked-select` | `standard` | Masked select operation in the indexing family. |
| `indexingMergeSorted` | `alpha.indexing.merge-sorted` | `standard` | Merge sorted operation in the indexing family. |
| `indexingNarrow` | `alpha.indexing.narrow` | `standard` | Narrow operation in the indexing family. |
| `indexingNonzero` | `alpha.indexing.nonzero` | `standard` | Nonzero operation in the indexing family. |
| `indexingNthElement` | `alpha.indexing.nth-element` | `standard` | Nth element operation in the indexing family. |
| `indexingOneHot` | `alpha.indexing.one-hot` | `standard` | One hot operation in the indexing family. |
| `indexingPartition` | `alpha.indexing.partition` | `standard` | Partition operation in the indexing family. |
| `indexingPut` | `alpha.indexing.put` | `standard` | Put operation in the indexing family. |
| `indexingPutAlongDim` | `alpha.indexing.put-along-dim` | `standard` | Put along dim operation in the indexing family. |
| `indexingRadixSort` | `alpha.indexing.radix-sort` | `standard` | Radix sort operation in the indexing family. |
| `indexingSampleWithoutReplacement` | `alpha.indexing.sample-without-replacement` | `standard` | Sample without replacement operation in the indexing family. |
| `indexingScatter` | `alpha.indexing.scatter` | `standard` | Scatter operation in the indexing family. |
| `indexingScatterAdd` | `alpha.indexing.scatter-add` | `standard` | Scatter add operation in the indexing family. |
| `indexingScatterMax` | `alpha.indexing.scatter-max` | `standard` | Scatter max operation in the indexing family. |
| `indexingScatterMean` | `alpha.indexing.scatter-mean` | `standard` | Scatter mean operation in the indexing family. |
| `indexingScatterMin` | `alpha.indexing.scatter-min` | `standard` | Scatter min operation in the indexing family. |
| `indexingScatterMul` | `alpha.indexing.scatter-mul` | `standard` | Scatter mul operation in the indexing family. |
| `indexingScatterReduce` | `alpha.indexing.scatter-reduce` | `standard` | Scatter reduce operation in the indexing family. |
| `indexingSearchSorted` | `alpha.indexing.search-sorted` | `standard` | Search sorted operation in the indexing family. |
| `indexingSegmentIds` | `alpha.indexing.segment-ids` | `standard` | Segment ids operation in the indexing family. |
| `indexingSelect` | `alpha.indexing.select` | `standard` | Select operation in the indexing family. |
| `indexingSlice` | `alpha.indexing.slice` | `standard` | Slice operation in the indexing family. |
| `indexingSliceUpdate` | `alpha.indexing.slice-update` | `standard` | Slice update operation in the indexing family. |
| `indexingSort` | `alpha.indexing.sort` | `standard` | Sort operation in the indexing family. |
| `indexingStableSort` | `alpha.indexing.stable-sort` | `standard` | Stable sort operation in the indexing family. |
| `indexingTake` | `alpha.indexing.take` | `standard` | Take operation in the indexing family. |
| `indexingTakeAlongDim` | `alpha.indexing.take-along-dim` | `standard` | Take along dim operation in the indexing family. |
| `indexingTopK` | `alpha.indexing.top-k` | `standard` | Top k operation in the indexing family. |
| `indexingUnique` | `alpha.indexing.unique` | `standard` | Unique operation in the indexing family. |
| `indexingUniqueConsecutive` | `alpha.indexing.unique-consecutive` | `standard` | Unique consecutive operation in the indexing family. |
| `indexingWhere` | `alpha.indexing.where` | `standard` | Where operation in the indexing family. |

## `memory_retrieval` (31)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `memoryRetrievalAssociativeMemoryRead` | `alpha.memory_retrieval.associative-memory-read` | `standard` | Associative memory read operation in the memory retrieval family. |
| `memoryRetrievalAssociativeMemoryWrite` | `alpha.memory_retrieval.associative-memory-write` | `standard` | Associative memory write operation in the memory retrieval family. |
| `memoryRetrievalContentAddressedRead` | `alpha.memory_retrieval.content-addressed-read` | `standard` | Content addressed read operation in the memory retrieval family. |
| `memoryRetrievalConversationPassportDecode` | `alpha.memory_retrieval.conversation-passport-decode` | `research` | Conversation passport decode operation in the memory retrieval family. |
| `memoryRetrievalConversationPassportEncode` | `alpha.memory_retrieval.conversation-passport-encode` | `research` | Conversation passport encode operation in the memory retrieval family. |
| `memoryRetrievalDeltaMemoryUpdate` | `alpha.memory_retrieval.delta-memory-update` | `standard` | Delta memory update operation in the memory retrieval family. |
| `memoryRetrievalEpisodicAppend` | `alpha.memory_retrieval.episodic-append` | `standard` | Episodic append operation in the memory retrieval family. |
| `memoryRetrievalEpisodicCompact` | `alpha.memory_retrieval.episodic-compact` | `standard` | Episodic compact operation in the memory retrieval family. |
| `memoryRetrievalEpisodicEvict` | `alpha.memory_retrieval.episodic-evict` | `standard` | Episodic evict operation in the memory retrieval family. |
| `memoryRetrievalEpisodicExactSpanRead` | `alpha.memory_retrieval.episodic-exact-span-read` | `standard` | Episodic exact span read operation in the memory retrieval family. |
| `memoryRetrievalEpisodicLookup` | `alpha.memory_retrieval.episodic-lookup` | `standard` | Episodic lookup operation in the memory retrieval family. |
| `memoryRetrievalEpisodicTopK` | `alpha.memory_retrieval.episodic-top-k` | `standard` | Episodic top k operation in the memory retrieval family. |
| `memoryRetrievalFastWeightMemoryRead` | `alpha.memory_retrieval.fast-weight-memory-read` | `standard` | Fast weight memory read operation in the memory retrieval family. |
| `memoryRetrievalFastWeightMemoryWrite` | `alpha.memory_retrieval.fast-weight-memory-write` | `standard` | Fast weight memory write operation in the memory retrieval family. |
| `memoryRetrievalHashTableLookup` | `alpha.memory_retrieval.hash-table-lookup` | `standard` | Hash table lookup operation in the memory retrieval family. |
| `memoryRetrievalInvertedFileSearch` | `alpha.memory_retrieval.inverted-file-search` | `standard` | Inverted file search operation in the memory retrieval family. |
| `memoryRetrievalKvMemoryRead` | `alpha.memory_retrieval.kv-memory-read` | `standard` | Kv memory read operation in the memory retrieval family. |
| `memoryRetrievalKvMemoryWrite` | `alpha.memory_retrieval.kv-memory-write` | `standard` | Kv memory write operation in the memory retrieval family. |
| `memoryRetrievalLocationAddressedRead` | `alpha.memory_retrieval.location-addressed-read` | `standard` | Location addressed read operation in the memory retrieval family. |
| `memoryRetrievalMemoryConfidenceUpdate` | `alpha.memory_retrieval.memory-confidence-update` | `standard` | Memory confidence update operation in the memory retrieval family. |
| `memoryRetrievalMemoryConflictDetect` | `alpha.memory_retrieval.memory-conflict-detect` | `standard` | Memory conflict detect operation in the memory retrieval family. |
| `memoryRetrievalMemoryProvenanceAttach` | `alpha.memory_retrieval.memory-provenance-attach` | `standard` | Memory provenance attach operation in the memory retrieval family. |
| `memoryRetrievalMemoryRestore` | `alpha.memory_retrieval.memory-restore` | `standard` | Memory restore operation in the memory retrieval family. |
| `memoryRetrievalMemorySnapshot` | `alpha.memory_retrieval.memory-snapshot` | `standard` | Memory snapshot operation in the memory retrieval family. |
| `memoryRetrievalProductQuantizedSearch` | `alpha.memory_retrieval.product-quantized-search` | `standard` | Product quantized search operation in the memory retrieval family. |
| `memoryRetrievalTypedBindingDelete` | `alpha.memory_retrieval.typed-binding-delete` | `standard` | Typed binding delete operation in the memory retrieval family. |
| `memoryRetrievalTypedBindingRead` | `alpha.memory_retrieval.typed-binding-read` | `standard` | Typed binding read operation in the memory retrieval family. |
| `memoryRetrievalTypedBindingResolve` | `alpha.memory_retrieval.typed-binding-resolve` | `standard` | Typed binding resolve operation in the memory retrieval family. |
| `memoryRetrievalTypedBindingSupersede` | `alpha.memory_retrieval.typed-binding-supersede` | `standard` | Typed binding supersede operation in the memory retrieval family. |
| `memoryRetrievalTypedBindingWrite` | `alpha.memory_retrieval.typed-binding-write` | `standard` | Typed binding write operation in the memory retrieval family. |
| `memoryRetrievalVectorIndexSearch` | `alpha.memory_retrieval.vector-index-search` | `standard` | Vector index search operation in the memory retrieval family. |

## `normalization` (22)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `normalizationBatchNorm` | `alpha.normalization.batch-norm` | `standard` | Batch norm operation in the normalization family. |
| `normalizationCosineNorm` | `alpha.normalization.cosine-norm` | `standard` | Cosine norm operation in the normalization family. |
| `normalizationDeepNorm` | `alpha.normalization.deep-norm` | `standard` | Deep norm operation in the normalization family. |
| `normalizationFusedAddNorm` | `alpha.normalization.fused-add-norm` | `standard` | Fused add norm operation in the normalization family. |
| `normalizationFusedNormLinear` | `alpha.normalization.fused-norm-linear` | `standard` | Fused norm linear operation in the normalization family. |
| `normalizationGroupNorm` | `alpha.normalization.group-norm` | `standard` | Group norm operation in the normalization family. |
| `normalizationInstanceNorm` | `alpha.normalization.instance-norm` | `standard` | Instance norm operation in the normalization family. |
| `normalizationLayerNorm` | `alpha.normalization.layer-norm` | `standard` | Layer norm operation in the normalization family. |
| `normalizationLocalResponseNorm` | `alpha.normalization.local-response-norm` | `standard` | Local response norm operation in the normalization family. |
| `normalizationMeanOnlyNorm` | `alpha.normalization.mean-only-norm` | `standard` | Mean only norm operation in the normalization family. |
| `normalizationNormBackward` | `alpha.normalization.norm-backward` | `standard` | Norm backward operation in the normalization family. |
| `normalizationOnlineNorm` | `alpha.normalization.online-norm` | `standard` | Online norm operation in the normalization family. |
| `normalizationPowerNorm` | `alpha.normalization.power-norm` | `standard` | Power norm operation in the normalization family. |
| `normalizationResidualLayerNorm` | `alpha.normalization.residual-layer-norm` | `standard` | Residual layer norm operation in the normalization family. |
| `normalizationResidualRmsNorm` | `alpha.normalization.residual-rms-norm` | `standard` | Residual rms norm operation in the normalization family. |
| `normalizationRmsNorm` | `alpha.normalization.rms-norm` | `standard` | Rms norm operation in the normalization family. |
| `normalizationScaleNorm` | `alpha.normalization.scale-norm` | `standard` | Scale norm operation in the normalization family. |
| `normalizationSpectralNorm` | `alpha.normalization.spectral-norm` | `standard` | Spectral norm operation in the normalization family. |
| `normalizationStreamingNorm` | `alpha.normalization.streaming-norm` | `standard` | Streaming norm operation in the normalization family. |
| `normalizationUnitNorm` | `alpha.normalization.unit-norm` | `standard` | Unit norm operation in the normalization family. |
| `normalizationWeightNorm` | `alpha.normalization.weight-norm` | `standard` | Weight norm operation in the normalization family. |
| `normalizationWelfordNorm` | `alpha.normalization.welford-norm` | `standard` | Welford norm operation in the normalization family. |

## `optimizer` (39)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `optimizerAdadelta` | `alpha.optimizer.adadelta` | `standard` | Adadelta operation in the optimizer family. |
| `optimizerAdafactor` | `alpha.optimizer.adafactor` | `standard` | Adafactor operation in the optimizer family. |
| `optimizerAdagrad` | `alpha.optimizer.adagrad` | `standard` | Adagrad operation in the optimizer family. |
| `optimizerAdam` | `alpha.optimizer.adam` | `standard` | Adam operation in the optimizer family. |
| `optimizerAdamW` | `alpha.optimizer.adam-w` | `standard` | Adam w operation in the optimizer family. |
| `optimizerAmsgrad` | `alpha.optimizer.amsgrad` | `standard` | Amsgrad operation in the optimizer family. |
| `optimizerBlockwiseOptimizerUpdate` | `alpha.optimizer.blockwise-optimizer-update` | `standard` | Blockwise optimizer update operation in the optimizer family. |
| `optimizerConservationProjectedUpdate` | `alpha.optimizer.conservation-projected-update` | `standard` | Conservation projected update operation in the optimizer family. |
| `optimizerDistributedShampoo` | `alpha.optimizer.distributed-shampoo` | `standard` | Distributed shampoo operation in the optimizer family. |
| `optimizerEkfac` | `alpha.optimizer.ekfac` | `standard` | Ekfac operation in the optimizer family. |
| `optimizerEma` | `alpha.optimizer.ema` | `standard` | Ema operation in the optimizer family. |
| `optimizerEventTriggeredOptimizerStep` | `alpha.optimizer.event-triggered-optimizer-step` | `standard` | Event triggered optimizer step operation in the optimizer family. |
| `optimizerGradientTrustRatio` | `alpha.optimizer.gradient-trust-ratio` | `standard` | Gradient trust ratio operation in the optimizer family. |
| `optimizerKfac` | `alpha.optimizer.kfac` | `standard` | Kfac operation in the optimizer family. |
| `optimizerLamb` | `alpha.optimizer.lamb` | `standard` | Lamb operation in the optimizer family. |
| `optimizerLars` | `alpha.optimizer.lars` | `standard` | Lars operation in the optimizer family. |
| `optimizerLearningRateSchedule` | `alpha.optimizer.learning-rate-schedule` | `standard` | Learning rate schedule operation in the optimizer family. |
| `optimizerLion` | `alpha.optimizer.lion` | `standard` | Lion operation in the optimizer family. |
| `optimizerLookahead` | `alpha.optimizer.lookahead` | `standard` | Lookahead operation in the optimizer family. |
| `optimizerLossScaleUpdate` | `alpha.optimizer.loss-scale-update` | `standard` | Loss scale update operation in the optimizer family. |
| `optimizerLowRankOptimizerUpdate` | `alpha.optimizer.low-rank-optimizer-update` | `standard` | Low rank optimizer update operation in the optimizer family. |
| `optimizerMuon` | `alpha.optimizer.muon` | `research` | Muon operation in the optimizer family. |
| `optimizerNadam` | `alpha.optimizer.nadam` | `standard` | Nadam operation in the optimizer family. |
| `optimizerNaturalGradient` | `alpha.optimizer.natural-gradient` | `standard` | Natural gradient operation in the optimizer family. |
| `optimizerNesterov` | `alpha.optimizer.nesterov` | `standard` | Nesterov operation in the optimizer family. |
| `optimizerNewtonSchulz` | `alpha.optimizer.newton-schulz` | `standard` | Newton schulz operation in the optimizer family. |
| `optimizerOptimizerStateOffload` | `alpha.optimizer.optimizer-state-offload` | `standard` | Optimizer state offload operation in the optimizer family. |
| `optimizerOptimizerStateQuantize` | `alpha.optimizer.optimizer-state-quantize` | `standard` | Optimizer state quantize operation in the optimizer family. |
| `optimizerOptimizerStateShard` | `alpha.optimizer.optimizer-state-shard` | `standard` | Optimizer state shard operation in the optimizer family. |
| `optimizerOrthogonalizedMomentum` | `alpha.optimizer.orthogonalized-momentum` | `standard` | Orthogonalized momentum operation in the optimizer family. |
| `optimizerPolarSigmaDeltaUpdate` | `alpha.optimizer.polar-sigma-delta-update` | `research` | Polar sigma delta update operation in the optimizer family. |
| `optimizerPolyakAverage` | `alpha.optimizer.polyak-average` | `standard` | Polyak average operation in the optimizer family. |
| `optimizerRadam` | `alpha.optimizer.radam` | `standard` | Radam operation in the optimizer family. |
| `optimizerRmsprop` | `alpha.optimizer.rmsprop` | `standard` | Rmsprop operation in the optimizer family. |
| `optimizerSgd` | `alpha.optimizer.sgd` | `standard` | Sgd operation in the optimizer family. |
| `optimizerSgdMomentum` | `alpha.optimizer.sgd-momentum` | `standard` | Sgd momentum operation in the optimizer family. |
| `optimizerShampoo` | `alpha.optimizer.shampoo` | `standard` | Shampoo operation in the optimizer family. |
| `optimizerSophia` | `alpha.optimizer.sophia` | `standard` | Sophia operation in the optimizer family. |
| `optimizerWeightDecay` | `alpha.optimizer.weight-decay` | `standard` | Weight decay operation in the optimizer family. |

## `pooling_resample` (19)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `poolingResampleAdaptiveAvgPool` | `alpha.pooling_resample.adaptive-avg-pool` | `standard` | Adaptive avg pool operation in the pooling resample family. |
| `poolingResampleAdaptiveMaxPool` | `alpha.pooling_resample.adaptive-max-pool` | `standard` | Adaptive max pool operation in the pooling resample family. |
| `poolingResampleAreaResize` | `alpha.pooling_resample.area-resize` | `standard` | Area resize operation in the pooling resample family. |
| `poolingResampleAvgPool` | `alpha.pooling_resample.avg-pool` | `standard` | Avg pool operation in the pooling resample family. |
| `poolingResampleBicubicResize` | `alpha.pooling_resample.bicubic-resize` | `standard` | Bicubic resize operation in the pooling resample family. |
| `poolingResampleBilinearResize` | `alpha.pooling_resample.bilinear-resize` | `standard` | Bilinear resize operation in the pooling resample family. |
| `poolingResampleFractionalMaxPool` | `alpha.pooling_resample.fractional-max-pool` | `standard` | Fractional max pool operation in the pooling resample family. |
| `poolingResampleGlobalAvgPool` | `alpha.pooling_resample.global-avg-pool` | `standard` | Global avg pool operation in the pooling resample family. |
| `poolingResampleGridSample` | `alpha.pooling_resample.grid-sample` | `standard` | Grid sample operation in the pooling resample family. |
| `poolingResampleLanczosResize` | `alpha.pooling_resample.lanczos-resize` | `standard` | Lanczos resize operation in the pooling resample family. |
| `poolingResampleLinearResize` | `alpha.pooling_resample.linear-resize` | `standard` | Linear resize operation in the pooling resample family. |
| `poolingResampleLpPool` | `alpha.pooling_resample.lp-pool` | `standard` | Lp pool operation in the pooling resample family. |
| `poolingResampleMaxPool` | `alpha.pooling_resample.max-pool` | `standard` | Max pool operation in the pooling resample family. |
| `poolingResampleMinPool` | `alpha.pooling_resample.min-pool` | `standard` | Min pool operation in the pooling resample family. |
| `poolingResampleNearestResize` | `alpha.pooling_resample.nearest-resize` | `standard` | Nearest resize operation in the pooling resample family. |
| `poolingResampleRoiAlign` | `alpha.pooling_resample.roi-align` | `standard` | Roi align operation in the pooling resample family. |
| `poolingResampleRoiPool` | `alpha.pooling_resample.roi-pool` | `standard` | Roi pool operation in the pooling resample family. |
| `poolingResampleStochasticPool` | `alpha.pooling_resample.stochastic-pool` | `standard` | Stochastic pool operation in the pooling resample family. |
| `poolingResampleUnpool` | `alpha.pooling_resample.unpool` | `standard` | Unpool operation in the pooling resample family. |

## `position_encoding` (15)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `positionEncodingAbsolutePositionEmbedding` | `alpha.position_encoding.absolute-position-embedding` | `standard` | Absolute position embedding operation in the position encoding family. |
| `positionEncodingAlibiBias` | `alpha.position_encoding.alibi-bias` | `standard` | Alibi bias operation in the position encoding family. |
| `positionEncodingDynamicNtkScaling` | `alpha.position_encoding.dynamic-ntk-scaling` | `standard` | Dynamic ntk scaling operation in the position encoding family. |
| `positionEncodingLearnedPhasePosition` | `alpha.position_encoding.learned-phase-position` | `standard` | Learned phase position operation in the position encoding family. |
| `positionEncodingLongRope` | `alpha.position_encoding.long-rope` | `standard` | Long rope operation in the position encoding family. |
| `positionEncodingPositionInterpolation` | `alpha.position_encoding.position-interpolation` | `standard` | Position interpolation operation in the position encoding family. |
| `positionEncodingRandomizedPosition` | `alpha.position_encoding.randomized-position` | `standard` | Randomized position operation in the position encoding family. |
| `positionEncodingRelativePositionBias` | `alpha.position_encoding.relative-position-bias` | `standard` | Relative position bias operation in the position encoding family. |
| `positionEncodingRotaryPosition` | `alpha.position_encoding.rotary-position` | `standard` | Rotary position operation in the position encoding family. |
| `positionEncodingRotaryPosition2d` | `alpha.position_encoding.rotary-position2d` | `standard` | Rotary position2d operation in the position encoding family. |
| `positionEncodingRotaryPositionInterleaved` | `alpha.position_encoding.rotary-position-interleaved` | `standard` | Rotary position interleaved operation in the position encoding family. |
| `positionEncodingSinusoidalPosition` | `alpha.position_encoding.sinusoidal-position` | `standard` | Sinusoidal position operation in the position encoding family. |
| `positionEncodingT5RelativeBias` | `alpha.position_encoding.t5-relative-bias` | `standard` | T5 relative bias operation in the position encoding family. |
| `positionEncodingXpos` | `alpha.position_encoding.xpos` | `standard` | Xpos operation in the position encoding family. |
| `positionEncodingYarn` | `alpha.position_encoding.yarn` | `standard` | Yarn operation in the position encoding family. |

## `probability_loss` (45)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `probabilityLossAdaptiveSoftmaxLoss` | `alpha.probability_loss.adaptive-softmax-loss` | `standard` | Adaptive softmax loss operation in the probability loss family. |
| `probabilityLossBinaryCrossEntropy` | `alpha.probability_loss.binary-cross-entropy` | `standard` | Binary cross entropy operation in the probability loss family. |
| `probabilityLossContrastiveLoss` | `alpha.probability_loss.contrastive-loss` | `standard` | Contrastive loss operation in the probability loss family. |
| `probabilityLossCosineEmbeddingLoss` | `alpha.probability_loss.cosine-embedding-loss` | `standard` | Cosine embedding loss operation in the probability loss family. |
| `probabilityLossCrfBackward` | `alpha.probability_loss.crf-backward` | `standard` | Crf backward operation in the probability loss family. |
| `probabilityLossCrfForward` | `alpha.probability_loss.crf-forward` | `standard` | Crf forward operation in the probability loss family. |
| `probabilityLossCrossEntropy` | `alpha.probability_loss.cross-entropy` | `standard` | Cross entropy operation in the probability loss family. |
| `probabilityLossCtcLoss` | `alpha.probability_loss.ctc-loss` | `standard` | Ctc loss operation in the probability loss family. |
| `probabilityLossDpoLoss` | `alpha.probability_loss.dpo-loss` | `standard` | Dpo loss operation in the probability loss family. |
| `probabilityLossEntmax15` | `alpha.probability_loss.entmax15` | `standard` | Entmax15 operation in the probability loss family. |
| `probabilityLossFocalLoss` | `alpha.probability_loss.focal-loss` | `standard` | Focal loss operation in the probability loss family. |
| `probabilityLossGumbelSoftmax` | `alpha.probability_loss.gumbel-softmax` | `standard` | Gumbel softmax operation in the probability loss family. |
| `probabilityLossHaltingCalibrationLoss` | `alpha.probability_loss.halting-calibration-loss` | `standard` | Halting calibration loss operation in the probability loss family. |
| `probabilityLossHellingerDistance` | `alpha.probability_loss.hellinger-distance` | `standard` | Hellinger distance operation in the probability loss family. |
| `probabilityLossHierarchicalSoftmaxLoss` | `alpha.probability_loss.hierarchical-softmax-loss` | `standard` | Hierarchical softmax loss operation in the probability loss family. |
| `probabilityLossHuberLoss` | `alpha.probability_loss.huber-loss` | `standard` | Huber loss operation in the probability loss family. |
| `probabilityLossInfoNceLoss` | `alpha.probability_loss.info-nce-loss` | `standard` | Info nce loss operation in the probability loss family. |
| `probabilityLossIpoLoss` | `alpha.probability_loss.ipo-loss` | `standard` | Ipo loss operation in the probability loss family. |
| `probabilityLossJsDivergence` | `alpha.probability_loss.js-divergence` | `standard` | Js divergence operation in the probability loss family. |
| `probabilityLossKlDivergence` | `alpha.probability_loss.kl-divergence` | `standard` | Kl divergence operation in the probability loss family. |
| `probabilityLossKtoLoss` | `alpha.probability_loss.kto-loss` | `standard` | Kto loss operation in the probability loss family. |
| `probabilityLossLabelSmoothedCrossEntropy` | `alpha.probability_loss.label-smoothed-cross-entropy` | `standard` | Label smoothed cross entropy operation in the probability loss family. |
| `probabilityLossLoadBalanceLoss` | `alpha.probability_loss.load-balance-loss` | `standard` | Load balance loss operation in the probability loss family. |
| `probabilityLossLogSoftmax` | `alpha.probability_loss.log-softmax` | `standard` | Log softmax operation in the probability loss family. |
| `probabilityLossMaeLoss` | `alpha.probability_loss.mae-loss` | `standard` | Mae loss operation in the probability loss family. |
| `probabilityLossMarginRankingLoss` | `alpha.probability_loss.margin-ranking-loss` | `standard` | Margin ranking loss operation in the probability loss family. |
| `probabilityLossMaskedSoftmax` | `alpha.probability_loss.masked-softmax` | `standard` | Masked softmax operation in the probability loss family. |
| `probabilityLossMemoryReconstructionLoss` | `alpha.probability_loss.memory-reconstruction-loss` | `standard` | Memory reconstruction loss operation in the probability loss family. |
| `probabilityLossMseLoss` | `alpha.probability_loss.mse-loss` | `standard` | Mse loss operation in the probability loss family. |
| `probabilityLossNoiseContrastiveEstimation` | `alpha.probability_loss.noise-contrastive-estimation` | `standard` | Noise contrastive estimation operation in the probability loss family. |
| `probabilityLossOnlineSoftmax` | `alpha.probability_loss.online-softmax` | `standard` | Online softmax operation in the probability loss family. |
| `probabilityLossOrpoLoss` | `alpha.probability_loss.orpo-loss` | `standard` | Orpo loss operation in the probability loss family. |
| `probabilityLossPreferenceLoss` | `alpha.probability_loss.preference-loss` | `standard` | Preference loss operation in the probability loss family. |
| `probabilityLossRenyiDivergence` | `alpha.probability_loss.renyi-divergence` | `standard` | Renyi divergence operation in the probability loss family. |
| `probabilityLossReverseKl` | `alpha.probability_loss.reverse-kl` | `standard` | Reverse kl operation in the probability loss family. |
| `probabilityLossRewardModelLoss` | `alpha.probability_loss.reward-model-loss` | `standard` | Reward model loss operation in the probability loss family. |
| `probabilityLossRouteConsistencyLoss` | `alpha.probability_loss.route-consistency-loss` | `standard` | Route consistency loss operation in the probability loss family. |
| `probabilityLossSampledSoftmaxLoss` | `alpha.probability_loss.sampled-softmax-loss` | `standard` | Sampled softmax loss operation in the probability loss family. |
| `probabilityLossSequenceRiskLoss` | `alpha.probability_loss.sequence-risk-loss` | `standard` | Sequence risk loss operation in the probability loss family. |
| `probabilityLossSmoothL1Loss` | `alpha.probability_loss.smooth-l1-loss` | `standard` | Smooth l1 loss operation in the probability loss family. |
| `probabilityLossSoftmax` | `alpha.probability_loss.softmax` | `standard` | Softmax operation in the probability loss family. |
| `probabilityLossSparsemax` | `alpha.probability_loss.sparsemax` | `standard` | Sparsemax operation in the probability loss family. |
| `probabilityLossTemperatureSoftmax` | `alpha.probability_loss.temperature-softmax` | `standard` | Temperature softmax operation in the probability loss family. |
| `probabilityLossTripletLoss` | `alpha.probability_loss.triplet-loss` | `standard` | Triplet loss operation in the probability loss family. |
| `probabilityLossWassersteinLoss` | `alpha.probability_loss.wasserstein-loss` | `standard` | Wasserstein loss operation in the probability loss family. |

## `quantization` (52)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `quantizationAsymmetricQuantize` | `alpha.quantization.asymmetric-quantize` | `standard` | Asymmetric quantize operation in the quantization family. |
| `quantizationBinaryCodeQuantize` | `alpha.quantization.binary-code-quantize` | `standard` | Binary code quantize operation in the quantization family. |
| `quantizationBlockScaleQuantize` | `alpha.quantization.block-scale-quantize` | `standard` | Block scale quantize operation in the quantization family. |
| `quantizationCodebookQuantize` | `alpha.quantization.codebook-quantize` | `standard` | Codebook quantize operation in the quantization family. |
| `quantizationComputeAmax` | `alpha.quantization.compute-amax` | `standard` | Compute amax operation in the quantization family. |
| `quantizationComputeScale` | `alpha.quantization.compute-scale` | `standard` | Compute scale operation in the quantization family. |
| `quantizationComputeZeroPoint` | `alpha.quantization.compute-zero-point` | `standard` | Compute zero point operation in the quantization family. |
| `quantizationDequantize` | `alpha.quantization.dequantize` | `standard` | Dequantize operation in the quantization family. |
| `quantizationDynamicQuantize` | `alpha.quantization.dynamic-quantize` | `standard` | Dynamic quantize operation in the quantization family. |
| `quantizationErrorFeedbackQuantize` | `alpha.quantization.error-feedback-quantize` | `research` | Error feedback quantize operation in the quantization family. |
| `quantizationFakeQuantize` | `alpha.quantization.fake-quantize` | `standard` | Fake quantize operation in the quantization family. |
| `quantizationHadamardQuantize` | `alpha.quantization.hadamard-quantize` | `standard` | Hadamard quantize operation in the quantization family. |
| `quantizationMicroscaleQuantize` | `alpha.quantization.microscale-quantize` | `standard` | Microscale quantize operation in the quantization family. |
| `quantizationOutlierRestore` | `alpha.quantization.outlier-restore` | `standard` | Outlier restore operation in the quantization family. |
| `quantizationOutlierSplit` | `alpha.quantization.outlier-split` | `standard` | Outlier split operation in the quantization family. |
| `quantizationPackBinary` | `alpha.quantization.pack-binary` | `standard` | Pack binary operation in the quantization family. |
| `quantizationPackFp8` | `alpha.quantization.pack-fp8` | `standard` | Pack fp8 operation in the quantization family. |
| `quantizationPackInt2` | `alpha.quantization.pack-int2` | `standard` | Pack int2 operation in the quantization family. |
| `quantizationPackInt4` | `alpha.quantization.pack-int4` | `standard` | Pack int4 operation in the quantization family. |
| `quantizationPackInt8` | `alpha.quantization.pack-int8` | `standard` | Pack int8 operation in the quantization family. |
| `quantizationPackNf4` | `alpha.quantization.pack-nf4` | `standard` | Pack nf4 operation in the quantization family. |
| `quantizationPackTernary` | `alpha.quantization.pack-ternary` | `standard` | Pack ternary operation in the quantization family. |
| `quantizationPerChannelQuantize` | `alpha.quantization.per-channel-quantize` | `standard` | Per channel quantize operation in the quantization family. |
| `quantizationPerColumnQuantize` | `alpha.quantization.per-column-quantize` | `standard` | Per column quantize operation in the quantization family. |
| `quantizationPerGroupQuantize` | `alpha.quantization.per-group-quantize` | `standard` | Per group quantize operation in the quantization family. |
| `quantizationPerRowQuantize` | `alpha.quantization.per-row-quantize` | `standard` | Per row quantize operation in the quantization family. |
| `quantizationPerTensorQuantize` | `alpha.quantization.per-tensor-quantize` | `standard` | Per tensor quantize operation in the quantization family. |
| `quantizationPerTokenQuantize` | `alpha.quantization.per-token-quantize` | `standard` | Per token quantize operation in the quantization family. |
| `quantizationProductQuantize` | `alpha.quantization.product-quantize` | `standard` | Product quantize operation in the quantization family. |
| `quantizationQuantizationCalibration` | `alpha.quantization.quantization-calibration` | `standard` | Quantization calibration operation in the quantization family. |
| `quantizationQuantizationErrorMetric` | `alpha.quantization.quantization-error-metric` | `standard` | Quantization error metric operation in the quantization family. |
| `quantizationQuantize` | `alpha.quantization.quantize` | `standard` | Quantize operation in the quantization family. |
| `quantizationRequantize` | `alpha.quantization.requantize` | `standard` | Requantize operation in the quantization family. |
| `quantizationResidualQuantize` | `alpha.quantization.residual-quantize` | `standard` | Residual quantize operation in the quantization family. |
| `quantizationRotationQuantize` | `alpha.quantization.rotation-quantize` | `standard` | Rotation quantize operation in the quantization family. |
| `quantizationRoundToNearestEven` | `alpha.quantization.round-to-nearest-even` | `standard` | Round to nearest even operation in the quantization family. |
| `quantizationRoundTowardNegative` | `alpha.quantization.round-toward-negative` | `standard` | Round toward negative operation in the quantization family. |
| `quantizationRoundTowardPositive` | `alpha.quantization.round-toward-positive` | `standard` | Round toward positive operation in the quantization family. |
| `quantizationRoundTowardZero` | `alpha.quantization.round-toward-zero` | `standard` | Round toward zero operation in the quantization family. |
| `quantizationSigmaDeltaQuantize` | `alpha.quantization.sigma-delta-quantize` | `research` | Sigma delta quantize operation in the quantization family. |
| `quantizationSmoothQuant` | `alpha.quantization.smooth-quant` | `standard` | Smooth quant operation in the quantization family. |
| `quantizationStaticQuantize` | `alpha.quantization.static-quantize` | `standard` | Static quantize operation in the quantization family. |
| `quantizationStochasticRound` | `alpha.quantization.stochastic-round` | `standard` | Stochastic round operation in the quantization family. |
| `quantizationSymmetricQuantize` | `alpha.quantization.symmetric-quantize` | `standard` | Symmetric quantize operation in the quantization family. |
| `quantizationUnpackBinary` | `alpha.quantization.unpack-binary` | `standard` | Unpack binary operation in the quantization family. |
| `quantizationUnpackFp8` | `alpha.quantization.unpack-fp8` | `standard` | Unpack fp8 operation in the quantization family. |
| `quantizationUnpackInt2` | `alpha.quantization.unpack-int2` | `standard` | Unpack int2 operation in the quantization family. |
| `quantizationUnpackInt4` | `alpha.quantization.unpack-int4` | `standard` | Unpack int4 operation in the quantization family. |
| `quantizationUnpackInt8` | `alpha.quantization.unpack-int8` | `standard` | Unpack int8 operation in the quantization family. |
| `quantizationUnpackNf4` | `alpha.quantization.unpack-nf4` | `standard` | Unpack nf4 operation in the quantization family. |
| `quantizationUnpackTernary` | `alpha.quantization.unpack-ternary` | `standard` | Unpack ternary operation in the quantization family. |
| `quantizationVectorQuantize` | `alpha.quantization.vector-quantize` | `standard` | Vector quantize operation in the quantization family. |

## `random_sampling` (39)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `randomSamplingBeamSearchStep` | `alpha.random_sampling.beam-search-step` | `standard` | Beam search step operation in the random sampling family. |
| `randomSamplingBernoulliRandom` | `alpha.random_sampling.bernoulli-random` | `standard` | Bernoulli random operation in the random sampling family. |
| `randomSamplingBetaRandom` | `alpha.random_sampling.beta-random` | `standard` | Beta random operation in the random sampling family. |
| `randomSamplingBinomialRandom` | `alpha.random_sampling.binomial-random` | `standard` | Binomial random operation in the random sampling family. |
| `randomSamplingCategoricalRandom` | `alpha.random_sampling.categorical-random` | `standard` | Categorical random operation in the random sampling family. |
| `randomSamplingContrastiveSearchStep` | `alpha.random_sampling.contrastive-search-step` | `standard` | Contrastive search step operation in the random sampling family. |
| `randomSamplingDirichletRandom` | `alpha.random_sampling.dirichlet-random` | `standard` | Dirichlet random operation in the random sampling family. |
| `randomSamplingDiverseBeamStep` | `alpha.random_sampling.diverse-beam-step` | `standard` | Diverse beam step operation in the random sampling family. |
| `randomSamplingDropoutMask` | `alpha.random_sampling.dropout-mask` | `standard` | Dropout mask operation in the random sampling family. |
| `randomSamplingEpsilonSample` | `alpha.random_sampling.epsilon-sample` | `standard` | Epsilon sample operation in the random sampling family. |
| `randomSamplingEtaSample` | `alpha.random_sampling.eta-sample` | `standard` | Eta sample operation in the random sampling family. |
| `randomSamplingExponentialRandom` | `alpha.random_sampling.exponential-random` | `standard` | Exponential random operation in the random sampling family. |
| `randomSamplingGammaRandom` | `alpha.random_sampling.gamma-random` | `standard` | Gamma random operation in the random sampling family. |
| `randomSamplingGumbelRandom` | `alpha.random_sampling.gumbel-random` | `standard` | Gumbel random operation in the random sampling family. |
| `randomSamplingHaltonRandom` | `alpha.random_sampling.halton-random` | `standard` | Halton random operation in the random sampling family. |
| `randomSamplingImportanceSample` | `alpha.random_sampling.importance-sample` | `standard` | Importance sample operation in the random sampling family. |
| `randomSamplingLangevinStep` | `alpha.random_sampling.langevin-step` | `standard` | Langevin step operation in the random sampling family. |
| `randomSamplingLogNormalRandom` | `alpha.random_sampling.log-normal-random` | `standard` | Log normal random operation in the random sampling family. |
| `randomSamplingMetropolisHastingsStep` | `alpha.random_sampling.metropolis-hastings-step` | `standard` | Metropolis hastings step operation in the random sampling family. |
| `randomSamplingMinPSample` | `alpha.random_sampling.min-psample` | `standard` | Min psample operation in the random sampling family. |
| `randomSamplingMirostatSample` | `alpha.random_sampling.mirostat-sample` | `standard` | Mirostat sample operation in the random sampling family. |
| `randomSamplingMultinomialRandom` | `alpha.random_sampling.multinomial-random` | `standard` | Multinomial random operation in the random sampling family. |
| `randomSamplingNormalRandom` | `alpha.random_sampling.normal-random` | `standard` | Normal random operation in the random sampling family. |
| `randomSamplingPcgRandom` | `alpha.random_sampling.pcg-random` | `standard` | Pcg random operation in the random sampling family. |
| `randomSamplingPhiloxRandom` | `alpha.random_sampling.philox-random` | `standard` | Philox random operation in the random sampling family. |
| `randomSamplingPoissonRandom` | `alpha.random_sampling.poisson-random` | `standard` | Poisson random operation in the random sampling family. |
| `randomSamplingRandomPermutation` | `alpha.random_sampling.random-permutation` | `standard` | Random permutation operation in the random sampling family. |
| `randomSamplingRejectionSample` | `alpha.random_sampling.rejection-sample` | `standard` | Rejection sample operation in the random sampling family. |
| `randomSamplingScrambledSobolRandom` | `alpha.random_sampling.scrambled-sobol-random` | `standard` | Scrambled sobol random operation in the random sampling family. |
| `randomSamplingSobolRandom` | `alpha.random_sampling.sobol-random` | `standard` | Sobol random operation in the random sampling family. |
| `randomSamplingSpeculativeAcceptReject` | `alpha.random_sampling.speculative-accept-reject` | `standard` | Speculative accept reject operation in the random sampling family. |
| `randomSamplingStochasticDepthMask` | `alpha.random_sampling.stochastic-depth-mask` | `standard` | Stochastic depth mask operation in the random sampling family. |
| `randomSamplingThreefryRandom` | `alpha.random_sampling.threefry-random` | `standard` | Threefry random operation in the random sampling family. |
| `randomSamplingTopKSample` | `alpha.random_sampling.top-ksample` | `standard` | Top ksample operation in the random sampling family. |
| `randomSamplingTopPSample` | `alpha.random_sampling.top-psample` | `standard` | Top psample operation in the random sampling family. |
| `randomSamplingTypicalSample` | `alpha.random_sampling.typical-sample` | `standard` | Typical sample operation in the random sampling family. |
| `randomSamplingUniformRandom` | `alpha.random_sampling.uniform-random` | `standard` | Uniform random operation in the random sampling family. |
| `randomSamplingXorshiftRandom` | `alpha.random_sampling.xorshift-random` | `standard` | Xorshift random operation in the random sampling family. |
| `randomSamplingXorwowRandom` | `alpha.random_sampling.xorwow-random` | `standard` | Xorwow random operation in the random sampling family. |

## `reduction` (43)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `reductionAll` | `alpha.reduction.all` | `standard` | All operation in the reduction family. |
| `reductionAmax` | `alpha.reduction.amax` | `standard` | Amax operation in the reduction family. |
| `reductionAminmax` | `alpha.reduction.aminmax` | `standard` | Aminmax operation in the reduction family. |
| `reductionAny` | `alpha.reduction.any` | `standard` | Any operation in the reduction family. |
| `reductionArgMax` | `alpha.reduction.arg-max` | `standard` | Arg max operation in the reduction family. |
| `reductionArgMin` | `alpha.reduction.arg-min` | `standard` | Arg min operation in the reduction family. |
| `reductionCompensatedSum` | `alpha.reduction.compensated-sum` | `standard` | Compensated sum operation in the reduction family. |
| `reductionCountNonzero` | `alpha.reduction.count-nonzero` | `standard` | Count nonzero operation in the reduction family. |
| `reductionDeterministicReduce` | `alpha.reduction.deterministic-reduce` | `standard` | Deterministic reduce operation in the reduction family. |
| `reductionFrobeniusNorm` | `alpha.reduction.frobenius-norm` | `standard` | Frobenius norm operation in the reduction family. |
| `reductionKahanSum` | `alpha.reduction.kahan-sum` | `standard` | Kahan sum operation in the reduction family. |
| `reductionL0Norm` | `alpha.reduction.l0-norm` | `standard` | L0 norm operation in the reduction family. |
| `reductionL1Norm` | `alpha.reduction.l1-norm` | `standard` | L1 norm operation in the reduction family. |
| `reductionL2Norm` | `alpha.reduction.l2-norm` | `standard` | L2 norm operation in the reduction family. |
| `reductionLogSumExp` | `alpha.reduction.log-sum-exp` | `standard` | Log sum exp operation in the reduction family. |
| `reductionLpNorm` | `alpha.reduction.lp-norm` | `standard` | Lp norm operation in the reduction family. |
| `reductionMaskedReduce` | `alpha.reduction.masked-reduce` | `standard` | Masked reduce operation in the reduction family. |
| `reductionMax` | `alpha.reduction.max` | `standard` | Max operation in the reduction family. |
| `reductionMaxWithIndex` | `alpha.reduction.max-with-index` | `standard` | Max with index operation in the reduction family. |
| `reductionMean` | `alpha.reduction.mean` | `standard` | Mean operation in the reduction family. |
| `reductionMin` | `alpha.reduction.min` | `standard` | Min operation in the reduction family. |
| `reductionMinWithIndex` | `alpha.reduction.min-with-index` | `standard` | Min with index operation in the reduction family. |
| `reductionMoments` | `alpha.reduction.moments` | `standard` | Moments operation in the reduction family. |
| `reductionNeumaierSum` | `alpha.reduction.neumaier-sum` | `standard` | Neumaier sum operation in the reduction family. |
| `reductionNuclearNorm` | `alpha.reduction.nuclear-norm` | `standard` | Nuclear norm operation in the reduction family. |
| `reductionPairwiseSum` | `alpha.reduction.pairwise-sum` | `standard` | Pairwise sum operation in the reduction family. |
| `reductionPartialReduce` | `alpha.reduction.partial-reduce` | `standard` | Partial reduce operation in the reduction family. |
| `reductionProduct` | `alpha.reduction.product` | `standard` | Product operation in the reduction family. |
| `reductionRaggedReduce` | `alpha.reduction.ragged-reduce` | `standard` | Ragged reduce operation in the reduction family. |
| `reductionReduceByKey` | `alpha.reduction.reduce-by-key` | `standard` | Reduce by key operation in the reduction family. |
| `reductionSegmentedMax` | `alpha.reduction.segmented-max` | `standard` | Segmented max operation in the reduction family. |
| `reductionSegmentedMean` | `alpha.reduction.segmented-mean` | `standard` | Segmented mean operation in the reduction family. |
| `reductionSegmentedMin` | `alpha.reduction.segmented-min` | `standard` | Segmented min operation in the reduction family. |
| `reductionSegmentedSum` | `alpha.reduction.segmented-sum` | `standard` | Segmented sum operation in the reduction family. |
| `reductionSoftmaxStatistics` | `alpha.reduction.softmax-statistics` | `standard` | Softmax statistics operation in the reduction family. |
| `reductionStandardDeviation` | `alpha.reduction.standard-deviation` | `standard` | Standard deviation operation in the reduction family. |
| `reductionSum` | `alpha.reduction.sum` | `standard` | Sum operation in the reduction family. |
| `reductionSumAbs` | `alpha.reduction.sum-abs` | `standard` | Sum abs operation in the reduction family. |
| `reductionSumSquares` | `alpha.reduction.sum-squares` | `standard` | Sum squares operation in the reduction family. |
| `reductionSuperaccumulatorSum` | `alpha.reduction.superaccumulator-sum` | `standard` | Superaccumulator sum operation in the reduction family. |
| `reductionTreeReduce` | `alpha.reduction.tree-reduce` | `standard` | Tree reduce operation in the reduction family. |
| `reductionVariance` | `alpha.reduction.variance` | `standard` | Variance operation in the reduction family. |
| `reductionWelfordMoments` | `alpha.reduction.welford-moments` | `standard` | Welford moments operation in the reduction family. |

## `routing_moe` (25)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `routingMoeCapacityRouter` | `alpha.routing_moe.capacity-router` | `standard` | Capacity router operation in the routing moe family. |
| `routingMoeCombineExpertOutputs` | `alpha.routing_moe.combine-expert-outputs` | `standard` | Combine expert outputs operation in the routing moe family. |
| `routingMoeDispatchToExperts` | `alpha.routing_moe.dispatch-to-experts` | `standard` | Dispatch to experts operation in the routing moe family. |
| `routingMoeExpertArchive` | `alpha.routing_moe.expert-archive` | `standard` | Expert archive operation in the routing moe family. |
| `routingMoeExpertAudit` | `alpha.routing_moe.expert-audit` | `standard` | Expert audit operation in the routing moe family. |
| `routingMoeExpertChoiceDispatch` | `alpha.routing_moe.expert-choice-dispatch` | `standard` | Expert choice dispatch operation in the routing moe family. |
| `routingMoeExpertChoiceRouter` | `alpha.routing_moe.expert-choice-router` | `standard` | Expert choice router operation in the routing moe family. |
| `routingMoeExpertDistill` | `alpha.routing_moe.expert-distill` | `standard` | Expert distill operation in the routing moe family. |
| `routingMoeExpertEvict` | `alpha.routing_moe.expert-evict` | `standard` | Expert evict operation in the routing moe family. |
| `routingMoeExpertLoadBalance` | `alpha.routing_moe.expert-load-balance` | `standard` | Expert load balance operation in the routing moe family. |
| `routingMoeExpertMerge` | `alpha.routing_moe.expert-merge` | `research` | Expert merge operation in the routing moe family. |
| `routingMoeExpertMitosis` | `alpha.routing_moe.expert-mitosis` | `research` | Expert mitosis operation in the routing moe family. |
| `routingMoeExpertPrefetch` | `alpha.routing_moe.expert-prefetch` | `standard` | Expert prefetch operation in the routing moe family. |
| `routingMoeExpertReplicaSelect` | `alpha.routing_moe.expert-replica-select` | `standard` | Expert replica select operation in the routing moe family. |
| `routingMoeGroupedExpertGemm` | `alpha.routing_moe.grouped-expert-gemm` | `standard` | Grouped expert gemm operation in the routing moe family. |
| `routingMoeHashRouter` | `alpha.routing_moe.hash-router` | `standard` | Hash router operation in the routing moe family. |
| `routingMoeHierarchicalRouter` | `alpha.routing_moe.hierarchical-router` | `standard` | Hierarchical router operation in the routing moe family. |
| `routingMoeRouterDiversity` | `alpha.routing_moe.router-diversity` | `standard` | Router diversity operation in the routing moe family. |
| `routingMoeRouterEntropy` | `alpha.routing_moe.router-entropy` | `standard` | Router entropy operation in the routing moe family. |
| `routingMoeRouterLocalityCost` | `alpha.routing_moe.router-locality-cost` | `standard` | Router locality cost operation in the routing moe family. |
| `routingMoeRouterZLoss` | `alpha.routing_moe.router-zloss` | `standard` | Router zloss operation in the routing moe family. |
| `routingMoeSinkhornRouter` | `alpha.routing_moe.sinkhorn-router` | `standard` | Sinkhorn router operation in the routing moe family. |
| `routingMoeSoftRouter` | `alpha.routing_moe.soft-router` | `standard` | Soft router operation in the routing moe family. |
| `routingMoeTokenChoiceDispatch` | `alpha.routing_moe.token-choice-dispatch` | `standard` | Token choice dispatch operation in the routing moe family. |
| `routingMoeTopKRouter` | `alpha.routing_moe.top-krouter` | `standard` | Top krouter operation in the routing moe family. |

## `scan` (25)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `scanAffineRecurrenceScan` | `alpha.scan.affine-recurrence-scan` | `standard` | Affine recurrence scan operation in the scan family. |
| `scanAssociativeScan` | `alpha.scan.associative-scan` | `standard` | Associative scan operation in the scan family. |
| `scanBidirectionalScan` | `alpha.scan.bidirectional-scan` | `standard` | Bidirectional scan operation in the scan family. |
| `scanBlellochScan` | `alpha.scan.blelloch-scan` | `standard` | Blelloch scan operation in the scan family. |
| `scanDecoupledLookbackScan` | `alpha.scan.decoupled-lookback-scan` | `standard` | Decoupled lookback scan operation in the scan family. |
| `scanDeltaRuleScan` | `alpha.scan.delta-rule-scan` | `standard` | Delta rule scan operation in the scan family. |
| `scanExclusiveScan` | `alpha.scan.exclusive-scan` | `standard` | Exclusive scan operation in the scan family. |
| `scanGatedDeltaRuleScan` | `alpha.scan.gated-delta-rule-scan` | `standard` | Gated delta rule scan operation in the scan family. |
| `scanHillisSteeleScan` | `alpha.scan.hillis-steele-scan` | `standard` | Hillis steele scan operation in the scan family. |
| `scanInclusiveScan` | `alpha.scan.inclusive-scan` | `standard` | Inclusive scan operation in the scan family. |
| `scanLinearRecurrenceScan` | `alpha.scan.linear-recurrence-scan` | `standard` | Linear recurrence scan operation in the scan family. |
| `scanMatrixRecurrenceScan` | `alpha.scan.matrix-recurrence-scan` | `standard` | Matrix recurrence scan operation in the scan family. |
| `scanParallelPrefixDoubling` | `alpha.scan.parallel-prefix-doubling` | `standard` | Parallel prefix doubling operation in the scan family. |
| `scanPrefixAnd` | `alpha.scan.prefix-and` | `standard` | Prefix and operation in the scan family. |
| `scanPrefixLogSumExp` | `alpha.scan.prefix-log-sum-exp` | `standard` | Prefix log sum exp operation in the scan family. |
| `scanPrefixMax` | `alpha.scan.prefix-max` | `standard` | Prefix max operation in the scan family. |
| `scanPrefixMin` | `alpha.scan.prefix-min` | `standard` | Prefix min operation in the scan family. |
| `scanPrefixOr` | `alpha.scan.prefix-or` | `standard` | Prefix or operation in the scan family. |
| `scanPrefixProduct` | `alpha.scan.prefix-product` | `standard` | Prefix product operation in the scan family. |
| `scanPrefixSum` | `alpha.scan.prefix-sum` | `standard` | Prefix sum operation in the scan family. |
| `scanPrefixXor` | `alpha.scan.prefix-xor` | `standard` | Prefix xor operation in the scan family. |
| `scanReverseScan` | `alpha.scan.reverse-scan` | `standard` | Reverse scan operation in the scan family. |
| `scanScanByKey` | `alpha.scan.scan-by-key` | `standard` | Scan by key operation in the scan family. |
| `scanSegmentedScan` | `alpha.scan.segmented-scan` | `standard` | Segmented scan operation in the scan family. |
| `scanSelectiveScan` | `alpha.scan.selective-scan` | `standard` | Selective scan operation in the scan family. |

## `sequence_recurrence` (41)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `sequenceRecurrenceDeltaRule` | `alpha.sequence_recurrence.delta-rule` | `standard` | Delta rule operation in the sequence recurrence family. |
| `sequenceRecurrenceDiagonalSsm` | `alpha.sequence_recurrence.diagonal-ssm` | `standard` | Diagonal ssm operation in the sequence recurrence family. |
| `sequenceRecurrenceEquilibriumSolve` | `alpha.sequence_recurrence.equilibrium-solve` | `speculative` | Equilibrium solve operation in the sequence recurrence family. |
| `sequenceRecurrenceEventDrivenMechanism` | `alpha.sequence_recurrence.event-driven-mechanism` | `research` | Event driven mechanism operation in the sequence recurrence family. |
| `sequenceRecurrenceFastWeightUpdate` | `alpha.sequence_recurrence.fast-weight-update` | `standard` | Fast weight update operation in the sequence recurrence family. |
| `sequenceRecurrenceFixedWaveMechanism` | `alpha.sequence_recurrence.fixed-wave-mechanism` | `standard` | Fixed wave mechanism operation in the sequence recurrence family. |
| `sequenceRecurrenceGatedDeltaRule` | `alpha.sequence_recurrence.gated-delta-rule` | `standard` | Gated delta rule operation in the sequence recurrence family. |
| `sequenceRecurrenceGru` | `alpha.sequence_recurrence.gru` | `standard` | Gru operation in the sequence recurrence family. |
| `sequenceRecurrenceH3Recurrence` | `alpha.sequence_recurrence.h3-recurrence` | `standard` | H3 recurrence operation in the sequence recurrence family. |
| `sequenceRecurrenceHebbianUpdate` | `alpha.sequence_recurrence.hebbian-update` | `standard` | Hebbian update operation in the sequence recurrence family. |
| `sequenceRecurrenceHyenaRecurrence` | `alpha.sequence_recurrence.hyena-recurrence` | `standard` | Hyena recurrence operation in the sequence recurrence family. |
| `sequenceRecurrenceImplicitStateUpdate` | `alpha.sequence_recurrence.implicit-state-update` | `speculative` | Implicit state update operation in the sequence recurrence family. |
| `sequenceRecurrenceIndRnn` | `alpha.sequence_recurrence.ind-rnn` | `standard` | Ind rnn operation in the sequence recurrence family. |
| `sequenceRecurrenceLinearAttentionRecurrence` | `alpha.sequence_recurrence.linear-attention-recurrence` | `standard` | Linear attention recurrence operation in the sequence recurrence family. |
| `sequenceRecurrenceLiquidTimeConstant` | `alpha.sequence_recurrence.liquid-time-constant` | `standard` | Liquid time constant operation in the sequence recurrence family. |
| `sequenceRecurrenceLstm` | `alpha.sequence_recurrence.lstm` | `standard` | Lstm operation in the sequence recurrence family. |
| `sequenceRecurrenceMambaSelectiveScan` | `alpha.sequence_recurrence.mamba-selective-scan` | `standard` | Mamba selective scan operation in the sequence recurrence family. |
| `sequenceRecurrenceMinimalGru` | `alpha.sequence_recurrence.minimal-gru` | `standard` | Minimal gru operation in the sequence recurrence family. |
| `sequenceRecurrenceMLstm` | `alpha.sequence_recurrence.m-lstm` | `standard` | M lstm operation in the sequence recurrence family. |
| `sequenceRecurrenceModernHopfieldUpdate` | `alpha.sequence_recurrence.modern-hopfield-update` | `standard` | Modern hopfield update operation in the sequence recurrence family. |
| `sequenceRecurrenceNeuralOdeStep` | `alpha.sequence_recurrence.neural-ode-step` | `standard` | Neural ode step operation in the sequence recurrence family. |
| `sequenceRecurrenceOjaUpdate` | `alpha.sequence_recurrence.oja-update` | `standard` | Oja update operation in the sequence recurrence family. |
| `sequenceRecurrenceOrthogonalRnn` | `alpha.sequence_recurrence.orthogonal-rnn` | `standard` | Orthogonal rnn operation in the sequence recurrence family. |
| `sequenceRecurrencePeepholeLstm` | `alpha.sequence_recurrence.peephole-lstm` | `standard` | Peephole lstm operation in the sequence recurrence family. |
| `sequenceRecurrenceQrnn` | `alpha.sequence_recurrence.qrnn` | `standard` | Qrnn operation in the sequence recurrence family. |
| `sequenceRecurrenceRetNetRecurrence` | `alpha.sequence_recurrence.ret-net-recurrence` | `standard` | Ret net recurrence operation in the sequence recurrence family. |
| `sequenceRecurrenceRwkv7StateUpdate` | `alpha.sequence_recurrence.rwkv7-state-update` | `standard` | Rwkv7 state update operation in the sequence recurrence family. |
| `sequenceRecurrenceRwkvWkv` | `alpha.sequence_recurrence.rwkv-wkv` | `standard` | Rwkv wkv operation in the sequence recurrence family. |
| `sequenceRecurrenceS4Kernel` | `alpha.sequence_recurrence.s4-kernel` | `standard` | S4 kernel operation in the sequence recurrence family. |
| `sequenceRecurrenceSelectiveStateUpdate` | `alpha.sequence_recurrence.selective-state-update` | `standard` | Selective state update operation in the sequence recurrence family. |
| `sequenceRecurrenceSimpleRnn` | `alpha.sequence_recurrence.simple-rnn` | `standard` | Simple rnn operation in the sequence recurrence family. |
| `sequenceRecurrenceSLstm` | `alpha.sequence_recurrence.s-lstm` | `standard` | S lstm operation in the sequence recurrence family. |
| `sequenceRecurrenceSru` | `alpha.sequence_recurrence.sru` | `standard` | Sru operation in the sequence recurrence family. |
| `sequenceRecurrenceStateCompress` | `alpha.sequence_recurrence.state-compress` | `standard` | State compress operation in the sequence recurrence family. |
| `sequenceRecurrenceStateExpand` | `alpha.sequence_recurrence.state-expand` | `standard` | State expand operation in the sequence recurrence family. |
| `sequenceRecurrenceStateReset` | `alpha.sequence_recurrence.state-reset` | `standard` | State reset operation in the sequence recurrence family. |
| `sequenceRecurrenceStateRestore` | `alpha.sequence_recurrence.state-restore` | `standard` | State restore operation in the sequence recurrence family. |
| `sequenceRecurrenceStateSnapshot` | `alpha.sequence_recurrence.state-snapshot` | `standard` | State snapshot operation in the sequence recurrence family. |
| `sequenceRecurrenceStructuredSsm` | `alpha.sequence_recurrence.structured-ssm` | `standard` | Structured ssm operation in the sequence recurrence family. |
| `sequenceRecurrenceUnitaryRnn` | `alpha.sequence_recurrence.unitary-rnn` | `standard` | Unitary rnn operation in the sequence recurrence family. |
| `sequenceRecurrenceXLstmBlock` | `alpha.sequence_recurrence.x-lstm-block` | `standard` | X lstm block operation in the sequence recurrence family. |

## `shape_layout` (49)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `shapeLayoutAlias` | `alpha.shape_layout.alias` | `standard` | Alias operation in the shape layout family. |
| `shapeLayoutAsStrided` | `alpha.shape_layout.as-strided` | `standard` | As strided operation in the shape layout family. |
| `shapeLayoutBitcast` | `alpha.shape_layout.bitcast` | `standard` | Bitcast operation in the shape layout family. |
| `shapeLayoutBlock` | `alpha.shape_layout.block` | `standard` | Block operation in the shape layout family. |
| `shapeLayoutBroadcastInDim` | `alpha.shape_layout.broadcast-in-dim` | `standard` | Broadcast in dim operation in the shape layout family. |
| `shapeLayoutBroadcastTo` | `alpha.shape_layout.broadcast-to` | `standard` | Broadcast to operation in the shape layout family. |
| `shapeLayoutClone` | `alpha.shape_layout.clone` | `standard` | Clone operation in the shape layout family. |
| `shapeLayoutContiguous` | `alpha.shape_layout.contiguous` | `standard` | Contiguous operation in the shape layout family. |
| `shapeLayoutCopy` | `alpha.shape_layout.copy` | `standard` | Copy operation in the shape layout family. |
| `shapeLayoutCrop` | `alpha.shape_layout.crop` | `standard` | Crop operation in the shape layout family. |
| `shapeLayoutDeinterleave` | `alpha.shape_layout.deinterleave` | `standard` | Deinterleave operation in the shape layout family. |
| `shapeLayoutDetach` | `alpha.shape_layout.detach` | `standard` | Detach operation in the shape layout family. |
| `shapeLayoutDevectorize` | `alpha.shape_layout.devectorize` | `standard` | Devectorize operation in the shape layout family. |
| `shapeLayoutExpandDims` | `alpha.shape_layout.expand-dims` | `standard` | Expand dims operation in the shape layout family. |
| `shapeLayoutFlatten` | `alpha.shape_layout.flatten` | `standard` | Flatten operation in the shape layout family. |
| `shapeLayoutFlip` | `alpha.shape_layout.flip` | `standard` | Flip operation in the shape layout family. |
| `shapeLayoutInterleave` | `alpha.shape_layout.interleave` | `standard` | Interleave operation in the shape layout family. |
| `shapeLayoutLayoutCast` | `alpha.shape_layout.layout-cast` | `standard` | Layout cast operation in the shape layout family. |
| `shapeLayoutMaterialize` | `alpha.shape_layout.materialize` | `standard` | Materialize operation in the shape layout family. |
| `shapeLayoutMemoryFormatCast` | `alpha.shape_layout.memory-format-cast` | `standard` | Memory format cast operation in the shape layout family. |
| `shapeLayoutMoveAxis` | `alpha.shape_layout.move-axis` | `standard` | Move axis operation in the shape layout family. |
| `shapeLayoutPack` | `alpha.shape_layout.pack` | `standard` | Pack operation in the shape layout family. |
| `shapeLayoutPad` | `alpha.shape_layout.pad` | `standard` | Pad operation in the shape layout family. |
| `shapeLayoutPermute` | `alpha.shape_layout.permute` | `standard` | Permute operation in the shape layout family. |
| `shapeLayoutReinterpret` | `alpha.shape_layout.reinterpret` | `standard` | Reinterpret operation in the shape layout family. |
| `shapeLayoutRepeat` | `alpha.shape_layout.repeat` | `standard` | Repeat operation in the shape layout family. |
| `shapeLayoutRepeatInterleave` | `alpha.shape_layout.repeat-interleave` | `standard` | Repeat interleave operation in the shape layout family. |
| `shapeLayoutReplicate` | `alpha.shape_layout.replicate` | `standard` | Replicate operation in the shape layout family. |
| `shapeLayoutReshape` | `alpha.shape_layout.reshape` | `standard` | Reshape operation in the shape layout family. |
| `shapeLayoutReshapeLike` | `alpha.shape_layout.reshape-like` | `standard` | Reshape like operation in the shape layout family. |
| `shapeLayoutReverse` | `alpha.shape_layout.reverse` | `standard` | Reverse operation in the shape layout family. |
| `shapeLayoutRoll` | `alpha.shape_layout.roll` | `standard` | Roll operation in the shape layout family. |
| `shapeLayoutRotate90` | `alpha.shape_layout.rotate90` | `standard` | Rotate90 operation in the shape layout family. |
| `shapeLayoutShard` | `alpha.shape_layout.shard` | `standard` | Shard operation in the shape layout family. |
| `shapeLayoutSlidingWindowView` | `alpha.shape_layout.sliding-window-view` | `standard` | Sliding window view operation in the shape layout family. |
| `shapeLayoutSqueeze` | `alpha.shape_layout.squeeze` | `standard` | Squeeze operation in the shape layout family. |
| `shapeLayoutSwapAxes` | `alpha.shape_layout.swap-axes` | `standard` | Swap axes operation in the shape layout family. |
| `shapeLayoutSwizzle` | `alpha.shape_layout.swizzle` | `standard` | Swizzle operation in the shape layout family. |
| `shapeLayoutTile` | `alpha.shape_layout.tile` | `standard` | Tile operation in the shape layout family. |
| `shapeLayoutTranspose` | `alpha.shape_layout.transpose` | `standard` | Transpose operation in the shape layout family. |
| `shapeLayoutUnblock` | `alpha.shape_layout.unblock` | `standard` | Unblock operation in the shape layout family. |
| `shapeLayoutUnflatten` | `alpha.shape_layout.unflatten` | `standard` | Unflatten operation in the shape layout family. |
| `shapeLayoutUnpack` | `alpha.shape_layout.unpack` | `standard` | Unpack operation in the shape layout family. |
| `shapeLayoutUnreplicate` | `alpha.shape_layout.unreplicate` | `standard` | Unreplicate operation in the shape layout family. |
| `shapeLayoutUnshard` | `alpha.shape_layout.unshard` | `standard` | Unshard operation in the shape layout family. |
| `shapeLayoutUnsqueeze` | `alpha.shape_layout.unsqueeze` | `standard` | Unsqueeze operation in the shape layout family. |
| `shapeLayoutUnswizzle` | `alpha.shape_layout.unswizzle` | `standard` | Unswizzle operation in the shape layout family. |
| `shapeLayoutVectorize` | `alpha.shape_layout.vectorize` | `standard` | Vectorize operation in the shape layout family. |
| `shapeLayoutView` | `alpha.shape_layout.view` | `standard` | View operation in the shape layout family. |

## `signal_transform` (39)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `signalTransformAnalyticSignal` | `alpha.signal_transform.analytic-signal` | `standard` | Analytic signal operation in the signal transform family. |
| `signalTransformAutocorrelation` | `alpha.signal_transform.autocorrelation` | `standard` | Autocorrelation operation in the signal transform family. |
| `signalTransformCepstrum` | `alpha.signal_transform.cepstrum` | `standard` | Cepstrum operation in the signal transform family. |
| `signalTransformChirpZTransform` | `alpha.signal_transform.chirp-ztransform` | `standard` | Chirp ztransform operation in the signal transform family. |
| `signalTransformContinuousWaveletTransform` | `alpha.signal_transform.continuous-wavelet-transform` | `standard` | Continuous wavelet transform operation in the signal transform family. |
| `signalTransformCrossCorrelation` | `alpha.signal_transform.cross-correlation` | `standard` | Cross correlation operation in the signal transform family. |
| `signalTransformCrossSpectrum` | `alpha.signal_transform.cross-spectrum` | `standard` | Cross spectrum operation in the signal transform family. |
| `signalTransformDct1` | `alpha.signal_transform.dct1` | `standard` | Dct1 operation in the signal transform family. |
| `signalTransformDct2` | `alpha.signal_transform.dct2` | `standard` | Dct2 operation in the signal transform family. |
| `signalTransformDct3` | `alpha.signal_transform.dct3` | `standard` | Dct3 operation in the signal transform family. |
| `signalTransformDct4` | `alpha.signal_transform.dct4` | `standard` | Dct4 operation in the signal transform family. |
| `signalTransformDst1` | `alpha.signal_transform.dst1` | `standard` | Dst1 operation in the signal transform family. |
| `signalTransformDst2` | `alpha.signal_transform.dst2` | `standard` | Dst2 operation in the signal transform family. |
| `signalTransformDst3` | `alpha.signal_transform.dst3` | `standard` | Dst3 operation in the signal transform family. |
| `signalTransformDst4` | `alpha.signal_transform.dst4` | `standard` | Dst4 operation in the signal transform family. |
| `signalTransformFft` | `alpha.signal_transform.fft` | `standard` | Fft operation in the signal transform family. |
| `signalTransformFft2` | `alpha.signal_transform.fft2` | `standard` | Fft2 operation in the signal transform family. |
| `signalTransformFftn` | `alpha.signal_transform.fftn` | `standard` | Fftn operation in the signal transform family. |
| `signalTransformFractionalDelay` | `alpha.signal_transform.fractional-delay` | `standard` | Fractional delay operation in the signal transform family. |
| `signalTransformFractionalFourierTransform` | `alpha.signal_transform.fractional-fourier-transform` | `standard` | Fractional fourier transform operation in the signal transform family. |
| `signalTransformHartleyTransform` | `alpha.signal_transform.hartley-transform` | `standard` | Hartley transform operation in the signal transform family. |
| `signalTransformHilbertTransform` | `alpha.signal_transform.hilbert-transform` | `standard` | Hilbert transform operation in the signal transform family. |
| `signalTransformIfft` | `alpha.signal_transform.ifft` | `standard` | Ifft operation in the signal transform family. |
| `signalTransformIfft2` | `alpha.signal_transform.ifft2` | `standard` | Ifft2 operation in the signal transform family. |
| `signalTransformIfftn` | `alpha.signal_transform.ifftn` | `standard` | Ifftn operation in the signal transform family. |
| `signalTransformInverseNumberTheoreticTransform` | `alpha.signal_transform.inverse-number-theoretic-transform` | `standard` | Inverse number theoretic transform operation in the signal transform family. |
| `signalTransformInverseShortTimeFourierTransform` | `alpha.signal_transform.inverse-short-time-fourier-transform` | `standard` | Inverse short time fourier transform operation in the signal transform family. |
| `signalTransformInverseWaveletTransform` | `alpha.signal_transform.inverse-wavelet-transform` | `standard` | Inverse wavelet transform operation in the signal transform family. |
| `signalTransformIrfft` | `alpha.signal_transform.irfft` | `standard` | Irfft operation in the signal transform family. |
| `signalTransformMelFilterbank` | `alpha.signal_transform.mel-filterbank` | `standard` | Mel filterbank operation in the signal transform family. |
| `signalTransformNumberTheoreticTransform` | `alpha.signal_transform.number-theoretic-transform` | `standard` | Number theoretic transform operation in the signal transform family. |
| `signalTransformOverlapAdd` | `alpha.signal_transform.overlap-add` | `standard` | Overlap add operation in the signal transform family. |
| `signalTransformOverlapSave` | `alpha.signal_transform.overlap-save` | `standard` | Overlap save operation in the signal transform family. |
| `signalTransformPolyphaseFilterbank` | `alpha.signal_transform.polyphase-filterbank` | `standard` | Polyphase filterbank operation in the signal transform family. |
| `signalTransformPowerSpectrum` | `alpha.signal_transform.power-spectrum` | `standard` | Power spectrum operation in the signal transform family. |
| `signalTransformResample` | `alpha.signal_transform.resample` | `standard` | Resample operation in the signal transform family. |
| `signalTransformRfft` | `alpha.signal_transform.rfft` | `standard` | Rfft operation in the signal transform family. |
| `signalTransformShortTimeFourierTransform` | `alpha.signal_transform.short-time-fourier-transform` | `standard` | Short time fourier transform operation in the signal transform family. |
| `signalTransformWaveletTransform` | `alpha.signal_transform.wavelet-transform` | `standard` | Wavelet transform operation in the signal transform family. |

## `solver_factorization` (54)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `solverFactorizationArnoldi` | `alpha.solver_factorization.arnoldi` | `standard` | Arnoldi operation in the solver factorization family. |
| `solverFactorizationBicgstab` | `alpha.solver_factorization.bicgstab` | `standard` | Bicgstab operation in the solver factorization family. |
| `solverFactorizationBidiagonalize` | `alpha.solver_factorization.bidiagonalize` | `standard` | Bidiagonalize operation in the solver factorization family. |
| `solverFactorizationBunchKaufman` | `alpha.solver_factorization.bunch-kaufman` | `standard` | Bunch kaufman operation in the solver factorization family. |
| `solverFactorizationCholesky` | `alpha.solver_factorization.cholesky` | `standard` | Cholesky operation in the solver factorization family. |
| `solverFactorizationConjugateGradient` | `alpha.solver_factorization.conjugate-gradient` | `standard` | Conjugate gradient operation in the solver factorization family. |
| `solverFactorizationEig` | `alpha.solver_factorization.eig` | `standard` | Eig operation in the solver factorization family. |
| `solverFactorizationEigh` | `alpha.solver_factorization.eigh` | `standard` | Eigh operation in the solver factorization family. |
| `solverFactorizationGaussSeidel` | `alpha.solver_factorization.gauss-seidel` | `standard` | Gauss seidel operation in the solver factorization family. |
| `solverFactorizationGeneralizedEig` | `alpha.solver_factorization.generalized-eig` | `standard` | Generalized eig operation in the solver factorization family. |
| `solverFactorizationGeneralizedSchur` | `alpha.solver_factorization.generalized-schur` | `standard` | Generalized schur operation in the solver factorization family. |
| `solverFactorizationGmres` | `alpha.solver_factorization.gmres` | `standard` | Gmres operation in the solver factorization family. |
| `solverFactorizationHessenberg` | `alpha.solver_factorization.hessenberg` | `standard` | Hessenberg operation in the solver factorization family. |
| `solverFactorizationInverseIteration` | `alpha.solver_factorization.inverse-iteration` | `standard` | Inverse iteration operation in the solver factorization family. |
| `solverFactorizationJacobiIteration` | `alpha.solver_factorization.jacobi-iteration` | `standard` | Jacobi iteration operation in the solver factorization family. |
| `solverFactorizationJacobiSvd` | `alpha.solver_factorization.jacobi-svd` | `standard` | Jacobi svd operation in the solver factorization family. |
| `solverFactorizationLanczos` | `alpha.solver_factorization.lanczos` | `standard` | Lanczos operation in the solver factorization family. |
| `solverFactorizationLdl` | `alpha.solver_factorization.ldl` | `standard` | Ldl operation in the solver factorization family. |
| `solverFactorizationLeastSquares` | `alpha.solver_factorization.least-squares` | `standard` | Least squares operation in the solver factorization family. |
| `solverFactorizationLobpcg` | `alpha.solver_factorization.lobpcg` | `standard` | Lobpcg operation in the solver factorization family. |
| `solverFactorizationLq` | `alpha.solver_factorization.lq` | `standard` | Lq operation in the solver factorization family. |
| `solverFactorizationLsmr` | `alpha.solver_factorization.lsmr` | `standard` | Lsmr operation in the solver factorization family. |
| `solverFactorizationLsqr` | `alpha.solver_factorization.lsqr` | `standard` | Lsqr operation in the solver factorization family. |
| `solverFactorizationLu` | `alpha.solver_factorization.lu` | `standard` | Lu operation in the solver factorization family. |
| `solverFactorizationLuCompletePivot` | `alpha.solver_factorization.lu-complete-pivot` | `standard` | Lu complete pivot operation in the solver factorization family. |
| `solverFactorizationLuPartialPivot` | `alpha.solver_factorization.lu-partial-pivot` | `standard` | Lu partial pivot operation in the solver factorization family. |
| `solverFactorizationLyapunovSolve` | `alpha.solver_factorization.lyapunov-solve` | `standard` | Lyapunov solve operation in the solver factorization family. |
| `solverFactorizationMatrixDeterminant` | `alpha.solver_factorization.matrix-determinant` | `standard` | Matrix determinant operation in the solver factorization family. |
| `solverFactorizationMatrixExponential` | `alpha.solver_factorization.matrix-exponential` | `standard` | Matrix exponential operation in the solver factorization family. |
| `solverFactorizationMatrixInverse` | `alpha.solver_factorization.matrix-inverse` | `standard` | Matrix inverse operation in the solver factorization family. |
| `solverFactorizationMatrixLogarithm` | `alpha.solver_factorization.matrix-logarithm` | `standard` | Matrix logarithm operation in the solver factorization family. |
| `solverFactorizationMatrixPower` | `alpha.solver_factorization.matrix-power` | `standard` | Matrix power operation in the solver factorization family. |
| `solverFactorizationMatrixPseudoInverse` | `alpha.solver_factorization.matrix-pseudo-inverse` | `standard` | Matrix pseudo inverse operation in the solver factorization family. |
| `solverFactorizationMatrixSquareRoot` | `alpha.solver_factorization.matrix-square-root` | `standard` | Matrix square root operation in the solver factorization family. |
| `solverFactorizationMinres` | `alpha.solver_factorization.minres` | `standard` | Minres operation in the solver factorization family. |
| `solverFactorizationMultigridVCycle` | `alpha.solver_factorization.multigrid-vcycle` | `standard` | Multigrid vcycle operation in the solver factorization family. |
| `solverFactorizationPivotedCholesky` | `alpha.solver_factorization.pivoted-cholesky` | `standard` | Pivoted cholesky operation in the solver factorization family. |
| `solverFactorizationPivotedQr` | `alpha.solver_factorization.pivoted-qr` | `standard` | Pivoted qr operation in the solver factorization family. |
| `solverFactorizationPolarDecomposition` | `alpha.solver_factorization.polar-decomposition` | `research` | Polar decomposition operation in the solver factorization family. |
| `solverFactorizationPowerIteration` | `alpha.solver_factorization.power-iteration` | `standard` | Power iteration operation in the solver factorization family. |
| `solverFactorizationPreconditionedConjugateGradient` | `alpha.solver_factorization.preconditioned-conjugate-gradient` | `standard` | Preconditioned conjugate gradient operation in the solver factorization family. |
| `solverFactorizationQl` | `alpha.solver_factorization.ql` | `standard` | Ql operation in the solver factorization family. |
| `solverFactorizationQr` | `alpha.solver_factorization.qr` | `standard` | Qr operation in the solver factorization family. |
| `solverFactorizationRandomizedSvd` | `alpha.solver_factorization.randomized-svd` | `standard` | Randomized svd operation in the solver factorization family. |
| `solverFactorizationRiccatiSolve` | `alpha.solver_factorization.riccati-solve` | `standard` | Riccati solve operation in the solver factorization family. |
| `solverFactorizationRq` | `alpha.solver_factorization.rq` | `standard` | Rq operation in the solver factorization family. |
| `solverFactorizationSchur` | `alpha.solver_factorization.schur` | `standard` | Schur operation in the solver factorization family. |
| `solverFactorizationSlogdet` | `alpha.solver_factorization.slogdet` | `standard` | Slogdet operation in the solver factorization family. |
| `solverFactorizationSolve` | `alpha.solver_factorization.solve` | `standard` | Solve operation in the solver factorization family. |
| `solverFactorizationSuccessiveOverRelaxation` | `alpha.solver_factorization.successive-over-relaxation` | `standard` | Successive over relaxation operation in the solver factorization family. |
| `solverFactorizationSvd` | `alpha.solver_factorization.svd` | `standard` | Svd operation in the solver factorization family. |
| `solverFactorizationSylvesterSolve` | `alpha.solver_factorization.sylvester-solve` | `standard` | Sylvester solve operation in the solver factorization family. |
| `solverFactorizationTriangularSolve` | `alpha.solver_factorization.triangular-solve` | `standard` | Triangular solve operation in the solver factorization family. |
| `solverFactorizationTridiagonalize` | `alpha.solver_factorization.tridiagonalize` | `standard` | Tridiagonalize operation in the solver factorization family. |

## `sparse_compute` (30)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `sparseComputeBlockSparseMatmul` | `alpha.sparse_compute.block-sparse-matmul` | `standard` | Block sparse matmul operation in the sparse compute family. |
| `sparseComputeFrontierExpand` | `alpha.sparse_compute.frontier-expand` | `standard` | Frontier expand operation in the sparse compute family. |
| `sparseComputeGraphAttention` | `alpha.sparse_compute.graph-attention` | `standard` | Graph attention operation in the sparse compute family. |
| `sparseComputeGraphMessagePass` | `alpha.sparse_compute.graph-message-pass` | `standard` | Graph message pass operation in the sparse compute family. |
| `sparseComputeMaskedSpmm` | `alpha.sparse_compute.masked-spmm` | `standard` | Masked spmm operation in the sparse compute family. |
| `sparseComputeNmSparseMatmul` | `alpha.sparse_compute.nm-sparse-matmul` | `standard` | Nm sparse matmul operation in the sparse compute family. |
| `sparseComputeSampledDenseDense` | `alpha.sparse_compute.sampled-dense-dense` | `standard` | Sampled dense dense operation in the sparse compute family. |
| `sparseComputeSddmm` | `alpha.sparse_compute.sddmm` | `standard` | Sddmm operation in the sparse compute family. |
| `sparseComputeSegmentSpmm` | `alpha.sparse_compute.segment-spmm` | `standard` | Segment spmm operation in the sparse compute family. |
| `sparseComputeSparseAttention` | `alpha.sparse_compute.sparse-attention` | `standard` | Sparse attention operation in the sparse compute family. |
| `sparseComputeSparseAxpby` | `alpha.sparse_compute.sparse-axpby` | `standard` | Sparse axpby operation in the sparse compute family. |
| `sparseComputeSparseBooleanMatmul` | `alpha.sparse_compute.sparse-boolean-matmul` | `standard` | Sparse boolean matmul operation in the sparse compute family. |
| `sparseComputeSparseCholesky` | `alpha.sparse_compute.sparse-cholesky` | `standard` | Sparse cholesky operation in the sparse compute family. |
| `sparseComputeSparseGather` | `alpha.sparse_compute.sparse-gather` | `standard` | Sparse gather operation in the sparse compute family. |
| `sparseComputeSparseLeastSquares` | `alpha.sparse_compute.sparse-least-squares` | `standard` | Sparse least squares operation in the sparse compute family. |
| `sparseComputeSparseMinPlusMatmul` | `alpha.sparse_compute.sparse-min-plus-matmul` | `standard` | Sparse min plus matmul operation in the sparse compute family. |
| `sparseComputeSparseQr` | `alpha.sparse_compute.sparse-qr` | `standard` | Sparse qr operation in the sparse compute family. |
| `sparseComputeSparseReduce` | `alpha.sparse_compute.sparse-reduce` | `standard` | Sparse reduce operation in the sparse compute family. |
| `sparseComputeSparseRefactor` | `alpha.sparse_compute.sparse-refactor` | `standard` | Sparse refactor operation in the sparse compute family. |
| `sparseComputeSparseRotate` | `alpha.sparse_compute.sparse-rotate` | `standard` | Sparse rotate operation in the sparse compute family. |
| `sparseComputeSparseScatter` | `alpha.sparse_compute.sparse-scatter` | `standard` | Sparse scatter operation in the sparse compute family. |
| `sparseComputeSparseSemiringMatmul` | `alpha.sparse_compute.sparse-semiring-matmul` | `standard` | Sparse semiring matmul operation in the sparse compute family. |
| `sparseComputeSparseSoftmax` | `alpha.sparse_compute.sparse-softmax` | `standard` | Sparse softmax operation in the sparse compute family. |
| `sparseComputeSparseTriangularSolve` | `alpha.sparse_compute.sparse-triangular-solve` | `standard` | Sparse triangular solve operation in the sparse compute family. |
| `sparseComputeSpgemm` | `alpha.sparse_compute.spgemm` | `standard` | Spgemm operation in the sparse compute family. |
| `sparseComputeSpmm` | `alpha.sparse_compute.spmm` | `standard` | Spmm operation in the sparse compute family. |
| `sparseComputeSpmv` | `alpha.sparse_compute.spmv` | `standard` | Spmv operation in the sparse compute family. |
| `sparseComputeSpsm` | `alpha.sparse_compute.spsm` | `standard` | Spsm operation in the sparse compute family. |
| `sparseComputeSpsv` | `alpha.sparse_compute.spsv` | `standard` | Spsv operation in the sparse compute family. |
| `sparseComputeSpvv` | `alpha.sparse_compute.spvv` | `standard` | Spvv operation in the sparse compute family. |

## `sparse_format` (28)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `sparseFormatBsrToCsr` | `alpha.sparse_format.bsr-to-csr` | `standard` | Bsr to csr operation in the sparse format family. |
| `sparseFormatCooToCsc` | `alpha.sparse_format.coo-to-csc` | `standard` | Coo to csc operation in the sparse format family. |
| `sparseFormatCooToCsr` | `alpha.sparse_format.coo-to-csr` | `standard` | Coo to csr operation in the sparse format family. |
| `sparseFormatCsrToBsr` | `alpha.sparse_format.csr-to-bsr` | `standard` | Csr to bsr operation in the sparse format family. |
| `sparseFormatCsrToCoo` | `alpha.sparse_format.csr-to-coo` | `standard` | Csr to coo operation in the sparse format family. |
| `sparseFormatCsrToCsc` | `alpha.sparse_format.csr-to-csc` | `standard` | Csr to csc operation in the sparse format family. |
| `sparseFormatDenseToBsc` | `alpha.sparse_format.dense-to-bsc` | `standard` | Dense to bsc operation in the sparse format family. |
| `sparseFormatDenseToBsr` | `alpha.sparse_format.dense-to-bsr` | `standard` | Dense to bsr operation in the sparse format family. |
| `sparseFormatDenseToCoo` | `alpha.sparse_format.dense-to-coo` | `standard` | Dense to coo operation in the sparse format family. |
| `sparseFormatDenseToCsc` | `alpha.sparse_format.dense-to-csc` | `standard` | Dense to csc operation in the sparse format family. |
| `sparseFormatDenseToCsr` | `alpha.sparse_format.dense-to-csr` | `standard` | Dense to csr operation in the sparse format family. |
| `sparseFormatDenseToDia` | `alpha.sparse_format.dense-to-dia` | `standard` | Dense to dia operation in the sparse format family. |
| `sparseFormatDenseToEll` | `alpha.sparse_format.dense-to-ell` | `standard` | Dense to ell operation in the sparse format family. |
| `sparseFormatDenseToSell` | `alpha.sparse_format.dense-to-sell` | `standard` | Dense to sell operation in the sparse format family. |
| `sparseFormatSparseCanonicalize` | `alpha.sparse_format.sparse-canonicalize` | `standard` | Sparse canonicalize operation in the sparse format family. |
| `sparseFormatSparseCoalesce` | `alpha.sparse_format.sparse-coalesce` | `standard` | Sparse coalesce operation in the sparse format family. |
| `sparseFormatSparseCompress2of4` | `alpha.sparse_format.sparse-compress2of4` | `standard` | Sparse compress2of4 operation in the sparse format family. |
| `sparseFormatSparseCountNnz` | `alpha.sparse_format.sparse-count-nnz` | `standard` | Sparse count nnz operation in the sparse format family. |
| `sparseFormatSparseDecompress2of4` | `alpha.sparse_format.sparse-decompress2of4` | `standard` | Sparse decompress2of4 operation in the sparse format family. |
| `sparseFormatSparsePermute` | `alpha.sparse_format.sparse-permute` | `standard` | Sparse permute operation in the sparse format family. |
| `sparseFormatSparsePruneBlock` | `alpha.sparse_format.sparse-prune-block` | `standard` | Sparse prune block operation in the sparse format family. |
| `sparseFormatSparsePruneNm` | `alpha.sparse_format.sparse-prune-nm` | `standard` | Sparse prune nm operation in the sparse format family. |
| `sparseFormatSparsePruneThreshold` | `alpha.sparse_format.sparse-prune-threshold` | `standard` | Sparse prune threshold operation in the sparse format family. |
| `sparseFormatSparsePruneTopK` | `alpha.sparse_format.sparse-prune-top-k` | `standard` | Sparse prune top k operation in the sparse format family. |
| `sparseFormatSparseReorder` | `alpha.sparse_format.sparse-reorder` | `standard` | Sparse reorder operation in the sparse format family. |
| `sparseFormatSparseSortIndices` | `alpha.sparse_format.sparse-sort-indices` | `standard` | Sparse sort indices operation in the sparse format family. |
| `sparseFormatSparseTranspose` | `alpha.sparse_format.sparse-transpose` | `standard` | Sparse transpose operation in the sparse format family. |
| `sparseFormatSparseValidate` | `alpha.sparse_format.sparse-validate` | `standard` | Sparse validate operation in the sparse format family. |

## `structured_linear` (28)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `structuredLinearBandedMultiply` | `alpha.structured_linear.banded-multiply` | `standard` | Banded multiply operation in the structured linear family. |
| `structuredLinearBlockCirculantMultiply` | `alpha.structured_linear.block-circulant-multiply` | `standard` | Block circulant multiply operation in the structured linear family. |
| `structuredLinearBlockDiagonalMultiply` | `alpha.structured_linear.block-diagonal-multiply` | `standard` | Block diagonal multiply operation in the structured linear family. |
| `structuredLinearButterflyTransform` | `alpha.structured_linear.butterfly-transform` | `standard` | Butterfly transform operation in the structured linear family. |
| `structuredLinearCauchyMultiply` | `alpha.structured_linear.cauchy-multiply` | `standard` | Cauchy multiply operation in the structured linear family. |
| `structuredLinearCayleyTransform` | `alpha.structured_linear.cayley-transform` | `standard` | Cayley transform operation in the structured linear family. |
| `structuredLinearCirculantMultiply` | `alpha.structured_linear.circulant-multiply` | `standard` | Circulant multiply operation in the structured linear family. |
| `structuredLinearCompanionMultiply` | `alpha.structured_linear.companion-multiply` | `standard` | Companion multiply operation in the structured linear family. |
| `structuredLinearDiagonalMultiply` | `alpha.structured_linear.diagonal-multiply` | `standard` | Diagonal multiply operation in the structured linear family. |
| `structuredLinearFastfoodTransform` | `alpha.structured_linear.fastfood-transform` | `standard` | Fastfood transform operation in the structured linear family. |
| `structuredLinearGivensSequence` | `alpha.structured_linear.givens-sequence` | `standard` | Givens sequence operation in the structured linear family. |
| `structuredLinearGivensTransform` | `alpha.structured_linear.givens-transform` | `standard` | Givens transform operation in the structured linear family. |
| `structuredLinearHaarTransform` | `alpha.structured_linear.haar-transform` | `standard` | Haar transform operation in the structured linear family. |
| `structuredLinearHadamardTransform` | `alpha.structured_linear.hadamard-transform` | `standard` | Hadamard transform operation in the structured linear family. |
| `structuredLinearHankelMultiply` | `alpha.structured_linear.hankel-multiply` | `standard` | Hankel multiply operation in the structured linear family. |
| `structuredLinearHouseholderProduct` | `alpha.structured_linear.householder-product` | `standard` | Householder product operation in the structured linear family. |
| `structuredLinearHouseholderTransform` | `alpha.structured_linear.householder-transform` | `standard` | Householder transform operation in the structured linear family. |
| `structuredLinearLiftingWaveletTransform` | `alpha.structured_linear.lifting-wavelet-transform` | `standard` | Lifting wavelet transform operation in the structured linear family. |
| `structuredLinearLowRankUpdate` | `alpha.structured_linear.low-rank-update` | `standard` | Low rank update operation in the structured linear family. |
| `structuredLinearMatrixSign` | `alpha.structured_linear.matrix-sign` | `standard` | Matrix sign operation in the structured linear family. |
| `structuredLinearOrthogonalRotation` | `alpha.structured_linear.orthogonal-rotation` | `standard` | Orthogonal rotation operation in the structured linear family. |
| `structuredLinearPolarFactor` | `alpha.structured_linear.polar-factor` | `research` | Polar factor operation in the structured linear family. |
| `structuredLinearRandomizedHadamardTransform` | `alpha.structured_linear.randomized-hadamard-transform` | `standard` | Randomized hadamard transform operation in the structured linear family. |
| `structuredLinearShermanMorrisonSolve` | `alpha.structured_linear.sherman-morrison-solve` | `standard` | Sherman morrison solve operation in the structured linear family. |
| `structuredLinearToeplitzMultiply` | `alpha.structured_linear.toeplitz-multiply` | `standard` | Toeplitz multiply operation in the structured linear family. |
| `structuredLinearVandermondeMultiply` | `alpha.structured_linear.vandermonde-multiply` | `standard` | Vandermonde multiply operation in the structured linear family. |
| `structuredLinearWalshHadamardTransform` | `alpha.structured_linear.walsh-hadamard-transform` | `standard` | Walsh hadamard transform operation in the structured linear family. |
| `structuredLinearWoodburySolve` | `alpha.structured_linear.woodbury-solve` | `standard` | Woodbury solve operation in the structured linear family. |

## `tensor_algebra` (25)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `tensorAlgebraCpDecomposition` | `alpha.tensor_algebra.cp-decomposition` | `standard` | Cp decomposition operation in the tensor algebra family. |
| `tensorAlgebraEinsum` | `alpha.tensor_algebra.einsum` | `standard` | Einsum operation in the tensor algebra family. |
| `tensorAlgebraFaceSplittingProduct` | `alpha.tensor_algebra.face-splitting-product` | `standard` | Face splitting product operation in the tensor algebra family. |
| `tensorAlgebraFoldTensor` | `alpha.tensor_algebra.fold-tensor` | `standard` | Fold tensor operation in the tensor algebra family. |
| `tensorAlgebraHadamardProduct` | `alpha.tensor_algebra.hadamard-product` | `standard` | Hadamard product operation in the tensor algebra family. |
| `tensorAlgebraHigherOrderSvd` | `alpha.tensor_algebra.higher-order-svd` | `standard` | Higher order svd operation in the tensor algebra family. |
| `tensorAlgebraHypergraphContract` | `alpha.tensor_algebra.hypergraph-contract` | `speculative` | Hypergraph contract operation in the tensor algebra family. |
| `tensorAlgebraKhatriRaoProduct` | `alpha.tensor_algebra.khatri-rao-product` | `standard` | Khatri rao product operation in the tensor algebra family. |
| `tensorAlgebraKroneckerProduct` | `alpha.tensor_algebra.kronecker-product` | `standard` | Kronecker product operation in the tensor algebra family. |
| `tensorAlgebraMatricize` | `alpha.tensor_algebra.matricize` | `standard` | Matricize operation in the tensor algebra family. |
| `tensorAlgebraModeProduct` | `alpha.tensor_algebra.mode-product` | `standard` | Mode product operation in the tensor algebra family. |
| `tensorAlgebraMpsContract` | `alpha.tensor_algebra.mps-contract` | `standard` | Mps contract operation in the tensor algebra family. |
| `tensorAlgebraNModeProduct` | `alpha.tensor_algebra.n-mode-product` | `standard` | N mode product operation in the tensor algebra family. |
| `tensorAlgebraPepsContract` | `alpha.tensor_algebra.peps-contract` | `standard` | Peps contract operation in the tensor algebra family. |
| `tensorAlgebraTensorContract` | `alpha.tensor_algebra.tensor-contract` | `standard` | Tensor contract operation in the tensor algebra family. |
| `tensorAlgebraTensorDot` | `alpha.tensor_algebra.tensor-dot` | `standard` | Tensor dot operation in the tensor algebra family. |
| `tensorAlgebraTensorNetworkContract` | `alpha.tensor_algebra.tensor-network-contract` | `standard` | Tensor network contract operation in the tensor algebra family. |
| `tensorAlgebraTensorOuter` | `alpha.tensor_algebra.tensor-outer` | `standard` | Tensor outer operation in the tensor algebra family. |
| `tensorAlgebraTensorRingDecomposition` | `alpha.tensor_algebra.tensor-ring-decomposition` | `standard` | Tensor ring decomposition operation in the tensor algebra family. |
| `tensorAlgebraTensorSvd` | `alpha.tensor_algebra.tensor-svd` | `standard` | Tensor svd operation in the tensor algebra family. |
| `tensorAlgebraTensorTrace` | `alpha.tensor_algebra.tensor-trace` | `standard` | Tensor trace operation in the tensor algebra family. |
| `tensorAlgebraTensorTrainContract` | `alpha.tensor_algebra.tensor-train-contract` | `standard` | Tensor train contract operation in the tensor algebra family. |
| `tensorAlgebraTensorTrainDecomposition` | `alpha.tensor_algebra.tensor-train-decomposition` | `standard` | Tensor train decomposition operation in the tensor algebra family. |
| `tensorAlgebraTensorTrainRound` | `alpha.tensor_algebra.tensor-train-round` | `standard` | Tensor train round operation in the tensor algebra family. |
| `tensorAlgebraTuckerDecomposition` | `alpha.tensor_algebra.tucker-decomposition` | `standard` | Tucker decomposition operation in the tensor algebra family. |

## `tensor_creation` (25)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `tensorCreationArange` | `alpha.tensor_creation.arange` | `standard` | Arange operation in the tensor creation family. |
| `tensorCreationBandPart` | `alpha.tensor_creation.band-part` | `standard` | Band part operation in the tensor creation family. |
| `tensorCreationComplexTensor` | `alpha.tensor_creation.complex-tensor` | `standard` | Complex tensor operation in the tensor creation family. |
| `tensorCreationDiag` | `alpha.tensor_creation.diag` | `standard` | Diag operation in the tensor creation family. |
| `tensorCreationDiagEmbed` | `alpha.tensor_creation.diag-embed` | `standard` | Diag embed operation in the tensor creation family. |
| `tensorCreationEmpty` | `alpha.tensor_creation.empty` | `standard` | Empty operation in the tensor creation family. |
| `tensorCreationEye` | `alpha.tensor_creation.eye` | `standard` | Eye operation in the tensor creation family. |
| `tensorCreationFromArray` | `alpha.tensor_creation.from-array` | `standard` | From array operation in the tensor creation family. |
| `tensorCreationFromBuffer` | `alpha.tensor_creation.from-buffer` | `standard` | From buffer operation in the tensor creation family. |
| `tensorCreationFromMappedMemory` | `alpha.tensor_creation.from-mapped-memory` | `standard` | From mapped memory operation in the tensor creation family. |
| `tensorCreationFull` | `alpha.tensor_creation.full` | `standard` | Full operation in the tensor creation family. |
| `tensorCreationIdentityMatrix` | `alpha.tensor_creation.identity-matrix` | `standard` | Identity matrix operation in the tensor creation family. |
| `tensorCreationLinspace` | `alpha.tensor_creation.linspace` | `standard` | Linspace operation in the tensor creation family. |
| `tensorCreationLogspace` | `alpha.tensor_creation.logspace` | `standard` | Logspace operation in the tensor creation family. |
| `tensorCreationMetaTensor` | `alpha.tensor_creation.meta-tensor` | `standard` | Meta tensor operation in the tensor creation family. |
| `tensorCreationOnes` | `alpha.tensor_creation.ones` | `standard` | Ones operation in the tensor creation family. |
| `tensorCreationQuantizedTensor` | `alpha.tensor_creation.quantized-tensor` | `standard` | Quantized tensor operation in the tensor creation family. |
| `tensorCreationRaggedTensor` | `alpha.tensor_creation.ragged-tensor` | `standard` | Ragged tensor operation in the tensor creation family. |
| `tensorCreationRandomTensor` | `alpha.tensor_creation.random-tensor` | `standard` | Random tensor operation in the tensor creation family. |
| `tensorCreationScalar` | `alpha.tensor_creation.scalar` | `standard` | Scalar operation in the tensor creation family. |
| `tensorCreationSparseTensor` | `alpha.tensor_creation.sparse-tensor` | `standard` | Sparse tensor operation in the tensor creation family. |
| `tensorCreationSymbolicTensor` | `alpha.tensor_creation.symbolic-tensor` | `standard` | Symbolic tensor operation in the tensor creation family. |
| `tensorCreationTril` | `alpha.tensor_creation.tril` | `standard` | Tril operation in the tensor creation family. |
| `tensorCreationTriu` | `alpha.tensor_creation.triu` | `standard` | Triu operation in the tensor creation family. |
| `tensorCreationZeros` | `alpha.tensor_creation.zeros` | `standard` | Zeros operation in the tensor creation family. |

## `ternary_math` (13)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `ternaryMathAddDiv` | `alpha.ternary_math.add-div` | `standard` | Add div operation in the ternary math family. |
| `ternaryMathAddMul` | `alpha.ternary_math.add-mul` | `standard` | Add mul operation in the ternary math family. |
| `ternaryMathClampTensor` | `alpha.ternary_math.clamp-tensor` | `standard` | Clamp tensor operation in the ternary math family. |
| `ternaryMathComplexFma` | `alpha.ternary_math.complex-fma` | `standard` | Complex fma operation in the ternary math family. |
| `ternaryMathConditionalSwap` | `alpha.ternary_math.conditional-swap` | `standard` | Conditional swap operation in the ternary math family. |
| `ternaryMathFma` | `alpha.ternary_math.fma` | `standard` | Fma operation in the ternary math family. |
| `ternaryMathLerp` | `alpha.ternary_math.lerp` | `standard` | Lerp operation in the ternary math family. |
| `ternaryMathMaskedFma` | `alpha.ternary_math.masked-fma` | `standard` | Masked fma operation in the ternary math family. |
| `ternaryMathMedianOfThree` | `alpha.ternary_math.median-of-three` | `standard` | Median of three operation in the ternary math family. |
| `ternaryMathMulAdd` | `alpha.ternary_math.mul-add` | `standard` | Mul add operation in the ternary math family. |
| `ternaryMathSaturatingFma` | `alpha.ternary_math.saturating-fma` | `standard` | Saturating fma operation in the ternary math family. |
| `ternaryMathSelectAndScatter` | `alpha.ternary_math.select-and-scatter` | `standard` | Select and scatter operation in the ternary math family. |
| `ternaryMathWhereTensor` | `alpha.ternary_math.where-tensor` | `standard` | Where tensor operation in the ternary math family. |

## `unary_math` (85)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `unaryMathAbs` | `alpha.unary_math.abs` | `standard` | Abs operation in the unary math family. |
| `unaryMathAcos` | `alpha.unary_math.acos` | `standard` | Acos operation in the unary math family. |
| `unaryMathAcosh` | `alpha.unary_math.acosh` | `standard` | Acosh operation in the unary math family. |
| `unaryMathAngle` | `alpha.unary_math.angle` | `standard` | Angle operation in the unary math family. |
| `unaryMathAsin` | `alpha.unary_math.asin` | `standard` | Asin operation in the unary math family. |
| `unaryMathAsinh` | `alpha.unary_math.asinh` | `standard` | Asinh operation in the unary math family. |
| `unaryMathAtan` | `alpha.unary_math.atan` | `standard` | Atan operation in the unary math family. |
| `unaryMathAtanh` | `alpha.unary_math.atanh` | `standard` | Atanh operation in the unary math family. |
| `unaryMathBesselI0` | `alpha.unary_math.bessel-i0` | `standard` | Bessel i0 operation in the unary math family. |
| `unaryMathBesselI1` | `alpha.unary_math.bessel-i1` | `standard` | Bessel i1 operation in the unary math family. |
| `unaryMathBesselJ0` | `alpha.unary_math.bessel-j0` | `standard` | Bessel j0 operation in the unary math family. |
| `unaryMathBesselJ1` | `alpha.unary_math.bessel-j1` | `standard` | Bessel j1 operation in the unary math family. |
| `unaryMathBitwiseNot` | `alpha.unary_math.bitwise-not` | `standard` | Bitwise not operation in the unary math family. |
| `unaryMathCbrt` | `alpha.unary_math.cbrt` | `standard` | Cbrt operation in the unary math family. |
| `unaryMathCeil` | `alpha.unary_math.ceil` | `standard` | Ceil operation in the unary math family. |
| `unaryMathCelu` | `alpha.unary_math.celu` | `standard` | Celu operation in the unary math family. |
| `unaryMathClamp` | `alpha.unary_math.clamp` | `standard` | Clamp operation in the unary math family. |
| `unaryMathClipByValue` | `alpha.unary_math.clip-by-value` | `standard` | Clip by value operation in the unary math family. |
| `unaryMathComplexAbs` | `alpha.unary_math.complex-abs` | `standard` | Complex abs operation in the unary math family. |
| `unaryMathConj` | `alpha.unary_math.conj` | `standard` | Conj operation in the unary math family. |
| `unaryMathCos` | `alpha.unary_math.cos` | `standard` | Cos operation in the unary math family. |
| `unaryMathCosh` | `alpha.unary_math.cosh` | `standard` | Cosh operation in the unary math family. |
| `unaryMathCountLeadingZeros` | `alpha.unary_math.count-leading-zeros` | `standard` | Count leading zeros operation in the unary math family. |
| `unaryMathCountTrailingZeros` | `alpha.unary_math.count-trailing-zeros` | `standard` | Count trailing zeros operation in the unary math family. |
| `unaryMathCube` | `alpha.unary_math.cube` | `standard` | Cube operation in the unary math family. |
| `unaryMathDigamma` | `alpha.unary_math.digamma` | `standard` | Digamma operation in the unary math family. |
| `unaryMathElu` | `alpha.unary_math.elu` | `standard` | Elu operation in the unary math family. |
| `unaryMathErf` | `alpha.unary_math.erf` | `standard` | Erf operation in the unary math family. |
| `unaryMathErfc` | `alpha.unary_math.erfc` | `standard` | Erfc operation in the unary math family. |
| `unaryMathErfinv` | `alpha.unary_math.erfinv` | `standard` | Erfinv operation in the unary math family. |
| `unaryMathExp` | `alpha.unary_math.exp` | `standard` | Exp operation in the unary math family. |
| `unaryMathExp10` | `alpha.unary_math.exp10` | `standard` | Exp10 operation in the unary math family. |
| `unaryMathExp2` | `alpha.unary_math.exp2` | `standard` | Exp2 operation in the unary math family. |
| `unaryMathExpm1` | `alpha.unary_math.expm1` | `standard` | Expm1 operation in the unary math family. |
| `unaryMathFindFirstSet` | `alpha.unary_math.find-first-set` | `standard` | Find first set operation in the unary math family. |
| `unaryMathFloor` | `alpha.unary_math.floor` | `standard` | Floor operation in the unary math family. |
| `unaryMathFrac` | `alpha.unary_math.frac` | `standard` | Frac operation in the unary math family. |
| `unaryMathGamma` | `alpha.unary_math.gamma` | `standard` | Gamma operation in the unary math family. |
| `unaryMathGeluExact` | `alpha.unary_math.gelu-exact` | `standard` | Gelu exact operation in the unary math family. |
| `unaryMathGeluQuick` | `alpha.unary_math.gelu-quick` | `standard` | Gelu quick operation in the unary math family. |
| `unaryMathGeluTanh` | `alpha.unary_math.gelu-tanh` | `standard` | Gelu tanh operation in the unary math family. |
| `unaryMathHardSigmoid` | `alpha.unary_math.hard-sigmoid` | `standard` | Hard sigmoid operation in the unary math family. |
| `unaryMathHardSwish` | `alpha.unary_math.hard-swish` | `standard` | Hard swish operation in the unary math family. |
| `unaryMathImag` | `alpha.unary_math.imag` | `standard` | Imag operation in the unary math family. |
| `unaryMathIsFinite` | `alpha.unary_math.is-finite` | `standard` | Is finite operation in the unary math family. |
| `unaryMathIsInf` | `alpha.unary_math.is-inf` | `standard` | Is inf operation in the unary math family. |
| `unaryMathIsNan` | `alpha.unary_math.is-nan` | `standard` | Is nan operation in the unary math family. |
| `unaryMathIsNormal` | `alpha.unary_math.is-normal` | `standard` | Is normal operation in the unary math family. |
| `unaryMathLeakyRelu` | `alpha.unary_math.leaky-relu` | `standard` | Leaky relu operation in the unary math family. |
| `unaryMathLgamma` | `alpha.unary_math.lgamma` | `standard` | Lgamma operation in the unary math family. |
| `unaryMathLog` | `alpha.unary_math.log` | `standard` | Log operation in the unary math family. |
| `unaryMathLog10` | `alpha.unary_math.log10` | `standard` | Log10 operation in the unary math family. |
| `unaryMathLog1p` | `alpha.unary_math.log1p` | `standard` | Log1p operation in the unary math family. |
| `unaryMathLog2` | `alpha.unary_math.log2` | `standard` | Log2 operation in the unary math family. |
| `unaryMathLogicalNot` | `alpha.unary_math.logical-not` | `standard` | Logical not operation in the unary math family. |
| `unaryMathLogSigmoid` | `alpha.unary_math.log-sigmoid` | `standard` | Log sigmoid operation in the unary math family. |
| `unaryMathMish` | `alpha.unary_math.mish` | `standard` | Mish operation in the unary math family. |
| `unaryMathNeg` | `alpha.unary_math.neg` | `standard` | Neg operation in the unary math family. |
| `unaryMathPopulationCount` | `alpha.unary_math.population-count` | `standard` | Population count operation in the unary math family. |
| `unaryMathPrelu` | `alpha.unary_math.prelu` | `standard` | Prelu operation in the unary math family. |
| `unaryMathReal` | `alpha.unary_math.real` | `standard` | Real operation in the unary math family. |
| `unaryMathReciprocal` | `alpha.unary_math.reciprocal` | `standard` | Reciprocal operation in the unary math family. |
| `unaryMathRelu` | `alpha.unary_math.relu` | `standard` | Relu operation in the unary math family. |
| `unaryMathRelu6` | `alpha.unary_math.relu6` | `standard` | Relu6 operation in the unary math family. |
| `unaryMathReverseBits` | `alpha.unary_math.reverse-bits` | `standard` | Reverse bits operation in the unary math family. |
| `unaryMathRound` | `alpha.unary_math.round` | `standard` | Round operation in the unary math family. |
| `unaryMathRoundEven` | `alpha.unary_math.round-even` | `standard` | Round even operation in the unary math family. |
| `unaryMathRsqrt` | `alpha.unary_math.rsqrt` | `standard` | Rsqrt operation in the unary math family. |
| `unaryMathSelu` | `alpha.unary_math.selu` | `standard` | Selu operation in the unary math family. |
| `unaryMathSigmoid` | `alpha.unary_math.sigmoid` | `standard` | Sigmoid operation in the unary math family. |
| `unaryMathSign` | `alpha.unary_math.sign` | `standard` | Sign operation in the unary math family. |
| `unaryMathSilu` | `alpha.unary_math.silu` | `standard` | Silu operation in the unary math family. |
| `unaryMathSin` | `alpha.unary_math.sin` | `standard` | Sin operation in the unary math family. |
| `unaryMathSinc` | `alpha.unary_math.sinc` | `standard` | Sinc operation in the unary math family. |
| `unaryMathSinh` | `alpha.unary_math.sinh` | `standard` | Sinh operation in the unary math family. |
| `unaryMathSinhc` | `alpha.unary_math.sinhc` | `standard` | Sinhc operation in the unary math family. |
| `unaryMathSoftplus` | `alpha.unary_math.softplus` | `standard` | Softplus operation in the unary math family. |
| `unaryMathSoftsign` | `alpha.unary_math.softsign` | `standard` | Softsign operation in the unary math family. |
| `unaryMathSqrt` | `alpha.unary_math.sqrt` | `standard` | Sqrt operation in the unary math family. |
| `unaryMathSquare` | `alpha.unary_math.square` | `standard` | Square operation in the unary math family. |
| `unaryMathTan` | `alpha.unary_math.tan` | `standard` | Tan operation in the unary math family. |
| `unaryMathTanh` | `alpha.unary_math.tanh` | `standard` | Tanh operation in the unary math family. |
| `unaryMathThreshold` | `alpha.unary_math.threshold` | `standard` | Threshold operation in the unary math family. |
| `unaryMathTrigamma` | `alpha.unary_math.trigamma` | `standard` | Trigamma operation in the unary math family. |
| `unaryMathTrunc` | `alpha.unary_math.trunc` | `standard` | Trunc operation in the unary math family. |

# Helios


## `device` (19)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `deviceCloseDevice` | `helios.device.close-device` | `standard` | Close device operation in the device family. |
| `deviceCreateContext` | `helios.device.create-context` | `standard` | Create context operation in the device family. |
| `deviceDestroyContext` | `helios.device.destroy-context` | `standard` | Destroy context operation in the device family. |
| `deviceEnumerateDevices` | `helios.device.enumerate-devices` | `standard` | Enumerate devices operation in the device family. |
| `deviceOpenDevice` | `helios.device.open-device` | `standard` | Open device operation in the device family. |
| `deviceQueryArchitecture` | `helios.device.query-architecture` | `standard` | Query architecture operation in the device family. |
| `deviceQueryCapabilities` | `helios.device.query-capabilities` | `standard` | Query capabilities operation in the device family. |
| `deviceQueryClocks` | `helios.device.query-clocks` | `standard` | Query clocks operation in the device family. |
| `deviceQueryMemory` | `helios.device.query-memory` | `standard` | Query memory operation in the device family. |
| `deviceQueryPower` | `helios.device.query-power` | `standard` | Query power operation in the device family. |
| `deviceQueryThermals` | `helios.device.query-thermals` | `standard` | Query thermals operation in the device family. |
| `deviceQueryTopology` | `helios.device.query-topology` | `standard` | Query topology operation in the device family. |
| `deviceResetContext` | `helios.device.reset-context` | `standard` | Reset context operation in the device family. |
| `deviceSetDeterminismPolicy` | `helios.device.set-determinism-policy` | `standard` | Set determinism policy operation in the device family. |
| `deviceSetExecutionMode` | `helios.device.set-execution-mode` | `standard` | Set execution mode operation in the device family. |
| `deviceSetMathMode` | `helios.device.set-math-mode` | `standard` | Set math mode operation in the device family. |
| `deviceSetPowerPolicy` | `helios.device.set-power-policy` | `standard` | Set power policy operation in the device family. |
| `deviceSetPrecisionPolicy` | `helios.device.set-precision-policy` | `standard` | Set precision policy operation in the device family. |
| `deviceSynchronizeDevice` | `helios.device.synchronize-device` | `standard` | Synchronize device operation in the device family. |

## `execution` (24)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `executionCancelExecution` | `helios.execution.cancel-execution` | `standard` | Cancel execution operation in the execution family. |
| `executionCollectMetrics` | `helios.execution.collect-metrics` | `standard` | Collect metrics operation in the execution family. |
| `executionCollectResult` | `helios.execution.collect-result` | `standard` | Collect result operation in the execution family. |
| `executionCollectTrace` | `helios.execution.collect-trace` | `standard` | Collect trace operation in the execution family. |
| `executionEnqueue` | `helios.execution.enqueue` | `standard` | Enqueue operation in the execution family. |
| `executionEnqueueBatch` | `helios.execution.enqueue-batch` | `standard` | Enqueue batch operation in the execution family. |
| `executionEnqueueIndirect` | `helios.execution.enqueue-indirect` | `standard` | Enqueue indirect operation in the execution family. |
| `executionExecute` | `helios.execution.execute` | `standard` | Execute operation in the execution family. |
| `executionExecuteActorWave` | `helios.execution.execute-actor-wave` | `standard` | Execute actor wave operation in the execution family. |
| `executionExecuteAsync` | `helios.execution.execute-async` | `standard` | Execute async operation in the execution family. |
| `executionExecuteGraph` | `helios.execution.execute-graph` | `standard` | Execute graph operation in the execution family. |
| `executionExecuteResident` | `helios.execution.execute-resident` | `standard` | Execute resident operation in the execution family. |
| `executionExecuteUntilQuiescent` | `helios.execution.execute-until-quiescent` | `standard` | Execute until quiescent operation in the execution family. |
| `executionFallbackExecution` | `helios.execution.fallback-execution` | `standard` | Fallback execution operation in the execution family. |
| `executionPollExecution` | `helios.execution.poll-execution` | `standard` | Poll execution operation in the execution family. |
| `executionRecoverExecution` | `helios.execution.recover-execution` | `standard` | Recover execution operation in the execution family. |
| `executionRetryExecution` | `helios.execution.retry-execution` | `standard` | Retry execution operation in the execution family. |
| `executionSetDeadline` | `helios.execution.set-deadline` | `standard` | Set deadline operation in the execution family. |
| `executionSetErrorBudget` | `helios.execution.set-error-budget` | `standard` | Set error budget operation in the execution family. |
| `executionSetEventBudget` | `helios.execution.set-event-budget` | `standard` | Set event budget operation in the execution family. |
| `executionSetExecutionBudget` | `helios.execution.set-execution-budget` | `standard` | Set execution budget operation in the execution family. |
| `executionSetExecutionPriority` | `helios.execution.set-execution-priority` | `standard` | Set execution priority operation in the execution family. |
| `executionSetMemoryBudget` | `helios.execution.set-memory-budget` | `standard` | Set memory budget operation in the execution family. |
| `executionWaitExecution` | `helios.execution.wait-execution` | `standard` | Wait execution operation in the execution family. |

## `fusion` (18)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `fusionDefuseForDeterminism` | `helios.fusion.defuse-for-determinism` | `standard` | Defuse for determinism operation in the fusion family. |
| `fusionDefuseForOccupancy` | `helios.fusion.defuse-for-occupancy` | `standard` | Defuse for occupancy operation in the fusion family. |
| `fusionDefuseForRegisterPressure` | `helios.fusion.defuse-for-register-pressure` | `standard` | Defuse for register pressure operation in the fusion family. |
| `fusionFuseActorTransition` | `helios.fusion.fuse-actor-transition` | `standard` | Fuse actor transition operation in the fusion family. |
| `fusionFuseAttention` | `helios.fusion.fuse-attention` | `standard` | Fuse attention operation in the fusion family. |
| `fusionFuseDequantize` | `helios.fusion.fuse-dequantize` | `standard` | Fuse dequantize operation in the fusion family. |
| `fusionFuseElementwiseChain` | `helios.fusion.fuse-elementwise-chain` | `standard` | Fuse elementwise chain operation in the fusion family. |
| `fusionFuseFft` | `helios.fusion.fuse-fft` | `standard` | Fuse fft operation in the fusion family. |
| `fusionFuseGemmEpilogue` | `helios.fusion.fuse-gemm-epilogue` | `standard` | Fuse gemm epilogue operation in the fusion family. |
| `fusionFuseGemmPrologue` | `helios.fusion.fuse-gemm-prologue` | `standard` | Fuse gemm prologue operation in the fusion family. |
| `fusionFuseMemoryLookup` | `helios.fusion.fuse-memory-lookup` | `standard` | Fuse memory lookup operation in the fusion family. |
| `fusionFuseNormResidual` | `helios.fusion.fuse-norm-residual` | `standard` | Fuse norm residual operation in the fusion family. |
| `fusionFuseOptimizer` | `helios.fusion.fuse-optimizer` | `standard` | Fuse optimizer operation in the fusion family. |
| `fusionFuseQuantize` | `helios.fusion.fuse-quantize` | `standard` | Fuse quantize operation in the fusion family. |
| `fusionFuseReductionEpilogue` | `helios.fusion.fuse-reduction-epilogue` | `standard` | Fuse reduction epilogue operation in the fusion family. |
| `fusionFuseSampling` | `helios.fusion.fuse-sampling` | `standard` | Fuse sampling operation in the fusion family. |
| `fusionFuseScan` | `helios.fusion.fuse-scan` | `standard` | Fuse scan operation in the fusion family. |
| `fusionFuseSparsePipeline` | `helios.fusion.fuse-sparse-pipeline` | `standard` | Fuse sparse pipeline operation in the fusion family. |

## `graph` (20)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `graphAddGraphDependency` | `helios.graph.add-graph-dependency` | `standard` | Add graph dependency operation in the graph family. |
| `graphAddGraphNode` | `helios.graph.add-graph-node` | `standard` | Add graph node operation in the graph family. |
| `graphCaptureGraph` | `helios.graph.capture-graph` | `standard` | Capture graph operation in the graph family. |
| `graphCloneGraph` | `helios.graph.clone-graph` | `standard` | Clone graph operation in the graph family. |
| `graphCompileGraph` | `helios.graph.compile-graph` | `standard` | Compile graph operation in the graph family. |
| `graphCreateGraph` | `helios.graph.create-graph` | `standard` | Create graph operation in the graph family. |
| `graphDestroyGraph` | `helios.graph.destroy-graph` | `standard` | Destroy graph operation in the graph family. |
| `graphFinalizeGraph` | `helios.graph.finalize-graph` | `standard` | Finalize graph operation in the graph family. |
| `graphFuseGraph` | `helios.graph.fuse-graph` | `standard` | Fuse graph operation in the graph family. |
| `graphInstantiateGraph` | `helios.graph.instantiate-graph` | `standard` | Instantiate graph operation in the graph family. |
| `graphOptimizeGraph` | `helios.graph.optimize-graph` | `standard` | Optimize graph operation in the graph family. |
| `graphPartitionGraph` | `helios.graph.partition-graph` | `standard` | Partition graph operation in the graph family. |
| `graphProfileGraph` | `helios.graph.profile-graph` | `standard` | Profile graph operation in the graph family. |
| `graphRemoveGraphDependency` | `helios.graph.remove-graph-dependency` | `standard` | Remove graph dependency operation in the graph family. |
| `graphRemoveGraphNode` | `helios.graph.remove-graph-node` | `standard` | Remove graph node operation in the graph family. |
| `graphReplayGraph` | `helios.graph.replay-graph` | `standard` | Replay graph operation in the graph family. |
| `graphScheduleGraph` | `helios.graph.schedule-graph` | `standard` | Schedule graph operation in the graph family. |
| `graphSerializeGraph` | `helios.graph.serialize-graph` | `standard` | Serialize graph operation in the graph family. |
| `graphUpdateGraphParameters` | `helios.graph.update-graph-parameters` | `standard` | Update graph parameters operation in the graph family. |
| `graphValidateGraph` | `helios.graph.validate-graph` | `standard` | Validate graph operation in the graph family. |

## `kernel_selection` (20)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `kernelSelectionAutotuneKernel` | `helios.kernel_selection.autotune-kernel` | `standard` | Autotune kernel operation in the kernel selection family. |
| `kernelSelectionBenchmarkKernel` | `helios.kernel_selection.benchmark-kernel` | `standard` | Benchmark kernel operation in the kernel selection family. |
| `kernelSelectionCacheTuningResult` | `helios.kernel_selection.cache-tuning-result` | `standard` | Cache tuning result operation in the kernel selection family. |
| `kernelSelectionEnumerateKernels` | `helios.kernel_selection.enumerate-kernels` | `standard` | Enumerate kernels operation in the kernel selection family. |
| `kernelSelectionExplainKernelChoice` | `helios.kernel_selection.explain-kernel-choice` | `standard` | Explain kernel choice operation in the kernel selection family. |
| `kernelSelectionInvalidateTuningResult` | `helios.kernel_selection.invalidate-tuning-result` | `standard` | Invalidate tuning result operation in the kernel selection family. |
| `kernelSelectionMatchKernel` | `helios.kernel_selection.match-kernel` | `standard` | Match kernel operation in the kernel selection family. |
| `kernelSelectionRegisterKernel` | `helios.kernel_selection.register-kernel` | `standard` | Register kernel operation in the kernel selection family. |
| `kernelSelectionScoreKernel` | `helios.kernel_selection.score-kernel` | `standard` | Score kernel operation in the kernel selection family. |
| `kernelSelectionSelectApproximateKernel` | `helios.kernel_selection.select-approximate-kernel` | `standard` | Select approximate kernel operation in the kernel selection family. |
| `kernelSelectionSelectDeterministicKernel` | `helios.kernel_selection.select-deterministic-kernel` | `standard` | Select deterministic kernel operation in the kernel selection family. |
| `kernelSelectionSelectExactKernel` | `helios.kernel_selection.select-exact-kernel` | `standard` | Select exact kernel operation in the kernel selection family. |
| `kernelSelectionSelectFallbackKernel` | `helios.kernel_selection.select-fallback-kernel` | `standard` | Select fallback kernel operation in the kernel selection family. |
| `kernelSelectionSelectGroupedKernel` | `helios.kernel_selection.select-grouped-kernel` | `standard` | Select grouped kernel operation in the kernel selection family. |
| `kernelSelectionSelectKernel` | `helios.kernel_selection.select-kernel` | `standard` | Select kernel operation in the kernel selection family. |
| `kernelSelectionSelectLowBitKernel` | `helios.kernel_selection.select-low-bit-kernel` | `standard` | Select low bit kernel operation in the kernel selection family. |
| `kernelSelectionSelectPersistentKernel` | `helios.kernel_selection.select-persistent-kernel` | `standard` | Select persistent kernel operation in the kernel selection family. |
| `kernelSelectionSelectResidentKernel` | `helios.kernel_selection.select-resident-kernel` | `standard` | Select resident kernel operation in the kernel selection family. |
| `kernelSelectionSelectSparseKernel` | `helios.kernel_selection.select-sparse-kernel` | `standard` | Select sparse kernel operation in the kernel selection family. |
| `kernelSelectionUnregisterKernel` | `helios.kernel_selection.unregister-kernel` | `standard` | Unregister kernel operation in the kernel selection family. |

## `observability` (20)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `observabilityBeginTrace` | `helios.observability.begin-trace` | `standard` | Begin trace operation in the observability family. |
| `observabilityCompareTrace` | `helios.observability.compare-trace` | `standard` | Compare trace operation in the observability family. |
| `observabilityDumpIr` | `helios.observability.dump-ir` | `standard` | Dump ir operation in the observability family. |
| `observabilityDumpLaunch` | `helios.observability.dump-launch` | `standard` | Dump launch operation in the observability family. |
| `observabilityDumpMemoryMap` | `helios.observability.dump-memory-map` | `standard` | Dump memory map operation in the observability family. |
| `observabilityDumpProgram` | `helios.observability.dump-program` | `standard` | Dump program operation in the observability family. |
| `observabilityDumpSass` | `helios.observability.dump-sass` | `standard` | Dump sass operation in the observability family. |
| `observabilityDumpTimeline` | `helios.observability.dump-timeline` | `standard` | Dump timeline operation in the observability family. |
| `observabilityEndTrace` | `helios.observability.end-trace` | `standard` | End trace operation in the observability family. |
| `observabilityMarkRange` | `helios.observability.mark-range` | `standard` | Mark range operation in the observability family. |
| `observabilityRecordCounter` | `helios.observability.record-counter` | `standard` | Record counter operation in the observability family. |
| `observabilityRecordEnergyEstimate` | `helios.observability.record-energy-estimate` | `standard` | Record energy estimate operation in the observability family. |
| `observabilityRecordGradientStats` | `helios.observability.record-gradient-stats` | `standard` | Record gradient stats operation in the observability family. |
| `observabilityRecordKernelStats` | `helios.observability.record-kernel-stats` | `standard` | Record kernel stats operation in the observability family. |
| `observabilityRecordMemoryStats` | `helios.observability.record-memory-stats` | `standard` | Record memory stats operation in the observability family. |
| `observabilityRecordNumericalError` | `helios.observability.record-numerical-error` | `standard` | Record numerical error operation in the observability family. |
| `observabilityRecordQueueStats` | `helios.observability.record-queue-stats` | `standard` | Record queue stats operation in the observability family. |
| `observabilityRecordRoutingStats` | `helios.observability.record-routing-stats` | `standard` | Record routing stats operation in the observability family. |
| `observabilityRecordTensorStats` | `helios.observability.record-tensor-stats` | `standard` | Record tensor stats operation in the observability family. |
| `observabilityReplayTrace` | `helios.observability.replay-trace` | `standard` | Replay trace operation in the observability family. |

## `planning` (30)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `planningEstimateBytes` | `helios.planning.estimate-bytes` | `standard` | Estimate bytes operation in the planning family. |
| `planningEstimateCommunication` | `helios.planning.estimate-communication` | `standard` | Estimate communication operation in the planning family. |
| `planningEstimateEnergy` | `helios.planning.estimate-energy` | `standard` | Estimate energy operation in the planning family. |
| `planningEstimateFlops` | `helios.planning.estimate-flops` | `standard` | Estimate flops operation in the planning family. |
| `planningEstimateLatency` | `helios.planning.estimate-latency` | `standard` | Estimate latency operation in the planning family. |
| `planningEstimateNumericalError` | `helios.planning.estimate-numerical-error` | `standard` | Estimate numerical error operation in the planning family. |
| `planningEstimateOccupancy` | `helios.planning.estimate-occupancy` | `standard` | Estimate occupancy operation in the planning family. |
| `planningEstimateRegisterPressure` | `helios.planning.estimate-register-pressure` | `standard` | Estimate register pressure operation in the planning family. |
| `planningEstimateSharedMemory` | `helios.planning.estimate-shared-memory` | `standard` | Estimate shared memory operation in the planning family. |
| `planningInferAliases` | `helios.planning.infer-aliases` | `standard` | Infer aliases operation in the planning family. |
| `planningInferEffects` | `helios.planning.infer-effects` | `standard` | Infer effects operation in the planning family. |
| `planningInferLayouts` | `helios.planning.infer-layouts` | `standard` | Infer layouts operation in the planning family. |
| `planningInferShapes` | `helios.planning.infer-shapes` | `standard` | Infer shapes operation in the planning family. |
| `planningInferTypes` | `helios.planning.infer-types` | `standard` | Infer types operation in the planning family. |
| `planningPlanActorPlacement` | `helios.planning.plan-actor-placement` | `standard` | Plan actor placement operation in the planning family. |
| `planningPlanAutotuneSearch` | `helios.planning.plan-autotune-search` | `standard` | Plan autotune search operation in the planning family. |
| `planningPlanCheckpoint` | `helios.planning.plan-checkpoint` | `standard` | Plan checkpoint operation in the planning family. |
| `planningPlanCollectives` | `helios.planning.plan-collectives` | `standard` | Plan collectives operation in the planning family. |
| `planningPlanEviction` | `helios.planning.plan-eviction` | `standard` | Plan eviction operation in the planning family. |
| `planningPlanFusion` | `helios.planning.plan-fusion` | `standard` | Plan fusion operation in the planning family. |
| `planningPlanMemory` | `helios.planning.plan-memory` | `standard` | Plan memory operation in the planning family. |
| `planningPlanOutputHead` | `helios.planning.plan-output-head` | `standard` | Plan output head operation in the planning family. |
| `planningPlanPipeline` | `helios.planning.plan-pipeline` | `standard` | Plan pipeline operation in the planning family. |
| `planningPlanPrecision` | `helios.planning.plan-precision` | `standard` | Plan precision operation in the planning family. |
| `planningPlanPrefetch` | `helios.planning.plan-prefetch` | `standard` | Plan prefetch operation in the planning family. |
| `planningPlanRecompute` | `helios.planning.plan-recompute` | `standard` | Plan recompute operation in the planning family. |
| `planningPlanResidentSet` | `helios.planning.plan-resident-set` | `standard` | Plan resident set operation in the planning family. |
| `planningPlanSparsity` | `helios.planning.plan-sparsity` | `standard` | Plan sparsity operation in the planning family. |
| `planningPlanTiling` | `helios.planning.plan-tiling` | `standard` | Plan tiling operation in the planning family. |
| `planningPlanVectorization` | `helios.planning.plan-vectorization` | `standard` | Plan vectorization operation in the planning family. |

## `program` (16)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `programCacheProgram` | `helios.program.cache-program` | `standard` | Cache program operation in the program family. |
| `programCompileProgram` | `helios.program.compile-program` | `standard` | Compile program operation in the program family. |
| `programCreateProgram` | `helios.program.create-program` | `standard` | Create program operation in the program family. |
| `programDeserializeProgram` | `helios.program.deserialize-program` | `standard` | Deserialize program operation in the program family. |
| `programDestroyProgram` | `helios.program.destroy-program` | `standard` | Destroy program operation in the program family. |
| `programHashProgram` | `helios.program.hash-program` | `standard` | Hash program operation in the program family. |
| `programInstallResidentProgram` | `helios.program.install-resident-program` | `standard` | Install resident program operation in the program family. |
| `programLinkProgram` | `helios.program.link-program` | `standard` | Link program operation in the program family. |
| `programLoadProgram` | `helios.program.load-program` | `standard` | Load program operation in the program family. |
| `programQueryProgramResources` | `helios.program.query-program-resources` | `standard` | Query program resources operation in the program family. |
| `programRefreshResidentWeights` | `helios.program.refresh-resident-weights` | `standard` | Refresh resident weights operation in the program family. |
| `programSerializeProgram` | `helios.program.serialize-program` | `standard` | Serialize program operation in the program family. |
| `programSpecializeProgram` | `helios.program.specialize-program` | `standard` | Specialize program operation in the program family. |
| `programUninstallResidentProgram` | `helios.program.uninstall-resident-program` | `standard` | Uninstall resident program operation in the program family. |
| `programUnloadProgram` | `helios.program.unload-program` | `standard` | Unload program operation in the program family. |
| `programValidateProgram` | `helios.program.validate-program` | `standard` | Validate program operation in the program family. |

## `tensor_storage` (21)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `tensorStorageAliasTensor` | `helios.tensor_storage.alias-tensor` | `standard` | Alias tensor operation in the tensor storage family. |
| `tensorStorageCopyTensor` | `helios.tensor_storage.copy-tensor` | `standard` | Copy tensor operation in the tensor storage family. |
| `tensorStorageCreateTensor` | `helios.tensor_storage.create-tensor` | `standard` | Create tensor operation in the tensor storage family. |
| `tensorStorageDestroyTensor` | `helios.tensor_storage.destroy-tensor` | `standard` | Destroy tensor operation in the tensor storage family. |
| `tensorStorageEvictTensor` | `helios.tensor_storage.evict-tensor` | `standard` | Evict tensor operation in the tensor storage family. |
| `tensorStorageExportTensor` | `helios.tensor_storage.export-tensor` | `standard` | Export tensor operation in the tensor storage family. |
| `tensorStorageFillTensor` | `helios.tensor_storage.fill-tensor` | `standard` | Fill tensor operation in the tensor storage family. |
| `tensorStorageImportTensor` | `helios.tensor_storage.import-tensor` | `standard` | Import tensor operation in the tensor storage family. |
| `tensorStorageMapTensor` | `helios.tensor_storage.map-tensor` | `standard` | Map tensor operation in the tensor storage family. |
| `tensorStorageMigrateTensor` | `helios.tensor_storage.migrate-tensor` | `standard` | Migrate tensor operation in the tensor storage family. |
| `tensorStoragePinTensor` | `helios.tensor_storage.pin-tensor` | `standard` | Pin tensor operation in the tensor storage family. |
| `tensorStoragePrefetchTensor` | `helios.tensor_storage.prefetch-tensor` | `standard` | Prefetch tensor operation in the tensor storage family. |
| `tensorStorageQueryTensorLayout` | `helios.tensor_storage.query-tensor-layout` | `standard` | Query tensor layout operation in the tensor storage family. |
| `tensorStorageQueryTensorResidency` | `helios.tensor_storage.query-tensor-residency` | `standard` | Query tensor residency operation in the tensor storage family. |
| `tensorStorageRestoreTensor` | `helios.tensor_storage.restore-tensor` | `standard` | Restore tensor operation in the tensor storage family. |
| `tensorStorageSnapshotTensor` | `helios.tensor_storage.snapshot-tensor` | `standard` | Snapshot tensor operation in the tensor storage family. |
| `tensorStorageSubviewTensor` | `helios.tensor_storage.subview-tensor` | `standard` | Subview tensor operation in the tensor storage family. |
| `tensorStorageUnmapTensor` | `helios.tensor_storage.unmap-tensor` | `standard` | Unmap tensor operation in the tensor storage family. |
| `tensorStorageUnpinTensor` | `helios.tensor_storage.unpin-tensor` | `standard` | Unpin tensor operation in the tensor storage family. |
| `tensorStorageValidateTensor` | `helios.tensor_storage.validate-tensor` | `standard` | Validate tensor operation in the tensor storage family. |
| `tensorStorageZeroTensor` | `helios.tensor_storage.zero-tensor` | `standard` | Zero tensor operation in the tensor storage family. |

# Prometheus


## `arith` (38)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `arithAbs` | `prometheus.arith.abs` | `standard` | Abs operation in the arith family. |
| `arithAdd` | `prometheus.arith.add` | `standard` | Add operation in the arith family. |
| `arithBitcast` | `prometheus.arith.bitcast` | `standard` | Bitcast operation in the arith family. |
| `arithBitExtract` | `prometheus.arith.bit-extract` | `standard` | Bit extract operation in the arith family. |
| `arithBitInsert` | `prometheus.arith.bit-insert` | `standard` | Bit insert operation in the arith family. |
| `arithBitwise` | `prometheus.arith.bitwise` | `standard` | Bitwise operation in the arith family. |
| `arithCeil` | `prometheus.arith.ceil` | `standard` | Ceil operation in the arith family. |
| `arithClamp` | `prometheus.arith.clamp` | `standard` | Clamp operation in the arith family. |
| `arithCompare` | `prometheus.arith.compare` | `standard` | Compare operation in the arith family. |
| `arithConvert` | `prometheus.arith.convert` | `standard` | Convert operation in the arith family. |
| `arithCos` | `prometheus.arith.cos` | `standard` | Cos operation in the arith family. |
| `arithDiv` | `prometheus.arith.div` | `standard` | Div operation in the arith family. |
| `arithErf` | `prometheus.arith.erf` | `standard` | Erf operation in the arith family. |
| `arithExp` | `prometheus.arith.exp` | `standard` | Exp operation in the arith family. |
| `arithExp2` | `prometheus.arith.exp2` | `standard` | Exp2 operation in the arith family. |
| `arithFloor` | `prometheus.arith.floor` | `standard` | Floor operation in the arith family. |
| `arithFma` | `prometheus.arith.fma` | `standard` | Fma operation in the arith family. |
| `arithFunnelShift` | `prometheus.arith.funnel-shift` | `standard` | Funnel shift operation in the arith family. |
| `arithLeadingZeros` | `prometheus.arith.leading-zeros` | `standard` | Leading zeros operation in the arith family. |
| `arithLog` | `prometheus.arith.log` | `standard` | Log operation in the arith family. |
| `arithLog2` | `prometheus.arith.log2` | `standard` | Log2 operation in the arith family. |
| `arithLogical` | `prometheus.arith.logical` | `standard` | Logical operation in the arith family. |
| `arithMax` | `prometheus.arith.max` | `standard` | Max operation in the arith family. |
| `arithMin` | `prometheus.arith.min` | `standard` | Min operation in the arith family. |
| `arithMul` | `prometheus.arith.mul` | `standard` | Mul operation in the arith family. |
| `arithNeg` | `prometheus.arith.neg` | `standard` | Neg operation in the arith family. |
| `arithPermuteBytes` | `prometheus.arith.permute-bytes` | `standard` | Permute bytes operation in the arith family. |
| `arithPopulationCount` | `prometheus.arith.population-count` | `standard` | Population count operation in the arith family. |
| `arithRem` | `prometheus.arith.rem` | `standard` | Rem operation in the arith family. |
| `arithRound` | `prometheus.arith.round` | `standard` | Round operation in the arith family. |
| `arithRsqrt` | `prometheus.arith.rsqrt` | `standard` | Rsqrt operation in the arith family. |
| `arithSaturate` | `prometheus.arith.saturate` | `standard` | Saturate operation in the arith family. |
| `arithSelect` | `prometheus.arith.select` | `standard` | Select operation in the arith family. |
| `arithSin` | `prometheus.arith.sin` | `standard` | Sin operation in the arith family. |
| `arithSqrt` | `prometheus.arith.sqrt` | `standard` | Sqrt operation in the arith family. |
| `arithSub` | `prometheus.arith.sub` | `standard` | Sub operation in the arith family. |
| `arithTanh` | `prometheus.arith.tanh` | `standard` | Tanh operation in the arith family. |
| `arithTrailingZeros` | `prometheus.arith.trailing-zeros` | `standard` | Trailing zeros operation in the arith family. |

## `async_actor` (34)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `asyncActorActorAwait` | `prometheus.async_actor.actor-await` | `standard` | Actor await operation in the async actor family. |
| `asyncActorActorCheckpoint` | `prometheus.async_actor.actor-checkpoint` | `standard` | Actor checkpoint operation in the async actor family. |
| `asyncActorActorQuiesce` | `prometheus.async_actor.actor-quiesce` | `standard` | Actor quiesce operation in the async actor family. |
| `asyncActorActorRestore` | `prometheus.async_actor.actor-restore` | `standard` | Actor restore operation in the async actor family. |
| `asyncActorActorSleep` | `prometheus.async_actor.actor-sleep` | `standard` | Actor sleep operation in the async actor family. |
| `asyncActorActorStateCommit` | `prometheus.async_actor.actor-state-commit` | `standard` | Actor state commit operation in the async actor family. |
| `asyncActorActorStateRead` | `prometheus.async_actor.actor-state-read` | `standard` | Actor state read operation in the async actor family. |
| `asyncActorActorStateRollback` | `prometheus.async_actor.actor-state-rollback` | `standard` | Actor state rollback operation in the async actor family. |
| `asyncActorActorStateWrite` | `prometheus.async_actor.actor-state-write` | `standard` | Actor state write operation in the async actor family. |
| `asyncActorActorWake` | `prometheus.async_actor.actor-wake` | `standard` | Actor wake operation in the async actor family. |
| `asyncActorActorYield` | `prometheus.async_actor.actor-yield` | `standard` | Actor yield operation in the async actor family. |
| `asyncActorBudgetConsume` | `prometheus.async_actor.budget-consume` | `standard` | Budget consume operation in the async actor family. |
| `asyncActorDeadlineCheck` | `prometheus.async_actor.deadline-check` | `standard` | Deadline check operation in the async actor family. |
| `asyncActorEventConsume` | `prometheus.async_actor.event-consume` | `standard` | Event consume operation in the async actor family. |
| `asyncActorEventCreate` | `prometheus.async_actor.event-create` | `standard` | Event create operation in the async actor family. |
| `asyncActorEventCreditAcquire` | `prometheus.async_actor.event-credit-acquire` | `standard` | Event credit acquire operation in the async actor family. |
| `asyncActorEventCreditRelease` | `prometheus.async_actor.event-credit-release` | `standard` | Event credit release operation in the async actor family. |
| `asyncActorEventDrop` | `prometheus.async_actor.event-drop` | `standard` | Event drop operation in the async actor family. |
| `asyncActorEventEmit` | `prometheus.async_actor.event-emit` | `standard` | Event emit operation in the async actor family. |
| `asyncActorEventForward` | `prometheus.async_actor.event-forward` | `standard` | Event forward operation in the async actor family. |
| `asyncActorEventMerge` | `prometheus.async_actor.event-merge` | `standard` | Event merge operation in the async actor family. |
| `asyncActorEventSplit` | `prometheus.async_actor.event-split` | `standard` | Event split operation in the async actor family. |
| `asyncActorLogicalWaveBegin` | `prometheus.async_actor.logical-wave-begin` | `standard` | Logical wave begin operation in the async actor family. |
| `asyncActorLogicalWaveEnd` | `prometheus.async_actor.logical-wave-end` | `standard` | Logical wave end operation in the async actor family. |
| `asyncActorMailboxCancel` | `prometheus.async_actor.mailbox-cancel` | `standard` | Mailbox cancel operation in the async actor family. |
| `asyncActorMailboxCommit` | `prometheus.async_actor.mailbox-commit` | `standard` | Mailbox commit operation in the async actor family. |
| `asyncActorMailboxCreate` | `prometheus.async_actor.mailbox-create` | `standard` | Mailbox create operation in the async actor family. |
| `asyncActorMailboxDestroy` | `prometheus.async_actor.mailbox-destroy` | `standard` | Mailbox destroy operation in the async actor family. |
| `asyncActorMailboxPeek` | `prometheus.async_actor.mailbox-peek` | `standard` | Mailbox peek operation in the async actor family. |
| `asyncActorMailboxReceive` | `prometheus.async_actor.mailbox-receive` | `standard` | Mailbox receive operation in the async actor family. |
| `asyncActorMailboxReserve` | `prometheus.async_actor.mailbox-reserve` | `standard` | Mailbox reserve operation in the async actor family. |
| `asyncActorMailboxSend` | `prometheus.async_actor.mailbox-send` | `standard` | Mailbox send operation in the async actor family. |
| `asyncActorMailboxTryReceive` | `prometheus.async_actor.mailbox-try-receive` | `standard` | Mailbox try receive operation in the async actor family. |
| `asyncActorQuiescenceCheck` | `prometheus.async_actor.quiescence-check` | `research` | Quiescence check operation in the async actor family. |

## `control_flow` (23)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `controlFlowAffineLoop` | `prometheus.control_flow.affine-loop` | `standard` | Affine loop operation in the control flow family. |
| `controlFlowBranch` | `prometheus.control_flow.branch` | `standard` | Branch operation in the control flow family. |
| `controlFlowBreakLoop` | `prometheus.control_flow.break-loop` | `standard` | Break loop operation in the control flow family. |
| `controlFlowConditionalBranch` | `prometheus.control_flow.conditional-branch` | `standard` | Conditional branch operation in the control flow family. |
| `controlFlowContinueLoop` | `prometheus.control_flow.continue-loop` | `standard` | Continue loop operation in the control flow family. |
| `controlFlowDoWhileLoop` | `prometheus.control_flow.do-while-loop` | `standard` | Do while loop operation in the control flow family. |
| `controlFlowFallbackRegion` | `prometheus.control_flow.fallback-region` | `standard` | Fallback region operation in the control flow family. |
| `controlFlowForLoop` | `prometheus.control_flow.for-loop` | `standard` | For loop operation in the control flow family. |
| `controlFlowGuardRegion` | `prometheus.control_flow.guard-region` | `standard` | Guard region operation in the control flow family. |
| `controlFlowIfRegion` | `prometheus.control_flow.if-region` | `standard` | If region operation in the control flow family. |
| `controlFlowLoop` | `prometheus.control_flow.loop` | `standard` | Loop operation in the control flow family. |
| `controlFlowParallelLoop` | `prometheus.control_flow.parallel-loop` | `standard` | Parallel loop operation in the control flow family. |
| `controlFlowPeelLoop` | `prometheus.control_flow.peel-loop` | `standard` | Peel loop operation in the control flow family. |
| `controlFlowPipelineLoop` | `prometheus.control_flow.pipeline-loop` | `standard` | Pipeline loop operation in the control flow family. |
| `controlFlowRetryRegion` | `prometheus.control_flow.retry-region` | `standard` | Retry region operation in the control flow family. |
| `controlFlowSelectRegion` | `prometheus.control_flow.select-region` | `standard` | Select region operation in the control flow family. |
| `controlFlowSoftwarePipeline` | `prometheus.control_flow.software-pipeline` | `standard` | Software pipeline operation in the control flow family. |
| `controlFlowSpeculateRegion` | `prometheus.control_flow.speculate-region` | `standard` | Speculate region operation in the control flow family. |
| `controlFlowSwitch` | `prometheus.control_flow.switch` | `standard` | Switch operation in the control flow family. |
| `controlFlowTileLoop` | `prometheus.control_flow.tile-loop` | `standard` | Tile loop operation in the control flow family. |
| `controlFlowTryRegion` | `prometheus.control_flow.try-region` | `standard` | Try region operation in the control flow family. |
| `controlFlowUnroll` | `prometheus.control_flow.unroll` | `standard` | Unroll operation in the control flow family. |
| `controlFlowWhileLoop` | `prometheus.control_flow.while-loop` | `standard` | While loop operation in the control flow family. |

## `cost_model` (22)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `costModelEstimateAtomicContention` | `prometheus.cost_model.estimate-atomic-contention` | `standard` | Estimate atomic contention operation in the cost model family. |
| `costModelEstimateBankConflicts` | `prometheus.cost_model.estimate-bank-conflicts` | `standard` | Estimate bank conflicts operation in the cost model family. |
| `costModelEstimateBarrierCost` | `prometheus.cost_model.estimate-barrier-cost` | `standard` | Estimate barrier cost operation in the cost model family. |
| `costModelEstimateCoalescing` | `prometheus.cost_model.estimate-coalescing` | `standard` | Estimate coalescing operation in the cost model family. |
| `costModelEstimateCycles` | `prometheus.cost_model.estimate-cycles` | `standard` | Estimate cycles operation in the cost model family. |
| `costModelEstimateDivergence` | `prometheus.cost_model.estimate-divergence` | `standard` | Estimate divergence operation in the cost model family. |
| `costModelEstimateEnergy` | `prometheus.cost_model.estimate-energy` | `standard` | Estimate energy operation in the cost model family. |
| `costModelEstimateGlobalBytes` | `prometheus.cost_model.estimate-global-bytes` | `standard` | Estimate global bytes operation in the cost model family. |
| `costModelEstimateInstructions` | `prometheus.cost_model.estimate-instructions` | `standard` | Estimate instructions operation in the cost model family. |
| `costModelEstimateL2Bytes` | `prometheus.cost_model.estimate-l2-bytes` | `standard` | Estimate l2 bytes operation in the cost model family. |
| `costModelEstimateLatency` | `prometheus.cost_model.estimate-latency` | `standard` | Estimate latency operation in the cost model family. |
| `costModelEstimateLocalMemory` | `prometheus.cost_model.estimate-local-memory` | `standard` | Estimate local memory operation in the cost model family. |
| `costModelEstimateNumericalError` | `prometheus.cost_model.estimate-numerical-error` | `standard` | Estimate numerical error operation in the cost model family. |
| `costModelEstimateOccupancy` | `prometheus.cost_model.estimate-occupancy` | `standard` | Estimate occupancy operation in the cost model family. |
| `costModelEstimateRegisters` | `prometheus.cost_model.estimate-registers` | `standard` | Estimate registers operation in the cost model family. |
| `costModelEstimateSharedMemory` | `prometheus.cost_model.estimate-shared-memory` | `standard` | Estimate shared memory operation in the cost model family. |
| `costModelEstimateThroughput` | `prometheus.cost_model.estimate-throughput` | `standard` | Estimate throughput operation in the cost model family. |
| `costModelRankSchedules` | `prometheus.cost_model.rank-schedules` | `standard` | Rank schedules operation in the cost model family. |
| `costModelSearchEpilogueFusion` | `prometheus.cost_model.search-epilogue-fusion` | `standard` | Search epilogue fusion operation in the cost model family. |
| `costModelSearchPipelineDepth` | `prometheus.cost_model.search-pipeline-depth` | `standard` | Search pipeline depth operation in the cost model family. |
| `costModelSearchTileShape` | `prometheus.cost_model.search-tile-shape` | `standard` | Search tile shape operation in the cost model family. |
| `costModelSearchWarpAssignment` | `prometheus.cost_model.search-warp-assignment` | `standard` | Search warp assignment operation in the cost model family. |

## `ir_module` (22)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `irModuleActor` | `prometheus.ir_module.actor` | `standard` | Actor operation in the ir module family. |
| `irModuleArgument` | `prometheus.ir_module.argument` | `standard` | Argument operation in the ir module family. |
| `irModuleAssert` | `prometheus.ir_module.assert` | `standard` | Assert operation in the ir module family. |
| `irModuleAssume` | `prometheus.ir_module.assume` | `standard` | Assume operation in the ir module family. |
| `irModuleBlock` | `prometheus.ir_module.block` | `standard` | Block operation in the ir module family. |
| `irModuleCall` | `prometheus.ir_module.call` | `standard` | Call operation in the ir module family. |
| `irModuleConstant` | `prometheus.ir_module.constant` | `standard` | Constant operation in the ir module family. |
| `irModuleDebugValue` | `prometheus.ir_module.debug-value` | `standard` | Debug value operation in the ir module family. |
| `irModuleExternal` | `prometheus.ir_module.external` | `standard` | External operation in the ir module family. |
| `irModuleFunction` | `prometheus.ir_module.function` | `standard` | Function operation in the ir module family. |
| `irModuleGlobal` | `prometheus.ir_module.global` | `standard` | Global operation in the ir module family. |
| `irModuleKernel` | `prometheus.ir_module.kernel` | `standard` | Kernel operation in the ir module family. |
| `irModuleMetadata` | `prometheus.ir_module.metadata` | `standard` | Metadata operation in the ir module family. |
| `irModuleModule` | `prometheus.ir_module.module` | `standard` | Module operation in the ir module family. |
| `irModuleRegion` | `prometheus.ir_module.region` | `standard` | Region operation in the ir module family. |
| `irModuleResult` | `prometheus.ir_module.result` | `standard` | Result operation in the ir module family. |
| `irModuleReturn` | `prometheus.ir_module.return` | `standard` | Return operation in the ir module family. |
| `irModuleSourceLocation` | `prometheus.ir_module.source-location` | `standard` | Source location operation in the ir module family. |
| `irModuleSymbol` | `prometheus.ir_module.symbol` | `standard` | Symbol operation in the ir module family. |
| `irModuleTrap` | `prometheus.ir_module.trap` | `standard` | Trap operation in the ir module family. |
| `irModuleUnreachable` | `prometheus.ir_module.unreachable` | `standard` | Unreachable operation in the ir module family. |
| `irModuleYield` | `prometheus.ir_module.yield` | `standard` | Yield operation in the ir module family. |

## `matrix` (25)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `matrixAnytimeBitplaneMatmul` | `prometheus.matrix.anytime-bitplane-matmul` | `research` | Anytime bitplane matmul operation in the matrix family. |
| `matrixBlockMatmul` | `prometheus.matrix.block-matmul` | `standard` | Block matmul operation in the matrix family. |
| `matrixBlockSparseMatmul` | `prometheus.matrix.block-sparse-matmul` | `standard` | Block sparse matmul operation in the matrix family. |
| `matrixFusedEpilogueMatmul` | `prometheus.matrix.fused-epilogue-matmul` | `standard` | Fused epilogue matmul operation in the matrix family. |
| `matrixGridMatmul` | `prometheus.matrix.grid-matmul` | `standard` | Grid matmul operation in the matrix family. |
| `matrixGroupedMatmul` | `prometheus.matrix.grouped-matmul` | `standard` | Grouped matmul operation in the matrix family. |
| `matrixMaskedMatmul` | `prometheus.matrix.masked-matmul` | `standard` | Masked matmul operation in the matrix family. |
| `matrixMixedInputMatmul` | `prometheus.matrix.mixed-input-matmul` | `standard` | Mixed input matmul operation in the matrix family. |
| `matrixMma` | `prometheus.matrix.mma` | `standard` | Mma operation in the matrix family. |
| `matrixMmaBf16` | `prometheus.matrix.mma-bf16` | `standard` | Mma bf16 operation in the matrix family. |
| `matrixMmaBinary` | `prometheus.matrix.mma-binary` | `standard` | Mma binary operation in the matrix family. |
| `matrixMmaFp16` | `prometheus.matrix.mma-fp16` | `standard` | Mma fp16 operation in the matrix family. |
| `matrixMmaInt4` | `prometheus.matrix.mma-int4` | `standard` | Mma int4 operation in the matrix family. |
| `matrixMmaInt8` | `prometheus.matrix.mma-int8` | `standard` | Mma int8 operation in the matrix family. |
| `matrixMmaSparse` | `prometheus.matrix.mma-sparse` | `standard` | Mma sparse operation in the matrix family. |
| `matrixMmaTf32` | `prometheus.matrix.mma-tf32` | `standard` | Mma tf32 operation in the matrix family. |
| `matrixOptimizerConsumedMatmul` | `prometheus.matrix.optimizer-consumed-matmul` | `research` | Optimizer consumed matmul operation in the matrix family. |
| `matrixPersistentMatmul` | `prometheus.matrix.persistent-matmul` | `standard` | Persistent matmul operation in the matrix family. |
| `matrixQuantizedMatmul` | `prometheus.matrix.quantized-matmul` | `standard` | Quantized matmul operation in the matrix family. |
| `matrixResidueCorrectedMatmul` | `prometheus.matrix.residue-corrected-matmul` | `research` | Residue corrected matmul operation in the matrix family. |
| `matrixSemiringMatmul` | `prometheus.matrix.semiring-matmul` | `standard` | Semiring matmul operation in the matrix family. |
| `matrixSimtMatmul` | `prometheus.matrix.simt-matmul` | `standard` | Simt matmul operation in the matrix family. |
| `matrixSplitKMatmul` | `prometheus.matrix.split-kmatmul` | `standard` | Split kmatmul operation in the matrix family. |
| `matrixStreamKMatmul` | `prometheus.matrix.stream-kmatmul` | `standard` | Stream kmatmul operation in the matrix family. |
| `matrixWarpMatmul` | `prometheus.matrix.warp-matmul` | `standard` | Warp matmul operation in the matrix family. |

## `memory` (38)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `memoryAddressSpaceCast` | `prometheus.memory.address-space-cast` | `standard` | Address space cast operation in the memory family. |
| `memoryAliasView` | `prometheus.memory.alias-view` | `standard` | Alias view operation in the memory family. |
| `memoryAlignmentAssume` | `prometheus.memory.alignment-assume` | `standard` | Alignment assume operation in the memory family. |
| `memoryAlloca` | `prometheus.memory.alloca` | `standard` | Alloca operation in the memory family. |
| `memoryAllocate` | `prometheus.memory.allocate` | `standard` | Allocate operation in the memory family. |
| `memoryAsyncCommit` | `prometheus.memory.async-commit` | `standard` | Async commit operation in the memory family. |
| `memoryAsyncCopy` | `prometheus.memory.async-copy` | `standard` | Async copy operation in the memory family. |
| `memoryAsyncWait` | `prometheus.memory.async-wait` | `standard` | Async wait operation in the memory family. |
| `memoryAtomicCompareExchange` | `prometheus.memory.atomic-compare-exchange` | `standard` | Atomic compare exchange operation in the memory family. |
| `memoryAtomicLoad` | `prometheus.memory.atomic-load` | `standard` | Atomic load operation in the memory family. |
| `memoryAtomicRmw` | `prometheus.memory.atomic-rmw` | `standard` | Atomic rmw operation in the memory family. |
| `memoryAtomicStore` | `prometheus.memory.atomic-store` | `standard` | Atomic store operation in the memory family. |
| `memoryBoundsCheck` | `prometheus.memory.bounds-check` | `standard` | Bounds check operation in the memory family. |
| `memoryCacheHint` | `prometheus.memory.cache-hint` | `standard` | Cache hint operation in the memory family. |
| `memoryCopy` | `prometheus.memory.copy` | `standard` | Copy operation in the memory family. |
| `memoryDeallocate` | `prometheus.memory.deallocate` | `standard` | Deallocate operation in the memory family. |
| `memoryEvictHint` | `prometheus.memory.evict-hint` | `standard` | Evict hint operation in the memory family. |
| `memoryFill` | `prometheus.memory.fill` | `standard` | Fill operation in the memory family. |
| `memoryGather` | `prometheus.memory.gather` | `standard` | Gather operation in the memory family. |
| `memoryLifetimeEnd` | `prometheus.memory.lifetime-end` | `standard` | Lifetime end operation in the memory family. |
| `memoryLifetimeStart` | `prometheus.memory.lifetime-start` | `standard` | Lifetime start operation in the memory family. |
| `memoryLoad` | `prometheus.memory.load` | `standard` | Load operation in the memory family. |
| `memoryMaskedLoad` | `prometheus.memory.masked-load` | `standard` | Masked load operation in the memory family. |
| `memoryMaskedStore` | `prometheus.memory.masked-store` | `standard` | Masked store operation in the memory family. |
| `memoryMemcpy` | `prometheus.memory.memcpy` | `standard` | Memcpy operation in the memory family. |
| `memoryMemmove` | `prometheus.memory.memmove` | `standard` | Memmove operation in the memory family. |
| `memoryMemoryFence` | `prometheus.memory.memory-fence` | `standard` | Memory fence operation in the memory family. |
| `memoryMemset` | `prometheus.memory.memset` | `standard` | Memset operation in the memory family. |
| `memoryNoAlias` | `prometheus.memory.no-alias` | `standard` | No alias operation in the memory family. |
| `memoryPointerAdd` | `prometheus.memory.pointer-add` | `standard` | Pointer add operation in the memory family. |
| `memoryPointerDiff` | `prometheus.memory.pointer-diff` | `standard` | Pointer diff operation in the memory family. |
| `memoryPrefetch` | `prometheus.memory.prefetch` | `standard` | Prefetch operation in the memory family. |
| `memoryReshapeView` | `prometheus.memory.reshape-view` | `standard` | Reshape view operation in the memory family. |
| `memoryScatter` | `prometheus.memory.scatter` | `standard` | Scatter operation in the memory family. |
| `memoryStore` | `prometheus.memory.store` | `standard` | Store operation in the memory family. |
| `memorySubview` | `prometheus.memory.subview` | `standard` | Subview operation in the memory family. |
| `memoryTransposeView` | `prometheus.memory.transpose-view` | `standard` | Transpose view operation in the memory family. |
| `memoryView` | `prometheus.memory.view` | `standard` | View operation in the memory family. |

## `parallel` (36)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `parallelArriveBarrier` | `prometheus.parallel.arrive-barrier` | `standard` | Arrive barrier operation in the parallel family. |
| `parallelBallot` | `prometheus.parallel.ballot` | `standard` | Ballot operation in the parallel family. |
| `parallelBlockBarrier` | `prometheus.parallel.block-barrier` | `standard` | Block barrier operation in the parallel family. |
| `parallelBlockBroadcast` | `prometheus.parallel.block-broadcast` | `standard` | Block broadcast operation in the parallel family. |
| `parallelBlockId` | `prometheus.parallel.block-id` | `standard` | Block id operation in the parallel family. |
| `parallelBlockReduce` | `prometheus.parallel.block-reduce` | `standard` | Block reduce operation in the parallel family. |
| `parallelBlockScan` | `prometheus.parallel.block-scan` | `standard` | Block scan operation in the parallel family. |
| `parallelClusterId` | `prometheus.parallel.cluster-id` | `standard` | Cluster id operation in the parallel family. |
| `parallelCooperativeGroup` | `prometheus.parallel.cooperative-group` | `standard` | Cooperative group operation in the parallel family. |
| `parallelGridBarrier` | `prometheus.parallel.grid-barrier` | `standard` | Grid barrier operation in the parallel family. |
| `parallelGridId` | `prometheus.parallel.grid-id` | `standard` | Grid id operation in the parallel family. |
| `parallelGridReduce` | `prometheus.parallel.grid-reduce` | `standard` | Grid reduce operation in the parallel family. |
| `parallelGridScan` | `prometheus.parallel.grid-scan` | `standard` | Grid scan operation in the parallel family. |
| `parallelLaneId` | `prometheus.parallel.lane-id` | `standard` | Lane id operation in the parallel family. |
| `parallelMatchAll` | `prometheus.parallel.match-all` | `standard` | Match all operation in the parallel family. |
| `parallelMatchAny` | `prometheus.parallel.match-any` | `standard` | Match any operation in the parallel family. |
| `parallelNamedBarrier` | `prometheus.parallel.named-barrier` | `standard` | Named barrier operation in the parallel family. |
| `parallelNumBlocks` | `prometheus.parallel.num-blocks` | `standard` | Num blocks operation in the parallel family. |
| `parallelNumThreads` | `prometheus.parallel.num-threads` | `standard` | Num threads operation in the parallel family. |
| `parallelNumWarps` | `prometheus.parallel.num-warps` | `standard` | Num warps operation in the parallel family. |
| `parallelShuffle` | `prometheus.parallel.shuffle` | `standard` | Shuffle operation in the parallel family. |
| `parallelShuffleDown` | `prometheus.parallel.shuffle-down` | `standard` | Shuffle down operation in the parallel family. |
| `parallelShuffleUp` | `prometheus.parallel.shuffle-up` | `standard` | Shuffle up operation in the parallel family. |
| `parallelShuffleXor` | `prometheus.parallel.shuffle-xor` | `standard` | Shuffle xor operation in the parallel family. |
| `parallelThreadId` | `prometheus.parallel.thread-id` | `standard` | Thread id operation in the parallel family. |
| `parallelTryWaitBarrier` | `prometheus.parallel.try-wait-barrier` | `standard` | Try wait barrier operation in the parallel family. |
| `parallelVoteAll` | `prometheus.parallel.vote-all` | `standard` | Vote all operation in the parallel family. |
| `parallelVoteAny` | `prometheus.parallel.vote-any` | `standard` | Vote any operation in the parallel family. |
| `parallelWaitBarrier` | `prometheus.parallel.wait-barrier` | `standard` | Wait barrier operation in the parallel family. |
| `parallelWarpBarrier` | `prometheus.parallel.warp-barrier` | `standard` | Warp barrier operation in the parallel family. |
| `parallelWarpBroadcast` | `prometheus.parallel.warp-broadcast` | `standard` | Warp broadcast operation in the parallel family. |
| `parallelWarpId` | `prometheus.parallel.warp-id` | `standard` | Warp id operation in the parallel family. |
| `parallelWarpReduce` | `prometheus.parallel.warp-reduce` | `standard` | Warp reduce operation in the parallel family. |
| `parallelWarpScan` | `prometheus.parallel.warp-scan` | `standard` | Warp scan operation in the parallel family. |
| `parallelWorkDonate` | `prometheus.parallel.work-donate` | `standard` | Work donate operation in the parallel family. |
| `parallelWorkSteal` | `prometheus.parallel.work-steal` | `standard` | Work steal operation in the parallel family. |

## `sparse` (20)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `sparseSparseBlockPack` | `prometheus.sparse.sparse-block-pack` | `standard` | Sparse block pack operation in the sparse family. |
| `sparseSparseBlockUnpack` | `prometheus.sparse.sparse-block-unpack` | `standard` | Sparse block unpack operation in the sparse family. |
| `sparseSparseCoalesce` | `prometheus.sparse.sparse-coalesce` | `standard` | Sparse coalesce operation in the sparse family. |
| `sparseSparseCompress` | `prometheus.sparse.sparse-compress` | `standard` | Sparse compress operation in the sparse family. |
| `sparseSparseConvert` | `prometheus.sparse.sparse-convert` | `standard` | Sparse convert operation in the sparse family. |
| `sparseSparseExpand` | `prometheus.sparse.sparse-expand` | `standard` | Sparse expand operation in the sparse family. |
| `sparseSparseInsert` | `prometheus.sparse.sparse-insert` | `standard` | Sparse insert operation in the sparse family. |
| `sparseSparseIterate` | `prometheus.sparse.sparse-iterate` | `standard` | Sparse iterate operation in the sparse family. |
| `sparseSparseLocate` | `prometheus.sparse.sparse-locate` | `standard` | Sparse locate operation in the sparse family. |
| `sparseSparseMask` | `prometheus.sparse.sparse-mask` | `standard` | Sparse mask operation in the sparse family. |
| `sparseSparseNmPack` | `prometheus.sparse.sparse-nm-pack` | `standard` | Sparse nm pack operation in the sparse family. |
| `sparseSparseNmUnpack` | `prometheus.sparse.sparse-nm-unpack` | `standard` | Sparse nm unpack operation in the sparse family. |
| `sparseSparsePrune` | `prometheus.sparse.sparse-prune` | `standard` | Sparse prune operation in the sparse family. |
| `sparseSparseReduce` | `prometheus.sparse.sparse-reduce` | `standard` | Sparse reduce operation in the sparse family. |
| `sparseSparseSddmm` | `prometheus.sparse.sparse-sddmm` | `standard` | Sparse sddmm operation in the sparse family. |
| `sparseSparseSemiring` | `prometheus.sparse.sparse-semiring` | `standard` | Sparse semiring operation in the sparse family. |
| `sparseSparseSort` | `prometheus.sparse.sparse-sort` | `standard` | Sparse sort operation in the sparse family. |
| `sparseSparseSpgemm` | `prometheus.sparse.sparse-spgemm` | `standard` | Sparse spgemm operation in the sparse family. |
| `sparseSparseSpmm` | `prometheus.sparse.sparse-spmm` | `standard` | Sparse spmm operation in the sparse family. |
| `sparseSparseSpmv` | `prometheus.sparse.sparse-spmv` | `standard` | Sparse spmv operation in the sparse family. |

## `structured` (33)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `structuredAttention` | `prometheus.structured.attention` | `standard` | Attention operation in the structured family. |
| `structuredBatchMatmul` | `prometheus.structured.batch-matmul` | `standard` | Batch matmul operation in the structured family. |
| `structuredContract` | `prometheus.structured.contract` | `standard` | Contract operation in the structured family. |
| `structuredConv` | `prometheus.structured.conv` | `standard` | Conv operation in the structured family. |
| `structuredEinsum` | `prometheus.structured.einsum` | `standard` | Einsum operation in the structured family. |
| `structuredExpertDispatch` | `prometheus.structured.expert-dispatch` | `standard` | Expert dispatch operation in the structured family. |
| `structuredFold` | `prometheus.structured.fold` | `standard` | Fold operation in the structured family. |
| `structuredGenericContraction` | `prometheus.structured.generic-contraction` | `standard` | Generic contraction operation in the structured family. |
| `structuredGenericConvolution` | `prometheus.structured.generic-convolution` | `standard` | Generic convolution operation in the structured family. |
| `structuredGenericElementwise` | `prometheus.structured.generic-elementwise` | `standard` | Generic elementwise operation in the structured family. |
| `structuredGenericGather` | `prometheus.structured.generic-gather` | `standard` | Generic gather operation in the structured family. |
| `structuredGenericHistogram` | `prometheus.structured.generic-histogram` | `standard` | Generic histogram operation in the structured family. |
| `structuredGenericReduction` | `prometheus.structured.generic-reduction` | `standard` | Generic reduction operation in the structured family. |
| `structuredGenericScan` | `prometheus.structured.generic-scan` | `standard` | Generic scan operation in the structured family. |
| `structuredGenericScatter` | `prometheus.structured.generic-scatter` | `standard` | Generic scatter operation in the structured family. |
| `structuredGenericSegment` | `prometheus.structured.generic-segment` | `standard` | Generic segment operation in the structured family. |
| `structuredGenericSemiring` | `prometheus.structured.generic-semiring` | `standard` | Generic semiring operation in the structured family. |
| `structuredGenericSort` | `prometheus.structured.generic-sort` | `standard` | Generic sort operation in the structured family. |
| `structuredGenericSparse` | `prometheus.structured.generic-sparse` | `standard` | Generic sparse operation in the structured family. |
| `structuredGenericTopK` | `prometheus.structured.generic-top-k` | `standard` | Generic top k operation in the structured family. |
| `structuredGenericWindow` | `prometheus.structured.generic-window` | `standard` | Generic window operation in the structured family. |
| `structuredGroupedMatmul` | `prometheus.structured.grouped-matmul` | `standard` | Grouped matmul operation in the structured family. |
| `structuredLayerNorm` | `prometheus.structured.layer-norm` | `standard` | Layer norm operation in the structured family. |
| `structuredMap` | `prometheus.structured.map` | `standard` | Map operation in the structured family. |
| `structuredMatmul` | `prometheus.structured.matmul` | `standard` | Matmul operation in the structured family. |
| `structuredPool` | `prometheus.structured.pool` | `standard` | Pool operation in the structured family. |
| `structuredReduce` | `prometheus.structured.reduce` | `standard` | Reduce operation in the structured family. |
| `structuredRmsNorm` | `prometheus.structured.rms-norm` | `standard` | Rms norm operation in the structured family. |
| `structuredScan` | `prometheus.structured.scan` | `standard` | Scan operation in the structured family. |
| `structuredSelectiveScan` | `prometheus.structured.selective-scan` | `standard` | Selective scan operation in the structured family. |
| `structuredSoftmax` | `prometheus.structured.softmax` | `standard` | Softmax operation in the structured family. |
| `structuredStencil` | `prometheus.structured.stencil` | `standard` | Stencil operation in the structured family. |
| `structuredZip` | `prometheus.structured.zip` | `standard` | Zip operation in the structured family. |

## `transform_pass` (46)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `transformPassAlgebraicSimplify` | `prometheus.transform_pass.algebraic-simplify` | `standard` | Algebraic simplify operation in the transform pass family. |
| `transformPassAsyncify` | `prometheus.transform_pass.asyncify` | `standard` | Asyncify operation in the transform pass family. |
| `transformPassBarrierInsert` | `prometheus.transform_pass.barrier-insert` | `standard` | Barrier insert operation in the transform pass family. |
| `transformPassBufferize` | `prometheus.transform_pass.bufferize` | `standard` | Bufferize operation in the transform pass family. |
| `transformPassCanonicalize` | `prometheus.transform_pass.canonicalize` | `standard` | Canonicalize operation in the transform pass family. |
| `transformPassCheckpoint` | `prometheus.transform_pass.checkpoint` | `standard` | Checkpoint operation in the transform pass family. |
| `transformPassCommonSubexpressionEliminate` | `prometheus.transform_pass.common-subexpression-eliminate` | `standard` | Common subexpression eliminate operation in the transform pass family. |
| `transformPassConstantFold` | `prometheus.transform_pass.constant-fold` | `standard` | Constant fold operation in the transform pass family. |
| `transformPassDeadCodeEliminate` | `prometheus.transform_pass.dead-code-eliminate` | `standard` | Dead code eliminate operation in the transform pass family. |
| `transformPassDebufferize` | `prometheus.transform_pass.debufferize` | `standard` | Debufferize operation in the transform pass family. |
| `transformPassDefuse` | `prometheus.transform_pass.defuse` | `standard` | Defuse operation in the transform pass family. |
| `transformPassDensify` | `prometheus.transform_pass.densify` | `standard` | Densify operation in the transform pass family. |
| `transformPassDependencyAnalyze` | `prometheus.transform_pass.dependency-analyze` | `standard` | Dependency analyze operation in the transform pass family. |
| `transformPassDequantize` | `prometheus.transform_pass.dequantize` | `standard` | Dequantize operation in the transform pass family. |
| `transformPassDeterminize` | `prometheus.transform_pass.determinize` | `standard` | Determinize operation in the transform pass family. |
| `transformPassDevectorize` | `prometheus.transform_pass.devectorize` | `standard` | Devectorize operation in the transform pass family. |
| `transformPassDoubleBuffer` | `prometheus.transform_pass.double-buffer` | `standard` | Double buffer operation in the transform pass family. |
| `transformPassFuse` | `prometheus.transform_pass.fuse` | `standard` | Fuse operation in the transform pass family. |
| `transformPassHazardResolve` | `prometheus.transform_pass.hazard-resolve` | `standard` | Hazard resolve operation in the transform pass family. |
| `transformPassInline` | `prometheus.transform_pass.inline` | `standard` | Inline operation in the transform pass family. |
| `transformPassLayoutPropagate` | `prometheus.transform_pass.layout-propagate` | `standard` | Layout propagate operation in the transform pass family. |
| `transformPassLayoutRewrite` | `prometheus.transform_pass.layout-rewrite` | `standard` | Layout rewrite operation in the transform pass family. |
| `transformPassMapToBlocks` | `prometheus.transform_pass.map-to-blocks` | `standard` | Map to blocks operation in the transform pass family. |
| `transformPassMapToThreads` | `prometheus.transform_pass.map-to-threads` | `standard` | Map to threads operation in the transform pass family. |
| `transformPassMapToWarps` | `prometheus.transform_pass.map-to-warps` | `standard` | Map to warps operation in the transform pass family. |
| `transformPassMonomorphize` | `prometheus.transform_pass.monomorphize` | `standard` | Monomorphize operation in the transform pass family. |
| `transformPassOperatorStrengthReduce` | `prometheus.transform_pass.operator-strength-reduce` | `standard` | Operator strength reduce operation in the transform pass family. |
| `transformPassOutline` | `prometheus.transform_pass.outline` | `standard` | Outline operation in the transform pass family. |
| `transformPassPrecisionLegalize` | `prometheus.transform_pass.precision-legalize` | `standard` | Precision legalize operation in the transform pass family. |
| `transformPassPrefetch` | `prometheus.transform_pass.prefetch` | `standard` | Prefetch operation in the transform pass family. |
| `transformPassQuantize` | `prometheus.transform_pass.quantize` | `standard` | Quantize operation in the transform pass family. |
| `transformPassRegisterAllocate` | `prometheus.transform_pass.register-allocate` | `standard` | Register allocate operation in the transform pass family. |
| `transformPassRematerialize` | `prometheus.transform_pass.rematerialize` | `standard` | Rematerialize operation in the transform pass family. |
| `transformPassRetile` | `prometheus.transform_pass.retile` | `standard` | Retile operation in the transform pass family. |
| `transformPassSchedule` | `prometheus.transform_pass.schedule` | `standard` | Schedule operation in the transform pass family. |
| `transformPassSemiringRewrite` | `prometheus.transform_pass.semiring-rewrite` | `standard` | Semiring rewrite operation in the transform pass family. |
| `transformPassSharedMemoryPlan` | `prometheus.transform_pass.shared-memory-plan` | `standard` | Shared memory plan operation in the transform pass family. |
| `transformPassSoftwarePipeline` | `prometheus.transform_pass.software-pipeline` | `standard` | Software pipeline operation in the transform pass family. |
| `transformPassSparsify` | `prometheus.transform_pass.sparsify` | `standard` | Sparsify operation in the transform pass family. |
| `transformPassSpecialize` | `prometheus.transform_pass.specialize` | `standard` | Specialize operation in the transform pass family. |
| `transformPassSymbolDeadCodeEliminate` | `prometheus.transform_pass.symbol-dead-code-eliminate` | `standard` | Symbol dead code eliminate operation in the transform pass family. |
| `transformPassTile` | `prometheus.transform_pass.tile` | `standard` | Tile operation in the transform pass family. |
| `transformPassTripleBuffer` | `prometheus.transform_pass.triple-buffer` | `standard` | Triple buffer operation in the transform pass family. |
| `transformPassUnroll` | `prometheus.transform_pass.unroll` | `standard` | Unroll operation in the transform pass family. |
| `transformPassVectorize` | `prometheus.transform_pass.vectorize` | `standard` | Vectorize operation in the transform pass family. |
| `transformPassVerify` | `prometheus.transform_pass.verify` | `standard` | Verify operation in the transform pass family. |

## `type_shape` (26)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `typeShapeBitcastType` | `prometheus.type_shape.bitcast-type` | `standard` | Bitcast type operation in the type shape family. |
| `typeShapeBroadcastShape` | `prometheus.type_shape.broadcast-shape` | `standard` | Broadcast shape operation in the type shape family. |
| `typeShapeCastType` | `prometheus.type_shape.cast-type` | `standard` | Cast type operation in the type shape family. |
| `typeShapeComplexType` | `prometheus.type_shape.complex-type` | `standard` | Complex type operation in the type shape family. |
| `typeShapeDimOf` | `prometheus.type_shape.dim-of` | `standard` | Dim of operation in the type shape family. |
| `typeShapeDynamicDimension` | `prometheus.type_shape.dynamic-dimension` | `standard` | Dynamic dimension operation in the type shape family. |
| `typeShapeElementTypeOf` | `prometheus.type_shape.element-type-of` | `standard` | Element type of operation in the type shape family. |
| `typeShapeEventType` | `prometheus.type_shape.event-type` | `standard` | Event type operation in the type shape family. |
| `typeShapeLayoutOf` | `prometheus.type_shape.layout-of` | `standard` | Layout of operation in the type shape family. |
| `typeShapeMailboxType` | `prometheus.type_shape.mailbox-type` | `standard` | Mailbox type operation in the type shape family. |
| `typeShapeMemrefType` | `prometheus.type_shape.memref-type` | `standard` | Memref type operation in the type shape family. |
| `typeShapePointerType` | `prometheus.type_shape.pointer-type` | `standard` | Pointer type operation in the type shape family. |
| `typeShapeQuantizedType` | `prometheus.type_shape.quantized-type` | `standard` | Quantized type operation in the type shape family. |
| `typeShapeRankOf` | `prometheus.type_shape.rank-of` | `standard` | Rank of operation in the type shape family. |
| `typeShapeRefineShape` | `prometheus.type_shape.refine-shape` | `standard` | Refine shape operation in the type shape family. |
| `typeShapeReinterpretType` | `prometheus.type_shape.reinterpret-type` | `standard` | Reinterpret type operation in the type shape family. |
| `typeShapeScalarType` | `prometheus.type_shape.scalar-type` | `standard` | Scalar type operation in the type shape family. |
| `typeShapeShapeOf` | `prometheus.type_shape.shape-of` | `standard` | Shape of operation in the type shape family. |
| `typeShapeSparseType` | `prometheus.type_shape.sparse-type` | `standard` | Sparse type operation in the type shape family. |
| `typeShapeStateType` | `prometheus.type_shape.state-type` | `standard` | State type operation in the type shape family. |
| `typeShapeStrideOf` | `prometheus.type_shape.stride-of` | `standard` | Stride of operation in the type shape family. |
| `typeShapeSymbolicDimension` | `prometheus.type_shape.symbolic-dimension` | `standard` | Symbolic dimension operation in the type shape family. |
| `typeShapeTensorType` | `prometheus.type_shape.tensor-type` | `standard` | Tensor type operation in the type shape family. |
| `typeShapeTileType` | `prometheus.type_shape.tile-type` | `standard` | Tile type operation in the type shape family. |
| `typeShapeTokenType` | `prometheus.type_shape.token-type` | `standard` | Token type operation in the type shape family. |
| `typeShapeVectorType` | `prometheus.type_shape.vector-type` | `standard` | Vector type operation in the type shape family. |

## `vector_tile` (32)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `vectorTileBroadcast` | `prometheus.vector_tile.broadcast` | `standard` | Broadcast operation in the vector tile family. |
| `vectorTileCompressStore` | `prometheus.vector_tile.compress-store` | `standard` | Compress store operation in the vector tile family. |
| `vectorTileConstantMask` | `prometheus.vector_tile.constant-mask` | `standard` | Constant mask operation in the vector tile family. |
| `vectorTileContract` | `prometheus.vector_tile.contract` | `standard` | Contract operation in the vector tile family. |
| `vectorTileCreateMask` | `prometheus.vector_tile.create-mask` | `standard` | Create mask operation in the vector tile family. |
| `vectorTileDeinterleave` | `prometheus.vector_tile.deinterleave` | `standard` | Deinterleave operation in the vector tile family. |
| `vectorTileExpandLoad` | `prometheus.vector_tile.expand-load` | `standard` | Expand load operation in the vector tile family. |
| `vectorTileExtract` | `prometheus.vector_tile.extract` | `standard` | Extract operation in the vector tile family. |
| `vectorTileExtractStridedSlice` | `prometheus.vector_tile.extract-strided-slice` | `standard` | Extract strided slice operation in the vector tile family. |
| `vectorTileGatherTile` | `prometheus.vector_tile.gather-tile` | `standard` | Gather tile operation in the vector tile family. |
| `vectorTileInsert` | `prometheus.vector_tile.insert` | `standard` | Insert operation in the vector tile family. |
| `vectorTileInsertStridedSlice` | `prometheus.vector_tile.insert-strided-slice` | `standard` | Insert strided slice operation in the vector tile family. |
| `vectorTileInterleave` | `prometheus.vector_tile.interleave` | `standard` | Interleave operation in the vector tile family. |
| `vectorTileLoadTile` | `prometheus.vector_tile.load-tile` | `standard` | Load tile operation in the vector tile family. |
| `vectorTileMaskedTransferRead` | `prometheus.vector_tile.masked-transfer-read` | `standard` | Masked transfer read operation in the vector tile family. |
| `vectorTileMaskedTransferWrite` | `prometheus.vector_tile.masked-transfer-write` | `standard` | Masked transfer write operation in the vector tile family. |
| `vectorTileMatrixFragmentLoad` | `prometheus.vector_tile.matrix-fragment-load` | `standard` | Matrix fragment load operation in the vector tile family. |
| `vectorTileMatrixFragmentMma` | `prometheus.vector_tile.matrix-fragment-mma` | `standard` | Matrix fragment mma operation in the vector tile family. |
| `vectorTileMatrixFragmentStore` | `prometheus.vector_tile.matrix-fragment-store` | `standard` | Matrix fragment store operation in the vector tile family. |
| `vectorTileMultiReduce` | `prometheus.vector_tile.multi-reduce` | `standard` | Multi reduce operation in the vector tile family. |
| `vectorTileOuterProduct` | `prometheus.vector_tile.outer-product` | `standard` | Outer product operation in the vector tile family. |
| `vectorTileScan` | `prometheus.vector_tile.scan` | `standard` | Scan operation in the vector tile family. |
| `vectorTileScatterTile` | `prometheus.vector_tile.scatter-tile` | `standard` | Scatter tile operation in the vector tile family. |
| `vectorTileShapeCast` | `prometheus.vector_tile.shape-cast` | `standard` | Shape cast operation in the vector tile family. |
| `vectorTileShuffle` | `prometheus.vector_tile.shuffle` | `standard` | Shuffle operation in the vector tile family. |
| `vectorTileSplat` | `prometheus.vector_tile.splat` | `standard` | Splat operation in the vector tile family. |
| `vectorTileStepVector` | `prometheus.vector_tile.step-vector` | `standard` | Step vector operation in the vector tile family. |
| `vectorTileStoreTile` | `prometheus.vector_tile.store-tile` | `standard` | Store tile operation in the vector tile family. |
| `vectorTileTransferRead` | `prometheus.vector_tile.transfer-read` | `standard` | Transfer read operation in the vector tile family. |
| `vectorTileTransferWrite` | `prometheus.vector_tile.transfer-write` | `standard` | Transfer write operation in the vector tile family. |
| `vectorTileTranspose` | `prometheus.vector_tile.transpose` | `standard` | Transpose operation in the vector tile family. |
| `vectorTileTypeCast` | `prometheus.vector_tile.type-cast` | `standard` | Type cast operation in the vector tile family. |

# Hephaestus


## `control_schedule` (22)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `controlScheduleInsertBlockBarrier` | `hephaestus.control_schedule.insert-block-barrier` | `standard` | Insert block barrier operation in the control schedule family. |
| `controlScheduleInsertDependencyBarrier` | `hephaestus.control_schedule.insert-dependency-barrier` | `standard` | Insert dependency barrier operation in the control schedule family. |
| `controlScheduleInsertMemoryBarrier` | `hephaestus.control_schedule.insert-memory-barrier` | `standard` | Insert memory barrier operation in the control schedule family. |
| `controlScheduleInsertNop` | `hephaestus.control_schedule.insert-nop` | `standard` | Insert nop operation in the control schedule family. |
| `controlScheduleInsertWarpBarrier` | `hephaestus.control_schedule.insert-warp-barrier` | `standard` | Insert warp barrier operation in the control schedule family. |
| `controlScheduleScheduleBasicBlock` | `hephaestus.control_schedule.schedule-basic-block` | `standard` | Schedule basic block operation in the control schedule family. |
| `controlScheduleScheduleInstructionWindow` | `hephaestus.control_schedule.schedule-instruction-window` | `standard` | Schedule instruction window operation in the control schedule family. |
| `controlScheduleScheduleSoftwarePipeline` | `hephaestus.control_schedule.schedule-software-pipeline` | `standard` | Schedule software pipeline operation in the control schedule family. |
| `controlScheduleSetDependencyMask` | `hephaestus.control_schedule.set-dependency-mask` | `standard` | Set dependency mask operation in the control schedule family. |
| `controlScheduleSetDivergenceHint` | `hephaestus.control_schedule.set-divergence-hint` | `standard` | Set divergence hint operation in the control schedule family. |
| `controlScheduleSetDualIssueHint` | `hephaestus.control_schedule.set-dual-issue-hint` | `standard` | Set dual issue hint operation in the control schedule family. |
| `controlScheduleSetIssueSlot` | `hephaestus.control_schedule.set-issue-slot` | `standard` | Set issue slot operation in the control schedule family. |
| `controlScheduleSetReadBarrier` | `hephaestus.control_schedule.set-read-barrier` | `standard` | Set read barrier operation in the control schedule family. |
| `controlScheduleSetReuseMask` | `hephaestus.control_schedule.set-reuse-mask` | `standard` | Set reuse mask operation in the control schedule family. |
| `controlScheduleSetStallCount` | `hephaestus.control_schedule.set-stall-count` | `standard` | Set stall count operation in the control schedule family. |
| `controlScheduleSetUniformPathHint` | `hephaestus.control_schedule.set-uniform-path-hint` | `standard` | Set uniform path hint operation in the control schedule family. |
| `controlScheduleSetWaitMask` | `hephaestus.control_schedule.set-wait-mask` | `standard` | Set wait mask operation in the control schedule family. |
| `controlScheduleSetWriteBarrier` | `hephaestus.control_schedule.set-write-barrier` | `standard` | Set write barrier operation in the control schedule family. |
| `controlScheduleSetYieldFlag` | `hephaestus.control_schedule.set-yield-flag` | `standard` | Set yield flag operation in the control schedule family. |
| `controlScheduleValidateBarrierLifetime` | `hephaestus.control_schedule.validate-barrier-lifetime` | `standard` | Validate barrier lifetime operation in the control schedule family. |
| `controlScheduleValidateControlEncoding` | `hephaestus.control_schedule.validate-control-encoding` | `standard` | Validate control encoding operation in the control schedule family. |
| `controlScheduleValidateHazards` | `hephaestus.control_schedule.validate-hazards` | `standard` | Validate hazards operation in the control schedule family. |

## `macro_kernel` (28)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `macroKernelMacroBlockReduce` | `hephaestus.macro_kernel.macro-block-reduce` | `research` | Macro block reduce operation in the macro kernel family. |
| `macroKernelMacroBlockScan` | `hephaestus.macro_kernel.macro-block-scan` | `research` | Macro block scan operation in the macro kernel family. |
| `macroKernelMacroDequantize` | `hephaestus.macro_kernel.macro-dequantize` | `research` | Macro dequantize operation in the macro kernel family. |
| `macroKernelMacroInt4Decode` | `hephaestus.macro_kernel.macro-int4-decode` | `research` | Macro int4 decode operation in the macro kernel family. |
| `macroKernelMacroLayerNorm` | `hephaestus.macro_kernel.macro-layer-norm` | `research` | Macro layer norm operation in the macro kernel family. |
| `macroKernelMacroMailboxReceive` | `hephaestus.macro_kernel.macro-mailbox-receive` | `research` | Macro mailbox receive operation in the macro kernel family. |
| `macroKernelMacroMailboxSend` | `hephaestus.macro_kernel.macro-mailbox-send` | `research` | Macro mailbox send operation in the macro kernel family. |
| `macroKernelMacroMatrixTileLoad` | `hephaestus.macro_kernel.macro-matrix-tile-load` | `research` | Macro matrix tile load operation in the macro kernel family. |
| `macroKernelMacroMatrixTileStore` | `hephaestus.macro_kernel.macro-matrix-tile-store` | `research` | Macro matrix tile store operation in the macro kernel family. |
| `macroKernelMacroMmaLoop` | `hephaestus.macro_kernel.macro-mma-loop` | `research` | Macro mma loop operation in the macro kernel family. |
| `macroKernelMacroOnlineSoftmax` | `hephaestus.macro_kernel.macro-online-softmax` | `research` | Macro online softmax operation in the macro kernel family. |
| `macroKernelMacroPersistentWorkLoop` | `hephaestus.macro_kernel.macro-persistent-work-loop` | `research` | Macro persistent work loop operation in the macro kernel family. |
| `macroKernelMacroPhilox` | `hephaestus.macro_kernel.macro-philox` | `research` | Macro philox operation in the macro kernel family. |
| `macroKernelMacroQuantize` | `hephaestus.macro_kernel.macro-quantize` | `research` | Macro quantize operation in the macro kernel family. |
| `macroKernelMacroQuiescenceCredit` | `hephaestus.macro_kernel.macro-quiescence-credit` | `research` | Macro quiescence credit operation in the macro kernel family. |
| `macroKernelMacroRmsNorm` | `hephaestus.macro_kernel.macro-rms-norm` | `research` | Macro rms norm operation in the macro kernel family. |
| `macroKernelMacroRotary` | `hephaestus.macro_kernel.macro-rotary` | `research` | Macro rotary operation in the macro kernel family. |
| `macroKernelMacroSelectiveScan` | `hephaestus.macro_kernel.macro-selective-scan` | `research` | Macro selective scan operation in the macro kernel family. |
| `macroKernelMacroSparse2of4Decode` | `hephaestus.macro_kernel.macro-sparse2of4-decode` | `research` | Macro sparse2of4 decode operation in the macro kernel family. |
| `macroKernelMacroSplitKReduce` | `hephaestus.macro_kernel.macro-split-kreduce` | `research` | Macro split kreduce operation in the macro kernel family. |
| `macroKernelMacroStreamK` | `hephaestus.macro_kernel.macro-stream-k` | `research` | Macro stream k operation in the macro kernel family. |
| `macroKernelMacroTernaryDecode` | `hephaestus.macro_kernel.macro-ternary-decode` | `research` | Macro ternary decode operation in the macro kernel family. |
| `macroKernelMacroVectorLoad` | `hephaestus.macro_kernel.macro-vector-load` | `research` | Macro vector load operation in the macro kernel family. |
| `macroKernelMacroVectorStore` | `hephaestus.macro_kernel.macro-vector-store` | `research` | Macro vector store operation in the macro kernel family. |
| `macroKernelMacroWarpReduce` | `hephaestus.macro_kernel.macro-warp-reduce` | `research` | Macro warp reduce operation in the macro kernel family. |
| `macroKernelMacroWarpScan` | `hephaestus.macro_kernel.macro-warp-scan` | `research` | Macro warp scan operation in the macro kernel family. |
| `macroKernelMacroWelford` | `hephaestus.macro_kernel.macro-welford` | `research` | Macro welford operation in the macro kernel family. |
| `macroKernelMacroWorkSteal` | `hephaestus.macro_kernel.macro-work-steal` | `research` | Macro work steal operation in the macro kernel family. |

## `module_binary` (21)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `moduleBinaryAlignSection` | `hephaestus.module_binary.align-section` | `standard` | Align section operation in the module binary family. |
| `moduleBinaryApplyRelocations` | `hephaestus.module_binary.apply-relocations` | `standard` | Apply relocations operation in the module binary family. |
| `moduleBinaryCreateModule` | `hephaestus.module_binary.create-module` | `standard` | Create module operation in the module binary family. |
| `moduleBinaryDecodeModule` | `hephaestus.module_binary.decode-module` | `standard` | Decode module operation in the module binary family. |
| `moduleBinaryDefineConstant` | `hephaestus.module_binary.define-constant` | `standard` | Define constant operation in the module binary family. |
| `moduleBinaryDefineFunction` | `hephaestus.module_binary.define-function` | `standard` | Define function operation in the module binary family. |
| `moduleBinaryDefineKernelEntry` | `hephaestus.module_binary.define-kernel-entry` | `standard` | Define kernel entry operation in the module binary family. |
| `moduleBinaryDefineRelocation` | `hephaestus.module_binary.define-relocation` | `standard` | Define relocation operation in the module binary family. |
| `moduleBinaryDefineSection` | `hephaestus.module_binary.define-section` | `standard` | Define section operation in the module binary family. |
| `moduleBinaryDefineSymbol` | `hephaestus.module_binary.define-symbol` | `standard` | Define symbol operation in the module binary family. |
| `moduleBinaryDeserializeModule` | `hephaestus.module_binary.deserialize-module` | `standard` | Deserialize module operation in the module binary family. |
| `moduleBinaryDestroyModule` | `hephaestus.module_binary.destroy-module` | `standard` | Destroy module operation in the module binary family. |
| `moduleBinaryDisassembleModule` | `hephaestus.module_binary.disassemble-module` | `standard` | Disassemble module operation in the module binary family. |
| `moduleBinaryEmitControlWord` | `hephaestus.module_binary.emit-control-word` | `standard` | Emit control word operation in the module binary family. |
| `moduleBinaryEmitData` | `hephaestus.module_binary.emit-data` | `standard` | Emit data operation in the module binary family. |
| `moduleBinaryEmitInstruction` | `hephaestus.module_binary.emit-instruction` | `standard` | Emit instruction operation in the module binary family. |
| `moduleBinaryEncodeModule` | `hephaestus.module_binary.encode-module` | `standard` | Encode module operation in the module binary family. |
| `moduleBinaryHashModule` | `hephaestus.module_binary.hash-module` | `standard` | Hash module operation in the module binary family. |
| `moduleBinaryResolveLabels` | `hephaestus.module_binary.resolve-labels` | `standard` | Resolve labels operation in the module binary family. |
| `moduleBinarySerializeModule` | `hephaestus.module_binary.serialize-module` | `standard` | Serialize module operation in the module binary family. |
| `moduleBinaryValidateModule` | `hephaestus.module_binary.validate-module` | `standard` | Validate module operation in the module binary family. |

## `register_allocation` (24)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `registerAllocationAssignPhysicalRegister` | `hephaestus.register_allocation.assign-physical-register` | `standard` | Assign physical register operation in the register allocation family. |
| `registerAllocationAssignRegisterTuple` | `hephaestus.register_allocation.assign-register-tuple` | `standard` | Assign register tuple operation in the register allocation family. |
| `registerAllocationCoalesceRegisters` | `hephaestus.register_allocation.coalesce-registers` | `standard` | Coalesce registers operation in the register allocation family. |
| `registerAllocationColorInterferenceGraph` | `hephaestus.register_allocation.color-interference-graph` | `standard` | Color interference graph operation in the register allocation family. |
| `registerAllocationComputeInterference` | `hephaestus.register_allocation.compute-interference` | `standard` | Compute interference operation in the register allocation family. |
| `registerAllocationComputeLiveness` | `hephaestus.register_allocation.compute-liveness` | `standard` | Compute liveness operation in the register allocation family. |
| `registerAllocationCreateBarrierRegister` | `hephaestus.register_allocation.create-barrier-register` | `standard` | Create barrier register operation in the register allocation family. |
| `registerAllocationCreatePinnedRegister` | `hephaestus.register_allocation.create-pinned-register` | `standard` | Create pinned register operation in the register allocation family. |
| `registerAllocationCreatePredicateRegister` | `hephaestus.register_allocation.create-predicate-register` | `standard` | Create predicate register operation in the register allocation family. |
| `registerAllocationCreateStateRegister` | `hephaestus.register_allocation.create-state-register` | `standard` | Create state register operation in the register allocation family. |
| `registerAllocationCreateTemporaryRegister` | `hephaestus.register_allocation.create-temporary-register` | `standard` | Create temporary register operation in the register allocation family. |
| `registerAllocationCreateUniformRegister` | `hephaestus.register_allocation.create-uniform-register` | `standard` | Create uniform register operation in the register allocation family. |
| `registerAllocationCreateVirtualRegister` | `hephaestus.register_allocation.create-virtual-register` | `standard` | Create virtual register operation in the register allocation family. |
| `registerAllocationCreateWeightRegister` | `hephaestus.register_allocation.create-weight-register` | `standard` | Create weight register operation in the register allocation family. |
| `registerAllocationForbidSpill` | `hephaestus.register_allocation.forbid-spill` | `standard` | Forbid spill operation in the register allocation family. |
| `registerAllocationLinearScanAllocate` | `hephaestus.register_allocation.linear-scan-allocate` | `standard` | Linear scan allocate operation in the register allocation family. |
| `registerAllocationReleaseRegister` | `hephaestus.register_allocation.release-register` | `standard` | Release register operation in the register allocation family. |
| `registerAllocationReloadRegister` | `hephaestus.register_allocation.reload-register` | `standard` | Reload register operation in the register allocation family. |
| `registerAllocationReportRegisterPressure` | `hephaestus.register_allocation.report-register-pressure` | `standard` | Report register pressure operation in the register allocation family. |
| `registerAllocationReserveRegister` | `hephaestus.register_allocation.reserve-register` | `standard` | Reserve register operation in the register allocation family. |
| `registerAllocationSpillRegister` | `hephaestus.register_allocation.spill-register` | `standard` | Spill register operation in the register allocation family. |
| `registerAllocationSplitLiveRange` | `hephaestus.register_allocation.split-live-range` | `standard` | Split live range operation in the register allocation family. |
| `registerAllocationValidateNoSpill` | `hephaestus.register_allocation.validate-no-spill` | `standard` | Validate no spill operation in the register allocation family. |
| `registerAllocationValidateRegisterBank` | `hephaestus.register_allocation.validate-register-bank` | `standard` | Validate register bank operation in the register allocation family. |

## `sass_bitfield` (21)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `sassBitfieldBinaryDotPrep` | `hephaestus.sass_bitfield.binary-dot-prep` | `standard` | Binary dot prep operation in the sass bitfield family. |
| `sassBitfieldBitFieldExtract` | `hephaestus.sass_bitfield.bit-field-extract` | `standard` | Bit field extract operation in the sass bitfield family. |
| `sassBitfieldBitFieldInsert` | `hephaestus.sass_bitfield.bit-field-insert` | `standard` | Bit field insert operation in the sass bitfield family. |
| `sassBitfieldBitMask` | `hephaestus.sass_bitfield.bit-mask` | `standard` | Bit mask operation in the sass bitfield family. |
| `sassBitfieldBitReverse` | `hephaestus.sass_bitfield.bit-reverse` | `standard` | Bit reverse operation in the sass bitfield family. |
| `sassBitfieldBytePermute` | `hephaestus.sass_bitfield.byte-permute` | `standard` | Byte permute operation in the sass bitfield family. |
| `sassBitfieldFindLeadingOne` | `hephaestus.sass_bitfield.find-leading-one` | `standard` | Find leading one operation in the sass bitfield family. |
| `sassBitfieldFunnelShift` | `hephaestus.sass_bitfield.funnel-shift` | `standard` | Funnel shift operation in the sass bitfield family. |
| `sassBitfieldInt4Decode` | `hephaestus.sass_bitfield.int4-decode` | `standard` | Int4 decode operation in the sass bitfield family. |
| `sassBitfieldInt4Pack` | `hephaestus.sass_bitfield.int4-pack` | `standard` | Int4 pack operation in the sass bitfield family. |
| `sassBitfieldLogicalAnd` | `hephaestus.sass_bitfield.logical-and` | `standard` | Logical and operation in the sass bitfield family. |
| `sassBitfieldLogicalLop3` | `hephaestus.sass_bitfield.logical-lop3` | `standard` | Logical lop3 operation in the sass bitfield family. |
| `sassBitfieldLogicalOr` | `hephaestus.sass_bitfield.logical-or` | `standard` | Logical or operation in the sass bitfield family. |
| `sassBitfieldLogicalXor` | `hephaestus.sass_bitfield.logical-xor` | `standard` | Logical xor operation in the sass bitfield family. |
| `sassBitfieldPackBits` | `hephaestus.sass_bitfield.pack-bits` | `standard` | Pack bits operation in the sass bitfield family. |
| `sassBitfieldPopulationCount` | `hephaestus.sass_bitfield.population-count` | `standard` | Population count operation in the sass bitfield family. |
| `sassBitfieldShiftLeft` | `hephaestus.sass_bitfield.shift-left` | `standard` | Shift left operation in the sass bitfield family. |
| `sassBitfieldShiftRightArithmetic` | `hephaestus.sass_bitfield.shift-right-arithmetic` | `standard` | Shift right arithmetic operation in the sass bitfield family. |
| `sassBitfieldShiftRightLogical` | `hephaestus.sass_bitfield.shift-right-logical` | `standard` | Shift right logical operation in the sass bitfield family. |
| `sassBitfieldTernaryDecode` | `hephaestus.sass_bitfield.ternary-decode` | `standard` | Ternary decode operation in the sass bitfield family. |
| `sassBitfieldUnpackBits` | `hephaestus.sass_bitfield.unpack-bits` | `standard` | Unpack bits operation in the sass bitfield family. |

## `sass_control` (26)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `sassControlBeginDivergence` | `hephaestus.sass_control.begin-divergence` | `standard` | Begin divergence operation in the sass control family. |
| `sassControlBranch` | `hephaestus.sass_control.branch` | `standard` | Branch operation in the sass control family. |
| `sassControlBranchIndexed` | `hephaestus.sass_control.branch-indexed` | `standard` | Branch indexed operation in the sass control family. |
| `sassControlBranchUniform` | `hephaestus.sass_control.branch-uniform` | `standard` | Branch uniform operation in the sass control family. |
| `sassControlBreakpoint` | `hephaestus.sass_control.breakpoint` | `standard` | Breakpoint operation in the sass control family. |
| `sassControlCall` | `hephaestus.sass_control.call` | `standard` | Call operation in the sass control family. |
| `sassControlExit` | `hephaestus.sass_control.exit` | `standard` | Exit operation in the sass control family. |
| `sassControlGetClock` | `hephaestus.sass_control.get-clock` | `standard` | Get clock operation in the sass control family. |
| `sassControlGetLaneId` | `hephaestus.sass_control.get-lane-id` | `standard` | Get lane id operation in the sass control family. |
| `sassControlGetSmId` | `hephaestus.sass_control.get-sm-id` | `standard` | Get sm id operation in the sass control family. |
| `sassControlGetSpecialRegister` | `hephaestus.sass_control.get-special-register` | `standard` | Get special register operation in the sass control family. |
| `sassControlGetWarpId` | `hephaestus.sass_control.get-warp-id` | `standard` | Get warp id operation in the sass control family. |
| `sassControlJump` | `hephaestus.sass_control.jump` | `standard` | Jump operation in the sass control family. |
| `sassControlJumpIndexed` | `hephaestus.sass_control.jump-indexed` | `standard` | Jump indexed operation in the sass control family. |
| `sassControlKillThread` | `hephaestus.sass_control.kill-thread` | `standard` | Kill thread operation in the sass control family. |
| `sassControlNanoSleep` | `hephaestus.sass_control.nano-sleep` | `standard` | Nano sleep operation in the sass control family. |
| `sassControlNop` | `hephaestus.sass_control.nop` | `standard` | Nop operation in the sass control family. |
| `sassControlPredicateToRegister` | `hephaestus.sass_control.predicate-to-register` | `standard` | Predicate to register operation in the sass control family. |
| `sassControlRegisterToPredicate` | `hephaestus.sass_control.register-to-predicate` | `standard` | Register to predicate operation in the sass control family. |
| `sassControlReturn` | `hephaestus.sass_control.return` | `standard` | Return operation in the sass control family. |
| `sassControlSelectPredicate` | `hephaestus.sass_control.select-predicate` | `standard` | Select predicate operation in the sass control family. |
| `sassControlSetPredicate` | `hephaestus.sass_control.set-predicate` | `standard` | Set predicate operation in the sass control family. |
| `sassControlSyncDivergence` | `hephaestus.sass_control.sync-divergence` | `standard` | Sync divergence operation in the sass control family. |
| `sassControlTrap` | `hephaestus.sass_control.trap` | `standard` | Trap operation in the sass control family. |
| `sassControlWarpSync` | `hephaestus.sass_control.warp-sync` | `standard` | Warp sync operation in the sass control family. |
| `sassControlYield` | `hephaestus.sass_control.yield` | `standard` | Yield operation in the sass control family. |

## `sass_convert` (15)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `sassConvertBfloatToFloat` | `hephaestus.sass_convert.bfloat-to-float` | `standard` | Bfloat to float operation in the sass convert family. |
| `sassConvertComplexPack` | `hephaestus.sass_convert.complex-pack` | `standard` | Complex pack operation in the sass convert family. |
| `sassConvertComplexUnpack` | `hephaestus.sass_convert.complex-unpack` | `standard` | Complex unpack operation in the sass convert family. |
| `sassConvertFloatToBfloat` | `hephaestus.sass_convert.float-to-bfloat` | `standard` | Float to bfloat operation in the sass convert family. |
| `sassConvertFloatToFloat` | `hephaestus.sass_convert.float-to-float` | `standard` | Float to float operation in the sass convert family. |
| `sassConvertFloatToHalf` | `hephaestus.sass_convert.float-to-half` | `standard` | Float to half operation in the sass convert family. |
| `sassConvertFloatToInt` | `hephaestus.sass_convert.float-to-int` | `standard` | Float to int operation in the sass convert family. |
| `sassConvertHalfToFloat` | `hephaestus.sass_convert.half-to-float` | `standard` | Half to float operation in the sass convert family. |
| `sassConvertIntToFloat` | `hephaestus.sass_convert.int-to-float` | `standard` | Int to float operation in the sass convert family. |
| `sassConvertIntToInt` | `hephaestus.sass_convert.int-to-int` | `standard` | Int to int operation in the sass convert family. |
| `sassConvertPackedConvert` | `hephaestus.sass_convert.packed-convert` | `standard` | Packed convert operation in the sass convert family. |
| `sassConvertRoundConvert` | `hephaestus.sass_convert.round-convert` | `standard` | Round convert operation in the sass convert family. |
| `sassConvertSaturatingConvert` | `hephaestus.sass_convert.saturating-convert` | `standard` | Saturating convert operation in the sass convert family. |
| `sassConvertScaleConvert` | `hephaestus.sass_convert.scale-convert` | `standard` | Scale convert operation in the sass convert family. |
| `sassConvertZeroPointConvert` | `hephaestus.sass_convert.zero-point-convert` | `standard` | Zero point convert operation in the sass convert family. |

## `sass_float` (22)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `sassFloatDoubleAdd` | `hephaestus.sass_float.double-add` | `standard` | Double add operation in the sass float family. |
| `sassFloatDoubleFma` | `hephaestus.sass_float.double-fma` | `standard` | Double fma operation in the sass float family. |
| `sassFloatDoubleMul` | `hephaestus.sass_float.double-mul` | `standard` | Double mul operation in the sass float family. |
| `sassFloatDoubleSetPredicate` | `hephaestus.sass_float.double-set-predicate` | `standard` | Double set predicate operation in the sass float family. |
| `sassFloatFloatAbs` | `hephaestus.sass_float.float-abs` | `standard` | Float abs operation in the sass float family. |
| `sassFloatFloatAdd` | `hephaestus.sass_float.float-add` | `standard` | Float add operation in the sass float family. |
| `sassFloatFloatCosApprox` | `hephaestus.sass_float.float-cos-approx` | `standard` | Float cos approx operation in the sass float family. |
| `sassFloatFloatExp2Approx` | `hephaestus.sass_float.float-exp2-approx` | `standard` | Float exp2 approx operation in the sass float family. |
| `sassFloatFloatFma` | `hephaestus.sass_float.float-fma` | `standard` | Float fma operation in the sass float family. |
| `sassFloatFloatLog2Approx` | `hephaestus.sass_float.float-log2-approx` | `standard` | Float log2 approx operation in the sass float family. |
| `sassFloatFloatMinMax` | `hephaestus.sass_float.float-min-max` | `standard` | Float min max operation in the sass float family. |
| `sassFloatFloatMul` | `hephaestus.sass_float.float-mul` | `standard` | Float mul operation in the sass float family. |
| `sassFloatFloatNeg` | `hephaestus.sass_float.float-neg` | `standard` | Float neg operation in the sass float family. |
| `sassFloatFloatReciprocalApprox` | `hephaestus.sass_float.float-reciprocal-approx` | `standard` | Float reciprocal approx operation in the sass float family. |
| `sassFloatFloatRound` | `hephaestus.sass_float.float-round` | `standard` | Float round operation in the sass float family. |
| `sassFloatFloatRsqrtApprox` | `hephaestus.sass_float.float-rsqrt-approx` | `standard` | Float rsqrt approx operation in the sass float family. |
| `sassFloatFloatSet` | `hephaestus.sass_float.float-set` | `standard` | Float set operation in the sass float family. |
| `sassFloatFloatSetPredicate` | `hephaestus.sass_float.float-set-predicate` | `standard` | Float set predicate operation in the sass float family. |
| `sassFloatFloatSinApprox` | `hephaestus.sass_float.float-sin-approx` | `standard` | Float sin approx operation in the sass float family. |
| `sassFloatFloatSqrt` | `hephaestus.sass_float.float-sqrt` | `standard` | Float sqrt operation in the sass float family. |
| `sassFloatFloatTanhApprox` | `hephaestus.sass_float.float-tanh-approx` | `standard` | Float tanh approx operation in the sass float family. |
| `sassFloatRangeReduce` | `hephaestus.sass_float.range-reduce` | `standard` | Range reduce operation in the sass float family. |

## `sass_half` (13)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `sassHalfBfloat162Add` | `hephaestus.sass_half.bfloat162-add` | `standard` | Bfloat162 add operation in the sass half family. |
| `sassHalfBfloat162Fma` | `hephaestus.sass_half.bfloat162-fma` | `standard` | Bfloat162 fma operation in the sass half family. |
| `sassHalfBfloat162Mul` | `hephaestus.sass_half.bfloat162-mul` | `standard` | Bfloat162 mul operation in the sass half family. |
| `sassHalfBfloat162SetPredicate` | `hephaestus.sass_half.bfloat162-set-predicate` | `standard` | Bfloat162 set predicate operation in the sass half family. |
| `sassHalfHalf2Add` | `hephaestus.sass_half.half2-add` | `standard` | Half2 add operation in the sass half family. |
| `sassHalfHalf2Fma` | `hephaestus.sass_half.half2-fma` | `standard` | Half2 fma operation in the sass half family. |
| `sassHalfHalf2MinMax` | `hephaestus.sass_half.half2-min-max` | `standard` | Half2 min max operation in the sass half family. |
| `sassHalfHalf2Mul` | `hephaestus.sass_half.half2-mul` | `standard` | Half2 mul operation in the sass half family. |
| `sassHalfHalf2Relu` | `hephaestus.sass_half.half2-relu` | `standard` | Half2 relu operation in the sass half family. |
| `sassHalfHalf2Set` | `hephaestus.sass_half.half2-set` | `standard` | Half2 set operation in the sass half family. |
| `sassHalfHalf2SetPredicate` | `hephaestus.sass_half.half2-set-predicate` | `standard` | Half2 set predicate operation in the sass half family. |
| `sassHalfPackedBfloatConvert` | `hephaestus.sass_half.packed-bfloat-convert` | `standard` | Packed bfloat convert operation in the sass half family. |
| `sassHalfPackedHalfConvert` | `hephaestus.sass_half.packed-half-convert` | `standard` | Packed half convert operation in the sass half family. |

## `sass_integer` (25)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `sassIntegerBorrowSub` | `hephaestus.sass_integer.borrow-sub` | `standard` | Borrow sub operation in the sass integer family. |
| `sassIntegerCarryAdd` | `hephaestus.sass_integer.carry-add` | `standard` | Carry add operation in the sass integer family. |
| `sassIntegerIntegerAbs` | `hephaestus.sass_integer.integer-abs` | `standard` | Integer abs operation in the sass integer family. |
| `sassIntegerIntegerAdd` | `hephaestus.sass_integer.integer-add` | `standard` | Integer add operation in the sass integer family. |
| `sassIntegerIntegerAdd3` | `hephaestus.sass_integer.integer-add3` | `standard` | Integer add3 operation in the sass integer family. |
| `sassIntegerIntegerAverage` | `hephaestus.sass_integer.integer-average` | `standard` | Integer average operation in the sass integer family. |
| `sassIntegerIntegerClamp` | `hephaestus.sass_integer.integer-clamp` | `standard` | Integer clamp operation in the sass integer family. |
| `sassIntegerIntegerCompare` | `hephaestus.sass_integer.integer-compare` | `standard` | Integer compare operation in the sass integer family. |
| `sassIntegerIntegerDivide` | `hephaestus.sass_integer.integer-divide` | `standard` | Integer divide operation in the sass integer family. |
| `sassIntegerIntegerDot2` | `hephaestus.sass_integer.integer-dot2` | `standard` | Integer dot2 operation in the sass integer family. |
| `sassIntegerIntegerDot4` | `hephaestus.sass_integer.integer-dot4` | `standard` | Integer dot4 operation in the sass integer family. |
| `sassIntegerIntegerMinMax` | `hephaestus.sass_integer.integer-min-max` | `standard` | Integer min max operation in the sass integer family. |
| `sassIntegerIntegerMultiply` | `hephaestus.sass_integer.integer-multiply` | `standard` | Integer multiply operation in the sass integer family. |
| `sassIntegerIntegerMultiplyAdd` | `hephaestus.sass_integer.integer-multiply-add` | `standard` | Integer multiply add operation in the sass integer family. |
| `sassIntegerIntegerMultiplyHigh` | `hephaestus.sass_integer.integer-multiply-high` | `standard` | Integer multiply high operation in the sass integer family. |
| `sassIntegerIntegerNeg` | `hephaestus.sass_integer.integer-neg` | `standard` | Integer neg operation in the sass integer family. |
| `sassIntegerIntegerReciprocalApprox` | `hephaestus.sass_integer.integer-reciprocal-approx` | `standard` | Integer reciprocal approx operation in the sass integer family. |
| `sassIntegerIntegerRemainder` | `hephaestus.sass_integer.integer-remainder` | `standard` | Integer remainder operation in the sass integer family. |
| `sassIntegerIntegerSad` | `hephaestus.sass_integer.integer-sad` | `standard` | Integer sad operation in the sass integer family. |
| `sassIntegerIntegerSaturate` | `hephaestus.sass_integer.integer-saturate` | `standard` | Integer saturate operation in the sass integer family. |
| `sassIntegerIntegerSetPredicate` | `hephaestus.sass_integer.integer-set-predicate` | `standard` | Integer set predicate operation in the sass integer family. |
| `sassIntegerIntegerSub` | `hephaestus.sass_integer.integer-sub` | `standard` | Integer sub operation in the sass integer family. |
| `sassIntegerLea` | `hephaestus.sass_integer.lea` | `standard` | Lea operation in the sass integer family. |
| `sassIntegerWideAdd` | `hephaestus.sass_integer.wide-add` | `standard` | Wide add operation in the sass integer family. |
| `sassIntegerWideMultiply` | `hephaestus.sass_integer.wide-multiply` | `standard` | Wide multiply operation in the sass integer family. |

## `sass_memory` (34)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `sassMemoryAcquireLoad` | `hephaestus.sass_memory.acquire-load` | `standard` | Acquire load operation in the sass memory family. |
| `sassMemoryAsyncCopyCommit` | `hephaestus.sass_memory.async-copy-commit` | `standard` | Async copy commit operation in the sass memory family. |
| `sassMemoryAsyncCopyWait` | `hephaestus.sass_memory.async-copy-wait` | `standard` | Async copy wait operation in the sass memory family. |
| `sassMemoryAsyncGlobalToShared` | `hephaestus.sass_memory.async-global-to-shared` | `standard` | Async global to shared operation in the sass memory family. |
| `sassMemoryAtomicGlobal` | `hephaestus.sass_memory.atomic-global` | `standard` | Atomic global operation in the sass memory family. |
| `sassMemoryAtomicShared` | `hephaestus.sass_memory.atomic-shared` | `standard` | Atomic shared operation in the sass memory family. |
| `sassMemoryCacheControl` | `hephaestus.sass_memory.cache-control` | `standard` | Cache control operation in the sass memory family. |
| `sassMemoryCacheInvalidate` | `hephaestus.sass_memory.cache-invalidate` | `standard` | Cache invalidate operation in the sass memory family. |
| `sassMemoryLoadConstant` | `hephaestus.sass_memory.load-constant` | `standard` | Load constant operation in the sass memory family. |
| `sassMemoryLoadGeneric` | `hephaestus.sass_memory.load-generic` | `standard` | Load generic operation in the sass memory family. |
| `sassMemoryLoadGlobal` | `hephaestus.sass_memory.load-global` | `standard` | Load global operation in the sass memory family. |
| `sassMemoryLoadGlobalVector` | `hephaestus.sass_memory.load-global-vector` | `standard` | Load global vector operation in the sass memory family. |
| `sassMemoryLoadLocal` | `hephaestus.sass_memory.load-local` | `standard` | Load local operation in the sass memory family. |
| `sassMemoryLoadShared` | `hephaestus.sass_memory.load-shared` | `standard` | Load shared operation in the sass memory family. |
| `sassMemoryLoadSharedMatrix` | `hephaestus.sass_memory.load-shared-matrix` | `standard` | Load shared matrix operation in the sass memory family. |
| `sassMemoryLoadUniformConstant` | `hephaestus.sass_memory.load-uniform-constant` | `standard` | Load uniform constant operation in the sass memory family. |
| `sassMemoryMemoryBarrierCta` | `hephaestus.sass_memory.memory-barrier-cta` | `standard` | Memory barrier cta operation in the sass memory family. |
| `sassMemoryMemoryBarrierGpu` | `hephaestus.sass_memory.memory-barrier-gpu` | `standard` | Memory barrier gpu operation in the sass memory family. |
| `sassMemoryMemoryBarrierSystem` | `hephaestus.sass_memory.memory-barrier-system` | `standard` | Memory barrier system operation in the sass memory family. |
| `sassMemoryNonCoherentLoad` | `hephaestus.sass_memory.non-coherent-load` | `standard` | Non coherent load operation in the sass memory family. |
| `sassMemoryPrefetchGlobal` | `hephaestus.sass_memory.prefetch-global` | `standard` | Prefetch global operation in the sass memory family. |
| `sassMemoryReduceGlobal` | `hephaestus.sass_memory.reduce-global` | `standard` | Reduce global operation in the sass memory family. |
| `sassMemoryReduceShared` | `hephaestus.sass_memory.reduce-shared` | `standard` | Reduce shared operation in the sass memory family. |
| `sassMemoryReleaseStore` | `hephaestus.sass_memory.release-store` | `standard` | Release store operation in the sass memory family. |
| `sassMemoryStoreGeneric` | `hephaestus.sass_memory.store-generic` | `standard` | Store generic operation in the sass memory family. |
| `sassMemoryStoreGlobal` | `hephaestus.sass_memory.store-global` | `standard` | Store global operation in the sass memory family. |
| `sassMemoryStoreGlobalVector` | `hephaestus.sass_memory.store-global-vector` | `standard` | Store global vector operation in the sass memory family. |
| `sassMemoryStoreLocal` | `hephaestus.sass_memory.store-local` | `standard` | Store local operation in the sass memory family. |
| `sassMemoryStoreShared` | `hephaestus.sass_memory.store-shared` | `standard` | Store shared operation in the sass memory family. |
| `sassMemoryStreamingLoad` | `hephaestus.sass_memory.streaming-load` | `standard` | Streaming load operation in the sass memory family. |
| `sassMemoryVolatileLoad` | `hephaestus.sass_memory.volatile-load` | `standard` | Volatile load operation in the sass memory family. |
| `sassMemoryVolatileStore` | `hephaestus.sass_memory.volatile-store` | `standard` | Volatile store operation in the sass memory family. |
| `sassMemoryWriteBackStore` | `hephaestus.sass_memory.write-back-store` | `standard` | Write back store operation in the sass memory family. |
| `sassMemoryWriteThroughStore` | `hephaestus.sass_memory.write-through-store` | `standard` | Write through store operation in the sass memory family. |

## `sass_tensor` (16)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `sassTensorBmmaBinary` | `hephaestus.sass_tensor.bmma-binary` | `standard` | Bmma binary operation in the sass tensor family. |
| `sassTensorHmmaBf16` | `hephaestus.sass_tensor.hmma-bf16` | `standard` | Hmma bf16 operation in the sass tensor family. |
| `sassTensorHmmaFp16` | `hephaestus.sass_tensor.hmma-fp16` | `standard` | Hmma fp16 operation in the sass tensor family. |
| `sassTensorHmmaTf32` | `hephaestus.sass_tensor.hmma-tf32` | `standard` | Hmma tf32 operation in the sass tensor family. |
| `sassTensorImmaInt4` | `hephaestus.sass_tensor.imma-int4` | `standard` | Imma int4 operation in the sass tensor family. |
| `sassTensorImmaInt8` | `hephaestus.sass_tensor.imma-int8` | `standard` | Imma int8 operation in the sass tensor family. |
| `sassTensorMatrixAccumulatorConvert` | `hephaestus.sass_tensor.matrix-accumulator-convert` | `standard` | Matrix accumulator convert operation in the sass tensor family. |
| `sassTensorMatrixAccumulatorEpilogue` | `hephaestus.sass_tensor.matrix-accumulator-epilogue` | `standard` | Matrix accumulator epilogue operation in the sass tensor family. |
| `sassTensorMatrixAccumulatorScale` | `hephaestus.sass_tensor.matrix-accumulator-scale` | `standard` | Matrix accumulator scale operation in the sass tensor family. |
| `sassTensorMatrixFillFragment` | `hephaestus.sass_tensor.matrix-fill-fragment` | `standard` | Matrix fill fragment operation in the sass tensor family. |
| `sassTensorMatrixLoadShared` | `hephaestus.sass_tensor.matrix-load-shared` | `standard` | Matrix load shared operation in the sass tensor family. |
| `sassTensorMatrixStoreShared` | `hephaestus.sass_tensor.matrix-store-shared` | `standard` | Matrix store shared operation in the sass tensor family. |
| `sassTensorMatrixTransposeFragment` | `hephaestus.sass_tensor.matrix-transpose-fragment` | `standard` | Matrix transpose fragment operation in the sass tensor family. |
| `sassTensorMmaSparseBf16` | `hephaestus.sass_tensor.mma-sparse-bf16` | `standard` | Mma sparse bf16 operation in the sass tensor family. |
| `sassTensorMmaSparseFp16` | `hephaestus.sass_tensor.mma-sparse-fp16` | `standard` | Mma sparse fp16 operation in the sass tensor family. |
| `sassTensorMmaSparseInt8` | `hephaestus.sass_tensor.mma-sparse-int8` | `standard` | Mma sparse int8 operation in the sass tensor family. |

## `sass_warp` (21)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `sassWarpBallot` | `hephaestus.sass_warp.ballot` | `standard` | Ballot operation in the sass warp family. |
| `sassWarpMatchAll` | `hephaestus.sass_warp.match-all` | `standard` | Match all operation in the sass warp family. |
| `sassWarpMatchAny` | `hephaestus.sass_warp.match-any` | `standard` | Match any operation in the sass warp family. |
| `sassWarpShuffleDown` | `hephaestus.sass_warp.shuffle-down` | `standard` | Shuffle down operation in the sass warp family. |
| `sassWarpShuffleIndex` | `hephaestus.sass_warp.shuffle-index` | `standard` | Shuffle index operation in the sass warp family. |
| `sassWarpShuffleUp` | `hephaestus.sass_warp.shuffle-up` | `standard` | Shuffle up operation in the sass warp family. |
| `sassWarpShuffleXor` | `hephaestus.sass_warp.shuffle-xor` | `standard` | Shuffle xor operation in the sass warp family. |
| `sassWarpVoteAll` | `hephaestus.sass_warp.vote-all` | `standard` | Vote all operation in the sass warp family. |
| `sassWarpVoteAny` | `hephaestus.sass_warp.vote-any` | `standard` | Vote any operation in the sass warp family. |
| `sassWarpVoteUniform` | `hephaestus.sass_warp.vote-uniform` | `standard` | Vote uniform operation in the sass warp family. |
| `sassWarpWarpBroadcast` | `hephaestus.sass_warp.warp-broadcast` | `standard` | Warp broadcast operation in the sass warp family. |
| `sassWarpWarpCompact` | `hephaestus.sass_warp.warp-compact` | `standard` | Warp compact operation in the sass warp family. |
| `sassWarpWarpElectLeader` | `hephaestus.sass_warp.warp-elect-leader` | `standard` | Warp elect leader operation in the sass warp family. |
| `sassWarpWarpExchange` | `hephaestus.sass_warp.warp-exchange` | `standard` | Warp exchange operation in the sass warp family. |
| `sassWarpWarpPrefixSum` | `hephaestus.sass_warp.warp-prefix-sum` | `standard` | Warp prefix sum operation in the sass warp family. |
| `sassWarpWarpReduceAdd` | `hephaestus.sass_warp.warp-reduce-add` | `standard` | Warp reduce add operation in the sass warp family. |
| `sassWarpWarpReduceAnd` | `hephaestus.sass_warp.warp-reduce-and` | `standard` | Warp reduce and operation in the sass warp family. |
| `sassWarpWarpReduceMax` | `hephaestus.sass_warp.warp-reduce-max` | `standard` | Warp reduce max operation in the sass warp family. |
| `sassWarpWarpReduceMin` | `hephaestus.sass_warp.warp-reduce-min` | `standard` | Warp reduce min operation in the sass warp family. |
| `sassWarpWarpReduceOr` | `hephaestus.sass_warp.warp-reduce-or` | `standard` | Warp reduce or operation in the sass warp family. |
| `sassWarpWarpReduceXor` | `hephaestus.sass_warp.warp-reduce-xor` | `standard` | Warp reduce xor operation in the sass warp family. |

# Chronos


## `dependency` (17)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `dependencyAddDependency` | `chronos.dependency.add-dependency` | `standard` | Add dependency operation in the dependency family. |
| `dependencyBarrierDependency` | `chronos.dependency.barrier-dependency` | `standard` | Barrier dependency operation in the dependency family. |
| `dependencyCriticalPath` | `chronos.dependency.critical-path` | `standard` | Critical path operation in the dependency family. |
| `dependencyCrossDeviceDependency` | `chronos.dependency.cross-device-dependency` | `standard` | Cross device dependency operation in the dependency family. |
| `dependencyCrossQueueDependency` | `chronos.dependency.cross-queue-dependency` | `standard` | Cross queue dependency operation in the dependency family. |
| `dependencyDependencyBatch` | `chronos.dependency.dependency-batch` | `standard` | Dependency batch operation in the dependency family. |
| `dependencyDependencyToken` | `chronos.dependency.dependency-token` | `standard` | Dependency token operation in the dependency family. |
| `dependencyDetectCycle` | `chronos.dependency.detect-cycle` | `standard` | Detect cycle operation in the dependency family. |
| `dependencyEventDependency` | `chronos.dependency.event-dependency` | `standard` | Event dependency operation in the dependency family. |
| `dependencyExecutionDependency` | `chronos.dependency.execution-dependency` | `standard` | Execution dependency operation in the dependency family. |
| `dependencyMemoryDependency` | `chronos.dependency.memory-dependency` | `standard` | Memory dependency operation in the dependency family. |
| `dependencyReadAfterWrite` | `chronos.dependency.read-after-write` | `standard` | Read after write operation in the dependency family. |
| `dependencyRemoveDependency` | `chronos.dependency.remove-dependency` | `standard` | Remove dependency operation in the dependency family. |
| `dependencyResolveDependencies` | `chronos.dependency.resolve-dependencies` | `standard` | Resolve dependencies operation in the dependency family. |
| `dependencyTopologicalOrder` | `chronos.dependency.topological-order` | `standard` | Topological order operation in the dependency family. |
| `dependencyWriteAfterRead` | `chronos.dependency.write-after-read` | `standard` | Write after read operation in the dependency family. |
| `dependencyWriteAfterWrite` | `chronos.dependency.write-after-write` | `standard` | Write after write operation in the dependency family. |

## `fence_semaphore` (17)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `fenceSemaphoreBinarySemaphore` | `chronos.fence_semaphore.binary-semaphore` | `standard` | Binary semaphore operation in the fence semaphore family. |
| `fenceSemaphoreCreateFence` | `chronos.fence_semaphore.create-fence` | `standard` | Create fence operation in the fence semaphore family. |
| `fenceSemaphoreCreateSemaphore` | `chronos.fence_semaphore.create-semaphore` | `standard` | Create semaphore operation in the fence semaphore family. |
| `fenceSemaphoreDestroyFence` | `chronos.fence_semaphore.destroy-fence` | `standard` | Destroy fence operation in the fence semaphore family. |
| `fenceSemaphoreDestroySemaphore` | `chronos.fence_semaphore.destroy-semaphore` | `standard` | Destroy semaphore operation in the fence semaphore family. |
| `fenceSemaphoreDoorbellSemaphore` | `chronos.fence_semaphore.doorbell-semaphore` | `standard` | Doorbell semaphore operation in the fence semaphore family. |
| `fenceSemaphoreEventfdBridge` | `chronos.fence_semaphore.eventfd-bridge` | `standard` | Eventfd bridge operation in the fence semaphore family. |
| `fenceSemaphoreGpuWrittenSemaphore` | `chronos.fence_semaphore.gpu-written-semaphore` | `standard` | Gpu written semaphore operation in the fence semaphore family. |
| `fenceSemaphorePollFence` | `chronos.fence_semaphore.poll-fence` | `standard` | Poll fence operation in the fence semaphore family. |
| `fenceSemaphoreSemaphoreAcquire` | `chronos.fence_semaphore.semaphore-acquire` | `standard` | Semaphore acquire operation in the fence semaphore family. |
| `fenceSemaphoreSemaphoreRelease` | `chronos.fence_semaphore.semaphore-release` | `standard` | Semaphore release operation in the fence semaphore family. |
| `fenceSemaphoreSemaphoreTryWait` | `chronos.fence_semaphore.semaphore-try-wait` | `standard` | Semaphore try wait operation in the fence semaphore family. |
| `fenceSemaphoreSignalFence` | `chronos.fence_semaphore.signal-fence` | `standard` | Signal fence operation in the fence semaphore family. |
| `fenceSemaphoreSignalSemaphore` | `chronos.fence_semaphore.signal-semaphore` | `standard` | Signal semaphore operation in the fence semaphore family. |
| `fenceSemaphoreTimelineSemaphore` | `chronos.fence_semaphore.timeline-semaphore` | `standard` | Timeline semaphore operation in the fence semaphore family. |
| `fenceSemaphoreWaitFence` | `chronos.fence_semaphore.wait-fence` | `standard` | Wait fence operation in the fence semaphore family. |
| `fenceSemaphoreWaitSemaphore` | `chronos.fence_semaphore.wait-semaphore` | `standard` | Wait semaphore operation in the fence semaphore family. |

## `profiling` (18)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `profilingBarrierWaitSample` | `chronos.profiling.barrier-wait-sample` | `standard` | Barrier wait sample operation in the profiling family. |
| `profilingCalibrateClock` | `chronos.profiling.calibrate-clock` | `standard` | Calibrate clock operation in the profiling family. |
| `profilingCorrelateHostGpuClock` | `chronos.profiling.correlate-host-gpu-clock` | `standard` | Correlate host gpu clock operation in the profiling family. |
| `profilingCounterSample` | `chronos.profiling.counter-sample` | `standard` | Counter sample operation in the profiling family. |
| `profilingDependencyTrace` | `chronos.profiling.dependency-trace` | `standard` | Dependency trace operation in the profiling family. |
| `profilingEventTrace` | `chronos.profiling.event-trace` | `standard` | Event trace operation in the profiling family. |
| `profilingExportChromeTrace` | `chronos.profiling.export-chrome-trace` | `standard` | Export chrome trace operation in the profiling family. |
| `profilingExportPerfettoTrace` | `chronos.profiling.export-perfetto-trace` | `standard` | Export perfetto trace operation in the profiling family. |
| `profilingLatencyHistogram` | `chronos.profiling.latency-histogram` | `standard` | Latency histogram operation in the profiling family. |
| `profilingMarkBegin` | `chronos.profiling.mark-begin` | `standard` | Mark begin operation in the profiling family. |
| `profilingMarkEnd` | `chronos.profiling.mark-end` | `standard` | Mark end operation in the profiling family. |
| `profilingMergeTraces` | `chronos.profiling.merge-traces` | `standard` | Merge traces operation in the profiling family. |
| `profilingOccupancySample` | `chronos.profiling.occupancy-sample` | `standard` | Occupancy sample operation in the profiling family. |
| `profilingQueueDepthSample` | `chronos.profiling.queue-depth-sample` | `standard` | Queue depth sample operation in the profiling family. |
| `profilingStallReasonSample` | `chronos.profiling.stall-reason-sample` | `standard` | Stall reason sample operation in the profiling family. |
| `profilingTimelineTrace` | `chronos.profiling.timeline-trace` | `standard` | Timeline trace operation in the profiling family. |
| `profilingTimestampGpu` | `chronos.profiling.timestamp-gpu` | `standard` | Timestamp gpu operation in the profiling family. |
| `profilingTimestampHost` | `chronos.profiling.timestamp-host` | `standard` | Timestamp host operation in the profiling family. |

## `quiescence` (16)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `quiescenceAcquireCredit` | `chronos.quiescence.acquire-credit` | `research` | Acquire credit operation in the quiescence family. |
| `quiescenceAwaitQuiescence` | `chronos.quiescence.await-quiescence` | `research` | Await quiescence operation in the quiescence family. |
| `quiescenceCheckQuiescence` | `chronos.quiescence.check-quiescence` | `research` | Check quiescence operation in the quiescence family. |
| `quiescenceCompletePendingIo` | `chronos.quiescence.complete-pending-io` | `research` | Complete pending io operation in the quiescence family. |
| `quiescenceCountInFlight` | `chronos.quiescence.count-in-flight` | `research` | Count in flight operation in the quiescence family. |
| `quiescenceCountMailboxItems` | `chronos.quiescence.count-mailbox-items` | `research` | Count mailbox items operation in the quiescence family. |
| `quiescenceCreateCreditPool` | `chronos.quiescence.create-credit-pool` | `research` | Create credit pool operation in the quiescence family. |
| `quiescenceDetectCreditLeak` | `chronos.quiescence.detect-credit-leak` | `research` | Detect credit leak operation in the quiescence family. |
| `quiescenceDetectDuplicateCredit` | `chronos.quiescence.detect-duplicate-credit` | `research` | Detect duplicate credit operation in the quiescence family. |
| `quiescenceDetectStaleCredit` | `chronos.quiescence.detect-stale-credit` | `research` | Detect stale credit operation in the quiescence family. |
| `quiescenceForceQuiescence` | `chronos.quiescence.force-quiescence` | `research` | Force quiescence operation in the quiescence family. |
| `quiescenceQuiescenceTimeout` | `chronos.quiescence.quiescence-timeout` | `research` | Quiescence timeout operation in the quiescence family. |
| `quiescenceQuiescenceTrace` | `chronos.quiescence.quiescence-trace` | `research` | Quiescence trace operation in the quiescence family. |
| `quiescenceRegisterPendingIo` | `chronos.quiescence.register-pending-io` | `research` | Register pending io operation in the quiescence family. |
| `quiescenceReleaseCredit` | `chronos.quiescence.release-credit` | `research` | Release credit operation in the quiescence family. |
| `quiescenceTransferCredit` | `chronos.quiescence.transfer-credit` | `research` | Transfer credit operation in the quiescence family. |

## `scheduler` (28)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `schedulerAssignQueue` | `chronos.scheduler.assign-queue` | `standard` | Assign queue operation in the scheduler family. |
| `schedulerAssignSmAffinityHint` | `chronos.scheduler.assign-sm-affinity-hint` | `standard` | Assign sm affinity hint operation in the scheduler family. |
| `schedulerBatchTasks` | `chronos.scheduler.batch-tasks` | `standard` | Batch tasks operation in the scheduler family. |
| `schedulerCancelTask` | `chronos.scheduler.cancel-task` | `standard` | Cancel task operation in the scheduler family. |
| `schedulerCoalesceTasks` | `chronos.scheduler.coalesce-tasks` | `standard` | Coalesce tasks operation in the scheduler family. |
| `schedulerConsumeBudget` | `chronos.scheduler.consume-budget` | `standard` | Consume budget operation in the scheduler family. |
| `schedulerCountercurrentSchedule` | `chronos.scheduler.countercurrent-schedule` | `research` | Countercurrent schedule operation in the scheduler family. |
| `schedulerCreateSchedule` | `chronos.scheduler.create-schedule` | `standard` | Create schedule operation in the scheduler family. |
| `schedulerDequeueTask` | `chronos.scheduler.dequeue-task` | `standard` | Dequeue task operation in the scheduler family. |
| `schedulerDestroySchedule` | `chronos.scheduler.destroy-schedule` | `standard` | Destroy schedule operation in the scheduler family. |
| `schedulerDonateTask` | `chronos.scheduler.donate-task` | `standard` | Donate task operation in the scheduler family. |
| `schedulerEnqueueTask` | `chronos.scheduler.enqueue-task` | `standard` | Enqueue task operation in the scheduler family. |
| `schedulerPersistentSchedule` | `chronos.scheduler.persistent-schedule` | `standard` | Persistent schedule operation in the scheduler family. |
| `schedulerPipelineSchedule` | `chronos.scheduler.pipeline-schedule` | `standard` | Pipeline schedule operation in the scheduler family. |
| `schedulerPrioritizeTask` | `chronos.scheduler.prioritize-task` | `standard` | Prioritize task operation in the scheduler family. |
| `schedulerProducerConsumerSchedule` | `chronos.scheduler.producer-consumer-schedule` | `standard` | Producer consumer schedule operation in the scheduler family. |
| `schedulerReplaySchedule` | `chronos.scheduler.replay-schedule` | `standard` | Replay schedule operation in the scheduler family. |
| `schedulerSetBudget` | `chronos.scheduler.set-budget` | `standard` | Set budget operation in the scheduler family. |
| `schedulerSetDeadline` | `chronos.scheduler.set-deadline` | `standard` | Set deadline operation in the scheduler family. |
| `schedulerSetDeterminism` | `chronos.scheduler.set-determinism` | `standard` | Set determinism operation in the scheduler family. |
| `schedulerSetFairness` | `chronos.scheduler.set-fairness` | `standard` | Set fairness operation in the scheduler family. |
| `schedulerSetPriority` | `chronos.scheduler.set-priority` | `standard` | Set priority operation in the scheduler family. |
| `schedulerSleepTask` | `chronos.scheduler.sleep-task` | `standard` | Sleep task operation in the scheduler family. |
| `schedulerSplitTask` | `chronos.scheduler.split-task` | `standard` | Split task operation in the scheduler family. |
| `schedulerStealTask` | `chronos.scheduler.steal-task` | `standard` | Steal task operation in the scheduler family. |
| `schedulerWakeTask` | `chronos.scheduler.wake-task` | `standard` | Wake task operation in the scheduler family. |
| `schedulerWavefrontSchedule` | `chronos.scheduler.wavefront-schedule` | `standard` | Wavefront schedule operation in the scheduler family. |
| `schedulerYieldTask` | `chronos.scheduler.yield-task` | `standard` | Yield task operation in the scheduler family. |

## `timeline` (13)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `timelineCreateTimeline` | `chronos.timeline.create-timeline` | `standard` | Create timeline operation in the timeline family. |
| `timelineDestroyTimeline` | `chronos.timeline.destroy-timeline` | `standard` | Destroy timeline operation in the timeline family. |
| `timelineExportTimeline` | `chronos.timeline.export-timeline` | `standard` | Export timeline operation in the timeline family. |
| `timelineImportTimeline` | `chronos.timeline.import-timeline` | `standard` | Import timeline operation in the timeline family. |
| `timelineNextTimelineValue` | `chronos.timeline.next-timeline-value` | `standard` | Next timeline value operation in the timeline family. |
| `timelinePollTimeline` | `chronos.timeline.poll-timeline` | `standard` | Poll timeline operation in the timeline family. |
| `timelineQueryTimeline` | `chronos.timeline.query-timeline` | `standard` | Query timeline operation in the timeline family. |
| `timelineResetTimeline` | `chronos.timeline.reset-timeline` | `standard` | Reset timeline operation in the timeline family. |
| `timelineSignalTimelineGpu` | `chronos.timeline.signal-timeline-gpu` | `standard` | Signal timeline gpu operation in the timeline family. |
| `timelineSignalTimelineHost` | `chronos.timeline.signal-timeline-host` | `standard` | Signal timeline host operation in the timeline family. |
| `timelineValidateMonotonicity` | `chronos.timeline.validate-monotonicity` | `standard` | Validate monotonicity operation in the timeline family. |
| `timelineWaitTimelineGpu` | `chronos.timeline.wait-timeline-gpu` | `standard` | Wait timeline gpu operation in the timeline family. |
| `timelineWaitTimelineHost` | `chronos.timeline.wait-timeline-host` | `standard` | Wait timeline host operation in the timeline family. |

## `watchdog_recovery` (17)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `watchdogRecoveryAbortChannel` | `chronos.watchdog_recovery.abort-channel` | `standard` | Abort channel operation in the watchdog recovery family. |
| `watchdogRecoveryAbortQueue` | `chronos.watchdog_recovery.abort-queue` | `standard` | Abort queue operation in the watchdog recovery family. |
| `watchdogRecoveryArmWatchdog` | `chronos.watchdog_recovery.arm-watchdog` | `standard` | Arm watchdog operation in the watchdog recovery family. |
| `watchdogRecoveryCaptureHangState` | `chronos.watchdog_recovery.capture-hang-state` | `standard` | Capture hang state operation in the watchdog recovery family. |
| `watchdogRecoveryDetectDeadlock` | `chronos.watchdog_recovery.detect-deadlock` | `standard` | Detect deadlock operation in the watchdog recovery family. |
| `watchdogRecoveryDetectHang` | `chronos.watchdog_recovery.detect-hang` | `standard` | Detect hang operation in the watchdog recovery family. |
| `watchdogRecoveryDetectLivelock` | `chronos.watchdog_recovery.detect-livelock` | `standard` | Detect livelock operation in the watchdog recovery family. |
| `watchdogRecoveryDetectTimeout` | `chronos.watchdog_recovery.detect-timeout` | `standard` | Detect timeout operation in the watchdog recovery family. |
| `watchdogRecoveryDisarmWatchdog` | `chronos.watchdog_recovery.disarm-watchdog` | `standard` | Disarm watchdog operation in the watchdog recovery family. |
| `watchdogRecoveryEmitFaultReport` | `chronos.watchdog_recovery.emit-fault-report` | `standard` | Emit fault report operation in the watchdog recovery family. |
| `watchdogRecoveryFailClosed` | `chronos.watchdog_recovery.fail-closed` | `standard` | Fail closed operation in the watchdog recovery family. |
| `watchdogRecoveryHeartbeat` | `chronos.watchdog_recovery.heartbeat` | `standard` | Heartbeat operation in the watchdog recovery family. |
| `watchdogRecoveryMarkDeviceLost` | `chronos.watchdog_recovery.mark-device-lost` | `standard` | Mark device lost operation in the watchdog recovery family. |
| `watchdogRecoveryRecoverDevice` | `chronos.watchdog_recovery.recover-device` | `standard` | Recover device operation in the watchdog recovery family. |
| `watchdogRecoveryResetSchedule` | `chronos.watchdog_recovery.reset-schedule` | `standard` | Reset schedule operation in the watchdog recovery family. |
| `watchdogRecoveryRetryFromCheckpoint` | `chronos.watchdog_recovery.retry-from-checkpoint` | `standard` | Retry from checkpoint operation in the watchdog recovery family. |
| `watchdogRecoveryRollbackEpoch` | `chronos.watchdog_recovery.rollback-epoch` | `standard` | Rollback epoch operation in the watchdog recovery family. |

## `wave_epoch` (20)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `waveEpochAdvanceEpoch` | `chronos.wave_epoch.advance-epoch` | `standard` | Advance epoch operation in the wave epoch family. |
| `waveEpochAdvanceWave` | `chronos.wave_epoch.advance-wave` | `standard` | Advance wave operation in the wave epoch family. |
| `waveEpochBeginEpoch` | `chronos.wave_epoch.begin-epoch` | `standard` | Begin epoch operation in the wave epoch family. |
| `waveEpochBeginWave` | `chronos.wave_epoch.begin-wave` | `standard` | Begin wave operation in the wave epoch family. |
| `waveEpochEndEpoch` | `chronos.wave_epoch.end-epoch` | `standard` | End epoch operation in the wave epoch family. |
| `waveEpochEndWave` | `chronos.wave_epoch.end-wave` | `standard` | End wave operation in the wave epoch family. |
| `waveEpochFederationEpoch` | `chronos.wave_epoch.federation-epoch` | `standard` | Federation epoch operation in the wave epoch family. |
| `waveEpochLamportClock` | `chronos.wave_epoch.lamport-clock` | `standard` | Lamport clock operation in the wave epoch family. |
| `waveEpochLogicalClock` | `chronos.wave_epoch.logical-clock` | `standard` | Logical clock operation in the wave epoch family. |
| `waveEpochModelEpoch` | `chronos.wave_epoch.model-epoch` | `standard` | Model epoch operation in the wave epoch family. |
| `waveEpochSequenceNumber` | `chronos.wave_epoch.sequence-number` | `standard` | Sequence number operation in the wave epoch family. |
| `waveEpochSessionEpoch` | `chronos.wave_epoch.session-epoch` | `standard` | Session epoch operation in the wave epoch family. |
| `waveEpochValidateEpoch` | `chronos.wave_epoch.validate-epoch` | `standard` | Validate epoch operation in the wave epoch family. |
| `waveEpochVectorClock` | `chronos.wave_epoch.vector-clock` | `standard` | Vector clock operation in the wave epoch family. |
| `waveEpochWaitWave` | `chronos.wave_epoch.wait-wave` | `standard` | Wait wave operation in the wave epoch family. |
| `waveEpochWaveBarrier` | `chronos.wave_epoch.wave-barrier` | `standard` | Wave barrier operation in the wave epoch family. |
| `waveEpochWaveComplete` | `chronos.wave_epoch.wave-complete` | `standard` | Wave complete operation in the wave epoch family. |
| `waveEpochWaveCreditAcquire` | `chronos.wave_epoch.wave-credit-acquire` | `standard` | Wave credit acquire operation in the wave epoch family. |
| `waveEpochWaveCreditRelease` | `chronos.wave_epoch.wave-credit-release` | `standard` | Wave credit release operation in the wave epoch family. |
| `waveEpochWaveRollback` | `chronos.wave_epoch.wave-rollback` | `standard` | Wave rollback operation in the wave epoch family. |

# Hermes


## `channel` (16)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `channelBindChannelAddressSpace` | `hermes.channel.bind-channel-address-space` | `standard` | Bind channel address space operation in the channel family. |
| `channelBindChannelEngine` | `hermes.channel.bind-channel-engine` | `standard` | Bind channel engine operation in the channel family. |
| `channelCaptureChannelFault` | `hermes.channel.capture-channel-fault` | `standard` | Capture channel fault operation in the channel family. |
| `channelCreateChannelGroup` | `hermes.channel.create-channel-group` | `standard` | Create channel group operation in the channel family. |
| `channelCreateComputeChannel` | `hermes.channel.create-compute-channel` | `standard` | Create compute channel operation in the channel family. |
| `channelDestroyChannelGroup` | `hermes.channel.destroy-channel-group` | `standard` | Destroy channel group operation in the channel family. |
| `channelDestroyComputeChannel` | `hermes.channel.destroy-compute-channel` | `standard` | Destroy compute channel operation in the channel family. |
| `channelDisableChannel` | `hermes.channel.disable-channel` | `standard` | Disable channel operation in the channel family. |
| `channelEnableChannel` | `hermes.channel.enable-channel` | `standard` | Enable channel operation in the channel family. |
| `channelQueryChannel` | `hermes.channel.query-channel` | `standard` | Query channel operation in the channel family. |
| `channelResetChannel` | `hermes.channel.reset-channel` | `standard` | Reset channel operation in the channel family. |
| `channelResumeChannel` | `hermes.channel.resume-channel` | `standard` | Resume channel operation in the channel family. |
| `channelSetChannelPriority` | `hermes.channel.set-channel-priority` | `standard` | Set channel priority operation in the channel family. |
| `channelSetChannelRunlist` | `hermes.channel.set-channel-runlist` | `standard` | Set channel runlist operation in the channel family. |
| `channelSetChannelTimeslice` | `hermes.channel.set-channel-timeslice` | `standard` | Set channel timeslice operation in the channel family. |
| `channelSuspendChannel` | `hermes.channel.suspend-channel` | `standard` | Suspend channel operation in the channel family. |

## `constant_parameters` (15)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `constantParametersAllocateConstantBank` | `hermes.constant_parameters.allocate-constant-bank` | `standard` | Allocate constant bank operation in the constant parameters family. |
| `constantParametersBindConstantBank` | `hermes.constant_parameters.bind-constant-bank` | `standard` | Bind constant bank operation in the constant parameters family. |
| `constantParametersFreeConstantBank` | `hermes.constant_parameters.free-constant-bank` | `standard` | Free constant bank operation in the constant parameters family. |
| `constantParametersPatchBudget` | `hermes.constant_parameters.patch-budget` | `standard` | Patch budget operation in the constant parameters family. |
| `constantParametersPatchPointer` | `hermes.constant_parameters.patch-pointer` | `standard` | Patch pointer operation in the constant parameters family. |
| `constantParametersPatchScalar` | `hermes.constant_parameters.patch-scalar` | `standard` | Patch scalar operation in the constant parameters family. |
| `constantParametersPatchShape` | `hermes.constant_parameters.patch-shape` | `standard` | Patch shape operation in the constant parameters family. |
| `constantParametersPatchStride` | `hermes.constant_parameters.patch-stride` | `standard` | Patch stride operation in the constant parameters family. |
| `constantParametersValidateParameterAbi` | `hermes.constant_parameters.validate-parameter-abi` | `standard` | Validate parameter abi operation in the constant parameters family. |
| `constantParametersWriteActorDescriptor` | `hermes.constant_parameters.write-actor-descriptor` | `standard` | Write actor descriptor operation in the constant parameters family. |
| `constantParametersWriteConstantParameter` | `hermes.constant_parameters.write-constant-parameter` | `standard` | Write constant parameter operation in the constant parameters family. |
| `constantParametersWriteKernelArguments` | `hermes.constant_parameters.write-kernel-arguments` | `standard` | Write kernel arguments operation in the constant parameters family. |
| `constantParametersWriteMailboxDescriptor` | `hermes.constant_parameters.write-mailbox-descriptor` | `standard` | Write mailbox descriptor operation in the constant parameters family. |
| `constantParametersWriteTensorDescriptor` | `hermes.constant_parameters.write-tensor-descriptor` | `standard` | Write tensor descriptor operation in the constant parameters family. |
| `constantParametersWriteTimelineDescriptor` | `hermes.constant_parameters.write-timeline-descriptor` | `standard` | Write timeline descriptor operation in the constant parameters family. |

## `gpfifo` (16)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `gpfifoCancelGpfifoEntries` | `hermes.gpfifo.cancel-gpfifo-entries` | `standard` | Cancel gpfifo entries operation in the gpfifo family. |
| `gpfifoCommitGpfifoEntries` | `hermes.gpfifo.commit-gpfifo-entries` | `standard` | Commit gpfifo entries operation in the gpfifo family. |
| `gpfifoCreateGpfifo` | `hermes.gpfifo.create-gpfifo` | `standard` | Create gpfifo operation in the gpfifo family. |
| `gpfifoDestroyGpfifo` | `hermes.gpfifo.destroy-gpfifo` | `standard` | Destroy gpfifo operation in the gpfifo family. |
| `gpfifoDetectGpfifoWrap` | `hermes.gpfifo.detect-gpfifo-wrap` | `standard` | Detect gpfifo wrap operation in the gpfifo family. |
| `gpfifoMapGpfifo` | `hermes.gpfifo.map-gpfifo` | `standard` | Map gpfifo operation in the gpfifo family. |
| `gpfifoReadGpfifoGet` | `hermes.gpfifo.read-gpfifo-get` | `standard` | Read gpfifo get operation in the gpfifo family. |
| `gpfifoRecoverGpfifo` | `hermes.gpfifo.recover-gpfifo` | `standard` | Recover gpfifo operation in the gpfifo family. |
| `gpfifoReserveGpfifoEntries` | `hermes.gpfifo.reserve-gpfifo-entries` | `standard` | Reserve gpfifo entries operation in the gpfifo family. |
| `gpfifoRestoreGpfifo` | `hermes.gpfifo.restore-gpfifo` | `standard` | Restore gpfifo operation in the gpfifo family. |
| `gpfifoRingDoorbell` | `hermes.gpfifo.ring-doorbell` | `standard` | Ring doorbell operation in the gpfifo family. |
| `gpfifoSnapshotGpfifo` | `hermes.gpfifo.snapshot-gpfifo` | `standard` | Snapshot gpfifo operation in the gpfifo family. |
| `gpfifoUnmapGpfifo` | `hermes.gpfifo.unmap-gpfifo` | `standard` | Unmap gpfifo operation in the gpfifo family. |
| `gpfifoWaitGpfifoSpace` | `hermes.gpfifo.wait-gpfifo-space` | `standard` | Wait gpfifo space operation in the gpfifo family. |
| `gpfifoWriteGpfifoEntry` | `hermes.gpfifo.write-gpfifo-entry` | `standard` | Write gpfifo entry operation in the gpfifo family. |
| `gpfifoWriteGpfifoPut` | `hermes.gpfifo.write-gpfifo-put` | `standard` | Write gpfifo put operation in the gpfifo family. |

## `launch` (20)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `launchCancelLaunch` | `hermes.launch.cancel-launch` | `standard` | Cancel launch operation in the launch family. |
| `launchCollectLaunchFault` | `hermes.launch.collect-launch-fault` | `standard` | Collect launch fault operation in the launch family. |
| `launchCollectLaunchResult` | `hermes.launch.collect-launch-result` | `standard` | Collect launch result operation in the launch family. |
| `launchFallbackLaunch` | `hermes.launch.fallback-launch` | `standard` | Fallback launch operation in the launch family. |
| `launchLaunchActorGrid` | `hermes.launch.launch-actor-grid` | `standard` | Launch actor grid operation in the launch family. |
| `launchLaunchCooperativeGrid` | `hermes.launch.launch-cooperative-grid` | `standard` | Launch cooperative grid operation in the launch family. |
| `launchLaunchGraph` | `hermes.launch.launch-graph` | `standard` | Launch graph operation in the launch family. |
| `launchLaunchKernel` | `hermes.launch.launch-kernel` | `standard` | Launch kernel operation in the launch family. |
| `launchLaunchKernelAsync` | `hermes.launch.launch-kernel-async` | `standard` | Launch kernel async operation in the launch family. |
| `launchLaunchKernelBatch` | `hermes.launch.launch-kernel-batch` | `standard` | Launch kernel batch operation in the launch family. |
| `launchLaunchKernelIndirect` | `hermes.launch.launch-kernel-indirect` | `standard` | Launch kernel indirect operation in the launch family. |
| `launchLaunchPersistentGrid` | `hermes.launch.launch-persistent-grid` | `standard` | Launch persistent grid operation in the launch family. |
| `launchLaunchResidentProgram` | `hermes.launch.launch-resident-program` | `standard` | Launch resident program operation in the launch family. |
| `launchLaunchWave` | `hermes.launch.launch-wave` | `standard` | Launch wave operation in the launch family. |
| `launchPollLaunch` | `hermes.launch.poll-launch` | `standard` | Poll launch operation in the launch family. |
| `launchRefreshResidentWeights` | `hermes.launch.refresh-resident-weights` | `standard` | Refresh resident weights operation in the launch family. |
| `launchRelaunch` | `hermes.launch.relaunch` | `standard` | Relaunch operation in the launch family. |
| `launchUpdateKernelParameters` | `hermes.launch.update-kernel-parameters` | `standard` | Update kernel parameters operation in the launch family. |
| `launchUpdateResidentState` | `hermes.launch.update-resident-state` | `standard` | Update resident state operation in the launch family. |
| `launchWaitLaunch` | `hermes.launch.wait-launch` | `standard` | Wait launch operation in the launch family. |

## `pushbuffer` (20)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `pushbufferAlignPushbuffer` | `hermes.pushbuffer.align-pushbuffer` | `standard` | Align pushbuffer operation in the pushbuffer family. |
| `pushbufferBeginPush` | `hermes.pushbuffer.begin-push` | `standard` | Begin push operation in the pushbuffer family. |
| `pushbufferCreatePushbuffer` | `hermes.pushbuffer.create-pushbuffer` | `standard` | Create pushbuffer operation in the pushbuffer family. |
| `pushbufferDestroyPushbuffer` | `hermes.pushbuffer.destroy-pushbuffer` | `standard` | Destroy pushbuffer operation in the pushbuffer family. |
| `pushbufferEmitCacheFlush` | `hermes.pushbuffer.emit-cache-flush` | `standard` | Emit cache flush operation in the pushbuffer family. |
| `pushbufferEmitComputeLaunch` | `hermes.pushbuffer.emit-compute-launch` | `standard` | Emit compute launch operation in the pushbuffer family. |
| `pushbufferEmitEngineBind` | `hermes.pushbuffer.emit-engine-bind` | `standard` | Emit engine bind operation in the pushbuffer family. |
| `pushbufferEmitGpuAddress` | `hermes.pushbuffer.emit-gpu-address` | `standard` | Emit gpu address operation in the pushbuffer family. |
| `pushbufferEmitImmediate` | `hermes.pushbuffer.emit-immediate` | `standard` | Emit immediate operation in the pushbuffer family. |
| `pushbufferEmitInlineData` | `hermes.pushbuffer.emit-inline-data` | `standard` | Emit inline data operation in the pushbuffer family. |
| `pushbufferEmitMethod` | `hermes.pushbuffer.emit-method` | `standard` | Emit method operation in the pushbuffer family. |
| `pushbufferEmitMethodIncrementing` | `hermes.pushbuffer.emit-method-incrementing` | `standard` | Emit method incrementing operation in the pushbuffer family. |
| `pushbufferEmitMethodNonIncrementing` | `hermes.pushbuffer.emit-method-non-incrementing` | `standard` | Emit method non incrementing operation in the pushbuffer family. |
| `pushbufferEmitSemaphoreAcquire` | `hermes.pushbuffer.emit-semaphore-acquire` | `standard` | Emit semaphore acquire operation in the pushbuffer family. |
| `pushbufferEmitSemaphoreRelease` | `hermes.pushbuffer.emit-semaphore-release` | `standard` | Emit semaphore release operation in the pushbuffer family. |
| `pushbufferEndPush` | `hermes.pushbuffer.end-push` | `standard` | End push operation in the pushbuffer family. |
| `pushbufferHashPushbuffer` | `hermes.pushbuffer.hash-pushbuffer` | `standard` | Hash pushbuffer operation in the pushbuffer family. |
| `pushbufferPatchPushbuffer` | `hermes.pushbuffer.patch-pushbuffer` | `standard` | Patch pushbuffer operation in the pushbuffer family. |
| `pushbufferReplayPushbuffer` | `hermes.pushbuffer.replay-pushbuffer` | `standard` | Replay pushbuffer operation in the pushbuffer family. |
| `pushbufferValidatePushbuffer` | `hermes.pushbuffer.validate-pushbuffer` | `standard` | Validate pushbuffer operation in the pushbuffer family. |

## `qmd` (21)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `qmdCloneQmd` | `hermes.qmd.clone-qmd` | `standard` | Clone qmd operation in the qmd family. |
| `qmdCreateQmd` | `hermes.qmd.create-qmd` | `standard` | Create qmd operation in the qmd family. |
| `qmdDecodeQmd` | `hermes.qmd.decode-qmd` | `standard` | Decode qmd operation in the qmd family. |
| `qmdDestroyQmd` | `hermes.qmd.destroy-qmd` | `standard` | Destroy qmd operation in the qmd family. |
| `qmdEncodeQmd` | `hermes.qmd.encode-qmd` | `standard` | Encode qmd operation in the qmd family. |
| `qmdPatchQmd` | `hermes.qmd.patch-qmd` | `standard` | Patch qmd operation in the qmd family. |
| `qmdSetQmdAcquire` | `hermes.qmd.set-qmd-acquire` | `standard` | Set qmd acquire operation in the qmd family. |
| `qmdSetQmdBarrierCount` | `hermes.qmd.set-qmd-barrier-count` | `standard` | Set qmd barrier count operation in the qmd family. |
| `qmdSetQmdBlockDimensions` | `hermes.qmd.set-qmd-block-dimensions` | `standard` | Set qmd block dimensions operation in the qmd family. |
| `qmdSetQmdCachePolicy` | `hermes.qmd.set-qmd-cache-policy` | `standard` | Set qmd cache policy operation in the qmd family. |
| `qmdSetQmdConstantBank` | `hermes.qmd.set-qmd-constant-bank` | `standard` | Set qmd constant bank operation in the qmd family. |
| `qmdSetQmdDependency` | `hermes.qmd.set-qmd-dependency` | `standard` | Set qmd dependency operation in the qmd family. |
| `qmdSetQmdGridDimensions` | `hermes.qmd.set-qmd-grid-dimensions` | `standard` | Set qmd grid dimensions operation in the qmd family. |
| `qmdSetQmdLaunchMode` | `hermes.qmd.set-qmd-launch-mode` | `standard` | Set qmd launch mode operation in the qmd family. |
| `qmdSetQmdLocalMemory` | `hermes.qmd.set-qmd-local-memory` | `standard` | Set qmd local memory operation in the qmd family. |
| `qmdSetQmdParameterAddress` | `hermes.qmd.set-qmd-parameter-address` | `standard` | Set qmd parameter address operation in the qmd family. |
| `qmdSetQmdProgramAddress` | `hermes.qmd.set-qmd-program-address` | `standard` | Set qmd program address operation in the qmd family. |
| `qmdSetQmdRegisterCount` | `hermes.qmd.set-qmd-register-count` | `standard` | Set qmd register count operation in the qmd family. |
| `qmdSetQmdRelease` | `hermes.qmd.set-qmd-release` | `standard` | Set qmd release operation in the qmd family. |
| `qmdSetQmdSharedMemory` | `hermes.qmd.set-qmd-shared-memory` | `standard` | Set qmd shared memory operation in the qmd family. |
| `qmdValidateQmd` | `hermes.qmd.validate-qmd` | `standard` | Validate qmd operation in the qmd family. |

## `telemetry_fault` (14)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `telemetryFaultCaptureLaunchTrace` | `hermes.telemetry_fault.capture-launch-trace` | `standard` | Capture launch trace operation in the telemetry fault family. |
| `telemetryFaultCaptureMethodTrace` | `hermes.telemetry_fault.capture-method-trace` | `standard` | Capture method trace operation in the telemetry fault family. |
| `telemetryFaultCapturePushbuffer` | `hermes.telemetry_fault.capture-pushbuffer` | `standard` | Capture pushbuffer operation in the telemetry fault family. |
| `telemetryFaultCaptureQmd` | `hermes.telemetry_fault.capture-qmd` | `standard` | Capture qmd operation in the telemetry fault family. |
| `telemetryFaultCaptureRegisterSnapshot` | `hermes.telemetry_fault.capture-register-snapshot` | `standard` | Capture register snapshot operation in the telemetry fault family. |
| `telemetryFaultClearFault` | `hermes.telemetry_fault.clear-fault` | `standard` | Clear fault operation in the telemetry fault family. |
| `telemetryFaultDecodeFault` | `hermes.telemetry_fault.decode-fault` | `standard` | Decode fault operation in the telemetry fault family. |
| `telemetryFaultDecodeTrap` | `hermes.telemetry_fault.decode-trap` | `standard` | Decode trap operation in the telemetry fault family. |
| `telemetryFaultEmitXidContext` | `hermes.telemetry_fault.emit-xid-context` | `standard` | Emit xid context operation in the telemetry fault family. |
| `telemetryFaultIsolateFaultingChannel` | `hermes.telemetry_fault.isolate-faulting-channel` | `standard` | Isolate faulting channel operation in the telemetry fault family. |
| `telemetryFaultProduceCrashBundle` | `hermes.telemetry_fault.produce-crash-bundle` | `standard` | Produce crash bundle operation in the telemetry fault family. |
| `telemetryFaultQueryChannelStatus` | `hermes.telemetry_fault.query-channel-status` | `standard` | Query channel status operation in the telemetry fault family. |
| `telemetryFaultQueryEngineStatus` | `hermes.telemetry_fault.query-engine-status` | `standard` | Query engine status operation in the telemetry fault family. |
| `telemetryFaultQueryFaultBuffer` | `hermes.telemetry_fault.query-fault-buffer` | `standard` | Query fault buffer operation in the telemetry fault family. |

## `work_transport` (19)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `workTransportAckWorkPacket` | `hermes.work_transport.ack-work-packet` | `research` | Ack work packet operation in the work transport family. |
| `workTransportBatchWorkPackets` | `hermes.work_transport.batch-work-packets` | `research` | Batch work packets operation in the work transport family. |
| `workTransportChecksumWorkPacket` | `hermes.work_transport.checksum-work-packet` | `research` | Checksum work packet operation in the work transport family. |
| `workTransportCoalesceWorkPackets` | `hermes.work_transport.coalesce-work-packets` | `research` | Coalesce work packets operation in the work transport family. |
| `workTransportCreateWorkQueue` | `hermes.work_transport.create-work-queue` | `research` | Create work queue operation in the work transport family. |
| `workTransportDequeueWorkPacket` | `hermes.work_transport.dequeue-work-packet` | `research` | Dequeue work packet operation in the work transport family. |
| `workTransportDestroyWorkQueue` | `hermes.work_transport.destroy-work-queue` | `research` | Destroy work queue operation in the work transport family. |
| `workTransportDonateWorkPacket` | `hermes.work_transport.donate-work-packet` | `research` | Donate work packet operation in the work transport family. |
| `workTransportDropStalePacket` | `hermes.work_transport.drop-stale-packet` | `research` | Drop stale packet operation in the work transport family. |
| `workTransportEnqueueActorEvent` | `hermes.work_transport.enqueue-actor-event` | `research` | Enqueue actor event operation in the work transport family. |
| `workTransportEnqueueCollectivePacket` | `hermes.work_transport.enqueue-collective-packet` | `research` | Enqueue collective packet operation in the work transport family. |
| `workTransportEnqueueMemoryPacket` | `hermes.work_transport.enqueue-memory-packet` | `research` | Enqueue memory packet operation in the work transport family. |
| `workTransportEnqueueTilePacket` | `hermes.work_transport.enqueue-tile-packet` | `research` | Enqueue tile packet operation in the work transport family. |
| `workTransportEnqueueWorkPacket` | `hermes.work_transport.enqueue-work-packet` | `research` | Enqueue work packet operation in the work transport family. |
| `workTransportNackWorkPacket` | `hermes.work_transport.nack-work-packet` | `research` | Nack work packet operation in the work transport family. |
| `workTransportRetryWorkPacket` | `hermes.work_transport.retry-work-packet` | `research` | Retry work packet operation in the work transport family. |
| `workTransportRouteWorkPacket` | `hermes.work_transport.route-work-packet` | `research` | Route work packet operation in the work transport family. |
| `workTransportStealWorkPacket` | `hermes.work_transport.steal-work-packet` | `research` | Steal work packet operation in the work transport family. |
| `workTransportTraceWorkPacket` | `hermes.work_transport.trace-work-packet` | `research` | Trace work packet operation in the work transport family. |

# Gaia


## `allocation` (19)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `allocationAllocateCodeMemory` | `gaia.allocation.allocate-code-memory` | `standard` | Allocate code memory operation in the allocation family. |
| `allocationAllocateCoherentMemory` | `gaia.allocation.allocate-coherent-memory` | `standard` | Allocate coherent memory operation in the allocation family. |
| `allocationAllocateConstantMemory` | `gaia.allocation.allocate-constant-memory` | `standard` | Allocate constant memory operation in the allocation family. |
| `allocationAllocateContiguousMemory` | `gaia.allocation.allocate-contiguous-memory` | `standard` | Allocate contiguous memory operation in the allocation family. |
| `allocationAllocatePinnedHostMemory` | `gaia.allocation.allocate-pinned-host-memory` | `standard` | Allocate pinned host memory operation in the allocation family. |
| `allocationAllocateProtectedMemory` | `gaia.allocation.allocate-protected-memory` | `standard` | Allocate protected memory operation in the allocation family. |
| `allocationAllocateQueueMemory` | `gaia.allocation.allocate-queue-memory` | `standard` | Allocate queue memory operation in the allocation family. |
| `allocationAllocateScratchMemory` | `gaia.allocation.allocate-scratch-memory` | `standard` | Allocate scratch memory operation in the allocation family. |
| `allocationAllocateSemaphoreMemory` | `gaia.allocation.allocate-semaphore-memory` | `standard` | Allocate semaphore memory operation in the allocation family. |
| `allocationAllocateSparseMemory` | `gaia.allocation.allocate-sparse-memory` | `standard` | Allocate sparse memory operation in the allocation family. |
| `allocationAllocateStateMemory` | `gaia.allocation.allocate-state-memory` | `standard` | Allocate state memory operation in the allocation family. |
| `allocationAllocateSystemMemory` | `gaia.allocation.allocate-system-memory` | `standard` | Allocate system memory operation in the allocation family. |
| `allocationAllocateVideoMemory` | `gaia.allocation.allocate-video-memory` | `standard` | Allocate video memory operation in the allocation family. |
| `allocationAllocateWorkspace` | `gaia.allocation.allocate-workspace` | `standard` | Allocate workspace operation in the allocation family. |
| `allocationFreeMemory` | `gaia.allocation.free-memory` | `standard` | Free memory operation in the allocation family. |
| `allocationQueryAllocation` | `gaia.allocation.query-allocation` | `standard` | Query allocation operation in the allocation family. |
| `allocationSetAllocationCompressibility` | `gaia.allocation.set-allocation-compressibility` | `standard` | Set allocation compressibility operation in the allocation family. |
| `allocationSetAllocationPriority` | `gaia.allocation.set-allocation-priority` | `standard` | Set allocation priority operation in the allocation family. |
| `allocationSetAllocationTag` | `gaia.allocation.set-allocation-tag` | `standard` | Set allocation tag operation in the allocation family. |

## `arena_pool` (19)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `arenaPoolArenaCheckpoint` | `gaia.arena_pool.arena-checkpoint` | `standard` | Arena checkpoint operation in the arena pool family. |
| `arenaPoolArenaFragmentationReport` | `gaia.arena_pool.arena-fragmentation-report` | `standard` | Arena fragmentation report operation in the arena pool family. |
| `arenaPoolArenaGuard` | `gaia.arena_pool.arena-guard` | `standard` | Arena guard operation in the arena pool family. |
| `arenaPoolArenaLeakReport` | `gaia.arena_pool.arena-leak-report` | `standard` | Arena leak report operation in the arena pool family. |
| `arenaPoolArenaPoison` | `gaia.arena_pool.arena-poison` | `standard` | Arena poison operation in the arena pool family. |
| `arenaPoolArenaRestore` | `gaia.arena_pool.arena-restore` | `standard` | Arena restore operation in the arena pool family. |
| `arenaPoolArenaRollback` | `gaia.arena_pool.arena-rollback` | `standard` | Arena rollback operation in the arena pool family. |
| `arenaPoolArenaSnapshot` | `gaia.arena_pool.arena-snapshot` | `standard` | Arena snapshot operation in the arena pool family. |
| `arenaPoolArenaValidate` | `gaia.arena_pool.arena-validate` | `standard` | Arena validate operation in the arena pool family. |
| `arenaPoolCreateArena` | `gaia.arena_pool.create-arena` | `standard` | Create arena operation in the arena pool family. |
| `arenaPoolCreatePool` | `gaia.arena_pool.create-pool` | `standard` | Create pool operation in the arena pool family. |
| `arenaPoolDestroyArena` | `gaia.arena_pool.destroy-arena` | `standard` | Destroy arena operation in the arena pool family. |
| `arenaPoolDestroyPool` | `gaia.arena_pool.destroy-pool` | `standard` | Destroy pool operation in the arena pool family. |
| `arenaPoolPoolAcquire` | `gaia.arena_pool.pool-acquire` | `standard` | Pool acquire operation in the arena pool family. |
| `arenaPoolPoolDefragment` | `gaia.arena_pool.pool-defragment` | `standard` | Pool defragment operation in the arena pool family. |
| `arenaPoolPoolRelease` | `gaia.arena_pool.pool-release` | `standard` | Pool release operation in the arena pool family. |
| `arenaPoolPoolTrim` | `gaia.arena_pool.pool-trim` | `standard` | Pool trim operation in the arena pool family. |
| `arenaPoolSuballocate` | `gaia.arena_pool.suballocate` | `standard` | Suballocate operation in the arena pool family. |
| `arenaPoolSubfree` | `gaia.arena_pool.subfree` | `standard` | Subfree operation in the arena pool family. |

## `copy_fill` (19)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `copyFillAsynchronousCopy` | `gaia.copy_fill.asynchronous-copy` | `standard` | Asynchronous copy operation in the copy fill family. |
| `copyFillBatchedCopy` | `gaia.copy_fill.batched-copy` | `standard` | Batched copy operation in the copy fill family. |
| `copyFillChecksumCopy` | `gaia.copy_fill.checksum-copy` | `research` | Checksum copy operation in the copy fill family. |
| `copyFillCopy2d` | `gaia.copy_fill.copy2d` | `standard` | Copy2d operation in the copy fill family. |
| `copyFillCopy3d` | `gaia.copy_fill.copy3d` | `standard` | Copy3d operation in the copy fill family. |
| `copyFillCopyCompressed` | `gaia.copy_fill.copy-compressed` | `standard` | Copy compressed operation in the copy fill family. |
| `copyFillCopyDecompressed` | `gaia.copy_fill.copy-decompressed` | `standard` | Copy decompressed operation in the copy fill family. |
| `copyFillCopyDeviceToDevice` | `gaia.copy_fill.copy-device-to-device` | `standard` | Copy device to device operation in the copy fill family. |
| `copyFillCopyDeviceToHost` | `gaia.copy_fill.copy-device-to-host` | `standard` | Copy device to host operation in the copy fill family. |
| `copyFillCopyGather` | `gaia.copy_fill.copy-gather` | `standard` | Copy gather operation in the copy fill family. |
| `copyFillCopyHostToDevice` | `gaia.copy_fill.copy-host-to-device` | `standard` | Copy host to device operation in the copy fill family. |
| `copyFillCopyPeerToPeer` | `gaia.copy_fill.copy-peer-to-peer` | `standard` | Copy peer to peer operation in the copy fill family. |
| `copyFillCopyScatter` | `gaia.copy_fill.copy-scatter` | `standard` | Copy scatter operation in the copy fill family. |
| `copyFillCopyStrided` | `gaia.copy_fill.copy-strided` | `standard` | Copy strided operation in the copy fill family. |
| `copyFillFillBytes` | `gaia.copy_fill.fill-bytes` | `standard` | Fill bytes operation in the copy fill family. |
| `copyFillFillWords` | `gaia.copy_fill.fill-words` | `standard` | Fill words operation in the copy fill family. |
| `copyFillPatternFill` | `gaia.copy_fill.pattern-fill` | `standard` | Pattern fill operation in the copy fill family. |
| `copyFillVerifiedCopy` | `gaia.copy_fill.verified-copy` | `standard` | Verified copy operation in the copy fill family. |
| `copyFillZeroMemory` | `gaia.copy_fill.zero-memory` | `standard` | Zero memory operation in the copy fill family. |

## `host_mapping` (14)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `hostMappingCreateStagingMap` | `gaia.host_mapping.create-staging-map` | `standard` | Create staging map operation in the host mapping family. |
| `hostMappingDestroyStagingMap` | `gaia.host_mapping.destroy-staging-map` | `standard` | Destroy staging map operation in the host mapping family. |
| `hostMappingFaultInPages` | `gaia.host_mapping.fault-in-pages` | `standard` | Fault in pages operation in the host mapping family. |
| `hostMappingFlushHostWrites` | `gaia.host_mapping.flush-host-writes` | `standard` | Flush host writes operation in the host mapping family. |
| `hostMappingInvalidateHostReads` | `gaia.host_mapping.invalidate-host-reads` | `standard` | Invalidate host reads operation in the host mapping family. |
| `hostMappingMapBar1` | `gaia.host_mapping.map-bar1` | `standard` | Map bar1 operation in the host mapping family. |
| `hostMappingMapHost` | `gaia.host_mapping.map-host` | `standard` | Map host operation in the host mapping family. |
| `hostMappingPinUserPages` | `gaia.host_mapping.pin-user-pages` | `standard` | Pin user pages operation in the host mapping family. |
| `hostMappingPrefaultPages` | `gaia.host_mapping.prefault-pages` | `standard` | Prefault pages operation in the host mapping family. |
| `hostMappingQueryMapping` | `gaia.host_mapping.query-mapping` | `standard` | Query mapping operation in the host mapping family. |
| `hostMappingSynchronizeMapping` | `gaia.host_mapping.synchronize-mapping` | `standard` | Synchronize mapping operation in the host mapping family. |
| `hostMappingUnmapBar1` | `gaia.host_mapping.unmap-bar1` | `standard` | Unmap bar1 operation in the host mapping family. |
| `hostMappingUnmapHost` | `gaia.host_mapping.unmap-host` | `standard` | Unmap host operation in the host mapping family. |
| `hostMappingUnpinUserPages` | `gaia.host_mapping.unpin-user-pages` | `standard` | Unpin user pages operation in the host mapping family. |

## `integrity` (16)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `integrityCaptureMemoryManifest` | `gaia.integrity.capture-memory-manifest` | `standard` | Capture memory manifest operation in the integrity family. |
| `integrityChecksumAllocation` | `gaia.integrity.checksum-allocation` | `research` | Checksum allocation operation in the integrity family. |
| `integrityCompareMemoryManifest` | `gaia.integrity.compare-memory-manifest` | `standard` | Compare memory manifest operation in the integrity family. |
| `integrityDetectDoubleFree` | `gaia.integrity.detect-double-free` | `standard` | Detect double free operation in the integrity family. |
| `integrityDetectLeak` | `gaia.integrity.detect-leak` | `standard` | Detect leak operation in the integrity family. |
| `integrityDetectOutOfBounds` | `gaia.integrity.detect-out-of-bounds` | `standard` | Detect out of bounds operation in the integrity family. |
| `integrityDetectOverlap` | `gaia.integrity.detect-overlap` | `standard` | Detect overlap operation in the integrity family. |
| `integrityDetectStaleMapping` | `gaia.integrity.detect-stale-mapping` | `standard` | Detect stale mapping operation in the integrity family. |
| `integrityDetectUseAfterFree` | `gaia.integrity.detect-use-after-free` | `standard` | Detect use after free operation in the integrity family. |
| `integrityGuardAllocation` | `gaia.integrity.guard-allocation` | `standard` | Guard allocation operation in the integrity family. |
| `integrityPoisonAllocation` | `gaia.integrity.poison-allocation` | `standard` | Poison allocation operation in the integrity family. |
| `integrityScrubSensitiveMemory` | `gaia.integrity.scrub-sensitive-memory` | `standard` | Scrub sensitive memory operation in the integrity family. |
| `integritySecureErase` | `gaia.integrity.secure-erase` | `standard` | Secure erase operation in the integrity family. |
| `integrityUnguardAllocation` | `gaia.integrity.unguard-allocation` | `standard` | Unguard allocation operation in the integrity family. |
| `integrityUnpoisonAllocation` | `gaia.integrity.unpoison-allocation` | `standard` | Unpoison allocation operation in the integrity family. |
| `integrityVerifyChecksum` | `gaia.integrity.verify-checksum` | `research` | Verify checksum operation in the integrity family. |

## `memory_policy` (16)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `memoryPolicyChooseCompression` | `gaia.memory_policy.choose-compression` | `standard` | Choose compression operation in the memory policy family. |
| `memoryPolicyChooseEvictionVictim` | `gaia.memory_policy.choose-eviction-victim` | `standard` | Choose eviction victim operation in the memory policy family. |
| `memoryPolicyChoosePageSize` | `gaia.memory_policy.choose-page-size` | `standard` | Choose page size operation in the memory policy family. |
| `memoryPolicyChoosePlacement` | `gaia.memory_policy.choose-placement` | `standard` | Choose placement operation in the memory policy family. |
| `memoryPolicyChoosePrefetchDistance` | `gaia.memory_policy.choose-prefetch-distance` | `standard` | Choose prefetch distance operation in the memory policy family. |
| `memoryPolicyChooseReplication` | `gaia.memory_policy.choose-replication` | `standard` | Choose replication operation in the memory policy family. |
| `memoryPolicyChooseStagingPath` | `gaia.memory_policy.choose-staging-path` | `standard` | Choose staging path operation in the memory policy family. |
| `memoryPolicyEnforceMemoryBudget` | `gaia.memory_policy.enforce-memory-budget` | `standard` | Enforce memory budget operation in the memory policy family. |
| `memoryPolicyEnforceNoAlias` | `gaia.memory_policy.enforce-no-alias` | `standard` | Enforce no alias operation in the memory policy family. |
| `memoryPolicyEnforceReadOnly` | `gaia.memory_policy.enforce-read-only` | `standard` | Enforce read only operation in the memory policy family. |
| `memoryPolicyEnforceSafetyReserve` | `gaia.memory_policy.enforce-safety-reserve` | `standard` | Enforce safety reserve operation in the memory policy family. |
| `memoryPolicyEnforceSessionQuota` | `gaia.memory_policy.enforce-session-quota` | `standard` | Enforce session quota operation in the memory policy family. |
| `memoryPolicyEstimateFragmentation` | `gaia.memory_policy.estimate-fragmentation` | `standard` | Estimate fragmentation operation in the memory policy family. |
| `memoryPolicyEstimateReuseDistance` | `gaia.memory_policy.estimate-reuse-distance` | `standard` | Estimate reuse distance operation in the memory policy family. |
| `memoryPolicyEstimateTransferCost` | `gaia.memory_policy.estimate-transfer-cost` | `standard` | Estimate transfer cost operation in the memory policy family. |
| `memoryPolicyEstimateWorkingSet` | `gaia.memory_policy.estimate-working-set` | `standard` | Estimate working set operation in the memory policy family. |

## `residency` (22)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `residencyEvictResident` | `gaia.residency.evict-resident` | `standard` | Evict resident operation in the residency family. |
| `residencyHotsetDemote` | `gaia.residency.hotset-demote` | `standard` | Hotset demote operation in the residency family. |
| `residencyHotsetPromote` | `gaia.residency.hotset-promote` | `standard` | Hotset promote operation in the residency family. |
| `residencyInstallCodeImage` | `gaia.residency.install-code-image` | `standard` | Install code image operation in the residency family. |
| `residencyInstallDistrictImage` | `gaia.residency.install-district-image` | `standard` | Install district image operation in the residency family. |
| `residencyInstallModelImage` | `gaia.residency.install-model-image` | `standard` | Install model image operation in the residency family. |
| `residencyInstallWeightImage` | `gaia.residency.install-weight-image` | `standard` | Install weight image operation in the residency family. |
| `residencyMakeNonResident` | `gaia.residency.make-non-resident` | `standard` | Make non resident operation in the residency family. |
| `residencyMakeResident` | `gaia.residency.make-resident` | `standard` | Make resident operation in the residency family. |
| `residencyMigrateToGpu` | `gaia.residency.migrate-to-gpu` | `standard` | Migrate to gpu operation in the residency family. |
| `residencyMigrateToHost` | `gaia.residency.migrate-to-host` | `standard` | Migrate to host operation in the residency family. |
| `residencyMigrateToPeer` | `gaia.residency.migrate-to-peer` | `standard` | Migrate to peer operation in the residency family. |
| `residencyPinResident` | `gaia.residency.pin-resident` | `standard` | Pin resident operation in the residency family. |
| `residencyPrefetchResident` | `gaia.residency.prefetch-resident` | `standard` | Prefetch resident operation in the residency family. |
| `residencyQueryResidency` | `gaia.residency.query-residency` | `standard` | Query residency operation in the residency family. |
| `residencyRefreshWeightImage` | `gaia.residency.refresh-weight-image` | `standard` | Refresh weight image operation in the residency family. |
| `residencySetEvictionPolicy` | `gaia.residency.set-eviction-policy` | `standard` | Set eviction policy operation in the residency family. |
| `residencySetResidencyPriority` | `gaia.residency.set-residency-priority` | `standard` | Set residency priority operation in the residency family. |
| `residencyTouchResident` | `gaia.residency.touch-resident` | `standard` | Touch resident operation in the residency family. |
| `residencyUninstallDistrictImage` | `gaia.residency.uninstall-district-image` | `standard` | Uninstall district image operation in the residency family. |
| `residencyUninstallModelImage` | `gaia.residency.uninstall-model-image` | `standard` | Uninstall model image operation in the residency family. |
| `residencyUnpinResident` | `gaia.residency.unpin-resident` | `standard` | Unpin resident operation in the residency family. |

## `session_state` (19)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `sessionStateAppendEpisodicRecord` | `gaia.session_state.append-episodic-record` | `standard` | Append episodic record operation in the session state family. |
| `sessionStateCloneSessionState` | `gaia.session_state.clone-session-state` | `standard` | Clone session state operation in the session state family. |
| `sessionStateCompactEpisodicRing` | `gaia.session_state.compact-episodic-ring` | `standard` | Compact episodic ring operation in the session state family. |
| `sessionStateCreateBindingTable` | `gaia.session_state.create-binding-table` | `standard` | Create binding table operation in the session state family. |
| `sessionStateCreateEpisodicRing` | `gaia.session_state.create-episodic-ring` | `standard` | Create episodic ring operation in the session state family. |
| `sessionStateCreateSessionState` | `gaia.session_state.create-session-state` | `standard` | Create session state operation in the session state family. |
| `sessionStateDeleteBinding` | `gaia.session_state.delete-binding` | `standard` | Delete binding operation in the session state family. |
| `sessionStateDeserializeSessionState` | `gaia.session_state.deserialize-session-state` | `standard` | Deserialize session state operation in the session state family. |
| `sessionStateDestroySessionState` | `gaia.session_state.destroy-session-state` | `standard` | Destroy session state operation in the session state family. |
| `sessionStateEvictEpisodicRecord` | `gaia.session_state.evict-episodic-record` | `standard` | Evict episodic record operation in the session state family. |
| `sessionStateInspectBindingTable` | `gaia.session_state.inspect-binding-table` | `standard` | Inspect binding table operation in the session state family. |
| `sessionStateMapSessionState` | `gaia.session_state.map-session-state` | `standard` | Map session state operation in the session state family. |
| `sessionStateMigrateSessionState` | `gaia.session_state.migrate-session-state` | `standard` | Migrate session state operation in the session state family. |
| `sessionStateRestoreSessionState` | `gaia.session_state.restore-session-state` | `standard` | Restore session state operation in the session state family. |
| `sessionStateSerializeSessionState` | `gaia.session_state.serialize-session-state` | `standard` | Serialize session state operation in the session state family. |
| `sessionStateSnapshotSessionState` | `gaia.session_state.snapshot-session-state` | `standard` | Snapshot session state operation in the session state family. |
| `sessionStateUnmapSessionState` | `gaia.session_state.unmap-session-state` | `standard` | Unmap session state operation in the session state family. |
| `sessionStateUpdateBindingTable` | `gaia.session_state.update-binding-table` | `standard` | Update binding table operation in the session state family. |
| `sessionStateValidateSessionIsolation` | `gaia.session_state.validate-session-isolation` | `standard` | Validate session isolation operation in the session state family. |

## `virtual_address` (18)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `virtualAddressCreateAddressSpace` | `gaia.virtual_address.create-address-space` | `standard` | Create address space operation in the virtual address family. |
| `virtualAddressCreatePeerAlias` | `gaia.virtual_address.create-peer-alias` | `standard` | Create peer alias operation in the virtual address family. |
| `virtualAddressCreateReadOnlyAlias` | `gaia.virtual_address.create-read-only-alias` | `standard` | Create read only alias operation in the virtual address family. |
| `virtualAddressCreateVaAlias` | `gaia.virtual_address.create-va-alias` | `standard` | Create va alias operation in the virtual address family. |
| `virtualAddressDestroyAddressSpace` | `gaia.virtual_address.destroy-address-space` | `standard` | Destroy address space operation in the virtual address family. |
| `virtualAddressDumpPageTable` | `gaia.virtual_address.dump-page-table` | `standard` | Dump page table operation in the virtual address family. |
| `virtualAddressFindFreeGpuVa` | `gaia.virtual_address.find-free-gpu-va` | `standard` | Find free gpu va operation in the virtual address family. |
| `virtualAddressFreeGpuVa` | `gaia.virtual_address.free-gpu-va` | `standard` | Free gpu va operation in the virtual address family. |
| `virtualAddressMapGpuVa` | `gaia.virtual_address.map-gpu-va` | `standard` | Map gpu va operation in the virtual address family. |
| `virtualAddressProtectGpuVa` | `gaia.virtual_address.protect-gpu-va` | `standard` | Protect gpu va operation in the virtual address family. |
| `virtualAddressQueryGpuVa` | `gaia.virtual_address.query-gpu-va` | `standard` | Query gpu va operation in the virtual address family. |
| `virtualAddressRemapGpuVa` | `gaia.virtual_address.remap-gpu-va` | `standard` | Remap gpu va operation in the virtual address family. |
| `virtualAddressReserveGpuVa` | `gaia.virtual_address.reserve-gpu-va` | `standard` | Reserve gpu va operation in the virtual address family. |
| `virtualAddressSetCompressionTag` | `gaia.virtual_address.set-compression-tag` | `standard` | Set compression tag operation in the virtual address family. |
| `virtualAddressSetMappingKind` | `gaia.virtual_address.set-mapping-kind` | `standard` | Set mapping kind operation in the virtual address family. |
| `virtualAddressSetPageSize` | `gaia.virtual_address.set-page-size` | `standard` | Set page size operation in the virtual address family. |
| `virtualAddressUnmapGpuVa` | `gaia.virtual_address.unmap-gpu-va` | `standard` | Unmap gpu va operation in the virtual address family. |
| `virtualAddressValidateGpuVa` | `gaia.virtual_address.validate-gpu-va` | `standard` | Validate gpu va operation in the virtual address family. |

# Aether


## `capability_compat` (16)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `capabilityCompatEmitCapabilityManifest` | `aether.capability_compat.emit-capability-manifest` | `standard` | Emit capability manifest operation in the capability compat family. |
| `capabilityCompatFailOnUnknownCompatibility` | `aether.capability_compat.fail-on-unknown-compatibility` | `standard` | Fail on unknown compatibility operation in the capability compat family. |
| `capabilityCompatLoadCompatibilityProfile` | `aether.capability_compat.load-compatibility-profile` | `standard` | Load compatibility profile operation in the capability compat family. |
| `capabilityCompatProbeChannelClass` | `aether.capability_compat.probe-channel-class` | `standard` | Probe channel class operation in the capability compat family. |
| `capabilityCompatProbeComputeClass` | `aether.capability_compat.probe-compute-class` | `standard` | Probe compute class operation in the capability compat family. |
| `capabilityCompatProbeDriverVersion` | `aether.capability_compat.probe-driver-version` | `standard` | Probe driver version operation in the capability compat family. |
| `capabilityCompatProbeGpuPciId` | `aether.capability_compat.probe-gpu-pci-id` | `standard` | Probe gpu pci id operation in the capability compat family. |
| `capabilityCompatProbeGspFirmwareVersion` | `aether.capability_compat.probe-gsp-firmware-version` | `standard` | Probe gsp firmware version operation in the capability compat family. |
| `capabilityCompatProbeKernelModuleVersion` | `aether.capability_compat.probe-kernel-module-version` | `standard` | Probe kernel module version operation in the capability compat family. |
| `capabilityCompatProbeMemoryClass` | `aether.capability_compat.probe-memory-class` | `standard` | Probe memory class operation in the capability compat family. |
| `capabilityCompatProbeSecurityMode` | `aether.capability_compat.probe-security-mode` | `standard` | Probe security mode operation in the capability compat family. |
| `capabilityCompatProbeSupportedAtomics` | `aether.capability_compat.probe-supported-atomics` | `standard` | Probe supported atomics operation in the capability compat family. |
| `capabilityCompatProbeSupportedFaultModes` | `aether.capability_compat.probe-supported-fault-modes` | `standard` | Probe supported fault modes operation in the capability compat family. |
| `capabilityCompatProbeSupportedMma` | `aether.capability_compat.probe-supported-mma` | `standard` | Probe supported mma operation in the capability compat family. |
| `capabilityCompatProbeSupportedPageSizes` | `aether.capability_compat.probe-supported-page-sizes` | `standard` | Probe supported page sizes operation in the capability compat family. |
| `capabilityCompatValidateCompatibilityProfile` | `aether.capability_compat.validate-compatibility-profile` | `standard` | Validate compatibility profile operation in the capability compat family. |

## `device_node` (14)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `deviceNodeCloseDeviceNode` | `aether.device_node.close-device-node` | `standard` | Close device node operation in the device node family. |
| `deviceNodeDuplicateDeviceFd` | `aether.device_node.duplicate-device-fd` | `standard` | Duplicate device fd operation in the device node family. |
| `deviceNodeMapDeviceRegion` | `aether.device_node.map-device-region` | `standard` | Map device region operation in the device node family. |
| `deviceNodeOpenControlDevice` | `aether.device_node.open-control-device` | `standard` | Open control device operation in the device node family. |
| `deviceNodeOpenGpuDevice` | `aether.device_node.open-gpu-device` | `standard` | Open gpu device operation in the device node family. |
| `deviceNodeOpenUvmDevice` | `aether.device_node.open-uvm-device` | `standard` | Open uvm device operation in the device node family. |
| `deviceNodePollDeviceNode` | `aether.device_node.poll-device-node` | `standard` | Poll device node operation in the device node family. |
| `deviceNodeQueryDeviceNode` | `aether.device_node.query-device-node` | `standard` | Query device node operation in the device node family. |
| `deviceNodeReadDeviceEvent` | `aether.device_node.read-device-event` | `standard` | Read device event operation in the device node family. |
| `deviceNodeSetCloseOnExec` | `aether.device_node.set-close-on-exec` | `standard` | Set close on exec operation in the device node family. |
| `deviceNodeSetNonBlocking` | `aether.device_node.set-non-blocking` | `standard` | Set non blocking operation in the device node family. |
| `deviceNodeUnmapDeviceRegion` | `aether.device_node.unmap-device-region` | `standard` | Unmap device region operation in the device node family. |
| `deviceNodeValidateDeviceMajorMinor` | `aether.device_node.validate-device-major-minor` | `standard` | Validate device major minor operation in the device node family. |
| `deviceNodeWriteDeviceCommand` | `aether.device_node.write-device-command` | `standard` | Write device command operation in the device node family. |

## `fault_diagnostics` (17)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `faultDiagnosticsAssembleCrashBundle` | `aether.fault_diagnostics.assemble-crash-bundle` | `standard` | Assemble crash bundle operation in the fault diagnostics family. |
| `faultDiagnosticsCaptureBarState` | `aether.fault_diagnostics.capture-bar-state` | `standard` | Capture bar state operation in the fault diagnostics family. |
| `faultDiagnosticsCaptureChannelState` | `aether.fault_diagnostics.capture-channel-state` | `standard` | Capture channel state operation in the fault diagnostics family. |
| `faultDiagnosticsCaptureFaultBuffer` | `aether.fault_diagnostics.capture-fault-buffer` | `standard` | Capture fault buffer operation in the fault diagnostics family. |
| `faultDiagnosticsCaptureGspCrashDump` | `aether.fault_diagnostics.capture-gsp-crash-dump` | `standard` | Capture gsp crash dump operation in the fault diagnostics family. |
| `faultDiagnosticsCaptureKernelLogContext` | `aether.fault_diagnostics.capture-kernel-log-context` | `standard` | Capture kernel log context operation in the fault diagnostics family. |
| `faultDiagnosticsCaptureObjectGraph` | `aether.fault_diagnostics.capture-object-graph` | `standard` | Capture object graph operation in the fault diagnostics family. |
| `faultDiagnosticsCapturePciState` | `aether.fault_diagnostics.capture-pci-state` | `standard` | Capture pci state operation in the fault diagnostics family. |
| `faultDiagnosticsClassifyFault` | `aether.fault_diagnostics.classify-fault` | `standard` | Classify fault operation in the fault diagnostics family. |
| `faultDiagnosticsDecodeRmEvent` | `aether.fault_diagnostics.decode-rm-event` | `standard` | Decode rm event operation in the fault diagnostics family. |
| `faultDiagnosticsDecodeXid` | `aether.fault_diagnostics.decode-xid` | `standard` | Decode xid operation in the fault diagnostics family. |
| `faultDiagnosticsFailClosed` | `aether.fault_diagnostics.fail-closed` | `standard` | Fail closed operation in the fault diagnostics family. |
| `faultDiagnosticsReadRmEvent` | `aether.fault_diagnostics.read-rm-event` | `standard` | Read rm event operation in the fault diagnostics family. |
| `faultDiagnosticsReadXid` | `aether.fault_diagnostics.read-xid` | `standard` | Read xid operation in the fault diagnostics family. |
| `faultDiagnosticsRecommendRecovery` | `aether.fault_diagnostics.recommend-recovery` | `standard` | Recommend recovery operation in the fault diagnostics family. |
| `faultDiagnosticsSubscribeRmEvent` | `aether.fault_diagnostics.subscribe-rm-event` | `standard` | Subscribe rm event operation in the fault diagnostics family. |
| `faultDiagnosticsUnsubscribeRmEvent` | `aether.fault_diagnostics.unsubscribe-rm-event` | `standard` | Unsubscribe rm event operation in the fault diagnostics family. |

## `gsp_rpc` (13)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `gspRpcCaptureGspLog` | `aether.gsp_rpc.capture-gsp-log` | `standard` | Capture gsp log operation in the gsp rpc family. |
| `gspRpcDecodeGspRpc` | `aether.gsp_rpc.decode-gsp-rpc` | `standard` | Decode gsp rpc operation in the gsp rpc family. |
| `gspRpcDetectGspTimeout` | `aether.gsp_rpc.detect-gsp-timeout` | `standard` | Detect gsp timeout operation in the gsp rpc family. |
| `gspRpcEncodeGspRpc` | `aether.gsp_rpc.encode-gsp-rpc` | `standard` | Encode gsp rpc operation in the gsp rpc family. |
| `gspRpcLoadGspFirmwareReference` | `aether.gsp_rpc.load-gsp-firmware-reference` | `standard` | Load gsp firmware reference operation in the gsp rpc family. |
| `gspRpcQueryGspStatus` | `aether.gsp_rpc.query-gsp-status` | `standard` | Query gsp status operation in the gsp rpc family. |
| `gspRpcQueryGspVersion` | `aether.gsp_rpc.query-gsp-version` | `standard` | Query gsp version operation in the gsp rpc family. |
| `gspRpcReceiveGspRpc` | `aether.gsp_rpc.receive-gsp-rpc` | `standard` | Receive gsp rpc operation in the gsp rpc family. |
| `gspRpcResetGsp` | `aether.gsp_rpc.reset-gsp` | `standard` | Reset gsp operation in the gsp rpc family. |
| `gspRpcSendGspRpc` | `aether.gsp_rpc.send-gsp-rpc` | `standard` | Send gsp rpc operation in the gsp rpc family. |
| `gspRpcTraceGspRpc` | `aether.gsp_rpc.trace-gsp-rpc` | `standard` | Trace gsp rpc operation in the gsp rpc family. |
| `gspRpcValidateGspCompatibility` | `aether.gsp_rpc.validate-gsp-compatibility` | `standard` | Validate gsp compatibility operation in the gsp rpc family. |
| `gspRpcWaitGspReady` | `aether.gsp_rpc.wait-gsp-ready` | `standard` | Wait gsp ready operation in the gsp rpc family. |

## `ioctl_transport` (15)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `ioctlTransportClassifyRmStatus` | `aether.ioctl_transport.classify-rm-status` | `standard` | Classify rm status operation in the ioctl transport family. |
| `ioctlTransportDecodeIoctl` | `aether.ioctl_transport.decode-ioctl` | `standard` | Decode ioctl operation in the ioctl transport family. |
| `ioctlTransportEncodeIoctl` | `aether.ioctl_transport.encode-ioctl` | `standard` | Encode ioctl operation in the ioctl transport family. |
| `ioctlTransportFuzzIoctl` | `aether.ioctl_transport.fuzz-ioctl` | `standard` | Fuzz ioctl operation in the ioctl transport family. |
| `ioctlTransportHandleIoctlError` | `aether.ioctl_transport.handle-ioctl-error` | `standard` | Handle ioctl error operation in the ioctl transport family. |
| `ioctlTransportInvokeIoctl` | `aether.ioctl_transport.invoke-ioctl` | `standard` | Invoke ioctl operation in the ioctl transport family. |
| `ioctlTransportInvokeIoctlAsync` | `aether.ioctl_transport.invoke-ioctl-async` | `standard` | Invoke ioctl async operation in the ioctl transport family. |
| `ioctlTransportInvokeIoctlRetry` | `aether.ioctl_transport.invoke-ioctl-retry` | `standard` | Invoke ioctl retry operation in the ioctl transport family. |
| `ioctlTransportQueryIoctlCapabilities` | `aether.ioctl_transport.query-ioctl-capabilities` | `standard` | Query ioctl capabilities operation in the ioctl transport family. |
| `ioctlTransportQueryIoctlVersion` | `aether.ioctl_transport.query-ioctl-version` | `standard` | Query ioctl version operation in the ioctl transport family. |
| `ioctlTransportRedactIoctl` | `aether.ioctl_transport.redact-ioctl` | `standard` | Redact ioctl operation in the ioctl transport family. |
| `ioctlTransportReplayIoctl` | `aether.ioctl_transport.replay-ioctl` | `standard` | Replay ioctl operation in the ioctl transport family. |
| `ioctlTransportTraceIoctl` | `aether.ioctl_transport.trace-ioctl` | `standard` | Trace ioctl operation in the ioctl transport family. |
| `ioctlTransportTranslateIoctlVersion` | `aether.ioctl_transport.translate-ioctl-version` | `standard` | Translate ioctl version operation in the ioctl transport family. |
| `ioctlTransportValidateIoctl` | `aether.ioctl_transport.validate-ioctl` | `standard` | Validate ioctl operation in the ioctl transport family. |

## `rm_channel` (17)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `rmChannelAllocateChannel` | `aether.rm_channel.allocate-channel` | `standard` | Allocate channel operation in the rm channel family. |
| `rmChannelAllocateChannelGroup` | `aether.rm_channel.allocate-channel-group` | `standard` | Allocate channel group operation in the rm channel family. |
| `rmChannelAllocateContextShare` | `aether.rm_channel.allocate-context-share` | `standard` | Allocate context share operation in the rm channel family. |
| `rmChannelAllocateEvent` | `aether.rm_channel.allocate-event` | `standard` | Allocate event operation in the rm channel family. |
| `rmChannelAllocateSemaphoreSurface` | `aether.rm_channel.allocate-semaphore-surface` | `standard` | Allocate semaphore surface operation in the rm channel family. |
| `rmChannelBindChannel` | `aether.rm_channel.bind-channel` | `standard` | Bind channel operation in the rm channel family. |
| `rmChannelFreeChannel` | `aether.rm_channel.free-channel` | `standard` | Free channel operation in the rm channel family. |
| `rmChannelFreeChannelGroup` | `aether.rm_channel.free-channel-group` | `standard` | Free channel group operation in the rm channel family. |
| `rmChannelFreeContextShare` | `aether.rm_channel.free-context-share` | `standard` | Free context share operation in the rm channel family. |
| `rmChannelFreeEvent` | `aether.rm_channel.free-event` | `standard` | Free event operation in the rm channel family. |
| `rmChannelFreeSemaphoreSurface` | `aether.rm_channel.free-semaphore-surface` | `standard` | Free semaphore surface operation in the rm channel family. |
| `rmChannelQueryChannelState` | `aether.rm_channel.query-channel-state` | `standard` | Query channel state operation in the rm channel family. |
| `rmChannelRegisterEvent` | `aether.rm_channel.register-event` | `standard` | Register event operation in the rm channel family. |
| `rmChannelResetChannelObject` | `aether.rm_channel.reset-channel-object` | `standard` | Reset channel object operation in the rm channel family. |
| `rmChannelSetChannelPriority` | `aether.rm_channel.set-channel-priority` | `standard` | Set channel priority operation in the rm channel family. |
| `rmChannelUnbindChannel` | `aether.rm_channel.unbind-channel` | `standard` | Unbind channel operation in the rm channel family. |
| `rmChannelUnregisterEvent` | `aether.rm_channel.unregister-event` | `standard` | Unregister event operation in the rm channel family. |

## `rm_client` (15)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `rmClientAllocRmObject` | `aether.rm_client.alloc-rm-object` | `standard` | Alloc rm object operation in the rm client family. |
| `rmClientCaptureRmObjectGraph` | `aether.rm_client.capture-rm-object-graph` | `standard` | Capture rm object graph operation in the rm client family. |
| `rmClientControlRmObject` | `aether.rm_client.control-rm-object` | `standard` | Control rm object operation in the rm client family. |
| `rmClientCreateRmClient` | `aether.rm_client.create-rm-client` | `standard` | Create rm client operation in the rm client family. |
| `rmClientCreateRootObject` | `aether.rm_client.create-root-object` | `standard` | Create root object operation in the rm client family. |
| `rmClientDestroyRmClient` | `aether.rm_client.destroy-rm-client` | `standard` | Destroy rm client operation in the rm client family. |
| `rmClientDuplicateRmClient` | `aether.rm_client.duplicate-rm-client` | `standard` | Duplicate rm client operation in the rm client family. |
| `rmClientEnumerateRmChildren` | `aether.rm_client.enumerate-rm-children` | `standard` | Enumerate rm children operation in the rm client family. |
| `rmClientFreeRmObject` | `aether.rm_client.free-rm-object` | `standard` | Free rm object operation in the rm client family. |
| `rmClientFreeRootObject` | `aether.rm_client.free-root-object` | `standard` | Free root object operation in the rm client family. |
| `rmClientQueryRmClient` | `aether.rm_client.query-rm-client` | `standard` | Query rm client operation in the rm client family. |
| `rmClientQueryRmObject` | `aether.rm_client.query-rm-object` | `standard` | Query rm object operation in the rm client family. |
| `rmClientSetRmClientPolicy` | `aether.rm_client.set-rm-client-policy` | `standard` | Set rm client policy operation in the rm client family. |
| `rmClientTranslateRmHandle` | `aether.rm_client.translate-rm-handle` | `standard` | Translate rm handle operation in the rm client family. |
| `rmClientValidateRmHandle` | `aether.rm_client.validate-rm-handle` | `standard` | Validate rm handle operation in the rm client family. |

## `rm_device` (18)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `rmDeviceAllocateDeviceObject` | `aether.rm_device.allocate-device-object` | `standard` | Allocate device object operation in the rm device family. |
| `rmDeviceAllocateSubdevice` | `aether.rm_device.allocate-subdevice` | `standard` | Allocate subdevice operation in the rm device family. |
| `rmDeviceFreeDeviceObject` | `aether.rm_device.free-device-object` | `standard` | Free device object operation in the rm device family. |
| `rmDeviceFreeSubdevice` | `aether.rm_device.free-subdevice` | `standard` | Free subdevice operation in the rm device family. |
| `rmDeviceQueryArchitectureInfo` | `aether.rm_device.query-architecture-info` | `standard` | Query architecture info operation in the rm device family. |
| `rmDeviceQueryBarInfo` | `aether.rm_device.query-bar-info` | `standard` | Query bar info operation in the rm device family. |
| `rmDeviceQueryClockInfo` | `aether.rm_device.query-clock-info` | `standard` | Query clock info operation in the rm device family. |
| `rmDeviceQueryDeviceInfo` | `aether.rm_device.query-device-info` | `standard` | Query device info operation in the rm device family. |
| `rmDeviceQueryEccInfo` | `aether.rm_device.query-ecc-info` | `standard` | Query ecc info operation in the rm device family. |
| `rmDeviceQueryEngineInfo` | `aether.rm_device.query-engine-info` | `standard` | Query engine info operation in the rm device family. |
| `rmDeviceQueryGspInfo` | `aether.rm_device.query-gsp-info` | `standard` | Query gsp info operation in the rm device family. |
| `rmDeviceQueryMemoryInfo` | `aether.rm_device.query-memory-info` | `standard` | Query memory info operation in the rm device family. |
| `rmDeviceQueryPciInfo` | `aether.rm_device.query-pci-info` | `standard` | Query pci info operation in the rm device family. |
| `rmDeviceQueryPowerInfo` | `aether.rm_device.query-power-info` | `standard` | Query power info operation in the rm device family. |
| `rmDeviceQueryResetStatus` | `aether.rm_device.query-reset-status` | `standard` | Query reset status operation in the rm device family. |
| `rmDeviceQuerySubdeviceInfo` | `aether.rm_device.query-subdevice-info` | `standard` | Query subdevice info operation in the rm device family. |
| `rmDeviceQueryThermalInfo` | `aether.rm_device.query-thermal-info` | `standard` | Query thermal info operation in the rm device family. |
| `rmDeviceQueryVbiosInfo` | `aether.rm_device.query-vbios-info` | `standard` | Query vbios info operation in the rm device family. |

## `rm_vaspace_memory` (15)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `rmVaspaceMemoryAllocateMemoryObject` | `aether.rm_vaspace_memory.allocate-memory-object` | `standard` | Allocate memory object operation in the rm vaspace memory family. |
| `rmVaspaceMemoryAllocateVaSpace` | `aether.rm_vaspace_memory.allocate-va-space` | `standard` | Allocate va space operation in the rm vaspace memory family. |
| `rmVaspaceMemoryBindMemoryToVaSpace` | `aether.rm_vaspace_memory.bind-memory-to-va-space` | `standard` | Bind memory to va space operation in the rm vaspace memory family. |
| `rmVaspaceMemoryDuplicateMemoryObject` | `aether.rm_vaspace_memory.duplicate-memory-object` | `standard` | Duplicate memory object operation in the rm vaspace memory family. |
| `rmVaspaceMemoryExportMemoryObject` | `aether.rm_vaspace_memory.export-memory-object` | `standard` | Export memory object operation in the rm vaspace memory family. |
| `rmVaspaceMemoryFreeMemoryObject` | `aether.rm_vaspace_memory.free-memory-object` | `standard` | Free memory object operation in the rm vaspace memory family. |
| `rmVaspaceMemoryFreeVaSpace` | `aether.rm_vaspace_memory.free-va-space` | `standard` | Free va space operation in the rm vaspace memory family. |
| `rmVaspaceMemoryImportMemoryObject` | `aether.rm_vaspace_memory.import-memory-object` | `standard` | Import memory object operation in the rm vaspace memory family. |
| `rmVaspaceMemoryMapMemoryObject` | `aether.rm_vaspace_memory.map-memory-object` | `standard` | Map memory object operation in the rm vaspace memory family. |
| `rmVaspaceMemoryQueryMemoryObject` | `aether.rm_vaspace_memory.query-memory-object` | `standard` | Query memory object operation in the rm vaspace memory family. |
| `rmVaspaceMemoryRegisterMemoryObject` | `aether.rm_vaspace_memory.register-memory-object` | `standard` | Register memory object operation in the rm vaspace memory family. |
| `rmVaspaceMemorySetMemoryObjectFlags` | `aether.rm_vaspace_memory.set-memory-object-flags` | `standard` | Set memory object flags operation in the rm vaspace memory family. |
| `rmVaspaceMemoryUnbindMemoryFromVaSpace` | `aether.rm_vaspace_memory.unbind-memory-from-va-space` | `standard` | Unbind memory from va space operation in the rm vaspace memory family. |
| `rmVaspaceMemoryUnmapMemoryObject` | `aether.rm_vaspace_memory.unmap-memory-object` | `standard` | Unmap memory object operation in the rm vaspace memory family. |
| `rmVaspaceMemoryUnregisterMemoryObject` | `aether.rm_vaspace_memory.unregister-memory-object` | `standard` | Unregister memory object operation in the rm vaspace memory family. |

## `uvm` (19)

| Export | Operation ID | Status | Summary |
|---|---|---|---|
| `uvmUvmDeinitialize` | `aether.uvm.uvm-deinitialize` | `standard` | Uvm deinitialize operation in the uvm family. |
| `uvmUvmFlushFaultBuffer` | `aether.uvm.uvm-flush-fault-buffer` | `standard` | Uvm flush fault buffer operation in the uvm family. |
| `uvmUvmInitialize` | `aether.uvm.uvm-initialize` | `standard` | Uvm initialize operation in the uvm family. |
| `uvmUvmMapExternalAllocation` | `aether.uvm.uvm-map-external-allocation` | `standard` | Uvm map external allocation operation in the uvm family. |
| `uvmUvmMigrate` | `aether.uvm.uvm-migrate` | `standard` | Uvm migrate operation in the uvm family. |
| `uvmUvmPrefetch` | `aether.uvm.uvm-prefetch` | `standard` | Uvm prefetch operation in the uvm family. |
| `uvmUvmQueryFaults` | `aether.uvm.uvm-query-faults` | `standard` | Uvm query faults operation in the uvm family. |
| `uvmUvmQueryPageResidency` | `aether.uvm.uvm-query-page-residency` | `standard` | Uvm query page residency operation in the uvm family. |
| `uvmUvmRegisterChannel` | `aether.uvm.uvm-register-channel` | `standard` | Uvm register channel operation in the uvm family. |
| `uvmUvmRegisterGpu` | `aether.uvm.uvm-register-gpu` | `standard` | Uvm register gpu operation in the uvm family. |
| `uvmUvmRegisterVaSpace` | `aether.uvm.uvm-register-va-space` | `standard` | Uvm register va space operation in the uvm family. |
| `uvmUvmServiceFault` | `aether.uvm.uvm-service-fault` | `standard` | Uvm service fault operation in the uvm family. |
| `uvmUvmSetAccessedBy` | `aether.uvm.uvm-set-accessed-by` | `standard` | Uvm set accessed by operation in the uvm family. |
| `uvmUvmSetPreferredLocation` | `aether.uvm.uvm-set-preferred-location` | `standard` | Uvm set preferred location operation in the uvm family. |
| `uvmUvmSetReadMostly` | `aether.uvm.uvm-set-read-mostly` | `standard` | Uvm set read mostly operation in the uvm family. |
| `uvmUvmUnmapExternalAllocation` | `aether.uvm.uvm-unmap-external-allocation` | `standard` | Uvm unmap external allocation operation in the uvm family. |
| `uvmUvmUnregisterChannel` | `aether.uvm.uvm-unregister-channel` | `standard` | Uvm unregister channel operation in the uvm family. |
| `uvmUvmUnregisterGpu` | `aether.uvm.uvm-unregister-gpu` | `standard` | Uvm unregister gpu operation in the uvm family. |
| `uvmUvmUnregisterVaSpace` | `aether.uvm.uvm-unregister-va-space` | `standard` | Uvm unregister va space operation in the uvm family. |
