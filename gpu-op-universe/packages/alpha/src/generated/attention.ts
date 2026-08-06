/* AUTO-GENERATED. Do not hand-edit; edit operation-registry.json. */
import { defineStub } from "../../../common/src/types";
import type { AttentionRequest } from "../../../common/src/types";

/**
 * alpha.attention.attention-backward
 * Attention backward operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionAttentionBackward = defineStub<AttentionRequest>("alpha.attention.attention-backward");

/**
 * alpha.attention.attention-checksum
 * Attention checksum operation in the attention family.
 * Status: research; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionAttentionChecksum = defineStub<AttentionRequest>("alpha.attention.attention-checksum");

/**
 * alpha.attention.attention-kv-append
 * Attention kv append operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionAttentionKvAppend = defineStub<AttentionRequest>("alpha.attention.attention-kv-append");

/**
 * alpha.attention.attention-kv-compact
 * Attention kv compact operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionAttentionKvCompact = defineStub<AttentionRequest>("alpha.attention.attention-kv-compact");

/**
 * alpha.attention.attention-kv-evict
 * Attention kv evict operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionAttentionKvEvict = defineStub<AttentionRequest>("alpha.attention.attention-kv-evict");

/**
 * alpha.attention.attention-kv-gather
 * Attention kv gather operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionAttentionKvGather = defineStub<AttentionRequest>("alpha.attention.attention-kv-gather");

/**
 * alpha.attention.attention-kv-quantize
 * Attention kv quantize operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionAttentionKvQuantize = defineStub<AttentionRequest>("alpha.attention.attention-kv-quantize");

/**
 * alpha.attention.attention-prefix-share
 * Attention prefix share operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionAttentionPrefixShare = defineStub<AttentionRequest>("alpha.attention.attention-prefix-share");

/**
 * alpha.attention.attention-score-only
 * Attention score only operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionAttentionScoreOnly = defineStub<AttentionRequest>("alpha.attention.attention-score-only");

/**
 * alpha.attention.attention-tree-share
 * Attention tree share operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionAttentionTreeShare = defineStub<AttentionRequest>("alpha.attention.attention-tree-share");

/**
 * alpha.attention.attention-value-only
 * Attention value only operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionAttentionValueOnly = defineStub<AttentionRequest>("alpha.attention.attention-value-only");

/**
 * alpha.attention.attention-with-retrieval
 * Attention with retrieval operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionAttentionWithRetrieval = defineStub<AttentionRequest>("alpha.attention.attention-with-retrieval");

/**
 * alpha.attention.bidirectional-attention
 * Bidirectional attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionBidirectionalAttention = defineStub<AttentionRequest>("alpha.attention.bidirectional-attention");

/**
 * alpha.attention.block-sparse-attention
 * Block sparse attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionBlockSparseAttention = defineStub<AttentionRequest>("alpha.attention.block-sparse-attention");

/**
 * alpha.attention.causal-attention
 * Causal attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionCausalAttention = defineStub<AttentionRequest>("alpha.attention.causal-attention");

/**
 * alpha.attention.chunked-attention
 * Chunked attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionChunkedAttention = defineStub<AttentionRequest>("alpha.attention.chunked-attention");

/**
 * alpha.attention.cosine-attention
 * Cosine attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionCosineAttention = defineStub<AttentionRequest>("alpha.attention.cosine-attention");

/**
 * alpha.attention.cross-attention
 * Cross attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionCrossAttention = defineStub<AttentionRequest>("alpha.attention.cross-attention");

/**
 * alpha.attention.delta-net-attention
 * Delta net attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionDeltaNetAttention = defineStub<AttentionRequest>("alpha.attention.delta-net-attention");

/**
 * alpha.attention.dilated-attention
 * Dilated attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionDilatedAttention = defineStub<AttentionRequest>("alpha.attention.dilated-attention");

/**
 * alpha.attention.entmax-attention
 * Entmax attention operation in the attention family.
 * Status: standard; target: architecture-agnostic, future-or-emulated; differentiability: differentiable-or-estimator.
 */
export const attentionEntmaxAttention = defineStub<AttentionRequest>("alpha.attention.entmax-attention");

/**
 * alpha.attention.flash-attention
 * Flash attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionFlashAttention = defineStub<AttentionRequest>("alpha.attention.flash-attention");

/**
 * alpha.attention.flash-attention-backward
 * Flash attention backward operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionFlashAttentionBackward = defineStub<AttentionRequest>("alpha.attention.flash-attention-backward");

/**
 * alpha.attention.flash-decoding
 * Flash decoding operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionFlashDecoding = defineStub<AttentionRequest>("alpha.attention.flash-decoding");

/**
 * alpha.attention.gated-delta-net
 * Gated delta net operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionGatedDeltaNet = defineStub<AttentionRequest>("alpha.attention.gated-delta-net");

/**
 * alpha.attention.gated-retention
 * Gated retention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionGatedRetention = defineStub<AttentionRequest>("alpha.attention.gated-retention");

/**
 * alpha.attention.global-local-attention
 * Global local attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionGlobalLocalAttention = defineStub<AttentionRequest>("alpha.attention.global-local-attention");

/**
 * alpha.attention.grouped-query-attention
 * Grouped query attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionGroupedQueryAttention = defineStub<AttentionRequest>("alpha.attention.grouped-query-attention");

/**
 * alpha.attention.hard-attention
 * Hard attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionHardAttention = defineStub<AttentionRequest>("alpha.attention.hard-attention");

/**
 * alpha.attention.hash-attention
 * Hash attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionHashAttention = defineStub<AttentionRequest>("alpha.attention.hash-attention");

/**
 * alpha.attention.kernel-linear-attention
 * Kernel linear attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionKernelLinearAttention = defineStub<AttentionRequest>("alpha.attention.kernel-linear-attention");

/**
 * alpha.attention.local-attention
 * Local attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionLocalAttention = defineStub<AttentionRequest>("alpha.attention.local-attention");

/**
 * alpha.attention.memory-compressed-attention
 * Memory compressed attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionMemoryCompressedAttention = defineStub<AttentionRequest>("alpha.attention.memory-compressed-attention");

/**
 * alpha.attention.monotonic-attention
 * Monotonic attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionMonotonicAttention = defineStub<AttentionRequest>("alpha.attention.monotonic-attention");

/**
 * alpha.attention.multi-head-attention
 * Multi head attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionMultiHeadAttention = defineStub<AttentionRequest>("alpha.attention.multi-head-attention");

/**
 * alpha.attention.multi-head-latent-attention
 * Multi head latent attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionMultiHeadLatentAttention = defineStub<AttentionRequest>("alpha.attention.multi-head-latent-attention");

/**
 * alpha.attention.multi-query-attention
 * Multi query attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionMultiQueryAttention = defineStub<AttentionRequest>("alpha.attention.multi-query-attention");

/**
 * alpha.attention.online-softmax-attention
 * Online softmax attention operation in the attention family.
 * Status: standard; target: architecture-agnostic, future-or-emulated; differentiability: differentiable-or-estimator.
 */
export const attentionOnlineSoftmaxAttention = defineStub<AttentionRequest>("alpha.attention.online-softmax-attention");

/**
 * alpha.attention.paged-attention
 * Paged attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionPagedAttention = defineStub<AttentionRequest>("alpha.attention.paged-attention");

/**
 * alpha.attention.performer-attention
 * Performer attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionPerformerAttention = defineStub<AttentionRequest>("alpha.attention.performer-attention");

/**
 * alpha.attention.prefix-attention
 * Prefix attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionPrefixAttention = defineStub<AttentionRequest>("alpha.attention.prefix-attention");

/**
 * alpha.attention.recurrent-attention
 * Recurrent attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionRecurrentAttention = defineStub<AttentionRequest>("alpha.attention.recurrent-attention");

/**
 * alpha.attention.retention
 * Retention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionRetention = defineStub<AttentionRequest>("alpha.attention.retention");

/**
 * alpha.attention.ring-attention
 * Ring attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionRingAttention = defineStub<AttentionRequest>("alpha.attention.ring-attention");

/**
 * alpha.attention.routing-attention
 * Routing attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionRoutingAttention = defineStub<AttentionRequest>("alpha.attention.routing-attention");

/**
 * alpha.attention.scaled-dot-product-attention
 * Scaled dot product attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionScaledDotProductAttention = defineStub<AttentionRequest>("alpha.attention.scaled-dot-product-attention");

/**
 * alpha.attention.sinkhorn-attention
 * Sinkhorn attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionSinkhornAttention = defineStub<AttentionRequest>("alpha.attention.sinkhorn-attention");

/**
 * alpha.attention.sliding-window-attention
 * Sliding window attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionSlidingWindowAttention = defineStub<AttentionRequest>("alpha.attention.sliding-window-attention");

/**
 * alpha.attention.sparsemax-attention
 * Sparsemax attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionSparsemaxAttention = defineStub<AttentionRequest>("alpha.attention.sparsemax-attention");

/**
 * alpha.attention.streaming-attention
 * Streaming attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionStreamingAttention = defineStub<AttentionRequest>("alpha.attention.streaming-attention");

/**
 * alpha.attention.strided-attention
 * Strided attention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionStridedAttention = defineStub<AttentionRequest>("alpha.attention.strided-attention");

/**
 * alpha.attention.top-kattention
 * Top kattention operation in the attention family.
 * Status: standard; target: architecture-agnostic; differentiability: differentiable-or-estimator.
 */
export const attentionTopKAttention = defineStub<AttentionRequest>("alpha.attention.top-kattention");
