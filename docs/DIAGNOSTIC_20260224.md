# Alpha Engine: Comprehensive Diagnostic Report

**Date**: 2026-02-24
**System**: Custom GPT training engine (TypeScript + Vulkan/SPIR-V GPU backend)
**Current run**: 8L/384D/8H batch=20 block=512, ~17.6M params, L4 24GB GPU

---

## 1. Architecture Overview

Alpha is a fully hand-written transformer training system. Zero external ML dependencies. The full stack:

```
TypeScript application layer
    |
@alpha/autograd  -- tape-based automatic differentiation
    |
@alpha/helios    -- Vulkan GPU backend
    |
helios_vk.node   -- C++ native addon (Vulkan API, buffer management)
    |
SPIR-V shaders   -- generated from TypeScript (no shader compiler)
    |
Vulkan driver    -- NVIDIA L4 GPU (24GB VRAM)
```

### Key Design Decisions
- **Lazy evaluation compute graph**: Ops are recorded, not executed immediately. Batched into single Vulkan command buffer submits (up to MAX_PENDING_OPS=64 ops per batch)
- **Timeline semaphores**: GPU work is tracked with monotonic timeline values. Buffers are only reused when their write operation completes
- **Two-tier buffer pool**: `bufferPool` for device-local allocations, `outputPool` for timeline-aware output regions
- **FinalizationRegistry + explicit release**: GPU buffers freed deterministically via `releaseFn` callbacks, with FR as safety net
- **Auto-tuned workgroup size**: Benchmarks WG sizes [64, 128, 256, 512] at init to pick optimal for the device

---

## 2. Current Training Run Status

```
Step: ~72/15000
Loss: 8.11 (down from 8.37 at step 1)
Throughput: ~895 tok/s
Step time: ~11.4s/iter
GPU ops/step: 1442 (1508 when clipping fires)
Output pool: 2.92 GB across 26 size classes, 134 regions
Model: 8 layers, 384 dim, 8 heads, 4096 vocab, 512 block size
Data: historic-chat-v2.txt (33MB, 12,992 conversations)
```

### Gradient Norm Observations
Most steps have grad_norm 0.7-1.8, but periodic spikes occur:
- Step 56: 13.1 (clip=0.38)
- Step 61: 146.9 (clip=0.034)
- Step 66: 37.5 (clip=0.13)
- Step 67: **1306.0** (clip=0.004)
- Step 68: 45.3 (clip=0.11)
- Step 72: **509.0** (clip=0.01)

These spikes suggest certain training batches contain pathological sequences (possibly very short or repetitive conversations). The gradClip=5.0 is catching them, but when clip_coef drops to 0.004, the effective learning rate is reduced by 250x for that step -- Adam momentum gets a near-zero update, which is wasteful but not destructive.

---

## 3. OOM Error Analysis

### Tested Configurations on L4 24GB

| Config | Batch | Status | Pool MB | Failure Point |
|--------|-------|--------|---------|---------------|
| 12L/512D/8H | 48 | OOM | -- | softmax forward (attention weight tensor 384MB) |
| 12L/512D/8H | 32 | OOM | 3,450 | softCap forward step 2 |
| 12L/512D/8H | 16 | OOM | 2,500 | mul in backward step 7 |
| 8L/384D/8H | 24 | OOM | 3,450 | clone in backward step 3 |
| 8L/384D/8H | 20 | STABLE | 2,920 | -- |
| 8L/384D/8H | 16 | STABLE | 2,350 | (underutilizing) |

### Why Backward Doubles Memory

The forward pass creates activations at each tape entry. During backward, each entry computes gradients which are *new* tensors. At any point during backward, you have:
1. Forward activations not yet processed (shrinking as backward progresses)
2. Gradient tensors being accumulated on parameters (growing)
3. Intermediate tensors within backward closures (transient)

Our tape clears activations as it walks backward (`releaseTensor(entry.output.data)`), but there's a peak at mid-backward where both halves overlap. For the 12L/512D model, the attention weight tensors are `[B, 8, 512, 512] * 4 bytes = B * 8MB` each, and with 12 layers that's 12 forward + 12 backward copies simultaneously alive at the peak.

### Memory Budget Breakdown (8L/384D/8H batch=20)

Per-step allocation (from gpu_mem logs):
- `160.0MB x 8` = 1280 MB -- matmul outputs (B*T x nEmbd or B*T x 4*nEmbd)
- `156.3MB x 6` = 938 MB -- similar matmul intermediates
- `60.0MB x 8` = 480 MB -- attention weights [20, 8, 512, 512]
- `15.0MB x 8` = 120 MB -- smaller projections
- `5.9MB x 7` = 41 MB -- layer norm / bias tensors
- `4.0MB x 8` = 32 MB -- misc
- `2.3MB x 8` = 18 MB -- misc
- `1.0MB x 8` = 8 MB -- scalars / small tensors

**Total output pool: ~2,925 MB of 24,576 MB VRAM (12% utilization)**

### The Real VRAM Bottleneck

We're only using 12% of VRAM. The remaining 88% is consumed by:
1. **Model parameters**: 17.6M params * 4 bytes = 70 MB
2. **Adam state**: 2 * 70 MB = 140 MB (m and v buffers)
3. **Vulkan driver overhead**: ~500-800 MB (descriptor sets, command buffers, pipeline caches)
4. **The actual VRAM used by intermediate tensors during computation**: During a step, new allocations happen before old ones are freed. The compute graph batches 64 ops, and each op's output region is alive until the next GC cycle.

The gap between 2.9GB pool and 24GB total suggests significant VRAM is unreachable -- either Vulkan driver reservations, or buffer fragmentation (many small allocations that can't be coalesced).

---

## 4. Dispatch Count Analysis

### Forward Pass (per step, 8L model)

Per transformer layer:
- 2x layerNorm (forward): 2 dispatches each = 4
- 6x matmul (Q,K,V,O projections + MLP fc1,fc2): 6 dispatches
- 6x transpose (for matmul): 6 dispatches (these are zero-copy reshapes? or GPU ops?)
- 6x reshape: metadata only, no GPU dispatch
- 1x scale (1/sqrt(d)): 1 dispatch
- 1x softCap: 1 dispatch (native kernel)
- 1x maskedFill: 1 dispatch
- 1x softmax: 1 multi-pass dispatch (max, sub, exp, sum, div)
- 2x dropout: 2 dispatches each
- 1x matmul (attn @ V): 1 dispatch
- 2x add (residuals): 2 dispatches
- 1x GELU: 1 dispatch

Per layer total: ~28-32 dispatches
8 layers: ~224-256 dispatches
+ embeddings, final LN, lmHead: ~20 dispatches
+ cross entropy: ~5 dispatches

**Forward total: ~250-280 dispatches**

### Backward Pass (per step)

Each forward op records a backward closure. Backward closures often create *additional* intermediate tensors:
- matmul backward: 2 transposes + 2 matmuls = 4 dispatches per matmul = 24/layer
- softmax backward: mul + sum + sub + mul = 4 dispatches
- layerNorm backward: ~8 dispatches
- add backward: 2 pass-throughs (free)
- GELU backward: mul dispatch
- softCap backward: 1 dispatch (native kernel)

Per layer backward: ~40-48 dispatches
8 layers: ~320-384 dispatches
+ embeddings, head, CE backward: ~30 dispatches

**Backward total: ~350-420 dispatches**

### Optimizer + Grad Norm

- Grad norm: 2 ops/param (mul, sum) * ~90 params = 180 dispatches
- Grad clip: 1 scale/param * 90 = 90 dispatches (when clipping)
- AdamW: 1 fused kernel/param * 90 = 90 dispatches

**Total GPU ops per step: ~870-1060 dispatches**

Measured: 1442 ops/step (no clip) to 1508 ops/step (with clip). The excess over our estimate comes from:
- broadcast/reduce operations we didn't count
- softmax being multi-pass (5-7 internal dispatches)
- layerNorm being multi-pass
- gradient accumulation overhead

### Dispatch Overhead

With MAX_PENDING_OPS=64, a step with 1442 ops triggers **22-23 flush cycles**. Each flush is one `batchBegin` + N `batchDispatch` + `batchSubmit` call to Vulkan. The per-flush overhead is ~100us (command buffer allocation + submit + signal). Total overhead: ~2.3ms.

This is small compared to the 11.4s step time, so **dispatch batching is not the bottleneck**.

---

## 5. Speed Optimization Opportunities

### Tier 1: High Impact (estimated 20-40% step time reduction)

#### 5a. Fused Flash Attention Kernel

**Current**: Attention is 6+ separate dispatches per layer: matmul(Q,K^T), scale, softCap, maskedFill, softmax, dropout, matmul(attn,V). Each writes full intermediate tensors to VRAM.

**Proposed**: Single fused kernel that:
1. Loads Q,K,V tiles from VRAM once
2. Computes QK^T, scales, caps, masks, softmax, and @V in shared memory
3. Writes only the final output to VRAM

**Impact**: Eliminates 5 VRAM round-trips per layer * 8 layers = 40 VRAM writes. The attention weight tensor `[B,H,T,T]` is 60MB at our config -- never needs to exist in VRAM. This is the single biggest optimization available.

**Complexity**: Very high. Requires writing SPIR-V that does tiled matrix operations with shared memory (`OpVariable` with Workgroup storage class), barrier synchronization, and careful numerical handling of softmax (online softmax / FlashAttention algorithm).

#### 5b. Matmul Tile Size Optimization

**Current**: Matmul kernel uses 16x16 tiles hardcoded. On NVIDIA L4 (Ada Lovelace), this is suboptimal -- the GPU has 58 SMs with 128 FP32 CUDA cores each.

**Proposed**: Auto-tune tile size (16, 32, or 64) based on matrix dimensions. For our common sizes (B*T x nEmbd = 10240 x 384), 32x32 tiles would halve the number of workgroup dispatches and improve shared memory utilization.

**Impact**: 15-25% improvement in matmul throughput. Since matmul dominates (12 per layer forward + backward), this compounds to 10-15% total step time.

#### 5c. Fused LayerNorm + Bias Kernel

**Current**: LayerNorm is a multi-pass operation (mean, variance, normalize, scale, shift). The backward is similarly multi-pass.

**Proposed**: Single-pass fused kernel using Welford's online algorithm for mean+variance, then normalize+scale+shift in one pass. Backward similarly fused.

**Impact**: Reduces per-layer dispatches by 3-4, saves VRAM bandwidth for intermediate mean/variance tensors. With 8 layers * 2 LN each = 16 layerNorms, this saves ~48-64 dispatches and associated VRAM writes.

### Tier 2: Medium Impact (estimated 10-20% improvement)

#### 5d. Fused Residual + Dropout + Add

**Current**: Each residual connection is: `dropout(projected, p, training)` then `add(x, dropResult)`. Two separate ops.

**Proposed**: Fused `residualDropoutAdd(x, projected, dropout_mask)` that does everything in one kernel.

**Impact**: Saves 2 dispatches per residual * 2 residuals per layer * 8 layers = 32 dispatches. More importantly, eliminates 32 intermediate tensors from VRAM.

#### 5e. Transpose Optimization

**Current**: Transpose allocates new output buffer and copies with reindexing.

**Proposed**: For matmul(A, transpose(B)), implement a `matmulTransposed` kernel that reads B in transposed order directly. This eliminates all explicit transpose dispatches for weight projections.

**Impact**: 6 transposes per layer * 8 layers = 48 dispatches eliminated. Each transpose is a full VRAM copy of the weight matrix.

#### 5f. Increase MAX_PENDING_OPS

**Current**: MAX_PENDING_OPS=64 causes 22+ flushes per step. While per-flush overhead is low (~100us), GPU pipeline utilization suffers from command buffer boundaries -- the GPU may briefly idle between batch submits.

**Proposed**: Increase to 256 or 512. The memory cost is negligible (just the PendingOp array). This lets more ops batch together, improving GPU occupancy.

**Impact**: 2-5% improvement from better GPU pipeline utilization. Nearly free to implement.

### Tier 3: Lower Impact but Good Hygiene

#### 5g. crossEntropyBackward Forces Flush

**Current** in `backend.ts`: The `crossEntropyBackward` method calls `graph.flush()` mid-operation to read logits back to CPU for the softmax computation. This breaks the compute graph batch and forces a GPU sync.

```typescript
crossEntropyBackward(logits, targets, gradOutput) {
  // ... forces flush via logits.data access (lazy tensor triggers readback)
  const probs = new Float32Array(C * N);
  // CPU softmax computation...
}
```

**Proposed**: Implement a GPU-native cross-entropy backward kernel that computes softmax + gradient on GPU without readback.

**Impact**: Eliminates 1 forced flush per step. Saves ~100-200us per step plus the CPU softmax compute time.

#### 5h. Buffer Pool Fragmentation

**Current**: `POOL_MAX_PER_SIZE=8` limits each size class to 8 buffers. With 26 size classes, the pool holds 134 buffers (2.9GB). But new size classes keep being created for different tensor shapes.

**Proposed**: Round buffer sizes up to power-of-2 boundaries to reduce size class proliferation. A 156.3MB tensor and a 160.0MB tensor could share the same 160MB pool.

**Impact**: Better buffer reuse, fewer Vulkan allocations over time. Reduces allocation churn from ~1000 allocs/step to potentially ~200.

---

## 6. Autograd Efficiency Analysis

### Tape Size Per Step

For an 8-layer model with B=20, T=512:
- Forward: ~120-150 tape entries (each op = one entry)
- Each entry stores: output Variable, input Variables[], backward closure

The backward closures capture references to forward tensors (e.g., `aData`, `bData` for matmul). These keep forward activations alive until backward processes them.

### Memory-Critical Backward Closures

**matmul backward** is the most memory-intensive:
```typescript
// backward: dL/dA = G @ B^T, dL/dB = A^T @ G
const tB = B.transpose(bData, ndimB - 2, ndimB - 1);  // creates new tensor
const tA = B.transpose(aData, ndimA - 2, ndimA - 1);  // creates new tensor
const ga = B.matmul(g, tB);   // creates new tensor
const gb = B.matmul(tA, g);   // creates new tensor
if (release) { release(tB); release(tA); }
```

Each matmul backward creates 4 new tensors (2 transposes + 2 matmuls). With 12 matmuls per layer * 8 layers = 96 matmul backwards, this creates **384 intermediate tensors** in backward alone. The `release` callback frees transposes immediately, but matmul outputs persist as gradients.

### Gradient Accumulation Pattern

When a Variable has multiple consumers (e.g., residual connections use `x` twice), the tape accumulates:
```typescript
input.grad = backend.add(input.grad, g);  // creates new tensor
releaseTensor(oldGrad);                     // frees old accumulation
releaseTensor(g);                           // frees the new gradient
```

This is correct but creates 2 temporary tensors per accumulation event. For an 8-layer model, the residual connection pattern means the `x` variable in each layer gets 2 gradient contributions.

### Potential Autograd Improvements

1. **In-place gradient accumulation**: Instead of `add(grad, g)` creating a new tensor, use an `addInPlace(grad, g)` kernel that modifies `grad` directly. Saves one allocation per accumulation.

2. **Matmul backward fusion**: Instead of separate `transpose + matmul`, implement `matmulBackward(g, A, B)` as a single fused kernel that reads A/B in the correct order without explicit transpose.

3. **Dead gradient elimination**: Some tape entries produce gradients for inputs that don't `requiresGrad`. The backward still computes those gradients before the tape discards them. Could skip computation entirely with a pre-pass.

---

## 7. Data Pipeline Analysis

### Current Data Loading

```typescript
class DataLoader {
  nextBatch() {
    // Creates new Int32Array(B * (T+1)) per call
    // Random offset into token array
    // Returns { inputs: TensorData, targets: TensorData }
  }
}
```

Each `nextBatch()` allocates a new `Int32Array` and two `TensorData` wrappers. At B=20, T=512, this is `20 * 513 * 4 = 40KB` per step. Not a bottleneck.

### Tokenization

The 33MB dataset is tokenized once at startup into a single `Int32Array` in memory. With BPE-4k, the token array is ~8M tokens. Random access into this array is O(1). No pipeline stall here.

---

## 8. Numerical Stability Assessment

### Current Safeguards
1. **softCap**: `tanh(x/30)*30` bounds attention scores to [-30, 30]. Prevents softmax from receiving extreme values.
2. **Gradient clipping**: gradClip=5.0 with full telemetry (clip_coef, clip_pct logged)
3. **NaN detection**: Forward loss NaN skips entire step. Grad norm NaN skips optimizer.
4. **Spike threshold**: Configurable absolute grad_norm cap (currently off, using only gradClip)
5. **AdamW moments in f32**: Even on GPU, moment buffers are Float32Array
6. **Grad norm computed in f32**: mul + sum stays in f32 precision

### Observed Issues

The gradient spikes (1306 at step 67, 509 at step 72) during early warmup are concerning. At lr=8e-6 (still in warmup), these shouldn't cause instability because:
1. Clipping reduces effective lr to 8e-6 * 0.004 = 3.2e-8 for those steps
2. Adam's momentum (beta1=0.9) smooths out single-step spikes
3. Loss is still decreasing monotonically (8.37 -> 8.11)

But the spike *frequency* matters. If >10% of steps are clipped, Adam's running mean is constantly being corrected, which slows convergence. Current clip_pct would be informative to watch.

### Recommendations
- **Lower beta2 from 0.95 to 0.9** if spikes persist past warmup -- faster second-moment adaptation
- **Consider spikeThreshold=50**: Skip the optimizer entirely for extreme spikes (>50x normal). This is better than clipping to 0.004x because it preserves Adam state
- **Monitor post-warmup behavior**: Spikes during warmup are somewhat normal. If they persist after step 1000 (warmup end), that signals a deeper issue

---

## 9. SPIR-V Kernel Inventory

### Current Kernels

| Kernel | Bindings | Push Constants | Vec4 | Purpose |
|--------|----------|---------------|------|---------|
| add | 3 (A,B,out) | len, 0 | yes | Element-wise add |
| sub | 3 | len, 0 | yes | Element-wise sub |
| mul | 3 | len, 0 | yes | Element-wise mul |
| div | 3 | len, 0 | yes | Element-wise div |
| scale | 2 (A,out) | len, scalar | yes | Multiply by scalar |
| neg | 2 | len, 0 | yes | Negate |
| exp | 2 | len, 0 | yes | Exponential |
| log | 2 | len, 0 | yes | Natural log |
| sqrt | 2 | len, 0 | yes | Square root |
| relu | 2 | len, 0 | yes | ReLU activation |
| gelu | 2 | len, 0 | yes | GELU activation |
| gelu_backward | 3 (input,grad,out) | len, 0 | yes | GELU gradient |
| relu_backward | 3 | len, 0 | yes | ReLU gradient |
| clamp | 2 | len, lo (hi via push) | no | Clamp to range |
| clamp_min | 2 | len, min | yes | Clamp minimum |
| clamp_backward | 3 | len, lo (hi via push) | no | Clamp gradient |
| softcap_forward | 2 | len, cap | no | tanh(x/cap)*cap |
| softcap_backward | 3 | len, cap | no | softCap gradient |
| embedding | 3 (weight,indices,out) | vocabSize, dim | no | Lookup table |
| embedding_backward | 3 | vocabSize, dim | no | Scatter gradients |
| softmax | multiple passes | varies | no | Multi-pass softmax |
| layernorm | multiple passes | varies | no | Layer normalization |
| layernorm_backward | multiple passes | varies | no | LN gradient |
| matmul | 3+ | M,N,K,... | no | Tiled matrix multiply |
| matmul_batched | 3+ | M,N,K,B | no | Batched matmul |
| cross_entropy | varies | N,C | no | CE loss forward |
| adamw | 5 (p,g,m,v,out) | many | no | Fused optimizer step |
| reduce_sum | 2 | len, 0 | no | Global sum reduction |
| masked_fill | 3 (A,mask,out) | len, value | no | Conditional fill |
| dropout | 3 (A,mask,out) | len, 0 | no | Apply dropout mask |
| clone | 2 | len, 0 | yes | Buffer copy |
| causal_mask | 1 | T | no | Generate causal mask |

### Missing Kernels (by priority)

1. **flash_attention**: Fused QK^T + scale + softCap + mask + softmax + dropout + @V
2. **matmul_transposed**: Matmul where B is read transposed (eliminates explicit transpose)
3. **residual_dropout_add**: Fused dropout + residual addition
4. **cross_entropy_backward_gpu**: GPU-native CE backward (currently falls back to CPU)
5. **layernorm_fused**: Single-pass forward using Welford's algorithm
6. **add_inplace**: In-place element-wise add for gradient accumulation

---

## 10. Comparison with Reference Implementations

### vs nanoGPT (Karpathy)
- nanoGPT uses PyTorch's Flash Attention (cuDNN) -- we don't have this, it's our biggest gap
- nanoGPT's training loop structure is similar (warmup + cosine decay + grad clip)
- Our hyperparams (beta2=0.95, gradClip=5.0, warmup=1000) match nanoGPT conventions
- nanoGPT achieves ~50K tok/s on A100 for 124M GPT-2. We get 895 tok/s on L4 for 17.6M. Accounting for hardware difference (A100 ~5x L4 for FP32), we're at roughly 1/10th efficiency, suggesting significant headroom

### vs llm.c (Karpathy)
- llm.c uses custom CUDA kernels with Flash Attention -- similar to what we should build
- llm.c reports ~300K tok/s on A100 for GPT-2 124M with custom kernels
- Key insight: custom kernels beat PyTorch by 3-6x. We have the same opportunity

### vs our previous run (20260224064526_4kwo)
- Previous: 6L/256D/8H batch=8 block=256, ~7.5M params, 750 tok/s
- Current: 8L/384D/8H batch=20 block=512, ~17.6M params, 895 tok/s
- 2.3x more params/step with only 19% throughput increase -- this means per-param efficiency improved
- Previous run had grad_norm spikes averaging 108K (max 301M). Current run has spikes at max 1306 -- 230x improvement from softCap + better clipping

---

## 11. Recommended Priority Actions

### Immediate (this week)
1. **Increase MAX_PENDING_OPS to 256** -- free performance, 5 min change
2. **Add vec4 variants for softcap kernels** -- currently scalar-only, leaving 4x on the table
3. **Implement size-class rounding in buffer pool** -- reduce allocation churn
4. **Monitor clip_pct** -- if >20% past warmup, investigate data quality or reduce lr

### Short term (1-2 weeks)
5. **Fused matmul_transposed kernel** -- eliminate all explicit transpose ops
6. **GPU-native cross_entropy_backward** -- stop the forced flush in backward
7. **Auto-tune matmul tile size** -- 32x32 or 64x64 for larger matrices
8. **Fused residual_dropout_add kernel** -- reduce dispatch count and VRAM pressure

### Medium term (2-4 weeks)
9. **Flash Attention kernel** -- single biggest performance win, eliminates attention intermediate VRAM
10. **In-place gradient accumulation kernel** -- reduce backward memory by ~30%
11. **Fused single-pass LayerNorm** -- Welford's algorithm, forward + backward

### Long term (1-2 months)
12. **Multi-GPU support** -- data parallelism with custom all-reduce over NCCL-free ring communication
13. **Mixed precision (FP16/BF16)** -- 2x memory reduction, requires careful loss scaling
14. **Activation checkpointing** -- trade compute for memory, enable larger models on same GPU

---

## 12. Current Codebase Strengths

1. **Excellent GPU memory management**: The `releaseFn` + `syncGpu()` pattern is production-grade. Deterministic cleanup prevents the OOM death spiral that plagues naive FinalizationRegistry-only approaches.

2. **Clean tape-based autograd**: The backward pass correctly handles gradient accumulation, activation release, and multi-consumer Variables. The code is readable and correct.

3. **Compute graph batching**: The lazy evaluation pattern with automatic flushing at 64 ops is well-designed. The per-op overhead of ~2us in batch mode is excellent.

4. **Comprehensive telemetry**: Per-step timing breakdown, GPU memory stats, gradient norm diagnostics, clipping telemetry -- all invaluable for debugging training issues.

5. **Native SPIR-V generation**: Generating shaders from TypeScript without any shader compiler dependency is a major engineering achievement. The code is maintainable and extensible.

6. **AdamW on GPU**: The fused `adamwStep` kernel avoids 5 separate dispatches per parameter. Moment buffers stay on GPU, avoiding upload/download per step.

---

## 13. Summary

**Where we are**: A working, training-from-scratch GPT system with custom GPU backend. Training is stable and loss is decreasing. The clipping telemetry shows gradient spikes are being handled correctly.

**Biggest bottleneck**: Individual kernel efficiency and lack of fusion. We dispatch ~1442 separate GPU operations per step. A fused Flash Attention kernel alone would eliminate ~40% of VRAM bandwidth pressure and reduce dispatches by ~200.

**VRAM utilization**: Only using 12% of L4's 24GB in our buffer pools. The gap is likely Vulkan driver overhead + allocation fragmentation. Larger models (12L/512D) OOM not because of total VRAM but because of peak transient memory during backward passes.

**Throughput gap**: ~10x behind reference implementations on equivalent hardware. The path to close this gap is clear: Flash Attention > matmul tuning > kernel fusion > mixed precision.

**Training stability**: Much improved from previous runs (max spike 1306 vs 301M). softCap + proper gradClip are working. Key metric to watch: clip_pct after warmup completes (step 1000).
