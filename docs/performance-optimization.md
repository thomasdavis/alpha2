# Performance Optimization: 307M Model on L4 (145 tok/s → target 500+ tok/s)

Current: **145 tok/s** on NVIDIA L4 24GB with helios Vulkan backend.
Config: 21L 1024d 16h swiglu, batch=1, accumSteps=4, block=512, activation checkpointing.

---

## Executive Summary

The biggest gains come from three categories:

| Category | Current State | Fix | Estimated Gain |
|----------|--------------|-----|----------------|
| **F16 compute** | F16 storage only, all compute in f32 | Native f16 matmul/attention kernels | **+80-120%** |
| **Matmul kernel** | 16x16 tile, no vec4, no tensor cores | Larger tiles + vec4 + subgroup ops | **+40-60%** |
| **Dispatch overhead** | Barriers between every op, sync readbacks | Dependency-aware barriers, deferred reads | **+25-40%** |

**Realistic target: 350-500 tok/s** (2.5-3.5x improvement) with the top optimizations.

---

## TIER 1: Highest Impact (each >30% improvement potential)

### 1. Native F16 Matmul and Attention Kernels (+80-120%)

**Problem**: The `f16: true` flag only casts activations to f16 for *storage*. All actual computation (matmul, attention, elementwise) runs in f32. The L4 GPU has 2x f16 throughput vs f32 (242 TFLOPS f16 vs 121 TFLOPS f32).

**Current code path** (`backend.ts` matmul):
```
f16 activation → cast to f32 (GPU dispatch) → f32 matmul (GPU dispatch) → cast to f16 (GPU dispatch)
```
That's 3 dispatches where 1 would suffice. The matmul kernel (`kernels/matmul.ts`) is hardcoded to f32:
- Loads f32 from global memory into shared memory tiles
- Computes f32 multiply-accumulate in inner loop
- Stores f32 result

**Fix**: Generate f16 matmul variants that:
1. Load f16 from global memory (half the bandwidth)
2. Convert to f32 in registers for accumulation (maintains precision)
3. Store result as f16 (half the bandwidth)
4. Fuse the cast+matmul into a single kernel dispatch

The SPIR-V generator (`spirv.ts`) already supports Float16 capability and f16 storage buffers. The f16 kernels in `kernels/f16.ts` prove the infrastructure exists.

**Implementation sketch** (`kernels/matmul.ts`):
```
// New: kernelMatmulF16 variant
// Shared memory tiles remain f32 (for accumulation precision)
// Global loads use OpLoad with f16 type, then OpFConvert to f32
// Global stores use OpFConvert f32→f16, then OpStore
// This halves global memory bandwidth with no precision loss in accumulation
```

Same approach for flash attention: load Q/K/V as f16, compute scores in f32, store output as f16. The attention kernel (`kernels/attention.ts`) already puts Q and O in per-thread registers — just change the load/store types.

**Why this is the #1 optimization**: Matmul dominates training time (~70% of GPU cycles for a transformer). Cutting memory bandwidth in half directly translates to ~2x throughput on bandwidth-bound operations. L4's memory bandwidth is 300 GB/s — f16 effectively gives you 600 GB/s equivalent throughput for the same data.

**Estimated improvement**: +80-120% throughput → **260-320 tok/s**

---

### 2. Improved Matmul Tiling & Vectorization (+40-60%)

**Problem**: Current matmul uses 16x16 tiles with scalar f32 loads. This underutilizes L4's capabilities.

**Current kernel** (`kernels/matmul.ts`):
- Tile size: 16×16 (configurable, 32×32 variant exists but unused for small M)
- Each thread computes 1 output element
- Scalar loads: 1 f32 per load instruction (4 bytes)
- Shared memory: 2×256 floats (2 KB) — very small, high occupancy but low throughput per thread
- K-dimension loop: sequential with barrier per tile

**Optimizations**:

**a) Vec4 loads for matmul** (+15-20%)
The element-wise kernels already have vec4 variants (`kernelAddVec4`, etc.) but matmul doesn't. Each thread should load 4 floats at once (128-bit loads). This is a 4x reduction in load instructions.

```
// Instead of: a_tile[ty][k] = A[row * K + (tileK + k)]  (4 bytes)
// Use:        a_tile[ty][k:k+4] = A_vec4[row * (K/4) + (tileK/4 + k/4)]  (16 bytes)
```

**b) Each thread computes multiple output elements** (+20-30%)
Instead of 1 output per thread, compute a 2×2 or 4×4 sub-tile per thread. This increases arithmetic intensity (more FLOPs per byte loaded from shared memory).

```
// Current: 16×16 threads → 16×16 output tile (1 element/thread)
// Proposed: 16×16 threads → 32×32 output tile (4 elements/thread, 2×2 sub-tile)
// Or:       8×8 threads → 32×32 output tile (16 elements/thread, 4×4 sub-tile)
```

**c) Double-buffered K-tiles** (+10-15%)
Load the next K-tile while computing on the current one. Requires 2x shared memory per tile but hides global memory latency.

```
// Load tile[0] → barrier → compute on tile[0] while loading tile[1] → barrier → ...
// Requires 4 shared memory tiles instead of 2 (still only ~4KB)
```

**d) Subgroup operations** (+5-10%)
The L4 supports Vulkan subgroup operations (wave size 32). Replace some shared memory + barrier patterns with subgroup shuffles for the inner reduction. Currently NO subgroup operations are used anywhere in the codebase.

**Estimated improvement**: +40-60% on top of f16 → **360-500 tok/s**

---

### 3. Dependency-Aware Barriers Instead of Blanket Barriers (+25-40%)

**Problem**: The batch dispatch system (`helios_vk.c` lines 1776-1786) inserts a full `VkMemoryBarrier` between EVERY dispatch in a batch, even when operations are independent.

**Current** (in `batchDispatch()`):
```c
if (batchDispatchCount > 0) {
    VkMemoryBarrier barrier = {
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    };
    vkCmdPipelineBarrier(batchCmdBuf, COMPUTE, COMPUTE, 0, 1, &barrier, ...);
}
```

This serializes ALL GPU work within a batch. If a batch has 256 ops (the max), and many are independent (e.g., computing gradients for different parameters), the GPU can't execute them in parallel.

**Fix**: Track buffer read/write sets per operation. Only insert barriers when operation N reads a buffer that operation M (M < N) wrote to. Independent operations can execute in parallel on different GPU compute units.

**Implementation**: In the `ComputeGraph` class (`backend.ts`), track `lastWriter: Map<bufferHandle, opIndex>` for each buffer. When adding an op to the batch:
```typescript
for (const inputBuf of op.inputs) {
    if (lastWriter.has(inputBuf)) {
        needsBarrier = true;
        break;
    }
}
```

Then in the C code, emit barriers only when the JS layer signals a dependency.

**Why this matters**: A transformer forward pass has many independent operations: layer norms, residual connections, different attention heads. These could overlap on the GPU if not serialized by unnecessary barriers.

**Estimated improvement**: +25-40%

---

## TIER 2: High Impact (each 10-25% improvement)

### 4. Eliminate CPU-GPU Synchronization in Training Loop (+10-20%)

**Problem**: The training loop forces CPU-GPU synchronization at multiple points per step.

**a) Loss readback per micro-batch** (trainer.ts:446):
```typescript
const microLoss = (loss.data.data as Float32Array)[0]; // GPU READBACK → CPU BLOCKS
```
This happens once per micro-batch (4 times per step with accumSteps=4). Each access triggers `graph.flush()` → `waitTimeline()` → `readBuffer()`. The CPU blocks waiting for the GPU.

**Fix**: Accumulate loss on GPU. Only read back the final accumulated value:
```typescript
// Instead of reading loss each micro-step, keep a running GPU sum
let lossAccum = backend.zeros([1]);
for (let microStep = 0; microStep < accumSteps; microStep++) {
    const { loss } = gptForward(...);
    backend.addInplace(lossAccum, backend.scale(loss.data, 1/accumSteps));
    tape.backward(loss, backend, releaseFn);
    tape.clear(releaseFn);
}
// Single readback at end
const lossVal = (lossAccum.data as Float32Array)[0];
```

**b) Gradient norm readback** (trainer.ts:520):
```typescript
const val = (sqNormParts[pi].data as Float32Array)[0]; // GPU SYNC for each parameter
```
Iterates over ALL parameters (~300+ tensors), reading each norm scalar from GPU. This should be reduced to GPU-side via a single reduction kernel.

**Fix**: Implement `backend.totalSumOfSquares(tensors[])` that computes the total norm in a single GPU dispatch, with one CPU readback at the end.

**c) Gradient clipping spot-check** (trainer.ts:645-650):
```typescript
const val = (s.data as Float32Array)[0]; // 3x GPU SYNC for Inf check
```

**Fix**: Batch all 3 checks into a single GPU dispatch, read once.

**Estimated improvement**: +10-20%

---

### 5. Fused AdamW Multi-Parameter Kernel (+5-15%)

**Problem**: The AdamW optimizer (`optimizers.ts`) issues a separate GPU dispatch for each parameter tensor (~300+ parameters for 307M model). Each dispatch has:
- Descriptor set allocation/update (~3µs)
- Pipeline barrier (~1µs)
- Command recording overhead

Total: ~300 × 4µs = 1.2ms overhead per optimizer step.

**Current** (`optimizers.ts:66-69`):
```typescript
for (const [name, param] of params) {
    this.backend.adamwStep(param, grad, m, v, lr, beta1, beta2, eps, wd, bc1, bc2);
}
```

**Fix**: Implement a multi-parameter AdamW that processes all parameters in a single dispatch using a parameter table:
```
// Single kernel: iterate over parameter table
// GPU memory layout: [param_offset, param_size, grad_offset, m_offset, v_offset] per param
// One dispatch: ceil(total_params / WG_SIZE) workgroups
```

This is feasible because AdamW is element-wise — each parameter element is independent. Packing all parameters into contiguous buffers allows a single dispatch.

**Estimated improvement**: +5-15%

---

### 6. Checkpoint Forward-Phase Optimization (+5-10%)

**Problem**: Activation checkpointing clones the layer output (`B.clone()`) at every layer during the forward pass. For 21 layers, that's 21 clone operations (each a full GPU memcpy).

**Current** (`checkpoint.ts:49`):
```typescript
const outputData = B.clone(tmpOutput.data); // Full tensor copy on GPU
```

**Fix**: Use in-place output reuse. The throwaway tape's output Variable wraps a GPU buffer. Instead of cloning, steal the buffer reference:
```typescript
// Don't clone — just take ownership of the buffer
const outputData = tmpOutput.data;
// Prevent tmpTape.clear() from releasing this specific buffer
tmpTape.clearExcept(ctx.release, tmpOutput);
```

This saves 21 × (2MB clone for [1,512,1024] at f32) = 42MB of GPU memcpy per forward pass.

**Estimated improvement**: +5-10%

---

### 7. Flash Attention Backward Optimization (+10-15%)

**Problem**: The attention backward kernel has high register pressure. For head_dim=64 (1024d / 16h), each thread stores `regQ[64]` and `regO[64]` — 512 bytes in registers. The L4 has 256KB register file per SM, shared among threads.

With Br=16 threads per workgroup and 512 bytes/thread = 8KB registers per workgroup. L4 can run ~32 concurrent workgroups per SM. This is fine for D=64 but becomes a problem if head_dim increases.

More importantly, the backward kernel creates dQ, dK, dV arrays — tripling register pressure. The backward pass recomputes attention scores, requiring:
- Load Q, K, V (shared memory)
- Recompute softmax (shared memory reduction)
- Compute dQ, dK, dV (accumulate in registers)
- Multiple barriers per key block

**Fix**:
- Split backward into separate dQ and dKV passes (reduces peak register usage by 50%)
- Use subgroup shuffle operations instead of shared memory for small reductions
- Pre-compute and store the softmax output (LSE) from forward pass to avoid recomputation

**Estimated improvement**: +10-15%

---

## TIER 3: Medium Impact (each 3-10% improvement)

### 8. Fused Softmax Backward Kernel (+3-5%)

**Problem**: Softmax backward (`ops.ts`) creates 5 intermediate GPU tensors:
```typescript
const sg = B.mul(out, g);           // Intermediate 1
const sumSg = B.sum(sg, axis, true); // Intermediate 2
const expanded = broadcastTo(...);    // Intermediate 3
const diff = B.sub(g, expanded);      // Intermediate 4
const result = B.mul(out, diff);      // Intermediate 5
```

That's 5 GPU dispatches + 5 buffer allocations per softmax backward. With 21 layers, that's 105 extra dispatches.

**Fix**: Implement `kernelSoftmaxBackward()` as a single fused kernel that computes `out * (g - sum(out * g))` in one dispatch with shared memory for the reduction.

### 9. Reduce QKV Slicing Overhead (+2-3%)

**Problem**: Each layer does 3 separate `slice()` operations to split QKV (`gpt.ts:240-242`), creating 3 tape entries with backward closures that use expensive `scatterSlice` or `padWithCat` fallback.

**Fix**: Implement `backend.splitQKV(qkvFlat, nEmbd)` as a single GPU dispatch that writes Q, K, V to three separate output buffers. Record as one tape entry with a fused backward.

### 10. Reduce GC Overhead (+2-5%)

**Problem**: With `gcEvery=1`, JavaScript GC runs every training step. For a multi-GB heap (788MB dataset + model params + optimizer state), this can take 1-3ms per step.

**Fix**: Use `gcEvery=10` or `gcEvery=50`. The explicit release callbacks (`releaseFn`) handle GPU memory deterministically — GC is only needed for JS-side cleanup. Every 10-50 steps is sufficient.

### 11. Build with -O3 and LTO (+1-3%)

**Problem**: The native addon is compiled with `-O2`:
```bash
gcc -shared -fPIC -O2 -Wall ... -o helios_vk.node helios_vk.c -ldl
```

**Fix**: Use `-O3 -flto -march=native`:
```bash
gcc -shared -fPIC -O3 -flto -march=native -Wall ... -o helios_vk.node helios_vk.c -ldl
```

This enables more aggressive inlining, vectorization, and link-time optimization of the native Vulkan dispatch path.

### 12. Output Pool Fragmentation (+2-3% VRAM efficiency)

**Problem**: The output pool rounds allocations to coarse bins:
```typescript
if (bytes <= 1_048_576) return Math.ceil(bytes / 262144) * 262144;  // 256KB bins
return Math.ceil(bytes / 4_194_304) * 4_194_304;  // 4MB bins above 1MB
```

A 1.1MB allocation rounds to 4MB — 72% wasted. With hundreds of intermediates, this adds up to 500MB-1GB of VRAM waste.

**Fix**: Use finer-grained bins: 64KB bins up to 1MB, 1MB bins up to 16MB, 4MB bins above.

---

## TIER 4: Architectural Changes (longer-term, highest ceiling)

### 13. Operator Fusion / Graph Compiler

Instead of dispatching each operation as a separate GPU kernel (matmul, add, layernorm, etc.), fuse chains of operations into single mega-kernels:

- **ResidualBlock fusion**: `LN → QKV matmul → attention → projection → residual add → LN → FFN` as one kernel
- **Backward fusion**: Fuse gradient chains to avoid intermediate buffer allocation
- Requires a small graph compiler that analyzes the tape and generates fused SPIR-V

This is what PyTorch's `torch.compile` and TensorRT do. It's a major effort but can yield another 2-3x improvement.

### 14. Pipeline Parallelism

With 21 layers, split the model across time: while layer 21's backward runs on micro-batch 1, layer 1's forward runs on micro-batch 2. This keeps the GPU fully utilized instead of alternating between forward and backward phases.

### 15. Persistent Kernel Approach

Instead of dispatching separate kernels per operation, launch a single persistent kernel that stays resident on the GPU and processes a stream of operations from a GPU-side command buffer. This eliminates all dispatch overhead.

---

## Implementation Priority

**Week 1**: Items 1 (f16 matmul), 4a (deferred loss readback), 10 (gcEvery), 11 (build flags)
- Expected: 145 → 280-320 tok/s

**Week 2**: Items 2 (matmul tiling), 3 (smart barriers), 5 (fused AdamW)
- Expected: 280 → 400-500 tok/s

**Week 3**: Items 6 (checkpoint optimization), 7 (attention backward), 8-9 (fused kernels)
- Expected: 400 → 500-600 tok/s

---

## Appendix: Current Architecture Reference

### Kernel Inventory

| Kernel | Op | Workgroup | Shared Mem | Vec4 | F16 |
|--------|----|-----------|-----------|------|-----|
| matmul | A×B^T | 16×16 | 2×256 f32 | No | No |
| flash_attn | QKV→O | Br=16 | Bc×D×2 f32 | No | No |
| elementwise | add/mul/etc | 256 | None | Yes | Storage only |
| softmax | per-row | 256 | 256 f32 | No | No |
| layernorm | per-sample | 256 | 256 f32 | No | No |
| adamw | per-element | 256 | None | No | No |
| reduction | sum/max | 256 | 256 f32 | No | No |

### GPU Dispatch Path
```
JS: backend.matmul(a, b)
  → graph.addOp(pipeline, [bufA, bufB, bufOut], groups, push)
  → (queued, up to 256 ops)
  → graph.flush()
    → vk.batchBegin()          // wait for previous batch
    → for each op:
        vk.batchDispatch(...)  // record to cmd buffer + barrier
    → vk.batchSubmit()         // single vkQueueSubmit + timeline signal
```

### Training Step Breakdown (estimated)
```
14,000ms total per step (accumSteps=4)
├── 4× forward pass:     ~4,000ms (28%)  ← matmul + attention dominate
├── 4× backward pass:    ~7,000ms (50%)  ← matmul + attention backward
├── optimizer step:       ~1,000ms (7%)
├── grad norm + clip:     ~500ms (4%)
├── GPU sync/readback:    ~500ms (4%)
├── GC + buffer mgmt:     ~500ms (4%)
└── data loading + misc:  ~500ms (4%)
```

### Files Referenced
- `packages/helios/src/backend.ts` — GPU backend, compute graph, buffer pools
- `packages/helios/src/kernels/matmul.ts` — Matmul SPIR-V generation
- `packages/helios/src/kernels/attention.ts` — Flash attention SPIR-V
- `packages/helios/src/kernels/elementwise.ts` — Elementwise ops
- `packages/helios/src/kernels/nn.ts` — LayerNorm, Softmax, CrossEntropy
- `packages/helios/src/kernels/optimizer.ts` — AdamW kernel
- `packages/helios/src/spirv.ts` — SPIR-V bytecode generator
- `packages/helios/native/helios_vk.c` — Vulkan native addon (2127 lines)
- `packages/model/src/gpt.ts` — Model forward pass
- `packages/train/src/trainer.ts` — Training loop
- `packages/autograd/src/ops.ts` — Differentiable operations
- `packages/autograd/src/checkpoint.ts` — Activation checkpointing
- `packages/train/src/optimizers.ts` — AdamW optimizer
