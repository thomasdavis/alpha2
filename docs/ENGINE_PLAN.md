# Alpha Engine: Phase 2 Build Plan

**Author**: Claude (Opus 4.6), with architectural direction from human lead
**Date**: 2026-02-24
**Baseline**: Run 20260224_114825 — 8L/384D/8H, 17.6M params, 895 tok/s, L4 24GB

---

## The Core Truth

Alpha has a working, stable, telemetry-rich training system built from scratch. The trainer logic, autograd, memory management, and clipping are all sound. The bottleneck is now singular:

**Kernel fragmentation and intermediate memory traffic.**

Not dispatch overhead. Not batch submit latency. Not trainer-level logic. The GPU is spending most of its time reading and writing intermediate tensors that only exist because operations aren't fused.

Every optimization in this plan targets that bottleneck.

---

## Principles

1. **Never break training stability.** Every change ships with a regression gate.
2. **Measure before and after.** No change lands without a benchmark comparison.
3. **Stage moonshots.** Flash Attention is 5 milestones, not 1.
4. **Fuse the hot path first.** Profile says where the time goes. Fuse those ops.
5. **Single-GPU efficiency before multi-GPU.** Scaling an inefficient kernel stack multiplies debugging pain.

---

## Tier 0 — Guardrails (before any kernel work)

These exist to protect us while we modify hot code paths.

### 0A. Golden Mini-Run Regression Test

**What**: A scripted 100-step training run that asserts invariants.

**Checks**:
- Loss at step 100 < loss at step 1
- No NaN in loss or grad_norm at any step
- All grad_norms are finite
- Checkpoint saves and reloads without error
- Reloaded checkpoint resumes training with loss within 0.1 of pre-save

**Implementation**:
- New script: `scripts/regression-test.ts`
- Uses a small fixed dataset (first 1MB of concordance.txt, deterministic seed)
- Runs on both `helios` and `cpu_ref` backends
- Compares final loss between backends (should be within 0.5 — different numerics but same trajectory)
- Exit code 0 = pass, 1 = fail
- Run before and after every kernel change

**Definition of done**: `npx tsx scripts/regression-test.ts` passes on both backends in under 60 seconds.

### 0B. Kernel Correctness Harness

**What**: Per-kernel comparison of GPU vs CPU output on random inputs.

**Kernels to test**:
- softCap forward/backward
- matmul (2D and batched)
- layerNorm forward/backward
- crossEntropy forward/backward
- gelu forward/backward
- embedding forward/backward
- softmax
- All future fused kernels

**Method**:
- Generate random Float32Array inputs (seeded RNG for reproducibility)
- Run through GPU backend, read back result
- Run through CPU ref backend
- Assert max absolute error < tolerance (1e-4 for f32, 1e-2 for f16)
- Assert relative error < tolerance where absolute values > 1.0

**Implementation**:
- New script: `scripts/kernel-test.ts`
- Test matrix: each kernel x multiple sizes [1, 64, 4096, 65536, 1048576]
- Report: per-kernel max_abs_error, mean_abs_error, max_rel_error
- Fail if any kernel exceeds tolerance

**Definition of done**: All current kernels pass. New kernels must be added to this harness before they can be used in training.

---

## Sprint A — Remove Forced Sync + Reduce Dispatch Count

**Goal**: Eliminate the worst architectural bottlenecks without writing any complex new algorithms. Every task here is a scoped kernel or a config change.

**Benchmark gate**: 100-step run, compare step_time and ops/step vs baseline.

### A1. GPU-Native `cross_entropy_backward`

**Problem**: `crossEntropyBackward` in `backend.ts` reads logits back to CPU via `.data` access, which forces `graph.flush()` + `waitTimeline()` + `readBuffer()`. This is a full GPU pipeline stall every single step, right in the middle of backward.

**Current code path**:
```
backward tape walks entries...
  -> crossEntropy backward closure fires
  -> accesses logits.data (lazy tensor)
  -> graph.flush() triggered
  -> GPU pipeline drains
  -> readBuffer() copies logits to CPU
  -> CPU computes softmax + gradient
  -> result uploaded back to GPU
  -> remaining backward entries resume
```

**Solution**: Write a SPIR-V kernel `cross_entropy_backward` that:
1. Takes logits [N, C], targets [N], gradOutput [1] as inputs
2. Computes softmax per row (online stable softmax: max, sub, exp, sum, div)
3. Subtracts 1.0 at the target index
4. Scales by gradOutput / N
5. Outputs gradLogits [N, C]

**Kernel design**:
- One workgroup per row (N rows total)
- Shared memory for row max and row sum
- Three passes within workgroup: (1) find max, (2) sum exp(x-max), (3) compute grad
- Push constants: N, C, gradScalar

**Files to modify**:
- `packages/helios/src/kernels.ts` — add `kernelCrossEntropyBackward`
- `packages/helios/src/backend.ts` — replace CPU path in `crossEntropyBackward()`
- `scripts/kernel-test.ts` — add CE backward test case

**Acceptance criteria**:
- No `.data` access on logits in CE backward path during training
- No `graph.flush()` triggered from CE backward
- Gradient values match CPU ref within 1e-4
- Step time decreases measurably (expected: 2-8%)
- 100-step regression test passes

**Risk**: Medium. Shared-memory SPIR-V is more complex than element-wise kernels. The softmax numerical stability (subtract max before exp) is critical to get right.

---

### A2. Vec4 `softcap_forward` and `softcap_backward`

**Problem**: softCap kernels are scalar-only. Every other element-wise kernel has vec4 variants that process 4 elements per thread invocation, giving ~4x throughput on aligned data.

**Current**: `softcap_forward` and `softcap_backward` in `kernels.ts` process one element per invocation.

**Solution**: Add `softcap_forward_vec4` and `softcap_backward_vec4` kernel generators following the same pattern as `kernelAdd` (which has both scalar and vec4). Update `backend.ts` to use vec4 when `(size & 3) === 0`.

**Files to modify**:
- `packages/helios/src/kernels.ts` — add vec4 variants
- `packages/helios/src/backend.ts` — update `softCap()` and `softCapBackward()` to select vec4

**Acceptance criteria**:
- Standalone softCap throughput benchmark improves (target: 3-4x)
- Numeric parity with scalar kernel within 1e-6
- Kernel test harness passes

**Risk**: Low. Straightforward pattern already established in other kernels.

---

### A3. Fused `residual_dropout_add` Kernel

**Problem**: Every transformer layer does this twice:
```typescript
const projDrop = dropout(ctx, projected, config.dropout, training);  // dispatch 1: mask gen + apply
x = add(ctx, x, projDrop);                                           // dispatch 2: element-wise add
```
That's 2 dispatches + 1 intermediate tensor (projDrop) per residual, times 2 residuals per layer, times 8 layers = 32 dispatches and 32 intermediate VRAM writes eliminated.

**Solution**: Single SPIR-V kernel `residual_dropout_add`:
- Inputs: residual [N], projected [N], dropout_mask [N] (or RNG seed + probability)
- Output: residual + projected * mask * (1/(1-p))
- Push constants: len, dropout_prob, training_flag, rng_seed

**Design choice — mask generation**:
Two options:
1. Pre-generate mask on CPU, pass as input buffer (current approach for dropout)
2. Generate mask in-kernel using deterministic RNG from push constants

Option 1 is safer (same RNG behavior), option 2 is faster (eliminates mask tensor entirely). Start with option 1 for correctness parity, upgrade to option 2 later.

**Files to modify**:
- `packages/helios/src/kernels.ts` — add `kernelResidualDropoutAdd`
- `packages/helios/src/backend.ts` — add `residualDropoutAdd()` method
- `packages/core/src/interfaces.ts` — add to Backend interface
- `packages/autograd/src/ops.ts` — add fused `residualDropoutAdd()` op
- `packages/model/src/gpt.ts` — use fused op in forward pass

**Backward design**:
The backward of `residual_dropout_add(x, projected, mask)` is:
- grad_x = upstream_grad (pass through)
- grad_projected = upstream_grad * mask * (1/(1-p))

This can be a single kernel `residual_dropout_add_backward` or just reuse the existing `mul` with the mask.

**Acceptance criteria**:
- Same dropout semantics (same mask values at same RNG state)
- ops/step drops by ~32
- Intermediate tensor count per step drops by ~32
- 100-step regression test passes
- Loss curve shape unchanged on 500-step comparison

**Risk**: Low-medium. The op itself is simple. The integration into `gpt.ts` forward pass requires careful threading of the dropout mask for backward.

---

### A4. Raise `MAX_PENDING_OPS` to 256

**Problem**: With 1442 ops/step and MAX_PENDING_OPS=64, we get 22+ flush cycles. While per-flush overhead is small (~100us), each flush creates a command buffer boundary that may cause brief GPU idle time.

**Solution**: Change `MAX_PENDING_OPS` from 64 to 256 in `backend.ts`. Make it configurable via an environment variable for tuning.

**Files to modify**:
- `packages/helios/src/backend.ts` — change constant, add env var override

**Acceptance criteria**:
- Flush count per step drops from ~22 to ~6
- No increase in peak VRAM usage
- No training regression

**Risk**: Very low. The only concern is command buffer size limits on some drivers, but 256 dispatches is well within Vulkan spec limits.

---

### A5. Buffer Pool Size-Class Rounding

**Problem**: 26 size classes in the output pool. Tensors of 156.3MB and 160.0MB get separate pools, wasting buffer reuse opportunities. Every new tensor shape creates a new size class.

**Solution**: Round allocation sizes up to the nearest power-of-2 boundary, or use coarse bins (e.g., round up to nearest 4MB for allocations > 1MB, nearest 256KB for smaller). This collapses many size classes into fewer bins.

**Implementation**:
```typescript
function roundPoolSize(bytes: number): number {
  if (bytes <= 4096) return 4096;
  if (bytes <= 1_048_576) return Math.ceil(bytes / 262144) * 262144;  // 256KB bins
  return Math.ceil(bytes / 4_194_304) * 4_194_304;  // 4MB bins
}
```

Apply in `acquireOutputRegion` and `acquireBuffer`.

**Files to modify**:
- `packages/helios/src/backend.ts` — add rounding in pool acquire functions

**Acceptance criteria**:
- Size class count drops from 26 to <15
- Total allocation count over 100 steps decreases
- No VRAM increase (rounding waste < 5% of pool size)
- Buffer reuse rate increases (visible in gpu_mem logs)

**Risk**: Low. Over-allocation is bounded by the bin size (max 4MB waste per buffer). Total waste for 134 buffers at worst: 134 * 4MB = 536MB, but most will waste far less.

---

### Sprint A Benchmark Gate

Run the current config (8L/384D/8H batch=20 block=512) for 200 steps, before and after Sprint A:

| Metric | Baseline | Target |
|--------|----------|--------|
| step_time | 11.4s | < 10.5s (8%+ reduction) |
| ops/step | 1442 | < 1400 |
| flush_count/step | 22 | < 8 |
| output pool size classes | 26 | < 15 |
| loss at step 200 | -- | within 0.1 of baseline |

---

## Sprint B — Attack Backward Memory Traffic

**Goal**: Reduce the number of intermediate tensors created during backward, and eliminate explicit transpose operations. This directly reduces peak VRAM and enables larger models/batches.

### B1. `matmul_transposed` Kernels

**Problem**: The forward pass does 6 matmuls per layer, and 5 of them involve an explicit `transpose()` of a weight matrix first:
```typescript
matmul(ctx, q3d, transpose(ctx, layer.attn.wq, 0, 1))  // Q projection
```
Each `transpose()` allocates a new GPU buffer and copies the entire weight matrix with reindexed access. For a [384, 384] weight, that's 589KB per transpose, 30 transposes per forward pass = 17.7MB of pure copy overhead.

In backward, `matmul backward` does 2 more transposes per matmul = 12 additional transposes per layer = 96 transposes total in backward.

**Solution**: Two new kernel variants:

1. **`matmul_nt`** (A @ B^T): Reads B with transposed indexing. Used for all weight projections in forward (`x @ W^T`).
2. **`matmul_tn`** (A^T @ B): Reads A with transposed indexing. Used in backward for gradient w.r.t. weights (`x^T @ grad`).

**Kernel design**:
Same tiled matmul structure as existing kernel, but the inner loop reads B[k][n] instead of B[n][k] (or A[m][k] instead of A[k][m] for the transposed variant). Shared memory tiling absorbs the access pattern change.

**Integration**:
- Add `matmulNT` and `matmulTN` to Backend interface
- Update `ops.ts` matmul to detect transposed inputs and use fused variant
- OR: add new `linearForward(x, weight)` op that does `x @ weight^T` as a single fused call, and `linearBackward` that returns both `grad @ weight` and `x^T @ grad`

The second approach (linear op) is cleaner because it captures the full pattern: every weight projection in the model is a linear layer.

**Files to modify**:
- `packages/helios/src/kernels.ts` — add `kernelMatmulNT`, `kernelMatmulTN`
- `packages/helios/src/backend.ts` — add `matmulNT()`, `matmulTN()` methods
- `packages/core/src/interfaces.ts` — add to Backend interface
- `packages/autograd/src/ops.ts` — add `linear()` op or update matmul to detect transposes
- `packages/model/src/gpt.ts` — use `linear()` for all weight projections

**Acceptance criteria**:
- Explicit transpose dispatch count drops from ~126/step to near zero
- Total ops/step drops by ~100+
- Output pool churn decreases (fewer transient transpose buffers)
- Numeric parity with existing matmul within 1e-4
- 100-step regression test passes

**Risk**: Medium. Matmul kernels are the most performance-sensitive code. Tiled matmul with transposed access patterns needs careful shared memory indexing. Test extensively against CPU ref.

---

### B2. In-Place Gradient Accumulation (`add_inplace`)

**Problem**: When a Variable has multiple gradient contributors (e.g., residual connections), the tape does:
```typescript
input.grad = backend.add(input.grad, g);  // allocates NEW tensor
releaseTensor(oldGrad);                     // frees old
releaseTensor(g);                           // frees contributor
```
This creates a temporary tensor for every accumulation. With 8 layers and 2 residual connections each, that's ~16 extra allocations in backward.

**Solution**: Add `addInPlace(target, source)` to the backend that modifies `target` directly:
```typescript
// In tape.ts backward:
if (input.grad) {
  backend.addInPlace(input.grad, g);  // modifies input.grad buffer directly
  releaseTensor(g);
  // no new allocation, no old grad to release
}
```

**Safety rules**:
- Only call on dedicated gradient buffers (never on forward activation data)
- The target buffer must not be in-flight on the GPU (must have completed any pending writes)
- Enforce this by requiring the buffer's `readyValue <= getCompleted()` before in-place write

**Kernel**: Simple element-wise `A[i] += B[i]`, dispatched as a 2-binding kernel with A as both input and output.

**Files to modify**:
- `packages/helios/src/kernels.ts` — add `kernelAddInPlace`
- `packages/helios/src/backend.ts` — add `addInPlace()` method
- `packages/core/src/interfaces.ts` — add to Backend interface
- `packages/autograd/src/tape.ts` — use in-place accumulation when backend supports it

**Acceptance criteria**:
- Allocation count per step decreases
- Peak VRAM usage during backward decreases
- No double-write or aliasing bugs (verified by kernel test harness)
- 100-step regression test passes

**Risk**: Medium. In-place GPU operations require timeline-aware safety. If the target buffer is still being read by a prior dispatch when we write to it, we get data corruption. The `readyValue` check is essential.

---

### B3. Dead-Gradient Elimination in Backward

**Problem**: Some tape entries compute gradients for inputs that don't require grad. The backward closure runs, allocates tensors, computes results, and then the tape discards them because `input.requiresGrad === false`.

Example: `maskedFill` backward computes a gradient for the mask input, which is a non-parameter constant. That gradient is computed and immediately thrown away.

**Solution**: Pass a `needsGrad` boolean array to backward closures, letting them skip unnecessary computation:

```typescript
// In tape.ts:
const needsGrad = entry.inputs.map(inp => inp.requiresGrad);
const inputGrads = entry.backward(outGrad, backend, releaseTensor, needsGrad);
```

Backward closures that check `needsGrad[i]` can skip expensive ops:
```typescript
// matmul backward:
backward: (g, B, release, needsGrad) => {
  const ga = needsGrad[0] ? B.matmul(g, tB) : null;  // skip if A doesn't need grad
  const gb = needsGrad[1] ? B.matmul(tA, g) : null;   // skip if B doesn't need grad
  return [ga, gb];
}
```

**Files to modify**:
- `packages/autograd/src/tape.ts` — extend backward signature with `needsGrad`
- `packages/autograd/src/ops.ts` — update backward closures for matmul, add, mul, div, embedding, layerNorm to check needsGrad

**Acceptance criteria**:
- Backward dispatch count decreases (target: 5-15% reduction)
- No change in parameter gradients (verified by comparing grad checksums)
- Step time decreases

**Risk**: Low. This is a pure optimization that skips unnecessary work. The only risk is accidentally skipping a needed gradient, which the regression test catches immediately.

---

### Sprint B Benchmark Gate

| Metric | Post-Sprint-A | Target |
|--------|---------------|--------|
| step_time | < 10.5s | < 9.0s (15%+ reduction) |
| ops/step | < 1400 | < 1200 |
| transpose ops/step | ~126 | < 10 |
| backward allocs/step | baseline | 20%+ reduction |
| max batch size (stable) | 20 | 24+ (test empirically) |

After Sprint B, attempt batch=24 (which previously OOM'd). If the reduced intermediate count drops peak VRAM enough, this gives us more tokens/step for free.

---

## Sprint C — Major Throughput Upgrade

**Goal**: Fuse the most bandwidth-heavy multi-pass operations into single kernels. These are harder to implement but deliver the largest per-kernel gains.

### C1. Fused LayerNorm Forward

**Problem**: LayerNorm is currently multi-pass: (1) compute mean, (2) compute variance, (3) normalize, (4) scale + shift. Each pass reads and writes the full tensor.

**Solution**: Single-pass kernel using Welford's online algorithm:
- Each workgroup processes one row (one token's embedding vector)
- Pass 1 (in shared memory): accumulate mean and M2 using Welford's
- Compute variance = M2 / dim, invStd = rsqrt(variance + eps)
- Pass 2: normalize, scale, shift in one read-write pass

**Kernel structure**:
- Workgroup size = min(dim, 256)
- Shared memory: partial sums for mean and M2 (or simpler: sum and sum-of-squares)
- For dim > workgroup size: each thread handles multiple elements, then reduce in shared memory
- Push constants: dim, eps, N (number of rows)
- Bindings: input [N, dim], weight [dim], bias [dim], output [N, dim]

**Files to modify**:
- `packages/helios/src/kernels.ts` — add `kernelLayerNormFused`
- `packages/helios/src/backend.ts` — replace multi-pass layerNorm with fused version

**Acceptance criteria**:
- Single dispatch per layerNorm (was 4-5)
- No intermediate mean/variance tensors in VRAM
- Numeric parity with multi-pass version within 1e-5
- 16 fewer dispatches per step (2 LN per layer x 8 layers)

---

### C2. Fused LayerNorm Backward

**Problem**: LayerNorm backward is similarly multi-pass and creates multiple intermediates.

**Solution**: Fused kernel that computes dx, dweight, dbias in a single pass per row, with shared memory for the reduction statistics.

This is more complex than forward because:
- dx requires the forward statistics (mean, invStd) which must be recomputed or cached
- dweight and dbias are reductions across all N rows

**Approach**: Recompute forward stats in backward (compute-vs-memory tradeoff — recomputation is cheaper than storing stats for all N rows).

**Acceptance criteria**:
- Single dispatch for LN backward
- Matches CPU ref within 1e-4
- Combined with C1, eliminates ~64 dispatches per step

---

### C3. Matmul Tile Auto-Tuning

**Problem**: Matmul uses hardcoded 16x16 tiles. For the L4's 58 SMs with 128 FP32 cores each, larger tiles would improve utilization.

**Solution**: Benchmark suite that tests tile sizes [8, 16, 32] on the actual training shapes:

| Shape | Operation |
|-------|-----------|
| [10240, 384] @ [384, 384] | Weight projection (Q,K,V,O) |
| [10240, 384] @ [384, 1536] | MLP fc1 (dim -> 4*dim) |
| [10240, 1536] @ [1536, 384] | MLP fc2 (4*dim -> dim) |
| [20, 8, 512, 48] @ [20, 8, 48, 512] | Attention scores |
| [20, 8, 512, 512] @ [20, 8, 512, 48] | Attention values |

**Implementation**:
- Auto-tune at device init (like workgroup size auto-tuning)
- Store optimal tile size per shape class (projection, MLP, attention)
- Select kernel variant based on matrix dimensions at dispatch time

**Files to modify**:
- `packages/helios/src/kernels.ts` — add tile-size-parameterized matmul generators
- `packages/helios/src/backend.ts` — shape-aware kernel selection in matmul dispatch

**Acceptance criteria**:
- Matmul throughput improves 15-25% on common shapes
- Overall step time improves 8-12%
- Correctness verified against CPU ref

---

### C4. Flash Attention — Forward Only (Prototype)

**Problem**: Attention is the single largest consumer of VRAM bandwidth:
```
Q @ K^T          -> [B, H, T, T] write (60MB)
scale            -> [B, H, T, T] write (60MB)
softCap          -> [B, H, T, T] write (60MB)
maskedFill       -> [B, H, T, T] write (60MB)
softmax          -> [B, H, T, T] write (60MB)
dropout          -> [B, H, T, T] write (60MB)
attn @ V         -> [B, H, T, D] write (15MB)
```
That's 375MB of intermediate VRAM writes per layer, 3GB total for 8 layers. The [B,H,T,T] attention matrix (60MB) is written 6 times and only the final [B,H,T,D] output (15MB) is actually needed.

**Solution**: Fused Flash Attention kernel (forward pass only as first milestone):

**Algorithm** (simplified FlashAttention-2):
```
For each query block Br of Q:
  Initialize O = 0, l = 0, m = -inf
  For each key block Bc of K, V:
    S = Br @ Bc^T / sqrt(d)         # in shared memory
    S = tanh(S/cap) * cap            # softCap in-register
    S = masked_fill(S, mask, -inf)   # causal mask
    m_new = max(m, rowmax(S))
    P = exp(S - m_new)               # in shared memory
    l_new = l * exp(m - m_new) + rowsum(P)
    O = O * exp(m - m_new) / l_new + P @ Vc / l_new
    m = m_new, l = l_new
  Write O to output
```

**Why forward-only first**:
- Validates the shared-memory tiling strategy
- Tests causal mask + softCap integration
- Measures actual VRAM savings
- Backward is 3x more complex (needs to recompute attention for gradient)

**Kernel design**:
- Block sizes: Br=64, Bc=64 (tune for L4 shared memory: 48KB)
- Shared memory: Q block [Br, D], K block [Bc, D], S block [Br, Bc]
- For D=48 (our head dim): Q block = 64*48*4 = 12KB, K block = 12KB, S block = 64*64*4 = 16KB. Total ~40KB. Fits in L4 shared memory.
- Workgroup: one workgroup per (batch, head, query_block) tuple

**Files to modify**:
- `packages/helios/src/kernels.ts` — add `kernelFlashAttentionForward`
- `packages/helios/src/backend.ts` — add `flashAttention()` method
- `packages/core/src/interfaces.ts` — add to Backend interface
- `packages/model/src/gpt.ts` — use in forward pass (keep old path as fallback)

**Acceptance criteria**:
- Attention output matches non-fused path within 1e-3 (relaxed tolerance for different reduction order)
- The [B,H,T,T] attention weight tensor never exists in VRAM
- Per-layer VRAM usage drops by ~345MB (from 375MB to ~30MB)
- Forward pass time decreases measurably

**Risk**: High. This is the most complex kernel in the system. Shared memory tiling, online softmax, causal masking, and softCap all interact. Budget significant testing time.

---

### Sprint C Benchmark Gate

| Metric | Post-Sprint-B | Target |
|--------|---------------|--------|
| step_time | < 9.0s | < 7.0s (20%+ reduction) |
| ops/step | < 1200 | < 900 |
| output pool MB | 2925 | < 2000 (with FA forward) |
| max batch size | 24 | 32+ (with FA VRAM savings) |
| tok/s | ~1000 | > 1400 |

---

## Sprint D — Complete Flash Attention + Production Hardening

### D1. Flash Attention Backward

**What**: The backward pass for Flash Attention requires recomputing the attention matrix block-by-block (same tiling as forward) because we deliberately don't store it. This is the classic compute-vs-memory tradeoff that makes Flash Attention work.

**Algorithm sketch**:
```
For each query block Br:
  Reload O, l, m from forward
  dO = upstream gradient for this block
  For each key block Bc:
    Recompute S, P as in forward
    dV += P^T @ dO
    dP = dO @ V^T
    dS = dP * P * (1 - softCap_deriv)  # softCap backward
    dQ += dS @ K
    dK += dS^T @ Q
```

**Staging**:
1. First: no softCap in backward (simpler attention backward)
2. Then: add softCap derivative
3. Then: add dropout backward
4. Each stage validated against non-fused backward path

### D2. Flash Attention — Add Dropout

Dropout in fused attention requires deterministic mask regeneration in backward (same mask as forward). Options:
- Store the dropout mask (defeats some memory savings)
- Use deterministic RNG seeded by (step, layer, block_idx) to regenerate mask
- The second approach is what FlashAttention-2 uses

### D3. Activation Checkpointing

**What**: Instead of storing all intermediate activations for backward, recompute them during backward. Trade ~30% extra compute for ~50% less VRAM.

**Why now**: After Flash Attention, the remaining VRAM consumers are:
- Layer outputs (for residual backward)
- MLP intermediates (for GELU backward)
- LayerNorm inputs (for LN backward)

Checkpointing stores only layer boundaries and recomputes everything else.

**Implementation**:
- Wrap each transformer layer in a `checkpoint()` function
- On forward: run layer, keep only output, discard intermediates
- On backward: re-run forward to recreate intermediates, then run backward
- The tape needs a "recompute segment" concept

**Impact**: Enables doubling model size or batch size on same GPU.

### D4. Mixed Precision (BF16/FP16)

**What**: Store activations and parameters in half precision, compute in FP32 where needed.

**Prerequisites**:
- L4 supports BF16 (Ada Lovelace)
- Need to verify `f16Supported` flag in device info
- Loss scaling for FP16 (dynamic loss scaling with overflow detection)

**Staging**:
1. BF16 storage for activations only (parameters stay FP32)
2. BF16 matmul with FP32 accumulation
3. Full mixed precision with loss scaling

**Impact**: 2x memory reduction for activations, potential 2x matmul throughput if using tensor cores.

---

## Sprint E — Multi-GPU (Future)

Only after single-GPU efficiency is maximized.

### E1. Data Parallelism

- Each GPU gets a different micro-batch
- Forward + backward independently
- All-reduce gradients across GPUs
- Custom ring all-reduce (not NCCL)

### E2. Gradient Compression

- Quantize gradients to FP16 or INT8 before all-reduce
- Top-K sparsification for communication reduction
- Error feedback (accumulate compression residuals)

### E3. Pipeline Parallelism (if needed for very large models)

- Split model layers across GPUs
- Micro-batch pipelining to keep all GPUs busy
- Custom bubble-filling schedule

---

## Implementation Sequencing (Calendar View)

```
Week 1:  Tier 0 (guardrails) + A4 (MAX_PENDING_OPS) + A5 (pool rounding)
         |-- regression test script
         |-- kernel test harness
         |-- config changes

Week 2:  A1 (CE backward kernel) + A2 (vec4 softCap)
         |-- new SPIR-V kernels
         |-- backend integration
         |-- benchmark comparison

Week 3:  A3 (residual_dropout_add) + Sprint A benchmark gate
         |-- fused kernel + autograd op
         |-- model integration
         |-- full Sprint A benchmark

Week 4:  B1 (matmul_transposed)
         |-- matmul_nt and matmul_tn kernels
         |-- linear() op in autograd
         |-- model forward pass rewrite

Week 5:  B2 (add_inplace) + B3 (dead gradient elimination)
         |-- in-place grad accumulation
         |-- tape backward optimization
         |-- Sprint B benchmark gate
         |-- attempt batch=24

Week 6:  C1 + C2 (fused LayerNorm forward + backward)
         |-- Welford's algorithm kernel
         |-- replace multi-pass LN

Week 7:  C3 (matmul tile auto-tune)
         |-- benchmark suite
         |-- shape-aware kernel selection

Week 8-9: C4 (Flash Attention forward prototype)
         |-- tiled attention kernel
         |-- shared memory management
         |-- extensive numerical testing

Week 10: Sprint C benchmark gate + production training run
         |-- full benchmark comparison
         |-- long training run with all optimizations
```

---

## Metrics Dashboard

Track these across all sprints:

### Per-Step Metrics (already have)
- `step_time` (ms)
- `ops/step` (GPU dispatch count)
- `tok/s` (throughput)
- `loss`, `val_loss`
- `grad_norm`, `clip_coef`, `clip_pct`
- `timing_fwd_ms`, `timing_bwd_ms`, `timing_optim_ms`, `timing_flush_ms`

### New Metrics (add)
- `transpose_ops/step` — count of explicit transpose dispatches
- `backward_allocs` — new GPU allocations during backward
- `pool_reuse_rate` — % of acquireOutputRegion calls that hit pool vs allocate
- `peak_vram_mb` — high-water mark per step
- `flush_count/step` — number of compute graph flushes
- `ce_backward_mode` — "gpu" or "cpu_fallback" (verify we're on GPU path)

### Benchmark Suite (run weekly)
- 100-step regression test (both backends)
- Kernel correctness harness (all kernels)
- Matmul throughput by shape class
- Attention forward throughput (fused vs unfused)
- End-to-end step time on reference config

---

## Non-Goals

These are explicitly out of scope for this phase:

1. **Distributed training** — single-GPU efficiency first
2. **New model architectures** (MoE, SSM) — optimize the transformer we have
3. **Inference optimization** — training throughput is the priority
4. **New tokenizers** — BPE-4k is working fine
5. **Web dashboard features** — the dashboard works, training engine needs the focus
6. **New training domains** — chat training is the test case
7. **ONNX/PyTorch interop** — we build from scratch, that's the point

---

## Definition of Done (Phase 2 Complete)

Phase 2 is complete when:

- [ ] All Sprint A-C tasks implemented and passing regression tests
- [ ] Step time < 7s on current config (8L/384D/8H batch=20 block=512)
- [ ] Throughput > 1400 tok/s on L4
- [ ] Can train 12L/512D/8H batch=16+ without OOM (enabled by Flash Attention VRAM savings)
- [ ] Zero forced CPU readbacks during training step
- [ ] Kernel test harness covers all GPU kernels with CPU ref comparison
- [ ] All new kernels have vec4 variants where applicable
- [ ] 1000-step training run produces equivalent loss curve to baseline

The target is not parity with PyTorch/CUDA. The target is **3-5x improvement over our current baseline** while maintaining the from-scratch philosophy. That puts us at ~3000-4500 tok/s on L4, which is respectable for a pure TypeScript + Vulkan stack.
