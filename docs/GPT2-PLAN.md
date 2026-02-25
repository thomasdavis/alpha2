# Scaling Alpha to GPT-2: Single-GPU Optimization Roadmap

A concrete plan for training a GPT-2 124M parameter model on a single A100/H100 using the Alpha codebase. Every optimization is grounded in what Alpha already has, what's missing, and what to build.

---

## Target Architecture

| Parameter | GPT-2 124M | Alpha Default |
|-----------|-----------|---------------|
| Layers | 12 | 6 |
| Hidden dim | 768 | 256 |
| Heads | 12 | 8 |
| Head dim | 64 | 32 |
| Vocab size | 50,257 | 256 |
| Context length | 1,024 | 256 |
| FFN inner dim | 3,072 | 1,024 |
| Parameters | 124.4M | ~4.9M |
| **Scale factor** | **~25x** | — |

### Exact Parameter Breakdown

```
Token embedding (wte):     50,257 * 768           = 38,597,376
Position embedding (wpe):  1,024 * 768            =    786,432
LM head:                   tied to wte             =          0  (weight tying)

Per transformer block (x12):
  LN1 weight + bias:       2 * 768                =      1,536
  Q projection:            768 * 768 + 768         =    590,592
  K projection:            768 * 768 + 768         =    590,592
  V projection:            768 * 768 + 768         =    590,592
  Output projection:       768 * 768 + 768         =    590,592
  LN2 weight + bias:       2 * 768                =      1,536
  FFN up:                  768 * 3,072 + 3,072     =  2,362,368
  FFN down:                3,072 * 768 + 768       =  2,360,064
  Per block:                                       =  7,087,872

12 blocks:                 12 * 7,087,872          = 85,054,464
Final LN:                  2 * 768                 =      1,536

Total:                                             = 124,439,808
```

---

## Memory Budget (Batch=8, Seq=1024, fp32 Naive)

| Component | Formula | Size |
|-----------|---------|------|
| Weights | 124.4M * 4 bytes | 498 MB |
| Gradients | 124.4M * 4 bytes | 498 MB |
| AdamW m (first moment) | 124.4M * 4 bytes | 498 MB |
| AdamW v (second moment) | 124.4M * 4 bytes | 498 MB |
| **Optimizer total** | | **1.99 GB** |
| Activations per layer | ~680 MB (attention matrix dominates) | |
| Attention matrices alone | B\*H\*T\*T\*4 = 8\*12\*1024\*1024\*4 | 402 MB/layer |
| 12 layers activations | | **~8.2 GB** |
| Logits (B\*T\*V) | 8\*1024\*50257\*4 | 1.6 GB |
| **Naive total** | | **~12.3 GB** |

This barely fits on a 16 GB GPU and leaves no headroom. On an 80 GB A100 it fits but wastes bandwidth and runs slowly. Every optimization below attacks a specific slice of this budget.

---

## The Optimizations

### 1. Weight Tying (Embedding ↔ LM Head)

**What:** Share the token embedding matrix `wte` [50257, 768] with the LM head projection. The LM head computes `logits = hidden @ wte.T` instead of using a separate `lmHead` parameter.

**Savings:** 50,257 * 768 * 4 bytes = **154 MB** weights + 154 MB gradients + 308 MB optimizer states = **616 MB total**.

**What exists in Alpha:** `gpt.ts` has a separate `lmHead` Variable. `collectParams()` treats it independently.

**What to change:**

In `initGPT()`, don't create `lmHead` — set `params.lmHead = params.wte` (same Variable reference).

In `gptForward()`, the final projection already does `matmul(flat, transpose(lmHead, 0, 1))`. With weight tying, this becomes `matmul(flat, transpose(wte, 0, 1))` — no code change needed if lmHead points to wte.

In the backward pass, gradients from both the embedding lookup and the output projection accumulate into the same Variable's `.grad` automatically — the tape already handles multi-use Variables via gradient accumulation.

In `collectParams()`, skip `lmHead` if it's the same reference as `wte`, so checkpoints don't double-serialize it.

**Complexity:** Low. ~20 lines changed.

---

### 2. Fused QKV Projection

**What:** Instead of 3 separate matmuls for Q, K, V, compute all three in one matmul by concatenating the weight matrices.

**Current state in `gpt.ts` (lines 138-141):**
```typescript
const q = reshape(matmul(q3d, transpose(wq, 0, 1)), [B, T, C])
const k = reshape(matmul(q3d, transpose(wk, 0, 1)), [B, T, C])
const v = reshape(matmul(q3d, transpose(wv, 0, 1)), [B, T, C])
```
This is 3 transposes + 3 matmuls = 6 GPU dispatches.

**Optimization:** Replace `wq`, `wk`, `wv` (each [768, 768]) with a single `wqkv` [768, 2304]. One matmul produces all three, then slice:

```typescript
const qkv = matmul(q3d, transpose(wqkv, 0, 1))  // [B*T, 2304]
const q = slice(qkv, 0, C)       // [B*T, 768]
const k = slice(qkv, C, 2*C)     // [B*T, 768]
const v = slice(qkv, 2*C, 3*C)   // [B*T, 768]
```

**Savings:**
- 6 dispatches → 1 dispatch + 3 slices (slices are CPU-side, zero-copy)
- Better GPU utilization: one large matmul [B\*T, 768] @ [768, 2304] vs three smaller ones
- At B=8, T=1024: 8K × 768 × 2304 = 14.4 GFLOPs in one dispatch vs 3 × 4.8 = 14.4 split across 3. Same FLOPs but fewer kernel launches and better memory coalescing on the weight matrix.

**What to change:** `LayerParams.attn` replaces `{wq, wk, wv, wo}` with `{wqkv, wo}`. `initGPT()` initializes `wqkv` as [2304, 768]. Forward pass uses single matmul + slice. Backward for slice is just placing gradients into the correct columns of the fused gradient.

The autograd `slice` operation needs a backward that scatters the gradient back into the fused tensor. This requires a new `sliceBackward` op or using the existing approach of building separate gradients and concatenating.

**Complexity:** Medium. ~50 lines changed in model, ~30 in autograd for slice backward.

---

### 3. Flash Attention

**What:** Fuse the entire attention computation (Q@K^T, mask, softmax, @V) into a single GPU kernel that never materializes the T×T attention matrix.

**Current state:** Attention is 12+ separate GPU dispatches per layer:
```
matmul(Q, K^T)     → write [B, H, T, T] to VRAM    (402 MB at B=8)
scale               → read + write [B, H, T, T]
maskedFill          → read + write [B, H, T, T]
softmax             → read + write [B, H, T, T]
matmul(attn, V)     → read [B, H, T, T] + [B, H, T, d], write [B, H, T, d]
```

The T×T matrix is 402 MB per layer. Written once, read 4 times, then discarded. This is pure bandwidth waste.

**Flash Attention algorithm:**

Process Q in row-blocks of size B_r, K/V in column-blocks of size B_c. Everything fits in shared memory:

```
For each Q block (B_r rows of Q):
  Initialize: O = 0, m = -inf, l = 0

  For each K,V block (B_c rows of K and V):
    Load Q_tile [B_r, d], K_tile [B_c, d], V_tile [B_c, d] into shared memory
    Compute S = Q_tile @ K_tile^T                     [B_r, B_c] in shared memory
    Apply causal mask: S[i,j] = -inf if global_j > global_i
    m_new = max(m, rowmax(S))
    P = exp(S - m_new)                                [B_r, B_c]
    l_new = exp(m - m_new) * l + rowsum(P)
    O = (l/l_new) * exp(m - m_new) * O + (1/l_new) * P @ V_tile
    m = m_new, l = l_new

  Store O to output [B_r, d]
  Store m, l for backward (2 floats per row — tiny)
```

**Tile sizes for Vulkan (shared memory = 48KB):**

```
d = 64 (head dim, fixed)
B_r = B_c = 32

Shared memory per tile:
  Q_tile:   32 * 64 * 4 = 8,192 bytes
  K_tile:   32 * 64 * 4 = 8,192 bytes
  V_tile:   32 * 64 * 4 = 8,192 bytes
  S_tile:   32 * 32 * 4 = 4,096 bytes
  O_accum:  32 * 64 * 4 = 8,192 bytes
  m, l:     32 * 4 * 2  =   256 bytes
  Total:                 = 37,120 bytes ✓ (fits in 48KB)
```

**Dispatch:**
```
Workgroup: 256 threads (covers one B_r×B_c tile computation)
Grid: (T/B_r, H, B) = (32, 12, 8) = 3,072 workgroups
```

**Memory savings:**
```
Standard:  402 MB/layer * 12 layers = 4.8 GB  (attention matrices)
Flash:     B*H*T*d*4 + 2*B*H*T*4 = 25.2 + 0.8 MB/layer * 12 = 312 MB
Saved:     4.5 GB
```

**Backward pass:** Flash Attention backward recomputes Q@K^T on the fly from stored Q, K, V (which are already kept for the linear projection backward). It only needs the saved m, l statistics (2 floats per row per head — negligible). No extra memory beyond what's already stored.

**SPIR-V implementation in Alpha:**

Add a new kernel generator in `kernels.ts`:

```typescript
function kernelFlashAttention(B_r: number, B_c: number, headDim: number): Uint32Array {
  const b = new SpirVBuilder();
  // ... workgroup setup, shared memory declarations
  // Main loop: iterate over K,V blocks
  //   Load tiles cooperatively
  //   Compute S = Q @ K^T in registers
  //   Online softmax update (m, l, O rescaling)
  //   Accumulate O += P @ V
  // Store O, m, l
  return b.finalize();
}
```

The SPIR-V assembler already supports shared memory (`Workgroup` storage class), barriers (`ControlBarrier`), and the necessary math ops (exp via GLSL.std.450). The main new requirement is register-level tiling within the kernel — each thread computes a subset of the B_r×B_c score tile.

**Integration with autograd:** Replace the current 5-op attention sequence with a single custom tape entry. The backward closure calls the flash attention backward kernel (also a single dispatch).

In `gpt.ts`, the maskedFill/softmax/matmul chain (lines 147-172) becomes:

```typescript
const attnOut = flashAttention(ctx, qH, kH, vH, headDim);  // single op
```

**Complexity:** High. ~300 lines of kernel code, ~100 lines of integration. This is the single highest-impact optimization.

---

### 4. Mixed Precision (fp16 Compute, fp32 Master Weights)

**What:** Store activations and gradients in fp16 (half the memory), compute matmuls in fp16 (faster on GPU hardware), keep master weights and optimizer states in fp32 (for numerical stability).

**Current state:** Alpha has fp16 kernel variants in `kernels.ts` (lines 287-428: `kernelBinaryOpF16`, `kernelUnaryOpF16`) and the C addon probes fp16 support (`shaderFloat16` + `storageBuffer16BitAccess`). But the training loop, autograd, and model all assume fp32.

**What changes:**

**Forward pass:** Cast fp32 weights to fp16 before matmuls. All intermediate activations are fp16. LayerNorm and softmax internal reductions stay fp32 (compute in fp32, cast output to fp16).

```
fp32 master weights ──cast──> fp16 weights for forward
fp16 activations throughout forward pass
Softmax: cast to fp32 internally, output fp16
LayerNorm: accumulate mean/var in fp32, output fp16
```

**Backward pass:** Gradients flow in fp16. The tape stores fp16 activations (half the memory for backward closures).

**Optimizer step:** Cast fp16 gradients to fp32, update fp32 master weights. The fused `adamw_step` kernel already operates on fp32 buffers — just need to add fp16→fp32 gradient cast before the step.

**Loss scaling (required for fp16, not bf16):**

Small gradients underflow fp16 (minimum positive value ~5.96e-8). Solution: scale the loss by a large factor before backward, then unscale gradients before the optimizer step.

```typescript
let lossScale = 32768;  // 2^15
const GROWTH_INTERVAL = 2000;
let goodSteps = 0;

// In training loop:
const scaledLoss = scale(loss, lossScale);
tape.backward(scaledLoss, backend, releaseTensor);

// Check for inf/nan in gradients (single GPU reduction)
const hasInf = checkGradOverflow(grads, backend);

if (hasInf) {
  lossScale *= 0.5;  // Halve scale
  goodSteps = 0;
  continue;  // Skip this step
}

// Unscale gradients
for (const [name, grad] of grads) {
  grads.set(name, backend.scale(grad, 1.0 / lossScale));
}

optimizer.step(params, grads);
goodSteps++;
if (goodSteps >= GROWTH_INTERVAL) {
  lossScale *= 2;  // Try doubling
  goodSteps = 0;
}
```

**Memory savings:**

```
                          fp32 only    Mixed precision
Weights (master, fp32):   498 MB       498 MB
Weights (fp16 copy):      0            249 MB
Gradients:                498 MB       249 MB (fp16)
Optimizer states:         996 MB       996 MB (must stay fp32)
Activations (12 layers):  8.2 GB       4.1 GB (fp16)
Logits (B*T*V):           1.6 GB       0.8 GB (fp16)
─────────────────────────────────────────────────
Total:                    ~11.8 GB     ~6.9 GB
Savings:                               ~4.9 GB
```

**Speed benefit:** fp16 matmuls are 2x faster on hardware with fp16 ALUs (all modern GPUs). With `VK_KHR_cooperative_matrix` (tensor cores via Vulkan), fp16 matmuls can be 8-16x faster than fp32.

**What to build:**

1. `castToF16` / `castToF32` GPU kernels in `kernels.ts` (trivial: load f32, store f16 or vice versa)
2. fp16 matmul kernel variant (or cooperative matrix kernel if available)
3. Loss scaling logic in `trainer.ts` (~30 lines)
4. fp16-aware `layerNorm` and `softmax` kernels (fp32 internal accumulation, fp16 I/O)
5. Dtype tracking in the compute graph — some tensors are fp16, some fp32

**Complexity:** Medium-high. ~200 lines across kernels, backend, trainer. The hardest part is managing the fp16/fp32 boundary cleanly through the autograd tape.

---

### 5. Activation Checkpointing

**What:** Don't store all intermediate activations for backward. Instead, store only the input to each transformer block. During backward, recompute that block's forward pass to regenerate the intermediates.

**Current state:** Alpha's tape stores every intermediate Variable created during forward. For 12 layers, this means all attention matrices, FFN intermediates, layer norm outputs, etc.

**How it works:**

```
Standard forward (all activations stored):
  Layer 0: store LN1, Q, K, V, scores, attn_weights, attn_out, LN2, FFN_h, FFN_out
  Layer 1: store same...
  ...
  Layer 11: store same...
  Memory: 12 * ~680 MB = 8.2 GB

Checkpointed forward:
  Layer 0: store ONLY x_0 (input to layer 0) = 25 MB
  Layer 1: store ONLY x_1 = 25 MB
  ...
  Layer 11: store ONLY x_11 = 25 MB
  Memory: 12 * 25 MB = 300 MB

During backward of layer i:
  Recompute layer i's forward from x_i → regenerate all intermediates
  Backward through layer i using those intermediates
  Free intermediates
  Peak: 300 MB + 680 MB (one layer) = 980 MB
```

**Savings:** 8.2 GB → ~1 GB. **Cost:** ~33% more FLOPs (each layer's forward is computed twice — once in forward, once in backward).

**Implementation in Alpha:**

The tape needs a `checkpoint` mechanism. During forward, instead of recording every op individually, record a single "recompute" entry per block:

```typescript
function checkpointedForward(
  ctx: { tape: Tape; backend: Backend },
  blockFn: (input: Variable) => Variable,
  input: Variable
): Variable {
  // Run forward WITHOUT recording to tape
  const detachedTape = new Tape();
  const detachedCtx = { tape: detachedTape, backend: ctx.backend };
  const output = blockFn(detachedCtx, input);

  // Record a single entry that recomputes on backward
  const result = new Variable(backend.clone(output.data), true);
  ctx.tape.record({
    output: result,
    inputs: [input],
    backward(outGrad, B) {
      // Recompute forward
      const recomputeTape = new Tape();
      const recomputeCtx = { tape: recomputeTape, backend: B };
      const recomputed = blockFn(recomputeCtx, input);
      // Backward through recomputed graph
      recomputed.grad = outGrad;
      recomputeTape.backward(recomputed, B, releaseTensor);
      return [input.grad!];
    }
  });

  detachedTape.clear();  // Don't keep intermediates
  return result;
}
```

Then in `gptForward()`, wrap each transformer block:

```typescript
for (const layer of params.layers) {
  x = checkpointedForward(ctx, (c, inp) => transformerBlock(c, inp, layer, config), x);
}
```

**Complexity:** Medium. ~80 lines for the checkpoint mechanism, ~20 lines to integrate into the forward pass.

---

### 6. Fused Cross-Entropy (Chunked Softmax)

**What:** The logits tensor before cross-entropy is [B\*T, V] = [8192, 50257] = 1.6 GB in fp32. Standard cross-entropy materializes the full softmax output (another 1.6 GB). Fused cross-entropy computes the loss and its gradient without ever materializing the full probability distribution.

**Current state in `ops.ts`:** `crossEntropy` calls `backend.softmax(logits)` which allocates a full [B\*T, V] tensor, then indexes into it.

**Fused approach:**

A single GPU kernel that:
1. Reads one row of logits [V=50257]
2. Computes max (for numerical stability)
3. Computes log-sum-exp
4. Computes loss = -(logit[target] - max - log_sum_exp)
5. Computes gradient = softmax(logits) - one_hot(target)
6. Writes loss (1 scalar) and gradient row [V] directly

The key insight: you never need the full [B\*T, V] softmax output. Each row is independent and can be processed by one workgroup.

**Even more memory-efficient (chunked logits):**

Instead of computing all V=50257 logits at once, compute them in chunks:

```
For chunk_start = 0 to V step CHUNK_SIZE:
  chunk_logits = hidden @ lmHead_weight[chunk_start : chunk_start + CHUNK_SIZE]^T
  // chunk_logits is [B*T, CHUNK_SIZE] — much smaller
  accumulate online softmax statistics (max, sum_exp)
  if target falls in this chunk, record the target logit
```

With CHUNK_SIZE=4096: memory per chunk = 8192 * 4096 * 4 = 128 MB instead of 1.6 GB.

**Savings:** 1.6 GB (softmax output) eliminated. Logits can also be chunked for another ~1.5 GB savings.

**Implementation:** New kernel `kernelFusedCrossEntropy` in `kernels.ts`. Each workgroup handles one row (one token position). The workgroup cooperatively loads the logit row, computes max, sum(exp), loss, and gradient row.

**Complexity:** Medium. ~150 lines of kernel code, ~50 lines to integrate as a custom autograd op.

---

### 7. GPU Embedding Backward

**What:** The embedding backward pass currently runs on CPU with a scatter loop. For V=50257 and batch 8 × seq 1024 = 8192 tokens, this is 8192 scatter-add operations into a [50257, 768] gradient matrix — slow on CPU.

**Current state in `ops.ts` (lines 161-176):**
```typescript
for (let i = 0; i < numIndices; i++) {
  const idx = idxArr[i];
  for (let d = 0; d < dim; d++) {
    gw[idx * dim + d] += gData[i * dim + d];  // CPU scatter-add
  }
}
```

**Fix:** Write a `kernelEmbeddingBackward` GPU kernel:

```
Bindings: grad_output [numTokens, dim], indices [numTokens], grad_weight [V, dim]
Grid: (numTokens, 1, 1)
Per thread: atomicAdd grad_weight[indices[gid], d] += grad_output[gid, d]
```

Vulkan supports `atomicAdd` on storage buffers (fp32). Each thread handles one token's gradient scatter. Atomic contention is low because tokens rarely map to the same vocab index in a single batch.

**Alternative (sort-based, no atomics):**
1. Sort (index, gradient_row) pairs by index
2. Segmented reduction: sum gradient rows with same index
3. Scatter results to grad_weight

This avoids atomics entirely but requires a GPU sort. For V=50257 and 8192 tokens, the sort approach may be overkill — atomicAdd is fine.

**Complexity:** Low. ~60 lines of kernel code, ~10 lines to replace the CPU loop.

---

### 8. Fused GELU Backward (Already Partially Exists)

**Current state:** `kernels.ts` has `kernelGeluBackward` (line 1025), and `backend.ts` exposes `geluBackward()`. The autograd `gelu` op in `ops.ts` (line 142) checks for `B.geluBackward`:

```typescript
if ((B as any).geluBackward) {
  return [(B as any).geluBackward(aData, g)];
}
```

**Issue:** The check uses `(B as any)` — it's not part of the official `Backend` interface. The optional methods `geluBackward?`, `reluBackward?`, `layerNormBackward?` are defined in the interface but the autograd code accesses them via `any` cast.

**Fix:** Clean up the type casts to use the optional interface methods properly. Verify the GPU kernel is actually being dispatched (add a log on first use). This is mostly a wiring issue, not a missing feature.

**Complexity:** Very low. ~10 lines.

---

### 9. Larger Matmul Tiles

**Current state:** `kernels.ts` uses 16×16 tiles with 256 threads per workgroup and 2×16×16×4 = 2 KB shared memory.

**Problem at GPT-2 scale:** The main matmuls are large:
- QKV projection: [8192, 768] @ [768, 2304] — 28.9 GFLOPs
- FFN up: [8192, 768] @ [768, 3072] — 38.7 GFLOPs
- FFN down: [8192, 3072] @ [3072, 768] — 38.7 GFLOPs

At 16×16 tiles, the GPU dispatches many small workgroups. Larger tiles improve compute-to-memory ratio.

**Optimization: 32×32 or 64×64 tiles with register blocking:**

```
Tile size: 64×64 output tile
Threads per workgroup: 256 (16×16 thread grid)
Each thread computes a 4×4 sub-tile in registers

Shared memory per tile:
  A tile: 64 × K_TILE × 4 bytes
  B tile: K_TILE × 64 × 4 bytes
  With K_TILE = 16: (64*16 + 16*64) * 4 = 8,192 bytes (8 KB) — fits easily

Register usage per thread:
  4×4 accumulators = 16 registers
  4 A values + 4 B values = 8 registers
  Total: ~24 registers per thread
```

This increases arithmetic intensity: each shared memory load serves 4×4 = 16 multiply-accumulate operations per thread instead of 1. The compute-to-memory ratio improves by ~16×.

**Implementation:** New `kernelMatmulTiled64` in `kernels.ts`. The SpirVBuilder already supports shared memory arrays and barriers. The main work is the register-level tiling logic within the SPIR-V emission.

**Complexity:** Medium. ~200 lines of kernel code. Must be carefully tested against the reference CPU backend.

---

### 10. Cooperative Matrix (Tensor Cores via Vulkan)

**What:** `VK_KHR_cooperative_matrix` exposes hardware matrix-multiply-accumulate (MMA) units (tensor cores on NVIDIA, matrix cores on AMD) through Vulkan. This is the Vulkan equivalent of CUDA's `wmma` / `mma.sync` instructions.

**Performance impact:** fp16 matmul goes from ~30 TFLOPS (shader fp16 arithmetic) to ~280+ TFLOPS (tensor cores) on A100. This is a **10× speedup** for matmuls, which dominate training FLOPs.

**Requirements:**
- Vulkan 1.1+ with `VK_KHR_cooperative_matrix` extension
- NVIDIA Turing+ (RTX 2000 series or newer), AMD RDNA3+
- SPIR-V `SPV_KHR_cooperative_matrix` capability

**Implementation in Alpha's SPIR-V assembler:**

Add cooperative matrix types and operations to `spirv.ts`:

```
OpTypeCooperativeMatrixKHR — declare matrix type (element type, scope, rows, cols, use)
OpCooperativeMatrixLoadKHR — load from memory to cooperative matrix
OpCooperativeMatrixStoreKHR — store cooperative matrix to memory
OpCooperativeMatrixMulAddKHR — fused multiply-add: C = A * B + C
```

Typical tile: 16×16 fp16 multiply, fp32 accumulate. The subgroup (warp on NVIDIA, wave on AMD) cooperatively holds the matrix fragments.

**Detection in helios_vk.c:** Already probes device features via `vkGetPhysicalDeviceFeatures2`. Add a check for `VK_KHR_cooperative_matrix` in the extension list and query supported matrix shapes via `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`.

**Complexity:** High. Requires SPIR-V assembler extensions, new kernel variants, and runtime feature detection. But the payoff is enormous — this single optimization can make training 3-5× faster overall.

---

### 11. Gradient Accumulation

**What:** Compute forward+backward on multiple micro-batches before doing an optimizer step. This allows a larger effective batch size without requiring more memory.

**Why it matters:** GPT-2 was trained with effective batch size 512 × 1024 = 524K tokens/step. On a single GPU with 80 GB, micro-batch of 8-16 fits. To match the original dynamics, accumulate 32-64 micro-steps.

In practice, effective batch sizes of 32-64 sequences (32K-64K tokens) work well for 124M models with appropriate LR adjustment.

**Current state:** Alpha's `trainer.ts` does one forward-backward-update per step. No accumulation.

**Implementation:**

```typescript
const accumSteps = config.accumSteps ?? 1;

for (let microStep = 0; microStep < accumSteps; microStep++) {
  const batch = trainLoader.nextBatch();
  const { loss } = gptForward(modelConfig, params, backend, tape, inputs, targets);

  // Scale loss by accumSteps so gradients average correctly
  const scaledLoss = scale(ctx, loss, 1.0 / accumSteps);
  tape.backward(scaledLoss, backend, releaseTensor);

  // Gradients accumulate in param.grad across micro-steps
  tape.clear(releaseTensor);
}

// Now do the optimizer step on accumulated gradients
optimizer.step(paramDataMap, gradMap);
```

**Key detail:** Gradients must be accumulated (added) across micro-steps, then the optimizer step is called once. The loss scaling by `1/accumSteps` ensures the gradient magnitude matches a single large-batch step.

**What to change:** Add `accumSteps` to TrainConfig. Wrap the forward-backward section of trainer.ts in a micro-step loop. Move optimizer.step() and gradient zeroing outside the loop.

**Memory impact:** None — activations from each micro-step are freed before the next. Only gradients persist across micro-steps (same memory as single-step).

**Complexity:** Low. ~30 lines in trainer.ts, ~5 lines in types.

---

### 12. Async Data Loading

**Current state:** `DataLoader.nextBatch()` in `data.ts` is synchronous — it blocks the CPU while building the batch. Since batches are constructed from an in-memory Int32Array (random index + memcpy), this is fast (~microseconds). But uploading the batch to the GPU is not overlapped with compute.

**Optimization:** Prefetch the next batch and upload it to a GPU staging buffer while the current step's forward-backward is running.

```typescript
class AsyncDataLoader {
  private nextBatchPromise: Promise<DataBatch> | null = null;

  prefetch(): void {
    this.nextBatchPromise = new Promise(resolve => {
      const batch = this.loader.nextBatch();
      // Upload to GPU staging buffer (uses transfer queue, overlaps with compute)
      this.backend.uploadBatchAsync(batch.inputs, batch.targets);
      resolve(batch);
    });
  }

  async getBatch(): Promise<DataBatch> {
    const batch = await this.nextBatchPromise!;
    this.prefetch();  // Start next immediately
    return batch;
  }
}
```

The C addon already supports a dedicated transfer queue (`hasAsyncTransfer` flag). Uploads on the transfer queue run concurrently with compute dispatches on the compute queue.

**Impact:** Negligible for in-memory datasets, but important for memory-mapped datasets that page from disk.

**Complexity:** Low. ~40 lines.

---

### 13. Memory-Mapped Binary Dataset

**Current state:** `data.ts` loads the entire tokenized dataset into a single `Int32Array` in memory. For OpenWebText (~9B tokens), this is 9B × 4 bytes = **36 GB of RAM**. This doesn't fit in memory on most machines.

**Solution:** Pre-tokenize to a flat binary file, then memory-map it:

```
File format: train.bin
Contents: [token_0: uint16][token_1: uint16]...[token_N: uint16]
Size: 9B tokens × 2 bytes = 18 GB (uint16 is sufficient for vocab < 65536)
```

The DataLoader uses `mmap` to map the file into the process's address space. The OS pages data in on demand and evicts under memory pressure. The full 18 GB file does NOT need to fit in RAM.

```typescript
import { openSync, fstatSync } from 'fs';

// Memory-map the token file
const fd = openSync('train.bin', 'r');
const size = fstatSync(fd).size;
const buffer = new SharedArrayBuffer(size);  // or use native mmap addon
const tokens = new Uint16Array(buffer);

// DataLoader samples random positions — OS handles page faults
```

**Pre-tokenization CLI command:**

```bash
alpha tokenize --input data/openwebtext.txt --output data/train.bin --tokenizer bpe --vocabSize 50257
```

**Complexity:** Medium. ~100 lines for the binary format + mmap DataLoader, ~50 lines for the tokenize CLI command.

---

### 14. Weight Decay Exclusions

**Current state:** Alpha's AdamW applies weight decay uniformly to all parameters.

**Problem:** Weight decay on biases and LayerNorm parameters hurts training. These should be excluded.

**Implementation:**

In the optimizer, split parameters into two groups:

```typescript
const decayParams = new Map<string, TensorData>();
const noDecayParams = new Map<string, TensorData>();

for (const [name, param] of params) {
  if (name.includes('bias') || name.includes('ln') || name.includes('lnF')) {
    noDecayParams.set(name, param);
  } else {
    decayParams.set(name, param);
  }
}
```

Then apply weight decay only to `decayParams`. The fused `adamw_step` kernel takes `weightDecay` as a push constant — just pass 0 for no-decay params.

**Complexity:** Low. ~20 lines in optimizers.ts.

---

### 15. Fused Residual + LayerNorm Kernel

**What:** Combine the residual add (`x = x + sublayer_output`) and the subsequent LayerNorm into a single kernel. This saves one full read-write pass over [B, T, 768].

**Current state:** These are two separate dispatches in every transformer block (twice per block — once before attention, once before MLP).

**Savings per kernel invocation:** 2 × B\*T\*d × 4 bytes = 2 × 25.2 MB = 50.4 MB bandwidth saved. Over 12 layers × 2 instances = 24 invocations = **1.2 GB bandwidth saved per step**.

**Implementation:** Single kernel that:
1. Reads `x` and `sublayer_output` from two input buffers
2. Computes `x = x + sublayer_output` in registers
3. Computes mean and variance (shared memory reduction)
4. Normalizes, scales by weight, shifts by bias
5. Writes both the updated `x` (for the next residual) and the normalized output

**Complexity:** Medium. ~120 lines of kernel code.

---

## Optimization Priority Order

Ranked by impact-to-effort ratio for getting to GPT-2 quality on a single GPU:

| Priority | Optimization | Memory Saved | Speed Impact | Effort |
|----------|-------------|-------------|-------------|--------|
| **1** | Flash Attention | 4.5 GB | 1.2-1.5× | High |
| **2** | Mixed Precision (fp16) | 4.9 GB | 1.5-2× | Med-High |
| **3** | Activation Checkpointing | 7.2 GB | 0.75× (more compute) | Medium |
| **4** | Fused Cross-Entropy | 3.1 GB | 1.1× | Medium |
| **5** | Weight Tying | 616 MB | — | Low |
| **6** | Fused QKV | — | 1.05× | Medium |
| **7** | GPU Embedding Backward | — | 1.1× | Low |
| **8** | Gradient Accumulation | enables training | — | Low |
| **9** | Larger Matmul Tiles | — | 1.3-1.5× | Medium |
| **10** | Weight Decay Exclusions | — | better convergence | Low |
| **11** | Fused Residual+LN | — | 1.05× | Medium |
| **12** | Async Data Loading | — | 1.02× | Low |
| **13** | Memory-Mapped Dataset | RAM savings | — | Medium |
| **14** | Cooperative Matrix | — | 3-5× matmuls | High |
| **15** | Clean up GELU backward wiring | — | 1.02× | Very Low |

### Minimum Viable Set

To train GPT-2 124M on a single 80 GB A100 without running out of memory:

**Must have:** Weight tying (#5) + Gradient accumulation (#8) + Weight decay exclusions (#10) + Clean GELU backward (#15)

These four are all low-effort and get you from "doesn't fit" to "runs but slow."

**Should have:** Flash Attention (#1) + Fused Cross-Entropy (#4) + GPU Embedding Backward (#7)

These three eliminate the biggest memory and performance bottlenecks. With these, training is practical.

**Nice to have:** Mixed precision (#2) + Activation Checkpointing (#3) + Larger tiles (#9)

These make training fast and memory-efficient enough to increase batch size or run on smaller GPUs.

---

## Training Configuration

### Recommended Hyperparameters

```typescript
const modelConfig: ModelConfig = {
  vocabSize: 50257,
  blockSize: 1024,
  nLayer: 12,
  nEmbd: 768,
  nHead: 12,
  dropout: 0.0,        // No dropout (modern practice for pre-training)
};

const trainConfig: TrainConfig = {
  iters: 300000,        // ~20B tokens at batch 64
  batchSize: 8,         // Micro-batch (actual GPU batch)
  accumSteps: 8,        // Effective batch = 64 sequences = 65K tokens/step
  lr: 6e-4,             // Peak LR (Chinchilla-optimal for 124M)
  beta1: 0.9,
  beta2: 0.95,          // Lower than default 0.999 — better for LLM pre-training
  eps: 1e-8,
  weightDecay: 0.1,     // Decoupled, higher than Alpha's default 0.01
  gradClip: 1.0,
  evalInterval: 1000,
  evalIters: 50,
  seed: 42,
  backend: "helios",
  tokenizer: "bpe",
  optimizer: "adamw",
};
```

### Learning Rate Schedule

```
Warmup:    2,000 steps (linear from 0 to 6e-4)
Decay:     Cosine from 6e-4 to 6e-5 (min_lr = 0.1 * peak_lr)
```

Alpha's current schedule (100-step warmup, cosine to 0.5× peak) should be updated to use 2,000-step warmup and decay to 0.1× peak for GPT-2 scale.

### Dataset

OpenWebText (~38 GB text, ~9B tokens with GPT-2 BPE tokenizer). Pre-tokenize to binary format. Train for 1-2 epochs.

### Expected Convergence

```
Step 0:       loss ~10.8  (random, -ln(1/50257))
Step 1K:      loss ~6.5   (learning token frequencies)
Step 5K:      loss ~4.5   (common words, basic grammar)
Step 20K:     loss ~3.8   (coherent short phrases)
Step 50K:     loss ~3.3   (grammatically correct sentences)
Step 100K:    loss ~3.1   (topically coherent paragraphs)
Step 200K:    loss ~3.0   (approaching GPT-2 124M quality)
Step 300K:    loss ~2.9   (diminishing returns)
```

Target validation perplexity: ~20-25 on held-out WebText.

### Expected Training Time

| GPU | MFU (realistic) | Steps/sec | Time to 300K steps | Cost |
|-----|-----------------|-----------|-------------------|------|
| A100 80GB (Vulkan, early) | 20-30% | 1.2-1.8 | 46-69 hrs | $51-$76 |
| A100 80GB (Vulkan, optimized) | 35-45% | 2.1-2.7 | 31-40 hrs | $34-$44 |
| H100 80GB (Vulkan, optimized) | 35-45% | 5.5-7.0 | 12-15 hrs | $30-$37 |
| H100 80GB (with coop matrix) | 45-55% | 7.5-9.5 | 9-11 hrs | $22-$27 |

FLOPs per step: `6 * 124.4M * 64 * 1024 ≈ 49 TFLOPs` (at effective batch 64).

---

## New Domain Definition

Add a `gpt2` domain to `@alpha/core/domains.ts`:

```typescript
{
  id: "gpt2",
  displayName: "GPT-2 124M",
  tokenizer: "bpe",
  samplePrompts: [
    "The meaning of life is",
    "In a shocking finding, scientists discovered",
    "Once upon a time, in a land far away,",
    "The president of the United States announced",
  ],
  modelDefaults: {
    vocabSize: 50257,
    blockSize: 1024,
    nLayer: 12,
    nEmbd: 768,
    nHead: 12,
    dropout: 0.0,
  },
  trainDefaults: {
    iters: 300000,
    batchSize: 8,
    lr: 6e-4,
    beta1: 0.9,
    beta2: 0.95,
    weightDecay: 0.1,
    gradClip: 1.0,
    evalInterval: 1000,
    evalIters: 50,
    backend: "helios",
  },
}
```

---

## Summary

Training a GPT-2 124M model on a single GPU with the Alpha codebase is feasible. The architecture is already correct — the transformer, autograd, and GPU backend all work. The gap is **scale optimization**: the current system is tuned for 5M-parameter models and needs specific optimizations to handle 25× more parameters efficiently.

The three highest-impact changes are:
1. **Flash Attention** — eliminates the O(T^2) memory bottleneck
2. **Mixed precision** — halves activation memory and doubles matmul speed
3. **Activation checkpointing** — trades 33% more compute for 8× less activation memory

With all three, GPT-2 124M trains comfortably on a single A100 in ~1-2 days at a cost of ~$30-50. Without them, it barely fits in memory and runs 3-5× slower.

The philosophy of Alpha — understanding everything, writing everything from scratch — makes these optimizations more work than calling `torch.compile()`. But it also means complete control over every byte of GPU memory and every microsecond of kernel execution. That's the trade-off, and it's the whole point.
