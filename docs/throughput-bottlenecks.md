# Throughput Bottlenecks: 148 tok/s on L4 (should be ~2000+)

**Model**: 307M params, 21L 1024d 16h swiglu, batch=1x4, block=512, helios backend
**GPU**: NVIDIA L4 (30.3 TFLOPS f32, 120 TFLOPS TF32 tensor cores, 24GB)
**Current**: ~148 tok/s (~14s/step, 9203 GPU dispatches/step)
**MFU**: ~0.8% — should be 15-30% for a custom Vulkan backend

The bottleneck is NOT compute. It's the dispatch infrastructure serializing CPU and GPU.

---

## 1. Serial Command Buffer Execution (2-3x fix)

**The single biggest problem.**

`helios_vk.c` has ONE command buffer and ONE descriptor pool. Every flush cycle:

```
batchBegin() → waitTimelineValue(lastDispatchTimeline)  // BLOCKS until GPU done
             → vkResetDescriptorPool()
             → vkResetCommandBuffer()
             → vkBeginCommandBuffer()
... record 256 dispatches ...
batchEnd()   → vkEndCommandBuffer()
             → vkQueueSubmit()
```

The CPU **cannot record the next batch until the GPU finishes the current one**. This means:

```
CPU:  [record 256 ops]  [idle........]  [record 256 ops]  [idle........]
GPU:  [idle........]     [execute 256]   [idle........]     [execute 256]
```

With 9203 ops / 256 per batch = **36 full GPU syncs per step**. Each sync drains the entire pipeline.

### Fix: Double-buffer (or triple-buffer) command buffers + descriptor pools

Allocate 2-3 of each. While GPU executes batch N using pool A, CPU records batch N+1 into pool B. Swap on flush. Only block when wrapping around.

```
CPU:  [record A]  [record B]  [record A]  [record B]  ...
GPU:             [execute A]  [execute B]  [execute A]  ...
```

**Files**: `helios_vk.c` — change static `batchCmdBuf`/`persistentDescPool` to arrays, add ring index

---

## 2. MAX_PENDING_OPS = 256 Causes 36 Flushes/Step (1.3-1.5x fix)

**`backend.ts:457`** — the compute graph auto-flushes every 256 ops because the descriptor pool's `maxSets = 256`.

9203 ops / 256 = 36 flushes. Each flush = 1 vkQueueSubmit + 1 full sync (see above).

### Fix: Increase to 2048-4096

- Bump `maxSets` in descriptor pool to 4096
- Bump `MAX_PENDING_OPS` to match
- Reduces flushes from 36 → 3-5 per step
- Each submit amortizes the fixed sync overhead over 8-16x more work

**Files**: `helios_vk.c` (descriptor pool size), `backend.ts` (MAX_PENDING_OPS constant)

---

## 3. Per-Dispatch Descriptor Set Allocation (1.2-1.5x fix)

Every `batchDispatch` call (9203/step):
1. `vkAllocateDescriptorSets()` — allocate fresh set from pool
2. `vkUpdateDescriptorSets()` — write buffer bindings

At ~2-3us per alloc+update: **~27ms/step** of pure descriptor overhead.

### Fix: Push descriptors

Use `VK_KHR_push_descriptor` — `vkCmdPushDescriptorSetKHR()` writes descriptors directly into the command buffer, no allocation needed. L4 supports this extension.

Alternative: pre-allocate descriptor sets and reuse via a ring buffer indexed by pipeline+buffer-signature hash.

**Files**: `helios_vk.c` — enable extension, replace allocate+update with push descriptor call

---

## 4. N-API Overhead: 9203 JS→C Crossings/Step (1.2-1.5x fix)

Each `batchDispatch` is a full N-API function call that:
- Parses 7 JS arguments
- Iterates a JS array to extract buffer handles (`napi_get_element` loop)
- Parses a TypedArray for write mask
- Does O(N*M) barrier tracking

Total per-dispatch: ~3-5us. At 9203 ops: **~28-46ms/step**.

### Fix: Batch the dispatch calls

Instead of 9203 individual N-API calls, pack all dispatches into a single call:

```typescript
// Current: 9203 individual calls
for (const op of pending) vk.batchDispatch(pipeline, bufs, gx, gy, gz, pc, writeMask);

// Proposed: 1 call with packed dispatch array
vk.batchDispatchMany(dispatchArray);  // ArrayBuffer with all dispatch descriptors
```

Pass a flat `ArrayBuffer` containing `[pipelineId, bufCount, buf0, buf1, ..., gX, gY, gZ, pcOffset, pcSize, writeMask]` per dispatch. Parse it in a single C loop with zero JS object access.

**Files**: `helios_vk.c` (add `napi_batchDispatchMany`), `backend.ts` (pack dispatches into ArrayBuffer)

---

## 5. Upload Blocks Entire GPU Pipeline (1.1-1.2x fix)

`helios_vk.c` line 1481:
```c
waitTimelineValue(lastDispatchTimeline);  // FULL SYNC before any upload
```

Every `uploadBuffer` for device-local memory drains the entire GPU pipeline. Happens when token indices, position embeddings get uploaded each microstep.

### Fix: Async upload via staging ring

Use a ring of staging buffers tracked by their own timeline values. Record the staging→device copy as a command in the batch command buffer instead of blocking.

```c
// Instead of: wait → map staging → copy → unmap
// Do: map free staging buffer → copy → record vkCmdCopyBuffer in batch
```

**Files**: `helios_vk.c` — add staging buffer ring, async copy recording

---

## 6. Unnecessary Transposes in Backward Pass (1.1-1.2x fix)

Every attention backward creates full tensor copies via `transpose()` dispatches. The matmul backward (autograd `ops.ts`) does:

```typescript
const tG = B.transpose(g);      // full copy of gradient
gb = B.matmul(tG, aData);       // could read transposed directly
```

For dim=1024, T=512, B=4, heads=16: each transpose copies 4MB. ~20 unnecessary transposes per layer x 21 layers = **~420 extra dispatches, ~1.7GB bandwidth wasted/step**.

### Fix: matmulTransposedA kernel

Add a matmul variant that reads A in transposed layout directly (opposite of existing `matmulTransposed` which transposes B). Eliminate the explicit transpose dispatch.

**Files**: `kernels/matmul.ts` (new variant), `backend.ts` (add `gpuMatmulTransposedA`), `ops.ts` (use in backward)

---

## 7. O(N^2) Barrier Tracking (minor, 1-2ms/step)

The per-buffer write tracking uses nested loops:
```c
for (uint32_t i = 0; i < bufCount; i++)
    for (uint32_t w = 0; w < batchBufWriteCount; w++)
        if (batchBufWrites[w].bufSlot == slot) ...
```

~320K comparisons per step.

### Fix: Direct-indexed array

Index by buffer slot (which is already a small integer). O(1) lookup instead of O(N).

```c
static uint32_t bufLastWriteDispatch[MAX_BUFFERS];  // indexed by slot
```

**Files**: `helios_vk.c`

---

## Expected Gains (cumulative)

| Fix | Multiplier | Cumulative tok/s |
|-----|-----------|------------------|
| Baseline | 1x | 148 |
| Double-buffer cmd bufs | 2-3x | 300-450 |
| + MAX_PENDING_OPS 2048 | 1.3x | 400-580 |
| + Push descriptors | 1.3x | 520-750 |
| + Batch N-API calls | 1.3x | 680-980 |
| + Async uploads | 1.1x | 750-1080 |
| + Eliminate transposes | 1.1x | 820-1190 |
| **Total without tensor cores** | **~6-8x** | **~900-1200** |
| + TF32 tensor cores (if driver supports) | 2-3x | **1800-3600** |

---

## Implementation Order

1. **Double-buffer + increase MAX_PENDING_OPS** — single change to `helios_vk.c`, biggest ROI
2. **Push descriptors** — straightforward Vulkan extension swap
3. **Batch N-API calls** — new `batchDispatchMany` function
4. **Async uploads** — staging ring buffer
5. **Transpose elimination** — new matmul kernel variant
6. **Tensor cores** — already implemented, needs driver support or TF32 path

Fixes 1-3 are purely in the Vulkan layer and don't touch the ML logic. They should get us to ~750-1000 tok/s. Fix 4-5 are moderate effort. Fix 6 depends on driver/hardware support.

---

## Theoretical Peak Reference

L4 f32 compute: 30.3 TFLOPS
307M model FLOPS per token (6 * params): ~1.84 GFLOP
Theoretical max at 100% MFU: ~16,500 tok/s
Realistic target (20-30% MFU): **3,300-5,000 tok/s**
Conservative target (10% MFU): **~1,650 tok/s**

We're at **0.8% MFU**. There's 10-20x headroom.
