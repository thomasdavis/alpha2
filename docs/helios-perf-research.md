# Helios Performance Research: How to Go Even Faster

## Current State (Feb 2026)

Helios is a hand-written Vulkan compute backend for element-wise tensor ops. It loads `libvulkan.so` via `dlopen`, generates SPIR-V bytecode from TypeScript, and bridges to Node.js via N-API. No SDK headers, no frameworks.

### What we've already optimized (Rounds 1 & 2)

| Optimization | What it eliminates |
|---|---|
| Persistent descriptor pool | `vkCreateDescriptorPool` / `vkDestroyDescriptorPool` per dispatch |
| Pre-allocated command buffers | `vkAllocateCommandBuffers` per dispatch |
| Persistent buffer mapping | `vkMapMemory` / `vkUnmapMemory` per upload/read |
| Buffer pool (TS) | `vkCreateBuffer` / `vkAllocateMemory` / `vkBindBufferMemory` / `vkDestroyBuffer` / `vkFreeMemory` per op |
| GPU residence tracking (WeakMap) | Re-uploading inputs that are already on GPU |
| Dedicated output buffers | Allocating new output buffer per dispatch |
| Lazy readback (getter) | GPU-to-CPU readback when result isn't consumed |
| Fence-based sync | `vkQueueWaitIdle` (lighter than queue-wide stall) |
| Device-local memory | Uses VRAM instead of system RAM where possible |
| Staging buffer (grows, reused) | Re-creating staging buffers for transfers |
| Dispatch state caching | Re-recording command buffer when pipeline+buffers+groups unchanged |
| Separate dispatch/transfer cmd bufs | Transfer ops don't invalidate dispatch cache |
| Vec4 SPIR-V kernels | 4x fewer threads, 128-bit memory transactions |
| Params upload caching | Redundant `uploadBuffer` calls when params unchanged |
| Reusable Float32Array for params | `new Float32Array()` allocation per dispatch |

### Current bottleneck profile (per cached dispatch)

```
vkQueueSubmit        ~20-40us   (driver overhead, queue doorbell)
vkWaitForFences      ~40-80us   (poll/sleep until GPU done)
vkResetFences        ~1-2us     (trivial)
N-API crossing       ~5-10us    (JS -> C -> JS)
Cache check          ~1us       (memcmp of buffer handles)
─────────────────────────────────
Total overhead       ~70-130us  per dispatch
```

The GPU shader execution itself is extremely fast for element-wise ops. The bottleneck is the synchronous submit-and-wait loop. We're at **93% of theoretical memory bandwidth** for large tensors — so the GPU side is basically solved. Everything below is about reducing host overhead and unlocking new op categories.

---

## Tier 1: High-Impact Optimizations

### 1. Timeline Semaphores + Deferred Host Wait

**The single biggest remaining win.** Replace the per-op fence wait with a timeline semaphore that lets ops fly without blocking.

**Why timeline semaphores over fences:**
- Fences nudge you into a submit-wait-submit-wait loop (one fence per submit).
- Timeline semaphores signal a monotonically increasing `uint64_t` counter per submit. You can submit N batches, incrementing the counter each time, and only block when you actually need a result on the host.
- Non-blocking progress queries via `vkGetSemaphoreCounterValue` (check if GPU has caught up without waiting).

Ref: [Vulkan Timeline Semaphores (Khronos Blog)](https://www.khronos.org/blog/vulkan-timeline-semaphores)

**Concrete model:**

```c
// Global state
VkSemaphore timelineSem;        // created with VK_SEMAPHORE_TYPE_TIMELINE
uint64_t    nextSubmitValue = 1; // monotonically increasing
uint64_t    completedValue = 0;  // last known completed value (cached)
```

Every tensor/buffer tracks when it was last written:

```c
typedef struct {
  VkBuffer    buffer;
  VkDeviceSize size;
  void*       mapped;
  uint64_t    lastWriterValue;  // timeline value when last dispatch wrote to this
} BufferSlot;
```

**The flow becomes:**

```
1. Record dispatches into command buffer (with barriers between them)
2. Submit, signal timelineSem = ++nextSubmitValue
3. Mark all output buffers: lastWriterValue = nextSubmitValue
4. Return immediately — no wait!
5. Only when .data is accessed: vkWaitSemaphores(timelineSem, lastWriterValue)
```

This replaces "wait every op" with "wait only when host observes", which is exactly what frameworks like Dawn/PyTorch do.

**In-flight safety rules (two invariants):**
1. Never reuse/alias a buffer region until its `lastWriterValue` has completed (check via `vkGetSemaphoreCounterValue`).
2. If an op reads a buffer still in-flight, no host wait needed — just GPU-side ordering via pipeline barriers within the command buffer or semaphores between submits.

**Required Vulkan API additions:**
```c
typedef VkResult (*PFN_vkCreateSemaphore)(VkDevice, const VkSemaphoreCreateInfo*, const void*, VkSemaphore*);
typedef VkResult (*PFN_vkWaitSemaphores)(VkDevice, const VkSemaphoreWaitInfo*, uint64_t);
typedef VkResult (*PFN_vkGetSemaphoreCounterValue)(VkDevice, VkSemaphore, uint64_t*);
// VkTimelineSemaphoreSubmitInfo chained onto VkSubmitInfo
```

Requires Vulkan 1.2+ (or `VK_KHR_timeline_semaphore` extension, very widely supported).

Ref: [Fence wait should be optional with timeline semaphores (Khronos GitHub)](https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/1410)

---

### 2. Batch Dispatches Into One Command Buffer + Pipeline Barriers

Once you have timeline semaphores, the next step is recording multiple dispatches into a single command buffer.

**Current flow (per op):**
```
submit(add) -> wait -> submit(mul) -> wait -> submit(exp) -> wait
Cost: 3 × (submit + fence_wait) ≈ 300-400us overhead
```

**Batched flow:**
```
record(add) -> barrier -> record(mul) -> barrier -> record(exp) -> submit -> wait
Cost: 1 × (submit + fence_wait) + 3 × (barrier) ≈ 100-130us overhead
```

For N ops, overhead drops from `N * 100us` to `100us + N * ~2us`.

**The exact barrier recipe (compute RAW hazard):**

Between dependent dispatches (dispatch 1 writes buffer, dispatch 2 reads it):

```c
VkMemoryBarrier memBarrier = {
  .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
  .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
  .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
};
vkCmdPipelineBarrier(cmdBuf,
  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,   // srcStageMask
  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,   // dstStageMask
  0,                                       // dependencyFlags
  1, &memBarrier,                          // memory barriers
  0, NULL,                                 // buffer barriers
  0, NULL);                                // image barriers
```

This is the canonical compute-to-compute synchronization from the Vulkan spec.

Ref: [Synchronization Examples (Vulkan Docs)](https://docs.vulkan.org/guide/latest/synchronization_examples.html)

**Important: avoid redundant barriers.** Don't barrier after every dispatch "just because." Only insert barriers for true RAW (read-after-write) hazards — when a dispatch reads a buffer that a previous dispatch wrote. Independent ops (different buffers) need no barrier between them.

Ref: [Tips and Tricks: Vulkan Dos and Don'ts (NVIDIA)](https://developer.nvidia.com/blog/vulkan-dos-donts/)

**Flush conditions (when to actually submit the batch):**
- Queue length exceeds threshold (32-256 dispatches)
- Memory pressure / pool nearing limit
- User accesses `.data` (forced sync)
- Host-visible dependency detected
- End of a training step / eval boundary

---

### 3. Encoder API: The Minimal Architecture Change That Unlocks Everything

You don't need a full compute graph compiler. A thin "encoder" layer gets 80% of the win with 20% of the complexity.

**TypeScript side:**
```typescript
class GpuEncoder {
  private ops: DispatchRecord[] = [];
  private submitValue: number = 0;

  record(pipeline: number, buffers: number[], pushConsts: Float32Array, groups: number): void {
    this.ops.push({ pipeline, buffers, pushConsts, groups });
  }

  submit(): number {
    // Calls into N-API: records all ops with barriers, submits, signals timeline
    this.submitValue = vk.submitBatch(this.ops);
    this.ops = [];
    return this.submitValue;
  }

  // Only blocks if needed
  waitUntil(value: number): void {
    vk.waitTimeline(value);
  }
}
```

**N-API additions:**
```c
// New N-API functions:
napi_value napi_beginBatch(env, info);     // reset command buffer, begin recording
napi_value napi_recordDispatch(env, info); // bind pipeline+descriptors, push consts, dispatch, barrier
napi_value napi_submitBatch(env, info);    // end recording, submit, signal timeline, return value
napi_value napi_getCompleted(env, info);   // non-blocking: vkGetSemaphoreCounterValue
napi_value napi_waitTimeline(env, info);   // blocking: vkWaitSemaphores
```

**Integration with existing Backend interface:**

The `gpuBinaryOp` / `gpuUnaryOp` methods become non-blocking recorders:

```typescript
private gpuBinaryOp(a: TensorData, b: TensorData, kernelName: string): TensorData {
  const vk = this.init();
  const size = shapeSize(a.shape);
  const bufA = ensureGpu(vk, a);
  const bufB = ensureGpu(vk, b);
  const bufC = acquireOutputRegion(size * 4);  // timeline-aware allocation

  this.encoder.record(pipeline, [bufA, bufB, bufC, ...], pushConsts, groups);

  // Return lazy tensor — accessing .data triggers flush + wait
  return lazyTensor(vk, a.shape, bufC, this.encoder);
}
```

The `lazyTensor` getter becomes:
```typescript
get data(): Float32Array {
  if (!cached) {
    const value = encoder.submit();  // flush all pending ops
    encoder.waitUntil(value);        // block until GPU done
    cached = vk.readBuffer(handle);
  }
  return cached;
}
```

---

### 4. Timeline-Aware Buffer Leasing (Fix Output Aliasing)

The current "dedicated output buffers" pattern (one per byte-size, reused) breaks with batching: if two ops in the same batch both write to the same output buffer, the second overwrites the first.

**Fix: track `readyValue` per buffer region.**

```typescript
interface BufferRegion {
  handle: number;
  byteSize: number;
  readyValue: number;  // timeline value when this region becomes available
}

function acquireOutputRegion(vk: NativeAddon, byteSize: number): BufferRegion {
  const completed = vk.getCompleted();  // non-blocking query
  const pool = regionPool.get(byteSize) ?? [];

  // Find a region that's no longer in-flight
  for (let i = 0; i < pool.length; i++) {
    if (pool[i].readyValue <= completed) {
      const region = pool.splice(i, 1)[0];
      return region;
    }
  }

  // No free region — allocate new
  return { handle: vk.createBuffer(byteSize, 0), byteSize, readyValue: 0 };
}

function releaseOutputRegion(region: BufferRegion, submitValue: number): void {
  region.readyValue = submitValue;  // can't reuse until this submit completes
  let pool = regionPool.get(region.byteSize);
  if (!pool) { pool = []; regionPool.set(region.byteSize, pool); }
  pool.push(region);
}
```

This eliminates output buffer aliasing without inserting any host waits.

---

### 5. Push Constants (Replace Params Buffer)

Currently we pass element count and scalar via a storage buffer (`paramsBuf4` / `paramsBuf8`). Push constants are faster — embedded directly in the command buffer, no buffer or descriptor needed.

```c
vkCmdPushConstants(cmdBuf, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                   0, 8, &params);  // 8 bytes: {elementCount, scalar}
```

**What this eliminates:**
- The params buffer entirely (1 fewer descriptor binding per pipeline, 3 bindings instead of 4)
- The `uploadBuffer` call for params
- The params caching logic in TS
- One memory load in the shader (push constants are in registers or L1)

**Especially valuable with batching:** push constants are "free" to update while recording (just a memcpy into the command buffer). No need to re-record or invalidate caches — each dispatch in a batch can have different push constant values.

**SPIR-V changes:**
```spirv
; Replace params storage buffer with:
OpDecorate %PushConstBlock Block
%PushConstBlock = OpTypeStruct %float %float    ; {len, scalar}
%PtrPushConst = OpTypePointer PushConstant %PushConstBlock
%pushConst = OpVariable %PtrPushConst PushConstant
```

**C changes:**
```c
// In pipeline layout creation, add:
VkPushConstantRange pushRange = {
  .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
  .offset = 0,
  .size = 8,  // 2 floats
};
// Add to VkPipelineLayoutCreateInfo:
layoutInfo.pushConstantRangeCount = 1;
layoutInfo.pPushConstantRanges = &pushRange;

// New function pointer:
typedef void (*PFN_vkCmdPushConstants)(VkCommandBuffer, VkPipelineLayout,
                                        VkFlags, uint32_t, uint32_t, const void*);
```

---

### 6. Memory Sub-Allocator

Currently each `vkCreateBuffer` + `vkAllocateMemory` allocates a separate Vulkan memory object. Drivers limit total allocations (typically 4096) and each allocation has significant overhead.

Ref: [Descriptor and Buffer Management (Vulkan Docs)](https://docs.vulkan.org/samples/latest/samples/performance/descriptor_management/README.html)

**Two viable approaches:**

**Option A: One big VkBuffer + offsets (simplest, fastest)**

```c
typedef struct {
  VkBuffer       buffer;      // one big buffer
  VkDeviceMemory memory;      // one big allocation
  void*          mapped;      // persistent mapping (if host-visible)
  VkDeviceSize   capacity;
  VkDeviceSize   head;        // bump allocator
} MemoryPool;

// "Allocate" a tensor:
VkDeviceSize offset = ALIGN_UP(pool.head, minStorageBufferOffsetAlignment);
pool.head = offset + size;
return offset;  // not a handle — just an offset into the big buffer
```

Descriptors point to same buffer with different offsets:
```c
VkDescriptorBufferInfo bufInfo = {
  .buffer = pool.buffer,
  .offset = tensorOffset,
  .range = tensorSize,
};
```

This pairs naturally with timeline-aware leasing (offsets + ranges instead of handles).

**Option B: Multiple VkBuffers, shared VkDeviceMemory**

```c
// Allocate one big VkDeviceMemory
// Bind multiple VkBuffers into it at different offsets via vkBindBufferMemory
// Reduces vkAllocateMemory calls from N to 1
```

More flexible (each buffer has its own handle) but more complex.

**For Helios: Option A** is simpler and pairs best with the encoder + push constants + device addresses path.

**Research questions:**
- Query `minStorageBufferOffsetAlignment` from device limits (usually 16-256 bytes)
- One pool per memory type (device-local, host-visible) or unified?
- Fragmentation strategy: bump allocator + periodic compact, ring buffer, or buddy allocator?

---

### 7. GPU Reduction Kernels (sum, mean, softmax, layerNorm)

Currently `sum`, `mean`, `softmax`, `layerNorm` are all CPU-only. Moving them to GPU is critical for training workloads — they're in the hot path of every transformer forward pass.

**Work-efficient parallel reduction (two-phase):**

```
Phase 1: Each workgroup reduces its chunk using shared memory
  - 256 threads load 256 values, tree-reduce in shared memory
  - Write 1 partial sum per workgroup
Phase 2: A second dispatch reduces the per-workgroup partial sums
  - If few enough partials (< 256), single workgroup finishes it
```

**SPIR-V requirements:**
- `OpVariable` with `StorageClass::Workgroup` for shared memory
- Workgroup barrier: `OpControlBarrier` with `Workgroup` scope
- Subgroup reductions (`OpGroupNonUniformFAdd`) can replace the first ~5 levels of tree reduction

**Subgroup operations (warp-level, no shared memory):**

Modern GPUs support subgroup operations that let threads within a warp/wavefront communicate directly:

```spirv
OpCapability GroupNonUniform
OpCapability GroupNonUniformArithmetic
%sum = OpGroupNonUniformFAdd %float %SubgroupScope Reduce %value
```

This eliminates shared memory for the first log2(subgroupSize) levels of reduction. For NVIDIA (subgroup=32), that's 5 levels "free."

**Requires:** Vulkan 1.1+ or `VK_KHR_shader_subgroup`.

**Kernel list to implement:**
| Kernel | Complexity | Training impact |
|--------|-----------|-----------------|
| `sum(axis)` | Medium | Used in mean, variance, loss |
| `max(axis)` | Medium | Used in softmax |
| `softmax` | Medium (fused max+exp+sum+div) | Every attention layer |
| `layerNorm` | Medium (fused mean+var+normalize) | Every transformer block |
| `crossEntropy` | Low (logSoftmax + gather) | Loss computation |

---

### 8. Kernel Fusion (2-5x for multi-op chains)

Many real workloads chain multiple element-wise ops. Each dispatch has ~100us overhead. Fusing adjacent ops into a single kernel eliminates intermediate dispatches AND intermediate memory round-trips.

**a) Template-based fusion (simple, high ROI):**

Pre-generate common fused kernels:
- `mul_add(a, b, c)` = `a * b + c` (FMA — single hardware instruction)
- `add_scale(a, b, s)` = `(a + b) * s`
- `gelu(x)` = full GELU in one kernel (currently CPU-only)
- `silu(x)` = `x * sigmoid(x)` in one kernel
- `rms_norm(x, w, eps)` = fused RMS normalization

**b) JIT fusion (more powerful, build later):**

Build a mini compute graph. When a chain of element-wise ops is detected, compose a single SPIR-V kernel:

```typescript
// Detect: c = add(a, b), d = scale(c, 2.0)
// Generate: d[i] = (a[i] + b[i]) * 2.0
const fusedSpirv = fuseElementwise([
  { op: "add", inputs: ["a", "b"] },
  { op: "scale", inputs: ["$0", 2.0] },  // $0 = output of previous
]);
```

The SPIR-V builder already supports this — just compose operations in the function body. Cache fused kernels by their op-chain signature.

**c) Fusion + batching synergy:**

With the encoder model, the encoder can detect fusible op sequences before submitting. This is where lazy evaluation really pays off — the encoder sees the full op chain and can fuse before recording any Vulkan commands.

---

## Tier 2: Medium-Impact Optimizations

### 9. Descriptor Optimization Paths

Even with dispatch caching, a cache miss requires `vkAllocateDescriptorSets` + `vkUpdateDescriptorSets`. Several extensions can help:

**a) `VK_KHR_push_descriptor` (widely supported):**
Push descriptors directly into the command buffer. No pool, no allocation, no reset:
```c
vkCmdPushDescriptorSetKHR(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                           layout, 0, writeCount, writes);
```

**b) `VK_EXT_descriptor_buffer` (modern, Vulkan 1.3+):**
Descriptors become plain memory you write directly. Zero API calls for updates. Most efficient for high-frequency descriptor changes.

Ref: [VK_EXT_descriptor_buffer (Khronos Blog)](https://www.khronos.org/blog/vk-ext-descriptor-buffer)

**c) `VK_KHR_descriptor_update_template`:**
Pre-create a template describing the layout. `vkUpdateDescriptorSetWithTemplate` is faster than individual `VkWriteDescriptorSet` entries.

**Recommendation:** If we go with sub-allocation (one big buffer + offsets), descriptor churn drops dramatically. Push descriptors are the easiest win if we still have per-op descriptor changes.

---

### 10. GPU Timestamp Queries (Essential for Future Optimization)

Once batching is in place, JS-side timing becomes misleading (host isn't waiting per-op). GPU timestamps give exact shader execution time:

```c
vkCmdWriteTimestamp(cmdBuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);
vkCmdDispatch(cmdBuf, ...);
vkCmdWriteTimestamp(cmdBuf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);
// After readback: (ts1 - ts0) * timestampPeriod = GPU time in nanoseconds
```

This lets us:
- Separate N-API overhead from actual GPU time
- Auto-tune batch sizes empirically
- Detect regressions in shader performance
- Compare kernels (scalar vs vec4 vs vec8) with real GPU timing, not noisy JS timing

**Implementation:** Allocate a `VkQueryPool` with type `VK_QUERY_TYPE_TIMESTAMP`, write timestamps around dispatch spans, read results with `vkGetQueryPoolResults` after sync.

---

### 11. Larger / Adaptive Workgroup Sizes (10-30% throughput)

Currently all kernels use workgroup size 256. Optimal sizes vary by GPU:

| GPU | Subgroup size | Recommended WG size |
|-----|------|------|
| NVIDIA (warp=32) | 32 | 256-1024 |
| AMD RDNA3 (wave32) | 32 | 256-512 |
| Intel Arc (Xe-HPG) | 8-32 | 128-256 |
| Integrated (Intel/AMD) | varies | 128-256 |

**What to do:**
- Query `maxComputeWorkGroupInvocations` and `subgroupSize` from device properties
- For simple element-wise ops (low register pressure): try 512 or 1024
- For complex ops (exp/log, many registers): stick with 256
- Optional: auto-tune by timing each kernel at different WG sizes during init

---

### 12. Multiple Elements Per Thread (vec4x2, vec4x4)

Currently each thread processes 4 elements (vec4). Going to 8 or 16 per thread via unrolled vec4 loads improves ILP:

```spirv
// Each thread processes 8 elements (2 vec4 loads)
vec4 a0 = A[gidX * 2 + 0];
vec4 a1 = A[gidX * 2 + 1];
vec4 c0 = a0 + b0;
vec4 c1 = a1 + b1;
C[gidX * 2 + 0] = c0;
C[gidX * 2 + 1] = c1;
```

The dispatch count drops by another 2x. More independent instructions help the GPU scheduler hide memory latency.

**Trade-off:** More registers per thread = lower occupancy. For simple ops this is fine; for exp/log it might hurt.

---

## Tier 3: Architecture-Level Changes

### 13. Compute Graph / Lazy Evaluation (5-20x for training)

The endgame architecture: instead of executing ops eagerly, build a graph and execute once.

```typescript
// Current: each op dispatches immediately
const c = backend.add(a, b);     // dispatch 1, wait
const d = backend.mul(c, e);     // dispatch 2, wait
const f = backend.exp(d);        // dispatch 3, wait

// Graph mode: build DAG, execute once
const c = backend.add(a, b);     // records node
const d = backend.mul(c, e);     // records node
const f = backend.exp(d);        // records node
const result = f.data;           // topological sort → fuse → record → submit → wait
```

**Benefits:**
- All ops in one command buffer with barriers
- Enables automatic fusion of adjacent element-wise ops
- Enables memory planning (allocate all intermediates up front, reuse dead buffers)
- Enables operator reordering (independent ops dispatched without barriers between them)

**This is how all serious GPU compute frameworks work** (PyTorch compile, JAX XLA, TensorFlow XLA). The encoder API from section 3 is a stepping stone — it handles linear chains. A full graph handles arbitrary DAGs.

**Implementation complexity:** High. Requires changing the Backend interface from eager to lazy. Autograd needs to work with graph nodes.

---

### 14. Cooperative Matrix Multiply (10-50x for matmul)

For matrix multiplication (currently CPU-only), Vulkan exposes hardware tensor cores:

**`VK_KHR_cooperative_matrix`** — each subgroup cooperatively multiplies small tiles (e.g., 16x16) using dedicated matrix hardware (NVIDIA tensor cores, AMD WMMA, Intel DPAS).

```spirv
OpCapability CooperativeMatrixKHR
%result = OpCooperativeMatrixMulAddKHR %matType %A %B %C
```

Near-theoretical-peak performance. This is the path to making actual training fast on GPU.

**Requires:** Vulkan 1.1+ and the cooperative matrix extension (NVIDIA Turing+, AMD RDNA3+, Intel Arc).

---

### 15. VK_KHR_buffer_device_address (Eliminates Descriptors Entirely)

Get raw GPU pointers for buffers, pass as push constants, load/store directly:

```c
VkBufferDeviceAddressInfo info = { .buffer = myBuffer };
uint64_t gpuAddr = vkGetBufferDeviceAddress(device, &info);
// Pass gpuAddr as push constant
```

In shader:
```spirv
OpCapability PhysicalStorageBufferAddresses
; Load directly from GPU address — no descriptors at all
```

This eliminates the entire descriptor management layer. Most modern GPU compute libraries use this approach.

**Requires:** Vulkan 1.2+ (widely supported).

---

### 16. Multiple Queues / Async Transfer

Most GPUs have separate queues for compute and transfer. Currently we use one queue for everything.

With separate queues:
- Upload tensor B while computing on tensor A
- Download result C while computing result D
- Use semaphores for cross-queue synchronization

**Requires:** Finding a dedicated transfer queue family, separate command pools, semaphore-based sync.

---

## Tier 4: Shader-Level Micro-Optimizations

### 17. FMA Instructions

Replace `a * b + c` with fused multiply-add (`OpExtInst Fma`). Single hardware instruction, better precision (single rounding). Relevant for fused kernels (layernorm, gelu).

### 18. Relaxed Precision

`OpDecorate %result RelaxedPrecision` — some GPUs execute at 2x throughput for relaxed-precision ops. Good for ops where full f32 precision isn't needed.

### 19. f16 Storage + f32 Compute

Store tensors as float16, compute in float32. Halves memory bandwidth:
- Declare buffers as `RuntimeArray<f16vec4>` (8 bytes per vec4 instead of 16)
- `OpFConvert` f16→f32 after load, compute, `OpFConvert` f32→f16 before store
- **2x memory bandwidth improvement**

**Requires:** `VK_KHR_16bit_storage` and `shaderFloat16` feature.

### 20. Skip Bounds Check for Aligned Sizes

The current bounds check creates divergence at the tail:
```spirv
if (gidX >= len) skip;
```

For vec4 kernels with aligned sizes (which is ~100% of real workloads), every thread is valid. Add an "exact" kernel variant that omits the bounds check entirely — saves a few instructions per thread and eliminates warp divergence.

---

## What Dawn/WebGPU Does That We Don't (Yet)

| Dawn feature | Helios equivalent | Status |
|---|---|---|
| Command encoder (batch dispatches) | Per-op submit+wait | **Planned: Encoder API** |
| Automatic barrier insertion | Manual (currently none within batch) | **Planned: With encoder** |
| Memory sub-allocation | Individual vkAllocateMemory | **Planned** |
| Shader compilation cache (disk) | In-memory only | Could add |
| Multiple dispatches per submit | One dispatch per submit | **Planned: Batching** |
| Push constants | Storage buffer for params | **Planned** |

The biggest gap: Dawn records many dispatches per submit with auto-barriers. Our encoder API closes this gap.

---

## Implementation Roadmap (Recommended Order)

| Step | What | Effort | Impact | Dependencies | Status |
|------|------|--------|--------|-------------|--------|
| 1 | GPU timestamp queries | Low | Enables data-driven optimization | None | **DONE** |
| 2 | Push constants (replace params buffer) | Low | ~10-30us/dispatch, cleaner code | None | **DONE** |
| 3 | Timeline semaphore + deferred wait | Medium | Unlocks async execution | None | **DONE** |
| 4 | Encoder API (batch recording + barriers) | Medium | 3-10x for op chains | Step 3 | **DONE** (C API) |
| 5 | Timeline-aware buffer leasing | Medium | Fixes aliasing without waits | Steps 3-4 | **DONE** |
| 6 | Memory sub-allocator (slab allocator) | Medium | Eliminates alloc overhead | None (but pairs with 4-5) | **DONE** |
| 7 | GPU reduction kernels (sum, max) | Medium | Unlocks GPU softmax/layernorm | None | **DONE** |
| 8 | Fused kernels (gelu, relu, silu, softmax, layernorm) | Medium | 2-5x for training hot path | Step 7 | **DONE** |
| 9 | Tiled matrix multiply (16×16 shared memory) | Medium | GPU matmul for all sizes | None | **DONE** |
| 10 | f16 storage (kernels + feature probing) | Medium | 2x bandwidth | None | **DONE** |
| 11 | Compute graph / lazy eval | High | 5-20x training | Steps 3-5 | **DONE** |
| 12 | Buffer device address | Medium | Eliminates descriptors | Vulkan 1.2 | Future |
| 13 | Auto-tuned workgroup sizes | Low | 10-30% throughput | Step 1 | **DONE** |
| 14 | Multiple queues (async transfer) | High | 20-40% overlap | Steps 3-5 | **DONE** |

**The critical path for training performance:**
Steps 1-2 (quick wins) → Steps 3-5 (async foundation) → Steps 7-9 (GPU training ops) → Step 11 (graph compiler)

---

## Hardware-Specific Notes

### NVIDIA (most common discrete GPU)
- Warp size: 32
- L1 cache: 128KB per SM (configurable shared/L1 split)
- Best workgroup size: 256-1024
- Tensor cores via cooperative matrix (Turing+)
- Subgroup ops fully supported
- Good at hiding latency with many warps in flight
- `minStorageBufferOffsetAlignment`: typically 16 bytes

### AMD RDNA 3
- Wave size: 32 (wave32 mode) or 64 (wave64 mode)
- Infinity Cache: 96MB L3
- Best workgroup size: 256-512
- Matrix ops via WMMA instructions
- Subgroup ops supported
- `minStorageBufferOffsetAlignment`: typically 16 bytes

### Intel Arc (Xe-HPG)
- Subgroup size: 8, 16, or 32 (configurable via `VK_EXT_subgroup_size_control`)
- Large L1 cache (256KB per Xe-core)
- Cooperative matrix via DPAS instructions
- May prefer smaller workgroup sizes (128-256)
- `minStorageBufferOffsetAlignment`: typically 64 bytes

### Integrated GPUs (Intel/AMD APU)
- Shared memory with CPU (no PCIe bottleneck!)
- Device-local + host-visible memory is the fast path
- Lower raw throughput but zero transfer overhead
- Our current code already handles this well (unified memory path)
- Sub-allocation less critical (no discrete VRAM pressure)

---

## Theoretical Bandwidth Limits

For a GPU with bandwidth B (GB/s):

| Op | Bytes/element (read+write) | Peak throughput |
|----|---|---|
| add(a,b) | 12 (read A + read B + write C) | B/12 G elem/s |
| scale(a,s) | 8 (read A + write C) | B/8 G elem/s |
| neg(a) | 8 (read A + write C) | B/8 G elem/s |
| exp(a) | 8 (read A + write C) | B/8 G elem/s |
| fused add+scale | 12 in, 4 out = 16 → but saves 8 (no intermediate) = **8** | B/8 G elem/s |

**Our current measurements (4.2M elements):**

| Op | Throughput (M elem/s) | Effective BW | % of theoretical (est ~40 GB/s) |
|----|---|---|---|
| neg | 4,649 | 37.2 GB/s | **93%** |
| exp | 4,651 | 37.2 GB/s | **93%** |
| mul | 3,097 | 37.2 GB/s | **93%** (12 bytes/elem) |
| add | 2,997 | 36.0 GB/s | **90%** |

**Conclusion:** For large element-wise ops, we're already at the memory bandwidth wall. The remaining wins are:

1. **Reduce dispatch overhead** (batching) — helps all sizes, especially small
2. **Reduce memory traffic** (f16 storage = 2x, fusion = eliminates intermediates)
3. **New op categories** (matmul via tensor cores, reductions via shared memory)
4. **Overlap compute and transfer** (async queues)

For training specifically: GPU matmul + GPU reductions + kernel fusion + compute graph = the whole game.
