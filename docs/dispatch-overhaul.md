# Helios Vulkan Dispatch Infrastructure Overhaul

## Status: DEPLOYED (2026-02-27)

Training restarted on `alpha-cognitive` with new code. Run ID: `concordance_v2_20260227074907_59am`.

## Problem

Training 307M param model on L4 at ~148 tok/s = ~0.8% MFU. Root cause: "one-frame-in-flight" compute — `batchBegin()` called `waitTimelineValue(lastDispatchTimeline)` which blocked CPU until GPU finished, then reset the single descriptor pool + single command buffer. With ~9000 dispatches/step and MAX_PENDING_OPS=256, this created ~36 full GPU pipeline drains per step.

## Changes Made (all 6 phases implemented)

### Phase 1: MAX_PENDING_OPS 256→2048
- `backend.ts`: `MAX_PENDING_OPS = 2048`
- `helios_vk.c`: descriptor pool maxSets 256→2048, descriptorCount 2048→16384
- Reduces flushes from ~36/step to ~5/step

### Phase 2: O(1) Barrier Tracking
- Replaced `batchBufWrites[256]` linear-scan array + `batchBufWriteCount` with:
  - `bufWriteGen[MAX_BUFFERS]` — generation stamp per buffer slot
  - `bufWriteDispatch[MAX_BUFFERS]` — dispatch index of last write
  - `bufWriteGeneration` — bumped on batchBegin
- Barrier check per buffer: single `bufWriteGen[slot] == bufWriteGeneration` check instead of scanning 2048 entries

### Phase 3: Multi-Frame-In-Flight Ring (3 slots)
- `BatchCtx` struct: `{ cmdPool, cmd, descPool, timelineValue }`
- `g_ring[RING_SIZE=3]`, `g_ringHead` advances on batchSubmit
- batchBegin only waits on the specific ring slot, not globally
- CPU can record batch N+1 while GPU executes batch N
- Single-dispatch path uses separate `singleDescPool` (small, 32 sets)

### Phase 4: batchDispatchMany (packed N-API call)
- Single `batchDispatchMany(ArrayBuffer, count)` replaces up to 2048 individual N-API calls
- Packed binary format per dispatch: pipeSlot(i32), bufCount(u16), flags(u16), gX(u32), [gZ(u32)], writeMask(u32), bufHandles[](i32), pushData(u8[])
- flags: bits[15:1]=gY, bit0=hasGZ
- backend.ts flush() packs all ops into single ArrayBuffer, falls back to per-dispatch loop if native method unavailable

### Phase 5: Push Descriptors (VK_KHR_push_descriptor)
- Probes for extension in device enumeration, enables if available
- Loads `vkCmdPushDescriptorSetKHR` via `vkGetDeviceProcAddr`
- Pipeline descriptor set layouts get `VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR`
- batchDispatchMany + batchDispatch use push descriptors when available, skip pool allocation
- Ring descriptor pools not created when push descriptors available
- `hasPushDescriptors` returned in initDevice info, added to device.ts NativeAddon type
- **L4 does NOT support this extension** — falls back to pool-based path

### Phase 6: Async Upload Staging Ring
- 4 slots x 16MB each, persistently mapped (`StagingSlot` struct)
- Device-local uploads ≤16MB use a free staging slot + async copy (no blocking wait)
- If batch is recording, copy command goes into current batch cmd buffer with transfer→compute barrier
- If no batch recording, standalone async submit via timeline semaphore
- Falls back to legacy blocking staging for >16MB uploads
- `initStagingRing()` called lazily on first device-local upload

## Files Modified

| File | Key Changes |
|------|-------------|
| `packages/helios/native/helios_vk.c` | Ring of 3 BatchCtx, push descriptors, batchDispatchMany, O(1) barriers, async staging ring |
| `packages/helios/src/backend.ts` | MAX_PENDING_OPS=2048, packed ArrayBuffer flush, feature detection |
| `packages/helios/src/device.ts` | NativeAddon interface: batchDispatchMany, hasPushDescriptors in initDevice return |

## Bug Fix: Inter-Batch Memory Barrier (2026-02-27)

The multi-frame ring (Phase 3) introduced a correctness bug: **missing memory barriers between batches**.

**Root cause**: `batchBegin()` increments `bufWriteGeneration`, clearing all write tracking. The first dispatch in batch N has no barrier to make batch N-1's shader writes visible. The old code was safe because `batchBegin()` drained ALL GPU work via `waitTimelineValue(lastDispatchTimeline)`. The ring-based code only waits for the specific ring slot (potentially 2 batches ago), so recent batch writes were not memory-visible.

**Symptoms**: Loss stuck at exactly 9.8923 from step 1, grad norms 150x smaller than expected (0.04 vs 6.0), model not learning at all. GPU was reading stale/uninitialized data.

**Fix**: Added a global `VkMemoryBarrier` at the start of each batch command buffer (after `vkBeginCommandBuffer`) to synchronize all prior compute shader writes:
```c
VkMemoryBarrier memBarrier = {
  .sType = 46, // VK_STRUCTURE_TYPE_MEMORY_BARRIER
  .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
  .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
};
fp_vkCmdPipelineBarrier(g_ring[slot].cmd,
  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
  0, 1, &memBarrier, 0, NULL, 0, NULL);
```

## Results

| Metric | Before (old code) | Ring w/o barrier (broken) | Ring w/ barrier (fixed) |
|--------|-------------------|--------------------------|------------------------|
| tok/s | 148 | 167 | **147** |
| Step 1 loss | 10.1112 | 9.8923 (stuck) | 10.1230 |
| Grad norm | ~6.0 | ~0.04 | ~6.0 |
| Learning | Yes | **No** | Yes |

The inter-batch barrier costs roughly the throughput gained by the ring overlap, making this approximately net-neutral on L4. On H100 with push descriptors (skipping descriptor pool allocation entirely), the ring overlap should provide a net gain since the barrier cost is fixed while the savings from eliminating per-dispatch pool allocation scale with dispatch count.

## Current Training Run

```
ssh <user>@<instance-ip>
# Run dir: ~/alpha/runs/concordance_300m_v3/
# Log: ~/alpha/runs/concordance_300m_v3/train.log
# Config: dim=1024, layers=21, heads=16, block=512, batch=1, accumSteps=4
# LR: 6e-4 (concordance domain default), warmup=1000, total iters=20000
# sampleInterval=200 (Discord inference samples)
```

## Future Work

- Kernel optimization: tiled/cooperative matmul for L4
- Profile remaining step time breakdown (compute vs memory vs dispatch)
- Test on H100 where push descriptors + ring overlap will have bigger impact
- Consider increasing batch size if memory allows (block=512 leaves headroom)
- Optimize inter-batch barrier: could track which buffers are "cross-batch dirty" to use per-buffer barriers instead of global barrier
