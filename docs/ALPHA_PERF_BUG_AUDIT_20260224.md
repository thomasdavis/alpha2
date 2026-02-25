# Alpha Audit (Performance, Bugs, Risk Review)

Date: 2026-02-24

Scope: core math/autograd/model/train pipeline, Helios GPU backend (TS + native C Vulkan bridge), and key web inference/runtime surfaces.

Method:
- Static code review of hot paths and contracts.
- Focused test run: `npm test -w @alpha/tests` (31 tests passed).
- Review aligned to the repo philosophy: keep core logic in TypeScript, reserve C for GPU/native plumbing.

## Executive Summary

The codebase has strong engineering direction and unusually ambitious from-scratch systems work (especially `packages/helios`). The highest-risk issues are not generic style problems; they are a small number of correctness bugs in broadcast semantics and training behavior, plus a native addon input-validation hole.

Top priorities:
1. Fix broadcast semantics (`autograd` + `helios` CPU/GPU broadcast paths).
2. Prevent activation checkpointing with nondeterministic dropout (or make dropout deterministic).
3. Restore attention dropout parity in the FlashAttention path.
4. Harden native Vulkan bridge buffer-array bounds (`>32` bindings currently unsafe).
5. Fix `DataLoader.nextBatch()` bounds/short-dataset behavior.

## Findings (Bugs / Correctness / Reliability)

### Critical

- Incorrect broadcast semantics in autograd/backends can produce wrong gradients and wrong tensor values.
  - Evidence:
    - `broadcastTo()` repeats by flat modulo (`i % srcSize`) in `packages/autograd/src/ops.ts:522`.
    - Helios `broadcast()` does the same in both GPU contract and CPU fallback in `packages/helios/src/backend.ts:2171`.
    - Helios `cpuBinaryOp()` also uses flat modulo for mismatched shapes in `packages/helios/src/backend.ts:2546`.
  - Why this is a bug:
    - Flat modulo is only correct for scalar broadcast and some trivial shape layouts.
    - It is wrong for common cases like broadcasting `[B, T, 1]` to `[B, T, C]` or `[B, 1, C]` to `[B, T, C]`.
    - This directly affects `softmax` backward (`packages/autograd/src/ops.ts:417`) and any path using broadcasted grads.
  - Fix:
    - Implement shape-aware broadcasting using strides (like `CpuRefBackend` already does for binary ops), and reuse that logic in `autograd.broadcastTo()` and Helios CPU/GPU broadcast kernels.

- Activation checkpointing is enabled in training even though dropout is nondeterministic, which can yield incorrect gradients.
  - Evidence:
    - Checkpoint implementation explicitly warns it is incompatible with nondeterministic ops in `packages/autograd/src/checkpoint.ts:16`.
    - `gptForward()` still applies checkpointing based only on flags/training mode in `packages/model/src/gpt.ts:215`.
    - Dropout masks use `Math.random()` in `packages/autograd/src/ops.ts:351` and `packages/autograd/src/ops.ts:373`.
  - Impact:
    - Recomputed forward during backward will use a different dropout mask, violating checkpoint assumptions.
  - Fix:
    - Short term: disallow `activationCheckpointing && training && dropout > 0`.
    - Better: use a deterministic RNG stream keyed by layer/op/step and thread it through dropout.

- FlashAttention training path drops attention dropout entirely (behavior mismatch vs non-flash path).
  - Evidence:
    - Standard path applies `dropout()` to attention weights in `packages/model/src/gpt.ts:152`.
    - Flash path calls `flashAttention(...)` with no dropout parameter in `packages/model/src/gpt.ts:126`.
    - Backend interface for FlashAttention has no dropout/training arg in `packages/core/src/interfaces.ts:110`.
  - Impact:
    - Training behavior changes depending on backend capability, which makes reproducibility and regularization inconsistent.
  - Fix:
    - Add dropout support to flash attention API/kernel path (preferred), or explicitly disable/zero attention dropout when flash is used and document it.

- Native addon has unbounded `bufCount` usage with fixed-size stack arrays (`[32]`) causing memory corruption risk.
  - Evidence:
    - `napi_dispatch` reads up to 32 entries but later loops over full `bufCount` and uses `memcmp`/`memcpy` with `bufCount` in `packages/helios/native/helios_vk.c:1568`.
    - Same pattern in `napi_batchDispatch` in `packages/helios/native/helios_vk.c:1732`.
    - Same pattern in `napi_gpuTime` in `packages/helios/native/helios_vk.c:1883`.
  - Impact:
    - Stack out-of-bounds writes/reads if JS passes more than 32 buffers.
    - Native crashes or silent corruption.
  - Fix:
    - Hard reject `bufCount > 32` before any copies/loops.
    - Also validate `bufCount == pipeline.numBindings` (or `<=` by explicit policy).

### High

- `DataLoader.nextBatch()` has an off-by-one sampling bug and no short-dataset guard.
  - Evidence:
    - `maxStart = tokens.length - T - 1` and `Math.floor(rng * maxStart)` in `packages/train/src/data.ts:59`.
  - Impact:
    - Last valid training window is never sampled.
    - If `tokens.length <= T`, `maxStart <= -1` and indexing becomes invalid/silent corruption.
  - Fix:
    - Validate `tokens.length >= T + 1` in constructor or `nextBatch()`.
    - Sample `start` from `[0, maxStart]` using `Math.floor(rng.next() * (maxStart + 1))`.

- Web inference crashes on prompts longer than `blockSize`.
  - Evidence:
    - `tokens` is allocated with `maxLen <= blockSize`, but `tokens.set(promptTokens)` writes full prompt in `apps/web/src/lib/engine.ts:278`.
  - Impact:
    - Long prompt requests can throw `RangeError` instead of truncating/sliding context.
  - Fix:
    - Truncate prompt tokens to last `blockSize` before `set()`, and report truncation if useful.

- `ensureInit()` has a race/stuck-on-failure initialization pattern.
  - Evidence:
    - `_initialized = true` is set before awaited work in `apps/web/src/lib/init.ts:11`.
  - Impact:
    - Concurrent requests can observe partially initialized state.
    - If `initEngine()` throws once, future requests permanently skip initialization.
  - Fix:
    - Replace boolean with a shared `Promise<void> | null` init latch, reset on failure.

- Helios `purgeBufferPools()` destroys pending buffers without waiting for GPU completion.
  - Evidence:
    - Comment says “wait for GPU if needed” but code destroys `pendingDestroys` immediately in `packages/helios/src/backend.ts:648`.
  - Impact:
    - Unsafe if called while GPU work is in-flight.
  - Fix:
    - `graph.flush()`, `waitTimeline(lastFlush)`, then process pending destroys safely.

- Native async submit ignores `vkQueueSubmit` result.
  - Evidence:
    - `submitCmdBufAsync()` calls `fp_vkQueueSubmit(...)` and unconditionally returns timeline value in `packages/helios/native/helios_vk.c:686`.
  - Impact:
    - On device-loss/submit failure, JS may wait on a timeline value that was never signaled.
  - Fix:
    - Check return code and propagate an error to JS.

### Medium

- Helios CPU fallback broadcasting and batched matmul do not match backend contract semantics.
  - Evidence:
    - `cpuBinaryOp()` uses flat modulo instead of shape-aware broadcast in `packages/helios/src/backend.ts:2546`.
    - `cpuMatmul()` assumes identical batch layout and uses `bOff = batch * K * N` in `packages/helios/src/backend.ts:2559`.
  - Impact:
    - Incorrect results for small tensors that fall back to CPU on Helios backend.
  - Fix:
    - Reuse `CpuRefBackend` stride-based semantics or extract shared TS reference helpers.

- Native pipeline creation leaks descriptor/pipeline-layout resources on failure paths.
  - Evidence:
    - `vkCreatePipelineLayout` failure path only destroys shader module in `packages/helios/native/helios_vk.c:1513`.
    - `vkCreateComputePipelines` failure path does not destroy created layout/descriptor set layout in `packages/helios/native/helios_vk.c:1533`.
  - Impact:
    - Native resource leaks on invalid SPIR-V / pipeline creation failures.
  - Fix:
    - Add structured cleanup for partially-built pipeline slot resources before return.

- Large-file tokenizer chunking can still split UTF-8 if no newline exists in chunk.
  - Evidence:
    - Chunking relies on newline search and falls back to raw chunk boundary in `packages/train/src/data.ts:170`.
  - Impact:
    - Potential replacement characters / tokenization drift on very long lines (JSONL edge cases, minified corpora).
  - Fix:
    - Carry incomplete UTF-8 suffix bytes to the next chunk (byte-level decoder buffering), independent of newline logic.

- Training metrics timing fields are internally inconsistent.
  - Evidence:
    - `timing_grad_clip_ms` and `timing_optim_ms` are both set to `_t5 - _t4` in `packages/train/src/trainer.ts:551`.
  - Impact:
    - Diagnostics can mislead optimization work.
  - Fix:
    - Split clip time and optimizer-step time into separate windows or rename fields to reflect combined timing.

## Performance Improvements (High-Value)

### Core / Training Path

- Cache position indices and causal masks instead of recreating every forward pass.
  - Evidence:
    - Position index tensor recreated every call in `packages/model/src/gpt.ts:195`.
    - Causal mask recreated every call in `packages/model/src/gpt.ts:207`.
  - Impact:
    - Repeated CPU allocations and uploads in the hottest path.
  - Recommendation:
    - Cache by `(B,T)` for position indices and by `T` for masks, keyed per backend/device.

- Dropout currently generates masks on CPU and uploads them every call.
  - Evidence:
    - `Math.random()` + `Float32Array` + `backend.clone()` in `packages/autograd/src/ops.ts:351` and `packages/autograd/src/ops.ts:388`.
  - Impact:
    - CPU bottleneck and host->device transfer overhead on GPU training.
  - Recommendation:
    - Add a GPU dropout mask kernel (TS-generated SPIR-V) with deterministic seed/counter push constants.

- Checkpoint save path builds one giant `Buffer` in memory before writing.
  - Evidence:
    - `Buffer.alloc(totalSize)` + full copy in `packages/train/src/checkpoint.ts:66`.
  - Impact:
    - Large temporary memory spikes and GC pressure for big checkpoints.
  - Recommendation:
    - Stream write header + tensor blobs to file (`FileHandle.write`) to avoid double-buffering.

- Trainer forces GC and GPU sync every step (good for VRAM safety, expensive for throughput).
  - Evidence:
    - Explicit `gc()` + `syncGpu()` in `packages/train/src/trainer.ts:517`.
  - Impact:
    - Reduces throughput and overlap; can dominate step time once memory is stable.
  - Recommendation:
    - Make this adaptive/configurable (e.g., every N steps or only when pool growth exceeds threshold).

- Per-step metrics write is fully awaited in hot loop.
  - Evidence:
    - `await metricsHandle.write(...)` in `packages/train/src/trainer.ts:632`.
  - Impact:
    - Synchronous backpressure from filesystem in the training loop.
  - Recommendation:
    - Buffer writes and flush periodically, or use a background writer queue.

### Inference / Web Path

- Sampling does full sort for top-k every generated token.
  - Evidence:
    - `Array.from(...).sort(...)` in `apps/web/src/lib/engine.ts:235`.
  - Impact:
    - O(V log V) per token for top-k; hurts latency on larger vocabularies.
  - Recommendation:
    - Use selection (`quickselect`) or a fixed-size heap; avoid materializing object arrays.

- No KV cache in autoregressive generation (full forward each token).
  - Evidence:
    - `sampleNextToken()` re-runs `gptForward()` on current context every token in `apps/web/src/lib/engine.ts:214`.
  - Impact:
    - O(T^2) decode behavior and poor interactive latency.
  - Recommendation:
    - Add a TS-first KV-cache path in model/inference API; GPU backend kernels can follow later.

- `scanLocalRuns()` does synchronous full-file reads of `metrics.jsonl` just to fetch the last line.
  - Evidence:
    - `readFileSync(...).split("\n")` in `apps/web/src/lib/engine.ts:114`.
  - Impact:
    - Startup stalls scale with run count and metrics file size.
  - Recommendation:
    - Tail-read last few KB and parse backwards, or persist last metrics snapshot separately.

- `ensureModel()` has no load de-duplication and caches only one model.
  - Evidence:
    - Single `loaded` slot in `apps/web/src/lib/engine.ts:61`; no in-flight promise cache in `apps/web/src/lib/engine.ts:184`.
  - Impact:
    - Duplicate checkpoint loads under concurrent cold requests; cache thrash across models.
  - Recommendation:
    - Add `Map<modelId, Promise<LoadedModel>>` for in-flight loads and an LRU cache (size 2-4).

## Test / Validation Notes

- Ran `npm test -w @alpha/tests`
  - Result: 4 test files, 31 tests passed.
- Current tests cover baseline tensor/autograd/tokenizer/rng behavior, but do not appear to cover:
  - Broadcasted gradient correctness in `softmax` backward on rank >2 shapes.
  - Helios backend semantics parity vs `CpuRefBackend`.
  - Trainer checkpointing + dropout interaction.
  - Native addon bounds validation and error-path cleanup.

## Recommended Fix Order (Pragmatic)

1. Correct broadcast semantics (`autograd.broadcastTo`, Helios CPU/GPU broadcast, Helios CPU binary fallback).
2. Guard checkpointing with dropout (or implement deterministic dropout RNG).
3. Fix FlashAttention dropout parity.
4. Harden native addon bounds checks (`bufCount <= 32`, validate buffer handles/counts, submit result checks).
5. Fix `DataLoader.nextBatch()` bounds and add a test for short datasets.
6. Fix web long-prompt handling and `ensureInit()` promise latch.
7. Implement hot-path perf wins (mask/index caching, checkpoint streaming write, top-k selection).

## Philosophy Fit (TS-first / C-for-GPU)

These recommendations preserve the current philosophy:
- Keep semantics/correctness fixes in TypeScript first (`autograd`, `model`, `train`, web).
- Use the native C layer only for hardening and GPU submission correctness.
- Add GPU kernels only where they eliminate real bottlenecks (dropout mask generation, optional later KV-cache kernels).

