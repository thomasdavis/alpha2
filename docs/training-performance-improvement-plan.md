# Training Performance Improvement Plan (Codebase Audit)

This document is a code-driven review of where training time is spent in Alpha and how to improve it.

Scope:
- Training throughput (`tok/s`, `ms/iter`) for `apps/cli train`
- CPU and `helios` GPU backends
- End-to-end step cost (data, forward, backward, grad norm/clip, optimizer, sync, eval/checkpoint/sample side work)

Primary audited code paths:
- `packages/train/src/trainer.ts`
- `packages/model/src/gpt.ts`
- `packages/autograd/src/tape.ts`
- `packages/autograd/src/ops.ts`
- `packages/helios/src/backend.ts`
- `packages/train/src/data.ts`
- `packages/train/src/checkpoint.ts`
- `apps/cli/src/commands/train.ts`

## Executive Summary

The codebase already has meaningful performance work (GPU graph batching, lazy readback, GPU-resident tensors, fused CE forward/backward, GPU AdamW), but training still pays significant overhead above the kernel layer.

The highest-impact opportunities are:

1. Remove or reduce per-step hard synchronization (`gc()`, `syncGpu()`, scalar readbacks)
2. Cache recurrent tensors in the model forward path (causal masks, positional indices)
3. Reduce extra passes over gradients (norm + clip + safety checks)
4. Cut JS allocation churn in hot loops (param maps, batches, tape objects)
5. Fuse more transformer work (QKV projection, attention mask/softmax, residual+bias chains)

The biggest architectural gap vs high-performance trainers is that training currently behaves like a sequence of micro graph breaks and maintenance passes, rather than a mostly asynchronous step with a single intentional synchronization point.

## How Training Works Today (Relevant to Performance)

### Step structure

Per step in `packages/train/src/trainer.ts`:
- Gradient accumulation loop (`Tape`, `DataLoader.nextBatch()`, `gptForward()`, `tape.backward()`) in `packages/train/src/trainer.ts:284`
- Optional gradient scaling for accumulation in `packages/train/src/trainer.ts:312`
- Gradient norm computation via extra backend ops and scalar readback in `packages/train/src/trainer.ts:327`
- Optional gradient clipping + extra safety checks in `packages/train/src/trainer.ts:395`
- Optimizer step in `packages/train/src/trainer.ts:430`
- Explicit grad release + `flush()` + `gc()` + `syncGpu()` in `packages/train/src/trainer.ts:440`
- Metrics/logging/write JSONL in `packages/train/src/trainer.ts:478`
- Periodic eval/checkpoint/sample in `packages/train/src/trainer.ts:512`, `packages/train/src/trainer.ts:554`, `packages/train/src/trainer.ts:563`

### Existing instrumentation (good)

The trainer already records a useful timing breakdown:
- `timing_data_ms`
- `timing_fwd_ms`
- `timing_bwd_ms`
- `timing_grad_norm_ms`
- `timing_grad_clip_ms`
- `timing_optim_ms`
- `timing_flush_ms`

See `packages/train/src/trainer.ts:481`.

This is enough to drive a disciplined optimization program without guessing.

## Observed Bottlenecks and Why They Matter

## 1. Per-step synchronization and GC are likely throttling GPU throughput

The trainer does all of the following every step on GPU runs:
- `flush()` (`packages/train/src/trainer.ts:447`)
- `globalThis.gc()` + `setImmediate()` (`packages/train/src/trainer.ts:449`)
- `syncGpu()` (which waits on timeline completion) (`packages/train/src/trainer.ts:458`)

`HeliosBackend.syncGpu()` explicitly waits for GPU completion (`vk.waitTimeline`) in `packages/helios/src/backend.ts:562`.

Why this hurts:
- It serializes step execution with the GPU, reducing overlap and batching benefits.
- It makes throughput sensitive to JS GC behavior.
- It can dominate step time when kernels are fast and the model is small/medium.

Why it exists:
- Memory safety / buffer reuse pressure.
- FinalizationRegistry-driven cleanup is not deterministic.

Improvement direction:
- Keep the memory-safety goal, but avoid full wait-every-step by using a budgeted sync policy:
  - `syncGpu()` every N steps
  - or only when pool/pending counters exceed thresholds from `gpuMemStats()`
  - or only on eval/checkpoint/sample boundaries

This is the highest leverage runtime-level change for `helios`.

## 2. The training loop forces graph breaks by reading scalars mid-step

Examples:
- `microLoss` reads `loss.data` each micro-step (`packages/train/src/trainer.ts:296`)
- grad norm reads each reduced scalar (`packages/train/src/trainer.ts:351`)
- post-clip safety checks read reduction scalars (`packages/train/src/trainer.ts:420`)

In Helios, reading `.data` on a lazy tensor flushes the compute graph and waits for GPU completion (`packages/helios/src/backend.ts:492`).

Why this hurts:
- Breaks large batched GPU submissions into smaller chunks.
- Adds extra host waits in the middle of a step.

Improvement direction:
- Fewer scalar reads in the hot path.
- Make some safety checks configurable/debug-only.
- Defer scalar reads until after optimizer when possible.

## 3. `gptForward()` allocates recurrent tensors every call (positions + causal mask)

In `packages/model/src/gpt.ts`:
- Position index tensor is rebuilt every forward (`packages/model/src/gpt.ts:117`)
- Causal mask is rebuilt every forward (`packages/model/src/gpt.ts:129`)

On `helios`, `causalMask()` is CPU-generated (`packages/helios/src/backend.ts:1865`) and then uploaded when used by `maskedFill()`.

Why this hurts:
- Repeated CPU allocations and memory writes in the hottest path
- Repeated host→GPU upload for a static tensor shape (same `T`)
- Extra GPU buffer lifetime pressure

Improvement direction:
- Cache causal masks by `T`
- Cache position indices by `(B,T)` or at least `[1,T]` and broadcast/expand
- Prefer backend-native cached GPU tensors for masks/positions on GPU runs

This is a clear, low-risk improvement.

## 4. Parameter traversal and map allocation are repeated in hot paths

`collectParams()` builds a fresh `Map<string, Variable>` every call (`packages/model/src/gpt.ts:209`).

It is called multiple times in a step:
- accumulation scaling (`packages/train/src/trainer.ts:314`)
- grad norm + optimizer prep (`packages/train/src/trainer.ts:328`)

Also used in checkpoints (`packages/train/src/checkpoint.ts:195`), where it is less important.

Why this hurts:
- Repeated JS object/map allocation and string-key iteration
- Extra GC pressure in the step loop

Improvement direction:
- Build a stable parameter list once after model init, e.g. `Array<[name, Variable]>`
- Reuse that list everywhere in trainer/checkpoint
- Keep a `Map` only where name lookup is truly needed

This is a low-risk JS-side optimization with consistent wins.

## 5. Gradient norm + clipping performs multiple extra full gradient passes

Current behavior in `packages/train/src/trainer.ts`:
- Compute norm by `mul + sum` per param (`packages/train/src/trainer.ts:333`)
- Optional clip by `scale` per grad (`packages/train/src/trainer.ts:395`)
- Optional post-clip Inf checks on top tensors (`packages/train/src/trainer.ts:406`)

Why this hurts:
- Extra kernel count and graph nodes per step
- Extra memory traffic over all gradients
- More scalar readbacks and graph flush points

Improvement direction:
- Make expensive safety checks optional (`trace` or `--strictNumerics`)
- Fuse gradient norm and clipping into optimizer/kernel path when using `helios`
- Compute norm less frequently if acceptable (e.g. every N steps) for some experiments

For small models the JS overhead may dominate; for larger models this becomes a memory-bandwidth tax.

## 6. DataLoader allocates fresh input/target arrays every batch

`DataLoader.nextBatch()` allocates new `Int32Array` buffers each call (`packages/train/src/data.ts:61`).

Why this hurts:
- Allocation churn inside the hottest CPU loop
- Increased GC pressure, especially with gradient accumulation

Improvement direction:
- Add reusable batch buffers in `DataLoader` (double-buffer if necessary)
- Optionally return views/wrappers reusing `TensorData` objects too

This is a simple, low-risk CPU optimization.

## 7. Transformer implementation is functional but not kernel-efficient yet

`gptForward()` in `packages/model/src/gpt.ts` is clear and correct, but performance-oriented fusion is limited:
- Q/K/V use separate matmuls (`packages/model/src/gpt.ts:139`)
- attention mask + softmax are separate ops (`packages/model/src/gpt.ts:155`, `packages/model/src/gpt.ts:170`)
- several reshape/transpose/materialization steps per block

Why this hurts:
- High op count and tape size
- More intermediate buffers
- More synchronization pressure from debug/scalar reads

Improvement direction:
- Fused QKV projection
- Fused attention score scaling + mask + softmax
- Fused attention backward (longer-term)
- Residual/add fusion where practical

This is higher complexity but necessary for major throughput gains.

## 8. Eval/checkpoint/sample defaults can materially reduce effective training throughput

The trainer does periodic:
- eval (`packages/train/src/trainer.ts:512`)
- checkpoint save (`packages/train/src/trainer.ts:554`)
- sample generation (`packages/train/src/trainer.ts:563`)

CLI exposes these in `apps/cli/src/commands/train.ts:41`.

Why this matters:
- Users often compare `tok/s` including these pauses
- Sample generation especially triggers extra flush/GC on GPU paths

Improvement direction:
- For perf benchmarking, disable or relax:
  - `evalInterval`
  - `evalIters`
  - `sampleInterval`
- Separate “steady-state train throughput” from “full workflow throughput”

## 9. Checkpoint save path builds one large in-memory buffer

`saveBinary()` in `packages/train/src/checkpoint.ts`:
- collects tensors
- allocates one `Buffer.alloc(totalSize)` (`packages/train/src/checkpoint.ts:69`)
- copies all tensors into it before writing

Why this hurts:
- Large RAM spikes for bigger models/checkpoints
- Pause time at checkpoint boundaries

Improvement direction:
- Stream checkpoint writes (header then tensor chunks)
- Optional background checkpoint compression/write worker

Not a per-step hot path, but meaningful for long runs and larger models.

## Improvement Roadmap (Prioritized)

## Phase 0: Measurement Hygiene (Do First)

Before changing code, standardize measurement so improvements are real.

### Recommended baseline commands

- Steady-state training benchmark:
  - disable eval/sampling for measurement runs
  - use fixed seed
  - warm up for a few steps before comparing

Example knobs (already supported by CLI):
- `--evalInterval` large (or `> iters`)
- `--evalIters=0`
- `--sampleInterval=0`
- `--trace=true` for short profiling runs only (not final throughput runs)

Relevant config parsing: `apps/cli/src/commands/train.ts:41`.

### Track these metrics

- `timing_fwd_ms`, `timing_bwd_ms`, `timing_grad_norm_ms`, `timing_flush_ms`
- `tokens_per_sec`
- `gpu_ops_count`
- `gpu_mem_pool_mb`

If `timing_flush_ms` is large/volatile, prioritize sync/GC policy changes before kernel work.

## Phase 1: Low-Risk, High-Confidence Wins

## 1. Cache causal masks and positional indices

### What to change

Add caches in the model forward path or backend:
- `causalMaskCache: Map<number, TensorData>` keyed by `T`
- `posIndexCache: Map<string, TensorData>` keyed by `${B}x${T}` (or better: cache `[1,T]` and broadcast)

### Why it helps

- Removes repeated CPU tensor creation from `gptForward()`
- Avoids repeated GPU uploads for masks/positions

### Code targets

- `packages/model/src/gpt.ts:117`
- `packages/model/src/gpt.ts:129`
- `packages/helios/src/backend.ts:1865`

### Expected impact

- Small-to-medium per-step improvement
- Larger improvement for shorter sequences/smaller models where overhead dominates

## 2. Reuse parameter traversal structures in trainer

### What to change

After model init, build a stable param list once:
- `const paramEntries = collectParamEntries(params)` (array)

Reuse it for:
- grad accumulation scaling
- grad norm prep
- optimizer prep
- zeroing grads

### Why it helps

- Cuts JS allocation churn
- Reduces string-key map overhead in step loop

### Code targets

- `packages/model/src/gpt.ts:209`
- `packages/train/src/trainer.ts:314`
- `packages/train/src/trainer.ts:328`

### Expected impact

- Modest but reliable; improves GC behavior and consistency

## 3. Reuse DataLoader batch buffers

### What to change

Modify `DataLoader` to maintain internal reusable `Int32Array` buffers for inputs/targets.

Optional:
- return stable `TensorData` wrappers and mutate `.data` contents
- use double-buffering if async consumers are introduced later

### Code targets

- `packages/train/src/data.ts:56`

### Expected impact

- Modest CPU improvement
- Less GC noise, especially with `gradAccumSteps > 1`

## 4. Make expensive gradient safety checks configurable

### What to change

Gate post-clip top-3 Inf checks (`packages/train/src/trainer.ts:406`) behind:
- `trace`
- or a new config flag (e.g. `strictNumerics`)

Keep NaN/Inf checks on loss and grad norm always-on.

### Why it helps

- Removes extra gradient passes and scalar readbacks in normal runs

### Expected impact

- Modest-to-medium depending on model size and clipping frequency

## Phase 2: Training Loop Runtime Policy Improvements (High Impact on Helios)

## 5. Relax per-step `gc()` + `syncGpu()` policy

### Current behavior

Every step:
- `flush()`
- `gc()`
- `syncGpu()` (waits)

See `packages/train/src/trainer.ts:447` and `packages/helios/src/backend.ts:562`.

### What to change

Introduce a sync policy, e.g.:
- `syncEveryNSteps` (default 1 for safety, tuneable)
- `syncWhenPoolMBExceeds`
- `syncWhenPendingDestroysExceeds`

Pseudo-policy:
- `flush()` every step (cheap and keeps graph bounded)
- `gc()` + `syncGpu()` only when memory diagnostics indicate pressure, or every N steps

### Why it helps

- Preserves correctness while reducing forced GPU idle time
- Lets graph batching deliver more benefit

### Risks

- VRAM growth/OOM if thresholds are too loose
- Needs guardrails using existing `gpuMemStats()`

### Expected impact

- Potentially large for GPU training throughput

## 6. Reduce mid-step scalar readbacks

### What to change

Prioritize eliminating reads that break batching:
- micro-loss read every accumulation micro-step (`packages/train/src/trainer.ts:296`)
- grad norm scalar reads across many tensors (`packages/train/src/trainer.ts:351`)

Options:
- only read `microLoss` on last micro-step (or sampled steps)
- compute and log loss every N steps in perf mode
- fuse grad norm into optimizer pre-pass (GPU)

### Why it helps

Each `.data` access on a lazy Helios tensor can flush and wait (`packages/helios/src/backend.ts:498`).

### Expected impact

- Medium-to-large on GPU depending on current graph break frequency

## Phase 3: Backend/Kernel Fusion for Major Gains

## 7. Fused gradient norm + clipping kernel path (Helios)

### What to change

Add a Helios path that:
- computes global grad norm (or partial norms reduced on GPU)
- computes clip coefficient
- scales grads in-place

Potential API shape:
- `backend.clipGradients?(grads: TensorData[], maxNorm: number): { norm: number, clipped: boolean }`

### Why it helps

- Replaces multiple per-parameter ops and host orchestration with fewer GPU passes
- Reduces graph size and scalar reads

### Expected impact

- Medium-to-large for larger models

## 8. Fused QKV projection in `gptForward`

### Current behavior

Three separate projections from the same `ln1Out`:
- `wq`, `wk`, `wv` matmuls (`packages/model/src/gpt.ts:139`)

### What to change

Store a packed projection weight per layer (or temporary packed view):
- `[3*nEmbd, nEmbd]`

Compute one matmul, then slice/reshape into Q/K/V.

### Why it helps

- Reduces matmul launches and intermediate tensors
- Better cache locality / GPU occupancy

### Cost

- Medium (model param format/checkpoint compatibility implications)

### Expected impact

- Medium-to-large depending on sequence/model size

## 9. Fused attention mask + softmax (and later attention backward)

### Current behavior

Attention pipeline is split:
- score matmul
- scale
- clamp
- maskedFill
- softmax

See `packages/model/src/gpt.ts:149` to `packages/model/src/gpt.ts:170`.

### What to change

Backend op(s):
- `maskedSoftmaxCausal(scores)` or `scaledMaskedSoftmax`

Longer term:
- fused attention forward/backward kernel family (FlashAttention-style direction)

### Why it helps

- Fewer intermediate buffers
- Fewer graph nodes and memory passes
- Lower tape size

### Expected impact

- Large for longer sequence lengths (`T`)

## 10. Mixed precision training (f16/bf16 path)

### Current state

Helios exposes `f16Supported` device info (`packages/helios/src/backend.ts:608` / `getDeviceInfo()` nearby) but training tensors/optimizer path are effectively `f32`.

### What to change

Phased approach:
- f16 activations/intermediates
- fp32 master weights + optimizer states
- loss scaling (dynamic)

### Why it helps

- Lower memory bandwidth pressure
- Higher throughput on supported GPUs

### Risks

- Numerical instability
- More complex kernel coverage

### Expected impact

- Potentially very large on GPU, but high implementation complexity

## Phase 4: Structural Runtime Improvements

## 11. Tape and autograd allocation reductions

### Observations

`Tape` allocates entries and arrays per op (`packages/autograd/src/tape.ts:39`, `packages/autograd/src/ops.ts:14`).
Backward often clones/adds gradients (`packages/autograd/src/tape.ts:74`).

### Improvement directions

- Tape entry object pooling
- Specialized backward paths for common op patterns (reduce allocations)
- More in-place accumulation when safe
- Fused composite ops in autograd to reduce tape length

### Expected impact

- Medium, especially for CPU and small/medium GPU models where JS overhead matters

## 12. Streamed/asynchronous checkpoints

### What to change

Replace `Buffer.alloc(totalSize)` checkpoint assembly with streaming writes.

Code target:
- `packages/train/src/checkpoint.ts:66`

Optional:
- checkpoint writer worker thread
- background checkpoint save while training continues (snapshot semantics required)

### Expected impact

- Improves pause time and memory spikes at checkpoint boundaries, not steady-state step speed

## Operational Recommendations (No Code Changes)

Use these when benchmarking throughput:

1. Disable samples during perf runs (`--sampleInterval=0`)
2. Disable eval during perf runs (`--evalIters=0`, large `--evalInterval`)
3. Turn off `trace` except short diagnostic runs
4. Use `gradAccumSteps` to increase effective batch without blowing VRAM, but measure `timing_flush_ms` as accumulation increases
5. Compare:
   - steady-state train throughput
   - full workflow throughput (with eval/checkpoint/sample)

These are all available via `apps/cli/src/commands/train.ts:41`.

## Suggested Implementation Order (Pragmatic)

1. Cache causal masks + position indices
2. Reuse param traversal list and DataLoader batch buffers
3. Add config flag for post-clip safety checks
4. Add sync policy (`gc/syncGpu` every N steps or memory-threshold based)
5. Reduce scalar reads in hot path
6. Fused grad norm + clip kernel path
7. Fused QKV
8. Fused masked softmax / attention kernels
9. Mixed precision

This order gives useful wins early without destabilizing training.

## Validation Plan for Each Change

For each optimization:

1. Run a fixed benchmark configuration (same model/data/seed)
2. Record:
   - `tokens_per_sec`
   - `ms_per_iter`
   - timing breakdown fields
   - peak/avg `gpu_mem_pool_mb`
3. Compare correctness:
   - loss curve shape over first N steps
   - no NaN/Inf regressions
4. Re-test with:
   - `cpu_ref`
   - `helios`
   - small and medium model sizes

Recommended acceptance rule:
- Keep changes only if they improve throughput or substantially reduce variance without harming stability.

## Notes on Existing Strengths (Worth Preserving)

These are already good and should not be regressed:
- GPU graph batching and lazy readback (`packages/helios/src/backend.ts:385`)
- GPU fused cross-entropy forward/backward (`packages/helios/src/backend.ts:1544`)
- GPU AdamW step (`packages/helios/src/backend.ts:2120`)
- Explicit GPU buffer release hooks in autograd/trainer (`packages/autograd/src/tape.ts:56`, `packages/train/src/trainer.ts:266`)
- Built-in step timing instrumentation (`packages/train/src/trainer.ts:481`)

## Related Existing Doc

For lower-level Helios runtime/backend optimization ideas, also see:
- `docs/helios-perf-research.md`

