# Alpha Framework Improvement Audit (Round 2)

Date: 2026-02-25
Repo: `/home/ajax/repos/models/alpha`

## Scope

Second-pass audit after the first round of fixes were reported as implemented.

Focus:
- correctness regressions / remaining bugs
- performance bottlenecks (training, inference, web serving, GPU backend)
- improvements consistent with project philosophy (TS-first; drop to C for GPU/native hot paths)

## Quick Validation (this pass)

Ran:

```bash
npm test -w @alpha/tests
npm run build -w @alpha/inference
npm run build -w @alpha/web
```

Results:
- `@alpha/tests`: 33 tests passed
- `@alpha/inference` build: passed
- `@alpha/web` production build: passed

## Verified Improvements Landed Since Prior Audit

These appear implemented and working in code:

- Broadcast helpers added and reused (`packages/core/src/broadcast.ts`, `packages/autograd/src/ops.ts`, `packages/helios/src/backend.ts:2570`)
- Deterministic dropout RNG + checkpoint replay handling (`packages/model/src/gpt.ts:117`, `packages/model/src/gpt.ts:240`)
- Position index / causal mask caching in GPT forward (`packages/model/src/gpt.ts:100`, `packages/model/src/gpt.ts:210`)
- `DataLoader.nextBatch()` bounds fix (`packages/train/src/data.ts:59`)
- Trainer timing metric split fix (observed in trainer timing fields during inspection)
- `@alpha/inference` fast path integrated into web (`apps/web/src/lib/engine.ts:15`, `apps/web/src/lib/engine.ts:163`)
- Prompt truncation added in web inference paths (`apps/web/src/lib/engine.ts:219`, `apps/web/src/app/v1/chat/completions/route.ts:31`, `apps/web/src/app/api/generate/route.ts:36`)

## Highest-Priority Remaining Bugs / Risks

### 1) Shared mutable inference state is reused across web requests (concurrency corruption)

Severity: **Critical**

`InferenceModel` mixes immutable weights with mutable KV cache + scratch buffers:
- mutable cache/scratch fields: `packages/inference/src/engine.ts:38`
- allocated on model creation: `packages/inference/src/engine.ts:217`

Web code caches a single `LoadedModel` globally and reuses `loaded.inference`:
- singleton cache: `apps/web/src/lib/engine.ts:69`
- `LoadedModel.inference`: `apps/web/src/lib/engine.ts:58`
- `ensureModel()` returns shared instance: `apps/web/src/lib/engine.ts:188`

Request handlers mutate shared state via `resetCache()`, `prefill()`, `decodeStep()`:
- `apps/web/src/lib/engine.ts:304`, `apps/web/src/lib/engine.ts:315`, `apps/web/src/lib/engine.ts:347`, `apps/web/src/lib/engine.ts:378`
- `apps/web/src/app/v1/chat/completions/route.ts:38`, `apps/web/src/app/v1/chat/completions/route.ts:80`
- `apps/web/src/app/api/generate/route.ts:43`, `apps/web/src/app/api/generate/route.ts:54`

Why this is real now:
- streaming routes explicitly yield with `setImmediate(...)`, allowing interleaving between concurrent requests (`apps/web/src/lib/engine.ts:381`, `apps/web/src/app/v1/chat/completions/route.ts:83`)
- one request can reset/overwrite KV cache and scratch buffers mid-generation of another

Recommended fix:
- split `InferenceModel` into:
  - `InferenceWeights` (immutable, shared)
  - `InferenceSession` (KV cache + scratch, per request or pooled)
- add a small session pool keyed by model id/config to preserve zero-allocation decode

### 2) Empty-prompt path can call `prefill()` with zero tokens (invalid offsets / undefined behavior)

Severity: **High**

`prefill()` assumes `tokens.length > 0`:
- `const T = tokens.length` with no guard: `packages/inference/src/engine.ts:268`
- computes `lastOff = (T - 1) * nEmbd`: `packages/inference/src/engine.ts:375`
- reads last token LN and logits from that offset: `packages/inference/src/engine.ts:377`, `packages/inference/src/engine.ts:381`

Web routes allow empty prompt strings and immediately call `prefill()`:
- `/api/generate` defaults prompt to `""`: `apps/web/src/app/api/generate/route.ts:20`
- then calls `prefill(...)`: `apps/web/src/app/api/generate/route.ts:44`
- chat route can build empty prompt from empty messages: `apps/web/src/app/v1/chat/completions/route.ts:29`, `apps/web/src/app/v1/chat/completions/route.ts:39`

Tokenizers can return zero-length encodings for empty text:
- char tokenizer: `packages/tokenizers/src/char.ts:66`
- word tokenizer: `packages/tokenizers/src/word.ts:79`
- BPE tokenizer path can also return empty result for empty text (`n=0`): `packages/tokenizers/src/bpe.ts:200`

Recommended fix:
- add explicit API-level handling for empty prompt:
  - reject with 400, or
  - synthesize a start token if tokenizer defines one
- hard-guard `prefill()` with `if (T <= 0) throw`

### 3) `@alpha/web` lazy init race / stuck-on-failure pattern remains

Severity: **High**

`_initialized = true` is set before awaited work:
- `apps/web/src/lib/init.ts:12`
- `apps/web/src/lib/init.ts:13`

If `initEngine()` throws, future calls may incorrectly return early and skip init forever.

Recommended fix:
- replace boolean with an in-flight promise (`initPromise`)
- only mark initialized after successful completion
- reset promise on failure

### 4) Helios native addon stack overflow / memory corruption risk for `bufCount > 32`

Severity: **High**

The native code reads `bufCount` from JS, stores handles in fixed stack arrays of 32, but loops/writes using `bufCount` without rejecting larger arrays:
- dispatch: `packages/helios/native/helios_vk.c:1568`, `packages/helios/native/helios_vk.c:1625`, `packages/helios/native/helios_vk.c:1660`
- batch dispatch: `packages/helios/native/helios_vk.c:1732`, `packages/helios/native/helios_vk.c:1782`, `packages/helios/native/helios_vk.c:1799`
- gpu timing path: `packages/helios/native/helios_vk.c:1883`, `packages/helios/native/helios_vk.c:1922`, `packages/helios/native/helios_vk.c:1939`

This can corrupt stack memory via `writes[i]`, `bufInfos[i]`, `memcmp/memcpy(...)` lengths.

Recommended fix:
- hard reject `bufCount == 0 || bufCount > 32` before any stack-array use
- validate each buffer handle before indexing `buffers[bufSlots[i]]`

### 5) Helios async submit ignores Vulkan submit errors

Severity: **High**

`submitCmdBufAsync()` ignores `vkQueueSubmit` return code:
- `packages/helios/native/helios_vk.c:680`
- `packages/helios/native/helios_vk.c:696`

On failure, code returns a timeline value that was never signaled, causing hangs / invalid waits.

Recommended fix:
- return `{ok, timeline}` style status or sentinel `0` and surface an N-API error
- do not advance observable timeline state on failed submit

### 6) `purgeBufferPools()` can destroy pending GPU buffers before they are safe

Severity: **High**

`pendingDestroys` explicitly tracks `readyValue` timeline safety:
- `packages/helios/src/backend.ts:385`
- `packages/helios/src/backend.ts:391`
- `packages/helios/src/backend.ts:406`

But `purgeBufferPools()` destroys all pending handles immediately:
- `packages/helios/src/backend.ts:651`
- `packages/helios/src/backend.ts:669`
- `packages/helios/src/backend.ts:671`

Recommended fix:
- call `syncGpu()` first, or
- wait to max pending timeline before destroying pending handles

### 7) FlashAttention training path still skips attention-dropout parity

Severity: **Medium-High** (training behavior mismatch)

Flash path:
- `packages/model/src/gpt.ts:140`
- `packages/model/src/gpt.ts:144`

Standard path applies attention dropout before `@V`:
- `packages/model/src/gpt.ts:166`
- `packages/model/src/gpt.ts:167`
- `packages/model/src/gpt.ts:168`

Effect:
- training with `ctx.backend.flashAttention` changes regularization behavior vs CPU/reference path

Recommended fix:
- add dropout support to fused flash attention (preferred), or
- disable flash path when `training && config.dropout > 0`

### 8) Helios CPU fallback `matmul` still lacks batch broadcast parity

Severity: **Medium**

Helios CPU fallback assumes identical batch prefixes:
- `packages/helios/src/backend.ts:2582`
- uses `bOff = batch * K * N` directly: `packages/helios/src/backend.ts:2593`

Reference CPU backend supports batch broadcasting:
- `packages/tensor/src/cpu_ref.ts:188`
- `packages/tensor/src/cpu_ref.ts:231`

Recommended fix:
- port/reference the broadcast batch-indexing logic from `cpu_ref`

## High-Impact Performance Improvements (TS-First Friendly)

### A) Split immutable weights from mutable inference sessions (also fixes concurrency)

Impact: **Very high** for correctness + throughput under concurrent serving

Today, every loaded model has exactly one mutable cache/scratch bundle (`packages/inference/src/engine.ts:29`), reused globally in web (`apps/web/src/lib/engine.ts:69`).

Proposed API shape:
- `prepareInferenceWeights(config, params) -> InferenceWeights`
- `createInferenceSession(weights) -> InferenceSession`
- `prefill(session, tokens)`
- `decodeStep(session, token, pos)`
- optional `SessionPool.acquire()/release()`

Benefits:
- no request races
- no forced `resetCache()` on shared object
- clearer scaling path to batching / worker threads

### B) Stop zero-filling full KV cache on every request

Impact: **High** for latency and CPU use

Current behavior:
- `resetCache()` fills all K/V caches every request (`packages/inference/src/engine.ts:237`)

This is O(`nLayer * nHead * blockSize * headDim`) memory traffic even for short prompts.

Recommended fix:
- track `cacheLen` (logical valid prefix)
- overwrite only positions actually used
- optionally zero only on debug/test paths

### C) Pool or preallocate prefill scratch buffers

Impact: **High** for allocation pressure / GC under serving load

`prefill()` allocates many large arrays each call:
- `x`, `lnBuf`, `Q`, `K`, `V`, `attnOut`, `scores`, `proj`, `mlpH`: `packages/inference/src/engine.ts:270`
- allocates `lastLn` every call: `packages/inference/src/engine.ts:376`

Recommended fix:
- move prefill scratch into `InferenceSession` (resizable)
- reuse buffers across requests
- at minimum reuse `lastLn`

### D) Replace top-k full sort + allocation with selection / heap, and add greedy path

Impact: **High** for decode throughput at large vocab sizes

Current `sampleFromLogits()`:
- allocates `sorted` every call: `packages/inference/src/engine.ts:533`
- full sort O(V log V): `packages/inference/src/engine.ts:535`
- computes `1 / temperature` with no guard: `packages/inference/src/engine.ts:525`

Recommended fix:
- if `temperature <= 0`, do argmax (greedy)
- use quickselect / fixed-size min-heap for top-k threshold (O(V) / O(V log k))
- add a second scratch buffer on model/session to avoid per-step allocation

### E) Add strict input validation to inference API (prevents late NaNs / OOB behavior)

Impact: **High** (reliability, debuggability)

Missing validations:
- `prefill()` no `T > 0` and no `T <= blockSize` guard (`packages/inference/src/engine.ts:268`)
- `decodeStep()` no `0 <= pos < blockSize` guard (`packages/inference/src/engine.ts:398`, `packages/inference/src/engine.ts:421`, `packages/inference/src/engine.ts:439`)
- `prepareInferenceModel()` no checkpoint parameter existence/shape checks (`packages/inference/src/engine.ts:180`, `packages/inference/src/engine.ts:193`)

Recommended fix:
- fail fast with explicit messages and shape details
- keep validation in `prepareInferenceModel()` and public entrypoints

### F) Web model loading: add in-flight de-duplication and multi-model cache

Impact: **Medium-High** for cold-start latency and model-switching UX

Current state:
- single cached model: `apps/web/src/lib/engine.ts:69`
- no in-flight load promise memoization: `apps/web/src/lib/engine.ts:188`

Consequences:
- concurrent cold requests can duplicate checkpoint loads
- switching between models thrashes and reloads

Recommended fix:
- `Map<modelId, Promise<LoadedModel>>` for in-flight loads
- LRU cache of loaded immutable weights (sessions pooled separately)

### G) Web metrics scan reads entire `metrics.jsonl` just to get the last line

Impact: **Medium** (startup latency on long runs)

Current behavior:
- `fs.readFileSync(...).trim().split("\n")`: `apps/web/src/lib/engine.ts:122`

Recommended fix:
- tail-read last N KB and parse last non-empty line
- or track latest metrics in a sidecar summary file during training

### H) Checkpoint save/load path still has avoidable memory spikes

Impact: **Medium-High** for larger models

Save path:
- allocates a single full-file buffer (`Buffer.alloc(totalSize)`): `packages/train/src/checkpoint.ts:69`

Load path:
- tensor extraction copies bytes via `ArrayBuffer.slice(...)`: `packages/train/src/checkpoint.ts:102`

Recommended fix:
- stream writes (header + tensor chunks) instead of assembling one giant buffer
- add header padding to 4-byte alignment so future loaders can create direct `Float32Array` views without copies

### I) Token cache write path doubles memory via `Buffer.concat`

Impact: **Medium** on large token caches

Current behavior:
- builds `Buffer.concat([header, tokenBuf])`: `packages/train/src/data.ts:251`, `packages/train/src/data.ts:254`

Recommended fix:
- write header then token buffer sequentially using a file handle / stream

## Architecture Improvements (Next Stage)

These are not immediate bugs, but they materially improve the framework’s trajectory.

### 1) Add inference parity tests (fast engine vs training/reference path)

Gap:
- no tests currently cover `@alpha/inference` behavior (no matches in `packages/tests/src` during search)

Add tests for:
- prefill logits parity vs reference forward on fixed weights/prompt
- decode step parity after KV cache buildup
- edge cases: empty prompt, long prompt truncation, `temperature=0`, `topk` boundaries

### 2) Add concurrency tests for web streaming generation

Reason:
- current biggest production bug is concurrency/race behavior

Test idea:
- two concurrent streams against same model
- assert deterministic isolation with per-session caches

### 3) Add benchmark harness + regression thresholds

Reason:
- framework is now performance-sensitive across CPU inference / GPU training

Track:
- prefill tok/s
- decode tok/s
- checkpoint save/load time and peak RSS
- GPU kernel timings (Helios)

This fits TS-first philosophy well (benchmark orchestration in TS, GPU kernels/native only where needed).

### 4) Introduce CPU inference weight packing / quantization as opt-in modules

Reason:
- biggest future CPU serving wins are memory bandwidth and cache locality

TS-first direction:
- start with blockwise int8 quantization + dequant matvec in TS for correctness + API
- later add optional native/WebAssembly SIMD backends if profiling proves necessary

## Recommended Fix Order (Pragmatic)

1. Split inference weights/session state and patch web handlers to use per-request sessions (fixes concurrency + enables pooling).
2. Add inference input validation (`prefill/decodeStep`) and handle empty prompts explicitly in web/CLI.
3. Fix Helios native `bufCount` bounds checks and `submitCmdBufAsync()` error handling.
4. Fix `purgeBufferPools()` pending-destroy safety.
5. Fix `sampleFromLogits()` (`temperature<=0`, top-k selection, remove per-step allocation).
6. Pool prefill buffers and replace full KV zero-fill with logical cache length.
7. Address FlashAttention dropout parity (fused dropout or guard).
8. Stream checkpoint/token-cache writes to remove large memory spikes.
9. Add inference parity + concurrency tests.

## Notes on Philosophy Alignment

These recommendations preserve the current design philosophy:
- core framework logic stays in TypeScript
- correctness and orchestration improvements are mostly TS-only
- C/native changes are limited to GPU/native safety and kernel submission reliability (`helios_vk.c`)

