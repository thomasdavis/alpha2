# PRD: Alpha to GPT-2 Class on L4

**Date:** 2026-03-07
**Project:** Alpha
**Objective:** Train a **GPT-2-class 124M parameter model** that produces coherent, formatted, non-gibberish text using the **Alpha from-scratch stack** (TypeScript + Vulkan + Helios) on an **L4-based path**, with minimal architectural compromise and no framework switch.

---

## 1. Mission

We are not proving that PyTorch can train GPT-2. We are proving that **Alpha can do it**.

The target is a **real 12-layer, 768-dim, 12-head transformer** in the GPT-2 124M class, trained end-to-end on the Alpha stack, producing coherent English text and demonstrating stable large-model training behavior.

This is not a toy milestone. This is the point where Alpha stops being "interesting infrastructure" and becomes a **credible training system**.

The L4 is not a reason to lower ambition. It is a constraint to engineer around.

---

## 2. Outcome We Want

By the end of this effort, Alpha should be able to:

- initialize, train, checkpoint, resume, and sample from a **124M GPT-2-class model**
- run stably enough on the current stack that long training is practical
- produce coherent text completions rather than collapse, repetition, or gibberish
- demonstrate that Alpha's optimizer, allocator, kernels, and training loop can sustain a serious transformer workload
- establish a repeatable path for future scaling beyond GPT-2 class

This is the first major "Alpha can really train" milestone.

---

## 3. Current State

### What Already Works

Alpha already has the essential skeleton of a real training system:

| Component | Status | Code Location |
|-----------|--------|---------------|
| Tokenizer pipeline (BPE) | Working | `packages/tokenizers/src/bpe.ts:113-413` |
| Model init (GPT decoder-only) | Working | `packages/model/src/gpt.ts:73-151` |
| Forward pass (attention + MLP + residual) | Working | `packages/model/src/gpt.ts:344-427` |
| Backward pass (autograd tape) | Working | `packages/autograd/src/tape.ts` |
| AdamW optimizer (CPU + GPU kernel) | Working | `packages/train/src/optimizers.ts:21-200`, `packages/helios/src/kernels/optimizer.ts:29` |
| Checkpoint save/load (binary format) | Working | `packages/train/src/checkpoint.ts:30-186` |
| Sampling (top-k, top-p, greedy) | Working | `packages/train/src/sample.ts:21-201` |
| Fleet deploy/train/resume/logs | Working | `apps/cli/src/commands/fleet.ts` |
| Cosine LR with warmup | Working | `packages/train/src/trainer.ts:1122-1130` |
| Dynamic loss scaling + NaN defense | Working | `packages/train/src/trainer.ts:1043-1072, 1219-1268, 1473-1490` |
| Flash Attention kernel | Working | `packages/helios/src/kernels/attention-coop.ts` |
| Activation checkpointing | Wired but OFF by default | `packages/model/src/gpt.ts:393-403`, `packages/autograd/src/checkpoint.ts:34-87` |
| Weight decay exclusion (`noDecayNames`) | Wired but has a bug | `apps/cli/src/commands/train.ts:318-339` |
| Concordance domain (GPT-2 target) | Defined but incomplete defaults | `packages/core/src/domains.ts:116-139` |

This matters because we are **not starting from zero**. We are **pushing an existing system through a scale barrier**.

### What Is Proven So Far

Small models already train and converge:

| Config | Params | Throughput | Loss | Status |
|--------|--------|------------|------|--------|
| 4L/128d/4h, block=256, batch=16, accum=2, f32 | 1.85M | 65K tok/s | Converges | Stable |
| 6L/256d/8h, block=256, batch=4, accum=2, f32 | 6.84M | 30K tok/s | Reaches ~5.0 | Needs lr<=1e-4 |
| 6L+/256d+, batch=8+, block=512 | - | - | - | OOM |

The remaining work is not "can Alpha train anything?" — it is: **can Alpha cross from small-model success into GPT-2-class training without collapsing under memory, allocator, or scale-pathology issues?**

---

## 4. Target Spec

### Model

| Parameter | Target | Code Reference |
|-----------|--------|----------------|
| nLayer | 12 | `packages/core/src/domains.ts:129` |
| nEmbd | 768 | `packages/core/src/domains.ts:130` |
| nHead | 12 | `packages/core/src/domains.ts:131` |
| ffnDim | 3072 (4 * 768) | `packages/model/src/gpt.ts:83-85` |
| blockSize | 512 initially, 1024 as stretch | `packages/core/src/domains.ts:128` |
| vocabSize | 64,000 | bpe-64k tokenizer |
| activation | GELU | Default in `initGPT` |
| params | ~124M (with weight tying) | Currently ~174M without tying |

### Training Defaults

| Parameter | Target | Rationale |
|-----------|--------|-----------|
| lr | 6e-4 | GPT-2 / nanoGPT standard |
| lrMin | 6e-5 | 10x decay floor |
| warmupIters | 2000 | ~1% of total steps |
| beta1 | 0.9 | Standard |
| beta2 | 0.95 | Standard for transformers |
| weightDecay | 0.1 | Applied to weight matrices only |
| gradClip | 1.0 | Standard |
| precision | Mixed if stable, f32 fallback | |
| blockSize | 512 first | |
| gradAccumSteps | Tuned to maximize effective batch | |
| activationCheckpointing | ON | Required for 12L on L4 |

### Quality Bar

Success means the model:
- emits coherent English text
- continues prompts in a stable and syntactically sane way
- does not instantly devolve into loops or junk
- respects basic formatting patterns in the data
- clearly improves over the current small-model outputs
- reaches a validation-loss regime plausibly in GPT-2-class territory

---

## 5. Strategic Position

The core idea is simple:

> We do not need perfect conditions. We need a disciplined path.

An L4 is enough to make this mission real provided we:

1. remove obvious model waste
2. fix optimizer correctness
3. treat memory as a first-class engineering problem
4. use gradient accumulation aggressively
5. enable activation checkpointing
6. train on enough data
7. keep the run stable and resumable

This effort is less about theoretical FLOPs and more about **stack discipline**.

---

## 6. L4 Memory Feasibility Analysis

### The Numbers

**124M model with weight tying, f32:**

| Component | Size | Notes |
|-----------|------|-------|
| Parameters | ~496 MB | 124M * 4 bytes |
| Optimizer state (m + v) | ~992 MB | 2 * 496 MB (AdamW) |
| Gradients | ~496 MB | Same as params |
| Forward activations (1 micro-batch) | ~300-500 MB | Depends on batch/block |
| Buffer + output pool cache | ~500 MB - 1 GB | Reusable between steps |
| **Total** | **~2.8-3.5 GB** | Well within L4 24GB |

**Key insight: raw VRAM is NOT the bottleneck. Allocation COUNT is.**

### The Real Constraint: Driver Allocation Limit

The L4 Vulkan driver has a hard limit of ~5500 concurrent `vkAllocateMemory` calls. Each tensor op creates 1 allocation.

| Pass | Estimated Allocations |
|------|-----------------------|
| Single forward (12L) | ~90-120 |
| Single backward (12L, no checkpointing) | ~270-480 |
| Single forward+backward | ~360-600 |
| With grad accum=4 (no syncGpu) | ~1440-2400 |

**Without mitigation:** 4 accumulation steps hit 1400+ live allocs. The slab allocator handles temporaries (`deviceTempPool`), but **persistent model params bypass the slab entirely** — each parameter buffer is an individual `vkAllocateMemory` call.

The critical slab bypass condition (`packages/helios/native/helios_vk.c:2071`):
```c
int slabCompatible = (useHostPool || temporary)  // persistent device-local → 0
  ? (memReq.memoryTypeBits & (1u << pool->memoryTypeIdx)) != 0
  : 0;
```

**Persistent device-local buffers (model params, optimizer state) = individual allocations. Not slab-managed.**

### Required Mitigations

| Mitigation | Impact | Status |
|------------|--------|--------|
| `syncGpu()` after every micro-step | Flushes deferred releases between accumulation steps | Already wired (`trainer.ts:1245-1248`) but only when `accumSteps > 1` |
| Activation checkpointing | Saves ~20-30% allocs + 60% activation memory | Fully wired, default OFF. Must turn ON |
| Weight tying | Removes ~50M params = fewer allocs + less optimizer state | NOT done yet |
| Conservative batch size | batch=1 or 2 with high accumulation | Config-only |
| Pool trimming at soft cap | Reclaims pool entries when live allocs > 8000 | Already wired (`backend.ts:487-511`) |

**Bottom line:** 124M on L4 is feasible with: weight tying + activation checkpointing ON + syncGpu every micro-step + batch=1-2 + high gradient accumulation. VRAM has headroom. Allocation count is the enemy.

---

## 7. Major Gaps to Close

## P0 — Required for Success

---

### 7.1 Weight Tying (wte == lmHead)

**Current code** — `initGPT` creates two separate matrices:

```ts
// packages/model/src/gpt.ts:78
const wte = initWeight(backend, rng, [vocabSize, nEmbd], std);
// packages/model/src/gpt.ts:148
const lmHead = initWeight(backend, rng, [vocabSize, nEmbd], std);
```

With vocabSize=64000 and nEmbd=768: **each matrix = 64000 * 768 * 4 = 188 MB**. Two copies = 376 MB of parameters. Plus optimizer state (m + v) for both = another 752 MB. Total waste: **~1.1 GB**.

`collectParamEntries` lists both as separate entries (`gpt.ts:435,437`):
```ts
entries.push(["wte", params.wte]);
// ...
entries.push(["lmHead", params.lmHead]);
```

The optimizer maintains duplicate `m` and `v` buffers and applies separate updates.

**Required changes:**

**File 1: `packages/model/src/gpt.ts`**

Line 148 — tie lmHead to wte:
```ts
// BEFORE:
const lmHead = initWeight(backend, rng, [vocabSize, nEmbd], std);
// AFTER:
const lmHead = wte;  // weight tying: share embedding with output projection
```

Line 437 — remove lmHead from param collection:
```ts
// REMOVE this line:
entries.push(["lmHead", params.lmHead]);
```

**File 2: `packages/train/src/checkpoint.ts`**

In `restoreParams` (line 220-250) — add backward compat for old checkpoints with separate lmHead:
```ts
// After the wq/wk/wv compat block, add:
// Backward compat: old checkpoints have separate lmHead tensor.
// With weight tying, lmHead == wte, so lmHead data should be ignored
// (wte already restored above). If checkpoint has lmHead but no wte,
// fall back to loading lmHead into wte.
if (name === "wte" && !saved) {
  const lmHeadSaved = checkpointParams["lmHead"];
  if (lmHeadSaved) {
    const arr = variable.data.data as Float32Array;
    for (let i = 0; i < arr.length; i++) arr[i] = lmHeadSaved.data[i];
  }
}
```

In `buildCheckpointState` (line 194-217) — no change needed because `collectParamEntries` won't include `lmHead` anymore.

**File 3: `apps/cli/src/commands/train.ts`**

Line 324 — remove `lmHead.weight` from noDecayNames (it won't exist as a param anymore):
```ts
// REMOVE:
noDecayNames.add("lmHead.weight");
```

Note: this line is already a bug — the actual param name is `"lmHead"`, not `"lmHead.weight"`. So it was never matching anyway.

**Outcome:** ~50M fewer params, ~1.1 GB less VRAM, fewer allocations, better convergence.

---

### 7.2 Fix noDecayNames Bug

**Current code** — `apps/cli/src/commands/train.ts:318-339`:

```ts
const noDecayNames = new Set<string>();
noDecayNames.add("wte");
noDecayNames.add("wpe");
noDecayNames.add("lnF.weight");
noDecayNames.add("lnF.bias");
noDecayNames.add("lmHead.weight");  // BUG: actual param name is "lmHead"
for (let i = 0; i < modelConfig.nLayer; i++) {
  noDecayNames.add(`layer.${i}.ln1.weight`);
  noDecayNames.add(`layer.${i}.ln1.bias`);
  noDecayNames.add(`layer.${i}.ln2.weight`);
  noDecayNames.add(`layer.${i}.ln2.bias`);
}
```

**Bug:** The param names from `collectParamEntries` (`gpt.ts:437`) are `"lmHead"`, not `"lmHead.weight"`. So weight decay IS being applied to lmHead (the no-decay exclusion never matches).

But with weight tying (7.1), lmHead disappears from the param list entirely, so this bug becomes moot. Just clean it up.

**Additional missing exclusions:** Attention biases (`bqkv`, `bo`) are not in the no-decay set. Check if these exist:

```ts
// packages/model/src/gpt.ts:34-41 (LayerParams interface)
attn: {
  wqkv: Variable;
  wo: Variable;
  bqkv?: Variable;  // Optional attention biases
  bo?: Variable;
};
```

If biases exist, they should be excluded from decay. Currently they are NOT in the no-decay set.

**Required change** — `apps/cli/src/commands/train.ts:318-339`:

```ts
const noDecayNames = new Set<string>();
noDecayNames.add("wte");
noDecayNames.add("wpe");
noDecayNames.add("lnF.weight");
noDecayNames.add("lnF.bias");
// lmHead removed (weight tying means it's the same as wte)
for (let i = 0; i < modelConfig.nLayer; i++) {
  noDecayNames.add(`layer.${i}.ln1.weight`);
  noDecayNames.add(`layer.${i}.ln1.bias`);
  noDecayNames.add(`layer.${i}.ln2.weight`);
  noDecayNames.add(`layer.${i}.ln2.bias`);
  // Attention biases (if present)
  noDecayNames.add(`layer.${i}.attn.bqkv`);
  noDecayNames.add(`layer.${i}.attn.bo`);
}
```

**Also required:** Log the decay groups at startup so we never have to guess:

```ts
const decayParams: string[] = [];
const noDecayParams: string[] = [];
for (const [name] of collectParamEntries(params)) {
  (noDecayNames.has(name) ? noDecayParams : decayParams).push(name);
}
console.log(`decay params (${decayParams.length}): ${decayParams.join(", ")}`);
console.log(`no-decay params (${noDecayParams.length}): ${noDecayParams.join(", ")}`);
```

**Outcome:** Correct optimization. Embeddings and norms stop getting weight decay. Visible audit trail.

---

### 7.3 Enable Activation Checkpointing by Default for Large Models

**Current code** — activation checkpointing is fully implemented but opt-in:

Forward pass (`packages/model/src/gpt.ts:393-403`):
```ts
if (activationCheckpointing && training) {
  const savedCounter = dropoutRng?.saveCounter();
  x = checkpoint(ctx, (innerCtx, inp) => {
    if (dropoutRng && savedCounter !== undefined) dropoutRng.restoreCounter(savedCounter);
    const innerCtxWithRng = { ...innerCtx, dropoutRng };
    const f32Inp = mixedPrecision ? castToF32(innerCtxWithRng, inp) : inp;
    return transformerBlock(innerCtxWithRng, f32Inp, layer, config, B, T, mask, training);
  }, x);
}
```

Trainer invocation (`packages/train/src/trainer.ts:1174`):
```ts
const { loss } = gptForward(activeModelConfig, params, backend, trainTape,
  batch.inputs, batch.targets, true,
  !!deps.activationCheckpointing,  // <-- opt-in boolean
  !!deps.mixedPrecision, ...);
```

**Impact on 124M:** Without checkpointing, a 12-layer backward pass stores all intermediate activations across all layers = massive allocation count and VRAM pressure. With checkpointing, only per-layer outputs are stored; intermediates are recomputed during backward. This trades ~33% more compute for ~60% less activation memory and ~20-30% fewer live allocations.

**Required change:**

In the CLI train command, auto-enable checkpointing when nLayer >= 8:
```ts
// apps/cli/src/commands/train.ts (where activationCheckpointing is set)
const activationCheckpointing = boolArg(kv, "activationCheckpointing")
  ?? (modelConfig.nLayer >= 8);  // auto-enable for large models
```

**Outcome:** 124M fits comfortably in L4 allocation budget. ~33% throughput cost but dramatically more stable.

---

### 7.4 Update Concordance Domain Defaults

**Current code** — `packages/core/src/domains.ts:116-139`:

```ts
concordance: {
  // ...
  modelDefaults: {
    blockSize: 1024,
    nLayer: 12,
    nEmbd: 768,
    nHead: 12,
  },
  trainDefaults: {
    tokenizer: "bpe-64k",
    lr: 6e-4,
    batchSize: 4,
    gradClip: 1.0,
    // MISSING: lrMin, warmupIters, beta2, gradAccumSteps, packed, etc.
  },
}
```

**Problem:** Missing critical training defaults. When these fall back to `defaultTrainConfig` in `packages/core/src/types.ts`, they may not match GPT-2 recipes.

**Required change:**

```ts
trainDefaults: {
  tokenizer: "bpe-64k",
  lr: 6e-4,
  lrMin: 6e-5,
  warmupIters: 2000,
  beta2: 0.95,
  eps: 1e-8,
  weightDecay: 0.1,
  batchSize: 2,          // conservative for L4
  gradAccumSteps: 16,    // effective batch = 2*16*512 = 16K tokens
  gradClip: 1.0,
  packed: true,
  sampleInterval: 200,
  evalInterval: 200,
  spikeThreshold: 0,     // disable spike recovery (interferes with training)
},
```

---

### 7.5 Large Real Dataset

A GPT-2-class model cannot be trained into quality on a tiny corpus.

**Current data:** `super_chat.txt` (91MB, ~25M tokens with BPE-64k). That is ~0.2 tokens per parameter for a 124M model. GPT-2 used ~80 tokens/param.

**Minimum viable:** ~500M-1B tokens. That requires ~2-4 GB of clean text.

**Candidate corpora:**
- OpenWebText2 (~17B tokens)
- FineWeb subsets
- The Pile subsets
- RedPajama subsets

**Data requirements before main run:**
1. Tokenize with bpe-64k and cache as `.tokens` binary
2. Verify train/val split integrity
3. Decode random token windows and inspect
4. Confirm no garbage concentration

**Data loading check:** `packages/train/src/data.ts:18-30` uses `Int32Array` for tokens. Verify it handles files >2GB (Node.js `Buffer.alloc` limit is ~2GB; may need streaming for very large datasets).

---

## P1 — Very Important

---

### 7.6 Mixed Precision Recovery

Mixed precision (f16 forward activations, f32 params/optimizer) would:
- halve inter-layer activation memory
- double effective batch size
- improve throughput 30-50%

**Current issue:** fp16 gradient overflow in backward pass. Even with dynamic loss scaling (initial 128.0, halving on NaN), grad norms explode to trillions.

The coop matmul pause during backward (`trainer.ts:1199-1216`) suggests matmul precision is part of the problem.

**Root cause investigation:**
1. Which op produces the first inf/nan? Add per-op overflow detection in backward
2. Is `crossEntropy` backward producing f16 intermediates that overflow?
3. Is the attention softmax backward losing precision at scale?
4. Is loss scale 128.0 too high for 124M-scale gradients?

**Policy:** Aggressively pursue mixed precision, but do NOT let it block the mission. Stable f32 training beats elegant f16 instability.

---

### 7.7 Allocation Discipline

Beyond the structural fixes (tying, checkpointing), reduce avoidable allocation churn:

**Key pressure points:**

1. **Per-step tensor creation in data loader** — `DataLoader.nextBatch()` (`data.ts`) should reuse batch buffers (it already has a `batchRing` — verify it works)

2. **Gradient accumulation sync** — `syncGpu()` after every micro-step (`trainer.ts:1245-1248`) is essential. Verify it fires for ALL accumulation configs, not just `accumSteps > 1`

3. **Output pool sizing** — Current limits (`backend.ts:462-465`):
   ```
   LIVE_ALLOC_SOFT_CAP = 8000
   LIVE_ALLOC_HARD_CAP = 10000
   ```
   For 124M, consider lowering soft cap to 6000 to trigger reclaim earlier:
   ```bash
   HELIOS_LIVE_ALLOC_SOFT_CAP=6000
   ```

4. **Persistent param alloc count** — With 12 layers, each having ~7 param tensors + 2 LN params = ~108 persistent allocations. Plus wte + wpe + lnF = ~111 total. Each gets individual `vkAllocateMemory`. This is acceptable (~2% of the 5500 budget).

---

### 7.8 Gradient Accumulation as Primary Scaling Tool

The L4 path depends on accumulation discipline. Physical batch must be tiny (1-2) to stay within allocation budget. Effective batch comes from accumulation.

**Target effective batch:**

```
batch=2 * gradAccumSteps=16 * blockSize=512 = 16,384 tokens/step
```

This is smaller than GPT-2's ~480K tokens/step, but sufficient for a first milestone. Can increase by:
- reducing block to 256 and raising batch
- raising accumulation steps (more wall-clock time per step)

**The math on training time:**

At 30K tok/s (current 6.84M model rate on L4):
- 1B tokens / 30K tok/s = ~9.3 hours
- 500M tokens = ~4.6 hours

At 10K tok/s (estimated for 124M, conservative):
- 1B tokens / 10K tok/s = ~27.8 hours
- 500M tokens = ~13.9 hours

**Feasible in 72 hours** even at pessimistic throughput estimates.

---

## P2 — Valuable but Secondary

### 7.9 Better Eval Harness

We need more than vibes:
- Validation loss every `evalInterval` steps (already wired)
- Perplexity tracking (= exp(val_loss))
- Fixed prompt battery for human inspection
- Sample logging to Discord

### 7.10 Run Schedule Semantics

If training extends across resumes, LR schedule must remain intentional:

```ts
// packages/train/src/trainer.ts:1067
const decayDenom = Math.max(1, totalIters - warmup);
```

If `totalIters` changes across resumes, the cosine schedule shifts. Consider making total schedule length configurable independently.

---

## 8. Execution Plan

### Phase 0 — Make the 124M Path Correct (Hours 0-4)

**Goal:** Eliminate waste, fix correctness, make the architecture GPT-2-ready.

| # | Task | File(s) | Lines | Impact |
|---|------|---------|-------|--------|
| 1 | Implement weight tying (lmHead = wte) | `packages/model/src/gpt.ts` | 148, 437 | -50M params, -1.1GB VRAM |
| 2 | Add checkpoint backward compat for tied weights | `packages/train/src/checkpoint.ts` | 220-250 | Old checkpoints still load |
| 3 | Fix noDecayNames bug (lmHead.weight -> remove) | `apps/cli/src/commands/train.ts` | 324 | Correct optimizer grouping |
| 4 | Add attention bias exclusions to noDecayNames | `apps/cli/src/commands/train.ts` | 325-329 | Complete no-decay coverage |
| 5 | Add startup log for decay/no-decay groups | `apps/cli/src/commands/train.ts` | After 339 | Audit trail |
| 6 | Auto-enable activation checkpointing for nLayer>=8 | `apps/cli/src/commands/train.ts` | Where flag is set | Fit 12L on L4 |
| 7 | Update concordance domain with full training defaults | `packages/core/src/domains.ts` | 132-137 | Complete config |
| 8 | Smoke test: 100 steps at 4L/128d to verify nothing broke | Local | - | Regression gate |

**Deliverable:** A codebase that is architecturally correct enough to launch a serious run.

---

### Phase 1 — Establish the L4 Scaling Path (Hours 4-12)

**Goal:** Prove the training loop can carry 124M on L4.

| # | Task | Details |
|---|------|---------|
| 1 | Launch 12L/768d/12h with activation checkpointing ON | batch=1, accum=8, block=512, f32 |
| 2 | Monitor first 200 steps: loss curve, alloc count, throughput | `fleet:logs -f` |
| 3 | If alloc OOM: lower HELIOS_LIVE_ALLOC_SOFT_CAP, increase syncGpu frequency | Env vars |
| 4 | If VRAM OOM: reduce batch to 1, reduce block to 256 | Config |
| 5 | If stable: increase batch/accum until allocation budget is saturated | Gradual escalation |
| 6 | Record throughput (tok/s) and project total tokens achievable in 48h | Decision gate |

**Stepping stone policy:** If 12L/768d causes intractable instability, immediately drop to **6L/512d** (same tokenizer, same corpus, same optimizer rules) to harden the exact stack components. This is not retreat — it accelerates success at 124M.

**Deliverable:** A stable baseline configuration for long-running 124M training on L4.

---

### Phase 2 — Get Real Data into the Machine (Hours 8-16, parallel with Phase 1)

**Goal:** Stop treating data as a debug artifact.

| # | Task | Details |
|---|------|---------|
| 1 | Source a large corpus (~1B+ tokens) | OpenWebText2, FineWeb, or Pile subset |
| 2 | Tokenize with bpe-64k | `packages/tokenizers/src/bpe.ts` |
| 3 | Cache as `.tokens` binary | `loadOrCacheTokens()` in `packages/train/src/data.ts` |
| 4 | Verify data quality: decode random windows, inspect | Manual |
| 5 | Verify train/val split integrity | Automated |
| 6 | Ensure data loader handles large files | Check `Int32Array` limits in `data.ts` |

**Deliverable:** A production-scale token stream ready for sustained training.

---

### Phase 3 — Long Run (Hours 16-66)

**Goal:** Train long enough for real model behavior to emerge.

| # | Task | Details |
|---|------|---------|
| 1 | Launch long-run training | `fleet:train` with final config |
| 2 | Checkpoint every 500 steps | Binary checkpoints to `runs/` |
| 3 | Sample every 200 steps | Discord webhook |
| 4 | Evaluate on fixed prompt set | Domain sample prompts |
| 5 | Monitor: loss, val_loss, throughput, alloc count, memory | Remote dashboard + `fleet:logs` |
| 6 | If loss plateaus: reduce LR | Only if evidence demands it |
| 7 | If loss spikes: diagnose gradients, reduce LR or batch | Don't panic-restart |
| 8 | Keep one conservative checkpoint line alive | Never risk everything on frontier |

**Resume protocol:** If L4 training needs chained runs, that is normal. Resumability is part of the system:
```bash
npm run fleet:resume -- <instance> --runtime=node
```

**Deliverable:** A real GPT-2-class Alpha checkpoint with coherent output.

---

### Phase 4 — Evaluation and Proof (Hours 66-72)

**Goal:** Make the milestone legible. Prove the stack crossed the line.

| # | Task | Details |
|---|------|---------|
| 1 | Run fixed prompt battery against best checkpoint | `sample --checkpoint=... --prompt=...` |
| 2 | Compare early checkpoints vs late checkpoints | Quality progression |
| 3 | Inspect formatting, coherence, repetition | Human eval |
| 4 | Record validation perplexity | exp(val_loss) |
| 5 | Prepare reproducible launch recipe | Document exact config |

**Deliverable:** A clear answer to: **Can Alpha train a GPT-2-class model on L4 and produce coherent output?**

---

## 9. Stepping-Stone Policy

We are aiming at 124M. That remains the target.

If a full 12L/768d launch exposes instability that prevents useful progress, use a **stepping-stone run**:

| Stepping Stone | Params | Purpose |
|----------------|--------|---------|
| 6L/512d/8h | ~25M | De-risk optimizer, allocator, data pipeline at meaningful scale |
| 8L/640d/10h | ~55M | Validate activation checkpointing at scale |
| 12L/768d/12h | ~124M | Full target |

Same tokenizer, same corpus, same optimizer rules, same checkpoint path, same sampling path, same L4 hardware. The stepping stone accelerates success at 124M, not replaces it.

---

## 10. Run Rules

1. **Always keep one known-good checkpoint line alive.** Never risk everything on a single frontier run.

2. **Never overwrite "best" with "latest."** The best checkpoint is the best checkpoint.

3. **Sample on schedule.** Do not wait until the end to discover the model is weird. Every 200 steps.

4. **Treat resume as normal.** Chained runs on L4 are acceptable. Resumability is part of the system.

5. **Measure what matters.** Every serious run must surface: step, loss, val_loss, throughput, alloc count, memory pressure, sample output.

---

## 11. Kill Criteria

Stop, restart, or reconfigure a run if:

- loss fails to improve after warmup + 2000 steps
- repeated NaN/Inf events despite loss scale reductions
- sample quality is clearly collapsing
- allocation behavior suggests inevitable OOM
- throughput is too poor to make the run strategically useful (<1K tok/s)
- checkpoint resume corrupts continuity

Stopping a bad run is not failure. Letting a doomed run waste the machine is failure.

---

## 12. Success Criteria

### Minimum Success
- 124M-class model launches and trains stably on Alpha/L4
- Checkpointing and resume work under the new model shape
- Samples are recognizably coherent English
- Loss trends downward

### Strong Success
- Samples become clearly structured and readable
- Repetition and gibberish materially reduced
- The model handles prompt continuation with sanity
- Validation perplexity < 50

### Full Milestone Success
- Alpha can legitimately claim GPT-2-class training capability
- The 124M path is repeatable
- The codebase is stronger after the effort, not just patched for one lucky run
- Validation perplexity < 40

---

## 13. Explicit Non-Goals

This project is **not** about:

- Switching to PyTorch, JAX, or another framework
- Rewriting the Vulkan backend
- Implementing multi-GPU / data parallel training
- Jumping to GPT-2 Medium/Large/XL (345M+)
- RLHF, instruction tuning, or post-training alignment
- Endless micro-optimization before proving the main thing
- Waiting for better hardware

One thing: **make Alpha train a real GPT-2-class model on the hardware we have.**

---

## 14. Implementation Checklist

Militant order. Do these in sequence. Do not skip.

```
[ ] 1. Weight tying: packages/model/src/gpt.ts:148 (lmHead = wte)
[ ] 2. Remove lmHead from collectParamEntries: packages/model/src/gpt.ts:437
[ ] 3. Checkpoint backward compat: packages/train/src/checkpoint.ts:220-250
[ ] 4. Fix noDecayNames: apps/cli/src/commands/train.ts:324 (remove lmHead.weight)
[ ] 5. Add attn bias exclusions: apps/cli/src/commands/train.ts:325-329
[ ] 6. Add decay group logging: apps/cli/src/commands/train.ts:~340
[ ] 7. Auto-enable activation checkpointing for nLayer>=8
[ ] 8. Update concordance domain defaults: packages/core/src/domains.ts:132-137
[ ] 9. npm run build && smoke test 100 steps at 4L/128d
[ ] 10. Source + tokenize large corpus (>500M tokens)
[ ] 11. Deploy to L4: npm run fleet:deploy -- <instance>
[ ] 12. Launch 12L/768d/12h with --activationCheckpointing=true
[ ] 13. Monitor 200 steps. Decision gate: stable? Proceed. OOM? Step down.
[ ] 14. Long run. 48+ hours. Checkpoint every 500. Sample every 200.
[ ] 15. Evaluate best checkpoint. Ship or iterate.
```

---

## 15. Key Code Paths

```
Model init + tying target:    packages/model/src/gpt.ts:73-151
Forward pass:                 packages/model/src/gpt.ts:344-427
Activation checkpointing:     packages/model/src/gpt.ts:393-403
Checkpoint impl:              packages/autograd/src/checkpoint.ts:34-87
Param collection:             packages/model/src/gpt.ts:433-467
Training loop:                packages/train/src/trainer.ts:563+
LR schedule:                  packages/train/src/trainer.ts:1122-1130
Grad accumulation:            packages/train/src/trainer.ts:1135-1251
syncGpu in accum loop:        packages/train/src/trainer.ts:1245-1248
Loss scaling:                 packages/train/src/trainer.ts:1043-1072
NaN defense:                  packages/train/src/trainer.ts:1219-1268, 1473-1490
Grad clipping:                packages/train/src/trainer.ts:1278-1470
Optimizer (AdamW):            packages/train/src/optimizers.ts:21-200
noDecayNames wiring:          packages/train/src/optimizers.ts:40-44, 138, 193
GPU AdamW kernel:             packages/helios/src/kernels/optimizer.ts:29
Checkpoint save:              packages/train/src/checkpoint.ts:30-91
Checkpoint load + compat:     packages/train/src/checkpoint.ts:95-250
Sampling:                     packages/train/src/sample.ts:21-201
Tokenizer (BPE):              packages/tokenizers/src/bpe.ts:113-413
Domain config (concordance):  packages/core/src/domains.ts:116-139
Allocator caps:               packages/helios/src/backend.ts:462-465
Pool trim / reclaim:          packages/helios/src/backend.ts:487-511
Output pool:                  packages/helios/src/backend.ts:739-810
Slab allocator (C):           packages/helios/native/helios_vk.c:818-844
Slab bypass condition:        packages/helios/native/helios_vk.c:2071
CLI optimizer wiring:         apps/cli/src/commands/train.ts:318-339
CLI train command:            apps/cli/src/commands/train.ts
Fleet commands:               apps/cli/src/commands/fleet.ts
Data loading:                 packages/train/src/data.ts
```

---

## 16. Final Framing

If we can train a 124M GPT-2-class transformer on Alpha with Helios on L4, then Alpha is no longer just promising infrastructure. It becomes a **working independent training stack** with a credible claim to real model development.

The mission is not to be comfortable. The mission is to cross the line.

**Alpha trains GPT-2 class on its own stack.** That is the milestone. That is the work.
