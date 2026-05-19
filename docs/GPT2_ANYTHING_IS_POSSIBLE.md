# GPT-2 Mission: Anything Is Possible

Date: 2026-03-07

## Non-Negotiable Goal

Train and ship a GPT-2-class model. No side quests.

## Core Principle

Anything is possible if we remove indecision:
- One target.
- One execution loop.
- One quality bar.

---

## Target Spec (GPT-2 Class)

Use this as the default architecture target:
- `layers=12`
- `dim=768`
- `heads=12`
- `block=1024` (fall back to `512` if needed for stability/throughput)
- tokenizer: GPT-style BPE (`bpe-64k` in this repo)

### Where This Is Defined In Code

**Domain config** (`packages/core/src/domains.ts:116-139`) — the `concordance` domain is the GPT-2-class target:

```ts
// packages/core/src/domains.ts lines 116-139
concordance: {
  blockSize: 1024,
  nLayer: 12,
  nEmbd: 768,
  nHead: 12,
  tokenizer: "bpe-64k",
  training: {
    lr: 6e-4,
    batchSize: 4,
    gradClip: 1.0,
  }
}
```

**ModelConfig type** (`packages/core/src/types.ts:51-74`):

```ts
interface ModelConfig {
  vocabSize: number;
  blockSize: number;      // context window (512 or 1024)
  nLayer: number;         // 12 for GPT-2 class
  nEmbd: number;          // 768 for GPT-2 class
  nHead: number;          // 12 for GPT-2 class
  dropout: number;
  ffnActivation: "gelu" | "silu" | "relu" | "swiglu" | "universal" | "kan_spline" | "composed";
  ffnDim?: number;
  softCap?: number;
  activationGraph?: unknown;
}
```

**TrainConfig type** (`packages/core/src/types.ts:77-139`):

```ts
interface TrainConfig {
  iters: number;
  batchSize: number;
  lr: number;             // peak learning rate
  lrMin: number;          // minimum LR for cosine decay
  warmupIters: number;
  beta1: number;          // 0.9
  beta2: number;          // 0.95
  eps: number;            // 1e-8
  weightDecay: number;    // 0.1
  gradClip: number;       // 1.0
  evalInterval: number;
  evalIters: number;
  sampleInterval: number;
  gradAccumSteps: number;
  spikeThreshold?: number;
  syncEvery?: number;
  gcEvery?: number;
  packed?: boolean;
  symbio?: unknown;
}
```

---

## Model Architecture (Decoder-Only Transformer)

The GPT model is defined in `packages/model/src/gpt.ts`. It's a standard decoder-only transformer with some custom additions.

### Parameter Structure (`packages/model/src/gpt.ts:21-51`)

```ts
interface GPTParams {
  wte: Variable;       // token embeddings [vocabSize, nEmbd]
  wpe: Variable;       // position embeddings [blockSize, nEmbd]
  layers: LayerParams[];
  lnF: { gamma: Variable; beta: Variable };  // final layer norm
  lmHead: Variable;    // output projection [nEmbd, vocabSize]
}

interface LayerParams {
  ln1: { gamma: Variable; beta: Variable };   // pre-attention layernorm
  ln2: { gamma: Variable; beta: Variable };   // pre-MLP layernorm
  attn: {
    wqkv: Variable;    // grouped QKV projection (single GEMM)
    wo: Variable;       // output projection
    bqkv?: Variable;
    bo?: Variable;
  };
  mlp: {
    // Multiple activation options: standard, SwiGLU, Universal Approximator, KAN Spline
    w1: Variable;       // up projection
    w2: Variable;       // down projection
    wGate?: Variable;   // gate projection (for SwiGLU)
  };
}
```

### Model Initialization (`packages/model/src/gpt.ts:73-151`)

`initGPT()` creates all parameters with proper initialization:
- Token embeddings: normal(0, 0.02)
- Position embeddings: normal(0, 0.01)
- Attention projections: normal(0, 0.02)
- Output projection: scaled by `1/sqrt(2 * nLayer)` for residual stream stability
- Configurable FFN activations (gelu, swiglu, etc.)

### Transformer Block (`packages/model/src/gpt.ts:220-333`)

Each block implements:
1. **Pre-norm** (LayerNorm) -> **Grouped QKV** (single GEMM instead of 3 separate projections) -> **Causal Self-Attention** (Flash Attention kernel or standard path) -> **Residual + Dropout**
2. **Pre-norm** (LayerNorm) -> **MLP** (with configurable activation: GELU, SwiGLU, etc.) -> **Residual + Dropout**

Supports:
- Flash Attention kernel with soft-capping
- Grouped QKV projection for efficiency
- Multiple activation function options

### Forward Pass (`packages/model/src/gpt.ts:344-427`)

`gptForward()` runs the full model:
1. Token embedding + position embedding lookup
2. Causal mask construction
3. Sequential transformer blocks (with optional activation checkpointing for memory savings)
4. Mixed precision support (f16 inter-layer activations)
5. Final layer norm -> LM head projection
6. Cross-entropy loss computation
7. Diagnostics (logit magnitude tracking)

### Parameter Counting (`packages/model/src/gpt.ts:430-481`)

`countParams()` — collects and counts all trainable parameters. For a 12L/768d/12h model this is ~124M parameters (GPT-2 class).

---

## Training Loop

The main training loop lives in `packages/train/src/trainer.ts`. The `train()` function (line 563) orchestrates everything.

### Training Setup (`packages/train/src/trainer.ts:593-794`)

```
1. Create run directory, save config                    (lines 593-603)
2. Load data with delimiter-aware splits                (lines 643-720)
3. Create validation buckets                            (lines 643-720)
4. Initialize model (initGPT) or resume from checkpoint (lines 722-753)
5. Collect infrastructure metadata                      (lines 756-794)
```

### Step Metrics (`packages/train/src/trainer.ts:191-249`)

Every training step records:
```ts
interface StepMetrics {
  step: number;
  loss: number;
  gradNorm: number;
  lr: number;
  tokPerSec: number;
  gpuAllocCount: number;
  gpuAllocBytes: number;
  gpuPoolEntries: number;
  clipRatio: number;
  lossScale: number;
  // ... symbio metrics, per-layer stats, etc.
}
```

### Learning Rate Schedule (`packages/train/src/trainer.ts:1122-1130`)

Cosine decay with linear warmup:
- Linear warmup from 0 to `lr` over `warmupIters` steps
- Cosine decay from `lr` to `lrMin` over remaining steps

### Gradient Accumulation (`packages/train/src/trainer.ts:1135-1251`)

```
For each microstep in gradAccumSteps:
  1. Get batch of data
  2. Forward pass (gptForward)
  3. Scale loss by (1/gradAccumSteps) * lossScale
  4. Backward pass (loss.backward())
  5. GPU-side loss accumulation
  6. Clear autograd tape
  7. Sync GPU
```

- Loss scaling applied during backward for mixed precision stability
- GPU-side loss accumulation avoids per-microstep CPU readback
- Single loss readback after all microsteps complete

### Dynamic Loss Scaling (`packages/train/src/trainer.ts:1043-1072, 1205-1216, 1473-1490`)

```
Initial scale: 128.0 (mixed precision) or 1.0 (f32)
On 200 consecutive good steps: double the scale (SCALE_GROWTH_INTERVAL)
On NaN detection: halve the scale, skip optimizer update
All gradients scaled by lossScale during backward
Gradients unscaled before optimizer step
```

### NaN Detection (`packages/train/src/trainer.ts:1219-1268, 1473-1490`)

Three-layer NaN defense:
1. **Forward pass**: check loss is finite after each microstep (lines 1219-1232)
2. **GPU loss accumulation**: single readback validation after all microsteps (lines 1254-1270)
3. **Backward pass**: check gradient norm is finite before optimizer step (lines 1473-1490)

On any NaN: halve loss scale, skip update, log warning.

### Gradient Clipping & Diagnostics (`packages/train/src/trainer.ts:1278-1470`)

```
1. Compute gradient norm (GPU reduction, CPU fallback, or totalSumSq fast path)
2. CPU recheck for suspicious gradient norms          (lines 1355-1410)
3. Per-parameter gradient norm diagnostics             (lines 1414-1468)
4. Per-layer gradient norm breakdown
5. Apply clipping coefficient in optimizer step
```

---

## Inference & Sampling

**File**: `packages/train/src/sample.ts:21-201`

Autoregressive token generation with:
- Sliding context window (handles sequences longer than blockSize)
- Temperature scaling
- **Greedy decoding**: argmax when temp <= 0
- **Top-k filtering** (lines 104-111): keep top-k logits, zero the rest
- **Top-p / nucleus sampling** (lines 114-165): keep tokens until cumulative probability exceeds threshold
- Per-step GPU buffer release to prevent OOM during long generation (lines 72-84)
- Periodic GPU flush for deferred buffer releases (line 84)

---

## GPU Backend: Helios (Vulkan Compute)

The entire GPU backend is custom-built — no CUDA, no cuDNN, no NCCL. Pure Vulkan compute shaders compiled from TypeScript to SPIR-V.

### Buffer Management (`packages/helios/src/backend.ts:451-611`)

**Buffer Pool** (lines 451-505):
```
Size-aware pool limits:
  <= 256KB  -> up to 256 entries
  <= 4MB    -> up to 32 entries
  > 4MB     -> up to 8 entries

Live allocation caps:
  SOFT_CAP = 8000 allocations
  HARD_CAP = 10000 allocations
```

**`acquireBuffer()`** (lines 524-572): Allocate from pool or create new Vulkan buffer
**`releaseBuffer()`** (lines 575-611): Return to pool with soft/hard caps, excess freed immediately

### Output Pool (Timeline-Aware) (`packages/helios/src/backend.ts:739-810`)

Per-size-class buffer pool with timeline tracking:
- `acquireOutputRegion()` (lines 765-788): allocate with readiness tracking
- `deferRelease()` (lines 794-810): deferred releases after compute graph flush

### OOM Handling (`packages/helios/src/backend.ts:487-511`)

`trimPoolsForAllocPressure()` — aggressive pool trimming when allocation count approaches driver limits. This is critical because the L4 driver caps at ~5500 live allocations.

### GPU Memory Stats (`packages/helios/src/backend.ts:1380-1461`)

```ts
gpuMemStats()    // live alloc count, total bytes, pool entries
poolBreakdown()  // per-size pool statistics
```

### Cooperative Matrix Multiply (`packages/helios/src/backend.ts:46-96`)

Environment-configurable cooperative matmul:
- Tile sizes, register tiling, split-K partitioning, double buffering
- Split-K for better SM occupancy (lines 85-88)
- Paused during backward pass to avoid f16 overflow (trainer.ts lines 1199-1216)

### Native Vulkan Addon (`packages/helios/native/helios_vk.c`)

**Slab Allocator** (lines 818-844):
```c
SLAB_INITIAL_SIZE = 64MB
SLAB_MAX_SIZE     = 256MB per slab
SLAB_POOL_MAX     = 8GB total per pool

// Two pools:
devicePool     // persistent params (weights, optimizer state)
deviceTempPool // temporary intermediates (activations, gradients)
```

**Slab Pool Structure** (lines 835-843):
```c
typedef struct {
  Slab slabs[MAX_SLABS];
  int slabCount;
  int memoryTypeIndex;
  uint64_t totalAllocated;
} SlabPool;
```

**Exposed JS Functions** (lines 1-20):
`initDevice()`, `createBuffer()`, `uploadBuffer()`, `readBuffer()`, `destroyBuffer()`, `createPipeline()`, `dispatch()`, `waitIdle()`, `destroy()`

No Vulkan SDK headers — minimal subset of type definitions hand-written (lines 32-327).

---

## Tokenizer (BPE)

**File**: `packages/tokenizers/src/bpe.ts`

### BPE Training Algorithm (lines 113-261)

```
1. Build base vocabulary from unique characters
2. Sample training corpus from full file (avoids merge bias)  (lines 131-134)
3. Iterative merge loop:
   a. Count all adjacent pairs
   b. Select most frequent pair
   c. Merge pair into new token
   d. Update pair counts incrementally (not full recount)
4. Continue until target vocab size reached
```

### Encoding (lines 270-413)

Two encoding paths depending on vocab size:
- **Small vocabs** (<1000 merges): O(M*N) simple scan (lines 281-283)
- **Large vocabs**: O(N log N) heap-based with doubly-linked list (lines 296-401)
  - Binary min-heap ordered by merge rank
  - Stale entry detection and skipping
  - Doubly-linked list for O(1) token removal after merge

### Reserved / Special Tokens (lines 87-96, 474-477)

Multi-character reserved tokens (e.g., `<|user|>`, `<|assistant|>`, `<|end_of_text|>`) act as hard merge boundaries — the BPE algorithm never merges across them.

### Decoding (lines 418-427)

Simple concatenation of vocab strings by token ID.

---

## Checkpoint System

**File**: `packages/train/src/checkpoint.ts`

### Binary Checkpoint Format (lines 4-12)

```
[4 bytes: magic "ALPH"]
[4 bytes: uint32 LE header JSON byte length]
[N bytes: header JSON (UTF-8)]
  - modelConfig, configHash, step, rngState, activationGraph
  - tensor manifest (name -> { offset, length, shape })
[remaining: concatenated raw Float32 tensor data]
```

### Save (`saveBinary()`, lines 30-91)

1. Collect all param tensors and optimizer state buffers (m, v for AdamW)
2. Build header with metadata + tensor manifest
3. Stream tensor data sequentially to avoid single large buffer allocation

### Load (`loadBinary()`, lines 95-186)

1. Detect format by magic bytes (`ALPH`)
2. Parse header JSON
3. Reconstruct tensor map from manifest
4. Backward compat: auto-converts old `wq`/`wk`/`wv` separate projections to grouped `wqkv`

### State Management

- `buildCheckpointState()` (lines 194-216): Serialize GPTParams + OptimizerState + RNG state
- `restoreParams()` (lines 220-250): Copy checkpoint tensor values into live Variable buffers, handles legacy format conversion

---

## Hardware Rule

- Preferred: H100
- Acceptable: A100 80GB
- Emergency fallback: L4 (for debugging and small-scale validation only)

---

## Build + Deploy Rule (Repo Standard)

Always deploy with compiled workflow:

```bash
npm run bun:compile
npm run fleet:deploy -- <instance-name>
```

The compile step (`npm run bun:compile`) is wired to:
1. Build `@alpha/cli` first
2. Build Helios native addon
3. Compile from `apps/cli/dist/main.js` into a standalone binary

---

## Fleet Deployment System

**File**: `apps/cli/src/commands/fleet.ts`

### Fleet Configuration (lines 22-27, 51-72)

```ts
{
  sshUser: string;
  sshKey: string;       // path to SSH private key
  deployDir: string;    // remote directory (e.g., /home/ajax/alpha)
  instances: {
    [name: string]: {
      host: string;     // IP address
      zone: string;     // GCP zone
      machine: string;  // machine type
      gpu: string;      // GPU type (L4, A100, H100)
      role: string;
      setupDone: boolean;
    }
  }
}
```

### Current Fleet (`fleet.json`)

| Instance | Host | Zone | GPU |
|----------|------|------|-----|
| alpha-bench-l4-coopdbg-20260228084511 | 136.113.161.152 | us-central1-b | L4 |
| alpha-train-20260307054233 | 35.243.129.105 | us-east1-b | L4 (ACTIVE) |

### Remote Training Runtime (lines 249-292)

`resolveTrainRuntime()` detects:
- **auto**: prefer compiled binary, fallback to node
- **binary**: force compiled binary (fail if missing)
- **node**: force Node.js with `--expose-gc`

### Deploy Process

1. SHA-256 check all artifacts (binary, native addon, source, env, flake files)
2. Only upload changed files (significantly speeds repeated deploys)
3. Never rsync `*.node` from mac to linux — must rebuild native on remote

---

## Training Commands

### Standard Training Run

```bash
npm run fleet:train -- <instance-name> \
  --runtime=binary \
  --dgc=true \
  --no-fallback=true \
  --sampleInterval=200 \
  --domain=concordance \
  --tokenizer=bpe-64k \
  --layers=12 \
  --dim=768 \
  --heads=12 \
  --block=512 \
  --batch=4 \
  --steps=20000
```

If binary runtime has Vulkan issues, switch `--runtime=node`.

### H100 Fast Path (Modal)

```bash
./scripts/modal-run.sh data/concordance-v2.txt \
  --backend=helios \
  --tokenizer=bpe-64k \
  --domain=concordance \
  --layers=12 \
  --dim=768 \
  --heads=12 \
  --block=512 \
  --batch=4 \
  --iters=20000
```

### Operational Commands

```bash
npm run fleet:status -- <instance-name>          # check status
npm run fleet:logs -- <instance-name> -f          # follow logs
npm run fleet:stop -- <instance-name>             # stop training
npm run fleet:resume -- <instance-name> --runtime=node  # resume from checkpoint
npm run fleet:download -- <instance-name> --run=<run>   # pull artifacts
```

---

## 72-Hour Execution Loop

Repeat this loop without interruption:

1. Launch run.
2. Watch logs and samples.
3. Keep only stable branches.
4. Resume from best checkpoint.
5. Kill weak branches early.

---

## Known Blockers & Current State

### L4 Allocator OOM

The L4 Vulkan driver caps at ~5500 live allocations. The slab allocator (`helios_vk.c:818-844`) mitigates this with 64MB-256MB slabs, but slab is currently disabled for device-local memory under pressure. Models larger than 4L/192d hit this wall.

**Impact**: Can't run the full 12L/768d/12h GPT-2 target on L4. Need H100/A100 or a slab allocator rewrite.

### fp16 Gradient Overflow

Even with dynamic loss scaling (initial scale 128.0, halving on NaN), gradient norms explode to trillions in f16 backward pass. Cooperative matmul is paused during backward (trainer.ts:1199-1216) but the issue persists. Training works fine in f32.

### Working Configurations on L4

| Config | Params | Throughput | Status |
|--------|--------|------------|--------|
| 4L/128d/4h, block=256, batch=16, accum=2, f32 | 1.85M | 65K tok/s | Working fast |
| 4L/192d/6h, block=256, batch=8, accum=2, f32 | 3M | ~1K tok/s | Working (GC pressure) |
| 6L+/256d+, any batch, accum>1 | - | - | OOM |

---

## Quality Bar (Simple and Brutal)

Model passes when it can:
- Respond to `Hello` naturally.
- Stay in user/assistant format.
- Answer basic factual/context prompts coherently.
- Avoid obvious repetitive gibberish.

---

## Scope Control

Do not add features.
Do not change objectives.
Do not restart from scratch after a stable checkpoint exists.

---

## Final Ship Condition

Ship the checkpoint that wins on:
1. Stability over long training windows.
2. Best validation trend.
3. Best human-judged sample coherence.

That is the model.

---

## Complete File Reference Index

| Component | File | Key Lines |
|-----------|------|-----------|
| Model definition | `packages/model/src/gpt.ts` | 21-51 (params), 73-151 (init), 220-333 (block), 344-427 (forward) |
| Training loop | `packages/train/src/trainer.ts` | 563+ (train fn), 1122-1130 (LR schedule), 1135-1251 (grad accum), 1278-1470 (grad clip) |
| Loss scaling | `packages/train/src/trainer.ts` | 1043-1072 (config), 1205-1216 (apply), 1473-1490 (NaN handling) |
| Sampling | `packages/train/src/sample.ts` | 21-201 (generation loop, top-k, top-p) |
| Checkpoints | `packages/train/src/checkpoint.ts` | 4-12 (format), 30-91 (save), 95-186 (load) |
| GPU backend | `packages/helios/src/backend.ts` | 451-611 (buffer pool), 739-810 (output pool), 487-511 (OOM) |
| Native Vulkan | `packages/helios/native/helios_vk.c` | 818-844 (slab allocator), 32-327 (Vulkan types) |
| BPE tokenizer | `packages/tokenizers/src/bpe.ts` | 113-261 (training), 270-413 (encoding), 87-96 (special tokens) |
| Domain configs | `packages/core/src/domains.ts` | 116-139 (concordance/GPT-2), 141-175 (chat), 214-252 (nanochat) |
| Type definitions | `packages/core/src/types.ts` | 51-74 (ModelConfig), 77-139 (TrainConfig) |
| Fleet commands | `apps/cli/src/commands/fleet.ts` | 22-72 (config), 249-292 (runtime resolution) |
| Fleet config | `fleet.json` | Instance definitions, SSH config |
