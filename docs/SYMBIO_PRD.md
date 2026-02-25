# PRD: `--symbio` Training Mode for Alpha

## 1. Overview

Add a `--symbio` CLI flag to `alpha train` that activates a full alternative training configuration preset inspired by the Symbiogenesis research. When enabled, the flag:

1. Switches the FFN activation function to **SwiGLU** (or a configurable alternative from `{gelu, silu, relu, swiglu}`)
2. Enables **CUSUM change-point monitoring** on gradient norms, clipping rates, throughput, and validation loss
3. Activates **adaptive batch sizing** that responds to CUSUM-detected regime shifts
4. Collects and reports **Symbiogenesis-inspired metrics**: weight entropy, free energy proxy, effective rank, mutual information profiles, and CUSUM statistics

All new metrics flow through the existing remote reporting pipeline (trainer → `StepMetrics` → remote reporter → `/api/ingest` → Turso DB → dashboard) with zero disruption to non-symbio runs.

### Motivation

Analysis of 68 training runs shows that quality varies significantly across configurations (valLoss range: 4.68–5.15+) while throughput remains architecture-bound (~7,643–7,803 tok/s for L6 D256 H8). This means config-level decisions — activation function, stability monitoring, adaptive hyperparameters — can improve results without sacrificing throughput. The `--symbio` flag packages these improvements into a single, reproducible preset.

### Design Principles

- **Single flag, batteries included** — `--symbio` activates everything; no need to set 15 individual flags
- **Override-friendly** — any individual setting can be overridden via explicit CLI flags after `--symbio`
- **Backward compatible** — runs without `--symbio` behave identically to today
- **Metrics-first** — every new signal flows through the existing pipeline to the dashboard
- **No new packages** — all changes live in existing packages to minimize surface area

---

## 2. CLI Interface

### Flag syntax

```bash
# Activate symbio preset with defaults
alpha train --data data/super_chat.txt --domain chat --symbio

# Symbio preset + override specific values
alpha train --data data/super_chat.txt --domain chat --symbio --lr=1e-4 --batch=32

# Symbio with custom config file
alpha train --data data/super_chat.txt --domain chat --symbio --symbio-config=configs/symbio-aggressive.json
```

### Override semantics

Resolution order (later wins):

1. Domain defaults from `@alpha/core`
2. `--symbio` preset values (see §3)
3. `--symbio-config=<path>` file values (if provided)
4. Explicit CLI flags (`--lr`, `--batch`, etc.)

This means `--symbio --lr=1e-4` uses the symbio preset but overrides learning rate. The user always has final say.

### New CLI flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--symbio` | boolean | `false` | Activate Symbiogenesis training preset |
| `--symbio-config` | string | — | Path to JSON config file with symbio overrides |
| `--activation` | string | — | FFN activation: `gelu`, `silu`, `relu`, `swiglu` |

### GCP script additions

`scripts/gcp_train.py` gets matching flags:

```python
parser.add_argument("--symbio", action="store_true", help="Activate Symbiogenesis preset")
parser.add_argument("--symbio-config", type=str, help="Symbio config JSON path")
parser.add_argument("--activation", type=str, help="FFN activation function")
```

These pass through directly to the Node CLI.

---

## 3. Symbio Config Preset

When `--symbio` is set, the following defaults apply before any explicit overrides:

### ModelConfig changes

| Field | Default | Symbio | Rationale |
|-------|---------|--------|-----------|
| `ffnActivation` | `"gelu"` | `"swiglu"` | Better loss/param efficiency in literature and our perf analysis |

### TrainConfig changes

| Field | Default | Symbio | Rationale |
|-------|---------|--------|-----------|
| `lr` | `3e-4` | `5e-5` | Best observed in 68-run analysis (run `20260223_171025`) |
| `beta2` | `0.95` | `0.95` | Confirmed optimal |
| `weightDecay` | `0.1` | `0.1` | Confirmed optimal |
| `gradClip` | `1.0` | `5.0` | Best observed; aggressive clipping hurts with SwiGLU |
| `warmupIters` | `0` | `500` | Stability for SwiGLU's gated activation |
| `spikeThreshold` | `0` | `10.0` | Enables grad spike detection |

### New symbio-specific fields (on `TrainConfig`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `symbio` | `boolean` | `false` | Whether symbio mode is active |
| `symbioConfig` | `SymbioConfig \| null` | `null` | Symbio-specific configuration |

### SymbioConfig type

```typescript
export interface SymbioConfig {
  /** CUSUM detection threshold for regime shifts */
  readonly cusumThreshold: number;       // default: 2.0
  /** CUSUM drift parameter */
  readonly cusumDrift: number;           // default: 0.5
  /** Steps between symbio metric collection (expensive metrics) */
  readonly metricsInterval: number;      // default: 50
  /** Enable adaptive batch sizing on CUSUM triggers */
  readonly adaptiveBatch: boolean;       // default: true
  /** Min batch size for adaptive sizing */
  readonly batchMin: number;             // default: 8
  /** Max batch size for adaptive sizing */
  readonly batchMax: number;             // default: 64
  /** Batch size step for adaptive changes */
  readonly batchStep: number;            // default: 4
  /** Enable weight entropy collection */
  readonly trackWeightEntropy: boolean;  // default: true
  /** Enable effective rank tracking */
  readonly trackEffectiveRank: boolean;  // default: true
  /** Enable free energy proxy */
  readonly trackFreeEnergy: boolean;     // default: true
  /** Enable mutual information profiles */
  readonly trackMIProfiles: boolean;     // default: false (expensive)
}
```

### symbio-config JSON file format

```json
{
  "cusumThreshold": 2.0,
  "cusumDrift": 0.5,
  "metricsInterval": 50,
  "adaptiveBatch": true,
  "batchMin": 8,
  "batchMax": 64,
  "batchStep": 4,
  "trackWeightEntropy": true,
  "trackEffectiveRank": true,
  "trackFreeEnergy": true,
  "trackMIProfiles": false
}
```

---

## 4. New Metrics

### 4.1 Clipping telemetry (fix existing gap)

Already computed in `trainer.ts` but not persisted. Must be added to DB.

| Metric | Type | Frequency | Description |
|--------|------|-----------|-------------|
| `clip_coef` | `float` | Every step | Gradient clipping coefficient (1.0 = no clipping) |
| `clip_pct` | `float` | Every step | Fraction of gradient norm clipped away |

### 4.2 CUSUM statistics

Computed every step on running statistics of `gradNorm`, `clip_coef`, `tokens_per_sec`, and `valLoss`.

| Metric | Type | Frequency | Description |
|--------|------|-----------|-------------|
| `cusum_grad_pos` | `float` | Every step | Positive CUSUM on log(gradNorm) — detects upward regime shifts |
| `cusum_grad_neg` | `float` | Every step | Negative CUSUM on log(gradNorm) — detects downward shifts |
| `cusum_clip_pos` | `float` | Every step | Positive CUSUM on clip_pct — detects persistent clipping onset |
| `cusum_tps_neg` | `float` | Every step | Negative CUSUM on tokens_per_sec — detects throughput collapse |
| `cusum_alerts` | `integer` | Every step | Bitmask of which CUSUM detectors fired this step |

#### CUSUM algorithm

Standard tabular CUSUM on the signal `x_t` with target mean `μ` (running exponential mean), drift `k`, and threshold `h`:

```
S_pos(t) = max(0, S_pos(t-1) + (x_t - μ) - k)
S_neg(t) = max(0, S_neg(t-1) - (x_t - μ) - k)
Alert when S_pos(t) > h or S_neg(t) > h
```

Parameters `k = cusumDrift` (default 0.5), `h = cusumThreshold` (default 2.0). Running mean `μ` uses EMA with α = 0.01.

#### CUSUM alert bitmask

| Bit | Signal | Direction | Meaning |
|-----|--------|-----------|---------|
| 0x01 | gradNorm | positive | Gradient explosion regime |
| 0x02 | gradNorm | negative | Gradient vanishing regime |
| 0x04 | clip_pct | positive | Persistent clipping onset |
| 0x08 | tokens_per_sec | negative | Throughput collapse |
| 0x10 | valLoss | positive | Validation loss divergence |

### 4.3 Symbio-specific metrics (expensive, collected at `metricsInterval`)

| Metric | Type | Frequency | Description |
|--------|------|-----------|-------------|
| `weight_entropy` | `float` | Every N steps | Shannon entropy of weight magnitude distribution across all params |
| `effective_rank` | `float` | Every N steps | Effective rank of the gradient covariance matrix (exponential of spectral entropy) |
| `free_energy` | `float` | Every N steps | Free energy proxy: `loss + temperature * weight_entropy` |
| `mi_input_output` | `float` | Every N steps | Mutual information estimate between layer inputs and outputs (optional, expensive) |

#### Metric definitions

**Weight entropy**: For all parameters flattened into vector `w`, compute histogram of `|w|` into 256 bins, normalize to probability distribution `p`, return `H(p) = -Σ p_i log(p_i)`.

**Effective rank**: For gradient matrix `G` (sampled subset of params), compute singular values `σ`, normalize to `p_i = σ_i / Σσ`, return `exp(H(p))` where `H` is Shannon entropy. Approximated via power iteration (no full SVD).

**Free energy proxy**: `F = L + T * H(w)` where `L` = training loss, `T` = temperature parameter (fixed at `lr * 1000` as a proxy for optimization temperature), `H(w)` = weight entropy.

**MI profiles**: Estimated via binned mutual information between input/output activation tensors at each layer boundary. Only collected when `trackMIProfiles: true` (off by default due to compute cost).

### 4.4 Adaptive batch sizing signals

| Metric | Type | Frequency | Description |
|--------|------|-----------|-------------|
| `adaptive_batch_size` | `integer` | Every step | Current effective batch size (changes on CUSUM triggers) |
| `batch_change_reason` | `string` | On change | Why batch size changed: `"cusum_grad_spike"`, `"cusum_throughput_drop"`, `"cusum_clip_onset"`, `"manual"` |

#### Adaptive batch logic

When `adaptiveBatch: true`:

- **CUSUM grad spike (0x01)**: Reduce batch by `batchStep`, clamp to `batchMin`. Smaller batches = more gradient updates = faster recovery from instability.
- **CUSUM throughput drop (0x08)**: Reduce batch by `batchStep`. GPU may be under memory pressure.
- **CUSUM clip onset (0x04)**: Increase batch by `batchStep`, clamp to `batchMax`. Larger batches = smoother gradients = less clipping.
- **No alerts for 200 steps**: Gradually restore toward original batch size (increase by `batchStep` per 200 calm steps).
- Batch changes take effect on the next step. Old batch size is logged for the step that triggered the change.

---

## 5. Data Flow

### Existing flow (unchanged for non-symbio runs)

```
Trainer step
  → assembles StepMetrics
  → writes to metrics.jsonl buffer (local)
  → calls onStep(metrics) callback
  → RemoteReporter.onStep() buffers metrics
  → POST /api/ingest { type: "metrics", metrics: batch }
  → API route inserts into Turso `metrics` table
  → Dashboard reads from DB via tRPC/fetch
```

### Extended flow for symbio runs

```
Trainer step
  → assembles StepMetrics (now includes clip_coef, clip_pct)
  → CUSUM monitor updates (every step)
    → computes cusum_grad_pos/neg, cusum_clip_pos, cusum_tps_neg
    → checks alert thresholds
    → if alert: triggers adaptive batch sizing
  → if step % metricsInterval === 0:
    → computes weight_entropy, effective_rank, free_energy
    → optionally computes mi_input_output
  → merges all symbio metrics into StepMetrics
  → existing flow continues (jsonl buffer, onStep, remote reporter)
  → POST /api/ingest — same endpoint, metrics batch now includes symbio fields
  → API route inserts into Turso — new columns, nullable for non-symbio runs
  → Dashboard detects symbio run via `runs.symbio` flag
    → renders symbio-specific charts and panels
```

### Key design decisions

1. **No separate symbio metrics table** — extend the existing `metrics` table with nullable columns. This keeps the pipeline unified and avoids join complexity on the dashboard.
2. **No separate ingest endpoint** — symbio metrics ride the same `POST /api/ingest` batch as regular metrics. The API inserts whatever columns are present.
3. **CUSUM state is ephemeral** — CUSUM accumulators live in trainer memory, not in DB. Only the current values and alert bitmask are persisted per step.
4. **Expensive metrics are sparse** — `weight_entropy`, `effective_rank`, `free_energy` are only populated every `metricsInterval` steps. Dashboard handles null gaps gracefully.

---

## 6. Files to Modify

### `packages/core/src/types.ts`

- Add `ffnActivation?: "gelu" | "silu" | "relu" | "swiglu"` to `ModelConfig` (default: `"gelu"`)
- Add `symbio?: boolean` and `symbioConfig?: SymbioConfig | null` to `TrainConfig`
- Add `SymbioConfig` interface (see §3)
- Add `ffnActivation` to `DEFAULT_MODEL_CONFIG` as `"gelu"`
- Add `symbio: false` and `symbioConfig: null` to `DEFAULT_TRAIN_CONFIG`

### `packages/model/src/gpt.ts`

- Replace hardcoded `matmulTransposedGelu` in FFN with activation dispatch
- Keep `matmulTransposedGelu` as the fast path when `ffnActivation === "gelu"`
- Add `silu` path using `matmulTransposed` + `silu` op
- Add `relu` path using `matmulTransposed` + `relu` op
- Add `swiglu` path: split `fc` into `fc_gate` and `fc_up` (both `nEmbd → 4*nEmbd`), compute `silu(x @ fc_gate) * (x @ fc_up)`, then project down via `fc_proj`. This changes parameter count — document in checkpoint format
- Read `ffnActivation` from `ModelConfig` passed through init

### `packages/model/src/init.ts`

- Handle SwiGLU parameter initialization: `fc_gate`, `fc_up`, `fc_proj` weight matrices
- Adjust parameter count calculation for SwiGLU (3 matrices instead of 2 in FFN)

### `packages/autograd/src/ops.ts`

- Add `silu` autograd op with forward and backward
- Add `swiglu` composite op (or implement via existing `silu` + `mul` ops)

### `packages/tensor/src/cpu_ref.ts`

- Add `siluBackward` implementation
- Verify `silu` forward op exists (it does per research)

### `packages/helios/src/backend.ts`

- Add `siluBackward` SPIR-V kernel
- Add fused `matmulTransposedSilu` kernel (if performance warrants, can defer to v2)

### `packages/train/src/trainer.ts`

- Import `SymbioConfig` from `@alpha/core`
- After assembling `StepMetrics`, run CUSUM update if `symbio` is enabled
- At `metricsInterval` steps, compute expensive symbio metrics (weight entropy, effective rank, free energy)
- Implement adaptive batch sizing logic responding to CUSUM alerts
- Add all new fields to `StepMetrics` interface

### `packages/train/src/cusum.ts` (new file)

- `CusumMonitor` class: maintains running state for one signal
- `CusumDashboard` class: orchestrates multiple monitors (gradNorm, clip_pct, tokens_per_sec, valLoss)
- Pure functions, no side effects, easily testable
- Export alert bitmask constants

### `packages/train/src/symbio-metrics.ts` (new file)

- `computeWeightEntropy(params: Tensor[]): number`
- `computeEffectiveRank(grads: Tensor[]): number`
- `computeFreeEnergy(loss: number, lr: number, weightEntropy: number): number`
- `computeMutualInformation(inputs: Tensor, outputs: Tensor): number`
- All functions are stateless, operate on tensor data directly

### `packages/train/src/remote-reporter.ts`

- No structural changes needed — already forwards the full `StepMetrics` object
- Verify that `flushBuffer` serializes all fields (it does — uses `JSON.stringify`)

### `packages/db/src/schema.ts`

- Add migration version 6 (see §7)

### `packages/db/src/types.ts`

- Add new columns to `DbMetric` interface
- Add `symbio` flag to `DbRun` interface

### `packages/db/src/metrics.ts`

- Extend `insertMetrics` to include all new columns in the INSERT statement
- Use existing batch insert pattern (500 rows per chunk)

### `apps/cli/src/commands/train.ts`

- Parse `--symbio`, `--symbio-config`, `--activation` flags
- Load and validate symbio config JSON if `--symbio-config` provided
- Apply symbio preset defaults (§3) before explicit flag overrides
- Pass `symbio: true` and `symbioConfig` through to trainer
- Log symbio activation to console and remote reporter

### `apps/web/src/components/run-detail-view.tsx`

- Detect symbio run from `run.symbio` flag
- Render symbio badge in run header
- Add symbio stats to the stat grid: weight entropy, effective rank, free energy, CUSUM alert count
- Add CUSUM chart panel (see §8)

### `apps/web/src/components/charts.tsx`

- Add `CusumChart` — dual-axis line chart showing CUSUM accumulator values with alert threshold lines
- Add `SymbioMetricsChart` — multi-line chart for weight entropy, effective rank, free energy over time
- Add `BatchSizeChart` — step chart showing adaptive batch size changes with trigger annotations

### `scripts/gcp_train.py`

- Add `--symbio`, `--symbio-config`, `--activation` argparse flags
- Pass through to Node CLI command string

### `apps/server/src/routes/ingest.ts` (or equivalent API route)

- No changes needed if the INSERT dynamically handles present fields
- If the INSERT is column-explicit, update to include new metric columns

---

## 7. DB Schema Migration

### Version 6: Symbio metrics

```sql
-- Clipping telemetry (fixing existing gap)
ALTER TABLE metrics ADD COLUMN clip_coef REAL;
ALTER TABLE metrics ADD COLUMN clip_pct REAL;

-- CUSUM statistics
ALTER TABLE metrics ADD COLUMN cusum_grad_pos REAL;
ALTER TABLE metrics ADD COLUMN cusum_grad_neg REAL;
ALTER TABLE metrics ADD COLUMN cusum_clip_pos REAL;
ALTER TABLE metrics ADD COLUMN cusum_tps_neg REAL;
ALTER TABLE metrics ADD COLUMN cusum_alerts INTEGER;

-- Symbio-specific metrics (sparse — only populated every metricsInterval steps)
ALTER TABLE metrics ADD COLUMN weight_entropy REAL;
ALTER TABLE metrics ADD COLUMN effective_rank REAL;
ALTER TABLE metrics ADD COLUMN free_energy REAL;
ALTER TABLE metrics ADD COLUMN mi_input_output REAL;

-- Adaptive batch sizing
ALTER TABLE metrics ADD COLUMN adaptive_batch_size INTEGER;

-- Run-level symbio flag
ALTER TABLE runs ADD COLUMN symbio INTEGER DEFAULT 0;
ALTER TABLE runs ADD COLUMN symbio_config TEXT;
ALTER TABLE runs ADD COLUMN ffn_activation TEXT;
```

All new columns are nullable. Non-symbio runs simply have `NULL` in these columns — zero storage overhead in SQLite for null values.

The `runs.symbio_config` column stores the JSON-serialized `SymbioConfig` for reproducibility.

The `runs.ffn_activation` column stores the activation function name (useful for comparing runs by activation even outside symbio mode).

---

## 8. Dashboard Additions

### Run list page

- Symbio badge: small indicator next to run ID for symbio-enabled runs
- Activation tag: show `ffnActivation` value if non-default (e.g., "SwiGLU" tag)

### Run detail page — new symbio section

Rendered only when `run.symbio === 1`. Appears after the existing timing stats grid.

#### Symbio stats grid (4 tiles)

| Tile | Value | Source |
|------|-------|--------|
| Weight Entropy | Latest value + sparkline | `weight_entropy` column |
| Effective Rank | Latest value + sparkline | `effective_rank` column |
| Free Energy | Latest value + trend arrow | `free_energy` column |
| CUSUM Alerts | Total count + last alert step | `cusum_alerts` column |

#### CUSUM monitor chart

- 4 lines: `cusum_grad_pos`, `cusum_grad_neg`, `cusum_clip_pos`, `cusum_tps_neg`
- Horizontal threshold line at `cusumThreshold` value
- Alert events marked as vertical dashed lines (same pattern as existing event markers)
- X-axis: training step. Y-axis: CUSUM accumulator value.

#### Symbio metrics chart

- 3 lines on separate y-axes: weight entropy, effective rank, free energy
- Sparse data (every `metricsInterval` steps) — interpolate or show points only
- X-axis: training step

#### Adaptive batch size chart

- Step chart (not line) showing batch size over time
- Trigger annotations showing reason for each change
- X-axis: training step. Y-axis: batch size.

#### Clip telemetry chart (shown for all runs that have clip data, not just symbio)

- Dual-axis: `clip_coef` (left, 0–1 range) and `clip_pct` (right, 0–1 range)
- X-axis: training step
- This chart benefits all runs, not just symbio — it fills the existing telemetry gap

---

## 9. SwiGLU Implementation Details

### Architecture change

Standard FFN (current):
```
h = gelu(x @ W_fc) @ W_proj
```
2 weight matrices per block: `W_fc` (nEmbd × 4·nEmbd), `W_proj` (4·nEmbd × nEmbd)

SwiGLU FFN:
```
h = (silu(x @ W_gate) ⊙ (x @ W_up)) @ W_proj
```
3 weight matrices per block: `W_gate` (nEmbd × 4·nEmbd), `W_up` (nEmbd × 4·nEmbd), `W_proj` (4·nEmbd × nEmbd)

### Parameter count impact

For L6 D256 H8:
- Standard FFN per block: `256×1024 + 1024×256 = 524,288` params
- SwiGLU FFN per block: `256×1024 + 256×1024 + 1024×256 = 786,432` params
- Per block increase: +262,144 params (+50%)
- Total model increase: ~1.57M additional params across 6 blocks

To keep parameter count comparable, the inner dimension can be reduced from `4·nEmbd` to `(8/3)·nEmbd` (rounded to nearest multiple of 64). This is the standard practice (used by LLaMA, etc.):
- Adjusted inner dim for D256: `ceil((8/3) * 256 / 64) * 64 = 704`
- Adjusted SwiGLU per block: `256×704 + 256×704 + 704×256 = 541,696` params (~3.3% more than standard, acceptable)

The `ModelConfig` should gain an optional `ffnDim` field to control this:

```typescript
export interface ModelConfig {
  // ... existing fields ...
  readonly ffnActivation?: "gelu" | "silu" | "relu" | "swiglu";
  readonly ffnDim?: number;  // defaults to 4*nEmbd for standard, (8/3)*nEmbd for swiglu
}
```

### Checkpoint compatibility

SwiGLU checkpoints are **not** shape-compatible with standard checkpoints. The checkpoint must record `ffnActivation` in its metadata so restore can validate:

- Loading a SwiGLU checkpoint into a standard model: **error** (different param shapes)
- Loading a standard checkpoint into a SwiGLU model: **error** (different param shapes)
- Checkpoint metadata includes `ffnActivation` field for validation

Future work (out of scope): checkpoint morphing/projection to transfer weights between activation types.

---

## 10. Backward Compatibility

### Non-symbio runs are identical

- `--symbio` defaults to `false`
- All new DB columns are nullable
- All new `StepMetrics` fields are optional
- Dashboard only renders symbio sections when `run.symbio === 1`
- No changes to model forward/backward for `ffnActivation === "gelu"` (default)
- Existing checkpoints load without modification (no `ffnActivation` in metadata = assume `"gelu"`)

### Remote reporter protocol

- Ingest endpoint already accepts arbitrary JSON fields in metrics batches
- New fields are silently ignored by old API versions (forward-compatible)
- Old metrics without new fields get `NULL` in new DB columns (backward-compatible)

### Checkpoint format

- Add `ffnActivation` to checkpoint metadata
- Missing field defaults to `"gelu"` for backward compatibility
- Shape validation catches mismatches at load time with clear error message

### Dashboard

- Old runs (pre-symbio) display exactly as before — no symbio section rendered
- New runs without `--symbio` also display as before
- Only `symbio === 1` runs get additional UI

---

## 11. Verification Plan

### Phase 1: Core types and activation dispatch

- [ ] `ffnActivation` field added to `ModelConfig` with default `"gelu"`
- [ ] `SymbioConfig` type defined in `@alpha/core`
- [ ] Model forward pass dispatches on `ffnActivation`
- [ ] `gelu` path uses existing fused `matmulTransposedGelu` (no regression)
- [ ] `silu` path produces correct output (compare against reference)
- [ ] `relu` path produces correct output
- [ ] `swiglu` path produces correct output with adjusted `ffnDim`
- [ ] All four paths produce gradients via autograd
- [ ] Parameter count matches expectations for each activation type

### Phase 2: CUSUM and symbio metrics

- [ ] `CusumMonitor` correctly detects known regime shifts in synthetic data
- [ ] Alert bitmask correctly encodes multiple simultaneous alerts
- [ ] `computeWeightEntropy` returns reasonable values (higher entropy for random weights, lower for structured)
- [ ] `computeEffectiveRank` returns values in [1, min(m,n)] range
- [ ] `computeFreeEnergy` combines loss and entropy correctly
- [ ] Adaptive batch sizing responds to CUSUM triggers with correct direction

### Phase 3: Pipeline integration

- [ ] `--symbio` flag parsed and applied in CLI
- [ ] Symbio preset overrides apply in correct order (preset → config file → explicit flags)
- [ ] `StepMetrics` includes all new fields when symbio is active
- [ ] `StepMetrics` does NOT include symbio fields when symbio is inactive (null/undefined)
- [ ] Remote reporter serializes and transmits symbio metrics
- [ ] Ingest endpoint persists new columns to DB
- [ ] Non-symbio runs unaffected (null columns, no extra compute)

### Phase 4: DB and dashboard

- [ ] Migration v6 applies cleanly on fresh and existing databases
- [ ] `clip_coef` and `clip_pct` persisted for ALL runs (not just symbio)
- [ ] Symbio badge appears on symbio runs in run list
- [ ] CUSUM chart renders with threshold lines and alert markers
- [ ] Symbio metrics chart handles sparse data (nulls between `metricsInterval` steps)
- [ ] Adaptive batch size chart shows step transitions
- [ ] Clip telemetry chart shows for any run with clip data

### Phase 5: End-to-end

- [ ] `alpha train --data data/super_chat.txt --domain chat --symbio --backend helios` completes a 1000-step run
- [ ] Metrics stream to remote dashboard in real time
- [ ] Dashboard shows full symbio section for the run
- [ ] Run with `--symbio --activation=gelu` uses gelu (override works)
- [ ] Run without `--symbio` is byte-identical in behavior to current code
- [ ] GCP script `python3 scripts/gcp_train.py --symbio --data data/super_chat.txt ...` works end-to-end

---

## 12. Implementation Order

Recommended sequence to minimize integration risk:

1. **Fix clip telemetry gap** — add `clip_coef`/`clip_pct` to DB schema and insert path. Benefits all runs immediately.
2. **Core types** — add `ffnActivation`, `SymbioConfig`, symbio fields to `ModelConfig`/`TrainConfig`.
3. **Activation dispatch** — implement `silu`, `relu`, `swiglu` paths in model and autograd.
4. **CUSUM module** — implement `CusumMonitor` and `CusumDashboard` in `packages/train`.
5. **Symbio metrics** — implement weight entropy, effective rank, free energy computations.
6. **Trainer integration** — wire CUSUM and symbio metrics into the training loop.
7. **Adaptive batch sizing** — implement batch adjustment logic responding to CUSUM.
8. **CLI flags** — add `--symbio`, `--symbio-config`, `--activation` to CLI and GCP script.
9. **DB migration** — version 6 with all new columns.
10. **Dashboard** — symbio badge, CUSUM chart, metrics chart, batch chart, clip chart.

Steps 1–3 can be tested independently. Steps 4–5 are also independent. Step 6 depends on 2–5. Steps 8–10 depend on 6.
