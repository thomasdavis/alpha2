# PRD: Symbiogenesis Mode for Alpha (`--symbio`)

Date: 2026-02-25
Status: **Approved**
Scope: Write Symbiogenesis-inspired training mode from scratch as a dedicated Alpha package, activated via `--symbio` CLI flag, with full metrics pipeline to remote DB and dashboard rendering

---

## 1. Decision

We will implement Symbiogenesis integration as:

- a dedicated `@alpha/symbiogenesis` package in the Alpha monorepo containing all symbio logic **written from scratch in TypeScript**
- a single CLI mode switch: `--symbio`
- an optional config file: `--symbio-config=<path>`
- an optional activation override: `--activation=<name>`

When `--symbio` is enabled, Alpha switches relevant internal seams automatically. The user does not manage individual seam flags.

**Everything is written from scratch.** The original Symbiogenesis Python repo (`symbiogenesis/`) is a reference for concepts, algorithms, and metric definitions. We do not import, wrap, call, or transpile any of its code. Every line of symbio logic in Alpha is new TypeScript, following Alpha's zero-dependency philosophy.

## 2. Problem Statement

Alpha has 68 training runs showing significant quality variance (valLoss 4.68–5.15+) at near-identical throughput (~7,643–7,803 tok/s). Config-level decisions — activation function, stability monitoring, adaptive hyperparameters — can improve results without sacrificing throughput. But:

- Only 32 of 68 runs have `valLoss` logged
- Only 3 of 68 runs have `clip_coef`/`clip_pct` in metrics
- Zero runs have stability monitoring, weight entropy, or change-point detection
- There is no mechanism to adapt batch size or detect regime shifts during training

We need a training mode that activates richer monitoring, adaptive behavior, and alternative model architectures — without scattering experimental code across every package.

## 3. Product Goal

```bash
# Normal Alpha — no change
alpha train --data data/super_chat.txt --domain chat --backend helios

# Symbiogenesis mode — batteries included
alpha train --data data/super_chat.txt --domain chat --backend helios --symbio

# Symbio with overrides
alpha train --data data/super_chat.txt --domain chat --backend helios --symbio --lr=1e-4 --activation=silu

# Symbio with custom config file
alpha train --data data/super_chat.txt --domain chat --backend helios --symbio --symbio-config=configs/symbio-aggressive.json
```

`--symbio` activates:
1. SwiGLU FFN activation (or configurable alternative)
2. CUSUM change-point monitoring on gradients, clipping, throughput, validation loss
3. Adaptive batch sizing responding to CUSUM alerts
4. Rich metric collection: weight entropy, free energy proxy, effective rank, MI profiles, population entropy, activation distribution, fitness scores
5. All metrics streamed to remote server, stored in DB, rendered on dashboard

## 4. Scope

### 4.1 In Scope (v1)

- `@alpha/symbiogenesis` package — all symbio logic written from scratch
- `--symbio`, `--symbio-config`, `--activation` CLI flags
- Internal seam switching (no user-facing seam flags)
- FFN activation support: `gelu`, `relu`, `silu`, `swiglu` in model, autograd, backends
- SwiGLU as the symbio default activation with `(8/3)*nEmbd` inner dim
- CUSUM change-point detection (inspired by `symbiogenesis/monitor.py:GelationMonitor`)
- Adaptive batch sizing responding to CUSUM triggers
- Symbio metric collection: weight entropy, effective rank, free energy, population entropy, MI profiles, activation distribution, fitness scores, clipping telemetry, CUSUM statistics
- **Every symbio metric persisted to remote DB** via the existing ingest pipeline
- DB schema migration (version 6) for all new columns
- Full dashboard rendering of all symbio metrics when viewing a symbio run
- Symbio FFN Activation Search orchestration (evolutionary, inspired by `symbiogenesis/population.py`)
- Symbio artifacts: `symbio-summary.json`, `symbio-candidates.jsonl`, `symbio-report.md`
- GCP script passthrough for all new flags
- Discord notifications for symbio events

### 4.2 Out of Scope (v1)

- Full Symbiogenesis phase parity (Kuramoto oscillators, reservoir computing, fusion strategies)
- Python subprocess bridge to the original Symbiogenesis repo
- User-facing generic adapter framework
- GPT checkpoint fusion/projection (seam defined, not implemented)
- Replacing Alpha's core trainer loop
- Per-layer activation vectors (global activation only in v1)

## 5. CLI Interface

### 5.1 New CLI Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--symbio` | boolean | `false` | Activate Symbiogenesis mode |
| `--symbio-config` | string | — | Path to JSON config file with symbio overrides |
| `--activation` | string | — | FFN activation: `gelu`, `silu`, `relu`, `swiglu` |

### 5.2 Override Semantics

Resolution order (later wins):

1. Domain defaults from `@alpha/core`
2. `--symbio` preset values (see §7)
3. `--symbio-config=<path>` file values (if provided)
4. Explicit CLI flags (`--lr`, `--batch`, `--activation`, etc.)

`--symbio --lr=1e-4` uses the symbio preset but overrides learning rate. The user always has final say.

### 5.3 CLI Behavior

- Without `--symbio`: Alpha uses existing default behavior only. No symbio code on any hot path.
- With `--symbio`: load symbio config, apply preset, activate monitoring, log seam switches to console and remote.
- If config load fails: fail fast with a clear error before any training starts.
- Run artifacts record `symbio=true`, the resolved config, and the activation function.
- When remote reporting is enabled, all symbio metrics flow through the same pipeline as regular step metrics.

### 5.4 Default Symbio Config Resolution

1. `--symbio-config=<path>` if provided
2. Hardcoded defaults in `@alpha/symbiogenesis/config/defaults.ts`
3. Fail with actionable error if validation fails

## 6. Architecture

### 6.1 Package: `@alpha/symbiogenesis`

Written from scratch. Inspired by the Python `symbiogenesis/` repo for concepts and algorithms but sharing zero code.

```
packages/symbiogenesis/
  src/
    config/
      schema.ts          — SymbioConfig type + defaults
      load.ts            — load from file, merge with defaults, validate
      preset.ts          — symbio preset overrides for ModelConfig/TrainConfig
    monitor/
      cusum.ts           — CusumMonitor class (inspired by monitor.py:GelationMonitor)
      dashboard.ts       — CusumDashboard: orchestrates multiple monitors
      adaptive-batch.ts  — batch sizing logic responding to CUSUM alerts
    metrics/
      weight-entropy.ts  — Shannon entropy of weight distributions (inspired by model.py:Unit.weight_entropy)
      effective-rank.ts  — SVD-based effective rank (inspired by model.py:Unit.effective_rank)
      free-energy.ts     — free energy proxy: loss + beta * entropy (inspired by training.py)
      mi-estimator.ts    — mutual information estimation (inspired by mi_estimator.py)
      population.ts      — population entropy, diversity, activation distribution
      fitness.ts         — multi-objective fitness functions (inspired by training.py:train_and_eval)
      collector.ts       — SymbioMetricsCollector: orchestrates all metric computation
    search/
      orchestrator.ts    — FFN activation search loop
      candidates.ts      — candidate generation and lifecycle
      ranking.ts         — fitness ranking with configurable weights
      report.ts          — artifact generation (summary, candidates, report)
    types.ts             — shared types
    index.ts             — public API
  package.json
  tsconfig.json
```

### 6.2 Execution Principle

Alpha remains the execution engine:
- Alpha trainer runs training steps
- Alpha model executes forward/backward
- Alpha backends run kernels
- Symbio package provides monitoring, metrics collection, adaptive logic, and search orchestration

Symbio mode does NOT duplicate the trainer loop. It hooks into the existing `onStep` callback and exposes functions the trainer calls at the right moments.

### 6.3 Internal Seam Model

When `--symbio` is enabled, Alpha internally routes these seams:

| Seam | Symbio OFF | Symbio ON |
|------|-----------|-----------|
| FFN activation | `gelu` (fused) | `swiglu` default (configurable) |
| Step monitoring | none | CUSUM dashboard on gradNorm, clip_pct, tokens_per_sec, valLoss |
| Batch sizing | fixed | adaptive (responds to CUSUM alerts) |
| Metric collection | standard StepMetrics | standard + all symbio metrics |
| Search orchestration | single run | evolutionary FFN activation search |
| Checkpoint transfer | same-shape restore | same-shape (seam exists for future projection) |

## 7. Symbio Config Preset

When `--symbio` is set, these defaults apply before any explicit overrides:

### 7.1 ModelConfig Overrides

| Field | Default | Symbio | Rationale |
|-------|---------|--------|-----------|
| `ffnActivation` | `"gelu"` | `"swiglu"` | Better loss/param efficiency |
| `ffnDim` | `4 * nEmbd` | `ceil((8/3) * nEmbd / 64) * 64` | Match param count to standard FFN |

### 7.2 TrainConfig Overrides

| Field | Default | Symbio | Rationale |
|-------|---------|--------|-----------|
| `lr` | `3e-4` | `5e-5` | Best observed across 68 runs |
| `gradClip` | `1.0` | `5.0` | Less aggressive clipping for gated activations |
| `warmupIters` | `0` | `500` | Stability for SwiGLU |
| `spikeThreshold` | `0` | `10.0` | Enable grad spike detection |

### 7.3 SymbioConfig Type

```typescript
export interface SymbioConfig {
  // -- CUSUM monitoring --
  readonly cusumSensitivity: number;     // default: 4.0 (std devs, from symbiogenesis config.py)
  readonly cusumBaselineWindow: number;  // default: 10 (steps for baseline mean/std)

  // -- Metric collection --
  readonly metricsInterval: number;      // default: 50 (steps between expensive metrics)
  readonly trackWeightEntropy: boolean;  // default: true
  readonly trackEffectiveRank: boolean;  // default: true
  readonly trackFreeEnergy: boolean;     // default: true
  readonly trackMIProfiles: boolean;     // default: false (expensive)
  readonly trackPopulationMetrics: boolean; // default: true
  readonly freeEnergyBeta: number;       // default: 0.01 (from symbiogenesis config.py)
  readonly miNumBins: number;            // default: 30 (from symbiogenesis mi_estimator.py)

  // -- Adaptive batch sizing --
  readonly adaptiveBatch: boolean;       // default: true
  readonly batchMin: number;             // default: 8
  readonly batchMax: number;             // default: 64
  readonly batchStep: number;            // default: 4
  readonly calmStepsBeforeRestore: number; // default: 200

  // -- Fitness / ranking --
  readonly fitnessAlpha: number;         // default: 1.0 (accuracy weight, from config.py)
  readonly complexityMode: "params" | "entropy" | "mdl"; // default: "entropy"
  readonly diversityBonus: number;       // default: 0.0
  readonly diversityDecay: "none" | "linear" | "cosine"; // default: "none"

  // -- FFN activation search --
  readonly searchMode: "ffn-activation-search" | "none"; // default: "ffn-activation-search"
  readonly activationPool: readonly string[]; // default: ["gelu", "relu", "silu", "swiglu"]
  readonly searchStrategy: "evolutionary" | "exhaustive"; // default: "evolutionary"
  readonly populationSize: number;       // default: 6
  readonly generations: number;          // default: 4
  readonly selectionStrategy: "topk" | "tournament"; // default: "topk"
  readonly tournamentK: number;          // default: 3
  readonly mutationRate: number;         // default: 0.25
  readonly stepsPerCandidate: number;    // default: 1000
  readonly rankBy: "valLoss" | "fitness"; // default: "valLoss"
  readonly perfWeight: number;           // default: 0.0
  readonly stabilityWeight: number;      // default: 0.0

  // -- Output --
  readonly writeReport: boolean;         // default: true
  readonly writeCandidates: boolean;     // default: true
  readonly writeSummary: boolean;        // default: true
}
```

### 7.4 Config File Format

```json
{
  "cusumSensitivity": 4.0,
  "metricsInterval": 50,
  "adaptiveBatch": true,
  "searchMode": "ffn-activation-search",
  "activationPool": ["gelu", "relu", "silu", "swiglu"],
  "searchStrategy": "evolutionary",
  "populationSize": 6,
  "generations": 4,
  "stepsPerCandidate": 1000,
  "rankBy": "valLoss"
}
```

Unspecified fields use defaults. Schema validated before training starts.

## 8. Metrics — Complete Catalog

Every metric listed here is computed when `--symbio` is active and **stored to the remote DB**. The dashboard renders all of them.

### 8.1 Clipping Telemetry (fix existing gap — stored for ALL runs)

Already computed in `trainer.ts` lines 604-605 but never persisted. Must be added to DB for every run, not just symbio.

| Metric | DB Column | Type | Frequency | Source |
|--------|-----------|------|-----------|--------|
| Gradient clip coefficient | `clip_coef` | REAL | every step | `trainer.ts` (existing) |
| Fraction of gradient clipped | `clip_pct` | REAL | every step | `trainer.ts` (existing) |

### 8.2 CUSUM Change-Point Statistics

Inspired by `symbiogenesis/monitor.py:GelationMonitor`. Four independent CUSUM monitors running in parallel, each using one-sided upper Page's test:

```
baseline: first cusumBaselineWindow steps establish mean μ and std σ
deviation(t) = (signal(t) - μ) / σ
S(t) = max(0, S(t-1) + deviation(t))
alert when S(t) > cusumSensitivity
```

| Metric | DB Column | Type | Frequency | Description |
|--------|-----------|------|-----------|-------------|
| CUSUM on gradNorm | `cusum_grad` | REAL | every step | Detects gradient norm regime shifts |
| CUSUM on clip_pct | `cusum_clip` | REAL | every step | Detects persistent clipping onset |
| CUSUM on tokens_per_sec | `cusum_tps` | REAL | every step | Detects throughput collapse |
| CUSUM on valLoss | `cusum_val` | REAL | on eval | Detects validation divergence |
| Alert bitmask | `cusum_alerts` | INTEGER | every step | Which monitors fired (see bitmask below) |
| Alert description | `cusum_alert_reason` | TEXT | on alert | Human-readable reason for the alert |

**CUSUM alert bitmask:**

| Bit | Signal | Meaning |
|-----|--------|---------|
| 0x01 | gradNorm | Gradient explosion/instability regime |
| 0x02 | clip_pct | Persistent clipping onset |
| 0x04 | tokens_per_sec | Throughput collapse |
| 0x08 | valLoss | Validation loss divergence |

### 8.3 Weight Entropy

Inspired by `symbiogenesis/model.py:Unit.weight_entropy`. Shannon entropy of weight magnitude distribution.

| Metric | DB Column | Type | Frequency | Description |
|--------|-----------|------|-----------|-------------|
| Weight entropy (bits) | `weight_entropy` | REAL | every metricsInterval | H = -Σ p_i log2(p_i) over |w| histogram (100 bins, matching symbiogenesis) |

Algorithm: flatten all model parameters into single vector, take absolute values, discretize into 100 uniform bins between 0 and max(|w|), compute normalized histogram, return Shannon entropy in bits. Identical to `symbiogenesis/model.py` approach but implemented in TypeScript operating on Alpha tensors.

### 8.4 Effective Rank

Inspired by `symbiogenesis/model.py:Unit.effective_rank`. SVD-based dimensionality measure.

| Metric | DB Column | Type | Frequency | Description |
|--------|-----------|------|-----------|-------------|
| Effective rank | `effective_rank` | REAL | every metricsInterval | Mean effective rank across weight matrices |

Algorithm: for each Linear weight matrix, compute SVD (or approximate via power iteration for large matrices), count singular values > 1% of max singular value, average across all layers. Matches the `symbiogenesis/model.py` definition.

### 8.5 Free Energy Proxy

Inspired by `symbiogenesis/training.py`. Combines loss with weight complexity.

| Metric | DB Column | Type | Frequency | Description |
|--------|-----------|------|-----------|-------------|
| Free energy | `free_energy` | REAL | every metricsInterval | F = loss + freeEnergyBeta * weight_entropy |

Uses `freeEnergyBeta` from SymbioConfig (default 0.01, matching `symbiogenesis/config.py:free_energy_beta`).

### 8.6 Population Entropy

Inspired by `symbiogenesis/monitor.py:IterationMetrics.population_entropy`. Thermodynamic entropy of fitness distribution.

| Metric | DB Column | Type | Frequency | Description |
|--------|-----------|------|-----------|-------------|
| Population entropy (nats) | `population_entropy` | REAL | every metricsInterval | Entropy of the fitness distribution across recent loss windows |

For single-run mode: computed over a sliding window of recent loss values, treating them as a probability distribution after softmax normalization. During search mode: computed over the candidate population's fitness distribution.

### 8.7 Activation Distribution

Inspired by `symbiogenesis/monitor.py:IterationMetrics.activation_distribution`.

| Metric | DB Column | Type | Frequency | Description |
|--------|-----------|------|-----------|-------------|
| Activation distribution | `activation_distribution` | TEXT (JSON) | every metricsInterval | JSON map of activation → count across layers |

For single-run mode: `{"swiglu": 6}` (all layers same). During search mode: aggregated across the current candidate population.

### 8.8 Mutual Information Profiles

Inspired by `symbiogenesis/mi_estimator.py`. Binned MI estimation.

| Metric | DB Column | Type | Frequency | Description |
|--------|-----------|------|-----------|-------------|
| I(X;T) — input-representation MI | `mi_input_repr` | REAL | every metricsInterval | Estimated bits (only when trackMIProfiles=true) |
| I(T;Y) — representation-label MI | `mi_repr_output` | REAL | every metricsInterval | Estimated bits (only when trackMIProfiles=true) |
| Compression ratio | `mi_compression` | REAL | every metricsInterval | I(T;Y) / I(X;T) |

Algorithm: matches `symbiogenesis/mi_estimator.py` — per-neuron independence assumption, uniform binning into `miNumBins` bins, joint entropy estimation. Adapted for transformer hidden states instead of feedforward network activations.

### 8.9 Fitness Scores

Inspired by `symbiogenesis/training.py:train_and_eval` fitness computation.

| Metric | DB Column | Type | Frequency | Description |
|--------|-----------|------|-----------|-------------|
| Fitness score | `fitness_score` | REAL | on eval | Multi-objective fitness value |
| Complexity score | `complexity_score` | REAL | every metricsInterval | Complexity component of fitness |

Three complexity modes (from `symbiogenesis/config.py:complexity_mode`):
- `"params"`: `1.0 / (1.0 + num_parameters / 1000.0)`
- `"entropy"`: `weight_entropy / max(1.0, log2(num_parameters))`
- `"mdl"`: `effective_rank * log2(max(2, num_parameters)) / num_parameters`

### 8.10 Adaptive Batch Sizing

| Metric | DB Column | Type | Frequency | Description |
|--------|-----------|------|-----------|-------------|
| Current batch size | `adaptive_batch_size` | INTEGER | every step | Effective batch size (changes on CUSUM triggers) |
| Batch change reason | `batch_change_reason` | TEXT | on change | Why: `"cusum_grad"`, `"cusum_clip"`, `"cusum_tps"`, `"restore"` |

Adaptive logic:
- **CUSUM grad alert (0x01)**: reduce batch by `batchStep`, clamp to `batchMin` — smaller batches recover faster from instability
- **CUSUM throughput alert (0x04)**: reduce batch by `batchStep` — GPU may be under memory pressure
- **CUSUM clip alert (0x02)**: increase batch by `batchStep`, clamp to `batchMax` — larger batches smooth gradients
- **No alerts for `calmStepsBeforeRestore` steps**: restore toward original batch by `batchStep`

### 8.11 Search Candidate Metrics (during FFN activation search)

| Metric | DB Column | Type | Frequency | Description |
|--------|-----------|------|-----------|-------------|
| Candidate ID | `symbio_candidate_id` | TEXT | every step (search mode) | Which candidate this step belongs to |
| Candidate activation | `symbio_candidate_activation` | TEXT | every step (search mode) | Activation being evaluated |
| Candidate generation | `symbio_generation` | INTEGER | every step (search mode) | Evolution generation number |
| Architecture diversity | `architecture_diversity` | REAL | per generation | Fraction of unique architectures in population |

### 8.12 Summary: All New DB Columns on `metrics` Table

| Column | Type | When populated |
|--------|------|---------------|
| `clip_coef` | REAL | every step, ALL runs |
| `clip_pct` | REAL | every step, ALL runs |
| `cusum_grad` | REAL | every step, symbio runs |
| `cusum_clip` | REAL | every step, symbio runs |
| `cusum_tps` | REAL | every step, symbio runs |
| `cusum_val` | REAL | on eval, symbio runs |
| `cusum_alerts` | INTEGER | every step, symbio runs |
| `cusum_alert_reason` | TEXT | on alert, symbio runs |
| `weight_entropy` | REAL | every metricsInterval, symbio runs |
| `effective_rank` | REAL | every metricsInterval, symbio runs |
| `free_energy` | REAL | every metricsInterval, symbio runs |
| `population_entropy` | REAL | every metricsInterval, symbio runs |
| `activation_distribution` | TEXT | every metricsInterval, symbio runs |
| `mi_input_repr` | REAL | every metricsInterval, symbio runs (when enabled) |
| `mi_repr_output` | REAL | every metricsInterval, symbio runs (when enabled) |
| `mi_compression` | REAL | every metricsInterval, symbio runs (when enabled) |
| `fitness_score` | REAL | on eval, symbio runs |
| `complexity_score` | REAL | every metricsInterval, symbio runs |
| `adaptive_batch_size` | INTEGER | every step, symbio runs |
| `batch_change_reason` | TEXT | on change, symbio runs |
| `symbio_candidate_id` | TEXT | every step, symbio search runs |
| `symbio_candidate_activation` | TEXT | every step, symbio search runs |
| `symbio_generation` | INTEGER | every step, symbio search runs |
| `architecture_diversity` | REAL | per generation, symbio search runs |

### 8.13 Summary: All New DB Columns on `runs` Table

| Column | Type | Description |
|--------|------|-------------|
| `symbio` | INTEGER | 0 or 1 — whether this is a symbio run |
| `symbio_config` | TEXT | JSON-serialized SymbioConfig |
| `ffn_activation` | TEXT | Activation function name |
| `symbio_winner` | TEXT | JSON: winning candidate summary (search mode) |
| `symbio_mode` | TEXT | Search mode used (e.g., `"ffn-activation-search"`) |

## 9. Data Flow

### 9.1 Existing Flow (unchanged for non-symbio runs)

```
Trainer step
  → assembles StepMetrics (step, loss, lr, gradNorm, timings, GPU stats)
  → writes to metrics.jsonl buffer (local, flushes every 50 steps)
  → calls onStep(metrics) callback
  → RemoteReporter.onStep() buffers (batch of 10)
  → POST /api/ingest { type: "metrics", runId, metrics: batch }
  → ingest route calls insertMetrics() → Turso metrics table
  → broadcastLive("metrics", ...) → SSE to dashboard
  → Dashboard polls /api/runs/{id}/metrics every 60s + SSE for live updates
```

### 9.2 Extended Flow for Symbio Runs

```
Trainer step
  → assembles StepMetrics (now includes clip_coef, clip_pct for ALL runs)
  → if symbio:
    → CusumDashboard.update(metrics) — updates all 4 monitors
      → computes cusum_grad, cusum_clip, cusum_tps, cusum_alerts
      → if alert: AdaptiveBatch.onAlert(alertMask) → adjusts batch size
    → if step % metricsInterval === 0:
      → SymbioMetricsCollector.collect(model, loss)
        → computeWeightEntropy(model.parameters())
        → computeEffectiveRank(model.parameters())
        → computeFreeEnergy(loss, config.freeEnergyBeta, weightEntropy)
        → computePopulationEntropy(recentLosses)
        → getActivationDistribution(model)
        → computeComplexity(model, config.complexityMode)
        → if trackMIProfiles: computeMI(model, batch)
    → merges all symbio fields into StepMetrics
  → existing flow continues unchanged (jsonl, onStep, remote reporter)
  → POST /api/ingest — same endpoint, batch now has extra nullable fields
  → insertMetrics() — extended SQL includes all new columns
  → Dashboard detects run.symbio === 1 → renders full symbio UI
```

### 9.3 Key Design Decisions

1. **Extend existing `metrics` table** — nullable columns for symbio fields. No separate `symbio_*` tables. Keeps pipeline unified, avoids join complexity, works with existing batch insert pattern.
2. **Same ingest endpoint** — `/api/ingest` already accepts `{ type: "metrics", metrics: [...] }`. New fields are just more keys on each metric object. No new endpoints.
3. **CUSUM state is ephemeral** — accumulators live in trainer memory. Only the current values and alerts are persisted per step. State can be reconstructed from the DB column history if needed.
4. **Expensive metrics are sparse** — weight entropy, effective rank, free energy only populate every `metricsInterval` steps. Dashboard handles null gaps.
5. **Clip telemetry for ALL runs** — `clip_coef` and `clip_pct` are always persisted, not just symbio. Fixes the existing 3-of-68-runs gap.

## 10. SwiGLU Implementation

### 10.1 Architecture Change

Standard FFN (current GELU):
```
h = gelu(x @ W_fc) @ W_proj
```
2 weight matrices: `W_fc` (nEmbd × 4·nEmbd), `W_proj` (4·nEmbd × nEmbd)

SwiGLU FFN:
```
h = (silu(x @ W_gate) ⊙ (x @ W_up)) @ W_proj
```
3 weight matrices: `W_gate` (nEmbd × ffnDim), `W_up` (nEmbd × ffnDim), `W_proj` (ffnDim × nEmbd)

### 10.2 Parameter Count Matching

To keep total params comparable, SwiGLU uses `ffnDim = ceil((8/3) * nEmbd / 64) * 64`:

For L6 D256 H8:
- Standard FFN per block: 2 × (256 × 1024) = 524,288 params
- SwiGLU at ffnDim=704: 2 × (256 × 704) + (704 × 256) = 541,696 params (+3.3%)

### 10.3 ModelConfig Extension

```typescript
export interface ModelConfig {
  readonly vocabSize: number;
  readonly blockSize: number;
  readonly nLayer: number;
  readonly nEmbd: number;
  readonly nHead: number;
  readonly dropout: number;
  readonly ffnActivation?: "gelu" | "silu" | "relu" | "swiglu";  // default: "gelu"
  readonly ffnDim?: number;  // default: 4*nEmbd; swiglu default: ceil((8/3)*nEmbd/64)*64
}
```

### 10.4 Checkpoint Compatibility

- SwiGLU checkpoints are NOT shape-compatible with standard GELU checkpoints (different param count and layout)
- Checkpoint metadata must include `ffnActivation` for validation at load time
- Missing `ffnActivation` in old checkpoints defaults to `"gelu"`
- Shape mismatch at load → clear error with explanation
- Future: checkpoint morphing/projection (out of scope for v1, seam exists)

## 11. DB Schema Migration

### Version 6: Symbio Metrics + Clipping Telemetry

```sql
-- Fix existing gap: clipping telemetry (for ALL runs)
ALTER TABLE metrics ADD COLUMN clip_coef REAL;
ALTER TABLE metrics ADD COLUMN clip_pct REAL;

-- CUSUM change-point statistics
ALTER TABLE metrics ADD COLUMN cusum_grad REAL;
ALTER TABLE metrics ADD COLUMN cusum_clip REAL;
ALTER TABLE metrics ADD COLUMN cusum_tps REAL;
ALTER TABLE metrics ADD COLUMN cusum_val REAL;
ALTER TABLE metrics ADD COLUMN cusum_alerts INTEGER;
ALTER TABLE metrics ADD COLUMN cusum_alert_reason TEXT;

-- Symbio metrics (sparse — every metricsInterval steps)
ALTER TABLE metrics ADD COLUMN weight_entropy REAL;
ALTER TABLE metrics ADD COLUMN effective_rank REAL;
ALTER TABLE metrics ADD COLUMN free_energy REAL;
ALTER TABLE metrics ADD COLUMN population_entropy REAL;
ALTER TABLE metrics ADD COLUMN activation_distribution TEXT;
ALTER TABLE metrics ADD COLUMN mi_input_repr REAL;
ALTER TABLE metrics ADD COLUMN mi_repr_output REAL;
ALTER TABLE metrics ADD COLUMN mi_compression REAL;
ALTER TABLE metrics ADD COLUMN fitness_score REAL;
ALTER TABLE metrics ADD COLUMN complexity_score REAL;

-- Adaptive batch sizing
ALTER TABLE metrics ADD COLUMN adaptive_batch_size INTEGER;
ALTER TABLE metrics ADD COLUMN batch_change_reason TEXT;

-- Search candidate tracking
ALTER TABLE metrics ADD COLUMN symbio_candidate_id TEXT;
ALTER TABLE metrics ADD COLUMN symbio_candidate_activation TEXT;
ALTER TABLE metrics ADD COLUMN symbio_generation INTEGER;
ALTER TABLE metrics ADD COLUMN architecture_diversity REAL;

-- Run-level symbio metadata
ALTER TABLE runs ADD COLUMN symbio INTEGER DEFAULT 0;
ALTER TABLE runs ADD COLUMN symbio_config TEXT;
ALTER TABLE runs ADD COLUMN ffn_activation TEXT;
ALTER TABLE runs ADD COLUMN symbio_winner TEXT;
ALTER TABLE runs ADD COLUMN symbio_mode TEXT;
```

All 24 new `metrics` columns and 5 new `runs` columns are nullable. Non-symbio runs have NULL — zero storage overhead in SQLite.

## 12. Code Changes Required

### 12.1 `packages/core/src/types.ts`

- Add `ffnActivation?: "gelu" | "silu" | "relu" | "swiglu"` to `ModelConfig`
- Add `ffnDim?: number` to `ModelConfig`
- Add `symbio?: boolean` to `TrainConfig`
- Add `symbioConfig?: SymbioConfig | null` to `TrainConfig`
- Add `ffnActivation` and `ffnDim` to `DEFAULT_MODEL_CONFIG` (`"gelu"`, `undefined`)
- Add `symbio: false` and `symbioConfig: null` to `DEFAULT_TRAIN_CONFIG`
- Export `SymbioConfig` type (defined in `@alpha/symbiogenesis`, re-exported through core)

### 12.2 `packages/model/src/gpt.ts`

- Replace hardcoded `matmulTransposedGelu` with activation dispatch on `config.ffnActivation`
- Preserve `matmulTransposedGelu` as the fast path when `ffnActivation === "gelu"` or undefined
- Add `silu` path: `matmulTransposed` + `silu` op
- Add `relu` path: `matmulTransposed` + `relu` op
- Add `swiglu` path: `silu(x @ W_gate) * (x @ W_up)` then `@ W_proj`
- Read `ffnActivation` from `ModelConfig`

### 12.3 `packages/model/src/init.ts`

- Handle SwiGLU parameter initialization: `fc_gate`, `fc_up`, `fc_proj` per block
- Adjust parameter count calculation for SwiGLU
- Use `ffnDim` when specified, otherwise compute default based on activation type

### 12.4 `packages/autograd/src/ops.ts`

- Add `silu` autograd op with forward and backward
- Add `swiglu` composite autograd op (or implement via `silu` + `mul` + `matmulTransposed`)

### 12.5 `packages/tensor/src/cpu_ref.ts`

- Add `siluBackward` implementation
- Verify `silu` forward exists (it does)

### 12.6 `packages/helios/src/backend.ts`

- Add `siluBackward` SPIR-V kernel
- Optionally add fused `matmulTransposedSilu` (can defer)

### 12.7 `packages/train/src/trainer.ts`

- Import symbio types from `@alpha/symbiogenesis`
- Extend `StepMetrics` interface with all new symbio fields (all optional)
- When `trainConfig.symbio`:
  - Create `CusumDashboard` and `AdaptiveBatch` at training start
  - Create `SymbioMetricsCollector` at training start
  - After assembling base StepMetrics, call `cusumDashboard.update(metrics)`
  - At `metricsInterval` steps, call `collector.collect(model, loss)`
  - Merge all symbio fields into the StepMetrics object
  - Apply adaptive batch size changes for next step
- When NOT symbio: zero overhead, no symbio code runs

### 12.8 `packages/train/src/remote-reporter.ts`

- No structural changes. Already `JSON.stringify`s the full `StepMetrics` object.
- New fields are automatically serialized and transmitted.
- Verify flush batch size handles larger metric payloads efficiently.

### 12.9 `packages/db/src/schema.ts`

- Add version 6 migration (see §11)

### 12.10 `packages/db/src/types.ts`

- Add all 24 new columns to `DbMetric` interface (all nullable)
- Add 5 new columns to `DbRun` interface

### 12.11 `packages/db/src/metrics.ts`

- Extend `insertMetrics` function signature with all new fields
- Extend INSERT SQL to include all new columns
- Keep batch insert pattern (500 rows per chunk)

### 12.12 `apps/server/src/app/api/ingest/route.ts`

- On `"run_start"`: persist `symbio`, `symbio_config`, `ffn_activation`, `symbio_mode` from the run metadata
- On `"metrics"`: no changes needed — `insertMetrics` already receives the full metric objects
- Store `infra` field (currently ignored — fix this while we're here)

### 12.13 `apps/cli/src/commands/train.ts`

- Parse `--symbio` (boolean), `--symbio-config` (string), `--activation` (string)
- When `--symbio`:
  - Load `SymbioConfig` from file or defaults
  - Apply symbio preset to ModelConfig and TrainConfig (§7)
  - Apply any explicit CLI overrides on top
  - Pass `symbio: true` and `symbioConfig` through to trainer
- When `--activation` (without --symbio): just set `ffnActivation` on ModelConfig
- Log symbio activation details to console

### 12.14 `apps/web/src/components/run-detail-view.tsx`

- Detect symbio run from `run.symbio === 1`
- Render all symbio-specific UI sections (see §13)
- Add symbio metrics to the poll cycle (already fetched as part of metrics)

### 12.15 `apps/web/src/components/charts.tsx`

- Add all new chart components (see §13)

### 12.16 `scripts/gcp_train.py`

- Add `--symbio` (store_true), `--symbio-config` (string), `--activation` (string) to argparse
- Pass through to Node CLI command

## 13. Dashboard — Symbio Run UI

When viewing a run where `run.symbio === 1`, the dashboard renders everything it renders for normal runs, **plus** the following symbio-specific sections. All charts follow existing Alpha dashboard patterns (dark theme, interactive hover tooltips, responsive grid).

### 13.1 Symbio Header Badge

- "Symbio" badge next to the run status badge in the header
- Activation function tag (e.g., "SwiGLU")
- Search mode tag if applicable (e.g., "FFN Activation Search")

### 13.2 Symbio Stats Grid (after existing timing stats grid)

8 tiles in `grid-cols-2 sm:grid-cols-4 lg:grid-cols-8`:

| Tile | Value | Source |
|------|-------|--------|
| Weight Entropy | latest value (bits) + sparkline | `weight_entropy` |
| Effective Rank | latest value + sparkline | `effective_rank` |
| Free Energy | latest value + trend arrow | `free_energy` |
| Population Entropy | latest value (nats) | `population_entropy` |
| Complexity | latest value + mode label | `complexity_score` |
| Fitness | latest value | `fitness_score` |
| CUSUM Alerts | total count + last alert step | `cusum_alerts` |
| Batch Size | current adaptive value | `adaptive_batch_size` |

### 13.3 CUSUM Monitor Chart

- 4 lines: `cusum_grad`, `cusum_clip`, `cusum_tps`, `cusum_val`
- Horizontal dashed line at `cusumSensitivity` threshold
- Alert events as vertical markers (same pattern as existing checkpoint/spike markers)
- Color-coded by signal type
- X-axis: step. Y-axis: CUSUM accumulator value.
- Hover tooltip shows all 4 values + alert status for that step.

### 13.4 Clip Telemetry Chart (shown for ANY run with clip data, not symbio-only)

- Dual-axis: `clip_coef` (left, 0–1) and `clip_pct` (right, 0–1)
- X-axis: step
- Fills the existing gap — this benefits all 68+ runs retroactively once clip data is persisted.

### 13.5 Symbio Metrics Charts (grid of 4 mini charts)

| Chart | Data | Style |
|-------|------|-------|
| Weight Entropy | `weight_entropy` over steps | Line, sparse points |
| Effective Rank | `effective_rank` over steps | Line, sparse points |
| Free Energy | `free_energy` over steps | Line with loss overlay |
| Fitness & Complexity | `fitness_score` + `complexity_score` | Dual-axis |

All handle sparse data (points only at `metricsInterval` steps).

### 13.6 Adaptive Batch Size Chart

- Step chart (not line) showing `adaptive_batch_size` over time
- Trigger annotations from `batch_change_reason` at each transition
- X-axis: step. Y-axis: batch size.

### 13.7 Information Plane (when MI data present)

Inspired by `symbiogenesis/visualize.py:plot_information_plane`.

- Scatter plot: `mi_input_repr` (X) vs `mi_repr_output` (Y)
- Points colored by step (gradient from early=blue to late=red)
- Shows the information bottleneck trajectory over training

### 13.8 MI Trajectory Chart (when MI data present)

- 3 lines: `mi_input_repr`, `mi_repr_output`, `mi_compression`
- X-axis: step

### 13.9 Activation Distribution Chart (search mode)

Inspired by `symbiogenesis/visualize.py:plot_activation_distribution`.

- Stacked area chart showing activation type proportions over generations
- Parsed from `activation_distribution` JSON column

### 13.10 Search Candidate Comparison (search mode)

- Table/cards showing each candidate: activation, generation, best valLoss, fitness, throughput, stability
- Winner highlighted
- Sortable by any metric column

### 13.11 Symbio Config Panel (collapsible, like existing raw JSON section)

- Rendered JSON of the resolved `SymbioConfig`
- Shows what symbio settings were active for this run

## 14. FFN Activation Search — Orchestration

### 14.1 Purpose

Evaluate FFN activation choices and select the best under a fixed budget. Inspired by `symbiogenesis/population.py` evolutionary search but adapted for transformer activation functions instead of network topology.

### 14.2 Search Flow

```
1. Parse activationPool and evolution config
2. Generate initial population (one candidate per activation, plus duplicates to fill populationSize)
3. For each generation:
   a. Train each candidate for stepsPerCandidate steps
   b. Evaluate fitness (valLoss + optional perf/stability weights)
   c. Select top-k candidates (or tournament selection)
   d. Mutate: swap activation with mutationRate probability
   e. Record metrics, write candidates.jsonl
4. Rank final population
5. Write symbio-summary.json, symbio-report.md
6. If winner differs from initial activation, log recommendation
```

### 14.3 Fitness Function

Inspired by `symbiogenesis/training.py:train_and_eval`:

```
fitness = fitnessAlpha * (-best_val_loss)
        + perfWeight * log(tokens_per_sec_mean)
        + stabilityWeight * (-instability_score)
```

Where `instability_score` includes:
- Spike rate (gradNorm > spikeThreshold)
- Clip saturation (persistent low clip_coef)
- NaN/skip events
- Step-time variance after warmup

### 14.4 Artifacts

- `symbio-summary.json`: mode, config, winner activation, winner metrics, population final state
- `symbio-candidates.jsonl`: one line per candidate per generation, full metrics
- `symbio-report.md`: human-readable summary with rankings, deltas, recommendations

## 15. GCP Script Changes

`scripts/gcp_train.py` additions:

```python
parser.add_argument("--symbio", action="store_true", help="Activate Symbiogenesis mode")
parser.add_argument("--symbio-config", type=str, help="Symbio config JSON path")
parser.add_argument("--activation", type=str, help="FFN activation function")
```

Passed through to Node CLI. Example:

```bash
python3 scripts/gcp_train.py --data data/super_chat.txt --domain chat \
  --iters 50000 --batch 20 --block 512 --dim 384 --heads 8 --layers 8 \
  --backend helios --zone us-central1-b --machine-type g2-standard-4 \
  --symbio --stop-after
```

## 16. Backward Compatibility

| Concern | Guarantee |
|---------|-----------|
| Non-symbio runs | Identical behavior. No symbio code on hot path. All new DB columns nullable. |
| Existing checkpoints | Load without modification. Missing `ffnActivation` defaults to `"gelu"`. |
| Remote reporter | New fields silently included in JSON. Old API ignores unknown fields. New API stores them. |
| Dashboard | Old runs render exactly as before. Symbio sections gated on `run.symbio === 1`. |
| DB migration | All ALTERs are additive nullable columns. No data loss. Forward-only migration. |
| Ingest endpoint | Backward-compatible — existing metric payloads still work. Extra fields accepted when present. |

## 17. Performance Requirements

### 17.1 Baseline Safety

- Zero symbio overhead when `--symbio` is off
- No conditional checks on the training hot path when symbio is disabled
- Alpha default training path remains behaviorally identical

### 17.2 Symbio Overhead Budget

- CUSUM update: <0.1ms per step (4 running accumulators, trivial math)
- Expensive metrics at metricsInterval=50: <50ms per collection (amortized <1ms per step)
- Adaptive batch logic: <0.01ms per step (conditional on alert bitmask)
- MI profiles (when enabled): may cost 100-500ms — only collected at metricsInterval, off by default
- Total symbio overhead target: <2% of step time

### 17.3 FFN Activation Safety

- GELU path preserves fused `matmulTransposedGelu` — zero regression
- SwiGLU path may be 10-30% slower due to 3 matmuls vs 2 — documented in artifacts
- Non-GELU correctness before optimization in v1

## 18. Testing Requirements

### 18.1 Unit Tests

- CUSUM monitor detects known regime shifts in synthetic data
- CUSUM alert bitmask correctly encodes multiple simultaneous alerts
- Weight entropy returns higher values for random weights, lower for structured
- Effective rank returns values in [1, min(m,n)] range
- Free energy combines loss and entropy correctly
- Adaptive batch sizing responds to alerts with correct direction and clamping
- Fitness function produces expected rankings for known inputs
- Complexity modes (`params`, `entropy`, `mdl`) each produce valid scores

### 18.2 Integration Tests

- `--symbio` flag parsed and applied, preset overrides in correct order
- `--symbio --lr=1e-4` overrides preset LR
- `--activation=silu` without `--symbio` works (just sets activation)
- `StepMetrics` includes all symbio fields when active
- `StepMetrics` does NOT include symbio fields when inactive
- Remote reporter serializes and transmits symbio metrics
- Ingest endpoint persists new columns
- DB migration v6 applies cleanly on fresh and existing databases
- Non-symbio ingest still works after migration

### 18.3 Model Tests

- `ffnActivation=gelu` produces same output as current code (regression guard)
- `ffnActivation=silu` forward and backward correct on cpu_ref
- `ffnActivation=relu` forward and backward correct on cpu_ref
- `ffnActivation=swiglu` forward and backward correct on cpu_ref
- SwiGLU parameter count matches expectation for given ffnDim

### 18.4 End-to-End Tests

- `alpha train --data <small_file> --symbio --iters=100 --backend cpu_ref` completes
- Metrics written to local JSONL include symbio fields
- Remote reporter transmits symbio metrics when ALPHA_REMOTE_URL is set
- Dashboard renders symbio UI sections for the resulting run
- Non-symbio run after migration renders unchanged

### 18.5 Dashboard Tests

- Non-symbio run pages render unchanged (regression guard)
- Symbio badge appears only for symbio runs
- CUSUM chart renders with threshold lines and alert markers
- Symbio metrics charts handle sparse data without errors
- Adaptive batch chart shows step transitions
- Clip telemetry chart appears for any run with clip_coef data
- Information plane and MI charts render when MI data present
- Search candidate table renders with correct rankings
- Partial/in-progress symbio data renders without crashing

## 19. Rollout Plan

### Phase A: Core Types + Clip Telemetry Fix

- Add `ffnActivation`, `ffnDim` to `ModelConfig` in `@alpha/core`
- Add `symbio`, `symbioConfig` to `TrainConfig`
- Define `SymbioConfig` type
- Add `clip_coef` and `clip_pct` to DB schema (migration v6, first two ALTERs)
- Add `clip_coef` and `clip_pct` to `insertMetrics` and `DbMetric`
- Persist clip telemetry for ALL runs

Exit criteria: all runs start storing clip data. New types available.

### Phase B: FFN Activation Support

- Activation dispatch in `gpt.ts` (gelu fast path preserved)
- `silu` autograd op + `siluBackward` in cpu_ref and helios
- SwiGLU forward/backward path
- Parameter init for SwiGLU (`fc_gate`, `fc_up`, `fc_proj`)
- Checkpoint metadata includes `ffnActivation`

Exit criteria: `alpha train --activation=swiglu` completes a single run correctly.

### Phase C: `@alpha/symbiogenesis` Package + CLI

- Package skeleton with all modules from §6.1
- CUSUM monitoring (`cusum.ts`, `dashboard.ts`)
- Adaptive batch sizing (`adaptive-batch.ts`)
- Metric collectors (weight entropy, effective rank, free energy, etc.)
- `--symbio`, `--symbio-config`, `--activation` CLI parsing
- Symbio preset application with override semantics
- Trainer integration: hook symbio monitoring into step loop

Exit criteria: `alpha train --symbio --iters=100` completes with all symbio metrics in local JSONL.

### Phase D: DB Migration + Remote Pipeline

- Full migration v6 (all 29 new columns)
- Extended `insertMetrics` with all symbio columns
- Ingest route stores `symbio`, `symbio_config`, `ffn_activation` on run
- Verify remote reporter transmits all symbio fields
- GCP script `--symbio` passthrough

Exit criteria: symbio metrics visible in Turso DB after a remote-reported training run.

### Phase E: Dashboard UI

- Symbio badge and activation tag in run header
- Symbio stats grid (8 tiles)
- CUSUM monitor chart
- Clip telemetry chart (all runs)
- Symbio metrics mini charts (weight entropy, effective rank, free energy, fitness)
- Adaptive batch size chart
- Information plane and MI trajectory (conditional)
- Symbio config panel
- Activation distribution chart (search mode)

Exit criteria: full symbio dashboard renders for a symbio run. Non-symbio runs unchanged.

### Phase F: FFN Activation Search

- Search orchestrator
- Candidate generation, evaluation, selection, mutation
- Fitness ranking with configurable weights
- Artifact generation (summary, candidates, report)
- Search candidate columns in DB
- Search candidate comparison UI on dashboard

Exit criteria: `alpha train --symbio` runs full evolutionary FFN activation search and displays results on dashboard.

## 20. Acceptance Criteria (v1)

1. `alpha train` without `--symbio` preserves current behavior exactly. Zero overhead.
2. `alpha train --symbio` activates Symbiogenesis mode with SwiGLU default and all monitoring.
3. All 24 symbio metric columns persisted to remote DB via existing pipeline.
4. All 5 run-level symbio columns persisted on run creation.
5. `clip_coef` and `clip_pct` stored for ALL runs (not just symbio).
6. Dashboard renders full symbio UI when viewing a symbio run: stats grid, CUSUM chart, metrics charts, adaptive batch chart, clip chart, config panel.
7. Dashboard renders search candidate comparison and activation distribution for search-mode runs.
8. Non-symbio run pages render unchanged.
9. FFN activation dispatch works for `gelu`, `silu`, `relu`, `swiglu` with GELU fused path preserved.
10. CUSUM monitoring detects regime shifts and triggers adaptive batch sizing.
11. GCP script passes `--symbio` through correctly.
12. Symbio artifacts written to run directory.
13. DB migration v6 applies cleanly on existing production database.

## 21. Risks and Mitigations

**Risk:** Symbio mode grows into a hidden second training stack.
**Mitigation:** Alpha trainer remains the only step-execution engine. Symbio package is monitoring/orchestration only.

**Risk:** `--symbio` becomes a black box.
**Mitigation:** Full resolved config logged at startup. Every metric stored to DB. Dashboard shows all internal state.

**Risk:** Symbio metrics bloat the metrics table.
**Mitigation:** All nullable, NULL = zero storage in SQLite. Expensive metrics are sparse (every 50 steps). CUSUM values are 4 floats per step — negligible.

**Risk:** Migration breaks existing DB.
**Mitigation:** All ALTERs add nullable columns — purely additive. Test on copy of production DB before deploy.

**Risk:** SwiGLU slower than GELU.
**Mitigation:** Expected and documented. Throughput deltas included in fitness ranking. GELU fused path unchanged.

**Risk:** Baseline regression from activation dispatch.
**Mitigation:** `ffnActivation === "gelu"` (or undefined) always takes the existing `matmulTransposedGelu` fused path. No dispatch overhead for default.

**Risk:** MI profile computation too expensive.
**Mitigation:** Off by default (`trackMIProfiles: false`). Only runs at `metricsInterval` when enabled. Budget: <500ms per collection.

## 22. Open Decisions — Resolved

| # | Question | Decision |
|---|----------|----------|
| 1 | Package name? | `@alpha/symbiogenesis` |
| 2 | Default config location? | Hardcoded defaults in package. No external file required. |
| 3 | Winner auto-apply? | Report only in v1. User decides whether to use winner activation for full run. |
| 4 | DB schema approach? | Extend existing `metrics` table with nullable columns. No separate tables. |

---

## Appendix A: Write Everything From Scratch

**This section is a binding constraint on implementation.**

All Symbiogenesis logic in Alpha must be written from scratch in TypeScript. The original Python `symbiogenesis/` repo is inspiration only:

| Symbiogenesis Python | Alpha TypeScript (write from scratch) |
|---------------------|--------------------------------------|
| `monitor.py:GelationMonitor` | `packages/symbiogenesis/src/monitor/cusum.ts` — CUSUM algorithm reimplemented for Alpha's signal types (gradNorm, clip_pct, tokens_per_sec, valLoss instead of avg_depth/avg_width) |
| `model.py:Unit.weight_entropy` | `packages/symbiogenesis/src/metrics/weight-entropy.ts` — same 100-bin histogram approach, operating on Alpha `Tensor` arrays |
| `model.py:Unit.effective_rank` | `packages/symbiogenesis/src/metrics/effective-rank.ts` — SVD-based with 1% threshold, operating on Alpha weight matrices |
| `training.py:train_and_eval` (fitness) | `packages/symbiogenesis/src/metrics/fitness.ts` — same 3 complexity modes (params/entropy/mdl), adapted for transformer loss/params |
| `training.py` (free energy) | `packages/symbiogenesis/src/metrics/free-energy.ts` — `loss + beta * entropy`, same formula |
| `mi_estimator.py` | `packages/symbiogenesis/src/metrics/mi-estimator.ts` — binned MI with per-neuron independence, adapted for transformer hidden states |
| `config.py:Config` | `packages/symbiogenesis/src/config/schema.ts` — SymbioConfig type with Alpha-relevant subset of fields |
| `population.py` (evolutionary search) | `packages/symbiogenesis/src/search/orchestrator.ts` — evolutionary search over activation functions instead of network topology |
| `monitor.py:IterationMetrics` | Extended `StepMetrics` in `packages/train/src/trainer.ts` — all metrics as optional fields |
| `visualize.py` (16 plot types) | Dashboard components in `apps/web/src/components/` — React/Canvas charts matching existing Alpha style |

**Do not:**
- Import, require, or dynamically load any Python code
- Transpile or auto-convert Python to TypeScript
- Shell out to Python subprocesses
- Use any Python ML/stats libraries via FFI
- Copy-paste Python code and "port" it line-by-line — understand the algorithm, then write idiomatic TypeScript

**Do:**
- Read the Python source to understand algorithms, thresholds, and design decisions
- Reimplement the same mathematical algorithms in TypeScript
- Use Alpha's existing tensor operations, autograd, and backend infrastructure
- Follow Alpha's patterns: ESM, readonly interfaces, pure functions, zero dependencies
- Adapt concepts for the transformer/GPT domain (the Python repo targets small feedforward networks)
