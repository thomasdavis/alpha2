# Symbiogenesis for Alpha: Performance + Results Strategy (2026-02-25)

## Executive Summary

Symbiogenesis is useful for `alpha`, but not because the Python MLP fusion code can be copied directly into the GPT training path.

The real leverage is:

1. A symbiogenesis-style outer loop for `alpha` run optimization (config evolution under fixed compute budgets).
2. Symbiogenesis-style instrumentation and change-point detection for online trainer control (stability + throughput).
3. Symbiogenesis-style weight transfer, adapted to `alpha` checkpoints as a safe parameter morph/projection layer (for cheaper architecture sweeps).

This can improve both:

- `results` (best val loss/perplexity, stability, reproducibility)
- `performance` (tokens/sec under stable settings, better runtime policy selection)

The highest-ROI path is a TypeScript package that evolves `ModelConfig + TrainConfig + runtime policy` using `alpha`'s existing CLI/trainer/checkpoint/metrics infrastructure.

## What I Researched

### Symbiogenesis (implemented features, not just docs)

I audited the local `models/alpha/symbiogenesis` code/docs/tests and confirmed the core ideas are implemented and tested:

- Population + fusion + replacement loop: `models/alpha/symbiogenesis/symbiogenesis/main.py`, `models/alpha/symbiogenesis/symbiogenesis/population.py`, `models/alpha/symbiogenesis/symbiogenesis/fusion.py`
- Multi-objective fitness / free-energy-ish scoring: `models/alpha/symbiogenesis/symbiogenesis/training.py`
- CUSUM gelation detection + metrics history: `models/alpha/symbiogenesis/symbiogenesis/monitor.py`
- MI / information bottleneck metrics: `models/alpha/symbiogenesis/symbiogenesis/mi_estimator.py`
- Kuramoto population dynamics: `models/alpha/symbiogenesis/symbiogenesis/kuramoto.py`
- Reservoir front-end (fixed random recurrent features): `models/alpha/symbiogenesis/symbiogenesis/reservoir.py`
- Compute-budget-equivalent baselines and benchmark harness: `models/alpha/symbiogenesis/symbiogenesis/baselines.py`, `models/alpha/symbiogenesis/symbiogenesis/benchmark.py`
- Broad integration test coverage across phases: `models/alpha/symbiogenesis/tests/test_integration.py`

Important caveat: I could not execute the Python tests here because `pytest` is not installed in this environment (the package declares it in dev extras).

### Alpha (integration surface for a symbiogenesis-style optimizer)

`alpha` already has most of what an outer-loop optimizer needs:

- Structured configs: `ModelConfig` / `TrainConfig` (`models/alpha/packages/core/src/types.ts`)
- Fixed-architecture init + parameter collection: `models/alpha/packages/model/src/gpt.ts`
- Rich per-step trainer telemetry + callbacks + metrics JSONL + checkpoints: `models/alpha/packages/train/src/trainer.ts`
- CLI train command with domain defaults and all key knobs exposed: `models/alpha/apps/cli/src/commands/train.ts`
- Local DB sync and queryable metrics schema: `models/alpha/packages/db/src/schema.ts`, `models/alpha/packages/db/src/metrics.ts`, `models/alpha/packages/db/src/sync.ts`
- Bench command (including train benchmark): `models/alpha/apps/cli/src/commands/bench.ts`

## Hard Data From Existing Alpha Runs (local `runs/*`)

I scanned local run artifacts (`config.json` + `metrics.jsonl`) and extracted run-level summaries from **68 runs**.

### Coverage Snapshot

- Total runs with both `config.json` and `metrics.jsonl`: `68`
- Domain counts:
  - `(none)`: `30`
  - `novels`: `23`
  - `chat`: `13`
  - `concordance`: `2`
- Backend counts:
  - `helios`: `39`
  - `cpu_ref`: `29`

### Instrumentation Coverage Gaps (matters for any optimizer)

- Runs with any `valLoss` logged: `32`
- Runs without `valLoss`: `36`
- Runs with `clip_coef` / `clip_pct` telemetry present in `metrics.jsonl`: `3`
- Runs with GPU utilization telemetry present: `5`

This is the biggest practical blocker to a symbiogenesis-style optimizer: historical fitness quality is inconsistent because evaluation and stability telemetry are incomplete across runs.

### Chat + Helios Findings (most relevant to your current quality/perf goals)

`chat` + `helios` runs found: `13`

- `6` runs are the target architecture `L6 D256 H8 B256`
- `5` runs are `L4 D128 H4 B256`
- `1` run is `L4 D128 H4 B128`
- `1` run is `L2 D64 H4 B64`

#### Strong signal: throughput is nearly flat across several 6x256 chat configs

For `L6 D256 H8 B256`, `batch=16`, `helios`, mean throughput is roughly:

- `~7643 - 7803 tok/s`

while best validation loss varies significantly:

- best observed `valLoss`: `4.6771`
- worse configs in same family: `5.15+`

This is exactly where symbiogenesis helps:

- search can improve `results` substantially without giving up much `performance`, because throughput is mostly architecture/runtime-bound and similar across nearby optimizer hyperparameters.

#### Best sampled chat result (within this local run set)

Best `chat+helios` val loss observed in local runs:

- Run: `20260223_171025`
- Config family: `L6 D256 H8 B256`, `batch=16`
- `lr=5e-5`, `beta2=0.95`, `weightDecay=0.1`, `gradClip=5`
- Best `valLoss=4.6771` at step `18000`

#### Stability is still a major issue in good runs

Even good-value runs show large gradient spikes in the mined metrics (using `gradNorm` thresholds and max values), consistent with the instability analysis in:

- `models/alpha/TRAINING_STABILITY.md`

This reinforces the need for:

- stability-aware fitness (not just best val loss)
- online change-point detection and branching/pruning

## What Symbiogenesis Can Improve in Alpha (Ranked by ROI)

## 1. Outer-Loop Evolution of Alpha Run Configs (Highest ROI)

### Why this is the direct fit

Symbiogenesis's strongest proven contributions are not “biology-inspired math” but engineering patterns:

- compute-budget-equivalent comparisons
- diversity maintenance
- adaptive population sizing
- multi-objective fitness
- multi-seed aggregation
- strong monitoring + run summaries

`alpha` already has the infrastructure to support this with minimal core-model changes.

### What evolves (genome)

Start with config-level evolution only.

`ModelConfig` genes (small, bounded):

- `nLayer`
- `nEmbd`
- `nHead` (must divide `nEmbd`)
- `blockSize`
- `dropout`

`TrainConfig` genes (high ROI first):

- `lr`, `lrMin`, `warmupIters`
- `beta2`
- `weightDecay`
- `gradClip`
- `batchSize`, `gradAccumSteps`
- `packed`
- `syncEvery`, `gcEvery`
- `spikeThreshold`

Optional categorical genes (later):

- backend (`cpu_ref`/`helios`) for correctness-vs-speed phases
- domain preset family as initialization prior

### Why this improves performance and results simultaneously

Symbiogenesis in `alpha` should optimize a joint objective, not “best val loss” alone.

A practical fitness for `alpha`:

```text
fitness = - val_loss_best
          + a * log(tokens_per_sec_mean)
          - b * instability_penalty
          - c * compute_cost_penalty
          - d * params_penalty
```

Where:

- `instability_penalty` can include:
  - spike rate (`gradNorm > threshold`)
  - clip saturation rate / low `clip_coef`
  - NaN/skip events
  - high step-time variance after warmup
- `compute_cost_penalty` can be fixed-token or fixed-walltime normalization term

This is the direct `alpha` analog of symbiogenesis's multi-objective fitness and free-energy-like scoring.

### Compute budget equivalence (copy this from symbiogenesis)

Symbiogenesis explicitly normalizes baselines by compute budget. `alpha` should do the same.

Use two modes:

1. `fixed_token_budget` (fair for model quality comparisons)
2. `fixed_walltime_budget` or `fixed_gpu_seconds` (fair for throughput/perf optimization)

Do not use raw `iters` as the primary budget when comparing configs with different:

- `batchSize`
- `gradAccumSteps`
- `blockSize`
- backend/runtime policy

## 2. Symbiogenesis-Style Online Trainer Control (High ROI, Low Intrusion)

### Why this matters for Alpha now

`alpha` trainer already emits rich per-step metrics:

- `loss`, `valLoss`
- `gradNorm`
- `tokens_per_sec`, `ms_per_iter`
- timing breakdowns
- GPU utilization / VRAM / pool metrics
- `clip_coef`, `clip_pct`

and already supports adaptive runtime controls (`syncEvery`, `gcEvery`) and spike skipping.

The missing piece is robust online detection and response logic.

### Port the best symbiogenesis monitoring idea: CUSUM/change-point detection

Do not port “gelation” literally.

Port the change-point detector pattern to `alpha` on:

- `gradNorm` (stability regime shift)
- `clip_coef` / `clip_pct` (persistent clipping onset)
- `tokens_per_sec` or `ms_per_iter` (throughput collapse)
- `timing_flush_ms` / GPU pool metrics (memory pressure regime shifts)

Example triggers:

- `CUSUM(gradNorm_log)` crosses threshold -> branch run or reduce LR / increase grad clip strictness / enable spike threshold
- `CUSUM(tokens_per_sec)` downward shift -> mutate runtime policy (`syncEvery`, `gcEvery`, `packed`, `accum`)
- `CUSUM(valLoss)` plateau + stable throughput -> exploit branch (lower LR / longer schedule)

### Result

This turns `alpha` from a static trainer into a controlled dynamical system without touching model kernels.

## 3. Weight Transfer / Projection for Alpha Checkpoints (Medium-High ROI)

### Why this is the real “symbiogenesis” path for Alpha

Symbiogenesis gains a lot from fusion + projection-based weight transfer. The `alpha` equivalent is:

- parameter morphing across nearby GPT architectures
- warm-starting children from parent checkpoints

This is where search gets much cheaper.

### Current state in Alpha (important)

`alpha` has checkpoint save/load and restore, but restore is effectively same-shape copy (plus legacy QKV compatibility), not morph-aware.

`restoreParams()` currently copies element-by-element into destination arrays without general shape projection logic.

### Recommended addition: `restoreParamsProjected()`

Implement a safe projection/truncation/zero-pad path (symbiogenesis-style) for nearby architecture changes.

High-value morphs:

1. `nLayer` changes
- copy shared prefix layers
- initialize extra layers fresh (or from cloned parent layer)

2. `nEmbd` width changes
- project matrices with overlap copy + zero/init pad
- affects embeddings, QKV, `wo`, MLP, layer norms, `lmHead`

3. `blockSize` changes
- copy overlapping rows of `wpe` (positional embeddings)

4. `nHead` changes with constant `nEmbd`
- parameter shapes mostly unchanged in this implementation
- forward semantics change due head partitioning only
- extremely cheap search branch (great candidate for symbio mutations)

### Why this helps performance too

Warm-started children require fewer steps to reach informative validation metrics.

That means:

- lower search cost
- more candidate evaluations per GPU-hour
- better quality frontier under fixed compute

## 4. Adaptive Population + Diversity for Alpha Experiments (Medium ROI)

### Why to port this

Your local run history already shows many repeated nearby configs. Symbiogenesis's diversity and adaptive population ideas solve exactly that.

Port these patterns to `alpha` experiment orchestration:

- diversity bonus for novel config regions
- population growth when improvements continue
- population shrink when search stagnates
- replacement based on Pareto dominance, not single scalar only

### Alpha-specific diversity dimensions

Diversity should be measured over:

- architecture tuple: `(nLayer, nEmbd, nHead, blockSize)`
- training dynamics tuple: `(lr, beta2, weightDecay, gradClip, warmupIters)`
- runtime policy tuple: `(batch, accum, packed, syncEvery, gcEvery)`

This avoids spending 80% of search on near-duplicates.

## 5. Activation/Architecture Evolution for Alpha (Constrained, Later)

### What to import from symbiogenesis

Symbiogenesis Phase 10 shows activation evolution can matter a lot in its MLP setting.

For `alpha`, do not start with arbitrary per-layer activation mutation in the transformer.

Instead, define a constrained set of architecture/operator variants:

- FFN activation: GELU vs SiLU
- FFN structure: standard MLP vs SwiGLU (later, requires parameter/layout changes)
- attention stabilization toggles:
  - QK-LayerNorm (high quality/stability upside)
  - softcap strength variants
- dropout policy variants

These are much more likely to improve real transformer results than porting ELU selection from the MLP benchmark.

## What Not to Port (Now)

Low immediate ROI for `alpha` perf/results:

- Kuramoto population dynamics in the trainer core
- Reservoir computing front-ends in GPT training
- MI/IB metrics as online optimization targets (too noisy/expensive for first pass)

They are useful research modules, but they do not attack your current bottlenecks as directly as config evolution + stability control + checkpoint morphing.

## Concrete Alpha Implementation Plan (TypeScript, staged)

## Phase A: Build the Outer-Loop Search Harness (No model changes)

### Deliverable

New package/command:

- `packages/symbio-search` (or `packages/evolve`)
- CLI command: `alpha evolve`

### Core loop (symbiogenesis-inspired, alpha-native)

1. Initialize population of candidate configs from:
   - domain defaults (`getDomain()`)
   - known-good seeds from past runs
   - random perturbations
2. Train each candidate for a short budget (fixed tokens or walltime)
3. Evaluate fitness from `metrics.jsonl` + `valLoss`
4. Select parents (fitness + diversity)
5. Fuse/mutate configs
6. Replace weak candidates
7. Continue until budget exhausted

### Integration points in Alpha

- Train execution via existing `runTrain`/CLI path
- Metrics from `metrics.jsonl`
- Run metadata from `config.json`
- Optional DB-backed run cache via `@alpha/db`

### Immediate constraints (enforce these)

- require `evalInterval` and validation data (or explicit split) for all search runs
- require telemetry completeness (`valLoss`, `gradNorm`, timing fields)
- fixed token budget mode by default

## Phase B: Add Stability-Aware Fitness + Change-Point Monitors

### Deliverable

Scoring + anomaly module consuming `StepMetrics`

Add:

- CUSUM detectors on `gradNorm`, `ms_per_iter`, `tokens_per_sec`
- plateau detector on `valLoss`
- instability penalties
- early pruning of bad candidates

### Why this matters now

Your chat runs show similar throughput across configs but large stability/val differences. Stability-aware pruning will save a lot of wasted GPU time.

## Phase C: Checkpoint Morphing / Projection (True Symbiogenesis Transfer)

### Deliverable

New checkpoint restore modes:

- `exact` (current behavior)
- `prefix` (layer subset copy)
- `project` (shape projection/truncation/padding)

### Candidate child generation modes

- `fresh`
- `warm_start_exact`
- `warm_start_projected`

### Safety requirements

- shape checks before copy
- explicit initialization policy for unmatched regions
- no optimizer-state reuse across shape changes (reset optimizer state)

## Phase D: Runtime Policy Evolution (Performance-first branch)

### Deliverable

A population specialized for throughput / efficiency over a fixed architecture:

- `batchSize`
- `gradAccumSteps`
- `syncEvery`
- `gcEvery`
- `packed`
- optional `fp16` and checkpointing toggles

### Objective

Maximize:

- `tokens/sec`

Subject to:

- no loss divergence relative to baseline
- no OOM / pool runaway
- acceptable `valLoss`

This is the cleanest way to use symbiogenesis for Helios performance gains without touching kernels.

## Engineering Gaps to Fix Before/Alongside This Work

## 1. Persist missing stability telemetry in local DB

Trainer emits `clip_coef` / `clip_pct`, but local DB schema + insert path currently do not store them.

Add columns + ingestion for:

- `clip_coef`
- `clip_pct`
- optional `spike_skip` count or per-step flag

Without this, historical search/analysis is blind to clipping dynamics.

## 2. Enforce evaluation availability for search runs

A large fraction of local runs have no `valLoss`. That makes them unusable for quality optimization.

Search harness should:

- require `valData` or auto-split
- enforce minimum `evalInterval`
- mark runs invalid for fitness if no validation points exist

## 3. Add lineage metadata for evolved runs

To make symbiogenesis useful operationally, store:

- parent run IDs
- mutation/fusion operator
- warm-start mode (`fresh`/`exact`/`project`)
- search generation index

This can be JSON in `config.json` first, DB schema later.

## Alpha-Specific “Fusion” Operators (recommended)

These are practical and aligned with the codebase.

### Config fusion (cheap, robust)

- Numeric params: weighted average in log-space for LR-like params
- Categorical params: parent vote or random inheritance
- Architecture:
  - `nLayer`: choose one parent or bounded interpolation
  - `nEmbd`: choose from discrete supported set
  - `nHead`: choose divisors of `nEmbd`
  - `blockSize`: choose from allowed set

### Runtime fusion (performance policy)

- `batchSize`, `gradAccumSteps`, `syncEvery`, `gcEvery`, `packed`

These have immediate performance impact and are easy to evaluate.

### Checkpoint projection fusion (later)

Use parent checkpoint + child config to create projected initialization.

This is the direct analog of symbiogenesis projection weight transfer.

## Example Fitness Functions (use both)

## Quality-first (fixed token budget)

```text
fitness_q = -best_val_loss
            - 0.05 * log1p(max_grad_norm)
            - 0.2  * clip_rate
            + 0.02 * log(tokens_per_sec_mean)
```

## Efficiency-first (fixed walltime budget)

```text
fitness_p = 0.5 * log(tokens_per_sec_mean)
            - 0.5 * best_val_loss
            - 0.25 * instability_score
            - 0.05 * log(param_count)
```

This mirrors symbiogenesis's multi-objective / free-energy framing while staying grounded in `alpha` telemetry.

## Expected Wins (Realistic)

## Near-term (1-2 weeks, no kernel work)

- Better chat validation loss under same GPU-hours via evolved config search
- Faster convergence to stable settings (fewer dead-end runs)
- Stronger reproducibility and better benchmark hygiene
- Runtime policy gains from evolving `syncEvery/gcEvery/packed/accum` on target hardware

## Mid-term (2-6 weeks)

- Cheaper architecture sweeps via projected checkpoint warm starts
- Better quality/perf Pareto frontier for each domain
- Domain-specific evolved defaults that can replace hand-tuned presets in `domains.ts`

## Recommended First Experiment (high signal)

Target: `chat`, `helios`, `L6 D256 H8 B256`, fixed architecture.

Evolve only:

- `lr`, `lrMin`, `warmupIters`
- `beta2`, `weightDecay`, `gradClip`
- `spikeThreshold`
- `syncEvery`, `gcEvery`

Budget:

- fixed walltime (e.g. 2-4 minutes per candidate)
- mandatory validation every N steps

Why first:

- local data already shows throughput is nearly constant across this family
- quality/stability differences are large
- no checkpoint morphing required

## Bottom Line

If the goal is improving `alpha` performance and results, the best use of symbiogenesis is:

- not porting the Python research stack wholesale,
- but porting its search logic, monitoring discipline, and transfer strategy into `alpha`'s existing TypeScript trainer ecosystem.

The winning sequence is:

1. config evolution + budget fairness
2. stability-aware fitness + change-point monitors
3. checkpoint projection warm-starts
4. constrained architecture/operator evolution

That path is directly compatible with the code you already have and addresses real issues visible in your current run history.
