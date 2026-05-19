# GPT-3 in 72 Hours: Full Research Notes (Alpha Repo)

Date: 2026-03-07
Repo: `/Users/ajaxdavis/repos/alpha2`

## TL;DR

- GPT-3 class in 72 hours is not feasible in this codebase in its current state.
- GPT-2 class is potentially feasible only with A100/H100 and tight execution discipline.
- On current L4-only fleet, highest-probability outcome in 72 hours is a strong small/medium chat model, not GPT-2 class parity.

## What I reviewed

- Core training/config scripts and domains.
- Fleet deployment/runtime behavior.
- Recent internal stability/perf reports.
- Existing GPT-2 planning doc.
- H100/A100 orchestration paths (`modal` and `gcp`).

## Key constraints (hard reality)

### 1. Current active fleet is L4 only

From `fleet.json`, both configured instances are `g2-standard-4` with L4 GPU.

Implication:
- GPT-3 class is out of scope.
- GPT-2 class is likely too aggressive for 72h on these nodes given current stack efficiency.

### 2. Repo itself describes multi-GPU/distributed as not production-ready yet

The strategy/scaling docs explicitly position multi-GPU and distributed functionality as a roadmap effort.

Implication:
- You cannot rely on mature data-parallel scaling today to brute-force a GPT-3 timeline.

### 3. Recent L4 stability notes still show scaling pain at larger configs

Recent working notes indicate:
- Stable config is small.
- Larger configs trigger allocation pressure/perf collapse.
- fp16 stability issues still exist in some paths.

Implication:
- Even if architecture knobs allow bigger models, practical sustained training throughput is still the gating factor.

## Feasibility matrix (72h)

### GPT-3 class

Status: Not feasible.

Reasons:
- No mature distributed training pipeline in practice.
- L4-only active fleet.
- Current perf/stability baseline is not near GPT-3 training economics.

### GPT-2 class

Status: Maybe, only with A100/H100 and disciplined run plan.

Needed:
- Use H100/A100 path immediately.
- Keep long uninterrupted loops with strict monitoring.
- Avoid known instability configurations.
- Use strong dataset/tokenizer setup and enough token budget.

### Best realistic result in 72h on current L4 fleet

Status: Feasible.

Outcome:
- Better chat coherence and lower loss than current super_chat/nanochat baselines.
- Not GPT-2 benchmark parity.

## Critical code-level findings

### 1. Tied embedding/head appears to be double-traversed in param list

Observed:
- `lmHead` is tied to `wte` in model init.
- Parameter collection still includes both `wte` and `lmHead` entries.
- Trainer fast optimizer path iterates collected param entries directly.

Likely impact:
- Potential duplicate update and/or duplicate optimizer-state handling on tied weights.
- Could distort optimization and convergence behavior.

### 2. Weight decay exclusion naming mismatch likely exists

Observed:
- No-decay set includes `lmHead.weight`.
- Collected param key appears to be `lmHead`.

Likely impact:
- Intended no-decay policy may not apply to tied head as expected.

### 3. GPT-2-ish domain/script exists, but execution shape differs from canonical GPT-2 recipes

Observed:
- `concordance` domain has 12x768 defaults.
- `train-concordance.sh` runs `block=256` and 2000 steps default.

Implication:
- This is useful for stress/scale progression, but not equivalent to a canonical GPT-2 training budget.

## Infrastructure paths that matter now

### A) Fleet path (repo canonical)

- Build and deploy with compiled workflow (`npm run bun:compile`, then `npm run fleet:deploy -- <instance>`).
- Train with required mandates: DGC enabled, no fallback, sample interval 200, Discord webhook configured.

Caveat:
- Recent note says compiled binary runtime had Vulkan init issues on a current L4 host; node runtime was more reliable there.

### B) GCP A100 path

- `scripts/gcp_train.py` supports `a2-ultragpu-1g` (A100 80GB).
- Full lifecycle automation exists (provision, sync, build, train, download).

### C) Modal H100 path

- `scripts/modal_train.py` runs on H100.
- Important limitation: function timeout is 6 hours, so 72h objective requires chained resumptions.

## 72-hour execution strategy (recommended)

## Phase 0 (first 2-4 hours)

1. Fix tied-weight traversal/decay mismatch before expensive runs.
2. Do a short validation run to confirm:
   - stable loss curve,
   - no NaN/divergence,
   - expected throughput,
   - checkpoint resume integrity.
3. Verify reporting pipeline (remote + Discord samples every 200 steps).

## Phase 1 (hours 4-18)

1. Launch a throughput soak on target hardware (prefer H100, second A100).
2. Measure effective tokens/day for your intended config.
3. Project 72h total token budget and compare against quality target.

Decision gate:
- If projected budget is insufficient for GPT-2 class trajectory, pivot quickly to a smaller model/data objective that can deliver meaningful quality.

## Phase 2 (hours 18-72)

1. Run long loops with resume hygiene and strict config consistency.
2. Keep sample/eval cadence fixed for reliable trend analysis.
3. Avoid speculative architecture changes mid-run unless clear regression appears.
4. Use checkpoint branches for A/B tests; keep one conservative baseline branch alive.

## What "success" should mean in 72h

Pick one target explicitly:

### Target A: GPT-2-class push (H100/A100 only)

Success criteria:
- Stable long-run training with no catastrophic divergence.
- Validation/samples trend toward a pre-agreed quality bar.
- Final checkpoint demonstrably better than current repo baselines.

### Target B: High-quality small/medium chat model (L4 compatible)

Success criteria:
- Strongly improved coherence and helpfulness on fixed prompt set.
- Lower and stable validation loss vs prior runs.
- Reliable production-ish loop with reporting and resumability.

## Risk register

1. Runtime instability under larger configs.
   - Mitigation: early soak tests, conservative fallback branch.
2. Resume/tokenizer/config incompatibility.
   - Mitigation: strict compatibility checks and frozen artifacts per run family.
3. Overfitting to loss without sample quality gains.
   - Mitigation: fixed sample battery + periodic human eval gates.
4. 72h wasted on impossible objective.
   - Mitigation: hard go/no-go at hour 18 based on projected token budget.

## Suggested immediate next step

Implement and test the tied-weight optimizer/decay fix first, then run a 1-2 hour calibrated benchmark on H100/A100 path to determine whether the GPT-2 objective remains viable inside 72 hours.

## Notes on external reference context

If you want strict GPT-2-grade comparison against NanoChat claims, use upstream NanoChat reference docs/scripts as baseline context:

- https://github.com/karpathy/nanochat
- https://github.com/karpathy/nanochat/blob/master/runs/speedrun.sh

(Those references are for benchmark framing, not a statement that current Alpha setup is equivalent.)
