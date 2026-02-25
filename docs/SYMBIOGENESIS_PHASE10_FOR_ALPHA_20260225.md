# Symbiogenesis Phase 10 for Alpha (Activation Evolution Only)

Date: 2026-02-25

## Scope

This document is only about **Symbiogenesis Phase 10** (per-layer activation function evolution) and how it could be used to improve `alpha`'s:

- model quality (validation loss / perplexity)
- training stability
- throughput-adjusted performance (quality per GPU-hour)

It is not a general symbiogenesis report.

## What Phase 10 Actually Is in Symbiogenesis

Phase 10 in `models/alpha/symbiogenesis` adds **activation function evolution** as part of the architecture genome.

Implemented mechanics:

- `activation_pool` and `activation_mutation_rate` in config (`models/alpha/symbiogenesis/symbiogenesis/config.py:68`)
- per-layer activation lists on each `Unit` (`models/alpha/symbiogenesis/symbiogenesis/model.py:27`)
- activation registry with 7 activation types (`relu`, `gelu`, `silu`, `tanh`, `leaky_relu`, `elu`, `sigmoid`) (`models/alpha/symbiogenesis/symbiogenesis/model.py:11`)
- fusion propagates activation lists and mutates them during child creation (`models/alpha/symbiogenesis/symbiogenesis/fusion.py:36`, `models/alpha/symbiogenesis/symbiogenesis/fusion.py:41`, `models/alpha/symbiogenesis/symbiogenesis/fusion.py:62`)
- initial population seeds activation choice from the pool (`models/alpha/symbiogenesis/symbiogenesis/population.py:56`)
- activation distributions are tracked and plotted (`models/alpha/symbiogenesis/symbiogenesis/population.py:209`, `models/alpha/symbiogenesis/symbiogenesis/main.py:116`, `models/alpha/symbiogenesis/symbiogenesis/visualize.py:484`)

Evidence surface (repo-local):

- model tests verify default ReLU behavior and that all seven activations produce outputs (`models/alpha/symbiogenesis/tests/test_model.py:210`, `models/alpha/symbiogenesis/tests/test_model.py:260`)
- integration tests verify full activation-evolution runs, restricted pools, backward compatibility, and mutation-driven diversity (`models/alpha/symbiogenesis/tests/test_integration.py:505`, `models/alpha/symbiogenesis/tests/test_integration.py:519`, `models/alpha/symbiogenesis/tests/test_integration.py:534`, `models/alpha/symbiogenesis/tests/test_integration.py:546`)

Doc-reported Phase 10 benchmark claim (MNIST MLP population):

- activation evolution outperformed ReLU-only baseline on the reported setup
- ELU dominated the final population
- doc also claims backward compatibility with prior test suite (`models/alpha/symbiogenesis/docs/Phase10_Results.md:45`, `models/alpha/symbiogenesis/docs/Phase10_Results.md:64`, `models/alpha/symbiogenesis/docs/Phase10_Results.md:66`, `models/alpha/symbiogenesis/docs/Phase10_Results.md:67`, `models/alpha/symbiogenesis/docs/Phase10_Results.md:77`, `models/alpha/symbiogenesis/docs/Phase10_Results.md:108`)

Important context:

- those results are on a **PyTorch MLP search framework**, not a transformer with fused GPU kernels
- the transferable idea is the **search dimension (activation choice)**, not the exact winners (for example ELU)

## Why Phase 10 Could Matter for Alpha

`alpha` currently fixes the transformer FFN activation to GELU in the hot path:

- GPT MLP uses `matmulTransposedGelu(...)` directly in the forward pass (`models/alpha/packages/model/src/gpt.ts:13`, `models/alpha/packages/model/src/gpt.ts:182`)

That means:

- `alpha` already made one activation choice globally (GELU)
- there is no built-in way to test whether that choice is best for each domain / model scale / runtime regime

Phase 10's core contribution to `alpha` is:

- turn activation choice into a search dimension
- rank choices by **quality + stability + throughput**

This is more relevant than it sounds because your existing `alpha` runs already show:

- similar throughput in some chat-run families
- large validation-loss differences and instability differences

Activation choice is one more knob that can move that frontier.

## Direct Port vs Alpha Reality (Critical)

## What does not port directly

Symbiogenesis Phase 10 uses per-hidden-layer activations in an MLP where activations are cheap to swap in PyTorch.

`alpha` is different:

- transformer FFN path is fused around GELU (`matmulTransposedGelu`)
- Helios performance depends on kernel fusion and dispatch count
- backend activation support is not symmetric with autograd support

## What Alpha supports today (activation-related)

Backend interface has forward ops for:

- `gelu`, `relu`, `silu` (`models/alpha/packages/core/src/interfaces.ts:68`, `models/alpha/packages/core/src/interfaces.ts:69`, `models/alpha/packages/core/src/interfaces.ts:70`)

CPU and Helios backends implement forward ops for:

- `gelu`, `relu`, `silu` (`models/alpha/packages/tensor/src/cpu_ref.ts:422`, `models/alpha/packages/tensor/src/cpu_ref.ts:429`, `models/alpha/packages/tensor/src/cpu_ref.ts:433`, `models/alpha/packages/helios/src/backend.ts:985`, `models/alpha/packages/helios/src/backend.ts:993`, `models/alpha/packages/helios/src/backend.ts:998`)

Autograd wrappers currently expose:

- `relu`
- `gelu`
- fused `matmulTransposedGelu`

and do not expose a general `silu` autograd op in `ops.ts` (`models/alpha/packages/autograd/src/ops.ts:250`, `models/alpha/packages/autograd/src/ops.ts:312`, `models/alpha/packages/autograd/src/ops.ts:337`).

Backward kernels available in the backend interface:

- `geluBackward`
- `reluBackward`

No `siluBackward` exists in the current interface (`models/alpha/packages/core/src/interfaces.ts:96`, `models/alpha/packages/core/src/interfaces.ts:97`).

Implication:

- a Phase 10-style activation search in `alpha` is feasible
- but only if it is implemented as an **alpha-native Phase 10**, not a literal copy of symbiogenesis Phase 10

## The Right Phase 10 for Alpha

Treat Phase 10 in `alpha` as **FFN activation evolution**, not “arbitrary per-neuron MLP activation mutation”.

## Phase 10A (recommended first target)

Global FFN activation choice for all transformer blocks:

- `gelu` (baseline)
- `silu` (likely strongest next candidate)
- `relu` (ablation / speed-oriented baseline)

Why this is the right first move:

- minimal checkpoint/config surface change
- small search space
- maps to existing backend support
- easy to benchmark fairly
- keeps the research question clean

## Phase 10B (after A works)

Per-block FFN activation vector:

- one activation per transformer block
- same shortlist (`gelu`, `silu`, optionally `relu`)

This is the closest transformer analog to symbiogenesis's per-layer activation genome.

## Phase 10C (more valuable than adding 7 scalar activations)

Operator-family evolution:

- `gelu_mlp`
- `silu_mlp`
- `swiglu` (later)
- optional stabilization toggles (QK-LN, softcap variants)

This is likely to produce larger gains than porting `elu/tanh/sigmoid` into the transformer FFN.

## Performance Consequences (and Why They Matter)

Phase 10 in `alpha` can improve results but also hurt throughput if implemented naively.

## Current fast path

`alpha` uses fused `matmulTransposedGelu` in the FFN expansion step (`models/alpha/packages/autograd/src/ops.ts:337`, `models/alpha/packages/model/src/gpt.ts:182`).

Benefits of this current path:

- fewer tape entries
- fewer intermediate tensors
- fewer GPU dispatches
- a dedicated GELU backward path can be used

## Naive Phase 10 port would regress speed

If you replace the fused path with generic:

- `matmulTransposed(...)`
- then `activation(...)`

you will likely increase:

- forward dispatch count
- backward work / intermediate lifetime
- memory traffic
- step time variance

This can erase quality wins if measured per walltime.

## Phase 10 for Alpha must be performance-aware

Any activation search in `alpha` should optimize:

- quality
- stability
- throughput

not just quality.

## Phase 10 Implementation Strategy for Alpha (Practical)

## Stage 1: Quality-first, low engineering risk

Goal:

- test whether activation choice is worth pursuing at all

Approach:

- add `ffnActivation` to `ModelConfig`
- implement generic (non-fused) FFN activation path for `relu` and `silu`
- keep GELU on fused path
- benchmark on short runs with fixed token budget and fixed architecture

This gives a quick signal on:

- validation-loss movement
- stability movement
- throughput penalty

## Stage 2: Recover performance for winners

If `silu` or another variant shows promise:

- add autograd `silu(...)`
- add `siluBackward` to backend interface + CPU/Helios implementations
- add `matmulTransposedSilu` fused op (analogous to GELU path)

At that point, rerun the same benchmark matrix and compare:

- `gelu_fused` vs `silu_generic` vs `silu_fused`

## Stage 3: Evolve, don't hand-pick

Once the activation choices are implemented:

- run a symbiogenesis-style outer loop over activation configs
- use multi-objective fitness (quality + throughput + stability)
- optionally add per-block activation vectors

## What to Change in Alpha (Phase 10-only roadmap)

## 1. Add activation config to core model types

Files:

- `models/alpha/packages/core/src/types.ts`
- `models/alpha/packages/core/src/domains.ts`
- `models/alpha/apps/cli/src/commands/train.ts`
- `models/alpha/packages/train/src/checkpoint.ts` (checkpoint already stores `modelConfig`, so compatibility impact is manageable)

Recommended config shape (start simple):

- `ffnActivation: "gelu" | "silu" | "relu"`

Later:

- `ffnActivations?: readonly ActivationName[]` (per-block vector)

## 2. Refactor GPT MLP forward path to dispatch by activation

File:

- `models/alpha/packages/model/src/gpt.ts`

Current:

- hard-coded `matmulTransposedGelu(...)`

Target:

- `if (gelu && fused available) use fused`
- else `matmulTransposed(...)` + chosen activation op

This preserves GELU fast path while enabling Phase 10 experiments.

## 3. Fill activation support gaps in autograd/backends

Files:

- `models/alpha/packages/autograd/src/ops.ts`
- `models/alpha/packages/core/src/interfaces.ts`
- `models/alpha/packages/tensor/src/cpu_ref.ts`
- `models/alpha/packages/helios/src/backend.ts`
- `models/alpha/packages/helios/src/kernels/*`

Minimum viable:

- autograd `silu(...)`
- CPU fallback backward for SiLU

Performance path:

- `siluBackward` in backends
- fused `matmulTransposedSilu`

## 4. Add Phase 10 benchmarking support

Files:

- `models/alpha/apps/cli/src/commands/bench.ts`
- new experiment script/package (recommended)

Run matrix (fixed architecture first):

- `chat` domain, `L6 D256 H8 B256`
- `ffnActivation in {gelu, silu, relu}`
- same token budget
- same eval cadence

Record:

- best / last `valLoss`
- `tokens_per_sec`
- `ms_per_iter`
- `timing_fwd_ms`, `timing_bwd_ms`
- `gradNorm`
- `clip_coef`, `clip_pct` (see telemetry gap below)

## 5. Fix telemetry persistence before serious Phase 10 search

`alpha` trainer emits clip telemetry:

- `clip_coef`
- `clip_pct`

(`models/alpha/packages/train/src/trainer.ts:76`, `models/alpha/packages/train/src/trainer.ts:77`, `models/alpha/packages/train/src/trainer.ts:604`, `models/alpha/packages/train/src/trainer.ts:605`)

But local DB metric row types/inserts do not currently persist those fields:

- `models/alpha/packages/db/src/types.ts`
- `models/alpha/packages/db/src/metrics.ts`

This matters because Phase 10 search should penalize instability and clipping saturation.

## Phase 10 Fitness for Alpha (Recommended)

Use a multi-objective Phase 10 score, not val loss alone.

Example (fixed token budget):

```text
fitness_phase10 =
  - best_val_loss
  + 0.03 * log(tokens_per_sec_mean)
  - 0.15 * clip_rate
  - 0.05 * log1p(max_grad_norm)
```

Example (fixed walltime budget):

```text
fitness_phase10_wall =
  - best_val_loss_at_budget
  - 0.10 * instability_score
  + 0.08 * log(tokens_per_sec_mean)
```

This is the `alpha` analog of symbiogenesis Phase 10 selection pressure:

- let activation choice compete
- but only keep choices that improve the real Pareto frontier

## How to Use Symbiogenesis Phase 10 Concepts Without Overfitting to MLP Results

## Transferable concepts

- activation choice as an evolvable gene
- mutation + inheritance of activation assignments
- activation distribution tracking over population/history
- backward-compatibility tests (default behavior preserved)

## Non-transferable assumptions

- ELU dominance on MNIST MLPs does not imply ELU is good for `alpha` transformers
- 7-activation pool is not automatically sensible for Helios performance
- MLP activation search cost is much lower than transformer activation search cost

## Local Alpha Run Reality (why this needs discipline)

From a local scan of existing `runs/*` (2026-02-25):

- many runs lack `valLoss`
- only a small minority include clipping telemetry in `metrics.jsonl`

For Phase 10 work, enforce:

- validation logging on every run
- stable metric schema
- fixed budgets

Otherwise the search will optimize noise.

## Decision Gates (use these to avoid wasted work)

## Gate 1: Is activation choice worth it at all?

Run:

- `gelu` vs `silu` vs `relu` on fixed architecture and budget

Pass if:

- at least one non-GELU option improves `valLoss` or stability materially
- throughput penalty is acceptable (or recoverable with fusion)

## Gate 2: Is performance regression recoverable?

If a generic activation wins quality but loses throughput:

- implement fused path for the winner (`matmulTransposedSilu`)
- rerun benchmark

Pass if:

- winner remains attractive on quality-per-walltime

## Gate 3: Is per-block evolution justified?

Only pursue per-block activation vectors if:

- global activation choice shows clear signal
- per-block variants outperform global variants under fixed compute

## Recommended Phase 10 Plan (Alpha-specific)

1. Add `ffnActivation` config (`gelu|silu|relu`) and CLI flag.
2. Refactor GPT FFN path to support generic activations while preserving GELU fused fast path.
3. Add autograd SiLU support.
4. Run fixed-budget Phase 10 benchmark matrix on `chat` and one smaller architecture.
5. If SiLU wins, add fused `matmulTransposedSilu`.
6. Only then build symbiogenesis-style activation evolution (global, then per-block).

## Bottom Line

Symbiogenesis Phase 10 is applicable to `alpha`, but the correct translation is:

- **activation evolution in transformer FFN/operator choices**, measured with a performance-aware fitness,

not:

- blindly porting a 7-activation MLP mutation system into the Helios training path.

The highest-probability win is:

- `gelu` vs `silu` (and possibly `relu` as a speed baseline),
- with quality + stability + throughput measured together,
- and a fused kernel path added only for winners.
