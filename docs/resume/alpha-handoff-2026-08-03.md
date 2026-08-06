# Alpha performance research — handoff index

**Frozen:** 2026-08-03 · **For:** the next model or researcher taking this over
**Purpose:** a complete pointer to every artifact produced, with status, so nothing has to be rediscovered.

This is an index, not an argument. If you want the reasoning, read the briefing (§2). If you want to reproduce or extend, everything you need is in §4 and §5.

---

## 1. What this program is

Alpha is a ~97M-parameter language model trained from scratch on one rented RTX 4090, using a from-scratch Vulkan/SPIR-V training engine (Helios) driven from TypeScript. The product goal is a genuinely good small conversational model. **This program is about the cost of getting there**, not the model's quality.

Two standing goals, both committed to the repo:

| Goal | File | Target |
|---|---|---|
| Cost | `alpha2/docs/resume/GOAL-EXTREME-PERFORMANCE-2026-08-03.md` | ≥10× reduction in GPU-dollars to a fixed held-out loss |
| Throughput (sub-goal) | `alpha2/docs/resume/GOAL-THROUGHPUT-2026-08-03.md` | 7,254 → 30,000 tokens/s committed, 45,000 stretch |

---

## 2. Documents, in reading order

All are in `/mnt/donto-data/donto-resources/research/alpha-helios-reimagined/` and the first four are published at `https://alpha.donto.org/research/`.

| # | Document | What it is |
|---|---|---|
| 1 | **[alpha-program-briefing-2026-08-03.md](https://alpha.donto.org/research/alpha-program-briefing-2026-08-03.html)** | **Start here.** Complete self-contained explanation of the whole program for someone who has never seen it. No questions in it. |
| 2 | [alpha-open-problems-2026-08-03.md](https://alpha.donto.org/research/alpha-open-problems-2026-08-03.html) | 54 posed open mathematical questions in ten families, each anchored to a measured constant |
| 3 | [alpha-response-to-harmonic-2026-08-03.md](https://alpha.donto.org/research/alpha-response-to-harmonic-2026-08-03.html) | Reply to an external critique; contains the retraction |
| 4 | `REIMAGINING-ALPHA-PERFORMANCE-2026-08-03.md` | Main internal record: roofline, four-factor decomposition, falsifications, ranked portfolio, and a table of ridiculous ideas with verdicts |
| 5 | `X8-THE-MISSING-HALF-OF-THE-STEP.md` | Diagnosis of the 51% of step time no kernel accounts for, entirely from preserved logs |
| 6 | `TWO-PRINCIPLES-2026-08-03.md` | Two generalisable methodological principles that came out of the retraction |
| 7 | `GOAL-THROUGHPUT-2026-08-03.md` | The throughput ladder and its gates |
| 8 | `README.md` | Directory index |

---

## 3. Every result, with status

### Established

| Result | Value | Where |
|---|---|---|
| Helios runs at **5.74% of FP32 peak**, 2.87% of BF16 tensor rating; gap is 12–22× | 4.74 TFLOP/s achieved | x1 |
| Workload is **compute-bound by 3.6×** (291 vs 82 FLOP/byte ridge) — bandwidth is not the excuse | — | x1 |
| **Attention is only 10.8% of arithmetic** at S=1024; any replacement caps at 1.12× | — | x1 |
| **51% of the step is unattributed to any kernel**; only 5–7 submissions and 1–3 waits per step, and the op graph is *static* (1,444 identical ops on 17 of 20 steps) rebuilt in TypeScript every iteration | host-bound, unoverlapped | X8 |
| **Batch 24 is 2.30× above the gradient noise scale** (B_simple = 10,674 tokens); batch 10 reaches the same loss with **1.65× less arithmetic** | ~29 GPU-h, ~$20 | x6, x6b |
| **Muon reaches AdamW's final held-out loss in 2.24× fewer tokens**, LR swept on both arms until each optimum was interior | +0.4699 nats | x7, x7b |
| Muon is far less LR-sensitive: 0.0135 nats across a 10× range vs AdamW's 0.408 | — | x7b |
| **int4 weight gradients train identically to FP32 closed-loop**; all four reduced-precision arms within 0.007 nats | int4 at −0.0005 | x13 |
| Stochastic rounding is **empirically unbiased**: bias/single-draw = 0.2479–0.2502 vs theoretical 1/√16 = 0.2500, across nine formats | — | x9 |
| **Curvature-aware bit allocation beats uniform** at fewer average bits (K-FAC metric) | 0.0092 vs 0.0254 | x14 |
| Norm-importance sampling gives a **1.57× variance reduction**, rate-independent | — | x11 |

### Closed — do not re-propose without overturning a measurement

| Direction | Why closed | Re-opens |
|---|---|---|
| Linear/oscillator/SSM attention as a **retrofit** | Attention is rank 684/1024 at 1% error; within-row logit spread 10.02 vs Alman–Song threshold 2.63 = **3.8× into the SETH-hard regime** | Trained from scratch under the constraint |
| FMM / sparse attention at S=1024 | Far-field blocks need rank 40/128; 37% of mass is out there; 1.53× on attention = **1.04× end to end** | S ≳ 16,384 (~2.3× there) |
| Any attention replacement at S=1024 | 10.8% of arithmetic; ceiling 1.12× | Long context |
| Low-rank / structured weights | 92.8% of full rank; factorisation costs **1.34× more** than dense | As its converse — argues *for* MoE |
| **Gauge-Quotient Muon** | Gradients are **exactly orthogonal to gauge orbits by invariance** — a theorem. Quantisation error in the gauge is perfectly isotropic (1.00× fair share) | Never; this closure is structural |

### Retracted

**Weight-gradient token subsampling (claimed 1.35×).** x5 measured the noise floor symmetrically while x9/x11 measured errors asymmetrically against the exact gradient, and the two were compared. On one consistent metric the floor is 0.844 — the error from simply halving the batch — and **no sampling rate stays under it**. Full account in x12 and document 3.

Generalised lesson, worth more than the result: *a method saving fraction f of arithmetic must be compared against the trivial method that simply uses f less data, batch, or steps.*

### Unfinished — the most valuable open item

**The consumed-metric attention audit** (`x10b_consumed_metric.py`, ready to run). Every attention closure above is stated in Frobenius norm on a matrix the model never consumes — it consumes `A·V` and then fifteen more layers. The first attempt (`x10`) was stopped as underpowered; see `results/x10_metric_mismatch.NOTE.md` for exactly what was wrong (needs paired per-batch differences and ≥64 sequences). **Until this runs, treat the attention closures as provisional.**

---

## 4. Every experiment script

All are self-contained, take no arguments, write Markdown to stdout and JSON to `../results/`. Interpreter: `/mnt/donto-data/alpha-corpora/.venv/bin/python` (torch 2.13 CPU, numpy 2.4.6, transformers 5.14.1). **None requires a GPU.**

| Script | Question |
|---|---|
| `x1_energy_budget.py` | Exact FLOP accounting, roofline position, four-factor cost decomposition |
| `x2_attention_structure.py` | Is attention low-rank or sparse? *(§1 superseded by x3)* |
| `x3_far_field_rank.py` | The corrected FMM criterion — off-diagonal block rank; exact logit spread |
| `x4_weight_spectra.py` | Are trained weight matrices using their full rank? |
| `x5_gradient_noise_floor.py` | How precisely does SGD need dW? *(metric superseded by x12)* |
| `x6_noise_scale.py` | Gradient noise scale / critical batch size |
| `x6b_batch_tradeoff.py` | What the contracted batch size costs |
| `x7_optimizer_geometry.py` | Muon vs AdamW at matched tokens |
| `x7b_adamw_bracket.py` | Bracketing the AdamW optimum so the comparison is fair |
| `x9_gradient_precision.py` | How many mantissa bits does a gradient need? Bias vs variance |
| `x10_metric_mismatch.py` | First consumed-metric attempt — **stopped, superseded** |
| `x10b_consumed_metric.py` | Redesigned consumed-metric audit — **ready, not yet run** |
| `x11_importance_sampling.py` | The norm-importance sampler |
| `x12_unified_metric.py` | **Reconciles x5/x9/x11 onto one metric; contains the retraction** |
| `x13_closed_loop_precision.py` | Closed-loop training with quantised weight gradients |
| `x14_curvature_budget.py` | Curvature-aware vs uniform bit allocation (K-FAC) |
| `x15_gauge_energy.py` | Gauge energy in gradients and quantisation error |
| `x16_throughput_ladder.py` | Derives the throughput target ladder from the measured profile |

---

## 5. Reproduction

```bash
cd /mnt/donto-data/donto-resources/research/alpha-helios-reimagined
sha256sum -c ARTIFACTS.sha256          # 63 artifacts
cd experiments
PY=/mnt/donto-data/alpha-corpora/.venv/bin/python
$PY x1_energy_budget.py                # closed-form, seconds
$PY x16_throughput_ladder.py           # closed-form, seconds
# the rest load the real checkpoint; minutes to ~1 hour each, nice them
```

**Identities used throughout:**

```
checkpoint  /mnt/donto-data/alpha-runs/hf-alpha-60m-base-c333bf2-20260728   57,688,576 params
held-out    /mnt/donto-data/alpha-corpora/pretrain-text/foundation-val-005-64m.txt
            17e30fa2e50e1a1f116cceed95381b76edd1be595d402c4dd053bd55a7eafd60
train shard /mnt/donto-data/alpha-corpora/pretrain-text/pretrain-000.txt
            d993342b0bb55198c520f1f761bb0aad2812b2d8fb9c6347b4e6f9d622794d9c
```

The held-out shard is the contracted validation slice, read for measurement only. No bytes were used to train, generate, or select anything.

**Foundation shape** (from `alpha2/scripts/run_foundation_candidate_full.sh`): 18 layers, d=640, 10 heads × 64, FFN 1,728, vocab 12,288, S=1,024, batch 24 → 24,576 tokens/step, 79,020 steps, 1,941,995,520 tokens, 97,098,880 parameters.

---

## 6. What to do next, in order

| # | Action | Cost | Why |
|---:|---|---|---|
| 1 | Print `host_build_ms` beside `dispatch_gpu_us` in the trainer | ~1 h, local, **free** | Confirms or refutes the host-bound model that the entire throughput ladder rests on. Nothing else should start first. |
| 2 | Microbenchmark the FP32-accumulate cooperative-matrix path on this die | **~$0.70**, 1 h | Decides whether BF16 tensor cores are worth 2× or **worth nothing** — if FP32-accumulate is half-rate, well-tiled FP32 beats them and the whole mixed-precision effort should be cancelled |
| 3 | Run `x10b_consumed_metric.py` | free, ~30 min | The only thing that can re-open the attention closures |
| 4 | R1: record the static graph once and replay | days | Largest single rung (1.84×), and it makes every later kernel gain visible |
| 5 | R2 → R3 → R4, reprofiling after each | — | Reaches the committed 30,000 tok/s in pure FP32 |

Two known unknowns worth stating plainly: the closed-loop precision result is at **250 steps against a 79,020-step contract** (316× shorter horizon, and error accumulation scales with step count), and the Muon result is at **8M-parameter proxy scale** — a direction and a sign, not a transferable multiplier.

---

## 7. Method, so it can be copied or criticised

- Device-independent claims get measured on CPU, against the real checkpoint and real held-out text, **before anything is rented**. Every result here cost zero GPU dollars.
- Matched comparisons only: same tokens, same data order, same shape, with the interacting hyperparameter swept on *both* arms.
- Predictions are labelled as predictions and carry falsification conditions.
- Negative results are archived permanently. Five closures and one retraction are the most useful output of this program.
- A result is a measurement, not an argument.

---

## 8. Honest scoreboard

Eleven results established, five directions closed (one by proof), one result retracted by our own follow-up, one experiment stopped as underpowered and rewritten but not yet re-run. Two bugs were caught by unit tests before they contaminated results; one bug killed a run and was fixed and re-run. Nothing in this program has been validated on the actual GPU, because no GPU time has been spent.
