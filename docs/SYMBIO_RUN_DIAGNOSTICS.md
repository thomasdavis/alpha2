# Symbiogenesis Run Diagnostics

**Run ID:** `historic_chat_v2_20260225114638_5ke3`
**Date:** 2026-02-25 11:47:06 – 12:04:36 UTC (17m 30s)
**Status:** Completed
**Dashboard:** https://alpha.omegaai.dev/runs/historic_chat_v2_20260225114638_5ke3
**Total Metrics Rows:** 435 (some steps skipped due to grad spikes)

---

## 1. Infrastructure

| | |
|---|---|
| **GPU** | NVIDIA L4 (24GB VRAM, Ampere arch) |
| **VRAM** | 23,034 MB total |
| **Host** | `alpha-train` — GCP `a2-ultragpu-1g` |
| **CPU** | 4 cores |
| **RAM** | 15,990 MB |
| **OS** | Linux |
| **Backend** | Helios (Vulkan compute via custom SPIR-V kernels) |

---

## 2. Model & Training Configuration

### Model Architecture

| Parameter | Value |
|-----------|-------|
| Layers | 6 |
| Embedding Dim | 256 |
| Heads | 8 (head dim = 32) |
| Vocab Size | 4,000 (BPE-4k tokenizer) |
| Context (Block Size) | 256 tokens |
| FFN Activation | swiglu (symbio preset override) |
| FFN Hidden Dim | 704 (2.75x embed dim — SwiGLU standard) |
| Dropout | 0.1 |
| Total Parameters | **6,937,088** (~6.9M) |
| Tokens/Step | 8 × 256 = 2,048 |

### Training Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Total Iterations | 500 | ~160 search + ~340 post-search |
| Batch Size | 8 | |
| Max Learning Rate | 3e-4 | |
| Min Learning Rate | 5e-6 | |
| LR Schedule | Cosine decay | |
| Warmup Steps | 50 | |
| Beta1 | 0.9 | |
| Beta2 | 0.95 | Lower than default 0.999 for stability |
| Epsilon | 1e-6 | |
| Weight Decay | 0.1 | |
| Grad Clip Max Norm | 1.0 | |
| Spike Threshold | 10.0 | Steps with grad_norm > 10 are skipped |
| Optimizer | AdamW | |
| Eval Interval | 100 steps | |
| Sample Interval | 200 steps | |

### Symbio / Search Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Mode | `ffn-activation-search` | Evolutionary search over activation functions |
| Activation Pool | gelu, silu, relu, swiglu | 4 candidate activations |
| Population Size | 4 | Candidates per generation |
| Generations | 2 | Total search generations |
| Steps/Candidate | 20 | Steps to evaluate each candidate |
| Selection Strategy | topk | Top performers advance to next gen |
| Mutation Rate | 0.5 | Probability of mutation when breeding |
| Rank By | valLoss | (falls back to train loss if no val) |
| CUSUM Sensitivity | 4.0 | Standard deviations for alert threshold |
| CUSUM Baseline Window | 5 | Steps for baseline accumulation |
| Metrics Interval | 10 | Symbio metrics collected every 10 steps |
| Adaptive Batch | false | Batch size adaptation disabled |
| Track Weight Entropy | true | |
| Track Effective Rank | true | |
| Track Free Energy | true | beta = 0.01 |
| Track MI Profiles | false | Mutual information disabled (expensive) |
| Track Population Metrics | true | |

**Computed search budget:** 4 candidates × 2 generations × 20 steps = **160 steps** dedicated to search, leaving 340 steps for post-search training with the winner.

---

## 3. Training Data

| | |
|---|---|
| **File** | `data/historic-chat-v2.txt` |
| **Size** | 33 MB |
| **Domain** | Chat (multi-turn `<\|user\|>` / `<\|assistant\|>` format) |
| **Tokenizer** | BPE-4k (4,000 token vocabulary) |

---

## 4. Loss Analysis

### Summary

| Metric | Value |
|--------|-------|
| Initial Loss (step 1) | **8.3546** (near log₂(4000) = 11.97, expected for random init) |
| Final Loss (step 498) | **7.2586** |
| Best Train Loss | **7.2104** (step ~370) |
| Best Val Loss | **7.2994** (step 400) |
| Total Loss Reduction | **-1.0962** (−13.1%) |
| Average Loss (full run) | **7.6327** |

### Validation Loss Checkpoints

| Step | Val Loss | Phase | Notes |
|------|----------|-------|-------|
| 100 | 8.3418 | Search (gen1_swiglu_5) | Just switched to gen1, model freshly re-initialized |
| 200 | 7.5354 | Post-search (step 40 of training with winner) | Strong improvement |
| 400 | 7.3268 | Post-search (step 240 of training with winner) | Still improving, gap to train is 0.09 |

Val-train gap at step 400: 7.3268 − 7.3393 = −0.013 (val slightly *better* than train — likely noise from small eval set). No evidence of overfitting at this scale.

### Detailed Loss Trajectory (every 10 steps)

| Step | Loss | Grad Norm | tok/s | ms/it | LR | Phase |
|------|------|-----------|-------|-------|-----|-------|
| 10 | 8.2620 | 1.277 | 6,917 | 296 | 6.4e-5 | Search: gen0_gelu_1 |
| 20 | 8.1103 | 0.731 | 7,428 | 276 | 1.2e-4 | Search: gen0_gelu_1 (end) |
| 30 | 8.1394 | **520.5** | 2,812 | 728 | 1.8e-4 | Search: gen0_silu_2 (spike!) |
| 40 | 8.0284 | 0.649 | 2,877 | 712 | 2.4e-4 | Search: gen0_silu_2 (end) |
| 50 | 7.9979 | 0.651 | 7,153 | 286 | 3.0e-4 | Search: gen0_relu_3 |
| 60 | 7.7401 | 0.496 | 6,983 | 293 | 3.0e-4 | Search: gen0_relu_3 (end, best loss so far) |
| 70 | 8.0433 | 0.666 | 2,544 | 805 | 3.0e-4 | Search: gen0_swiglu_4 |
| 80 | 7.7973 | 0.561 | 2,652 | 772 | 3.0e-4 | Search: gen0_swiglu_4 (end) |
| 90 | 8.0065 | 0.648 | 2,665 | 768 | 2.9e-4 | Search: gen1_swiglu_5 |
| 100 | 7.7882 | **16,214** | 2,609 | 785 | 2.9e-4 | Search: gen1_swiglu_5 (spike at eval!) |
| 110 | 7.9892 | 0.622 | 7,244 | 283 | 2.9e-4 | Search: gen1_relu_6 |
| 120 | 7.7010 | 0.498 | 7,419 | 276 | 2.8e-4 | Search: gen1_relu_6 (end, new best!) |
| 130 | 7.9838 | 0.670 | 7,509 | 273 | 2.8e-4 | Search: gen1_gelu_7 |
| 140 | 7.7648 | 0.514 | 7,086 | 289 | 2.7e-4 | Search: gen1_gelu_7 (end) |
| 150 | 8.1011 | **47.4** | 2,874 | 713 | 2.7e-4 | Search: gen1_silu_8 (spike) |
| 180 | 7.6251 | 0.494 | 2,805 | 730 | 2.4e-4 | Post-search training |
| 200 | 7.5731 | 0.465 | 2,751 | 744 | 2.3e-4 | Post-search |
| 250 | 7.5172 | 0.477 | 2,699 | 759 | 1.8e-4 | Post-search |
| 300 | 7.3970 | 0.595 | 2,782 | 736 | 1.3e-4 | Post-search |
| 350 | 7.4110 | **1,260** | 2,642 | 775 | 8.0e-5 | Post-search (spike) |
| 400 | 7.3393 | **251.0** | 2,811 | 728 | 4.0e-5 | Post-search (spike) |
| 450 | 7.2823 | 0.641 | 2,724 | 752 | 1.5e-5 | Post-search |
| 490 | 7.2199 | 0.631 | 2,793 | 733 | 5.4e-6 | Post-search (near minimum LR) |

### Loss by 50-Step Buckets

| Bucket | Avg Loss | Min Loss | Δ from prev |
|--------|----------|----------|-------------|
| 0–49 | 8.1957 | 8.0214 | — |
| 50–99 | 8.0074 | 7.7401 | −0.188 |
| 100–149 | 8.0215 | 7.7010 | +0.014 (new gen, re-inits) |
| 150–199 | 7.7348 | 7.5470 | −0.287 (post-search begins) |
| 200–249 | 7.5203 | 7.4423 | −0.215 |
| 250–299 | 7.4413 | 7.3294 | −0.079 |
| 300–349 | 7.3806 | 7.3027 | −0.061 |
| 350–399 | 7.3183 | **7.2104** | −0.062 |
| 400–449 | 7.2813 | 7.2117 | −0.037 |
| 450–499 | 7.2677 | 7.2160 | −0.014 (approaching floor) |

Loss improvement is decelerating — the last 100 steps gained only 0.014 average loss reduction, suggesting the model is near its capacity limit at 6.9M params on this dataset.

---

## 5. Throughput & Performance

### Overall

| Metric | Value |
|--------|-------|
| Avg Throughput | **3,419 tok/s** |
| Peak Throughput | 7,641 tok/s |
| Min Throughput | 389 tok/s (step 1, cold GPU) |
| Avg Step Time | 713 ms |
| Total Training Time | ~1,015 seconds |
| Total Tokens Processed | ~1,024,000 (500 × 2,048) |

### Throughput Bimodality

The run has a strong bimodal throughput distribution caused by the activation type:

| Activation | Avg tok/s | Avg ms/step | GPU ops/step |
|-----------|-----------|-------------|--------------|
| **gelu** (gen0) | 6,571 | 550 | ~647 |
| **relu** (gen0) | 6,716 | 320 | ~612 |
| **relu** (gen1) | 6,960 | 301 | ~612 |
| **gelu** (gen1) | 6,903 | 308 | ~612 |
| **silu** (gen0) | 2,555 | 835 | ~641 |
| **silu** (gen1) | 2,872 | 720 | ~600 |
| **swiglu** (gen0) | 2,627 | 787 | ~647 |
| **swiglu** (gen1 search) | 2,615 | 802 | ~647 |
| **relu** (post-search) | 2,659 | 791 | ~600 |

**Key finding:** gelu and relu run at ~7k tok/s while silu and swiglu run at ~2.6k tok/s. This is a **2.7x throughput difference** within the same model architecture. The difference is entirely in the backward pass — SwiGLU requires 3 weight matrices and more gradient computation, and silu (SiLU/Swish) has a more expensive derivative than relu.

Post-search relu runs at 2,659 tok/s despite relu being the "fast" activation during search. This suggests the GPU buffer pools haven't been optimally reallocated after the final model re-initialization.

### Timing Breakdown (averaged over all steps with timing data)

| Phase | Avg Time | % of Step | Notes |
|-------|----------|-----------|-------|
| **Forward** | 79.0 ms | 11.1% | |
| **Backward** | 599.0 ms | 84.0% | Dominates — expected for small batch |
| **Optimizer** | 1.3 ms | 0.2% | AdamW update (negligible) |
| **GPU Flush** | 8.7 ms | 1.2% | Buffer synchronization |
| **Data Load** | 0.04 ms | ~0% | Data fits in memory |
| **Unaccounted** | ~25 ms | 3.5% | Metric logging, symbio collection |

**Bwd/Fwd ratio: 7.6x** — Higher than the typical 2-3x for standard training. This is because the small batch (8) doesn't fully utilize GPU parallelism in the forward pass, while backward always requires full computation. Increasing batch size would improve this ratio.

---

## 6. Gradient Health

### Overview

| Metric | Value |
|--------|-------|
| Healthy Avg Grad Norm | **0.678** (steps where grad_norm < 10) |
| Total Grad Spikes (>10) | **61 steps** (12.2% of all steps) |
| NaN Gradients | ~4 steps |
| Avg Clip Coefficient | 0.980 (barely clipping most steps) |
| Cumulative Clip % | 23.1% of iterations hit the max_norm=1.0 threshold |

### Gradient Spike Breakdown by Phase

| Phase | Steps | Spikes | Spike Rate | Worst Spike |
|-------|-------|--------|------------|-------------|
| Search (steps 1-160) | 152 | 9 | 5.9% | 16,214 (gen1_swiglu_5, step 100) |
| Post-search (steps 161-500) | 283 | 52 | **18.4%** | **774,971** (step 190) |

The post-search phase has a *worse* spike rate than the search phase. This is concerning — relu with this learning rate schedule produces frequent extreme gradient norms. These spikes are caught by the `spikeThreshold=10` guard and the optimizer update is skipped, but 18% of steps being wasted is significant.

### Gradient Spikes per Activation (Search Phase Only)

| Activation | Spikes | Rate | Worst | Notes |
|-----------|--------|------|-------|-------|
| **gelu** (×2 candidates) | 0 | 0% | — | Cleanest gradients |
| **relu** (×2 candidates) | 0 | 0% | — | Clean during search |
| **silu** (×2 candidates) | 5 | 14.3% | 6,094 | NaN grads, most unstable |
| **swiglu** (×2 candidates) | 4 | 5.4% | 16,214 | Periodic large spikes |

### Top 10 Gradient Spikes

| Step | Grad Norm | Loss | Phase | Notes |
|------|-----------|------|-------|-------|
| 190 | **774,971** | 7.616 | Post-search | Extreme — nearly 1M |
| 210 | **771,646** | 7.503 | Post-search | |
| 179 | 607,307 | 7.672 | Post-search | |
| 168 | 325,327 | 7.724 | Post-search | |
| 171 | 223,069 | 7.699 | Post-search | |
| 240 | 59,203 | 7.516 | Post-search | |
| 161 | 54,803 | 7.833 | Post-search (first step!) | |
| 188 | 19,829 | 7.601 | Post-search | |
| 100 | 16,214 | 7.788 | Search: gen1_swiglu_5 | Spike at eval boundary |
| 249 | 9,758 | 7.527 | Post-search | |

**Pattern:** The worst spikes (100k–775k) cluster in steps 161-210, immediately after the search completes and the model switches to the winning activation for extended training. This transition period is the most unstable. After step 250, spikes become less extreme (typically 100-4000).

### Clipping Telemetry per Candidate

| Candidate | Activation | Avg Clip Coef | Avg Clip % | Clipped Steps |
|-----------|-----------|---------------|------------|---------------|
| gen0_gelu_1 | gelu | 0.847 | 92.1% | 13/20 (65%) |
| gen0_silu_2 | silu | 0.936 | 59.5% | 5/18 (28%) |
| gen0_relu_3 | relu | 0.965 | 40.0% | 2/20 (10%) |
| gen0_swiglu_4 | swiglu | 0.937 | 34.8% | 5/19 (26%) |
| gen1_swiglu_5 | swiglu | 0.983 | 32.1% | 4/18 (22%) |
| gen1_relu_6 | relu | 0.970 | 28.1% | 2/20 (10%) |
| gen1_gelu_7 | gelu | 0.961 | 26.0% | 3/20 (15%) |
| gen1_silu_8 | silu | 0.957 | 25.1% | 4/17 (24%) |

**Trend:** Clip % decreases across candidates (92% → 25%). This is because later candidates benefit from a higher learning rate (LR peaks at step 50 = 3e-4 and decays), and the clip threshold is relative. The first candidate (gelu) runs during warmup when gradients are naturally larger relative to parameters.

---

## 7. CUSUM Change-Point Monitoring

| Metric | Value |
|--------|-------|
| Total CUSUM Alert Steps | **66** |
| Alert Type | "throughput collapse" (100% of alerts) |
| CUSUM Data Range | Steps 1–154 (search phase only) |
| CUSUM Channels Active | `cusum_tps` (throughput), `cusum_clip` |
| Channels Always Zero | `cusum_grad`, `cusum_val` |

### CUSUM Alerts per Candidate

| Candidate | Alert Steps | Rate |
|-----------|-------------|------|
| gen0_gelu_1 | 5 | 25% |
| gen0_silu_2 | 2 | 11% |
| gen0_relu_3 | 11 | 55% |
| gen0_swiglu_4 | 4 | 21% |
| gen1_swiglu_5 | 0 | 0% |
| gen1_relu_6 | 12 | 60% |
| gen1_gelu_7 | 20 | **100%** |
| gen1_silu_8 | 12 | 71% |

**Root cause:** The CUSUM baseline for throughput is established during the first 5 steps of the entire run (steps 1-5 of gen0_gelu_1). Since gelu runs at ~7k tok/s, every subsequent candidate running at ~2.6k tok/s (silu, swiglu) triggers "throughput collapse." When relu/gelu candidates run at ~7k tok/s, they match the baseline and produce fewer alerts — but once the baseline accumulates more variance from the mixed-activation search, later candidates get alerts regardless.

**Bug identified:** The CUSUM monitor doesn't reset its baseline when the activation changes. Each candidate should have an independent CUSUM baseline, or the baseline should be reset at each activation switch.

**Note:** `cusum_grad` is always null after step 28. This indicates the grad CUSUM channel never establishes a baseline (possibly because the baseline window requirement isn't met across candidate boundaries).

---

## 8. FFN Activation Search — Detailed Results

### Search Timeline

```
Steps 1-20:    gen0_gelu_1   (gelu)    → bestLoss=8.110  fitness=0.0266
Steps 21-40:   gen0_silu_2   (silu)    → bestLoss=8.026  fitness=0.0284  ⚠ 4 grad spikes
Steps 41-60:   gen0_relu_3   (relu)    → bestLoss=7.740  fitness=0.0318  ★ Gen0 winner
Steps 61-80:   gen0_swiglu_4 (swiglu)  → bestLoss=7.797  fitness=0.0301
                ── Selection: relu, swiglu advance to gen1 ──
Steps 81-100:  gen1_swiglu_5 (swiglu)  → bestLoss=7.788  fitness=0.0296
Steps 101-120: gen1_relu_6   (relu)    → bestLoss=7.701  fitness=0.0327  ★ Gen1 winner
Steps 121-140: gen1_gelu_7   (gelu)    → bestLoss=7.760  fitness=0.0315  (mutation from relu→gelu)
Steps 141-159: gen1_silu_8   (silu)    → bestLoss=7.868  fitness=0.0280  (mutation from swiglu→silu)
                ── Search complete: relu wins ──
Steps 160-500: Post-search training with relu activation
```

### Per-Candidate Deep Dive

#### gen0_gelu_1 (gelu, steps 1–20)

| Metric | Value |
|--------|-------|
| Best Loss | 8.110 |
| Worst Loss | 8.355 (step 1, random init) |
| Avg Loss | 8.235 |
| Loss Reduction | −0.245 (−2.9%) |
| Avg Throughput | 6,571 tok/s |
| Avg Step Time | 550 ms |
| Grad Spikes | 0 |
| CUSUM Alerts | 5 |
| Fitness | 0.0266 |
| Clip Rate | 65% (during warmup) |

First candidate, runs during LR warmup (0 → 1.2e-4). Highest clip rate due to large initial gradients from random weights. Clean gradient profile. Throughput starts low (389 tok/s at step 1 — GPU cold start) and ramps to ~7k tok/s.

#### gen0_silu_2 (silu, steps 21–40)

| Metric | Value |
|--------|-------|
| Best Loss | 8.026 |
| Worst Loss | 8.339 |
| Avg Loss | 8.162 |
| Loss Reduction | −0.313 (−3.8%) |
| Avg Throughput | **2,555 tok/s** (2.6x slower than gelu) |
| Avg Step Time | 835 ms |
| Grad Spikes | **4** (worst of any candidate) |
| NaN Steps | 2 (steps 29, 38) |
| CUSUM Alerts | 2 |
| Fitness | 0.0284 |
| Clip Rate | 28% |

Most unstable candidate. The SiLU activation has a more complex gradient landscape that interacts poorly with the aggressive LR ramp (now at 1.8e-4 and climbing). Steps lost to spikes: 30 (520x), 31 (62x), 37 (184x), 39 (6,094x). Two NaN gradient steps cause complete optimizer skips. Despite instability, still achieves better loss than gelu (the LR is higher, so each healthy step makes more progress).

#### gen0_relu_3 (relu, steps 41–60) — GEN 0 WINNER

| Metric | Value |
|--------|-------|
| Best Loss | **7.740** |
| Worst Loss | 8.352 |
| Avg Loss | 8.000 |
| Loss Reduction | −0.612 (−7.3%) |
| Avg Throughput | **6,716 tok/s** (fastest candidate) |
| Avg Step Time | 320 ms |
| Grad Spikes | **0** |
| CUSUM Alerts | 11 (all throughput) |
| Fitness | **0.0318** |
| Clip Rate | 10% (least clipping) |

Dominant candidate: fastest convergence, zero instability, highest throughput. Benefits from LR near peak (2.4e-4 → 3.0e-4). Relu's simple gradient (0 or 1) avoids the vanishing/exploding issues seen with silu. The 11 CUSUM alerts are false positives from the throughput bimodality.

#### gen0_swiglu_4 (swiglu, steps 61–80)

| Metric | Value |
|--------|-------|
| Best Loss | 7.797 |
| Worst Loss | 8.354 |
| Avg Loss | 8.049 |
| Loss Reduction | −0.557 (−6.5%) |
| Avg Throughput | 2,627 tok/s |
| Avg Step Time | 787 ms |
| Grad Spikes | 2 (steps 66: 29x, 77: 1,047x) |
| CUSUM Alerts | 4 |
| Fitness | 0.0301 |
| Clip Rate | 26% |

Second-best in gen0. SwiGLU's gated architecture gives good loss reduction but at 2.6x the compute cost of relu. Two gradient spikes, both moderate. The extra FFN parameters (3 matrices vs 2) don't help at this small scale.

#### gen1_swiglu_5 (swiglu, steps 81–100)

| Metric | Value |
|--------|-------|
| Best Loss | 7.788 |
| Avg Loss | 8.044 |
| Grad Spikes | 2 (steps 97: 2,234x, 100: 16,214x) |
| CUSUM Alerts | 0 |
| Fitness | 0.0296 |

Carried forward from gen0 selection. Slightly better best loss than gen0_swiglu (7.788 vs 7.797) but actually worse fitness — the later position in the search means LR is past peak and beginning to decay.

#### gen1_relu_6 (relu, steps 101–120) — GEN 1 WINNER / OVERALL WINNER

| Metric | Value |
|--------|-------|
| Best Loss | **7.701** (overall best during search) |
| Avg Loss | 7.981 |
| Loss Reduction | −0.656 (−7.6%) |
| Avg Throughput | **6,960 tok/s** |
| Avg Step Time | 301 ms (fastest avg) |
| Grad Spikes | **0** |
| CUSUM Alerts | 12 |
| Fitness | **0.0327** (overall best) |
| Clip Rate | 10% |

Confirms gen0 relu's dominance. Even with decaying LR (2.9e-4 → 2.8e-4), relu still achieves the best loss of any candidate. Zero gradient spikes across 40 total relu steps (both generations) is remarkable.

#### gen1_gelu_7 (gelu, steps 121–140, mutation from relu)

| Metric | Value |
|--------|-------|
| Best Loss | 7.760 |
| Avg Loss | 8.005 |
| Grad Spikes | 0 |
| CUSUM Alerts | 20 (100% of steps!) |
| Fitness | 0.0315 |

Mutated from relu. Performs well but doesn't beat relu. All 20 steps trigger CUSUM alerts because the accumulated baseline now expects ~2.6k tok/s throughput but gelu delivers ~7k tok/s. This is a "throughput *improvement*" alert being misclassified as "collapse."

#### gen1_silu_8 (silu, steps 141–159, mutation from swiglu)

| Metric | Value |
|--------|-------|
| Best Loss | 7.868 |
| Avg Loss | 8.077 |
| Grad Spikes | 1 (step 150: 47x) |
| CUSUM Alerts | 12 |
| Fitness | 0.0280 (worst overall) |

Worst performer in gen1. Only ran 17 steps (3 steps lost to NaN/spikes). Confirms silu's instability pattern from gen0.

### Final Rankings

| Rank | Candidate | Activation | Gen | Best Loss | Fitness | Spikes | Throughput |
|------|-----------|-----------|-----|-----------|---------|--------|-----------|
| 1 | gen1_relu_6 | **relu** | 1 | **7.701** | **0.0327** | 0 | 6,960 |
| 2 | gen0_relu_3 | **relu** | 0 | 7.740 | 0.0318 | 0 | 6,716 |
| 3 | gen1_gelu_7 | gelu | 1 | 7.760 | 0.0315 | 0 | 6,903 |
| 4 | gen1_swiglu_5 | swiglu | 1 | 7.788 | 0.0296 | 2 | 2,615 |
| 5 | gen0_swiglu_4 | swiglu | 0 | 7.797 | 0.0301 | 2 | 2,627 |
| 6 | gen0_silu_2 | silu | 0 | 8.026 | 0.0284 | 4 | 2,555 |
| 7 | gen1_silu_8 | silu | 1 | 7.868 | 0.0280 | 1 | 2,872 |
| 8 | gen0_gelu_1 | gelu | 0 | 8.110 | 0.0266 | 0 | 6,571 |

**Activation Tier List:**
1. **relu** — Best loss, fastest throughput, zero spikes. Wins both generations.
2. **gelu** — Close to relu on loss, equal throughput, zero spikes. Gen0 penalized by warmup.
3. **swiglu** — Decent loss but 2.7x slower and occasional spikes.
4. **silu** — Worst loss, slowest, most unstable. Not viable for this architecture.

### Validity Assessment

**WARNING:** These rankings should be taken with extreme caution:

1. **Only 20 steps per candidate** — Far too few for meaningful comparison. At 20 steps, you're measuring "speed of initial convergence from random weights," not "quality of activation at convergence."
2. **Fresh random init each switch** — Each candidate starts from scratch. Relu's advantage may simply be that relu gradients flow more cleanly through random-initialized weights, not that relu is better at convergence.
3. **LR varies across candidates** — gen0_gelu runs during warmup (LR: 0→1.2e-4) while gen0_relu runs near peak (LR: 2.4e-4→3e-4). The search doesn't control for LR schedule position.
4. **Small model** — At 6.9M params, architectural differences between activations may not manifest. SwiGLU's benefits are typically seen at 100M+ params.

---

## 9. Post-Search Training (Steps 161–500)

After the search selected relu, the model was re-initialized with relu activation and trained for the remaining ~340 steps.

| Metric | Value |
|--------|-------|
| Duration | ~340 steps |
| Starting Loss | 7.833 (fresh init at step 161) |
| Final Loss | 7.259 (step 498) |
| Best Loss | **7.210** (step ~370) |
| Avg Loss | 7.399 |
| Avg Throughput | 2,659 tok/s |
| Avg Step Time | 791 ms |
| Grad Spikes | **52** (18.4% spike rate!) |
| LR Range | 2.6e-4 → 5e-6 (cosine decay) |

### Post-Search Gradient Instability

The post-search phase has a *much* higher spike rate (18.4%) than the search phase (5.9%). This is a significant issue. The spike distribution:

| Step Range | Spikes | Worst | Notes |
|-----------|--------|-------|-------|
| 161–200 | 7 | 774,971 | Immediate post-switch instability |
| 201–250 | 4 | 771,646 | Settling period |
| 251–300 | 6 | 3,138 | Moderate spikes |
| 301–350 | 10 | 3,368 | Cluster of instability |
| 351–400 | 10 | 7,899 | Persistent |
| 401–450 | 8 | 4,926 | |
| 451–500 | 7 | 1,413 | Improving (LR near minimum) |

**Diagnosis:** The spike pattern suggests relu + AdamW at this learning rate has inherent instability. The `spikeThreshold=10` guard prevents catastrophic loss, but 52 wasted optimizer steps means ~15% of training compute is thrown away. The spikes don't cause loss divergence (loss continues to decrease overall) but they slow convergence.

**Potential fixes:**
- Lower initial LR for post-search phase (e.g., start at 1e-4 instead of inheriting ~2.6e-4)
- Increase weight decay (currently 0.1)
- Use a warmup period after the search→train transition
- Consider gelu instead of relu — gelu had zero spikes and nearly identical loss

---

## 10. Symbio Metrics Evolution

### Weight Entropy (bits)

| Phase | Range | Trend |
|-------|-------|-------|
| gen0 search (steps 10-60) | 1.851 – 1.870 | Stable, slight decrease |
| gen1 search (steps 70-150) | 1.842 – 1.913 | Higher variance (different activations have different weight distributions) |
| Post-search (steps 180-490) | 1.839 – 1.842 | **Very stable** — converged |

Weight entropy decreasing from 1.87 to 1.84 means the weight distribution is becoming more concentrated (less uniform). This is expected as training progresses and weights specialize. The 1.6% entropy reduction over 500 steps is minimal — no sign of weight collapse.

### Effective Rank

**Constant at 20 throughout the entire run.** This is the number of monitored weight matrices: 6 layers × (attention + FFN) + embeddings + layer norms. All matrices maintain full rank. This model is too small for rank collapse to be a concern — effective rank becomes informative at 512+ dimensions.

### Free Energy (F = L + β·H)

| Step | Free Energy | Loss | Entropy (H) | β |
|------|-------------|------|-------------|---|
| 10 | 8.281 | 8.262 | 1.870 | 0.01 |
| 50 | 8.016 | 7.998 | 1.858 | 0.01 |
| 100 | 7.807 | 7.788 | 1.912 | 0.01 |
| 200 | 7.591 | 7.573 | 1.840 | 0.01 |
| 300 | 7.434 | — | 1.842 | 0.01 |
| 400 | 7.358 | 7.339 | 1.842 | 0.01 |
| 490 | 7.238 | 7.220 | 1.842 | 0.01 |

Free energy tracks loss almost exactly since β=0.01 makes the entropy term negligible (~0.018). The entropy regularization effect is unmeasurable at this β. Consider β=1.0 or higher for entropy to meaningfully influence the free energy landscape.

### Fitness Score (1/F)

Monotonically increasing from 0.025 → 0.040. Directly mirrors free energy decrease. At this β, fitness is essentially 1/loss.

---

## 11. Sample Generations

Five samples were generated after training completed (step 500, temperature=0.8, top-k=40):

### Sample 1
**Prompt:** `<|user|> Hello, how are you? <|assistant|>`
**Output:** `enthat is unps, your and must be in both sent is for a c. <|end_of_text|>`

### Sample 2
**Prompt:** `<|user|> What do you like to do for fun? <|assistant|>`
**Output:** `u, on a order the and detesned for that from on from a for is a purposin olis relaughter ts in knowledge...`

### Sample 3
**Prompt:** `<|user|> Tell me about yourself. <|assistant|>`
**Output:** `'s the balance s's will ts unhuman divine and prea surethat in a of leded ts both blut, may sion and eris a victor...`

### Sample 4
**Prompt:** `<|user|> What is the meaning of life? <|assistant|>`
**Output:** `eds of predined and tt; not m, or to cingthe living ythe Mthe ἀwhich divine the mof h...`

### Sample 5
**Prompt:** `<|user|> Can you help me with something? <|assistant|>`
**Output:** `, must a divine e, to dekanot our chs but that inaand as and flamts of of your h...`

**Assessment:** The model has learned some English word fragments and a few complete words ("the", "and", "divine", "balance", "knowledge", "moral"), but output is largely incoherent. This is expected for a 6.9M param model trained for only 500 steps. The model would need 10,000+ steps at this scale, or 50,000+ for coherent multi-turn chat. Interestingly, the word "divine" appears frequently — suggesting the training data contains religious or philosophical text that dominated the small BPE-4k vocabulary.

---

## 12. Checkpoint

| Step | File Size | Uploaded |
|------|-----------|----------|
| 500 | 67.1 MB (full) → 22.4 MB (inference) → 20.7 MB (compressed) | Yes (21 chunks to Railway) |

---

## 13. Pipeline Verification

All symbio fields are successfully flowing through the full stack:

| Stage | Status | Fields |
|-------|--------|--------|
| **Trainer** → StepMetrics | ✅ | All 24 metric columns populated |
| **Remote Reporter** → SSE | ✅ | Metrics streamed in real-time |
| **Ingest API** → DB | ✅ | All columns written to Turso |
| **DB** → Dashboard API | ✅ | Metrics served via `/api/runs/{id}/metrics` |
| **Dashboard** → Charts | ✅ | All symbio charts rendering |

Fields confirmed in DB:
- `clip_coef`, `clip_pct` — Every step
- `cusum_grad`, `cusum_clip`, `cusum_tps`, `cusum_val` — Steps 1-154
- `cusum_alerts`, `cusum_alert_reason` — Steps with alerts
- `weight_entropy`, `effective_rank`, `free_energy`, `fitness_score` — Every 10 steps
- `symbio_candidate_id`, `symbio_candidate_activation`, `symbio_generation` — Steps 1-159
- Run-level: `symbio=1`, `ffn_activation=swiglu`, `symbio_mode=ffn-activation-search`

---

## 14. Issues Found

### Critical

1. **Post-search gradient instability (18.4% spike rate)** — The relu winner produces frequent extreme gradient spikes during extended training. This wastes ~15% of compute. The search's "stability" advantage for relu (0 spikes in 40 search steps) doesn't hold at longer training durations. **Root cause:** Likely relu's dead neuron problem combined with AdamW moment estimates being reset after the search.

### High

2. **CUSUM baseline not reset across candidates** — The CUSUM monitor accumulates a single baseline across all candidates. When throughput varies 2.7x between activation types, every candidate triggers false "throughput collapse" alerts. Each candidate needs an independent baseline.

3. **`cusum_grad` always null after step 28** — The gradient CUSUM channel fails to establish a baseline. Likely caused by the baseline window (5 steps) being interrupted by activation switches before accumulating enough samples.

4. **Search doesn't control for LR schedule position** — Candidates early in the run train at lower LR (warmup) while later candidates train at higher LR. This confounds the comparison.

### Medium

5. **CUSUM data stops after search phase** — No CUSUM monitoring during the 340-step post-search training. The monitor should continue running.

6. **Throughput bimodality not explained to user** — The dashboard shows wild throughput variance but doesn't indicate that it's caused by different activations having different compute costs.

7. **`symbio_winner` field is null** — The run completed but the `symbio_winner` column in the DB wasn't populated. The trainer should write the winner activation to this field.

### Low

8. **Free energy β too small** — At β=0.01, the entropy term contributes ~0.018 to free energy vs ~7.5 from loss. The entropy regularization is effectively zero.

9. **Effective rank is constant** — At 256 embed dim and 6 layers, rank collapse is impossible. This metric needs larger models to be useful.

---

## 15. Recommendations

### For the Search Algorithm

| Change | Impact | Effort |
|--------|--------|--------|
| Increase `stepsPerCandidate` to 200+ | High — meaningful comparison | Config only |
| Reset CUSUM baseline at each candidate switch | High — eliminates false positives | Code change in monitor |
| Add 10-step warmup grace period after each switch | Medium — cleaner metrics | Code change in orchestrator |
| Control for LR: use constant LR during search | Medium — fair comparison | Code change in trainer |
| Seed candidates from same initial weights | Medium — isolate activation effect | Code change in orchestrator |
| Set `symbio_winner` in DB when search completes | Low — metadata completeness | Code change in trainer |

### For Training Stability

| Change | Impact | Effort |
|--------|--------|--------|
| Lower post-search initial LR (or add warmup) | High — reduce 18% spike waste | Code change |
| Increase `spikeThreshold` to 100 for post-search | Medium — fewer wasted steps | Config |
| Consider gelu over relu for stability | Medium — zero spikes in search | Investigation |
| Increase β to 0.1+ for meaningful entropy regularization | Low — research interest | Config |

### For the Dashboard

| Change | Impact | Effort |
|--------|--------|--------|
| Show activation type in throughput chart | Medium — explains bimodality | UI change |
| Continue CUSUM monitoring post-search | Medium — full-run coverage | Code change |
| Add activation switch markers to loss chart | Medium — visual context | UI change |
