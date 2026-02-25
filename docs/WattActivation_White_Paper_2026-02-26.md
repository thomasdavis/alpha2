# WattActivation

**Author:** Lisa Watts  
**Document Type:** Technical White Paper (Commit-Derived Repository Analysis)  
**Repository:** `alpha`  
**Analysis Date:** 2026-02-26  
**Commit Inclusion Rule:** `git log --since="24 hours ago"` (committer time)  
**Observed Commit Interval:** 2026-02-25T07:12:09+10:00 to 2026-02-26T03:33:22+10:00 (local `+10:00`)

## Abstract

This white paper presents a commit-level empirical analysis of the `alpha` repository across the most recent 24-hour lookback window, with an emphasis on the co-evolution of hardware-aware training kernels, activation-function experimentation, symbiogenetic search infrastructure, and observability-driven visualization systems. The analysis covers **59 commits** by **2 authors**, totaling **56,137 insertions**, **14,755 deletions**, and **70,892 lines of text churn** across **424 unique files**.

The central result is a rapid convergence of three previously separable concerns into a unified research-engineering stack: (1) GPU execution efficiency (`packages/helios`), (2) activation-search / symbiogenesis logic (`packages/symbiogenesis` plus model/trainer integrations), and (3) high-bandwidth observability interfaces (`apps/web`). This convergence culminates in the introduction and visualization of expanded activation spaces (including universal approximator and KAN spline variants), lineage-aware evolutionary UIs, and evolutionary-aware loss overlays.

The report is intentionally scientific in format: methods, quantitative results, subsystem concentration, temporal phase analysis, threats to validity, and a complete appendix listing every included commit.

## Keywords

Activation evolution, symbiogenesis, GPU kernels, FlashAttention, mixed precision, training observability, lineage visualization, D3, KAN spline, universal approximator, repository mining.

## 1. Executive Summary

1. The 24-hour window shows a multi-phase engineering sprint spanning kernel/runtime performance, inference packaging, training correctness, and a dense symbiogenesis + visualization expansion.
2. `packages/helios` and `apps/web` dominate code-subsystem churn, indicating a strong hardware/UX co-design loop around training experimentation.
3. Symbiogenesis evolves from a switchable training mode to a multi-package platform with metrics, orchestration, lineage tracking, composed activations, and specialized visual analytics.
4. Documentation and artifact churn are large enough to distort raw metrics unless normalized; this report explicitly separates raw and source-only views.
5. The terminal UI/chart commits in the window introduce evolutionary-aware loss interpretation (`Evo Best`, `Evo Overfit`) to better model nonstationary loss trajectories under activation switching.

## 2. Methods

### 2.1 Data Source and Inclusion Criteria

The analysis was generated directly from local Git history using committer timestamps. A commit was included if it satisfied:

- `committer_time >= now() - 24h`

The resulting corpus spans `20.35` hours of observed commit activity within the 24-hour lookback period (i.e., there was a quiet gap before the first included commit).

### 2.2 Metrics

We define:

- **Insertions (I):** added lines from `git show --numstat`
- **Deletions (D):** deleted lines from `git show --numstat`
- **Churn (C):** `C = I + D`
- **Net delta (N):** `N = I - D`
- **File-touch event:** one file reported in one commit

A source-normalized view is reported to separate executable-system change from documentation and artifacts.

### 2.3 Normalization and Validity Controls

To reduce interpretive distortion, the source-normalized subset excludes:

- `.histchat/` batch JSONL artifacts
- `docs/` and root markdown files
- `package-lock.json`
- static diagrams/assets (e.g., `.svg`, `.html`, image files)

This matters because the window contains both a large docs/artifact ingestion event and a later docs reorganization commit.

## 3. Corpus Statistics

### 3.1 Core Metrics

| Metric | Value |
|---|---|
| Commit count (N) | 59 |
| Observed commit window | 2026-02-25T07:12:09+10:00 to 2026-02-26T03:33:22+10:00 |
| Observed active span | 20.35 h |
| Authors | Thomas Davis (56), Ajax Davis (3) |
| Raw insertions / deletions | 56,137 / 14,755 |
| Raw net delta | +41,382 |
| Raw churn (insertions+deletions) | 70,892 |
| Unique files touched | 424 |
| File-touch events | 576 |
| Median commit size (churn) | 137 |
| Mean commit size (churn) | 1,201.6 |
| P90 commit size (churn) | 2,315 |
| Commit intensity | 2.90 commits/hour |
| Median inter-commit gap | 8.4 min |
| Mean inter-commit gap | 21.1 min |

### 3.2 Source-Normalized Metrics

| Metric | Value |
|---|---|
| Source-only filter | Excludes `.histchat/`, `docs/`, root `*.md`, `package-lock.json`, static diagrams/assets |
| Source insertions / deletions | 25,193 / 11,430 |
| Source net delta | +13,763 |
| Source churn | 36,623 |
| Source churn share of raw churn | 51.7% |
| Source unique files | 117 |
| Source file-touch events | 259 |
| Documentation markdown churn (`.md`) | 12,604 |
| Batch/chat JSONL churn (`.jsonl`) | 12,997 |

### 3.3 Author Distribution

| Author | Commits | Share |
|---|---|---|
| Thomas Davis | 56 | 94.9% |
| Ajax Davis | 3 | 5.1% |

## 4. Temporal Dynamics (Commit Cadence)

The commit cadence is burst-like rather than stationary. The median inter-commit gap is **8.4 minutes**, while the mean is **21.1 minutes**, indicating skew from batch and milestone commits.

### 4.1 Hourly Commit Density

| Hour (local +10) | Commits |
|---|---|
| 2026-02-25 07:00 | 9 |
| 2026-02-25 08:00 | 2 |
| 2026-02-25 10:00 | 2 |
| 2026-02-25 15:00 | 8 |
| 2026-02-25 16:00 | 5 |
| 2026-02-25 17:00 | 4 |
| 2026-02-25 18:00 | 1 |
| 2026-02-25 19:00 | 2 |
| 2026-02-25 21:00 | 3 |
| 2026-02-25 22:00 | 3 |
| 2026-02-25 23:00 | 3 |
| 2026-02-26 00:00 | 5 |
| 2026-02-26 01:00 | 4 |
| 2026-02-26 02:00 | 4 |
| 2026-02-26 03:00 | 4 |

### 4.2 Phase Segmentation

| Phase | Commits | Insertions | Deletions | Churn | Start | End |
|---|---|---|---|---|---|---|
| Phase I: Kernel & Training Performance Foundation | 9 | 10,127 | 5,879 | 16,006 | 2026-02-25T07:12:09+10:00 | 2026-02-25T07:48:42+10:00 |
| Phase II: Inference Packaging and HF Surface Expansion | 4 | 18,314 | 1,954 | 20,268 | 2026-02-25T08:18:28+10:00 | 2026-02-25T10:48:34+10:00 |
| Phase III: Runtime Correctness, Throughput, and UI Unification | 20 | 4,391 | 2,675 | 7,066 | 2026-02-25T15:48:20+10:00 | 2026-02-25T19:52:45+10:00 |
| Phase IV: Symbiogenesis Platform, Visualization, and Activation Science | 25 | 12,697 | 4,247 | 16,944 | 2026-02-25T21:12:40+10:00 | 2026-02-26T03:31:21+10:00 |
| Phase V: Documentation Reorganization | 1 | 10,608 | 0 | 10,608 | 2026-02-26T03:33:22+10:00 | 2026-02-26T03:33:22+10:00 |

## 5. Subsystem Concentration and Hotspots

### 5.1 Churn Concentration by Major Code Subsystem (`apps/*`, `packages/*`)

| Subsystem | Churn | Touches |
|---|---|---|
| packages/helios | 14,645 | 28 |
| apps/web | 12,302 | 57 |
| packages/symbiogenesis | 2,190 | 37 |
| packages/inference | 955 | 7 |
| packages/train | 705 | 21 |
| packages/autograd | 649 | 17 |
| packages/model | 569 | 11 |
| apps/hf | 378 | 6 |
| packages/db | 241 | 13 |
| packages/tensor | 221 | 3 |
| apps/cli | 159 | 14 |
| packages/core | 158 | 15 |

Interpretation:

- `packages/helios`: **44.1%** of `apps/* + packages/*` churn
- `apps/web`: **37.0%** of `apps/* + packages/*` churn
- `packages/symbiogenesis`: **6.6%** of `apps/* + packages/*` churn

This distribution supports a co-design interpretation: low-level compute and high-level observability advanced together.

### 5.2 File-Level Hotspots

| File | Churn | Touches |
|---|---|---|
| package-lock.json | 6,250 | 8 |
| apps/web/src/components/symbio-charts.tsx | 6,237 | 14 |
| packages/helios/src/kernels.ts | 4,760 | 1 |
| packages/helios/src/kernels/nn.ts | 2,213 | 2 |
| packages/helios/src/kernels/attention.ts | 1,836 | 1 |
| apps/web/src/components/radial-viz.tsx | 1,769 | 3 |
| apps/web/src/components/run-detail-view.tsx | 1,662 | 10 |
| packages/helios/src/kernels/elementwise.ts | 1,608 | 1 |
| architecture-diagram.html | 1,366 | 1 |
| apps/web/src/components/charts.tsx | 1,334 | 5 |
| packages/helios/src/kernels/matmul.ts | 1,127 | 2 |
| docs/SYMBIO_PHENOTYPE_ANALYSIS.md | 1,092 | 1 |
| docs/PRD_SYMBIOGENESIS_ADAPTER_LAYER_20260225.md | 1,090 | 1 |
| architecture.svg | 1,052 | 1 |
| docs/SYMBIO_JUDGE_EVOLUTION.md | 962 | 1 |

### 5.3 Extension-Level Churn Composition

| Extension | Churn | Touches |
|---|---|---|
| .ts | 22,395 | 178 |
| .jsonl | 12,997 | 261 |
| .md | 12,604 | 45 |
| .tsx | 11,701 | 36 |
| .json | 6,452 | 35 |
| .html | 1,366 | 1 |
| .py | 1,156 | 7 |
| .svg | 1,052 | 1 |
| .d2 | 659 | 1 |
| .mjs | 293 | 2 |
| .sh | 139 | 2 |
| .c | 42 | 1 |

## 6. Results: Technical Contributions by Theme

### 6.1 Thematic Incidence (Multi-Label Commit Classification)

| Thematic incidence (multi-label) | Commits tagged | Share of N |
|---|---|---|
| GPU Kernels & Runtime | 11 | 18.6% |
| Training Pipeline & Infra | 16 | 27.1% |
| Inference Packaging / Serving | 5 | 8.5% |
| Dashboard / Charting / UX | 20 | 33.9% |
| Symbiogenesis / Activation Evolution | 20 | 33.9% |
| Documentation / Repo Organization | 5 | 8.5% |

These labels are intentionally overlapping. A single commit may contribute simultaneously to, for example, training infrastructure and dashboard/UX.

### 6.2 GPU Kernels and Runtime Substrate (Helios)

The window opens with a high-churn runtime transformation centered on `packages/helios`, including FlashAttention-related modularization, fused operations, matmul tiling refinements, mixed-precision support, additional GPU kernels (`slice`, `scatterSlice`, dropout mask), and runtime safety hardening. This is a performance substrate build-out, not isolated micro-tuning.

### 6.3 Training Pipeline, Correctness, and Operability

Training-loop commits establish a stronger operational baseline via activation checkpointing, deterministic dropout RNG, parity fixes, DataLoader correctness fixes, streamed/buffered metric pathways, adaptive sync/GC policy, and improved training script flag propagation. These changes reduce confounds and operational fragility during rapid experimentation.

### 6.4 Inference Packaging and Surface Expansion

The extraction of `@alpha/inference`, HF app addition, CLI integration, and Docker build updates indicate architectural decoupling between model execution for serving and the training/research stack. This reduces coupling cost for future deployment-oriented evolution.

### 6.5 Symbiogenesis as a First-Class Platform

The symbiogenesis commits collectively add a real platform surface: config schemas/loaders, metrics collectors, search orchestration, ranking/reporting, trainer wiring, DB support, and dashboard telemetry. The evidence suggests a transition from “feature flag” to sustained experimental mode.

### 6.6 Activation-Space Expansion and Composed Activation Search

The introduction of universal approximator and KAN spline activations expands the activation search space beyond fixed classical choices. Subsequent commits add validation support, configuration retuning, and eventually composed activation graph mutation with autograd evaluation and DB migration support. This is a shift from categorical activation selection toward structural activation-program evolution.

### 6.7 Observability and Evolutionary Visualization Stack

The web/UI layer evolves rapidly, especially in `apps/web/src/components/symbio-charts.tsx` and `apps/web/src/components/charts.tsx`, adding activation-switch logs, pinned-step synchronization, lineage tree interactions, pie lineage nodes, hereditary links, Sankey flow views, D3 migration, and phase-change UX improvements.

The final chart commit in the window introduces evolutionary-aware loss overlays (`Evo Best`, `Evo Overfit`) to accommodate nonstationary trajectories caused by activation switching. This is scientifically notable because the visualization semantics are explicitly updated to match the underlying search process.

## 7. WattActivation Interpretation (Scientific Framing)

### 7.1 Thesis

> **WattActivation** is the co-design principle that activation-function search becomes materially more useful when hardware efficiency, training-loop instrumentation, and visualization semantics improve in lockstep.

The commit sequence supports this thesis by showing repeated coupling between:

- kernel/runtime efficiency work,
- activation-search and symbiogenesis feature expansion,
- and observability/UX redesigns that preserve interpretability under evolutionary dynamics.

### 7.2 Why the “Watt” Prefix Is Defensible

The corpus is not only about activations. It materially invests in the energy/performance substrate of experimentation: FlashAttention enablement, fused kernels, mixed precision, grouped QKV projection, checkpointing, and GPU-specific runtime safety. The result is a practical research posture where activation science is inseparable from computational efficiency.

## 8. Representative High-Impact Commits

| Time | Commit | Author | Churn | Files | Subject |
|---|---|---|---|---|---|
| 2026-02-25T07:12:09+10:00 | aea09957 | Thomas Davis | 14,827 | 29 | Flash Attention kernels + kernel module split + training improvements |
| 2026-02-25T08:18:28+10:00 | 147fd82c | Ajax Davis | 3,356 | 12 | Extract inference engine into @alpha/inference package |
| 2026-02-25T10:48:34+10:00 | 4ffa6a90 | Ajax Davis | 16,800 | 274 | Add HF app, docs, architecture diagrams, and clean up stale files |
| 2026-02-25T17:54:59+10:00 | c28d27c7 | Thomas Davis | 2,315 | 3 | Extract shared chart components, unify training page with run detail view |
| 2026-02-25T19:52:45+10:00 | 2eee2e5f | Thomas Davis | 1,861 | 15 | Add utility scripts, data tools, and GPU test harnesses |
| 2026-02-25T21:12:40+10:00 | 9170b719 | Thomas Davis | 2,751 | 44 | Implement Symbiogenesis mode: --symbio flag with full monitoring, metrics, and FFN activation search |
| 2026-02-25T22:56:46+10:00 | b461932c | Thomas Davis | 2,363 | 6 | Add evolutionary metrics, interactive charts, help icons, and UI improvements |
| 2026-02-25T23:54:07+10:00 | e6477261 | Thomas Davis | 2,230 | 13 | Add universal approximator & KAN spline activations, evolutionary tree UI, symbio config update |
| 2026-02-26T00:51:05+10:00 | 073bce81 | Thomas Davis | 1,748 | 6 | Add 3D radial training visualization (Three.js) and activation evolution docs |
| 2026-02-26T01:04:55+10:00 | f3c3591a | Thomas Davis | 1,839 | 4 | Replace 3D Three.js viz with 2D canvas radial activation oscillator |
| 2026-02-26T02:12:47+10:00 | d6c58535 | Thomas Davis | 852 | 17 | Add composed activation evolution: structural graph mutations, autograd evaluation, DB migration v8 |
| 2026-02-26T03:12:22+10:00 | 04b0fd43 | Thomas Davis | 1,739 | 3 | Replace amcharts with D3, reimagine evolutionary search UX, enhance phase changes |
| 2026-02-26T03:31:21+10:00 | 9db7c6a8 | Thomas Davis | 181 | 1 | Add evolutionary loss overlays to training chart |
| 2026-02-26T03:33:22+10:00 | 608bd4d0 | Thomas Davis | 10,608 | 33 | Move root markdown files into docs |

## 9. Discussion

### 9.1 Engineering Pattern Observed

The commit stream follows a strong exploratory loop:

1. Build performance headroom and runtime correctness.
2. Expand search/model capability.
3. Expand observability and visualization semantics.
4. Retune configurations and UX based on observed dynamics.

### 9.2 Scientific Maturity Signals

Observed maturity signals include deterministic RNG handling, validation-path updates for newly introduced activation types, DB schema evolution for lineage/metrics, and synchronized multi-chart inspection patterns.

### 9.3 Open Questions (Outside the Scope of Commit Mining)

This white paper analyzes repository change events, not experiment outcomes. It cannot by itself establish:

- realized throughput gains under standardized workloads,
- accuracy/quality effects of activation search variants,
- training stability under mixed precision and checkpointing at scale,
- or generalization of the new UI semantics to larger evolutionary histories.

## 10. Threats to Validity

1. Commit messages and churn do not directly measure runtime speed or model quality.
2. Some commits aggregate multiple conceptual changes.
3. Documentation/artifact-heavy commits inflate raw churn if not normalized.
4. Rename detection and path normalization affect file- and extension-level accounting.
5. The analysis is a single-window snapshot and may not reflect longer-run development behavior.

## 11. Conclusion

Across **59 commits** in an observed **20.35-hour active interval**, the `alpha` repository exhibits a coherent convergence of GPU runtime engineering, training infrastructure, activation-search science, and scientific visualization design. The strongest technical arc is the maturation of symbiogenesis from a feature flag into an observable evolutionary platform, capped by chart semantics that explicitly accommodate activation-switching-induced nonstationarity.

In that sense, **WattActivation** is not merely a title; it is an accurate systems-level description of the engineering trajectory captured in this commit window.

## 12. Reproducibility Appendix (Commands)

```bash
git log --since='24 hours ago' --reverse --pretty=format:'%H	%ct	%cI	%an	%s'
git show --numstat --format= --find-renames --find-copies <commit>
```

## Appendix A. Complete Commit Ledger (All Included Commits)

| # | Commit time (+10) | Hash | Author | Files | Ins | Del | Churn | Subject |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-02-25T07:12:09+10:00 | aea09957 | Thomas Davis | 29 | 9125 | 5702 | 14827 | Flash Attention kernels + kernel module split + training improvements |
| 2 | 2026-02-25T07:12:09+10:00 | 96fd5d9a | Thomas Davis | 6 | 174 | 62 | 236 | Activation checkpointing: trade compute for memory in transformer layers |
| 3 | 2026-02-25T07:12:09+10:00 | 05ff9644 | Thomas Davis | 5 | 269 | 7 | 276 | Fused sumOfSquares kernel: save 85 GPU ops/step in gradient norm computation |
| 4 | 2026-02-25T07:12:09+10:00 | 94451c58 | Thomas Davis | 3 | 35 | 16 | 51 | Parameterize matmul tile size: tile=32 for large matrices (2x memory efficiency) |
| 5 | 2026-02-25T07:12:09+10:00 | 88009740 | Thomas Davis | 6 | 389 | 67 | 456 | Mixed precision foundation: f16 dtype, cast kernels (f32↔f16), castDtype backend method |
| 6 | 2026-02-25T07:12:09+10:00 | 3f0f4233 | Thomas Davis | 5 | 97 | 14 | 111 | Mixed precision training: f16 activations, dynamic loss scaling, --fp16 flag |
| 7 | 2026-02-25T07:12:09+10:00 | e2290d94 | Thomas Davis | 1 | 6 | 0 | 6 | Add --fp16 and --checkpoint flags to GCP training script |
| 8 | 2026-02-25T07:23:22+10:00 | ebf4cb35 | Thomas Davis | 4 | 30 | 10 | 40 | Fix web dashboard crash: null-safe toLocaleString for nullable DB fields |
| 9 | 2026-02-25T07:48:42+10:00 | 64ad7c3e | Thomas Davis | 1 | 2 | 1 | 3 | Fix training page crash: null-safe formatNumber for SSE data |
| 10 | 2026-02-25T08:18:28+10:00 | 147fd82c | Ajax Davis | 12 | 1462 | 1894 | 3356 | Extract inference engine into @alpha/inference package |
| 11 | 2026-02-25T08:25:43+10:00 | 0f9f5c88 | Ajax Davis | 1 | 1 | 0 | 1 | Add @alpha/inference to web Dockerfile build |
| 12 | 2026-02-25T10:29:33+10:00 | 107a50a7 | Thomas Davis | 3 | 89 | 22 | 111 | Wire fast inference engine into CLI sample command |
| 13 | 2026-02-25T10:48:34+10:00 | 4ffa6a90 | Ajax Davis | 274 | 16762 | 38 | 16800 | Add HF app, docs, architecture diagrams, and clean up stale files |
| 14 | 2026-02-25T15:48:20+10:00 | 5c1460d4 | Thomas Davis | 2 | 27 | 9 | 36 | Fix DataLoader off-by-one, timing metrics split, and add event loop yield |
| 15 | 2026-02-25T15:48:20+10:00 | 06eb1832 | Thomas Davis | 3 | 130 | 26 | 156 | Add deterministic dropout RNG and fix FlashAttention dropout parity |
| 16 | 2026-02-25T15:48:20+10:00 | da145568 | Thomas Davis | 4 | 125 | 57 | 182 | Extract broadcast helpers to @alpha/core and add correctness tests |
| 17 | 2026-02-25T15:48:20+10:00 | 4d7e560e | Thomas Davis | 6 | 383 | 214 | 597 | Split inference engine into InferenceWeights + InferenceSession |
| 18 | 2026-02-25T15:48:21+10:00 | 194798f4 | Thomas Davis | 2 | 77 | 18 | 95 | Helios safety: native bounds checks, submit error handling, purge sync |
| 19 | 2026-02-25T15:48:21+10:00 | c7ed286c | Thomas Davis | 2 | 23 | 20 | 43 | Stream checkpoint writes and fix remote metric reporting |
| 20 | 2026-02-25T15:48:21+10:00 | 686ca4d6 | Thomas Davis | 4 | 98 | 39 | 137 | Descriptive run names, GCP training improvements, CLAUDE.md cleanup |
| 21 | 2026-02-25T15:56:29+10:00 | aba7b766 | Thomas Davis | 4 | 52 | 12 | 64 | Adaptive sync/GC policy and buffered metrics writer |
| 22 | 2026-02-25T16:07:37+10:00 | 635b46da | Thomas Davis | 5 | 86 | 21 | 107 | Grouped QKV projection: single GEMM instead of three |
| 23 | 2026-02-25T16:36:10+10:00 | ec9a4efe | Thomas Davis | 1 | 64 | 0 | 64 | Dashboard: auto-detect overfitting onset on loss chart |
| 24 | 2026-02-25T16:38:23+10:00 | f111fe50 | Thomas Davis | 1 | 291 | 0 | 291 | Add project README |
| 25 | 2026-02-25T16:44:38+10:00 | 39f76c8e | Thomas Davis | 8 | 131 | 11 | 142 | Fused MLP op and sequence packing DataLoader |
| 26 | 2026-02-25T16:49:57+10:00 | ed604160 | Thomas Davis | 1 | 13 | 4 | 17 | Relax overfit detection: 2% rise OR 3+ consecutive val_loss increases |
| 27 | 2026-02-25T17:20:46+10:00 | 02dcf6a9 | Thomas Davis | 1 | 262 | 24 | 286 | Dashboard: toggleable event markers on loss chart |
| 28 | 2026-02-25T17:50:48+10:00 | fcfd0222 | Thomas Davis | 8 | 603 | 28 | 631 | GPU slice, scatterSlice, and dropout mask kernels |
| 29 | 2026-02-25T17:54:59+10:00 | c28d27c7 | Thomas Davis | 3 | 130 | 2185 | 2315 | Extract shared chart components, unify training page with run detail view |
| 30 | 2026-02-25T17:55:24+10:00 | c66cb59f | Thomas Davis | 1 | 5 | 1 | 6 | Enable FlashAttention during dropout training |
| 31 | 2026-02-25T18:03:26+10:00 | f748ae9b | Thomas Davis | 1 | 4 | 4 | 8 | Charts: start x-axis from iteration 0 |
| 32 | 2026-02-25T19:33:57+10:00 | 29bee418 | Thomas Davis | 1 | 26 | 2 | 28 | Run detail page: poll for fresh data every 60 seconds |
| 33 | 2026-02-25T19:52:45+10:00 | 2eee2e5f | Thomas Davis | 15 | 1861 | 0 | 1861 | Add utility scripts, data tools, and GPU test harnesses |
| 34 | 2026-02-25T21:12:40+10:00 | 9170b719 | Thomas Davis | 44 | 2720 | 31 | 2751 | Implement Symbiogenesis mode: --symbio flag with full monitoring, metrics, and FFN activation search |
| 35 | 2026-02-25T21:22:47+10:00 | e27c6392 | Thomas Davis | 3 | 5 | 4 | 9 | Fix Docker build: add symbiogenesis package, fix post-search sample crash |
| 36 | 2026-02-25T21:52:53+10:00 | a6e22d44 | Thomas Davis | 1 | 192 | 8 | 200 | Add activation switch log table to symbio dashboard |
| 37 | 2026-02-25T22:01:29+10:00 | 0c1471dd | Thomas Davis | 1 | 2 | 2 | 4 | Fix CUSUM chart filtering on cusum_grad only |
| 38 | 2026-02-25T22:42:24+10:00 | f56aedbb | Thomas Davis | 4 | 1146 | 587 | 1733 | Make all run detail charts interactive with synced pinned step markers |
| 39 | 2026-02-25T22:56:46+10:00 | b461932c | Thomas Davis | 6 | 1546 | 817 | 2363 | Add evolutionary metrics, interactive charts, help icons, and UI improvements |
| 40 | 2026-02-25T23:18:02+10:00 | 9f5f8770 | Thomas Davis | 1 | 22 | 14 | 36 | Add pinned step markers to all symbio charts for consistent cross-chart sync |
| 41 | 2026-02-25T23:54:07+10:00 | e6477261 | Thomas Davis | 13 | 2105 | 125 | 2230 | Add universal approximator & KAN spline activations, evolutionary tree UI, symbio config update |
| 42 | 2026-02-25T23:59:16+10:00 | 773c9245 | Thomas Davis | 1 | 6 | 4 | 10 | Update symbio search config: 6 activations, 3000 steps/candidate, diversity bonus |
| 43 | 2026-02-26T00:00:56+10:00 | e3420806 | Thomas Davis | 1 | 2 | 2 | 4 | Update symbio config: generations=20, population=50000 |
| 44 | 2026-02-26T00:06:59+10:00 | c70a1e3b | Thomas Davis | 1 | 1 | 1 | 2 | Add universal and kan_spline to valid activation pool in validation |
| 45 | 2026-02-26T00:14:12+10:00 | 632b6b78 | Thomas Davis | 1 | 1 | 1 | 2 | Change run detail poll interval from 60s to 15s |
| 46 | 2026-02-26T00:30:56+10:00 | f5c202fa | Thomas Davis | 1 | 3 | 3 | 6 | Symbio config: pop=6, steps=30, gens=12 for rapid evolution (~30min) |
| 47 | 2026-02-26T00:51:05+10:00 | 073bce81 | Thomas Davis | 6 | 1744 | 4 | 1748 | Add 3D radial training visualization (Three.js) and activation evolution docs |
| 48 | 2026-02-26T01:04:55+10:00 | f3c3591a | Thomas Davis | 4 | 628 | 1211 | 1839 | Replace 3D Three.js viz with 2D canvas radial activation oscillator |
| 49 | 2026-02-26T01:24:12+10:00 | ac5e1b5f | Thomas Davis | 4 | 304 | 2 | 306 | Add candidate lineage columns to DB and traditional lineage tree chart |
| 50 | 2026-02-26T01:25:13+10:00 | 05358177 | Thomas Davis | 1 | 20 | 0 | 20 | Add symbio config for 50k novels run |
| 51 | 2026-02-26T01:34:28+10:00 | ae5cf3a1 | Thomas Davis | 1 | 2 | 2 | 4 | Symbio config: 20 steps/candidate, 416 generations for rapid evolution |
| 52 | 2026-02-26T02:12:47+10:00 | d6c58535 | Thomas Davis | 17 | 807 | 45 | 852 | Add composed activation evolution: structural graph mutations, autograd evaluation, DB migration v8 |
| 53 | 2026-02-26T02:15:23+10:00 | 16beddb8 | Thomas Davis | 1 | 1 | 1 | 2 | Update composed symbio config: 250 generations for 50k total steps (8×25×250) |
| 54 | 2026-02-26T02:38:19+10:00 | e05f8712 | Thomas Davis | 2 | 48 | 22 | 70 | Fix NaN color crash and null element errors in dashboard charts |
| 55 | 2026-02-26T02:45:21+10:00 | 876ee07a | Thomas Davis | 1 | 177 | 55 | 232 | Add pan/zoom to lineage tree: scroll to zoom, drag to pan, minimap, fit button |
| 56 | 2026-02-26T03:07:52+10:00 | bcaa7f5b | Thomas Davis | 1 | 476 | 125 | 601 | Pie chart lineage nodes, hereditary lines, Sankey diagram, incremental DAG updates |
| 57 | 2026-02-26T03:12:22+10:00 | 04b0fd43 | Thomas Davis | 3 | 565 | 1174 | 1739 | Replace amcharts with D3, reimagine evolutionary search UX, enhance phase changes |
| 58 | 2026-02-26T03:31:21+10:00 | 9db7c6a8 | Thomas Davis | 1 | 174 | 7 | 181 | Add evolutionary loss overlays to training chart |
| 59 | 2026-02-26T03:33:22+10:00 | 608bd4d0 | Thomas Davis | 33 | 10608 | 0 | 10608 | Move root markdown files into docs |
