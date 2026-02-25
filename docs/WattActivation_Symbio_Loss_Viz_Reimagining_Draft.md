# WattActivation: Symbio Loss Visualization Reimagining (Draft)

Author: Lisa Watts
Date: 2026-02-26
Status: Draft (Design + Implementation Direction)

## 1. Problem Statement

The current symbiogenesis loss chart can imply a false story: a single model training continuously while activation "takes over". In reality, symbio search evaluates multiple candidates and frequently re-initializes model weights and optimizer state. A single continuous line across global steps visually suggests continuity that does not exist.

This is a representation mismatch.

## 2. First Principles (System Philosophy)

1. Symbio is a search process, not a single-trajectory training run.
2. Candidate-local dynamics and global search progress are different signals and must be shown separately (but linked).
3. Diversity pressure is not noise; it is an intentional control mechanism that can oppose convergence.
4. Visual continuity must reflect causal continuity. If weights reset, the line should not imply uninterrupted state continuity.
5. Selection semantics matter more than legacy training heuristics during active search.

## 3. Ground Truth Semantics (What the Data Means)

In symbio search:
- Each candidate window is a fresh evaluation episode (new weights, reset optimizer state).
- Global `step` is scheduler/runtime chronology, not candidate-local optimization time.
- Loss resets near candidate switches are expected and often desirable (fair comparison under fresh initialization).
- The meaningful global signal is the frontier (best-so-far / search envelope), not the raw stitched loss line alone.

## 4. Visualization Model (Target)

### 4.1 Primary View: Search Trajectory + Frontier (Unified)

A single chart (not duplicated panels) with layered semantics:
- Layer A: Candidate-local loss segments (broken at candidate boundaries)
- Layer B: Validation points/segments (also broken at candidate boundaries)
- Layer C: Search frontier (`Evo Best` / running best envelope)
- Layer D: Candidate switch events (vertical markers)
- Layer E: Per-candidate failure/overfit bands (`Evo Overfit`)
- Layer F: Optional global diagnostics (checkpoints, warmup, spikes)

Interpretation rule:
- Read Layer A locally (within each segment)
- Read Layer C globally (across the full run)

### 4.2 Convergence vs Diversity (Already Added)

The convergence tug-of-war view is conceptually correct and should remain a core companion panel.

Role in the system:
- Explains why frontier movement may stall even when search remains active
- Distinguishes productive exploration from search churn
- Provides a control-theoretic interpretation of diversity regularization pressure

### 4.3 Candidate Window Strip (Next Iteration)

Add a compact strip directly under the primary loss chart:
- One rectangle per candidate window
- Width = steps evaluated
- Color = activation family (or lineage family)
- Opacity/saturation = fitness percentile or best-val percentile
- Border = selected / pinned candidate

Purpose:
- Make the discrete episode structure visually explicit
- Reduce reliance on reading switch markers only

### 4.4 Frontier Attribution View (Next Iteration)

Add an optional mini-chart or legend-like table answering:
- Which candidate contributed each frontier drop?
- How much did it improve the frontier?
- Was the improvement retained by descendants?

This turns the frontier from a line into an explainable search outcome.

## 5. Interaction Design

### 5.1 Hover / Pin Semantics

Hovering a point should show:
- Global step
- Candidate ID
- Candidate activation / generation (if present)
- Candidate-local loss and val loss (raw)
- Whether point is near a switch event
- Frontier value at that step (optional next step)

Pinned interactions should synchronize:
- Loss chart
- Convergence tug-of-war (already implemented)
- Lineage / timeline panels

### 5.2 Marker Controls (Taxonomy)

Controls should be grouped by semantics, not by "traditional vs evolutionary":
- Search semantics: `Activation Switch`, `Evo Best`, `Evo Overfit`
- Validation / selection: `Best Val`, `Overfit`
- Run diagnostics: `Checkpoints`, `Warmup End`, `Grad Spikes`, `Loss Spikes`

Each control must have a clickable help icon explaining:
- Detection logic
- Intended interpretation
- Symbio caveats (global stitched heuristic vs per-candidate truth)

## 6. Anti-Patterns to Avoid

1. Duplicating the same loss/gradient chart into two panels with different overlays only.
2. Labeling symbio search trajectories as if they were a single continuous model optimization path.
3. Over-emphasizing global heuristics (e.g., overfit on stitched series) without caveats.
4. Hiding candidate episode boundaries while showing only switch markers.

## 7. Implementation Plan (Pragmatic)

### Phase 0 (Completed in this iteration)
- Remove duplicated loss panels
- Use a single unified chart with symbio-aware labeling
- Group marker pills by semantics
- Keep help tooltips on all marker pills
- Keep convergence tug-of-war panel

### Phase 1 (High value, low risk)
- Add explicit candidate window strip under the unified chart
- Add frontier attribution tooltip/table for pinned step
- Add candidate-local step index to tooltip (`local_step_in_candidate`)

### Phase 2 (Richer search-native view)
- Add candidate-normalized x-axis mode (0..N within candidate)
- Add small multiples for top-K candidate trajectories by generation/family
- Add frontier decomposition (drop events by lineage)

### Phase 3 (Experimental)
- Compare "fresh-init fairness" mode vs "warm-start transfer" mode visually if trainer semantics evolve
- Add uncertainty bands for early-step candidate variance across same activation families

## 8. Technical Notes (Current Code Reality)

The trainer currently re-initializes on candidate switches, so resets in raw loss are expected. The visualization should expose this explicitly rather than smoothing it away.

Relevant code paths:
- Candidate switch re-init: `packages/train/src/trainer.ts` (search orchestrator switch block)
- Raw step loss logging: `packages/train/src/trainer.ts` (`metrics.loss = lossVal`)
- UI loss chart rendering: `apps/web/src/components/charts.tsx`
- Activation switch extraction: `apps/web/src/components/symbio-charts.tsx`

## 9. Success Criteria

A correct symbio loss visualization should let a user answer, at a glance:
- Are candidates improving locally?
- Is the search frontier still moving?
- Is diversity pressure helping exploration or stalling convergence?
- Which candidate families are actually responsible for gains?
- Are we observing real search progress, or just repeated resets with no frontier movement?

---

This draft intentionally prioritizes semantic correctness over aesthetic continuity.
