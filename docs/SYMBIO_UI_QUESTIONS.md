# Symbiogenesis UI Feedback & Questions

Date: 2026-02-25
Run: `historic_chat_v2_20260225123612_2ao8` (50k symbio, adaptiveBatch, trackMIProfiles)

---

## Training Behavior Questions

### Why does loss spike at the beginning when switching activators?

When the evolutionary search switches from one activation function to another (e.g. GeLU -> SiLU), the model weights were optimized for the previous activation's nonlinearity. The new activation has a different shape (different gradients, different output range), so the existing weights are suddenly mismatched. This causes an immediate loss spike as the optimizer needs several steps to adapt the weights to the new activation landscape.

Think of it like switching the engine in a car while driving -- the transmission ratios are all wrong for the new engine, so performance drops until everything re-adapts. With only 20 steps per candidate in the test config (`stepsPerCandidate: 20`), there's barely enough time to recover before the next switch.

**Mitigation ideas:**
- Increase `stepsPerCandidate` to give each candidate more recovery time (1000+ recommended)
- Use weight rescaling when switching (normalize by activation output statistics)
- Warm-restart the optimizer state on activation switch
- Use cosine annealing within each candidate's evaluation window

### What is the CUSUM Change-Point Monitor chart?

CUSUM (Cumulative Sum) is a statistical process control method for detecting regime shifts in time series. The chart shows four independent monitors:

- **cusum_grad** (gradient norm) -- detects when gradient magnitude suddenly changes regime, indicating the loss landscape has fundamentally shifted
- **cusum_clip** (clipping percentage) -- detects onset of persistent gradient clipping, suggesting training is hitting instability boundaries
- **cusum_tps** (tokens per second) -- detects throughput collapse, indicating memory pressure or compute bottlenecks
- **cusum_val** (validation loss) -- detects validation loss divergence, catching overfitting or training collapse

**Algorithm:** Page's test -- accumulates standardized deviations from a baseline. When the cumulative sum exceeds a sensitivity threshold (default 4.0 std devs), an alert fires. The baseline is established from the first N steps (default 10).

**Interpreting the chart:**
- Flat near zero = stable training, no regime change detected
- Rising curve = deviation accumulating, approaching alert
- Spike/peak = alert fired, regime shift detected
- The higher the peak, the more severe the shift

### What does the Search Candidates table mean?

The table shows all candidates ever created during the evolutionary activation search. Each row is a candidate that was trained and evaluated:

| Field | Meaning |
|-------|---------|
| ID | Unique identifier: `gen{N}_{activation}_{counter}` |
| Activation | Which FFN activation function this candidate uses |
| Generation | Which evolutionary generation (0 = initial, 1+ = offspring) |
| Best Loss | Lowest training loss achieved during evaluation |
| Best Val Loss | Lowest validation loss achieved |
| Fitness | Multi-objective score: accuracy - complexity penalty |
| Steps | How many training steps this candidate ran |
| Alive | Whether this candidate is in the current generation |

### Why does the Search Candidates table have duplicate entries?

Duplicates are **by design** in the evolutionary algorithm:

1. **Elite preservation:** Top-performing parents are cloned into the next generation with the same activation, creating a new candidate with identical activation but fresh evaluation
2. **Mutation can produce same activation:** With only 4 activations in the pool, mutation (random re-selection) has a 25% chance of picking the same one
3. **Pool cycling:** If `populationSize > activationPool.length`, the initial population wraps around, creating multiple candidates per activation

Each "duplicate" has a unique ID (`gen0_silu_1` vs `gen1_silu_3`) and independently measured fitness, so they represent different evaluation runs of the same architecture.

### Why is effective rank flat?

Effective rank measures how many singular values of the weight matrices are significant (>1% of the largest). In the early training phase with a small model, this can appear flat for several reasons:

1. **Initialization already near full rank** -- Xavier/He initialization creates matrices that are approximately full rank from the start
2. **Small model dimensions** -- with `dim=64, heads=2, layers=2`, the weight matrices are small (64x64), so there aren't many singular values to begin with
3. **Short evaluation window** -- with `stepsPerCandidate: 20`, there's not enough training to meaningfully change the rank structure
4. **Sparse measurement** -- effective rank is measured every `metricsInterval` steps (10 in test config), so few data points exist

**Expected behavior at scale:** With larger models (256+ dim) trained for thousands of steps, effective rank typically starts high, dips during initial learning (as the model discovers low-rank structure), then gradually increases as it learns more complex representations.

### Why are architecture choices evenly distributed instead of converging?

The architecture distribution shows how many layers use each activation type. In the current implementation, **all layers in a candidate use the same activation**, so the distribution reflects the population composition, not per-layer choices.

Even distribution indicates:
1. **Insufficient selection pressure** -- With `stepsPerCandidate: 20` and `generations: 2` in the test config, there's not enough training per candidate to differentiate performance between activations
2. **Small fitness differences** -- All activations may perform similarly on this task/model size, so selection is nearly random
3. **High mutation rate** -- `mutationRate: 0.5` in test config means half of offspring get random activations, counteracting convergence

**To get convergence:** Increase `stepsPerCandidate` to 1000+, increase `generations` to 4+, and decrease `mutationRate` to 0.15-0.25. The current 50k run should show much better convergence.

---

## UI Feature Requests (Implemented)

1. Loss curve: activator switch event lines with hover tooltips (from/to activation, metadata)
2. Copy as JSON button for entire run metrics
3. Sample generations table showing all samples with checkpoint association
4. All panes expanded by default
5. Question mark help icon on every chart panel with full description
6. Activation switch log: click rows to see from/to fitness values
7. Full evolutionary metrics section: populations, tables, fitness charts, chronological evolution view
8. Phase changes/gelation visualization
9. Harmonic oscillator feedback visualization
10. Architecture convergence visualization
11. Fix blank batch size in symbio section

---

## Technical Notes

- Symbiogenesis package: `packages/symbiogenesis/src/`
- Metrics flow: trainer -> remote reporter -> `/api/ingest` -> Turso metrics table (45 columns)
- Expensive metrics collected every `metricsInterval` steps (sparse)
- CUSUM metrics collected every step (dense)
- MI profiles only when `trackMIProfiles: true`
- Current run config: `configs/symbio-test-search.json`
