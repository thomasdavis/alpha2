# Symbio Future Addendum: Crazier Activator Mutation Ideas (Alpha-Oriented)

Date: 2026-02-25
Status: Research notes
Related: `models/alpha/SYMBIO_FUTURE.md`

## Purpose

This addendum extends `SYMBIO_FUTURE.md` with more aggressive ideas for mutating activators in Symbio mode, while still fitting Alpha's constraints:

- Alpha remains the execution engine
- Symbio owns orchestration/evolution policy
- metrics must flow through Alpha's existing remote pipeline and DB/web surfaces

## 1. Next-Level Activator Mutation Ideas

## 1.1 Derivative-First Genomes

Mutate the derivative `f'(x)` instead of the activation `f(x)`, then integrate to recover `f(x)`.

Why it is useful:

- easier to enforce monotonicity / slope bounds
- easier to constrain Lipschitz behavior for stability
- mutation can target curvature directly (plateaus, spikes, asymmetry)

Genome examples:

- piecewise slope bins
- spline over derivative space
- rational derivative with bounded denominator

## 1.2 Histogram-Aware Local Surgery

Mutations should target the x-ranges the model actually uses (from FFN preactivation histograms), not the whole domain.

Example:

- if 95% of values are in `[-1.5, 0.7]`, mutate that band aggressively
- leave far tails mostly unchanged

Why it is useful:

- much more sample-efficient
- less likely to damage behavior in irrelevant regions
- naturally compatible with Alpha metrics and per-layer histograms

## 1.3 Spectral Activators (Fourier / Wavelet)

Represent the activation using a basis expansion over a bounded interval:

- Fourier coefficients (global smooth patterns)
- wavelet coefficients (localized bumps / kinks)

Mutation operators:

- add/remove frequency components
- phase shifts
- localized coefficient perturbation

Why it is useful:

- compact genome
- easy to regularize complexity
- can discover nonstandard bump-like activations

## 1.4 Latent Activator Manifold

Learn a latent space over activator shapes (known + discovered curves), then mutate in latent space and decode to a curve.

Why it is useful:

- smooth crossover between families (GELU-like, SiLU-like, etc.)
- avoids brittle hand-built program trees early
- enables semantic interpolation and novelty search in latent space

Practical path:

- bootstrap with standard activations + spline-discovered winners
- train a small shape autoencoder offline
- evolve latent codes online

## 1.5 Dual-Genome Activators (Train-Time vs Deploy-Time)

Store two representations for each activator candidate:

- expressive training phenotype (spline/program/rational)
- kernel-friendly deploy approximation (piecewise linear or low-order rational)

Fitness includes:

- training outcome
- runtime cost
- approximation error between rich and deploy forms

Why it is useful:

- evolution can be expressive without giving up Helios performance forever
- creates a path to future fused kernels

## 1.6 Conditional Activators (Context-Gated)

Activation is conditioned on lightweight context:

- layer index / depth
- training phase
- recent stats (variance, saturation rate)

Example:

- blend two curves with a scalar gate derived from layer stats

Why it is useful:

- keeps representation small while enabling adaptive behavior
- can replace hard switches with continuous adaptation

## 1.7 Stateful Activators (Short-Memory)

Activator uses an EMA of recent preactivation statistics per layer.

Examples of internal state:

- EMA mean
- EMA variance
- saturation ratio
- high-magnitude hit rate

Behavior idea:

- more linear under unstable gradients
- more saturating during stable late-stage convergence

Why it is useful:

- turns activation into a dynamic controller, not a fixed curve

## 1.8 Forward/Backward Co-Evolution

Evolve forward activation shape and backward surrogate derivative separately.

Concept:

- `f(x)` used in forward pass
- `g(x)` used as backward surrogate (instead of exact derivative)

Why it is useful:

- can improve gradient flow without sacrificing forward expressiveness
- useful when exact derivative is numerically hostile

Risk:

- easy to destabilize optimization; must enforce strict guardrails

## 1.9 Activation Grammar with Reusable Macros

Extend the activation DSL to support reusable subexpressions ("macros") evolved over time.

Examples:

- `soft_clip(x, k)`
- `swish_like(x, a)`
- `bump(x, c, w)`

Why it is useful:

- reduces genome bloat
- enables compositional reuse across layers/candidates
- makes evolved programs more interpretable than raw GP trees

## 1.10 Speciation by Shape Phenotype

Cluster activators by shape features and maintain species diversity in the population.

Phenotype features:

- monotonic vs non-monotonic
- inflection count
- tail growth rate
- slope at zero
- saturation asymmetry

Why it is useful:

- prevents early collapse to one family
- improves exploration even with a small activation pool

## 1.11 Adversarial Activation Mutation (Robustness Pressure)

Score activators under mild training perturbations:

- LR spike
- batch size change
- gradient noise injection
- temporary throughput pressure

Why it is useful:

- selects activators that are robust, not just lucky on a short window
- helps avoid winners that fail after switching or at scale

## 1.12 Regime-Scheduled Activation Genomes

Evolve a policy over time, not a single activator.

Genome encodes:

- activation sequence (or set of candidate states)
- trigger conditions (CUSUM, plateau, clip rate, step windows)
- switch warmup behavior

Why it is useful:

- directly models changing training regimes
- aligns with the CUSUM/periodic challenger concepts in `SYMBIO_FUTURE.md`

## 1.13 Co-Evolve Activation + FFN Width Ratio

Mutate activation choice and FFN expansion ratio together.

Why it is useful:

- some activators only perform well at certain widths
- activation and FFN capacity are coupled design choices

Cost:

- requires checkpoint morphing/projection infrastructure for efficient switching

## 1.14 Jacobian-Targeted Mutations

Use layer Jacobian proxies in fitness and mutation pressure.

Targets:

- gradient amplification
- gradient attenuation
- conditioning proxies

Why it is useful:

- activation search becomes stability-aware at a more fundamental level than `valLoss` alone

## 1.15 Activation Ecology (Niche Assignment Across Layers)

Encourage different layers (or groups of layers) to occupy different activation niches.

Examples:

- early layers: compressive
- middle layers: expressive
- late layers: stabilizing / near-linear

Why it is useful:

- avoids all layers converging to the same behavior when specialization is better
- fits naturally with per-layer activation vectors / learnable per-layer curves

## 2. Mutation Operators (Representation-Agnostic)

These are useful across spline/rational/DSL/latent representations.

- `localized_bump_inject`: add a small bump in an active x-region
- `tail_policy_swap`: swap tail behavior (linear, soft-clip, saturating)
- `symmetry_break`: mutate positive and negative sides independently
- `slope_cap_mutate`: adjust max slope / local derivative clamps
- `inflection_insert_remove`: add/remove a curvature change
- `semantic_crossover_center`: crossover only the central region shape
- `semantic_crossover_tails`: crossover only tail behavior
- `simplify_if_close`: compress a curve if a simpler approximation is near-equivalent
- `novelty_push`: mutate away from dominant species centroid
- `stability_repair`: post-mutation repair to satisfy slope/denominator constraints

## 3. Fitness Signals for Alpha (What to Log and Optimize)

Because Symbio metrics must go through Alpha's existing remote pipeline and into DB/web, mutation ideas should assume structured metrics from day one.

## 3.1 Core Fitness Terms

- `valLoss` (primary)
- `tokens_per_sec` (performance penalty/reporting)
- `clip_pct` / `clip_coef` (stability)
- `gradNorm` dynamics (stability)
- failure/NaN/abort rate

## 3.2 Activator-Specific Fitness Terms

- activation complexity cost (number of params, tree nodes, coefficients)
- approximation cost (if dual-genome)
- shape novelty score (distance from existing population/species)
- saturation rate in active regions
- Jacobian stability proxy
- switch cost / recovery cost after mutation or regime transition

## 3.3 Metrics/Event Payloads Worth Persisting

These should be designed for the existing remote pipeline and DB/web rendering.

Per candidate / generation:

- `symbio_mode`
- `search_mode` (`ffn-activation-search`)
- `generation`
- `candidate_id`
- `parent_ids`
- `species_id` (if speciation enabled)
- `mutation_ops[]`
- `activation_repr_type` (`discrete`, `spline`, `rational`, `dsl`, `latent`, etc.)
- `fitness_total`
- `fitness_terms` (structured)
- `winner_flag`

Phenotype summaries:

- curve descriptors (slope at 0, asymmetry, inflection count, tail policy)
- complexity score
- deploy approximation error (if dual-genome)

Runtime/stability summaries:

- `valLoss`
- `tokens_per_sec`
- `clip_pct`
- `gradNorm` summary
- switch events / warmup events

## 4. Practical "Crazy but Buildable" Combos

## 4.1 Best Near-Term Crazy Combo

`Histogram-aware spline genome` + `speciation` + `regime-scheduled switching`

Why:

- high novelty potential
- still interpretable
- aligns with CUSUM and challenger windows already in `SYMBIO_FUTURE.md`
- easy to visualize on web run pages

## 4.2 Best Performance-Conscious Combo (Alpha/Helios Friendly)

`Dual-genome activators` + `simplify_if_close` mutation + `deploy approximation penalty`

Why:

- explores rich activators during search
- preserves a path to fast approximate execution
- keeps "crazy" search compatible with production performance goals

## 4.3 Most Research-Aggressive Combo

`Forward/backward co-evolution` + `activation grammar macros` + `Jacobian-targeted fitness`

Why:

- likely to discover genuinely new training dynamics
- highest risk and hardest to debug
- should only be attempted after pipeline/DB/web Symbio telemetry is solid

## 5. Suggested Implementation Sequence (If You Want to Go Hard)

1. Add phenotype descriptors + mutation-op logging to Symbio metrics schema.
2. Implement histogram-aware mutations for the existing activation representation first.
3. Add speciation (phenotype clustering + diversity quotas).
4. Add regime-scheduled genomes (CUSUM/periodic challenger triggers become part of the genome).
5. Add dual-genome deploy approximations for performance-aware evolution.
6. Add spline or rational learned activators as a richer representation.
7. Only then test forward/backward co-evolution.

## 6. Web UI Ideas (to Match the New Metrics)

If Symbio metrics are in the DB, the run page can render:

- generation timeline with winners and switch events
- species distribution over generations
- mutation operator histogram (which mutations actually help)
- phenotype descriptor trends (slope at zero, asymmetry, complexity)
- activation shape viewer (for spline/rational/approx curves)

These become critical when activator mutation gets more complex than `gelu|relu|silu`.

## 7. Bottom Line

The biggest jump beyond the current plan is to stop treating activators as a flat enum and start treating them as evolvable objects with:

- a representation (shape/program/latent code)
- mutation operators
- phenotype descriptors
- deployment cost
- regime-switch behavior

That is where Symbio can become genuinely differentiated instead of just "activation search with extra steps."
