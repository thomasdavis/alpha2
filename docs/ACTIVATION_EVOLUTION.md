# Evolutionary Activation Search — Universal & KAN Spline

## Overview

Alpha's symbiogenesis system evolves activation functions through natural selection. Instead of picking one activation (gelu, silu, etc.) and hoping it's optimal, the system breeds a population of candidates — each using a different activation — trains them competitively, and lets the best survive.

The two newest activations, **Universal Approximator** and **KAN Spline**, are particularly interesting because they have **learnable parameters** that co-evolve with the model weights during training.

---

## The Activation Pool

Six activations compete:

| Activation | Prefix | Learnable Params | FFN Dim |
|-----------|--------|-----------------|---------|
| gelu | G | none | 4 * nEmbd |
| silu | S | none | 4 * nEmbd |
| relu | R | none | 4 * nEmbd |
| swiglu | Sw | none | (8/3) * nEmbd |
| **universal** | U | 2 vectors/layer | 4 * nEmbd |
| **kan_spline** | K | 5 vectors/layer | 4 * nEmbd |

---

## Universal Approximator

### What it is

A learnable per-channel gating mechanism that blends a nonlinear path (SiLU) with a linear identity path:

```
f(x) = silu(x) * gate + x * skip
```

Where `gate` and `skip` are **trainable vectors** of shape `[1, ffnDim]` that broadcast across the batch.

### Initialization

```
gate = [1.0, 1.0, ..., 1.0]   (ffnDim values)
skip = [0.0, 0.0, ..., 0.0]   (ffnDim values)
```

At initialization, `f(x) = silu(x) * 1 + x * 0 = silu(x)`. It starts as plain SiLU, then the model learns per-channel how much nonlinearity vs identity it wants.

### Forward pass (per transformer block)

```
h = x @ fc1                    # project to ffnDim
h_silu = silu(h)               # nonlinear path
gated = h_silu * gate          # scale nonlinearity per channel
skipped = h * skip             # identity/residual per channel
h_act = gated + skipped        # combine
out = h_act @ fc2              # project back to nEmbd
```

### What it can learn

- **gate=1, skip=0** → pure SiLU (starting point)
- **gate=0, skip=1** → pure linear (identity)
- **gate=0.7, skip=0.3** → blend of nonlinear + linear per channel
- Different channels can learn different blends — some channels might be mostly linear while others are aggressively nonlinear

### Why it's interesting for evolution

During evolutionary search, each Universal candidate starts identical (as SiLU). But over the 30 training steps of evaluation, the gate/skip vectors drift in different directions depending on the random seed and data ordering. When a Universal candidate mutates into a child (still Universal but different seed), the child starts fresh and may find a completely different gate/skip landscape. This means Universal candidates have high intra-activation diversity — two Universal candidates can behave very differently.

---

## KAN Spline

### What it is

A 5-basis function approximator inspired by Kolmogorov-Arnold Networks. Instead of using B-splines (which would require a custom autograd op), it uses existing activation functions as basis functions:

```
f(x) = c0*silu(x) + c1*relu(x) + c2*gelu(x) + c3*x + c4*x^2
```

Each coefficient `c0`-`c4` is a **trainable vector** of shape `[1, ffnDim]`.

### Initialization

```
c0 = 0.5    (silu weight — half contribution)
c1 = 0.0    (relu weight — off)
c2 = 0.5    (gelu weight — half contribution)
c3 = 0.0    (identity weight — off)
c4 = 0.0    (quadratic weight — off)
```

At initialization, `f(x) = 0.5*silu(x) + 0.5*gelu(x)`. A smooth blend of two well-known activations.

### Forward pass (per transformer block)

```
h = x @ fc1                      # project to ffnDim
h_silu = silu(h) * c0            # basis 1: smooth nonlinear
h_relu = relu(h) * c1            # basis 2: piecewise linear
h_gelu = gelu(h) * c2            # basis 3: smooth nonlinear
h_id   = h * c3                  # basis 4: identity
h_sq   = (h * h) * c4            # basis 5: quadratic
h_act  = h_silu + h_relu + h_gelu + h_id + h_sq
out    = h_act @ fc2             # project back to nEmbd
```

### What it can learn

- **c0=1, rest=0** → pure SiLU
- **c2=1, rest=0** → pure GELU
- **c1=1, rest=0** → pure ReLU
- **c3=1, rest=0** → linear (no activation)
- **c0=0.3, c2=0.3, c3=0.2, c4=0.2** → a custom activation that doesn't exist in any textbook
- Each channel independently learns its own activation shape — channel 42 might be mostly ReLU while channel 100 is a quadratic-GELU blend

### Why it's interesting for evolution

KAN Spline is the most expressive activation in the pool. With 5 degrees of freedom per channel, it can approximate any of the other activations in the pool (gelu, silu, relu are all subsets). This gives it a theoretical advantage — but also a risk: more parameters means more potential for overfitting on short evaluation windows. The evolutionary pressure tests whether that expressiveness actually helps on real data.

---

## How Evolution Works

### Candidate Naming

Each candidate gets a human-readable name based on its activation and lineage:

- **Gen 0**: `{Prefix}-{GreekLetter}` — e.g., `G-Alpha` (first gelu), `S-Beta` (first silu), `U-Gamma` (first universal), `K-Delta` (first kan_spline)
- **Gen 1+**: `{Prefix}-{ParentSuffix}.{Gen}` — e.g., `G-Alpha.1` (gelu child of Alpha), `K-Delta.2` (kan_spline child of Delta, gen 2)

Every candidate tracks its full lineage: `[parentId, grandparentId, ...]` for tree visualization.

### Generation Lifecycle

```
┌─────────────────────────────────────┐
│  GENERATION 0                       │
│                                     │
│  G-Alpha  → train 30 steps → loss   │
│  S-Beta   → train 30 steps → loss   │
│  R-Gamma  → train 30 steps → loss   │
│  Sw-Delta → train 30 steps → loss   │
│  U-Epsilon→ train 30 steps → loss   │
│  K-Zeta   → train 30 steps → loss   │
│                                     │
│  RANK by validation loss             │
│  SELECT top 3 (topk, k=pop/2)       │
│                                     │
│  BREED 6 offspring:                  │
│    3 mutations (swap activation)     │
│    3 clones (keep activation)        │
│                                     │
└───────────┬─────────────────────────┘
            │
            ▼
┌─────────────────────────────────────┐
│  GENERATION 1                       │
│                                     │
│  S-Beta.1   → (clone of Beta)       │
│  K-Zeta.1   → (mutation: now gelu)  │
│  U-Epsilon.1→ (clone of Epsilon)    │
│  S-Beta.1b  → (mutation: now relu)  │
│  K-Zeta.1b  → (clone of Zeta)      │
│  U-Epsilon.1b→(mutation: now swiglu)│
│                                     │
│  ... train, rank, select, breed ... │
└───────────┬─────────────────────────┘
            │
            ▼
        (repeat for 12 generations)
            │
            ▼
┌─────────────────────────────────────┐
│  WINNER SELECTED                    │
│  Best valLoss across all gens       │
│  Train remaining ~47,840 steps      │
└─────────────────────────────────────┘
```

### What happens at each candidate switch

When the orchestrator says "next candidate":

1. **Model is reinitialized** with the new activation's architecture (different ffnDim for swiglu, different learnable params for universal/kan_spline)
2. **Fresh random weights** — seeded by `trainSeed + currentStep` so each candidate gets a deterministic but different initialization
3. **Optimizer state reset** — Adam momentum/variance cleared so the new candidate starts clean
4. **Forward cache cleared** — position embeddings and attention masks recalculated

This means each candidate is evaluated independently. They don't inherit weights from the previous candidate — it's a fair comparison from scratch.

### Mutation mechanics

When a parent is selected for mutation:
- A random activation is chosen from the pool **excluding** the parent's activation
- New candidate gets the new activation, parent's ID in lineage, and a fresh name
- If Universal mutates to KAN Spline: the 2 learnable vectors (gate, skip) are replaced by 5 learnable vectors (c0-c4). Completely different parameter space
- If KAN Spline mutates to GELU: all 5 learnable vectors are dropped. Pure fixed activation

### Diversity bonus

The system tracks **architecture diversity** — what fraction of the population uses each activation type. A diversity bonus (default 5%) rewards candidates from under-represented activations:

```
effective_fitness = raw_fitness + diversityBonus * (1 - frequency_of_this_activation)
```

**Cosine decay** reduces this bonus over generations:

```
bonus_at_gen_g = diversityBonus * cos(π * g / (2 * totalGenerations))
```

Early generations: full diversity bonus → explore widely. Late generations: near-zero bonus → converge on quality.

---

## Selective Pressure

Selective pressure is the core evolutionary force that drives the activation search toward better architectures. It works through three interlocking mechanisms.

### 1. Evaluation pressure (per-candidate)

Each candidate is trained for exactly `stepsPerCandidate` steps (30 in the current run). During those steps, the system records:

- **bestLoss**: The minimum training loss seen across all 30 steps
- **bestValLoss**: The minimum validation loss (if eval interval fires)
- **fitnessScore**: A composite metric combining loss improvement rate, gradient health, and throughput

The candidate has exactly 30 steps to prove itself. Activations that converge faster in the early steps have a measurable advantage — this is intentional. In practice, the LR is still in warmup during gen 0 (lr ~5e-6 at step 30), so the signal is noisy. But across 6 candidates, relative ordering is still meaningful: an activation that drops from 8.37 to 8.10 in 30 steps is clearly better than one that only reaches 8.30.

**With 30 steps, what can each activation demonstrate?**
- **Fixed activations** (gelu, silu, relu, swiglu): The activation shape is frozen. All the signal comes from how well that fixed nonlinearity interacts with the random weight initialization and the data distribution.
- **Universal approximator**: Gate and skip vectors receive gradient updates for 30 steps. At lr ~5e-6, the gate might drift from 1.0 to ~0.9997 — barely perceptible. The Universal activation is essentially SiLU for all practical purposes in gen 0.
- **KAN spline**: Five coefficient vectors receive gradients. Same story — 30 steps at warmup LR means the coefficients barely move from their (0.5, 0, 0.5, 0, 0) initialization. KAN spline is essentially (silu+gelu)/2 in gen 0.

This means **in the early generations, Universal and KAN Spline have no real advantage over their starting-point equivalents**. Their power emerges only if they survive to later generations where the LR is higher and they accumulate more training steps across the lineage.

### 2. Selection pressure (per-generation)

After all candidates in a generation are evaluated, the population is ranked by `valLoss` (or `bestLoss` if no validation was triggered). Then:

```
parents = top_k(ranked, k = ceil(populationSize / 2))
```

With `populationSize=6` and `topk` selection, the **top 3 candidates survive**. The bottom 3 are eliminated. This is a 50% kill rate per generation — aggressive selective pressure.

The surviving 3 parents produce 6 offspring:
- **3 mutations** (50% mutation rate): Parent's activation is swapped to a random different one
- **3 clones**: Parent's activation is preserved

This creates an interesting dynamic:

```
Gen 0:  G-Alpha(gelu)  S-Beta(silu)  R-Gamma(relu)  Sw-Delta(swiglu)  U-Epsilon(universal)  K-Zeta(kan_spline)
        ───rank───
        Top 3: [silu, gelu, swiglu]  (hypothetical)
        Kill:  [relu, universal, kan_spline]

Gen 1:  S-Beta.1(silu)     ← clone of Beta
        G-Alpha.1(gelu)    ← clone of Alpha
        Sw-Delta.1(swiglu) ← clone of Delta
        S-Beta.1b(relu)    ← mutation of Beta → now relu
        G-Alpha.1b(kan_spline) ← mutation of Alpha → now kan_spline
        Sw-Delta.1b(universal) ← mutation of Delta → now universal
```

Notice: even though relu, universal, and kan_spline were eliminated, mutations can **resurrect** them by assigning them to offspring of winning parents. The activation dies but gets another chance with a different random seed.

### 3. Diversity pressure (cross-generation)

The diversity bonus injects counter-selective pressure that prevents premature convergence:

```
adjusted_score = raw_score - diversityBonus * (1 - activation_frequency) * decay(gen)
```

(Note: for valLoss ranking where lower is better, the bonus is subtracted to improve the candidate's ranking.)

**Cosine decay schedule:**
```
Gen 0:  decay = cos(0)     = 1.00  → full 5% diversity bonus
Gen 3:  decay = cos(π/8)   = 0.92  → 4.6% bonus
Gen 6:  decay = cos(π/4)   = 0.71  → 3.5% bonus
Gen 9:  decay = cos(3π/8)  = 0.38  → 1.9% bonus
Gen 11: decay = cos(11π/24)= 0.13  → 0.6% bonus
Gen 12: decay = cos(π/2)   = 0.00  → no bonus, pure meritocracy
```

**Effect**: In gen 0-3, a rare activation (say, kan_spline appearing in only 1/6 of the population) gets a ~4-5% fitness boost. This can save it from elimination even if its raw loss is slightly worse. By gen 9+, the bonus is nearly gone and only raw performance matters.

### Interaction between pressures

The three pressures interact to produce an explore-then-exploit dynamic:

1. **Gen 0-3** (high diversity bonus): All 6 activations get a fair shot. Even weaker activations survive through diversity protection. The population stays heterogeneous.

2. **Gen 4-7** (declining bonus): The consistently worst activations start getting eliminated despite the bonus. A pattern emerges — maybe 2-3 activation types dominate. Mutations still inject variety.

3. **Gen 8-11** (minimal bonus): Pure performance-driven selection. If one activation type is genuinely better, it colonizes most of the population. The tree chart shows long lineages from a common ancestor.

4. **Gen 12** (search complete): The single best candidate across all 12 generations is selected as the winner. The remaining ~47,840 training steps all use this winning activation.

---

## Expected Outcomes for Current Run

**Run**: `super_chat_20260225143238_sa7h` — 17.4M params, L4 GPU, 50K iters

### Timeline

| Time | Step | Event |
|------|------|-------|
| 0:00 | 1 | G-Alpha (gelu) begins gen 0 |
| 0:25 | 30 | G-Alpha done, S-Beta (silu) starts |
| 0:50 | 60 | S-Beta done, R-Gamma (relu) starts |
| 1:15 | 90 | R-Gamma done, Sw-Delta (swiglu) starts |
| 1:40 | 120 | Sw-Delta done, U-Epsilon (universal) starts |
| 2:05 | 150 | U-Epsilon done, K-Zeta (kan_spline) starts |
| ~2:30 | 180 | **Gen 0 complete** — rank, select top 3, breed gen 1 |
| ~5:00 | 360 | **Gen 1 complete** — first evolutionary offspring evaluated |
| ~7:30 | 540 | Gen 2 complete |
| ~15:00 | 1080 | Gen 5 — diversity bonus declining, dominance patterns emerge |
| ~25:00 | 1800 | Gen 9 — near-zero diversity bonus, convergence |
| ~30:00 | 2160 | **Gen 12 complete** — winner selected |
| ~30:00-11:00hrs | 2160-50000 | Winner trains for remaining ~47,840 steps |

### Predictions

**Most likely winner**: **SwiGLU** or **SiLU**

Reasoning:
- At 30 steps per candidate during warmup (lr ~5-7e-6), the evaluation window is too short for Universal and KAN Spline's learnable parameters to differentiate themselves from their fixed-activation starting points
- SwiGLU has a structural advantage: the gated architecture (up_proj * gate_proj) provides more expressive capacity even without learnable activation parameters
- SiLU is a strong baseline that consistently performs well in transformer architectures
- GELU is close to SiLU but historically slightly worse for small models

**Most likely to be eliminated first**: **ReLU**

Reasoning:
- ReLU's hard zero below x=0 causes "dead neuron" problems, especially at low learning rates during warmup where neurons that start negative may never recover
- All other activations are smooth (differentiable everywhere), giving them better gradient flow

**Wildcard**: **KAN Spline**

Reasoning:
- Starts as (silu+gelu)/2, which is actually a reasonable activation
- Even without learning (30 steps is too few), the averaged activation might outperform either silu or gelu alone
- If it survives to later generations where LR is higher (gen 4+ at lr ~2e-5), the coefficients could learn something genuinely novel

**Expected tree shape**:
- Gen 0-3: Bushy, diverse — all 6 activations represented
- Gen 4-7: Narrowing — 2-3 dominant lineages with long branches
- Gen 8-12: Near-monoculture — one activation type dominates, with occasional mutation branches that quickly die off
- Final tree depth: 12 (one branch per generation for the winning lineage)

**Expected architecture_diversity metric**:
- Gen 0: 1.0 (perfect — one of each)
- Gen 3: ~0.7 (some types eliminated)
- Gen 6: ~0.4 (2-3 types remaining)
- Gen 9: ~0.2 (near monoculture)
- Gen 12: ~0.17-0.3 (dominated by winner, mutations provide noise)

---

## What to watch on the dashboard

### Evolutionary Tree Chart
- Color-coded by activation type: gelu (green), silu (orange), relu (red), swiglu (purple), universal (pink), kan_spline (cyan)
- Click nodes to see fitness, loss, generation
- Watch lineages form — if Universal keeps producing winning children, its branch grows deep

### Activation Switch Log
- Shows every candidate transition with from/to activation
- Click "tree" button to navigate to that candidate's node
- Look for patterns: does the system keep switching back to the same activation?

### Key metrics
- `symbio_candidate_activation` — which activation is currently training
- `architecture_diversity` — how varied the current population is (1.0 = perfectly diverse, 0.17 = monoculture)
- `symbio_generation` — current evolutionary generation

---

## Current run config

```json
{
  "populationSize": 6,
  "generations": 12,
  "stepsPerCandidate": 30,
  "mutationRate": 0.5,
  "diversityBonus": 0.05,
  "diversityDecay": "cosine"
}
```

- **6 candidates per generation** (one per activation type in gen 0)
- **30 steps per candidate** (~25 seconds each)
- **12 generations** of evolution (~30 minutes total)
- **50% mutation rate**: half of offspring get a new activation, half clone parent's
- **Search budget**: 2,160 steps out of 50,000 total → winner trains for ~47,840 remaining steps

### Evolutionary Tree Chart
- Color-coded by activation type: gelu (green), silu (orange), relu (red), swiglu (purple), universal (pink), kan_spline (cyan)
- Click nodes to see fitness, loss, generation
- Watch lineages form — if Universal keeps producing winning children, its branch grows deep

### Activation Switch Log
- Shows every candidate transition with from/to activation
- Click "tree" button to navigate to that candidate's node
- Look for patterns: does the system keep switching back to the same activation?

### Key metrics
- `symbio_candidate_activation` — which activation is currently training
- `architecture_diversity` — how varied the current population is (1.0 = perfectly diverse, 0.17 = monoculture)
- `symbio_generation` — current evolutionary generation

---

## Current run config

```json
{
  "populationSize": 6,
  "generations": 12,
  "stepsPerCandidate": 30,
  "mutationRate": 0.5,
  "diversityBonus": 0.05,
  "diversityDecay": "cosine"
}
```

- **6 candidates per generation** (one per activation type in gen 0)
- **30 steps per candidate** (~25 seconds each)
- **12 generations** of evolution (~30 minutes total)
- **50% mutation rate**: half of offspring get a new activation, half clone parent's
- **Search budget**: 2,160 steps out of 50,000 total → winner trains for ~47,840 remaining steps
