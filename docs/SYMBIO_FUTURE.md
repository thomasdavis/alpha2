# Symbio Future: Beyond v1

Date: 2026-02-25
Status: Research notes — not scheduled, not committed
Depends on: `PRD_SYMBIOGENESIS_ADAPTER_LAYER_20260225.md` (v1) shipping first

---

## 1. Continuous Selective Pressure

### Problem

v1's evolutionary search is a one-shot tournament: run each activation for N steps early in training, pick a winner, use it forever. What's optimal at step 2000 may not be optimal at step 40000. Loss landscape curvature changes as the model trains — gated activations (SwiGLU) may dominate early while simpler activations (GELU) may converge better late.

### 1a. CUSUM-Triggered Re-Search

The CUSUM monitors from v1 are already watching gradNorm, clip_pct, tokens_per_sec, and valLoss. When one fires — loss plateau, gradient instability, throughput collapse — that's a natural point to re-evaluate the activation choice.

```
step 0–4000:      initial evolutionary search → winner = silu
step 4000–18000:  training with silu, CUSUM monitors running
step 18000:       CUSUM fires on valLoss (plateau detected)
step 18000–22000: re-search from current checkpoint
                  fork → evaluate each activation × 1000 steps
                  maybe gelu now wins at this loss landscape
step 22000+:      continue with new winner (or confirm silu)
```

Reactive — only re-evaluates when something goes wrong. The CUSUM alert is the selective pressure signal.

New config fields:
```typescript
readonly cusumTriggersResearch: boolean;  // default: true
readonly researchBudgetSteps: number;     // default: 1000 per activation
readonly switchWarmup: number;            // default: 200
```

### 1b. Periodic Challenger Windows

At fixed intervals, run short challenger evaluations regardless of CUSUM state:

```
every challengerInterval steps:
  save checkpoint
  for each activation in pool (excluding current):
    fork from checkpoint, train challengerBudget steps, record valLoss
  if any challenger beats current by > challengerThreshold:
    switch activation (with warmup)
  else:
    continue
```

Proactive — applies pressure on a schedule. Catches slow degradation that CUSUM might miss (gradual rather than sudden regime shift).

New config fields:
```typescript
readonly challengerInterval: number;      // default: 10000
readonly challengerBudget: number;        // default: 500
readonly challengerThreshold: number;     // default: 0.02 (2% valLoss improvement)
```

### 1c. Diversity-Aware Population Maintenance

Inspired by `symbiogenesis/population.py:adapt_size()` and `config.py:diversity_bonus`. Maintain a lightweight shadow population score throughout training:

```
shadow scores: { gelu: 0.8, silu: 0.95, relu: 0.6, swiglu: 0.92 }

every metricsInterval:
  current activation's score = actual fitness
  other scores decay toward neutral (staleness)
  population_entropy = H(softmax(scores))

  if entropy < convergence_threshold:
    diversity_bonus activates → adds bonus to underrepresented activations
    if bonus pushes a challenger above current → trigger re-evaluation

every challengerInterval:
  mandatory re-evaluation for all activations
  update shadow scores with real data
```

The diversity bonus decays over training (linear or cosine, from `symbiogenesis/config.py:diversity_decay`) — early training explores, late training exploits. But it never drops to zero.

### 1d. Combined Approach (Recommended)

Use 1a + 1b together:
- Periodic challengers every 10000 steps (catches slow drift)
- CUSUM-triggered re-search on regime shifts (catches sudden changes)
- Either mechanism can trigger a switch, both log to DB and dashboard

---

## 2. Learnable Activation Functions

### Problem

Discrete search over {gelu, silu, relu, swiglu} is limited by human imagination. Every activation function is just a curve ℝ → ℝ. There's no reason the optimal curve for a specific model size, dataset, and training stage should be one that a human named.

### 2a. Learnable B-Spline Activations

Define the activation as a cubic B-spline with trainable control points:

```
activation(x) = Σ cᵢ · Bᵢ(x)
```

where `cᵢ` are learnable parameters (per layer) and `Bᵢ` are cubic B-spline basis functions over a fixed knot grid.

Implementation:
- 16 control points per layer spanning [-6, 6]
- Linear extrapolation outside the range
- Initialize control points to match SiLU/GELU shape
- Control points are regular parameters — AdamW updates, weight decay
- Fully differentiable: gradients flow to both x and cᵢ

Parameter cost: 16 floats per layer × 6 layers = 96 parameters. Negligible vs millions of weight params.

What this enables:
- At step 5000 the curve might look like SiLU
- At step 30000 it might look like nothing anyone's ever named
- Different layers discover different shapes
- The curve co-evolves with the weights continuously — no discrete search needed

Alpha implementation:
```typescript
// New autograd op
export function splineActivation(ctx: Ctx, x: Variable, controlPoints: Variable): Variable {
  // Forward: cubic B-spline interpolation
  // Backward: gradients w.r.t. x AND w.r.t. controlPoints
}

// In gpt.ts SwiGLU path:
// Before: silu(x @ W_gate) * (x @ W_up)
// After:  spline(x @ W_gate, controlPoints[layer]) * (x @ W_up)

// ModelConfig:
readonly ffnActivation?: "gelu" | "silu" | "relu" | "swiglu" | "learned";
```

This is related to **KAN (Kolmogorov-Arnold Networks)** — learnable splines replacing fixed activations — but applied selectively to just the FFN activation, not the entire architecture.

### 2b. Rational Function Activations (Padé Approximation)

Represent the activation as a ratio of two learnable polynomials:

```
activation(x) = (a₀ + a₁x + a₂x² + a₃x³) / (1 + |b₁x| + |b₂x²|)
```

- 6 learnable parameters per layer (aᵢ, bᵢ)
- Absolute values in denominator guarantee no division by zero
- Degree (3,2) can exactly represent ReLU, closely approximate GELU and SiLU
- Can also represent functions with asymmetric saturation, bumps, or inflection points that no standard activation has

This is **PAU (Padé Activation Units)** from the literature. Proven to match or beat all fixed activations across benchmarks.

The denominator enables shapes that element-wise activations can't express — e.g., an activation that saturates hard on the negative side but is linear positive with a slight bump at x=1.

### 2c. Micro-MLP Activations

Most general approach: the activation is a tiny neural network.

```
activation(x) = MLP(x; θ)    where θ is a 2-layer MLP, width 8
```

- 25 parameters per layer (8 + 8 + 8 + 1)
- Universal approximator for any continuous function on a compact domain
- Can learn discontinuities, oscillations, multiple bumps
- Initialize to approximate SiLU, then let it evolve
- Implementation is just matmul + relu + matmul — ops Alpha already has

Most expressive but hardest to interpret. The spline and rational approaches produce curves you can plot and understand. The micro-MLP is a black box.

### 2d. Recommended Path

**Start with 2a (B-spline)**. Reasons:
- Simplest to implement (one new autograd op)
- Easiest to visualize (plot control points at each checkpoint)
- Lowest risk (initialize from known-good shape)
- Natural regularization (weight decay on control points biases toward simpler curves)
- Dashboard visualization is immediately compelling: watch the activation curve morph over training

Then if splines hit expressiveness limits, graduate to 2b (rational). Only go to 2c (micro-MLP) if there's evidence that smooth functions aren't enough.

### Integration with v1 Symbio

The discrete evolutionary search from v1 becomes the **initialization strategy**:
1. Evolutionary search picks the best starting activation (e.g., SiLU)
2. Initialize spline control points to match that shape
3. Training continues with learnable spline — the curve refines continuously
4. CUSUM monitors track control point drift as a regime-shift signal
5. Weight entropy and effective rank apply to control points too

---

## 3. Per-Layer Activation Vectors

### Problem

v1 uses a single global activation across all transformer blocks. But different layers may want different nonlinearities — early layers doing feature extraction may prefer a different shape than late layers doing classification.

### 3a. Per-Block Activation Selection

Each transformer block gets its own activation choice from the pool:

```typescript
readonly ffnActivations?: readonly ("gelu" | "silu" | "relu" | "swiglu")[];
// e.g., ["swiglu", "swiglu", "silu", "silu", "gelu", "gelu"] for L6
```

The evolutionary search becomes combinatorial: 4 activations × 6 layers = 4096 possible configurations. Needs smarter search than exhaustive (genetic algorithm, simulated annealing, or Bayesian optimization).

Checkpoint compatibility: checkpoint must store the full activation vector. Loading a checkpoint with different layer count requires truncation or padding.

### 3b. Per-Block Learnable Activations

Combine §2a with §3a: each block gets its own learnable spline. No search needed — each layer's curve evolves independently via backprop.

```
Block 0: activation = spline(x; c₀)   — might evolve toward GELU-like
Block 3: activation = spline(x; c₃)   — might evolve toward SiLU-like
Block 5: activation = spline(x; c₅)   — might evolve toward something novel
```

16 control points × 6 layers = 96 learnable activation parameters total.

This is the endgame for activation evolution: every layer discovers its own optimal nonlinearity, continuously, without any discrete search. The symbiogenesis Phase 10 vision (per-layer activation evolution) implemented not through population dynamics but through gradient descent.

---

## 4. Compositional Activation Evolution

### Problem

The activation pool is a flat list of human-designed functions. What if the optimal activation is a composition or blend of known activations?

### 4a. Weighted Blending

```
activation(x) = w₁·gelu(x) + w₂·silu(x) + w₃·relu(x)
where w₁ + w₂ + w₃ = 1 (softmax)
```

The weights are learnable per layer. Gradient descent finds the optimal blend. This is a poor man's version of the spline approach but easier to interpret: "layer 3 is 70% SiLU + 30% GELU".

### 4b. Sequential Composition

Inspired by symbiogenesis fusion strategies — sequential fusion concatenates layers, so sequential activation composition nests functions:

```
activation(x) = silu(gelu(x))           — sequential compose
activation(x) = 0.7·silu(x) + 0.3·x    — residual activation
activation(x) = silu(x) · sigmoid(x)    — gated activation
```

The evolutionary search mutates by:
- Swapping leaf activations (gelu → silu)
- Adding/removing composition depth
- Adding/removing residual connections
- Changing blend weights

This is a **tree-structured search** over activation function programs. More expressive than a flat pool, less black-box than a learned spline.

### 4c. DSL for Activation Functions

Define a small DSL for expressing activation functions as composable primitives:

```
primitives: x, sigmoid(x), tanh(x), relu(x), exp(x), abs(x), x²
operators: +, *, compose, gate(a,b) = a * sigmoid(b)
```

A genetic programming search evolves activation function programs. Each candidate is a small expression tree. Fitness is training loss after N steps.

```
generation 0: [relu(x), sigmoid(x)*x, tanh(x)]
generation 1: [sigmoid(x)*x, sigmoid(x)*x + 0.1*x, tanh(sigmoid(x)*x)]
generation 2: [sigmoid(x)*x + 0.1*x, x*sigmoid(1.2*x), ...]
```

The system could discover SwiGLU-like gating (`x * sigmoid(x)` = SiLU) from primitives without ever being told about it. Or discover something entirely new.

---

## 5. Checkpoint Morphing and Weight Transfer

### Problem

Switching activation functions mid-training requires restarting from scratch (different param shapes for SwiGLU vs GELU). v1 treats this as a hard boundary. Checkpoint morphing would enable seamless transitions.

### 5a. Shape-Preserving Activation Switch

For activations that don't change parameter shapes (GELU ↔ SiLU ↔ ReLU), switching mid-training just means changing the nonlinearity. Weights carry over directly. Add a warmup period after the switch to let the optimizer state adapt.

Already possible in v1 with manual checkpoint editing. Should be automated via the challenger mechanism (§1).

### 5b. Shape-Changing Activation Switch (GELU ↔ SwiGLU)

SwiGLU has 3 weight matrices vs GELU's 2. Switching requires weight projection:

GELU → SwiGLU:
```
W_gate = W_fc          (copy)
W_up   = W_fc          (copy or random init)
W_proj = W_proj        (copy, but dimensions may differ if ffnDim changes)
```

SwiGLU → GELU:
```
W_fc   = (W_gate + W_up) / 2    (average, or just pick one)
W_proj = W_proj                  (truncate/pad if ffnDim differs)
```

Inspired by `symbiogenesis/fusion.py:weight_transfer_mode`:
- `"copy"`: only transfer exact-shape matches
- `"projection"`: truncation/zero-padding for dimension mismatches
- `"partial"`: copy what fits, init the rest fresh

### 5c. Progressive Morphing

Instead of an abrupt switch, gradually morph from one activation to another:

```
activation(x; t) = (1-t) * gelu(x) + t * silu(x)
```

where `t` goes from 0 → 1 over `switchWarmup` steps. The weights have time to adapt to the changing nonlinearity.

---

## 6. Thermodynamic Training Dynamics

### Problem

v1 tracks metrics (weight entropy, free energy, population entropy) but doesn't use them to drive decisions beyond CUSUM alerts. The symbiogenesis repo's future directions document (`Future_Research_Directions.md`) outlines a thermodynamic framework where these metrics become active control signals.

### 6a. Temperature-Controlled Exploration

Map learning rate to temperature. Population entropy (or weight entropy) determines whether to explore or exploit:

```
if population_entropy < low_threshold:
  increase lr (raise temperature, more exploration)
  increase diversity_bonus
if population_entropy > high_threshold:
  decrease lr (lower temperature, more exploitation)
  decrease diversity_bonus
```

This is simulated annealing applied to the training dynamics, with the entropy metric as the thermostat.

### 6b. Free Energy as Training Objective

Instead of pure loss minimization, directly minimize free energy:

```
F = loss + β · weight_entropy
```

The `β` parameter controls the complexity-accuracy tradeoff. v1 computes free energy as a metric. This would make it the actual training objective by adding the entropy term to the loss function.

Effect: the model is incentivized to find simpler weight configurations that achieve the same loss. This is a form of regularization that's theoretically grounded in the Free Energy Principle (Friston, 2010) and related to variational inference.

### 6c. Heat Capacity Monitoring

Heat capacity `C = dE/dT` measures how sensitive average fitness is to changes in exploration pressure. Peaks in heat capacity signal phase transitions — moments where the model's learning dynamics are fundamentally changing.

In practice: track `d(loss) / d(lr)` over a sliding window. Large values indicate the model is at a critical point where small hyperparameter changes have outsized effects. These are the moments where adaptive decisions (activation switch, batch size change, LR schedule adjustment) have the most impact.

---

## 7. Mutual Information Guided Architecture

### Problem

v1 collects MI profiles (I(X;T), I(T;Y), compression ratio) as metrics. The information bottleneck theory predicts that these metrics can guide architectural decisions.

### 7a. Information Bottleneck Regularization

Add an IB regularizer to the loss function (from `symbiogenesis/mi_estimator.py:ib_regularizer`):

```
total_loss = task_loss + β_ib · Σ var(activations_per_layer)
```

The variance proxy approximates I(X;T) and encourages the model to compress information through bottleneck layers. This is the **Variational Information Bottleneck (VIB)** from Alemi et al. 2017, adapted for transformers.

### 7b. MI-Guided Layer Width

If I(T;Y) at a particular layer is much lower than I(X;T), that layer is retaining too much irrelevant information. Signal to the evolutionary search: this layer could be narrower.

If I(T;Y) ≈ I(X;T), the layer is passing everything through without compression. Signal: this layer's activation might be too linear (try something with more saturation).

These signals can inform the challenger evaluation: when re-evaluating activations, weight candidates by their MI profile improvement.

### 7c. Information Plane Trajectory Analysis

Track the (I(X;T), I(T;Y)) trajectory over training on the dashboard. The trajectory reveals:
- **Fitting phase**: I(T;Y) increases (learning the task)
- **Compression phase**: I(X;T) decreases (discarding irrelevant info)
- **Overfitting**: I(X;T) increases again without I(T;Y) improving

Detecting the transition from fitting to compression (or the onset of overfitting) via the information plane is a potential CUSUM signal for triggering LR reduction or early stopping.

---

## 8. Multi-GPU Evolutionary Search

### Problem

v1 search is sequential: evaluate one activation at a time. With multiple GPUs (see `scale.md` roadmap), candidates could evaluate in parallel.

### 8a. Parallel Candidate Evaluation

On a multi-GPU setup, run each candidate on a separate GPU simultaneously:

```
GPU 0: train with gelu    for 1000 steps
GPU 1: train with silu    for 1000 steps    (in parallel)
GPU 2: train with relu    for 1000 steps    (in parallel)
GPU 3: train with swiglu  for 1000 steps    (in parallel)
→ compare results, pick winner
→ all GPUs resume with winner for full training
```

Wall-clock cost of search drops from `N × stepsPerCandidate` to `1 × stepsPerCandidate`.

### 8b. Asynchronous Population Evolution

Maintain a persistent population across GPUs. Each GPU trains a different candidate continuously. Periodically synchronize, exchange best performers, mutate underperformers. This is **population-based training (PBT)** — but evolving activation functions instead of hyperparameters.

---

## 9. Dashboard Visualization Additions

These visualizations would be needed for the features above:

### 9a. Activation Shape Viewer (for §2)

Interactive chart showing the learned activation curve at each layer, at each checkpoint. Slider to scrub through training steps and watch the curve morph. Overlay the standard activation shapes (GELU, SiLU, ReLU) as reference lines.

### 9b. Activation Switch Timeline (for §1)

Horizontal timeline showing which activation was active at each step range, with markers for challenger evaluations and switch events. Color-coded by activation. Shows the "evolutionary history" of the run's activation function.

### 9c. Per-Layer Activation Heatmap (for §3)

Matrix visualization: rows = layers, columns = steps, color = activation type (or spline distance from standard shapes). Shows how per-layer activations diverge over training.

### 9d. Information Plane Animation (for §7)

Animated scatter plot of (I(X;T), I(T;Y)) over training, colored by step. Shows fitting → compression → overfitting phases. One point per layer, trails showing trajectory.

---

## Priority Ordering

If building beyond v1, the recommended sequence:

1. **§1d Combined selective pressure** (CUSUM + periodic challengers) — highest impact for lowest effort, uses v1 infrastructure directly
2. **§2a Learnable B-spline activations** — transforms discrete search into continuous optimization, unlocks genuinely novel activations
3. **§3b Per-block learnable activations** — natural extension of 2a, each layer finds its own curve
4. **§5a Shape-preserving activation switch** — enables §1 to actually switch activations mid-training
5. **§6a Temperature-controlled exploration** — makes the thermodynamic metrics from v1 actively useful
6. **§9a Activation shape viewer** — essential dashboard support for §2
7. Everything else follows as research directions

---

## References

- Molina et al. (2020). "Padé Activation Units: End-to-end Learning of Flexible Activation Functions in Deep Networks." ICLR.
- Liu et al. (2024). "KAN: Kolmogorov-Arnold Networks." arXiv:2404.19756.
- Ramachandran et al. (2017). "Searching for Activation Functions." arXiv:1710.05941 (discovered SiLU/Swish via NAS).
- Alemi et al. (2017). "Deep Variational Information Bottleneck." ICLR.
- Friston, K. (2010). "The free-energy principle: a unified brain theory?" Nature Reviews Neuroscience.
- Shwartz-Ziv & Tishby (2017). "Opening the Black Box of Deep Neural Networks via Information." arXiv:1703.00810.
- Jaderberg et al. (2017). "Population Based Training of Neural Networks." arXiv:1711.09846.
