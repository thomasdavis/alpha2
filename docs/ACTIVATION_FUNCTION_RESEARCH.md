# Activation Function Evolution: From Symbiogenesis to Alpha

**Date**: 2026-02-25
**Context**: Research into applying Symbiogenesis's per-layer activation function evolution findings to Alpha's GPT training system.

---

## Table of Contents

1. [What Symbiogenesis Proved](#1-what-symbiogenesis-proved)
2. [What Modern LLMs Actually Use](#2-what-modern-llms-actually-use)
3. [SwiGLU: The Modern Standard](#3-swiglu-the-modern-standard)
4. [Alpha's Current State](#4-alphas-current-state)
5. [The Gap Between Symbiogenesis and Transformers](#5-the-gap-between-symbiogenesis-and-transformers)
6. [What Actually Transfers](#6-what-actually-transfers)
7. [Concrete Upgrade Paths for Alpha](#7-concrete-upgrade-paths-for-alpha)
8. [Implementation Anatomy](#8-implementation-anatomy)
9. [Expected Impact](#9-expected-impact)
10. [Research Frontiers](#10-research-frontiers)

---

## 1. What Symbiogenesis Proved

Symbiogenesis (Phase 10) introduced per-layer activation function evolution via natural selection on small MLP populations. Each hidden layer carries its own activation from a pool of 7 (ReLU, GELU, SiLU, Tanh, LeakyReLU, ELU, Sigmoid). Activations propagate through fusion, mutate at a configurable per-layer rate, and compete through fitness-based replacement.

### The MNIST Benchmark

| Metric | Activation Evolution | ReLU-Only | Delta |
|--------|---------------------|-----------|-------|
| Train Accuracy | 92.86% | 66.95% | **+25.91%** |
| Test Accuracy | 69.20% | 58.60% | **+10.60%** |
| Best Architecture | `[31]` (ELU) | `[6]` (ReLU) | — |
| Avg Depth | 1.0 | 2.2 | -1.2 |

### Final Activation Distribution

| Activation | Fraction |
|------------|----------|
| **ELU** | **90%** |
| GELU | 10% |
| Everything else | 0% |

### Key Findings

1. **ELU convergence was unguided** — all 7 activations started with equal probability. Natural selection over 50 iterations converged 90% of layers to ELU without any prior bias.

2. **Better activations substitute for depth** — a single wide ELU layer (`[31]`) outperformed multiple narrow ReLU layers (avg depth 2.2). The activation function quality compensated for reduced depth.

3. **GELU filled a minority niche** — 10% persistence suggests complementary value (regularization or diversity) that pure ELU populations cannot provide.

4. **The mechanism is simple**: inherit activations from parents via fusion, randomly mutate each layer with probability 0.15, let fitness-based selection do the rest. No gradient-based activation search, no meta-learning.

---

## 2. What Modern LLMs Actually Use

### The Industry Map

| Model | Year | Activation | MLP Type |
|-------|------|-----------|----------|
| GPT-2 | 2019 | GELU | Standard 2-matrix FFN |
| GPT-3 | 2020 | GELU | Standard 2-matrix FFN |
| BERT | 2018 | GELU | Standard 2-matrix FFN |
| Falcon | 2023 | GELU | Standard 2-matrix FFN |
| Phi-2 | 2023 | GELU (NewGELU) | Standard 2-matrix FFN |
| **PaLM** | **2022** | **SwiGLU** | **3-matrix gated FFN** |
| **PaLM 2** | **2023** | **SwiGLU** | **3-matrix gated FFN** |
| **Llama 1/2/3** | **2023-24** | **SwiGLU** | **3-matrix gated FFN** |
| **Mistral** | **2023** | **SwiGLU** | **3-matrix gated FFN** |
| **Gemma** | **2024** | **GeGLU** | **3-matrix gated FFN** |
| Phi-3 | 2024 | SiLU | Standard FFN |

**The inflection point was 2022.** Pre-2022 models overwhelmingly use GELU in a standard 2-matrix FFN. Post-2022 frontier models have converged on **SwiGLU** (or GeGLU) with a **3-matrix gated FFN**. PaLM validated SwiGLU at 540B scale; Llama made it the open-source default.

### The Trend

Alpha currently uses the GPT-2 era architecture: GELU in a 2-matrix FFN. Every frontier model since 2022 has moved to gated linear units. This is the single biggest architectural gap between Alpha and modern practice.

---

## 3. SwiGLU: The Modern Standard

### What Is It?

SwiGLU (Shazeer, 2020, "GLU Variants Improve Transformer") replaces the standard FFN's single activation with a **gated** activation. The key insight is element-wise gating with a second linear projection:

**Standard FFN (Alpha's current approach):**
```
FFN(x) = GELU(x · W1) · W2
```
Two weight matrices: W1 (`nEmbd → 4*nEmbd`) and W2 (`4*nEmbd → nEmbd`).

**SwiGLU:**
```
FFN(x) = [SiLU(x · W_gate) ⊙ (x · W_up)] · W_down
```
Three weight matrices: W_gate and W_up (`nEmbd → hidden_dim`) and W_down (`hidden_dim → nEmbd`).

The `⊙` is element-wise multiplication. The gate branch decides **which features to pass through**, the up branch computes **the features themselves**. This multiplicative interaction is fundamentally more expressive than a single-branch activation.

### The GLU Family

All GLU variants share the same structure, differing only in the gating activation:

| Variant | Formula | Used By |
|---------|---------|---------|
| GLU | σ(xW) ⊙ xV | Original (Dauphin, 2017) |
| ReGLU | ReLU(xW) ⊙ xV | — |
| **SwiGLU** | **SiLU(xW) ⊙ xV** | **Llama, PaLM, Mistral** |
| GeGLU | GELU(xW) ⊙ xV | Gemma |

### The 2/3 Dimension Adjustment

Three matrices = 50% more parameters per layer than two matrices. To keep total parameter count constant, the hidden dimension is reduced:

- Standard FFN: `hidden_dim = 4 * nEmbd` (expansion ratio 4x)
- SwiGLU: `hidden_dim = (8/3) * nEmbd` (expansion ratio ~2.67x), often rounded to nearest multiple of 64 or 256

With this adjustment, total MLP parameters are approximately equal:
- Standard: `2 × nEmbd × 4*nEmbd = 8 × nEmbd²`
- SwiGLU: `3 × nEmbd × (8/3)*nEmbd = 8 × nEmbd²`

In Llama specifically: `hidden_dim = round((8/3) * nEmbd / 256) * 256`.

### Why Is SwiGLU Better?

1. **Gating as learned feature selection** — the network learns which features to amplify vs. suppress, a richer operation than uniformly applying an activation.

2. **Richer gradient pathways** — both the gate and value branches receive gradients during backprop, effectively doubling the gradient highways through the FFN.

3. **Consistent perplexity improvement** — Shazeer's original paper showed SwiGLU and GeGLU consistently produced the best perplexities across pretraining and fine-tuning benchmarks (T5 on GLUE, SuperGLUE, SQuAD).

4. **Validated at scale** — PaLM (540B), Llama 1/2/3, Mistral 7B, Gemma all ship with GLU variants. Not a paper result — a production reality.

### Benchmark Data

**Shazeer 2020** (T5, matched parameter count):
- SwiGLU and GeGLU were the best-performing variants across all benchmarks
- Consistent ~0.5-1.0 perplexity point improvement over standard GELU FFN

**xIELU paper 2024** (1.1B Llama, 126B tokens):

| Activation | Perplexity (126B tokens) |
|-----------|--------------------------|
| SiLU (no gating) | 18.576 (at 4B) |
| SwiGLU | 10.517 |
| ReLU² | 10.352 |
| xIELU | 10.207 |

**Critical finding**: plain SiLU (no gating) scores 18.576 perplexity vs. SwiGLU's 17.583 at 4B tokens. **The gating mechanism is doing substantial work** — it's not just the activation function, it's the architectural change.

---

## 4. Alpha's Current State

### The MLP Block

From `packages/model/src/gpt.ts:179-184`:
```typescript
// 2) LN → MLP → Residual
const ln2Out = layerNorm(ctx, x, layer.ln2.weight, layer.ln2.bias, 1e-5);
const flat = reshape(ctx, ln2Out, [Batch * T, nEmbd]);
const h = matmulTransposedGelu(ctx, flat, layer.mlp.fc1);      // [B*T, 4*nEmbd]
const mlpOut = reshape(ctx, matmulTransposed(ctx, h, layer.mlp.fc2), [B, T, nEmbd]);
return residualDropoutAdd(ctx, x, mlpOut, config.dropout, training);
```

- Two matrices: `fc1 [4*nEmbd, nEmbd]` and `fc2 [nEmbd, 4*nEmbd]`
- GELU is hardcoded — no config option to change it
- Expansion ratio: exactly 4x
- `matmulTransposedGelu` is a fused **autograd** op (one tape entry), but at the GPU level it's still two dispatches: `kernelMatmulTransposed` then `kernelGelu`/`kernelGeluVec4`

### What Exists in the GPU Stack

| Component | GELU | ReLU | SiLU |
|-----------|------|------|------|
| Forward kernel (SPIR-V) | `kernelGelu` | `kernelRelu` | `kernelSilu` |
| Vec4 forward kernel | `kernelGeluVec4` | `kernelReluVec4` | `kernelSiluVec4` |
| Backward kernel (SPIR-V) | `kernelGeluBackward` | `kernelReluBackward` | **MISSING** |
| Backend `.forward()` | `backend.gelu()` | `backend.relu()` | `backend.silu()` |
| Backend `.backward()` | `backend.geluBackward()` | `backend.reluBackward()` | **MISSING** |
| Autograd op | `gelu()` in ops.ts | `relu()` in ops.ts | **MISSING** |
| Fused matmul+act | `matmulTransposedGelu()` | — | — |

**SiLU has forward-only support.** The forward kernel exists (both scalar and Vec4), and `backend.silu()` is wired up. But there is no backward kernel, no `siluBackward` on the Backend interface, and no `silu()` autograd op. SiLU cannot participate in training today.

### ModelConfig

```typescript
export interface ModelConfig {
  readonly vocabSize: number;
  readonly blockSize: number;
  readonly nLayer: number;
  readonly nEmbd: number;
  readonly nHead: number;
  readonly dropout: number;
}
```

No `activation` field. No `mlpType` field. No `mlpHiddenDim` field.

---

## 5. The Gap Between Symbiogenesis and Transformers

Symbiogenesis's activation evolution results are compelling but don't directly transfer to transformer language models. Here's why:

### Different Architectures, Different Constraints

| Factor | Symbiogenesis | Alpha (Transformer) |
|--------|---------------|---------------------|
| Architecture | Shallow MLP (1-8 layers) | Deep transformer (6-8 blocks × 2 sublayers) |
| Normalization | None | LayerNorm before every sublayer |
| Task | MNIST classification | Language modeling |
| Parameters | ~18K | ~10-30M |
| Activation role | The only nonlinearity | One component in a rich residual pipeline |

### Why ELU Won on MNIST but Wouldn't Win in Transformers

ELU's advantages — smooth negative saturation, self-normalizing properties — are most relevant for:
- Models **without LayerNorm** (ELU provides centering that LayerNorm already gives)
- **Shallow networks** where dead neurons are a bigger problem
- **Small-scale tasks** where activation function choice has outsized impact

Transformers use LayerNorm extensively, which already handles the centering/normalizing benefits ELU provides. No production transformer has ever shipped with ELU.

**However**: the xIELU activation (2024, arXiv:2411.13010), which outperforms SwiGLU at 1.1B scale, is **derived by integrating ELU**. So ELU's mathematical structure appears in modified form in cutting-edge research — Symbiogenesis may have independently identified ELU's latent potential.

### Why Per-Layer Activation Variation Doesn't Transfer (Yet)

No production LLM uses different activations at different layers. Every model uses one activation uniformly. Research on "Transformers with Learnable Activation Functions" (2022, arXiv:2208.14111) showed that **learned activations do substantially vary between layers** of a pre-trained model, suggesting the potential exists — but no one has demonstrated practical gains from explicit per-layer activation selection in transformers at scale.

### What DOES Transfer

The core insight from Symbiogenesis that transfers is not "use ELU" or "vary activations per-layer" — it's:

1. **Activation function choice matters more than people think** — the +25.91% training accuracy gap is a wake-up call. Alpha uses the GPT-2 era activation (GELU) when the industry moved to SwiGLU 3 years ago.

2. **Selection pressure finds the right activation** — you don't need to know which activation is best a priori. Symbiogenesis proved that fitness-based selection converges on the optimal choice. For Alpha, the "selection pressure" is empirical ablation: train identical models with different activations, keep the winner.

3. **The gating principle** — Symbiogenesis's parallel fusion (where one parent gates another) mirrors the gating mechanism in SwiGLU. The evolutionary success of gating as a fusion strategy aligns with the empirical success of gating in the FFN.

---

## 6. What Actually Transfers

### Principle 1: Don't Default to GELU — It's Outdated

Alpha uses GELU because GPT-2 used GELU. GPT-2 is from 2019. Every frontier model since PaLM (2022) uses a GLU variant. This is the lowest-hanging fruit.

### Principle 2: Gating > Plain Activation

Both Symbiogenesis (parallel fusion gating) and the LLM literature (SwiGLU) converge on the same insight: **multiplicative gating is more expressive than additive activation**. A gate that learns which features to pass through outperforms a fixed nonlinearity applied uniformly.

- Symbiogenesis: parallel fusion picks the "dominant" parent per layer (gating by fitness)
- SwiGLU: learned element-wise gate per feature dimension
- Same principle, different granularity

### Principle 3: Ablation is Cheap, Wrong Defaults are Expensive

Symbiogenesis's entire value proposition is that you can search the activation space cheaply through population-based evolution. Alpha can't do population-based search (one model at a time), but it can run ablation studies: train the same model config with GELU vs. SwiGLU vs. GeGLU and compare validation loss curves. At Alpha's scale (~10-30M params, ~50K iterations), a single ablation takes hours, not weeks.

### Principle 4: The Activation-Depth Tradeoff

Symbiogenesis found that better activations reduce the need for depth. This maps to a practical question for Alpha: **if Alpha switches to SwiGLU, can it achieve the same loss with fewer layers?** Fewer layers = less VRAM = larger batch size = better GPU utilization. This is a direct scaling win.

---

## 7. Concrete Upgrade Paths for Alpha

### Path A: GELU → SwiGLU (Recommended)

The highest-impact change. Replaces the 2-matrix GELU FFN with a 3-matrix SwiGLU FFN, matching Llama/PaLM/Mistral architecture.

**What changes:**
- `ModelConfig` gets an `activation` field (`"gelu" | "swiglu"`)
- `LayerParams.mlp` gets a third weight matrix (`gate`, `up`, `down` instead of `fc1`, `fc2`)
- MLP forward: `matmulTransposedGelu(x, fc1)` → `silu(matmulTransposed(x, gate)) ⊙ matmulTransposed(x, up)`
- Hidden dim: `4 * nEmbd` → `round((8/3) * nEmbd / 64) * 64`
- New autograd `silu()` op
- New SPIR-V `kernelSiluBackward` (scalar + Vec4)
- New `siluBackward` on Backend interface

**Expected impact:** ~0.5-1.0 perplexity point improvement (extrapolating from Shazeer's T5 results). Roughly equivalent parameter count. The gating mechanism provides richer gradient flow and learned feature selection.

**Complexity:** Moderate. ~300-400 lines of new code across 6-7 files. The SiLU backward kernel is straightforward (`sigmoid(x) * (1 + x * (1 - sigmoid(x)))`). The hardest part is refactoring the MLP forward pass and ensuring checkpoint compatibility.

### Path B: GELU → GeGLU (Simpler Variant)

Same structural change as SwiGLU, but uses GELU as the gating activation instead of SiLU. Advantage: **no new backward kernel needed** — GELU backward already exists.

**What changes:**
- Same as Path A except the gate branch uses `gelu()` instead of `silu()`
- No `kernelSiluBackward`, no `siluBackward`, no `silu()` autograd op needed
- Forward: `gelu(matmulTransposed(x, gate)) ⊙ matmulTransposed(x, up)`

**Expected impact:** Essentially equal to SwiGLU (Shazeer showed GeGLU and SwiGLU are within noise). Gemma uses GeGLU at scale, validating it.

**Complexity:** Lower than Path A. No new GPU kernels. The main work is the MLP restructuring.

### Path C: Configurable Activation (GELU/SiLU/ReLU)

Keep the 2-matrix FFN structure, but make the activation function configurable. This enables ablation studies without the structural MLP change.

**What changes:**
- `ModelConfig` gets an `activation` field (`"gelu" | "silu" | "relu"`)
- `transformerBlock` dispatches to the appropriate fused op or separate ops
- New autograd `silu()` op + backward kernel (same as Path A)
- Optionally: `matmulTransposedSilu` fused autograd op for parity with `matmulTransposedGelu`

**Expected impact:** Small. Plain SiLU (without gating) scored 18.576 vs. SwiGLU's 17.583 in the xIELU paper. The gating is what matters, not the specific activation function. Still useful for ablation data.

**Complexity:** Low-moderate. ~150-200 lines.

### Path D: Full Activation Evolution (Research Mode)

Port Symbiogenesis's evolutionary mechanism: maintain a population of model configs with different activations, train each, select survivors. This is a hyperparameter search tool, not a model architecture change.

**What changes:**
- A script that launches multiple training runs with different activation configs
- Tracks fitness (validation loss) and reports the winner
- Could vary both activation function and expansion ratio

**Expected impact:** Finds the empirically optimal activation for Alpha's specific model/data combination. The search itself is expensive (N training runs), but the finding is permanent.

**Complexity:** Low (it's a script, not a model change). But requires multiple training runs.

### Recommendation

**Path A (SwiGLU) first, then Path D (ablation) to validate.**

SwiGLU is the industry standard for a reason. Implementing it brings Alpha's architecture to 2023 parity. Then run a GELU-vs-SwiGLU ablation to quantify the improvement on Alpha's specific task/data. If SwiGLU wins (extremely likely based on all available evidence), keep it as the default.

---

## 8. Implementation Anatomy

### What Exists vs. What's Missing (for SwiGLU)

```
                   Forward          Backward         Autograd Op
                   ───────          ────────         ───────────
GELU               kernelGelu       kernelGeluBwd    gelu(), matmulTransposedGelu()
ReLU               kernelRelu       kernelReluBwd    relu()
SiLU               kernelSilu       ✗ MISSING        ✗ MISSING
                   kernelSiluVec4   ✗ MISSING
```

### New Components Needed

**1. `kernelSiluBackward` SPIR-V kernel** (`packages/helios/src/kernels/elementwise.ts` or `nn.ts`)

SiLU derivative: `d/dx [x · σ(x)] = σ(x) · (1 + x · (1 - σ(x)))`

Where `σ(x) = 1 / (1 + exp(-x))`.

3-binding kernel (input, gradOutput, gradInput), same pattern as `kernelGeluBackward`. Need both scalar and Vec4 variants.

**2. `siluBackward` on Backend interface** (`packages/core/src/interfaces.ts`)

```typescript
siluBackward?(input: TensorData, gradOutput: TensorData): TensorData;
```

**3. `siluBackward` on Helios backend** (`packages/helios/src/backend.ts`)

Route to `gpuBinaryOp("silu_backward")` with CPU fallback.

**4. `silu()` autograd op** (`packages/autograd/src/ops.ts`)

Same pattern as `gelu()`: forward via `backend.silu()`, backward via `backend.siluBackward()` with CPU inline fallback.

**5. MLP restructuring** (`packages/model/src/gpt.ts`)

Replace:
```typescript
// Current: 2-matrix GELU FFN
const h = matmulTransposedGelu(ctx, flat, layer.mlp.fc1);
const mlpOut = matmulTransposed(ctx, h, layer.mlp.fc2);
```

With:
```typescript
// SwiGLU: 3-matrix gated FFN
const gateProj = matmulTransposed(ctx, flat, layer.mlp.gate);
const upProj = matmulTransposed(ctx, flat, layer.mlp.up);
const activated = mul(ctx, silu(ctx, gateProj), upProj);
const mlpOut = matmulTransposed(ctx, activated, layer.mlp.down);
```

**6. Weight initialization** (`packages/model/src/gpt.ts`)

```typescript
// Current:
mlp: {
  fc1: initWeight(backend, rng, [4 * nEmbd, nEmbd], std),
  fc2: initWeight(backend, rng, [nEmbd, 4 * nEmbd], std / Math.sqrt(2 * nLayer)),
}

// SwiGLU:
const hiddenDim = Math.round((8 / 3) * nEmbd / 64) * 64;
mlp: {
  gate: initWeight(backend, rng, [hiddenDim, nEmbd], std),
  up:   initWeight(backend, rng, [hiddenDim, nEmbd], std),
  down: initWeight(backend, rng, [nEmbd, hiddenDim], std / Math.sqrt(2 * nLayer)),
}
```

**7. `ModelConfig` update** (`packages/core/src/types.ts`)

```typescript
export interface ModelConfig {
  readonly vocabSize: number;
  readonly blockSize: number;
  readonly nLayer: number;
  readonly nEmbd: number;
  readonly nHead: number;
  readonly dropout: number;
  readonly activation?: "gelu" | "swiglu";  // default "gelu" for backward compat
}
```

### Checkpoint Compatibility

Adding `activation` to `ModelConfig` with a default value preserves backward compatibility — existing checkpoints (which don't have the field) will default to `"gelu"` and load as before. New SwiGLU checkpoints will have different weight names (`gate`/`up`/`down` vs. `fc1`/`fc2`) and shapes, so they're inherently incompatible with GELU checkpoints. This is correct and expected.

---

## 9. Expected Impact

### Perplexity Improvement

Based on Shazeer (2020) and subsequent work:
- At T5-Base scale (~220M): SwiGLU improved perplexity by ~0.5-1.0 points
- At 1.1B scale (xIELU paper): SwiGLU at 10.517 vs. GELU baseline higher
- Improvement is consistent across model sizes and tends to be **more pronounced at smaller scales** where every architectural choice matters more

Alpha's models are 10-30M parameters — smaller than any published benchmark. The relative improvement could be larger or smaller. Ablation is the only way to know for sure.

### Training Efficiency

The gating mechanism in SwiGLU provides richer gradient flow. This can manifest as:
- **Faster convergence** — reaching a given loss target in fewer iterations
- **Better final loss** — converging to a lower minimum at the same iteration count
- **More stable training** — the multiplicative gate acts as a soft feature selection, potentially reducing gradient spikes

### Parameter Efficiency

With the 2/3 dimension adjustment, SwiGLU has approximately the same total parameter count as the standard FFN. But the parameters are "working harder" — three specialized projection matrices vs. two general ones. Empirically this translates to better loss per parameter.

### Compute Cost

SwiGLU replaces one fused matmul+GELU dispatch with:
- Two matmul dispatches (gate and up projections) — but each is 2/3 the size
- One SiLU dispatch
- One element-wise multiply
- One matmul dispatch (down projection) — 2/3 the size

Net FLOPs are approximately equal (three 2/3-sized matmuls ≈ two full-sized matmuls). The activation function cost is negligible (<1% of total compute). Memory overhead is slightly higher due to the intermediate gate/up tensors, but these can be released immediately after the element-wise multiply.

---

## 10. Research Frontiers

### Things Worth Watching

**xIELU (2024)** — Derives activation functions via integration. xIELU outperforms SwiGLU by ~3% perplexity at 1.1B scale. Uses a trainable activation that integrates ELU for negative inputs. Connects back to Symbiogenesis's discovery that ELU has latent potential. Not yet validated at frontier scale, but promising.

**Expanded Gating Ranges (2024)** — Instead of gating in [0, 1], gates in [-α, 1+α] with learned α. Consistent improvement across all activation types. Simple modification to SwiGLU that could be added later.

**ReLU Strikes Back (Apple, ICLR 2024)** — Shows that LLMs trained with SiLU/GELU can be fine-tuned to use ReLU with minimal accuracy loss, unlocking 3x inference speedup from activation sparsity. Suggests a two-phase approach: train with SwiGLU for quality, distill to ReLU for inference speed.

**Learnable Activation Functions (2022)** — Per-layer learned activations (Recurrent Activation Functions) show that optimal activations vary significantly between layers. No practical deployment yet, but validates the per-layer intuition from Symbiogenesis.

### The Symbiogenesis Connection to Future Work

Symbiogenesis proved that evolutionary pressure naturally converges on optimal activations. The mechanism — population-based search with fitness selection — could be adapted for Alpha as an architecture search tool:

1. Define a search space: {GELU FFN, SwiGLU, GeGLU, ReGLU} × {expansion ratios} × {nLayer, nEmbd combinations}
2. Train a small population of models (3-5) for N iterations each
3. Select the fittest configuration
4. Train the full run with the winning config

This is essentially Symbiogenesis's Phase 10 applied at the model-config level rather than the per-layer level. The search space is small enough that even exhaustive ablation is practical at Alpha's current scale.

---

## References

- Shazeer, N. (2020). "GLU Variants Improve Transformer." arXiv:2002.05202
- Touvron et al. (2023). "LLaMA: Open and Efficient Foundation Language Models." arXiv:2302.13971
- Chowdhery et al. (2022). "PaLM: Scaling Language Modeling with Pathways." arXiv:2204.02311
- Mirzadeh et al. (2024). "ReLU Strikes Back: Exploiting Activation Sparsity in LLMs." ICLR 2024, arXiv:2310.04564
- Nair et al. (2024). "Expanded Gating Ranges Improve Activation Functions." arXiv:2405.20768
- Heim et al. (2024). "Deriving Activation Functions via Integration: xIELU." arXiv:2411.13010
- Bai et al. (2022). "Transformers with Learnable Activation Functions." arXiv:2208.14111
- Hendrycks & Gimpel (2016). "Gaussian Error Linear Units (GELUs)." arXiv:1606.08415
- Dauphin et al. (2017). "Language Modeling with Gated Convolutional Networks." ICML 2017
- Symbiogenesis Phase 10 Results (2026). `symbiogenesis/docs/Phase10_Results.md`
