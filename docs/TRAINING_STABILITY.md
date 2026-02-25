# Training Stability Research

## Problem

Training a 6.8M param GPT-2 style model (6 layers, 256 dim, 8 heads, BPE-4K vocab) on chat data. Gradient norms explode after warmup:

- During warmup (first 200 steps): grad norms stable at 0.8-4.0
- After warmup: avg 108K, max 301M, 75%+ of steps > 50
- `layer.0.attn.wo` consistently the largest gradient contributor
- Loss still decreases (8.3 → 5.9) despite instability — gradient direction after clipping is informative but optimizer state is corrupted

## Root Causes

### 1. gradClip=5.0 is too high (should be 1.0)

The standard for transformer pretraining is **gradClip=1.0** (nanoGPT, llm.c, GPT-2, GPT-3). With gradClip=5.0, Adam's internal momentum (m) and variance (v) estimates get corrupted by pre-clip gradient magnitudes in the millions. The clipped gradient direction is still useful (which is why loss decreases), but the optimizer's adaptive learning rate per-parameter becomes miscalibrated.

Our codebase default was 1.0 before we raised it to 5.0 for the chat domain.

### 2. beta2=0.999 is too slow (should be 0.95)

beta2 controls the EMA of squared gradients (second moment) in Adam. With beta2=0.999, this adapts over ~1000 steps. With beta2=0.95, it adapts in ~20 steps.

References using beta2=0.95:
- nanoGPT (Karpathy)
- LLaMA (Meta)
- "Small-scale proxies for large-scale Transformer training instabilities" (ICLR 2024, Wortsman et al.)
- MAE (Meta)

When a gradient spike occurs with beta2=0.999, the denominator barely updates, so the spike passes through nearly unattenuated. With 0.95, the denominator quickly absorbs the spike and naturally dampens the effective per-parameter learning rate.

### 3. lr=3e-4 is too high for our batch size

LR values in the literature assume much larger batch sizes:
- GPT-2 (124M): lr=2.5e-4, batch=512×1024 = 524K tokens/step
- GPT-3 (125M): lr=6e-4, batch=500K tokens/step
- nanoGPT Shakespeare: lr=1e-3, batch=64×256 = 16K tokens/step

We use batch=16×256 = 4,096 tokens/step — 128x smaller than GPT-2's batch.

Square-root batch scaling: `lr = 2.5e-4 × sqrt(4096/524288)` ≈ **2.2e-5**
From nanoGPT Shakespeare: `lr = 1e-3 × sqrt(4096/16384)` ≈ **5e-4**

Practical range: **5e-5 to 2e-4**, start with **1e-4**.

### 4. weight_decay=0.01 is low (should be 0.1)

nanoGPT uses weight_decay=0.1 (10x higher). Higher weight decay keeps weight norms bounded, which indirectly limits gradient norms. The standard range for transformer pretraining is 0.01-0.1.

Must ensure LN parameters and biases are excluded from weight decay (we have noDecayNames support for this).

### 5. layer.0.attn.wo gradient dominance

This is a known pattern. The first layer's output projection accumulates gradients from all downstream layers via the residual stream. The "attention logit growth" phenomenon causes softmax gradients to spike through wo when attention scores become large.

**Long-term fix**: QK-LayerNorm (apply LayerNorm to Q and K before computing attention scores). Used by Qwen 3, Gemma 3, OLMo 3, GLM 4.5. This directly prevents attention logit growth.

Our code already has logit capping (`clamp(ctx, rawScores, -30, 30)`) but this creates hard gradient walls rather than smooth normalization.

## Recommended Changes (Priority Order)

| Setting | Current | Recommended | Impact |
|---------|---------|-------------|--------|
| gradClip | 5.0 | **1.0** | Prevents Adam state corruption from massive pre-clip norms |
| beta2 | 0.999 | **0.95** | Fast adaptation to gradient scale changes |
| lr | 3e-4 | **1e-4** | Appropriate for our small batch size |
| weightDecay | 0.01 | **0.1** | Keeps weight norms bounded |
| warmup | 2000 (4%) | 4000 (8%) | More gradual LR ramp |

### Future architecture changes
- **QK-LayerNorm**: Apply LN to Q, K before attention score computation
- **Per-layer gradient clipping**: Clip individual parameter gradients, not just global norm

## Implementation Audit

Thorough code review of all training-critical components. No code changes made.

### AdamW Optimizer (`packages/train/src/optimizers.ts`)

**Status: CORRECT.** All three code paths (CPU, Helios CPU fallback, GPU SPIR-V kernel) implement identical math matching the canonical Loshchilov & Hutter 2019 algorithm.

- Update formula: correct (weight decay → moment update → bias correction → param update)
- Weight decay: truly decoupled (applied to params, not grads)
- Bias correction: correct (`1 - beta^t`, step starts at 1)
- GPU/CPU parity: verified — all three paths are semantically equivalent
- noDecayNames: correctly plumbed from CLI → optimizer, properly skips weight decay for matching params

**Minor concern**: `defaultTrainConfig.eps = 1e-6` is unusual for f32 training (standard is 1e-8). Not a bug but slightly conservative — biases effective LR downward for params with small second moments.

### Gradient Clipping (`packages/train/src/trainer.ts`)

**Status: CORRECT.** Matches PyTorch's `clip_grad_norm_`.

- Global L2 norm: correct — `sqrt(sum(g^2))` across all params
- Clip coefficient: `gradClip / gradNorm` — correct (no epsilon, but guard condition prevents div-by-zero)
- Logged gradient norm is PRE-clip — correct for monitoring
- Order: backward → accumulate → scale by 1/K → compute norm → clip → optimizer step → zero grads — correct
- No dedicated GPU kernel for grad norm (composes `mul` + `sum` + `scale`) — correct but could be optimized

**No bugs found.**

### Attention (`packages/model/src/gpt.ts`)

**Status: CORRECT.**

- QKV projection: separate Q/K/V matrices (no bias), functionally equivalent to GPT-2's fused c_attn
- Score computation: `Q @ K^T / sqrt(d_k)` — correct
- Logit capping: `clamp(scores, -30, 30)` — PaLM/Gemma technique, intentional
- Causal mask: `-1e9` fill value (not -Inf) — safe for f32, avoids NaN edge cases
- Softmax: numerically stable (subtract max before exp)
- Multi-head reshape and output projection: correct
- Pre-LN architecture (LN before attention/MLP) — more stable than Post-LN, correct choice
- No dropout anywhere — design choice, not a bug
- No bias terms on any linear layers — matches modern practice (LLaMA, GPT-J)

### Cross-Entropy Loss (`packages/helios/src/backend.ts`, `kernels.ts`)

**Status: CORRECT.** Fused kernel is numerically stable.

- Forward: log-sum-exp trick (subtract max → exp → sum → log) — stable
- Mean reduction: sum per-row losses / N — correct
- Backward: `(softmax - one_hot) / N` — correct standard gradient

**Minor concern**: GPU backward ignores upstream gradient `_gradOutput` parameter. Currently safe because CE is always the terminal loss node, but would break if loss were composed with regularization terms. The CPU fallback path correctly uses the upstream gradient — inconsistency between paths.

### Learning Rate Schedule (`packages/train/src/trainer.ts`)

**Status: CORRECT.**

- Linear warmup: ramps from `lrMin` to `lr_max` — correct, continuous
- Cosine decay: `lrMin + (lr - lrMin) * 0.5 * (1 + cos(pi * progress))` — standard formula
- Warmup-to-decay transition: continuous at boundary — correct

**Concerns**:
- `lrMin` defaults to 0 (nanoGPT defaults to `lr/10`). Decaying to 0 can be risky for long runs
- `warmupIters=0` triggers auto-warmup `min(2000, iters/5)` instead of disabling warmup — surprising behavior

### Weight Initialization (`packages/model/src/gpt.ts`)

**Status: CORRECT.** Matches nanoGPT conventions.

- Base std = 0.02 — correct (GPT-2 standard)
- Residual projections (`wo`, `fc2`) scaled by `1/sqrt(2*nLayer)` — correct
- Non-residual projections (`wq`, `wk`, `wv`, `fc1`) use base std — correct
- LayerNorm: weight=1, bias=0 — correct
- Embeddings (`wte`, `wpe`): N(0, 0.02) — correct
- `lmHead`: N(0, 0.02), no weight tying with `wte` — matches nanoGPT

### Weight Decay Exclusion (`apps/cli/src/commands/train.ts`)

**Status: BUG FOUND.**

Currently excluded from weight decay:
- `lnF.weight`, `lnF.bias` — correct
- `layer.N.ln1.weight`, `layer.N.ln1.bias` — correct
- `layer.N.ln2.weight`, `layer.N.ln2.bias` — correct

**Missing from noDecayNames**:
- **`wte` (token embeddings)** — nanoGPT excludes embeddings from weight decay
- **`wpe` (position embeddings)** — same rationale

Embeddings are lookup tables, not standard weight matrices. Applying weight decay to them degrades token/position representations. This is a bug — both should be added to `noDecayNames`.

No bias terms exist on linear layers in this model, so no missing bias exclusions.

## References

- nanoGPT: https://github.com/karpathy/nanoGPT
- llm.c GPT-2 reproduction: https://github.com/karpathy/llm.c/discussions/481
- Small-scale proxies for large-scale Transformer training instabilities (ICLR 2024): https://arxiv.org/abs/2309.14322
- GPT-3 paper (LR/batch scaling table): https://arxiv.org/pdf/2005.14165
- Stabilizing Transformer Training by Preventing Attention Entropy Collapse: https://proceedings.mlr.press/v202/zhai23a
