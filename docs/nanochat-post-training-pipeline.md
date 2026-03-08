# Post-Training Pipeline for Nanochat-Class Models

## Abstract

This document specifies a post-training pipeline for sub-1B parameter language models trained on the nanochat architecture. The pipeline transforms a pretrained base model into a capable conversational agent through four sequential phases: **Protocol Adaptation**, **Supervised Finetuning**, **Reinforcement Learning with Verifiable Rewards**, and **Self-Play Refinement**. Each phase has a distinct purpose, operates on different data distributions, and targets different failure modes.

The core insight, drawn from Karpathy's work, is that small model failures are overwhelmingly **protocol failures** — the model has latent knowledge but cannot surface it through the expected interaction format. Post-training is therefore primarily about teaching the *shape* of tasks, not injecting new knowledge.

---

## Architecture of the Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         BASE MODEL (pretrained)                     │
│                    depth=N, trained on web text                      │
│                    knows language, has world knowledge               │
│                    but: no concept of conversation, tasks, tools     │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1: PROTOCOL ADAPTATION (midtraining)                         │
│                                                                     │
│  Purpose:  Teach the model that conversations exist                 │
│  Data:     Mixed web + conversation format data                     │
│  Tokens:   ~500K-1M conversations                                   │
│  Duration: Short (minutes on 8xH100, ~1hr on 1xL4)                 │
│  Key:      Introduce special tokens, multi-turn structure,          │
│            multiple-choice format, tool-use syntax                  │
│                                                                     │
│  Loss:     Standard cross-entropy on ALL tokens (no masking)        │
│            Model adapts to distribution, not just completions       │
│                                                                     │
│  This phase bridges the distribution gap between web text           │
│  and conversation format. Without it, SFT fights two battles        │
│  simultaneously (format + quality) and wins neither cleanly.        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 2: SUPERVISED FINETUNING (SFT)                               │
│                                                                     │
│  Purpose:  Teach quality responses and task-specific behavior       │
│  Data:     Curated conversations (high quality only)                │
│  Duration: 1 epoch over mixture (minutes to hours)                  │
│  Key:      Loss masking — only supervise assistant completions       │
│            Best-fit packing — no token waste, no cropping           │
│            Domain-matched format (inference-identical)               │
│                                                                     │
│  Loss:     Cross-entropy on assistant tokens only (mask=1)          │
│            User prompts, special tokens, tool outputs → mask=0      │
│                                                                     │
│  The model learns WHAT to say, not just HOW to format.              │
│  Data quality is everything here — "cherry pick the most            │
│  beautiful/good data" (Karpathy).                                   │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 3: REINFORCEMENT LEARNING (RL with verifiable rewards)       │
│                                                                     │
│  Purpose:  Improve on tasks with objective correctness criteria     │
│  Data:     Problem sets with verifiable answers (math, code)        │
│  Method:   Simplified GRPO — on-policy, no trust region, no KL     │
│  Duration: ~1 hour+ (many rollouts per problem)                     │
│  Key:      Model generates its own solutions, gets binary reward,   │
│            learns to self-correct through policy gradient            │
│                                                                     │
│  Loss:     L = -∑ log p(token) × advantage(sequence)               │
│            advantage = reward - mean(rewards_in_group)               │
│                                                                     │
│  Only applied to domains with unambiguous correctness.              │
│  Not for style, personality, or open-ended generation.              │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 4: SELF-PLAY REFINEMENT (optional, experimental)             │
│                                                                     │
│  Purpose:  Iterative improvement without new human data             │
│  Method:   Generate → evaluate → train on improved completions      │
│  Key:      The model becomes its own teacher through self-play      │
│                                                                     │
│  Only pursue if Phase 3 shows the model can reliably self-evaluate. │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Protocol Adaptation (Midtraining)

### Why This Phase Exists

A base model trained on web text has no concept of:
- Conversation turns (who is speaking, when to stop)
- Special tokens that delimit roles (`<|user_start|>`, `<|assistant_end|>`)
- Multiple-choice quiz format (mapping options to letters)
- Tool-use syntax (when to invoke Python, how to read outputs)

Karpathy's key insight: *"The issue is not that the model doesn't have the knowledge, it's that it doesn't understand how Multiple Choice works to surface that knowledge."*

If you skip midtraining and go straight to SFT, the model must simultaneously learn:
1. That special tokens exist and have structural meaning
2. That conversations have turn-taking structure
3. What high-quality responses look like

This conflation produces worse results than separating format learning (Phase 1) from quality learning (Phase 2).

### Token Format

```
<|bos|>
<|user_start|>What is 2+2?<|user_end|>
<|assistant_start|>Let me calculate that.
<|python_start|>2+2<|python_end|>
<|output_start|>4<|output_end|>
The answer is 4.<|assistant_end|>
<|user_start|>Thanks!<|user_end|>
<|assistant_start|>You're welcome!<|assistant_end|>
```

Nine special tokens total:
| Token | Purpose |
|-------|---------|
| `<|bos|>` | Document boundary |
| `<|user_start|>` / `<|user_end|>` | User turn delimiters |
| `<|assistant_start|>` / `<|assistant_end|>` | Assistant turn delimiters |
| `<|python_start|>` / `<|python_end|>` | Tool invocation |
| `<|output_start|>` / `<|output_end|>` | Tool output (not generated by model) |

These tokens are **absent during pretraining** and introduced here. The model learns their structural role through exposure.

### Data Mixture

| Dataset | Rows | Purpose |
|---------|------|---------|
| SmolTalk | 460K | General conversation distribution |
| MMLU (auxiliary_train) | 100K × N epochs | Multiple-choice format comprehension |
| GSM8K | 8K × N epochs | Math reasoning + tool-use syntax |
| Identity conversations | 1-2K | Personality/identity anchoring |
| Spelling tasks | 200K+ | Character-level reasoning |

### Loss Strategy

**Full-sequence cross-entropy** — no masking. The model needs to learn the distribution of conversations, including user turns, special tokens, and tool outputs. This is distribution adaptation, not instruction following.

### Hyperparameters

- Continue from pretrained optimizer state (warm-start momentum buffers)
- Inherit LR from pretraining but apply `init_lr_frac=0.8` scaling
- `warmdown_ratio=0.5` — linear decay over second half
- `weight_decay=0.0` — pretraining already decayed to zero
- Single epoch through the mixture (no overfitting on format data)

---

## Phase 2: Supervised Finetuning (SFT)

### Why This Phase Is Different From Midtraining

Midtraining teaches format. SFT teaches quality. The critical differences:

1. **Loss masking**: Only assistant completions are supervised (mask=1). User prompts, special tokens, and tool outputs get mask=0 (targets set to -1, ignore_index).

2. **Data quality**: Midtraining uses bulk data for distribution coverage. SFT uses cherry-picked, high-quality responses.

3. **Format matching**: SFT data looks exactly like inference — each batch row starts with `<|bos|>`, conversations are packed with best-fit algorithm, and padding positions are masked. This eliminates the train/test format mismatch.

### Loss Masking Strategy

```python
# mask=1: supervise (assistant text, python code, assistant_end token)
# mask=0: don't supervise (user text, special tokens, tool outputs)

# The assistant_end token IS supervised — the model must learn
# to stop generating, not just what to generate.

# Tool outputs (between output_start/output_end) are NOT supervised
# because they come from runtime execution, not the model.
```

### Data Packing: Best-Fit Algorithm

Naive padding wastes ~40% of compute on empty tokens. Naive cropping discards long conversations. Nanochat uses **best-fit packing**:

1. Maintain a buffer of tokenized conversations
2. For each row in the batch, find the largest conversation that fits entirely
3. Pack multiple conversations into one row (each starting with `<|bos|>`)
4. When no conversation fits the remaining space, pad (never crop)
5. Padding positions have targets masked with -1

This guarantees **zero token waste** — every conversation is trained on completely.

### Optimizer Warm-Start

The SFT optimizer loads momentum buffers from the pretrained/midtrained checkpoint but resets learning rates. This preserves the curvature information from pretraining while allowing SFT to use its own LR schedule.

```python
# Load pretrained optimizer state (momentum, variance)
optimizer.load_state_dict(pretrained_optimizer_state)
# But reset LRs to SFT values (pretraining decayed them to ~0)
for group, lr in zip(optimizer.param_groups, sft_learning_rates):
    group["lr"] = lr
```

### Learning Rate Schedule

```
LR
│  ┌──────────────────────────┐
│ /                            \
│/                              \
└──────────────────────────────────→ progress
  warmup     constant      warmdown
  (0%)       (0-50%)       (50-100%)
```

- `warmup_ratio=0.0` (no warmup — momentum is already warm)
- `warmdown_ratio=0.5` (linear decay over second half)
- `init_lr_frac=0.8` (start at 80% of base LR)
- `final_lr_frac=0.0` (decay to zero)

### Evaluation During SFT

**ChatCORE** metric — a normalized accuracy score across:
- ARC-Easy, ARC-Challenge, MMLU (categorical, baseline=0.25)
- GSM8K, HumanEval, SpellingBee (generative, baseline=0.0)

Score = mean of centered accuracies: `(acc - baseline) / (1 - baseline)`

Ranges from 0 (random) to 1 (perfect). Evaluated every 200 steps.

---

## Phase 3: Reinforcement Learning

### Why RL After SFT

SFT teaches the model to imitate. But imitation has a ceiling — the model can never exceed the quality of its training data. RL allows the model to discover novel solutions by:

1. Generating candidate solutions (rollouts)
2. Evaluating them with an objective reward function
3. Increasing probability of successful strategies

This is only effective for domains with **verifiable rewards** — problems where correctness can be checked automatically (math answers, code execution, factual lookups).

### Algorithm: Simplified GRPO

Nanochat uses a radically simplified variant of Group Relative Policy Optimization. What's removed is as important as what's kept:

| Standard GRPO | Nanochat's Version |
|---------------|-------------------|
| KL penalty to reference model | **Removed** — no reference model needed |
| PPO ratio clipping | **Removed** — on-policy, no importance weighting |
| Value function baseline | **Removed** — advantages from reward mean-shift only |
| Z-score normalization | **Removed** — only mean subtraction |
| Entropy bonus | **Removed** — pure reward optimization |

What remains is essentially **REINFORCE with a group baseline**:

```
For each problem:
  1. Generate K completions (K=16 default)
  2. Score each with binary reward r ∈ {0, 1}
  3. Compute advantage: a = r - mean(rewards)
  4. Policy gradient: L = -∑ log p(token) × a / num_valid_tokens
  5. Gradient step
```

### Mathematical Formulation

```
Given problem x, generate K responses {y₁, ..., yₖ} ~ π_θ(·|x)
Compute rewards: rᵢ = R(x, yᵢ) ∈ {0, 1}
Compute advantages: aᵢ = rᵢ - (1/K) ∑ rⱼ

Policy gradient loss:
  L(θ) = - (1/K) ∑ᵢ aᵢ · ∑ₜ log π_θ(yᵢₜ | x, yᵢ<ₜ)

Update: θ ← θ - α∇L(θ)
```

### Why This Simplification Works at Small Scale

1. **No KL needed**: The model is small enough that it won't drift catastrophically in a few RL steps. KL regularization solves a problem that doesn't exist here.

2. **On-policy is sufficient**: With 16 samples per problem and binary rewards, the signal-to-noise ratio is already decent. PPO's off-policy corrections add complexity without proportional benefit.

3. **Mean-shift is enough**: Z-score normalization can amplify noise when all rewards are similar. Mean subtraction is more stable for binary rewards.

### Tool Use During RL

The inference engine intercepts `<|python_start|>...<|python_end|>` tokens and executes the enclosed expression via a sandboxed calculator. Results are injected as `<|output_start|>result<|output_end|>`. The model learns to leverage this tool through RL — it discovers that using the calculator produces correct answers more reliably than mental arithmetic.

The calculator supports:
- Arithmetic expressions
- String `.count()` operations (for tasks like "how many r's in strawberry")
- Sandboxed execution with memory limits, timeouts, and blocked dangerous operations

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| num_samples | 16 | Enough for stable advantage estimation with binary rewards |
| max_new_tokens | 256 | GSM8K solutions rarely exceed this |
| temperature | 1.0 | Full exploration during rollout |
| top_k | 50 | Prevent degenerate low-probability tokens |
| init_lr_frac | 0.05 | Very small — RL updates should be gentle |
| LR schedule | Linear warmdown to 0 | |
| examples_per_step | 16 | Problems per gradient step |
| num_epochs | 1 | Single pass to avoid reward hacking |

---

## Phase 4: Self-Play Refinement (Experimental)

### Concept

After RL establishes that the model can solve problems and evaluate correctness, self-play extends this to domains without pre-existing datasets:

1. **Generate problems**: Model creates new problems in learned domains
2. **Solve problems**: Model generates candidate solutions
3. **Verify solutions**: Either through tool execution or model-as-judge
4. **Train on successes**: Policy gradient on verified correct solutions

This is related to SPIN (Self-Play Fine-Tuning) and SeRL (Self-Play RL):

- **SPIN**: The current model generates "negative" examples, human data provides "positive" examples, and the model learns to distinguish. Equivalent to DPO loss with self-generated negatives.
- **SeRL**: Two modules — self-instruction (generate problems) and self-rewarding (evaluate solutions) — bootstrap from limited data.

### When to Use Self-Play

Only pursue if:
1. Phase 3 RL shows reliable reward signal (>50% solve rate on training set)
2. The model can generate syntactically valid problems in the target domain
3. Verification is cheap (tool execution, not human judgment)

For a contest deadline, this phase is likely too experimental. Focus on Phases 1-3.

---

## Data Strategy

### The Mixture Principle

Each phase uses data calibrated to its purpose:

```
Phase 1 (Protocol): BROAD + STRUCTURAL
  └─ SmolTalk (460K general conversations)
  └─ MMLU auxiliary (100K × 3 epochs = 300K MC examples)
  └─ GSM8K (8K × 4 epochs = 32K math examples)
  └─ Spelling tasks (280K)
  └─ Identity data (2K)
  Total: ~1M examples, no quality filter

Phase 2 (SFT): NARROW + HIGH-QUALITY
  └─ SmolTalk (460K, but this IS the quality filter)
  └─ MMLU auxiliary (100K × 3 epochs)
  └─ GSM8K (8K × 4 epochs)
  └─ Identity data (2K × 2 epochs)
  └─ Spelling tasks (280K)
  Total: ~1M examples, assistant-only loss masking

Phase 3 (RL): TARGETED + VERIFIABLE
  └─ GSM8K train (8K problems, 16 rollouts each)
  └─ Any domain with binary reward function
  Total: 8K × 16 = 128K rollouts
```

### Identity Data

Synthetic conversations that teach the model who it is. Generated from templates with variations. 1-2K conversations, oversampled to 2× in the training mixture. This is cheap but important — without it, the model has no consistent identity and may claim to be GPT-4, Siri, etc.

### Scaling the Data

For small models, the bottleneck is format comprehension, not knowledge. Karpathy's data ratios are computed from the observation that:

- **MMLU** needs ~3 epochs before the model reliably maps options to letters
- **GSM8K** needs ~4 epochs before the model reliably uses tool syntax
- **SmolTalk** provides breadth — 1 epoch is sufficient

Oversampling structured tasks relative to free-form conversation is deliberate and important.

---

## Optimizer Design

### Muon + AdamW (Combined)

Nanochat uses a split optimizer:
- **Muon** for weight matrices (matrix_lr=0.02): Better for large matrix parameters
- **AdamW** for embeddings (embedding_lr=0.3) and unembeddings (unembedding_lr=0.004)

Learning rates are scaled by `1/√(model_dim / reference_dim)` to maintain stable updates as model width changes (µP-style transfer).

### Momentum Warm-Start

Critical insight: The optimizer state from pretraining contains valuable curvature information in the momentum and variance buffers. SFT loads these buffers but resets learning rates. This is why `warmup_ratio=0.0` works — the optimizer already has good momentum estimates.

### Weight Decay

Pretraining ramps weight decay from `0.2 × √(batch_size/ref_batch) × (ref_dim/model_dim)` down to zero during the warmdown phase. SFT continues with `weight_decay=0.0`. RL also uses zero weight decay. The model's norm is already well-calibrated from pretraining.

---

## Evaluation Framework

### During Training

| Metric | Frequency | What It Measures |
|--------|-----------|-----------------|
| Train loss (EMA) | Every step | Optimization health |
| Validation BPB | Every 200 steps | Generalization on held-out chat data |
| ChatCORE | Every 200 steps | Aggregate task performance (0-1 scale) |

### ChatCORE Breakdown

```
ChatCORE = mean of centered accuracies across 6 tasks:

Categorical (baseline=0.25):
  - ARC-Easy: Grade-school science (easy)
  - ARC-Challenge: Grade-school science (hard)
  - MMLU: Broad knowledge multiple-choice

Generative (baseline=0.0):
  - GSM8K: Grade-school math with tool use
  - HumanEval: Code generation
  - SpellingBee: Character-level reasoning

Centered accuracy = (acc - baseline) / (1 - baseline)
Score of 0 = random chance, 1 = perfect
```

### Post-Training Evaluation

After the full pipeline:
1. **DCLM CORE** — primary benchmark for GPT-2-class comparison
2. **ChatCORE** — conversational capability aggregate
3. **Manual chat testing** — qualitative sanity via CLI or WebUI

---

## Implementation Checklist

### For a Contest (2-3 day timeline on 1×L4):

```
Day 1 (hours 0-8):
  □ Train tokenizer (vocab=32768, ~2 min)
  □ Download data (80 shards, ~5 min)
  □ Pretrain depth=8 base model (~75 min)
  □ Pretrain depth=12 base model (~5 hrs)

Day 1-2 (hours 8-16):
  □ SFT on depth=8 checkpoint (~20 min)
  □ SFT on depth=12 checkpoint (~40 min)
  □ Evaluate both with ChatCORE
  □ Pick the better one for RL

Day 2 (hours 16-24):
  □ RL on GSM8K (1-2 hrs)
  □ Final evaluation
  □ Deploy web UI
  □ Generate report
```

### For Production (unlimited time, multi-GPU):

```
□ Pretrain depth=20-24 base model (2-3 hrs on 8×H100)
□ Midtraining phase (8 min)
□ SFT phase (7 min)
□ RL phase (1.5 hrs)
□ Self-play refinement (experimental, hours)
□ Full benchmark suite
□ Safety evaluation
□ Deploy with monitoring
```

---

## Key Principles

1. **Separate format from quality**. Midtraining teaches structure, SFT teaches substance. Don't conflate them.

2. **Mask what you don't control**. User prompts, tool outputs, and structural tokens should not contribute to loss during SFT. The model should only be supervised on what it actually generates at inference time.

3. **Warm-start everything**. Optimizer momentum, learning rates, model weights — carry forward as much state as possible between phases. Cold starts waste compute.

4. **RL only where you can verify**. Don't apply RL to open-ended generation. Use it exclusively for domains with automated correctness checking (math, code execution, factual lookup).

5. **Data quality dominates data quantity for SFT**. A small amount of excellent data beats a large amount of mediocre data. This is the opposite of pretraining, where scale is king.

6. **The model knows more than it shows**. Most failures in small models are protocol failures, not knowledge failures. Teach the protocol explicitly through structured data.

7. **Simple RL works**. You don't need PPO, KL penalties, value functions, or trust regions for small models doing simple tasks. REINFORCE with a group baseline is sufficient.

---

## Advanced Techniques (Beyond Nanochat Baseline)

The following techniques are drawn from SOTA research (2025-2026) and can be layered on top of the nanochat baseline for additional gains.

### NEFTune: Noise-Augmented Embeddings

Add uniform random noise to embedding vectors during SFT forward passes. Original results: LLaMA-2-7B on AlpacaEval jumped from 29.79% to 64.69%. The mechanism is regularization — models overfit less to instruction data specifics. **SymNoise** (2025) extends this with symmetric noise, pushing to 69.04%. Essentially free — no extra compute or data required.

Implementation: Add `noise ~ Uniform(-α, α)` to token embeddings after the embedding layer, only during training. Scale α ≈ 5/√(hidden_dim × seq_len).

### WSD Learning Rate Schedule

Warmup-Stable-Decay is replacing cosine as the preferred schedule. Three phases: warmup, constant plateau, then short decay. Key advantage: the stable phase can run indefinitely without pre-specifying total steps. You branch off with a decay phase whenever you want a checkpoint.

**WSM** (July 2025) goes further — skip online decay entirely and merge checkpoints from the stable phase with theoretically-derived weights. Shows +3.5% MATH, +2.9% HumanEval, +5.5% MMLU-Pro over standard WSD.

### Two-Stage Curriculum SFT

Grounded in cognitive science, this approach splits SFT into two stages:

1. **Stage 1**: Train on reasoning-enhanced data with explicit chain-of-thought. Build strong inferential capabilities.
2. **Stage 2**: Train on standard prompt-response pairs *without* intermediate reasoning. Model learns to apply reasoning implicitly.

This is particularly effective for small models (<1B params) where explicit reasoning chains serve as scaffolding that can later be internalized.

### Online vs Offline Preference Optimization

If pursuing DPO instead of GRPO, **online DPO** dramatically outperforms offline:
- AlpacaEval 2.0: 83.1% winrate (online) vs 53.2% (offline)
- MATH500: 58.7% vs 53.7%

Semi-online DPO offers a middle ground — generate new completions from the current policy periodically, rather than using a fixed preference dataset.

### Guided GRPO for Small Models (G2RPO-A)

Standard GRPO struggles with small models because they lack the base capability to generate enough correct solutions for meaningful advantage estimation. **G2RPO-A** injects ground-truth guidance into thinking trajectories during rollouts, with adaptive guidance strength that decreases as the model improves. Substantially outperforms vanilla GRPO on math and code benchmarks for <1B models.

### Scaffolded GRPO (Scaf-GRPO)

Progressive training framework: provide minimal hints only when independent learning plateaus. Tiered hints range from abstract concepts to concrete steps. Boosts Qwen2.5-Math-7B on AIME24 by 44.3% relative over vanilla GRPO. Particularly relevant for small models that need scaffolding to discover correct solution strategies.

### Prompt Curriculum Learning for RL

Select training prompts of **intermediate difficulty** — where the model has roughly 50% success rate — for maximum sample efficiency. Uses a concurrently-updated value model to estimate prompt difficulty on-policy. More effective than random prompt sampling for RL convergence.

### Knowledge Distillation

For small models, distillation from a larger teacher can be more effective than training from scratch:
- **Offline on-policy distillation**: Teacher and student share tokenizer; transfer at logit level
- **Distilling Step-by-Step**: Train on both labels AND rationales from teacher — outperforms larger models with less data
- **DA-KD** (ICML 2025): Dynamically adjust distillation dataset based on sample difficulty

### Model Merging (Post-RL)

After training separate RL specialists (math, code, reasoning), merge them:
- **TIES-Merging**: Resolves sign conflicts and removes redundant parameters before averaging. 1-5 point improvement over naive averaging.
- **DARE**: Randomly drop up to 90% of delta parameters and rescale. Combined with TIES, this is current best practice.
- Important: these methods work for SFT-trained deltas but are **mismatched for RL-trained models** — use with caution.

### Self-Play (SPIN)

Self-Play Fine-Tuning generates negative examples from the current model while using human data as positive examples. Equivalent to DPO loss with self-generated negatives. Can be iterated: each round's model generates new negatives for the next. Particularly effective when human preference data is scarce.

---

## The Karpathy Doctrine (Synthesized)

From the nanochat Discussion #1, the "Deep Dive into LLMs" talk, and the "2025 LLM Year in Review":

> *"The issue is not that the model doesn't have the knowledge, it's that it doesn't understand how Multiple Choice works to surface that knowledge."*

> *"RLHF is just barely RL."* — Reward models are "crappy proxy objectives." Contrast with AlphaGo where RL optimizes against a verifiable reward (winning). RLHF optimizes against a neural net's guess about preferences.

> *"Most of the capability progress of 2025 was defined by longer RL runs, not larger models."* — Scale is not the primary lever anymore.

His evolving view: RLVR (RL from Verifiable Rewards) is "the de facto new major stage" — unlike RLHF, verifiable rewards can't be gamed, allowing much longer optimization. Models trained via RLVR "learn to break down problem solving into intermediate calculations" and develop strategies "that would have been very difficult to teach directly."

But long-term, he's bearish on RL as currently practiced: reward functions are "super sus" and genuine breakthroughs require "fundamentally different learning mechanisms" — interactive environments where models "act and observe consequences."

---

## References

- [karpathy/nanochat](https://github.com/karpathy/nanochat) — source implementation
- [nanochat Discussion #1](https://github.com/karpathy/nanochat/discussions/1) — Karpathy's design philosophy
- [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300) — Group Relative Policy Optimization
- [G2RPO-A](https://arxiv.org/html/2508.13023v1) — Guided GRPO for small models
- [Scaf-GRPO](https://arxiv.org/abs/2510.19807) — Scaffolded GRPO with progressive hints
- [MS-GRPO](https://openreview.net/forum?id=ktHj6YazEE) — Multi-scale advantage aggregation
- [SPIN](https://arxiv.org/abs/2401.01335) — Self-Play Fine-Tuning
- [SeRL](https://arxiv.org/abs/2505.20347) — Self-Play RL with limited data
- [Post-Training for SLMs via Knowledge Distillation](https://arxiv.org/html/2509.26497) — Two-stage curriculum SFT
- [Prompt Curriculum Learning](https://openreview.net/forum?id=zqOCacBD3P) — Difficulty-aware prompt selection for RL
- [NEFTune](https://arxiv.org/abs/2310.05914) — Noisy embeddings improve instruction finetuning
- [WSM: Decay-Free LR via Checkpoint Merging](https://arxiv.org/html/2507.17634) — Beyond WSD scheduling
- [TIES-Merging](https://developer.nvidia.com/blog/an-introduction-to-model-merging-for-llms/) — Model merging for specialists
- [Tulu 3](https://arxiv.org/abs/2411.15124) — Open LM post-training recipes
- [RLVR Explained](https://www.promptfoo.dev/blog/rlvr-explained/) — Verifiable rewards for RL
- [Online vs Offline RL for LLMs](https://cameronrwolfe.substack.com/p/online-rl) — Why online DPO dominates
- [DA-KD: Difficulty-Aware Knowledge Distillation](https://icml.cc/virtual/2025/poster/45516) — ICML 2025
- [Karpathy: "RLHF is just barely RL"](https://x.com/karpathy) — Aug 2024 thread
- [Karpathy: 2025 LLM Year in Review](https://karpathy.ai) — Dec 2025 blog post
