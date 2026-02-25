# Chat Model Training Plan

## Goal

Train a language model on conversational data so it can generate coherent chat responses via `alpha.omegaai.dev/chat`.

## What We Have

- **Training data**: `data/chat_combined.txt` (64MB, 213K conversations) — multi-turn dialogues with `<|user|>` / `<|assistant|>` turn markers
- **Chat domain config** in `packages/core/src/domains.ts` with BPE-4K tokenizer, 6L/256D/8H model defaults
- **Working training pipeline**: CLI, helios GPU backend (Vulkan/SPIR-V), remote metrics, checkpoint upload to Railway
- **Working inference server**: loads checkpoints, serves streaming chat via AI SDK, OpenAI-compatible endpoints
- **Cloud GPU script**: `scripts/gcp_train.py` (GCP A100 80GB ~$1.10/hr)

## Training Config

```bash
python scripts/gcp_train.py \
  --data data/chat_combined.txt \
  --domain chat \
  --iters 50000 \
  --batch 16 \
  --block 256 \
  --dim 256 \
  --heads 8 \
  --layers 6 \
  --backend helios \
  --eval-interval 500 \
  --stop-after
```

Chat domain defaults (set in `packages/core/src/domains.ts`):
- `lr: 3e-4`, `lrMin: 3e-5` (cosine decay with 10:1 ratio)
- `warmupIters: 1000` (linear warmup)
- `beta2: 0.95`, `weightDecay: 0.1`
- `gradClip: 1.0`, `batchSize: 16`
- `sampleInterval: 500`

Expected: ~7M params, ~3 epochs through dataset, ~30 min on A100, ~$0.55

## What "Working" Looks Like

1. Loss below 3.0 (ideally ~2.5)
2. Sample generations that form real English and follow turn structure
3. Coherent multi-word responses to chat prompts
4. Checkpoint uploaded and accessible at `alpha.omegaai.dev/chat`

## Strategy

1. **Pilot**: 5K steps — verify loss < 4.0 and coherent samples
2. **Full run**: 50K steps if pilot is healthy
3. Verify at `alpha.omegaai.dev/chat`

## Local Pilot Results (Intel Iris Xe iGPU)

Run ID: `20260223085609_er4v` | 4L/128D/4H | BPE-4K | 3000 steps | chat_tiny.txt (2MB)

| Metric | Step 100 | Step 500 | Step 1000 | Step 2000 | Step 3000 |
|--------|----------|----------|-----------|-----------|-----------|
| Loss | 8.0 | 7.3 | 6.5 | 5.2 | 5.2 |
| Grad norm | 0.5-0.7 | 0.9-1.1 | 1.5-2.0 | 5-70 | 50-3000+ |
| LR | 6e-5 | 1.5e-4 | 3e-4 (peak) | 1.6e-4 | 3e-5 (min) |

**Verdict:** Improved hyperparams (warmup, cosine decay, beta2=0.95) completely eliminated the divergence seen in 5 prior attempts. Model produces English-ish fragments but needs bigger model + full dataset on cloud GPU for coherence.
