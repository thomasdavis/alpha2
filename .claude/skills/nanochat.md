---
name: nanochat
description: Train, evaluate, and iterate on a chat model that can actually converse. Deploys to L4 GPU, trains on SODA data, generates inference samples, fixes issues, and iterates until the model talks well.
disable-model-invocation: true
---

# NanoChat — Chat Model Training Loop

You are a training agent. Your goal: make a small GPT model that can hold real conversations — answer questions, discuss topics, respond naturally. You deploy to an L4 GPU, train, evaluate samples, diagnose issues, fix them, and repeat.

## Architecture

- **Model**: GPT with SwiGLU, 16 layers, dim=768, 12 heads, ~120M params
- **Tokenizer**: bpe-chat-4k (4000 vocab with `<|user|>`, `<|assistant|>`, `<|end_of_text|>` special tokens)
- **Data**: SODA dataset (868MB, high-quality multi-turn conversations)
- **Backend**: Helios (Vulkan compute shaders) on NVIDIA L4 GPU
- **Block size**: 512 tokens
- **Activation checkpointing**: enabled (trades compute for memory)
- **Mixed precision**: fp16 enabled

## Target Instance

SSH to the L4 benchmark instance:
```
ssh -i ~/.ssh/google_compute_engine ajax@136.113.161.152
```
- Remote dir: `~/alpha`
- VK ICD: `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd_headless.json`
- Node: v22

## The Loop

### Step 1: Build and Deploy

Build locally, sync to remote:
```bash
# Local build
npm run build

# Full sync (excludes large data files already on remote)
scripts/nanochat-loop.sh --iters=20000
```

Or manually sync + run:
```bash
rsync -az --delete \
  --exclude='node_modules' --exclude='.git' --exclude='runs' \
  --exclude='data/concordance*' --exclude='data/soda.txt' \
  --exclude='data/chat.txt' --exclude='.turbo' --exclude='perf' \
  -e "ssh -i ~/.ssh/google_compute_engine" \
  ./ ajax@136.113.161.152:~/alpha/

ssh -i ~/.ssh/google_compute_engine ajax@136.113.161.152 \
  'cd ~/alpha && npm install && npm run build'
```

### Step 2: Train

Run on the L4:
```bash
ssh -i ~/.ssh/google_compute_engine ajax@136.113.161.152 'cd ~/alpha && \
  source .env.local && \
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd_headless.json && \
  bash scripts/train-nanochat.sh 2>&1'
```

To resume from checkpoint:
```bash
NANOCHAT_RESUME=runs/<run>/checkpoint-N.json bash scripts/train-nanochat.sh
```

Training reports to https://alpha.omegaai.dev and Discord webhook.

### Step 3: Evaluate Samples

After training (or during, via checkpoints), run inference:
```bash
node apps/cli/dist/main.js sample \
  --checkpoint=runs/<run>/checkpoint-N.json \
  --prompt='<|user|> Hello! Who are you? <|assistant|>' \
  --steps=200 --temperature=0.7 --topk=40
```

Test multiple prompts to evaluate quality:
```
<|user|> What is photosynthesis? <|assistant|>
<|user|> Tell me a joke. <|assistant|>
<|user|> I'm feeling sad today. <|assistant|>
<|user|> How do computers work? <|assistant|>
<|user|> What's the capital of France? <|assistant|>
```

### Step 4: Diagnose and Fix

Common issues and fixes:

**Model outputs gibberish**: Too few training steps, lr too high, or data issue.
- Fix: Train longer, reduce lr, check data integrity.

**Model repeats itself**: Temperature too low, or model collapsed.
- Fix: Increase temperature, check for loss plateau, increase dropout.

**Model doesn't follow chat format**: Tokenizer doesn't have special tokens, or data format wrong.
- Fix: Verify tokenizer has `<|user|>`, `<|assistant|>`, `<|end_of_text|>` in vocab. Check data with `scripts/validate-chat-data.ts`.

**Loss stops decreasing**: Learning rate schedule exhausted, or model capacity saturated.
- Fix: Increase model size, use cosine annealing with restarts, increase data.

**OOM on L4**: Model too large for 24GB.
- Fix: Reduce batch size, enable activation checkpointing (`--checkpoint=true`), reduce block size.

**GPU smoke test hang**: Known bug with BPE + data >= ~33MB.
- Fix: Use cached tokenizer artifacts (`--tokenizerArtifacts=artifacts/nanochat-tok.json`), or use a smaller data chunk for tokenizer training.

### Step 5: Iterate

After evaluating, decide what to change:
1. **More training**: Increase `--iters` and resume from checkpoint
2. **Better data**: Mix in more datasets, clean low-quality conversations
3. **Hyperparameter tuning**: Adjust lr, dropout, batch size, warmup
4. **Model architecture**: Change layers/dim/heads if model is too small/large
5. **Data augmentation**: Add identity conversations, factual Q&A, reasoning tasks

Go back to Step 1 and repeat.

## Key Files

- `packages/core/src/domains.ts` — nanochat domain config
- `scripts/train-nanochat.sh` — training script
- `scripts/nanochat-loop.sh` — deploy + train loop
- `packages/train/src/trainer.ts` — training loop implementation
- `packages/train/src/sample.ts` — autograd-based sampling
- `packages/inference/src/engine.ts` — fast KV-cache inference
- `packages/model/src/gpt.ts` — GPT architecture
- `packages/tokenizers/src/bpe.ts` — BPE tokenizer
- `data/soda.txt` — SODA conversational dataset (868MB)

## Chat Format

```
<|user|> user message here <|assistant|> assistant response here<|end_of_text|>
```

Multi-turn:
```
<|user|> first question <|assistant|> first answer<|end_of_text|>
<|user|> followup question <|assistant|> followup answer<|end_of_text|>
```

## Quality Targets

- **Step 1000**: Model should produce grammatical English in chat format
- **Step 5000**: Model should give topically relevant responses
- **Step 10000**: Model should handle multi-turn conversations coherently
- **Step 20000**: Model should discuss diverse topics with reasonable accuracy

## Rules

1. Always check sample quality after significant training progress
2. Never train blind — evaluate inference checkpoints regularly
3. If loss spikes, investigate before continuing
4. One change at a time for hyperparameter tuning
5. Document what worked and what didn't
6. Use the fast inference engine for quick sampling (not autograd)
7. Keep training data clean — validate chat format before training
