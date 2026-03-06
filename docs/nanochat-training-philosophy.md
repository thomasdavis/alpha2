# Nanochat Training Philosophy

Getting a small GPT to hold conversations on an L4 GPU using Helios (Vulkan compute shaders).

## Goal
Train a 56M parameter SwiGLU GPT on conversational data (SODA/super_chat format) until it can produce coherent multi-turn dialogue. Target: GPT-2 level fluency at chat scale.

## Architecture
- **Model**: GPT with SwiGLU activation, 16 layers, dim=512, 8 heads, block=512
- **Tokenizer**: BPE-chat-4k (4000 vocab, chat-specific with `<|user|>`, `<|assistant|>`, `<|end_of_text|>` tokens)
- **Parameters**: ~56M (no weight tying — lmHead separate from wte)
- **FFN dim**: ceil(8/3 * 512 / 64) * 64 = 1408 (SwiGLU formula)

## Training Configuration (Current Best)
```
lr=3e-4, lrMin=3e-5, warmupIters=500
batch=4, block=512, packed=true
checkpoint=true (activation checkpointing)
gradClip=1.0, spikeThreshold=10000
beta2=0.999, optimizer=adamw
```

## Key Learnings

### Learning Rate
- **lr=6e-4 is too high** for this model+data combo on Helios. Causes gradient explosion around step 100-160.
- **lr=3e-4 is stable** with the same architecture and data. Gradient norms stay around 3400-3500.
- The instability manifests as a sudden grad_norm spike (3500 → 40000+) followed by cascading NaN or divergence.

### Cooperative Matmul (Tensor Cores via VK_KHR_cooperative_matrix)
Helios supports cooperative matrix multiply which uses tensor cores on NVIDIA L4. This converts f32 inputs to f16 for the matrix multiply. Key findings:

1. **Forward pass f16 is lossy enough to destabilize training.** Even with perfect f16 clamping and f32 backward, the f16 forward pass changes the loss landscape enough across 16 layers to cause divergence at lr=3e-4. This was confirmed by comparing:
   - No coop (f32 everywhere): stable at lr=3e-4, ~2% intermittent NaN
   - Coop forward only (f16 fwd, f32 bwd): diverges at step ~104
   - Full coop (f16 fwd + bwd): diverges at step ~100

2. **f16 backward pass is catastrophic.** Gradient values can exceed f16 max (65504), producing Inf. Even with clamping, clamped gradients are incorrect and corrupt optimizer state. Fixed by disabling coop during backward, but forward alone is still problematic.

3. **Coop matmul gives ~10% speedup** (1100 vs 1000 tok/s) but the precision cost is too high for this model size. Larger models with naturally smaller per-element values may tolerate f16 better.

4. **Cache eviction bug (fixed)**: The f16 input cache in `getCoopInputBuffer()` could evict buffers mid-operation when `castDtype()` triggered a graph auto-flush at MAX_PENDING_OPS=2048. Fixed by moving eviction to safe points only.

**Current decision**: Coop matmul disabled (`HELIOS_DISABLE_COOP_MAT=1`) for training stability. Investigate mixed-precision (fp16 forward, f32 backward with loss scaling) as a proper solution later.

### Gradient Instability
- ~2-7% of steps produce NaN gradients even with f32 matmul. This is a known Helios/SwiGLU interaction.
- `spikeThreshold=10000` catches extreme spikes and skips those optimizer steps.
- The spike handler backs off learning rate progressively (lr_scale: 1.0 → 0.5 → 0.25 → 0.125 → 0.1 min).
- Loss remains finite through NaN gradient steps; model recovers if spikes are intermittent.

### Data
- **super_chat.txt** (94MB): Multi-turn conversations with `<|user|>`/`<|assistant|>` format
- Sequence packing enabled: ~11375 steps per epoch with batch=4, block=512
- Delimiter-aware 90/10 train/val split (splits on `<|end_of_text|>` boundaries)
- ~24.8M tokens estimated — very low tokens/param ratio (0.73). Model will overfit eventually; maxDatasetPasses=50 limits this.

### GPU Memory
- L4 has 24GB VRAM; model + optimizer + activations fit with checkpoint=true
- Activation checkpointing trades ~33% more compute for O(layers + ops_per_layer) memory
- `syncEvery=1` and `gcEvery=1` ensure timely GPU buffer cleanup

## Monitoring
Training metrics and events stream to `https://alpha.omegaai.dev` for remote monitoring.

### Events API
`GET /api/runs/{runId}/events` returns diagnostic events:
- `training_started` — config snapshot
- `grad_norm_nan` — every NaN occurrence (rate-limited: first 5, then every 10th), includes cumulative count
- `spike_skip` — every gradient spike skip with grad_norm, threshold, lr_scale
- `gpu_diagnostics` — emitted at eval intervals (every 500 steps): buffer pool stats, VRAM, coop matmul stats, NaN count
- `checkpoint_saved` / `checkpoint_uploaded` — model snapshots
- `sample_generation_failed` — inference errors

### Metrics API
Step metrics include: loss, lr, gradNorm, tokens/sec, ms/iter, gpu_ops_count, clip_coef, timing breakdowns (fwd/bwd/optim/flush), gpu_mem_pool_mb.

## Workflow
1. Launch training on L4 bench instance (136.113.161.152)
2. Monitor via events API and SSH log tailing
3. Checkpoints saved every 500 steps with remote upload
4. Inference samples generated every 1000 steps
5. Eval on held-out val split every 500 steps

## What "GPT-2 Level" Means
For a 56M chat model, we're targeting:
- Grammatically correct responses
- Appropriate turn-taking (responds as assistant, not user)
- Topic coherence within a conversation
- Reasonable diversity (not just repeating training data)
- Loss below ~4.0 on held-out validation

We're NOT targeting:
- Factual accuracy
- Complex reasoning
- Long-term coherence across many turns
- Instruction following beyond basic chat
