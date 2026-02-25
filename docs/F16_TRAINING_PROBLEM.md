# GPU Training Instability — Diagnosis & Fix

## Status: RESOLVED

The root cause was **not** f16 precision. The Helios GPU backend uses **f32 compute throughout** — all matmuls, softmax, CE, optimizer states are f32. The f16 storage kernels exist in `kernels.ts` but are never dispatched.

## Actual Root Causes

### 1. Aggressive gradient clipping (gradClip=1.0)

With a healthy baseline grad_norm of ~2 at step 500, gradClip=1.0 fires on *every step* and distorts Adam's second moment estimate `v`. This causes miscalibrated updates that compound into instability. By step 1000, grad norms reach 300+ and clipCoef = 1/300 = 0.003, making effective LR ~3e-7.

**Fix**: Raised gradClip to 5.0 for chat domain. Allows healthy gradients through while still protecting against spikes.

### 2. Weight decay on LayerNorm params

Weight decay was applied uniformly to all parameters including LayerNorm weights/biases. Standard practice excludes these since LN params have different optimization dynamics.

**Fix**: Added `noDecayNames` set to AdamW. LN and bias params get `weightDecay=0`.

### 3. Numerically imprecise CE forward path

The CE forward used a 5-op chain (`softmax → clamp_min → log → pick → negate`) which loses precision for small probabilities and requires intermediate buffers.

**Fix**: Fused CE forward kernel (`ce_fwd_fused`) computes log-sum-exp directly in a single kernel dispatch.

## Changes Made

- `packages/core/src/domains.ts`: gradClip 1.0 → 5.0 for chat domain
- `packages/train/src/optimizers.ts`: noDecayNames support in AdamW
- `apps/cli/src/commands/train.ts`: Build no-decay set from model config
- `packages/helios/src/kernels.ts`: Fused CE forward SPIR-V kernel
- `packages/helios/src/backend.ts`: Wire fused CE kernel
- `packages/train/src/trainer.ts`: Per-layer grad norm diagnostics (trace mode)
