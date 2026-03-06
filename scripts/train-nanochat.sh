#!/usr/bin/env bash
# Train a chat model on SODA conversational data.
#
# ~845MB high-quality multi-turn conversations, BPE-chat-4k tokenizer,
# SwiGLU GPT with 16 layers, dim=512 (~56M params).
# Designed for L4 GPU (24GB VRAM).
#
# For remote dashboard reporting + checkpoint upload, set:
#   ALPHA_REMOTE_URL=https://alpha.omegaai.dev
#   ALPHA_REMOTE_SECRET=<your-secret>

set -euo pipefail
cd "$(dirname "$0")/.."

# Load .env.local if present (for ALPHA_REMOTE_URL/SECRET)
if [ -f .env.local ]; then
  set -a
  source .env.local
  set +a
fi

# Default to SODA, allow override
DATA="${NANOCHAT_DATA:-data/soda-chat.txt}"
ITERS="${NANOCHAT_ITERS:-20000}"
RESUME="${NANOCHAT_RESUME:-}"

RESUME_FLAG=""
if [ -n "$RESUME" ]; then
  RESUME_FLAG="--resume=$RESUME"
fi

node --expose-gc apps/cli/dist/main.js train \
  --data="$DATA" \
  --domain=nanochat \
  --tokenizer=bpe-chat-4k \
  --backend=helios \
  --iters="$ITERS" \
  --batch=4 \
  --block=512 \
  --dim=512 \
  --heads=8 \
  --layers=16 \
  --activation=swiglu \
  --lr=6e-4 \
  --lrMin=6e-5 \
  --warmupIters=500 \
  --accumSteps=1 \
  --beta2=0.999 \
  --gradClip=1.0 \
  --evalInterval=500 \
  --sampleInterval=1000 \
  --spikeThreshold=0 \
  --packed=true \
  --dropout=0 \
  --checkpoint=true \
  --fp16=false \
  --gpuProfile=none \
  --minGpuSize=1 \
  --postSamples=true \
  --maxDatasetPasses=50 \
  --symbio=false \
  --syncEvery=1 \
  --gcEvery=1 \
  $RESUME_FLAG \
  "$@"
