#!/usr/bin/env bash
# Train a GPT-2 scale model on concordance-v2 (Strong's Bible concordance).
#
# ~788MB text, BPE-64k tokenizer, GPT-2 architecture (117M params).
# Requires GPU backend (helios) — runs on L4/A100/H100.
# Estimated ~33 minutes for 2000 steps on L4 GPU.
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

node --expose-gc apps/cli/dist/main.js train \
  --data=data/concordance-v2.txt \
  --domain=concordance \
  --tokenizer=bpe-64k \
  --backend=helios \
  --iters=2000 \
  --batch=4 \
  --block=256 \
  --dim=768 \
  --heads=12 \
  --layers=12 \
  --lr=3e-4 \
  --gradClip=1.0 \
  --evalInterval=200 \
  "$@"
