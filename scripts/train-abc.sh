#!/usr/bin/env bash
# Train an ABC music notation model.
#
# 1K folk tunes (~282KB), char tokenizer (vocab ~103).
# 4 layers, 128 dim, 4 heads, blockSize 256.
# ~282KB / (256*8) = ~138 batches/epoch → 2000 iters ≈ 14.5 epochs.
# LR 0.0003 (lower for stability with char-level data).
# Estimated ~14 hours on CPU.

set -euo pipefail
cd "$(dirname "$0")/.."

npx tsx apps/cli/src/main.ts train \
  --data=data/abc-small.txt \
  --domain=abc \
  --iters=2000 \
  --batch=8 \
  --lr=0.0003 \
  --evalInterval=200 \
  --evalIters=10 \
  --runDir=outputs/abc-char-run2
