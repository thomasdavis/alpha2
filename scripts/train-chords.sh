#!/usr/bin/env bash
# Train a chord progression model.
#
# ~120K params, ~20 minutes on CPU.
# Uses word-level tokenizer (vocab ~170), small architecture.

set -euo pipefail
cd "$(dirname "$0")/.."

npx tsx apps/cli/src/main.ts train \
  --data=data/chords.txt \
  --domain=chords \
  --iters=2000 \
  --batch=8 \
  --lr=0.001 \
  --evalInterval=200 \
  --evalIters=10 \
  --runDir=outputs/chords-run
