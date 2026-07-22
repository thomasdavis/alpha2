#!/usr/bin/env bash
#
# End-to-end proof of the from-scratch → HuggingFace chain (GOAL.md Stage 3, G3):
#
#   Alpha train (cpu_ref, Llama-form)  →  ALPH checkpoint
#     →  alpha export-hf               →  model.safetensors + config.json + tokenizer
#     →  alpha logits (cpu_ref)        →  alpha_logits.json
#     →  scripts/verify_hf_export.py   →  Alpha-forward == transformers-forward
#
# This is a WIRING test (tiny model, ~30 steps) — it proves the plumbing before
# any GPU money is spent, NOT model quality. Re-runnable; all outputs under $RUN.
#
# Prereqs: `nice -n19 npx tsc -b` (built CLI dist), and the uv venv at
# /mnt/donto-data/alpha-corpora/.venv with torch-cpu + transformers + safetensors.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="${RUN:-/mnt/donto-data/alpha-runs/g3-e2e}"
VENV="/mnt/donto-data/alpha-corpora/.venv"
CORPUS_SRC="/mnt/donto-data/alpha-corpora/pretrain-text/pretrain-000.txt"
CLI="node $ROOT/apps/cli/dist/main.js"

mkdir -p "$RUN"
[ -f "$RUN/corpus.txt" ] || nice -n19 head -c 5000000 "$CORPUS_SRC" > "$RUN/corpus.txt"

echo "== 1/4 train tiny alpha_llama (4L/128d/4H rmsnorm+rope+tied+swiglu, bpe-byte-4k) =="
nice -n19 $CLI train \
  --data="$RUN/corpus.txt" --tokenizer=bpe-byte-4k \
  --layers=4 --dim=128 --heads=4 --block=128 \
  --activation=swiglu --normType=rmsnorm --posEnc=rope --tieEmbeddings=true --ropeTheta=10000 \
  --steps=30 --batch=8 --lr=3e-4 --warmupIters=2 --seed=42 \
  --evalInterval=30 --evalIters=3 --backend=cpu_ref --postSamples=false --remote=false \
  --runDir="$RUN/run"

CKPT="$RUN/run/checkpoint-30.json"

echo "== 2/4 export to HF LlamaForCausalLM =="
nice -n19 $CLI export-hf --checkpoint="$CKPT" --out="$RUN/hf"

echo "== 3/4 dump Alpha cpu_ref logits for the golden-token test =="
nice -n19 $CLI logits --checkpoint="$CKPT" --prompt-file="$RUN/prompts.txt" --json --out="$RUN/alpha_logits.json"

echo "== 4/4 verify Alpha-forward == transformers-forward =="
"$VENV/bin/python" "$ROOT/scripts/verify_hf_export.py" \
  --export-dir="$RUN/hf" --alpha-logits="$RUN/alpha_logits.json" --tol=1e-3
