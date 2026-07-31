#!/usr/bin/env bash
# Deterministic development evaluation for one repair-v2 checkpoint.
# The sealed-final suite must never be passed here during checkpoint selection.

set -euo pipefail

checkpoint=${1:?checkpoint required}
chat_prompts=${2:?development chat prompts required}
out_dir=${3:?new output directory required}
max_tokens=${4:-128}
qa_probe=${5:-/mnt/donto-data/alpha-corpora/frozen-eval-v1/candidates/closed-book-qa.jsonl}

for required in "$checkpoint" "$chat_prompts" "$qa_probe" apps/cli/dist/main.js; do
  [[ -f $required ]] || { echo "required file missing: $required" >&2; exit 1; }
done
[[ ! -e $out_dir ]] || { echo "output directory already exists: $out_dir" >&2; exit 1; }
[[ $max_tokens =~ ^[1-9][0-9]*$ ]] || { echo "MAX_TOKENS must be a positive integer" >&2; exit 2; }

node apps/cli/dist/main.js eval-frozen \
  --checkpoint="$checkpoint" \
  --chat="$chat_prompts" \
  --qa="$qa_probe" \
  --out="$out_dir" \
  --maxTokens="$max_tokens" \
  --qaMaxTokens=1 \
  --qaLimit=1

npx tsx scripts/audit_frozen_chat_failures.ts \
  --prompts "$chat_prompts" \
  --results "$out_dir/chat-results.jsonl" \
  --summary "$out_dir/summary.json" \
  --out "$out_dir/stratified-audit.json"
