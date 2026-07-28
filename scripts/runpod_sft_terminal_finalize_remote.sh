#!/usr/bin/env bash
# Fail-closed terminal verification/evaluation/export pipeline, executed on the RunPod host.

set -euo pipefail

mode=${1:?usage: runpod_sft_terminal_finalize_remote.sh MODE REPO RUN SELECTION BASE_EVAL FROZEN_ROOT PYTHON SOURCE_COMMIT}
repo=${2:?remote repo required}
run=${3:?remote run required}
selection=${4:?SFT selection report required}
base_eval=${5:?base frozen-eval directory required}
frozen_root=${6:?frozen-eval root required}
python_bin=${7:?verification Python required}
source_commit=${8:?source commit required}

[[ $mode == preflight || $mode == finalize ]] || { echo "invalid mode: $mode" >&2; exit 2; }
[[ $repo == /workspace/alpha2 ]] || { echo "unexpected repo: $repo" >&2; exit 2; }
[[ $run =~ ^/workspace/alpha2/runs/[A-Za-z0-9._-]+$ ]] || { echo "invalid run: $run" >&2; exit 2; }
[[ $selection =~ ^/workspace/alpha2/runs/[A-Za-z0-9._/-]+\.json$ ]] || { echo "invalid selector: $selection" >&2; exit 2; }
[[ $base_eval =~ ^/workspace/alpha2/runs/[A-Za-z0-9._-]+$ ]] || { echo "invalid base eval: $base_eval" >&2; exit 2; }
[[ $frozen_root == /runpod/data/frozen-eval-v1 ]] || { echo "unexpected frozen root: $frozen_root" >&2; exit 2; }
[[ $python_bin =~ ^/workspace/[A-Za-z0-9._/-]+/bin/python$ ]] || { echo "invalid Python: $python_bin" >&2; exit 2; }
[[ $source_commit =~ ^[0-9a-f]{40}$ ]] || { echo "invalid source commit: $source_commit" >&2; exit 2; }

checkpoint="$run/checkpoint-30322.json"
verification="$run/terminal-sft-verification.json"
analysis="$run/flagship-sft-analysis.json"
chat_eval="$run/frozen-eval-chat"
pair_analysis="$run/frozen-eval-pair-analysis.json"
hf_export="$run/hf-alpha-60m-chat"
alpha_logits="$run/chat-alpha-logits.json"
hf_parity_log="$run/hf-export-parity.log"
status_file="$run/terminal-finalizer-status.json"
manifest_file="$run/terminal-artifact-sha256.txt"

required_files=(
  "$repo/apps/cli/dist/main.js"
  "$repo/scripts/verify_flagship_sft_inputs.ts"
  "$repo/scripts/analyze_flagship_sft.ts"
  "$repo/scripts/analyze_frozen_eval_pair.ts"
  "$repo/scripts/verify_hf_export.py"
  "$selection"
  "$base_eval/summary.json"
  "$base_eval/chat-results.jsonl"
  "$base_eval/qa-results.jsonl"
  "$frozen_root/MANIFEST.json"
  "$frozen_root/final/chat-prompts.jsonl"
  "$frozen_root/final/closed-book-qa.jsonl"
  "$python_bin"
  "$run/sft-contract.json"
  "$run/config.json"
  "$run/metrics.jsonl"
)
for file in "${required_files[@]}"; do
  [[ -f $file ]] || { echo "required file missing: $file" >&2; exit 2; }
done

cd "$repo"
actual_commit=$(git rev-parse HEAD)
[[ $actual_commit == "$source_commit" ]] || {
  echo "remote source $actual_commit != contracted $source_commit" >&2
  exit 2
}
"$python_bin" -c 'import numpy, safetensors, torch, transformers; print("terminal Python dependencies=PASS")'

if [[ $mode == preflight ]]; then
  rows=$(wc -l < "$run/metrics.jsonl")
  printf 'terminal finalizer preflight=PASS source=%s rows=%s run=%s\n' "$actual_commit" "$rows" "$run"
  exit 0
fi

rows=$(wc -l < "$run/metrics.jsonl")
[[ $rows == 30322 ]] || { echo "terminal metric rows $rows != 30322" >&2; exit 3; }
[[ -f $checkpoint ]] || { echo "terminal checkpoint missing: $checkpoint" >&2; exit 3; }
train_pid=$(ps -eo pid=,comm=,args= | awk -v run="$run" \
  '$2 == "node" && index($0, "apps/cli/dist/main.js train") && index($0, run) { print $1; exit }')
[[ -z $train_pid ]] || { echo "trainer PID $train_pid is still active" >&2; exit 3; }

echo "== terminal input/checkpoint audit =="
if [[ ! -f $verification ]]; then
  verification_tmp="$verification.$$.tmp"
  nice -n 19 ionice -c3 npx tsx scripts/verify_flagship_sft_inputs.ts \
    --data /runpod/data/alpha-sft-v2/sft-v2.txt \
    --manifest /runpod/data/alpha-sft-v2/sft-v2.txt.manifest.json \
    --lengthAudit /runpod/data/alpha-sft-v2/length-audit.json \
    --maskAudit /runpod/data/alpha-sft-v2/mask-audit.json \
    --tokenizer /workspace/alpha2/artifacts/g2-bpe-byte-12k.json \
    --baseCheckpoint "$checkpoint" \
    --expectedBaseStep 30322 > "$verification_tmp"
  mv "$verification_tmp" "$verification"
fi
node -e '
  const fs = require("fs");
  const value = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
  if (value.result !== "PASS" || value.base_checkpoint?.step !== 30322 ||
      value.base_checkpoint?.parameter_elements !== 57688576 ||
      value.base_checkpoint?.finite_parameter_elements !== 57688576) process.exit(1);
' "$verification"

echo "== terminal SFT analyzer =="
if [[ ! -f $analysis ]]; then
  analysis_tmp="$analysis.$$.tmp"
  nice -n 19 ionice -c3 npx tsx scripts/analyze_flagship_sft.ts \
    --run "$run" \
    --out "$analysis_tmp" \
    --sourceCommit "$source_commit" \
    --selectionReport "$selection" \
    --terminalVerification "$verification"
  mv "$analysis_tmp" "$analysis"
fi
node -e '
  const fs = require("fs");
  const value = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
  if (value.result !== "PASS" || value.rows !== 30322 || value.checkpoint?.parameter_elements !== 57688576) process.exit(1);
' "$analysis"

echo "== frozen 100-chat / 200-QA evaluation =="
if [[ ! -f $chat_eval/summary.json ]]; then
  set +e
  nice -n 19 ionice -c3 node apps/cli/dist/main.js eval-frozen \
    --checkpoint="$checkpoint" \
    --chat="$frozen_root/final/chat-prompts.jsonl" \
    --qa="$frozen_root/final/closed-book-qa.jsonl" \
    --out="$chat_eval" \
    --maxTokens=128 \
    --qaMaxTokens=64 2>&1 | tee "$run/frozen-eval-chat.log"
  eval_status=${PIPESTATUS[0]}
  set -e
  [[ $eval_status == 0 ]] || { echo "frozen evaluation failed: $eval_status" >&2; exit "$eval_status"; }
fi

echo "== recomputed frozen base/chat machine gate =="
pair_tmp="$pair_analysis.$$.tmp"
set +e
nice -n 19 ionice -c3 npx tsx scripts/analyze_frozen_eval_pair.ts \
  --base "$base_eval" \
  --chat "$chat_eval" \
  --manifest "$frozen_root/MANIFEST.json" \
  --out "$pair_tmp"
pair_status=$?
set -e
[[ -f $pair_tmp ]] || { echo "pair analyzer produced no report" >&2; exit 4; }
pair_result=$(node -e '
  const fs = require("fs");
  const value = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
  process.stdout.write(value.result);
' "$pair_tmp")
if [[ $pair_status == 0 && $pair_result != PASS ]] || [[ $pair_status != 0 && $pair_result != FAIL ]]; then
  echo "pair analyzer exit/result mismatch: $pair_status/$pair_result" >&2
  exit 4
fi
if [[ -f $pair_analysis ]]; then
  cmp -s "$pair_tmp" "$pair_analysis" || { echo "existing pair analysis differs" >&2; exit 4; }
  rm -- "$pair_tmp"
else
  mv "$pair_tmp" "$pair_analysis"
fi

echo "== standard Hugging Face export =="
if [[ ! -d $hf_export ]]; then
  hf_tmp="$hf_export.$$.tmp"
  nice -n 19 ionice -c3 node apps/cli/dist/main.js export-hf \
    --checkpoint="$checkpoint" --out="$hf_tmp"
  mv "$hf_tmp" "$hf_export"
fi
for file in model.safetensors config.json generation_config.json tokenizer.json tokenizer_config.json chat_template.jinja; do
  [[ -f $hf_export/$file ]] || { echo "HF export missing $file" >&2; exit 5; }
done

echo "== Alpha/Transformers terminal export parity =="
if [[ ! -f $alpha_logits ]]; then
  logits_tmp="$alpha_logits.$$.tmp"
  nice -n 19 ionice -c3 node apps/cli/dist/main.js logits \
    --checkpoint="$checkpoint" --prompt="Hello" --json --out="$logits_tmp"
  mv "$logits_tmp" "$alpha_logits"
fi
parity_tmp="$hf_parity_log.$$.tmp"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 19 ionice -c3 "$python_bin" scripts/verify_hf_export.py \
  --export-dir="$hf_export" --alpha-logits="$alpha_logits" --tol=1e-3 > "$parity_tmp" 2>&1
grep -F 'RESULT               : PASS' "$parity_tmp"
mv "$parity_tmp" "$hf_parity_log"

PAIR_RESULT="$pair_result" PAIR_STATUS="$pair_status" SOURCE_COMMIT="$source_commit" \
  RUN_PATH="$run" CHECKPOINT_PATH="$checkpoint" ANALYSIS_PATH="$analysis" \
  PAIR_PATH="$pair_analysis" HF_PATH="$hf_export" node -e '
    const fs = require("fs");
    const crypto = require("crypto");
    const hash = (p) => crypto.createHash("sha256").update(fs.readFileSync(p)).digest("hex");
    const out = {
      schema: "alpha-sft-terminal-finalizer-v1",
      result: "PASS",
      completed_utc: new Date().toISOString(),
      source_commit: process.env.SOURCE_COMMIT,
      run: process.env.RUN_PATH,
      checkpoint: { path: process.env.CHECKPOINT_PATH, sha256: hash(process.env.CHECKPOINT_PATH) },
      sft_analysis: { path: process.env.ANALYSIS_PATH, sha256: hash(process.env.ANALYSIS_PATH) },
      machine_d3: { result: process.env.PAIR_RESULT, analyzer_exit_code: Number(process.env.PAIR_STATUS), path: process.env.PAIR_PATH, sha256: hash(process.env.PAIR_PATH) },
      semantic_review: "PENDING_HUMAN_REVIEW",
      hf_export: process.env.HF_PATH,
    };
    process.stdout.write(JSON.stringify(out, null, 2) + "\n");
  ' > "$status_file.$$.tmp"
mv "$status_file.$$.tmp" "$status_file"

echo "== seal remote artifact manifest =="
(
  cd "$run"
  find . -type f ! -name 'terminal-artifact-sha256.txt' ! -name '*.tmp' -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum > "$manifest_file.$$.tmp"
  mv "$manifest_file.$$.tmp" "$manifest_file"
)

cat "$status_file"
echo "terminal finalizer operational pipeline=PASS machine_d3=$pair_result semantic_review=PENDING_HUMAN_REVIEW"
