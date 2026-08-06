#!/usr/bin/env bash
set -euo pipefail

# Evaluate every checkpoint declared by an Alpha V12 recipe contract. The
# individual evaluator owns checkpoint/freeze/hash/parity validation; this
# wrapper only provides serial, resumable orchestration on one GPU.

run_dir=${1:?usage: run_chat_recipe_v12_evaluations.sh RUN_DIR EVALUATION_FREEZE OUT_ROOT PYTHON}
evaluation_freeze=${2:?evaluation freeze required}
out_root=${3:?output root required}
python=${4:?Python interpreter required}
batch_size=${BATCH_SIZE:-32}

for required in \
  "$run_dir/recipe-contract.json" \
  "$evaluation_freeze" \
  "$python" \
  apps/cli/dist/main.js \
  scripts/evaluate_chat_semantic_v4_checkpoint.ts; do
  [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 2; }
done
if [[ ! "$batch_size" =~ ^[0-9]+$ ]] || (( batch_size < 1 )); then
  echo "BATCH_SIZE must be a positive integer" >&2
  exit 2
fi

mapfile -t checkpoints < <(
  RUN_CONTRACT="$run_dir/recipe-contract.json" node -e '
    const fs = require("node:fs");
    const contract = JSON.parse(fs.readFileSync(process.env.RUN_CONTRACT, "utf8"));
    if (contract.schema !== "alpha-chat-recipe-v12-pilot-contract-v1") {
      throw new Error(`unexpected V12 contract schema: ${contract.schema}`);
    }
    const checkpoints = contract.selection?.checkpoints;
    if (!Array.isArray(checkpoints) || checkpoints.length === 0 ||
        checkpoints.some((step) => !Number.isSafeInteger(step) || step <= 0)) {
      throw new Error("V12 contract has no valid selection checkpoints");
    }
    for (const step of checkpoints) process.stdout.write(`${step}\n`);
  '
)

mkdir -p "$out_root"
for step in "${checkpoints[@]}"; do
  checkpoint="$run_dir/checkpoint-$step.json"
  out_dir="$out_root/step-$step"
  [[ -f "$checkpoint" ]] || { echo "missing declared checkpoint: $checkpoint" >&2; exit 2; }

  if [[ -f "$out_dir/evaluation-manifest.json" ]]; then
    echo "step $step already complete: $out_dir/evaluation-manifest.json"
    continue
  fi

  resume_args=()
  if [[ -d "$out_dir" ]]; then
    resume_args+=(--resume)
  fi
  echo "evaluating V12 checkpoint step $step"
  npx tsx scripts/evaluate_chat_semantic_v4_checkpoint.ts \
    --checkpoint "$checkpoint" \
    --run-contract "$run_dir/recipe-contract.json" \
    --evaluation-freeze "$evaluation_freeze" \
    --out-dir "$out_dir" \
    --python "$python" \
    --batch-size "$batch_size" \
    "${resume_args[@]}"
done

echo "all declared V12 checkpoints evaluated: $out_root"
