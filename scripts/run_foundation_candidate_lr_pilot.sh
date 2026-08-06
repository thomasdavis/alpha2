#!/usr/bin/env bash
# Run one bounded LR arm for the measured one-GPU Alpha foundation candidate.
# Usage:
#   scripts/run_foundation_candidate_lr_pilot.sh <lr> <train-text> <heldout-text> <tokenizer> <run-dir> [resume-checkpoint]

set -euo pipefail

learning_rate=${1:?learning rate required}
train_data=${2:?training text required}
val_data=${3:?held-out validation text required}
tokenizer=${4:?tokenizer artifact required}
run_dir=${5:?run directory required}
resume_checkpoint=${6:-}
learning_rate_min=$(PILOT_LR="$learning_rate" node -e '
  const lr = Number(process.env.PILOT_LR);
  if (![1e-3, 2e-3, 3e-3].includes(lr)) throw new Error(`unsupported pilot LR: ${process.env.PILOT_LR}`);
  process.stdout.write(String(lr / 10));
')

for required in "$train_data" "$val_data" "$tokenizer" apps/cli/dist/main.js scripts/prepare_resume_metrics.ts; do
  [[ -f "$required" ]] || { echo "required file missing: $required" >&2; exit 1; }
done
if [[ "$train_data" -ef "$val_data" ]]; then
  echo "training and validation files must be distinct" >&2
  exit 2
fi
if [[ -n "$resume_checkpoint" ]]; then
  if [[ ! -d "$run_dir" || ! -f "$resume_checkpoint" || ! -f "$run_dir/pilot-contract.json" ]]; then
    echo "resume requires an existing run directory, raw checkpoint, and pilot contract" >&2
    exit 1
  fi
elif [[ -e "$run_dir" ]]; then
  echo "run directory already exists; pass a raw checkpoint as argument 6 to resume: $run_dir" >&2
  exit 1
fi

export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd_headless.json}
export HELIOS_DISABLE_COOP_MAT=1
export HELIOS_WG_SIZE=64
export HELIOS_MAX_OUTPUT_POOL_ENTRIES=512
export ALPHA_GPU_METRICS_SAMPLE_EVERY=25
export ALPHA_FAIL_ON_SMOKE_TEST=1

mkdir -p "$run_dir"
source_commit=$(git rev-parse HEAD)
train_sha256=$(sha256sum "$train_data" | awk '{print $1}')
val_sha256=$(sha256sum "$val_data" | awk '{print $1}')
tokenizer_sha256=$(sha256sum "$tokenizer" | awk '{print $1}')
resume_args=()

if [[ -n "$resume_checkpoint" ]]; then
  SOURCE_COMMIT="$source_commit" TRAIN_PATH="$train_data" TRAIN_SHA256="$train_sha256" \
  VAL_PATH="$val_data" VAL_SHA256="$val_sha256" TOKENIZER_PATH="$tokenizer" \
  TOKENIZER_SHA256="$tokenizer_sha256" PILOT_LR="$learning_rate" PILOT_LR_MIN="$learning_rate_min" \
  CONTRACT_PATH="$run_dir/pilot-contract.json" node -e '
    const fs = require("node:fs");
    const c = JSON.parse(fs.readFileSync(process.env.CONTRACT_PATH, "utf8"));
    const expected = {
      schema: "alpha-foundation-candidate-lr-pilot-v1",
      expected_params: 97098880,
      expected_steps: 384,
      expected_tokens: 9437184,
      learning_rate: Number(process.env.PILOT_LR),
      learning_rate_min: Number(process.env.PILOT_LR_MIN),
      source_commit: process.env.SOURCE_COMMIT,
    };
    for (const [key, value] of Object.entries(expected)) {
      if (c[key] !== value) throw new Error(`resume contract ${key}: ${c[key]} != ${value}`);
    }
    if (c.train_data?.path !== process.env.TRAIN_PATH || c.train_data?.sha256 !== process.env.TRAIN_SHA256) throw new Error("resume training-data mismatch");
    if (c.val_data?.path !== process.env.VAL_PATH || c.val_data?.sha256 !== process.env.VAL_SHA256) throw new Error("resume validation-data mismatch");
    if (c.tokenizer?.path !== process.env.TOKENIZER_PATH || c.tokenizer?.sha256 !== process.env.TOKENIZER_SHA256) throw new Error("resume tokenizer mismatch");
  '
  npx tsx scripts/prepare_resume_metrics.ts \
    --run "$run_dir" --checkpoint "$resume_checkpoint" --sourceCommit "$source_commit"
  resume_args=(--resume="$resume_checkpoint")
else
  contract_tmp="$run_dir/pilot-contract.json.tmp"
  SOURCE_COMMIT="$source_commit" TRAIN_PATH="$train_data" TRAIN_SHA256="$train_sha256" \
  VAL_PATH="$val_data" VAL_SHA256="$val_sha256" TOKENIZER_PATH="$tokenizer" \
  TOKENIZER_SHA256="$tokenizer_sha256" PILOT_LR="$learning_rate" PILOT_LR_MIN="$learning_rate_min" \
  CONTRACT_TMP="$contract_tmp" node -e '
    const fs = require("node:fs");
    const contract = {
      schema: "alpha-foundation-candidate-lr-pilot-v1",
      expected_params: 97098880,
      expected_steps: 384,
      batch_size: 24,
      block_size: 1024,
      grad_accum_steps: 1,
      expected_tokens: 9437184,
      minimum_train_tokens: 9437184,
      learning_rate: Number(process.env.PILOT_LR),
      learning_rate_min: Number(process.env.PILOT_LR_MIN),
      warmup_iters: 38,
      eval_interval: 96,
      checkpoint_interval: 192,
      source_commit: process.env.SOURCE_COMMIT,
      train_data: { path: process.env.TRAIN_PATH, sha256: process.env.TRAIN_SHA256 },
      val_data: { path: process.env.VAL_PATH, sha256: process.env.VAL_SHA256 },
      tokenizer: { path: process.env.TOKENIZER_PATH, sha256: process.env.TOKENIZER_SHA256 },
      started_utc: new Date().toISOString(),
    };
    fs.writeFileSync(process.env.CONTRACT_TMP, JSON.stringify(contract, null, 2) + "\n", { flag: "wx" });
  '
  mv "$contract_tmp" "$run_dir/pilot-contract.json"
fi

set +e
nice -n 5 ionice -c 2 -n 7 node --expose-gc apps/cli/dist/main.js train \
  --data="$train_data" \
  --valData="$val_data" \
  --requireValData=true \
  --domain=alpha_llama \
  --tokenizerArtifacts="$tokenizer" \
  --vocabSize=12288 \
  --block=1024 \
  --layers=18 \
  --dim=640 \
  --heads=10 \
  --dropout=0 \
  --activation=swiglu \
  --ffnDim=1728 \
  --normType=rmsnorm \
  --posEnc=rope \
  --ropeTheta=10000 \
  --tieEmbeddings=true \
  --batch=24 \
  --accumSteps=1 \
  --steps=384 \
  --lr="$learning_rate" \
  --lrMin="$learning_rate_min" \
  --warmupIters=38 \
  --beta1=0.9 \
  --beta2=0.95 \
  --eps=1e-8 \
  --weightDecay=0.1 \
  --gradClip=1.0 \
  --spikeThreshold=0 \
  --seed=42 \
  --backend=helios \
  --gpuProfile=none \
  --optim=adamw \
  --fp16=false \
  --minGpuSize=1 \
  --no-fallback=true \
  --strictPlanning=false \
  --evalInterval=96 \
  --checkpointInterval=192 \
  --evalIters=4 \
  --logEvery=12 \
  --sampleInterval=0 \
  --postSamples=false \
  --remote=false \
  --packed=true \
  --symbio=false \
  --trace=true \
  --runDir="$run_dir" \
  "${resume_args[@]}" 2>&1 | tee -a "$run_dir/train.log"
train_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$train_status" > "$run_dir/exit-code.txt"
if (( train_status != 0 )); then
  exit "$train_status"
fi

RUN_DIR="$run_dir" node -e '
  const fs = require("node:fs");
  const path = require("node:path");
  const config = JSON.parse(fs.readFileSync(path.join(process.env.RUN_DIR, "config.json"), "utf8"));
  if (config.totalParams !== 97098880) throw new Error(`parameter count ${config.totalParams} != 97098880`);
  const rows = fs.readFileSync(path.join(process.env.RUN_DIR, "metrics.jsonl"), "utf8").trim().split("\n");
  if (rows.length !== 384) throw new Error(`metric rows ${rows.length} != 384`);
'

(
  cd "$run_dir"
  : > checkpoint-raw.sha256
  : > checkpoint-zst.sha256
  for checkpoint in checkpoint-192.json checkpoint-384.json; do
    [[ -f "$checkpoint" ]] || { echo "missing completed checkpoint: $checkpoint" >&2; exit 1; }
    sha256sum "$checkpoint" >> checkpoint-raw.sha256
    zstd -T0 -6 --rm "$checkpoint"
    zstd -t "$checkpoint.zst"
    sha256sum "$checkpoint.zst" >> checkpoint-zst.sha256
  done
  while read -r expected checkpoint; do
    actual=$(zstd -dc "$checkpoint.zst" | sha256sum | awk '{print $1}')
    [[ "$actual" == "$expected" ]] || { echo "decompressed hash mismatch: $checkpoint" >&2; exit 1; }
  done < checkpoint-raw.sha256
)
