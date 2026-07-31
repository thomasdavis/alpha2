#!/usr/bin/env bash
# Bounded corrective SFT for Alpha's response-initiation failure.
#
# Usage:
#   scripts/run_chat_repair.sh TRAIN DEV MANIFEST TOKENIZER INIT_CHECKPOINT RUN_DIR \
#     [STEPS=2200] [LR=0.00005] [BATCH=32]

set -euo pipefail

train_data=${1:?training conversations required}
dev_data=${2:?development conversations required}
manifest=${3:?corpus manifest required}
tokenizer=${4:?Alpha tokenizer artifact required}
init_checkpoint=${5:?initial checkpoint required}
run_dir=${6:?new run directory required}
steps=${7:-2200}
learning_rate=${8:-0.00005}
batch_size=${9:-32}

for required in "$train_data" "$dev_data" "$manifest" "$tokenizer" "$init_checkpoint" apps/cli/dist/main.js; do
  [[ -f $required ]] || { echo "required file missing: $required" >&2; exit 1; }
done
[[ ! -e $run_dir ]] || { echo "run directory already exists: $run_dir" >&2; exit 1; }
[[ $steps =~ ^[1-9][0-9]*$ ]] || { echo "STEPS must be a positive integer" >&2; exit 2; }
[[ $batch_size =~ ^[1-9][0-9]*$ ]] || { echo "BATCH must be a positive integer" >&2; exit 2; }

train_sha=$(sha256sum "$train_data" | awk '{print $1}')
dev_sha=$(sha256sum "$dev_data" | awk '{print $1}')
tokenizer_sha=$(sha256sum "$tokenizer" | awk '{print $1}')
checkpoint_sha=$(sha256sum "$init_checkpoint" | awk '{print $1}')
manifest_sha=$(sha256sum "$manifest" | awk '{print $1}')
manifest_train_sha=$(node -e 'const m=require(process.argv[1]); process.stdout.write(m.outputs.train.sha256)' "$manifest")
manifest_dev_sha=$(node -e 'const m=require(process.argv[1]); process.stdout.write(m.outputs.dev.sha256)' "$manifest")
[[ $train_sha == "$manifest_train_sha" ]] || { echo "training corpus hash mismatch" >&2; exit 1; }
[[ $dev_sha == "$manifest_dev_sha" ]] || { echo "development corpus hash mismatch" >&2; exit 1; }

source_commit=$(git rev-parse HEAD)
mkdir -p "$run_dir"
contract_tmp="$run_dir/repair-contract.json.tmp"
SOURCE_COMMIT="$source_commit" TRAIN_DATA="$train_data" TRAIN_SHA="$train_sha" \
DEV_DATA="$dev_data" DEV_SHA="$dev_sha" MANIFEST="$manifest" MANIFEST_SHA="$manifest_sha" \
TOKENIZER="$tokenizer" TOKENIZER_SHA="$tokenizer_sha" INIT_CHECKPOINT="$init_checkpoint" \
CHECKPOINT_SHA="$checkpoint_sha" STEPS="$steps" LR="$learning_rate" BATCH="$batch_size" \
CONTRACT_TMP="$contract_tmp" node -e '
  const fs = require("node:fs");
  const contract = {
    schema: "alpha-chat-repair-contract-v1",
    purpose: "repair free response initiation and natural conversation",
    sourceCommit: process.env.SOURCE_COMMIT,
    initializedFrom: {
      path: process.env.INIT_CHECKPOINT,
      sha256: process.env.CHECKPOINT_SHA,
    },
    inputs: {
      train: { path: process.env.TRAIN_DATA, sha256: process.env.TRAIN_SHA },
      dev: { path: process.env.DEV_DATA, sha256: process.env.DEV_SHA },
      manifest: { path: process.env.MANIFEST, sha256: process.env.MANIFEST_SHA },
      tokenizer: { path: process.env.TOKENIZER, sha256: process.env.TOKENIZER_SHA },
    },
    training: {
      steps: Number(process.env.STEPS),
      blockSize: 512,
      batchSize: Number(process.env.BATCH),
      learningRate: Number(process.env.LR),
      learningRateMin: Number(process.env.LR) / 10,
      warmupSteps: 100,
      checkpointInterval: 200,
      deterministicEpochShuffle: true,
      equalConversationWeight: true,
      answerStartTokens: 4,
      answerStartMultiplier: 8,
      eosBoostedAsAnswerStart: false,
    },
    selection: {
      suite: "eval/chat-repair-dev-v1",
      generation: "greedy",
      rule: "aggregate nonempty, clean stopping, no loops or role leaks, then human conversational inspection",
      finalSuiteHeldUntilSelection: true,
    },
    startedUtc: new Date().toISOString(),
  };
  fs.writeFileSync(process.env.CONTRACT_TMP, JSON.stringify(contract, null, 2) + "\n", { flag: "wx" });
'
mv "$contract_tmp" "$run_dir/repair-contract.json"

export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd_headless.json}
export HELIOS_DISABLE_COOP_MAT=1
export HELIOS_WG_SIZE=64
export HELIOS_MAX_OUTPUT_POOL_ENTRIES=512
export ALPHA_FAIL_ON_SMOKE_TEST=1
export ALPHA_ALLOW_RESUME_MISMATCH=1 # deliberate 1024 -> 512 RoPE context migration; weights are shape-compatible
export ALPHA_SFT_SHUFFLE=1
export ALPHA_SFT_BALANCE_CONVERSATIONS=1
export ALPHA_SFT_START_TOKENS=4
export ALPHA_SFT_START_WEIGHT=8
export ALPHA_SAMPLE_FROM_CHECKPOINT=0
export ALPHA_GPU_METRICS_SAMPLE_EVERY=25

exec nice -n 5 ionice -c2 -n7 node --expose-gc apps/cli/dist/main.js train \
  --data="$train_data" \
  --valData="$dev_data" \
  --requireValData=true \
  --sft=true \
  --domain=alpha_llama \
  --tokenizerArtifacts="$tokenizer" \
  --initCheckpoint="$init_checkpoint" \
  --vocabSize=12288 \
  --block=512 \
  --layers=16 \
  --dim=512 \
  --heads=8 \
  --dropout=0 \
  --activation=swiglu \
  --ffnDim=1408 \
  --normType=rmsnorm \
  --posEnc=rope \
  --ropeTheta=10000 \
  --tieEmbeddings=true \
  --backend=helios \
  --optim=adamw \
  --batch="$batch_size" \
  --accumSteps=1 \
  --steps="$steps" \
  --lr="$learning_rate" \
  --lrMin="$(node -e 'process.stdout.write(String(Number(process.argv[1]) / 10))' "$learning_rate")" \
  --warmupIters=100 \
  --beta1=0.9 \
  --beta2=0.95 \
  --eps=1e-8 \
  --weightDecay=0.1 \
  --gradClip=1.0 \
  --evalInterval=100 \
  --checkpointInterval=200 \
  --evalIters=10 \
  --sampleInterval=100 \
  --logEvery=10 \
  --seed=42 \
  --strictPlanning=false \
  --remote=false \
  --postSamples=false \
  --runDir="$run_dir"
