#!/usr/bin/env bash
# Contracted 1B-token Alpha Llama pretraining run on the proven RTX 3090 profile.
# Usage:
#   scripts/run_flagship_pretrain.sh <selected-lr> <data-manifest> <tokenizer-artifact> <run-dir> [resume-checkpoint]

set -euo pipefail

learning_rate=${1:?selected learning rate required}
data_manifest=${2:?pretraining shard manifest required}
tokenizer=${3:?tokenizer artifact required}
run_dir=${4:?run directory required}
resume_checkpoint=${5:-}
contract_only=${ALPHA_CONTRACT_ONLY:-0}
[[ $contract_only == 0 || $contract_only == 1 ]] || { echo "ALPHA_CONTRACT_ONLY must be 0 or 1" >&2; exit 2; }
if [[ $contract_only == 1 && -n "$resume_checkpoint" ]]; then
  echo "ALPHA_CONTRACT_ONLY cannot be combined with resume" >&2
  exit 2
fi
learning_rate_min=$(FLAGSHIP_LR="$learning_rate" node -e '
  const lr = Number(process.env.FLAGSHIP_LR);
  if (!Number.isFinite(lr) || lr <= 0) throw new Error(`invalid learning rate: ${process.env.FLAGSHIP_LR}`);
  if (![1e-3, 2e-3, 3e-3].includes(lr)) throw new Error(`learning rate was not selected from the contracted sweep: ${lr}`);
  process.stdout.write(String(lr / 10));
')

for required in "$data_manifest" "$tokenizer" apps/cli/dist/main.js scripts/prepare_resume_metrics.ts; do
  [[ -f "$required" ]] || { echo "required file missing: $required" >&2; exit 1; }
done
if [[ -n "$resume_checkpoint" ]]; then
  if [[ ! -d "$run_dir" || ! -f "$resume_checkpoint" || ! -f "$run_dir/flagship-contract.json" ]]; then
    echo "resume requires an existing run directory, checkpoint, and flagship contract" >&2
    exit 1
  fi
elif [[ -e "$run_dir" ]]; then
  echo "run directory already exists; pass its checkpoint as argument 5 to resume: $run_dir" >&2
  exit 1
fi

export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd_headless.json}
export HELIOS_DISABLE_COOP_MAT=1
export HELIOS_WG_SIZE=64
export HELIOS_MAX_OUTPUT_POOL_ENTRIES=512
export ALPHA_GPU_METRICS_SAMPLE_EVERY=100

mkdir -p "$run_dir"
source_commit=$(git rev-parse HEAD)
manifest_sha256=$(sha256sum "$data_manifest" | awk '{print $1}')
tokenizer_sha256=$(sha256sum "$tokenizer" | awk '{print $1}')
resume_args=()

if [[ -n "$resume_checkpoint" ]]; then
  SOURCE_COMMIT="$source_commit" MANIFEST_PATH="$data_manifest" MANIFEST_SHA256="$manifest_sha256" \
  TOKENIZER_PATH="$tokenizer" TOKENIZER_SHA256="$tokenizer_sha256" FLAGSHIP_LR="$learning_rate" \
  FLAGSHIP_LR_MIN="$learning_rate_min" CONTRACT_PATH="$run_dir/flagship-contract.json" node -e '
    const fs = require("node:fs");
    const c = JSON.parse(fs.readFileSync(process.env.CONTRACT_PATH, "utf8"));
    const expected = {
      schema: "alpha-flagship-contract-v1",
      expected_params: 57688576,
      expected_steps: 61036,
      expected_tokens: 1000013824,
      minimum_train_tokens: 1000013824,
      learning_rate: Number(process.env.FLAGSHIP_LR),
      learning_rate_min: Number(process.env.FLAGSHIP_LR_MIN),
      source_commit: process.env.SOURCE_COMMIT,
    };
    for (const [key, value] of Object.entries(expected)) {
      if (c[key] !== value) throw new Error(`resume contract ${key}: ${c[key]} != ${value}`);
    }
    if (c.data_manifest?.path !== process.env.MANIFEST_PATH || c.data_manifest?.sha256 !== process.env.MANIFEST_SHA256) throw new Error("resume data-manifest contract mismatch");
    if (c.tokenizer?.path !== process.env.TOKENIZER_PATH || c.tokenizer?.sha256 !== process.env.TOKENIZER_SHA256) throw new Error("resume tokenizer contract mismatch");
  '
  npx tsx scripts/prepare_resume_metrics.ts \
    --run "$run_dir" --checkpoint "$resume_checkpoint" --sourceCommit "$source_commit"
  resume_args=(--resume="$resume_checkpoint")
else
  contract_tmp="$run_dir/flagship-contract.json.tmp"
  SOURCE_COMMIT="$source_commit" MANIFEST_PATH="$data_manifest" MANIFEST_SHA256="$manifest_sha256" \
  TOKENIZER_PATH="$tokenizer" TOKENIZER_SHA256="$tokenizer_sha256" FLAGSHIP_LR="$learning_rate" \
  FLAGSHIP_LR_MIN="$learning_rate_min" CONTRACT_TMP="$contract_tmp" node -e '
    const fs = require("node:fs");
    const manifest = JSON.parse(fs.readFileSync(process.env.MANIFEST_PATH, "utf8"));
    if (manifest.schema !== "alpha-pretrain-shards-v1" || !Array.isArray(manifest.shards) || manifest.shards.length < 2) {
      throw new Error("invalid pretraining shard manifest");
    }
    const contract = {
      schema: "alpha-flagship-contract-v1",
      expected_params: 57688576,
      expected_steps: 61036,
      batch_size: 16,
      block_size: 1024,
      grad_accum_steps: 1,
      expected_tokens: 1000013824,
      minimum_train_tokens: 1000013824,
      learning_rate: Number(process.env.FLAGSHIP_LR),
      learning_rate_min: Number(process.env.FLAGSHIP_LR_MIN),
      warmup_iters: 610,
      source_commit: process.env.SOURCE_COMMIT,
      data_manifest: { path: process.env.MANIFEST_PATH, sha256: process.env.MANIFEST_SHA256, shards: manifest.shards },
      tokenizer: { path: process.env.TOKENIZER_PATH, sha256: process.env.TOKENIZER_SHA256 },
      started_utc: new Date().toISOString(),
    };
    fs.writeFileSync(process.env.CONTRACT_TMP, JSON.stringify(contract, null, 2) + "\n", { flag: "wx" });
  '
  mv "$contract_tmp" "$run_dir/flagship-contract.json"
fi

if [[ $contract_only == 1 ]]; then
  echo "flagship contract prepared without launching training: $run_dir/flagship-contract.json"
  exit 0
fi

exec nice -n 5 ionice -c 2 -n 7 node --expose-gc apps/cli/dist/main.js train \
  --dataManifest="$data_manifest" \
  --verifyDataHashes=true \
  --domain=alpha_llama \
  --tokenizerArtifacts="$tokenizer" \
  --vocabSize=12288 \
  --block=1024 \
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
  --batch=16 \
  --accumSteps=1 \
  --steps=61036 \
  --lr="$learning_rate" \
  --lrMin="$learning_rate_min" \
  --warmupIters=610 \
  --beta1=0.9 \
  --beta2=0.95 \
  --eps=1e-8 \
  --weightDecay=0.1 \
  --gradClip=1.0 \
  --spikeThreshold=0 \
  --seed=42 \
  --backend=helios \
  --gpuProfile=none \
  --fp16=false \
  --minGpuSize=1 \
  --no-fallback=true \
  --strictPlanning=true \
  --evalInterval=500 \
  --checkpointInterval=1000 \
  --evalIters=5 \
  --logEvery=25 \
  --sampleInterval=0 \
  --postSamples=false \
  --remote=false \
  --packed=true \
  --runDir="$run_dir" \
  "${resume_args[@]}"
