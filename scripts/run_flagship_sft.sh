#!/usr/bin/env bash
# Contracted one-epoch, assistant-only SFT run initialized from the flagship base checkpoint.
# Usage:
#   scripts/run_flagship_sft.sh <selected-lr> <sft-data> <sft-manifest> <tokenizer-artifact> \
#     <base-checkpoint> <run-dir> [resume-checkpoint]

set -euo pipefail

learning_rate=${1:?selected SFT learning rate required}
sft_data=${2:?SFT data required}
sft_manifest=${3:?SFT manifest required}
tokenizer=${4:?tokenizer artifact required}
base_checkpoint=${5:?flagship base checkpoint required}
run_dir=${6:?run directory required}
resume_checkpoint=${7:-}
length_audit=$(dirname "$sft_manifest")/length-audit.json
mask_audit=$(dirname "$sft_manifest")/mask-audit.json
contract_only=${ALPHA_CONTRACT_ONLY:-0}
[[ $contract_only == 0 || $contract_only == 1 ]] || { echo "ALPHA_CONTRACT_ONLY must be 0 or 1" >&2; exit 2; }
if [[ $contract_only == 1 && -n "$resume_checkpoint" ]]; then
  echo "ALPHA_CONTRACT_ONLY cannot be combined with resume" >&2
  exit 2
fi
learning_rate_min=$(SFT_LR="$learning_rate" node -e '
  const lr = Number(process.env.SFT_LR);
  if (!Number.isFinite(lr) || lr <= 0) throw new Error(`invalid learning rate: ${process.env.SFT_LR}`);
  if (![1e-4, 3e-4, 1e-3].includes(lr)) throw new Error(`learning rate was not selected from the contracted SFT sweep: ${lr}`);
  process.stdout.write(String(lr / 10));
')

for required in "$sft_data" "$sft_manifest" "$length_audit" "$mask_audit" "$tokenizer" \
  "$base_checkpoint" apps/cli/dist/main.js scripts/prepare_resume_metrics.ts \
  scripts/verify_flagship_sft_inputs.ts; do
  [[ -f "$required" ]] || { echo "required file missing: $required" >&2; exit 1; }
done
if [[ -n "$resume_checkpoint" ]]; then
  if [[ ! -d "$run_dir" || ! -f "$resume_checkpoint" || ! -f "$run_dir/sft-contract.json" ]]; then
    echo "resume requires an existing run directory, checkpoint, and SFT contract" >&2
    exit 1
  fi
elif [[ -e "$run_dir" ]]; then
  echo "run directory already exists; pass its checkpoint as argument 7 to resume: $run_dir" >&2
  exit 1
fi

export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd_headless.json}
export HELIOS_DISABLE_COOP_MAT=1
export HELIOS_WG_SIZE=64
export HELIOS_MAX_OUTPUT_POOL_ENTRIES=512
export ALPHA_GPU_METRICS_SAMPLE_EVERY=100
export ALPHA_SFT_VAL_FRACTION=0.05

mkdir -p "$run_dir"
source_commit=$(git rev-parse HEAD)
resume_args=()
init_args=()

if [[ -n "$resume_checkpoint" ]]; then
  sft_data_sha256=$(nice -n 10 ionice -c 2 -n 7 sha256sum "$sft_data" | awk '{print $1}')
  sft_manifest_sha256=$(nice -n 10 ionice -c 2 -n 7 sha256sum "$sft_manifest" | awk '{print $1}')
  length_audit_sha256=$(nice -n 10 ionice -c 2 -n 7 sha256sum "$length_audit" | awk '{print $1}')
  mask_audit_sha256=$(nice -n 10 ionice -c 2 -n 7 sha256sum "$mask_audit" | awk '{print $1}')
  tokenizer_sha256=$(nice -n 10 ionice -c 2 -n 7 sha256sum "$tokenizer" | awk '{print $1}')
  base_checkpoint_sha256=$(nice -n 10 ionice -c 2 -n 7 sha256sum "$base_checkpoint" | awk '{print $1}')
  SOURCE_COMMIT="$source_commit" SFT_DATA="$sft_data" SFT_DATA_SHA256="$sft_data_sha256" \
  SFT_MANIFEST="$sft_manifest" SFT_MANIFEST_SHA256="$sft_manifest_sha256" \
  LENGTH_AUDIT="$length_audit" LENGTH_AUDIT_SHA256="$length_audit_sha256" \
  MASK_AUDIT="$mask_audit" MASK_AUDIT_SHA256="$mask_audit_sha256" \
  TOKENIZER="$tokenizer" TOKENIZER_SHA256="$tokenizer_sha256" \
  BASE_CHECKPOINT="$base_checkpoint" BASE_CHECKPOINT_SHA256="$base_checkpoint_sha256" \
  SFT_LR="$learning_rate" SFT_LR_MIN="$learning_rate_min" CONTRACT_PATH="$run_dir/sft-contract.json" node -e '
    const fs = require("node:fs");
    const c = JSON.parse(fs.readFileSync(process.env.CONTRACT_PATH, "utf8"));
    const expected = {
      schema: "alpha-flagship-sft-contract-v1",
      expected_params: 57688576,
      expected_steps: 30322,
      padded_train_tokens: 496795648,
      train_conversations: 485150,
      validation_conversations: 26278,
      learning_rate: Number(process.env.SFT_LR),
      learning_rate_min: Number(process.env.SFT_LR_MIN),
      source_commit: process.env.SOURCE_COMMIT,
    };
    for (const [key, value] of Object.entries(expected)) {
      if (c[key] !== value) throw new Error(`resume contract ${key}: ${c[key]} != ${value}`);
    }
    const files = {
      corpus: [process.env.SFT_DATA, process.env.SFT_DATA_SHA256],
      manifest: [process.env.SFT_MANIFEST, process.env.SFT_MANIFEST_SHA256],
      length_audit: [process.env.LENGTH_AUDIT, process.env.LENGTH_AUDIT_SHA256],
      mask_audit: [process.env.MASK_AUDIT, process.env.MASK_AUDIT_SHA256],
      tokenizer: [process.env.TOKENIZER, process.env.TOKENIZER_SHA256],
      base_checkpoint: [process.env.BASE_CHECKPOINT, process.env.BASE_CHECKPOINT_SHA256],
    };
    for (const [key, [filePath, sha256]] of Object.entries(files)) {
      if (c.inputs?.[key]?.path !== filePath || c.inputs?.[key]?.sha256 !== sha256) {
        throw new Error(`resume ${key} contract mismatch`);
      }
    }
  '
  npx tsx scripts/prepare_resume_metrics.ts \
    --run "$run_dir" --checkpoint "$resume_checkpoint" --sourceCommit "$source_commit"
  resume_args=(--resume="$resume_checkpoint")
else
  verification_tmp=$(mktemp "$run_dir/sft-input-verification.XXXXXX.tmp")
  nice -n 10 ionice -c 2 -n 7 npx tsx scripts/verify_flagship_sft_inputs.ts \
    --data "$sft_data" \
    --manifest "$sft_manifest" \
    --lengthAudit "$length_audit" \
    --maskAudit "$mask_audit" \
    --tokenizer "$tokenizer" \
    --baseCheckpoint "$base_checkpoint" \
    --expectedBaseStep 61036 > "$verification_tmp"
  contract_tmp="$run_dir/sft-contract.json.tmp"
  SOURCE_COMMIT="$source_commit" VERIFICATION_PATH="$verification_tmp" SFT_LR="$learning_rate" \
  SFT_LR_MIN="$learning_rate_min" CONTRACT_TMP="$contract_tmp" node -e '
    const fs = require("node:fs");
    const verification = JSON.parse(fs.readFileSync(process.env.VERIFICATION_PATH, "utf8"));
    if (verification.result !== "PASS") throw new Error("SFT input verification did not pass");
    const contract = {
      schema: "alpha-flagship-sft-contract-v1",
      expected_params: 57688576,
      expected_steps: 30322,
      batch_size: 16,
      block_size: 1024,
      grad_accum_steps: 1,
      padded_train_tokens: 496795648,
      train_conversations: 485150,
      validation_conversations: 26278,
      validation_fraction: 0.05,
      learning_rate: Number(process.env.SFT_LR),
      learning_rate_min: Number(process.env.SFT_LR_MIN),
      warmup_iters: 303,
      source_commit: process.env.SOURCE_COMMIT,
      inputs: {
        corpus: verification.corpus,
        manifest: verification.manifest,
        length_audit: verification.length_audit,
        mask_audit: verification.mask_audit,
        tokenizer: verification.tokenizer,
        base_checkpoint: verification.base_checkpoint,
      },
      started_utc: new Date().toISOString(),
    };
    fs.writeFileSync(process.env.CONTRACT_TMP, JSON.stringify(contract, null, 2) + "\n", { flag: "wx" });
  '
  mv "$verification_tmp" "$run_dir/sft-input-verification.json"
  mv "$contract_tmp" "$run_dir/sft-contract.json"
  init_args=(--initCheckpoint="$base_checkpoint")
fi

if [[ $contract_only == 1 ]]; then
  echo "SFT contract prepared without launching training: $run_dir/sft-contract.json"
  exit 0
fi

exec nice -n 5 ionice -c 2 -n 7 node --expose-gc apps/cli/dist/main.js train \
  --data="$sft_data" \
  --sft=true \
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
  --steps=30322 \
  --lr="$learning_rate" \
  --lrMin="$learning_rate_min" \
  --warmupIters=303 \
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
  --strictPlanning=false \
  --evalInterval=500 \
  --checkpointInterval=1000 \
  --evalIters=5 \
  --logEvery=25 \
  --sampleInterval=0 \
  --postSamples=false \
  --remote=false \
  --packed=false \
  --runDir="$run_dir" \
  "${resume_args[@]}" \
  "${init_args[@]}"
