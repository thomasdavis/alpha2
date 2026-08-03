#!/usr/bin/env bash
# Run the selected 97M Alpha foundation for one non-repeating ~20-token/parameter pass.
# Usage:
#   scripts/run_foundation_candidate_full.sh <lr-selection> <data-manifest> <heldout-text> <tokenizer> <run-dir> [resume-checkpoint]

set -euo pipefail

selection_report=${1:?LR selection report required}
data_manifest=${2:?pretraining shard manifest required}
val_data=${3:?held-out validation text required}
tokenizer=${4:?tokenizer artifact required}
run_dir=${5:?run directory required}
resume_checkpoint=${6:-}
contract_only=${ALPHA_CONTRACT_ONLY:-0}

[[ $contract_only == 0 || $contract_only == 1 ]] || { echo "ALPHA_CONTRACT_ONLY must be 0 or 1" >&2; exit 2; }
if [[ $contract_only == 1 && -n "$resume_checkpoint" ]]; then
  echo "ALPHA_CONTRACT_ONLY cannot be combined with resume" >&2
  exit 2
fi
for required in "$selection_report" "$data_manifest" "$val_data" "$tokenizer" apps/cli/dist/main.js scripts/prepare_resume_metrics.ts; do
  [[ -f "$required" ]] || { echo "required file missing: $required" >&2; exit 1; }
done

tokenizer_sha256=$(sha256sum "$tokenizer" | awk '{print $1}')
learning_rate=$(SELECTION_REPORT="$selection_report" TOKENIZER_SHA256="$tokenizer_sha256" node -e '
  const fs = require("node:fs");
  const report = JSON.parse(fs.readFileSync(process.env.SELECTION_REPORT, "utf8"));
  if (report.schema !== "alpha-foundation-candidate-lr-sweep-v1" || report.result !== "PASS") {
    throw new Error("invalid foundation LR selection report");
  }
  const expectedRates = [1e-3, 2e-3, 3e-3];
  if (!expectedRates.includes(report.selected_learning_rate)) {
    throw new Error(`invalid selected learning rate: ${report.selected_learning_rate}`);
  }
  if (!Array.isArray(report.ranking) || report.ranking.length !== 3 ||
      report.ranking[0]?.learning_rate !== report.selected_learning_rate) {
    throw new Error("LR selection ranking is incomplete or inconsistent");
  }
  const rates = report.ranking.map((entry) => entry.learning_rate).sort((a, b) => a - b);
  if (JSON.stringify(rates) !== JSON.stringify(expectedRates)) throw new Error("LR candidate set drifted");
  if (!/^[0-9a-f]{40}$/.test(report.source_commit ?? "")) throw new Error("pilot source commit is invalid");
  if (report.tokenizer_sha256 !== process.env.TOKENIZER_SHA256) throw new Error("tokenizer hash differs from LR pilot");
  process.stdout.write(String(report.selected_learning_rate));
')
learning_rate_min=$(FULL_LR="$learning_rate" node -e '
  process.stdout.write(String(Number(process.env.FULL_LR) / 10));
')

if [[ -n "$resume_checkpoint" ]]; then
  if [[ ! -d "$run_dir" || ! -f "$resume_checkpoint" || ! -f "$run_dir/foundation-contract.json" ]]; then
    echo "resume requires an existing run directory, raw checkpoint, and foundation contract" >&2
    exit 1
  fi
elif [[ -e "$run_dir" ]]; then
  echo "run directory already exists; pass a raw checkpoint as argument 6 to resume: $run_dir" >&2
  exit 1
fi

export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd_headless.json}
export HELIOS_DISABLE_COOP_MAT=1
export HELIOS_FLASH_FWD_PREFER_COOP2=0
export HELIOS_WG_SIZE=64
export HELIOS_MATMUL_REG4X2=1
export HELIOS_MATMUL_REG4X2_TRANSPOSED_B=1
export HELIOS_MATMUL_TRANSPOSED_B_COALESCED=1
export HELIOS_MATMUL_REG2X2=1
export HELIOS_MAX_OUTPUT_POOL_ENTRIES=512
export ALPHA_GPU_METRICS_SAMPLE_EVERY=100
export ALPHA_FAIL_ON_SMOKE_TEST=1

mkdir -p "$run_dir"
source_commit=$(git rev-parse HEAD)
manifest_sha256=$(sha256sum "$data_manifest" | awk '{print $1}')
val_sha256=$(sha256sum "$val_data" | awk '{print $1}')
selection_sha256=$(sha256sum "$selection_report" | awk '{print $1}')
resume_args=()

if [[ -n "$resume_checkpoint" ]]; then
  SOURCE_COMMIT="$source_commit" SELECTION_PATH="$selection_report" SELECTION_SHA256="$selection_sha256" \
  MANIFEST_PATH="$data_manifest" MANIFEST_SHA256="$manifest_sha256" VAL_PATH="$val_data" VAL_SHA256="$val_sha256" \
  TOKENIZER_PATH="$tokenizer" TOKENIZER_SHA256="$tokenizer_sha256" FULL_LR="$learning_rate" \
  FULL_LR_MIN="$learning_rate_min" CONTRACT_PATH="$run_dir/foundation-contract.json" node -e '
    const fs = require("node:fs");
    const c = JSON.parse(fs.readFileSync(process.env.CONTRACT_PATH, "utf8"));
    const expected = {
      schema: "alpha-foundation-full-contract-v1",
      expected_params: 97098880,
      expected_steps: 79020,
      expected_tokens: 1941995520,
      minimum_train_tokens: 1941995520,
      learning_rate: Number(process.env.FULL_LR),
      learning_rate_min: Number(process.env.FULL_LR_MIN),
      source_commit: process.env.SOURCE_COMMIT,
      engine: {
        backend: "helios",
        accelerator_api: "vulkan",
        kernel_policy: "layout-portfolio-r42c-r2-v2",
        environment: {
          HELIOS_DISABLE_COOP_MAT: "1",
          HELIOS_FLASH_FWD_PREFER_COOP2: "0",
          HELIOS_WG_SIZE: "64",
          HELIOS_MATMUL_REG4X2: "1",
          HELIOS_MATMUL_REG4X2_TRANSPOSED_B: "1",
          HELIOS_MATMUL_TRANSPOSED_B_COALESCED: "1",
          HELIOS_MATMUL_REG2X2: "1",
          HELIOS_MAX_OUTPUT_POOL_ENTRIES: "512",
        },
      },
    };
    for (const [key, value] of Object.entries(expected)) {
      if (JSON.stringify(c[key]) !== JSON.stringify(value)) {
        throw new Error(`resume contract ${key}: ${JSON.stringify(c[key])} != ${JSON.stringify(value)}`);
      }
    }
    if (c.lr_selection?.path !== process.env.SELECTION_PATH || c.lr_selection?.sha256 !== process.env.SELECTION_SHA256) throw new Error("resume LR-selection mismatch");
    if (c.data_manifest?.path !== process.env.MANIFEST_PATH || c.data_manifest?.sha256 !== process.env.MANIFEST_SHA256) throw new Error("resume data-manifest mismatch");
    if (c.validation?.path !== process.env.VAL_PATH || c.validation?.sha256 !== process.env.VAL_SHA256) throw new Error("resume validation mismatch");
    if (c.tokenizer?.path !== process.env.TOKENIZER_PATH || c.tokenizer?.sha256 !== process.env.TOKENIZER_SHA256) throw new Error("resume tokenizer mismatch");
  '
  npx tsx scripts/prepare_resume_metrics.ts \
    --run "$run_dir" --checkpoint "$resume_checkpoint" --sourceCommit "$source_commit"
  resume_args=(--resume="$resume_checkpoint")
else
  contract_tmp="$run_dir/foundation-contract.json.tmp"
  SOURCE_COMMIT="$source_commit" SELECTION_PATH="$selection_report" SELECTION_SHA256="$selection_sha256" \
  MANIFEST_PATH="$data_manifest" MANIFEST_SHA256="$manifest_sha256" VAL_PATH="$val_data" VAL_SHA256="$val_sha256" \
  TOKENIZER_PATH="$tokenizer" TOKENIZER_SHA256="$tokenizer_sha256" FULL_LR="$learning_rate" \
  FULL_LR_MIN="$learning_rate_min" CONTRACT_TMP="$contract_tmp" node -e '
    const fs = require("node:fs");
    const path = require("node:path");
    const report = JSON.parse(fs.readFileSync(process.env.SELECTION_PATH, "utf8"));
    const manifest = JSON.parse(fs.readFileSync(process.env.MANIFEST_PATH, "utf8"));
    if (manifest.schema !== "alpha-pretrain-shards-v1" || !Array.isArray(manifest.shards) || manifest.shards.length !== 4) {
      throw new Error("foundation manifest must contain exactly four training shards");
    }
    const validationResolved = path.resolve(process.env.VAL_PATH);
    const manifestDir = path.dirname(path.resolve(process.env.MANIFEST_PATH));
    if (manifest.shards.some((shard) => path.resolve(manifestDir, shard.path) === validationResolved)) {
      throw new Error("held-out validation is also present in the training manifest");
    }
    const contract = {
      schema: "alpha-foundation-full-contract-v1",
      expected_params: 97098880,
      expected_steps: 79020,
      batch_size: 24,
      block_size: 1024,
      grad_accum_steps: 1,
      expected_tokens: 1941995520,
      minimum_train_tokens: 1941995520,
      learning_rate: Number(process.env.FULL_LR),
      learning_rate_min: Number(process.env.FULL_LR_MIN),
      warmup_iters: 790,
      eval_interval: 500,
      checkpoint_interval: 1000,
      eval_iters: 5,
      source_commit: process.env.SOURCE_COMMIT,
      engine: {
        backend: "helios",
        accelerator_api: "vulkan",
        kernel_policy: "layout-portfolio-r42c-r2-v2",
        environment: {
          HELIOS_DISABLE_COOP_MAT: "1",
          HELIOS_FLASH_FWD_PREFER_COOP2: "0",
          HELIOS_WG_SIZE: "64",
          HELIOS_MATMUL_REG4X2: "1",
          HELIOS_MATMUL_REG4X2_TRANSPOSED_B: "1",
          HELIOS_MATMUL_TRANSPOSED_B_COALESCED: "1",
          HELIOS_MATMUL_REG2X2: "1",
          HELIOS_MAX_OUTPUT_POOL_ENTRIES: "512",
        },
      },
      lr_selection: {
        path: process.env.SELECTION_PATH,
        sha256: process.env.SELECTION_SHA256,
        pilot_source_commit: report.source_commit,
        selected_learning_rate: Number(process.env.FULL_LR),
      },
      data_manifest: { path: process.env.MANIFEST_PATH, sha256: process.env.MANIFEST_SHA256, shards: manifest.shards },
      validation: { path: process.env.VAL_PATH, sha256: process.env.VAL_SHA256, wholly_held_out: true },
      tokenizer: { path: process.env.TOKENIZER_PATH, sha256: process.env.TOKENIZER_SHA256 },
      started_utc: new Date().toISOString(),
    };
    fs.writeFileSync(process.env.CONTRACT_TMP, JSON.stringify(contract, null, 2) + "\n", { flag: "wx" });
  '
  mv "$contract_tmp" "$run_dir/foundation-contract.json"
fi

if [[ $contract_only == 1 ]]; then
  echo "foundation contract prepared without launching training: $run_dir/foundation-contract.json"
  exit 0
fi

set +e
nice -n 5 ionice -c 2 -n 7 node --expose-gc apps/cli/dist/main.js train \
  --dataManifest="$data_manifest" \
  --verifyDataHashes=true \
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
  --steps=79020 \
  --lr="$learning_rate" \
  --lrMin="$learning_rate_min" \
  --warmupIters=790 \
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
  --strictPlanning=true \
  --evalInterval=500 \
  --checkpointInterval=1000 \
  --evalIters=5 \
  --logEvery=25 \
  --sampleInterval=0 \
  --postSamples=false \
  --remote=false \
  --packed=true \
  --symbio=false \
  --trace=false \
  --runDir="$run_dir" \
  "${resume_args[@]}"
train_status=$?
set -e
printf '%s\n' "$train_status" > "$run_dir/exit-code.txt"
exit "$train_status"
