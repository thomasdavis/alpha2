#!/usr/bin/env bash
# Equal-token Stage-3 architecture pilot on the proven RTX 3090 Vulkan profile.
#
# Usage (on the bootstrapped pod):
#   scripts/run_g3_pilot.sh llama /workspace/data/pretrain-000.txt \
#     /workspace/alpha2/artifacts/g2-bpe-byte-12k.json /workspace/alpha2/runs/g3-llama-100m
#   scripts/run_g3_pilot.sh gpt2  ... /workspace/alpha2/runs/g3-gpt2-100m
# Resume after preemption with the exact same command plus the checkpoint as argument 5:
#   scripts/run_g3_pilot.sh llama ... /workspace/alpha2/runs/g3-llama-100m \
#     /workspace/alpha2/runs/g3-llama-100m/checkpoint-3000.json
#
# Both controls see 6,104 * 16 * 1,024 = 100,007,936 tokens with identical
# tokenizer, batches, optimizer, seed, and schedule. Exact initialized sizes:
#   llama: 57,688,576 params (16L, RMSNorm, RoPE, tied, soft-cap off)
#   gpt2:  58,094,592 params (14L, LayerNorm, learned pos, untied, soft-cap 30)
# Difference: 0.704%, close enough to make equal-token loss curves meaningful.

set -euo pipefail

variant=${1:?variant required: llama or gpt2}
data=${2:?training text path required}
tokenizer=${3:?tokenizer artifact path required}
run_dir=${4:?run directory required}
resume_checkpoint=${5:-}
pilot_lr=${ALPHA_PILOT_LR:-3e-4}
pilot_lr_min=$(PILOT_LR="$pilot_lr" node -e '
  const lr = Number(process.env.PILOT_LR);
  if (!Number.isFinite(lr) || lr <= 0) throw new Error(`invalid ALPHA_PILOT_LR: ${process.env.PILOT_LR}`);
  process.stdout.write(String(lr / 10));
')

case "$variant" in
  llama)
    expected_params=57688576
    architecture_args=(
      --layers=16
      --normType=rmsnorm
      --posEnc=rope
      --ropeTheta=10000
      --tieEmbeddings=true
    )
    ;;
  gpt2)
    expected_params=58094592
    architecture_args=(
      --layers=14
      --normType=layernorm
      --posEnc=learned
      --tieEmbeddings=false
    )
    ;;
  *)
    echo "unknown variant: $variant (expected llama or gpt2)" >&2
    exit 2
    ;;
esac

for required in "$data" "$tokenizer" apps/cli/dist/main.js; do
  if [[ ! -f "$required" ]]; then
    echo "required file missing: $required" >&2
    exit 1
  fi
done
if [[ -n "$resume_checkpoint" ]]; then
  if [[ ! -d "$run_dir" || ! -f "$resume_checkpoint" || ! -f "$run_dir/pilot-contract.json" ]]; then
    echo "resume requires an existing run directory, checkpoint, and pilot contract" >&2
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
data_sha256=$(sha256sum "$data" | awk '{print $1}')
tokenizer_sha256=$(sha256sum "$tokenizer" | awk '{print $1}')
contract_tmp="$run_dir/pilot-contract.json.tmp"
resume_args=()
if [[ -n "$resume_checkpoint" ]]; then
  VARIANT="$variant" EXPECTED_PARAMS="$expected_params" SOURCE_COMMIT="$source_commit" \
  DATA_PATH="$data" DATA_SHA256="$data_sha256" TOKENIZER_PATH="$tokenizer" \
  TOKENIZER_SHA256="$tokenizer_sha256" PILOT_LR="$pilot_lr" PILOT_LR_MIN="$pilot_lr_min" \
  CONTRACT_PATH="$run_dir/pilot-contract.json" node -e '
    const fs = require("node:fs");
    const c = JSON.parse(fs.readFileSync(process.env.CONTRACT_PATH, "utf8"));
    const expected = {
      variant: process.env.VARIANT,
      expected_params: Number(process.env.EXPECTED_PARAMS),
      expected_steps: 6104,
      expected_tokens: 100007936,
      minimum_train_tokens: 100007936,
      learning_rate: Number(process.env.PILOT_LR),
      learning_rate_min: Number(process.env.PILOT_LR_MIN),
      source_commit: process.env.SOURCE_COMMIT,
    };
    for (const [key, value] of Object.entries(expected)) {
      if (c[key] !== value) throw new Error(`resume contract ${key}: ${c[key]} != ${value}`);
    }
    if (c.data?.path !== process.env.DATA_PATH || c.data?.sha256 !== process.env.DATA_SHA256) throw new Error("resume data contract mismatch");
    if (c.tokenizer?.path !== process.env.TOKENIZER_PATH || c.tokenizer?.sha256 !== process.env.TOKENIZER_SHA256) throw new Error("resume tokenizer contract mismatch");
  '
  npx tsx scripts/prepare_resume_metrics.ts \
    --run "$run_dir" --checkpoint "$resume_checkpoint" --sourceCommit "$source_commit"
  resume_args=(--resume="$resume_checkpoint")
else
  VARIANT="$variant" EXPECTED_PARAMS="$expected_params" SOURCE_COMMIT="$source_commit" \
  DATA_PATH="$data" DATA_SHA256="$data_sha256" TOKENIZER_PATH="$tokenizer" \
  TOKENIZER_SHA256="$tokenizer_sha256" CONTRACT_TMP="$contract_tmp" PILOT_LR="$pilot_lr" \
  PILOT_LR_MIN="$pilot_lr_min" node -e '
  const fs = require("node:fs");
  const contract = {
    schema: "alpha-g3-pilot-contract-v1",
    variant: process.env.VARIANT,
    expected_params: Number(process.env.EXPECTED_PARAMS),
    expected_steps: 6104,
    batch_size: 16,
    block_size: 1024,
    grad_accum_steps: 1,
    expected_tokens: 100007936,
    minimum_train_tokens: 100007936,
    learning_rate: Number(process.env.PILOT_LR),
    learning_rate_min: Number(process.env.PILOT_LR_MIN),
    source_commit: process.env.SOURCE_COMMIT,
    data: { path: process.env.DATA_PATH, sha256: process.env.DATA_SHA256 },
    tokenizer: { path: process.env.TOKENIZER_PATH, sha256: process.env.TOKENIZER_SHA256 },
    started_utc: new Date().toISOString(),
  };
  fs.writeFileSync(process.env.CONTRACT_TMP, JSON.stringify(contract, null, 2) + "\n", { flag: "wx" });
'
  mv "$contract_tmp" "$run_dir/pilot-contract.json"
fi
exec nice -n 5 ionice -c 2 -n 7 node --expose-gc apps/cli/dist/main.js train \
  --data="$data" \
  --domain=alpha_llama \
  --tokenizerArtifacts="$tokenizer" \
  --vocabSize=12288 \
  --block=1024 \
  --dim=512 \
  --heads=8 \
  --dropout=0 \
  --activation=swiglu \
  --ffnDim=1408 \
  --batch=16 \
  --accumSteps=1 \
  --steps=6104 \
  --lr="$pilot_lr" \
  --lrMin="$pilot_lr_min" \
  --warmupIters=500 \
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
  --evalInterval=500 \
  --checkpointInterval=1000 \
  --evalIters=5 \
  --logEvery=25 \
  --sampleInterval=0 \
  --postSamples=false \
  --remote=false \
  --packed=true \
  --runDir="$run_dir" \
  "${resume_args[@]}" \
  "${architecture_args[@]}"
