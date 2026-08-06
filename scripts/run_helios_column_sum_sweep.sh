#!/usr/bin/env bash

# Correctness-gated physical-device sweep for RMSNorm's row-parallel column
# reduction. The mirrored order limits warm-up bias, while separate timestamped
# and sustained phases keep diagnostic dispatch timing distinct from production
# tokens/s.

set -euo pipefail

: "${TRAIN_DATA:?set TRAIN_DATA to one immutable pretraining shard}"
: "${VAL_DATA:?set VAL_DATA to the held-out validation corpus}"
: "${TOKENIZER:?set TOKENIZER to the byte-BPE tokenizer artifact}"

OUT_ROOT=${OUT_ROOT:-/tmp/alpha-helios-column-sum-sweep}
PROFILE_STEPS=${PROFILE_STEPS:-1}
SUSTAINED_STEPS=${SUSTAINED_STEPS:-20}
NODE_BIN=${NODE_BIN:-node}

for value_name in PROFILE_STEPS SUSTAINED_STEPS; do
  value=${!value_name}
  if [[ ! "$value" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    echo "$value_name must be a positive integer" >&2
    exit 2
  fi
done
for file in "$TRAIN_DATA" "$VAL_DATA" "$TOKENIZER" apps/cli/dist/main.js scripts/summarize_helios_profile.mjs; do
  [[ -f "$file" ]] || { echo "required file missing: $file" >&2; exit 2; }
done
if [[ -e "$OUT_ROOT" ]]; then
  echo "OUT_ROOT already exists; choose a new immutable evidence directory: $OUT_ROOT" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT"
sha256sum "$TRAIN_DATA" "$VAL_DATA" "$TOKENIZER" > "$OUT_ROOT/INPUT-HASHES.sha256"
git rev-parse HEAD > "$OUT_ROOT/SOURCE-COMMIT.txt"
git status --porcelain=v1 --untracked-files=all > "$OUT_ROOT/SOURCE-STATUS.txt"
git diff --binary HEAD > "$OUT_ROOT/SOURCE-DIFF.patch"
{
  "$NODE_BIN" --version
  npm --version
  uname -a
} > "$OUT_ROOT/RUNTIME.txt"
sha256sum \
  packages/helios/src/backend.ts \
  packages/helios/src/kernels/index.ts \
  packages/helios/src/kernels/reduction.ts \
  packages/train/src/trainer.ts \
  apps/cli/dist/main.js \
  package-lock.json \
  > "$OUT_ROOT/SOURCE-FILES.sha256"

run_row() {
  local name=$1
  local row_lanes=$2
  local steps=$3
  local timestamped=$4
  local run_dir="$OUT_ROOT/$name"
  local log="$run_dir/console.log"
  mkdir "$run_dir"

  local -a env_args=(
    "HELIOS_DISABLE_COOP_MAT=1"
    "HELIOS_FLASH_FWD_PREFER_COOP2=0"
    "HELIOS_WG_SIZE=64"
    "HELIOS_MATMUL_REG4X2=1"
    "HELIOS_MATMUL_REG4X2_TRANSPOSED_B=1"
    "HELIOS_MATMUL_TRANSPOSED_B_COALESCED=1"
    "HELIOS_MATMUL_TRANSPOSED_A_COALESCED=1"
    "HELIOS_MATMUL_REG2X2=1"
    "HELIOS_MAX_OUTPUT_POOL_ENTRIES=512"
    "HELIOS_COLUMN_SUM_ROW_LANES=$row_lanes"
    "HELIOS_PROFILE_GPU_OPS=1"
    "HELIOS_PROFILE_GPU_TIMESTAMPS=$timestamped"
    "ALPHA_GPU_METRICS_SAMPLE_EVERY=1"
    "ALPHA_DISABLE_CHECKPOINTS=1"
    "ALPHA_FAIL_ON_SMOKE_TEST=1"
    "ALPHA_SAMPLE_FROM_CHECKPOINT=0"
  )
  if [[ -n "${VK_ICD_FILENAMES:-}" ]]; then
    env_args+=("VK_ICD_FILENAMES=$VK_ICD_FILENAMES")
  elif [[ -f /etc/vulkan/icd.d/nvidia_icd_headless.json ]]; then
    env_args+=("VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd_headless.json")
  fi
  printf '%s\n' "${env_args[@]}" > "$run_dir/controlled-environment.txt"
  printf 'name=%s row_lanes=%s steps=%s timestamped=%s\n' \
    "$name" "$row_lanes" "$steps" "$timestamped" > "$run_dir/row.txt"

  set +e
  env "${env_args[@]}" nice -n 5 ionice -c 2 -n 7 \
    "$NODE_BIN" --expose-gc apps/cli/dist/main.js train \
      --data="$TRAIN_DATA" \
      --valData="$VAL_DATA" \
      --requireValData=true \
      --domain=alpha_llama \
      --tokenizerArtifacts="$TOKENIZER" \
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
      --steps="$steps" \
      --lr=0.002 \
      --lrMin=0.0002 \
      --warmupIters=790 \
      --beta1=0.9 \
      --beta2=0.95 \
      --eps=0.00000001 \
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
      --evalInterval="$steps" \
      --checkpointInterval="$steps" \
      --evalIters=1 \
      --logEvery=1 \
      --sampleInterval=0 \
      --postSamples=false \
      --remote=false \
      --packed=true \
      --symbio=false \
      --trace=true \
      --runDir="$run_dir" \
      2>&1 | tee "$log"
  local status=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$status" > "$run_dir/exit-code.txt"
  if (( status != 0 )); then
    echo "row failed: $name (exit $status)" >&2
    return "$status"
  fi

  if [[ "$timestamped" == "1" ]]; then
    "$NODE_BIN" scripts/summarize_helios_profile.mjs "$log" > "$run_dir/profile-summary.md"
    "$NODE_BIN" scripts/summarize_helios_profile.mjs --format json "$log" > "$run_dir/profile-summary.json"
  fi
}

# Mirrored exact-profile order reduces systematic warm-up/order bias.
run_row profile-control-a 0  "$PROFILE_STEPS" 1
run_row profile-lanes4-a  4  "$PROFILE_STEPS" 1
run_row profile-lanes8-a  8  "$PROFILE_STEPS" 1
run_row profile-lanes16-a 16 "$PROFILE_STEPS" 1
run_row profile-lanes16-b 16 "$PROFILE_STEPS" 1
run_row profile-lanes8-b  8  "$PROFILE_STEPS" 1
run_row profile-lanes4-b  4  "$PROFILE_STEPS" 1
run_row profile-control-b 0  "$PROFILE_STEPS" 1

# The sustained phase retains the same mirrored order and disables timestamp
# instrumentation. All rows still record operation identities and trajectories.
run_row sustained-control-a 0  "$SUSTAINED_STEPS" 0
run_row sustained-lanes4-a  4  "$SUSTAINED_STEPS" 0
run_row sustained-lanes8-a  8  "$SUSTAINED_STEPS" 0
run_row sustained-lanes16-a 16 "$SUSTAINED_STEPS" 0
run_row sustained-lanes16-b 16 "$SUSTAINED_STEPS" 0
run_row sustained-lanes8-b  8  "$SUSTAINED_STEPS" 0
run_row sustained-lanes4-b  4  "$SUSTAINED_STEPS" 0
run_row sustained-control-b 0  "$SUSTAINED_STEPS" 0

find "$OUT_ROOT" -type f ! -name ARTIFACTS.sha256 -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$OUT_ROOT/ARTIFACTS.sha256"
sha256sum -c "$OUT_ROOT/ARTIFACTS.sha256"
