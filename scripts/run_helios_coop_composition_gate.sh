#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_root="${1:-/mnt/donto-data/donto-resources/benchmarks/alpha-helios-coop-composition-${timestamp}}"

: "${TRAIN_DATA:?set TRAIN_DATA to the frozen training prefix}"
: "${VAL_DATA:?set VAL_DATA to the wholly held-out validation prefix}"
: "${TOKENIZER:?set TOKENIZER to the exact tokenizer artifact}"
: "${PROFITABLE_SHAPES:?set PROFITABLE_SHAPES to a comma-separated MxNxK allow-list}"

steps="${STEPS:-3}"
batch="${BATCH:-10}"
eval_interval="${EVAL_INTERVAL:-0}"
eval_iters="${EVAL_ITERS:-1}"
control_shapes="${CONTROL_SHAPES:-}"
mkdir -p "$output_root"
cd "$repo_root"

source_commit="${SOURCE_COMMIT_OVERRIDE:-}"
if [[ -z "$source_commit" ]]; then
  source_commit="$(git rev-parse HEAD)"
fi
printf '%s\n' "$source_commit" > "$output_root/SOURCE-COMMIT.txt"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git status --short > "$output_root/SOURCE-STATUS.txt"
  git diff --binary > "$output_root/SOURCE-DIFF.patch"
else
  printf '%s\n' "source package supplied with SOURCE_COMMIT_OVERRIDE=$source_commit" > "$output_root/SOURCE-STATUS.txt"
  : > "$output_root/SOURCE-DIFF.patch"
fi
sha256sum "$TRAIN_DATA" "$VAL_DATA" "$TOKENIZER" > "$output_root/INPUT-HASHES.sha256"
sha256sum \
  packages/helios/src/backend.ts \
  packages/helios/src/kernels/matmul-coop.ts \
  packages/helios/native/helios_vk.c \
  packages/train/src/trainer.ts \
  scripts/run_helios_coop_composition_gate.sh \
  scripts/summarize_helios_coop_bisect.mjs \
  package-lock.json > "$output_root/SOURCE-HASHES.sha256"

{
  date -u --iso-8601=seconds
  uname -a
  node --version
  npm --version
  df -h / /workspace 2>/dev/null || df -h /
  free -h
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,power.limit,clocks.max.sm,clocks.max.memory --format=csv,noheader
} > "$output_root/HOST.txt" 2>&1

HELIOS_NATIVE_FORCE_REBUILD=1 npm run build -w @alpha/helios > "$output_root/build.log" 2>&1
npx turbo build --filter=@alpha/cli... >> "$output_root/build.log" 2>&1

run_row() {
  local name="$1"
  local coop_mode="$2"
  local shape_allow="$3"
  local reclaim_mode="$4"
  local row_root="$output_root/$name"
  mkdir -p "$row_root"

  local -a controlled_env=(
    "VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd_headless.json}"
    "HELIOS_DISABLE_DGC=1"
    "HELIOS_DISABLE_TEMP_SLABS=0"
    "HELIOS_TEMP_SLAB_POOL_MB=12288"
    "HELIOS_EXACT_BUFFER_SIZES=0"
    "HELIOS_FLASH_FWD_PREFER_COOP2=0"
    "HELIOS_WG_SIZE=64"
    "HELIOS_MATMUL_REG4X2=1"
    "HELIOS_MATMUL_REG4X2_TRANSPOSED_B=1"
    "HELIOS_MATMUL_TRANSPOSED_B_COALESCED=1"
    "HELIOS_MATMUL_TRANSPOSED_A_COALESCED=1"
    "HELIOS_MATMUL_REG2X2=1"
    "HELIOS_OUTPUT_POOL_LARGE_PER_CLASS=48"
    "HELIOS_MAX_OUTPUT_POOL_ENTRIES=512"
    "HELIOS_PROFILE_GPU_OPS=1"
    "HELIOS_PROFILE_GPU_TIMESTAMPS=0"
    "HELIOS_COOP_REPORT_SHAPES=1"
    "HELIOS_ENABLE_COOP_BACKWARD=${HELIOS_ENABLE_COOP_BACKWARD:-0}"
    "HELIOS_COOP_PRECAST_F16_INPUT=${HELIOS_COOP_PRECAST_F16_INPUT:-1}"
    "HELIOS_COOP_SHAPE_ALLOW=$shape_allow"
    "ALPHA_GPU_METRICS_SAMPLE_EVERY=1"
    "ALPHA_DISABLE_CHECKPOINTS=1"
    "ALPHA_FAIL_ON_SMOKE_TEST=1"
    "ALPHA_SAMPLE_FROM_CHECKPOINT=0"
  )
  local -a reclaim_args=("--syncEvery=0" "--gcEvery=0")
  if [[ "$coop_mode" == "off" ]]; then
    controlled_env+=("HELIOS_DISABLE_COOP_MAT=1")
  else
    controlled_env+=("HELIOS_DISABLE_COOP_MAT=0")
  fi
  if [[ "$reclaim_mode" == "forced" ]]; then
    controlled_env+=(
      "ALPHA_ADAPTIVE_PURGE_LIVE_ALLOCS_THRESHOLD=800"
      "ALPHA_ADAPTIVE_PURGE_MIN_INTERVAL=1"
    )
    reclaim_args=("--syncEvery=1" "--gcEvery=1")
  fi
  printf '%s\n' "${controlled_env[@]}" > "$row_root/CONTROLLED-ENVIRONMENT.txt"
  printf '%s\n' "${reclaim_args[@]}" > "$row_root/CONTROLLED-ARGS.txt"

  set +e
  env "${controlled_env[@]}" nice -n 5 ionice -c 2 -n 7 \
    node --expose-gc apps/cli/dist/main.js train \
      --data="$TRAIN_DATA" --valData="$VAL_DATA" --requireValData=true --sft=false \
      --domain=alpha_llama --tokenizerArtifacts="$TOKENIZER" \
      --vocabSize=12288 --block=1024 --layers=18 --dim=640 --heads=10 --dropout=0 \
      --activation=swiglu --ffnDim=1728 --normType=rmsnorm --posEnc=rope --ropeTheta=10000 \
      --tieEmbeddings=true --backend=helios --gpuProfile=none --optim=adamw \
      --batch="$batch" --accumSteps=1 --steps="$steps" --lr=0.002 --lrMin=0.0002 \
      --warmupIters=790 --beta1=0.9 --beta2=0.95 --eps=0.00000001 --weightDecay=0.1 \
      --gradClip=1.0 --spikeThreshold=50 --evalInterval="$eval_interval" --checkpointInterval=1000000 \
      --evalIters="$eval_iters" --sampleInterval=0 --logEvery=1 --seed=42 --strictPlanning=false \
      --remote=false --fp16=false --minGpuSize=1 --no-fallback=true --packed=true \
      --symbio=false --postSamples=false --trace=true "${reclaim_args[@]}" \
      --runDir="$row_root" > "$row_root/console.log" 2>&1
  local status=$?
  set -e
  printf '%s\n' "$status" > "$row_root/exit-code.txt"
}

run_row baseline_fp32 off "" default
if [[ -n "$control_shapes" ]]; then
  run_row control_composition on "$control_shapes" default
fi
run_row profitable_four_default on "$PROFITABLE_SHAPES" default
if [[ "${SOAK_ONLY:-0}" != "1" ]]; then
  run_row cooperative_all_default on "" default
  run_row cooperative_all_forced_reclaim on "" forced
  run_row profitable_four_forced_reclaim on "$PROFITABLE_SHAPES" forced
fi

node scripts/summarize_helios_coop_bisect.mjs --root "$output_root"

# Write the final path before hashing so the runner's captured stdout cannot
# mutate a file that the manifest already claims to cover.
printf '%s\n' "$output_root"
(
  cd "$output_root"
  find . -type f ! -name ARTIFACTS.sha256 ! -name hash-check.log -print0 \
    | sort -z | xargs -0 sha256sum > ARTIFACTS.sha256
  sha256sum -c ARTIFACTS.sha256 > hash-check.log
)
