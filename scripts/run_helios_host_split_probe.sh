#!/usr/bin/env bash
set -euo pipefail

: "${TRAIN_DATA:?set TRAIN_DATA to an immutable pretraining text file}"
: "${VAL_DATA:?set VAL_DATA to an immutable held-out validation text file}"
: "${TOKENIZER:?set TOKENIZER to the byte-BPE tokenizer artifact}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_dir="${1:-/mnt/donto-data/donto-resources/benchmarks/alpha-helios-host-split-${timestamp}}"
steps="${HOST_SPLIT_STEPS:-3}"

if [[ ! "$steps" =~ ^[0-9]+$ ]] || (( steps < 2 )); then
  echo "HOST_SPLIT_STEPS must be an integer >= 2" >&2
  exit 2
fi
for required in "$TRAIN_DATA" "$VAL_DATA" "$TOKENIZER"; do
  [[ -f "$required" ]] || { echo "required input missing: $required" >&2; exit 2; }
done
if [[ -e "$output_dir" ]]; then
  echo "output directory already exists: $output_dir" >&2
  exit 2
fi

mkdir -p "$output_dir/run"
cd "$repo_root"

git rev-parse HEAD > "$output_dir/SOURCE-COMMIT.txt"
git status --porcelain=v1 --untracked-files=all > "$output_dir/SOURCE-STATUS.txt"
git diff --binary HEAD > "$output_dir/SOURCE-DIFF.patch"
sha256sum "$TRAIN_DATA" "$VAL_DATA" "$TOKENIZER" > "$output_dir/INPUT-HASHES.sha256"
sha256sum \
  packages/helios/src/backend.ts \
  packages/helios/native/helios_vk.c \
  packages/train/src/trainer.ts \
  apps/cli/dist/main.js \
  scripts/summarize_helios_profile.mjs \
  scripts/run_helios_host_split_probe.sh \
  package-lock.json > "$output_dir/SOURCE-HASHES.sha256"

{
  date -u --iso-8601=seconds
  uname -a
  node --version
  npm --version
  df -h / /workspace 2>/dev/null || df -h /
  free -h
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,power.limit,clocks.max.sm,clocks.max.memory --format=csv,noheader
} > "$output_dir/HOST.txt" 2>&1

controlled_env=(
  "VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd_headless.json}"
  # DGC is not selected by this 1,444-op training graph (the profiler reports
  # dgc=0), so do not reserve its preprocess buffer during this tight-VRAM
  # measurement. This changes setup memory only, not the dispatched graph.
  "HELIOS_DISABLE_DGC=1"
  "HELIOS_DISABLE_COOP_MAT=1"
  "HELIOS_FLASH_FWD_PREFER_COOP2=0"
  "HELIOS_WG_SIZE=64"
  "HELIOS_MATMUL_REG4X2=1"
  "HELIOS_MATMUL_REG4X2_TRANSPOSED_B=1"
  "HELIOS_MATMUL_TRANSPOSED_B_COALESCED=1"
  "HELIOS_MATMUL_TRANSPOSED_A_COALESCED=1"
  "HELIOS_MATMUL_REG2X2=1"
  "HELIOS_MAX_OUTPUT_POOL_ENTRIES=512"
  "HELIOS_PROFILE_GPU_OPS=1"
  "HELIOS_PROFILE_GPU_TIMESTAMPS=1"
  "ALPHA_GPU_METRICS_SAMPLE_EVERY=1"
  "ALPHA_DISABLE_CHECKPOINTS=1"
  "ALPHA_FAIL_ON_SMOKE_TEST=1"
  "ALPHA_SAMPLE_FROM_CHECKPOINT=0"
)
printf '%s\n' "${controlled_env[@]}" > "$output_dir/CONTROLLED-ENVIRONMENT.txt"

set +e
env "${controlled_env[@]}" nice -n 5 ionice -c 2 -n 7 \
  node --expose-gc apps/cli/dist/main.js train \
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
    --runDir="$output_dir/run" \
    > "$output_dir/console.log" 2>&1
train_status=$?
set -e
printf '%s\n' "$train_status" > "$output_dir/exit-code.txt"
if (( train_status != 0 )); then
  tail -100 "$output_dir/console.log" >&2
  exit "$train_status"
fi

node scripts/summarize_helios_profile.mjs --format json "$output_dir/console.log" \
  > "$output_dir/profile-summary.json"
node scripts/summarize_helios_profile.mjs "$output_dir/console.log" \
  > "$output_dir/profile-summary.md"

jq -e --argjson steps "$steps" '
  .sampleCount == $steps and
  .averages.hostBuildMs != null and
  .averages.gpuBlockingMs != null and
  .averages.coreStepMs != null and
  .averages.dispatchGpuUs > 0 and
  .averages.hostBuildMs >= 0 and
  .averages.gpuBlockingMs >= 0
' "$output_dir/profile-summary.json" > /dev/null

jq -r '
  "# Helios exact-foundation host/GPU split\n",
  "**Samples:** " + (.sampleCount | tostring),
  "**Average core step:** " + (.averages.coreStepMs | tostring) + " ms",
  "**Average host build:** " + (.averages.hostBuildMs | tostring) + " ms (" + ((100 * .averages.hostBuildMs / .averages.coreStepMs) | tostring) + "%)",
  "**Average synchronous GPU blocking:** " + (.averages.gpuBlockingMs | tostring) + " ms (" + ((100 * .averages.gpuBlockingMs / .averages.coreStepMs) | tostring) + "%)",
  "**Average timestamped GPU dispatch:** " + ((.averages.dispatchGpuUs / 1000) | tostring) + " ms",
  "**Average timestamped GPU batch:** " + ((.averages.batchGpuUs / 1000) | tostring) + " ms",
  "",
  "`host_build_ms + gpu_blocking_ms` partitions the measured pre-metrics step wall. Timestamped dispatch time is an independent sum of kernel intervals and may exceed blocking wall when useful host/GPU overlap exists. This probe uses the exact 18-layer, d=640, FFN=1728, context=1024, batch=24 foundation shape and the selected portable FP32 kernel policy."
' "$output_dir/profile-summary.json" > "$output_dir/README.md"

(
  cd "$output_dir"
  find . -type f \
    ! -name ARTIFACTS.sha256 \
    ! -name hash-check.log \
    -print0 \
    | sort -z \
    | xargs -0 sha256sum > ARTIFACTS.sha256
  sha256sum -c ARTIFACTS.sha256 > hash-check.log
)

echo "$output_dir"
