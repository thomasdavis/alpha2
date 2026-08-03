#!/usr/bin/env bash
set -euo pipefail

: "${TRAIN_DATA:?set TRAIN_DATA to an immutable pretraining text file}"
: "${VAL_DATA:?set VAL_DATA to an immutable held-out validation text file}"
: "${TOKENIZER:?set TOKENIZER to the byte-BPE tokenizer artifact}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_dir="${1:-/mnt/donto-data/donto-resources/benchmarks/alpha-helios-allocator-factorial-${timestamp}}"
steps="${ALLOCATOR_FACTORIAL_STEPS:-12}"
warmup="${ALLOCATOR_FACTORIAL_WARMUP:-3}"

if [[ ! "$steps" =~ ^[0-9]+$ ]] || (( steps < 4 )); then
  echo "ALLOCATOR_FACTORIAL_STEPS must be an integer >= 4" >&2
  exit 2
fi
if [[ ! "$warmup" =~ ^[0-9]+$ ]] || (( warmup < 0 || warmup >= steps )); then
  echo "ALLOCATOR_FACTORIAL_WARMUP must be an integer in [0, steps)" >&2
  exit 2
fi
for required in "$TRAIN_DATA" "$VAL_DATA" "$TOKENIZER"; do
  [[ -f "$required" ]] || { echo "required input missing: $required" >&2; exit 2; }
done
if [[ -e "$output_dir" ]]; then
  echo "output directory already exists: $output_dir" >&2
  exit 2
fi

mkdir -p "$output_dir/modes"
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
  scripts/summarize_helios_allocator_factorial.mjs \
  scripts/run_helios_allocator_factorial.sh \
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

cat_description="$output_dir/EXPERIMENT.txt"
printf '%s\n' \
  "Exact foundation graph: 18 layers, d=640, FFN=1728, context=1024, batch=24." \
  "Factorial variables: coarse versus exact allocation classes; native temporary slabs versus individual VkDeviceMemory allocations." \
  "Every mode runs in a fresh Node process. GPU timestamp queries are disabled so the outcome is sustained end-to-end token throughput." \
  "The first ${warmup} steps are excluded from summary statistics." \
  > "$cat_description"

modes=(coarse_slabs exact_slabs coarse_individual exact_individual)
overall_status=0
for mode in "${modes[@]}"; do
  mode_dir="$output_dir/modes/$mode"
  mkdir -p "$mode_dir/run"
  case "$mode" in
    coarse_slabs) exact=0; disable_slabs=0 ;;
    exact_slabs) exact=1; disable_slabs=0 ;;
    coarse_individual) exact=0; disable_slabs=1 ;;
    exact_individual) exact=1; disable_slabs=1 ;;
    *) echo "unknown mode: $mode" >&2; exit 2 ;;
  esac
  jq -n \
    --arg mode "$mode" \
    --argjson exact "$exact" \
    --argjson disableSlabs "$disable_slabs" \
    --argjson steps "$steps" \
    --argjson warmup "$warmup" \
    '{mode:$mode, exact_buffer_sizes:($exact == 1), temporary_slabs:($disableSlabs == 0), steps:$steps, warmup_excluded:$warmup}' \
    > "$mode_dir/MODE.json"

  controlled_env=(
    "VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd_headless.json}"
    "HELIOS_DISABLE_DGC=1"
    "HELIOS_DISABLE_TEMP_SLABS=$disable_slabs"
    "HELIOS_EXACT_BUFFER_SIZES=$exact"
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
    "HELIOS_PROFILE_GPU_TIMESTAMPS=0"
    "ALPHA_GPU_METRICS_SAMPLE_EVERY=1"
    "ALPHA_DISABLE_CHECKPOINTS=1"
    "ALPHA_FAIL_ON_SMOKE_TEST=1"
    "ALPHA_SAMPLE_FROM_CHECKPOINT=0"
  )
  printf '%s\n' "${controlled_env[@]}" > "$mode_dir/CONTROLLED-ENVIRONMENT.txt"

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
      --evalInterval=1000000 \
      --checkpointInterval=1000000 \
      --evalIters=1 \
      --logEvery=1 \
      --sampleInterval=0 \
      --postSamples=false \
      --remote=false \
      --packed=true \
      --symbio=false \
      --trace=true \
      --runDir="$mode_dir/run" \
      > "$mode_dir/console.log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" > "$mode_dir/exit-code.txt"
  if (( status != 0 )); then
    overall_status=1
    tail -100 "$mode_dir/console.log" >&2
  fi
done

node scripts/summarize_helios_allocator_factorial.mjs "$output_dir" --warmup "$warmup" --format json > "$output_dir/summary.json"
node scripts/summarize_helios_allocator_factorial.mjs "$output_dir" --warmup "$warmup" > "$output_dir/README.md"

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

cat "$output_dir/README.md"
echo "$output_dir"
exit "$overall_status"
