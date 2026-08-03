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
baseline="${ALLOCATOR_FACTORIAL_BASELINE:-exact_individual}"
workload_batch="${ALLOCATOR_FACTORIAL_BATCH:-24}"
workload_block="${ALLOCATOR_FACTORIAL_BLOCK:-1024}"
workload_layers="${ALLOCATOR_FACTORIAL_LAYERS:-18}"
workload_dim="${ALLOCATOR_FACTORIAL_DIM:-640}"
workload_heads="${ALLOCATOR_FACTORIAL_HEADS:-10}"
workload_ffn_dim="${ALLOCATOR_FACTORIAL_FFN_DIM:-1728}"

if [[ ! "$steps" =~ ^[0-9]+$ ]] || (( steps < 4 )); then
  echo "ALLOCATOR_FACTORIAL_STEPS must be an integer >= 4" >&2
  exit 2
fi
if [[ ! "$warmup" =~ ^[0-9]+$ ]] || (( warmup < 0 || warmup >= steps )); then
  echo "ALLOCATOR_FACTORIAL_WARMUP must be an integer in [0, steps)" >&2
  exit 2
fi
for workload_value in \
  "$workload_batch" "$workload_block" "$workload_layers" \
  "$workload_dim" "$workload_heads" "$workload_ffn_dim"; do
  if [[ ! "$workload_value" =~ ^[0-9]+$ ]] || (( workload_value < 1 )); then
    echo "allocator workload dimensions must be positive integers" >&2
    exit 2
  fi
done
if (( workload_dim % workload_heads != 0 )); then
  echo "ALLOCATOR_FACTORIAL_DIM must be divisible by ALLOCATOR_FACTORIAL_HEADS" >&2
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
  scripts/summarize_helios_graph_trace.mjs \
  scripts/analyze_helios_buffer_lifetimes.mjs \
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

foundation_shape=false
if (( workload_batch == 24 && workload_block == 1024 && workload_layers == 18 &&
      workload_dim == 640 && workload_heads == 10 && workload_ffn_dim == 1728 )); then
  foundation_shape=true
fi
jq -n \
  --argjson batch "$workload_batch" \
  --argjson block "$workload_block" \
  --argjson layers "$workload_layers" \
  --argjson dim "$workload_dim" \
  --argjson heads "$workload_heads" \
  --argjson ffnDim "$workload_ffn_dim" \
  --argjson foundationShape "$foundation_shape" \
  '{batch:$batch, block:$block, layers:$layers, dim:$dim, heads:$heads, ffn_dim:$ffnDim, exact_foundation_shape:$foundationShape}' \
  > "$output_dir/WORKLOAD.json"

cat_description="$output_dir/EXPERIMENT.txt"
printf '%s\n' \
  "Workload: ${workload_layers} layers, d=${workload_dim}, FFN=${workload_ffn_dim}, context=${workload_block}, batch=${workload_batch}; exact_foundation_shape=${foundation_shape}." \
  "Factorial variables: coarse versus exact allocation classes; native temporary slabs versus individual VkDeviceMemory allocations." \
  "Every mode runs in a fresh Node process. GPU timestamp queries are disabled so the outcome is sustained end-to-end token throughput." \
  "The first ${warmup} steps are excluded from summary statistics." \
  > "$cat_description"

read -r -a modes <<< "${ALLOCATOR_FACTORIAL_MODES:-coarse_slabs exact_slabs coarse_individual exact_individual}"
overall_status=0
for mode in "${modes[@]}"; do
  mode_dir="$output_dir/modes/$mode"
  mkdir -p "$mode_dir/run"
  case "$mode" in
    coarse_slabs) exact=0; disable_slabs=0; slab_mb=8192; large_per_class=8; pool_entries=512 ;;
    exact_slabs) exact=1; disable_slabs=0; slab_mb=8192; large_per_class=8; pool_entries=512 ;;
    coarse_individual) exact=0; disable_slabs=1; slab_mb=8192; large_per_class=8; pool_entries=512 ;;
    exact_individual) exact=1; disable_slabs=1; slab_mb=8192; large_per_class=8; pool_entries=512 ;;
    slab8_pool8) exact=0; disable_slabs=0; slab_mb=8192; large_per_class=8; pool_entries=512 ;;
    slab16_pool8) exact=0; disable_slabs=0; slab_mb=16384; large_per_class=8; pool_entries=512 ;;
    slab8_pool32) exact=0; disable_slabs=0; slab_mb=8192; large_per_class=32; pool_entries=1024 ;;
    slab8_pool64) exact=0; disable_slabs=0; slab_mb=8192; large_per_class=64; pool_entries=2048 ;;
    slab12_pool48) exact=0; disable_slabs=0; slab_mb=12288; large_per_class=48; pool_entries=1536 ;;
    slab16_pool32) exact=0; disable_slabs=0; slab_mb=16384; large_per_class=32; pool_entries=1024 ;;
    slab16_pool64) exact=0; disable_slabs=0; slab_mb=16384; large_per_class=64; pool_entries=2048 ;;
    *) echo "unknown mode: $mode" >&2; exit 2 ;;
  esac
  jq -n \
    --arg mode "$mode" \
    --argjson exact "$exact" \
    --argjson disableSlabs "$disable_slabs" \
    --argjson slabMb "$slab_mb" \
    --argjson largePerClass "$large_per_class" \
    --argjson poolEntries "$pool_entries" \
    --argjson steps "$steps" \
    --argjson warmup "$warmup" \
    '{mode:$mode, exact_buffer_sizes:($exact == 1), temporary_slabs:($disableSlabs == 0), temp_slab_pool_mb:$slabMb, output_pool_large_per_class:$largePerClass, output_pool_entries:$poolEntries, steps:$steps, warmup_excluded:$warmup}' \
    > "$mode_dir/MODE.json"

  controlled_env=(
    "VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd_headless.json}"
    "HELIOS_DISABLE_DGC=1"
    "HELIOS_DISABLE_TEMP_SLABS=$disable_slabs"
    "HELIOS_TEMP_SLAB_POOL_MB=$slab_mb"
    "HELIOS_EXACT_BUFFER_SIZES=$exact"
    "HELIOS_DISABLE_COOP_MAT=1"
    "HELIOS_FLASH_FWD_PREFER_COOP2=0"
    "HELIOS_WG_SIZE=64"
    "HELIOS_MATMUL_REG4X2=1"
    "HELIOS_MATMUL_REG4X2_TRANSPOSED_B=1"
    "HELIOS_MATMUL_TRANSPOSED_B_COALESCED=1"
    "HELIOS_MATMUL_TRANSPOSED_A_COALESCED=1"
    "HELIOS_MATMUL_REG2X2=1"
    "HELIOS_OUTPUT_POOL_LARGE_PER_CLASS=$large_per_class"
    "HELIOS_MAX_OUTPUT_POOL_ENTRIES=$pool_entries"
    "HELIOS_PROFILE_GPU_OPS=1"
    "HELIOS_PROFILE_GRAPH_SIGNATURE=1"
    "HELIOS_PROFILE_GRAPH_TRACE=${HELIOS_PROFILE_GRAPH_TRACE:-0}"
    "HELIOS_PROFILE_GPU_TIMESTAMPS=0"
    "ALPHA_GPU_METRICS_SAMPLE_EVERY=1"
    "ALPHA_WRITE_GRAPH_TRACE=${ALPHA_WRITE_GRAPH_TRACE:-0}"
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
      --block="$workload_block" \
      --layers="$workload_layers" \
      --dim="$workload_dim" \
      --heads="$workload_heads" \
      --dropout=0 \
      --activation=swiglu \
      --ffnDim="$workload_ffn_dim" \
      --normType=rmsnorm \
      --posEnc=rope \
      --ropeTheta=10000 \
      --tieEmbeddings=true \
      --batch="$workload_batch" \
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
  if [[ -s "$mode_dir/run/gpu-graph-trace.jsonl" ]]; then
    node scripts/summarize_helios_graph_trace.mjs \
      "$mode_dir/run/gpu-graph-trace.jsonl" --json \
      > "$mode_dir/graph-trace-summary.json"
    node scripts/summarize_helios_graph_trace.mjs \
      "$mode_dir/run/gpu-graph-trace.jsonl" \
      > "$mode_dir/GRAPH-TRACE.md"
    node scripts/analyze_helios_buffer_lifetimes.mjs \
      "$mode_dir/run/gpu-graph-trace.jsonl" --json \
      > "$mode_dir/buffer-lifetime-analysis.json"
    node scripts/analyze_helios_buffer_lifetimes.mjs \
      "$mode_dir/run/gpu-graph-trace.jsonl" \
      > "$mode_dir/BUFFER-LIFETIME-ANALYSIS.md"
  fi
  if (( status != 0 )); then
    overall_status=1
    tail -100 "$mode_dir/console.log" >&2
  fi
done

node scripts/summarize_helios_allocator_factorial.mjs "$output_dir" --warmup "$warmup" --baseline "$baseline" --format json > "$output_dir/summary.json"
node scripts/summarize_helios_allocator_factorial.mjs "$output_dir" --warmup "$warmup" --baseline "$baseline" > "$output_dir/README.md"

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
