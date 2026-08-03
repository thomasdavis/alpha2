#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_root="${1:-/mnt/donto-data/donto-resources/benchmarks/alpha-helios-coop-shape-bisect-${timestamp}}"

: "${TRAIN_DATA:?set TRAIN_DATA to the frozen training prefix}"
: "${VAL_DATA:?set VAL_DATA to the wholly held-out validation prefix}"
: "${TOKENIZER:?set TOKENIZER to the exact tokenizer artifact}"

steps="${STEPS:-2}"
batch="${BATCH:-10}"
mkdir -p "$output_root"
cd "$repo_root"

git rev-parse HEAD > "$output_root/SOURCE-COMMIT.txt"
git status --short > "$output_root/SOURCE-STATUS.txt"
git diff --binary > "$output_root/SOURCE-DIFF.patch"
sha256sum "$TRAIN_DATA" "$VAL_DATA" "$TOKENIZER" > "$output_root/INPUT-HASHES.sha256"
sha256sum \
  packages/helios/src/backend.ts \
  packages/helios/src/kernels/matmul-coop.ts \
  packages/helios/native/helios_vk.c \
  packages/train/src/trainer.ts \
  scripts/run_helios_coop_shape_bisect.sh \
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
npx tsc -b >> "$output_root/build.log" 2>&1

run_row() {
  local name="$1"
  local coop_mode="$2"
  local shape_allow="${3:-}"
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
    "HELIOS_ENABLE_COOP_BACKWARD=0"
    "HELIOS_COOP_SHAPE_ALLOW=$shape_allow"
    "ALPHA_GPU_METRICS_SAMPLE_EVERY=1"
    "ALPHA_DISABLE_CHECKPOINTS=1"
    "ALPHA_FAIL_ON_SMOKE_TEST=1"
    "ALPHA_SAMPLE_FROM_CHECKPOINT=0"
  )
  if [[ "$coop_mode" == "off" ]]; then
    controlled_env+=("HELIOS_DISABLE_COOP_MAT=1")
  else
    controlled_env+=("HELIOS_DISABLE_COOP_MAT=0")
  fi
  printf '%s\n' "${controlled_env[@]}" > "$row_root/CONTROLLED-ENVIRONMENT.txt"

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
      --gradClip=1.0 --spikeThreshold=50 --evalInterval=0 --checkpointInterval=1000000 \
      --evalIters=1 --sampleInterval=0 --logEvery=1 --seed=42 --strictPlanning=false \
      --remote=false --fp16=false --minGpuSize=1 --no-fallback=true --packed=true \
      --symbio=false --postSamples=false --trace=true --runDir="$row_root" \
      > "$row_root/console.log" 2>&1
  local status=$?
  set -e
  printf '%s\n' "$status" > "$row_root/exit-code.txt"
}

run_row baseline_fp32 off
run_row cooperative_all on

mapfile -t shapes < <(node - "$output_root/cooperative_all/console.log" <<'NODE'
const fs = require("node:fs");
const log = fs.readFileSync(process.argv[2], "utf8");
const lines = [...log.matchAll(/^coop_shapes: (.+)$/mg)];
if (lines.length === 0) process.exit(0);
const rows = JSON.parse(lines.at(-1)[1]);
const shapes = [...new Set(rows.map((row) => `${row.M}x${row.N}x${row.K}`))].sort();
for (const shape of shapes) process.stdout.write(`${shape}\n`);
NODE
)

for shape in "${shapes[@]}"; do
  run_row "shape_${shape}" on "$shape"
done

node scripts/summarize_helios_coop_bisect.mjs --root "$output_root"

(
  cd "$output_root"
  find . -type f ! -name ARTIFACTS.sha256 ! -name hash-check.log -print0 \
    | sort -z | xargs -0 sha256sum > ARTIFACTS.sha256
  sha256sum -c ARTIFACTS.sha256 > hash-check.log
)

echo "$output_root"
