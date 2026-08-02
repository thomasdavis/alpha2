#!/usr/bin/env bash
set -euo pipefail

# Correctness-gated end-to-end throughput sweep for Alpha's real chat workload.
# Run only on an idle NVIDIA host after building the repository. Every row uses
# the same checkpoint, token stream, optimizer schedule, and seed. Risky rows
# are retained as evidence even when they fail.

: "${TRAIN_DATA:?set TRAIN_DATA to the rendered training corpus}"
: "${VAL_DATA:?set VAL_DATA to the rendered validation corpus}"
: "${TOKENIZER:?set TOKENIZER to the native tokenizer artifact}"
: "${INIT_CHECKPOINT:?set INIT_CHECKPOINT to the clean native checkpoint}"

OUT_ROOT=${OUT_ROOT:-/tmp/alpha-chat-throughput-sweep}
STEPS=${STEPS:-30}
SKIP_STEPS=${SKIP_STEPS:-5}
NODE_BIN=${NODE_BIN:-node}

if [[ ! "$STEPS" =~ ^[0-9]+$ ]] || (( STEPS < 10 )); then
  echo "STEPS must be an integer >= 10" >&2
  exit 2
fi
if [[ ! "$SKIP_STEPS" =~ ^[0-9]+$ ]] || (( SKIP_STEPS < 1 || SKIP_STEPS >= STEPS )); then
  echo "SKIP_STEPS must be an integer in [1, STEPS)" >&2
  exit 2
fi
for file in "$TRAIN_DATA" "$VAL_DATA" "$TOKENIZER" "$INIT_CHECKPOINT"; do
  [[ -f "$file" ]] || { echo "missing input: $file" >&2; exit 2; }
done
[[ -f apps/cli/dist/main.js ]] || { echo "build apps/cli before running the sweep" >&2; exit 2; }

mkdir "$OUT_ROOT"
sha256sum "$TRAIN_DATA" "$VAL_DATA" "$TOKENIZER" "$INIT_CHECKPOINT" > "$OUT_ROOT/INPUT-HASHES.sha256"

run_row() {
  local name=$1 wg=$2 coop=$3 fp16=$4 block=$5 batch=$6 pool=$7
  local run_dir="$OUT_ROOT/$name"
  local log="$OUT_ROOT/$name.log"
  mkdir "$run_dir"
  printf 'row=%s wg=%s coop=%s fp16=%s block=%s batch=%s pool=%s\n' \
    "$name" "$wg" "$coop" "$fp16" "$block" "$batch" "$pool" | tee "$run_dir/row.txt"

  local -a env_args=(
    "HELIOS_WG_SIZE=$wg"
    "HELIOS_MAX_OUTPUT_POOL_ENTRIES=$pool"
    "ALPHA_FAIL_ON_SMOKE_TEST=1"
    "ALPHA_SAMPLE_FROM_CHECKPOINT=0"
    "ALPHA_GPU_METRICS_SAMPLE_EVERY=1"
    "HELIOS_PROFILE_GPU_OPS=1"
  )
  if [[ "$coop" == "off" ]]; then
    env_args+=("HELIOS_DISABLE_COOP_MAT=1")
  else
    env_args+=("HELIOS_DISABLE_COOP_MAT=0")
  fi
  if [[ "$block" != "1024" ]]; then
    env_args+=("ALPHA_ALLOW_RESUME_MISMATCH=1")
  fi

  set +e
  env "${env_args[@]}" \
    "$NODE_BIN" --expose-gc apps/cli/dist/main.js train \
      --data="$TRAIN_DATA" --valData="$VAL_DATA" --requireValData=true --sft=false \
      --domain=alpha_llama --tokenizerArtifacts="$TOKENIZER" --initCheckpoint="$INIT_CHECKPOINT" \
      --vocabSize=12288 --block="$block" --layers=16 --dim=512 --heads=8 --dropout=0 \
      --activation=swiglu --ffnDim=1408 --normType=rmsnorm --posEnc=rope --ropeTheta=10000 \
      --tieEmbeddings=true --backend=helios --gpuProfile=none --optim=adamw \
      --batch="$batch" --accumSteps=1 --steps="$STEPS" --lr=0.0003 --lrMin=0.00003 \
      --warmupIters=10 --beta1=0.9 --beta2=0.95 --eps=0.00000001 --weightDecay=0.1 \
      --gradClip=1.0 --spikeThreshold=0 --evalInterval="$STEPS" \
      --checkpointInterval="$STEPS" --evalIters=1 --sampleInterval=0 --logEvery=1 \
      --seed=1337 --strictPlanning=false --remote=false --fp16="$fp16" --minGpuSize=1 \
      --no-fallback=true --packed=true --symbio=false --postSamples=false --trace=true \
      --runDir="$run_dir" 2>&1 | tee "$log"
  local status=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$status" > "$run_dir/exit-code.txt"
}

run_row b0_fp32_wg64          64  off false 1024 16  512
run_row b1_fp32_wg128        128  off false 1024 16  512
run_row b2_fp32_wg256        256  off false 1024 16  512
run_row b3_fp32_wg128_pool768 128 off false 1024 16  768
run_row b4_coop_wg128        128  on  false 1024 16  512
run_row b5_mixed_wg128       128  on  true  1024 16  512
run_row b6_fp32_block512     128  off false 512  32  512

python3 scripts/summarize_chat_throughput_sweep.py \
  --root "$OUT_ROOT" --skip-steps "$SKIP_STEPS" --exclude-final \
  | tee "$OUT_ROOT/SUMMARY.md"
