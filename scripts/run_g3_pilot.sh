#!/usr/bin/env bash
# Equal-token Stage-3 architecture pilot on the proven RTX 3090 Vulkan profile.
#
# Usage (on the bootstrapped pod):
#   scripts/run_g3_pilot.sh llama /workspace/data/g1-pretrain-128m.txt \
#     /workspace/alpha2/artifacts/g2-bpe-byte-12k.json /workspace/alpha2/runs/g3-llama-100m
#   scripts/run_g3_pilot.sh gpt2  ... /workspace/alpha2/runs/g3-gpt2-100m
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

case "$variant" in
  llama)
    architecture_args=()
    ;;
  gpt2)
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
if [[ -e "$run_dir" ]]; then
  echo "run directory already exists; refusing to mix pilot artifacts: $run_dir" >&2
  exit 1
fi

export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd_headless.json}
export HELIOS_DISABLE_COOP_MAT=1
export HELIOS_WG_SIZE=64
export HELIOS_MAX_OUTPUT_POOL_ENTRIES=512
export ALPHA_GPU_METRICS_SAMPLE_EVERY=100

mkdir -p "$run_dir"
exec nice -n 5 ionice -c 2 -n 7 node --expose-gc apps/cli/dist/main.js train \
  --data="$data" \
  --domain=alpha_llama \
  --tokenizerArtifacts="$tokenizer" \
  --batch=16 \
  --accumSteps=1 \
  --steps=6104 \
  --lr=3e-4 \
  --lrMin=3e-5 \
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
  --evalIters=5 \
  --logEvery=25 \
  --sampleInterval=0 \
  --postSamples=false \
  --remote=false \
  --packed=true \
  --runDir="$run_dir" \
  "${architecture_args[@]}"
