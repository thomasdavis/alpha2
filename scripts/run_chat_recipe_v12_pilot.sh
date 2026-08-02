#!/usr/bin/env bash
# Clean-base packed full-sequence Smol-SmolTalk pilot for Alpha V12.
#
# Usage:
#   scripts/run_chat_recipe_v12_pilot.sh \
#     TRAIN TEST MANIFEST TOKENIZER EVAL_FREEZE CLEAN_CHECKPOINT RUN_DIR LR [STEPS=2000]

set -euo pipefail

train_data=${1:?training text required}
test_data=${2:?validation text required}
manifest=${3:?corpus manifest required}
tokenizer=${4:?tokenizer artifact required}
eval_freeze=${5:?evaluation freeze required}
init_checkpoint=${6:?clean pretrained checkpoint required}
run_dir=${7:?new run directory required}
learning_rate=${8:?peak learning rate required}
steps=${9:-2000}

for required in "$train_data" "$test_data" "$manifest" "$tokenizer" "$eval_freeze" \
  "$init_checkpoint" apps/cli/dist/main.js; do
  [[ -f $required ]] || { echo "required file missing: $required" >&2; exit 1; }
done
[[ ! -e $run_dir ]] || { echo "run directory already exists: $run_dir" >&2; exit 1; }
[[ -z $(git status --porcelain) ]] || {
  echo "V12 training requires a clean committed worktree" >&2
  exit 1
}
[[ $steps == 2000 ]] || {
  echo "the frozen V12 pilot length is exactly 2,000 steps" >&2
  exit 2
}
case "$learning_rate" in
  0.0003|3e-4|0.001|1e-3) ;;
  *) echo "the frozen V12 pilot LRs are 3e-4 and 1e-3" >&2; exit 2 ;;
esac

learning_rate_min=$(node -e '
  const lr = Number(process.argv[1]);
  if (!Number.isFinite(lr) || lr <= 0) throw new Error("learning rate must be positive");
  process.stdout.write(String(lr / 10));
' "$learning_rate")

sha256() { sha256sum "$1" | awk '{print $1}'; }
train_sha=$(sha256 "$train_data")
test_sha=$(sha256 "$test_data")
manifest_sha=$(sha256 "$manifest")
tokenizer_sha=$(sha256 "$tokenizer")
eval_freeze_sha=$(sha256 "$eval_freeze")
checkpoint_sha=$(sha256 "$init_checkpoint")
expected_checkpoint_sha=08e14fa9604bf1b46ebcd5df37933c84d2496c1d05d9e4b32ebad98792cc6049
expected_tokenizer_sha=c310343a185aecb572b8b6568b55179df248f4adec009d14a9496da354090b24
expected_render_tokenizer_sha=37372c9b1bdbf7d9655444e90247bef957018d0d7ff0b668d1330e28d97c44cf
[[ $checkpoint_sha == "$expected_checkpoint_sha" ]] || {
  echo "clean pretrained checkpoint hash mismatch: $checkpoint_sha" >&2
  exit 1
}
[[ $tokenizer_sha == "$expected_tokenizer_sha" ]] || {
  echo "tokenizer hash mismatch: $tokenizer_sha" >&2
  exit 1
}

MANIFEST="$manifest" TRAIN_SHA="$train_sha" TEST_SHA="$test_sha" \
RENDER_TOKENIZER_SHA="$expected_render_tokenizer_sha" EVAL_FREEZE="$eval_freeze" node - <<'NODE'
const corpus = require(process.env.MANIFEST);
const freeze = require(process.env.EVAL_FREEZE);
const assert = (condition, message) => { if (!condition) throw new Error(message); };
assert(corpus.schema === "alpha-chat-recipe-v12-corpus-v1", "unexpected corpus schema");
assert(corpus.result === "PASS", "corpus build did not pass");
assert(corpus.train?.output?.sha256 === process.env.TRAIN_SHA, "training hash mismatch");
assert(corpus.test?.output?.sha256 === process.env.TEST_SHA, "test hash mismatch");
assert(corpus.tokenizer?.sha256 === process.env.RENDER_TOKENIZER_SHA, "render-length tokenizer mismatch");
assert(corpus.post_build_train_test_overlap === 0, "train/test overlap remains");
assert(corpus.rendering?.order === "preserve_upstream", "training order drifted");
assert(corpus.rendering?.system_policy === "fold_into_first_user", "system rendering drifted");
assert(corpus.train?.counts?.accepted > 400000, "training population unexpectedly small");
assert(corpus.test?.counts?.accepted > 20000, "validation population unexpectedly small");
assert(freeze.schema === "alpha-chat-semantic-repair-v4-evaluation-freeze-v1", "unexpected evaluation freeze");
assert(freeze.status === "development-visible; inherited-final-sealed-unexecuted", "evaluation freeze is not selection-safe");
NODE

source_commit=$(git rev-parse HEAD)
warmup_steps=$((steps / 10))
((warmup_steps < 1)) && warmup_steps=1
mkdir -p "$run_dir"
contract_tmp="$run_dir/recipe-contract.json.tmp"
SOURCE_COMMIT="$source_commit" TRAIN_DATA="$train_data" TRAIN_SHA="$train_sha" \
TEST_DATA="$test_data" TEST_SHA="$test_sha" MANIFEST="$manifest" \
MANIFEST_SHA="$manifest_sha" TOKENIZER="$tokenizer" TOKENIZER_SHA="$tokenizer_sha" \
EVAL_FREEZE="$eval_freeze" EVAL_FREEZE_SHA="$eval_freeze_sha" \
INIT_CHECKPOINT="$init_checkpoint" CHECKPOINT_SHA="$checkpoint_sha" STEPS="$steps" \
LR="$learning_rate" LR_MIN="$learning_rate_min" WARMUP="$warmup_steps" \
CONTRACT_TMP="$contract_tmp" node - <<'NODE'
const fs = require("node:fs");
const steps = Number(process.env.STEPS);
const contract = {
  schema: "alpha-chat-recipe-v12-pilot-contract-v1",
  purpose: "replicate the packed full-sequence Smol-SmolTalk recipe from Alpha's clean base",
  eligibleForCheckpointSelection: true,
  sourceCommit: process.env.SOURCE_COMMIT,
  sourceTreeDirty: false,
  initializedFrom: { path: process.env.INIT_CHECKPOINT, sha256: process.env.CHECKPOINT_SHA },
  inputs: {
    train: { path: process.env.TRAIN_DATA, sha256: process.env.TRAIN_SHA },
    validation: { path: process.env.TEST_DATA, sha256: process.env.TEST_SHA },
    manifest: { path: process.env.MANIFEST, sha256: process.env.MANIFEST_SHA },
    tokenizer: { path: process.env.TOKENIZER, sha256: process.env.TOKENIZER_SHA },
    evaluationFreeze: { path: process.env.EVAL_FREEZE, sha256: process.env.EVAL_FREEZE_SHA },
  },
  intervention: {
    changed: [
      "complete rendered chat is supervised rather than assistant tokens only",
      "conversations are packed without padding",
      "public Smol-SmolTalk source is tested from the clean base",
    ],
    unchangedAcrossPilotArms: [
      "model parameters", "architecture", "tokenizer", "data bytes", "packed windows",
      "optimizer family", "seed", "evaluation", "decoding",
    ],
    onlyArmDifference: "learning rate",
    excluded: [
      "synthetic V12 curriculum", "assistant-only loss", "RCR-UL", "Symbiogenesis",
      "sealed-final tuning", "validation-loss-only selection",
    ],
  },
  training: {
    steps,
    blockSize: 1024,
    batchSize: 16,
    gradientAccumulationSteps: 1,
    objective: "full-sequence next-token cross entropy",
    packed: true,
    optimizer: "AdamW reset from clean pretrained parameters",
    learningRate: Number(process.env.LR),
    learningRateMin: Number(process.env.LR_MIN),
    warmupSteps: Number(process.env.WARMUP),
    checkpointInterval: 250,
    seed: 1337,
    fp16: false,
    symbio: false,
  },
  selection: {
    checkpoints: Array.from({ length: Math.floor(steps / 250) }, (_, index) => (index + 1) * 250),
    primary: "blinded held-out free-conversation quality and semantic contingency",
    lossMaySelect: false,
    nonemptyOutputMaySelect: false,
    mustCompareAgainst: ["clean base", "best retained public Alpha"],
    publishOnlyGenuineLocalWinner: true,
    sealedFinalRemainsClosedUntilSelection: true,
  },
  startedUtc: new Date().toISOString(),
};
fs.writeFileSync(process.env.CONTRACT_TMP, JSON.stringify(contract, null, 2) + "\n", { flag: "wx" });
NODE
mv "$contract_tmp" "$run_dir/recipe-contract.json"

export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd_headless.json}
export HELIOS_DISABLE_COOP_MAT=1
export HELIOS_WG_SIZE=64
export HELIOS_MAX_OUTPUT_POOL_ENTRIES=512
export ALPHA_FAIL_ON_SMOKE_TEST=1
export ALPHA_SAMPLE_FROM_CHECKPOINT=0
export ALPHA_GPU_METRICS_SAMPLE_EVERY=25

exec nice -n 5 ionice -c2 -n7 node --expose-gc apps/cli/dist/main.js train \
  --data="$train_data" --valData="$test_data" --requireValData=true --sft=false \
  --domain=alpha_llama --tokenizerArtifacts="$tokenizer" --initCheckpoint="$init_checkpoint" \
  --vocabSize=12288 --block=1024 --layers=16 --dim=512 --heads=8 --dropout=0 \
  --activation=swiglu --ffnDim=1408 --normType=rmsnorm --posEnc=rope --ropeTheta=10000 \
  --tieEmbeddings=true --backend=helios --gpuProfile=none --optim=adamw --batch=16 --accumSteps=1 \
  --steps="$steps" --lr="$learning_rate" --lrMin="$learning_rate_min" --warmupIters="$warmup_steps" \
  --beta1=0.9 --beta2=0.95 --eps=0.00000001 --weightDecay=0.1 --gradClip=1.0 \
  --spikeThreshold=0 --evalInterval=250 --checkpointInterval=250 --evalIters=5 \
  --sampleInterval=0 --logEvery=25 --seed=1337 --strictPlanning=false --remote=false \
  --fp16=false --minGpuSize=1 --no-fallback=true --packed=true --symbio=false \
  --postSamples=false --runDir="$run_dir"
