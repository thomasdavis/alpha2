#!/usr/bin/env bash
# Finite Alpha chat-repair v2 continuation pilot.
#
# Usage:
#   scripts/run_chat_repair_v2.sh \
#     TRAIN DEV_CORPUS CORPUS_MANIFEST TOKENIZER INIT_CHECKPOINT \
#     EVAL_FREEZE_MANIFEST RUN_DIR \
#     [STEPS=800] [LR=0.00001] [LR_MIN=0.000002] [BATCH=16] \
#     [START_WEIGHT=8] [START_TOKENS=4] [END_WEIGHT=4] [WARMUP=50]

set -euo pipefail

train_data=${1:?training conversations required}
dev_data=${2:?development conversations required}
corpus_manifest=${3:?corpus manifest required}
tokenizer=${4:?Alpha tokenizer artifact required}
init_checkpoint=${5:?initial checkpoint required}
eval_manifest=${6:?frozen evaluation manifest required}
run_dir=${7:?new run directory required}
steps=${8:-800}
learning_rate=${9:-0.00001}
learning_rate_min=${10:-0.000002}
batch_size=${11:-16}
start_weight=${12:-8}
start_tokens=${13:-4}
end_weight=${14:-4}
warmup_steps=${15:-50}

for required in \
  "$train_data" "$dev_data" "$corpus_manifest" "$tokenizer" \
  "$init_checkpoint" "$eval_manifest" apps/cli/dist/main.js; do
  [[ -f $required ]] || { echo "required file missing: $required" >&2; exit 1; }
done
[[ ! -e $run_dir ]] || { echo "run directory already exists: $run_dir" >&2; exit 1; }
for pair in \
  "STEPS:$steps" "BATCH:$batch_size" "START_WEIGHT:$start_weight" \
  "START_TOKENS:$start_tokens" "END_WEIGHT:$end_weight" "WARMUP:$warmup_steps"; do
  name=${pair%%:*}
  value=${pair#*:}
  [[ $value =~ ^[1-9][0-9]*$ ]] || { echo "$name must be a positive integer" >&2; exit 2; }
done
node -e '
  for (const [name, value] of [["LR", process.argv[1]], ["LR_MIN", process.argv[2]]]) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed) || parsed <= 0) throw new Error(`${name} must be finite and positive`);
  }
  if (Number(process.argv[2]) > Number(process.argv[1])) throw new Error("LR_MIN must not exceed LR");
' "$learning_rate" "$learning_rate_min"

sha256() { sha256sum "$1" | awk '{print $1}'; }
train_sha=$(sha256 "$train_data")
dev_sha=$(sha256 "$dev_data")
tokenizer_sha=$(sha256 "$tokenizer")
checkpoint_sha=$(sha256 "$init_checkpoint")
corpus_manifest_sha=$(sha256 "$corpus_manifest")
eval_manifest_sha=$(sha256 "$eval_manifest")

TRAIN_SHA="$train_sha" DEV_SHA="$dev_sha" \
CORPUS_MANIFEST="$corpus_manifest" EVAL_MANIFEST="$eval_manifest" node -e '
  const corpus = require(process.env.CORPUS_MANIFEST);
  const freeze = require(process.env.EVAL_MANIFEST);
  const assert = (condition, message) => { if (!condition) throw new Error(message); };
  assert(corpus.schema === "alpha-chat-repair-corpus-v2", "unexpected corpus manifest schema");
  assert(corpus.outputs?.train?.sha256 === process.env.TRAIN_SHA, "training corpus hash mismatch");
  assert(corpus.outputs?.dev?.sha256 === process.env.DEV_SHA, "development corpus hash mismatch");
  assert(freeze.schema === "alpha-chat-repair-v2-eval-freeze-v1", "unexpected evaluation manifest schema");
  assert(freeze.status === "development-visible; final-sealed-unexecuted", "evaluation final is not sealed");
  assert(freeze.inputs?.train?.sha256 === process.env.TRAIN_SHA, "evaluation freeze used another train corpus");
  assert(freeze.inputs?.devCorpus?.sha256 === process.env.DEV_SHA, "evaluation freeze used another dev corpus");
  assert(freeze.outputs?.development?.sha256, "evaluation freeze has no development suite hash");
  assert(freeze.outputs?.qualitativePanel?.sha256, "evaluation freeze has no qualitative panel hash");
  assert(freeze.outputs?.sealedFinal?.sha256, "evaluation freeze has no sealed final hash");
'

source_commit=$(git rev-parse HEAD)
source_tree_dirty=false
[[ -z $(git status --porcelain) ]] || source_tree_dirty=true
mkdir -p "$run_dir"
contract_tmp="$run_dir/repair-contract.json.tmp"
SOURCE_COMMIT="$source_commit" SOURCE_TREE_DIRTY="$source_tree_dirty" \
TRAIN_DATA="$train_data" TRAIN_SHA="$train_sha" DEV_DATA="$dev_data" DEV_SHA="$dev_sha" \
CORPUS_MANIFEST="$corpus_manifest" CORPUS_MANIFEST_SHA="$corpus_manifest_sha" \
EVAL_MANIFEST="$eval_manifest" EVAL_MANIFEST_SHA="$eval_manifest_sha" \
TOKENIZER="$tokenizer" TOKENIZER_SHA="$tokenizer_sha" INIT_CHECKPOINT="$init_checkpoint" \
CHECKPOINT_SHA="$checkpoint_sha" STEPS="$steps" LR="$learning_rate" LR_MIN="$learning_rate_min" \
BATCH="$batch_size" START_WEIGHT="$start_weight" START_TOKENS="$start_tokens" \
END_WEIGHT="$end_weight" WARMUP="$warmup_steps" CONTRACT_TMP="$contract_tmp" node -e '
  const fs = require("node:fs");
  const freeze = require(process.env.EVAL_MANIFEST);
  const contract = {
    schema: "alpha-chat-repair-contract-v2",
    purpose: "repair conversational stopping and repetition while preserving response initiation",
    sourceCommit: process.env.SOURCE_COMMIT,
    sourceTreeDirty: process.env.SOURCE_TREE_DIRTY === "true",
    initializedFrom: {
      path: process.env.INIT_CHECKPOINT,
      sha256: process.env.CHECKPOINT_SHA,
    },
    inputs: {
      train: { path: process.env.TRAIN_DATA, sha256: process.env.TRAIN_SHA },
      devCorpus: { path: process.env.DEV_DATA, sha256: process.env.DEV_SHA },
      corpusManifest: { path: process.env.CORPUS_MANIFEST, sha256: process.env.CORPUS_MANIFEST_SHA },
      evalFreezeManifest: { path: process.env.EVAL_MANIFEST, sha256: process.env.EVAL_MANIFEST_SHA },
      tokenizer: { path: process.env.TOKENIZER, sha256: process.env.TOKENIZER_SHA },
    },
    training: {
      steps: Number(process.env.STEPS),
      blockSize: 1024,
      batchSize: Number(process.env.BATCH),
      learningRate: Number(process.env.LR),
      learningRateMin: Number(process.env.LR_MIN),
      warmupSteps: Number(process.env.WARMUP),
      checkpointInterval: 200,
      deterministicEpochShuffle: true,
      equalConversationWeight: true,
      answerStartTokens: Number(process.env.START_TOKENS),
      answerStartMultiplier: Number(process.env.START_WEIGHT),
      answerEndMultiplier: Number(process.env.END_WEIGHT),
      eosBoostedAsAnswerStart: false,
      gpuGateRequired: true,
      smallTensorCpuFallbackThreshold: 4096,
    },
    selection: {
      developmentSuite: freeze.outputs.development,
      qualitativePanel: freeze.outputs.qualitativePanel,
      sealedFinalSuite: freeze.outputs.sealedFinal,
      generation: "deterministic greedy",
      rule: "select on development generation: nonempty plus EOS plus no role leak, then lower loops and direct human conversational inspection",
      finalSuiteHeldUntilOneCheckpointIsSelected: true,
    },
    startedUtc: new Date().toISOString(),
  };
  fs.writeFileSync(process.env.CONTRACT_TMP, JSON.stringify(contract, null, 2) + "\n", { flag: "wx" });
'
mv "$contract_tmp" "$run_dir/repair-contract.json"

export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd_headless.json}
export HELIOS_DISABLE_COOP_MAT=1
export HELIOS_WG_SIZE=64
export HELIOS_MAX_OUTPUT_POOL_ENTRIES=512
export ALPHA_FAIL_ON_SMOKE_TEST=1
# The selected repair checkpoint has the same parameter shapes but a 512-token
# runtime contract. RoPE has no learned position table, so v2 deliberately
# restores the pretrained parent's 1,024-token context.
export ALPHA_ALLOW_RESUME_MISMATCH=1
export ALPHA_SFT_SHUFFLE=1
export ALPHA_SFT_BALANCE_CONVERSATIONS=1
export ALPHA_SFT_START_TOKENS="$start_tokens"
export ALPHA_SFT_START_WEIGHT="$start_weight"
export ALPHA_SFT_END_WEIGHT="$end_weight"
export ALPHA_SAMPLE_FROM_CHECKPOINT=0
export ALPHA_GPU_METRICS_SAMPLE_EVERY=25

exec nice -n 5 ionice -c2 -n7 node --expose-gc apps/cli/dist/main.js train \
  --data="$train_data" \
  --valData="$dev_data" \
  --requireValData=true \
  --sft=true \
  --domain=alpha_llama \
  --tokenizerArtifacts="$tokenizer" \
  --initCheckpoint="$init_checkpoint" \
  --vocabSize=12288 \
  --block=1024 \
  --layers=16 \
  --dim=512 \
  --heads=8 \
  --dropout=0 \
  --activation=swiglu \
  --ffnDim=1408 \
  --normType=rmsnorm \
  --posEnc=rope \
  --ropeTheta=10000 \
  --tieEmbeddings=true \
  --backend=helios \
  --optim=adamw \
  --batch="$batch_size" \
  --accumSteps=1 \
  --steps="$steps" \
  --lr="$learning_rate" \
  --lrMin="$learning_rate_min" \
  --warmupIters="$warmup_steps" \
  --beta1=0.9 \
  --beta2=0.95 \
  --eps=1e-8 \
  --weightDecay=0.1 \
  --gradClip=1.0 \
  --evalInterval=200 \
  --checkpointInterval=200 \
  --evalIters=10 \
  --sampleInterval=0 \
  --logEvery=10 \
  --seed=42 \
  --strictPlanning=false \
  --remote=false \
  --postSamples=false \
  --runDir="$run_dir"
