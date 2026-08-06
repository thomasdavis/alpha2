#!/usr/bin/env bash
# Finite clean-base ablation for Alpha's reviewed semantic-chat curriculum.
#
# This deliberately changes only initialization and exposure duration relative
# to v4: it starts from the clean pretrained parent rather than continuing the
# roleplay-heavy public chat checkpoint.  Intermediate checkpoints remain the
# selection unit; reaching the final step is not evidence of improvement.
#
# Usage:
#   scripts/run_chat_semantic_repair_v5_clean_base.sh \
#     TRAIN DEV CORPUS_MANIFEST EVAL_FREEZE TOKENIZER INIT_CHECKPOINT RUN_DIR \
#     [STEPS=1600] [LR=0.00005] [LR_MIN=0.000005]

set -euo pipefail

train_data=${1:?training conversations required}
dev_data=${2:?development conversations required}
corpus_manifest=${3:?corpus manifest required}
eval_manifest=${4:?v4 evaluation freeze required}
tokenizer=${5:?Alpha tokenizer artifact required}
init_checkpoint=${6:?clean pretrained checkpoint required}
run_dir=${7:?new run directory required}
steps=${8:-1600}
learning_rate=${9:-0.00005}
learning_rate_min=${10:-0.000005}

for required in \
  "$train_data" "$dev_data" "$corpus_manifest" "$eval_manifest" \
  "$tokenizer" "$init_checkpoint" apps/cli/dist/main.js; do
  [[ -f $required ]] || { echo "required file missing: $required" >&2; exit 1; }
done
[[ ! -e $run_dir ]] || { echo "run directory already exists: $run_dir" >&2; exit 1; }
[[ -z $(git status --porcelain) ]] || { echo "v5 training requires a clean committed worktree" >&2; exit 1; }
[[ $steps =~ ^[1-9][0-9]*$ ]] || { echo "steps must be a positive integer" >&2; exit 2; }
node -e '
  const [lr, floor] = process.argv.slice(1).map(Number);
  if (![lr, floor].every((value) => Number.isFinite(value) && value > 0)) throw new Error("invalid LR");
  if (floor > lr) throw new Error("LR_MIN must not exceed LR");
' "$learning_rate" "$learning_rate_min"

sha256() { sha256sum "$1" | awk '{print $1}'; }
train_sha=$(sha256 "$train_data")
dev_sha=$(sha256 "$dev_data")
corpus_manifest_sha=$(sha256 "$corpus_manifest")
eval_manifest_sha=$(sha256 "$eval_manifest")
tokenizer_sha=$(sha256 "$tokenizer")
checkpoint_sha=$(sha256 "$init_checkpoint")
expected_checkpoint_sha=08e14fa9604bf1b46ebcd5df37933c84d2496c1d05d9e4b32ebad98792cc6049
expected_tokenizer_sha=c310343a185aecb572b8b6568b55179df248f4adec009d14a9496da354090b24
[[ $checkpoint_sha == "$expected_checkpoint_sha" ]] || {
  echo "clean pretrained checkpoint hash mismatch: $checkpoint_sha" >&2; exit 1;
}
[[ $tokenizer_sha == "$expected_tokenizer_sha" ]] || {
  echo "native tokenizer hash mismatch: $tokenizer_sha" >&2; exit 1;
}

TRAIN_SHA="$train_sha" DEV_SHA="$dev_sha" CORPUS_MANIFEST_SHA="$corpus_manifest_sha" \
CORPUS_MANIFEST="$corpus_manifest" EVAL_MANIFEST="$eval_manifest" node - <<'NODE'
const corpus = require(process.env.CORPUS_MANIFEST);
const freeze = require(process.env.EVAL_MANIFEST);
const assert = (condition, message) => { if (!condition) throw new Error(message); };
assert(corpus.schema === "alpha-chat-semantic-repair-v4-corpus-manifest-v1", "unexpected corpus schema");
assert(corpus.source_tree_dirty === false && typeof corpus.source_commit === "string", "corpus source provenance is not clean and committed");
assert(corpus.outputs?.train?.sha256 === process.env.TRAIN_SHA, "training corpus hash mismatch");
assert(corpus.outputs?.dev?.sha256 === process.env.DEV_SHA, "development corpus hash mismatch");
assert(corpus.sources?.["gpt-5.4"] > 0, "corpus has no synthetic semantic-chat rows");
assert(freeze.schema === "alpha-chat-semantic-repair-v4-evaluation-freeze-v1", "unexpected evaluation freeze");
assert(freeze.status === "development-visible; inherited-final-sealed-unexecuted", "sealed state changed");
assert(freeze.inputs?.corpus_manifest?.sha256 === process.env.CORPUS_MANIFEST_SHA, "evaluation binds another corpus");
assert(freeze.sealed_final?.sha256 === "8b71ab5f8843b14a8bbe56a473ea9cd0672b873024632c023abbe4935e48eb1d", "sealed final identity changed");
NODE

source_commit=$(git rev-parse HEAD)
mkdir -p "$run_dir"
contract_tmp="$run_dir/repair-contract.json.tmp"
SOURCE_COMMIT="$source_commit" TRAIN_DATA="$train_data" TRAIN_SHA="$train_sha" \
DEV_DATA="$dev_data" DEV_SHA="$dev_sha" CORPUS_MANIFEST="$corpus_manifest" \
CORPUS_MANIFEST_SHA="$corpus_manifest_sha" EVAL_MANIFEST="$eval_manifest" \
EVAL_MANIFEST_SHA="$eval_manifest_sha" TOKENIZER="$tokenizer" TOKENIZER_SHA="$tokenizer_sha" \
INIT_CHECKPOINT="$init_checkpoint" CHECKPOINT_SHA="$checkpoint_sha" STEPS="$steps" \
LR="$learning_rate" LR_MIN="$learning_rate_min" CONTRACT_TMP="$contract_tmp" node - <<'NODE'
const fs = require("node:fs");
const steps = Number(process.env.STEPS);
const contract = {
  schema: "alpha-chat-semantic-repair-contract-v5",
  purpose: "test whether v4 semantic failure is path dependence from the roleplay-heavy selected chat checkpoint",
  eligibleForCheckpointSelection: true,
  sourceCommit: process.env.SOURCE_COMMIT,
  sourceTreeDirty: false,
  initializedFrom: { path: process.env.INIT_CHECKPOINT, sha256: process.env.CHECKPOINT_SHA },
  inputs: {
    train: { path: process.env.TRAIN_DATA, sha256: process.env.TRAIN_SHA },
    development: { path: process.env.DEV_DATA, sha256: process.env.DEV_SHA },
    corpusManifest: { path: process.env.CORPUS_MANIFEST, sha256: process.env.CORPUS_MANIFEST_SHA },
    evaluationFreeze: { path: process.env.EVAL_MANIFEST, sha256: process.env.EVAL_MANIFEST_SHA },
    tokenizer: { path: process.env.TOKENIZER, sha256: process.env.TOKENIZER_SHA },
  },
  intervention: {
    changed: [
      "initialization: clean pretrained parent instead of selected roleplay-heavy chat checkpoint",
      "finite exposure window extended to make early and late clean-base states observable",
    ],
    unchanged: [
      "architecture", "tokenizer", "chat template", "loss implementation",
      "reviewed v4 train and development bytes", "answer-start weighting", "EOS weighting", "decoding",
    ],
    excluded: ["SODA bulk replay", "RCR unlikelihood", "sealed-final tuning", "validation-loss selection"],
  },
  training: {
    steps, blockSize: 512, batchSize: 16,
    gradientAccumulationSteps: 1, optimizer: "AdamW reset from clean pretrained parameters",
    learningRate: Number(process.env.LR), learningRateMin: Number(process.env.LR_MIN),
    warmupSteps: 100, checkpointInterval: 200, deterministicEpochShuffle: true,
    equalConversationWeight: true, answerStartTokens: 4, answerStartMultiplier: 8,
    answerEndMultiplier: 2, rcrUlWeight: 0, fp16: false,
  },
  selection: {
    checkpoints: Array.from({ length: Math.floor(steps / 200) }, (_, index) => (index + 1) * 200),
    rule: "select only on matched held-out free-generation meaning and conversational behavior; validation loss and final-step status cannot select",
    matchedPublicBaselineRequired: true,
    matchedV4EvidenceRequired: true,
    sealedFinalRemainsClosedUntilSelection: true,
  },
  resourceBound: {
    localProjectArtifactPauseBytes: 15 * 1024 ** 3,
    policy: "pause before retained Alpha v4 plus v5 project artifacts exceed this bound",
  },
  startedUtc: new Date().toISOString(),
};
fs.writeFileSync(process.env.CONTRACT_TMP, JSON.stringify(contract, null, 2) + "\n", { flag: "wx" });
NODE
mv "$contract_tmp" "$run_dir/repair-contract.json"

export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd_headless.json}
export HELIOS_DISABLE_COOP_MAT=1
export HELIOS_WG_SIZE=64
export HELIOS_MAX_OUTPUT_POOL_ENTRIES=512
export ALPHA_FAIL_ON_SMOKE_TEST=1
# The clean parent used block 1024; weights are shape-compatible with the
# deliberately retained 512-token conversational context.
export ALPHA_ALLOW_RESUME_MISMATCH=1
export ALPHA_SFT_SHUFFLE=1
export ALPHA_SFT_BALANCE_CONVERSATIONS=1
export ALPHA_SFT_START_TOKENS=4
export ALPHA_SFT_START_WEIGHT=8
export ALPHA_SFT_END_WEIGHT=2
export ALPHA_SAMPLE_FROM_CHECKPOINT=0
export ALPHA_GPU_METRICS_SAMPLE_EVERY=10

exec nice -n 5 ionice -c2 -n7 node --expose-gc apps/cli/dist/main.js train \
  --data="$train_data" --valData="$dev_data" --requireValData=true --sft=true \
  --domain=alpha_llama --tokenizerArtifacts="$tokenizer" --initCheckpoint="$init_checkpoint" \
  --vocabSize=12288 --block=512 --layers=16 --dim=512 --heads=8 --dropout=0 \
  --activation=swiglu --ffnDim=1408 --normType=rmsnorm --posEnc=rope --ropeTheta=10000 \
  --tieEmbeddings=true --backend=helios --optim=adamw --batch=16 --accumSteps=1 \
  --steps="$steps" --lr="$learning_rate" --lrMin="$learning_rate_min" --warmupIters=100 \
  --beta1=0.9 --beta2=0.95 --eps=0.00000001 --weightDecay=0.1 --gradClip=1.0 \
  --spikeThreshold=50 --evalInterval=200 --checkpointInterval=200 --evalIters=10 \
  --sampleInterval=0 --logEvery=10 --seed=42 --strictPlanning=false --remote=false \
  --postSamples=false --runDir="$run_dir"
