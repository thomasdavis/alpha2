#!/usr/bin/env bash
# Finite direct-semantic bridge from Alpha's mechanically stable U1 checkpoint.

set -euo pipefail

train_data=${1:?training conversations required}
dev_data=${2:?development conversations required}
corpus_manifest=${3:?v7 corpus manifest required}
eval_manifest=${4:?evaluation freeze required}
tokenizer=${5:?Alpha tokenizer artifact required}
init_checkpoint=${6:?U1 checkpoint required}
run_dir=${7:?new run directory required}
steps=${8:-2800}
learning_rate=${9:-0.00001}
learning_rate_min=${10:-0.000001}

for required in "$train_data" "$dev_data" "$corpus_manifest" "$eval_manifest" "$tokenizer" "$init_checkpoint" apps/cli/dist/main.js; do
  [[ -f $required ]] || { echo "required file missing: $required" >&2; exit 1; }
done
[[ ! -e $run_dir ]] || { echo "run directory already exists: $run_dir" >&2; exit 1; }
[[ -z $(git status --porcelain) ]] || { echo "v7 training requires a clean committed worktree" >&2; exit 1; }
[[ $steps =~ ^[1-9][0-9]*$ && $((steps % 400)) -eq 0 ]] || { echo "steps must be a positive multiple of 400" >&2; exit 2; }

sha256() { sha256sum "$1" | awk '{print $1}'; }
train_sha=$(sha256 "$train_data")
dev_sha=$(sha256 "$dev_data")
corpus_manifest_sha=$(sha256 "$corpus_manifest")
eval_manifest_sha=$(sha256 "$eval_manifest")
tokenizer_sha=$(sha256 "$tokenizer")
checkpoint_sha=$(sha256 "$init_checkpoint")
expected_checkpoint_sha=0453a842b264c80c3578bc419c3dc94b46420aca30cad93593d62c812f5710fb
expected_tokenizer_sha=c310343a185aecb572b8b6568b55179df248f4adec009d14a9496da354090b24
[[ $checkpoint_sha == "$expected_checkpoint_sha" ]] || { echo "U1 checkpoint hash mismatch: $checkpoint_sha" >&2; exit 1; }
[[ $tokenizer_sha == "$expected_tokenizer_sha" ]] || { echo "tokenizer hash mismatch: $tokenizer_sha" >&2; exit 1; }

TRAIN_SHA="$train_sha" DEV_SHA="$dev_sha" CORPUS_MANIFEST="$corpus_manifest" EVAL_MANIFEST="$eval_manifest" node - <<'NODE'
const corpus = require(process.env.CORPUS_MANIFEST);
const freeze = require(process.env.EVAL_MANIFEST);
const assert = (condition, message) => { if (!condition) throw new Error(message); };
assert(corpus.schema === "alpha-chat-bridge-v7-corpus-v1", "unexpected corpus schema");
assert(corpus.sourceTreeDirty === false, "corpus was not built from a clean committed tree");
assert(corpus.outputs?.train?.sha256 === process.env.TRAIN_SHA, "training corpus hash mismatch");
assert(corpus.outputs?.dev?.sha256 === process.env.DEV_SHA, "development corpus hash mismatch");
assert(corpus.rows?.bySplitAndOrigin?.train?.reviewedSemantic === 5104, "reviewed semantic train count drift");
assert(corpus.rows?.bySplitAndOrigin?.train?.canonicalDirect === 40000, "direct train count drift");
assert(corpus.invariants?.semanticOrTopicStringFilterApplied === false, "unexpected semantic string filter");
assert(corpus.invariants?.sealedFinalInspected === false, "sealed final was inspected");
assert(freeze.schema === "alpha-chat-semantic-repair-v4-evaluation-freeze-v1", "unexpected evaluation freeze");
assert(freeze.status === "development-visible; inherited-final-sealed-unexecuted", "sealed state changed");
assert(freeze.sealed_final?.sha256 === "8b71ab5f8843b14a8bbe56a473ea9cd0672b873024632c023abbe4935e48eb1d", "sealed final identity changed");
NODE

source_commit=$(git rev-parse HEAD)
mkdir -p "$run_dir"
contract_tmp="$run_dir/repair-contract.json.tmp"
SOURCE_COMMIT="$source_commit" TRAIN_DATA="$train_data" TRAIN_SHA="$train_sha" DEV_DATA="$dev_data" DEV_SHA="$dev_sha" \
CORPUS_MANIFEST="$corpus_manifest" CORPUS_MANIFEST_SHA="$corpus_manifest_sha" EVAL_MANIFEST="$eval_manifest" \
EVAL_MANIFEST_SHA="$eval_manifest_sha" TOKENIZER="$tokenizer" TOKENIZER_SHA="$tokenizer_sha" \
INIT_CHECKPOINT="$init_checkpoint" CHECKPOINT_SHA="$checkpoint_sha" STEPS="$steps" LR="$learning_rate" \
LR_MIN="$learning_rate_min" CONTRACT_TMP="$contract_tmp" node - <<'NODE'
const fs = require("node:fs");
const steps = Number(process.env.STEPS);
const contract = {
  schema: "alpha-chat-bridge-contract-v7",
  purpose: "add direct semantic response competence while preserving U1 autoregressive stability",
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
    changed: "one finite pass over a balanced direct-answer and reviewed semantic-chat bridge corpus",
    unchanged: ["U1 parameters", "architecture", "tokenizer", "chat template", "positive SFT loss", "decoding"],
    excluded: ["bulk long-form SFT", "SODA roleplay", "new unlikelihood pressure", "sealed-final tuning", "BLAH answer training"],
  },
  training: {
    steps, blockSize: 512, batchSize: 16, gradientAccumulationSteps: 1,
    optimizer: "AdamW reset from U1 parameters",
    learningRate: Number(process.env.LR), learningRateMin: Number(process.env.LR_MIN),
    warmupSteps: 200, checkpointInterval: 400, deterministicEpochShuffle: true,
    equalConversationWeight: true, answerStartTokens: 4, answerStartMultiplier: 8,
    answerEndMultiplier: 2, rcrUlWeight: 0, fp16: false,
  },
  selection: {
    checkpoints: Array.from({ length: Math.floor(steps / 400) }, (_, index) => (index + 1) * 400),
    rule: "matched untouched free-generation semantics, contingency, loop behavior, and stopping; loss cannot select",
    matchedPublicBaselineRequired: true,
    matchedU1EvidenceRequired: true,
    sealedFinalRemainsClosedUntilSelection: true,
  },
  startedUtc: new Date().toISOString(),
};
fs.writeFileSync(process.env.CONTRACT_TMP, JSON.stringify(contract, null, 2) + "\n", { flag: "wx" });
NODE
mv "$contract_tmp" "$run_dir/repair-contract.json"

export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd_headless.json}
export HELIOS_DISABLE_COOP_MAT=1 HELIOS_WG_SIZE=64 HELIOS_MAX_OUTPUT_POOL_ENTRIES=512
export ALPHA_FAIL_ON_SMOKE_TEST=1 ALPHA_ALLOW_RESUME_MISMATCH=1
export ALPHA_SFT_SHUFFLE=1 ALPHA_SFT_BALANCE_CONVERSATIONS=1
export ALPHA_SFT_START_TOKENS=4 ALPHA_SFT_START_WEIGHT=8 ALPHA_SFT_END_WEIGHT=2
export ALPHA_SAMPLE_FROM_CHECKPOINT=0 ALPHA_GPU_METRICS_SAMPLE_EVERY=25

exec nice -n 5 ionice -c2 -n7 node --expose-gc apps/cli/dist/main.js train \
  --data="$train_data" --valData="$dev_data" --requireValData=true --sft=true \
  --domain=alpha_llama --tokenizerArtifacts="$tokenizer" --initCheckpoint="$init_checkpoint" \
  --vocabSize=12288 --block=512 --layers=16 --dim=512 --heads=8 --dropout=0 \
  --activation=swiglu --ffnDim=1408 --normType=rmsnorm --posEnc=rope --ropeTheta=10000 \
  --tieEmbeddings=true --backend=helios --optim=adamw --batch=16 --accumSteps=1 \
  --steps="$steps" --lr="$learning_rate" --lrMin="$learning_rate_min" --warmupIters=200 \
  --beta1=0.9 --beta2=0.95 --eps=0.00000001 --weightDecay=0.1 --gradClip=1.0 \
  --spikeThreshold=50 --evalInterval=400 --checkpointInterval=400 --evalIters=10 \
  --sampleInterval=0 --logEvery=25 --seed=42 --strictPlanning=false --remote=false \
  --postSamples=false --runDir="$run_dir"
