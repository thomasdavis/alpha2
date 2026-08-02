#!/usr/bin/env bash
# Finite full-token instruction-pretraining pilot over the reviewed V8 dialogue.
#
# This is deliberately not another assistant-only SFT arm. It asks whether the
# same accepted conversations teach more foundational language/instruction
# structure when every next-token transition is supervised. A later, separate
# finishing stage may restore assistant-only weighting if this stage wins.

set -euo pipefail

train_data=${1:?training conversations required}
dev_data=${2:?development conversations required}
corpus_manifest=${3:?V8 corpus manifest required}
evaluation_freeze=${4:?evaluation freeze required}
tokenizer=${5:?Alpha tokenizer artifact required}
init_checkpoint=${6:?U1 checkpoint required}
run_dir=${7:?new run directory required}
steps=${8:-200}
learning_rate=${9:-0.000005}
learning_rate_min=${10:-0.0000005}

for required in \
  "$train_data" "$dev_data" "$corpus_manifest" "$evaluation_freeze" \
  "$tokenizer" "$init_checkpoint" apps/cli/dist/main.js; do
  [[ -f $required ]] || { echo "required file missing: $required" >&2; exit 1; }
done
[[ ! -e $run_dir ]] || { echo "run directory already exists: $run_dir" >&2; exit 1; }
[[ -z $(git status --porcelain) ]] || { echo "V9 pilot requires a clean committed worktree" >&2; exit 1; }
[[ $steps =~ ^[1-9][0-9]*$ && $((steps % 50)) -eq 0 ]] || {
  echo "steps must be a positive multiple of 50" >&2
  exit 2
}

sha256() { sha256sum "$1" | awk '{print $1}'; }
train_sha=$(sha256 "$train_data")
dev_sha=$(sha256 "$dev_data")
corpus_manifest_sha=$(sha256 "$corpus_manifest")
evaluation_freeze_sha=$(sha256 "$evaluation_freeze")
tokenizer_sha=$(sha256 "$tokenizer")
checkpoint_sha=$(sha256 "$init_checkpoint")

expected_train_sha=7e2c0b1256cdbdb6ff4fcc9ccd567a2b745a34e9b60b9fa8529539a940b42bef
expected_dev_sha=f693a82cff22b8a94bc5f2c3c55a46bb3a83d545737de57b40db54983ada28d1
expected_checkpoint_sha=0453a842b264c80c3578bc419c3dc94b46420aca30cad93593d62c812f5710fb
expected_tokenizer_sha=c310343a185aecb572b8b6568b55179df248f4adec009d14a9496da354090b24
expected_freeze_sha=3e5a35d01644961bf464c627b527cf99290b1ed6f56467ebaccfbe86a4c66908

[[ $train_sha == "$expected_train_sha" ]] || { echo "training corpus hash mismatch: $train_sha" >&2; exit 1; }
[[ $dev_sha == "$expected_dev_sha" ]] || { echo "development corpus hash mismatch: $dev_sha" >&2; exit 1; }
[[ $checkpoint_sha == "$expected_checkpoint_sha" ]] || { echo "U1 checkpoint hash mismatch: $checkpoint_sha" >&2; exit 1; }
[[ $tokenizer_sha == "$expected_tokenizer_sha" ]] || { echo "tokenizer hash mismatch: $tokenizer_sha" >&2; exit 1; }
[[ $evaluation_freeze_sha == "$expected_freeze_sha" ]] || { echo "evaluation freeze hash mismatch: $evaluation_freeze_sha" >&2; exit 1; }

TRAIN_SHA="$train_sha" DEV_SHA="$dev_sha" CORPUS_MANIFEST="$corpus_manifest" \
EVALUATION_FREEZE="$evaluation_freeze" node - <<'NODE'
const corpus = require(process.env.CORPUS_MANIFEST);
const freeze = require(process.env.EVALUATION_FREEZE);
const assert = (condition, message) => { if (!condition) throw new Error(message); };
assert(corpus.schema === "alpha-chat-foundations-v8-corpus-v1", "unexpected corpus schema");
assert(corpus.outputs?.train?.sha256 === process.env.TRAIN_SHA, "training corpus/manifest drift");
assert(corpus.outputs?.dev?.sha256 === process.env.DEV_SHA, "development corpus/manifest drift");
assert(corpus.rows?.train === 5141 && corpus.rows?.dev === 615, "corpus row counts drifted");
assert(corpus.invariants?.allCandidatesReviewedExactlyOnce === true, "review population incomplete");
assert(corpus.invariants?.onlyIndependentlyAcceptedSyntheticDataTrains === true, "unreviewed data can train");
assert(corpus.invariants?.wholeBatchDevelopmentHoldout === true, "whole-batch development holdout not proven");
assert(corpus.invariants?.exactNormalizedVisiblePromptExclusion === true, "visible overlap exclusion not proven");
assert(corpus.invariants?.replayDataIncluded === false, "public replay data entered training");
assert(corpus.invariants?.sealedFinalInspected === false, "sealed final was inspected");
assert(freeze.schema === "alpha-chat-semantic-repair-v4-evaluation-freeze-v1", "unexpected evaluation freeze");
assert(freeze.status === "development-visible; inherited-final-sealed-unexecuted", "sealed state changed");
assert(freeze.sealed_final?.sha256 === "8b71ab5f8843b14a8bbe56a473ea9cd0672b873024632c023abbe4935e48eb1d", "sealed final identity changed");
NODE

source_commit=$(git rev-parse HEAD)
mkdir -p "$run_dir"
contract_tmp="$run_dir/experiment-contract.json.tmp"
SOURCE_COMMIT="$source_commit" TRAIN_DATA="$train_data" TRAIN_SHA="$train_sha" \
DEV_DATA="$dev_data" DEV_SHA="$dev_sha" CORPUS_MANIFEST="$corpus_manifest" \
CORPUS_MANIFEST_SHA="$corpus_manifest_sha" EVALUATION_FREEZE="$evaluation_freeze" \
EVALUATION_FREEZE_SHA="$evaluation_freeze_sha" TOKENIZER="$tokenizer" TOKENIZER_SHA="$tokenizer_sha" \
INIT_CHECKPOINT="$init_checkpoint" CHECKPOINT_SHA="$checkpoint_sha" STEPS="$steps" \
LR="$learning_rate" LR_MIN="$learning_rate_min" CONTRACT_TMP="$contract_tmp" node - <<'NODE'
const fs = require("node:fs");
const steps = Number(process.env.STEPS);
const contract = {
  schema: "alpha-chat-foundations-v9-ipt-pilot-contract-v1",
  purpose: "test full-token instruction pretraining on the exact reviewed V8 dialogues before any further assistant-only SFT",
  eligibleForCheckpointSelection: false,
  sourceCommit: process.env.SOURCE_COMMIT,
  sourceTreeDirty: false,
  initializedFrom: { path: process.env.INIT_CHECKPOINT, sha256: process.env.CHECKPOINT_SHA },
  inputs: {
    train: { path: process.env.TRAIN_DATA, sha256: process.env.TRAIN_SHA },
    development: { path: process.env.DEV_DATA, sha256: process.env.DEV_SHA },
    corpusManifest: { path: process.env.CORPUS_MANIFEST, sha256: process.env.CORPUS_MANIFEST_SHA },
    evaluationFreeze: { path: process.env.EVALUATION_FREEZE, sha256: process.env.EVALUATION_FREEZE_SHA },
    tokenizer: { path: process.env.TOKENIZER, sha256: process.env.TOKENIZER_SHA },
  },
  intervention: {
    changed: "assistant-only masks removed so every next-token transition in the reviewed chat trajectories is supervised",
    unchanged: ["U1 parameters", "V8 dialogue bytes", "architecture", "tokenizer", "chat template", "decoding", "frozen evaluation"],
    excluded: ["public-output replay", "exact BLAH prompt training", "sealed-final tuning", "RCR-UL during this causal stage"],
  },
  training: {
    stage: "full-token instruction pretraining pilot",
    steps,
    blockSize: 512,
    batchSize: 16,
    packed: true,
    assistantOnlyMask: false,
    optimizer: "AdamW reset from U1 parameters",
    learningRate: Number(process.env.LR),
    learningRateMin: Number(process.env.LR_MIN),
    warmupSteps: 20,
    checkpointInterval: 50,
    fp16: false,
  },
  gates: {
    stageASelection: "visible free-generation semantics plus structural prerequisites; validation loss cannot select",
    stageBAllowedOnlyIf: "at least one checkpoint shows semantic improvement without loop, EOS, role-leak, or nonempty regression",
    blahRerunOnlyAfter: "a completed stage-B candidate beats the public checkpoint locally",
    sealedFinalRemainsClosed: true,
  },
  startedUtc: new Date().toISOString(),
};
fs.writeFileSync(process.env.CONTRACT_TMP, JSON.stringify(contract, null, 2) + "\n", { flag: "wx" });
NODE
mv "$contract_tmp" "$run_dir/experiment-contract.json"

export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd_headless.json}
export HELIOS_DISABLE_COOP_MAT=1 HELIOS_WG_SIZE=64 HELIOS_MAX_OUTPUT_POOL_ENTRIES=512
export ALPHA_FAIL_ON_SMOKE_TEST=1 ALPHA_ALLOW_RESUME_MISMATCH=1
export ALPHA_SAMPLE_FROM_CHECKPOINT=0 ALPHA_GPU_METRICS_SAMPLE_EVERY=25

exec nice -n 5 ionice -c2 -n7 node --expose-gc apps/cli/dist/main.js train \
  --data="$train_data" --valData="$dev_data" --requireValData=true --sft=false --packed=true \
  --domain=alpha_llama --tokenizerArtifacts="$tokenizer" --initCheckpoint="$init_checkpoint" \
  --vocabSize=12288 --block=512 --layers=16 --dim=512 --heads=8 --dropout=0 \
  --activation=swiglu --ffnDim=1408 --normType=rmsnorm --posEnc=rope --ropeTheta=10000 \
  --tieEmbeddings=true --backend=helios --optim=adamw --batch=16 --accumSteps=1 \
  --steps="$steps" --lr="$learning_rate" --lrMin="$learning_rate_min" --warmupIters=20 \
  --beta1=0.9 --beta2=0.95 --eps=0.00000001 --weightDecay=0.1 --gradClip=1.0 \
  --spikeThreshold=50 --evalInterval=50 --checkpointInterval=50 --evalIters=10 \
  --sampleInterval=0 --logEvery=10 --seed=42 --strictPlanning=false --remote=false \
  --postSamples=false --runDir="$run_dir"
