#!/usr/bin/env bash
# Finite reviewed-synthetic repair from Alpha's mechanically stable U1 checkpoint.

set -euo pipefail

train_data=${1:?training conversations required}
dev_data=${2:?development conversations required}
corpus_manifest=${3:?v8 corpus manifest required}
train_mask_audit=${4:?exhaustive train mask audit required}
dev_mask_audit=${5:?exhaustive development mask audit required}
eval_manifest=${6:?evaluation freeze required}
tokenizer=${7:?Alpha tokenizer artifact required}
init_checkpoint=${8:?U1 checkpoint required}
rcr_ul_data=${9:?row-matched RCR-UL cohort required}
rcr_ul_manifest=${10:?RCR-UL remap manifest required}
run_dir=${11:?new run directory required}
steps=${12:-1200}
learning_rate=${13:-0.00001}
learning_rate_min=${14:-0.000001}

for required in \
  "$train_data" "$dev_data" "$corpus_manifest" "$train_mask_audit" \
  "$dev_mask_audit" "$eval_manifest" "$tokenizer" "$init_checkpoint" \
  "$rcr_ul_data" "$rcr_ul_manifest" apps/cli/dist/main.js; do
  [[ -f $required ]] || { echo "required file missing: $required" >&2; exit 1; }
done
[[ ! -e $run_dir ]] || { echo "run directory already exists: $run_dir" >&2; exit 1; }
[[ -z $(git status --porcelain) ]] || { echo "v8 training requires a clean committed worktree" >&2; exit 1; }
[[ $steps =~ ^[1-9][0-9]*$ && $((steps % 200)) -eq 0 ]] || {
  echo "steps must be a positive multiple of 200" >&2
  exit 2
}

sha256() { sha256sum "$1" | awk '{print $1}'; }
train_sha=$(sha256 "$train_data")
dev_sha=$(sha256 "$dev_data")
corpus_manifest_sha=$(sha256 "$corpus_manifest")
train_mask_sha=$(sha256 "$train_mask_audit")
dev_mask_sha=$(sha256 "$dev_mask_audit")
eval_manifest_sha=$(sha256 "$eval_manifest")
tokenizer_sha=$(sha256 "$tokenizer")
checkpoint_sha=$(sha256 "$init_checkpoint")
rcr_ul_sha=$(sha256 "$rcr_ul_data")
rcr_ul_manifest_sha=$(sha256 "$rcr_ul_manifest")
expected_checkpoint_sha=0453a842b264c80c3578bc419c3dc94b46420aca30cad93593d62c812f5710fb
expected_tokenizer_sha=c310343a185aecb572b8b6568b55179df248f4adec009d14a9496da354090b24
[[ $checkpoint_sha == "$expected_checkpoint_sha" ]] || { echo "U1 checkpoint hash mismatch: $checkpoint_sha" >&2; exit 1; }
[[ $tokenizer_sha == "$expected_tokenizer_sha" ]] || { echo "tokenizer hash mismatch: $tokenizer_sha" >&2; exit 1; }

TRAIN_SHA="$train_sha" DEV_SHA="$dev_sha" CORPUS_MANIFEST="$corpus_manifest" \
TRAIN_MASK_AUDIT="$train_mask_audit" DEV_MASK_AUDIT="$dev_mask_audit" \
EVAL_MANIFEST="$eval_manifest" RCR_UL_SHA="$rcr_ul_sha" RCR_UL_MANIFEST="$rcr_ul_manifest" \
node - <<'NODE'
const fs = require("node:fs");
const crypto = require("node:crypto");
const corpus = require(process.env.CORPUS_MANIFEST);
const trainMask = require(process.env.TRAIN_MASK_AUDIT);
const devMask = require(process.env.DEV_MASK_AUDIT);
const freeze = require(process.env.EVAL_MANIFEST);
const rcr = require(process.env.RCR_UL_MANIFEST);
const assert = (condition, message) => { if (!condition) throw new Error(message); };
const fileHash = (path) => crypto.createHash("sha256").update(fs.readFileSync(path)).digest("hex");
assert(corpus.schema === "alpha-chat-foundations-v8-corpus-v1", "unexpected corpus schema");
assert(corpus.sourceTreeDirty === false, "corpus was not built from a clean committed tree");
assert(corpus.outputs?.train?.sha256 === process.env.TRAIN_SHA, "training corpus hash mismatch");
assert(corpus.outputs?.dev?.sha256 === process.env.DEV_SHA, "development corpus hash mismatch");
assert(corpus.rows?.train > 4000 && corpus.rows?.dev > 400, "v8 corpus unexpectedly small");
assert(corpus.invariants?.allCandidatesReviewedExactlyOnce === true, "review population incomplete");
assert(corpus.invariants?.onlyIndependentlyAcceptedSyntheticDataTrains === true, "unreviewed data can train");
assert(corpus.invariants?.wholeBatchDevelopmentHoldout === true, "whole-batch holdout not proven");
assert(corpus.invariants?.exactNormalizedVisiblePromptExclusion === true, "visible overlap exclusion not proven");
assert(corpus.invariants?.replayDataIncluded === false, "replay data entered v8");
assert(corpus.invariants?.sealedFinalInspected === false, "sealed final was inspected");
for (const [label, audit, expectedPath, expectedSha, expectedRows] of [
  ["train", trainMask, process.env.TRAIN_MASK_AUDIT, process.env.TRAIN_SHA, corpus.rows.train],
  ["dev", devMask, process.env.DEV_MASK_AUDIT, process.env.DEV_SHA, corpus.rows.dev],
]) {
  assert(audit.schema === "alpha-sft-mask-audit-v1" && audit.result === "PASS", `${label} mask audit did not pass`);
  assert(audit.corpus?.sha256 === expectedSha && audit.corpus?.rows === expectedRows, `${label} mask audit corpus drift`);
  assert(audit.selection?.rows_sampled === expectedRows, `${label} mask audit is not exhaustive`);
  assert(audit.selection?.block_size === 512, `${label} mask audit block drift`);
  assert(audit.mask_checks?.rows_over_block_size === 0, `${label} contains overlength rows`);
  assert(audit.mask_checks?.assistant_only_state_machine === "PASS", `${label} assistant mask failed`);
  assert(audit.mask_checks?.role_markers_atomic === "PASS", `${label} role markers are not atomic`);
  assert(audit.mask_checks?.final_eot_supervised === "PASS", `${label} final EOS is not supervised`);
  assert(fileHash(expectedPath).length === 64, `${label} audit is unreadable`);
}
assert(rcr.schema === "alpha-chat-foundations-v8-rcr-ul-remap-v1", "unexpected RCR-UL remap schema");
assert(rcr.status === "complete-and-immutable", "RCR-UL remap is incomplete");
assert(rcr.sourceTreeDirty === false, "RCR-UL remap was built from a dirty tree");
assert(rcr.output?.sha256 === process.env.RCR_UL_SHA, "RCR-UL data hash mismatch");
assert(rcr.output?.rows === corpus.rows.train, "RCR-UL/positive row count mismatch");
assert(rcr.inputs?.positive?.sha256 === process.env.TRAIN_SHA, "RCR-UL positive corpus mismatch");
assert(rcr.summary?.totalPenaltyPositions > 0, "RCR-UL has no penalty positions");
assert(freeze.schema === "alpha-chat-semantic-repair-v4-evaluation-freeze-v1", "unexpected evaluation freeze");
assert(freeze.status === "development-visible; inherited-final-sealed-unexecuted", "sealed state changed");
assert(freeze.sealed_final?.sha256 === "8b71ab5f8843b14a8bbe56a473ea9cd0672b873024632c023abbe4935e48eb1d", "sealed final identity changed");
NODE

source_commit=$(git rev-parse HEAD)
mkdir -p "$run_dir"
contract_tmp="$run_dir/repair-contract.json.tmp"
SOURCE_COMMIT="$source_commit" TRAIN_DATA="$train_data" TRAIN_SHA="$train_sha" \
DEV_DATA="$dev_data" DEV_SHA="$dev_sha" CORPUS_MANIFEST="$corpus_manifest" \
CORPUS_MANIFEST_SHA="$corpus_manifest_sha" TRAIN_MASK_AUDIT="$train_mask_audit" \
TRAIN_MASK_SHA="$train_mask_sha" DEV_MASK_AUDIT="$dev_mask_audit" DEV_MASK_SHA="$dev_mask_sha" \
EVAL_MANIFEST="$eval_manifest" EVAL_MANIFEST_SHA="$eval_manifest_sha" TOKENIZER="$tokenizer" \
TOKENIZER_SHA="$tokenizer_sha" INIT_CHECKPOINT="$init_checkpoint" CHECKPOINT_SHA="$checkpoint_sha" \
RCR_UL_DATA="$rcr_ul_data" RCR_UL_SHA="$rcr_ul_sha" RCR_UL_MANIFEST="$rcr_ul_manifest" \
RCR_UL_MANIFEST_SHA="$rcr_ul_manifest_sha" STEPS="$steps" LR="$learning_rate" \
LR_MIN="$learning_rate_min" CONTRACT_TMP="$contract_tmp" node - <<'NODE'
const fs = require("node:fs");
const steps = Number(process.env.STEPS);
const contract = {
  schema: "alpha-chat-foundations-contract-v8",
  purpose: "install compact foundational conversational competence while retaining U1 loop resistance",
  eligibleForCheckpointSelection: true,
  sourceCommit: process.env.SOURCE_COMMIT,
  sourceTreeDirty: false,
  initializedFrom: { path: process.env.INIT_CHECKPOINT, sha256: process.env.CHECKPOINT_SHA },
  inputs: {
    train: { path: process.env.TRAIN_DATA, sha256: process.env.TRAIN_SHA },
    development: { path: process.env.DEV_DATA, sha256: process.env.DEV_SHA },
    corpusManifest: { path: process.env.CORPUS_MANIFEST, sha256: process.env.CORPUS_MANIFEST_SHA },
    trainMaskAudit: { path: process.env.TRAIN_MASK_AUDIT, sha256: process.env.TRAIN_MASK_SHA },
    developmentMaskAudit: { path: process.env.DEV_MASK_AUDIT, sha256: process.env.DEV_MASK_SHA },
    evaluationFreeze: { path: process.env.EVAL_MANIFEST, sha256: process.env.EVAL_MANIFEST_SHA },
    tokenizer: { path: process.env.TOKENIZER, sha256: process.env.TOKENIZER_SHA },
    rcrUlData: { path: process.env.RCR_UL_DATA, sha256: process.env.RCR_UL_SHA },
    rcrUlManifest: { path: process.env.RCR_UL_MANIFEST, sha256: process.env.RCR_UL_MANIFEST_SHA },
  },
  intervention: {
    changed: "one finite pass schedule over independently reviewed GPT-5.4 foundational dialogue with GPT-5.5 adjudication",
    unchanged: ["U1 parameters", "architecture", "tokenizer", "chat template", "decoding", "frozen evaluation"],
    retained: ["U1-derived RCR-UL repetition trajectories at weight 0.5"],
    excluded: ["public-output replay", "bulk long-form SFT", "programming curriculum", "sealed-final tuning", "exact BLAH prompt training"],
  },
  training: {
    steps,
    blockSize: 512,
    batchSize: 16,
    gradientAccumulationSteps: 1,
    optimizer: "AdamW reset from U1 parameters",
    learningRate: Number(process.env.LR),
    learningRateMin: Number(process.env.LR_MIN),
    warmupSteps: 100,
    checkpointInterval: 200,
    deterministicEpochShuffle: true,
    equalConversationWeight: true,
    answerStartTokens: 4,
    answerStartMultiplier: 4,
    answerEndMultiplier: 2,
    rcrUlWeight: 0.5,
    rcrUlEpsilon: 1e-6,
    fp16: false,
  },
  selection: {
    checkpoints: Array.from({ length: Math.floor(steps / 200) }, (_, index) => (index + 1) * 200),
    rule: "untouched free-generation semantic correctness, contingency, loop behavior, and stopping; neither loss nor isolated nonempty output can select",
    matchedPublicBaselineRequired: true,
    matchedU1EvidenceRequired: true,
    blahRerunOnlyAfterLocalSelection: true,
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
export ALPHA_SFT_START_TOKENS=4 ALPHA_SFT_START_WEIGHT=4 ALPHA_SFT_END_WEIGHT=2
export ALPHA_SAMPLE_FROM_CHECKPOINT=0 ALPHA_GPU_METRICS_SAMPLE_EVERY=25

exec nice -n 5 ionice -c2 -n7 node --expose-gc apps/cli/dist/main.js train \
  --data="$train_data" --valData="$dev_data" --requireValData=true --sft=true \
  --domain=alpha_llama --tokenizerArtifacts="$tokenizer" --initCheckpoint="$init_checkpoint" \
  --vocabSize=12288 --block=512 --layers=16 --dim=512 --heads=8 --dropout=0 \
  --activation=swiglu --ffnDim=1408 --normType=rmsnorm --posEnc=rope --ropeTheta=10000 \
  --tieEmbeddings=true --backend=helios --optim=adamw --batch=16 --accumSteps=1 \
  --steps="$steps" --lr="$learning_rate" --lrMin="$learning_rate_min" --warmupIters=100 \
  --beta1=0.9 --beta2=0.95 --eps=0.00000001 --weightDecay=0.1 --gradClip=1.0 \
  --spikeThreshold=50 --evalInterval=200 --checkpointInterval=200 --evalIters=10 \
  --sampleInterval=0 --logEvery=25 --seed=42 --strictPlanning=false --remote=false \
  --rcrUlData="$rcr_ul_data" --rcrUlWeight=0.5 --rcrUlEpsilon=0.000001 \
  --postSamples=false --runDir="$run_dir"
