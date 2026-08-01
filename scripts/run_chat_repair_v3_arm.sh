#!/usr/bin/env bash
# One fail-closed Alpha chat-repair-v3 arm. C0 and U1 must invoke this exact
# launcher from the same clean commit with weights 0.0 and 0.5 respectively.
#
# Usage:
#   scripts/run_chat_repair_v3_arm.sh \
#     ARM RCR_WEIGHT POSITIVE_COHORT NEGATIVE_COHORT RCR_MANIFEST \
#     FREEZE_MANIFEST DEV_SFT TOKENIZER INIT_CHECKPOINT RUN_DIR

set -euo pipefail

arm=${1:?arm label C0 or U1 required}
rcr_weight=${2:?RCR-UL weight required}
positive_data=${3:?positive cohort required}
negative_data=${4:?negative RCR-UL cohort required}
rcr_manifest=${5:?RCR-UL manifest required}
freeze_manifest=${6:?v3 freeze manifest required}
dev_data=${7:?teacher-forced development corpus required}
tokenizer=${8:?Alpha tokenizer artifact required}
init_checkpoint=${9:?initial checkpoint required}
run_dir=${10:?new run directory required}

[[ $arm == C0 || $arm == U1 ]] || { echo "ARM must be C0 or U1" >&2; exit 2; }
[[ $arm != C0 || $rcr_weight == 0 || $rcr_weight == 0.0 ]] || {
  echo "C0 requires RCR_WEIGHT=0.0" >&2; exit 2;
}
[[ $arm != U1 || $rcr_weight == 0.5 ]] || { echo "U1 requires RCR_WEIGHT=0.5" >&2; exit 2; }
for required in \
  "$positive_data" "$negative_data" "$rcr_manifest" "$freeze_manifest" \
  "$dev_data" "$tokenizer" "$init_checkpoint" apps/cli/dist/main.js; do
  [[ -f $required ]] || { echo "required file missing: $required" >&2; exit 1; }
done
[[ ! -e $run_dir ]] || { echo "run directory already exists: $run_dir" >&2; exit 1; }
[[ -z $(git status --porcelain) ]] || { echo "v3 training requires a clean committed worktree" >&2; exit 1; }

sha256() { sha256sum "$1" | awk '{print $1}'; }
positive_sha=$(sha256 "$positive_data")
negative_sha=$(sha256 "$negative_data")
rcr_manifest_sha=$(sha256 "$rcr_manifest")
freeze_manifest_sha=$(sha256 "$freeze_manifest")
dev_sha=$(sha256 "$dev_data")
tokenizer_sha=$(sha256 "$tokenizer")
checkpoint_sha=$(sha256 "$init_checkpoint")
expected_checkpoint_sha=399f776b49acc0c8834ff8a7f2390454e2c5f2d833a264e3f83ff546e973cfec
[[ $checkpoint_sha == "$expected_checkpoint_sha" ]] || {
  echo "initial checkpoint hash mismatch: $checkpoint_sha" >&2; exit 1;
}

POSITIVE_SHA="$positive_sha" NEGATIVE_SHA="$negative_sha" CHECKPOINT_SHA="$checkpoint_sha" \
RCR_MANIFEST="$rcr_manifest" FREEZE_MANIFEST="$freeze_manifest" node - <<'NODE'
const rcr = require(process.env.RCR_MANIFEST);
const freeze = require(process.env.FREEZE_MANIFEST);
const assert = (condition, message) => { if (!condition) throw new Error(message); };
assert(freeze.schema === "alpha-chat-repair-v3-freeze-v1", "unexpected v3 freeze schema");
assert(freeze.outputs?.positive_cohort?.sha256 === process.env.POSITIVE_SHA, "positive cohort differs from freeze");
assert(freeze.counts?.rollout_selected === 4096, "freeze does not bind 4096 rollout identities");
assert(freeze.counts?.development_selected === 96, "freeze does not bind the 96-case selector");
assert(rcr.schema === "alpha-rcr-ul-cohort-manifest-v1", "unexpected RCR-UL manifest schema");
assert(rcr.status === "complete-and-immutable", "RCR-UL cohort is not complete");
assert(rcr.inputs?.checkpoint_sha256 === process.env.CHECKPOINT_SHA, "RCR-UL checkpoint identity mismatch");
assert(rcr.inputs?.positive_cohort?.sha256 === process.env.POSITIVE_SHA, "RCR-UL positive cohort mismatch");
assert(rcr.outputs?.negative_cohort?.sha256 === process.env.NEGATIVE_SHA, "RCR-UL negative cohort mismatch");
assert(rcr.summary?.rows === 4096, "RCR-UL cohort does not contain 4096 rows");
assert(rcr.summary?.eligible_negative_rows > 0, "RCR-UL cohort has no eligible negative rows");
assert(rcr.summary?.total_penalty_positions > 0, "RCR-UL cohort has no penalty positions");
NODE

source_commit=$(git rev-parse HEAD)
mkdir -p "$run_dir"
contract_tmp="$run_dir/repair-contract.json.tmp"
ARM="$arm" RCR_WEIGHT="$rcr_weight" SOURCE_COMMIT="$source_commit" \
POSITIVE_DATA="$positive_data" POSITIVE_SHA="$positive_sha" \
NEGATIVE_DATA="$negative_data" NEGATIVE_SHA="$negative_sha" \
RCR_MANIFEST="$rcr_manifest" RCR_MANIFEST_SHA="$rcr_manifest_sha" \
FREEZE_MANIFEST="$freeze_manifest" FREEZE_MANIFEST_SHA="$freeze_manifest_sha" \
DEV_DATA="$dev_data" DEV_SHA="$dev_sha" TOKENIZER="$tokenizer" TOKENIZER_SHA="$tokenizer_sha" \
INIT_CHECKPOINT="$init_checkpoint" CHECKPOINT_SHA="$checkpoint_sha" CONTRACT_TMP="$contract_tmp" node - <<'NODE'
const fs = require("node:fs");
const freeze = require(process.env.FREEZE_MANIFEST);
const rcr = require(process.env.RCR_MANIFEST);
const contract = {
  schema: "alpha-chat-repair-contract-v3",
  purpose: "test rollout-conditioned repetition unlikelihood without changing the positive corpus or execution path",
  arm: process.env.ARM,
  sourceCommit: process.env.SOURCE_COMMIT,
  sourceTreeDirty: false,
  initializedFrom: { path: process.env.INIT_CHECKPOINT, sha256: process.env.CHECKPOINT_SHA },
  inputs: {
    positiveCohort: { path: process.env.POSITIVE_DATA, sha256: process.env.POSITIVE_SHA },
    negativeCohort: { path: process.env.NEGATIVE_DATA, sha256: process.env.NEGATIVE_SHA },
    rcrManifest: { path: process.env.RCR_MANIFEST, sha256: process.env.RCR_MANIFEST_SHA },
    freezeManifest: { path: process.env.FREEZE_MANIFEST, sha256: process.env.FREEZE_MANIFEST_SHA },
    teacherForcedDev: { path: process.env.DEV_DATA, sha256: process.env.DEV_SHA },
    tokenizer: { path: process.env.TOKENIZER, sha256: process.env.TOKENIZER_SHA },
  },
  training: {
    steps: 400,
    blockSize: 1024,
    batchSize: 16,
    gradientAccumulationSteps: 1,
    optimizer: "AdamW reset from identical initial parameters",
    learningRate: 5e-6,
    learningRateMin: 1e-6,
    warmupSteps: 50,
    checkpointInterval: 50,
    deterministicEpochShuffle: true,
    equalConversationWeight: true,
    answerStartTokens: 4,
    answerStartMultiplier: 8,
    answerEndMultiplier: 2,
    rcrUlWeight: Number(process.env.RCR_WEIGHT),
    rcrUlEpsilon: 1e-6,
    matchedNegativeBranchAlwaysExecutes: true,
    fp16: false,
  },
  selection: {
    developmentSelector: freeze.outputs.development_selector,
    qualitativePanel: freeze.outputs.development_panel,
    priorV2Regression: "eligible-69 only",
    generation: "deterministic greedy, 128 tokens",
    rule: "free conversational behavior first; validation loss cannot select a checkpoint",
    sealedFinalRemainsClosed: true,
  },
  cohortSummary: rcr.summary,
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
export ALPHA_ALLOW_RESUME_MISMATCH=1
export ALPHA_SFT_SHUFFLE=1
export ALPHA_SFT_BALANCE_CONVERSATIONS=1
export ALPHA_SFT_START_TOKENS=4
export ALPHA_SFT_START_WEIGHT=8
export ALPHA_SFT_END_WEIGHT=2
export ALPHA_SAMPLE_FROM_CHECKPOINT=0
export ALPHA_GPU_METRICS_SAMPLE_EVERY=1

exec nice -n 5 ionice -c2 -n7 node --expose-gc apps/cli/dist/main.js train \
  --data="$positive_data" \
  --valData="$dev_data" \
  --requireValData=true \
  --sft=true \
  --rcrUlData="$negative_data" \
  --rcrUlWeight="$rcr_weight" \
  --rcrUlEpsilon=0.000001 \
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
  --batch=16 \
  --accumSteps=1 \
  --steps=400 \
  --lr=0.000005 \
  --lrMin=0.000001 \
  --warmupIters=50 \
  --beta1=0.9 \
  --beta2=0.95 \
  --eps=0.00000001 \
  --weightDecay=0.1 \
  --gradClip=1.0 \
  --spikeThreshold=50 \
  --evalInterval=50 \
  --checkpointInterval=50 \
  --evalIters=10 \
  --sampleInterval=0 \
  --logEvery=10 \
  --seed=42 \
  --strictPlanning=false \
  --remote=false \
  --postSamples=false \
  --runDir="$run_dir"
