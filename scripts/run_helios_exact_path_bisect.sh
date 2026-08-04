#!/usr/bin/env bash
#
# X45 — staged exact-path bisection for X25–X31.
#
# Purpose
# -------
# Handoff §12 Phase B requires validating X25–X31 "one stage at a time against
# the legacy path", because combining them makes a regression unbisectable.
# This is that runner. It exists so a rental is push-button rather than
# improvised under a termination deadline.
#
# Two modes, deliberately separated (operating method §7: exact-path work and
# changed-mathematics work must never be mixed):
#
#   --local      correctness only. Runs the X43 parity lane per stage on a
#                software Vulkan device. Free. Answers "does this stage change
#                the numbers?" and nothing else.
#   --physical   adds a warm throughput measurement per stage on the target
#                device. Answers "what is this stage worth?".
#
# Run --local first, always. Operating method §3.5: never rent a GPU to answer
# a question a free measurement can already kill.
#
# The gate
# --------
# X25–X31 are EXACT transforms — fusions and layout changes that must not alter
# arithmetic. So the parity gate is BIT-EXACT loss against the legacy baseline,
# not a tolerance. A stage that moves the loss sequence at all has changed the
# mathematics and is a failure, regardless of speed. This is the strongest
# boring explanation (§3.4): a "faster" fused path that is merely doing less
# work would show up here and nowhere else.
#
# Usage:
#   scripts/run_helios_exact_path_bisect.sh --local
#   TRAIN_DATA=... VAL_DATA=... TOKENIZER=... OUT_ROOT=... \
#     scripts/run_helios_exact_path_bisect.sh --physical
set -euo pipefail

MODE="${1:---local}"
STEPS="${STEPS:-20}"
BATCH="${BATCH:-10}"
BLOCK="${BLOCK:-1024}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# ── Stage table ─────────────────────────────────────────────────────────────
# Cumulative: each stage adds one mechanism to all previous ones. Stage 0 is
# the legacy path with every exact-path mechanism disabled.
#
# Flags are DISABLE_-style, so "enabling" stage N means dropping its flag from
# the disable set.
STAGE_NAMES=(
  "s0_legacy_all_disabled"
  "s1_residual_add_rmsnorm"          # X27
  "s2_qkv_head_major_rope"           # X28
  "s3_qkv_combined_backward"         # X29
  "s4_flash_token_major"             # X30
  "s5_flash_grouped_qkv_backward"    # X31
  "s6_device_resident_grad_norm"     # X26
)
# Disable-flags still active at each stage (space separated, empty = none)
STAGE_DISABLES=(
  "HELIOS_DISABLE_RESIDUAL_ADD_RMSNORM HELIOS_DISABLE_QKV_HEAD_MAJOR_ROPE HELIOS_DISABLE_QKV_COMBINED_BACKWARD HELIOS_DISABLE_FLASH_TOKEN_MAJOR HELIOS_DISABLE_FLASH_GROUPED_QKV_BACKWARD ALPHA_DISABLE_TOTAL_SUMSQ"
  "HELIOS_DISABLE_QKV_HEAD_MAJOR_ROPE HELIOS_DISABLE_QKV_COMBINED_BACKWARD HELIOS_DISABLE_FLASH_TOKEN_MAJOR HELIOS_DISABLE_FLASH_GROUPED_QKV_BACKWARD ALPHA_DISABLE_TOTAL_SUMSQ"
  "HELIOS_DISABLE_QKV_COMBINED_BACKWARD HELIOS_DISABLE_FLASH_TOKEN_MAJOR HELIOS_DISABLE_FLASH_GROUPED_QKV_BACKWARD ALPHA_DISABLE_TOTAL_SUMSQ"
  "HELIOS_DISABLE_FLASH_TOKEN_MAJOR HELIOS_DISABLE_FLASH_GROUPED_QKV_BACKWARD ALPHA_DISABLE_TOTAL_SUMSQ"
  "HELIOS_DISABLE_FLASH_GROUPED_QKV_BACKWARD ALPHA_DISABLE_TOTAL_SUMSQ"
  "ALPHA_DISABLE_TOTAL_SUMSQ"
  ""
)

OUT_ROOT="${OUT_ROOT:-$repo_root/.bisect-out/$(date -u +%Y%m%dT%H%M%SZ)}"
[ -e "$OUT_ROOT" ] && { echo "OUT_ROOT exists, refusing: $OUT_ROOT"; exit 2; }
mkdir -p "$OUT_ROOT"

# ── Predeclare the contract BEFORE running anything (§20) ───────────────────
commit="$(git rev-parse HEAD)"
dirty="$(git status --porcelain | wc -l)"
git diff HEAD > "$OUT_ROOT/dirty.patch" || true
cat > "$OUT_ROOT/CONTRACT.json" <<EOF
{
  "schema": "alpha-helios-x45-exact-path-bisect-v1",
  "declared_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "mode": "$MODE",
  "source_commit": "$commit",
  "dirty_files": $dirty,
  "stages": $(printf '%s\n' "${STAGE_NAMES[@]}" | python3 -c 'import json,sys;print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))'),
  "gate": {
    "kind": "bit-exact",
    "rule": "every stage must reproduce stage 0 loss sequence exactly; X25-X31 are exact transforms",
    "rationale": "a fused path that is faster because it does less work fails here and nowhere else"
  },
  "steps": $STEPS,
  "batch": $BATCH,
  "block": $BLOCK
}
EOF
echo "contract written: $OUT_ROOT/CONTRACT.json"

run_stage_local() {
  local name="$1"
  local disables="$2"
  local dir="$OUT_ROOT/$name"
  mkdir -p "$dir"
  local env_args=(ALPHA_PARITY_ALLOW_SOFTWARE_DEVICE=1 HELIOS_DISABLE_COOP_MAT=1)
  for f in $disables; do env_args+=("$f=1"); done
  printf '%s\n' "${env_args[@]}" > "$dir/env.txt"
  set +e
  env "${env_args[@]}" npx vitest run --root packages/tests parity-helios \
    > "$dir/parity.log" 2>&1
  echo $? > "$dir/exit-code.txt"
  set -e
  grep -oE "Tests  [0-9]+ failed \| [0-9]+ passed|Tests  [0-9]+ passed" "$dir/parity.log" \
    | tail -1 > "$dir/result.txt" || echo "NO RESULT" > "$dir/result.txt"
  printf '  %-32s %s\n' "$name" "$(cat "$dir/result.txt")"
}

run_stage_physical() {
  local name="$1"
  local disables="$2"
  local dir="$OUT_ROOT/$name"
  mkdir -p "$dir"
  : "${TRAIN_DATA:?set TRAIN_DATA}"; : "${VAL_DATA:?set VAL_DATA}"; : "${TOKENIZER:?set TOKENIZER}"
  local env_args=(HELIOS_HOST_TIMING=1)
  for f in $disables; do env_args+=("$f=1"); done
  printf '%s\n' "${env_args[@]}" > "$dir/env.txt"
  set +e
  env "${env_args[@]}" node --expose-gc apps/cli/dist/main.js train \
    --backend=helios --data="$TRAIN_DATA" --valData="$VAL_DATA" --requireValData=true \
    --domain=alpha_llama --tokenizerArtifacts="$TOKENIZER" --vocabSize=12288 \
    --block="$BLOCK" --layers=18 --dim=640 --heads=10 --activation=swiglu --ffnDim=1728 \
    --normType=rmsnorm --posEnc=rope --ropeTheta=10000 --tieEmbeddings=true \
    --batch="$BATCH" --accumSteps=1 --steps="$STEPS" --lr=0.002 --trace=true \
    > "$dir/console.log" 2>&1
  echo $? > "$dir/exit-code.txt"
  set -e
  grep -oE "loss=[0-9.]+" "$dir/console.log" > "$dir/losses.txt" || true
  grep -oE "[0-9]+ tok/s" "$dir/console.log" | tr -d ' tok/s' > "$dir/tps.txt" || true
  local med
  med=$(sort -n "$dir/tps.txt" 2>/dev/null | awk '{a[NR]=$1} END{if(NR)print a[int((NR+1)/2)]}')
  echo "${med:-NA}" > "$dir/warm_median_tps.txt"
  printf '  %-32s median %s tok/s  exit %s\n' "$name" "${med:-NA}" "$(cat "$dir/exit-code.txt")"
}

echo
echo "== X45 exact-path bisection ($MODE) =="
for i in "${!STAGE_NAMES[@]}"; do
  if [ "$MODE" = "--physical" ]; then
    run_stage_physical "${STAGE_NAMES[$i]}" "${STAGE_DISABLES[$i]}"
  else
    run_stage_local "${STAGE_NAMES[$i]}" "${STAGE_DISABLES[$i]}"
  fi
done

# ── Bit-exact parity check against stage 0 ──────────────────────────────────
echo
echo "== parity against s0_legacy_all_disabled =="
base="$OUT_ROOT/${STAGE_NAMES[0]}"
for i in "${!STAGE_NAMES[@]}"; do
  [ "$i" = "0" ] && continue
  d="$OUT_ROOT/${STAGE_NAMES[$i]}"
  if [ "$MODE" = "--physical" ]; then
    if diff -q "$base/losses.txt" "$d/losses.txt" >/dev/null 2>&1; then
      echo "  PASS bit-exact  ${STAGE_NAMES[$i]}"
    else
      echo "  FAIL DIVERGED   ${STAGE_NAMES[$i]}  <-- exact transform changed the mathematics"
    fi
  else
    if [ "$(cat "$base/result.txt")" = "$(cat "$d/result.txt")" ]; then
      echo "  PASS same-parity ${STAGE_NAMES[$i]}"
    else
      echo "  FAIL parity-diff ${STAGE_NAMES[$i]}"
    fi
  fi
done

find "$OUT_ROOT" -type f ! -name ARTIFACTS.sha256 | sort | xargs sha256sum > "$OUT_ROOT/ARTIFACTS.sha256"
echo
echo "artifacts: $OUT_ROOT"
echo "hash manifest: $(wc -l < "$OUT_ROOT/ARTIFACTS.sha256") files"
