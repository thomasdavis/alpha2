#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STEPS="${1:-100}"
ITERATIONS="${2:-20}"
RUN_RETRIES="${RUN_RETRIES:-0}"
RUN_CONTINUE_ON_OK="${RUN_CONTINUE_ON_OK:-0}"
BACKEND="${BACKEND:-helios}"
TUNE_REQUIRE_OK="${TUNE_REQUIRE_OK:-1}"
PERF_DIR="perf"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_FILE="${PERF_DIR}/tune-adaptive-env-${TS}.csv"
BEST_FILE="${PERF_DIR}/best-adaptive-env-${TS}.env"

mkdir -p "$PERF_DIR"

# Candidate tuples:
# mem_poll sync_interval deferred_threshold pending_threshold gpu_metrics_sample_every
COMBOS=(
  "14 11 32 28 50"
  "12 10 28 24 50"
  "16 12 36 32 50"
  "10 10 28 24 50"
  "14 10 28 24 50"
  "12 11 32 28 50"
  "16 10 28 24 50"
  "12 10 32 24 50"
  "12 10 24 20 50"
  "10 11 32 24 50"
  "14 11 32 28 75"
  "12 10 28 24 75"
  "16 12 36 32 75"
  "10 10 28 24 75"
  "14 10 28 24 75"
  "12 11 32 28 75"
  "16 10 28 24 75"
  "12 10 32 24 75"
  "12 10 24 20 75"
  "10 11 32 24 75"
)

echo "iter,mem_poll,sync_interval,deferred_threshold,pending_threshold,gpu_metrics_sample_every,avg_tok_s,last_tok_s,status,speedup_pct,run_log" > "$OUT_FILE"

combo_count="${#COMBOS[@]}"

for ((iter=1; iter<=ITERATIONS; iter++)); do
  idx=$(( (iter - 1) % combo_count ))
  read -r mem sync dthr pthr gsample <<< "${COMBOS[$idx]}"

  ALPHA_ADAPTIVE_MEM_STATS_POLL_EVERY="$mem" \
  ALPHA_ADAPTIVE_SYNC_MIN_INTERVAL="$sync" \
  ALPHA_ADAPTIVE_SYNC_DEFERRED_THRESHOLD="$dthr" \
  ALPHA_ADAPTIVE_SYNC_PENDING_THRESHOLD="$pthr" \
  ALPHA_GPU_METRICS_SAMPLE_EVERY="$gsample" \
  BACKEND="$BACKEND" \
  RUN_RETRIES="$RUN_RETRIES" \
  RUN_CONTINUE_ON_OK="$RUN_CONTINUE_ON_OK" \
  scripts/run-compiled-benchmark.sh "$STEPS" >/tmp/tune-adaptive-env-${TS}-${iter}.log 2>&1 || true

  source perf/last-benchmark.env

  echo "${iter},${mem},${sync},${dthr},${pthr},${gsample},${AVG_TOK_S},${LAST_TOK_S},${STATUS},${SPEEDUP_PCT},${RUN_LOG}" >> "$OUT_FILE"
  echo "[tune ${iter}/${ITERATIONS}] mem=${mem} sync=${sync} d=${dthr} p=${pthr} gsample=${gsample} -> avg_tok_s=${AVG_TOK_S} status=${STATUS}"
done

best_line=""
best_rank=-1
best_tok=0

while IFS=, read -r iter mem sync dthr pthr gsample avg_tok_s _ status _ run_log; do
  [[ "$iter" == "iter" ]] && continue

  if [[ "$TUNE_REQUIRE_OK" == "1" && "$status" != "ok" ]]; then
    continue
  fi

  rank=0
  case "$status" in
    ok) rank=3 ;;
    unstable) rank=2 ;;
    smoke_fail) rank=1 ;;
    *) rank=0 ;;
  esac

  take=0
  if (( rank > best_rank )); then
    take=1
  elif (( rank == best_rank )); then
    if awk -v a="$avg_tok_s" -v b="$best_tok" 'BEGIN{exit !(a>b)}'; then
      take=1
    fi
  fi

  if (( take == 1 )); then
    best_rank="$rank"
    best_tok="$avg_tok_s"
    best_line="$iter,$mem,$sync,$dthr,$pthr,$gsample,$avg_tok_s,$status,$run_log"
  fi
done < "$OUT_FILE"

if [[ -z "$best_line" ]]; then
  echo "No acceptable candidate found (TUNE_REQUIRE_OK=${TUNE_REQUIRE_OK})."
  echo "SWEEP_FILE=${OUT_FILE}"
  exit 1
fi

IFS=, read -r best_iter best_mem best_sync best_dthr best_pthr best_gsample best_avg best_status best_log <<< "$best_line"

cat > "$BEST_FILE" <<EOF
ALPHA_ADAPTIVE_MEM_STATS_POLL_EVERY=${best_mem}
ALPHA_ADAPTIVE_SYNC_MIN_INTERVAL=${best_sync}
ALPHA_ADAPTIVE_SYNC_DEFERRED_THRESHOLD=${best_dthr}
ALPHA_ADAPTIVE_SYNC_PENDING_THRESHOLD=${best_pthr}
ALPHA_GPU_METRICS_SAMPLE_EVERY=${best_gsample}
EOF

echo "SWEEP_FILE=${OUT_FILE}"
echo "BEST=${best_line}"
echo "BEST_ENV=${BEST_FILE}"
echo "APPLY_WITH:"
echo "  source ${BEST_FILE}"
