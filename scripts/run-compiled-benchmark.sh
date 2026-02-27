#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STEPS="${1:-100}"
BACKEND="${BACKEND:-helios}"
DATA_PATH="${DATA_PATH:-data/abc-small.txt}"
BATCH="${BATCH:-2}"
BLOCK="${BLOCK:-64}"
LAYERS="${LAYERS:-2}"
DIM="${DIM:-128}"
HEADS="${HEADS:-4}"
LOG_EVERY="${LOG_EVERY:-25}"
HELIOS_WG_SIZE="${HELIOS_WG_SIZE:-256}"
LR="${LR:-0.0001}"
GRAD_CLIP="${GRAD_CLIP:-0}"
COMPILE_TIMEOUT_SEC="${COMPILE_TIMEOUT_SEC:-180}"

timestamp_utc() {
  date -u +"%Y%m%dT%H%M%SZ"
}

now_ms() {
  date +%s%3N
}

TS="$(timestamp_utc)"
RUN_DIR="runs/compiled-binary-${BACKEND}-${TS}"
PERF_DIR="perf"
HISTORY_FILE="${PERF_DIR}/compiled-loop-history.csv"
COMPILE_LOG="${PERF_DIR}/compile-${TS}.log"
RUN_LOG="${PERF_DIR}/run-${TS}.log"
SUMMARY_FILE="${PERF_DIR}/last-benchmark.env"
METRICS_FILE="${RUN_DIR}/metrics.jsonl"

mkdir -p "$PERF_DIR"

if [[ ! -f "$HISTORY_FILE" ]]; then
  echo "timestamp,git_commit,steps,backend,compile_ms,run_ms,avg_tok_s,last_tok_s,speedup_pct,run_dir,log_file,status" > "$HISTORY_FILE"
fi

echo "[bench] compile start (steps=${STEPS}, backend=${BACKEND})"
compile_start_ms="$(now_ms)"
if ! timeout "${COMPILE_TIMEOUT_SEC}" npm run bun:compile >"$COMPILE_LOG" 2>&1; then
  compile_end_ms="$(now_ms)"
  compile_ms="$((compile_end_ms - compile_start_ms))"
  git_commit="$(git rev-parse --short HEAD)"
  echo "${TS},${git_commit},${STEPS},${BACKEND},${compile_ms},0,0.000,0.000,0.00,${RUN_DIR},${COMPILE_LOG},compile_failed" >> "$HISTORY_FILE"
  cat > "$SUMMARY_FILE" <<EOF
TIMESTAMP=${TS}
GIT_COMMIT=${git_commit}
STEPS=${STEPS}
BACKEND=${BACKEND}
COMPILE_MS=${compile_ms}
RUN_MS=0
AVG_TOK_S=0.000
LAST_TOK_S=0.000
SPEEDUP_PCT=0.00
RUN_DIR=${RUN_DIR}
RUN_LOG=${COMPILE_LOG}
HELIOS_WG_SIZE=${HELIOS_WG_SIZE}
LR=${LR}
GRAD_CLIP=${GRAD_CLIP}
STATUS=compile_failed
EOF
  echo "[bench] compile failed or timed out after ${COMPILE_TIMEOUT_SEC}s"
  echo "[bench] compile log tail:"
  tail -n 40 "$COMPILE_LOG"
  exit 1
fi
compile_end_ms="$(now_ms)"
compile_ms="$((compile_end_ms - compile_start_ms))"
echo "[bench] compile done in ${compile_ms} ms"

echo "[bench] run start (log: ${RUN_LOG}, helios_wg=${HELIOS_WG_SIZE})"
run_start_ms="$(now_ms)"
set +e
HELIOS_WG_SIZE="${HELIOS_WG_SIZE}" ./.bun-out/alpha train \
  --data="${DATA_PATH}" \
  --backend="${BACKEND}" \
  --steps="${STEPS}" \
  --lr="${LR}" \
  --batch="${BATCH}" \
  --block="${BLOCK}" \
  --layers="${LAYERS}" \
  --dim="${DIM}" \
  --heads="${HEADS}" \
  --gradClip="${GRAD_CLIP}" \
  --accumSteps=1 \
  --evalInterval="${STEPS}" \
  --evalIters=1 \
  --sampleInterval=0 \
  --logEvery="${LOG_EVERY}" \
  --postSamples=false \
  --remote=false \
  --trace=false \
  --runDir="${RUN_DIR}" >"$RUN_LOG" 2>&1
run_status="$?"
set -e
run_end_ms="$(now_ms)"
run_ms="$((run_end_ms - run_start_ms))"

avg_tok_s="0.000"
last_tok_s="0.000"
if [[ -f "$METRICS_FILE" ]]; then
  tok_series="$(grep -Eo '"tokens_per_sec":[0-9.+-eE]+' "$METRICS_FILE" | awk -F: '{print $2}' || true)"
else
  tok_series=""
fi
if [[ -z "$tok_series" ]]; then
  tok_series="$(grep -Eo '([0-9]+([.][0-9]+)?) tok/s' "$RUN_LOG" | awk '{print $1}' || true)"
fi
if [[ -n "$tok_series" ]]; then
  avg_tok_s="$(echo "$tok_series" | awk 'BEGIN{s=0;n=0} {s+=$1;n++} END{if(n>0) printf "%.3f", s/n; else printf "0.000"}')"
  last_tok_s="$(echo "$tok_series" | awk 'END{if(NR>0) printf "%.3f", $1; else printf "0.000"}')"
fi

prev_avg_tok_s="$(awk -F, 'NR>1 && $12=="ok" {v=$7} END{print v}' "$HISTORY_FILE")"

speedup_pct="0.00"
if [[ -n "$prev_avg_tok_s" && "$prev_avg_tok_s" != "0" && "$prev_avg_tok_s" != "0.000" ]]; then
  speedup_pct="$(awk -v cur="$avg_tok_s" -v prev="$prev_avg_tok_s" 'BEGIN{printf "%.2f", ((cur-prev)/prev)*100}')"
fi

git_commit="$(git rev-parse --short HEAD)"
status_label="ok"
if [[ "$run_status" -ne 0 ]]; then
  status_label="failed"
elif rg -q 'loss=NaN|Inf/NaN|smoke_test: FAIL' "$RUN_LOG"; then
  status_label="unstable"
fi

echo "${TS},${git_commit},${STEPS},${BACKEND},${compile_ms},${run_ms},${avg_tok_s},${last_tok_s},${speedup_pct},${RUN_DIR},${RUN_LOG},${status_label}" >> "$HISTORY_FILE"

cat > "$SUMMARY_FILE" <<EOF
TIMESTAMP=${TS}
GIT_COMMIT=${git_commit}
STEPS=${STEPS}
BACKEND=${BACKEND}
COMPILE_MS=${compile_ms}
RUN_MS=${run_ms}
AVG_TOK_S=${avg_tok_s}
LAST_TOK_S=${last_tok_s}
SPEEDUP_PCT=${speedup_pct}
RUN_DIR=${RUN_DIR}
RUN_LOG=${RUN_LOG}
HELIOS_WG_SIZE=${HELIOS_WG_SIZE}
LR=${LR}
GRAD_CLIP=${GRAD_CLIP}
STATUS=${status_label}
EOF

echo "[bench] run status: ${status_label}"
echo "[bench] run time: ${run_ms} ms"
echo "[bench] avg tok/s: ${avg_tok_s}"
echo "[bench] last tok/s: ${last_tok_s}"
echo "[bench] speedup vs previous: ${speedup_pct}%"
echo "[bench] summary: ${SUMMARY_FILE}"

if [[ "$run_status" -ne 0 ]]; then
  echo "[bench] last 40 log lines:"
  tail -n 40 "$RUN_LOG"
  exit "$run_status"
fi
