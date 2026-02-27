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
LR="${LR:-0.00003}"
GRAD_CLIP="${GRAD_CLIP:-0}"
COMPILE_TIMEOUT_SEC="${COMPILE_TIMEOUT_SEC:-60}"
COMPILE_RETRIES="${COMPILE_RETRIES:-2}"
RUN_RETRIES="${RUN_RETRIES:-2}"
SKIP_COMPILE_IF_FRESH="${SKIP_COMPILE_IF_FRESH:-1}"

timestamp_utc() {
  date -u +"%Y%m%dT%H%M%SZ"
}

now_ms() {
  date +%s%3N
}

TS="$(timestamp_utc)"
RUN_DIR_BASE="runs/compiled-binary-${BACKEND}-${TS}"
PERF_DIR="perf"
HISTORY_FILE="${PERF_DIR}/compiled-loop-history.csv"
COMPILE_LOG="${PERF_DIR}/compile-${TS}.log"
SUMMARY_FILE="${PERF_DIR}/last-benchmark.env"

mkdir -p "$PERF_DIR"

if [[ ! -f "$HISTORY_FILE" ]]; then
  echo "timestamp,git_commit,steps,backend,compile_ms,run_ms,avg_tok_s,last_tok_s,speedup_pct,run_dir,log_file,status" > "$HISTORY_FILE"
fi

echo "[bench] compile start (steps=${STEPS}, backend=${BACKEND})"
compile_start_ms="$(now_ms)"
compile_ok="0"
attempts="$((COMPILE_RETRIES + 1))"
: >"$COMPILE_LOG"
should_compile="1"
if [[ "$SKIP_COMPILE_IF_FRESH" == "1" && -x ".bun-out/alpha" && -f ".bun-out/helios_vk.node" ]]; then
  latest_src_mtime="$(
    find apps/cli/src packages scripts \
      -type f \
      \( -name '*.ts' -o -name '*.js' -o -name '*.mjs' -o -name '*.c' -o -name '*.h' \) \
      -printf '%T@\n' | awk '($1 > max) { max = $1 } END { if (max == "") max = 0; print max }'
  )"
  alpha_mtime="$(stat -c '%Y' .bun-out/alpha 2>/dev/null || echo 0)"
  native_mtime="$(stat -c '%Y' .bun-out/helios_vk.node 2>/dev/null || echo 0)"
  latest_src_sec="${latest_src_mtime%%.*}"
  latest_src_sec="${latest_src_sec:-0}"
  if [[ "$alpha_mtime" -ge "$latest_src_sec" && "$native_mtime" -ge "$latest_src_sec" ]]; then
    should_compile="0"
    echo "[bench] compile skip: existing .bun-out artifacts are fresh" | tee -a "$COMPILE_LOG"
  fi
fi

if [[ "$should_compile" == "1" ]]; then
  for attempt in $(seq 1 "$attempts"); do
    echo "[bench] compile attempt ${attempt}/${attempts}" | tee -a "$COMPILE_LOG"
    if timeout "${COMPILE_TIMEOUT_SEC}" npm run bun:compile >>"$COMPILE_LOG" 2>&1; then
      compile_ok="1"
      break
    fi
    if [[ "$attempt" -lt "$attempts" ]]; then
      echo "[bench] compile attempt ${attempt} failed (timeout ${COMPILE_TIMEOUT_SEC}s) — retrying..." | tee -a "$COMPILE_LOG"
    fi
  done
else
  compile_ok="1"
fi
if [[ "$compile_ok" != "1" ]]; then
  compile_end_ms="$(now_ms)"
  compile_ms="$((compile_end_ms - compile_start_ms))"
  git_commit="$(git rev-parse --short HEAD)"
  echo "${TS},${git_commit},${STEPS},${BACKEND},${compile_ms},0,0.000,0.000,0.00,${RUN_DIR_BASE},${COMPILE_LOG},compile_failed" >> "$HISTORY_FILE"
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
RUN_DIR=${RUN_DIR_BASE}
RUN_LOG=${COMPILE_LOG}
HELIOS_WG_SIZE=${HELIOS_WG_SIZE}
LR=${LR}
GRAD_CLIP=${GRAD_CLIP}
STATUS=compile_failed
EOF
  echo "[bench] compile failed after ${attempts} attempt(s), timeout ${COMPILE_TIMEOUT_SEC}s each"
  echo "[bench] compile log tail:"
  tail -n 40 "$COMPILE_LOG"
  exit 1
fi
compile_end_ms="$(now_ms)"
compile_ms="$((compile_end_ms - compile_start_ms))"
echo "[bench] compile done in ${compile_ms} ms"

selected_run_status=1
selected_status_label="failed"
selected_run_ms=0
selected_avg_tok_s="0.000"
selected_last_tok_s="0.000"
selected_run_dir="${RUN_DIR_BASE}"
selected_run_log="${PERF_DIR}/run-${TS}.log"

run_attempts="$((RUN_RETRIES + 1))"
for run_try in $(seq 1 "$run_attempts"); do
  if [[ "$run_try" -eq 1 ]]; then
    suffix=""
  else
    suffix="-try${run_try}"
  fi
  run_dir="${RUN_DIR_BASE}${suffix}"
  run_log="${PERF_DIR}/run-${TS}${suffix}.log"
  metrics_file="${run_dir}/metrics.jsonl"

  echo "[bench] run attempt ${run_try}/${run_attempts} start (log: ${run_log}, helios_wg=${HELIOS_WG_SIZE})"
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
    --runDir="${run_dir}" >"$run_log" 2>&1
  run_status="$?"
  set -e
  run_end_ms="$(now_ms)"
  run_ms="$((run_end_ms - run_start_ms))"

  avg_tok_s="0.000"
  last_tok_s="0.000"
  if [[ -f "$metrics_file" ]]; then
    tok_series="$(grep -Eo '"tokens_per_sec":[0-9.+-eE]+' "$metrics_file" | awk -F: '{print $2}' || true)"
  else
    tok_series=""
  fi
  if [[ -z "$tok_series" ]]; then
    tok_series="$(grep -Eo '([0-9]+([.][0-9]+)?) tok/s' "$run_log" | awk '{print $1}' || true)"
  fi
  if [[ -n "$tok_series" ]]; then
    avg_tok_s="$(echo "$tok_series" | awk 'BEGIN{s=0;n=0} {s+=$1;n++} END{if(n>0) printf "%.3f", s/n; else printf "0.000"}')"
    last_tok_s="$(echo "$tok_series" | awk 'END{if(NR>0) printf "%.3f", $1; else printf "0.000"}')"
  fi

  status_label="ok"
  if [[ "$run_status" -ne 0 ]]; then
    status_label="failed"
  elif rg -q 'loss=NaN|Inf/NaN|smoke_test: FAIL' "$run_log"; then
    status_label="unstable"
  fi

  selected_run_status="$run_status"
  selected_status_label="$status_label"
  selected_run_ms="$run_ms"
  selected_avg_tok_s="$avg_tok_s"
  selected_last_tok_s="$last_tok_s"
  selected_run_dir="$run_dir"
  selected_run_log="$run_log"

  if [[ "$status_label" == "ok" ]]; then
    break
  fi
  if [[ "$run_try" -lt "$run_attempts" ]]; then
    echo "[bench] run attempt ${run_try} ended with status=${status_label} — retrying..."
  fi
done

run_status="$selected_run_status"
status_label="$selected_status_label"
run_ms="$selected_run_ms"
avg_tok_s="$selected_avg_tok_s"
last_tok_s="$selected_last_tok_s"
RUN_DIR="$selected_run_dir"
RUN_LOG="$selected_run_log"

prev_avg_tok_s="$(awk -F, -v steps="$STEPS" -v backend="$BACKEND" 'NR>1 && $3==steps && $4==backend && $12=="ok" {v=$7} END{print v}' "$HISTORY_FILE")"

speedup_pct="0.00"
if [[ -n "$prev_avg_tok_s" && "$prev_avg_tok_s" != "0" && "$prev_avg_tok_s" != "0.000" ]]; then
  speedup_pct="$(awk -v cur="$avg_tok_s" -v prev="$prev_avg_tok_s" 'BEGIN{printf "%.2f", ((cur-prev)/prev)*100}')"
fi

git_commit="$(git rev-parse --short HEAD)"

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
