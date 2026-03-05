#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

INSTANCE="${1:-alpha-bench-l4-coopdbg-20260228084511}"
OUT_MD="${2:-docs/l4-historic-v2-inference-checks.md}"
REMOTE_ROOT="${3:-/home/ajax/alpha-repo}"
INTERVAL="${4:-200}"
TITLE="${INFERENCE_CHECKS_TITLE:-$(basename "$OUT_MD" .md | tr '-' ' ')}"

# Keep prompts simple and deterministic so diffs are easy to review.
PROMPTS=(
  "The "
  "Once upon a time"
  "He walked into"
)

strip_ansi() {
  sed -E 's/\x1B\[[0-9;]*[A-Za-z]//g'
}

fleet_run_raw() {
  local cmd="$1"
  timeout "${FLEET_TIMEOUT:-45s}" node apps/cli/dist/main.js fleet run "$INSTANCE" -- "$cmd"
}

fleet_capture() {
  local cmd="$1"
  fleet_run_raw "$cmd" | strip_ansi
}

fleet_last_line() {
  local cmd="$1"
  fleet_capture "$cmd" | awk 'NF{line=$0} END{print line}'
}

safe_key() {
  echo "$1" | tr '/: ' '___' | tr -cd 'A-Za-z0-9._-'
}

decode_b64() {
  printf '%s' "$1" | base64 --decode
}

post_discord_samples() {
  local webhook="$1"
  local run_id="$2"
  local step="$3"
  local decoded_tsv="$4"
  node - "$webhook" "$run_id" "$step" "$decoded_tsv" <<'NODE'
const [webhook, runId, step, sampleFile] = process.argv.slice(2);
const fs = require("node:fs");

const raw = fs.readFileSync(sampleFile, "utf8").trim();
if (!raw) process.exit(0);
const rows = raw.split("\n").map((line) => {
  const tab = line.indexOf("\t");
  if (tab < 0) return null;
  return { prompt: line.slice(0, tab), output: line.slice(tab + 1) };
}).filter(Boolean);
if (!rows.length) process.exit(0);

const fields = rows.slice(0, 3).map((r, i) => ({
  name: `Prompt ${i + 1}: "${r.prompt}"`,
  value: `\`\`\`\n${r.output.slice(0, 300)}${r.output.length > 300 ? "..." : ""}\n\`\`\``,
}));

const payload = {
  embeds: [{
    title: `📝 Inference Samples (Step ${step})`,
    color: 0xff9800,
    description: `Run \`${runId}\``,
    fields,
    timestamp: new Date().toISOString(),
  }],
};

fetch(webhook, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload),
  signal: AbortSignal.timeout(10000),
}).then(() => process.exit(0)).catch(() => process.exit(0));
NODE
}

log_path="$(fleet_last_line "cat ${REMOTE_ROOT}/train.log.path 2>/dev/null || true")"
run_dir="$(fleet_last_line "cat ${REMOTE_ROOT}/train.run.path 2>/dev/null || true")"
pid_line="$(fleet_last_line "cat ${REMOTE_ROOT}/train.pid 2>/dev/null || true")"
proc_info="$(
  fleet_capture "ps -p \$(cat ${REMOTE_ROOT}/train.pid 2>/dev/null) -o pid,etime,pcpu,pmem,cmd --no-headers 2>/dev/null || true" \
    | awk '/^[[:space:]]*[0-9]+[[:space:]]/ {print; exit}'
)"

if [[ -z "${run_dir}" ]]; then
  echo "No remote train.run.path found on ${INSTANCE}" >&2
  exit 1
fi

if [[ -z "${log_path}" ]]; then
  log_path="unknown"
fi

if [[ -n "${proc_info}" ]]; then
  train_status="running"
else
  train_status="not_running"
fi

tmp_log="$(mktemp)"
tmp_ckpts="$(mktemp)"
tmp_sorted="$(mktemp)"
trap 'rm -f "$tmp_log" "$tmp_ckpts" "$tmp_sorted"' EXIT

fleet_capture "cat '${log_path}' 2>/dev/null || true" > "$tmp_log"
run_id="$(
  awk -F': ' '/^run_id:/ {print $2; exit}' "$tmp_log" | sed 's/[[:space:]]*$//'
)"
if [[ -z "${run_id}" ]]; then
  run_id="unknown"
fi

# Persist sampled outputs so repeated updates only compute new checkpoints.
state_dir="perf/inference-check-cache"
mkdir -p "$state_dir"
state_key="$(safe_key "${INSTANCE}_${run_dir}")"
state_file="${state_dir}/${state_key}.tsv"
posted_steps_file="${state_dir}/${state_key}.discord-posted"
touch "$state_file"
touch "$posted_steps_file"

if [[ -z "${DISCORD_WEBHOOK_URL:-}" && -f ".env.local" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env.local"
  set +a
fi
discord_webhook="${DISCORD_WEBHOOK_URL:-}"

fleet_capture "cd '${REMOTE_ROOT}' && ls -1 '${run_dir}'/checkpoint-*.json 2>/dev/null || true" \
  | awk '/checkpoint-[0-9]+\.json$/ {print}' > "$tmp_ckpts"

declare -A step_counts=()
declare -A existing_prompt=()
declare -A posted_steps=()

while IFS=$'\t' read -r step _prompt_b64 _out_b64; do
  [[ -z "${step}" ]] && continue
  prompt_decoded="$(decode_b64 "${_prompt_b64}")"
  existing_prompt["${step}|${prompt_decoded}"]=1
  step_counts["$step"]="$(( ${step_counts[$step]:-0} + 1 ))"
done < "$state_file"

while IFS= read -r posted_step; do
  [[ -z "$posted_step" ]] && continue
  posted_steps["$posted_step"]=1
done < "$posted_steps_file"

prompt_target="${#PROMPTS[@]}"

while IFS= read -r ckpt; do
  [[ -z "$ckpt" ]] && continue
  if [[ "$ckpt" =~ checkpoint-([0-9]+)\.json$ ]]; then
    step="${BASH_REMATCH[1]}"
  else
    continue
  fi
  if (( step % INTERVAL != 0 )); then
    continue
  fi
  step_new_samples=0
  for prompt in "${PROMPTS[@]}"; do
    key="${step}|${prompt}"
    if [[ -n "${existing_prompt[$key]:-}" ]]; then
      continue
    fi
    sample_out="$(
      fleet_capture "cd '${REMOTE_ROOT}' && node apps/cli/dist/main.js sample --checkpoint='${ckpt}' --backend=cpu_ref --steps=40 --prompt='${prompt}' --temp=0.8 --topk=40 --topp=1.0" \
        | awk '
            /^---$/ { if (!started) { started=1; next } }
            started {
              if ($0 ~ /^--- stats ---$/) exit
              print
            }
          ' \
        | sed '/^[[:space:]]*$/d'
    )"
    if [[ -z "$sample_out" ]]; then
      sample_out="(no output captured)"
    fi
    prompt_b64="$(printf '%s' "$prompt" | base64 -w0)"
    out_b64="$(printf '%s' "$sample_out" | base64 -w0)"
    printf '%s\t%s\t%s\n' "$step" "$prompt_b64" "$out_b64" >> "$state_file"
    existing_prompt["$key"]=1
    step_counts["$step"]="$(( ${step_counts[$step]:-0} + 1 ))"
    step_new_samples=1
  done

  if [[ "$step_new_samples" -eq 1 ]] && [[ -n "$discord_webhook" ]] && [[ -z "${posted_steps[$step]:-}" ]] && (( ${step_counts[$step]:-0} >= prompt_target )); then
    step_tmp="$(mktemp)"
    grep -E "^${step}"$'\t' "$state_file" | while IFS=$'\t' read -r s p_b64 o_b64; do
      p_dec="$(decode_b64 "$p_b64")"
      o_dec="$(decode_b64 "$o_b64")"
      printf '%s\t%s\n' "$p_dec" "$o_dec"
    done > "$step_tmp"
    post_discord_samples "$discord_webhook" "$run_id" "$step" "$step_tmp"
    rm -f "$step_tmp"
    echo "$step" >> "$posted_steps_file"
    posted_steps["$step"]=1
  fi
done < "$tmp_ckpts"

# Backfill Discord posts for any fully-captured steps not yet posted.
if [[ -n "$discord_webhook" ]]; then
  while read -r step count; do
    [[ -z "$step" ]] && continue
    if (( count < prompt_target )); then
      continue
    fi
    if [[ -n "${posted_steps[$step]:-}" ]]; then
      continue
    fi
    step_tmp="$(mktemp)"
    grep -E "^${step}"$'\t' "$state_file" | while IFS=$'\t' read -r s p_b64 o_b64; do
      p_dec="$(decode_b64 "$p_b64")"
      o_dec="$(decode_b64 "$o_b64")"
      printf '%s\t%s\n' "$p_dec" "$o_dec"
    done > "$step_tmp"
    post_discord_samples "$discord_webhook" "$run_id" "$step" "$step_tmp"
    rm -f "$step_tmp"
    echo "$step" >> "$posted_steps_file"
    posted_steps["$step"]=1
  done < <(awk -F $'\t' '{ if ($1 != "") c[$1]++ } END { for (s in c) print s, c[s] }' "$state_file" | sort -n -k1,1)
fi

sort -t $'\t' -k1,1n "$state_file" > "$tmp_sorted"

mkdir -p "$(dirname "$OUT_MD")"
{
  echo "# ${TITLE} (Every ${INTERVAL} Steps)"
  echo
  echo "- Updated (UTC): $(date -u '+%Y-%m-%d %H:%M:%S')"
  echo "- Fleet instance: \`$INSTANCE\`"
  echo "- Remote run id: \`$run_id\`"
  echo "- Remote run dir: \`$run_dir\`"
  echo "- Remote log: \`$log_path\`"
  echo "- Train PID: \`${pid_line:-unknown}\`"
  echo "- Status: \`$train_status\`"
  if [[ -n "${proc_info}" ]]; then
    echo "- Process: \`$proc_info\`"
  fi
  echo

  if [[ ! -s "$tmp_sorted" ]]; then
    echo "_No checkpoint-based inference entries captured yet._"
    echo
    echo "Expected first entry once \`checkpoint-${INTERVAL}.json\` is available."
  else
    last_step=""
    while IFS=$'\t' read -r step prompt_b64 out_b64; do
      [[ -z "$step" ]] && continue
      prompt="$(decode_b64 "$prompt_b64")"
      output="$(decode_b64 "$out_b64")"
      if [[ "$step" != "$last_step" ]]; then
        echo "## Step $step"
        echo
        last_step="$step"
      fi
      echo "### Prompt"
      echo "\`$prompt\`"
      echo
      echo "### Output"
      echo '```text'
      printf '%s\n' "$output"
      echo '```'
      echo
    done < "$tmp_sorted"
  fi
} > "$OUT_MD"

echo "Updated $OUT_MD"
