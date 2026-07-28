#!/usr/bin/env bash
# Wait for a contracted SFT run, finalize it remotely, mirror+verify all artifacts, then remove its pod.

set -euo pipefail

host=${1:?usage: runpod_sft_terminal_watch.sh HOST PORT POD_ID REMOTE_RUN LOCAL_RUN SOURCE_COMMIT [INTERVAL]}
port=${2:?SSH port required}
pod_id=${3:?pod ID required}
remote_run=${4:?remote run required}
local_run=${5:?local run required}
source_commit=${6:?source commit required}
interval=${7:-60}

[[ $host =~ ^[A-Za-z0-9.-]+$ ]] || { echo "invalid host: $host" >&2; exit 2; }
[[ $port =~ ^[0-9]+$ ]] || { echo "invalid port: $port" >&2; exit 2; }
[[ $pod_id =~ ^[A-Za-z0-9]+$ ]] || { echo "invalid pod ID: $pod_id" >&2; exit 2; }
[[ $remote_run =~ ^/workspace/alpha2/runs/[A-Za-z0-9._-]+$ ]] || { echo "invalid remote run: $remote_run" >&2; exit 2; }
[[ $local_run =~ ^/mnt/donto-data/alpha-runs/[A-Za-z0-9._/-]+$ ]] || { echo "invalid local run: $local_run" >&2; exit 2; }
[[ $source_commit =~ ^[0-9a-f]{40}$ ]] || { echo "invalid source commit: $source_commit" >&2; exit 2; }
[[ $interval =~ ^[0-9]+$ && $interval -ge 30 ]] || { echo "interval must be >= 30" >&2; exit 2; }
[[ ${RUNPOD_FINALIZER_ONCE:-0} == 0 || ${RUNPOD_FINALIZER_ONCE:-0} == 1 ]] || {
  echo "RUNPOD_FINALIZER_ONCE must be 0 or 1" >&2
  exit 2
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
remote_helper_local="$script_dir/runpod_sft_terminal_finalize_remote.sh"
ssh_key=${RUNPOD_SSH_KEY:-/home/ajax/.runpod/ssh/runpodctl-ssh-key}
remote_repo=/workspace/alpha2
selection=/workspace/alpha2/runs/sft-lr-sweep-analysis-c333bf2-20260728.json
base_eval=/workspace/alpha2/runs/frozen-eval-base-flagship-20260728
frozen_root=/runpod/data/frozen-eval-v1
python_bin=/workspace/alpha2-hf-verify-venv/bin/python
remote_helper="/workspace/alpha2-sft-terminal-finalize-$pod_id.sh"
ssh_opts=(-i "$ssh_key" -p "$port" -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=no)
scp_opts=(-i "$ssh_key" -P "$port" -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=no)
rsync_ssh="ssh -i $ssh_key -p $port -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=no"

[[ -f $remote_helper_local ]] || { echo "remote helper missing: $remote_helper_local" >&2; exit 2; }
[[ -f $ssh_key ]] || { echo "SSH key missing: $ssh_key" >&2; exit 2; }
command -v runpodctl >/dev/null || { echo "runpodctl missing" >&2; exit 2; }
command -v rsync >/dev/null || { echo "rsync missing" >&2; exit 2; }

mkdir -p "$local_run"
exec > >(tee -a "$local_run/terminal-finalizer-host.log") 2>&1

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

log "terminal watcher start pod=$pod_id remote=$host:$port:$remote_run local=$local_run"
scp -q "${scp_opts[@]}" "$remote_helper_local" "root@$host:$remote_helper"
ssh "${ssh_opts[@]}" "root@$host" bash "$remote_helper" preflight \
  "$remote_repo" "$remote_run" "$selection" "$base_eval" "$frozen_root" "$python_bin" "$source_commit"

if [[ ${RUNPOD_FINALIZER_ONCE:-0} == 1 ]]; then
  log "one-shot preflight complete"
  exit 0
fi

consecutive_ssh_failures=0
while true; do
  if ! state=$(ssh "${ssh_opts[@]}" "root@$host" "
    run='$remote_run'
    metrics=\"\$run/metrics.jsonl\"
    rows=0; test -f \"\$metrics\" && rows=\$(wc -l < \"\$metrics\")
    pid=\$(ps -eo pid=,comm=,args= | awk -v run=\"\$run\" '\$2 == \"node\" && index(\$0, \"apps/cli/dist/main.js train\") && index(\$0, run) { print \$1; exit }')
    checkpoint=0; test -f \"\$run/checkpoint-30322.json\" && checkpoint=1
    printf '%s\\t%s\\t%s\\n' \"\$rows\" \"\${pid:-0}\" \"\$checkpoint\"
  "); then
    consecutive_ssh_failures=$((consecutive_ssh_failures + 1))
    log "WARNING terminal watcher SSH failure count=$consecutive_ssh_failures"
    if (( consecutive_ssh_failures >= 10 )); then
      log "ERROR repeated SSH failure; pod left untouched"
      exit 3
    fi
    sleep "$interval"
    continue
  fi
  consecutive_ssh_failures=0
  IFS=$'\t' read -r rows train_pid checkpoint <<<"$state"
  log "terminal watcher rows=$rows train_pid=$train_pid terminal_checkpoint=$checkpoint"
  if [[ $train_pid == 0 ]]; then
    if [[ $rows == 30322 && $checkpoint == 1 ]]; then
      break
    fi
    log "ERROR trainer ended before exact terminal state; pod left untouched"
    exit 4
  fi
  sleep "$interval"
done

log "exact terminal state observed; starting remote finalization"
ssh "${ssh_opts[@]}" "root@$host" bash "$remote_helper" finalize \
  "$remote_repo" "$remote_run" "$selection" "$base_eval" "$frozen_root" "$python_bin" "$source_commit"

log "remote finalization complete; mirroring run"
nice -n 10 ionice -c2 -n7 rsync -az --partial -e "$rsync_ssh" \
  "root@$host:$remote_run/" "$local_run/"

[[ -f $local_run/terminal-artifact-sha256.txt ]] || {
  log "ERROR mirrored artifact manifest missing; pod left untouched"
  exit 5
}
log "verifying every mirrored artifact against remote manifest"
(
  cd "$local_run"
  nice -n 19 ionice -c3 sha256sum -c terminal-artifact-sha256.txt
)
node -e '
  const fs = require("fs");
  const value = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
  if (value.result !== "PASS" || !["PASS", "FAIL"].includes(value.machine_d3?.result) ||
      value.semantic_review !== "PENDING_HUMAN_REVIEW") process.exit(1);
' "$local_run/terminal-finalizer-status.json"

log "all unique terminal artifacts are local and hash-verified; removing scoped pod $pod_id"
runpodctl remove pod "$pod_id"
log "scoped pod removal requested successfully; semantic review and any chat publication remain manual"
