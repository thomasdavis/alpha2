#!/usr/bin/env bash
# Continuously pull a RunPod training directory and prove metrics are advancing.
# Auto-termination is opt-in: set RUNPOD_POD_ID and TERMINATE_ON_STALL=1.

set -euo pipefail

host=${1:?usage: runpod_run_guard.sh HOST PORT REMOTE_RUN_DIR LOCAL_RUN_DIR [INTERVAL_SECONDS] [STALE_SECONDS]}
port=${2:?port required}
remote_run=${3:?remote run directory required}
local_run=${4:?local run directory required}
interval=${5:-300}
stale_seconds=${6:-1800}
ssh_key=${RUNPOD_SSH_KEY:-/home/ajax/.runpod/ssh/runpodctl-ssh-key}
pod_id=${RUNPOD_POD_ID:-}
terminate_on_stall=${TERMINATE_ON_STALL:-0}
remote_keep_checkpoints=${REMOTE_KEEP_CHECKPOINTS:-0}
guard_once=${RUNPOD_GUARD_ONCE:-0}

[[ $host =~ ^[A-Za-z0-9.-]+$ ]] || { echo "invalid host: $host" >&2; exit 2; }
[[ $port =~ ^[0-9]+$ ]] || { echo "invalid port: $port" >&2; exit 2; }
[[ $remote_run =~ ^/workspace/alpha2/runs/[A-Za-z0-9._/-]+$ ]] || {
  echo "remote run must be a concrete path under /workspace/alpha2/runs" >&2; exit 2;
}
[[ $local_run =~ ^/mnt/donto-data/alpha-runs/[A-Za-z0-9._/-]+$ ]] || {
  echo "local run must be a concrete path under /mnt/donto-data/alpha-runs" >&2; exit 2;
}
[[ $interval =~ ^[0-9]+$ && $interval -ge 10 ]] || { echo "interval must be >= 10 seconds" >&2; exit 2; }
[[ $stale_seconds =~ ^[0-9]+$ && $stale_seconds -ge 60 ]] || { echo "stale timeout must be >= 60 seconds" >&2; exit 2; }
[[ $remote_keep_checkpoints =~ ^[0-9]+$ ]] || { echo "REMOTE_KEEP_CHECKPOINTS must be an integer" >&2; exit 2; }
if (( remote_keep_checkpoints == 1 )); then
  echo "REMOTE_KEEP_CHECKPOINTS must be 0 (disabled) or at least 2" >&2
  exit 2
fi
[[ $guard_once == 0 || $guard_once == 1 ]] || { echo "RUNPOD_GUARD_ONCE must be 0 or 1" >&2; exit 2; }
[[ -f $ssh_key ]] || { echo "SSH key missing: $ssh_key" >&2; exit 2; }
if [[ $terminate_on_stall == 1 && ! $pod_id =~ ^[A-Za-z0-9]+$ ]]; then
  echo "TERMINATE_ON_STALL=1 requires a concrete RUNPOD_POD_ID" >&2
  exit 2
fi

mkdir -p "$local_run"
guard_log="$local_run/puller.log"
ssh_opts=(-i "$ssh_key" -p "$port" -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=no)
rsync_ssh="ssh -i $ssh_key -p $port -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=no"

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$guard_log"
}

pull_once() {
  nice -n 10 ionice -c2 -n7 rsync -az --partial \
    -e "$rsync_ssh" "root@$host:$remote_run/" "$local_run/"
}

prune_remote_checkpoints() {
  (( remote_keep_checkpoints >= 2 )) || return 0
  local remote_listing
  if ! remote_listing=$(ssh "${ssh_opts[@]}" "root@$host" \
    "find '$remote_run' -maxdepth 1 -type f -name 'checkpoint-[0-9]*.json' -printf '%f\\n' | sort -V"); then
    log "WARNING could not list remote checkpoints; nothing pruned"
    return 1
  fi
  local -a checkpoints=()
  mapfile -t checkpoints <<<"$remote_listing"
  local remove_count=$((${#checkpoints[@]} - remote_keep_checkpoints))
  (( remove_count > 0 )) || return 0
  for ((index = 0; index < remove_count; index++)); do
    local base=${checkpoints[$index]}
    [[ $base =~ ^checkpoint-[0-9]+\.json$ ]] || {
      log "WARNING refusing unexpected remote checkpoint name: $base"
      return 1
    }
    local local_file="$local_run/$base"
    if [[ ! -f $local_file ]]; then
      log "WARNING local mirror missing $base; remote copy retained"
      return 1
    fi
    local remote_record remote_size remote_sha local_size local_sha
    if ! remote_record=$(ssh "${ssh_opts[@]}" "root@$host" \
      "stat -c %s '$remote_run/$base'; sha256sum '$remote_run/$base' | awk '{print \$1}'"); then
      log "WARNING could not hash remote $base; remote copy retained"
      return 1
    fi
    remote_size=$(sed -n '1p' <<<"$remote_record")
    remote_sha=$(sed -n '2p' <<<"$remote_record")
    local_size=$(stat -c %s "$local_file")
    local_sha=$(sha256sum "$local_file" | awk '{print $1}')
    if [[ $remote_size != "$local_size" || $remote_sha != "$local_sha" ]]; then
      log "WARNING mirror verification failed for $base; remote copy retained"
      return 1
    fi
    ssh "${ssh_opts[@]}" "root@$host" "rm -- '$remote_run/$base'"
    log "pruned remote $base after size+sha256 verified local mirror"
  done
}

final_pull() {
  if pull_once; then
    prune_remote_checkpoints || true
    log "final pull complete"
  else
    log "WARNING final pull failed"
  fi
}

last_rows=-1
last_progress_epoch=$(date +%s)
log "guard start remote=$host:$port:$remote_run local=$local_run interval=${interval}s stale=${stale_seconds}s"

while true; do
  if ! pull_once; then
    log "WARNING rsync failed; connectivity failure is not treated as a training stall"
    sleep "$interval"
    continue
  fi
  prune_remote_checkpoints || true

  if ! status=$(ssh "${ssh_opts[@]}" "root@$host" "
    metrics='$remote_run/metrics.jsonl'
    if test -f \"\$metrics\"; then rows=\$(wc -l < \"\$metrics\"); bytes=\$(stat -c %s \"\$metrics\"); else rows=0; bytes=0; fi
    pid=\$(ps -eo pid=,comm=,args= | awk '\$2 == \"node\" && index(\$0, \"apps/cli/dist/main.js train\") { print \$1; exit }')
    rss=0
    if test -n \"\$pid\"; then rss=\$(ps -o rss= -p \"\$pid\" | tr -d ' '); fi
    printf '%s\\t%s\\t%s\\t%s\\n' \"\$rows\" \"\$bytes\" \"\${pid:-0}\" \"\${rss:-0}\"
  "); then
    log "WARNING status SSH failed; connectivity failure is not treated as a training stall"
    sleep "$interval"
    continue
  fi

  IFS=$'\t' read -r rows bytes train_pid rss_kb <<<"$status"
  now=$(date +%s)
  if (( rows > last_rows )); then
    last_rows=$rows
    last_progress_epoch=$now
  fi
  stale_for=$((now - last_progress_epoch))
  log "metrics_rows=$rows metrics_bytes=$bytes train_pid=$train_pid rss_kb=$rss_kb stale_for=${stale_for}s"

  if [[ $guard_once == 1 ]]; then
    log "one-shot guard completed"
    exit 0
  fi

  if [[ $train_pid == 0 ]]; then
    log "remote training process ended; performing final pull and exiting"
    final_pull
    exit 0
  fi
  if (( stale_for >= stale_seconds )); then
    log "ERROR metrics have not advanced for ${stale_for}s while PID $train_pid is alive"
    final_pull
    if [[ $terminate_on_stall == 1 ]]; then
      log "terminating pod $pod_id after verified metric stall"
      runpodctl remove pod "$pod_id"
    else
      log "pod left running because TERMINATE_ON_STALL is not enabled"
    fi
    exit 3
  fi
  sleep "$interval"
done
