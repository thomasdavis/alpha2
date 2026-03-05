#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/home/ajax/alpha-repo}"
LOOP_SCRIPT="${2:-$ROOT/super_chat_small_chunk_loop.sh}"
CHECK_EVERY="${CHECK_EVERY:-30}"
STALE_MINUTES="${STALE_MINUTES:-20}"
LOG="${WATCHDOG_LOG:-$ROOT/train_super_chat_watchdog.log}"

mkdir -p "$ROOT"

log(){
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"
}

is_loop_alive(){
  if [[ ! -f "$ROOT/train.loop.pid" ]]; then
    return 1
  fi
  local pid
  pid="$(cat "$ROOT/train.loop.pid" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 1
  ps -p "$pid" -o cmd= 2>/dev/null | grep -q "super_chat_small_chunk_loop.sh"
}

latest_ckpt_epoch(){
  local latest
  latest="$(ls -1 "$ROOT"/runs/super_chat_small_20260305_035247/checkpoint-*.json 2>/dev/null | sort -V | tail -n1 || true)"
  if [[ -z "$latest" ]]; then
    echo 0
    return
  fi
  stat -c %Y "$latest" 2>/dev/null || echo 0
}

start_loop(){
  if [[ ! -x "$LOOP_SCRIPT" ]]; then
    chmod +x "$LOOP_SCRIPT" || true
  fi
  (cd "$ROOT" && nohup bash "$LOOP_SCRIPT" >> train_super_chat_small_chunk_loop.nohup.log 2>&1 & echo $! > train.loop.pid)
  sleep 2
  local pid
  pid="$(cat "$ROOT/train.loop.pid" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && ps -p "$pid" >/dev/null 2>&1; then
    log "loop-started pid=$pid"
  else
    log "loop-start-failed"
  fi
}

log "watchdog-start root=$ROOT loop_script=$LOOP_SCRIPT check_every=${CHECK_EVERY}s stale_minutes=${STALE_MINUTES}"
last_ckpt_epoch="$(latest_ckpt_epoch)"

while true; do
  if ! is_loop_alive; then
    log "loop-not-alive; restarting"
    start_loop
  fi

  current_ckpt_epoch="$(latest_ckpt_epoch)"
  now_epoch="$(date +%s)"

  if [[ "$current_ckpt_epoch" -gt 0 ]]; then
    if [[ "$current_ckpt_epoch" -gt "$last_ckpt_epoch" ]]; then
      last_ckpt_epoch="$current_ckpt_epoch"
      log "checkpoint-progress epoch=$current_ckpt_epoch"
    else
      age=$(( now_epoch - current_ckpt_epoch ))
      stale_limit=$(( STALE_MINUTES * 60 ))
      if (( age > stale_limit )); then
        log "checkpoint-stale age=${age}s (> ${stale_limit}s); restarting loop"
        pkill -f super_chat_small_chunk_loop.sh || true
        sleep 1
        start_loop
      fi
    fi
  fi

  sleep "$CHECK_EVERY"
done
