#!/usr/bin/env bash
# NanoChat restart-based training loop.
#
# Passes --iters=TOTAL_STEPS so the LR schedule (warmup+cosine) spans the
# full run.  Kills the process every SEGMENT_STEPS to avoid GPU buffer leak
# hitting the Vulkan purge threshold (~6000 live allocs) which corrupts memory.
# Checkpoints at evalInterval=50 ensure minimal lost work.
#
# Usage: bash scripts/nanochat-train-loop.sh [TOTAL_STEPS] [SEGMENT_STEPS]
#
# Environment:
#   NANOCHAT_DATA  - data file (default: data/soda-chat.txt)

cd "$(dirname "$0")/.."

TOTAL_STEPS="${1:-20000}"
SEGMENT_STEPS="${2:-300}"
DATA="${NANOCHAT_DATA:-data/soda-chat.txt}"

# Disable spike LR backoff — gradClip handles gradient clipping,
# and the permanent LR scale reduction prevents proper warmup
export ALPHA_SPIKE_LR_BACKOFF=1.0

# Load .env.local if present
if [ -f .env.local ]; then
  set -a
  source .env.local
  set +a
fi

RUN_DIR=""
RESUME=""
COMPLETED=0

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

find_latest_checkpoint() {
  local dir="$1"
  ls -1t "$dir"/checkpoint-*.json 2>/dev/null | head -1
}

# Auto-detect existing checkpoint to resume from (find latest checkpoint across all run dirs)
for CANDIDATE_DIR in $(ls -1td runs/soda_chat_* 2>/dev/null); do
  AUTO_CKPT=$(find_latest_checkpoint "$CANDIDATE_DIR") || true
  if [ -n "$AUTO_CKPT" ]; then
    STEP_NUM=$(echo "$AUTO_CKPT" | grep -oP 'checkpoint-\K[0-9]+') || true
    if [ -n "$STEP_NUM" ] && [ "$STEP_NUM" -gt 0 ]; then
      RUN_DIR="$CANDIDATE_DIR"
      RESUME="$AUTO_CKPT"
      COMPLETED="$STEP_NUM"
      log "Auto-resuming from existing checkpoint: step=$STEP_NUM path=$AUTO_CKPT"
      break
    fi
  fi
done

log "NanoChat training loop: total=$TOTAL_STEPS segment=$SEGMENT_STEPS data=$DATA starting_at=$COMPLETED"

while [ "$COMPLETED" -lt "$TOTAL_STEPS" ]; do
  RESUME_FLAG=""
  RUN_DIR_FLAG=""
  if [ -n "$RESUME" ]; then
    RESUME_FLAG="--resume=$RESUME"
    RUN_DIR=$(dirname "$RESUME")
    RUN_DIR_FLAG="--runDir=$RUN_DIR"
  fi

  KILL_AT=$((COMPLETED + SEGMENT_STEPS))
  log "Segment: step $COMPLETED -> kill at $KILL_AT (total $TOTAL_STEPS)"

  # Run training in background so we can monitor and kill it
  node --expose-gc apps/cli/dist/main.js train \
    --data="$DATA" \
    --domain=nanochat \
    --tokenizer=bpe-chat-4k \
    --backend=helios \
    --iters="$TOTAL_STEPS" \
    --batch=4 \
    --block=512 \
    --dim=512 \
    --heads=8 \
    --layers=16 \
    --activation=swiglu \
    --lr=6e-4 \
    --lrMin=6e-5 \
    --warmupIters=500 \
    --accumSteps=1 \
    --beta2=0.999 \
    --gradClip=1.0 \
    --evalInterval=50 \
    --sampleInterval=99999 \
    --spikeThreshold=0 \
    --packed=true \
    --dropout=0 \
    --checkpoint=true \
    --fp16=false \
    --gpuProfile=none \
    --minGpuSize=1 \
    --postSamples=false \
    --maxDatasetPasses=50 \
    --symbio=false \
    --syncEvery=1 \
    --gcEvery=1 \
    $RESUME_FLAG \
    $RUN_DIR_FLAG \
    2>&1 &
  TRAIN_PID=$!

  # Monitor the process — watch for the kill-at step in the log
  # We pipe through tee so we can see output while also monitoring
  KILLED=false
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 5
    # Find the run directory if we don't have it yet
    if [ -z "$RUN_DIR" ]; then
      RUN_DIR=$(ls -1td runs/soda_chat_* 2>/dev/null | head -1) || true
    fi
    # Check latest checkpoint to see current step
    if [ -n "$RUN_DIR" ]; then
      LATEST_CKPT=$(find_latest_checkpoint "$RUN_DIR") || true
      if [ -n "$LATEST_CKPT" ]; then
        CURRENT_STEP=$(echo "$LATEST_CKPT" | grep -oP 'checkpoint-\K[0-9]+') || true
        if [ -n "$CURRENT_STEP" ] && [ "$CURRENT_STEP" -ge "$KILL_AT" ]; then
          log "Reached step $CURRENT_STEP >= $KILL_AT — killing to reset GPU memory"
          kill "$TRAIN_PID" 2>/dev/null
          KILLED=true
          break
        fi
      fi
    fi
  done

  # Wait for the process to finish
  wait "$TRAIN_PID" 2>/dev/null
  TRAIN_EXIT=$?

  if [ "$TRAIN_EXIT" -eq 0 ] && [ "$KILLED" = false ]; then
    log "Training completed successfully"
    COMPLETED="$TOTAL_STEPS"
    break
  fi

  if [ "$KILLED" = false ]; then
    log "Training exited with code $TRAIN_EXIT"
  fi
  sleep 3

  # Find latest checkpoint
  if [ -z "$RUN_DIR" ]; then
    for CANDIDATE_DIR in $(ls -1td runs/soda_chat_* 2>/dev/null); do
      CKPT=$(find_latest_checkpoint "$CANDIDATE_DIR") || true
      if [ -n "$CKPT" ]; then
        RUN_DIR="$CANDIDATE_DIR"
        break
      fi
    done
  fi

  if [ -n "$RUN_DIR" ]; then
    LATEST_CKPT=$(find_latest_checkpoint "$RUN_DIR") || true
    if [ -n "$LATEST_CKPT" ]; then
      STEP_NUM=$(echo "$LATEST_CKPT" | grep -oP 'checkpoint-\K[0-9]+') || true
      if [ -n "$STEP_NUM" ] && [ "$STEP_NUM" -gt "$COMPLETED" ]; then
        COMPLETED="$STEP_NUM"
        RESUME="$LATEST_CKPT"
        log "Resuming from checkpoint: step=$STEP_NUM path=$LATEST_CKPT"
      else
        log "ERROR: No progress since last restart (step=$COMPLETED, ckpt=$STEP_NUM)"
        sleep 10
      fi
    else
      log "ERROR: No checkpoint found in $RUN_DIR"
      break
    fi
  else
    log "ERROR: No run directory found"
    break
  fi

  log "Progress: $COMPLETED/$TOTAL_STEPS steps"
  sleep 2
done

log "Training complete: $COMPLETED/$TOTAL_STEPS steps"

if [ -n "$RESUME" ]; then
  log "Final checkpoint: $RESUME"
  log ""
  log "To sample from the model:"
  log "  node apps/cli/dist/main.js sample --checkpoint=$RESUME --backend=cpu_ref --slow --prompt='<|user|> Hello! <|assistant|>' --steps=100 --temperature=0.7"
fi
