#!/usr/bin/env bash
# NanoChat training loop — deploys to L4, trains, evaluates, iterates.
#
# Usage:
#   scripts/nanochat-loop.sh [--host=IP] [--iters=N] [--resume=PATH]
#
# This script:
#   1. Syncs code + data to the L4 instance
#   2. Builds the project remotely
#   3. Launches training with reporting enabled
#   4. Monitors progress and downloads checkpoints
#   5. Runs inference samples from latest checkpoint
#
# Environment:
#   NANOCHAT_HOST     - SSH host (default: from fleet.json bench instance)
#   NANOCHAT_ITERS    - Training iterations (default: 20000)
#   NANOCHAT_DATA     - Data file path on remote (default: data/soda-chat.txt)

set -euo pipefail
cd "$(dirname "$0")/.."

# ── Parse args ──────────────────────────────────────────────────────────
HOST="${NANOCHAT_HOST:-136.113.161.152}"
ITERS="${NANOCHAT_ITERS:-20000}"
DATA="${NANOCHAT_DATA:-data/soda-chat.txt}"
RESUME=""
SAMPLE_INTERVAL=500
SSH_KEY="${HOME}/.ssh/google_compute_engine"
SSH_USER="ajax"
REMOTE_DIR="/home/ajax/alpha"
LOCAL_RUNS="runs/nanochat"

for arg in "$@"; do
  case "$arg" in
    --host=*) HOST="${arg#*=}" ;;
    --iters=*) ITERS="${arg#*=}" ;;
    --resume=*) RESUME="${arg#*=}" ;;
    --data=*) DATA="${arg#*=}" ;;
  esac
done

SSH_CMD="ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -i $SSH_KEY ${SSH_USER}@${HOST}"
SCP_CMD="scp -o StrictHostKeyChecking=no -i $SSH_KEY"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

# ── Step 1: Check connectivity ─────────────────────────────────────────
log "Checking L4 instance at $HOST..."
if ! $SSH_CMD "nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader" 2>/dev/null; then
  echo "ERROR: Cannot reach L4 instance at $HOST"
  exit 1
fi
log "L4 instance is reachable."

# ── Step 2: Sync code ──────────────────────────────────────────────────
log "Syncing project to $HOST:$REMOTE_DIR..."
rsync -az --delete \
  --exclude='node_modules' \
  --exclude='.git' \
  --exclude='runs' \
  --exclude='data/concordance*' \
  --exclude='data/soda-chat.txt' \
  --exclude='data/chat.txt' \
  --exclude='data/chat_combined.txt' \
  --exclude='data/books' \
  --exclude='data/bfdb*' \
  --exclude='data/all_tunes*' \
  --exclude='.turbo' \
  --exclude='perf' \
  --exclude='movies' \
  --exclude='third_party' \
  --exclude='symbiogenesis' \
  --exclude='.tmp-venv' \
  -e "ssh -o StrictHostKeyChecking=no -i $SSH_KEY" \
  ./ "${SSH_USER}@${HOST}:${REMOTE_DIR}/"

log "Code synced."

# ── Step 3: Check if SODA data exists on remote, upload if not ────────
log "Checking for SODA data on remote..."
HAS_SODA=$($SSH_CMD "[ -f $REMOTE_DIR/data/soda-chat.txt ] && echo yes || echo no" 2>/dev/null)
if [ "$HAS_SODA" = "no" ]; then
  log "Uploading SODA data (868MB) — this will take a few minutes..."
  rsync -az --progress \
    -e "ssh -o StrictHostKeyChecking=no -i $SSH_KEY" \
    data/soda-chat.txt "${SSH_USER}@${HOST}:${REMOTE_DIR}/data/soda-chat.txt"
  log "SODA data uploaded."
else
  log "SODA data already present on remote."
fi

# ── Step 4: Install deps + build on remote ────────────────────────────
log "Building project on remote..."
$SSH_CMD "cd $REMOTE_DIR && npm install --prefer-offline 2>&1 | tail -3 && npm run build 2>&1 | tail -5"
log "Build complete."

# ── Step 5: Launch training ───────────────────────────────────────────
RESUME_FLAG=""
if [ -n "$RESUME" ]; then
  RESUME_FLAG="--resume=$RESUME"
fi

# Create .env.local on remote for reporting
$SSH_CMD "cat > $REMOTE_DIR/.env.local << 'ENVEOF'
ALPHA_REMOTE_URL=https://alpha.omegaai.dev
ALPHA_REMOTE_SECRET=74c80a29940afdda7ceba133a650eb17f11d4960adde
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/1475248227711324210/XcCPGSSnzF8zbKlINWR1pO6UnmGd1NIG3FlUBmcDvjQMVwEp8oTBSRIDxouG9ieNq6Vo
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd_headless.json
ENVEOF"

log "Starting NanoChat training: iters=$ITERS data=$DATA"
log "Training will report to https://alpha.omegaai.dev"
log "Samples generated every $SAMPLE_INTERVAL steps"
log "─────────────────────────────────────────────────"

# Run training in foreground so we see output
$SSH_CMD "cd $REMOTE_DIR && \
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd_headless.json && \
  source .env.local && \
  NANOCHAT_ITERS=$ITERS NANOCHAT_DATA=$DATA \
  bash scripts/train-nanochat.sh $RESUME_FLAG 2>&1" | tee "runs/nanochat-training-$(date +%Y%m%d_%H%M%S).log"

EXIT_CODE=${PIPESTATUS[0]}

# ── Step 6: Download checkpoints ──────────────────────────────────────
log "Downloading checkpoints from remote..."
mkdir -p "$LOCAL_RUNS"

# Find the latest run directory
LATEST_RUN=$($SSH_CMD "ls -1td $REMOTE_DIR/runs/nanochat-* 2>/dev/null | head -1" 2>/dev/null || true)
if [ -n "$LATEST_RUN" ]; then
  RUN_NAME=$(basename "$LATEST_RUN")
  log "Downloading run: $RUN_NAME"
  rsync -az \
    -e "ssh -o StrictHostKeyChecking=no -i $SSH_KEY" \
    "${SSH_USER}@${HOST}:${LATEST_RUN}/" "$LOCAL_RUNS/$RUN_NAME/"
  log "Checkpoints downloaded to $LOCAL_RUNS/$RUN_NAME"

  # Show latest checkpoint
  LATEST_CKPT=$(ls -1t "$LOCAL_RUNS/$RUN_NAME"/checkpoint-*.json 2>/dev/null | head -1 || true)
  if [ -n "$LATEST_CKPT" ]; then
    log "Latest checkpoint: $LATEST_CKPT"
    log ""
    log "To chat with the model:"
    log "  node apps/cli/dist/main.js sample --checkpoint=$LATEST_CKPT --prompt='<|user|> Hello! <|assistant|>' --steps=200 --temperature=0.7"
  fi
fi

if [ "$EXIT_CODE" -ne 0 ]; then
  log "Training exited with code $EXIT_CODE"
  exit $EXIT_CODE
fi

log "Training complete!"
