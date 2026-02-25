#!/usr/bin/env bash
#
# modal-run.sh — One-command training on Modal H100.
#
# Usage:
#   ./scripts/modal-run.sh data/books_all.txt --iters=5000 --backend=helios
#   ./scripts/modal-run.sh books_all.txt --iters=10000 --batch=128
#
# The first argument is the dataset (local path or just filename if already uploaded).
# All remaining arguments are passed directly as training args.
#
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <dataset> [training args...]"
  echo ""
  echo "Examples:"
  echo "  $0 data/books_all.txt --iters=5000 --backend=helios"
  echo "  $0 concordance.txt --iters=20000 --batch=128 --dim=512 --layers=12"
  echo ""
  echo "Training args: --iters --batch --block --dim --heads --layers --lr"
  echo "               --backend --tokenizer --domain"
  exit 1
fi

DATA="$1"
shift

# Parse known flags, pass the rest through
ITERS=1000; BATCH=64; BLOCK=256; DIM=256; HEADS=8; LAYERS=6
LR=3e-4; BACKEND=helios; TOKENIZER=bpe; DOMAIN=""; UPLOAD=false

EXTRA_ARGS=""
for arg in "$@"; do
  case "$arg" in
    --iters=*)     ITERS="${arg#*=}" ;;
    --batch=*)     BATCH="${arg#*=}" ;;
    --block=*)     BLOCK="${arg#*=}" ;;
    --dim=*)       DIM="${arg#*=}" ;;
    --heads=*)     HEADS="${arg#*=}" ;;
    --layers=*)    LAYERS="${arg#*=}" ;;
    --lr=*)        LR="${arg#*=}" ;;
    --backend=*)   BACKEND="${arg#*=}" ;;
    --tokenizer=*) TOKENIZER="${arg#*=}" ;;
    --domain=*)    DOMAIN="${arg#*=}" ;;
    --upload)      UPLOAD=true ;;
    *)             EXTRA_ARGS="$EXTRA_ARGS $arg" ;;
  esac
done

# Build modal run command
CMD="modal run scripts/modal_train.py"
CMD="$CMD --data $DATA"
CMD="$CMD --iters $ITERS --batch $BATCH --block $BLOCK"
CMD="$CMD --dim $DIM --heads $HEADS --layers $LAYERS"
CMD="$CMD --lr $LR --backend $BACKEND --tokenizer $TOKENIZER"

if [ -n "$DOMAIN" ]; then
  CMD="$CMD --domain $DOMAIN"
fi

if [ "$UPLOAD" = true ]; then
  CMD="$CMD --upload true"
fi

echo "============================================================"
echo "Alpha Modal Training"
echo "============================================================"
echo "Dataset:   $DATA"
echo "Backend:   $BACKEND"
echo "Model:     ${DIM}d ${HEADS}h ${LAYERS}L"
echo "Training:  ${ITERS} iters, batch=${BATCH}, block=${BLOCK}"
echo "Command:   $CMD"
echo "============================================================"
echo ""

exec $CMD
