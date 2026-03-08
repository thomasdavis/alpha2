#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

cd ~/nanochat
source .venv/bin/activate

echo "=== NANOCHAT FULL PIPELINE ==="
echo "$(date)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo ""

# Report reset
python -m nanochat.report reset

# Data + Tokenizer
echo "=== Downloading data (8 shards for tokenizer) ==="
python -u -m nanochat.dataset -n 8

echo "=== Downloading 80 shards in background ==="
python -u -m nanochat.dataset -n 80 &
DATA_PID=$!

echo "=== Training tokenizer ==="
python -u -m scripts.tok_train
python -u -m scripts.tok_eval

echo "=== Waiting for data download ==="
wait $DATA_PID || true

# Identity data
if [ ! -f "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" ]; then
    echo "=== Downloading identity data ==="
    curl -sL -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
        https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

echo ""
echo "=== DEPTH=12 BASE TRAINING (286M params, ~7.8 hours) ==="
echo "$(date)"
python -u -m scripts.base_train \
    --depth=12 \
    --device-batch-size=8 \
    --window-pattern=L

echo ""
echo "=== BASE EVAL ==="
echo "$(date)"
python -u -m scripts.base_eval --device-batch-size=8

echo ""
echo "=== POST-TRAINING PIPELINE ==="
echo "$(date)"
python -u -m scripts.post_train \
    --depth=12 \
    --device-batch-size=8 \
    --midtrain-steps=500 \
    --sft-s1-steps=2000 \
    --sft-s2-steps=1000 \
    --rl-num-steps=200 \
    --selfplay-rounds=3 \
    --neftune-alpha=5.0

echo ""
echo "=== NANOCHAT BUILT-IN SFT (for comparison) ==="
echo "$(date)"
python -u -m scripts.chat_sft --device-batch-size=8
python -u -m scripts.chat_eval -i sft

echo ""
echo "=== ALL DONE ==="
echo "$(date)"
echo "python -m scripts.chat_cli"
echo "python -m scripts.chat_web --port 8000"
