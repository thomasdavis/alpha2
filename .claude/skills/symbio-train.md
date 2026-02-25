Run symbiogenesis training on historic.txt via GCP.

## Launch

Export remote metrics env vars and run:

```bash
source .env.local && \
  export ALPHA_REMOTE_URL="$ALPHA_REMOTE_URL" && \
  export ALPHA_REMOTE_SECRET="$ALPHA_REMOTE_SECRET" && \
  export DISCORD_WEBHOOK_URL="$DISCORD_WEBHOOK_URL" && \
  python3 scripts/gcp_train.py \
  --data data/historic.txt \
  --domain chat \
  --iters 50000 \
  --batch 20 \
  --block 512 \
  --dim 384 \
  --heads 8 \
  --layers 8 \
  --lr 5e-5 \
  --backend helios \
  --tokenizer bpe \
  --warmup 1000 \
  --beta2 0.95 \
  --grad-clip 1.0 \
  --eval-interval 500 \
  --sample-interval 300 \
  --symbio \
  --symbio-config configs/symbio-composed-novels.json \
  --zone us-central1-b \
  --machine-type g2-standard-4
```

`gcp_train.py` reads `ALPHA_REMOTE_URL`, `ALPHA_REMOTE_SECRET`, and `DISCORD_WEBHOOK_URL` from `os.environ` and forwards them to the training process so live metrics stream to https://alpha.omegaai.dev/training and Discord gets notified. Inference samples post to Discord every 300 steps.

The instance stays running after training completes so you can download the checkpoint or upload it to the remote. Stop it manually when done:

```bash
python3 scripts/gcp_train.py --action stop
```

## Monitoring

Check the training log:

```bash
gcloud compute ssh alpha-train --project=GCP_PROJECT --zone=us-central1-b \
  --command="tail -50 ~/alpha/runs/train_*.log | tail -50"
```

## If the run fails, plateaus, or overfits

1. **Kill the run:**
   ```bash
   gcloud compute ssh alpha-train --project=GCP_PROJECT --zone=us-central1-b \
     --command="pkill -9 -f 'node.*train'"
   ```

2. **Diagnose** by tailing the log. Look for:
   - **Crash/failure:** stack trace or process exit in the log
   - **Gradient instability:** >25% of steps showing spike skips or NaN grad_norm
   - **Overfitting:** val_loss rising while train loss keeps dropping
   - **Plateau:** loss stuck for 2000+ steps with no improvement
   - **Symbio stagnation:** CUSUM "throughput collapse" on every step, no new candidates improving

3. **Fix the issue** in the codebase — refactor the training loop, adjust the symbio config, add new algorithms, fix bugs, whatever is needed.

4. **Rebuild on the instance:**
   ```bash
   gcloud compute ssh alpha-train --project=GCP_PROJECT --zone=us-central1-b \
     --command="cd ~/alpha && npm install --ignore-scripts && node packages/helios/native/build.mjs && npx turbo build --filter=@alpha/cli"
   ```
   Or just re-run `gcp_train.py` which syncs code and rebuilds automatically.

5. **Relaunch** using the command above.
