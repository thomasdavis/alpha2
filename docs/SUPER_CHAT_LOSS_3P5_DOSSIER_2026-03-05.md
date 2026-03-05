# Super Chat Loss 3.0-3.5 Dossier (L4, Helios)

Date: 2026-03-05 (UTC)
Audience: internal + external second opinion (ChatGPT/Claude)

## 1) Objective

Primary target:
- Train `data/super_chat.txt` to **validation loss 3.0-3.5** with coherent chat samples.

Secondary targets:
- Keep runs stable (no allocator/live-allocation runaway, no silent stalls).
- Keep telemetry reliable over API + Discord (samples every 200 steps).
- Keep run reproducible on Fleet-managed L4.

---

## 2) Current Run Status (Evidence)

Run ID:
- `super_chat_20260305075936_434f`

Canonical API endpoints:
- Run list: `https://alpha.omegaai.dev/api/runs`
- Run metrics: `https://alpha.omegaai.dev/api/runs/super_chat_20260305075936_434f/metrics`
- Run samples: `https://alpha.omegaai.dev/api/runs/super_chat_20260305075936_434f/samples`
- Dashboard page: `https://alpha.omegaai.dev/runs/super_chat_20260305075936_434f`

Observed from API:
- `latest_step`: **843 / 12000**
- `status`: **active** (but stale)
- `updated_at`: **2026-03-05 08:05:56 UTC**
- `best val_loss`: **4.7467** (at step 800)
- `best train loss`: **4.7091**

Why this is suspicious:
- Run is labeled `active` but timestamp is stale and step does not move.
- Fleet SSH to configured L4 instances currently times out.
- This suggests a stale status record and a dead or unreachable worker.

---

## 3) Metric Forensics

### 3.1 Full run summary (843 points)

- Start loss: `7.6209` (step 1)
- Best loss: `4.7091` (step 800)
- Last loss: `6.2145` (step 843)
- Validation checkpoints recorded at steps: 200, 400, 600, 800
  - step 200: val `6.0416`
  - step 400: val `4.8791`
  - step 600: val `4.7754`
  - step 800: val `4.7467` (best)

Interpretation:
- Training improved quickly through step ~800.
- After that, run destabilized and regressed.

### 3.2 Instability events

Detected high-grad spikes (`grad_norm > 1000`):
- step 832: loss 6.2298, grad_norm 108,734, clip_coef 9.20e-6
- step 840: loss 6.2241, grad_norm 75,875, clip_coef 1.32e-5
- step 841: loss 7.6679, grad_norm 234,137, clip_coef 4.27e-6
- step 843: loss 6.2145, grad_norm 116,340, clip_coef 8.60e-6

These are catastrophic-scale spikes relative to normal regime (`grad_norm ~0.15-0.9`).

### 3.3 Throughput drift (performance symptom)

Average by phase:
- Steps 1-200: 59,788 tok/s, 104.6 ms/iter
- Steps 201-600: 58,824 tok/s, 105.8 ms/iter
- Steps 601-843: 47,786 tok/s, 139.3 ms/iter

Interpretation:
- Late-run slowdown correlates with instability/pressure, not just normal variance.

### 3.4 Sample quality snapshot from API

`/samples` currently returns only earliest (step~start) gibberish outputs, e.g. repetitive malformed text.

Interpretation:
- Either no later sample ingestion happened before stall, or API is not serving latest samples for this run.
- This makes sample-quality tracking unreliable without checking logs/checkpoints directly.

---

## 4) Data and Split Diagnostics

Dataset: `data/super_chat.txt`

Measured stats:
- Conversations: `226,110`
- Mean chars/conversation: `399.4`
- p50: `240`, p90: `550`, p99: `3323`, max: `5607`
- Marker counts:
  - `<|user|>`: `609,740`
  - `<|assistant|>`: `609,740`
  - `<|end_of_text|>`: `226,110`

Critical distribution issue (contiguous split risk):
- Mean length first 90% records: `264.26`
- Mean length last 10% records: `1615.63`

Interpretation:
- A contiguous tail validation split is strongly biased and can distort training/validation behavior.
- Deterministic delimiter-aware split is required for meaningful val signals.

---

## 5) Code Audit (Key Files + Evidence)

### 5.1 Domain config currently used (`super_chat`)

File:
- `packages/core/src/domains.ts` (lines ~178-210)

Current defaults:
- model: `4L / 128D / 4H / block=256 / dropout=0`
- train: `lr=1e-4`, `lrMin=1e-5`, `warmup=1600`, `batch=12`, `accum=2`, `gradClip=0.5`, `spikeThreshold=100`, `evalInterval=200`, `sampleInterval=200`

Risk:
- LR schedule appears aggressive for long-run stability on this setup.

---

### 5.2 Tokenizer training sample bias fix

File:
- `packages/tokenizers/src/bpe.ts` (lines ~23-42, 93-97)

Implemented behavior:
- `ALPHA_BPE_MAX_TRAIN_CHARS` env support
- `buildTrainingSample()` now uses 32 windows across corpus (not prefix-only)

Snippet:
```ts
const maxTrainChars = readPositiveIntEnv("ALPHA_BPE_MAX_TRAIN_CHARS", 500_000);
const trainText = buildTrainingSample(input, maxTrainChars);
```

Impact:
- Reduces merge bias from early-file-only BPE training.

---

### 5.3 Deterministic delimiter-aware train/val split

File:
- `packages/train/src/trainer.ts` (lines ~280-317, 475-497)

Implemented behavior:
- Uses `<|end_of_text|>` hash-based split by default (`ALPHA_TEXT_SPLIT_MODE=auto`)
- Controlled by:
  - `ALPHA_TEXT_SPLIT_MODE`
  - `ALPHA_TEXT_SPLIT_VAL_FRACTION`
  - `ALPHA_TEXT_SPLIT_DELIMITER`

Snippet:
```ts
const delimiter = process.env.ALPHA_TEXT_SPLIT_DELIMITER ?? "<|end_of_text|>";
const byDelimiter = canUseDelimiterSplit
  ? splitByDelimiterDeterministic(rawText, delimiter, splitFraction, trainConfig.seed)
  : null;
```

Impact:
- Prevents tail-skew validation pathology on super_chat.

---

### 5.4 Sample generation + plateau/overfit signaling + Discord posting

Files:
- `packages/train/src/trainer.ts` (lines ~1596-1653)
- `packages/train/src/remote-reporter.ts` (lines ~350-371)

Implemented behavior:
- Samples every `sampleInterval` (default 200 for super_chat)
- Trend analysis computes plateau/overfitting warnings from eval window
- `onSamples` sends to API ingest and Discord

Snippet:
```ts
if (deps.onSamples) await deps.onSamples(samples, stepNum, trend ?? undefined);
```

Discord snippet:
```ts
await sendDiscord(discordWebhook, [{
  title: `📝 Inference Samples${step ? ` (Step ${step})` : ""}`,
  fields: sampleFields,
}]);
```

Gap:
- API `/samples` for this run appears stale; verify backend ingestion/read path consistency.

---

### 5.5 Adaptive memory pressure controls and purge path

File:
- `packages/train/src/trainer.ts` (lines ~809-813, 1229-1255)

Current defaults in code:
- `ALPHA_ADAPTIVE_SYNC_MIN_INTERVAL` default `6`
- `ALPHA_ADAPTIVE_SYNC_DEFERRED_THRESHOLD` default `28`
- `ALPHA_ADAPTIVE_SYNC_LIVE_ALLOCS_THRESHOLD` default `5200`
- `ALPHA_ADAPTIVE_PURGE_LIVE_ALLOCS_THRESHOLD` default `6000`

Snippet:
```ts
const adaptiveSyncPressure = !!(
  memStatsStep && (
    memStatsStep.deferredReleases > ADAPTIVE_SYNC_DEFERRED_THRESHOLD ||
    (memStatsStep.liveAllocs ?? 0) > ADAPTIVE_SYNC_LIVE_ALLOCS_THRESHOLD
  )
);
```

Impact:
- Trainer now has explicit live-allocation pressure controls.
- Still needs host stability + per-run tuning to avoid long-run degradations.

---

## 6) Root Cause Hypotheses (Ranked)

1. **Run lifecycle/infra reliability issue**
- Evidence: stale `active` run + unreachable Fleet hosts.
- If worker dies or disconnects, training quality work is blocked.

2. **Late-stage optimization instability**
- Evidence: catastrophic grad spikes after step ~832 despite earlier steady improvements.
- Likely interaction of LR schedule, clipping behavior, and backend memory pressure.

3. **Model/data objective mismatch for target loss 3.0-3.5**
- Current 1.33M config may underfit conversational complexity at desired quality floor.
- Need capacity and/or longer horizon with stable schedule.

4. **Sample telemetry visibility gap**
- API samples not reflecting later checkpoints makes quality diagnosis harder.

---

## 7) Plan to Reach Loss 3.0-3.5 (Concrete)

### Phase A: Re-establish stable training execution (must do first)

1. Bring up one known-good L4 Fleet instance.
2. Verify run process health + heartbeat every minute.
3. Confirm metrics and samples appear via API endpoints.
4. Confirm Discord receives sample payload every 200 steps.

Exit criteria:
- 1,000 uninterrupted steps with no stall and no catastrophic grad spikes.

### Phase B: Run matrix (small number of high-value configs)

Run all with:
- delimiter split enabled (`ALPHA_TEXT_SPLIT_MODE=auto`)
- BPE artifacts cached
- `sampleInterval=200`, `evalInterval=200`

Config R1 (stability-first baseline):
- 4L/128D/4H
- `lr=1.0e-4`, `lrMin=1.0e-5`, `warmup=1600`, `gradClip=0.5`
- `batch=12`, `accum=2`

Config R2 (capacity bump, conservative LR):
- 6L/192D/6H
- `lr=8.0e-5`, `lrMin=8.0e-6`, `warmup=2000`, `gradClip=0.4`
- `batch=8`, `accum=3`

Config R3 (capacity bump v2):
- 6L/256D/8H (if memory stable)
- `lr=6.0e-5`, `lrMin=6.0e-6`, `warmup=2400`, `gradClip=0.35`
- `batch=6`, `accum=4`

Selection criteria after 1200-2000 steps:
- best val_loss
- spike frequency (`grad_norm > 1000` count)
- sample coherence at steps 400/800/1200

### Phase C: Hold/kill policy

Kill run if either is true:
- No new best val_loss for 6 eval windows AND trend says plateau likely.
- 3+ catastrophic spikes (`grad_norm > 1000`) within 200 steps.

Continue run if:
- val_loss improving OR stable with better sample coherence.

---

## 8) Fleet + API Repro Commands

### 8.1 Verify Fleet config and instance

```bash
cat fleet.json
npm run fleet:status -- <instance>
npm run fleet:run -- <instance> -- "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"
```

### 8.2 Start stable super_chat run (example)

```bash
ALPHA_TEXT_SPLIT_MODE=auto \
ALPHA_TEXT_SPLIT_VAL_FRACTION=0.1 \
ALPHA_TEXT_SPLIT_DELIMITER='<|end_of_text|>' \
ALPHA_BPE_MAX_TRAIN_CHARS=800000 \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
HELIOS_WG_SIZE=256 \
npm run fleet:train -- <instance> -- \
  --domain=super_chat \
  --data=data/super_chat.txt \
  --backend=helios \
  --gpuProfile=l4 \
  --steps=12000 \
  --lr=1e-4 --lrMin=1e-5 --warmupIters=1600 \
  --gradClip=0.5 \
  --batch=12 --accumSteps=2 \
  --evalInterval=200 --evalIters=10 \
  --sampleInterval=200 \
  --postSamples=false \
  --tokenizerArtifacts=runs/tokenizer-artifacts-super-chat-bpe2k-v2.json \
  --runDir=runs/super_chat_loss35_r1
```

### 8.3 API checks (must pass)

```bash
curl -sS https://alpha.omegaai.dev/api/runs | jq '[.[]|select(.id|test("super_chat"))][:5]'
curl -sS https://alpha.omegaai.dev/api/runs/<run_id>/metrics | jq 'length, .[-1]'
curl -sS https://alpha.omegaai.dev/api/runs/<run_id>/samples | jq 'length, .[-1]'
```

---

## 9) Why previous low-loss-but-bad-samples happened

Observed pattern from prior reports:
- scalar loss decreased, but generations stayed repetitive/gibberish.

Most likely causes in this stack:
- Overly easy short-context fitting + weak generalization signal.
- Tokenizer artifacts biased by insufficient corpus coverage (now partially fixed).
- Run instability/corruption events (spikes) after early progress.
- Sample API serving stale entries, masking real trend.

So "loss is low" was not enough; we need:
- validation trend,
- coherent sample snapshots,
- and stability metrics together.

---

## 10) Open Issues to Validate Next

1. Is `/api/runs/:id/samples` returning latest samples after step 200?
2. Is the stale `active` run status cleaned up when workers die?
3. Are Fleet instance definitions current (hosts reachable)?
4. Does long-run memory pressure still rise monotonically on new code path?

---

## 11) Minimal Ask for External Second Opinion

Please review and answer:
1. Is the run instability likely optimizer schedule, allocator pressure, or both?
2. Are the proposed R1-R3 configs reasonable for reaching val loss 3.0-3.5 on this corpus?
3. Would you change tokenizer strategy (2k BPE vs 4k) given current loss/sample behavior?
4. What additional single metric would best predict "good samples" beyond val loss in this setup?

---

## 12) Changes Implemented After This Analysis

These were applied directly to reduce recurrence of the exact issues identified above.

1. Run stale/heartbeat lifecycle hardening
- `packages/db/src/runs.ts`
  - Added automatic status refresh on reads:
    - `active -> completed` when `latest_step >= total_iters`
    - `active -> stale` when no update for `ALPHA_RUN_STALE_SECS` (default 180s, min 30s)
  - `getRun()` and `listRuns()` now call this refresh.
- `packages/train/src/remote-reporter.ts`
  - Added periodic heartbeat POST to `/api/ingest` (`type=heartbeat`, default every 60s).
- `apps/web/src/app/api/ingest/route.ts`
  - Added `heartbeat` ingest handler to keep active runs fresh while process is alive.

2. Samples endpoint freshness fix
- `packages/db/src/samples.ts`
  - Changed samples insert from `INSERT OR IGNORE` to `ON CONFLICT(run_id, idx) DO UPDATE`.
  - Samples at indices 0..N now overwrite with latest outputs; `/api/runs/:id/samples` no longer freezes at first sample set.

3. Best validation-loss tracking fix
- `apps/web/src/app/api/ingest/route.ts`
  - Previously `best_val_loss` only read from the last metric in each batch.
  - Now computes min validation loss across the whole posted metrics batch (`valLoss` or `val_loss`), then updates run progress.

4. Spike circuit breaker strengthened
- `packages/train/src/trainer.ts`
  - Existing spike-skip behavior retained.
  - Added LR backoff on spikes:
    - `ALPHA_SPIKE_LR_BACKOFF` (default `0.5`)
    - `ALPHA_SPIKE_LR_MIN_SCALE` (default `0.1`)
  - Added per-step batch fingerprint logging on spike:
    - `batch_hash` and `tok_range` to identify pathological batches.
  - Added persistent runtime LR scaling after spike events.

5. Length-bucket validation diagnostics
- `packages/train/src/trainer.ts`
  - Added optional delimiter-based val bucket loaders (`short/medium/long`) for non-large-file mode.
  - Enabled with `ALPHA_VAL_BUCKET_EVAL=1`.
  - Tunable with `ALPHA_VAL_BUCKET_EVAL_ITERS` (default 2).
  - Logs per-bucket eval losses at eval points:
    - `[val_bucket] short=... | medium=... | long=...`

6. Super-chat default stability tuning
- `packages/core/src/domains.ts` (`super_chat` train defaults)
  - `lr: 1e-4` (was `2e-4`)
  - `lrMin: 1e-5` (was `2e-5`)
  - `warmupIters: 1600` (was `1200`)
  - `gradClip: 0.5` (was `1.0`)
  - `spikeThreshold: 100` (new)

---

## 13) Follow-up Execution Update (2026-03-05 09:32 UTC)

### Additional hardening implemented

1. Remote metrics durability
- `packages/train/src/remote-reporter.ts`
  - Metric flush is serialized (`flushInFlight`) to avoid overlapping writes.
  - Failed metric batches are re-queued instead of dropped.
  - Added bounded backlog guard (`MAX_BUFFERED_METRICS=5000`) with explicit warning on forced drop.
  - `complete()` now drains buffered metrics deterministically before marking run complete.
  - `sendSamples()` now includes `step` in ingest payload.

2. Adaptive sync/purge behavior
- `packages/train/src/trainer.ts`
  - Need-sync path now executes sync first, then re-reads memory stats.
  - Pool purge only occurs if pressure remains high after sync.
  - Added `ALPHA_ADAPTIVE_PURGE_MIN_INTERVAL` (default `max(16, ADAPTIVE_SYNC_MIN_INTERVAL)`).
  - Goal: prevent repeated purge thrash and long stalls under allocator pressure.

3. Training startup no longer blocked by run registration
- `packages/train/src/trainer.ts`
  - `onStart` callback is now non-blocking and error-tolerant.
  - Remote ingest latency/outage can no longer delay reaching step 1.

4. Fleet launch robustness updates
- `apps/cli/src/commands/fleet.ts`
  - L4 launch path auto-exports:
    - `HELIOS_WG_SIZE=256` if unset
    - `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json` if unset
  - Train/resume launcher now skips Nix shell when using compiled `./alpha` prefix.

### Infra truth discovered

- On this specific host, compiled binary path (`/home/ajax/alpha/alpha`) still fails during GPU init:
  - `vkCreateInstance failed`
- Reliable path remains Node dist from `/home/ajax/alpha-repo`:
  - `node --expose-gc apps/cli/dist/main.js train ...`

### Current active run (healthy progression)

- `run_id`: `super_chat_20260305092634_zg6e`
- run dir: `runs/super_chat_loss35_stable_repo_20260305_192629`
- host: `alpha-bench-l4-coopdbg-20260228084511` (NVIDIA L4, driver 590.48.01)
- core flags: `fp16=false`, `sampleInterval=200`, `evalInterval=200`, `syncEvery=1`, `gcEvery=1`

Observed trajectory:
- Step 200: `loss=6.7767`, `val_loss=6.7645`
- Step 400: `loss=5.8079`, `val_loss=5.7971`
- Step 500: `loss=5.3706`
- Step ~660 (live): recent losses around `~4.93`

Key stability signals:
- No catastrophic grad-norm spikes through this window.
- API run telemetry now advances continuously (past prior step-10 freeze behavior).

Update after step ~950:
- Repeated allocator-triggered purges (liveAllocs > ~7400) were followed by extreme gradient spikes (`12k` → `96k`) and loss jump.
- Branch was stopped and resumed conservatively from checkpoint-800 using:
  - `lr=4e-5`, `lrMin=4e-6`, `warmupIters=1000`, `gradClip=0.4`, `spikeThreshold=50`.
  - New run dir: `runs/super_chat_resume800_conservative_20260305_193522`.
  - New run id after resume: `super_chat_20260305093527_jdrl`.
  - Resume branch is active and stable through ~step 1560 with `loss≈4.7177`, `val_loss≈4.7509` at step 1400.

Further update:
- To prevent repeat failure around step ~1750, a stricter branch was resumed from checkpoint-1600:
  - `run_id: super_chat_20260305103105_q8r3`
  - `batch=8`, `accum=2`, `lr=2e-5`, `lrMin=2e-6`, `gradClip=0.3`, `spikeThreshold=30`
  - tighter allocator/adaptive caps (`HELIOS_LIVE_ALLOC_*`, `ALPHA_ADAPTIVE_*`).
- Current observed status: step ~2340, `loss≈4.7115`, `best_val_loss≈4.7146`, no catastrophic spike recurrence yet.

### Remaining risks still under watch

1. Long-run allocator pressure:
- `liveAllocs` still trends upward over time; continue monitoring past 1k/2k steps.

2. Sample freshness on hosted API:
- Local code now supports sample overwrite; production server deploy state should be verified if stale sample behavior persists.

3. Loss target gap:
- Current trajectory is materially improved, but still above 3.5.
- Next decision point is after the next few eval checkpoints (>=800, 1000, 1200) with sample coherence review.
