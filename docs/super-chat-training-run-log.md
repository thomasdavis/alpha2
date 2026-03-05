# Super Chat Training Run Log (L4)

Last updated: 2026-03-05 09:32 UTC

## Objective

Train on `data/super_chat.txt` with Helios on L4, keep remote reporting + Discord samples every 200 steps, and avoid the long-run allocator cap crash.

## Environment

- Instance: `alpha-bench-l4-coopdbg-20260228084511` (`136.113.161.152`)
- GPU: NVIDIA L4 (`driver 590.48.01`)
- Runtime path used for training: `/home/ajax/alpha-repo` (Node dist)
- Dataset: `/home/ajax/alpha-repo/data/super_chat.txt`
- Remote reporting: `https://alpha.omegaai.dev`
- Discord sample posting: enabled via `DISCORD_WEBHOOK_URL`

## Code Changes Applied In This Pass

1. Autograd shared-gradient release ordering fix:
- `packages/autograd/src/tape.ts`
- Grad tensors shared across multiple trainable inputs are now reference-counted per entry before release, preventing premature release/reclone behavior.

2. Helios allocator pressure handling:
- `packages/helios/src/backend.ts`
- Added pool trimming under live-allocation pressure.
- Added retry-on-allocation-failure after aggressive trim.
- Added pool entry caps:
  - `HELIOS_MAX_BUFFER_POOL_ENTRIES` (default 2048)
  - `HELIOS_MAX_OUTPUT_POOL_ENTRIES` (default 2048)
- Added allocator pressure env controls:
  - `HELIOS_LIVE_ALLOC_SOFT_CAP` (default 7000)
  - `HELIOS_LIVE_ALLOC_HARD_CAP` (default 7800)
- Fixed OOM accounting bug: failed allocation no longer leaves `_liveAllocCount` incremented.

3. Training-side adaptive memory defaults tightened:
- `packages/train/src/trainer.ts`
- Default adaptive polling/sync/purge thresholds lowered vs prior values.

4. Inference check exporter generalized:
- `scripts/update-l4-inference-checks-md.sh`
- Heading now derives from output filename instead of hardcoded “historic v2”.

## Launch/Debug Timeline

### A) Fleet binary path still fails Vulkan init

`fleet train` via `/home/ajax/alpha/alpha` still fails on this host with:
- `vkCreateInstance failed`

Action:
- Continued training via `/home/ajax/alpha-repo` + `node --expose-gc apps/cli/dist/main.js`.

### B) First `super_chat` run (fp16 path): immediate NaNs

Config:
- chat domain-like shape, `layers=6 dim=256 heads=8 block=256 batch=4 accum=2`
- `fp16=true` (effective from L4 profile)

Outcome:
- `loss=NaN` from first steps.
- Run discarded.

### C) Stable fp32 run from scratch

Run:
- `run_id: super_chat_20260305033552_t4qf`
- run dir: `runs/super_chat_l4_stable_20260305_133548`
- log: `train_super_chat_stable_20260305_133548.log`

Config:
- same model shape as above
- `--fp16=false --packed=false`
- `--syncEvery=1 --gcEvery=1`
- `--sampleInterval=200 --evalInterval=200 --evalIters=3`

Result:
- Healthy descent to step 600:
  - step 200: `loss=7.6971`, `val_loss=7.6433`
  - step 400: `loss=7.0320`, `val_loss=6.9274`
  - step 600: `loss=6.5155`, `val_loss=6.5140`
- Then allocator cap failure at step 650:
  - live allocs reached 8192
  - `Fatal: Error: Max buffers reached`

### D) Resume from checkpoint-600 with strict alloc caps

Run:
- `run_id: super_chat_20260305034257_9p2l`
- resume from `checkpoint-600.json`

Extra env used:
- `ALPHA_ADAPTIVE_MEM_STATS_POLL_EVERY=4`
- `ALPHA_ADAPTIVE_SYNC_MIN_INTERVAL=2`
- `ALPHA_ADAPTIVE_SYNC_LIVE_ALLOCS_THRESHOLD=5600`
- `ALPHA_ADAPTIVE_PURGE_LIVE_ALLOCS_THRESHOLD=6200`
- `HELIOS_LIVE_ALLOC_SOFT_CAP=6000`
- `HELIOS_LIVE_ALLOC_HARD_CAP=6800`
- `HELIOS_MAX_OUTPUT_POOL_ENTRIES=1024`
- `HELIOS_MAX_BUFFER_POOL_ENTRIES=1024`

Result:
- Continued cleanly to ~1200.
- Later gradient/loss explosion (~step 1150+):
  - `loss` jumped to ~8.3
  - huge grad norms and `clip≈0`
- Branch considered unstable for quality; stopped.

### E) Resume from checkpoint-1000 with LR/clip fix (current active)

Run:
- `run_id: super_chat_20260305034747_aw21`
- resume from `checkpoint-1000.json`
- log: `/home/ajax/alpha-repo/train_super_chat_resume_lrfix_20260305_034746.log`

Config changes:
- `--lr=2e-5`
- `--lrMin=2e-6`
- `--gradClip=0.3`
- same strict alloc env caps as section D

Current observed progress:
- step 1200: `loss=6.6129`, `val_loss=6.4100`
- step 1400: `loss=6.5851`, `val_loss=6.4569`
- step 1500: `loss=6.5227`
- step 1550: drift/explosion recurs (`loss≈8.32`, huge grad norm), correlated with live alloc climbing above ~6800 again.

## Inference Sampling Artifacts

Generated and updated via:

```bash
bash scripts/update-l4-inference-checks-md.sh \
  alpha-bench-l4-coopdbg-20260228084511 \
  docs/super-chat-inference-checks.md \
  /home/ajax/alpha-repo \
  200
```

Output file:
- `docs/super-chat-inference-checks.md`

This file includes checkpoint-based samples for steps 200/400/600/800/1000 (and later as checkpoints appear).

## Current Assessment

1. Numerics can be made stable (fp32 path), but long-run allocator growth still exists.
2. Strict pool/threshold controls delay crashes but do not fully eliminate growth.
3. Current 6.84M-parameter setup is data-starved for this token budget (~4.5 tokens/param), which aligns with weak sample quality and poor loss floor.

## Immediate Next Actions

1. Shift to a smaller model (higher tokens/param) for quality target runs.
2. Add restart cadence (checkpoint-chunked execution) so runs do not approach 8k live allocs in a single process.
3. Keep eval/sample cadence at 200 (Discord/reporting requirement), but restart before allocator pressure enters unstable range.

## F) Small-model pivot (current active branch)

Reason:
- 6.84M-parameter run remained allocator-sensitive and sample quality stayed poor.
- Token budget warning was severe for that shape.

New run:
- `run_id: super_chat_20260305035248_ri37`
- run dir: `runs/super_chat_small_20260305_035247`
- log: `/home/ajax/alpha-repo/train_super_chat_small_20260305_035247.log`
- params: `1,845,504` (~7 MB)

Config highlights:
- `layers=4 dim=128 heads=4 block=256`
- `batch=8 accum=2` (effective batch 16)
- `steps=20000`
- `lr=1e-4, lrMin=1e-5, warmup=1000`
- `fp16=false, packed=false`
- `sampleInterval=200, evalInterval=200, evalIters=3`
- same strict allocator env controls as above

Early trajectory:
- step 200: `loss=7.7744`, `val_loss=7.7307`
- step 400: `loss=7.0617`, `val_loss=6.9408`
- step 600: `loss=6.6763`, `val_loss=6.5151`

Observed behavior:
- Convergence is materially better than the earlier unstable resumed branch.
- Allocator live count still trends upward (`~5349` live by step 600), so restart cadence remains necessary for very long runs.

## G) Chunked Auto-Resume Loop Enabled (Current Mode)

To prevent the per-process allocator runaway from reaching the 8192 handle cap, training is now running in chunked mode:

- Loop process: `/home/ajax/alpha-repo/super_chat_small_chunk_loop.sh`
- Loop PID file: `/home/ajax/alpha-repo/train.loop.pid`
- Active run dir: `runs/super_chat_small_20260305_035247`
- Chunk size: `600` steps
- Final target: `20000` steps
- Sample/eval cadence preserved: every `200` steps

Loop behavior:

1. Finds latest checkpoint in run dir.
2. Starts next process with `--steps = latest + 600` (or final target).
3. Writes active child pid to `train.pid`, active chunk log to `train.log.path`.
4. Waits for process exit, then continues with the next checkpoint.

This keeps each process below the observed allocator-instability window while preserving continuous remote reporting and Discord sample posting.

## H) Crash Root Cause Confirmed + Hardening Applied

Observed root cause for loop stops:

- Chunk training completed successfully, but process exited non-zero during post-training sample generation.
- Failure path was unguarded in CLI post-sample loop, so a single OOM could terminate the chunk process.

Actions applied:

1. CLI hardening:
- File: `apps/cli/src/commands/train.ts`
- Post-training samples are now disabled by default on GPU backends (`--postSamples` still overrides).
- Post-training sample generation is now wrapped in `try/catch` per prompt so failures do not crash the run.

2. Runtime policy on L4 chunk loop:
- Keep `--postSamples=false` in chunked training command.
- Continue step-level sampling (`sampleInterval=200`) for remote/Discord reporting.

3. Supervisor + watchdog (temporary during diagnosis):
- Loop process: `/home/ajax/alpha-repo/super_chat_small_chunk_loop.sh`
- Watchdog process: `/home/ajax/alpha-repo/super_chat_loop_watchdog.sh`
- Watchdog log: `/home/ajax/alpha-repo/train_super_chat_watchdog.log`
- This was used to prove continuity and isolate the post-sample crash path.

Live confirmation (UTC):

- `04:43:51`: chunk `1200 -> 1800` exited code 0.
- `04:46:24`: chunk `1800 -> 2400` exited code 0.
- `04:46:26`: next chunk `2400 -> 3000` started automatically.

Current state:

- Training is active and advancing checkpoints in chunk mode.
- No post-training sample OOM crash after chunk completion.

## I) Auto-Restart Policy Updated

Per user direction, auto-restart has been disabled:

- Stopped watchdog and loop supervisors.
- Removed `train.loop.pid` / `train.watchdog.pid`.
- Left exactly one active trainer process (`train.pid`) running fail-fast.

Policy now:

- If training crashes, it stays down.
- We diagnose root cause first, then restart manually with targeted fixes.

## J) 2026-03-05 Recovery Pass (Current Active Run)

### What was fixed in this pass

1. Remote ingest resilience:
- `packages/train/src/remote-reporter.ts`
- Metrics flush is now serialized and retry-safe (failed batches are re-queued instead of silently dropped).
- Added bounded metric buffer (`MAX_BUFFERED_METRICS=5000`) to avoid unbounded growth on remote outages.
- `sendSamples(...)` now includes `step` in ingest payload so sample events advance run progress correctly.

2. Adaptive memory pressure behavior:
- `packages/train/src/trainer.ts`
- Sync now occurs before purge decisions, then memory is re-probed.
- Added purge cooldown control:
  - `ALPHA_ADAPTIVE_PURGE_MIN_INTERVAL` (default max(16, sync interval)).
- This prevents repeated purge thrash under live-allocation pressure.

3. Startup non-blocking remote register:
- `packages/train/src/trainer.ts`
- `onStart` (run_start ingest) is now non-blocking and cannot stall training startup on network hiccups.

4. Fleet launch robustness:
- `apps/cli/src/commands/fleet.ts`
- L4 launcher now auto-exports:
  - `HELIOS_WG_SIZE=256` (if unset)
  - `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json` (if unset)
- Train/resume launcher now skips Nix shell when using compiled `./alpha` prefix.

### Infra finding (important)

- Compiled binary path `/home/ajax/alpha/alpha` still fails Vulkan init on this host with:
  - `vkCreateInstance failed`
- Node dist path in `/home/ajax/alpha-repo` remains the reliable execution path for now.

### Current active run

- Process:
  - host: `alpha-bench-l4-coopdbg-20260228084511`
  - pid: `217102`
  - cwd: `/home/ajax/alpha-repo`
- Run:
  - `run_id: super_chat_20260305092634_zg6e`
  - run dir: `runs/super_chat_loss35_stable_repo_20260305_192629`
  - log: `train_super_chat_loss35_stable_repo_20260305_192629.log`
- Config:
  - `domain=super_chat`
  - `steps=12000`
  - `sampleInterval=200`
  - `evalInterval=200`
  - `fp16=false`
  - `syncEvery=1`
  - `gcEvery=1`
  - `lr=1e-4`, `lrMin=1e-5`, `warmup=1600`, `gradClip=0.5`, `spikeThreshold=100`

### Current trajectory snapshot

- Step 200:
  - `loss=6.7767`, `val_loss=6.7645`
- Step 400:
  - `loss=5.8079`, `val_loss=5.7971`
- Step 500:
  - `loss=5.3706`
- API telemetry now advances past the historical “step=10” stall:
  - `/api/runs` shows latest step moving (e.g. `~589+` during this pass)
  - `/api/runs/<run_id>/metrics?last=5` shows recent step/loss entries in real time.

### Immediate watchpoints

1. Gradient spike recurrence:
- No catastrophic spike observed through step ~500 in this run.
- Continue watching `grad_norm` and skip-step events.

2. Allocator pressure trend:
- `liveAllocs` still trends upward during long runs; this remains the top long-run risk.
- Sync-before-purge has reduced early thrash behavior, but long-horizon monitoring is still required.

3. Sample quality vs loss:
- Early samples still noisy but improving from step 200 to 400.
- Keep step-200 cadence and compare with eval trend for plateau/overfit decisions.

## K) Step~950 Spike Event + Conservative Resume Branch

Observed on run `super_chat_20260305092634_zg6e`:

- Healthy descent through step 800:
  - step 800: `loss=4.7936`, `val_loss=4.8460`
- Allocator pressure then crossed purge threshold repeatedly:
  - `liveAllocs` exceeded ~7400 and triggered repeated pool purges.
- Catastrophic gradient event around steps 949–953:
  - `grad_norm` spikes: `12258`, `65252`, `76552`, `92637`, `96547`
  - skip-step circuit breaker fired multiple times
  - runtime LR scale decayed to floor (`0.10`)
  - loss jumped (`~6.23`) indicating branch corruption risk.

Action taken:

1. Stopped this branch at ~step 950.
2. Started a conservative resume from stable checkpoint 800:
- source checkpoint:
  - `runs/super_chat_loss35_stable_repo_20260305_192629/checkpoint-800.json`
- new run dir:
  - `runs/super_chat_resume800_conservative_20260305_193522`
- key config changes:
  - `lr=4e-5` (down from schedule peak around `6e-5`)
  - `lrMin=4e-6`
  - `warmupIters=1000`
  - `gradClip=0.4`
  - `spikeThreshold=50`
  - `syncEvery=1`, `gcEvery=1`
  - `evalInterval=200`, `sampleInterval=200`

Status at this log update:

- Conservative resume is active under run id:
  - `super_chat_20260305093527_jdrl`
- Recent trajectory:
  - step `1400`: `loss=4.7402`, `val_loss=4.7509`
  - step `1560`: `loss≈4.7177` (latest API snapshot)
- No catastrophic spike recurrence observed so far on this conservative branch.
- This repo path (`/home/ajax/alpha-repo`) remains the reliable runtime path; compiled `/home/ajax/alpha/alpha` still fails Vulkan instance creation on this host.

## L) Strict Resume Branch From Checkpoint-1600 (Current)

After recurrence near step ~1750 on the previous branch, a stricter resume was launched from checkpoint-1600:

- source checkpoint:
  - `runs/super_chat_resume800_conservative_20260305_193522/checkpoint-1600.json`
- run dir:
  - `runs/super_chat_resume1600_strict_20260305_203101`
- run id:
  - `super_chat_20260305103105_q8r3`

Stricter config:

- `batch=8`, `accumSteps=2`
- `lr=2e-5`, `lrMin=2e-6`
- `warmupIters=1000`
- `gradClip=0.3`
- `spikeThreshold=30`
- `evalInterval=200`, `evalIters=10`, `sampleInterval=200`
- `syncEvery=1`, `gcEvery=1`

Allocator/pressure env clamps:

- `ALPHA_ADAPTIVE_SYNC_MIN_INTERVAL=2`
- `ALPHA_ADAPTIVE_SYNC_LIVE_ALLOCS_THRESHOLD=6200`
- `ALPHA_ADAPTIVE_PURGE_LIVE_ALLOCS_THRESHOLD=6800`
- `ALPHA_ADAPTIVE_PURGE_MIN_INTERVAL=8`
- `HELIOS_LIVE_ALLOC_SOFT_CAP=6200`
- `HELIOS_LIVE_ALLOC_HARD_CAP=7000`
- `HELIOS_MAX_OUTPUT_POOL_ENTRIES=1024`
- `HELIOS_MAX_BUFFER_POOL_ENTRIES=1024`

Latest observed state:

- step ~2340
- `loss≈4.7115`
- `best_val_loss≈4.7146`
- no catastrophic grad spike recurrence yet on this branch.

## M) 2026-03-05 Late Loop: Fleet Launch Failure + Event API Validation + New Live Run

### 1) Fleet binary launcher failure reproduced

Using `npm run fleet:train -- alpha-bench-l4-coopdbg-20260228084511 ...` on `/home/ajax/alpha` started and then exited immediately with:

- `error: vkCreateInstance failed`

Confirmed this was happening in the compiled `./alpha` path despite L4 defaults.

### 2) Fleet launcher hardening (local code)

Updated `apps/cli/src/commands/fleet.ts`:

- strip literal `--` separator from forwarded flags (prevents `./alpha train -- --foo=...`)
- prefer headless ICD when auto-setting Vulkan env on L4:
  - `/etc/vulkan/icd.d/nvidia_icd_headless.json`
  - fallback `/usr/share/vulkan/icd.d/nvidia_icd.json`

### 3) Recovery path to keep training unblocked

Because compiled-binary Vulkan init still failed on this host, switched to runtime from:

- `/home/ajax/alpha-repo`
- `node --expose-gc apps/cli/dist/main.js train ...`

Synced latest local source files to `/home/ajax/alpha-repo` and rebuilt key workspaces:

- `@alpha/core`
- `@alpha/tokenizers`
- `@alpha/autograd`
- `@alpha/helios`
- `@alpha/train`
- `@alpha/cli`

This ensured the active training process used latest allocator/autograd/event code, not stale dist artifacts.

### 4) New run started with conservative stability caps

Instance:

- `alpha-bench-l4-coopdbg-20260228084511` (NVIDIA L4, driver 590.48.01)

Run:

- run id: `super_chat_20260305132457_f0hn`
- run dir: `runs/super_chat_l4_loopfix_20260305_132456`
- log: `/home/ajax/alpha-repo/train_super_chat_loopfix_20260305_132456.log`

Config highlights:

- `lr=2e-5`, `lrMin=2e-6`, `warmupIters=1000`
- `gradClip=0.3`, `spikeThreshold=30`
- `batch=8`, `accumSteps=2`, `block=256`
- `evalInterval=200`, `sampleInterval=200`, `postSamples=true`
- `syncEvery=1`, `gcEvery=1`

Allocator/env caps used:

- `HELIOS_LIVE_ALLOC_SOFT_CAP=6200`
- `HELIOS_LIVE_ALLOC_HARD_CAP=7000`
- `HELIOS_MAX_BUFFER_POOL_ENTRIES=256`
- `HELIOS_MAX_OUTPUT_POOL_ENTRIES=256`
- `ALPHA_ADAPTIVE_SYNC_LIVE_ALLOCS_THRESHOLD=5800`
- `ALPHA_ADAPTIVE_PURGE_LIVE_ALLOCS_THRESHOLD=6200`

### 5) Live status snapshot (current)

Observed in log/API during this pass:

- progressed through `step 200, 400, 600`
- at API check: `step ~779`, `loss ~7.35`
- memory pressure stayed under hard cap with active purges instead of immediate allocator crash
- samples are emitted at 200-step cadence in log

### 6) Event log endpoint verified end-to-end

Confirmed endpoint now returns real run events:

- `GET /api/runs/super_chat_20260305132457_f0hn/events?last=20`
- returned non-empty events including:
  - `gpu_mem_purge`
  - `spike_skip`

Also validated ingest writes by posting a manual probe event and reading it back.

### 7) Outstanding

- Older stale super_chat runs are still marked `active` in API listings; status cleanup path needs deployment alignment on the web side.
- Compiled `/home/ajax/alpha/alpha` Vulkan startup issue remains unresolved on this host; node launcher from `/home/ajax/alpha-repo` is currently the reliable path.

## N) Resume-800 Branch Stabilized (run `super_chat_20260305132939_uvfs`)

After `super_chat_20260305132457_f0hn` entered repeated spike-skip loops around step ~930, training was stopped and resumed from:

- `runs/super_chat_l4_loopfix_20260305_132456/checkpoint-800.json`

Resume branch launch:

- run id: `super_chat_20260305132939_uvfs`
- run dir: `runs/super_chat_l4_loopfix_resume800_20260305_132938`
- log: `/home/ajax/alpha-repo/train_super_chat_loopfix_resume800_20260305_132938.log`

Resume settings:

- `lr=1e-5`, `lrMin=1e-6`, `warmupIters=200`
- `gradClip=0.25`, `spikeThreshold=100`
- `ALPHA_SPIKE_LR_BACKOFF=0.25`, `ALPHA_SPIKE_LR_MIN_SCALE=0.02`
- allocator caps retained (`HELIOS_LIVE_ALLOC_*`, adaptive purge/sync thresholds)

Observed behavior (good):

- passed prior failure zone (900+) without catastrophic spike storm
- continuing descent:
  - step 1000: `loss=7.1951`, `val_loss=7.1284`
  - step 1200: `loss=6.9425`, `val_loss=6.9699`
  - step 1400: eval/checkpoint/samples emitted normally
- latest API snapshot during this pass:
  - step `1434`, `loss=6.7774`, `grad_norm=0.3659`

Event API validation on this run:

- `GET /api/runs/super_chat_20260305132939_uvfs/events?last=10`
- returned expected training events:
  - `training_started`
  - `eval_summary`
  - `checkpoint_saved`
  - `samples_generated`

This confirms event ingestion + retrieval are functioning with the rebuilt `/home/ajax/alpha-repo` runtime path.

## O) Grad-Norm Fast-Path Issue Isolaton + Mitigation

Observed pattern across multiple branches:

- around step ~900-1700, grad norm suddenly locked to a near-constant huge value (`~11.8k` / `~11.9k`)
- repeated `spike_skip` events on nearly every step
- LR scale decayed to floor and run stopped making progress

Key finding:

- Resuming from checkpoint-1600 with `ALPHA_DISABLE_TOTAL_SUMSQ=1` removed the deterministic spike loop in the same step range.

Validation run:

- run id: `super_chat_20260305133407_q029`
- run dir: `runs/super_chat_l4_loopfix_resume1600_nototalsq_20260305_133407`
- resumed from: `runs/super_chat_l4_loopfix_resume800_20260305_132938/checkpoint-1600.json`
- env includes: `ALPHA_DISABLE_TOTAL_SUMSQ=1`

Observed in this branch:

- step 1700: `loss=6.7610`, `grad_norm=0.280`
- step 1800: `loss=6.8006`, `val_loss=6.6956`
- step 2200: eval/checkpoint/sample events all emitted normally
- no repeated high-constant spike_skip storm in this region

Action merged into code:

- `packages/train/src/trainer.ts` now includes an automatic fallback guard:
  - if repeated near-identical spike norms occur in fast-path mode,
  - disable `totalSumSq` grad-norm fast path and switch to per-parameter fallback,
  - emit `grad_norm_fastpath_disabled` event.

This allows recovery without manual env intervention in future runs.

## P) Root Cause Pass + Corrective Actions (2026-03-05 14:30 UTC)

### Confirmed root cause #1 (critical): resume config mismatch

The collapsed branch was resumed from a 2k-vocab checkpoint using 4k tokenizer artifacts, which silently changed the active model vocab shape and destabilized training.

Evidence:

- Good lineage config (`runs/super_chat_resume1600_strict_20260305_203101/config.json`): `vocabSize=2000`, tokenizer artifacts loaded from `runs/tokenizer-artifacts-super-chat-bpe2k-v2.json`.
- Bad lineage config (`runs/super_chat_l4_resume2400_nospike_20260305_133915/config.json`): `vocabSize=4000`, `packed=false`, `spikeThreshold=0`.
- Metrics shift happens exactly at first step of bad resume: stable run ended around loss ~4.71; bad resume starts ~6.67 then drifts into 8.x regime.

Code fix added:

- File: `packages/train/src/trainer.ts`
- New guard: `validateResumeModelCompatibility(...)`
- Enforces checkpoint/current model compatibility on resume (`vocabSize`, `blockSize`, `nLayer`, `nEmbd`, `nHead`, `ffnActivation`, `ffnDim`, `dropout`).
- Default behavior: throw and refuse resume on mismatch.
- Explicit override for migration experiments only: `ALPHA_ALLOW_RESUME_MISMATCH=1`.

### Run relaunch after fix

Launched from known-good checkpoint with matching tokenizer:

- Run dir: `runs/super_chat_rca_resume2400_20260305_140835`
- Resume: `runs/super_chat_resume1600_strict_20260305_203101/checkpoint-2400.json`
- Tokenizer artifacts: `runs/tokenizer-artifacts-super-chat-bpe2k-v2.json`
- Run id: `super_chat_20260305140836_zbq9`

Observed:

- Healthy zone recovered initially (loss ~4.69-4.78, val ~4.70-4.73).
- Later recurrence: giant spike-skip storm around step ~3355+.

### Secondary issue observed: late-run giant spike storms

Pattern:

- repeated large grad_norms (e.g., ~19k/~42k repeating) with many `spike_skip` events
- strong correlation with repeated `gpu_mem_purge` events before spike storm windows
- eventually loss jumps into ~7.7 plateau regime

This persisted even when `ALPHA_DISABLE_TOTAL_SUMSQ=1`, so totalSumSq fast path is not the only trigger.

Additional hardening added:

- File: `packages/train/src/trainer.ts`
- Improved fast-path auto-disable heuristics:
  - near-identical giant spikes now use relative tolerance (`1%`) instead of absolute `1e-3`
  - added massive-spike window detector (`>=3` huge spikes in 64-step window) to force fallback

### Follow-up run with hardening + totalSumSq disabled

- Run dir: `runs/super_chat_rca_resume3200_20260305_141719`
- Resume: `runs/super_chat_rca_resume2400_20260305_140835/checkpoint-3200.json`
- Env: `ALPHA_DISABLE_TOTAL_SUMSQ=1`
- Run id: `super_chat_20260305141720_xw0u`

Observed:

- Stable through 3400/3600/3800/4000 with val ~4.70-4.72.
- Spike storm still recurred around ~4100 with repeated giant norms and skip events.

### Current diagnostic branch (in progress)

To isolate whether norm/skip machinery itself is the collapse trigger, launched a controlled branch disabling grad-norm gating:

- Run dir: `runs/super_chat_rca_resume4000_nognorm_20260305_142600`
- Resume: `runs/super_chat_rca_resume3200_20260305_141719/checkpoint-4000.json`
- Env: `ALPHA_DISABLE_GRAD_NORM=1`, `ALPHA_DISABLE_TOTAL_SUMSQ=1`
- CLI: `--gradClip=0 --spikeThreshold=0`
- Status: startup tokenization phase at time of this log update (pre-step metrics)


## Q) Native Allocator Cap Crash Identified + Fixed (2026-03-05 14:43 UTC)

Observed crash on nopurge branch:

- Run: `super_chat_20260305143308_rhoj`
- Run dir: `runs/super_chat_rca_resume4800_nopurge_20260305_143307`
- Crash point: around step `5725`
- Error tail:
  - `[helios OOM] acquireBuffer failed: requesting 4.0MB`
  - `Max buffers reached`
  - live allocations at crash: `8192`

Root cause:

- Native hard cap in `packages/helios/native/helios_vk.c`:
  - `#define MAX_BUFFERS 8192`
- This cap was reached during long uninterrupted training despite low VRAM usage, causing fatal allocator failure.

Fix applied:

- File updated: `packages/helios/native/helios_vk.c`
- Changed:
  - `MAX_BUFFERS` from `8192` -> `65536`
- Remote rebuild completed on L4 host:
  - rebuilt `@alpha/helios` native addon
  - rebuilt `@alpha/train` and `@alpha/cli`

Post-fix relaunch:

- Resume checkpoint: `runs/super_chat_rca_resume4800_nopurge_20260305_143307/checkpoint-5600.json`
- New run dir: `runs/super_chat_rca_resume5600_nopurge_20260305_144259`
- Launch profile keeps the currently stable anti-collapse settings:
  - `ALPHA_DISABLE_GRAD_NORM=1`
  - `ALPHA_DISABLE_TOTAL_SUMSQ=1`
  - high live-alloc purge thresholds (`ALPHA_ADAPTIVE_*`, `HELIOS_LIVE_ALLOC_*`) to avoid purge-storm collapse path

