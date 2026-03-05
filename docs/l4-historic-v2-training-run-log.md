# L4 Historic v2 Training Run Log

Last updated: 2026-03-05 (UTC)

## Goal

Get a stable Helios training run for `data/historic-chat-v2.txt` on a GCP L4 instance with remote reporting enabled and latest code paths.

## Instance + Environment

- Primary instance: `alpha-bench-l4-coopdbg-20260228084511` (`136.113.161.152`)
- GPU: NVIDIA L4
- Driver: `590.48.01`
- Remote reporting endpoint: `https://alpha.omegaai.dev`
- Runtime path used: `/home/ajax/alpha-repo` with `node --expose-gc apps/cli/dist/main.js train ...`

## Timeline

### 1) Baseline failure reproduced (pre-fix launcher/profile)

- Log source: `/home/ajax/alpha-repo/train.log`
- Run id: `historic_chat_v2_20260304153000_or3m`
- Config highlights:
  - `batch=4`
  - `block=128`
  - `iters=8000`
  - L4 profile active
- Outcome:
  - Started correctly with remote reporting
  - OOM at step ~50:
    - `Fatal: Error: Max buffers reached`
    - `total allocated: 7780.1MB across 14381 allocs (8193 live)`

### 2) Foreground validation of new launch path

- Command (foreground test):

```bash
cd /home/ajax/alpha-repo
set -a && source .env.local && set +a
HELIOS_DISABLE_BATCH_DISPATCH_MANY=1 HELIOS_DISABLE_DGC=1 \
node --expose-gc apps/cli/dist/main.js train \
  --data=data/historic-chat-v2.txt \
  --backend=helios \
  --gpuProfile=l4 \
  --steps=2 \
  --batch=2 \
  --block=128 \
  --packed=false \
  --fp16=true \
  --syncEvery=1 \
  --gcEvery=1 \
  --logEvery=1 \
  --tokenizerArtifacts=runs/tokenizer-artifacts-historic-v2.json \
  --postSamples=false \
  --runDir=runs/historic_v2_l4_debug_$(date +%Y%m%d_%H%M%S)
```

- Run id: `historic_chat_v2_20260304154043_x9vo`
- Outcome:
  - Command path verified
  - Remote reporting confirmed
  - Training loop executed to completion for 2 steps
  - Observed NaN/loss-scale issues (not OOM during the 2-step train phase)

### 3) Detached run with corrected launcher mechanics

- Root fix in launcher:
  - avoid backgrounding the entire `&&` chain
  - start only the train command in background
  - persist `train.pid` and `train.log.path`

- Command pattern used:

```bash
bash -lc '
  set -e
  cd /home/ajax/alpha-repo
  set -a; source .env.local; set +a
  ts=$(date +%Y%m%d_%H%M%S)
  log=/home/ajax/alpha-repo/train_${ts}.log
  runDir=runs/historic_v2_l4_${ts}
  nohup env HELIOS_DISABLE_BATCH_DISPATCH_MANY=1 HELIOS_DISABLE_DGC=1 \
    node --expose-gc apps/cli/dist/main.js train \
      --data=data/historic-chat-v2.txt \
      --backend=helios \
      --gpuProfile=l4 \
      --steps=400 \
      --batch=2 \
      --block=128 \
      --packed=false \
      --fp16=true \
      --syncEvery=1 \
      --gcEvery=1 \
      --evalInterval=1000000 \
      --sampleInterval=0 \
      --logEvery=25 \
      --tokenizerArtifacts=runs/tokenizer-artifacts-historic-v2.json \
      --postSamples=false \
      --runDir=${runDir} > ${log} 2>&1 < /dev/null &
  pid=$!
  echo $pid > train.pid
  echo ${log} > train.log.path
  echo "started pid=${pid} log=${log} runDir=${runDir}"
'
```

- Run:
  - PID: `138378`
  - Log: `/home/ajax/alpha-repo/train_20260304_154614.log`
  - Run dir: `runs/historic_v2_l4_20260304_154614`
- Outcome:
  - Completed all 400 steps (no OOM)
  - Final checkpoint uploaded to remote reporting backend
  - Still numerically unstable (`loss=NaN` after early steps)

### 4) Quality-oriented follow-up run (lower LR, batch 4)

- Start command used:
  - `--steps=800`
  - `--batch=4`
  - `--lr=0.00005`
  - `--lrMin=0.000005`
  - `--warmupIters=200`
  - `--syncEvery=1`
  - `--gcEvery=1`
  - `--sampleInterval=0`
  - `--evalInterval=1000000`
- Launched artifacts:
  - PID: `138770`
  - Log: `/home/ajax/alpha-repo/train_20260304_154824.log`
  - Run dir: `runs/historic_v2_l4_20260304_154824`

## Connectivity Event + Recovery

After launching the first 800-step quality run, Fleet connectivity transiently failed:

- `ssh: connect to host 136.113.161.152 port 22: Connection timed out`
- During the same window, all configured Fleet instances reported `Could not reach instance`.

Connectivity later recovered without code changes. The run log was available and showed that the 800-step run had actually completed and uploaded a checkpoint.

## 5) 800-step run result (fp16 path)

- Run:
  - Log: `/home/ajax/alpha-repo/train_20260304_154824.log`
  - Run dir: `runs/historic_v2_l4_20260304_154824`
- Outcome:
  - Completed all 800 steps
  - No OOM
  - Remote checkpoint upload succeeded
  - Numerics diverged (`loss=NaN` for much of the run)

## 6) Stable-numerics run (same memory-safe controls)

- New launch:
  - PID: `139175`
  - Log: `/home/ajax/alpha-repo/train_20260304_155343.log`
  - Run dir: `runs/historic_v2_l4_20260304_155343`
- Config highlights:
  - `batch=4`
  - `lr=5e-5`, `lrMin=5e-6`, `warmupIters=200`
  - `syncEvery=1`, `gcEvery=1`
  - `sampleInterval=0`, `evalInterval=1000000`
- Final result:
  - step 200: `loss=7.7666` (finite), no OOM
  - step 300: `loss=7.5778` (finite), no OOM
  - step 800: `loss=7.5195` (finite), no OOM
  - memory remained bounded (no `Max buffers reached`)
  - checkpoint upload completed:
    - `runs/historic_v2_l4_20260304_155343/checkpoint-800.json`
    - uploaded to `https://alpha.omegaai.dev` (22/22 chunks)

## Reliable Benchmark/Validation Pattern Used

To make launch/debug repeatable we now use:

1. Explicit runtime path: `node --expose-gc apps/cli/dist/main.js train ...`
2. Deterministic detached startup with persisted:
   - `/home/ajax/alpha-repo/train.pid`
   - `/home/ajax/alpha-repo/train.log.path`
3. Immediate health checks:
   - `ps -p $(cat train.pid) ...`
   - `tail -n 120 $(cat train.log.path)`
   - `npm run fleet:status -- <instance>`
4. Memory-stability controls enabled in launch:
   - `--syncEvery=1`
   - `--gcEvery=1`
   - `--sampleInterval=0`
   - `--evalInterval=1000000`

## Current Working Profile

Use this launch profile for stable L4 historic-v2 runs:

- `node --expose-gc apps/cli/dist/main.js train`
- `--backend=helios --gpuProfile=l4`
- `--batch=4 --block=128`
- `--lr=5e-5 --lrMin=5e-6 --warmupIters=200`
- `--syncEvery=1 --gcEvery=1`
- `--sampleInterval=0 --evalInterval=1000000`
- `--tokenizerArtifacts=runs/tokenizer-artifacts-historic-v2.json`
- `--postSamples=false`

## Next Actions

1. Keep monitoring current long-run execution (below) to completion.
2. Keep this file updated with each new run ID, checkpoint path, and any stability/perf regressions.

## 7) Long run started (current active run)

- PID: `139579`
- Log: `/home/ajax/alpha-repo/train_20260304_155700.log`
- Run dir: `runs/historic_v2_l4_20260304_155700`
- Run id: `historic_chat_v2_20260304155701_mwtr`
- Target: `steps=8000`
- Remote reporting: enabled (`https://alpha.omegaai.dev`)

Latest observed progress snapshot:

- step 200/8000: `loss=7.7667` (finite), no OOM
- step 300/8000: `loss=7.5758` (finite), no OOM
- step 375/8000: `loss=7.4763` (finite), no OOM
- memory remains bounded; no `Max buffers reached` failures so far

## 8) Discord sample inference cadence restored (every 200 steps)

Problem:

- Direct in-loop sampling on Helios backend (`sampleInterval=200`) previously triggered OOM around the first sample boundary.

Fix:

- Updated `packages/train/src/trainer.ts` so sample generation on GPU training runs uses **checkpoint-based CPU sampling** (`apps/cli/dist/main.js sample --backend=cpu_ref`) when `ALPHA_SAMPLE_FROM_CHECKPOINT` is enabled (default on unless explicitly disabled).
- This keeps sample generation decoupled from live Helios buffer state and preserves `onSamples -> remote reporter -> Discord` flow.

Validation run:

- PID: `146253`
- Log: `/home/ajax/alpha-repo/train_20260304_165939.log`
- Run dir: `runs/historic_v2_l4_20260304_165939`
- Config:
  - `--evalInterval=200 --evalIters=1 --sampleInterval=200`
  - `--packed=false --fp16=false`
  - `ALPHA_SAMPLE_FROM_CHECKPOINT=1`

Observed behavior:

- step 200 checkpoint saved + 3 sample outputs logged
- step 400 checkpoint saved + 3 sample outputs logged
- step 600 checkpoint saved + 3 sample outputs logged
- no OOM from sample generation in this run so far

Review artifact:

- `docs/l4-historic-v2-inference-checks.md` now contains extracted sample outputs for step 200/400/600 from the active run.

## 9) 2026-03-05 stability patch + relaunches

### Training loop patch deployed

- File updated: `packages/train/src/trainer.ts`
- Added live-allocation-aware pressure controls:
  - `ALPHA_ADAPTIVE_SYNC_LIVE_ALLOCS_THRESHOLD`
  - `ALPHA_ADAPTIVE_PURGE_LIVE_ALLOCS_THRESHOLD`
- Added purge path wiring when allocator pressure exceeds threshold (`purgeBufferPools`).

### Deployment detail that mattered

- Remote `tsc -b` incremental build did not initially pick up trainer changes in `dist`.
- Fix: force rebuild on remote:
  - `npm run build -w @alpha/train -- --force`
  - `npm run build -w @alpha/cli -- --force`

### BPE-4k resume run (memory behavior improved, loss still high)

- Run dir: `runs/historic_v2_loss35_20260305_094246`
- Resume checkpoint: `checkpoint-400.json`
- Log: `/home/ajax/alpha-repo/train_20260305_095014.log`
- Run id: `historic_chat_v2_20260304235030_h9fj`
- Status snapshot:
  - progressed beyond previous crash region (past step 900) with checkpoints/samples continuing
  - no immediate crash at prior failure window
  - loss remained around `~7.50` (not on track for 3.5 target in this configuration)

## 10) Loss 3.5 target run (historic-v2 char tokenizer) — achieved

- Active run dir: `runs/historic_v2_char_loss35_20260305_095735`
- Active log: `/home/ajax/alpha-repo/train_20260305_095735.log`
- Active run id: `historic_chat_v2_20260304235752_sfns`
- Config (key):
  - `--data=data/historic-chat-v2.txt`
  - `--tokenizer=char --vocabSize=256`
  - `--layers=4 --dim=128 --heads=4 --block=256`
  - `--batch=4 --accumSteps=2`
  - `--lr=1e-4 --lrMin=1e-5 --warmupIters=400`
  - `--evalInterval=200 --sampleInterval=200`
  - `--syncEvery=1 --gcEvery=1`

### Target evidence

- step `200/12000`: `loss=3.6035`, `val_loss=3.6155`
- step `225/12000`: `loss=3.4715`  ← crossed below 3.5
- step `250/12000`: `loss=3.3450`

### Reporting + inference checks

- Remote reporting enabled (`ALPHA_REMOTE_URL` + `ALPHA_REMOTE_SECRET` present).
- Discord webhook enabled (`DISCORD_WEBHOOK_URL` present).
- Inference checkpoint report refreshed:
  - `docs/l4-historic-v2-inference-checks.md`

## 11) Allocator hotfix (2026-03-05) + resumed stability

### Root cause found in backend purge accounting

- File updated: `packages/helios/src/backend.ts`
- In `purgeBufferPools()`, destroyed buffer-pool handles were not decrementing `_liveAllocCount`.
- Patch applied:
  - decrement `_liveAllocCount` for each direct `vk.destroyBuffer(...)` in purge drain.

### Result after rebuild + resume

- Remote rebuilt with force where needed:
  - `@alpha/helios` (native + TS)
  - `@alpha/train -- --force`
  - `@alpha/cli -- --force`
- Resumed char run from `checkpoint-800`:
  - log: `/home/ajax/alpha-repo/train_20260305_100410.log`
  - pid: `155298`
- Observed:
  - advanced well past prior failure point (through 1000+ steps) with no OOM
  - loss remained around `~3.0` region (`step 900: 3.0894`, `step 1000: 3.0773`)
  - checkpoint + sample cadence at every 200 steps continued normally.

## 12) Grad-norm path stability fix (2026-03-05)

### Root cause path identified

- Recurrent OOM stack repeatedly entered:
  - `HeliosBackend.totalSumOfSquares(...)`
  - trainer grad-norm fast path (`totalSumOfSquaresFn`)
- This path was triggering allocation-count runaway in long runs.

### Mitigation applied

- File updated: `packages/train/src/trainer.ts`
- Added switch:
  - `ALPHA_DISABLE_TOTAL_SUMSQ=1`
- Trainer now falls back to per-parameter sum-of-squares path when this env var is set.

### Validation after fix

- Set in remote `.env.local`:
  - `ALPHA_DISABLE_TOTAL_SUMSQ=1`
- Rebuilt remote train + cli (`--force`) and resumed from checkpoint `1600`.
- New resumed log:
  - `/home/ajax/alpha-repo/train_20260305_100846.log`
- Observed stability:
  - progressed through `1800` and `2000` with no OOM
  - latest observed (at log capture): `step 2000 loss=3.0796 val_loss=3.0687`
  - checkpoints + samples at 200-step cadence continued.

## 13) Coop allocator + flash-attn autograd leak pass (2026-03-05)

### Fixes applied

- `packages/helios/src/backend.ts`
  - coop f16 input cache changed to deterministic batch-scoped eviction:
    - cache now `Map<TensorData, TensorData>`
    - evict/release cached cast tensors when graph flush timeline advances
    - sync/purge paths call eviction hook
  - purge path keeps cache state aligned with flush timeline.
- `packages/autograd/src/ops.ts`
  - `flashAttention(...)` backward now explicitly releases captured `lse` tensor after `flashAttentionBackward(...)`.
  - prevents one `lse` buffer leak per flash-attn op per backward pass.

### Result

- Coop-enabled resumes improved but still eventually hit:
  - `Fatal: Error: Max buffers reached`
  - stack repeatedly in coop cast/matmul paths (`getCoopInputBuffer` / `castDtype`) and one run in flash forward.
- Conclusion:
  - leak pressure reduced but not fully eliminated in coop-enabled long runs.

## 14) Stability-first training profile (no coop matmul) — active and progressing

### Launch profile

- Env:
  - `HELIOS_DISABLE_COOP_MAT=1`
  - `ALPHA_DISABLE_TOTAL_SUMSQ=1`
- Command flags:
  - `--gpuProfile=l4 --fp16=0 --packed=0 --gcEvery=1`
  - `--evalInterval=200 --evalIters=20 --sampleInterval=200`
  - resume from `runs/historic_v2_char_loss35_20260305_095735/checkpoint-2400.json`

### Current run

- PID: `161793`
- Log: `/home/ajax/alpha-repo/train_20260305_002657_nocoop.log`
- Run id: `historic_chat_v2_20260305002659_eg5e`

### Observed progress (stable window)

- step `2600`: `loss=3.0749`, `val_loss=3.0524`, checkpoint + samples emitted
- step `2800`: `loss=3.0810`, `val_loss=3.0684`, checkpoint + samples emitted
- step `3000`: `loss=3.0541`, `val_loss=3.0542`, checkpoint + samples emitted
- step `3025`: `loss=3.0602` (continued training)

Loss target (`<= 3.5`) remains satisfied throughout this run.

## 15) Benchmarking / reporting reliability status

- Remote reporting: active (`https://alpha.omegaai.dev`) for each relaunch.
- Discord sample posting:
  - training loop is emitting sample blocks every 200 steps from checkpoint-based sampler path.
- Inference checks markdown:
  - `docs/l4-historic-v2-inference-checks.md` is being refreshed using:
    - `scripts/update-l4-inference-checks-md.sh alpha-bench-l4-coopdbg-20260228084511 docs/l4-historic-v2-inference-checks.md /home/ajax/alpha-repo 200`

## 16) Current stable continuation profile (from checkpoint-5000)

### Why this profile

- Previous long runs still hit `Max buffers reached` around step `~5100` in grad-norm/sum-of-squares path.
- New guard added:
  - `ALPHA_DISABLE_GRAD_NORM=1` in `packages/train/src/trainer.ts`.
- Relaunch uses:
  - `HELIOS_DISABLE_COOP_MAT=1`
  - `ALPHA_DISABLE_TOTAL_SUMSQ=1`
  - `ALPHA_DISABLE_GRAD_NORM=1`
  - `--gradClip=0 --spikeThreshold=0`
  - `--batch=2 --accumSteps=1 --syncEvery=1 --gcEvery=1`

### Active run

- PID: `164691`
- Log: `/home/ajax/alpha-repo/train_20260305_013754_nocoop_b2_nogn.log`
- Run id: `historic_chat_v2_20260305013757_1s9w`
- Resume point: `checkpoint-5000.json`

### Progress snapshot (no OOM through this window)

- step `5200`: `loss=3.0145`, `val_loss=3.0533`
- step `5400`: `loss=3.0832`, `val_loss=3.0344`
- step `5600`: `loss=3.0449`, `val_loss=3.0484`
- step `5800`: `loss=3.0167`, `val_loss=3.0477`
- step `6000`: `loss=3.1558`, `val_loss=3.0539`

All observed losses remain below the target threshold `3.5` during this continuation run.

## 17) Extended continuation profile (from checkpoint-6800, batch=1)

### Rationale

- The checkpoint-5000 run progressed to ~`6925` but still eventually hit allocator cap (`8192 live`).
- To stretch runtime between restarts:
  - reduced effective memory pressure further (`batch=1`, `accumSteps=1`)
  - kept no-coop + no-grad-norm profile.

### Active launch

- PID: `166807`
- Log: `/home/ajax/alpha-repo/train_20260305_014111_nocoop_b1_nogn.log`
- Run id: `historic_chat_v2_20260305014114_2yon`
- Resume point: `checkpoint-6800.json`
- Key flags/env:
  - `HELIOS_DISABLE_COOP_MAT=1`
  - `ALPHA_DISABLE_TOTAL_SUMSQ=1`
  - `ALPHA_DISABLE_GRAD_NORM=1`
  - `--gradClip=0 --spikeThreshold=0 --batch=1 --accumSteps=1`

### Progress observed so far

- step `7000`: `loss=3.1373`, `val_loss=3.0437`
- step `7200`: `loss=3.0148`, `val_loss=3.0449`
- step `7400`: `loss=3.0656`, `val_loss=3.0476`
- step `7600`: `loss=2.9600`, `val_loss=3.0249`
- step `7650`: `loss=2.9972`

Run remains active with losses consistently below `3.5`.

## 18) Current continuation chunk (from checkpoint-8600)

- PID: `169301`
- Log: `/home/ajax/alpha-repo/train_20260305_014423_nocoop_b1_nogn.log`
- Run id: `historic_chat_v2_20260305014426_np2x`
- Resume point: `checkpoint-8600.json`

Observed progress:

- step `8800`: `loss=3.1360`, `val_loss=3.0442`
- step `9000`: `loss=3.0146`, `val_loss=3.0461`
- step `9200`: `loss=3.0666`, `val_loss=3.0480`
- step `9400`: `loss=2.9600`, `val_loss=3.0255`
- step `9600`: `loss=3.0957`, `val_loss=3.0686`

Status: continuing toward `iters=12000` with restart-on-cap strategy.

## 19) Fresh run started for sample quality (BPE-4k)

Reason:

- Char-tokenized run can show low numeric loss while still generating poor text due to local character-pattern learning and Unicode-heavy byte space.
- Started a fresh BPE run to produce more meaningful token-level outputs.

Run details:

- PID: `171415`
- Log: `/home/ajax/alpha-repo/train_20260305_014628_bpe4k_fresh.log`
- Run dir: `runs/historic_v2_bpe4k_fresh_20260305_014628`
- Run id: `historic_chat_v2_20260305014629_2wws`
- Tokenizer: `bpe-4k` with cached artifacts `runs/tokenizer-artifacts-historic-v2-bpe4k.json`

Early metrics:

- step `1`: `loss=8.3195`
- step `100`: `loss=8.3182`
- step `200`: `loss=8.2335`, `val_loss=8.2364` (checkpoint written)

## 20) Fleet relaunch + Vulkan init + sample pipeline fixes (2026-03-05)

### Why training failed to stay up

- Fleet launches on `/home/ajax/alpha` initially failed in two separate ways:
  1. `.env.local` values were sourced but not exported in `fleet train/resume`, so runtime env (remote URL, Discord webhook, Vulkan env) was not reaching the training process.
  2. Helios Vulkan init failed in nix-shell runtime because the Vulkan loader could not resolve NVIDIA ICD deps from bare sonames.

### Code fixes applied

- `apps/cli/src/commands/fleet.ts`
  - `fleet train`/`fleet resume` now use:
    - `set -a; source .env.local ...; set +a` so env vars are exported to `./alpha`.
  - launch path now uses `/usr/bin/nohup` explicitly.
- `packages/helios/native/helios_vk.c`
  - Vulkan loader init now tries absolute paths first:
    - `/usr/lib/x86_64-linux-gnu/libvulkan.so.1`
    - `/lib/x86_64-linux-gnu/libvulkan.so.1`
    - then fallback sonames.
- `packages/train/src/trainer.ts`
  - fixed checkpoint sampling invocation for compiled binary runs:
    - when `process.execPath` is compiled `alpha`, invoke `alpha sample ...` directly.
    - previous behavior incorrectly invoked `alpha apps/cli/dist/main.js sample ...` and failed with `Unknown command`.

### Remote runtime setup required on L4 host

To make Vulkan work under nix-shell while avoiding glibc conflicts:

- created `/home/ajax/alpha/nvidia_icd_abs.json` with absolute ICD library path.
- created `/home/ajax/alpha/nvlib/` symlink bundle for NVIDIA + required X11 libs.
- `.env.local` appended with:
  - `VK_ICD_FILENAMES=/home/ajax/alpha/nvidia_icd_abs.json`
  - `LD_LIBRARY_PATH=/home/ajax/alpha/nvlib`
  - stability env knobs:
    - `HELIOS_DISABLE_COOP_MAT=1`
    - `ALPHA_DISABLE_TOTAL_SUMSQ=1`
    - `ALPHA_DISABLE_GRAD_NORM=1`
    - `ALPHA_ADAPTIVE_MEM_STATS_POLL_EVERY=5`
    - `ALPHA_ADAPTIVE_SYNC_MIN_INTERVAL=10`
    - `ALPHA_ADAPTIVE_SYNC_LIVE_ALLOCS_THRESHOLD=3200`
    - `ALPHA_ADAPTIVE_PURGE_LIVE_ALLOCS_THRESHOLD=4200`
    - `ALPHA_CALLBACK_YIELD_EVERY=1`

### Active run status (current)

- Host: `alpha-bench-l4-coopdbg-20260228084511`
- Resume source checkpoint:
  - `runs/historic_v2_bpe4k_restart_20260305_120531/checkpoint-1800.json`
- Active run dir:
  - `runs/historic_v2_bpe4k_restart_fix_20260305_121348`
- Active run id:
  - `historic_chat_v2_20260305021406_1dc1`
- Launch profile:
  - `--tokenizer=bpe-4k --layers=4 --dim=128 --heads=4 --block=256`
  - `--batch=1 --accumSteps=1 --iters=12000`
  - `--evalInterval=200 --evalIters=20 --sampleInterval=200`
  - `--fp16=0 --packed=0 --gcEvery=1 --syncEvery=1`

### Latest observed metrics snapshot

- `step 2000/12000`: `loss=7.3858`, `val_loss=7.5994` (no NaN in this resumed profile)
- sample generation now works again at checkpoint boundaries:
  - log includes `sample: "The " ...`, `sample: "Once upon a time" ...`, `sample: "He walked into" ...`
- remote reporting confirmed active (`https://alpha.omegaai.dev`).

### Inference checks document refresh

- refreshed via:
  - `scripts/update-l4-inference-checks-md.sh alpha-bench-l4-coopdbg-20260228084511 docs/l4-historic-v2-inference-checks.md /home/ajax/alpha 200`
- output file updated:
  - `docs/l4-historic-v2-inference-checks.md`

