# Alpha — GPT training system

Domain: **alpha.omegaai.dev**
API endpoints: **https://alpha.omegaai.dev/api** (dashboard, inference, training ingest, uploads)
OpenAI-compatible: **https://alpha.omegaai.dev/v1** (chat completions, model list)

## Philosophy

Alpha is **hand-written from scratch**. Every component — tensors, autograd, model, tokenizers, training loop — is custom TypeScript with minimal to no external dependencies. This is intentional.

- **Pure JavaScript/TypeScript** — no heavy ML frameworks, no ONNX runtime wrappers, no "just use PyTorch"
- **Super configurable** — every layer of the stack should be tunable and swappable
- **Zero dependencies** — prefer writing code over adding packages. No npm deps for core functionality. If we need GPU access, we write the native addon and SPIR-V assembler ourselves
- **Understand everything** — no black boxes. If it's in the codebase, we wrote it and we understand it
- **Scratch-built** — the value is in building it, not in importing it
- **Never take shortcuts** — if a feature needs a new GPU kernel, write the SPIR-V kernel. Don't compose workarounds from existing ops when a dedicated kernel would be faster. We build everything from scratch and should embrace the complexity — new kernels, new ops, new backends. Aim to do it better than existing solutions, not just equivalent
- **Build it right** — expect to implement complex things. A proper `clamp` kernel is better than chaining `neg` + `clamp_min` + `neg`. A fused attention kernel is better than 6 separate dispatches. Don't fear the SPIR-V
- **Scale from scratch** — this system is designed to eventually run on 100+ H100s at 90%+ utilization. Every scaling feature (data parallelism, tensor parallelism, pipeline parallelism, collective ops, gradient compression) must be custom-built. No NCCL, no DeepSpeed, no Megatron imports. We write our own all-reduce, our own ring communication, our own sharded optimizer. See `scale.md` for the full roadmap

## Architecture

Monorepo: `npm` workspaces + Turbo. TypeScript ESM throughout.

```
packages/core       — types, configs, domains, RNG
packages/tensor     — CPU tensor backend
packages/helios     — GPU compute backend (Vulkan native addon + SPIR-V from TS)
packages/autograd   — automatic differentiation
packages/model      — GPT model (init, forward, params)
packages/tokenizers — BPE, char, word tokenizers
packages/train      — training loop, checkpointing, data loading
packages/db         — Turso/libsql database layer
apps/server         — inference server (Hono, AI SDK provider)
apps/web            — Next.js dashboard (Tailwind, App Router, dark theme)
apps/hf             — HF Spaces inference server (standalone, optimized, no autograd)
apps/tui            — terminal dashboard (Ink/React)
apps/cli            — CLI commands
```

## Do

- **Maximize GPU utilization** — training should use at least 80% of available GPU resources (VRAM, compute). Don't waste money on rented GPUs by under-utilizing them. Tune batch size, block size, and model dimensions to fill the GPU.
- Use `@alpha/core` types (`ModelConfig`, `TrainConfig`, `DomainConfig`) as source of truth
- Read existing code before modifying — patterns are intentional
- Use `@libsql/client` for all DB access via `@alpha/db` (createDb/getDb)
- **Always use the alpha remote** when running training — set `ALPHA_REMOTE_URL` and `ALPHA_REMOTE_SECRET` env vars so live metrics stream to https://alpha.omegaai.dev/training for real-time monitoring
- Set env vars in `.env.local` for local dev (gitignored)
- Set prod env vars via `railway variables set`
- Run `npx tsc -b packages/X/tsconfig.json` to build a single package
- Run `npx tsc -p packages/X/tsconfig.json --noEmit` to type-check without emitting
- Keep modules small and focused — one concern per file
- Use `INSERT OR IGNORE` / `ON CONFLICT` for idempotent DB writes
- Batch DB inserts in chunks (500 rows) for metrics

## Don't

- Don't hardcode secrets — use `TURSO_DATABASE_URL` / `TURSO_AUTH_TOKEN` env vars
- Don't import from `dist/` — always import from package names (`@alpha/core`, `@alpha/db`)
- Don't add `effect` dependency to new packages unless necessary (only core uses it)
- Don't break the TUI or server — they must keep working after changes
- Don't use CJS — everything is ESM (`"type": "module"`)
- Don't run `PRAGMA journal_mode=WAL` on remote Turso connections
- Don't duplicate types — if `@alpha/core` has it, import it

## Chat Training Data

Primary dataset: **`data/super_chat.txt`** (91MB, 226k conversations) — combined from `dailydialog_proper.txt`, `discord_chat.txt`, and `historic-chat-v2.txt`. Validated and clean:
- Every conversation starts with `<|user|>` and ends with `<|assistant|>` turn + `<|end_of_text|>`
- Strict user/assistant alternation
- No empty turns, no conversations ending on a user turn

Run `npx tsx scripts/validate-chat-data.ts <file>` to validate any chat data file, or `--fix` to auto-clean it.

## GPU Training (GCP)

Training runs on **GCP** via `scripts/gcp_train.py`. The script handles the full lifecycle: provision instance, sync code, build, upload dataset, train, download results.

See **`training-runs.md`** for active/recent run tracking and GPU quota status.

### Machine types

| Machine | GPU | Cost | Notes |
|---------|-----|------|-------|
| `g2-standard-4` | L4 24GB | ~$0.70/hr | Default fallback |
| `a2-ultragpu-1g` | A100 80GB | ~$1.10/hr | Needs quota (currently 0) |

### SSH access

```bash
export PATH="$HOME/google-cloud-sdk/bin:$PATH"
gcloud compute ssh alpha-train --project=$GCP_PROJECT --zone=<zone> --command="<cmd>"
```

Code: `~/alpha/`, logs: `~/alpha/runs/<run_dir>.log`, checkpoints: `~/alpha/runs/<run_dir>/checkpoint-<step>.json`

Inference from checkpoint while training (use `cpu_ref` — GPU is locked):
```bash
gcloud compute ssh alpha-train --project=$GCP_PROJECT --zone=<zone> \
  --command="cd ~/alpha && node apps/cli/dist/main.js sample \
    --checkpoint=runs/<run_dir>/checkpoint-<step>.json \
    --backend=cpu_ref --steps=80 --temp=0.8 --topk=40 \
    --prompt='<|user|> Hello <|assistant|>'"
```

### Commands

```bash
# Train chat model on L4:
python3 scripts/gcp_train.py --data data/super_chat.txt --domain chat \
  --iters 50000 --batch 20 --block 512 --dim 384 --heads 8 --layers 8 \
  --backend helios --zone us-central1-b --machine-type g2-standard-4 --stop-after

# Train on A100 (when quota available):
python3 scripts/gcp_train.py --data data/super_chat.txt --domain chat \
  --iters 50000 --batch 20 --block 512 --dim 384 --heads 8 --layers 8 \
  --backend helios --zone us-central1-c --stop-after

# Instance management:
python3 scripts/gcp_train.py --action status     # check instance
python3 scripts/gcp_train.py --action stop        # stop (disk persists)
python3 scripts/gcp_train.py --action start       # resume stopped instance
python3 scripts/gcp_train.py --action ssh         # interactive SSH
python3 scripts/gcp_train.py --action delete      # destroy instance + disk
```

### Remote metrics

Export these env vars before running `gcp_train.py` (it reads `os.environ`, not `.env.local`):
- `ALPHA_REMOTE_URL` — API server for metrics streaming
- `ALPHA_REMOTE_SECRET` — auth token (same as UPLOAD_SECRET)
- `DISCORD_WEBHOOK_URL` — Discord webhook for notifications

Values are in `.env.local`.

## Key env vars

| Var | Where | Purpose |
|-----|-------|---------|
| `TURSO_DATABASE_URL` | .env.local, Railway | Turso libsql connection URL |
| `TURSO_AUTH_TOKEN` | .env.local, Railway | Turso auth token (rw) |
| `UPLOAD_SECRET` | Railway | Auth token for ingest/upload endpoints |
| `ALPHA_REMOTE_URL` | .env.local, training pod | API server URL for metrics streaming (use `https://alpha.omegaai.dev`) |
| `ALPHA_REMOTE_SECRET` | .env.local, training pod | Same as UPLOAD_SECRET on server |
| `DISCORD_WEBHOOK_URL` | .env.local, training instance | Discord webhook for training notifications + inference samples |

## Discord Notifications

Training posts to Discord via `DISCORD_WEBHOOK_URL` (set in `.env.local`) when:
- A run starts (model config, params, hyperparams)
- Inference samples are generated (at each checkpoint interval)
- Training completes

When the user asks to "post to Discord", always post to the **#training-and-evals** channel. Channel and guild IDs are configured via environment variables. This channel is used for training updates, run summaries, eval results, and team communication about model progress.

## Deploy

### Railway (dashboard + API)

All deployments are on **Railway**.

- **Single service** (web + API consolidated): `railway service alpha-web && railway up`
- Training runs stored in `outputs/` locally, synced to Turso via `@alpha/db` syncFromDisk

### HF Spaces (inference)

Live at **https://ajaxdavis-alpha-v0-historic.hf.space** — OpenAI-compatible inference API on free `cpu-basic` (2 vCPU, 16GB RAM).

**Architecture**: `apps/hf/` is a standalone Hono server with a dedicated inference engine (`inference.ts`) that bypasses all training machinery (no autograd, no tape). Uses KV cache, tiled matmul, zero-alloc decode loop. ~50ms/token on cpu-basic.

**Deploy**: Push to the HF Space git repo. HF rebuilds the Docker image automatically.

```bash
# Clone, update, push
git clone https://huggingface.co/spaces/ajaxdavis/alpha-v0-historic /tmp/hf-space
# Copy updated files from apps/hf/ and packages/{core,tokenizers}/ into the clone
# The HF repo mirrors: Dockerfile, README.md, package.json, turbo.json, tsconfig.base.json,
# package-lock.json, packages/{core,tokenizers}/, apps/hf/
cd /tmp/hf-space && git add -A && git commit -m "description" && git push
```

- Dockerfile downloads checkpoint from HF model repo `ajaxdavis/alpha-v0-historic` at build time
- Only `@alpha/core` and `@alpha/tokenizers` are needed (no tensor/autograd/model)
- Regenerate `package-lock.json` in the HF repo after changing deps: `npm install --package-lock-only`

**Test**:
```bash
# Health
curl https://ajaxdavis-alpha-v0-historic.hf.space/
# Models
curl https://ajaxdavis-alpha-v0-historic.hf.space/v1/models
# Chat completion (timed)
curl -w "\nTime: %{time_total}s\n" -X POST https://ajaxdavis-alpha-v0-historic.hf.space/v1/chat/completions \
  -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":30}'
# Streaming
curl -N -X POST https://ajaxdavis-alpha-v0-historic.hf.space/v1/chat/completions \
  -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":30,"stream":true}'
```

## DB

Turso cloud: set via `TURSO_DATABASE_URL` env var (see `.env.local`)

Tables: `runs`, `metrics` (WITHOUT ROWID), `checkpoints` (WITHOUT ROWID), `domains`, `schema_version`

View: `run_summary` (joins runs + checkpoint/metric counts)

Schema managed by version-based migrations in `packages/db/src/schema.ts`.

## Canonical Commands

- Install: `npm install`
- Build all: `npm run build`
- Typecheck: `npm run typecheck`
- Compile binary: `npm run bun:compile`
- Benchmark loop: `scripts/run-compiled-benchmark.sh 100`
- Adaptive tuning sweep: `npm run perf:tune:adaptive`
- Helios vs CUDA (local): `npm run bench:cuda -- --iters=12`
- Helios vs CUDA (fleet L4 cycle): `npm run fleet:bench:cuda -- --shutdown=delete`

## Fleet Operations (Remote Instances)

Prereqs:

- `fleet.json` must exist at repo root (copy `fleet.json.example` and fill host/key/user).
- Build CLI before Fleet operations: `npm run build -w @alpha/cli`.

Core commands:

- Dashboard: `npm run fleet`
- Status: `npm run fleet:status -- <instance>`
- Deploy binary/native addon: `npm run fleet:deploy -- <instance>`
- Force remote native rebuild: `npm run fleet:deploy -- <instance> --rebuild-native`
- Deploy all instances: `npm run fleet -- deploy --all`
- First-time setup (Nix + train shell warmup): `npm run fleet:setup -- <instance>`
- Start training: `npm run fleet:train -- <instance> --data=<path> ...`
- Resume latest run: `npm run fleet:resume -- <instance>`
- Resume specific run: `npm run fleet:resume -- <instance> --run=<run-name>`
- Stop training: `npm run fleet:stop -- <instance>`
- Logs (last lines): `npm run fleet:logs -- <instance>`
- Logs follow: `npm run fleet:logs -- <instance> -f`
- SSH: `npm run fleet:ssh -- <instance>`
- Run remote command: `npm run fleet:run -- <instance> -- <cmd>`
- Sync SSH key: `npm run fleet -- sync-keys <instance>`
- Download run dir: `npm run fleet:download -- <instance> --run=<run-name>`

Expected workflow:

1. `npm run fleet -- sync-keys <instance>` (once per machine).
2. `npm run fleet:deploy -- <instance>` (builds via `bun:compile`, uploads `alpha` + prebuilt `helios_vk.node`, uploads `.env.local` if present).
3. `npm run fleet:setup -- <instance>` only when Nix shell tooling is required (training env warmup, remote native rebuild, etc.).
4. `npm run fleet:train -- <instance> ...` or `npm run fleet:resume -- <instance>`.
5. Monitor with `fleet logs`, stop with `fleet stop`, and pull artifacts with `fleet download`.

Operational notes:

- `fleet train`/`fleet resume` run detached via `nohup` and write `train.log` under remote `deployDir`.
- On L4 instances, Fleet auto-applies default flags unless explicitly overridden.
- Use `--force` on `fleet train`/`fleet resume` only when intentionally bypassing running-process checks.
- `fleet deploy` now skips unchanged uploads via SHA-256 checks (binary/addon/source/env/flake files), which significantly speeds repeated deploy loops.
- Coop matmul is default-on when supported; set `HELIOS_DISABLE_COOP_MAT=1` for forced-disable diagnostics.

## Compiled Binary Rules

- Always compile via `npm run bun:compile`.
- Do not compile directly from `apps/cli/src/main.ts`.
- `bun:compile` is intentionally wired to:
  1. build `@alpha/cli` first,
  2. build Helios native addon,
  3. compile from `apps/cli/dist/main.js`.

## Performance Loop Gate (Required)

For any performance change:
1. Run `scripts/run-compiled-benchmark.sh 100`.
2. Run 3 compiled inference prompts from latest checkpoint.
3. Record/verify metrics in `perf/compiled-loop-history.csv`.
4. Commit only after both pass.

## Benchmark Policy

- Default strict smoke behavior is expected:
  - `FAIL_ON_SMOKE_TEST=1` in benchmark harness.
- Status ranking for attempt selection:
  - `ok > unstable > smoke_fail > failed`
- Diagnostic override when needed:
  - `FAIL_ON_SMOKE_TEST=0 scripts/run-compiled-benchmark.sh 100`

## Adaptive Tuning

Runtime knobs (no source edits needed):

- `ALPHA_ADAPTIVE_MEM_STATS_POLL_EVERY`
- `ALPHA_ADAPTIVE_SYNC_MIN_INTERVAL`
- `ALPHA_ADAPTIVE_SYNC_DEFERRED_THRESHOLD`
- `ALPHA_ADAPTIVE_SYNC_PENDING_THRESHOLD`
- `ALPHA_GPU_METRICS_SAMPLE_EVERY`
- `ALPHA_CALLBACK_YIELD_EVERY`
- `ALPHA_FAIL_ON_SMOKE_TEST`

Tuner behavior:

- `scripts/tune-adaptive-env-loop.sh` defaults to `TUNE_REQUIRE_OK=1`.
- If no `ok` candidate exists, sweep fails (non-zero) by design.

## Tokenizer Artifact Caching

- Use `--tokenizerArtifacts=<path>` for repeated runs.
- Behavior:
  - if file exists: load artifacts,
  - else: build tokenizer artifacts and save to path.
- Benchmark harness defaults to:
  - `TOKENIZER_ARTIFACTS=perf/tokenizer-artifacts-benchmark.json`

## Working Tree Discipline

- Do not revert unrelated user changes.
- If unrelated modified files exist, leave them untouched unless explicitly requested.
- Keep perf artifacts under `perf/` and run logs under `runs/`.

## Documentation To Keep In Sync

- `docs/README.md`
- `feedback-loop.md`
- `docs/compiled-binary-usage.md`

Update docs when changing benchmark policy, compile flow, or perf-tuning controls.
