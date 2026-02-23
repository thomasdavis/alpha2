# Alpha — GPT training system

Domain: **alpha.omegaai.dev**

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

## GPU Training (RunPod)

Training runs on **RunPod** GPU pods via `scripts/runpod_train.py`. The script handles the full lifecycle: provision pod, sync code, build, upload dataset, train, download results.

```bash
# Train on concordance dataset with H100:
python scripts/runpod_train.py --data data/concordance.txt --domain concordance \
  --iters 50000 --batch 8 --block 256 --dim 256 --heads 8 --layers 6 \
  --backend helios --gpu H100

# Pod management:
python scripts/runpod_train.py --action status     # check pod
python scripts/runpod_train.py --action stop        # stop (saves money)
python scripts/runpod_train.py --action ssh         # interactive SSH
python scripts/runpod_train.py --action terminate   # destroy pod
```

- Pods reuse automatically — the script finds an existing `alpha-train` pod before creating a new one
- Environment setup (Node.js, Vulkan) is cached on the pod's workspace volume
- Code is synced via rsync on each run; datasets are uploaded once and reused
- Pass `--stop-after` to auto-stop the pod when training completes
- SSH public key must be added to RunPod account settings (or is injected via `PUBLIC_KEY` env var)

## GPU Training (GCP)

Training also runs on **GCP** A100 80GB instances via `scripts/gcp_train.py`. Same structure as RunPod but roughly half the cost (~$1.10/hr vs $1.99-2.39/hr).

Prerequisites:
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

```bash
# Train on concordance dataset with A100:
python scripts/gcp_train.py --data data/concordance.txt --domain concordance \
  --iters 50000 --batch 8 --block 256 --dim 256 --heads 8 --layers 6 \
  --backend helios --stop-after

# Train in a different zone:
python scripts/gcp_train.py --data data/concordance.txt --iters 5000 --zone us-west1-b

# Instance management:
python scripts/gcp_train.py --action status     # check instance
python scripts/gcp_train.py --action stop        # stop (disk persists, ~$34/mo)
python scripts/gcp_train.py --action start       # resume stopped instance
python scripts/gcp_train.py --action ssh         # interactive SSH
python scripts/gcp_train.py --action delete      # destroy instance + disk
```

- Uses `a2-ultragpu-1g` machine type (A100 80GB included, no separate `--accelerator` needed)
- Deep Learning VM image has NVIDIA drivers pre-installed — setup only needs Node.js + Vulkan
- Boot disk (200GB SSD) persists across stop/start cycles
- SSH keys bootstrapped automatically via `gcloud compute ssh` on first connection
- Same CLI args as RunPod (except `--zone` instead of `--gpu`)

## Key env vars

| Var | Where | Purpose |
|-----|-------|---------|
| `TURSO_DATABASE_URL` | .env.local, Railway | Turso libsql connection URL |
| `TURSO_AUTH_TOKEN` | .env.local, Railway | Turso auth token (rw) |
| `UPLOAD_SECRET` | Railway | Auth token for ingest/upload endpoints |
| `ALPHA_REMOTE_URL` | .env.local, training pod | API server URL for metrics streaming (use `ALPHA_REMOTE_URL`, NOT the web dashboard URL) |
| `ALPHA_REMOTE_SECRET` | .env.local, training pod | Same as UPLOAD_SECRET on server |
| `RUNPOD_API_KEY` | .env.local | RunPod API key for GPU pod provisioning |
| `RUNPOD_VOLUME_ID` | .env.local | (optional) RunPod network volume for persistent storage |
| `DISCORD_WEBHOOK_URL` | .env.local, training pod | Discord webhook for training notifications + inference samples |

## Discord Notifications

Training posts to Discord via webhook when:
- A run starts (model config, params, hyperparams)
- Inference samples are generated (at each checkpoint interval)
- Training completes

Webhook: `REDACTED_DISCORD_WEBHOOK_URL`

Set `DISCORD_WEBHOOK_URL` in `.env.local` on training pods. The remote reporter sends embeds with run info, sample outputs, and completion status.

## Deploy

All deployments are on **Railway** (project `REDACTED_PROJECT`). Root `vercel.json` is a legacy leftover.

- **Server**: `railway service alpha2 && railway up`
- **Web dashboard**: `railway service alpha-web && railway up`
- Training runs stored in `outputs/` locally, synced to Turso via `@alpha/db` syncFromDisk

## DB

Turso cloud: `TURSO_DATABASE_URL`

Tables: `runs`, `metrics` (WITHOUT ROWID), `checkpoints` (WITHOUT ROWID), `domains`, `schema_version`

View: `run_summary` (joins runs + checkpoint/metric counts)

Schema managed by version-based migrations in `packages/db/src/schema.ts`.
