# Alpha — GPT training system

Domain: **alpha.omegaai.dev**

## Philosophy

Alpha is **hand-written from scratch**. Every component — tensors, autograd, model, tokenizers, training loop — is custom TypeScript with minimal to no external dependencies. This is intentional.

- **Pure JavaScript/TypeScript** — no heavy ML frameworks, no ONNX runtime wrappers, no "just use PyTorch"
- **Super configurable** — every layer of the stack should be tunable and swappable
- **Zero dependencies** — prefer writing code over adding packages. No npm deps for core functionality. If we need GPU access, we write the native addon and SPIR-V assembler ourselves
- **Understand everything** — no black boxes. If it's in the codebase, we wrote it and we understand it
- **Scratch-built** — the value is in building it, not in importing it

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

## Key env vars

| Var | Where | Purpose |
|-----|-------|---------|
| `TURSO_DATABASE_URL` | .env.local, Railway | Turso libsql connection URL |
| `TURSO_AUTH_TOKEN` | .env.local, Railway | Turso auth token (rw) |
| `UPLOAD_SECRET` | Railway | Auth token for ingest/upload endpoints |
| `ALPHA_REMOTE_URL` | training machine | Remote server URL for live metrics streaming |
| `ALPHA_REMOTE_SECRET` | training machine | Same as UPLOAD_SECRET on server |

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
