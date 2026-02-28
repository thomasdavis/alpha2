# AGENTS.md

Project-specific operating guide for coding agents in this repo.

## Scope

- Repo: `alpha` (TypeScript-first GPT stack + optional C Vulkan addon).
- Primary optimization focus: training throughput + compiled binary reliability.
- Language policy: default to TypeScript; use C only where native/Vulkan bridge is required.

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
