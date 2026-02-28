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
