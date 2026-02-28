# Alpha

Alpha is a from-scratch GPT training and inference system built primarily in TypeScript, with a hand-written C Vulkan addon for GPU compute (`Helios`).

- No PyTorch / TensorFlow / ONNX
- Custom tensor backend + autograd
- GPT model, tokenizers, training loop, checkpointing, eval, sampling
- Optional Vulkan GPU backend with TS-generated SPIR-V kernels
- CLI, web dashboard/API, and TUI

## Why This Exists

Alpha is an engineering-first project focused on understanding and controlling the entire ML stack:

- TypeScript-first runtime and model code
- explicit, inspectable math and kernels
- minimal black boxes
- pluggable backends and tokenizers

If you want a compact system you can read, modify, and benchmark end-to-end, this repo is designed for that.

## What’s In The Repo

### Apps

- `apps/cli` — main `alpha` CLI (`train`, `sample`, `eval`, `bench`, `datagen`, `tokenizer build`)
- `apps/web` — Next.js dashboard + API routes (including OpenAI-compatible endpoints)
- `apps/tui` — terminal dashboard
- `apps/hf` — Hugging Face integration/deployment helper app

### Core Packages

- `packages/core` — shared types, configs, registries, RNG, shape utilities
- `packages/tensor` — CPU reference tensor backend
- `packages/autograd` — tape-based reverse-mode autodiff
- `packages/model` — GPT decoder-only transformer
- `packages/tokenizers` — BPE / char / word tokenizers
- `packages/train` — training loop, optimizers, checkpointing, data pipeline
- `packages/inference` — optimized pure-TS inference engine (KV cache, preallocated buffers)
- `packages/helios` — Vulkan GPU backend (TS runtime + TS-generated SPIR-V + native C addon)
- `packages/tests` — unit tests

## Features

- Train GPT-style models from text corpora
- Evaluate and sample from checkpoints
- Domain presets (`abc`, `chords`, `concordance`, etc.)
- GPU training with Vulkan (`helios`) or CPU fallback (`cpu_ref`)
- Mixed-precision activation storage (`--fp16=true`)
- Activation checkpointing (`--checkpoint=true`)
- Compiled standalone CLI binary (`bun --compile`) + Helios native sidecar
- Benchmark harness with retry/selection/history logging (`perf/compiled-loop-history.csv`)
- Runtime adaptive tuning knobs via env vars (no source edits required)
- Fail-fast GPU smoke-test mode for clean benchmarking
- Tokenizer artifact caching for repeated training/benchmark loops
- Benchmarks for ops / end-to-end / training
- OpenAI-compatible chat API routes in the web app

## Project Status

Alpha is an active experimental/open-source project.

- APIs and CLI flags may evolve
- performance work is ongoing
- some subsystems are optimized, others are still being hardened

## Prerequisites

### Required (all users)

- Node.js `>=20` (see `package.json`)
- npm `>=10`

### Optional (Helios GPU backend)

For `--backend=helios`:

- Linux (recommended)
- Vulkan-capable GPU + working driver/runtime
- `gcc`
- Node headers / dev package (for native addon build)

The native build script compiles `packages/helios/native/helios_vk.c` directly with `gcc`.

## Quick Start (CPU)

### 1. Install dependencies

```bash
npm install
```

### 2. See CLI help

```bash
npm run -w @alpha/cli dev -- --help
```

### 3. Train a small model (CPU reference backend)

Use the included sample dataset for a quick smoke test:

```bash
npm run train:dev -- \
  --data=datasets/test.txt \
  --backend=cpu_ref \
  --tokenizer=char \
  --iters=50 \
  --batch=8 \
  --block=64 \
  --dim=64 \
  --layers=2 \
  --heads=4 \
  --evalInterval=25
```

This writes outputs to `runs/<run_id>/` (config, metrics, checkpoints).

### 4. Sample from a checkpoint

```bash
npm run -w @alpha/cli dev -- sample \
  --checkpoint=runs/<run_id>/checkpoint-50.json \
  --prompt="The " \
  --steps=100
```

## Quick Start (GPU with Helios)

### 1. Build the native Vulkan addon

```bash
npm run -w @alpha/helios build:native
```

Or build the whole package:

```bash
npm run -w @alpha/helios build
```

### 2. Train with the GPU backend

```bash
npm run train:dev -- \
  --data=datasets/test.txt \
  --backend=helios \
  --tokenizer=char \
  --iters=200 \
  --batch=16 \
  --block=128 \
  --dim=128 \
  --layers=4 \
  --heads=4 \
  --fp16=true
```

Notes:

- `helios` is optional; CPU backend remains useful for correctness checks and small runs.
- If the native addon is missing or Vulkan is unavailable, use `--backend=cpu_ref`.

## Compiled Binary Workflow

Build compiled binary + native sidecar:

```bash
npm run bun:compile
```

This now does:
1. Workspace TS build for `@alpha/cli` (fresh `dist` artifacts).
2. Helios native build.
3. `bun --compile` from `apps/cli/dist/main.js`.
4. Copies `helios_vk.node` next to `./.bun-out/alpha`.

Quick check:

```bash
./.bun-out/alpha --help
```

See also:
- `docs/compiled-binary-usage.md`

## Performance Loop (Compiled Benchmark)

Main benchmark command:

```bash
scripts/run-compiled-benchmark.sh 100
```

Outputs:
- `perf/compiled-loop-history.csv`
- `perf/last-benchmark.env`
- `perf/run-<timestamp>.log`

Default benchmark safety behavior:
- `FAIL_ON_SMOKE_TEST=1` (strict fail-fast on GPU smoke failures)
- attempt scoring preference: `ok > unstable > smoke_fail > failed`

Optional diagnostic mode:

```bash
FAIL_ON_SMOKE_TEST=0 scripts/run-compiled-benchmark.sh 100
```

Automated adaptive tuning sweep (20 loops):

```bash
npm run perf:tune:adaptive
```

By default this tuner only accepts `status=ok` candidates and exits non-zero if none are valid.

## CLI Commands

Main CLI entrypoint (`apps/cli/src/main.ts`) supports:

- `alpha tokenizer build`
- `alpha train`
- `alpha sample`
- `alpha eval`
- `alpha bench`
- `alpha datagen`

Examples:

```bash
npm run -w @alpha/cli dev -- tokenizer build --type=bpe --input=datasets/test.txt --vocabSize=2000 --out=artifacts/tokenizer.json
npm run -w @alpha/cli dev -- train --data=datasets/test.txt --backend=cpu_ref --iters=100
npm run -w @alpha/cli dev -- eval --checkpoint=runs/<run_id>/checkpoint-100.json --data=datasets/test.txt
npm run -w @alpha/cli dev -- bench --suite=ops --backend=cpu_ref
```

### Important CLI Parsing Note

The current CLI argument parser expects `--key=value` style flags (for example `--iters=100`, not `iters=100`).

### Useful Training Flags For Looping

- `--tokenizerArtifacts=<path>`  
  Reuses tokenizer artifacts across runs; if missing, builds once and saves to that path.

## Web Dashboard / API

Start the web app (Next.js):

```bash
npm run web
```

The web app includes:

- run/model views
- training dashboards
- inference pages
- OpenAI-compatible API routes (`/v1/*`)

See:

- `docs/openai-compatible-api.md`
- `docs/guide-openai-compatible-api.md`

## TUI Dashboard

```bash
npm run tui
```

## Benchmarks and Tests

### Run tests

```bash
npm test
```

Or just the core unit suite:

```bash
npm run -w @alpha/tests test
```

### Run benchmarks

```bash
npm run -w @alpha/cli dev -- bench --suite=ops --backend=cpu_ref
npm run -w @alpha/cli dev -- bench --suite=train
scripts/run-compiled-benchmark.sh 100
npm run perf:tune:adaptive
```

### Runtime Performance Env Knobs

These are read at runtime by the trainer:

- `ALPHA_ADAPTIVE_MEM_STATS_POLL_EVERY`
- `ALPHA_ADAPTIVE_SYNC_MIN_INTERVAL`
- `ALPHA_ADAPTIVE_SYNC_DEFERRED_THRESHOLD`
- `ALPHA_ADAPTIVE_SYNC_PENDING_THRESHOLD`
- `ALPHA_GPU_METRICS_SAMPLE_EVERY`
- `ALPHA_CALLBACK_YIELD_EVERY`
- `ALPHA_FAIL_ON_SMOKE_TEST`

## Training Tips

- Start with `cpu_ref` for correctness and small experiments.
- Move to `helios` when your dataset/model is large enough for GPU payoff.
- Use `--trace=true` only when profiling; it adds overhead.
- `--fp16=true` and `--checkpoint=true` can reduce VRAM pressure for larger runs.
- Use domain presets (`--domain=abc`, `--domain=chords`, etc.) to get sane defaults quickly.

## Documentation

Start here:

- `ARCHITECTURE.md` — high-level system overview
- `docs/helios-perf-research.md` — GPU backend performance notes
- `docs/openai-compatible-api.md` — API behavior
- `docs/training-performance-improvement-plan.md` (if present in your branch) / repo diagnostics and perf notes

Additional repo notes and diagnostics may exist in root `.md` files (run reports, audits, plans).

## Development Workflow

### Build everything

```bash
npm run build
```

### Typecheck (workspace build without emit)

```bash
npm run typecheck
```

### Clean workspace outputs

```bash
npm run clean
```

## Contributing

Contributions are welcome, especially in:

- training performance
- Helios kernel work (matmul, attention, mixed precision)
- tests and backend parity checks
- docs and examples
- tooling and developer ergonomics

If you open a PR, include:

- what changed
- why it changed
- how you verified it (tests/benchmarks)

## License

A root `LICENSE` file is not currently present in this repository. Add one before publishing broadly as an OSS package/project.

## Acknowledgments

Alpha is intentionally influenced by “small, readable ML systems” ideas, while pushing further into a custom TS + Vulkan stack.
