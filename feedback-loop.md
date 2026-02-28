# Performance Feedback Loop

## Goal
Ship safe throughput improvements quickly with a tight local validation loop.

## Loop (repeat every cycle)
1. Identify the current bottleneck.
2. Implement one focused optimization.
3. Build native + compiled binary.
4. Run a 100-iteration local GPU benchmark with the compiled binary.
5. Run a compiled-binary inference check with 3 fixed questions.
6. Record cooperative-matmul hit telemetry (`coop_matmul`) from the run log.
7. If both pass, commit.
8. Pick the next bottleneck and repeat.

## 1) Identify Issues
- Use recent training logs and trace timings (`fwd`, `bwd`, `gradnorm`, `clip`, `optim`, `flush`, `gpu_ops`).
- Prioritize by highest step-time contribution and lowest implementation risk.
- Only change one major optimization area per cycle so regressions are easy to isolate.

## 2) Build Compiled Binary
```bash
npm run bun:compile
```

This script compiles Helios native C first, then runs Bun compile to produce:
- `./.bun-out/alpha`

## 3) 100-Iteration Local GPU Benchmark (Compiled Binary)
Primary benchmark loop (records timings + speedup vs previous run):

```bash
scripts/run-compiled-benchmark.sh 100
```

Outputs:
- `perf/compiled-loop-history.csv` (benchmark history)
- `perf/last-benchmark.env` (latest metrics for commit message)
- `perf/run-<timestamp>.log` (full run log)
- `coop_matmul` line in run log (direct/padded/rewrite coop dispatch counts + hit rate)
- benchmark tokenizer cache: `perf/tokenizer-artifacts-benchmark.json` (auto-built once, then reused)

Adaptive runtime tuning knobs (no source edit/recompile needed):
- `ALPHA_ADAPTIVE_MEM_STATS_POLL_EVERY`
- `ALPHA_ADAPTIVE_SYNC_MIN_INTERVAL`
- `ALPHA_ADAPTIVE_SYNC_DEFERRED_THRESHOLD`
- `ALPHA_ADAPTIVE_SYNC_PENDING_THRESHOLD`
- `ALPHA_GPU_METRICS_SAMPLE_EVERY`
- `ALPHA_CALLBACK_YIELD_EVERY`
- `ALPHA_FAIL_ON_SMOKE_TEST` (default strict in benchmark script)

Benchmark script smoke policy:
- default: `FAIL_ON_SMOKE_TEST=1` (fail fast; marks run `smoke_fail` quickly)
- override for diagnostics only: `FAIL_ON_SMOKE_TEST=0 scripts/run-compiled-benchmark.sh 100`

Quick smoke alternative (5 iterations) for sanity checks:

```bash
./.bun-out/alpha train \
  --data=data/abc-small.txt \
  --backend=helios \
  --steps=5 \
  --batch=2 \
  --block=64 \
  --layers=2 \
  --dim=128 \
  --heads=4 \
  --accumSteps=1 \
  --evalInterval=5 \
  --evalIters=1 \
  --sampleInterval=0 \
  --postSamples=false \
  --remote=false \
  --trace=true \
  --runDir=runs/feedback-loop
```

Pass criteria (required):
- `scripts/run-compiled-benchmark.sh 100` completes all 100 iterations.
- No crash/OOM/native-load failure.
- Loss/grad_norm values are finite.
- No obvious performance regression in per-step timing breakdown.
- `coop_matmul` telemetry is present and parseable in the run log.

Optional smoke criteria (if you also run the 5-step command):
- Binary starts and runs all 5 iterations.

## 4) End-of-Loop Inference Check (3 Questions)
Run a final inference sanity check from the latest checkpoint using exactly 3 prompts.

```bash
CKPT="$(ls -t runs/feedback-loop/checkpoint-*.json | head -n1)"
./.bun-out/alpha sample --checkpoint="$CKPT" --prompt="What is 2+2?" --steps=64
./.bun-out/alpha sample --checkpoint="$CKPT" --prompt="Finish: The quick brown fox" --steps=64
./.bun-out/alpha sample --checkpoint="$CKPT" --prompt="Name one GPU optimization idea." --steps=64
```

Pass criteria:
- All 3 sample commands complete successfully.
- No crash/OOM/native-load failure.
- Each output is non-empty and not obviously corrupted/repeating a single token forever.

## 5) Commit Rules
- Commit only after both gates pass: 100-iteration compiled benchmark + 3-question inference check.
- Commit message must include speedup percentage vs previous benchmark run.
- Commit message format:
  - `perf: <short optimization summary> (+X.XX% tok/s)`
- Include what changed, why, and validation command/output summary.

## 6) Next Optimization Selection
- Re-rank bottlenecks after each successful cycle.
- Prefer wins that remove full-tensor passes, reduce dispatch count, or reduce sync/readback.
- Keep sampling behavior unchanged during training.

Automated 20-loop adaptive sweep:

```bash
npm run perf:tune:adaptive
```

This writes:
- `perf/tune-adaptive-env-<timestamp>.csv`
- `perf/best-adaptive-env-<timestamp>.env`
