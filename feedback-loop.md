# Performance Feedback Loop

## Goal
Ship safe throughput improvements quickly with a tight local validation loop.

## Loop (repeat every cycle)
1. Identify the current bottleneck.
2. Implement one focused optimization.
3. Build native + compiled binary.
4. Run a 5-iteration local GPU smoke test with the compiled binary.
5. If it passes, commit.
6. Pick the next bottleneck and repeat.

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

Pass criteria:
- Binary starts and runs all 5 iterations.
- No crash/OOM/native-load failure.
- Loss/grad_norm values are finite.
- No obvious performance regression in per-step timing breakdown.

## 4) Commit Rules
- Commit only after the compiled-binary GPU test passes.
- Commit message must include speedup percentage vs previous benchmark run.
- Commit message format:
  - `perf: <short optimization summary> (+X.XX% tok/s)`
- Include what changed, why, and validation command/output summary.

## 5) Next Optimization Selection
- Re-rank bottlenecks after each successful cycle.
- Prefer wins that remove full-tensor passes, reduce dispatch count, or reduce sync/readback.
- Keep sampling behavior unchanged during training.
