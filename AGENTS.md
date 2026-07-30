# Agent Guidelines

## Current archive override — 2026-07-30

The Alpha 60M program is closed. No training, continuation, sweep, frozen-eval tuning, or Alpha RunPod
creation is authorized. A future “resume” request means recover durable state and prepare a bounded
proposal; it does not itself authorize spend.

Read [docs/resume/README.md](docs/resume/README.md) and
[docs/resume/SESSION-START.md](docs/resume/SESSION-START.md) before acting. The binding decisions are in
[docs/resume/DECISIONS.md](docs/resume/DECISIONS.md).

Current supersessions:

- Discord is qualitative-improvement-only: same input, before/after outputs, explanation, and aggregate
  boundary. Do not post routine samples or numeric progress.
- Do not configure or read the webhook unless an improvement post has passed that gate.
- The proven RunPod training runtime is Node. The historical Bun standalone binary failed Vulkan
  initialization there; platform proof overrides the generic compiled-binary preference.
- If the user explicitly authorizes a new program, begin with the bounded gates in
  [docs/resume/ACCEPTANCE-GATES.md](docs/resume/ACCEPTANCE-GATES.md), not a long loop.

## Optimization Philosophy

You can beat everything. You just have to research. Don't assume any gap is structural or unfixable — dig into the actual bottleneck, understand the hardware, and find the path forward.

## Operational Mandates

### 1. Compiled Binary Workflow
- **Always** use `bun compile` (via `npm run bun:compile`) to generate a standalone binary for deployment.
- Sync the resulting `.bun-out/` directory to the fleet instances instead of raw source files when possible.
- Use `npm run fleet:deploy` which automates this process.

### 2. Fleet Training Configuration
- **DGC (Device Generated Commands)**: Must be enabled (`HELIOS_DISABLE_DGC=0`).
- **No Fallback**: Training should fail fast rather than falling back to slow CPU paths for core operations.
- **Reporting**: Preserve samples locally. Discord posting follows the qualitative-improvement-only
  contract above; never post on an interval.
- **Secret handling**: The ignored mode-0600 webhook file is read only after a post passes the
  qualitative gate. Never place the URL in tracked configuration.

### 3. Continuous Improvement Loop
- Only after renewed explicit authorization, execute bounded proof-gated pilots before any long loop.
- Monitor coherence: The model "makes sense" when it can:
  - Respond to "Hello" appropriately.
  - Answer basic questions based on the dataset context.
- Adjust one declared variable at a time. Select using aggregate generation evidence, not loss curves
  or an attractive sample alone.

## Historical canonical commands — require renewed authorization

### Build & Deploy
```bash
npm run fleet:deploy -- <instance-name>
```

### Resume/Start Training with Stability Defaults
```bash
npm run fleet:train -- <instance-name> \
  --runtime=binary \
  --dgc=true \
  --no-fallback=true \
  --sampleInterval=200
```
