# Agent Guidelines

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
- **Reporting**: Always configure `DISCORD_WEBHOOK_URL` in `.env.local`.
- **Inference Samples**: The trainer should post samples to Discord every 200 steps.

### 3. Continuous Improvement Loop
- Execute training in long loops.
- Monitor coherence: The model "makes sense" when it can:
  - Respond to "Hello" appropriately.
  - Answer basic questions based on the dataset context.
- Adjust hyperparameters (LR, capacity, softCap) based on loss curves and sample quality.

## Canonical Commands

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
