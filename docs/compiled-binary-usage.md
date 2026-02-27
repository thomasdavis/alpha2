# Using The Compiled `alpha` Binary

This is a quick guide you can paste into Slack.

## What you get
- Compiled executable: `./.bun-out/alpha`
- GPU native addon sidecar: `./.bun-out/helios_vk.node`

Both files are required for `--backend=helios`.

## Build the binary
From repo root:

```bash
npm run bun:compile
```

This does:
1. Builds Helios native C addon.
2. Compiles CLI with `bun --compile`.
3. Copies `helios_vk.node` next to the executable.

## Sanity check
```bash
./.bun-out/alpha --help
```

## 5-iteration smoke test (CPU)
```bash
./.bun-out/alpha train \
  --data=data/abc-small.txt \
  --backend=cpu_ref \
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
  --runDir=runs/compiled-binary-cpu
```

## 5-iteration smoke test (GPU / Helios)
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
  --runDir=runs/compiled-binary-helios
```

Notes:
- `--postSamples=false` keeps smoke tests stable/fast by skipping post-train generation.
- `--remote=false` disables remote reporting even if `.env.local` is present.

## Running from another folder
You can run the binary from anywhere as long as:
- `alpha` and `helios_vk.node` stay in the same directory.
- You pass valid paths for `--data`, `--checkpoint`, `--runDir`.

Example:
```bash
cd /tmp
/path/to/.bun-out/alpha --help
```

## Common commands
Generate text:
```bash
./.bun-out/alpha sample --checkpoint=runs/compiled-binary-cpu/checkpoint-5.json --prompt="The " --steps=100
```

Evaluate checkpoint:
```bash
./.bun-out/alpha eval --checkpoint=runs/compiled-binary-cpu/checkpoint-5.json --data=data/abc-small.txt
```

## Troubleshooting
- `native addon not found`:
  - Re-run `npm run bun:compile`.
  - Ensure `helios_vk.node` is next to `alpha`.
- GPU issues:
  - Verify Vulkan-capable drivers are installed.
  - Use `--backend=cpu_ref` to confirm CLI path first.
- Permission denied:
  - `chmod +x ./.bun-out/alpha`
