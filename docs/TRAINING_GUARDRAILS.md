# Training Guardrails

This guide is a practical checklist for kicking off training runs with sane data/compute planning.

## Why

Small models can look "stable" while being under-trained or overfitting reused data.  
The highest-impact early checks are:

- token budget vs parameter count
- dataset reuse pressure
- validation hygiene
- resume behavior

Alpha now runs startup planning checks in `alpha train` and can fail fast in strict mode.

## Startup Checks In Alpha

Before training starts, Alpha estimates:

- `model_params` (architecture-based estimate)
- `planned_tokens = iters * batch * block * accumSteps`
- `tokens/param`
- approximate dataset token count from file size + tokenizer heuristic
- approximate dataset passes (`planned_tokens / dataset_tokens`)

It warns on:

- low token budget for model size (`tokens/param`)
- high dataset reuse (many effective passes)
- `--valData` missing (auto 90/10 split fallback) or invalid (same file path)
- `evalInterval >= iters`
- tiny effective batch on GPU

It errors on:

- very low `tokens/param` (<5)
- validation path equal to training path
- dataset reuse above configured hard limit
- missing `--valData` when `--requireValData=true`

## Recommended Command Pattern

```bash
npm run train:dev -- \
  --data=data/train.txt \
  --valData=data/val.txt \
  --backend=helios \
  --strictPlanning=true \
  --requireValData=true \
  --minTokensPerParam=20 \
  --warnDatasetPasses=8 \
  --maxDatasetPasses=20
```

Notes:

- `--strictPlanning=true` fails only when startup checks hit hard errors.
- `--requireValData=true` enforces a dedicated validation file.
- If you do not provide `--valData`, Alpha still runs by auto-splitting the data file 90/10.

## Resume Guidance

Alpha checkpoints include optimizer state and RNG state.  
When you resume from `--resume=<checkpoint>`, trainer restores both.

Implication:

- normal resume should not need LR warmup reset just because of restart
- if you intentionally reset optimizer state, treat that as a fresh optimization phase and use warmup

## Data Order Guidance

Alpha already supports:

- random window sampling (default)
- packed sequential cursor mode (`--packed=true`)

For contiguous-token training, this usually covers the same failure mode as "fixed ordering" concerns in many small-project training setups.

## Practical Rule Of Thumb

1. Keep `tokens/param` at or above your chosen target (default 20).
2. Keep effective dataset passes in a reasonable band.
3. Always monitor validation (prefer dedicated `--valData` in serious runs).
4. Resume from full checkpoint state whenever possible.
