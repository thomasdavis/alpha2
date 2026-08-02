# Alpha foundation-candidate feasibility and LR-pilot contract

Date: 2026-08-02

## Decision

The one-GPU architecture feasibility test is complete. The initially proposed
136,867,584-parameter configuration is rejected for the current paid envelope:
it sustained only 2,613.1 tokens/sec at batch 16 and occupied 43,488 MiB of the
49,140 MiB exposed device memory. A 2B-token run would take about 8.86 days and
cost about $146.70 at the active pod's $0.69/hour before post-training or
evaluation.

The bounded LR pilot will instead use this measured candidate:

```text
layers:          18
hidden width:    640
attention heads: 10 (64 dimensions/head)
SwiGLU FFN:      1,728
vocabulary:      12,288, tied embeddings
context:         1,024 tokens
parameters:      97,098,880
precision:       FP32
symbiogenesis:   disabled
```

At batch 24 it sustained 3,563.7 tokens/sec, 6.2% above its batch-16 result.
Batch 32 failed before the first step with an exact allocator exhaustion, so
batch 24 is the measured ceiling. The candidate is authorized for the three
bounded LR arms below, not yet for the multi-day foundation run.

## Measured configurations

Each successful row trained 30 full-context steps with the native TypeScript,
Vulkan, autograd, AdamW, tokenizer, and checkpoint path. These are end-to-end
training rates, not isolated matrix multiplication benchmarks.

| Architecture | Batch | Mean tok/s | Median tok/s | Mean step | Observed GPU allocation | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 136,867,584 params, 18x768 | 16 | 2,613.1 | 2,618.8 | 6,270.9 ms | 43,488 MiB | reject on cost |
| 97,098,880 params, 18x640 | 16 | 3,355.7 | 3,356.3 | 4,883.3 ms | 28,247 MiB | finite reference |
| 97,098,880 params, 18x640 | 24 | 3,563.7 | 3,586.4 | 6,901.5 ms | below device limit | select pilot batch |
| 97,098,880 params, 18x640 | 32 | n/a | n/a | no completed step | 49,153.9 MiB at failure | reject; OOM |

For the selected row, the synchronized phase medians were approximately
1,352.3 ms forward, 4,499.1 ms backward, and 1,048.3 ms gradient norm. The
larger batch improves useful tokens per step but does not change the earlier
finding that backward propagation dominates optimization opportunity.

## Full-run budget implication

Twenty tokens per parameter requires at least 1,941,977,600 training tokens.
At batch 24 and block 1,024, 79,020 steps produce exactly 1,941,995,520 tokens,
or 20.00018 tokens per parameter. Extrapolating only from the measured batch-24
rate gives about 6.31 uninterrupted days and $104.45 of GPU rental. That leaves
some of the operator's paid envelope for LR selection, conversational
post-training, checkpoint evaluation, and failures; the 137M architecture did
not.

This is capacity planning, not a quality claim. The eventual checkpoint must
still beat the public Alpha in free conversation after post-training.

## Corpus separation

Training uses the first four sealed, non-repeating pretraining shards. The
actual manifest is adjacent to the corpus:

```text
/mnt/donto-data/alpha-corpora/pretrain-text/foundation-2b-manifest.json
SHA-256 be6975e2ffe327beafdc35174321c79a778b3ac33e248eba28ab591081dcb2e0
```

`pretrain-005.txt` is excluded from training. A deterministic newline-aligned
64 MiB prefix is frozen for cheap, independent validation:

```text
foundation-val-005-64m.txt
bytes   67,108,687
SHA-256 17e30fa2e50e1a1f116cceed95381b76edd1be595d402c4dd053bd55a7eafd60
```

Its construction manifest has SHA-256
`f010da477d29189211d04ee05253906310658e0b61aac06069d48c84be24f384`.
The source shard is bound to SHA-256
`8a77915edafa6303086132272fa45d28af243050da26d2b32c86c3a59d89723e`.
No validation bytes may enter foundation training or later synthetic chat
generation.

## LR-pilot contract

Run three independent initializations from the same seed and corpus window:

| Arm | Peak LR | Minimum LR | Steps | Tokens | Evaluations | Checkpoints |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| L1 | 1e-3 | 1e-4 | 384 | 9,437,184 | 96, 192, 288, 384 | 192, 384 |
| L2 | 2e-3 | 2e-4 | 384 | 9,437,184 | 96, 192, 288, 384 | 192, 384 |
| L3 | 3e-3 | 3e-4 | 384 | 9,437,184 | 96, 192, 288, 384 | 192, 384 |

All other architecture, optimizer, seed, batching, validation, and backend
settings are identical. Selection is the lowest mean held-out loss over the
final three aligned evaluations; final held-out loss then lower LR break ties.
This short random-initialization pilot selects a stable learning rate. It does
not select a publishable model and cannot demonstrate chattiness.

The executable contracts are:

```text
scripts/freeze_pretrain_validation_slice.ts
scripts/run_foundation_candidate_lr_pilot.sh
scripts/analyze_foundation_candidate_lr_sweep.ts
scripts/pretokenize_pretrain_shard.ts
```

The pretokenizer builds the identical hash-keyed cache consumed by the trainer
without allocating a model or touching the GPU. After the remaining training
shards arrive, separate low-priority processes may prepare their caches in
parallel; the paid long run then verifies and loads those exact caches instead
of spending its first hour serially tokenizing four multi-gigabyte files.

Each completed checkpoint is compressed losslessly only after the native run
returns success. Both the raw digest and compressed digest are retained, and a
streaming decompression hash check must pass before the raw file is removed.
This keeps the three-arm evidence below the operator's 15 GiB review threshold.

## Correctness and publication gates

Before the long run:

1. all three arms must complete finite with zero allocator overflow;
2. the source, train, validation, and tokenizer hashes must match across arms;
3. validation cadence and metric rows must be exact;
4. lossless checkpoint integrity must pass;
5. the LR analyzer must select one of the predeclared rates;
6. the long-run contract must support a wholly held-out validation file while
   training across the four-shard manifest;
7. retained new artifacts must remain under 15 GiB, or work pauses for review.

No LR-pilot output is posted to Discord, Hugging Face, or BLAH. Those channels
receive a new version only after a completed foundation plus conversational
post-training candidate produces a genuine behavioral improvement.

## Canonical feasibility evidence

```text
/mnt/donto-data/donto-resources/benchmarks/alpha-foundation-feasibility-20260802/
```

The archive contains exact configs, metrics, logs, and raw-checkpoint SHA-256
identities for all successful rows, plus the exact batch-32 OOM. Representative
metric hashes:

| Row | `metrics.jsonl` SHA-256 |
| --- | --- |
| 137M batch 16 | `5b73522a234c20b7070761c95ac3e71a2c6d440d3c0c9ca6e010850a9685e1ad` |
| 97M batch 16 | `7c4d9ae20ace6336154c3e8ab47637e23e852fcf679bc25757b1e79b056ae989` |
| 97M batch 24 | `0c0be4eb937c057c69bfb20ab90152da7b5896f7afc38fac13e160cc9e4e8cc3` |

The disposable remote feasibility directories were deleted only after this
canonical copy and the hashes were verified. The GPU is idle pending the
committed pilot code and input-integrity checks.
