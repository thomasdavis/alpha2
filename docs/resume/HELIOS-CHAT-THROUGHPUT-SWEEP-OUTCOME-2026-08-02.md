# Helios chat-training throughput sweep — outcome

Date: 2026-08-02

## Decision

The bounded eight-row RTX 4090 sweep is complete. It did **not** find a safe
threefold throughput improvement. Alpha's existing full-context FP32 recipe
remains the correctness reference at 5,333.6 mean tokens/sec. Enlarging the
output pool was statistically indistinguishable, workgroups 128 and 256 were
slower, and the only faster finite row changed the product contract from a
1,024-token context to 512 tokens.

The cooperative-matrix and mixed-precision rows are rejected. They did not
merely follow a different numerical trajectory: both began around loss 6.95
instead of the FP32 reference near 2.74, the cooperative row produced
non-finite gradients and skipped updates, and both eventually exhausted GPU
memory. They must not be used for training until the forward and backward
paths pass the existing NVIDIA parity suite.

The synchronized attribution row established where to optimize next. At full
context, median forward time was 502 ms and median backward time was 2,682.5
ms. Gradient norm was only 15 ms and AdamW was 2 ms once GPU work was charged
to the phase that actually submitted it. Backward propagation therefore owns
roughly 84% of the measured forward-plus-backward wall time. Optimizer fusion
or CPU/data work cannot deliver the desired multiplier.

## Executed matrix

Every row used the same clean checkpoint, token stream, tokenizer, seed, 30
steps, and first training windows. Five warm-up steps and the final
evaluation/checkpoint step were excluded from the throughput summary.

| Row | Exit | Median tok/s | Mean tok/s | Median ms | Last loss | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| B0 FP32, WG64, block 1024 | 0 | 5,328 | 5,333.6 | 3,075 | 2.3646 | correctness reference |
| B1 FP32, WG128, block 1024 | 0 | 5,130 | 5,129.9 | 3,193.5 | 2.3639 | slower |
| B2 FP32, WG256, block 1024 | 0 | 5,151 | 5,110.0 | 3,181 | 2.3626 | slower |
| B3 FP32, WG128, pool 768 | 0 | 5,359 | 5,311.4 | 3,057.5 | 2.3638 | no meaningful gain |
| B4 cooperative forward | 1 | 4,844 | 4,870.7 | 3,382 | 6.9600 | wrong and non-finite |
| B5 cooperative mixed precision | 1 | 2,253 | 2,259.2 | 7,272 | 6.9512 | wrong, slower, OOM |
| B6 FP32, block 512, batch 32 | 0 | 5,552.5 | 5,458.3 | 2,950.5 | 2.4913 | context trade-off, not selected |
| B7 FP32 with phase synchronization | 0 | 5,111.5 | 5,033.6 | 3,205.5 | 2.3646 | attribution only |

B6's mean gain over B0 is only 2.34%, and its median gain is 4.21%. It also
halves the conversation context, so it is not a same-model implementation
win. B3 differs from B0 by -0.42% in mean throughput despite a slightly higher
median. Neither justifies changing the production training recipe.

## Operation evidence

The full-context reference issued 2,162 GPU operations per ordinary step:

```text
unary       938
reduce_sum  459
matmul      259
binary      161
backward    116
optimizer   114
inplace      81
layernorm    33
softmax       1
```

The dominant individual kernels were 680 `scale`, 231 `sum_reduce`, 162
`sum_sq_reduce_stride`, 128 `transpose`, 114 `adamw_step`, and 113 `add`
dispatches. The synchronized row used six command flushes, three of which
waited, with 360.3 operations per flush.

The evidence redirects optimization toward the backward graph: eliminate or
fuse repeated scale/reduction/transpose envelopes and reduce intermediate
materialization. Workgroup selection, output-pool size, optimizer cost, and
shortening context cannot provide the requested multiplier.

## Artifact integrity

Canonical archive:

    /mnt/donto-data/donto-resources/benchmarks/alpha-helios-chat-throughput-20260802/

The archive is 1.2 GiB and contains complete logs, configs, metrics, exit
statuses, and losslessly compressed optimizer-bearing checkpoints for all six
successful rows. `zstd -t` passed for every checkpoint and each compressed
digest matches its row ledger. The two failed rows contain their exact logs and
non-zero exit status but no partial checkpoint.

| Artifact | SHA-256 |
| --- | --- |
| `SUMMARY.md` | `84f52591b97e9542a5f9988517da6730d5cc766df1ef0b4a9990481be58e2663` |
| `summary.json` | `a28c9c53799d3054bd8947f6cdffa0b6f27c1f0fbffedb42bd6f8a8d1feb1c3f` |
| `ARTIFACTS.sha256` | `4ed1c05cee21aca81f9710b03c00477b0d6a11d8d60cc8475bedc15f90dcbcb5` |

After the local copy and all integrity checks passed, the exact 1.2 GiB remote
sweep directory was removed to recover pod overlay space. The canonical
mounted-drive archive is retained and recoverable; no model run or unique
evidence was deleted.

## Consequence for the model plan

Do not authorize a multi-day foundation run by extrapolating a hypothetical
3x speedup. Benchmark the proposed architecture itself at full context, then
choose a model/token budget that fits the paid envelope. Conversational
sequence-level distillation remains part of the plan because it can transfer
teacher behavior without requiring Alpha to memorize the teacher's factual
world model. Every candidate still has to beat the frozen public model in free
generation before publication.
