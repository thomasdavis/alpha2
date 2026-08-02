# Helios chat-training throughput and capacity audit

Date: 2026-08-02

## Decision

Alpha's current chat work has two separable bottlenecks:

1. **model/data capacity:** the clean base is too weak for ordinary SFT to
   install dependable conversational semantics;
2. **training-system efficiency:** the native FP32 Vulkan training path leaves
   substantial accelerator throughput unused despite reporting a busy GPU.

The current 2,000-step V12 `1e-3` arm remains unchanged because it is a bounded
causal control. Changing block length, precision, kernels, or model size in the
middle would destroy the comparison. Once it completes, Helios will receive a
same-seed, same-token-window throughput sweep before a longer or larger training
run is authorized.

The project should no longer assume that the present parameter count is the
ideal product size. The next serious model should test a larger one-GPU
foundation or a distilled student. More parameters alone are not a cure: they
must be paired with adequate foundation exposure or teacher supervision.

## Measured baseline

Active workload:

```text
architecture: 16 layers, width 512, 8 heads, SwiGLU FFN 1408
sequence:     batch 16 x block 1024 = 16,384 tokens/optimizer step
precision:    FP32 training
backend:      native Helios Vulkan
attention:    fused flash-attention path available
coop matrix:  disabled by HELIOS_DISABLE_COOP_MAT=1
workgroup:    HELIOS_WG_SIZE=64
packing:      enabled
optimizer:    AdamW
```

Across 27 non-evaluation steady-state log samples from steps 100 through 825:

| Measurement | Observed |
| --- | ---: |
| Mean throughput | 5,330.8 tokens/sec |
| Median throughput | 5,377 tokens/sec |
| Minimum / maximum | 4,895 / 5,520 tokens/sec |
| Mean step time | 3,076.5 ms |
| GPU operations per ordinary step | 1,934 |
| GPU operations at an eval checkpoint | 3,409-3,637 |

This is a measured baseline, not an estimate. A 20-second `nvidia-smi dmon`
sample during ordinary training averaged 42.5% SM utilization, 6.15%
memory-controller utilization, and 196.05 W from a 450 W power limit. Individual
one-second samples ranged from 0% to 100% SM use while framebuffer allocation
swung from 16,158 MiB to 48,526 MiB. This
sawtooth is more informative than a single utilization snapshot: the device is
alternating between useful kernels and underfilled/idle periods while temporary
storage turns over. The combination—low average power, low memory utilization,
intermittent SM idleness, and 1,934 operations per step—strongly suggests
latency, occupancy, dispatch, allocation-lifetime, and unfused-memory-traffic
limits rather than full math saturation.

The Node process used roughly 57% of one CPU core on a 256-vCPU host, so input
tokenization and CPU availability are not the primary limiter. Data is already
token-cached and packed.

As a rough transformer-training utilization proxy, `6 * parameters * tokens`
is about 5.67 teraFLOP per current step, or only about 1.84 effective TFLOP/sec
at the measured 3.0765 seconds. This proxy omits architecture-specific attention
and kernel work, so it is not a hardware-profiler result, but it confirms that
the observed step rate is nowhere near the device's arithmetic ceiling.

## Why the historical 50K-60K numbers do not transfer

An older Helios dossier reports approximately 59K tokens/sec, but that workload
used a four-layer, width-128 model with a 256-token block. It performs far less
work per token than the current 16-layer, width-512, 1,024-context model. That
number is useful proof that the runtime can move tokens quickly on a tiny graph;
it is not evidence that the current model should achieve the same throughput.

## Concrete inefficiencies visible now

### 1. Tensor-core paths are disabled

Training runs with `HELIOS_DISABLE_COOP_MAT=1` and `--fp16=false`. The current
stable path therefore uses FP32 SIMT kernels instead of the device's fastest
matrix hardware. This is the largest plausible multiplicative opportunity.

The flags were not disabled arbitrarily. Historical Alpha runs found that
f16-cast cooperative forward passes changed the loss trajectory and could
diverge, while f16 backward gradients could overflow. Any reactivation must
pass loss, gradient, and checkpoint parity—not merely a speed test.

### 2. Dispatch count is high

An ordinary step records 1,934 GPU operations, approximately 121 per transformer
layer before accounting for work shared outside blocks. Even though Helios
batches command submission and enables Vulkan device-generated commands, each
kernel still incurs scheduling and global-memory costs. A 3x whole-step gain is
unlikely from workgroup tuning alone; it likely requires fewer operations and
fused transformer subpaths.

### 3. Temporary-buffer churn is high

Each 25-step telemetry interval reports roughly 76,000 temporary creations and
destructions, or about 3,040 each per optimizer step. The native slab allocator
prevents those logical tensors from becoming the same number of physical VRAM
allocations, but the lifecycle and handle-management work remains substantial.

The output pool is pinned at 512 entries and about 6.4 GiB. A typical 25-step
window showed approximately 17.5K pool hits against 75.9K misses/overflows. This
does not prove that simply enlarging the pool is faster—doing so can exhaust
VRAM—but it identifies a benchmarkable allocator/lifetime bottleneck.

### 4. Workgroup size is inherited from a conservative recipe

The active run forces workgroup size 64. Older NVIDIA recipes often used 128 or
256. The correct value is kernel- and device-specific, so it must be selected by
an identical-workload sweep rather than historical habit.

### 5. Context length has a real compute price

Attention cost grows quadratically with sequence length. A `block=512,
batch=32` test keeps 16,384 tokens per step while substantially reducing
attention work per token. It is a legitimate product trade-off only if the
shorter context still supports the required conversation; it is not a free
implementation optimization. It will therefore be reported separately from
same-model kernel improvements.

## Post-control benchmark matrix

Every row will use the same clean checkpoint, corpus, tokenizer, seed, optimizer
settings, and first token windows. Warm-up steps will be excluded from the
median. Each risky row must be compared against the baseline loss/gradient
trajectory and native checkpoint outputs.

| Row | Change | Question |
| --- | --- | --- |
| B0 | FP32, coop off, WG 64 | Reproduce the 5.33K baseline |
| B1 | FP32, coop off, WG 128 | Does workgroup tuning help? |
| B2 | FP32, coop off, WG 256 | Does the historical NVIDIA setting help? |
| B3 | FP32, coop off, larger bounded output pool | Does reduced buffer churn justify VRAM? |
| B4 | Coop forward enabled, backward safety retained | Is tensor-core forward fast *and* numerically usable? |
| B5 | Mixed precision with loss scaling | Can the true high-throughput path remain finite? |
| B6 | Block 512, batch 32, FP32 | What is the context-length/product trade-off? |
| B7 | B0 plus a GPU synchronization at each measured phase boundary | Which phase owns GPU wall time? This row is attribution-only, not a throughput contender. |

For B4 and B5, the experiment stops immediately on non-finite loss, gradients,
or a correctness mismatch. A fast-but-wrong kernel is a failed row.

Helios normally evaluates lazily, so ordinary trace timers measure CPU graph
construction and charge accumulated GPU waiting to the next readback; they are
not honest kernel-phase timers. B7 deliberately synchronizes after forward,
backward, and optimizer work to produce an attribution profile. Those barriers
change batching and make B7 slower, so only B0-B6 may be compared for end-to-end
throughput. Opt-in backend instrumentation records operation kind, the dominant
kernel names, command flushes, waited flushes, DGC flushes, and operations per
flush. Both profilers are disabled in ordinary training. Kernel work should be
optimized only after the bounded evidence identifies the dominant phase and
operation mix.

## Likely optimization order

1. Tune WG and bounded pool settings because they are low-risk and easy to
   falsify.
2. Isolate and repair cooperative-matrix/mixed-precision correctness, keeping
   FP32 master weights and numerically sensitive reductions.
3. Reduce temporary allocation churn through deterministic lifetimes and
   reuse—not by hiding leaks under a larger pool.
4. Fuse high-frequency envelopes such as norm/residual/linear epilogues and
   activation paths to reduce the 1,934-operation step.
5. Re-benchmark full training, not only isolated GEMMs.

The target is at least 3x whole-step throughput, but it is a target rather than
an invented promise. Only median end-to-end tokens/sec with matching numerical
behavior counts.

The economic difference is material. At 5,330.8 tokens/sec, exposing a model to
5 billion tokens would take about 10.86 uninterrupted days and cost about
`$179.77` at the active pod's `$0.69/hour`. At exactly 3x throughput it would
take about 3.62 days and cost about `$59.92`, before evaluation and failures.
This is why throughput work belongs before the larger foundation run rather
than after it.

## Capacity decision

A chatty model does not have a strict parameter threshold. Very small models can
produce coherent dialogue when heavily distilled, while substantially larger
models can remain poor when undertrained. Alpha's evidence now says the current
combination—present size, approximately one billion foundation tokens, then
ordinary SFT—is insufficient.

The primary next-foundation candidate is now concrete:

```text
layers:          18
hidden width:    768
attention heads: 12 (64 dimensions/head)
SwiGLU FFN:      2048
vocabulary:      12,288, tied embeddings
parameters:      136,867,584
```

The exact count follows Alpha's implemented tied Llama-form parameterization:
`vocab*d + layers*(4*d*d + 3*d*ffn + 2*d) + d`.

This candidate is not an arbitrary scaling exercise. The staged pretraining
corpus already contains six sealed shards, 11,700,002,843 characters and an
estimated 3.0B tokens. The previous minimum run's manifest selected only the
first three. Using all six once gives the proposed model approximately 21.9
tokens per parameter, matching the project's declared minimum planning ratio
without repeating data. The corpus manifest and every shard hash already live
under `/mnt/donto-data/alpha-corpora/pretrain-text/`.

The all-shard selection is now materialized as
`foundation-3b-manifest.json`, SHA-256
`e2a24b0ca48790bb68ca9f2a326f68c64bd92c3b568dcbcb5d4b262f7ca732a4`.
It references, but does not duplicate, the six files sealed by the original
`MANIFEST.sha256`.

At a hypothetical 3x improvement to the current kernel path and naive inverse
parameter-count scaling, this larger model would process roughly 6.7K tokens/sec
and the 3B-token pass would take about 5.2 days or `$85` at `$0.69/hour`. This is
only capacity planning—the real architecture must be benchmarked because larger
matrix shapes may use the GPU more efficiently than linear scaling predicts.

After foundation pretraining, the comparison should include conversational
sequence-level distillation from a strong teacher as well as ordinary SFT. A
roughly 295M positive-control architecture remains scientifically useful, but a
full 20-token/parameter run is unlikely to fit the current paid-compute envelope
unless the measured speedup materially exceeds 3x.

Selection remains behavioral. A larger checkpoint that lowers validation loss
but still cannot answer, stop, preserve a conversational role, or follow short
instructions is not progress.

## Authorization gate

No long larger-model run begins until:

- the V12 control has an honest free-generation outcome;
- the throughput sweep has a reproducible winner;
- that winner passes the NVIDIA kernel/parity suite;
- expected retained artifacts remain below the 15 GiB review threshold;
- and the larger/distillation run has an immutable token, compute, evaluation,
  and stop contract.
