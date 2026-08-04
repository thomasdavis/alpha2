# X40 — the operation graph is dominated by reductions and the optimizer, not GEMMs

**Date:** 2026-08-04
**Status:** analysis of a preserved **physical RTX 3090** trace. No code changed, no speedup claimed.
**Follows:** X39, which established that host cost is per-dispatch dominated.
**Source trace:** `benchmarks/alpha-helios-3090-portfolio-20260803/alpha-helios-graph-trace-3090-b10-20260803-r4/modes/slab8_pool64/run/gpu-graph-trace.jsonl`, step 4, graph `70191bafab6319b9`, 1,703 operations.

---

## 1. The question X39 left

X39 showed that the host phases scaling with operation count — `desc_update`, `barrier`, `push_const`,
`cmd_dispatch`, `decode`, `bind` — are **65.3% of host time**, each with exactly one call per dispatch.
Host cost is therefore a function of *how many operations the graph contains*.

That makes "which operations should be removed" a measurable question rather than an intuition. This scans the
real graph and answers it.

## 2. The result

**By operation kind:**

| Kind | Operations | Share |
|---|---:|---:|
| **reduce_sum** | **518** | **30.4%** |
| unary | 326 | 19.1% |
| matmul | 291 | 17.1% |
| binary | 181 | 10.6% |
| backward | 130 | 7.6% |
| optimizer | 128 | 7.5% |
| inplace | 91 | 5.3% |
| layernorm | 37 | 2.2% |

**Reductions are nearly twice the operation count of all matrix multiplication combined.**

**Top kernels:**

| Kernel | Count | Share |
|---|---:|---:|
| `sum_reduce` | 259 | 15.2% |
| `sum_sq_reduce_stride` | 182 | 10.7% |
| `transpose` | 144 | 8.5% |
| `adamw_step` | 128 | 7.5% |
| `add` | 127 | 7.5% |
| `matmul_transposed_R42C` | 91 | 5.3% |
| `matmul_R42` | 91 | 5.3% |
| `matmul_transposed_a_R42C` | 91 | 5.3% |
| `sum_sq_reduce` | 74 | 4.3% |

**Back-to-back runs of a single kernel** — these need no new fused kernel, only a batched dispatch:

| Kernel | Runs | Ops | Longest run | Dispatches removable |
|---|---:|---:|---:|---:|
| `adamw_step` | 1 | 128 | **128** | 127 |
| `add` | 1 | 127 | **127** | 126 |
| `transpose` | 36 | 108 | 3 | 72 |
| `rope` | 36 | 72 | 2 | 36 |
| `matmul_transposed_R42C` | 18 | 36 | 2 | 18 |
| `scatter_slice_2d` | 18 | 36 | 2 | 18 |

**Removable by same-kernel run batching alone: 401 dispatches, 23.5% of the graph.**

## 3. The finding that redirects effort

The campaign has correctly identified that **GEMMs dominate GPU time** — the handoff records the top three GEMMs
plus attention dKV at 84.59% of warmed dispatch share. X23 and X29–X31 attack exactly that.

But GEMMs are only **17.1% of operation count**. Host cost scales with operation count, not with GPU time.

> **These are two different bottlenecks requiring two different fixes, and optimizing GEMM arithmetic does not
> reduce host cost at all.**

`adamw_step` is the clearest case: **128 consecutive dispatches**, one per parameter tensor, in a single
unbroken run. `add` is a second run of 127. Together that is 253 dispatches — 14.9% of the entire graph — doing
elementwise work on independent tensors that could be issued as one dispatch each.

This is the well-established multi-tensor-apply pattern (NVIDIA Apex's fused optimizers, PyTorch's `_foreach_*`
family). It is not novel; what is new is the measurement showing how much of *this* graph it covers.

The gradient-norm chain is the other half. The pairs

```
sum_sq_reduce_stride -> sum_reduce   182 occurrences
sum_reduce -> sum_sq_reduce_stride   180
sum_reduce -> sum_sq_reduce           74
sum_sq_reduce -> sum_reduce           74
```

are a per-tensor two-pass sum-of-squares tree, run for every parameter tensor to produce one global scalar for
gradient clipping. That accounts for the 518 reduction operations, **30.4% of the graph**, to compute a single
number.

## 4. What this does and does not license

**Does:** it identifies, from physical data, that ~24% of dispatches are removable by batching alone and that
reductions plus the optimizer are ~38% of the graph. Combined with X39's finding that per-dispatch phases are
65.3% of host time, the expected host-time reduction from removing a dispatch is roughly proportional.

**Does not:** claim a speedup. Specifically:

- The trace records execution order, kernel identity, buffer count, write mask, dispatch geometry and logical
  shape — but **not buffer identities**. Adjacency is a strong prior for producer→consumer inside a transformer
  block, but it is not proof. Each candidate needs its dependency confirmed in the backend.
- Host time is ~21.6% of the step on the 3090 after the X21 slab-cap promotion (344.55 ms host of 1,594.44 ms).
  So removing 100% of per-dispatch host cost is bounded by roughly 1.16× end to end, not by the 23.5% figure.
- Batched dispatches still perform the same arithmetic. GPU time falls only to the extent that per-dispatch
  launch overhead and barrier stalls were real, which X39 could not measure on a physical device.
- The 41.5% "top 5 pairs" figure double-counts overlapping candidates and is an upper bound only.

**The honest expected value is a host-side win of up to ~1.15× end to end, not a transformative one.** It is
attractive because it is exact-arithmetic, locally implementable, and independent of the GEMM work already in
flight — not because it is large.

## 5. Goal set from this

> **Reduce the training step's operation count from 1,703 by at least 30%, with bit-exact loss and gradient
> parity, by batching independent same-kernel dispatches — starting with the optimizer and gradient-norm
> reduction, which together are 38% of the graph.**

Ordered by (dispatches removed) ÷ (implementation risk):

| # | Change | Dispatches removed | Risk |
|---:|---|---:|---|
| 1 | Multi-tensor `adamw_step` — one dispatch over all 128 parameter tensors | 127 | low; tensors are independent, no cross-dependency |
| 2 | Multi-tensor gradient accumulation `add` | 126 | low; same structure |
| 3 | Fused multi-tensor sum-of-squares for the global grad norm | up to ~380 | medium; changes the reduction tree, needs numerical care |
| 4 | Batch the 36 `transpose` runs and 36 `rope` pairs | 108 | medium; needs dependency confirmation |

Items 1 and 2 alone remove **253 dispatches, 14.9% of the graph**, and are the cheapest exact-arithmetic
changes available. They require no new numerical behaviour: the same per-tensor arithmetic, issued once.

**Gate:** bit-exact loss and gradient parity against the current path on an identical checkpoint, batch, seed
and token IDs, plus the physical-run acceptance protocol in the handoff §13. A dispatch-count reduction that
changes the loss trajectory is a failure, not a trade.

## 6. Reproduce

```bash
node scripts/x40-fusion-opportunity-scan.mjs \
  /mnt/donto-data/donto-resources/benchmarks/alpha-helios-3090-portfolio-20260803/\
alpha-helios-graph-trace-3090-b10-20260803-r4/modes/slab8_pool64/run/gpu-graph-trace.jsonl
```

The scan is deterministic and reads only preserved evidence; it needs no device.
