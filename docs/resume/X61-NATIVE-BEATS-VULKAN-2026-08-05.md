# X61 — the native stack overtakes Vulkan, 90 → 2943 tok/s

**Date:** 2026-08-05 · **Card:** RTX 3070 (RunPod `alpha-bridge`, $0.13/hr) · **Gate:** green, 40 passed / 0 skipped, 5 layer suites

## Result

Benchmark is `packages/tests/bench-native-vs-vulkan.mjs`: 2 layers, 64 embd, 4 heads,
vocab 64, 32 tokens a step, forward and backward.

| backend | tok/s | step | against Vulkan |
|---|---|---|---|
| **helios-native** | **2943** (2948, 2837, 2875, 2724) | 10.9 ms | **4.5x ahead** |
| helios-vulkan | 612 (542–651) | 52.3 ms | — |
| cpu_ref | 1471 | 21.8 ms | — |

Loss identical to `cpu_ref` at 4.1834 on every backend and every run.

Starting point was 90 tok/s / 354.6 ms, i.e. **0.14x of Vulkan**. Overall **32.7x**.

## The instrument had to be fixed first

Two faults, both of which had produced published numbers:

- **The backends interfered.** All three ran in one process, and with the native
  channel open the same Vulkan binary measured 142 tok/s where it measures 628.
  Now one backend per process: `node bench-native-vs-vulkan.mjs native|vulkan|cpu`.
- **Warmup was too short.** This card idles at **210 MHz against 2100 max** and
  `nvidia-smi` cannot lock clocks inside the container. Ten steps is 0.85 s and
  left it part-ramped: a cold process measured 127 tok/s against a warm 628, a
  4.9x error. Warmup is now by TIME (3 s), which is where the medians stop moving.

Anything measured before this is not comparable.

## What actually cost, in the order it was found

### 1. The driver — 65% of the step

Every backend call cost 500–900 µs *regardless of the work*. `reshape` launches
nothing and cost 496 µs; `zeros` cost 869. A cost that does not vary with the
work is not the work.

| | |
|---|---|
| alloc 4 KB, fresh | **802.3 µs** |
| alloc 4 KB, from the pool | **1.0 µs** |
| flush, nothing pending | 0.1 µs |
| one kernel, enqueued and drained | 28.9 µs |

800x, and it is three ioctls and an `mmap`. Nothing frees intermediates, so all
~283 allocations a step paid it: **~227 ms of a 349 ms step**, in RM, doing no
arithmetic. `stats` had been carrying the evidence all along — `allocations 5671,
pooled 0`.

Tensors are now carved from **slabs mapped once**. `SLAB_BYTES` is 4 MiB and that
is measured, not chosen: `gaia_alloc` asks for physically contiguous pages and
64 MiB — the arithmetic answer — failed outright. `tools/slab_probe.c` walks the
sizes; 4 MiB is the kernel's `MAX_ORDER` ceiling.

`carved` is now reported beside `allocations`, because slabs made the latter rare
and it would otherwise have gone quiet on exactly the fault it exists to reveal.

### 2. `reshape` was draining the queue

The comment above it calls it "the one operation that is free". It was written
`data: da.data`, and that spelling READS the getter — which is where the queue
barrier lives. 50 drains a step, 18.4 ms, more than any real kernel. It also
flattened the barrier away, so a later read of the view would not have flushed at
all. Carrying the getter across fixes both.

### 3. Write-combined memory — the whole remaining gap

Two wrong turns first, both killed by measurement and both worth keeping:

- **Drains are cheap.** 8 kernels drain in ~51 µs however long the host idled
  first, and the same 32 kernels with the same 8 ms of host work cost 8.6 ms
  whether split across 16 drains or 1. Batching harder would have bought nothing.
- **Stall counts were not the problem.** `hp_ctrl_safe()` encoded stall 15, the
  maximum. Lowering it to 7 on measured evidence changed 383 tok/s into 386.

Then the measurement that mattered: fire the step's *real* launch mix — same
kernels, same shapes, same counts — and drain once. **6.92 ms.** Drain every four
launches: 7.06 ms. A step's kernels are 7 ms of an 85 ms step. **The backend was
host-bound**, and the 32 ms the profile attributed to drains was never GPU time.

`layerNorm` gave it away: 5 calls, 12.1 ms, against 6.6 µs of GPU work each. It
broadcasts a weight and a bias, and `expand()` does that in a JavaScript loop over
device memory. Over 2048 floats:

| | write-combined | plain | ratio |
|---|---|---|---|
| read | 225.7 µs | 1.4 µs | **161x** |
| copy | 387.9 µs | 1.0 µs | **388x** |
| write | 1.2 µs | 1.0 µs | 1.2x |

Write-combining is a one-way street: right for a pushbuffer, which the host writes
and the GPU reads; wrong for a tensor, which the host reads constantly — every
broadcast, slice, concatenation and permutation, and every CPU fallback in
autograd. `gaia_alloc` asked for `WRITE_COMBINE` for everything.

**Tensor slabs are now CACHED**; pushbuffer, QMD and constant bank keep
write-combining. Device reads went 225.7 µs → 1.4 µs. Safe for the same reason
CUDA's pinned memory is cacheable by default — on x86 the root complex snoops —
and the known-answer suite proves it, since it writes an input on the host, runs a
kernel, and reads the result back.

## Kept although it paid nothing: the stall table

`tools/stall_probe.c` sweeps the stall on a producer whose consumer reads its
result and asks the hardware where the answer stops being right:

| instruction | minimum |
|---|---|
| IADD3 / IMAD / FFMA / SHF+LOP3 | 4 |
| MOV c[] | 5 |
| IMAD.WIDE, HADD2 | 0 |
| **ISETP → @P** | **13** |

The default is now 7. It bought no throughput and it is kept for what it caught:
**ISETP needs thirteen where the ALU needs four**, and every ISETP in the tree is
written `hp_ctrl_safe()` — matmul's loop, all three reductions, the causal mask,
dropout, cross-entropy. A blanket lowering justified by the ALU figure would have
handed all of them 7, and a stale predicate does not fault — it masks the wrong
elements and returns a plausible number. `sm86_flow.c` clamps it in the emitter,
where no call site can disagree.

A single warp is the worst case, which is why a probe is enough: the stall spaces
one warp's own stream, and additional resident warps can only add delay.

## Bugs found on the way

- **The handle wrapped at 65,535.** 16 bits of index, and `make_handle` masked
  `index+1` into the field — so the last slot produced a non-zero handle that
  resolved to nothing. The caller's "did it fail" check passes and it surfaces
  later as *"allocated handle has no view"*: a message about the view, from a bug
  in the index. Now 20 bits of index, 12 of generation, refusing one short of the
  mask.
- **A reshape shares its buffer.** Two TensorData objects, one `NativeBuffer`, and
  a live/dead flag — so releasing either view freed the memory the other used.
  This is what `tensor.c` meant by the tape "naively freeing tensors the graph
  still references"; it was not the tape. Reference counted now.

## What is still wrong — the next thing

**Nothing frees intermediates, in the benchmark *or* in training.**

`trainer.ts` builds its release callback with
`typeof backend.releaseGpuTensor === "function"`. This backend spells the method
`release`, so the probe fails, `releaseFn` stays undefined, and a real training
run reclaims nothing: `carved` climbs ~283 a step until the tensor table is gone.
299 slabs — 1.2 GB — after a few hundred benchmark steps.

The alias is one line and **is deliberately not added yet**. With reference
counting a naive release policy now gets further than it did — past matmul's
backward — and then dies in gradient accumulation, so a lifetime is still wrong
somewhere between the tape and this backend. Adding it would turn a trainer that
is leaky and correct into one that is tidy and wrong, failing mid-run.
`packages/tests/micro-release-recycles.mjs` is the harness that will settle it.

**After that**, the step is roughly GPU-bound: ~7 ms of kernels in an 11 ms step,
and the matmul is nearly all of it — 250 µs for 32x64x256, against 5–35 µs for
everything else. The 16 fused backward methods autograd probes for
(`layerNormBackward`, `geluBackward`, `rmsNormBackward`, `clampBackward`,
`embeddingBackward`, …) are implemented by Vulkan and by none of this backend, so
every backward still falls back to JavaScript. That is the next block of work, and
it is now worth doing — before the memory fix it would have been invisible under
the 161x.

## Tools added

| | |
|---|---|
| `tools/slab_probe.c` | largest contiguous sysmem allocation this machine gives |
| `tools/stall_probe.c` | minimum safe stall per instruction form, by known answer |
| `tests/profile-native-step.mjs` | per-method cost, drain attribution, and who reads `.data` |
| `tests/micro-native-costs.mjs` | the fixed costs: alloc, flush, enqueue |
| `tests/micro-drain-gap.mjs` | whether a host gap makes a drain dear (it does not) |
| `tests/micro-kernel-shapes.mjs` | per-launch GPU cost at the model's real shapes |
| `tests/micro-step-mix.mjs` | a whole step's launches, one drain |
| `tests/micro-host-memory.mjs` | host read/write speed on device-mapped memory |
| `tests/micro-release-recycles.mjs` | does the pool recycle when something frees |

## Method note

Every wrong turn here was a plausible theory that a measurement killed, and each
was cheaper to test than to implement. The two that would have cost the most —
batching harder, and writing a scheduler — were closed for about twenty minutes
each. The one that mattered was found by asking a question with no theory in it:
*what does the host actually cost?*
