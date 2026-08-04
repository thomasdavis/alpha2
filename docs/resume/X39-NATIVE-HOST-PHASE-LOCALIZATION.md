# X39 — native host-interval localization

**Date:** 2026-08-04
**Status:** instrumentation landed and locally exercised. **No physical-device measurement.** No speedup claimed.
**Answers:** Phase A items 5 and 6 of the 2026-08-04 handoff — "instrument the native interval beneath JS
packing" and "print host subintervals beside device dispatch time".
**Device:** llvmpipe (LLVM 21.1.8) via lavapipe. Absolute microseconds do **not** transfer to NVIDIA.

---

## 1. Why this exists

X38 rejected the JavaScript field-packing hypothesis: a static packed template made the encoder 3.57× faster
and saved 228.36 µs, which is **0.0663%** of the measured 344.55 ms host interval. That left the interval
unlocalized, with the explicit instruction that it "must be located in native Vulkan object lifecycle,
descriptor work, command recording/submission, waits, or other measured boundaries — not assumed".

So the native dispatch path is now instrumented directly.

## 2. What was built

`napi_batchExecuteAllImpl` in `packages/helios/native/helios_vk.c` is split into twelve disjoint phases with
a `CLOCK_MONOTONIC` accumulator, exposed to JS through two new addon entry points:

```
getHostTiming()   -> { enabled, batches, dispatches, clockReads, phases: { name: { us, calls } } }
resetHostTiming()
```

Surfaced as `HeliosBackend.getNativeHostTiming()` / `resetNativeHostTiming()`, and printed by the trainer as a
`[host_phases]` line beside the existing `[gpu_ops]` telemetry.

**Everything is gated on `HELIOS_HOST_TIMING=1`.** With the variable unset, `htBegin()` returns 0 after one
cached branch and `htEnd()` returns immediately, so the default dispatch path is unchanged.

The critical design decision is that **`ring_wait` is measured separately**. It is the wait for a ring slot's
prior submission to complete — a GPU-completion wait, not host work. Folding it into "host time" would recreate
exactly the misattribution X8 warned about, and §4 shows it would have been a catastrophic error here.

## 3. Method, and its one real compromise

The trainer **cannot** run on this box. Its capability guard rejects llvmpipe: device type 4 is not a discrete
or integrated GPU, and native subgroup size 8 is incompatible with the 32-lane kernel layouts. That guard is
correct — running anyway would produce silently wrong gradients — and it was **not weakened**.

Instead `scripts/x39-host-phase-probe.mjs` drives the identical native path with a synthetic dispatch stream:
1,703 dispatches per batch (the real per-step operation count), 3 storage bindings each, one push constant,
rotating buffers so the write-tracking and barrier logic behaves realistically. Command recording, descriptor
work and submission do not depend on kernel arithmetic being correct or on the device being a real GPU.

**What transfers:** the phase decomposition; the per-dispatch call counts; whether a phase is per-batch or
per-dispatch; whether descriptor sets are allocated per dispatch.
**What does not transfer:** absolute microseconds. Mesa's host-side driver cost is not NVIDIA's.

## 4. Result

20 batches × 1,703 dispatches = 34,060 dispatches.

| Phase | total µs | calls | µs/call | % of host |
|---|---:|---:|---:|---:|
| `ring_wait` | 2,545,532 | 20 | 127,276.6 | *excluded — GPU wait* |
| `cmd_begin` | 36,637 | 20 | 1,831.9 | 34.6% |
| `desc_update` | 24,757 | 34,060 | 0.727 | 23.4% |
| `barrier` | 24,035 | 34,060 | 0.706 | 22.7% |
| `push_const` | 9,372 | 34,060 | 0.275 | 8.9% |
| `cmd_dispatch` | 6,382 | 34,060 | 0.187 | 6.0% |
| `decode` | 2,534 | 34,060 | 0.074 | 2.4% |
| `bind` | 2,012 | 34,060 | 0.059 | 1.9% |
| `submit` | 124 | 20 | 6.179 | 0.1% |
| `cmd_end` | 8 | 20 | 0.408 | 0.0% |
| `pool_reset` | 4 | 20 | 0.205 | 0.0% |
| `desc_alloc` | 0 | **0** | — | 0.0% |

Host total excluding `ring_wait`: **105,864 µs = 3.108 µs per dispatch**.
Instrumentation self-cost: ~37,792 µs, **1.43%** of measured time, reported rather than assumed.

## 5. What this establishes

**(a) Separating the GPU wait was not a formality.** `ring_wait` is 2.55 s against 106 ms of host work — a
24× ratio. Any instrument that charged it to the host would have reported "96% host-bound" and sent the next
operator to optimize a wait for the GPU to finish. This is the single most important structural result here,
and it is a warning about how the earlier ~51% figure should be read on any device.

**(b) Host cost is per-dispatch dominated.** The phases that scale with operation count — `desc_update`,
`barrier`, `push_const`, `cmd_dispatch`, `decode`, `bind` — are **65.3% of host time** and all have 34,060
calls. Host cost is therefore a function of *how many operations the graph contains*, not of tensor size.
That is direct support for the operation-count reduction already underway in X27–X31, and it predicts those
fusions reduce host time roughly in proportion to the operations they remove.

**(c) X38's conclusion is independently corroborated from the other side.** X38 measured the JS encoder;
this measures the native decoder. `decode` is **2.4%** of host time. Two independent measurements of the
pack/unpack path both say it is not where the interval lives.

**(d) `desc_alloc` recorded zero calls**, because this device exposes `VK_KHR_push_descriptor` and the code
takes the push-descriptor branch. On a device *without* that extension the same workload would issue 34,060
`vkAllocateDescriptorSets` calls per 20 batches. That branch is untested for cost and is a portability risk
worth checking on the target device before assuming the measured profile applies.

**(e) `barrier` at 22.7% is a new, specific target.** A `vkCmdPipelineBarrier` is emitted per dispatch whenever
an input was written in the current generation. Nothing here proves it is reducible, but it is the second
largest per-dispatch phase and it had not previously been named as a candidate.

## 6. What this does NOT establish

- **`cmd_begin` at 34.6% is almost certainly a Mesa artifact.** 1,831 µs for a single
  `vkResetCommandBuffer` + `vkBeginCommandBuffer` pair is implausible on a production NVIDIA driver, where
  these are typically microseconds. **Do not carry this number to the 3090.** It is the clearest example of a
  figure that does not transfer.
- No claim is made about the real host interval's composition on a physical device. The transferable content
  is the decomposition and the call counts, not the times.
- No speedup was implemented or measured.

## 7. Verification

- `packages/tests`: **251 passed, 59 physical-gated, 0 failed** after the change.
- Default-path overhead: not resolvable above run-to-run variance on this shared host
  (timing off: 98.5, 97.9 ms/batch; timing on: 85.6, 106.0 ms/batch).
- Native addon rebuilds clean; the one compiler warning (`submitCmdBufSync` unused) pre-dates this change.
- `getHostTiming()` reports `enabled: false` and `clockReads: 0` when the environment variable is unset.

## 8. Next

The instrumentation is ready for a physical run. Under Phase B, `HELIOS_HOST_TIMING=1` on the real foundation
shape produces the same table with numbers that *do* transfer, and settles:

1. whether `cmd_begin` is genuinely per-batch-expensive on NVIDIA or a Mesa artifact (expected: artifact);
2. what fraction of real host time is per-dispatch, which sets the ceiling on fusion work;
3. whether `barrier` is worth attacking;
4. whether the target device takes the push-descriptor branch at all.

Until then, the actionable inference is (b): **host time scales with operation count**, so operation-count
reduction is the lever, and X27–X31 are pointed the right way.

## 9. Files

```
packages/helios/native/helios_vk.c          twelve-phase accumulator + 2 addon entry points
packages/helios/src/device.ts               NativeHostTiming type, addon signatures
packages/helios/src/backend.ts              getNativeHostTiming / resetNativeHostTiming
packages/train/src/trainer.ts               [host_phases] telemetry line, env-gated
scripts/x39-host-phase-probe.mjs            standalone probe, runs on any Vulkan device
```
