# X42 — the bottleneck has moved; correcting X40's stated value

**Date:** 2026-08-04
**Status:** correction and Phase A item 7 answer. No code changed. **Contains a correction to X40.**
**Closes:** handoff §12 Phase A item 7 ("attack the currently binding measured constraint, then re-profile;
do not keep optimizing yesterday's bottleneck after it moves") and item 8.

---

## 1. The correction

X40 identified 253 consecutive dispatches (`adamw_step` ×128, `add` ×127) removable by multi-tensor batching,
and stated the expected value as "a host-side win of up to ~1.15× end to end."

**That figure was wrong.** 1.15× is approximately the bound for eliminating *all* host build time. It is not
the value of removing the 253 dispatches identified. The correct arithmetic, from the X21 selected 12 GiB
policy on the RTX 3090:

```
step          1594.44 ms
  host build   344.55 ms   21.61%
  gpu blocking 1249.49 ms  78.37%
```

Removing a fraction *f* of the 1,703 dispatches removes *f* of the per-dispatch host cost. X39 measured
per-dispatch phases at 65.3% of host on llvmpipe; if `cmd_begin` is the Mesa artifact X39 flagged, the
per-dispatch share on NVIDIA would approach 100% of host. Both bounds:

| Change | Dispatches | % of graph | Step time saved | End to end |
|---|---:|---:|---:|---:|
| Multi-tensor `adamw_step` only | 127 | 7.5% | 16.8–25.7 ms | **1.011×–1.016×** |
| `adamw_step` + `add` | 253 | 14.9% | 33.4–51.2 ms | **1.021×–1.033×** |
| All same-kernel run batching | 401 | 23.5% | 53.0–81.1 ms | **1.034×–1.054×** |
| *Ceiling: all host build eliminated* | — | — | 344.55 ms | *1.276×* |

**Multi-tensor AdamW is worth 1.1–1.6%, not 15%.** The entire operation-count program is worth 3.4–5.4%.

## 2. Why this matters more than the number

Item 7 carries an explicit warning: *do not keep optimizing yesterday's bottleneck after it moves.*

It has moved. The host interval **was** the dominant cost — X20 measured 3,216 ms of host build against
1,479 ms of GPU blocking on the L40S, 68.5% host. X21's device-adaptive 12 GiB slab cap then cut host build
27.43% and promoted it as the default. On the current 3090 configuration host is **21.61%** and GPU blocking
is **78.37%**.

So the binding measured constraint is now **GPU arithmetic**, and the handoff's own §6 records where inside it:
the top three GEMMs plus attention dKV are 84.59% of warmed dispatch share.

Against a remaining gap of **6.59×** to the 50,000 tokens/s target, the entire host-side program — including
perfect elimination of all host build time, which nothing proposes — contributes at most 1.276×. The other
5.16× has to come from the arithmetic path.

## 3. Consequence: the item-7 implementation is not warranted, and that is the finding

Implementing multi-tensor AdamW would mean:

- writing a new buffer-device-address SPIR-V kernel, because 128 tensors × 4 buffers is 512 descriptor
  bindings and far past `maxPerStageDescriptorStorageBuffers`;
- new per-step address-table upload and dispatch plumbing;
- all of it on the **optimizer** — the single most correctness-critical path in the trainer, where a silent
  error corrupts every subsequent step rather than failing loudly.

For 1.1–1.6%. Against a constraint that is 78.4% of the step and untouched by the change.

**That trade is not justified, and declining it is the correct execution of item 7**, not a failure to execute
it. The instruction was to attack the binding constraint; the binding constraint is arithmetic, and the
honest first step of attacking it was the arithmetic in §1 that shows where it now lives.

X40's *measurements* stand — reductions really are 30.4% of the operation graph against GEMMs at 17.1%, and
253 dispatches really are consecutive and independent. Only its stated value was wrong. The finding remains
useful as the correct target *if and when* host becomes binding again, for instance on a device with a much
faster arithmetic path where the 21.61% host share would rise as a fraction.

## 4. Item 8 — source-guided implementation

Item 8 permits continued source-guided implementation "only when the mechanism maps faithfully to Alpha's
graph and optimizer."

Multi-tensor apply maps faithfully — it is the established Apex fused-optimizer / PyTorch `_foreach_*` pattern
and Alpha's optimizer has exactly the independent-tensor structure it assumes. It is nevertheless declined by
§3's arithmetic rather than by any mapping objection.

The source-guided work that *would* address the binding constraint is the arithmetic path already implemented
in X25–X31 and unvalidated. Item 9 holds ordinary math exact until physical parity of X25–X31 is established,
and that parity requires Phase B, which is unauthorized. **So no new source-guided implementation is warranted
locally**: the mechanisms worth implementing are already implemented and waiting on physical validation, and
implementing more of them before validating any would deepen the pile of unvalidated work the handoff
explicitly flags as a hazard.

## 5. Phase A status

| Item | State |
|---|---|
| 1–3. Verify revision, manifest, clean tree; read records | done |
| 4. Reproduce X38 before reinterpreting it | done — X41, 3/3 byte-parity |
| 5. Instrument the native interval | done — X39 |
| 6. Print host subintervals beside dispatch time | done — X39 |
| **7. Attack the binding constraint, then re-profile** | **done — this record.** Constraint identified as GPU arithmetic (78.37% of step); host-side implementation declined at 1.1–1.6% |
| **8. Source-guided implementation where it maps faithfully** | **done — this record.** None warranted locally; the mechanisms that address the constraint are X25–X31, already implemented and blocked on physical parity |
| 9. Keep ordinary math exact until X25–X31 parity is physical | holding — this is why 7 and 8 resolve as they do |
| 10. Verify `x10b` is still open | done — X41; open, deprioritized, 1.12× ceiling at S=1024 |

**Phase A is complete.** Every remaining lever of consequence is behind Phase B authorization.

## 6. What the next authorized session should do

The handoff's §12 Phase B order is unchanged and is now the whole remaining program. Its first two steps are
cheap and decisive:

1. Rent the cheapest suitable RTX 3090 with explicit auto-termination.
2. Capture the X37 cooperative tuple list **before** writing any float32-input shader; if no eligible
   F32-input tuple exists, close the TF32 branch immediately.
3. Validate X25–X31 **one stage at a time** against the legacy path — not combined — so staged bisection can
   identify regressions.

Two measurements should be added to that run at near-zero cost, both now instrumented:

- `HELIOS_HOST_TIMING=1` for the X39 phase table with numbers that transfer, settling whether `cmd_begin` is a
  Mesa artifact and what the real per-dispatch share is;
- the X40 scan against a fresh trace, to confirm the operation mix has not shifted under X25–X31.

## 7. Method note

This record exists because a stated value was checked before it was implemented. The measurement in X40 was
sound; the number attached to it was not, and the gap between them was three hours of arithmetic away from
being several days of work on the optimizer.

That is the trivial-baseline discipline from `TWO-PRINCIPLES-2026-08-03.md` applied to ourselves: before
building the clever thing, compute what it is worth against the thing it competes with.
