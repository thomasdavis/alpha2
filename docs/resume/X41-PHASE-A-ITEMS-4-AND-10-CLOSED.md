# X41 — Phase A items 4 and 10 closed

**Date:** 2026-08-04
**Status:** verification record. No code changed, no speedup claimed.
**Closes:** handoff §12 Phase A item 4 (reproduce X38 before reinterpreting it) and item 10 (verify whether
`x10b_consumed_metric.py` is still an open current question).

---

## 1. Item 4 — X38 reproduced

The handoff requires reproducing the X38 microbenchmark *before* changing its interpretation. X39 extended
that interpretation (arguing the native `decode` phase at 2.4% of host time independently corroborates X38
from the other side), so the reproduction was owed.

**Input verified first.** The declared trace SHA-256 matches the file on disk exactly:

```
9c1cf2c71a5eaea5a1cec333b57bdb1fd9def2aa6f11125906e5f4856c161663
```

**Result:**

**Result, across the original and two independent reproductions:**

| Metric | Recorded 06:58Z | Repro #1 | Repro #2 | Verdict |
|---|---:|---:|---:|---|
| Packed-output parity SHA-256 | `5e00a088…3df1e6` | same | same | **3/3 exact** |
| Parity pass | true | true | true | match |
| `reuse_then_patch` speedup | 3.567× | 3.068× | 2.867× | 2.87–3.57× |
| `copy_then_patch` speedup | 2.543× | 2.046× | — | — |
| Saving per step | 228.36 µs | 196.70 µs | 171.67 µs | 172–228 µs |
| Fraction of the 344.55 ms host build | 0.0663% | 0.0571% | **0.0498%** | **0.050–0.066%** |

**The byte-parity hash reproduces exactly on all three runs**, so the transformation under test is verifiably
identical each time. Only the timings move, by about 30%.

That spread is host load, not a discrepancy in the measured object: this is a CPU microbenchmark on a shared
box carrying other tenants, where load average reached ~18 on 8 cores the same day. The quantity that would
signal a changed transformation — the output hash — never moved. Reporting the range rather than the original
point estimate is the honest form of this result.

**The conclusion is unchanged and slightly strengthened.** X38 rejected static packed-dispatch encoding as
immaterial at 0.0663% of the host interval. Three runs put it at 0.050–0.066%, and the two reproductions both
land *below* the original. Every figure in that range is three orders of magnitude below anything that could
explain the host bottleneck.

X39's extension of that interpretation is therefore licensed: two independent measurements of the pack/unpack
path — X38 on the JavaScript encoder, X39 on the native decoder at 2.4% of host time — agree that byte
packing is not where the interval lives.

## 2. Item 10 — `x10b_consumed_metric.py` is still open, and still correctly ranked below host work

The handoff asks whether this remains a current open question, warning that "old reports described it as
unfinished; it should not outrank the host-bound localization unless current evidence has changed."

**Has it been run?** No. `experiments/x10b_consumed_metric.py` is present; `results/` contains only
`x10_metric_mismatch.NOTE.md`, the record of the earlier underpowered attempt being stopped. No result
artifact exists for the redesigned version.

**Has the evidence changed?** No. Nothing in X20–X40 revisits the attention closures B1–B3. The attention
work in that range is entirely about making the *existing* attention faster, not about whether it could be
approximated:

| Record | Subject |
|---|---|
| X22 | dKV tile verdict, K32 and dKV-v2 rejected |
| X27 | residual-add plus RMSNorm fusion |
| X28 | grouped-QKV head layout and RoPE fusion |
| X29 | combined QKV/Flash backward |
| X30 | token-major Flash output |
| X35 | direct grouped-QKV Flash rewrite, deferred |

**So the question is open but correctly deprioritized**, and the accounting says why. Attention scores are
10.8% of training arithmetic at the contracted sequence length of 1,024, so even a free and perfect attention
replacement is capped at **1.12× end to end**. That ceiling does not depend on the metric the closures were
measured in — it is an arithmetic share, not an approximation-quality claim. Re-opening B1–B3 in the consumed
metric could change *whether* a cheaper attention is available; it cannot change *how much it would be worth*
at S=1024.

Meanwhile the host-bound localization it was being ranked against produced X39 and X40, which identified a
different and larger structural target: per-dispatch host phases at 65.3% of host time, and an operation graph
that is 30.4% reductions against 17.1% GEMMs.

**Verdict: leave `x10b` unrun.** It stays on the register as an open question with a stated ceiling. It should
be run when either (a) a long-context Alpha is being designed, where the attention share rises and the ceiling
with it, or (b) the host and operation-count work is exhausted and 1.12× becomes the largest remaining item.

## 3. Phase A status after this record

| Item | State |
|---|---|
| 1–3. Verify revision, manifest, clean tree; read current records | done |
| **4. Reproduce X38 before reinterpreting it** | **done — this record** |
| 5. Instrument the native interval beneath JS packing | done — X39 |
| 6. Print host subintervals beside device dispatch time | done — X39 |
| 7. Attack the binding constraint, then re-profile | X40 identifies it; implementation not started |
| 8. Source-guided implementation where the mechanism maps faithfully | not started |
| 9. Keep ordinary math exact until X25–X31 parity is physical | holding |
| **10. Verify whether `x10b` is still an open question** | **done — this record; open, deprioritized** |

Phase A is complete except item 7's implementation and item 8. Phase B remains unauthorized.

## 4. Reproduce

```bash
cd /mnt/donto-data/donto-resources/research/alpha-helios-reimagined/experiments
node x38_static_packed_dispatch_template.mjs
```

Deterministic in its transformation (the output hash is stable); its timing figures vary with host load.
