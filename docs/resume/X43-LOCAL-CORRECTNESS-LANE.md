# X43 — a local correctness lane: 30 of 33 parity tests now run without a GPU

**Date:** 2026-08-04
**Status:** capability added and verified locally. No speedup claimed. **The training capability guard is
untouched.**
**Cost:** $0.06 of wasted rental, described in §5, which is what prompted this.

---

## 1. Why this exists

The operator observed that the previous agent made faster progress by *"improving all the algorithms by
testing them locally and seeing all the problems with Helios before kicking off big runs."*

That is visible in the record — X20 was offline lifetime analysis, X38 a CPU microbenchmark over a preserved
trace, and X25–X31 were all implemented and locally verified before anything was rented. GPU money was spent
to **measure** things already known to be correct.

This session did the opposite: rented a pod to go and find out, and burned $0.06 on a host that never became
reachable. The observation is correct and this record acts on it.

## 2. What was blocking local validation

`packages/tests/src/parity-helios.test.ts` contains the numerical parity suite — per-op forward parity against
`cpu_ref`, tiny-GPT forward and backward parity where *every parameter gradient* is compared, AdamW step
parity, mixed-precision cast roundtrips, and rmsNorm/rope autograd parity.

The entire suite is gated on `assessHeliosTrainingDevice(...).supported`, so on this box it skipped in full.
**33 correctness tests, none of them ever run locally.**

That gate is correct for training and over-broad for testing:

- The guard exists because a wrong subgroup width corrupts gradients **silently** during training.
- Every assertion in this suite compares against `cpu_ref`. A wrong result fails **loudly**.

Those are opposite failure modes, so the same restriction is not warranted for the second.

## 3. The change

An opt-in environment flag, `ALPHA_PARITY_ALLOW_SOFTWARE_DEVICE=1`, lets the **correctness suite only** run on
a software Vulkan device.

- `assessHeliosTrainingDevice` is **not modified**. The trainer still refuses to train on such a device.
- It is deliberately **not** applied to the performance suite, where software timings are meaningless.
- When active the suite prints that the run is correctness-only and confers no promotion.

## 4. Result

| Configuration | Passed | Failed |
|---|---:|---:|
| Software device, defaults | 22 | 11 |
| `+ HELIOS_DISABLE_COOP_MAT=1` | **30** | **3** |

**The 11 → 3 improvement** is entirely the cooperative-matrix path. lavapipe advertises
`coopMatSupported: true` but `vkCreateComputePipelines` fails with `VkResult=-13` on the cooperative shaders,
which assume a 32-lane layout. That is a genuine hardware limitation, correctly surfaced.

**The remaining 3 failures are subgroup-width artifacts, and the evidence is the failing element indices.**
The test tensors are `[6,10]`, so rows are 10 wide, and lavapipe's subgroup is 8:

| Failing test | Element | Row | Row-local index |
|---|---:|---:|---:|
| `softmax axis=-1 [6,10]` | 38 | 3 | **8** |
| `crossEntropyMaskedBackward` | 39 | 3 | **9** |
| `crossEntropyUnlikelihoodMaskedBackward` | 39 | 3 | **9** |

A row of 10 processed by a subgroup of width 8 gives lanes 0–7 as one full subgroup and lanes **8–9** as a
2-lane remainder. **All three failures land in that remainder and nowhere else.** The softmax kernel family
carries 26 subgroup operations, and the source comments the register-resident variant as "single-pass read,
subgroup reduce". Selecting the vec4 variant does not help because it is subgroup-dependent too.

On hardware with subgroup 32 a row of 10 sits entirely inside one subgroup and this remainder path is never
exercised, which is why these three pass on a real GPU and fail here. **They are artifacts of the test device,
not defects in Helios.**

## 5. What this is worth

30 of 33 numerical parity tests — including *every parameter gradient* of a tiny-GPT backward pass and a full
AdamW step — can now be validated on this box, for free, in about 20 seconds.

The value is not speed; it is **arriving at rented hardware already correct**. Every numerical regression this
lane catches is a regression that would otherwise have been discovered at $0.22/hour, on a machine with a
two-hour termination deadline, in the middle of a staged X25–X31 bisection.

That matters directly for the current program: X25–X31 are *implemented but never physically validated*, and
the handoff's Phase B plan validates them one stage at a time on rented hardware. Running each stage through
this lane first converts part of that paid bisection into free local work.

## 6. Honest limits

- **This does not license promotion.** A pass here is correctness on a software device, not physical
  validation, and the four states — implemented, locally verified, physically measured, quality-promoted —
  remain distinct. This lane moves work from the first state to the second and no further.
- The cooperative-matrix path, which is where the current arithmetic frontier lives (X23, X32, X37), **cannot**
  be validated here at all. Those still need hardware.
- Three softmax-family tests must be excluded, with the reason recorded above rather than suppressed.
- Software-device numerics are not bit-identical to NVIDIA in general; the suite's tolerances
  (`FWD_REL_TOL 1e-3`, `GRAD_REL_TOL 1e-2`) are what make the comparison meaningful, and a test that passes
  here could still drift on hardware within those bounds.

## 7. Reproduce

```bash
cd /mnt/donto-data/workspace/alpha2
ALPHA_PARITY_ALLOW_SOFTWARE_DEVICE=1 \
HELIOS_DISABLE_COOP_MAT=1 \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
npx vitest run --root packages/tests parity-helios
# expect 30 passed, 3 failed (softmax family, subgroup-8 remainder)
```
