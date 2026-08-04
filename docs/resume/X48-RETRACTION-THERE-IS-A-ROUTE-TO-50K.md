# X48 — retraction: there *is* a route to 50,000, and X44/X47 were reasoning from the wrong baseline

**Date:** 2026-08-04
**Evidence level:** E1 (arithmetic over measured quantities). Free.
**Status:** **retracts the central claim of X44 and X47.** No measurement is overturned; the inference from
them was wrong.

---

## 1. What I got wrong

X44 concluded the identified levers cap at ~30,600 tok/s. X47 then closed the last candidate and concluded:

> "There is now no identified mechanism that could take Helios from ~30,600 to 50,000 tokens/s."

**That was wrong, and it was wrong in a specific, avoidable way: I reasoned about increments to the current
implementation and never asked what the hardware permits.**

Model FLOPs per step are fixed by the shape: 653.2 MFLOP/token × 10,240 tokens = **6.689 TFLOP**. Against the
RTX 3090's tensor peaks (71.2 TFLOP/s with FP32 accumulate, 142.3 with FP16 accumulate):

| Point | Step ms | TFLOP/s | % of FP32-acc peak | % of FP16-acc peak |
|---|---:|---:|---:|---:|
| Current baseline 6,422 | 1594.5 | 4.19 | 5.9% | 2.9% |
| X24 quality-bearing 7,762 | 1319.2 | 5.07 | 7.1% | 3.6% |
| **X44 "ceiling" 30,600** | 334.6 | 19.99 | **28.1%** | 14.0% |
| **Target 50,000** | 204.8 | 32.66 | **45.9%** | 23.0% |

A well-tuned trainer routinely sustains **35–50% MFU**.

So 50,000 tok/s is **45.9% MFU** — ambitious but squarely inside the normal range for a competent
implementation, and only **23.0%** if FP16 accumulation is usable anywhere. Meanwhile the "ceiling" I declared
sits at **28.1% MFU**, which is *below* what good implementations reach.

**I declared a ceiling beneath ordinary engineering quality and called it a limit.** The 3.94× figure was never
a bound on the hardware; it was a bound on patching today's code, presented as though it were the former.

The tell was visible and I walked past it: the cooperative microbenchmark reaches **101.6–118.7 TFLOP/s**,
which is **71–83% of the FP16-accumulate peak**. Individual GEMMs already run near hardware limits while the
whole step runs at 7.1%. A 10–12× gap between kernel efficiency and step efficiency is not a missing
mechanism — it is everything *around* the kernels.

## 2. The second error: pricing levers against a profile I intend to change

X42 dismissed operation-count reduction at 1.034–1.054× because host build is 21.61% of *today's* step. That
arithmetic was right and the conclusion did not follow, because the plan is to make GEMMs 5× faster, which
inverts the composition.

Composing the levers **sequentially**, where each fix changes what the next is worth:

| Stage | Step ms | tok/s | host | GEMM | other GPU |
|---|---:|---:|---:|---:|---:|
| Today | 1594.4 | 6,422 | 21.6% | 66.3% | 12.1% |
| + cooperative 5× on GEMM | 748.5 | 13,681 | **46.0%** | 28.2% | 25.7% |
| + host graph replay → ~0 | 403.9 | 25,351 | 0% | 52.3% | **47.7%** |
| + 4× on non-GEMM GPU work | 259.5 | 39,457 | 0% | 81.5% | 18.5% |
| + GEMM 7× | **199.1** | **51,424** | 0% | 75.8% | 24.2% |

After cooperative lands, **host becomes 46.0% of the step** — the single largest term, and the thing X42
priced at 1.03× and set aside. After host is removed, **non-GEMM GPU work becomes 47.7%** — the reductions,
unary and transpose operations X40 measured at 30.4% + 19.1% + 8.5% of the operation graph and X42 also set
aside.

I made exactly the error handoff item 7 warns about — *"do not keep optimizing yesterday's bottleneck after it
moves"* — in reverse: I **dismissed tomorrow's bottleneck using today's profile.**

## 3. The route

Four stages, none of which is a new mechanism. All are already-identified work that was individually
dismissed for being too small against a static profile:

1. **Cooperative arithmetic at whole-step level** — 4.99–5.81× measured per GEMM (X-coop-accum); currently
   promoted only for forward input conversion, since backward failed cost-to-quality (X24).
2. **Host elimination via static graph replay** — the step's op graph is static (X8: 1,444 identical ops on
   17 of 20 steps). X21 rejected *static buffer slots*; graph/command replay is a different mechanism and was
   never tried.
3. **≈4× on non-GEMM GPU work** — the 518 reductions, 326 unary and 144 transpose operations from X40. Note
   this is a *requirement derived from the budget*, not an achieved result.
4. **GEMM ≈7×** — beyond the 5× measured, requiring FP16 accumulate or better tiling.

## 4. What this is, and firmly is not

**This is a budget, not a prediction.** It states what each stage must deliver for the total to reach the
target. It is emphatically **not** a claim that 50,000 will be achieved, and it must not be read as one.

The program's own rule applies to this document above all: **never compose unvalidated multipliers.** Every
row of the §2 table is a composition, and none of it is measured end to end. Specifically:

- The 5× cooperative figure is a **per-GEMM microbenchmark**, never demonstrated at whole-step level.
- The **7× GEMM** stage likely needs FP16 accumulation, which already failed the backward quality gate in
  X24 — it may be available in forward only, in which case this rung is smaller.
- The **4× on non-GEMM work** is derived from the budget and has no supporting measurement whatsoever.
- Host → 0 is unreachable in practice; the table uses it as a bound.

So the correct reading is: **the target is not blocked by a missing mechanism. It is blocked by execution
quality across four known fronts**, and each front needs its own paired measurement before any of it is
believed.

## 5. What survives from X44/X47

The measurements all stand, and so do the individual closures *as stated*:

- dKV tuning really does cap at 1.09–1.31× (X47) — it is not a route **on its own**.
- Sequence packing really is void (X46) — the loader has no padding.
- Batch really is 2.30× above the noise scale (X6).
- Sub-FP16 backward really did fail the quality gate (X24).

What does **not** survive is the synthesis: *"therefore no route exists."* Those four were candidates for a
single 1.63× lever on top of a fixed 3.94×. The actual route is not one more lever — it is raising MFU from
7.1% to ~46% by composing work already identified.

**X44's revised target of 30,000 should therefore be treated as a near-term milestone, not as the ceiling.**
50,000 stays live.

## 6. Method note

The failure mode here is worth naming, because it is subtle and I repeated it three times (X42, X44, X47).

Bounding a prize before building is right, and it correctly killed the JS-encoding and gauge-quotient work.
But a bound computed against the current composition is only valid **if the composition stays fixed**. When a
program intends to change the composition, each lever must be priced against the profile that will exist
*when it lands*, not the one that exists now.

The general form: **an Amdahl bound is a statement about a fixed profile, not about a system.** Cite the
profile it was computed against, and recompute it after every accepted change.

And the check I should have run first, which would have caught all three at once: **compare the target to the
hardware's roofline before declaring anything unreachable.** 50,000 tok/s is 23–46% MFU. That number was one
division away the entire time.
