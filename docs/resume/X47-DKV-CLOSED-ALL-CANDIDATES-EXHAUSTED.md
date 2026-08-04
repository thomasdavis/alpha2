# X47 — dKV closed by bound; all four routes to 50,000 are now exhausted

**Date:** 2026-08-04
**Evidence level:** E1 (arithmetic over measured shares). Free, no device.
**Status:** closes the last of X44's four candidates. **The 50,000 target has no identified route.**

---

## 1. The candidate

X44 listed "attention dKV work-partition redesign" as a route to the 1.63× residual, noting both prior
attempts were already falsified in X22:

- **dKV-v2** (four query rows per loop body, for instruction-level parallelism): numerically valid but
  **4.50× slower** than the selected kernel.
- **Non-square tiles**: showed an apparent 44% speedup that was a **false positive from skipped causal work**.
  The kernel used the key-tile ordinal as the query-tile ordinal and loaded one query row per invocation, an
  identity valid only when `Br == Bc`. Correcting the causal block indexing and staging all `Br` rows restored
  exact trajectory behaviour and removed the speedup. The generator now rejects `Br` not an integer multiple
  of `Bc`, and the selected 32×32 tile remains fastest.

So the mechanism question was: could a *third*, genuinely different work-partition succeed where those failed?

## 2. That question does not need answering, because the ceiling is too low

Before designing a third mechanism, bound what a perfect one could be worth. Using the X21 selected 12 GiB
policy (step 1,594.44 ms = 344.55 ms host + 1,249.49 ms GPU blocking):

| dKV share of dispatch | ms of step | % of step | Ceiling if **eliminated entirely** |
|---:|---:|---:|---:|
| 10.0% | 124.9 | 7.8% | 1.085× |
| 15.9% | 198.7 | 12.5% | 1.142× |
| 20.0% | 249.9 | 15.7% | 1.186× |
| 25.0% | 312.4 | 19.6% | 1.244× |
| 30.0% | 374.8 | 23.5% | 1.307× |

**For total elimination of dKV alone to supply the 1.63× residual, dKV would have to be 49.3% of warmed
dispatch.** X22 measures the top three GEMM layouts alone at **67.28%**, so dKV cannot be anywhere near 49.3%.

Two things make this robust:

1. **The conclusion does not depend on dKV's exact share.** Across the entire plausible range it caps between
   1.09× and 1.31×, all short of 1.63×.
2. **Elimination is impossible in principle.** dKV is required gradient work — the key and value gradients
   must be computed. The table bounds a physically unreachable best case, so the real ceiling is strictly
   lower.

X22 reached the same conclusion in prose — *"Reaching 50,000 tokens/s cannot come from dKV tile tuning or
another allocator percentage point"* — and this puts a number on it.

## 3. All four candidates are now closed

| # | Candidate | Closed by | Basis |
|---:|---|---|---|
| 1 | Attention dKV work-partition redesign | **X47 (this)** | ≤1.31× even if eliminated entirely; would need 49.3% of dispatch; two mechanisms falsified in X22 |
| 2 | Sequence packing | X46 | Void — both loader paths are padding-free by construction, no pad token exists |
| 3 | Larger effective batch | X6 | Batch is already 2.30× above the measured gradient noise scale |
| 4 | Arithmetic below FP16 inputs | X24 | Backward quality gate already failed at FP16; reverses after step 125 |

> **There is now no identified mechanism that could take Helios from ~30,600 to 50,000 tokens/s on an
> RTX 3090.**

This is the strongest form the X44 revision can take. It is not "we have not found one yet" — it is that every
route anyone named has been closed by a specific measurement or bound, and each closure is recorded with the
number that closed it.

## 4. What this does and does not mean

**Does not mean 50,000 is impossible.** It means reaching it requires a mechanism outside the set anyone has
proposed. Candidates would have to come from outside the current frame — a different model shape with better
arithmetic intensity, a different device, or a fundamentally different training formulation. Each would need
its own E0 mechanism audit before any implementation.

**Does mean the 30,000 target in X44 should be adopted.** 30,000 sits just under the ~30,600 ceiling and is
reachable by finishing work already begun: physically validating X25–X31 (locally verified in X43/X45, never
measured on hardware) and realising the cooperative arithmetic advantage at whole-step level.

**Does not weaken the program.** Four closed routes with numbers attached are worth more than four open ones
with hope attached. The next agent will not spend a rental rediscovering that dKV tuning caps at 1.14×.

## 5. Method note

This closure cost one arithmetic table and no hardware. The temptation was to design a third dKV work
partition, because two failures suggest a third attempt — but the operating method's §3.5 ladder asks for the
cheapest evidence that can kill the idea, and the cheapest evidence here is a ceiling computed from shares
already measured.

The generalisable form: **when two attempts at a mechanism have failed, bound the prize before designing a
third.** If a perfect version cannot reach the target, the failures were never the problem.

## 6. Reproduce

```bash
python3 - <<'PY'
step, gpu = 1594.44, 1249.49
for share in (0.10, 0.159, 0.20, 0.25, 0.30):
    frac = gpu*share/step
    print(f"{share*100:5.1f}% of dispatch -> {1/(1-frac):.3f}x if eliminated entirely")
print(f"share needed for 1.63x: {100*(1-1/1.63)*step/gpu:.1f}% of dispatch")
PY
```
