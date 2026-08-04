# X46 — sequence packing is void: the pretraining loader has no padding

**Date:** 2026-08-04
**Evidence level:** E0 (source and mechanism audit). Free. No device, no measurement needed.
**Status:** closes the last unexplored lever named in X44. **Negative result.**

---

## 1. Why this was asked

X44 revised the RTX 3090 target from 50,000 to ~30,000 tok/s, on the arithmetic that all identified levers
combine to ~3.94× against a 6.44× requirement. It listed four candidates that could supply the residual 1.63×,
and rated three of them as having evidence against them already:

| Candidate | Status in X44 |
|---|---|
| Attention dKV work-partition redesign | both prior attempts falsified (X22) |
| Larger effective batch | already 2.3× past the gradient noise scale |
| Arithmetic below FP16 inputs | quality gate already failed once at FP16 (X24) |
| **Sequence packing** | **"the only candidate with no evidence against it"** |

So sequence packing was the last live route to 50,000. The operating method's rule is to establish the
necessary condition at the cheapest evidence level that can kill it (§3.5), and the necessary condition here
is simply *that padding waste exists*. That is a source audit, not a measurement.

## 2. Result: there is no padding to remove

`packages/train/src/data.ts` has two batch-formation paths and **neither pads**.

**Default path — `nextBatchRandom`** (the constructor default is `packed = false`):

```ts
const maxStart = this.tokens.length - T;
for (let b = 0; b < B; b++) {
  const start = Math.floor(this.rng.next() * maxStart);
  inputs.set(this.tokens.subarray(start, start + T), dst);
  targets.set(this.tokens.subarray(start + 1, start + T + 1), dst);
}
```

Every sample is a contiguous `T`-token window drawn from a flat token array. Document boundaries are not
respected and no sequence is padded to length — **padding-free by construction**.

**Alternate path — `nextBatchPacked`**: `B` cursors advance sequentially through the corpus, and the source
comments it as "All documents naturally packed contiguously (no wasted padding)."

**And there is no pad token at all.** A search for `padToken`, `PAD` or `pad_token` across the pretraining
loader and the trainer returns nothing.

## 3. Consequence

**Sequence packing cannot yield any throughput on Alpha pretraining, because the mechanism it removes is not
present.** Every one of the 10,240 tokens in a step is a real corpus token contributing to the loss.

That closes X44's last unexplored candidate. The position is now:

> **The ~30,600 tok/s ceiling stands, and there is no longer any identified mechanism that could supply the
> residual 1.63× to 50,000.**

This does not make 50,000 impossible. It means reaching it now requires a mechanism nobody in this program has
yet named — which is a materially stronger statement than X44 was able to make, and it strengthens the case
for the revised 30,000 target rather than weakening it.

## 4. Scope — where packing *would* matter

This finding is specific to **pretraining throughput**, which is what the 50,000 target measures.

It does **not** apply to chat/SFT data, where conversations are variable-length and padding is real. The
handoff's V12 recipe explicitly concerns "packed full-sequence causal loss for two epochs" against Alpha's
earlier "one un-packed assistant-only pass", so packing is a live and important question *there*. It is simply
not a lever on the engine's tokens/s.

Anyone revisiting this should keep the two phases separate: a packing win in the SFT phase changes tokens-to-
quality, not tokens-per-second.

## 5. Method note

The audit cost minutes and required no hardware. Had it been deferred until a rental, it would have consumed
part of a bounded run to discover that the proposed optimisation targets something that does not exist.

This is the operating method's §3.5 ladder working as intended: an E0 mechanism audit killed a candidate that
would otherwise have been carried into an E4 device experiment. It also illustrates §3.3 — an analogy is not a
mechanism. "Sequence packing helps transformers" is true in general and void here, and the difference is one
`grep` away.

## 6. Reproduce

```bash
cd /mnt/donto-data/workspace/alpha2
sed -n '183,235p' packages/train/src/data.ts          # both batch paths
rg -n "padToken|PAD|pad_token" packages/train/src      # returns nothing
```
