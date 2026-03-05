# Super Chat Inference Quality RCA (Comprehensive)

Date: 2026-03-05
Repo: `alpha`
Authoring context: L4 training/reporting pipeline with Helios backend and remote metrics at `https://alpha.omegaai.dev`

---

## 1. Executive Summary

This document answers one question:

> Why is `super_chat` still producing weak/gibberish inference even when loss improved and runtime stability got better?

Short answer:

1. The system had multiple overlapping failure modes (resume mismatch, allocator cap crashes, spike storms, stale run state), and many were fixed.
2. The **best stable completed run** now exists (`super_chat_20260305150148_iz8f`, step 12000), but it **plateaus around val loss ~4.70-4.73** and still produces weak samples.
3. The current setup is likely bottlenecked by a combination of:
- small model capacity (`4L x 128d x 4h`, ~1.33M params),
- tokenizer/domain choices that are not chat-optimal for this corpus profile,
- training objective/quality metrics not tightly aligned with chat coherence,
- and historically noisy lifecycle/monitoring that made bad runs look "active".

This is no longer just "it crashed". It is now mostly a **quality objective + capacity + tokenizer/recipe** problem.

---

## 2. Current Ground Truth (from API and logs)

### 2.1 Best stable completed run in this pass chain

Run: `super_chat_20260305150148_iz8f`

- status: `completed`
- steps: `8401 -> 12000`
- model: `nLayer=4, nEmbd=128, nHead=4, block=256, vocab=2000`
- train config highlights: `lr=3e-6`, `lrMin=3e-7`, `warmup=10`, `packed=true`, `gradAccum=2`, `gradClip=0`, `spikeThreshold=0`

Metrics summary (API-derived):

- `loss_first=4.7515`
- `loss_last=4.7570`
- `loss_min=4.6369`
- `val_first=4.7002 @ step 8600`
- `val_best=4.7002 @ step 8600`
- `val_last=4.7255 @ step 12000`
- loss start/end avg (200-step windows): `4.7182 -> 4.7114` (small change)
- val drift from best to end: `+0.0253`

Operational events for this run:

- `training_started=1`
- `eval_summary=18`
- `checkpoint_saved=18`
- `samples_generated=18`
- `train_status_warning=8`
- `training_complete=1`

Interpretation:

- Runtime reliability is much better than earlier crash-heavy phases.
- Optimization quality is near-flat in the tail and not translating to coherent chat outputs.

### 2.2 Example unstable run signature

Run: `super_chat_20260305075936_434f`

- status in API: `active` (but effectively stale/dead)
- steps observed: `1 -> 843`
- config highlights: `lr=2e-4`, `warmup=1200`, `gradClip=1`, `spikeThreshold=0`, `packed=true`

Key metrics:

- `loss_min=4.7091 @ step 800`
- first catastrophic spike: `step 832`, `grad_norm=108734`, `loss=6.2298`, `ms/iter=378.6`
- max spike: `grad_norm=234137 @ step 841`, `loss=7.6679`, `ms/iter=462.5`
- spikes: `grad_norm > 100` occurred `10` times; `> 1000` occurred `4` times

Top spike rows:

| step | loss | grad_norm | ms/iter | tok/s |
|---:|---:|---:|---:|---:|
| 841 | 7.6679 | 234137.5 | 462.5 | 13283 |
| 843 | 6.2145 | 116340.4 | 365.2 | 16823 |
| 832 | 6.2298 | 108734.2 | 378.6 | 16229 |
| 840 | 6.2241 | 75875.3 | 279.0 | 22020 |

Interpretation:

- This is a classic "descend then numeric event" trace, not a smooth optimization trajectory.

### 2.3 Remote status integrity remains noisy

From `/api/runs?limit=400`, `super_chat_*` statuses currently include many stale actives:

- `active: 34`
- `completed: 8`
- `stale: 1`

Interpretation:

- Lifecycle state has improved but still needs stricter stale/run-death reconciliation so dashboard truth is unambiguous.

---

## 3. Why "loss looks lower" but inference is still bad

A frequent confusion in these runs:

- "Loss is around 4.7, why are samples still bad?"

Cross-entropy to perplexity:

- `loss=4.70 -> perplexity ~= 109.9`
- `loss=4.75697 -> perplexity ~= 116.4`
- target `loss=3.5 -> perplexity ~= 33.1`
- target `loss=3.0 -> perplexity ~= 20.1`

So 4.7 is not "good chat quality" yet. It is still a high-entropy regime where outputs can look noisy/repetitive/subword-like, especially with stochastic decoding (`temperature=0.8`, `topk=40`).

---

## 4. High-Confidence Root Causes (ranked)

## RC1) Model/recipe currently plateaus before chat-coherent regime

Evidence:

- Best stable completed run ends around `val ~4.7255` with weak samples.
- Tail movement is tiny over thousands of steps.

Confidence: High.

## RC2) Tokenizer/domain configuration is likely suboptimal for this corpus

Evidence:

- `super_chat` defaults still use tokenizer `bpe` (2k target) and a very small base model.
- Corpus has very large character diversity (`unique_chars=1876`), which heavily consumes vocab budget in char-first BPE.
- For 2k BPE, merges are only `124` in this dataset profile.

Observed from local tokenizer build on `data/super_chat.txt`:

- vocab size: `2000`
- merge count: `124`
- role token exact entries:
  - `<|user|>`: false (encodes to 3 tokens)
  - `<|assistant|>`: false (encodes to 3 tokens)
  - `<|end_of_text|>`: true (encodes to 1 token)

Also measured token compression ratio on first 1M chars:

- 2k BPE ratio: `~0.4786 tokens/char`
- 4k BPE ratio: `~0.2437 tokens/char`

Interpretation:

- 2k setup is likely too constrained for this corpus character distribution and chat token patterns.
- 4k gives much better compression/context efficiency, but prior 4k runs used unstable hyperparams and did not isolate tokenizer benefit cleanly.

Confidence: High.

## RC3) Earlier optimization instability caused bad branches and misleading progress

Evidence:

- Multiple runs with catastrophic grad spikes.
- allocator pressure/cap issues historically interrupted long trajectories.
- many runs marked active but stale/dead in practice.

Confidence: High.

## RC4) Quality telemetry was weaker than needed (improving, still incomplete)

Evidence:

- samples API rows currently have `step=null`, making timeline attribution weaker.
- event endpoint exists and works for newer runs, but old runs may have empty events.

Confidence: Medium-high.

## RC5) Inference checks are stochastic and can hide/overstate quality

Evidence:

- training sampling uses fixed `temperature=0.8`, `topk=40` at eval intervals.
- weak models under stochastic decoding can look much worse than greedy.

Confidence: Medium.

---

## 5. Dataset Characterization (`data/super_chat.txt`)

Measured directly from local file:

- file chars: `93,925,740`
- conversations (split by `<|end_of_text|>`): `226,110`
- mean chars/conversation: `399.4`
- p50: `240`
- p90: `550`
- p99: `3323`
- min/max chars: `42 / 5607`
- unique chars: `1876`
- non-ascii chars: `240,159` (`0.256%`)
- unique non-ascii code points: `1778`
- conversations containing non-ascii: `62,813` (`27.78%`)

Conversation marker checks:

- starts with `<|user|>`: `100%`
- has `<|assistant|>`: `100%`
- assistant before user ordering errors: `0`

Marker counts in full corpus:

- `<|user|>` count: `609,740`
- `<|assistant|>` count: `609,740`
- `<|end_of_text|>` count: `226,110`

Interpretation:

- Structural markers are present and balanced.
- Character diversity is unusually high for a small-vocab char-first BPE setup.

---

## 6. Code-Level Evidence (snippets)

### 6.1 Tokenizer training starts from full unique-char base vocab

`packages/tokenizers/src/bpe.ts`

```ts
// Base vocabulary: sorted unique characters from full input.
const baseChars = [...new Set(input)].sort();
this._vocab = [...baseChars];
...
const numMerges = this._targetVocabSize - baseChars.length;
for (let m = 0; m < numMerges; m++) {
  ...
}
```

Implication:

- With `targetVocab=2000` and `baseChars=1876`, only `124` merge slots remain.

### 6.2 Super-chat domain defaults are still tiny model + 2k BPE

`packages/core/src/domains.ts`

```ts
"super_chat": {
  tokenizer: "bpe",
  modelDefaults: {
    blockSize: 256,
    nLayer: 4,
    nEmbd: 128,
    nHead: 4,
    dropout: 0.0,
  },
  trainDefaults: {
    tokenizer: "bpe",
    lr: 1e-4,
    warmupIters: 1600,
    gradClip: 0.5,
    spikeThreshold: 100,
    packed: true,
    sampleInterval: 200,
    evalInterval: 200,
  }
}
```

Implication:

- This default is likely optimized for stability/cost, not chat coherence target quality.

### 6.3 Training samples are generated with stochastic decode by default

`packages/train/src/trainer.ts`

```ts
const sampleCfg: SampleConfig = { steps: 50, temperature: 0.8, topk: 40 };
```

Implication:

- Good for qualitative checks, but not a deterministic quality gate.

### 6.4 Samples persistence currently does not store per-sample step

`packages/db/src/schema.ts` + `packages/db/src/samples.ts`

```sql
CREATE TABLE IF NOT EXISTS samples (
  run_id TEXT NOT NULL,
  idx INTEGER NOT NULL,
  prompt TEXT NOT NULL,
  output TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (run_id, idx)
)
```

```ts
INSERT INTO samples (run_id, idx, prompt, output) VALUES ...
```

Implication:

- API sample rows can show `step=null`, making per-step sample timeline and regressions harder to inspect.

### 6.5 Event logs endpoint exists and supports tailing filters

`apps/web/src/app/api/runs/[id]/events/route.ts`

```ts
const last = parseIntParam(query.get("last"));
const fromId = parseIntParam(query.get("fromId"));
const fromStep = parseIntParam(query.get("fromStep"));
const limit = parseIntParam(query.get("limit"));
...
const events = await getEvents(client, id, { last, fromId, fromStep, limit, level, kind });
```

Implication:

- This should be the primary debug timeline surface for instability analysis.

---

## 7. Observability Gaps That Still Block Fast Iteration

1. Sample rows lack a canonical step key in DB.
- Impact: hard to align sample degradation with exact optimizer events.

2. Stale "active" runs still clutter API truth.
- Impact: wasted debugging cycles on dead runs.

3. Quality telemetry relies heavily on freeform text sample inspection.
- Missing hard metrics tied to coherence/degeneracy.

4. No first-class run health score in API combining:
- plateau warnings,
- spike counts,
- val drift,
- throughput degradation,
- event freshness.

---

## 8. What Is Most Likely Blocking Good Inference Right Now

Priority order:

1. **Optimization plateau in current tiny model recipe**
- stable but near-flat loss dynamics around 4.7x

2. **Tokenizer-vocab pressure from high char diversity in 2k mode**
- too few learned merges, limited abstraction of chat patterns

3. **Insufficient quality-aligned evaluation**
- stochastic sample snapshots without deterministic gates can mislead

4. **Residual lifecycle noise (stale active runs) and sample-step missingness**
- slows diagnosis and run selection

---

## 9. Recommended Action Plan (for ChatGPT review and implementation)

## Phase A: Make quality diagnostics decision-grade (fast, low risk)

1. Add `step` column to `samples` table and ingest path.
- Include `(run_id, step, idx)` uniqueness.
- Backfill best-effort with created_at ordering where step missing.

2. Add deterministic sample set at eval time.
- Add a second sample pass with `temperature=0` (greedy).
- Keep current stochastic pass for style drift checks.

3. Add quality metrics endpoint fields per eval window:
- repeated 3/4-gram rate,
- unique token ratio,
- role-token correctness rate (`<|assistant|>` continuity),
- long-bucket val loss (already computable via bucket loaders).

4. Tighten stale-run policy.
- Mark run stale if no metric/event heartbeat beyond threshold.

## Phase B: Tokenizer and recipe experiments (highest ROI for quality)

1. Run clean A/B on tokenizer size with stable recipe:
- A: 2k BPE (current)
- B: 4k BPE
- Keep architecture and schedule fixed to isolate tokenizer effect.

2. Add optional reserved special tokens in tokenizer build path.
- Explicitly reserve `<|user|>`, `<|assistant|>`, `<|end_of_text|>` as single tokens.
- Do not rely on merges to discover them.

3. Re-test model capacity with stable hyperparams:
- current 4L/128 baseline,
- 6L/192 candidate,
- keep packed mode and new stability guards.

4. Use conservative LR + explicit decay + spike handling:
- if grad spikes: skip step + event + optional temporary LR backoff.

## Phase C: Quality stop/go criteria

Use these as run gates (not just raw loss):

- val loss trend (windowed) must improve or remain neutral without widening gap
- greedy sample coherence must improve across fixed prompts
- repetition metric must trend down
- event log must show no catastrophic spikes for sustained windows

---

## 10. Concrete API Queries Used in This RCA

Latest super_chat runs snapshot:

```bash
curl -sS 'https://alpha.omegaai.dev/api/runs?limit=400' \
  | jq -r '[.[] | select(.run_id|startswith("super_chat_"))] | length'
```

Metrics for specific run:

```bash
curl -sS 'https://alpha.omegaai.dev/api/runs/super_chat_20260305150148_iz8f/metrics?limit=4000' | jq .
```

Event tail for specific run:

```bash
curl -sS 'https://alpha.omegaai.dev/api/runs/super_chat_20260305150148_iz8f/events?limit=100' | jq .
```

Samples for specific run:

```bash
curl -sS 'https://alpha.omegaai.dev/api/runs/super_chat_20260305150148_iz8f/samples?limit=20' | jq .
```

---

## 11. Additional Notes for External Review (ChatGPT handoff)

Questions to ask for second-opinion depth:

1. Given this corpus distribution and char diversity, is 2k BPE inherently too constrained?
2. Should special role tokens be hard-reserved before BPE merges?
3. For 1.33M params, what realistic val-loss floor should we expect on this dataset?
4. What is the best low-risk path to reach `val ~3.5`:
- tokenizer change,
- capacity increase,
- schedule/optimizer change,
- or data normalization?
5. Which single additional metric best predicts sample coherence early?

---

## 12. Bottom Line

The project moved from "crashing and blind" to "stable and measurable".

The remaining problem is mainly **model+tokenizer+objective quality**, not just infra survival.

If the goal is genuinely good chat inference (and eventually loss 3.0-3.5), the next iteration must be quality-driven with clean A/Bs, deterministic eval signals, and tokenizer/model choices aligned to this dataset's character and conversation profile.

