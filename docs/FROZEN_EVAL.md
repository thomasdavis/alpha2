# Frozen evaluation

Alpha freezes its evaluation data before flagship training. The suite contains no benchmark data:

- 100 prompts balanced across clean candidates from the `HuggingFaceTB/smol-smoltalk`
  test split and `OpenAssistant/oasst2` validation split;
- 200 closed-book questions derived from structured facts in FineWiki pages;
- 500 held-out validation documents from each source in premix shard 4, outside training shards 0–3.

`scripts/build_frozen_eval.py` creates oversized deterministic candidate pools. `scripts/audit_13gram.rs`
then scans the actual training text and reports any candidate with an exact 13-word overlap. Finalization
admits only candidates absent from those reports. Chat candidates are tokenized with the canonical 12,288
token HF tokenizer and capped at 896 prompt tokens, reserving 128 positions in the 1,024-token flagship
context for greedy generation.

## Build and audit

```bash
EVAL=/mnt/donto-data/alpha-corpora/frozen-eval-v1
PY=/mnt/donto-data/alpha-corpora/.venv/bin/python
TOK=/mnt/donto-data/alpha-corpora/tokenizers/hf-bpe-byte-12k-20260722/tokenizer.json

nice -n19 ionice -c3 "$PY" scripts/build_frozen_eval.py \
  --smoltalk-test /mnt/donto-data/alpha-corpora/sft/smol-smoltalk/data/test-00000-of-00001.parquet \
  --oasst2-validation /mnt/donto-data/alpha-corpora/sft/oasst2/data/validation-00000-of-00001-1deeef95c3248fe0.parquet \
  --finewiki /mnt/donto-data/alpha-corpora/eval-sources/finewiki-10M/data/train-00000-of-00001.parquet \
  --premix-heldout /mnt/donto-data/alpha-corpora/eval-sources/premix-heldout/data/train-00004-of-00100.parquet \
  --hf-tokenizer-json "$TOK" \
  --out "$EVAL"

rustc +1.88 -O scripts/audit_13gram.rs -o /tmp/alpha-audit-13gram
nice -n19 ionice -c3 /tmp/alpha-audit-13gram \
  "$EVAL/audit/sft-eval-docs.txt" "$EVAL/audit/sft-overlap.tsv" \
  /mnt/donto-data/alpha-corpora/sft-text-v2/sft-v2.txt
nice -n19 ionice -c3 /tmp/alpha-audit-13gram \
  "$EVAL/audit/pretrain-eval-docs.txt" "$EVAL/audit/pretrain-overlap.tsv" \
  /mnt/donto-data/alpha-corpora/pretrain-text/pretrain-{000,001,002,003,004,005}.txt

nice -n19 ionice -c3 "$PY" scripts/build_frozen_eval.py \
  --smoltalk-test /mnt/donto-data/alpha-corpora/sft/smol-smoltalk/data/test-00000-of-00001.parquet \
  --oasst2-validation /mnt/donto-data/alpha-corpora/sft/oasst2/data/validation-00000-of-00001-1deeef95c3248fe0.parquet \
  --finewiki /mnt/donto-data/alpha-corpora/eval-sources/finewiki-10M/data/train-00000-of-00001.parquet \
  --premix-heldout /mnt/donto-data/alpha-corpora/eval-sources/premix-heldout/data/train-00004-of-00100.parquet \
  --hf-tokenizer-json "$TOK" \
  --out "$EVAL" \
  --pretrain-overlap-report "$EVAL/audit/pretrain-overlap.tsv" \
  --sft-overlap-report "$EVAL/audit/sft-overlap.tsv"
```

`MANIFEST.json` records every source and output SHA-256. Do not edit the final JSONL/text files by hand;
change the builder or seed and repeat both audits.

## Score a checkpoint

```bash
node apps/cli/dist/main.js eval-frozen \
  --checkpoint=/path/to/checkpoint.json \
  --chat=/mnt/donto-data/alpha-corpora/frozen-eval-v1/final/chat-prompts.jsonl \
  --qa=/mnt/donto-data/alpha-corpora/frozen-eval-v1/final/closed-book-qa.jsonl \
  --out=/mnt/donto-data/alpha-runs/FLAGSHIP/frozen-eval
```

The evaluator loads the checkpoint once and uses Alpha's zero-allocation inference engine with greedy
decoding. It records per-case output plus:

- assistant response non-emptiness, EOS termination, and generated `<|user|>` leakage;
- token 4-gram repeat rate and the number of samples at or above the 0.20 loop threshold;
- normalized exact match, answer containment, and token F1 for the 200 closed-book questions.

The full D3 structural pass is non-empty + EOS-terminated + no user-role leak. The v2 `summary.json`
hashes the checkpoint, both frozen inputs, and both detailed output JSONL files; it also records the
atomic EOS/user control IDs so every structural flag can be recomputed from generated token IDs.

After evaluating the final base and chat checkpoints against the exact same frozen files, recompute the
machine-verifiable gate and the base-vs-chat deltas:

```bash
nice -n10 ionice -c2 -n7 npx tsx scripts/analyze_frozen_eval_pair.ts \
  --base /mnt/donto-data/alpha-runs/FLAGSHIP/frozen-eval-base \
  --chat /mnt/donto-data/alpha-runs/FLAGSHIP/frozen-eval-chat \
  --manifest /mnt/donto-data/alpha-corpora/frozen-eval-v1/MANIFEST.json \
  --out /mnt/donto-data/alpha-runs/FLAGSHIP/frozen-eval-pair.json
```

The analyzer requires all 100 chat and 200 QA cases, binds both runs to the final frozen-manifest
chat/QA hashes, exact base step 61,036 and chat step 30,322, identical architecture/input hashes/case
order, untampered detailed outputs, at least 95 structural
passes, zero samples at or above 0.20 4-gram repetition, and finite recomputed QA scores. Its PASS is
explicitly scoped to this machine-verifiable portion of D3. Conversational coherence remains a separate
review of the generated text; token statistics are never mislabeled as a semantic-quality judgment.

## Prepare the blinded semantic review

Only after terminal SFT and frozen generation are complete, create the review packet from the exact
manifest-bound inputs and outputs:

```bash
nice -n19 ionice -c3 npx tsx scripts/prepare_frozen_chat_semantic_review.ts \
  --prompts /mnt/donto-data/alpha-corpora/frozen-eval-v1/final/chat-prompts.jsonl \
  --results /mnt/donto-data/alpha-runs/FLAGSHIP/frozen-eval-chat/chat-results.jsonl \
  --summary /mnt/donto-data/alpha-runs/FLAGSHIP/frozen-eval-chat/summary.json \
  --manifest /mnt/donto-data/alpha-corpora/frozen-eval-v1/MANIFEST.json \
  --out /mnt/donto-data/alpha-runs/FLAGSHIP/frozen-chat-semantic-review.json
```

The preparer rejects the wrong checkpoint step, manifest/input/output hash drift, missing or duplicate
cases, changed case order, and malformed prompt/result rows. The packet contains all 100 prompts and
model responses but deliberately excludes the held-out reference answers. Review each case as `PASS`
(intelligible and relevant; simplistic or factually weak is allowed), `BORDERLINE` (understandable but
substantially irrelevant, contradictory, or fragmented), or `FAIL` (gibberish, word salad, role
confusion, empty output, or degenerate repetition). Preserve the per-case verdicts and rationales plus
the reviewer, UTC timestamp, and overall rationale; then change the packet status to `COMPLETE` and run:

```bash
nice -n19 ionice -c3 npx tsx scripts/finalize_frozen_chat_semantic_review.ts \
  --review /mnt/donto-data/alpha-runs/FLAGSHIP/frozen-chat-semantic-review.json \
  --out /mnt/donto-data/alpha-runs/FLAGSHIP/frozen-chat-semantic-review-report.json
```

The semantic gate is predeclared as at least 80 `PASS` cases and zero `FAIL` cases; `BORDERLINE` is
reserved for comprehensible but substantially flawed answers. The finalizer re-reads and hashes every
sealed input, proves the packet still contains the exact prompts/outputs/machine flags in exact order,
requires a verdict and rationale for all 100 cases, and preserves a machine-readable PASS or FAIL report.
Machine structure and closed-book factual accuracy remain separately reported gates.

## Terminal result — 2026-07-30

The final step-30,322 SFT checkpoint was evaluated once against the unchanged frozen inputs. This is
the terminal result; the operator then closed training and prohibited further runs.

| Metric | Base step 61,036 | Chat step 30,322 |
|---|---:|---:|
| Structural pass | 0 / 100 | 2 / 100 |
| EOS termination | 0 / 100 | 94 / 100 |
| Nonempty | 99 / 100 | 8 / 100 |
| User-role leaks | 0 / 100 | 0 / 100 |
| Degenerate loops | 99 / 100 | 6 / 100 |
| Mean / max 4-gram repeat rate | 0.81256 / 0.98400 | 0.04904 / 0.98400 |
| QA exact / contained | 0 / 200 · 1 / 200 | 0 / 200 · 0 / 200 |
| QA mean token F1 | 0.000238 | 0.000000 |
| Blinded semantic review | not run | 0 PASS / 0 BORDERLINE / 100 FAIL |

The machine gate and semantic gate both failed. Of the chat outputs, 92 were empty, six were degenerate
loops, and the remaining two were unusable fragments (`#### 512 ` and a bare code fence). Publication
as a failed-quality research artifact was a later explicit operator override; it does not change the
gate result.

Canonical evidence:

- frozen manifest SHA-256 `bf6e6ea4e7fb9ccffd2bab6283de42fe33e681679883da06d691f06cb867ac68`;
- chat results SHA-256 `bc369665e98ec49ae141e271508fa289d6fcbc7acc14fe8632360ba1f64fe161`;
- QA results SHA-256 `82d3254f02f7c900e395ae82387256097a9926c4e651544215a993af5a5d0cd7`;
- chat summary SHA-256 `c4751b33d19f09fbb84f223397af63897975980dfcf52172e9e18905ae955930`;
- pair-analysis SHA-256 `92da0b3bf5bd984c579ded700c1b2f9bfe928fe010a5352f65d1a15aea3d48c6`;
- semantic-report SHA-256 `35cc1a87fad2c4f258cfdbd5859d0a0106c0f2c1e8bdd0d6e5ada303a0ffc1e9`.
