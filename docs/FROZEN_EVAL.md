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

The full D3 structural pass is non-empty + EOS-terminated + no user-role leak. `summary.json` hashes the
checkpoint and both frozen inputs so results cannot silently drift between checkpoints.
