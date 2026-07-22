# SFT corpus v2

Alpha's canonical chat corpus is `/mnt/donto-data/alpha-corpora/sft-text-v2/sft-v2.txt`.
It contains one rendered conversation per line with atomic
`<|user|>`, `<|assistant|>`, and `<|end_of_text|>` markers. System prompts are folded into the first user
turn. Exact duplicate rendered conversations are removed by SHA-256.

The mix is dynamic rather than keyed to source-name conditionals: SmolTalk is the backbone, the selected
SmolTalk2 no-think splits add everyday and system-following conversations, OASST2 contributes the
best-ranked English path through each training tree, and a deterministic hash sample from SODA is capped
below 5% of final rows.

## Build

```bash
PY=/mnt/donto-data/alpha-corpora/.venv/bin/python
TOK=/mnt/donto-data/alpha-corpora/tokenizers/hf-bpe-byte-12k-20260722/tokenizer.json

nice -n10 ionice -c2 -n7 "$PY" scripts/build_sft_corpus.py \
  --smoltalk /mnt/donto-data/alpha-corpora/sft/smol-smoltalk/data \
  --smoltalk2 /mnt/donto-data/alpha-corpora/sft/smoltalk2 \
  --oasst2 /mnt/donto-data/alpha-corpora/sft/oasst2 \
  --soda /mnt/donto-data/alpha-corpora/sft/soda/train.parquet \
  --hf-tokenizer-json "$TOK" \
  --max-tokens 1024 \
  --out /mnt/donto-data/alpha-corpora/sft-text-v2/sft-v2.txt
```

The builder uses the actual Hugging Face export of Alpha's tokenizer. If a conversation exceeds the
1,024-token training context, it removes complete trailing user/assistant pairs until the row fits. It
never cuts an assistant response or its EOS in the middle. A single overlong pair is skipped honestly.
The output and manifest are atomically replaced; the manifest hashes every source, tokenizer, and output.

## Gates

```bash
nice -n10 ionice -c2 -n7 npx tsx scripts/validate-chat-data.ts \
  /mnt/donto-data/alpha-corpora/sft-text-v2/sft-v2.txt

nice -n10 ionice -c2 -n7 "$PY" scripts/audit_sft_lengths.py \
  --data /mnt/donto-data/alpha-corpora/sft-text-v2/sft-v2.txt \
  --manifest /mnt/donto-data/alpha-corpora/sft-text-v2/sft-v2.txt.manifest.json \
  --tokenizer "$TOK" \
  --out /mnt/donto-data/alpha-corpora/sft-text-v2/length-audit.json

nice -n10 ionice -c2 -n7 npx tsx scripts/verify_sft_masks.ts \
  --data /mnt/donto-data/alpha-corpora/sft-text-v2/sft-v2.txt \
  --manifest /mnt/donto-data/alpha-corpora/sft-text-v2/sft-v2.txt.manifest.json \
  --tokenizer /mnt/donto-data/alpha-corpora/tokenizers/bpe-byte-12k-20260722.json \
  --out /mnt/donto-data/alpha-corpora/sft-text-v2/mask-audit.json \
  --every 500 --block 1024
```

Final 2026-07-22 result: 511,428 structurally clean conversations; SHA-256
`ffad0a376c7eac2e0ec91f0901ec1ff87cba67cc298222828ce3df1a3e60b3fb`; SODA 4.828%; exact token
lengths min/p50/p95/p99/max = 16/657/978/1,014/1,024; zero rows over the context bound. The mask audit
checks every source boundary and each 500th row against the real training implementation: role markers are
atomic, user/scaffolding targets are masked, assistant targets and final EOS are supervised, and no sampled
row exceeds the block.
