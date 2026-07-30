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

## Flagship SFT run

Do not choose the post-training LR by intuition. From the final 61,036-step base checkpoint, run three
sequential, guarded pilots through the same verified inputs:

```bash
scripts/run_sft_lr_pilot.sh 0.0001 "$SFT" "$SFT.manifest.json" "$TOKENIZER" "$BASE" "$RUN/lr1e4"
scripts/run_sft_lr_pilot.sh 0.0003 "$SFT" "$SFT.manifest.json" "$TOKENIZER" "$BASE" "$RUN/lr3e4"
scripts/run_sft_lr_pilot.sh 0.0010 "$SFT" "$SFT.manifest.json" "$TOKENIZER" "$BASE" "$RUN/lr1e3"
```

Each pilot is exactly 2,000 steps × 16 × 1,024 = 32,768,000 padded tokens, with eight aligned held-out
evaluations, complete allocator telemetry, a full terminal checkpoint, immutable corpus/audit/tokenizer/
base hashes, and safe resume. Never run them concurrently on one inference/GPU path.

```bash
npx tsx scripts/analyze_sft_lr_sweep.ts \
  --lr1e4 "$RUN/lr1e4" --lr3e4 "$RUN/lr3e4" --lr1e3 "$RUN/lr1e3" \
  --out "$RUN/sft-lr-selection.json"

ALPHA_SFT_SELECTION_REPORT="$RUN/sft-lr-selection.json" \
  scripts/run_flagship_sft.sh <selected-lr> "$SFT" "$SFT.manifest.json" \
  "$TOKENIZER" "$BASE" "$RUN/one-epoch"
```

The selector requires exact equal inputs/commit/architecture/split/cadence, 2,000 finite rows per run,
zero allocator overflow, and 650–750 MiB terminal checkpoints. It ranks the final three aligned
validation losses, then final validation loss and lower LR as deterministic tie-breaks. The one-epoch
launcher will not accept a merely allowed numeric LR: it verifies the report's selected LR, source
commit, and all six input hashes before launching 30,322 assistant-only steps.

## Completed flagship outcome — 2026-07-30

The strict sweep selected `3e-4` from `{1e-4, 3e-4, 1e-3}`. The contracted run then completed exactly
one epoch: 30,322 steps, batch 16, block 1,024, and 496,795,648 padded tokens over 485,150 training and
26,278 held-out conversations. All rows were finite and consecutive; every one of 57,688,576 parameters
was finite/nonzero; 305 allocator samples reported zero overflow. Median post-warmup throughput was
3,847.23 tok/s, final train/held-out loss was 1.7579851/1.6439665, and the final-three validation mean
was 1.7073496.

The execution contract passed but the downstream chat-quality gate failed; see `docs/FROZEN_EVAL.md`.
No further SFT run is authorized. The exact terminal checkpoint is SHA-256 `6c279d086d...`; a slightly
better surviving full checkpoint at step 29,000 is also retained. Both, plus the base checkpoint,
optimizer/RNG state, corpus manifest/audits, contracts, full metric stream, and failure reports are
available at `ajaxdavis/alpha-60m-training-checkpoints`, immutable revision
`7198d1a1f094ffe88d06399ea99fecbd78fa8b66`. Local restart instructions are in
`/mnt/donto-data/alpha-runs/alpha-60m-continuation-c333bf2-20260730/RESUME.md`.
