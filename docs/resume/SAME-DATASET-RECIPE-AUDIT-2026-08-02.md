# Same-dataset recipe audit — project correction

Date: 2026-08-02

## Outcome

The proposed V12 synthetic contrast-family generation is parked. No V12 data was
generated and no V12 GPU training began.

The next justified experiment is a clean-base replication of the successful
public Smol-SmolTalk training pattern: packed full-sequence causal modeling over
the complete existing Smol-SmolTalk source for two passes, with early
free-generation checks and an identical-window learning-rate pilot.

## Why the plan changed

The external and local evidence separates three comparisons:

1. **Same pretraining composition at one billion tokens.** The public codelion
   reference is also severely repetitive when treated as a chatbot. Alpha's
   clean base is not uniquely broken at this exposure.
2. **Same SFT dataset.** Hugging Face's public SmolLM2 SFT-only checkpoint trained
   on Smol-SmolTalk produces normal conversational answers under its exact chat
   template.
3. **Same exact shuffled pretraining dataset in a full pipeline.** The public
   one-GPU nanochat run uses more shards, packed SmolTalk plus targeted tasks, and
   a later reinforcement stage.

The historical SmolLM2 recipe commit proves that its SFT stage used
`packing=True`, complete rendered-conversation next-token loss, and two epochs.
Alpha's flagship used `packed=false`, assistant-only loss, and one conversation
pass. V11 used full-sequence loss but only for 2.46 million positions over 10,862
synthetic conversations and initialized from V8; it did not test the public
recipe.

## Exact evidence

Canonical research artifact:

```text
/mnt/donto-data/donto-resources/research/alpha-same-dataset-recipe-audit-20260802/
```

Read its `README.md` for source revisions, direct model outputs, configuration
comparison, falsification conditions, runtime estimate, and artifact hashes.

## Revised order of work

1. build immutable Smol-SmolTalk-only train and test renderings from the already
   staged parquet files — **complete**;
2. verify hashes, role structure, special-token boundaries, train/test separation,
   and tokenizer parity — **complete** (exact packed stream count is recorded by
   the native trainer cache at pilot launch);
3. run a short identical-window packed full-sequence LR pilot from the clean base;
4. evaluate every pilot checkpoint through free generation and the unchanged
   held-out semantic panel;
5. run two full corpus passes only for a viable learning rate;
6. apply assistant-only response-policy recovery only if the full-sequence parent
   shows a semantic gain;
7. publish a new Alpha version only if it beats the current best model locally;
8. resume bespoke synthetic data generation only if the existing-data recipe is
   insufficient and the failure tells us what new distribution is required.

The untracked V12 generator/reviewer drafts are retained as parked future work.
They are not the active experiment.
