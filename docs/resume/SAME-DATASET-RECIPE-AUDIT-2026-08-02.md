# Same-dataset recipe audit — project correction

Date: 2026-08-02

## Outcome

The proposed bespoke synthetic contrast-family generation is parked. The
active V12 experiment is instead a clean-base replication of the public
Smol-SmolTalk SFT pattern: packed full-sequence causal modeling over the complete
existing Smol-SmolTalk source, with early free-generation checks and two
identical-window learning-rate pilots before any full two-pass schedule.

The `3e-4` pilot completed and was rejected. Its best mechanical window remained
far behind the selected public Alpha baseline and produced circular, repetitive,
semantically wrong answers. The predeclared `1e-3` public-recipe arm is now in
progress. No checkpoint from either arm may justify a longer run until it passes
the frozen free-generation viability gate.

## Why the plan changed

The external and local evidence separates three comparisons:

1. **Same pretraining composition at one billion tokens.** The public codelion
   reference is also severely repetitive when treated as a chatbot. Alpha's
   clean base is not uniquely broken at this exposure.
2. **Same SFT dataset.** Hugging Face's public SmolLM2 SFT-only checkpoint trained
   on Smol-SmolTalk produces normal conversational answers under its exact chat
   template. Its base, however, was pretrained on two trillion tokens. Alpha's
   clean base saw approximately one billion. The public result therefore does
   not show that the SFT dataset can install conversational semantics into a
   similarly underexposed base; it shows that it can align a vastly stronger
   foundation.
3. **Same exact shuffled pretraining dataset in a full pipeline.** The public
   one-GPU nanochat run uses more shards, packed SmolTalk plus targeted tasks, and
   a later reinforcement stage.

The historical SmolLM2 recipe commit proves that its SFT stage used
`packing=True`, complete rendered-conversation next-token loss, and two epochs.
Alpha's flagship used `packed=false`, assistant-only loss, and one conversation
pass. V11 used full-sequence loss but only for 2.46 million positions over 10,862
synthetic conversations and initialized from V8; it did not test the public
recipe.

The central correction is therefore that "same dataset" is not the same
experiment when the parent models differ by roughly 2,000 times in token
exposure. Dataset format remains worth testing, but a negative short-pilot result
points toward base-model exposure or teacher distillation rather than another
blind post-training mixture.

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
3. run a short identical-window packed full-sequence LR pilot from the clean
   base — **`3e-4` complete and rejected; `1e-3` in progress**;
4. evaluate every pilot checkpoint through free generation and the unchanged
   held-out semantic panel — **complete for `3e-4`; pending for `1e-3`**;
5. run two full corpus passes only for a viable learning rate;
6. apply assistant-only response-policy recovery only if the full-sequence parent
   shows a semantic gain;
7. publish a new Alpha version only if it beats the current best model locally;
8. resume bespoke synthetic data generation only if the existing-data recipe is
   insufficient and the failure tells us what new distribution is required.

The synthetic generator/reviewer drafts are retained as parked future work. They
are not the active experiment. If both public-recipe learning-rate arms fail,
the next research contract should test distillation or greater foundation
exposure rather than assuming that more passes over the same SFT rows will
create the missing semantic competence.
