# Alpha chat recipe V12 — frozen experiment contract

Date frozen: 2026-08-02

## Decision

V12 is a controlled replication of the closest public chat-training recipe that
uses Alpha's already staged Smol-SmolTalk source. It is not a new synthetic-data
run and it does not use the parked V12 synthetic generator drafts.

The experiment starts from the clean pretraining checkpoint, presents complete
rendered conversations as one packed token stream, applies ordinary next-token
loss to every token, and tests two learning rates on identical data windows. A
full run is permitted only when a pilot checkpoint improves free conversation
without replacing Alpha's existing failure mode with immediate EOS, copying,
generic filler, or a different repetition loop.

## Research basis

The canonical audit is:

```text
/mnt/donto-data/donto-resources/research/alpha-same-dataset-recipe-audit-20260802/
```

It established the following relevant facts:

1. a public base trained on the same 50/30/20 pretraining composition at roughly
   the same token exposure is also repetitive when used directly as a chatbot;
2. Hugging Face's public SFT-only checkpoint trained on the exact
   `HuggingFaceTB/smol-smoltalk` source produces coherent conversational output;
3. the corresponding public SmolLM2 SFT recipe used packed complete-chat
   next-token training for two epochs;
4. Alpha's original flagship instead used padded assistant-only training for one
   conversation pass;
5. V11's full-sequence arm was only a very short continuation of a post-trained
   checkpoint and therefore did not test the public recipe;
6. the four upstream Smol-SmolTalk training shards are already heavily
   interleaved by source, so this run does not add a speculative source-order
   intervention.

This evidence does not prove the public recipe will repair Alpha. It does make
it the lowest-confound, highest-value experiment before authoring a new
synthetic curriculum.

## Frozen inputs

### Parent

The parent is the immutable clean pretraining step-61,036 checkpoint already
used by Alpha's earlier clean-base comparisons. The runner must record its
absolute path, SHA-256, checkpoint step, tokenizer fingerprint, architecture
configuration, and parameter-tensor fingerprint before GPU initialization.

No SFT, chat-repair, V8, V10, or V11 weights may initialize this run.

### Dataset

Training source:

```text
/mnt/donto-data/alpha-corpora/sft/smol-smoltalk/data/train-*.parquet
```

Validation source:

```text
/mnt/donto-data/alpha-corpora/sft/smol-smoltalk/data/test-00000-of-00001.parquet
```

Dataset identity is the SHA-256 of every parquet file plus the recorded upstream
dataset repository and immutable revision from the audit. The builder emits one
canonical text file per split and an immutable manifest.

Rendering and length checks use the standard Hugging Face `tokenizer.json`
export of Alpha's tokenizer, SHA-256
`37372c9b1bdbf7d9655444e90247bef957018d0d7ff0b668d1330e28d97c44cf`.
Training uses the corresponding native Alpha artifact, SHA-256
`c310343a185aecb572b8b6568b55179df248f4adec009d14a9496da354090b24`.
The exporter is already parity-tested; V12 additionally records corpus-scale
role-token checks before launch. The two serializations must never be passed to
the other's loader.

The upstream training order is preserved. It is already highly interleaved and
changing it would introduce a second experimental variable. Exact rendered
duplicates are removed. Any exact train/test rendering overlap is excluded from
validation and reported.

### Rendering

The tokenizer has only three conversation control tokens. Model-visible text is
therefore rendered as:

```text
<|user|> USER <|assistant|> ASSISTANT [more alternating turns] <|end_of_text|>
```

System messages are folded into the first user message using the existing
Alpha rendering convention. Empty, malformed, non-alternating, or marker-
injecting rows are rejected and counted. Long conversations retain the longest
valid prefix ending in an assistant turn and fitting the 1,024-token context.

No hidden metadata, source labels, loss-mask syntax, or synthetic system persona
is shown to the model.

## Frozen optimization comparison

All arms use:

- the same clean parent and tokenizer;
- the same rendered train and validation bytes;
- ordinary full-sequence next-token loss (`sft=false`);
- packed sequential loading (`packed=true`);
- context length 1,024 and batch size 16;
- AdamW, the existing stable full-precision Vulkan path, fixed seed 1337;
- no Symbiogenesis, auxiliary loss, RCR-UL, dropout, or response-start weighting;
- fresh optimizer and schedule state.

Pilot arms:

| Arm | Peak LR | Minimum LR | Pilot length | Checkpoints |
|---|---:|---:|---:|---|
| `lr3e-4` | 3e-4 | 3e-5 | 2,000 steps | 250, 500, 1,000, 2,000 |
| `lr1e-3` | 1e-3 | 1e-4 | 2,000 steps | 250, 500, 1,000, 2,000 |

Warmup is 10% of the pilot. The full-run warmup is 10% of one corpus pass. A
full run uses two complete packed corpus passes, rounded up to the next whole
step, with evaluation/checkpointing at useful fractions of each pass.

The two pilots must consume identical packed windows in identical order. The
only intended difference is learning rate.

## Evaluation

Loss is diagnostic, not the selector. Every checkpoint is decoded freely under
the same deterministic and sampled settings used for prior Alpha comparisons.

The evaluation includes:

- the frozen development and semantic-contingency prompts;
- ordinary short chat, including `What is DNA?`;
- multi-turn follow-up and reference resolution;
- answer-and-stop prompts;
- ambiguity, correction, intent, and conceptual distinction prompts;
- prompt-length bands;
- repetition, copy, EOS, truncation, and role-marker checks;
- a blinded comparison against the clean base and the best retained public
  Alpha checkpoint.

Checkpoint reports must separate:

- nonempty output from meaningful conversation;
- prompt copying from a relevant response;
- lexical variety from semantic contingency;
- chosen stopping from evaluator token-limit truncation;
- genuine improvement from a new failure signature.

## Selection and stopping

The pilot has three possible outcomes:

1. **Viable parent:** one arm shows a clear conversational gain across multiple
   prompt families without a severe regression. Continue that arm to two corpus
   passes.
2. **Ambiguous:** gains are narrow, seed-sensitive, or offset by a comparably
   serious new failure. Stop and inspect rather than spending the full run.
3. **Rejected:** both arms remain semantically inert or degenerate. Do not train
   longer merely because validation loss falls. Use the observed failure to
   decide whether the missing variable is foundation exposure, response-policy
   recovery, data composition, or bespoke synthetic data.

After a viable full-sequence parent exists, a short assistant-only response-
policy recovery stage may be separately contracted. It must not be silently
folded into this experiment.

## Publication contract

A checkpoint is uploaded as a new Hugging Face and BLAH model version only when
it is a genuine local winner. Existing BLAH model records are never repointed.
The exact model revision, chat template, inference settings, and evaluation
evidence travel with the new version.

Discord receives a sample only for a genuine improvement, accompanied by the
old output, new output, prompt, checkpoint identity, and the specific reason it
is better. Routine progress and failed runs stay in the scientific ledger.

## Artifact and recovery contract

The canonical experiment root is:

```text
/mnt/donto-data/donto-resources/research/alpha-chat-recipe-v12-20260802/
```

Large rendered corpora and token caches live under:

```text
/mnt/donto-data/alpha-corpora/chat-recipe-v12/
```

The experiment records source hashes, corpus hashes, manifests, runner
configuration, source commit, environment, metrics, checkpoints, evaluation
outputs, selection decisions, and null results. Retained experiment artifacts
are measured after each stage; work pauses before retained artifacts exceed the
user's 15 GiB review threshold.

The parked files named `chat-foundations-v12-*` describe a possible future
synthetic curriculum. They generated no data, are not inputs to this run, and
will be renamed when that line of work resumes.
