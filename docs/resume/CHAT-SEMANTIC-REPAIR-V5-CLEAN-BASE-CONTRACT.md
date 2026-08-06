# Alpha semantic repair v5: clean-base path-dependence contract

Date: 2026-08-02

## Decision

Semantic repair v4 is rejected. It completed its finite GPU run and passed the
mechanical execution checks, but it did not improve the capability that matters.
Step 400 was its best mechanical checkpoint, yet it still generated circular or
false explanations such as describing DNA as a type of DNA made of cells. The
sealed final suite remains untouched, and v4 must not replace the public model.

The next experiment changes initialization, not the reviewed curriculum. V5
starts from the clean pretrained parent rather than from the public chat
checkpoint whose post-training data was dominated by roleplay. This tests the
specific hypothesis that v4's semantic curriculum was being pulled through an
entrenched conversational trajectory instead of installing clean answer
behavior.

## Frozen identities

| Object | SHA-256 |
| --- | --- |
| clean pretrained parent | `08e14fa9604bf1b46ebcd5df37933c84d2496c1d05d9e4b32ebad98792cc6049` |
| tokenizer | `c310343a185aecb572b8b6568b55179df248f4adec009d14a9496da354090b24` |
| v4 corpus manifest | `f0595e4d9775cf87d8fa7b1e402bf0c311bcb6cffacc686276a543783377a10b` |
| v4 train bytes | `a060479969d8ed5b51896af9becbfb7f7bbfdc2d18f61d4071d9dbacc2707938` |
| v4 development bytes | `0d4fee07dde9ceb36f0cf23ab2d4747207682d4705115c7b8a60afd43788c9b0` |
| evaluation freeze | `3e5a35d01644961bf464c627b527cf99290b1ed6f56467ebaccfbe86a4c66908` |

The launcher verifies these bindings before it writes the run contract.

## Intervention

Changed:

- initialization is the clean pretrained parent;
- the finite observation window is 1,600 steps, with candidates every 200
  steps so early competence can be selected before late overfitting.

Held constant:

- architecture and tokenizer;
- chat rendering and causal SFT loss;
- exact training and development corpus bytes;
- answer-start and EOS weighting;
- deterministic shuffle, decoding, and visible evaluation suites.

Excluded:

- roleplay-heavy SODA replay;
- RCR unlikelihood loss;
- tuning on BLAH prompts or judge answers;
- validation-loss selection;
- access to the sealed final suite before a candidate passes visible review.

## Finite training contract

- backend: native Helios Vulkan on NVIDIA;
- steps: 1,600 maximum;
- batch: 16;
- context: 512;
- optimizer: reset AdamW;
- learning rate: `5e-5` to `5e-6` with 100 warmup steps;
- answer start: first four assistant tokens weighted 8x;
- assistant end token: weighted 2x;
- checkpoints: 200, 400, 600, 800, 1000, 1200, 1400, 1600;
- free-generation selection only.

The final training step is not privileged. A checkpoint can be retained only if
its actual answers improve meaning, conversational contingency, and clean
stopping over the public baseline and the rejected v4 evidence.

## Promotion gate

A v5 checkpoint is ineligible if any of these are true:

- it remains circular on explanations such as DNA, promises, or ambiguity;
- fewer loops are achieved by empty, generic, or evasive answers;
- it adopts a canned empathy script instead of responding to the user;
- mechanical scores improve while human semantic inspection does not;
- ordinary conversation regresses;
- export parity or checkpoint identity fails.

Only after a candidate clears this gate may the sealed final suite be executed.
Only after the sealed result is acceptable may Hugging Face or BLAH be updated.

## Resource boundary

The retained v4 and v5 project artifacts must remain below 15 GiB. Derived HF
exports used only for parity are reproducible and need not be mirrored locally.
If retained artifacts approach the boundary, execution pauses before producing
another checkpoint rather than silently consuming more storage.

## Reproduction

The fail-closed launcher is:

```text
scripts/run_chat_semantic_repair_v5_clean_base.sh
```

The matched evaluator remains:

```text
scripts/evaluate_chat_semantic_v4_checkpoint.ts
```

It now distinguishes v4 and v5 from their immutable run-contract schemas and
checks the appropriate initialization hash for each.
