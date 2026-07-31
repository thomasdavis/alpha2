# Alpha chat repair v2 — evidence and experiment contract

**Status:** active design gate; no paid pod is live  
**Product:** an effective, natural conversational model  
**Constraint:** one rented GPU; model scale is not a benchmark or identity

## Decision in one paragraph

The next run will not blindly extend the published checkpoint. It will restore the pretrained model's original
1,024-token context, continue from the best response-start checkpoint on a new direct-assistant-dominant corpus,
and evaluate every short checkpoint on a newly frozen, source- and length-stratified development suite. The new
corpus sharply reduces SODA roleplay, removes a dominant repeated greeting without matching literal phrases,
caps repeated answer signatures and exact assistant turns dynamically, rejects already-looping targets, and
keeps OASST2 and concise direct answers prominent. The first paid experiment is a bounded continuation pilot.
Only a measured reduction in loops with reliable EOS stopping authorizes a longer continuation; failure triggers
one clean-base control rather than repeated tuning on the same outputs.

## Product gate

Alpha is not done because it emits nonempty text. A successful candidate must:

1. answer nearly every generation-eligible prompt;
2. stop cleanly rather than hitting the context or token limit;
3. avoid repeated phrase loops;
4. respond to the latest user move rather than merely produce grammatical topic words;
5. remain natural on short greetings, follow-ups, direct questions, disagreements, and requests for explanation;
6. preserve these behaviors over several turns.

Closed-book factual recall is diagnostic only. It cannot compensate for a bad conversation, and poor factual
recall is not by itself a failure of this stage.

## What the first repair actually established

The first corrective run fixed two real problems:

- generation now ends the prompt exactly at `<|assistant|>`, matching the learned token boundary;
- assistant starts, ordinary answer tokens, and EOS are no longer treated as one undifferentiated token average.

The selected checkpoint produced a good exact sample on the ordinary greeting `Hey, how is your day going?`:

> It’s going well, thank you. How about you?

The untouched 100-prompt result nevertheless reported 55 structural passes, 70 nonempty responses, 56 EOS
terminations, and 31 repetition loops. That aggregate hid two different failure classes.

### Context-contract failure

The checkpoint was trained and exported with a 512-token block even though its clean pretrained parent used a
1,024-token block. The frozen prompt suite legitimately contained longer conversations.

The reproducible joined audit found:

| Measurement | Result |
|---|---:|
| Frozen prompts | 100 |
| Prompts too long to generate even one token | 29 |
| Empty outputs | 30 |
| Empty outputs attributable to over-context prompts | 29 |
| Generation-eligible prompts | 71 |
| Nonempty on generation-eligible prompts | 70 / 71 |
| EOS on generation-eligible prompts | 56 / 71 |
| Loops on generation-eligible prompts | 31 / 71 |

Therefore the headline 30 empty answers must not be interpreted as 30 answer-initiation failures. Answer
initiation is almost repaired inside the declared context. Repetition and stopping are not.

Canonical machine audit:

`/mnt/donto-data/donto-resources/research/alpha-chat-repair-20260731/frozen-step1200-failure-audit.json`

### Decoding is not the remedy

A deterministic 12-prompt exploratory comparison tested greedy decoding, a four-gram block, nucleus sampling,
and nucleus sampling with a repetition penalty. A four-gram block forced the measured loop count to zero by
definition, but several generations remained wandering or nonsensical. Nucleus sampling retained loops. This is
useful as an inference safety control, but it is not evidence that the model improved.

Canonical exploratory artifact:

`/mnt/donto-data/donto-resources/research/alpha-chat-repair-20260731/step1200-decoding-exploration.json`

## What was wrong with the first repair corpus

The first corpus contained 34,880 conversations, of which 30,000 were SODA. That supplied natural turn-taking,
but it made roleplayed social dialogue 86% of the learning population while direct assistant behavior was a
minority.

The target text itself rarely contained within-answer four-gram loops. The larger problem was distributional
concentration:

| Corpus property | Result |
|---|---:|
| Assistant turns | 114,801 |
| Exact duplicate assistant turns after the first occurrence | 12,118 (10.6%) |
| Most common first-four-token signature | 7,730 turns (6.7%) |
| Everyday turns sharing its most common four-token start | 26.1% |
| SODA turns sharing its most common four-token start | 7.7% |

The everyday source repeats a single greeting in almost every conversation. SODA repeats a small set of social
openers and acknowledgements thousands of times. This is useful evidence because the model's failures are also
high-probability phrase attractors: once a locally plausible phrase begins, it re-enters the same continuation.

Canonical corpus audit:

`/mnt/donto-data/donto-resources/research/alpha-chat-repair-20260731/corpus-repetition-audit.json`

## Research synthesis

The intervention follows five findings from adjacent work.

1. Maximum-likelihood generation can over-assign probability to frequent and repeated sequences.
   [Unlikelihood training](https://arxiv.org/abs/1908.04319) reduced repetitive generation while largely
   preserving perplexity, and dialogue-specific experiments reduced target and context repetition
   substantially ([Li et al., ACL 2020](https://aclanthology.org/2020.acl-main.428/)).
2. Repetition in the training distribution itself is strongly associated with degeneration; removing or
   penalizing repeated training signals can reduce the behavior
   ([Li et al., 2023](https://arxiv.org/abs/2310.10226)).
3. Targeted teacher demonstrations can improve grammaticality and cohesion in small multi-turn models, but
   conversational contingency remains difficult
   ([ContingentChat](https://aclanthology.org/2025.babylm-main.25/)).
4. Dialogue-only training can improve a narrow continuation measure without producing broad communicative
   competence; targeted post-training performs better than treating dialogue volume as the intervention
   ([Padovani et al., 2025](https://aclanthology.org/2025.babylm-main.29/)).
5. Synthetic quality is not determined only by teacher strength. Hugging Face reported that stronger teachers
   under the same generation prompts did not automatically improve the student; prompt and corpus design were
   the useful intervention ([SmolLM data study](https://huggingface.co/blog/smollm)).

This evidence supports de-templating and direct response training now. It also supports a later unlikelihood or
contrastive arm if a clean data-only experiment still loops. Implementing a new loss in the from-scratch
CPU/Vulkan stack before testing the data diagnosis would bundle two interventions and add kernel risk, so it is
not part of the first v2 pilot.

## Corpus v2 contract

The v2 builder uses only already staged, licensed sources. AlphaCorpus remains paused and contributes zero rows.

Planned source roles:

| Source | Role in the mixture |
|---|---|
| Smol Magpie ultra-short train split | concise direct assistant responses and bounded follow-ups |
| OASST2 train split | human-authored assistant behavior and longer explanatory answers |
| Everyday conversations | short multi-turn continuity after removing the dominant greeting pair |
| SODA | minority social-conversation seasoning, no longer the corpus identity |

All selection operations are content-independent or distribution-derived:

- strict alternating user/assistant structure;
- whole-conversation token bound of 1,024;
- per-answer token bound;
- rejection of target answers already above the loop threshold;
- deterministic source sampling by hashed identity;
- a per-source cap on exact assistant-turn reuse;
- a per-source cap on first-four-token answer signatures;
- removal of a leading user/assistant pair only when the same exact pair occupies at least half of that source;
- exact conversation deduplication across sources;
- deterministic conversation-level train/development split.

The greeting transform does not search for `hello`, `how can I help`, or any other hand-maintained phrase. It is
derived from frequency and recorded by hash, preventing a literal string rule from becoming the data model.

## Evaluation firewall

The old 100-prompt final suite has now served its scientific purpose and is visible to the active diagnosis. It
cannot remain the untouched selector for another run.

Repair v2 freezes two new disjoint suites from held-out source splits:

- a visible development suite used for checkpoint selection;
- a sealed final suite executed only after one checkpoint is selected.

Both suites:

- exclude every ID in the published frozen final;
- reserve at least 128 token positions for generation inside the 1,024-token block;
- reject exact full-conversation overlap with the new train and corpus-development files;
- stratify by source, prompt length, and conversation-turn count;
- preserve exact prompt bytes, references, hashes, and row counts.

Checkpoint selection uses generated behavior, not validation loss. The primary automatic measurements are:

- nonempty rate;
- EOS termination rate;
- role-marker leakage;
- per-response repeated four-gram rate;
- fraction crossing the loop threshold;
- results stratified by prompt length, source, and turn count.

Automatic gates are necessary but not sufficient. A small fixed panel of exact outputs is reviewed for directness,
contingency, coherence, and naturalness. A decoder trick that only suppresses an n-gram is not a model win.

## Paid experiment sequence

### Gate 0 — local inputs

Before renting anything:

1. finish corpus construction;
2. run structural validation and mask audits at block 1,024;
3. freeze development and final suites;
4. hash all inputs;
5. run repository tests affected by the builders and evaluator;
6. write the exact command and stop conditions below with no placeholders.

### Pilot A — continue the best response-start checkpoint

Purpose: test whether corrected context and de-templated direct-assistant data repair stopping and repetition
without paying to reacquire answer initiation.

Provisional finite contract:

| Setting | Value |
|---|---:|
| Initialization | published selected repair checkpoint |
| Context | 1,024 |
| Batch | largest proven batch that leaves allocator headroom; start at 16 |
| Steps | 800 maximum |
| Checkpoints | 200, 400, 600, 800 |
| LR | `1e-5` cosine to `2e-6` |
| Warmup | 50 |
| First assistant content tokens | first 4 at 8x |
| Assistant EOS | 4x |
| Conversation weighting | equal |
| Epoch order | deterministic shuffle |

The EOS multiplier is intentionally separated from response-start weighting. A higher EOS weight is allowed only
because the first four content tokens remain independently protected; empty-first-token behavior is checked at
every candidate.

Pilot A succeeds only if a checkpoint, on the new development suite:

- answers at least 99% of generation-eligible prompts;
- reduces loops materially from the context-eligible published baseline;
- raises EOS termination without increasing empty answers;
- improves or preserves the exact qualitative panel;
- shows no role leakage or numerical instability.

### Pilot B — one clean-base control, only if needed

If Pilot A cannot escape the SODA-shaped attractors, run one clean-base arm on the same corpus and evaluator. This
tests whether the selected repair checkpoint has harmful path dependence. It is not an open sweep.

### Longer continuation

Only the better pilot initialization may continue. Stop when development generation stops improving for two
successive checkpoints or when the predeclared maximum is reached. The final suite is then run exactly once.

## GPU decision

No pod is currently live. The known RTX 4090 path completed the prior finite run and is the default because it is
operationally proven. A larger-memory GPU is justified only if the block-1,024 batch probe demonstrates that the
extra memory produces a meaningfully larger effective batch or higher measured tokens per dollar. Hardware name
or peak FLOPS is not sufficient; Alpha's custom Vulkan kernels must demonstrate real throughput on the rented
device before the full pilot proceeds.

Every launched process must be verified by advancing steps, bounded RSS/VRAM, output checkpoints, and measured
tokens per second. A created pod or a living PID is not proof of useful work.

## Discord publication rule

Discord receives no routine checkpoint spam. Post only when a new checkpoint improves against its declared
baseline. Each message must contain:

- the exact input;
- the exact output;
- the checkpoint identity;
- the measured improvement;
- why the change plausibly caused it;
- any remaining failure visible in the same evaluation.

## Stop conditions

Immediately stop the paid run for:

- NaN or non-finite loss/gradients;
- a failed NVIDIA/Vulkan gate;
- no advancing steps or output evidence;
- unbounded memory growth or swap/host degradation;
- worsening empty-response behavior at two consecutive checkpoints;
- no loop improvement at two consecutive checkpoints after the initial adaptation period;
- a project-owned artifact increase beyond the operator's 15 GiB pause threshold.

On every stop, preserve the checkpoint, optimizer state, metrics, exact config, corpus hashes, evaluation output,
pod identity, and termination evidence before deciding what comes next.
