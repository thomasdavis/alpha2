# Alpha chat repair v2 — bounded negative result and recovery record

**Executed:** 2026-07-31

**Status:** complete; no v2 checkpoint selected

**Public chat checkpoint:** unchanged

**Sealed-final suite:** never executed or inspected

**Alpha paid compute:** terminated after verified recovery

**Discord:** no improvement announcement; one later user-requested message was explicitly labelled as a test

## Executive result

Chat repair v2 answered the narrow causal question it was designed to test, but it did not produce a better
conversational model.

Two bounded experiments were run against a development suite frozen before training:

1. **Pilot A** continued the published step-1,200 repair checkpoint for 800 steps on a cleaner, broader,
   1,024-token conversation corpus.
2. **Pilot B** repeated the intervention for 1,600 steps from the original clean pre-SFT checkpoint, after Pilot A
   rejected the simple-continuation hypothesis.

Both experiments made response initiation reliable. Every selector checkpoint produced a nonempty response on
all 96 development prompts. Neither experiment beat the published checkpoint on the exact 69 prompts that all
candidates could generate. Repetition increased, EOS control weakened, and the fixed qualitative panel contained
parroting, nonanswers, semantic errors, and repeated-phrase attractors. Lower teacher-forced validation loss did
not repair free conversation.

**Decision:** reject every v2 checkpoint. Keep the previously published step-1,200 checkpoint as the best honest
public artifact, despite its own quality gate remaining `FAIL`. Preserve v2 as a negative result and restartable
research branch. Do not spend more GPU by blindly continuing either v2 trajectory.

## What remained selected

| Artifact | Immutable identity |
|---|---|
| Native selected checkpoint | `399f776b49acc0c8834ff8a7f2390454e2c5f2d833a264e3f83ff546e973cfec` |
| Public standard model | `ajaxdavis/alpha-60m-chat` revision `ab1c5be13a12c0feb2d5e2c9af89bd5924a0e8b0` |
| Original corrective recovery archive | revision `ffc447e8a0f2240d42ceb0abfd18ab5b427d5e60` |
| Static Space | revision `d87e0950baf0a16ccd2859c2cee6314602ba2881` |

This is not a claim that the selected model is generally effective. Its untouched result remains 55/100
structural passes, 70/100 nonempty responses, 31 loop flags, and 0/200 closed-book QA exact. It remains selected
only because the newer candidates were worse on controlled comparable conversation behavior.

## Frozen experiment inputs

The exact execution source was clean commit:

    64a5e724ad90806e4c55e877180cb2c9bf2e1153

The v2 corpus root is:

    /mnt/donto-data/donto-resources/research/alpha-chat-repair-v2-r2-20260731/

It contains 24,701 unique conversations:

| Split / source | Conversations |
|---|---:|
| Train | 23,529 |
| Corpus development | 1,172 |
| Smol Magpie | 12,532 |
| SODA | 8,000 |
| OASST2 | 2,131 |
| SmolTalk2 everyday conversation | 2,038 |

The corpus has 72,416 assistant turns. Its audits found:

- zero target answers crossing the free-generation loop threshold;
- 1.25% exact target duplication after the first occurrence;
- maximum first-four-token answer-start concentration of 0.28%;
- structurally valid role boundaries and assistant-only loss masks under the native tokenizer;
- no use of the paused AlphaCorpus candidates.

### Immutable data and evaluation hashes

| Artifact | SHA-256 |
|---|---|
| Rendered train text | `5307b6ec210a172f853f7d5ba353727e1a6f065a337154954b049d989f403f63` |
| Rendered corpus-development text | `cf0b94f41d78144b4496a5569b42d6ceb08086709e13b745c652bbea0701f5b9` |
| Corpus manifest | `45637671411ca2a1fdd349a3660401acf828b41b8749198377af16ca8c60cd39` |
| Tokenizer | `c310343a185aecb572b8b6568b55179df248f4adec009d14a9496da354090b24` |
| Visible 96-prompt development suite | `156f70d6f374a006b668b7d6c2edd54f541097b0d015e313c11033d0cd098f33` |
| Fixed 12-prompt qualitative panel | `4e5b1e3025087d7a2d57282f8f8b548f4def9e07996ad2f5f7a3e93e8e9759ea` |
| Disjoint sealed-final 150-prompt suite | `8b71ab5f8843b14a8bbe56a473ea9cd0672b873024632c023abbe4935e48eb1d` |
| Evaluation-freeze manifest | `60f1b06acfd281caabfc2d3f423ef15762a7900a681a76326340f0729dd0d190` |

The development suite, panel membership, comparable-ID set, and sealed-final suite were frozen before either
pilot. Greedy decoding used a 128-token completion limit. Only development checkpoints declared in advance were
evaluated. Because no candidate passed development selection, the sealed-final suite remained untouched.

## Preflight and runtime integrity

The exact NVIDIA gate executed 46/46 expected assertions with zero failure and zero skip before training. The
run used Alpha's own Helios/Vulkan training path, and model-sized CPU fallback was forbidden. Local consolidated
tests before execution were 212 pass with 46 NVIDIA-gated skips; TypeScript typechecking was clean.

Every checkpoint, metric log, configuration, contract, and evaluation output was copied from the pod and
SHA-256 verified before compute termination. All training steps were finite. No allocator free-range overflow
was recorded.

## Published-baseline measurement on the v2 selector

The published checkpoint could attempt only 69 of the 96 prompts because its earlier serving/training contract
used a 512-token context. The remaining 27 prompts were over capacity and receive no negative or positive
selection credit. On the exact 69 generation-eligible prompts, the baseline measured:

| Structural | Nonempty | EOS | Role leaks | Loops | Mean 4-gram repeat |
|---:|---:|---:|---:|---:|---:|
| 55/69 | 68/69 | 56/69 | 0 | 24 | 0.15870 |

The full 96-row baseline file retains the over-capacity rows and reports 55 structural, 68 nonempty, 56 EOS,
24 loops, and zero role leaks. The exact eligible-ID comparison, rather than the misleading 96-row denominator,
was the selection basis.

Canonical evidence:

    /mnt/donto-data/alpha-runs/alpha-chat-repair-v2-20260731/pilot-a/evaluations/baseline-step1200-development/

## Pilot A — continuation from the published checkpoint

Pilot A continued the selected corrective checkpoint for 800 bounded steps. It used a learning rate of `1e-5`
cosine-decayed to `2e-6`, batch size 16, context 1,024, first-four assistant content weighting at 8x, and EOS
weighting at 4x. Checkpoints 200, 400, 600, and 800 were evaluated.

### Exact 69-prompt comparison

| Candidate | Structural | Nonempty | EOS | Role leaks | Loops | Mean 4-gram repeat |
|---|---:|---:|---:|---:|---:|---:|
| Published baseline | 55 | 68 | 56 | 0 | 24 | 0.15870 |
| Pilot A step 200 | 58 | 69 | 58 | 0 | 32 | 0.24688 |
| Pilot A step 400 | 60 | 69 | 60 | 0 | 32 | 0.23140 |
| Pilot A step 600 | 60 | 69 | 60 | 0 | 33 | 0.23613 |
| Pilot A step 800 | 56 | 69 | 56 | 0 | 30 | 0.24667 |

Pilot A solved the single empty response but increased loop flags by 6–9 and materially increased repeated
four-grams. It was rejected. This disproved the narrow hypothesis that simply continuing the response-start
checkpoint on the corrected corpus would yield stable conversation.

## Pilot B — clean-base causal control

Pilot B initialized from the original pre-SFT checkpoint:

    08e14fa9604bf1b46ebcd5df37933c84d2496c1d05d9e4b32ebad98792cc6049

Its executed configuration was:

| Setting | Value |
|---|---:|
| Steps | 1,600 |
| Context | 1,024 |
| Batch | 16 conversations |
| Optimizer | AdamW |
| Learning rate | `5e-5` cosine to `5e-6` |
| Warmup | 100 steps |
| First assistant content tokens | first 4 at 8x |
| Assistant EOS | 4x |
| Conversation weighting | equal |
| Epoch order | deterministic shuffle |
| Development checkpoints | 400, 800, 1,200, 1,600 only |

The run completed 1,600/1,600 finite steps in 7,130.7 seconds on one RTX 3090. Internal validation loss fell to
2.0204 at step 1,600. That improvement did not translate into conversational selection.

### Exact 69-prompt comparison

| Candidate | Structural | Nonempty | EOS | Role leaks | Loops | Mean 4-gram repeat |
|---|---:|---:|---:|---:|---:|---:|
| Published baseline | 55 | 68 | 56 | 0 | 24 | 0.15870 |
| Clean base step 400 | 51 | 69 | 51 | 0 | 35 | 0.30246 |
| Clean base step 800 | 53 | 69 | 53 | 0 | 32 | 0.27948 |
| Clean base step 1,200 | 51 | 69 | 51 | 0 | 31 | 0.27176 |
| Clean base step 1,600 | 52 | 69 | 52 | 0 | 29 | 0.24566 |

### Full 96-prompt diagnostic view

| Candidate | Structural | Nonempty | EOS | Role leaks | Loops | Mean 4-gram repeat |
|---|---:|---:|---:|---:|---:|---:|
| Step 400 | 74 | 96 | 74 | 0 | 48 | 0.28741 |
| Step 800 | 72 | 96 | 72 | 0 | 44 | 0.28837 |
| Step 1,200 | 69 | 96 | 69 | 0 | 45 | 0.29488 |
| Step 1,600 | 73 | 96 | 73 | 0 | 42 | 0.26226 |

Step 1,600's raw generation file is SHA-256
`6e6f4ea1dd450d568def290a6ca7c6fb6dee9adee2c47ea113641dfdb693e9a3`.

No clean-base checkpoint beat the published baseline's 24 loop flags or 55/69 structural passes on the exact
shared IDs. Pilot B was rejected.

## Qualitative evidence

The automatic rejection is supported by direct inspection. Representative step-1,600 outputs include:

### Wrong semantic relation

**Input:** `icecream or ice scream?`

**Output:** `Yes, you can use ice screams to escape.`

The reply is nonempty and stops, but it misunderstands the user rather than resolving the lexical contrast.

### Repetition attractor

**Input:** after an air-traffic-control exchange, `How is the wind for landing 27L?`

**Output:** begins `The wind for landing 27L is 1,000 feet...` and repeats the same false phrase until the
completion cap.

### Generic or incomplete completion

Other frozen-panel examples contain charging-cable parroting, an incomplete anime response, incorrect modular
arithmetic, an empty Python code fence, and surface templates such as `Here's a revised version:` without the
revision. These are failures of relevance and semantic continuation, not merely an aggressive repetition
threshold.

The exact 12-case panels for all declared checkpoints are in each evaluation directory. They show source
responses only as context; Alpha was judged on directness, contingency, coherence, naturalness, stopping, and
absence of loops, not lexical imitation.

## Diagnosis established by v2

1. **Response initiation is separable from conversational competence.** First-content and EOS weighting made all
   v2 checkpoints answer every development prompt, but it did not make the answers useful.
2. **The remaining failure is semantic and dynamical.** Alpha frequently parrots, emits a learned preamble without
   its payload, or enters a high-probability repeated phrase.
3. **The context limit was not the whole cause.** The exact shared-ID comparison remained worse after moving to
   a 1,024-token context.
4. **Ordinary social turn-taking is easier than instruction and semantic continuation.** On shared IDs,
   everyday-conversation loop flags stayed at 2. At clean-base step 800, OASST2 rose from 9 baseline loops to 13,
   and Smol Magpie rose from 13 to 17.
5. **Initialization alone is not the cause.** Continuation from the selected checkpoint and restart from the clean
   base both failed under the corrected mixture.
6. **Clean target text is not sufficient.** No target crossed the loop threshold, but autoregressive generation
   still developed loops.
7. **Teacher-forced validation loss is an invalid sole selector.** Loss improved while the conversational gate
   remained worse than baseline.
8. **No sealed-final claim is available.** Because development selection failed, inspecting the sealed suite would
   have spent its evidentiary value without a candidate.

This rules out more blind continuation on the same objective. A future run needs a declared intervention aimed at
semantic contingency and stable autoregressive dynamics, with the same strict comparable-ID and qualitative
selection discipline.

## Recovery artifacts

Canonical mounted run:

    /mnt/donto-data/alpha-runs/alpha-chat-repair-v2-20260731/

The clean-base run retains every 200-step checkpoint with optimizer and RNG state:

    /mnt/donto-data/alpha-runs/alpha-chat-repair-v2-20260731/pilot-b/run/

The run manifest SHA-256 is:

    48023b6874d3571f22802d122b8d895df034b3b0e71c466c27371800d719ce0d

### All clean-base checkpoints

| Step | SHA-256 |
|---:|---|
| 200 | `689c377420dc928b34200aef75fe06738160a6c8c45adafe8ea5fb9c2488bad4` |
| 400 | `276d8fe12f30ffa9acc80336712baa5aac4d459b89e585ad536352fe61574332` |
| 600 | `d0b440470c7863afa75e470fea19548f8bc9ddc15c951e380c2f54ee416151dd` |
| 800 | `fc83b3cd8493e1b554a436a61025a80a13359317e0ad0327ec0320ebafafa0b4` |
| 1,000 | `f50b47c61788a69305ad94ea6bd428762e3ce8c2fe8e75e3139d231fe62b8f5a` |
| 1,200 | `ffac13d2fde9de551224c9764e26a0f36acf2b5acc64a6b30fad0ef092afdce1` |
| 1,400 | `fd1968b554b0b460ae9e0d49fc8e1a1da0b701d0a027976ba0e5c826dd1ca930` |
| 1,600 | `1aa3e071d1999254903b95b1c46cd3ab8907f826ebf3cf3c2078c7c52c318be8` |

### Public v2 recovery archive

The negative-result archive is public under the existing training-checkpoint repository:

    https://huggingface.co/ajaxdavis/alpha-60m-training-checkpoints/tree/c1117378c0bc8b81b408be09c000f80ea9f027d7/chat-repair-v2-20260731

Immutable revision:

    c1117378c0bc8b81b408be09c000f80ea9f027d7

The archive contains 53 files: two optimizer-bearing checkpoints, the exact run contract/configuration/metrics,
all four clean-base evaluation directories, logs, pod-removal proof, frozen input manifests, audits, tokenizer,
and a complete checksum manifest.

| Public recovery artifact | SHA-256 |
|---|---|
| Step 800 native checkpoint | `fc83b3cd8493e1b554a436a61025a80a13359317e0ad0327ec0320ebafafa0b4` |
| Step 1,600 native checkpoint | `1aa3e071d1999254903b95b1c46cd3ab8907f826ebf3cf3c2078c7c52c318be8` |
| Public archive README | `3235310d50eb4da238d8658106eb484abb3d2f96068f259730f6b7f6206ec953` |
| Public `CHECKSUMS.sha256` | `b733f5704e722faadd2e6e46cd9505be44e7952da75d3d001aa65ac92cc6cf5f` |

Anonymous verification at the immutable revision found exactly 53 nested files, matched both checkpoint LFS
SHA-256 values and sizes, and downloaded the README and checksum manifest byte-identically to the local archive.

The public archive intentionally retains only step 800—the strongest clean-base structural checkpoint—and step
1,600—the final continuation state. Every intermediate checkpoint remains on the mounted canonical run.

## Live public-serving verification after v2 rejection

The unchanged selected artifact was reverified after v2 publication:

- `alpha2-hf-backend.service` was enabled, active/running, zero-restart, and exited no failed main process;
- `/health` reported model `ajaxdavis/alpha-60m-chat`, 57,688,576 parameters, checkpoint step 1,200, and
  `quality_gate: FAIL`;
- `/evidence` reported selected checkpoint SHA
  `399f776b49acc0c8834ff8a7f2390454e2c5f2d833a264e3f83ff546e973cfec`, 55/100 structural, 30 empty, 31
  loops, and QA 0/200;
- anonymous Hub metadata still reported model revision
  `ab1c5be13a12c0feb2d5e2c9af89bd5924a0e8b0` and running Space revision
  `d87e0950baf0a16ccd2859c2cee6314602ba2881`;
- the static page returned its exact `x-repo-commit` for that Space revision;
- a real Chromium session loaded the public Space, submitted `Hey, how is your day going?`, and displayed
  `It's going well, thank you. How about you?`;
- the rendered page retained the visible quality-fail status, one main landmark, and no horizontal overflow.

Browser screenshot:

    /mnt/donto-data/alpha-runs/alpha-chat-repair-v2-20260731/public-verification/space-live-selected-step1200-20260731.png

Screenshot SHA-256:

    54a56df6d34bcfac0e68727953feb4bc2846c77226730f876ae9a2f19d685d14

This proves that rejecting v2 did not accidentally change or disguise the serving path. It does not upgrade the
model's failed quality verdict.

## Repository closeout validation

- local Markdown link validation passed across all 15 changed dossier files;
- `git diff --check` passed;
- secret-pattern scanning found no webhook URL, Hugging Face token, RunPod key, or private key in the intended
  tracked files;
- both ignored webhook files remained mode 0600 and outside Git;
- `npm run typecheck` passed;
- the consolidated `@alpha/tests` suite passed 212 tests with 46 expected local NVIDIA-gated skips and zero
  failures.

The repository-wide `npm test` Turbo wrapper remains configuration-red because several library workspaces run
Vitest despite containing no local test files; Vitest exits 1 with `No test files found`. During that wrapper run,
the 26-test corpus package passed before Turbo stopped at the first empty workspace. This is not represented as a
test assertion failure, and it is not silently reported as a green root wrapper.

## Paid-compute closeout

The Alpha pod was `omn3hktwqs7r5l`, one RTX 3090 at $0.22/hour. It was removed only after the full run, every
checkpoint, all four evaluations, and remote logs were local and hash-verified. The before/after listings are:

    /mnt/donto-data/alpha-runs/alpha-chat-repair-v2-20260731/pilot-b/runpod-before-termination.txt
    /mnt/donto-data/alpha-runs/alpha-chat-repair-v2-20260731/pilot-b/runpod-after-termination.txt

The immediate after listing retained only unrelated pod `7pk5wnwgtazb0z`; that workload was deliberately
untouched. A later final documentation audit found the live RunPod list empty. No paid Alpha compute remains
live, and neither observation authorizes actions against another project's future pods.

## Discord record

No checkpoint passed the qualitative-improvement gate, so no v2 improvement announcement was posted. After
closeout, the operator explicitly requested a webhook test using the last frozen panel sample. The message was
labelled `TEST`, identified step 1,600 as rejected, included the exact `icecream or ice scream?` input/output, and
stated that semantic contingency failed. That test did not supersede the improvement-only posting rule.

## Scientifically valid restart

A future session should:

1. verify the public or mounted checkpoint by exact SHA-256;
2. treat step 1,600 as a branch state, not a selected model;
3. retain the frozen v2 development and sealed-final hashes;
4. define a new intervention aimed at semantic contingency or sequence dynamics before spending GPU;
5. create a new development selector rather than repeatedly tuning against these outputs;
6. compare against the published baseline on exact shared IDs;
7. select by generated conversation and fixed qualitative inspection, never loss alone;
8. execute a new sealed final only after one candidate passes all development gates.

The goal—an effective, genuinely chatty Alpha—remains unachieved. This record closes the authorized v2
experiment honestly; it does not redefine the failure as success.
