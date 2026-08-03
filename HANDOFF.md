# HANDOFF — Alpha foundation and Helios acceleration active; best honest chat checkpoint unchanged, state as of 2026-08-03

## ACTIVE GOAL — make the original Alpha model reliably chatty

### 2026-08-03 superseding execution state

The active path is now a better-trained small foundation followed by distillation and chat post-training. The
full foundation run has **not** begun. Its 2,058,181,632-token packed train cache and fixed validation cache are
verified, a matched three-arm pilot selected peak learning rate `0.002`, and the current candidate contract is
97,098,880 parameters, sequence length 1,024, batch 24, 79,020 optimizer steps, and 1,941,995,520 planned
training tokens. Symbiogenesis is disabled for this run. Do not describe the verified caches or LR pilot as a
trained foundation model.

Helios optimization now has five selected, numerically validated results. The exact per-dispatch Vulkan profiler
first guided a portable 2 x 2 register-blocked GEMM that reduced one-step dispatch time by 36.9% and raised
matched steady median throughput from the historical 3,579 to 4,513 tokens/s. Corrected physical-kernel labels
then exposed 637 `scale_vec4x2` calls as autograd gradient copies rather than useful arithmetic. The tape now
moves the final consumer's gradient buffer and clones only genuine aliases. A same-source trace-on control is
available with `ALPHA_DISABLE_GRADIENT_BUFFER_MOVE=1`; it measured 4,121.0 tokens/s versus 6,123.2 with ownership
forwarding (+48.6%). A longer trace-off production run measured 18 warm steps at p10/median/p90 6,432.6 /
6,567.7 / 6,666.5 tokens/s. Its median is +45.5% over the prior register-blocked baseline and +83.5% over the
historical path.

A third, portable 4 x 2 register-blocked GEMM was then evaluated per physical layout. It is faster for ordinary
and transposed-A multiplication but slower for transposed-B, so the selected portfolio uses R4x2 / R2 / R4x2 for
ordinary / transposed-B / transposed-A respectively. Across 18 warm production steps, it raised median throughput
from 6,567.7 to 6,836.8 tokens/s (+4.10%), with p10/p90 6,638.4 / 6,970.6. Maximum loss and gradient-norm
differences were `9.537e-7` and `4.308e-8`; the terminal held-out loss, learning rate, and clipping coefficients
matched. The complete RTX 4090 suite passed 29 files / 283 tests.

The remaining transposed-B path was then redesigned around physical memory layout. R42C makes adjacent X
invocations read contiguous K elements of B and transposes only into shared memory. A paired R2C control was
correct but neutral; R42C cut transposed-B time from 570,078.2 to 467,672.1 us (-17.96%) and full exact dispatch
time from 1,759,004.2 to 1,640,182.0 us (-6.75%). Its 18-warm-step p10/median/p90 was 6,844.8 / 7,048.9 /
7,200.8 tokens/s, another 3.10% median gain. Maximum loss and gradient-norm differences from the prior portfolio
were `9.537e-7` and `3.681e-8`; terminal validation, learning rate, and clipping coefficients matched. The full
physical suite again passed 29 files / 283 tests.

The transposed-A path now has the matching physical-layout remap. R42C-A makes adjacent X invocations read
contiguous M elements from physical `[K,M]` A and transposes only into the shared `[32,16]` tile. Across five
matched exact profiles, the original R42 controls took 336,395.8 and 338,954.0 us while three R42C-A samples took
290,239.8-292,475.6 us; candidate median is 13.61% below the control midpoint. In a conservative candidate-first
20-step production comparison, warm median throughput rose from 7,085.0 to 7,253.8 tokens/s (+2.38%). Loss was
exact across all steps and maximum gradient-norm drift was `2.154e-8`. The physical suite passed 29 files / 283
tests. Selected source commit is `028e9b31524e6d89b2caee76dad2ae47b8896e03`; complete evidence is under
`/mnt/donto-data/donto-resources/benchmarks/alpha-helios-matmul-transposed-a-coalesced-20260803/`.

Matched control/candidate losses and validation loss were exact; maximum gradient-norm difference was
`6.913e-7`. A later one-ulp replay difference was traced to legal nondeterministic ordering in repeated-token
embedding-gradient atomics, not hidden with a tolerance. Bounded replay shapes now use a fixed-order gather,
while real training retains the fast scatter. The failing case passed in 10 fresh GPU processes and the
default-on path then passed all 29 physical-GPU test files and 283 tests. A simple four-query dKV
unroll was separately rejected after making dKV 74.7% slower and the complete dispatch graph 15.6% slower. These
are engine gains, not behavioral model gains, so they have not triggered Discord, Hugging Face, or BLAH
publication.

The register-blocked portfolio remains explicitly selected with `HELIOS_MATMUL_REG4X2=1`,
`HELIOS_MATMUL_REG4X2_TRANSPOSED_B=1`, `HELIOS_MATMUL_TRANSPOSED_B_COALESCED=1`,
`HELIOS_MATMUL_TRANSPOSED_A_COALESCED=1`, and
`HELIOS_MATMUL_REG2X2=1` while more devices are measured. R4x2 needs only
ordinary scalar FP32 Vulkan compute, a 16 x 8 workgroup, 128 invocations, and 4 KiB of shared memory; the R2
fallback needs a 16 x 16 workgroup and 256 invocations. Their awkward-dimension numerical smokes also pass Mesa llvmpipe. That is useful portability
evidence but **not** physical AMD proof. The current RunPod catalog visible to this account offers NVIDIA GPUs
only. AMD support remains active work: Vulkan-on-Radeon first, plus a backend-neutral HIP/ROCm lowering for
Instinct rentals that do not expose production Vulkan.

The dedicated Alpha pod is currently `wtupxv15debnvh`, an RTX 4090 at USD 0.69/hour. It was live and idle after
the 2026-08-03 gates; pod state and price are volatile and must be rechecked. At the sustained median 7,253.8
tokens/s, the current full-token contract would take about 74.37 hours before validation/checkpoint overhead and
cost about USD 51.31 at that price. This is materially better but still not the accepted engine endpoint. The
next exact targets are a CODA-controlled GEMM-epilogue slice, correct reduced-precision matrix acceleration, a
real attention-backward redesign, column-sum/reductions, transposes, and operation-graph quotienting. Finish the
correctness-gated optimization/accelerator decision before starting the multi-day run.

Read these first:

- [Helios exact profiler and register-blocking evidence](docs/resume/HELIOS-PROFILER-REGISTER-BLOCKING-EVIDENCE-2026-08-03.md)
- [Helios optimization and AMD compatibility program](docs/resume/HELIOS-OPTIMIZATION-AND-AMD-PROGRAM-2026-08-03.md)
- [Foundation candidate feasibility and LR contract](docs/resume/FOUNDATION-CANDIDATE-FEASIBILITY-2026-08-02.md)
- [Current state](docs/resume/CURRENT-STATE.md)

The older historical sections below remain evidence, but any statement that no Alpha pod is billing, no paid run
is authorized, or the LR pilot is merely planned is superseded by this execution state.

The operator explicitly returned the project to the original product goal on 2026-07-31: Alpha should be a
small, natural conversational model. AlphaCorpus remains a valuable side project, but its D5 review workflow is
paused and is not the active model-training objective. Do not restart corpus generation, human-review pipeline
work, or public explorer work merely because the historical section below is detailed.

The latest authoritative records are the
[v2 mechanism analysis](docs/resume/CHAT-REPAIR-V2-MECHANISM-ANALYSIS-2026-08-01.md) and the unexecuted
[v3 experiment contract](docs/resume/CHAT-REPAIR-V3-EXPERIMENT-CONTRACT.md). The full v2 execution record remains
[Chat Repair v2 2026-07-31](docs/resume/CHAT-REPAIR-V2-2026-07-31.md). Repair v2 is complete and negative:

- Pilot A continued the published checkpoint for 800 bounded steps. Pilot B then ran the one predeclared
  clean-base control for 1,600 steps from SHA
  `08e14fa9604bf1b46ebcd5df37933c84d2496c1d05d9e4b32ebad98792cc6049`.
- Every v2 selector checkpoint answered all 96 prompts, so the response-initiation intervention worked. On the
  exact 69 prompts shared with the public baseline, however, the baseline had 24 loop flags; Pilot A's best was
  30 and the clean-base control's best was 29. Qualitative panels showed parroting, semantic errors, generic
  nonanswers, and repeated-phrase attractors.
- No v2 checkpoint was selected. The 150-prompt sealed-final suite was never executed or inspected. The public
  model, Space, and backend deliberately remain on the earlier selected step 1,200.
- All eight clean-base checkpoints, metrics, four evaluations, and logs are mounted under
  `/mnt/donto-data/alpha-runs/alpha-chat-repair-v2-20260731/`. Step 800 and terminal step 1,600 are additionally
  public with optimizer/RNG state at training-archive revision
  `c1117378c0bc8b81b408be09c000f80ea9f027d7`, path `chat-repair-v2-20260731/`.
- Anonymous verification found 53 public files and exact LFS hashes for both retained checkpoints. The public
  checksum manifest SHA-256 is `b733f5704e722faadd2e6e46cd9505be44e7952da75d3d001aa65ac92cc6cf5f`.
- Alpha pod `omn3hktwqs7r5l` was removed only after verified recovery. RunPod `7pk5wnwgtazb0z`, visible in the
  immediate termination proof, belonged to an unrelated workload and was untouched; it was no longer listed at
  the final documentation audit. Always recheck live pod ownership rather than treating either observation as
  permanent.
- No v2 improvement announcement was posted. One later operator-requested webhook test used an exact rejected
  sample and was clearly labelled as a test; it did not change the improvement-only rule.
- Do not continue either v2 branch blindly. A future paid run requires explicit authorization and a genuinely
  new finite intervention aimed at semantic contingency or autoregressive stability.

Repair v3 was subsequently authorized, executed, recovered, and rejected for release:

- The full 4,096-row CUDA rollout ledger, repetition mask, selection-ineligible paired probe, matched C0/U1 arms,
  and declared development evaluations completed on one secure RTX 4090. The real NVIDIA suite passed 50/50 after
  an exact-zero gradient-sign correction in `72079db`; native tokenizer binding was corrected in `8c0fe06` before
  selectable training.
- C0 and U1 each completed 400/400 finite steps from the same selected checkpoint with the same data order. U1
  step 400 reduced fresh-development loops from 35 to 6, fixed 29 C0 loop cases, introduced zero new paired loops,
  preserved 96/96 nonempty responses, and therefore passed the mechanical gate.
- A 48-triad blinded human-review packet is preserved under
  `/mnt/donto-data/donto-resources/research/alpha-chat-repair-v3-blind-review-20260801/`; its key remains sealed and
  human review remains pending. The sealed final was not opened.
- A matched live probe through `evals.blah.dev` was sufficient to reject promotion: U1 step 400 remained circular
  or wrong on explanation and conceptual distinction, ignored pragmatic intent, and did not improve ambiguity or
  identity reasoning. It fixed an autoregressive repetition symptom without adding the missing semantic ability.
- The complete remote artifact tree was mirrored locally: 273/273 files, 13,896,740,932 bytes, and every SHA-256
  matched. Pod `13ot2p3prx36th` was removed only after that proof. No Alpha RunPod is billing. Canonical execution
  record: `/mnt/donto-data/alpha-runs/alpha-chat-repair-v3-20260801/RUN.md`.
- The transient local U1 serving unit was stopped after the Blah probe. The public Hugging Face model, public
  backend, Space, and Blah production registration remain on selected step 1,200.

The active local preflight is now
[semantic-chat repair v4](docs/resume/CHAT-SEMANTIC-REPAIR-V4-PREFLIGHT.md). It measures a concrete data mismatch:
the 34,880-row corrective corpus was 86% SODA role-play and contained only 519 one-exchange conversations, while
the existing SmolTalk source held 186,043 direct user/assistant pairs that the repair builder excluded by design.
V4 tests a compact, structurally generated and reviewed `gpt-5.4` semantic-chat curriculum plus limited natural
multi-turn replay. The structured generation smoke passed. The first production attempt was stopped after 15/50
batches because an embedding/manual audit exposed cross-batch paraphrase clusters that the per-batch validator
could not see. Those attempts are preserved and excluded. Planned-v2 now uses one `gpt-5.5` global semantic
blueprint before `gpt-5.4` realization, then performs stronger-model review, semantic-overlap, tokenizer, mask,
and human-sample audits. No v4 GPU run or public model change has occurred.

The selected checkpoint's earlier recovery record is
[Chat Repair 2026-07-31](docs/resume/CHAT-REPAIR-2026-07-31.md). Its short version:

- The archived terminal checkpoint really did have an answer-initiation failure, but the evaluator, native API,
  Hugging Face template, and Space also appended an erroneous generation-only space after `<|assistant|>`.
  With byte-level BPE that created a standalone token at a boundary absent from SFT and could turn otherwise
  viable checkpoints into code-fence/forum attractors. Commit `cf4ad61` fixes every generation path and adds
  regressions; historical assistant turns retain their necessary content-leading space.
- A compact 34,880-conversation corrective corpus was built from existing staged sources, deterministically
  shuffled and split. The trainer now weights conversations equally, gives the first four assistant content
  tokens an 8x multiplier, gives terminal EOS 2x weight, and never counts EOS as an answer-start token.
- The full corrective run completed on a single RTX 4090 from the clean pre-SFT base. Checkpoint 1,200 is the
  selected candidate: SHA-256 `399f776b49acc0c8834ff8a7f2390454e2c5f2d833a264e3f83ff546e973cfec`.
  On the 48-case repair-development suite it produced 48/48 nonempty, 48/48 EOS-terminated responses, no role
  leaks, and five repetition loops. Examples include “It’s going well, thanks. How about you?”, “What happened?”,
  and “The elevator opened onto a beach.” This is a genuine structural recovery, not semantic mastery.
- Later checkpoints did not improve the overall conversational trade-off. Validation loss was lowest later, but
  free generation became more repetitive or failed to stop, so checkpoint selection correctly followed actual
  generation rather than teacher-forced loss.
- The untouched suite measured 55/100 structural passes, 70/100 nonempty replies, 56/100 EOS terminations,
  31 loops, and 0/200 exact or contained QA answers. The development suite substantially overstated
  generalization. The model is more conversational than the archived terminal release, but the final gate FAILS.
- The selected native checkpoint and full evidence are public at recovery revision
  `ffc447e8a0f2240d42ceb0abfd18ab5b427d5e60`; the standard Transformers model is revision
  `ab1c5be13a12c0feb2d5e2c9af89bd5924a0e8b0`; the static Space is revision
  `d87e0950baf0a16ccd2859c2cee6314602ba2881`.
- The public backend serves step 1,200 and exact checkpoint SHA
  `399f776b49acc0c8834ff8a7f2390454e2c5f2d833a264e3f83ff546e973cfec`, with no fallback model. A real browser
  submitted “Hey, how is your day going?” and received “It’s going well, thank you. How about you?”
- Pod `ksotbczj60mntk` was removed after the untouched outputs were copied, hashed, and recomputed;
  `runpodctl pod list` was empty immediately afterward.

Selected-repair local evidence root:

    /mnt/donto-data/alpha-runs/alpha-chat-repair-20260731/

Rejected-v2 local evidence root:

    /mnt/donto-data/alpha-runs/alpha-chat-repair-v2-20260731/

Current corpus root:

    /mnt/donto-data/donto-resources/research/alpha-chat-repair-20260731/

The generated replies are often shallow, vague, wrong, empty, or repetitive. Do not describe the selected or
rejected candidates as philosophically intelligent, generally capable, or reliably chatty. The selected model
demonstrates a partial recovery of conversational initiation; v2 proves that reliable initiation alone does not
produce semantic contingency or stable conversation.

---

## PAUSED SIDE PROJECT — AlphaCorpus D5 review pipeline

### Previous planning goal

The current project goal is the canonical suite at
[docs/synthetic-curriculum-prd/README.md](docs/synthetic-curriculum-prd/README.md). It defines Alpha as a small,
chatty, fact-light conversational model and makes construction of a Donto-inspired, deeply categorized
synthetic corpus and SQLite scientific ledger a principal half of the program. The first planned training study
is synthetic-only; model scale is governed by one-GPU feasibility, not a fixed parameter target.

**Current state:** the bounded D2–D4 implementation and D5 generation portion are complete. Read
[Execution 01](docs/synthetic-curriculum-prd/EXECUTION-01-LEDGER-AND-CALIBRATION.md). The canonical mounted
ledger is `/mnt/donto-data/donto-resources/research/alpha2-corpus`; it contains 48 GPT-5.4 candidates across six
quarantined families, 42 structurally valid and six retained rejections, with no human-accepted/training rows.
The public read-only explorer is specified by [PRD-11](docs/synthetic-curriculum-prd/PRD-11-PUBLIC-CORPUS-EXPLORER.md)
and its release evidence belongs in [Execution 02](docs/synthetic-curriculum-prd/EXECUTION-02-PUBLIC-CORPUS-EXPLORER.md).
It is live at `https://alpha.donto.org/corpus` and dynamically exposes all 135 tables and five views without
changing any candidate's stage. The public proxy blocks every legacy application API and every non-read method.
The latest audit packet is under the ledger's `releases/audit/` directory. Human conceptual adjudication is
still next, now governed by
[PRD-12](docs/synthetic-curriculum-prd/PRD-12-D5-HUMAN-ADJUDICATION.md) and its complete
[review instrument](docs/synthetic-curriculum-prd/APPENDIX-D-D5-REVIEW-INSTRUMENT.md). The census is all 48
candidates: blind conversational Pass A, contract-aware Pass B, six family syntheses, and separate content and
schema dispositions for the six `unknown_secondary_lens` rejections. No model critic has run, so do not call
those structural dispositions critic false accepts or rejects. The unratified next choices are in
[Decision Packet 01](docs/synthetic-curriculum-prd/DECISION-PACKET-01-D5-NEXT-STEP.md).

The review workflow is now operational and recorded in
[Execution 03](docs/synthetic-curriculum-prd/EXECUTION-03-D5-HUMAN-REVIEW-WORKFLOW.md). The first blinded Pass A
population has 12 assigned candidates. Its original rubric-v1 assignments are preserved as superseded; the
current rubric-v2 packet is at
`/mnt/donto-data/donto-resources/research/alpha2-corpus/releases/review/alpha-calibration-v1-a-rubric-v2/`.
All assignment responses and the reviewer competence/session-condition declaration are blank: there are still
zero human reviews, declarations, adjudications, release members, and training exposures. The current packet
SHA-256 is `8c6a99c8c4dc1d74ceca0e75eb1767bb3229b9c9f2529c26d800b58e62b66f92`. Submit through the local
`review-submit` CLI only after a real human completes the form; never fill it with a model while labeling the
result human.

The same blinded packet is now practical to review at `https://alpha.donto.org/corpus/review`. The original
deployed workspace is recorded in
[Execution 04](docs/synthetic-curriculum-prd/EXECUTION-04-D5-REVIEW-WORKSPACE.md), autosaves only in browser
local storage, validates against the same executable rubric as `review-submit`, and downloads JSON. It has no
public submit action; the app and Caddy return 405 to non-read methods. The original workspace release source
was `cc9be14`; human responses are still pending.

The aggregate-pipeline release was introduced at
`8f25d51c362921480d68e37a22d57a9ee39d47d9`, recorded in
[Execution 09](docs/synthetic-curriculum-prd/EXECUTION-09-D5-PIPELINE-VISIBILITY.md). The review dashboard
derives the complete aggregate D5 pipeline from SQLite: Pass A 0/48 with 12 open, hidden repeats 0/6, Pass B
0/48, Pass C 0/6, structural dispositions 0/6, and Pass D 0/1. It explicitly says the open packet is one
session within the 48-candidate census and exposes none of the lineage that would compromise blind review.
Public desktop and 390 px browser checks passed and POST remained 405. That release remains a historical
rollback point for the later immutable-envelope and workspace releases.

The local Pass B preparer is additionally fail-closed at revision
`b52792b4e0821852e500466be7f0640cf6f60b67`; see
[Execution 10](docs/synthetic-curriculum-prd/EXECUTION-10-D5-PASS-B-BLINDNESS-GATE.md). An audit found that the
old candidate-level predicate would reveal contracts for completed candidates before the remaining blind
census and repeat presentations. Pass B now requires all 48 Pass A reviews, all six repeat-stability rows, and
zero open first-class Pass A sessions for the same reviewer/rubric. A live premature Pass B command exited
nonzero at 0/48 and 0/6, while the canonical SQLite SHA-256 remained exactly
`7184a38a4213e319008d8f8f2b170f6d3c4c5d934b581c2afa9d7aad6c4847ce` before and after.

The immutable-envelope production release was introduced at
`e07477b934897b71f241724a230e2ccd6320e0c9`; see
[Execution 11](docs/synthetic-curriculum-prd/EXECUTION-11-D5-IMMUTABLE-REVIEW-ENVELOPE.md). The importer and
browser now canonicalize a packet with every response reset, then require every remaining byte of meaning to
match an exact exported packet. A changed assistant message with the original candidate hash and a changed
opaque presentation identity both fail without writing a review or submission artifact. A real public browser
tamper probe inserted a sentinel into local storage, reloaded, and observed the source packet restored in both
the DOM and storage. That checkpoint's blank packet remains preserved at SHA-256
`6740d83545335ec520989452eb2619bead4d95af62e681c7dfcd7e9245132c48`; human reviews remain zero. The 22/22
corpus suite, optimized web build, canary, manifest, public 200/405 boundaries, ledger integrity, foreign keys,
and blob validation pass. The project-owned footprint is 35.76 MiB.

The local downstream importers are additionally hardened at revision `6a33410`; see
[Execution 12](docs/synthetic-curriculum-prd/EXECUTION-12-D5-ALL-PACKET-ENVELOPE-BINDING.md). Pass C family
synthesis plus structural dispositions and Pass D closeout now use the same exact exported-envelope contract
as A/B. Completed-but-timestamp-altered packets write zero scientific or raw-submission evidence, and accepted
events retain separate envelope and submission hashes. No public redeploy was necessary because the public
workspace exposes only the already-protected A/B instrument. The canonical SQLite main-file SHA remains
`7184a38a4213e319008d8f8f2b170f6d3c4c5d934b581c2afa9d7aad6c4847ce`; every human-evidence count remains zero.

The future D6 substrate is now specified—but not executed—in
[PRD-13](docs/synthetic-curriculum-prd/PRD-13-EVALUATION-FIREWALL-AND-FREEZE.md). The live schema currently has
only a preliminary `evaluation_output` placeholder. PRD-13 defines the missing evaluation suite, family, item,
expectation, review, metric, run, output, contamination, split-closure, human-study, and private-payload
relations. It preserves the operator's public-all-tables goal by publishing safe metadata and commitments while
keeping private evaluation plaintext in an unserved encrypted vault. Do not migrate or populate D6 before the
human D5 closeout and a bounded operator decision.

The other current planning-only refinement is
[PRD-14](docs/synthetic-curriculum-prd/PRD-14-RESPONSE-POLICY-CONTROL-PLANE.md). An aggregate-only SQLite query
found 48 distinct legacy `intendedResponsePolicy` strings across 48 current candidates and zero
`response_policy_target` rows; it did not reveal candidate-to-policy mappings. PRD-14 specifies an extensible
policy graph, compositional targets, natural-language compilation, observed-behavior evidence, and release
balance. Do not backfill or publish individual mappings before D5 blindness is complete.

Execution 13 records the prior workspace-hardening release and preserves its corrections. Its claim that the
position value was an opaque identity was wrong: revision `c4e7c4d...` stored a packet-scoped numeric index.

The Execution-14 production review application was
`ccdd9e6f00129a28571fd1158b42acf508fb0991`; see
[Execution 14](docs/synthetic-curriculum-prd/EXECUTION-14-D5-REVIEW-SESSION-PROVENANCE.md). It retains the
navigation and accessibility improvements, now stores the active opaque item identity, requires competence
scope plus start/end/interruption/fatigue/condition evidence, and preserves older v1 packets as readable but
explicitly incomplete. Migration 8 brings the ledger to eight migrations, 131 tables, five views, and 190
triggers. The 23/23 corpus suite, typecheck, optimized build, immutable canary, desktop/mobile public Chromium
checks, legacy-packet index proof, and 200/405 method boundary pass. The deployed immutable release is
`/home/ajax/alpha2-web-releases/ccdd9e6f00129a28571fd1158b42acf508fb0991`; its manifest SHA-256 is
`927f62f0240228ef5cd36016199c66a953796f6706a72f0cfbb03c0b1acdf147`. The main SQLite SHA-256 is
`d2ccec649ab4aaeb0aac427391de0366a2b16b852477b2aa0da1724dd2ce9d19`; 12 Pass A assignments remain open,
all human/downstream evidence remains zero, and the project-owned tree is 42,336,230 bytes (40.37 MiB).
The pre-migration backup is
`backups/pre-d5-session-declarations-20260731T054250Z.sqlite` at SHA-256
`ee96b8a8573df70a1ab4b0d4b56176a4b765a062cf3a82fde85d574ae73fec38`; migration 8 digest is
`0374db80ce8ff18195c7e8f1ce57b78bac6f13b9c9f92f6f23014bb93f8b0f51`.

The current production review application is
`2c6da67121938c8c033e7d0e30ddaefd0f0949ba`; see
[Execution 15](docs/synthetic-curriculum-prd/EXECUTION-15-D5-REVIEW-EVIDENCE-COMPLETENESS.md). Rubric v2 makes
the two immediate-comprehension judgments, evidence for every dimension, explicit non-numeric states, and the
complete finding repair contract executable and append-only. Twelve blank v1 assignments remain linked to
their twelve v2 replacements over the same candidate-content hashes. The ledger has ten migrations, 135
tables, five views, and 198 triggers. The current SQLite SHA-256 is
`0695bbf651d74c227931016fbe14e617337872d05ebf494f33aa264973dd327b`; every human, adjudication, release,
training, and authorization count remains zero. Pass B's contract-indexed Appendix D matrices are the next
implementation seam and must not be approximated by the current generic dimensions.

The D5 deterministic profile is now first-class and recorded in
[Execution 05](docs/synthetic-curriculum-prd/EXECUTION-05-D5-SURFACE-EVIDENCE.md). Current run
`analysis_run_a32f01644e1a96a9ea601b35b35626aa` is tied to exact revision
`002e123f71cdc30ef883b6fb6222a47240578001` and input snapshot
`8ce2bf64152c674f17b707137774ad7be42eb0ad5c2a51ec3953a196f75ac03e`. It contributes 236 metrics,
2,256 similarity edges, and 488 signatures. The earlier run
`analysis_run_e6e5e45da332d9482be1ed7ca586780f` contains an incorrect operator-supplied revision and is
explicitly superseded by `analysis_run_correction_d74d355ddcfc337474809c4dd1b618e0`; never cite it as
current.

The Pass C family-synthesis and structural-rejection workflow is implemented at revision `66783ae` and
recorded in
[Execution 06](docs/synthetic-curriculum-prd/EXECUTION-06-D5-FAMILY-SYNTHESIS-WORKFLOW.md). Migration 5 adds
five first-class tables and eight append-only triggers, taking the canonical ledger to five migrations, 117
tables, four views, and 168 triggers. A pre-migration SQLite backup has SHA-256
`c4ae7e7b0720fd4c69f97b3331d53946e4b8dfbc9cc15869f2d0af959c58450e`. The live Pass C preparation correctly
failed because A/B evidence is incomplete and created zero assignments, syntheses, structural dispositions,
release members, or training exposures. All new relations return HTTP 200 in the public explorer. The
canonical ledger was clean with zero FK/blob failures and a 24.03 MiB project-owned footprint at that
checkpoint.

The hidden-repeat implementation is now recorded in
[Execution 07](docs/synthetic-curriculum-prd/EXECUTION-07-D5-BLINDED-REPEAT-PRESENTATIONS.md) at revision
`249a00f`. Migration 6 took that checkpoint to six migrations, 122 tables, five views, and 174 triggers.
Its pre-migration backup SHA-256 is
`c914d560405642ce641570ed8794172ec3a36edc2948ed3d27bda800059b2918`. The original 12-item packet remains
byte-identical at SHA-256 `6740d83545335ec520989452eb2619bead4d95af62e681c7dfcd7e9245132c48`.
All presentation/repeat rows remain zero until a real human completes Pass A; all six new public relations
return HTTP 200. The footprint at that checkpoint was 29.68 MiB.

The final fail-closed Pass D campaign-closeout path is implemented at revision `6cd4921` and recorded in
[Execution 08](docs/synthetic-curriculum-prd/EXECUTION-08-D5-CAMPAIGN-CLOSEOUT-WORKFLOW.md). Migration 7
brings the canonical ledger to seven migrations, 129 tables, five views, and 186 triggers. Its pre-migration
backup SHA-256 is `ad4afb5622f30adca5c00df4b2425805bfe5235c250c8cc11f097f267002d5e3`; migration digest is
`d64055b6b6f1572d2a3fd1bd2f57760b6b164fe3e77aaa76be04c9cb4b7ab275`. The live preparation command fails
at the first missing sealed Pass A review and creates zero closeout assignments. Closeouts, adjudications,
recommended states, failure clusters, release members, and training exposures remain zero. SQL enforces
`execution_authorized = 0`; Pass D can recommend later evidence stages but can never start them. All seven
new relations return HTTP 200 in the public explorer. Integrity, foreign keys, and blob checks are clean; the
project-owned footprint was 35.39 MiB at that checkpoint. After the Execution 09 browser evidence, the complete
project-owned tree is 35.65 MiB. Integrity, foreign keys, and blobs remain clean. The immediate
authority-bearing action remains real human Pass A.

Do not expand the corpus, invoke GPT-5.5, train, provision GPU infrastructure, mutate live Donto, publish a
different artifact, or send additional ad hoc Discord messages without another bounded operator instruction.
The public read-only `/corpus` publication is the explicit exception recorded on 2026-07-30. The former factual
progress timer `alpha-corpus-discord-progress.timer` was disabled and stopped when AlphaCorpus was paused on
2026-07-31; its first historical post returned HTTP 204 at 09:09 UTC. GPT-5.6-sol is the counsel tier and GPT-5.4 is the initial worker. New corpus work pauses if the
project-owned artifact tree exceeds 15 GiB; this does not create a global disk rule.

---

For any future agent. **Read `GOAL.md` first** (repo root): the archived Alpha 60M program is closed, its
execution gates passed, and its chat-quality gate failed. This file preserves the exact restart/publication
state and the chronological paid-run evidence. Box operating rules live in `/home/ajax/CLAUDE.md`; roadmap
state is mirrored in `/home/ajax/TODO.md`.

**Future resume entry point:** [docs/resume/README.md](docs/resume/README.md). Read its session checklist,
current state, and decisions before reading the historical live record below. A bare “resume” request is
not authorization to create a pod.

---

## CURRENT CLOSEOUT — no Alpha GPU or training run is live

- **Operator decision:** no more model runs after the sealed terminal evaluation. Do not provision a
  RunPod or continue training unless the user gives a new explicit authorization and a new continuation
  contract is written. The Alpha pod `gp4m6s8m06bhen` was removed by the fail-closed finalizer at
  2026-07-30 00:17:45 UTC. RunPod then showed only an unrelated Wajarri pod; it was not touched.
- **Terminal SFT:** exact source `c333bf247fbe87b85d01f3d34789b46615dd1034`, 30,322/30,322 finite
  steps, one assistant-only epoch, 496,795,648 padded tokens, median post-warmup 3,847.23 tok/s, final
  train/held-out loss 1.7579851/1.6439665, all 57,688,576 parameters finite/nonzero, and zero allocator
  overflow. Native checkpoint SHA-256 is `6c279d086d8c0679495e38ebec8a473ac23d16bfb3b93516e144712963fecbc8`.
- **D3 FAILED:** sealed chat result is 2/100 structural, 94/100 EOS, 8/100 nonempty, six degenerate
  loops, and mean/max 4-gram repetition 0.04904/0.984. Closed-book QA is 0/200 exact, 0 contained,
  mean F1 0. A reference-blinded inspection of every output returned 0 PASS / 0 BORDERLINE / 100 FAIL:
  92 were empty, six looped, and the two remaining fragments were unusable. Machine report SHA
  `92da0b3b...`; semantic report SHA `35cc1a87...`. This is not a chatty model.
- **Standard public model:** [ajaxdavis/alpha-60m-chat](https://huggingface.co/ajaxdavis/alpha-60m-chat),
  immutable revision `b481f46924b7a4777a029de1ffb44c06cc925d4c`, weights SHA `6bb34908...`.
  Anonymous empty-cache stock Transformers loaded `LlamaForCausalLM`, exact parameter/hash counts,
  CPU-only and without custom code. Plain text produced `Hello ever`; the message-list pipeline emitted
  an empty assistant string, consistent with the sealed failure.
- **Native restart archive:**
  [ajaxdavis/alpha-60m-training-checkpoints](https://huggingface.co/ajaxdavis/alpha-60m-training-checkpoints),
  revision `7198d1a1f094ffe88d06399ea99fecbd78fa8b66`. It contains full optimizer/RNG/tokenizer checkpoints
  for base step 61,036 (`08e14fa9...`), best surviving SFT step 29,000 (`03eaac3e...`), and terminal
  SFT step 30,322 (`6c279d08...`), plus the complete contracts, metrics, audits, and failed evals.
  Local hardlink bundle: `/mnt/donto-data/alpha-runs/alpha-60m-continuation-c333bf2-20260730/`;
  `MANIFEST.sha256` is verified and `RESUME.md` gives the future RunPod procedure.
- **Public Space:** [ajaxdavis/alpha-60m-chat](https://huggingface.co/spaces/ajaxdavis/alpha-60m-chat),
  immutable Space revision `be0bd0428631d1585b13ddf9e93a8ed2d9254606`. Hugging Face rejected
  Docker `cpu-basic` without PRO, so the free static Space calls the exact Alpha CPU inference backend
  at `https://donto.org/alpha-60m`. Runtime source is `5bd723d`; service
  `alpha2-hf-backend.service` is nice 19, CPU-capped, 3GB memory-capped, loopback-only behind Caddy,
  and checkpoint-hash pinned. Desktop and 390px browser checks passed with no overflow; the UI prompt
  flow displayed the real empty EOS in 1.883s. Screenshots and public API proof are under
  `/mnt/donto-data/alpha-runs/alpha-60m-space-runtime-5bd723d-20260730/`.
- **Discord:** no post was sent at terminal because the user required improvement-only posts and none
  of the final outputs improved qualitatively. The ignored webhook file remains local and mode 0600.
- **Canonical evidence:** terminal run
  `/mnt/donto-data/alpha-runs/flagship-sft-c333bf2-20260728/`; publication reports live directly under
  `/mnt/donto-data/alpha-runs/hf-*-20260730*`. Final project tag is
  `alpha-60m-archive-20260730`.

## Historical live recovery record — no endpoints or pods below are current

- **Active pod `gp4m6s8m06bhen`**, RTX 3090 community, **$0.22/hr**. SSH:
  `ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p 10784 root@99.69.17.69`.
- The same immutable flagship run is live at
  `/workspace/alpha2/runs/flagship-1b-e561f66-20260724` on exact detached source
  `e561f66c7a88a5294e9cb74a4fc3afd6be167d4f`. Trainer PID 2173 is nice 10; launch log is
  `/workspace/alpha2-run-logs/flagship-1b-e561f66-20260724.recovery2.log`.
- The original host was unexpectedly marked `Exited by user` by RunPod at 21:49:25 UTC after step
  28,900 but before checkpoint 29,000. No stop was issued by this session and the guard did not have
  auto-termination enabled. Its exact 28,900-row prefix is preserved locally as
  `metrics.pre-recovery-28900.jsonl` and
  `metrics.pre-resume-checkpoint-28000-through-28900.jsonl` at SHA-256 `bec96f18…`; those abandoned
  900 rows are evidence only and are not part of the canonical continuation.
- Recovery fell back to the last fully audited checkpoint 28,000: 692,528,817 bytes, SHA-256
  `b9f80989…`, all 114 tensors / 57,688,576 parameters finite and nonzero. Recovery2 independently
  passed the current NVIDIA gate **46/46**, hash-verified all 5,976,889,749 source bytes, rebuilt all
  six train/validation token caches, and recorded the truncation in `resume-ledger.jsonl` before
  resuming at step 28,000.
- The canonical recovery replay has re-passed the full step-28,500 gate: 28,500 consecutive finite
  rows / 466,944,000 tokens (46.6938%), p10/median 3,728.9936/3,858.6430 tok/s, all 286 allocator
  samples present, exactly 34 temporary slabs, and zero free-range overflow. Train/held-out loss is
  3.2907429/3.3982231; held-out differs from the abandoned trajectory's corresponding gate by only
  +0.0035649. Remote/mounted metrics match at 10,264,519 bytes / SHA-256 `9a9edd57…`. Steady-state
  recovery steps 28,101–28,500 held RSS at 7,936–7,948MB and ArrayBuffers at 6,995–6,996MB. GPU
  utilization was 100%, `/runpod` had 8.7GB free, and account balance was `$44.7557173066`.
- Checkpoint 29,000 is the first new durable point beyond the failed host. All 29,000 rows are finite
  and consecutive, covering 475,136,000 tokens (47.5129%); p10/median is
  3,729.7739/3,859.2590 tok/s; all 291 allocator samples report 34 slabs/zero overflow. Train/held-out
  loss is 3.2767162/3.4017447. Exact metrics `5a1e0af4…` and the 692,528,817-byte checkpoint
  `2e66f8d3…` match remote/mounted; native audit `a977b8aa…` passed all 114 parameter tensors /
  57,688,576 elements finite and nonzero. Safe retention leaves 27k/28k/29k locally and 28k/29k
  remotely. Training resumed through 29,050 at 100% GPU; balance was `$44.5628176547`.
- **Memory discriminator RESOLVED for live optimizer buffers:** pre-save steps 28,501–29,000 held
  ArrayBuffers/RSS exactly 6,996/7,948MB. After checkpoint 29,000 released all 228 cloned optimizer
  buffers, steps 29,001–30,000 held ArrayBuffers exactly 7,292MB. Checkpoint 30,000 again released all
  228 clones; steps 30,001–30,050 returned to exactly 7,292MB, so the +296MB recovery increment did
  not repeat per save. RSS established a higher 8,471–8,472MB plateau (+~220MB from immediately
  pre-save) without live ArrayBuffer/external growth; keep watching it at checkpoint 31,000, but the
  64GB host has ample headroom and there is no reason to disturb the immutable run.
- Step 29,500 also passed: 29,500 finite/consecutive rows, 483,328,000 tokens (48.3321%), p10/median
  3,730.4182/3,859.8298 tok/s, 296 complete allocator samples, 34 slabs, and zero overflow.
  Train/held-out loss is 3.2149160/3.4070656, only +0.0053210 from checkpoint 29,000. Exact metrics
  match remote/mounted at `07b03e0c…`. Across the entire post-checkpoint 500-row window,
  ArrayBuffers stayed exactly 7,292MB and RSS stayed within 8,179–8,258MB, ruling out per-step
  growth. Balance was `$44.4181244601`.
- **Checkpoint 30,000 PASSED; new held-out run best:** 30,000 finite/consecutive rows cover
  491,520,000 tokens (49.1513%); p10/median is 3,731.2399/3,860.6254 tok/s; all 301 allocator samples
  report 34 slabs and zero overflow. The last 500 rows averaged loss/gradient norm
  3.3288895/0.2294206. Train/held-out loss is 3.1875825/3.3639263, improving 0.0431393 from step
  29,500 and 0.0041162 from the former step-27,000 best. Exact 30,000-row metrics `f4a39944…` and the
  692,528,817-byte checkpoint `1625c7d6…` match remote/mounted. Independent native audit
  `6ec3b0ff…` passed all 114 parameter tensors / 57,688,576 elements finite and nonzero. Safe
  retention is exactly 28k/29k/30k on both sides. Training resumed through 30,050; balance
  `$44.2492298008`, total account burn `$0.303/hr`, and mounted disk has 85GB free.
- **Step 30,500 PASSED; another held-out run best:** 30,500 finite/consecutive rows cover
  499,712,000 tokens (49.9705%); p10/median is 3,731.9190/3,861.3858 tok/s; all 306 allocator
  samples report exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.3303287/0.2359238 and held ArrayBuffers/external memory exactly 7,292/7,294MB, with RSS
  8,417–8,490MB. Train/held-out loss is 3.2476387/3.3596032, improving 0.0043231 from checkpoint
  30,000. Exact remote/mounted metrics match at `c9efd9ed…`; guard remains active/zero-restart.
  Balance is `$43.9518146859`. Total account burn rose to `$0.75/hr` because the unrelated
  `mobtranslate-wajarri-v1-gpu-preflight-a3-20260726` A40 pod is now running; Alpha itself remains
  the scoped RTX 3090 at `$0.22/hr`, and the unrelated pod was not touched.
- **Checkpoint 31,000 PASSED; new held-out run best and over halfway:** 31,000 finite/consecutive rows
  cover 507,904,000 tokens (50.7897%); p10/median is 3,732.8800/3,862.6160 tok/s; all 311
  allocator samples report exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient
  norm 3.3137590/0.2355993 and held RSS/ArrayBuffers/external exactly 8,490/7,292/7,294MB.
  Train/held-out loss is 3.3144221/3.3412647, improving 0.0183384 from step 30,500. Exact metrics
  `f6092046…` and the 692,528,817-byte checkpoint `8372b814…` match remote/mounted; native audit
  `948de04f…` passed all 114 tensors / 57,688,576 elements finite and nonzero. The save released all
  228 clones; steps 31,001–31,050 returned to exactly 7,292/7,294MB ArrayBuffers/external while RSS
  settled only 48MB higher at 8,538MB. Safe retention is exactly 29k/30k/31k on both sides. Balance
  `$43.5891647432`; only the Alpha pod is running and total account burn returned to `$0.303/hr`.
- **Step 31,500 PASSED; substantial new held-out run best:** 31,500 finite/consecutive rows cover
  516,096,000 tokens (51.6089%); p10/median is 3,733.7679/3,863.6001 tok/s; all 316 allocator
  samples report exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.3260559/0.2355921 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,467–8,538MB. Train/held-out loss is 3.4092531/3.2892870, improving 0.0519777 from checkpoint
  31,000. Exact remote/mounted metrics match at `c63ee5b8…`; guard remains active/zero-restart.
  Balance is `$43.3246554467`, only Alpha is running, and total account burn remains `$0.303/hr`.
- **Checkpoint 32,000 PASSED; held-out remains second-best:** 32,000 finite/consecutive rows cover
  524,288,000 tokens (52.4281%); p10/median is 3,734.6678/3,864.4482 tok/s; all 321 allocator
  samples report exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.3093260/0.2391542 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,537–8,538MB. Train/held-out loss is 3.3595934/3.3350753: +0.0457882 from the unusually strong
  31,500 window but 0.0061894 better than checkpoint 31,000. Exact metrics `08eb6938…` and the
  692,528,817-byte checkpoint `e82ac311…` match remote/mounted; native audit `79ba1fa4…` passed all
  114 tensors / 57,688,576 elements finite and nonzero. The save returned ArrayBuffers directly to
  baseline; steps 32,001–32,050 held 7,292/7,294MB ArrayBuffers/external and RSS only 1MB higher at
  8,539MB. Retention is exactly 30k/31k/32k on both sides. Balance `$42.9987368539`, total burn
  `$0.303/hr`, mounted disk 84GB free.
- **Step 32,500 PASSED:** 32,500 finite/consecutive rows cover 532,480,000 tokens (53.2473%);
  p10/median is 3,735.5363/3,865.5490 tok/s; all 326 allocator samples report exactly 34 slabs/zero
  overflow. The last 500 rows averaged loss/gradient norm 3.3296892/0.2432705 and held
  ArrayBuffers/external exactly 7,292/7,294MB, with RSS 8,466–8,539MB. Train/held-out loss is
  3.1392970/3.3509093, +0.0158340 from checkpoint 32,000 and +0.0616222 from the sharp step-31,500
  best; all invariants remain green, so this is normal five-batch variance pending checkpoint 33,000.
  Exact remote/mounted metrics match at `441c237d…`; guard remains active/zero-restart. Balance
  `$42.8003199855`; total burn is `$0.75/hr` because unrelated Wajarri pod `b21dbqjy0t3gir` is
  running at `$0.44/hr`. Alpha remains `$0.22/hr`; the unrelated pod was not touched.
- **Checkpoint 33,000 PASSED; held-out recovered:** 33,000 finite/consecutive rows cover 540,672,000
  tokens (54.0665%); p10/median is 3,736.2691/3,866.5968 tok/s; all 331 allocator samples report
  exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.2952673/0.2418966 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,465–8,538MB. Train/held-out loss is 3.2223387/3.3299192, improving 0.0209901 from step 32,500
  and 0.0051561 from checkpoint 32,000; it remains 0.0406322 above the sharp step-31,500 best. Exact
  metrics `addc830f…` and the 692,528,817-byte checkpoint `000d1d09…` match remote/mounted; native
  audit `9893f0c0…` passed all 114 tensors / 57,688,576 elements finite and nonzero. Post-save steps
  33,001–33,050 held the exact 7,292/7,294MB live-buffer baseline and the 8,528MB pre-save RSS,
  adding no new plateau. Retention is 31k/32k/33k both sides. Balance `$42.523076967`, total burn
  `$0.303/hr`, mounted disk 82GB free.
- **Step 33,500 PASSED; held-out improved again:** 33,500 finite/consecutive rows cover 548,864,000
  tokens (54.8856%); p10/median is 3,737.3560/3,867.5338 tok/s; all 336 allocator samples report
  exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.2911672/0.2427404 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,469–8,541MB. Train/held-out loss is 3.1827683/3.3272114, improving 0.0027078 from checkpoint
  33,000; it remains 0.0379244 above the sharp step-31,500 best. Exact remote/mounted metrics match
  at `20feca37…`; the guard remains active/zero-restart. Balance `$42.35422643`, only Alpha is
  running, total burn is `$0.303/hr`, and mounted disk has 83GB free.
- **Checkpoint 34,000 PASSED; save and memory gate clean:** 34,000 finite/consecutive rows cover
  557,056,000 tokens (55.7048%); p10/median is 3,738.1565/3,868.6421 tok/s; all 341 allocator
  samples report exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.2653971/0.2541467 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,465–8,539MB. Train/held-out loss is 3.3614364/3.3465386, a +0.0193273 wobble from step 33,500
  while remaining 0.0043707 better than step 32,500. Exact metrics `87ef124c…` and the
  692,528,817-byte checkpoint `2d63169b…` match remote/mounted; native audit `3102c2b0…` passed all
  114 tensors / 57,688,576 elements finite/nonzero. Steps 34,001–34,050 returned to exactly
  7,292/7,294MB buffers and RSS 8,538MB. Retention is 32k/33k/34k both sides. Balance
  `$42.185329143`, only Alpha is running, total burn is `$0.303/hr`, and mounted disk has 82GB free.
- **Step 34,500 PASSED; held-out wobble on watch:** 34,500 finite/consecutive rows cover 565,248,000
  tokens (56.5240%); p10/median is 3,739.0630/3,869.5629 tok/s; all 346 allocator samples report
  exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.2622366/0.2468041 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,467–8,540MB. Train/held-out loss is 3.3501410/3.3995402, a +0.0530016 wobble from checkpoint
  34,000 and +0.1102532 above the sharp step-31,500 best. This is comparable to earlier recovered
  five-batch variance; do not intervene on one read. Exact remote/mounted metrics match at
  `9eb80597…`; guard remains active/zero-restart. Balance `$41.8946601577`; total burn is `$0.75/hr`
  because unrelated Wajarri pod `2q7ky3hpzbsw17` is running at `$0.44/hr`. Alpha remains `$0.22/hr`
  and the unrelated pod was not touched.
- **Checkpoint 35,000 PASSED; wobble resolved to a new run best:** 35,000 finite/consecutive rows
  cover 573,440,000 tokens (57.3432%); p10/median is 3,739.8840/3,870.6404 tok/s; all 351
  allocator samples report exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient
  norm 3.3145007/0.2491159 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,467–8,539MB. Train/held-out loss is 3.3514276/3.2819459, improving 0.1175943 from step 34,500
  and setting a new run best by 0.0073411 over step 31,500. Exact metrics `c4144895…` and the
  692,528,817-byte checkpoint `df9dc23a…` match remote/mounted; native audit `ce6e46a3…` passed all
  114 tensors / 57,688,576 elements finite/nonzero. Steps 35,001–35,050 returned to exactly
  7,292/7,294MB buffers and RSS 8,539MB. Retention is 33k/34k/35k both sides. Balance
  `$41.6485739058`, only Alpha is running, total burn is `$0.303/hr`, and mounted disk has 78GB free.
- **Step 35,500 PASSED; five-batch variance on watch:** 35,500 finite/consecutive rows cover
  581,632,000 tokens (58.1624%); p10/median is 3,740.7007/3,871.8695 tok/s; all 356 allocator
  samples report exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.3062677/0.2554765 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,464–8,541MB. Train/held-out loss is 3.2625861/3.3820238: +0.1000779 from the unusually strong
  checkpoint-35,000 best, but already 0.0175164 better than the prior 34,500 wobble. Exact
  remote/mounted metrics match at `f31899c0…`; guard remains active/zero-restart. Balance
  `$41.458165665`; total burn is `$0.75/hr` because unrelated Wajarri pod `9u5z7t9uv6e8ac` is
  running at `$0.44/hr`. Alpha remains `$0.22/hr` and the unrelated pod was not touched.
- **Checkpoint 36,000 PASSED; elevated validation persists but hard gates are green:** 36,000
  finite/consecutive rows cover 589,824,000 tokens (58.9816%); p10/median is
  3,741.3099/3,873.2473 tok/s; all 361 allocator samples report exactly 34 slabs/zero overflow.
  The last 500 rows averaged loss/gradient norm 3.2717362/0.2514415 and held
  ArrayBuffers/external exactly 7,292/7,294MB, with RSS 8,465–8,540MB. Train/held-out loss is
  3.2520704/3.3931745: +0.0111507 from step 35,500 and +0.1112286 from the sharp 35,000 best. Two
  elevated windows are now explicit, but this remains within established oscillation and no hard
  stop condition fired. Exact metrics `8fb078fa…` and 692,528,817-byte checkpoint `696a20f8…` match
  remote/mounted; native audit `b5321d32…` passed all 114 tensors / 57,688,576 elements
  finite/nonzero. Steps 36,001–36,050 returned to exactly 7,292/7,294MB buffers and RSS 8,540MB.
  Retention is 34k/35k/36k both sides. Balance `$41.0687693316`, only Alpha is running, total burn
  is `$0.303/hr`, and mounted disk has 76GB free.
- **Step 36,500 PASSED; elevated trend recovered:** 36,500 finite/consecutive rows cover
  598,016,000 tokens (59.8008%); p10/median is 3,741.9697/3,874.4000 tok/s; all 366 allocator
  samples report exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.2918009/0.2540529 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,467–8,550MB. Train/held-out loss is 3.3547421/3.3283298, improving 0.0648447 from checkpoint
  36,000 and 0.0536940 from step 35,500; it remains 0.0463839 above the sharp 35,000 best. Exact
  remote/mounted metrics match at `747f7b02…`; guard remains active/zero-restart. Balance
  `$40.9240139261`, only Alpha is running, total burn is `$0.303/hr`, and mounted disk has 74GB free.
- **Checkpoint 37,000 PASSED; new run best and over 60%:** 37,000 finite/consecutive rows cover
  606,208,000 tokens (60.6200%); p10/median is 3,742.6444/3,875.4751 tok/s; all 371 allocator
  samples report exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.2631735/0.2608119 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,528–8,539MB. Train/held-out loss is 3.2379632/3.2644020, improving 0.0639278 from step 36,500
  and setting a new run best by 0.0175439 over checkpoint 35,000. Exact metrics `e0e49b59…` and the
  692,528,817-byte checkpoint `5fddd499…` match remote/mounted; native audit `a8419c04…` passed all
  114 tensors / 57,688,576 elements finite/nonzero. Steps 37,001–37,050 returned to exactly
  7,292/7,294MB buffers and RSS 8,540MB. Retention is 35k/36k/37k both sides. Balance
  `$40.5897592151`; total burn is `$0.75/hr` because unrelated Wajarri pod `2d55zbgwjg13ta` is
  running at `$0.44/hr`. Alpha remains `$0.22/hr`, and mounted disk has 74GB free.
- **Step 37,500 PASSED; validation remains near the new best:** 37,500 finite/consecutive rows cover
  614,400,000 tokens (61.4392%); p10/median is 3,743.6359/3,876.5917 tok/s; all 376 allocator
  samples report exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.2731337/0.2579882 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,465–8,542MB. Train/held-out loss is 3.2774620/3.2806316, only 0.0162296 above the step-37,000
  run best. Exact remote/mounted metrics match at `4182bd2a…`; the guard remains active with zero
  restarts. Balance is `$40.2212785873`; total burn is `$0.75/hr` because unrelated Wajarri pod
  `2d55zbgwjg13ta` remains running at `$0.44/hr`. Alpha remains `$0.22/hr`, and mounted disk has
  73GB free.
- **Checkpoint 38,000 PASSED; validation improved and save/memory gate clean:** 38,000
  finite/consecutive rows cover 622,592,000 tokens (62.2583%); p10/median is
  3,744.5871/3,877.7341 tok/s; all 381 allocator samples report exactly 34 slabs/zero overflow.
  The last 500 rows averaged loss/gradient norm 3.2679434/0.2623056 and held
  ArrayBuffers/external exactly 7,292/7,294MB, with RSS 8,540–8,541MB. Train/held-out loss is
  3.3129361/3.2791747, improving 0.0014569 from step 37,500 and remaining only 0.0147727 above the
  run best. Exact metrics `fc9dfc4d…` and 692,528,817-byte checkpoint `e792bb50…` match
  remote/mounted; native audit `0dc9b5e7…` passed all 114 tensors / 57,688,576 elements
  finite/nonzero. Steps 38,001–38,050 returned exactly to 7,292/7,294MB buffers and 8,541MB RSS.
  Retention is 36k/37k/38k both sides. Balance `$40.0002825224`; only Alpha is running, total burn
  is `$0.303/hr`, and mounted disk has 73GB free.
- **Step 38,500 PASSED; validation remains tightly clustered near best:** 38,500 finite/consecutive
  rows cover 630,784,000 tokens (63.0775%); p10/median is 3,745.4257/3,879.0059 tok/s; all 386
  allocator samples report exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient
  norm 3.2529682/0.2654853 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,466–8,544MB. Train/held-out loss is 3.2997167/3.2809323, a negligible +0.0017577 wobble from
  checkpoint 38,000 and only 0.0165303 above the run best. Exact remote/mounted metrics match at
  `bfc478d3…`; the guard remains active with zero restarts. Balance is `$39.831379613`; only Alpha is
  running, total burn is `$0.303/hr`, and mounted disk has 72GB free.
- **Checkpoint 39,000 PASSED; substantial new run best:** 39,000 finite/consecutive rows cover
  638,976,000 tokens (63.8967%); p10/median is 3,746.2924/3,880.1787 tok/s; all 391 allocator
  samples report exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.2383468/0.2667200 and held ArrayBuffers at 7,292–7,293MB, external at 7,294MB, and RSS
  8,466–8,541MB. Train/held-out loss is 3.2750547/3.1773408, improving 0.1035915 from step 38,500
  and setting a new run best by 0.0870612 over checkpoint 37,000. Exact metrics `f34dbdeb…` and
  692,528,817-byte checkpoint `7f78da25…` match remote/mounted; native audit `ec06cd64…` passed all
  114 tensors / 57,688,576 elements finite/nonzero. Steps 39,001–39,050 returned exactly to
  7,292/7,294MB buffers and 8,530MB RSS. Retention is 37k/38k/39k both sides. Balance
  `$39.6626626648`; only Alpha is running, total burn is `$0.303/hr`, and mounted disk has 71GB free.
- **Step 39,500 PASSED; five-batch variance on watch:** 39,500 finite/consecutive rows cover
  647,168,000 tokens (64.7159%); p10/median is 3,747.1650/3,881.3546 tok/s; all 396 allocator
  samples report exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.2490023/0.2657663 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,466–8,543MB. Train/held-out loss is 3.2509217/3.2830912, +0.1057504 from the unusually sharp
  checkpoint-39,000 best but only +0.0021588 from step 38,500. This is within the established
  five-batch variance, so checkpoint 40,000 is the discriminator and no intervention is justified.
  Exact remote/mounted metrics match at `f7396b0e…`; the guard remains active with zero restarts.
  Balance is `$39.4938992721`; only Alpha is running, total burn is `$0.303/hr`, and mounted disk has
  71GB free.
- **Checkpoint 40,000 PASSED; wobble resolved to another new run best:** 40,000
  finite/consecutive rows cover 655,360,000 tokens (65.5351%); p10/median is
  3,747.9657/3,882.3791 tok/s; all 401 allocator samples report exactly 34 slabs/zero overflow.
  The last 500 rows averaged loss/gradient norm 3.2550314/0.2709435 and held
  ArrayBuffers/external exactly 7,292/7,294MB, with RSS 8,530–8,542MB. Train/held-out loss is
  3.3543744/3.1690485, improving 0.1140427 from step 39,500 and setting a new run best by 0.0082923
  over checkpoint 39,000. Exact metrics `e83589fd…` and 692,528,817-byte checkpoint `e0f176cb…`
  match remote/mounted; native audit `5f48bcd8…` passed all 114 tensors / 57,688,576 elements
  finite/nonzero. Steps 40,001–40,050 returned exactly to 7,292/7,294MB buffers and 8,542MB RSS.
  Retention is 38k/39k/40k both sides. Balance `$39.325022885`; only Alpha is running, total burn is
  `$0.303/hr`, and mounted disk has 70GB free.
- **Step 40,500 PASSED; five-batch variance on watch:** 40,500 finite/consecutive rows cover
  663,552,000 tokens (66.3543%); p10/median is 3,748.6685/3,883.2726 tok/s; all 406 allocator
  samples report exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.2557079/0.2689958 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,469–8,543MB. Train/held-out loss is 3.1113625/3.2990192, +0.1299707 from the unusually sharp
  checkpoint-40,000 best and +0.0159281 from the prior high step-39,500 window. This remains
  established five-batch variance, so checkpoint 41,000 is the discriminator and no intervention is
  justified. Exact remote/mounted metrics match at `94f2cd00…`; the guard remains active with zero
  restarts. Balance is `$39.1802612462`; only Alpha is running, total burn is `$0.303/hr`, and
  mounted disk has 70GB free.
- **Checkpoint 41,000 PASSED; elevated validation persists but hard gates are green:** 41,000
  finite/consecutive rows cover 671,744,000 tokens (67.1735%); p10/median is
  3,749.5583/3,884.2409 tok/s; all 411 allocator samples report exactly 34 slabs/zero overflow.
  The last 500 rows averaged loss/gradient norm 3.2326558/0.2729599 and held ArrayBuffers at
  7,292–7,293MB, external at 7,294MB, and RSS 8,466–8,542MB. Train/held-out loss is
  3.0748420/3.3072725, +0.0082532 from step 40,500 and +0.1382240 from the sharp checkpoint-40,000
  best. Two elevated windows are explicit, but remain within earlier recovered oscillation and no
  hard stop fired. Exact metrics `7510063a…` and 692,528,817-byte checkpoint `1e560e77…` match
  remote/mounted; native audit `a13ffedc…` passed all 114 tensors / 57,688,576 elements
  finite/nonzero. Steps 41,001–41,050 returned exactly to 7,292/7,294MB buffers and 8,531MB RSS.
  Retention is 39k/40k/41k both sides. Balance `$38.9872020943`; only Alpha is running, total burn
  is `$0.303/hr`, and mounted disk has 70GB free.
- **Step 41,500 PASSED; elevated validation recovered materially:** 41,500 finite/consecutive rows
  cover 679,936,000 tokens (67.9927%); p10/median is 3,750.2319/3,884.9894 tok/s; all 416 allocator
  samples report exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.2260098/0.2746605 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,490–8,553MB. Train/held-out loss is 3.2313604/3.2499305, improving 0.0573420 from checkpoint
  41,000 and leaving the read 0.0808820 above the sharp checkpoint-40,000 best. This resolves the
  two-window elevated trend without pretending one five-batch read is a new optimum. Exact
  remote/mounted metrics match at `87d235f5…`; the trainer and guard remain healthy with zero guard
  restarts. Balance is `$38.8425750831`; only Alpha is running, total burn is `$0.303/hr`, and
  mounted disk has 70GB free.
- **Checkpoint 42,000 PASSED; validation recovery continued and save/memory gate is clean:** 42,000
  finite/consecutive rows cover 688,128,000 tokens (68.8118%); p10/median is
  3,751.0408/3,885.8204 tok/s; all 421 allocator samples report exactly 34 slabs/zero overflow. The
  last 500 rows averaged loss/gradient norm 3.2156622/0.2770711 and held RSS/ArrayBuffers/external
  exactly 8,553/7,292/7,294MB. Train/held-out loss is 3.2169719/3.2275744, improving 0.0223560 from
  step 41,500 and remaining 0.0585259 above the checkpoint-40,000 best. Exact metrics `c4814cca…`
  and 692,528,817-byte checkpoint `b5354669…` match remote/mounted; native audit `66c47c81…` passed
  all 114 tensors / 57,688,576 elements finite/nonzero. Steps 42,001–42,050 returned exactly to
  7,292/7,294MB buffers and 8,553MB RSS. Retention is 40k/41k/42k both sides. Balance
  `$38.6496333868`; only Alpha is running, total burn is `$0.303/hr`, and mounted disk has 70GB free.
- **Step 42,500 PASSED; modest five-batch variance on watch:** 42,500 finite/consecutive rows cover
  696,320,000 tokens (69.6310%); p10/median is 3,751.6747/3,886.5458 tok/s; all 426 allocator
  samples report exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.2126758/0.2811326 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,468–8,553MB. Train/held-out loss is 3.2306371/3.2531915, +0.0256171 from checkpoint 42,000 and
  +0.0841430 from the sharp checkpoint-40,000 best. This is modest established five-batch variance;
  checkpoint 43,000 is the discriminator. Exact remote/mounted metrics match at `e77ea69d…`; the
  trainer and guard remain healthy with zero guard restarts. Balance is `$38.5049024867`; only Alpha
  is running, total burn is `$0.303/hr`, and mounted disk has 70GB free.
- **Checkpoint 43,000 PASSED; validation wobble recovered and save/memory gate is clean:** 43,000
  finite/consecutive rows cover 704,512,000 tokens (70.4502%); p10/median is
  3,752.0443/3,886.8866 tok/s; all 431 allocator samples report exactly 34 slabs/zero overflow. The
  last 500 rows averaged loss/gradient norm 3.2033633/0.2803718 and held ArrayBuffers/external
  exactly 7,292/7,294MB, with RSS 8,468–8,542MB. Train/held-out loss is 3.2085137/3.2101865,
  improving 0.0430050 from step 42,500 and remaining only 0.0411380 above the checkpoint-40,000
  best. Exact metrics `1f08118e…` and 692,528,817-byte checkpoint `3da69bcb…` match remote/mounted;
  native audit `93a28428…` passed all 114 tensors / 57,688,576 elements finite/nonzero. Steps
  43,001–43,050 returned exactly to 7,292/7,294MB buffers and 8,539–8,540MB RSS. Retention is
  41k/42k/43k both sides. Balance `$38.3119838902`; only Alpha is running, total burn is
  `$0.303/hr`, and mounted disk has 69GB free.
- **Step 43,500 PASSED; validation improvement continued:** 43,500 finite/consecutive rows cover
  712,704,000 tokens (71.2694%); p10/median is 3,752.3466/3,886.9284 tok/s; all 436 allocator
  samples report exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.1912802/0.2818282 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,471–8,544MB. Train/held-out loss is 3.0849869/3.2006689, improving 0.0095176 from checkpoint
  43,000 and remaining only 0.0316204 above the checkpoint-40,000 best. Exact remote/mounted metrics
  match at `66ed2336…`; the trainer and guard remain healthy with zero guard restarts. Balance is
  `$38.167213268`; only Alpha is running, total burn is `$0.303/hr`, and mounted disk has 69GB free.
- **Checkpoint 44,000 PASSED; five-batch validation wobble on watch while hard gates stay clean:**
  44,000 finite/consecutive rows cover 720,896,000 tokens (72.0886%); p10/median is
  3,752.5666/3,886.8866 tok/s; all 441 allocator samples report exactly 34 slabs/zero overflow. The
  last 500 rows averaged loss/gradient norm 3.2118540/0.2841877 and held ArrayBuffers/external
  exactly 7,292/7,294MB, with RSS 8,468–8,543MB. Train/held-out loss is 3.1902018/3.2532575,
  +0.0525887 from step 43,500 and +0.0842090 from the checkpoint-40,000 best. This is one five-batch
  wobble after two improving windows; step 44,500 is the discriminator. Exact metrics `e1db3751…`
  and 692,528,817-byte checkpoint `a64189e6…` match remote/mounted; native audit `c9d89867…` passed
  all 114 tensors / 57,688,576 elements finite/nonzero. Steps 44,001–44,050 returned exactly to
  7,292/7,294MB buffers and 8,543MB RSS. Retention is 42k/43k/44k both sides. Balance
  `$37.9742922272`; only Alpha is running, total burn is `$0.303/hr`, and mounted disk has 69GB free.
- **Step 44,500 PASSED; elevated validation persists but hard gates remain green:** 44,500
  finite/consecutive rows cover 729,088,000 tokens (72.9078%); p10/median is
  3,752.2116/3,886.4798 tok/s; all 446 allocator samples report exactly 34 slabs/zero overflow. The
  last 500 rows averaged loss/gradient norm 3.2248817/0.2865226 and held ArrayBuffers/external
  exactly 7,292/7,294MB, with RSS 8,469–8,545MB. Train/held-out loss is 3.2194901/3.2662046,
  +0.0129471 from checkpoint 44,000 and +0.0971561 from the checkpoint-40,000 best. Two elevated
  windows are explicit, but this pattern has recovered before and no hard stop fired; checkpoint
  45,000 is the discriminator. Exact remote/mounted metrics match at `56436775…`; the trainer and
  guard remain healthy with zero guard restarts. Balance is `$37.8295331549`; only Alpha is running,
  total burn is `$0.303/hr`, and mounted disk has 69GB free.
- **Checkpoint 45,000 PASSED; elevated validation eased but remains on watch:** 45,000
  finite/consecutive rows cover 737,280,000 tokens (73.7270%); p10/median is
  3,751.7456/3,885.8743 tok/s; all 451 allocator samples report exactly 34 slabs/zero overflow. The
  last 500 rows averaged loss/gradient norm 3.2021399/0.2907239 and held
  RSS/ArrayBuffers/external at 8,543–8,544/7,292/7,294MB. Train/held-out loss is
  3.2578907/3.2536933, improving 0.0125113 from step 44,500 but remaining 0.0846448 above the
  checkpoint-40,000 best. The elevated phase is easing, not resolved; continue the aligned watch.
  Exact metrics `9e57f4e1…` and 692,528,817-byte checkpoint `dd8852f0…` match remote/mounted; native
  audit `372487d9…` passed all 114 tensors / 57,688,576 elements finite/nonzero. Steps
  45,001–45,050 returned exactly to 7,292/7,294MB buffers and 8,544MB RSS. Retention is
  43k/44k/45k both sides. Balance `$37.6365491698`; only Alpha is running, total burn is
  `$0.303/hr`, and mounted disk has 69GB free.
- **Step 45,500 PASSED; elevated phase resolved to a substantial new run best:** 45,500
  finite/consecutive rows cover 745,472,000 tokens (74.5462%); p10/median is
  3,751.4171/3,885.4225 tok/s; all 456 allocator samples report exactly 34 slabs/zero overflow. The
  last 500 rows averaged loss/gradient norm 3.1922874/0.2969672 and held ArrayBuffers/external
  exactly 7,292/7,294MB, with RSS 8,472–8,545MB. Train/held-out loss is 3.1899173/3.1329673,
  improving 0.1207261 from checkpoint 45,000 and setting a new run best by 0.0360812 over checkpoint
  40,000. Exact remote/mounted metrics match at `9bd00c17…`; the trainer and guard remain healthy
  with zero guard restarts. Balance is `$37.4677396994`; only Alpha is running, total burn is
  `$0.303/hr`, and mounted disk has 69GB free.
- **Checkpoint 46,000 PASSED; one-window variance from the sharp new best, hard gates clean:**
  46,000 finite/consecutive rows cover 753,664,000 tokens (75.3654%); p10/median is
  3,750.7619/3,884.4729 tok/s; all 461 allocator samples report exactly 34 slabs/zero overflow. The
  last 500 rows averaged loss/gradient norm 3.2024244/0.2999512 and held
  RSS/ArrayBuffers/external at 8,543–8,544/7,292/7,294MB. Train/held-out loss is
  3.2123423/3.2270148, +0.0940475 from the unusually sharp step-45,500 best but still 0.0266785
  better than checkpoint 45,000. This is one-window variance under continued watch. Exact metrics
  `b42b3010…` and 692,528,817-byte checkpoint `1ba70b29…` match remote/mounted; native audit
  `d2d2f123…` passed all 114 tensors / 57,688,576 elements finite/nonzero. Steps 46,001–46,050
  returned exactly to 7,292/7,294MB buffers and 8,544MB RSS. Retention is 44k/45k/46k both sides.
  Balance `$37.2747729474`; only Alpha is running, total burn is `$0.303/hr`, and mounted disk has
  69GB free.
- **Step 46,500 PASSED; post-best validation wobble recovered materially:** 46,500
  finite/consecutive rows cover 761,856,000 tokens (76.1845%); p10/median is
  3,750.0489/3,883.8174 tok/s; all 466 allocator samples report exactly 34 slabs/zero overflow. The
  last 500 rows averaged loss/gradient norm 3.2058733/0.2950835 and held ArrayBuffers/external
  exactly 7,292/7,294MB, with RSS 8,472–8,545MB. Train/held-out loss is 3.1847329/3.1806306,
  improving 0.0463842 from checkpoint 46,000 and remaining only 0.0476633 above the sharp step-45,500
  best. Exact remote/mounted metrics match at `5ddd24a5…`; the trainer and guard remain healthy with
  zero guard restarts. Balance is `$37.1299918752`; only Alpha is running, total burn is `$0.303/hr`,
  and mounted disk has 69GB free.
- **Checkpoint 47,000 PASSED; validation recovery continued and save/memory gate stayed clean:**
  47,000 finite/consecutive rows cover 770,048,000 tokens (77.0037%); p10/median is
  3,749.6407/3,883.1745 tok/s; all 471 allocator samples report exactly 34 slabs/zero overflow. The
  last 500 rows averaged loss/gradient norm 3.1795377/0.2960260 and held ArrayBuffers/external
  exactly 7,292/7,294MB, with RSS exactly 8,545MB. Train/held-out loss is 3.1541908/3.1714686,
  improving 0.0091619 from step 46,500 and remaining 0.0385014 above the sharp step-45,500 best.
  Exact metrics `b486260b…` and 692,528,817-byte checkpoint `5c1219a5…` match remote/mounted;
  native audit `e5243324…` passed all 114 tensors / 57,688,576 elements finite/nonzero. Steps
  47,001–47,050 returned exactly to 7,292/7,294MB buffers and 8,545MB RSS. Retention is
  45k/46k/47k both sides. Balance is `$36.9370448011`; only Alpha is running, total burn is
  `$0.303/hr`, and mounted disk has 69GB free.
- **Step 47,500 PASSED; small one-window validation variance, hard gates clean:** 47,500
  finite/consecutive rows cover 778,240,000 tokens (77.8229%); p10/median is
  3,749.0308/3,882.5621 tok/s; all 476 allocator samples report exactly 34 slabs/zero overflow. The
  last 500 rows averaged loss/gradient norm 3.1798035/0.3013766 and held ArrayBuffers/external
  exactly 7,292/7,294MB, with RSS 8,472–8,555MB. Train/held-out loss is 3.1647832/3.1811959,
  +0.0097273 from checkpoint 47,000 and only +0.0482286 above the sharp step-45,500 best. Exact
  remote/mounted metrics match at `af2c0a38…`; the trainer and guard remain healthy with zero guard
  restarts. Balance is `$36.7923223344`; only Alpha is running, total burn is `$0.303/hr`, and
  mounted disk has 69GB free.
- **Checkpoint 48,000 PASSED; sharp one-window validation spike is explicitly on watch:** 48,000
  finite/consecutive rows cover 786,432,000 tokens (78.6421%); p10/median is
  3,748.7797/3,881.9792 tok/s; all 481 allocator samples report exactly 34 slabs/zero overflow. The
  last 500 rows averaged loss/gradient norm 3.1739363/0.3014846 and held ArrayBuffers/external
  exactly 7,292/7,294MB, with RSS 8,471–8,544MB. Train/held-out loss is 3.1994004/3.3201346,
  +0.1389387 from step 47,500 and +0.1871673 above the sharp step-45,500 best. This is a serious
  one-window quality wobble, but not yet a corruption stop: train loss, gradients, weights, allocator,
  and memory all remain clean; step 48,500 is the discriminator. Exact metrics `356609b5…` and
  692,528,817-byte checkpoint `bf298cd4…` match remote/mounted; native audit `9cf6692b…` passed all
  114 tensors / 57,688,576 elements finite/nonzero. Steps 48,001–48,050 returned exactly to
  7,292/7,294MB buffers and 8,544MB RSS. Retention is 46k/47k/48k both sides. Balance is
  `$36.599386077`; only Alpha is running, total burn is `$0.303/hr`, and mounted disk has 69GB free.
- **Step 48,500 PASSED; checkpoint-48,000 validation spike resolved materially:** 48,500
  finite/consecutive rows cover 794,624,000 tokens (79.4613%); p10/median is
  3,748.1689/3,881.2362 tok/s; all 486 allocator samples report exactly 34 slabs/zero overflow. The
  last 500 rows averaged loss/gradient norm 3.1566103/0.3011098 and held ArrayBuffers/external
  exactly 7,292/7,294MB, with RSS 8,491–8,555MB. Train/held-out loss is 3.3287439/3.2004976;
  held-out recovered 0.1196370 from checkpoint 48,000, sits only +0.0193017 above step 47,500, and
  +0.0675303 above the sharp step-45,500 best. Exact remote/mounted metrics match at `d9170d63…`;
  the trainer and guard remain healthy with zero guard restarts. Balance is `$36.4305807622`; only
  Alpha is running, total burn is `$0.303/hr`, and mounted disk has 68GB free.
- **Checkpoint 49,000 PASSED; moderate validation variance remains on aligned watch:** 49,000
  finite/consecutive rows cover 802,816,000 tokens (80.2805%); p10/median is
  3,747.2236/3,880.4696 tok/s; all 491 allocator samples report exactly 34 slabs/zero overflow. The
  last 500 rows averaged loss/gradient norm 3.1829064/0.3071070 and held ArrayBuffers/external
  exactly 7,292/7,294MB, with RSS 8,470–8,555MB. Train/held-out loss is 3.4289570/3.2353356,
  +0.0348381 from step 48,500 and +0.1023684 above the sharp step-45,500 best. Exact metrics
  `edfdc19b…` and 692,528,817-byte checkpoint `ce31be53…` match remote/mounted; native audit
  `5d3b64c6…` passed all 114 tensors / 57,688,576 elements finite/nonzero. Steps 49,001–49,050
  returned exactly to 7,292/7,294MB buffers and 8,541MB RSS. Retention is 47k/48k/49k both sides.
  Balance is `$36.2375836381`; only Alpha is running, total burn is `$0.303/hr`, and mounted disk
  has 68GB free.
- **Step 49,500 PASSED; renewed two-window elevated validation phase on watch:** 49,500
  finite/consecutive rows cover 811,008,000 tokens (81.0997%); p10/median is
  3,746.8122/3,880.0823 tok/s; all 496 allocator samples report exactly 34 slabs/zero overflow. The
  last 500 rows averaged loss/gradient norm 3.1433593/0.3170738 and held ArrayBuffers/external
  exactly 7,292/7,294MB, with RSS 8,488–8,552MB. Train/held-out loss is 3.2038922/3.2817139;
  held-out is +0.0463783 from checkpoint 49,000, +0.0812163 from step 48,500, and +0.1487466 above
  the sharp step-45,500 best, while remaining below the transient checkpoint-48,000 spike. Exact
  remote/mounted metrics match at `d212e295…`; the trainer and guard remain healthy with zero guard
  restarts. Balance is `$36.0929247881`; only Alpha is running, total burn is `$0.303/hr`, and
  mounted disk has 68GB free. Checkpoint 50,000 is the discriminator.
- **Checkpoint 50,000 PASSED; renewed elevated validation phase resolved:** 50,000
  finite/consecutive rows cover 819,200,000 tokens (81.9189%); p10/median is
  3,746.7980/3,879.7418 tok/s; all 501 allocator samples report exactly 34 slabs/zero overflow. The
  last 500 rows averaged loss/gradient norm 3.1791822/0.3079951 and held ArrayBuffers/external
  exactly 7,292/7,294MB, with RSS exactly 8,552MB. Train/held-out loss is 3.1976542/3.2035347;
  held-out recovered 0.0781792 from step 49,500, sits only +0.0030372 above step 48,500, and
  +0.0705675 above the sharp step-45,500 best. Exact metrics `e5238d36…` and 692,528,817-byte
  checkpoint `4bdefac1…` match remote/mounted; native audit `9d6bf76b…` passed all 114 tensors /
  57,688,576 elements finite/nonzero. Steps 50,001–50,050 returned exactly to 7,292/7,294MB
  buffers and 8,553MB RSS. Retention is 48k/49k/50k both sides. Balance is `$35.8998268919`; only
  Alpha is running, total burn is `$0.303/hr`, and mounted disk has 68GB free.
- **Step 50,500 PASSED; post-recovery validation stabilized:** 50,500 finite/consecutive rows
  cover 827,392,000 tokens (82.7381%); p10/median is 3,746.7541/3,879.4267 tok/s; all 506
  allocator samples report exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient
  norm 3.1534839/0.3119011 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,473–8,553MB. Train/held-out loss is 3.1614351/3.2016599; held-out improved 0.0018748 from
  checkpoint 50,000, sits only +0.0011623 above step 48,500, and +0.0686926 above the sharp
  step-45,500 best. Exact remote/mounted metrics match at `1327de1f…`; the trainer and guard remain
  healthy with zero guard restarts. Balance is `$35.7310038549`; only Alpha is running, total burn
  is `$0.303/hr`, and mounted disk has 68GB free.
- **Checkpoint 51,000 PASSED; moderate one-window validation wobble on aligned watch:** 51,000
  finite/consecutive rows cover 835,584,000 tokens (83.5572%); p10/median is
  3,747.0509/3,879.4109 tok/s; all 511 allocator samples report exactly 34 slabs/zero overflow. The
  last 500 rows averaged loss/gradient norm 3.1456441/0.3094241 and held ArrayBuffers/external
  exactly 7,292/7,294MB, with RSS 8,470–8,545MB. Train/held-out loss is 3.2150030/3.2513193,
  +0.0496593 from step 50,500 and +0.1183520 above the sharp step-45,500 best. Exact metrics
  `57475440…` and 692,528,817-byte checkpoint `e5aeb795…` match remote/mounted; native audit
  `3047e2b1…` passed all 114 tensors / 57,688,576 elements finite/nonzero. Steps 51,001–51,050
  returned exactly to 7,292/7,294MB buffers and 8,542MB RSS. Retention is 49k/50k/51k both sides.
  Balance is `$35.5621934067`; only Alpha is running, total burn is `$0.303/hr`, and mounted disk
  has 67GB free.
- **Step 51,500 PASSED; validation recovered to near-run-best:** 51,500 finite/consecutive rows
  cover 843,776,000 tokens (84.3764%); p10/median is 3,747.4460/3,879.6410 tok/s; all 516
  allocator samples report exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient
  norm 3.1485280/0.3142221 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,471–8,546MB. Train/held-out loss is 3.1850824/3.1555889; held-out improved 0.0957304 from
  checkpoint 51,000 and is only +0.0226216 above the sharp step-45,500 best. Exact remote/mounted
  metrics match at `e841965c…`; the trainer and guard remain healthy with zero guard restarts.
  Balance is `$35.3933533196`; only Alpha is running, total burn is `$0.303/hr`, and mounted disk
  has 67GB free.
- **Checkpoint 52,000 PASSED; NEW RUN BEST:** 52,000 finite/consecutive rows cover 851,968,000
  tokens (85.1956%); p10/median is 3,747.7495/3,879.6506 tok/s; all 521 allocator samples report
  exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.1574955/0.3157661 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,527–8,545MB. Train/held-out loss is 3.0159855/3.1257953; held-out improved 0.0297936 from step
  51,500 and set a new run best by 0.0071720 over step 45,500. Exact metrics `3bc0a40c…` and
  692,528,817-byte checkpoint `9a2c585d…` match remote/mounted; native audit `e7fd7fa9…` passed all
  114 tensors / 57,688,576 elements finite/nonzero. Steps 52,001–52,050 returned exactly to
  7,292/7,294MB buffers and 8,545MB RSS. Retention is 50k/51k/52k both sides. Balance is
  `$35.2244717991`; only Alpha is running, total burn is `$0.303/hr`, and mounted disk has 66GB free.
- **Step 52,500 PASSED; NEW RUN BEST again:** 52,500 finite/consecutive rows cover 860,160,000
  tokens (86.0148%); p10/median is 3,748.1008/3,879.6693 tok/s; all 526 allocator samples report
  exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.1478990/0.3178515 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,470–8,547MB. Train/held-out loss is 3.1507206/3.1009022; held-out improved 0.0248931 from
  checkpoint 52,000 and is the new run best. Exact remote/mounted metrics match at `8972c1b3…`;
  the trainer and guard remain healthy with zero guard restarts. Balance is `$35.0556843897`; only
  Alpha is running, total burn is `$0.303/hr`, and mounted disk has 66GB free.
- **Checkpoint 53,000 PASSED; moderate one-window rebound from the new best:** 53,000
  finite/consecutive rows cover 868,352,000 tokens (86.8340%); p10/median is
  3,748.3551/3,879.7285 tok/s; all 531 allocator samples report exactly 34 slabs/zero overflow. The
  last 500 rows averaged loss/gradient norm 3.1392291/0.3207649 and held ArrayBuffers/external
  exactly 7,292/7,294MB, with RSS 8,534–8,546MB. Train/held-out loss is
  3.1202853/3.1644977, +0.0635955 from the exceptional step-52,500 best. Exact metrics
  `562e8403…` and 692,528,817-byte checkpoint `b2cb6865…` match remote/mounted; native audit
  `c56206c9…` passed all 114 tensors / 57,688,576 elements finite/nonzero. Steps 53,001–53,050
  returned exactly to 7,292/7,294MB buffers and held RSS at 8,546MB. Retention is 51k/52k/53k on
  both sides. Balance is `$34.8867962693`; only Alpha is running, total burn is `$0.303/hr`, and
  mounted disk has 66GB free.
- **Step 53,500 PASSED; two-window elevated validation phase on watch:** 53,500 finite/consecutive
  rows cover 876,544,000 tokens (87.6532%); p10/median is 3,748.5257/3,879.8256 tok/s; all 536
  allocator samples report exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient
  norm 3.1241615/0.3207876 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,474–8,548MB. Train/held-out loss is 3.2312453/3.2067587; held-out is +0.0422610 from checkpoint
  53,000 and +0.1058565 from the sharp step-52,500 best, while remaining 0.1133759 below the earlier
  checkpoint-48,000 spike. Exact remote/mounted metrics match at `41f392fa…`; the trainer and guard
  remain healthy with zero guard restarts. Balance is `$34.7179635157`; only Alpha is running, total
  burn is `$0.303/hr`, and mounted disk has 65GB free.
- **Checkpoint 54,000 PASSED; elevated validation phase eased but remains on watch:** 54,000
  finite/consecutive rows cover 884,736,000 tokens (88.4724%); p10/median is
  3,748.7694/3,879.6273 tok/s; all 541 allocator samples report exactly 34 slabs/zero overflow. The
  last 500 rows averaged loss/gradient norm 3.1495170/0.3400912 and held ArrayBuffers/external
  exactly 7,292/7,294MB with RSS exactly 8,546MB. Train/held-out loss is
  3.2111425/3.1928943; held-out improved 0.0138644 from step 53,500 but remains +0.0919921 above
  the sharp step-52,500 best. Exact metrics `34a0ab36…` and 692,528,817-byte checkpoint
  `3fb1913a…` match remote/mounted; native audit `6fc1ea2b…` passed all 114 tensors / 57,688,576
  elements finite/nonzero. Steps 54,001–54,050 returned exactly to 7,292/7,294MB buffers and held
  RSS at 8,546MB. Retention is 52k/53k/54k on both sides. Balance is `$34.5250273194`; only Alpha
  is running, total burn is `$0.303/hr`, and mounted disk has 65GB free.
- **Step 54,500 PASSED; elevated validation plateau eased marginally and remains on watch:** 54,500
  finite/consecutive rows cover 892,928,000 tokens (89.2916%); p10/median is
  3,748.7665/3,879.5157 tok/s; all 546 allocator samples report exactly 34 slabs/zero overflow. The
  last 500 rows averaged loss/gradient norm 3.1415791/0.3262672; ArrayBuffers/external varied only
  1MB at 7,292–7,293/7,294–7,295MB and RSS stayed within 8,471–8,546MB. Train/held-out loss is
  3.0450501/3.1912791; held-out improved 0.0016152 from checkpoint 54,000 and 0.0154796 from step
  53,500, while remaining +0.0903769 above the sharp step-52,500 best. Exact remote/mounted metrics
  match at `500ec2ef…`; the trainer and guard remain healthy with zero guard restarts. Balance is
  `$34.3803019805`; only Alpha is running, total burn is `$0.303/hr`, and mounted disk has 65GB free.
- **Checkpoint 55,000 PASSED; elevated validation plateau materially resolved toward baseline:**
  55,000 finite/consecutive rows cover 901,120,000 tokens (90.1108%); p10/median is
  3,748.6685/3,879.1190 tok/s; all 551 allocator samples report exactly 34 slabs/zero overflow. The
  last 500 rows averaged loss/gradient norm 3.1226775/0.3276708, held ArrayBuffers/external exactly
  7,292/7,294MB, and held RSS within 8,540–8,547MB. Train/held-out loss is
  3.1819386/3.1667840; held-out improved 0.0244951 from step 54,500 and is only +0.0022863 above
  checkpoint 53,000, while remaining +0.0658818 above the sharp step-52,500 best. Exact metrics
  `092e479f…` and 692,528,817-byte checkpoint `95e8cd31…` match remote/mounted; native audit
  `26888e73…` passed all 114 tensors / 57,688,576 elements finite/nonzero. Steps 55,001–55,050
  returned exactly to 7,292/7,294MB buffers and held RSS at 8,547MB. Retention is 53k/54k/55k on
  both sides. Balance is `$34.1873456174`; only Alpha is running, total burn is `$0.303/hr`, and
  mounted disk has 64GB free.
- **Step 55,500 PASSED; elevated validation plateau resolved:** 55,500 finite/consecutive rows cover
  909,312,000 tokens (90.9299%); p10/median is 3,748.7797/3,878.9848 tok/s; all 556 allocator
  samples report exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.1314820/0.3289836 and held ArrayBuffers/external exactly 7,292/7,294MB, with RSS
  8,471–8,548MB. Train/held-out loss is 3.0617723/3.1577666; held-out improved 0.0090174 from
  checkpoint 55,000, 0.0335125 from step 54,500, and is now 0.0067311 better than checkpoint
  53,000, while remaining +0.0568644 above the sharp step-52,500 best. Exact remote/mounted metrics
  match at `f8a382d0…`; the trainer and guard remain healthy with zero guard restarts. Balance is
  `$34.0426110506`; only Alpha is running, total burn is `$0.303/hr`, and mounted disk has 64GB free.
- **Checkpoint 56,000 PASSED; validation recovery continued near run best:** 56,000
  finite/consecutive rows cover 917,504,000 tokens (91.7491%); p10/median is
  3,749.1632/3,879.2154 tok/s; all 561 allocator samples report exactly 34 slabs/zero overflow. The
  last 500 rows averaged loss/gradient norm 3.1271782/0.3319130 and held ArrayBuffers/external
  exactly 7,292/7,294MB, with RSS 8,471–8,547MB. Train/held-out loss is
  3.2305961/3.1364248; held-out improved 0.0213418 from step 55,500 and 0.0303592 from checkpoint
  55,000, leaving only +0.0355226 above the sharp step-52,500 best. Exact metrics `15b11de0…` and
  692,528,817-byte checkpoint `41923a11…` match remote/mounted; native audit `b4b174ba…` passed all
  114 tensors / 57,688,576 elements finite/nonzero. Steps 56,001–56,050 returned exactly to
  7,292/7,294MB buffers and held RSS at 8,547MB. Retention is 54k/55k/56k on both sides. Balance
  is `$33.8496793765`; only Alpha is running, total burn is `$0.303/hr`, and mounted disk has 64GB
  free.
- **Step 56,500 PASSED; held-out returned to within 0.008 of the run best:** 56,500
  finite/consecutive rows cover 925,696,000 tokens (92.5683%); p10/median is
  3,749.6664/3,879.5791 tok/s; all 566 allocator samples report exactly 34 slabs/zero overflow. The
  last 500 rows averaged loss/gradient norm 3.1224420/0.3326293 and held RSS/ArrayBuffers/external
  exactly at 8,547/7,292/7,294MB. Train/held-out loss is 3.2060032/3.1087756; held-out improved
  0.0276492 from checkpoint 56,000 and is only +0.0078734 above the sharp step-52,500 run best.
  Exact remote/mounted metrics match at `adfd1a15…`. Balance is `$33.7049310597`; only Alpha is
  running, total burn is `$0.303/hr`, and mounted disk has 62GB free.
- **Checkpoint 57,000 PASSED; NEW RUN BEST:** 57,000 finite/consecutive rows cover 933,888,000
  tokens (93.3875%); p10/median is 3,750.2319/3,879.8896 tok/s; all 571 allocator samples report
  exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.1198608/0.3377949 and held RSS/ArrayBuffers/external exactly at 8,547/7,292/7,294MB.
  Train/held-out loss is 3.0678167/3.0660259; held-out improved 0.0427497 from step 56,500 and set
  a new run best by 0.0348763 over step 52,500. Exact metrics `f3d0c063…` and 692,528,817-byte
  checkpoint `eae1679e…` match remote/mounted; native audit `d6a277a5…` passed all 114 tensors /
  57,688,576 elements finite/nonzero. Steps 57,001–57,050 remained exactly at the same memory
  baseline. Retention is 55k/56k/57k on both sides. Balance is `$33.5119787299`; only Alpha is
  running, total burn is `$0.303/hr`, and mounted disk has 63GB free.
- **Step 57,500 PASSED; moderate one-window rebound after new best:** 57,500 finite/consecutive rows
  cover 942,080,000 tokens (94.2067%); p10/median is 3,750.7619/3,880.2897 tok/s; all 576
  allocator samples report exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient
  norm 3.0976251/0.3547236; ArrayBuffers/external held exactly at 7,292/7,294MB and RSS stayed
  within 8,474–8,547MB. Train/held-out loss is 3.2421117/3.1012069, +0.0351810 from the new
  checkpoint-57,000 best and only +0.0003047 above the former step-52,500 best. Exact
  remote/mounted metrics match at `b5120435…`. Balance is `$33.3672945799`; only Alpha is running,
  total burn is `$0.303/hr`, and mounted disk has 62GB free.
- **Checkpoint 58,000 PASSED; two-window elevated validation phase on watch:** 58,000
  finite/consecutive rows cover 950,272,000 tokens (95.0259%); p10/median is
  3,751.1933/3,880.8205 tok/s; all 581 allocator samples report exactly 34 slabs/zero overflow. The
  last 500 rows averaged loss/gradient norm 3.0931357/0.3387721; ArrayBuffers/external held exactly
  at 7,292/7,294MB and RSS stayed within 8,535–8,546MB. Train/held-out loss is
  3.1479688/3.1600277, +0.0588208 from step 57,500 and +0.0940019 from the checkpoint-57,000
  best. Exact metrics `5f6bdb84…` and 692,528,817-byte checkpoint `85b949ff…` match
  remote/mounted; native audit `97580626…` passed all 114 tensors / 57,688,576 elements
  finite/nonzero. Steps 58,001–58,050 returned exactly to the 8,547/7,292/7,294MB
  RSS/ArrayBuffers/external baseline. Retention is 56k/57k/58k on both sides. Balance is
  `$33.1743314336`; only Alpha is running, total burn is `$0.303/hr`, and mounted disk has 62GB free.
- **Step 58,500 PASSED; elevated validation phase resolved:** 58,500 finite/consecutive rows cover
  958,464,000 tokens (95.8451%); p10/median is 3,751.6908/3,881.2779 tok/s; all 586 allocator
  samples report exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.1321392/0.3434877; ArrayBuffers/external held exactly at 7,292/7,294MB and RSS stayed within
  8,493–8,556MB. Train/held-out loss is 3.2500243/3.1015048; held-out improved 0.0585229 from
  checkpoint 58,000, is only +0.0006026 above the former step-52,500 best, and remains +0.0354789
  above the checkpoint-57,000 run best. Exact remote/mounted metrics match at `6dbd87f6…`. Balance
  is `$33.0295856225`; only Alpha is running, total burn is `$0.303/hr`, and mounted disk has 62GB
  free.
- **Checkpoint 59,000 PASSED; validation recovery continued:** 59,000 finite/consecutive rows cover
  966,656,000 tokens (96.6643%); p10/median is 3,752.0499/3,881.7374 tok/s; all 591 allocator
  samples report exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.1191765/0.3458788 and held RSS/ArrayBuffers/external exactly at 8,556/7,292/7,294MB.
  Train/held-out loss is 2.8956242/3.0951898; held-out improved 0.0063150 from step 58,500 and is
  only +0.0291639 above the checkpoint-57,000 run best. Exact metrics `17af1cc2…` and
  692,528,817-byte checkpoint `a96b22e2…` match remote/mounted; native audit `4104dcc1…` passed all
  114 tensors / 57,688,576 elements finite/nonzero. Steps 59,001–59,050 held exactly at the
  8,557/7,292/7,294MB RSS/ArrayBuffers/external baseline. Retention is 57k/58k/59k on both sides.
  Balance is `$32.8366250428`; only Alpha is running, total burn is `$0.303/hr`, and mounted disk has
  62GB free.
- **Step 59,500 PASSED; validation returned near run best:** 59,500 finite/consecutive rows cover
  974,848,000 tokens (97.4835%); p10/median is 3,752.3330/3,882.0040 tok/s; all 596 allocator
  samples report exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.0926898/0.3416984; ArrayBuffers/external held exactly at 7,292/7,294MB and RSS stayed within
  8,485–8,557MB. Train/held-out loss is 2.9711595/3.0791674; held-out improved 0.0160223 from
  checkpoint 59,000 and is only +0.0131415 above the checkpoint-57,000 run best. Exact
  remote/mounted metrics match at `3cccaf99…`. Balance is `$32.6918561928`; only Alpha is running,
  total burn is `$0.303/hr`, and mounted disk has 62GB free.
- **Checkpoint 60,000 PASSED; late validation wobble recorded, every hard gate clean:** 60,000
  finite/consecutive rows cover 983,040,000 tokens (98.3026%); p10/median is
  3,752.6690/3,882.1724 tok/s; all 601 allocator samples report exactly 34 slabs/zero overflow. The
  last 500 rows averaged loss/gradient norm 3.1206065/0.3466359; RSS stayed within 8,527–8,556MB
  while ArrayBuffers/external held exactly at 7,292/7,294MB. Train/held-out loss is
  3.1976528/3.2482990, +0.1691316 from step 59,500 and +0.1822731 above the checkpoint-57,000 run
  best. This is an explicit late validation wobble, but the exact metrics `d15f007c…` and
  692,528,817-byte checkpoint `cd124a9c…` match remote/mounted; native audit `b66644dc…` passed all
  114 tensors / 57,688,576 elements finite/nonzero. Steps 60,001–60,050 returned to
  8,534–8,535/7,292/7,294MB RSS/ArrayBuffers/external memory. Retention is 58k/59k/60k both sides.
  Balance is `$32.4748066537`; only Alpha is running, total burn is `$0.303/hr`, and mounted disk has
  62GB free.
- **Step 60,500 PASSED; late wobble resolved to a NEW RUN BEST:** 60,500 finite/consecutive rows
  cover 991,232,000 tokens (99.1218%); p10/median is 3,752.8326/3,882.1851 tok/s; all 606 allocator
  samples report exactly 34 slabs/zero overflow. The last 500 rows averaged loss/gradient norm
  3.1226713/0.3479907; ArrayBuffers/external held exactly at 7,292/7,294MB and RSS stayed within
  8,475–8,549MB. Train/held-out loss is 3.1089549/3.0491906; held-out recovered 0.1991084 from
  checkpoint 60,000 and set a new run best by 0.0168353 over checkpoint 57,000. The final batch's
  pre-clip grad norm 1.278 was correctly clipped to coefficient 0.782; the aligned 500-row mean and
  every finite/system gate remain healthy. Exact remote/mounted metrics match at `c9179f8f…`.
  Balance is `$32.3541754074`; only Alpha is running, total burn is `$0.303/hr`, and mounted disk has
  62GB free.
- **Checkpoint 61,000 PASSED:** 61,000 finite/consecutive rows cover 999,424,000 tokens (99.9410%);
  p10/median is 3,752.9627/3,882.1910 tok/s; all 611 allocator samples report exactly 34 slabs/zero
  overflow. The last 500 rows averaged loss/gradient norm 3.1076607/0.3496741 and held
  RSS/ArrayBuffers/external exactly at 8,547/7,292/7,294MB. Train/held-out loss is
  3.2091236/3.1908423: validation rebounded from the exceptional 60,500 best but remained 0.0574567
  better than checkpoint 60,000. Exact metrics `cf2a2e4c…` and 692,528,817-byte checkpoint
  `8b2872ab…` match remote/mounted; native audit `c1171427…` passed all 114 tensors / 57,688,576
  elements finite/nonzero.
- **FLAGSHIP PRETRAIN TERMINAL PASS:** the immutable `e561f66` run completed all 61,036 rows and
  exactly 1,000,013,824 tokens. Fail-closed analyzer `5d65e518…` passed exact selector/manifest/
  tokenizer/contract binding, p10/median 3,753.1721/3,882.3479 tok/s after warmup, 612 complete
  allocator samples with zero overflow, final/last-100 train loss 2.9974854/3.1011362, final-three
  validation mean 3.1367731, and native scan of all 57,688,576 terminal parameters finite/nonzero.
  Terminal checkpoint is 692,528,817 bytes at `08e14fa9…`; canonical metrics are `7ff9feec…`.
- **Terminal-validation contract bug found and repaired without changing weights:** `e561f66`
  evaluated only cadence multiples while its analyzer correctly required off-cadence terminal 61,036.
  `4c5d1aa` fixes future terminal cadence. A 36-step replay proved that path but produced a different
  checkpoint (`039a260d…`) because Vulkan reductions are not bit-deterministic; it was rejected as
  canonical and preserved as named evidence. `c333bf2` adds a fail-closed eval-only mode; it loaded
  the sealed original `08e14fa9…` checkpoint, executed five validation batches and **zero training
  steps**, measured terminal val loss 3.1702865, and changed only `valLoss` on metric row 61,036.
  Repair evidence is `56e77083…`; original metrics remain preserved at `c383d24b…`.
- **Recovery2 downstream staging is complete and hash-verified:** `/runpod/data/alpha-sft-v2`
  contains the exact SFT corpus/manifest/length-audit/mask-audit at `ffad0a37…`/`e5d034ac…`/
  `1dc89d0f…`/`20c7a45f…`; the deployed tokenizer is `c310343a…`. Frozen manifest/chat/QA are
  `bf6e6ea4…`/`6c463deb…`/`bbbeec57…` under `/runpod/data/frozen-eval-v1`. The pod retained
  7.6GB free after staging.
- **SFT source certified on NVIDIA:** exact `c333bf2` passed all 46/46 GPU-gated tests on the live
  RTX 3090 with zero skipped/failed/todo tests. The complete report is mirrored at
  `/mnt/donto-data/alpha-runs/nvidia-gate-c333bf2-sft-20260727/` (`vitest-report.json` SHA-256
  `6143c60d…`). The pod intentionally remains clean and detached at `c333bf2`; documentation-only
  commits must not advance the SFT source because the LR selector and full-SFT contract bind it.
- **SFT LR pilot 1 (`1e-4`) COMPLETE:** all 2,000 rows are consecutive and finite; the eight aligned
  validations end at 1.8512929 and the final-three mean is 1.8586174. Final/last-100 train loss is
  1.9064003/1.8535267; median post-warmup throughput is 3,804.2311 tok/s; all 21 allocator samples
  are complete through step 2,000 with zero overflow. Terminal checkpoint is 692,528,815 bytes at
  `4f573f39…`; metrics are `11482619…`. The exact remote run is mirrored and hash-verified at
  `/mnt/donto-data/alpha-runs/sft-lr-pilot-1e4-c333bf2-20260727/`.
- **SFT LR sweep COMPLETE; strict selector chose `3e-4`:** all three assistant-only pilots completed
  2,000 consecutive finite rows, eight aligned validations, 21 allocator samples through terminal,
  and zero free-range overflow. Final-three validation means rank `3e-4` 1.7839965, `1e-3`
  1.8391179, `1e-4` 1.8586174. The canonical PASS report is
  `/mnt/donto-data/alpha-runs/sft-lr-sweep-analysis-c333bf2-20260728.json` (SHA-256 `06243d36...`)
  and binds source `c333bf2` plus the exact corpus/manifest/audits/tokenizer/base checkpoint. The
  `1e-3` terminal checkpoint is 692,528,815 bytes at `0fdb23db...`; metrics are `bd98562f...`, and
  both hashes match the sealed mounted mirror. All three pilot guards exited cleanly with no restarts.
- **Full flagship SFT is LIVE** at
  `/workspace/alpha2/runs/flagship-sft-c333bf2-20260728` on pod `gp4m6s8m06bhen`; mounted mirror is
  `/mnt/donto-data/alpha-runs/flagship-sft-c333bf2-20260728`. Its immutable contract passed and binds
  30,322 steps / 496,795,648 padded tokens, one assistant-only epoch, selected `3e-4` to `3e-5`,
  selector SHA `06243d36...`, exact `c333bf2`, and all input hashes. The first recovery gate passed:
  step 1,000 was finite at 3,963 tok/s with held-out loss 1.9429283 (improved from 2.0446098 at step
  500), all allocator telemetry still reported zero overflow, and the 692,528,815-byte checkpoint
  matched remote/mounted at SHA-256 `9149bc73...`. Training resumed cleanly through step 1,050 at
  4,005 tok/s. Guard `alpha2-flagship-sft-guard-20260728.service` uses 60-second verified pulls, a
  1,800-second metric-stall limit, matched keep-three retention, and pod-scoped auto-termination; it
  remains active with zero restarts. **Checkpoint 2,000 PASSED:** all 2,000 rows are consecutive and
  finite; train/held-out loss is 1.8980973/1.7896707, improving held-out by 0.1648041 from step 1,500,
  and throughput is 3,945 tok/s. All 21 allocator samples are complete with zero overflow. The
  692,528,815-byte checkpoint and 2,000-row metrics match remote/mounted at SHA-256 `1878ed9e...` /
  `ace1231c...`; independent audit `477cde8f...` passed all 114 tensors / 57,688,576 parameters finite
  and nonzero and re-bound every SFT input hash. Training resumed through step 2,125. External/
  ArrayBuffer memory returned to the 2,841/2,839MB live baseline at step 2,100, proving that the second
  save did not add another live-buffer plateau. RSS retained allocator pages at 4,373MB (+~314MB from
  pre-save) but moved only 9MB between steps 2,050 and 2,100.
- **Checkpoint 3,000 PASSED and resolved the memory discriminator:** all 3,000 rows are consecutive and
  finite; train/held-out loss is 1.9239737/1.7948370, only +0.0051663 from checkpoint 2,000 after the
  step-2,500 window at 1.8771862. P10/median post-warmup throughput is 3,801.65/3,927.61 tok/s; the
  last 500 rows average loss/gradient norm 1.8181615/0.4922099; all 31 allocator samples are complete
  with zero overflow. The checkpoint, native audit, and exact 3,000-row metrics prefix match
  remote/mounted at SHA-256 `5ad80097...`, `7af019aa...`, and `286052e7...`. The audit passed all
  114 tensors / 57,688,576 parameters finite/nonzero. Pre-save rows 2,501–2,999 held RSS at
  4,389–4,392MB and live external/ArrayBuffer memory at 2,841–2,843/2,839–2,841MB; post-save rows
  through 3,050 held 4,393/2,841–2,842/2,839–2,840MB. There is no repeated checkpoint-sized RSS or
  live-buffer growth. Training resumed through step 3,050 with guard active/zero-restart.
- **Step 3,500 held-out gate PASSED:** all 3,500 rows are consecutive and finite, covering 57,344,000
  padded tokens. Train/held-out loss is 1.6857041/1.8006272; held-out is only +0.0057902 from checkpoint
  3,000 and remains 0.0765590 better than step 2,500. P10/median post-warmup throughput is
  3,793.52/3,920.77 tok/s; the last 500 rows average loss/gradient norm 1.7803503/0.4910335. All 36
  allocator samples are present with zero overflow, and rows 3,001–3,500 hold RSS at 4,393–4,394MB,
  external at 2,841–2,842MB, and ArrayBuffers at 2,839–2,840MB. The exact 3,500-row remote/mounted
  metrics prefix matches at SHA-256 `0dd1d198...`.
- **Checkpoint 4,000 PASSED, including the first live keep-three prune:** all 4,000 rows are consecutive
  and finite, covering 65,536,000 padded tokens. Train/held-out loss is 1.8461738/1.7894577; held-out
  improved 0.0111696 from step 3,500 and set a narrow new run best by 0.0002130 over step 2,000.
  P10/median post-warmup throughput is 3,783.81/3,911.46 tok/s; the last 500 rows average loss/gradient
  norm 1.7611660/0.4883318. All 41 allocator samples are present with zero overflow. Checkpoint, native
  audit, and exact metrics prefix match remote/mounted at `da7e18b2...`, `c0cc3dd2...`, and
  `cd337b07...`; the audit passed all 114 tensors / 57,688,576 parameters finite/nonzero. Only after
  the 4,000 checkpoint's size+SHA mirror proof, the guard pruned checkpoint 1,000 remotely, then wrote
  `delete_committed`/`deleted` records for local SHA `9149bc73...`. Both sides retain 2,000/3,000/4,000.
  Rows 4,001–4,050 returned to 4,395/2,841–2,842/2,839–2,840MB RSS/external/ArrayBuffers, and training
  resumed through step 4,050.
- **Step 4,500 PASSED with one held-out wobble on watch:** 4,500 finite/consecutive rows cover
  73,728,000 padded tokens. P10/median post-warmup throughput is 3,771.38/3,901.98 tok/s; the last 500
  rows average loss/gradient norm 1.7536257/0.4881017; all 46 allocator samples have zero overflow.
  Train/held-out loss is 1.7010769/1.8321369, with held-out +0.0426792 from the narrow checkpoint-4,000
  best. This is one five-batch validation wobble, not a stop signal: steps are finite, gradients and
  throughput are stable, and rows 4,001–4,500 hold RSS exactly 4,395MB with external/ArrayBuffers within
  2,841–2,842/2,839–2,840MB. Exact remote/mounted metrics match at SHA-256 `4dcd2175...`; checkpoint
  5,000 is the discriminator. Balance was `$24.4667641212`; total account burn remained `$0.303/hr`.
- **Checkpoint 5,000 PASSED and resolved the held-out wobble positively:** 5,000 finite/consecutive
  rows cover 81,920,000 padded tokens. Train/held-out loss is 1.7386191/1.7725743; held-out recovered
  0.0595626 from step 4,500 and set a new run best by 0.0168833 over checkpoint 4,000. P10/median
  post-warmup throughput is 3,766.05/3,896.00 tok/s; last-500 loss/gradient norm is
  1.7463370/0.4936785; all 51 allocator samples report zero overflow. Exact checkpoint/native-audit/
  metrics-prefix hashes are `776b111d...` / `e0087402...` / `ee2f1db1...`; the audit passed all 114
  tensors and all 57,688,576 parameters finite/nonzero. The second live keep-three transition pruned
  remote checkpoint 2,000 only after mirror proof, then wrote `delete_committed`/`deleted` records and
  removed the matching local `1878ed9e...`; both sides retain 3,000/4,000/5,000. Rows 5,001–5,050
  returned to 4,395/2,841–2,842/2,839–2,840MB RSS/external/ArrayBuffers. Training resumed through
  5,075; both guards remain active/zero-restart. Balance was `$24.2738337305`; total burn `$0.303/hr`.
- **Step 5,500 PASSED and set another new held-out best:** all 5,500 rows are finite/consecutive and
  cover 90,112,000 padded tokens. Train/held-out loss is 1.8147178/1.7393485; held-out improved
  0.0332258 from checkpoint 5,000 and 0.0501091 over the prior checkpoint-4,000 best. P10/median
  post-warmup throughput is 3,743.68/3,886.21 tok/s; last-500 loss/gradient norm is
  1.7280470/0.4932815; all 56 allocator samples report zero overflow. Rows 5,001–5,500 stayed within
  4,395–4,396/2,841–2,842/2,839–2,840MB RSS/external/ArrayBuffers. Exact remote/mounted metrics
  match at SHA-256 `b9221eef...`; training resumed through 5,525 with both guards still
  active/zero-restart. Balance was `$24.1291633305`; total burn remained `$0.303/hr`.
- **Checkpoint 6,000 PASSED and set a third consecutive new held-out best:** 6,000 finite/consecutive
  rows cover 98,304,000 padded tokens. Train/held-out loss is 1.5110403/1.7103160; held-out improved
  0.0290325 from step 5,500 and 0.0622583 from checkpoint 5,000. P10/median post-warmup throughput is
  3,737.85/3,880.35 tok/s; last-500 loss/gradient norm is 1.7342749/0.4956843; all 61 allocator
  samples report zero overflow. Exact checkpoint/native-audit/metrics-prefix hashes are
  `e06801e3...` / `bac50efb...` / `9857b4e9...`; the audit passed all 114 tensors and all 57,688,576
  parameters finite/nonzero. The third live keep-three transition pruned remote checkpoint 3,000 only
  after mirror proof, then ledgered and removed matching local SHA `5ad80097...`; both sides retain
  4,000/5,000/6,000. Rows 6,001–6,050 returned to the stable
  4,396/2,841–2,842/2,839–2,840MB RSS/external/ArrayBuffer baseline. Training resumed through 6,050;
  both guards remain active/zero-restart. Balance was `$23.9362119787`; burn remained `$0.303/hr`.
- **Step 6,500 PASSED with the largest recent held-out improvement:** all 6,500 rows are
  finite/consecutive and cover 106,496,000 padded tokens. Train/held-out loss is
  1.5530798/1.6268690; held-out improved 0.0834470 from checkpoint 6,000 and set a fourth consecutive
  new run best. P10/median post-warmup throughput is 3,728.22/3,874.93 tok/s; last-500 loss/gradient
  norm is 1.7142389/0.4935493; all 66 allocator samples report zero overflow. Rows 6,001–6,500 held
  RSS exactly 4,396MB and external/ArrayBuffers within 2,841–2,842/2,839–2,840MB. Exact
  remote/mounted metrics match at SHA-256 `5f3bfa80...`; training resumed through 6,525 with both
  guards active/zero-restart. Balance was `$23.7673140194`; burn remained `$0.303/hr`.
- **Checkpoint 7,000 PASSED; one five-batch held-out wobble is on watch:** all 7,000 rows are
  finite/consecutive and cover 114,688,000 padded tokens (23.0855% of the one-epoch run).
  Train/held-out loss is 1.8896970/1.7260751. Held-out is +0.0992061 from the unusually sharp
  step-6,500 best but only +0.0157591 from checkpoint 6,000; every hard gate remains clean and step
  7,500 is the discriminator. P10/median throughput is 3,723.66/3,872.87 tok/s; last-500
  loss/gradient norm is 1.7021976/0.4947595; all 71 allocator samples report exactly 34 temporary
  slabs and zero free-range overflow. Exact checkpoint/native-audit/metrics-prefix hashes match
  remote/mounted at `a60b94c5...` / `3355ab03...` / `b4e7c2c8...`; the independent audit passed all
  114 tensors and all 57,688,576 parameters finite/nonzero. The guard pruned remote checkpoint 4,000
  only after size+SHA mirror proof, then wrote `delete_committed`/`deleted` records and removed the
  exact local `da7e18b2...`; both sides retain 5,000/6,000/7,000. Rows 7,001–7,050 are finite and
  held RSS exactly 4,398MB, external at 2,841–2,842MB, and ArrayBuffers at 2,839–2,840MB, proving a
  clean post-save return to baseline. Training resumed through 7,050 with both guards
  active/zero-restart. Balance was `$23.5742744231`; only Alpha is running and total burn remains
  `$0.303/hr`.
- **Step 7,500 PASSED all hard gates; elevated validation now spans two gates:** 7,500
  finite/consecutive rows cover 122,880,000 padded tokens (24.7345% of the one-epoch run).
  Train/held-out loss is 1.7767525/1.7546257. Held-out is +0.0285507 from checkpoint 7,000,
  +0.0443097 from checkpoint 6,000, and +0.1277568 from the unusually sharp step-6,500 best. This
  is a trend to watch at checkpoint 8,000, not an intervention trigger: gradients, throughput,
  allocator state, and host memory remain stable, and absolute held-out loss is still materially below
  the early-SFT gates. P10/median throughput is 3,713.89/3,866.68 tok/s; last-500 loss/gradient norm
  is 1.7073891/0.4980304; all 76 allocator samples report exactly 34 temporary slabs and zero
  free-range overflow. Rows 7,001–7,500 held RSS at 4,398–4,399MB, external at 2,841–2,842MB,
  and ArrayBuffers at 2,839–2,840MB. The exact 7,500-row metrics prefix matches remote/mounted at
  SHA-256 `2a607fff...`; training resumed through 7,525 and both guards remain active/zero-restart.
  Balance was `$23.4296932453`; only Alpha is running and total burn remains `$0.303/hr`.
- **Checkpoint 8,000 PASSED and resolved the elevated validation trend positively:** 8,000
  finite/consecutive rows cover 131,072,000 padded tokens (26.3835% of the one-epoch run).
  Train/held-out loss is 1.8891588/1.6951937. Held-out improved 0.0594321 from step 7,500,
  0.0308814 from checkpoint 7,000, and 0.0151223 from checkpoint 6,000; it is the run's second-best
  validation, remaining 0.0683247 above the unusually sharp step-6,500 best. P10/median throughput
  is 3,692.59/3,858.35 tok/s; last-500 loss/gradient norm is 1.6788371/0.4984117; all 81 allocator
  samples report exactly 34 temporary slabs and zero free-range overflow. Exact checkpoint/native-
  audit/metrics-prefix hashes match remote/mounted at `b4dfd9bd...` / `703625e2...` /
  `d4981b63...`; the independent audit passed all 114 tensors and all 57,688,576 parameters
  finite/nonzero. The guard pruned remote checkpoint 5,000 only after size+SHA mirror proof, then
  wrote `delete_committed`/`deleted` records and removed exact local SHA `776b111d...`; both sides
  retain 6,000/7,000/8,000. Rows 8,001–8,050 are finite and held RSS exactly 4,405MB, external at
  2,841–2,842MB, and ArrayBuffers at 2,839–2,840MB, proving a clean post-save return to baseline.
  Training resumed through 8,050 with both guards active/zero-restart. Balance was
  `$23.2366289601`; only Alpha is running and total burn remains `$0.303/hr`.
- **Step 8,500 PASSED and set a new validation best:** 8,500 finite/consecutive rows cover
  139,264,000 padded tokens (28.0325% of the one-epoch run). Train/held-out loss is
  1.5012096/1.6183248; held-out improved 0.0768689 from checkpoint 8,000 and 0.0085442 from the
  prior step-6,500 best. P10/median throughput is 3,673.73/3,847.85 tok/s; last-500 loss/gradient
  norm is 1.6793206/0.5022152. All 86 allocator samples report exactly 34 temporary slabs and zero
  free-range overflow. Rows 8,001–8,500 held RSS exactly at 4,405MB, external at
  2,841–2,842MB, and ArrayBuffers at 2,839–2,840MB. The exact metrics prefix matches
  remote/mounted at SHA-256 `886c93fb...`. Both guards remain active/zero-restart; balance was
  `$23.0678590897`, only Alpha is running, and total burn remains `$0.303/hr`.
- **Checkpoint 9,000 PASSED and set another new validation best:** 9,000 finite/consecutive rows
  cover 147,456,000 padded tokens (29.6814% of the one-epoch run). Train/held-out loss is
  1.5297173/1.6034031; held-out improved 0.0149217 from step 8,500 and 0.0234659 from the prior
  step-6,500 best. P10/median throughput is 3,668.81/3,841.06 tok/s; last-500 loss/gradient norm is
  1.6736896/0.4973364. All 91 allocator samples report exactly 34 temporary slabs and zero
  free-range overflow. Exact checkpoint/native-audit/metrics-prefix hashes match remote/mounted at
  `59b8a988...` / `4432adae...` / `5348f3d6...`; the independent audit passed all 114 tensors and
  all 57,688,576 parameters finite/nonzero. The guard pruned remote checkpoint 6,000 only after
  size+SHA mirror proof, then ledgered and removed exact local SHA `e06801e3...`; both sides retain
  7,000/8,000/9,000. Rows 9,001–9,050 are finite and held RSS exactly at 4,410MB, external at
  2,841–2,842MB, and ArrayBuffers at 2,839–2,840MB. Training resumed through 9,050 with both guards
  active/zero-restart. Balance was `$22.8748735153`; only Alpha is running and total burn remains
  `$0.303/hr`.
- **Step 9,500 PASSED hard gates with a sharp one-window validation wobble:** 9,500
  finite/consecutive rows cover 155,648,000 padded tokens (31.3304% of the one-epoch run).
  Train/held-out loss is 1.6326890/1.8127774; held-out is +0.2093743 from checkpoint 9,000 and
  +0.1944526 from step 8,500, while only +0.0581516 above the earlier step-7,500 window.
  P10/median throughput is 3,665.59/3,837.17 tok/s; last-500 loss/gradient norm is
  1.6553709/0.5062537. All 96 allocator samples report exactly 34 temporary slabs and zero
  free-range overflow. Rows 9,001–9,500 held RSS exactly at 4,410MB, external at
  2,841–2,842MB, and ArrayBuffers at 2,839–2,840MB. The exact metrics prefix matches
  remote/mounted at SHA-256 `ed96427a...`. Both guards remain active/zero-restart; balance was
  `$22.7060368504`, only Alpha is running, and total burn remains `$0.303/hr`. All hard numeric,
  memory, and allocator gates are clean, so native-audited checkpoint 10,000 is the discriminator;
  do not intervene from this single five-batch read alone.
- **Checkpoint 10,000 PASSED and resolved the step-9,500 validation wobble:** 10,000
  finite/consecutive rows cover 163,840,000 padded tokens (32.9794% of the one-epoch run).
  Train/held-out loss is 1.8357692/1.6254322; held-out improved 0.1873452 from step 9,500 and sits
  only 0.0220291 above the checkpoint-9,000 best and 0.0071074 above step 8,500. P10/median
  throughput is 3,664.84/3,834.85 tok/s; last-500 loss/gradient norm is 1.6609252/0.5050354.
  All 101 allocator samples report exactly 34 temporary slabs and zero free-range overflow. Exact
  checkpoint/native-audit/metrics-prefix hashes match remote/mounted at `dbc111d0...` /
  `5a2c2e24...` / `785ff316...`; the independent audit passed all 114 tensors and all 57,688,576
  parameters finite/nonzero. The guard pruned remote checkpoint 7,000 only after size+SHA mirror
  proof, then ledgered and removed exact local SHA `a60b94c5...`; both sides retain
  8,000/9,000/10,000. Rows 10,001–10,050 are finite and held RSS exactly at 4,410MB, external at
  2,841–2,842MB, and ArrayBuffers at 2,839–2,840MB. Training resumed through 10,050 with both guards
  active/zero-restart. Balance was `$22.5130707707`; only Alpha is running and total burn remains
  `$0.303/hr`.
- **Step 10,500 PASSED and set a substantial new validation best:** 10,500 finite/consecutive rows
  cover 172,032,000 padded tokens (34.6283% of the one-epoch run). Train/held-out loss is
  1.5839281/1.5531592; held-out improved 0.0722730 from checkpoint 10,000, 0.0502439 from the prior
  checkpoint-9,000 best, and 0.0651656 from step 8,500. P10/median throughput is
  3,664.88/3,832.67 tok/s; last-500 loss/gradient norm is 1.6468749/0.5014553. All 106 allocator
  samples report exactly 34 temporary slabs and zero free-range overflow. Rows 10,001–10,500 held
  RSS exactly at 4,410MB, external at 2,841–2,842MB, and ArrayBuffers at 2,839–2,840MB. The exact
  metrics prefix matches remote/mounted at SHA-256 `8a68ff76...`. Both guards remain
  active/zero-restart; balance was `$22.3684070318`, only Alpha is running, and total burn remains
  `$0.303/hr`.
- **Checkpoint 11,000 PASSED hard gates with a one-window validation wobble:** 11,000
  finite/consecutive rows cover 180,224,000 padded tokens (36.2773% of the one-epoch run).
  Train/held-out loss is 1.5290310/1.7124914; held-out is +0.1593322 from the unusually strong
  step-10,500 read and +0.0870592 from checkpoint 10,000, while remaining 0.1002860 below the prior
  step-9,500 spike. P10/median throughput is 3,664.16/3,830.42 tok/s; last-500 loss/gradient norm
  is 1.6328440/0.5097909. All 111 allocator samples report exactly 34 temporary slabs and zero
  free-range overflow. Exact checkpoint/native-audit/metrics-prefix hashes match remote/mounted at
  `442504c5...` / `0151ca1d...` / `3826f693...`; the independent audit passed all 114 tensors and
  all 57,688,576 parameters finite/nonzero. The guard pruned remote checkpoint 8,000 only after
  size+SHA mirror proof, then ledgered and removed exact local SHA `b4dfd9bd...`; both sides retain
  9,000/10,000/11,000. Rows 11,001–11,050 are finite and held RSS exactly at 4,411MB, external at
  2,841–2,842MB, and ArrayBuffers at 2,839–2,840MB. Training resumed through 11,050 with both guards
  active/zero-restart. Balance was `$22.1754871521`; only Alpha is running and total burn remains
  `$0.303/hr`. All hard gates are clean, so step 11,500 is the discriminator rather than
  intervention from one five-batch read.
- **Step 11,500 PASSED; checkpoint-11,000 validation wobble substantially recovered:** 11,500
  finite/consecutive rows cover 188,416,000 padded tokens (37.9263% of the one-epoch run).
  Train/held-out loss is 1.6845196/1.6471172; held-out improved 0.0653743 from checkpoint 11,000
  and is only +0.0216850 from checkpoint 10,000, though still +0.0939580 from the unusually strong
  step-10,500 best. P10/median throughput is 3,661.35/3,825.44 tok/s; last-500 loss/gradient norm
  is 1.6437370/0.5091599. All 116 allocator samples report exactly 34 temporary slabs and zero
  free-range overflow. Rows 11,001–11,500 held RSS exactly at 4,411MB, external at 2,841–2,842MB,
  and ArrayBuffers at 2,839–2,840MB. The exact metrics prefix matches remote/mounted at SHA-256
  `bf84b946...`. Training resumed through 11,550 at 3,850 tok/s with both guards active and zero
  restarts. Balance was `$22.0066488983`; only Alpha is running and total burn remains `$0.303/hr`.
  Next: native-audited checkpoint 12,000.
- **Checkpoint 12,000 PASSED; validation held flat after the recovery:** 12,000 finite/consecutive
  rows cover 196,608,000 padded tokens (39.5752% of the one-epoch run). Train/held-out loss is
  1.3504689/1.6495684; held-out is only +0.0024513 from step 11,500, remains 0.0629230 better than
  checkpoint 11,000, and is +0.0964092 from the unusually strong step-10,500 best. P10/median is
  3,660.85/3,824.20 tok/s; last-500 loss/gradient norm is 1.6164730/0.5103170. All 121 allocator
  samples report exactly 34 temporary slabs and zero free-range overflow. Rows 11,501–12,000 held
  RSS at 4,411–4,413MB, external at 2,841–2,843MB, and ArrayBuffers at 2,839–2,841MB. Exact
  checkpoint/native-audit/metrics-prefix hashes match remote/mounted at `310b319b...` /
  `64384ea6...` / `39ab674e...`; all 114 tensors and all 57,688,576 parameters passed
  finite/nonzero. The guard pruned remote checkpoint 9,000 only after size+SHA mirror proof, then
  ledgered and removed exact local SHA `59b8a988...`; both sides retain exactly
  10,000/11,000/12,000. The save released all 228 optimizer buffers and training resumed through
  12,050 at 100% GPU. Both guards remain active/zero-restart; balance was `$21.8377576613`, only
  Alpha is running, and total burn remains `$0.303/hr`. Next: step 12,500 held-out validation.
- **Step 12,500 PASSED hard gates; sharp one-window validation wobble:** 12,500 finite/consecutive
  rows cover 204,800,000 padded tokens (41.2242% of the one-epoch run). Train/held-out loss is
  1.6656476/1.7952452; held-out is +0.1456768 from checkpoint 12,000, +0.1481280 from step 11,500,
  and +0.2420860 from the step-10,500 best, while remaining 0.0175322 below the prior step-9,500
  spike. P10/median is 3,662.10/3,824.11 tok/s; last-500 loss/gradient norm is
  1.6185794/0.5072211. All 126 allocator samples report exactly 34 temporary slabs and zero
  free-range overflow. Rows 12,001–12,500 held RSS exactly at 4,414MB, external at 2,841–2,842MB,
  and ArrayBuffers at 2,839–2,840MB. Exact remote/mounted metrics match at `c67c529b...`.
  Training resumed through 12,550 at 3,659 tok/s with both guards active and zero restarts. Balance
  was `$21.6689838187`; only Alpha is running and total burn remains `$0.303/hr`. With execution
  gates clean, native-audited checkpoint 13,000 is the next discriminator rather than intervention
  from one five-batch read.
- **Checkpoint 13,000 PASSED; step-12,500 validation wobble decisively resolved:** 13,000
  finite/consecutive rows cover 212,992,000 padded tokens (42.8732% of the one-epoch run).
  Train/held-out loss is 1.5390872/1.6126210; held-out improved 0.1826242 from step 12,500 and
  0.0369474 from checkpoint 12,000. It is the third-best read of the run and the best since
  step 10,500, only +0.0092179 from checkpoint 9,000 and +0.0594618 from the run best.
  P10/median is 3,664.80/3,825.28 tok/s; last-500 loss/gradient norm is 1.6111589/0.5159380. All
  131 allocator samples report exactly 34 temporary slabs and zero free-range overflow. Rows
  12,501–13,000 held RSS at 4,414–4,415MB, external at 2,841–2,843MB, and ArrayBuffers at
  2,839–2,841MB. Exact checkpoint/native-audit/metrics-prefix hashes match remote/mounted at
  `5c0c404f...` / `c4382db0...` / `d9064535...`; all 114 tensors and all 57,688,576 parameters
  passed finite/nonzero. The guard pruned remote checkpoint 10,000 only after size+SHA mirror proof,
  then ledgered and removed exact local SHA `dbc111d0...`; both sides retain exactly
  11,000/12,000/13,000. The save released all 228 optimizer buffers and training resumed through
  13,050 at 3,797 tok/s. Both guards remain active/zero-restart; balance was `$21.5000564649`, only
  Alpha is running, and total burn remains `$0.303/hr`. Next: step 13,500 held-out validation.
- **Step 13,500 PASSED; second-best validation read of the run:** 13,500 finite/consecutive rows
  cover 221,184,000 padded tokens (44.5221% of the one-epoch run). Train/held-out loss is
  1.8516752/1.5671833; held-out improved 0.0454377 from checkpoint 13,000 and 0.0362198 from
  checkpoint 9,000, leaving it only +0.0140241 from the step-10,500 run best. P10/median is
  3,666.62/3,826.42 tok/s; last-500 loss/gradient norm is 1.6063555/0.5156075. All 136 allocator
  samples report exactly 34 temporary slabs and zero free-range overflow. Rows 13,001–13,500 held
  RSS exactly at 4,415MB, external at 2,841–2,842MB, and ArrayBuffers at 2,839–2,840MB. Exact
  remote/mounted metrics match at `90c810a0...`. Training resumed through 13,550 at 3,909 tok/s
  with both guards active and zero restarts. Balance was `$21.3312723556`; only Alpha is running
  and total burn remains `$0.303/hr`. Next: native-audited checkpoint 14,000.
- **Checkpoint 14,000 PASSED; small validation wobble, still third-best:** 14,000
  finite/consecutive rows cover 229,376,000 padded tokens (46.1711% of the one-epoch run).
  Train/held-out loss is 1.5563986/1.5784316; held-out is +0.0112483 from step 13,500, remains
  0.0341894 better than checkpoint 13,000, and is +0.0252724 from the step-10,500 run best.
  P10/median is 3,669.26/3,828.89 tok/s; last-500 loss/gradient norm is 1.5880093/0.5121112. All
  141 allocator samples report exactly 34 temporary slabs and zero free-range overflow. Rows
  13,501–14,000 held RSS at 4,415–4,416MB, external at 2,841–2,843MB, and ArrayBuffers at
  2,839–2,841MB. Exact checkpoint/native-audit/metrics-prefix hashes match remote/mounted at
  `ad42beef...` / `1141da8f...` / `b59e4113...`; all 114 tensors and all 57,688,576 parameters
  passed finite/nonzero. The guard pruned remote checkpoint 11,000 only after size+SHA mirror proof,
  then ledgered and removed exact local SHA `442504c5...`; both sides retain exactly
  12,000/13,000/14,000. The save released all 228 optimizer buffers and training resumed through
  14,050 at 3,947 tok/s. Both guards remain active/zero-restart; balance was `$21.138261176`, only
  Alpha is running, and total burn remains `$0.303/hr`. Next: step 14,500 held-out validation.
- **Step 14,500 PASSED; modest wobble, still fourth-best validation:** 14,500 finite/consecutive
  rows cover 237,568,000 padded tokens (47.8201% of the one-epoch run). Train/held-out loss is
  1.8460128/1.5946541; held-out is +0.0162225 from checkpoint 14,000 and +0.0274707 from step
  13,500, but remains 0.0179669 better than checkpoint 13,000 and 0.0087490 better than checkpoint
  9,000; it is +0.0414949 from the run best. P10/median is 3,671.46/3,831.03 tok/s; last-500
  loss/gradient norm is 1.5891173/0.5191094. All 146 allocator samples report exactly 34 temporary
  slabs and zero free-range overflow. Rows 14,001–14,500 held RSS exactly at 4,416MB, external at
  2,841–2,842MB, and ArrayBuffers at 2,839–2,840MB. Exact remote/mounted metrics match at
  `dad538dc...`. A transient SSH route loss at 04:27 UTC produced one fail-closed watcher/rsync
  warning; connectivity returned by 04:28, both guards observed the original PID at row 14,450 with
  unchanged RSS, and no restart or data loss occurred. Training resumed through 14,550 at
  3,952 tok/s with both guards active and zero restarts. Balance was `$20.9935850316`; only Alpha
  is running and total burn remains `$0.303/hr`. Next: native-audited checkpoint 15,000.
- **Checkpoint 15,000 PASSED; substantial new validation best:** 15,000 finite/consecutive rows
  cover 245,760,000 padded tokens (49.4690%). Train/held-out loss is 1.5967857/1.4783111; held-out
  improved 0.1163430 from step 14,500 and 0.0748481 from the prior step-10,500 run best. P10/median
  is 3,674.19/3,834.60 tok/s; last-500 loss/gradient norm is 1.5701235/0.5184629. All 151 allocator
  samples report exactly 34 temporary slabs/zero overflow. Rows 14,501–15,000 held RSS exactly at
  4,416MB, external at 2,841–2,843MB, and ArrayBuffers at 2,839–2,841MB. Exact remote/mounted
  checkpoint/native-audit/metrics-prefix hashes are `32962998...` / `46d2d334...` / `3f923dea...`;
  every one of 57,688,576 parameters passed finite/nonzero. Safe retention pruned checkpoint 12,000
  only after mirror proof and ledgered local SHA `310b319b...`; both sides retain exactly
  13,000/14,000/15,000. Training resumed through 15,150 at 3,674 tok/s with both guards active and
  zero restarts. Balance was `$20.7500386482`; only Alpha is running and total burn remains
  `$0.303/hr`. Next: step 15,500 held-out validation.
- **Checkpoint-15,000 ad hoc chat probe remains below bar and was posted honestly to Discord:** eight
  varied, non-frozen prompts were generated with deterministic greedy decoding and a 96-token cap.
  The complete result was 0/8 structural pass, 4/8 nonempty, four immediate-EOS empty replies, three
  repetition loops, and zero role leaks. The encouragement answer was recognizably conversational but
  repetitive; no flattering subset was selected. Mounted input/output hashes are `4c12151b...` /
  `e1ca5e2b...`; all nine webhook posts (provenance summary plus all eight results) were accepted. The
  user-supplied webhook lives only in ignored mode-0600 `.env.discord.local`; reusable tracked
  `scripts/post_discord_progress.sh` reads it and enforces JSON encoding plus a 1,900-byte cap. A
  temporary second 3090 was deleted immediately after source inspection showed the sampler is CPU
  inference; RunPod confirmed only the flagship pod remains running.
- **Step 15,500 PASSED; validation rebounded but remains in the healthy band:** 15,500
  finite/consecutive rows cover 253,952,000 padded tokens (51.1180%). Train/held-out loss is
  1.3062316/1.5797324. Held-out is +0.1014213 from checkpoint 15,000's unusually strong best, but
  only +0.0013008 from checkpoint 14,000, is 0.0149217 better than step 14,500, and remains the
  run's fifth-best read. P10/median is 3,675.60/3,835.96 tok/s; last-500 loss/gradient norm is
  1.5694094/0.5207373. All 156 allocator samples report exactly 34 temporary slabs/zero overflow.
  Rows 15,001–15,500 held RSS at 4,416–4,417MB, external at 2,841–2,842MB, and ArrayBuffers at
  2,839–2,840MB. Exact remote/mounted metrics match at `1be61a49...`; training resumed through
  15,525 at 3,879 tok/s with both guards active and zero restarts. Balance was `$20.6294777408`;
  only Alpha is running and total burn remains `$0.303/hr`. The saved Discord webhook accepted the
  884-byte progress report. Next: native-audited checkpoint 16,000.
- **Checkpoint 16,000 PASSED; second-best validation read of the run:** the exact 16,000-row prefix
  is finite/consecutive and covers 262,144,000 padded tokens (52.7670%). Train/held-out loss is
  1.5142699/1.4864452. Held-out improved 0.0932872 from step 15,500 and 0.0667140 from the old
  step-10,500 best; it is only +0.0081342 from checkpoint 15,000's new best. P10/median is
  3,676.60/3,836.08 tok/s; last-500 loss/gradient norm is 1.5701724/0.5196544. All 161 allocator
  samples report exactly 34 temporary slabs/zero overflow. Rows 15,501–16,000 held RSS exactly at
  4,417MB, external at 2,841–2,843MB, and ArrayBuffers at 2,839–2,841MB. Exact remote/mounted
  checkpoint/native-audit/metrics-prefix hashes are `df81015b...` / `9e1c748c...` / `c18fef4c...`;
  all 57,688,576 parameters passed finite/nonzero. The guard mirror-proved checkpoint 16,000, then
  pruned remote checkpoint 13,000 and two-state ledgered/deleted exact local SHA `5c0c404f...`;
  both sides retain exactly 14,000/15,000/16,000. Training resumed through 16,050 at 3,927 tok/s
  with both guards active/zero-restart. Balance was `$20.4606995593`; only Alpha is running and total
  burn remains `$0.303/hr`. The saved Discord webhook accepted the 904-byte report. Next: step
  16,500 held-out validation.
- **Step 16,500 PASSED; third-best validation result of the run:** the exact 16,500-row prefix is
  finite/consecutive and covers 270,336,000 padded tokens (54.4159%). Train/held-out loss is
  1.6954148/1.5252913. Held-out is +0.0388461 from checkpoint 16,000 and +0.0469802 from the
  checkpoint-15,000 best, while remaining 0.0278679 better than the former step-10,500 best.
  P10/median is 3,676.81/3,835.96 tok/s; last-500 loss/gradient norm is 1.5615495/0.5289970.
  All 166 allocator samples report exactly 34 temporary slabs/zero overflow. Rows 16,001–16,500
  held RSS exactly at 4,417MB, external at 2,841–2,842MB, and ArrayBuffers at 2,839–2,840MB.
  Exact remote/mounted metrics match at `6635c36c...`; training resumed through 16,525 at 3,846
  tok/s with the original PID and 100% GPU utilization. Both guards remain active/zero-restart;
  balance was `$20.2917975666`, only Alpha is running, and total burn remains `$0.303/hr`. The saved
  Discord webhook accepted the 865-byte report.
- **Checkpoint 17,000 PASSED and is byte+SHA mirrored; fourth-best validation:** the
  exact 17,000-row prefix is finite/consecutive and covers 278,528,000 padded tokens (56.0649%).
  Train/held-out loss is 1.2788125/1.5497097; P10/median is 3,677.31/3,836.17 tok/s; last-500
  loss/gradient norm is 1.5493912/0.5237756. All 171 allocator samples remain exactly 34 slabs/zero
  overflow; rows 16,501–17,000 held RSS exactly at 4,417MB and external/ArrayBuffers within
  2,841–2,843/2,839–2,841MB. Native audit passed all 114 tensors and 57,688,576 parameters
  finite/nonzero. Exact checkpoint/audit/remote-metrics-prefix hashes are `67d6fb8a...` /
  `61e3098d...` / `763577a0...`. The RunPod SSH data path briefly degraded below 0.3MB/min even for a
  raw bounded transfer, but the retry recovered at 07:47 UTC and exact local byte+SHA parity passed.
  Only then did the guard prune remote checkpoint 14,000 and two-state ledger/delete exact local SHA
  `ad42beef...`; both sides retain 15,000/16,000/17,000. Training resumed through 17,300 with the
  original PID. Guard/finalizer rsync now has 600/1,800-second hard bounds plus SSH keepalives,
  terminal transfer failure leaves the pod untouched, and failed guard mirroring still performs the
  status check. Syntax, negative-timeout validation, and fake-rsync one-shot control-flow tests passed.
  Both services are active with zero automatic restarts; balance `$20.0264510034`; only Alpha is
  running at `$0.303/hr` total account burn. Discord accepted the 1,093-byte report and 628-byte
  mirror-recovery correction. Next: step 17,500 validation.
- **Discord now receives qualitative improvements only (user directive 2026-07-29):** no routine
  checkpoint/loss/health posts and no regression posts. Every qualifying message must contain the
  controlled input, actual before/after outputs, and why the later response is better; preserve the
  complete result set locally. The saved poster requires the explicit `--qualitative-improvement`
  attestation or exits before reading the webhook. The same eight non-frozen prompts, greedy decoder, and 96-token cap at
  checkpoint 15,000 vs 17,000 improved structural pass 0/8→3/8, loops 3→1, and mean four-gram repeat
  0.3629→0.0444, while four replies stayed empty and encouragement became repetitive. Only casual chat
  was a clear semantic improvement: the 15k star-emoji loop became a relevant greeting and follow-up
  question with clean EOS. That single 988-byte paired example was accepted by Discord. Exact 15k/17k
  outputs are `e1ca5e2b...` / `832722ff...`; summaries are `d8cc6914...` / `f082d290...`.
- **Step 17,500 PASSED execution gates but did not improve quality; no Discord post:** 17,500
  finite/consecutive rows cover 286,720,000 padded tokens (57.7139%). Train/held-out loss is
  1.5036104/1.6043088, +0.0545991 from checkpoint 17,000 and +0.1259977 from the checkpoint-15,000
  best (11th-best run read). P10/median is 3,677.61/3,836.33 tok/s; last-500 loss/gradient norm is
  1.5333154/0.5266370; all 176 allocator samples remain exactly 34 slabs/zero overflow. Rows
  17,001–17,500 held RSS exactly at 4,417MB and external/ArrayBuffers at 2,841–2,842/2,839–2,840MB.
  Mounted metrics-prefix SHA is `e7aae708...`; the original trainer resumed through 17,550 at
  3,541 tok/s. Balance `$19.9540516978`; only Alpha runs at `$0.303/hr`. Next: checkpoint 18,000
  native audit and same-prompt comparison against 17,000; post only if outputs genuinely improve.
- **Checkpoint 18,000 PASSED mechanical gates but regressed quality; Discord correctly stayed silent:**
  18,000 finite/consecutive rows cover 294,912,000 padded tokens (59.3628%). Train/held-out loss is
  1.7388102/1.6347323, +0.0304235 from step 17,500 and +0.1564213 from the checkpoint-15,000 best
  (16th-best run read). P10/median is 3,677.83/3,836.69 tok/s; last-500 loss/gradient norm is
  1.5350078/0.5284160; all 181 allocator samples remain exactly 34 slabs/zero overflow. Rows
  17,501–18,000 held RSS exactly at 4,417MB and external/ArrayBuffers at 2,841–2,843/2,839–2,841MB.
  Checkpoint/audit/metrics-prefix match remote/mounted at `9ef764b9...` / `17b3d413...` /
  `2f3fbac6...`; all 114 tensors / 57,688,576 parameters are finite/nonzero. Byte+SHA mirror proof
  preceded remote checkpoint-15,000 prune and two-state local deletion of exact SHA `32962998...`;
  both sides retain 16k/17k/18k. Original PID resumed through 18,075 at 3,884 tok/s. Identical
  17k→18k eight-prompt generation regressed structural 3→1, EOS 7→6, nonempty 4→3, and mean repeat
  0.0444→0.1129; encouragement became a stronger loop. Exact 18k output/summary hashes are
  `ed0eb1f3...` / `13ea8bad...`. Balance `$19.7611629236`; only Alpha at `$0.303/hr`. Next:
  step 18,500; compare text again at native checkpoint 19,000 and post only a real improvement.
- **Step 18,500 PASSED; third-best held-out result, but no output evidence/no Discord post:** 18,500
  finite/consecutive rows cover 303,104,000 padded tokens (61.0118%). Train/held-out loss is
  1.5098559/1.5037231, improving 0.1310092 from checkpoint 18,000 and sitting only +0.0254120 from
  the 15,000 best (+0.0172779 from checkpoint 16,000). P10/median is 3,678.85/3,837.52 tok/s;
  last-500 loss/gradient norm is 1.5284904/0.5294629; all 186 allocator samples remain exactly
  34 slabs/zero overflow. Rows 18,001–18,500 held RSS exactly at 4,417MB and external/ArrayBuffers
  at 2,841–2,842/2,839–2,840MB. Mounted metrics-prefix SHA is `c9e56be8...`; original PID resumed
  through 18,525 at 3,875 tok/s. Balance `$19.6163806902`; only Alpha at `$0.303/hr`; both services
  active/zero-restart. Validation alone is not a qualitative Discord event, so nothing was posted.
  Next: native checkpoint 19,000 and controlled 18k→19k output comparison.
- **Checkpoint 19,000 PASSED every execution/native/mirror gate, but the complete chat probe regressed:**
  19,000 finite/consecutive rows cover 311,296,000 padded tokens (62.6608%). Train/held-out loss is
  1.4295553/1.6717061, with held-out +0.1679830 from step 18,500 and +0.1933950 from the 15,000 best
  (20th-best run read). P10/median is 3,681.04/3,840.26 tok/s; last-500 loss/gradient norm is
  1.5209734/0.5390623; all 191 allocator samples remain exactly 34 slabs/zero overflow. Rows
  18,501–19,000 held RSS exactly at 4,417MB and external/ArrayBuffers at 2,841–2,843/2,839–2,841MB.
  Checkpoint/audit/metrics-prefix match remote/mounted at `5f0e6b5...` / `484c834...` /
  `dabdb1b...`; all 114 tensors / 57,688,576 parameters are finite/nonzero. Mirror proof preceded
  remote checkpoint-16,000 pruning and two-state local deletion of exact SHA `df81015...`; both sides
  retain 17k/18k/19k. The original PID resumed through 19,100 at 100% GPU utilization; both host
  services are active/zero-restart. Balance `$19.4234416217`; only Alpha at `$0.303/hr`. Identical
  18k→19k eight-prompt generation regressed structural 1→0, EOS 6→5, loops 1→2, and mean repeat
  0.1129→0.1546. One narrow response did improve: encouragement changed from a 0.892-repeat
  `I'm so glad you're here` loop to a relevant, empathetic response at 0.054 repetition. Only that
  1,363-byte input/before/after/explanation was posted to Discord, explicitly bounded by the aggregate
  regression and remaining awkward/nonterminating text. Exact 19k output/summary hashes are
  `64463d9...` / `7ed1cbb...`; posted artifact SHA `e7a5671...`. Next: step 19,500; next text at 20k.
- **Step 19,500 PASSED and recovered much of the checkpoint-19,000 validation wobble; no Discord post:**
  19,500 finite/consecutive rows cover 319,488,000 padded tokens (64.3097%). Train/held-out loss is
  1.6173558/1.5617543; held-out improved 0.1099518 from checkpoint 19k, remains +0.0580312 from
  step 18.5k and +0.0834433 from the 15k best, and ranks seventh among 39 reads. P10/median is
  3,682.94/3,842.95 tok/s; last-500 loss/gradient norm is 1.5259041/0.5399490; all 196 allocator
  samples remain exactly 34 slabs/zero overflow. Rows 19,001–19,500 held RSS exactly at 4,418MB and
  external/ArrayBuffers at 2,841–2,842/2,839–2,840MB. Mounted metrics-prefix SHA is `584e12a...`;
  original PID resumed through 19,525 at 100% GPU utilization. Both host services are active with zero
  automatic restarts; balance `$19.2546072789`; only Alpha runs at `$0.303/hr`. No model text was
  generated at this half-checkpoint, so Discord remained silent. Next: native checkpoint 20k and the
  controlled 19k→20k output comparison.
- **Checkpoint 20,000 PASSED and set a new validation best; one bounded improvement went to Discord:**
  20,000 finite/consecutive rows cover 327,680,000 padded tokens (65.9587%). Train/held-out loss is
  1.5330499/1.4432350, improving 0.1185193 from step 19.5k and 0.0350760 from the prior 15k best
  (first among 40 reads). P10/median is 3,685.18/3,845.95 tok/s; last-500 loss/gradient norm is
  1.5040567/0.5367312; all 201 allocator samples remain exactly 34 slabs/zero overflow. Rows
  19,501–20,000 held RSS exactly at 4,418MB and external/ArrayBuffers at 2,841–2,842/2,839–2,840MB.
  Checkpoint/audit/metrics-prefix match remote/mounted at `eb23e88...` / `43e14dc...` /
  `84cc0f5...`; all 114 tensors / 57,688,576 parameters are finite/nonzero. Byte+SHA mirror proof
  preceded remote checkpoint-17,000 pruning and two-state local deletion of SHA `67d6fb8...`; both
  sides retain 18k/19k/20k. Original PID resumed through 20,075 at 100% GPU utilization; both host
  services remain active/zero-restart. Balance `$19.0857863807`; only Alpha at `$0.303/hr`. The same
  eight prompts improved structural 0→2, EOS 5→7, loops 2→0, and mean repeat 0.1546→0.0208, but
  nonempty remains only 3/8. Casual degraded to `🌟🌟` and cooking remained incorrect, so neither was
  shared. Encouragement was the only clear semantic gain—more coherent, zero repetition, clean EOS—so
  its 1,618-byte exact pair was the only Discord post, with five empty answers and remaining awkwardness
  disclosed. Exact 20k output/summary/post hashes `0f6ad68...` / `196844f...` / `e504f6d...`. Next:
  step 20.5k; next controlled text comparison at 21k.
- **Step 20,500 PASSED and remains in the top validation band; no output/no Discord post:** 20,500
  finite/consecutive rows cover 335,872,000 padded tokens (67.6077%). Train/held-out loss is
  1.4755288/1.4993019; held-out is +0.0560668 from checkpoint 20k's new best and only +0.0209908
  from the former 15k best, ranking fourth among 41 reads. P10/median is 3,687.17/3,848.49 tok/s;
  last-500 loss/gradient norm is 1.4918094/0.5368801; all 206 allocator samples remain exactly
  34 slabs/zero overflow. Rows 20,001–20,500 held RSS exactly at 4,418MB and external/ArrayBuffers
  at 2,841–2,842/2,839–2,840MB. Mounted metrics-prefix SHA is `1faa919...`; original PID resumed
  through 20,525 at 4,014 tok/s. Both host services are active/zero-restart; balance
  `$18.9410557252`; only Alpha at `$0.303/hr`. No text was generated, so Discord remained silent.
  Next: native checkpoint 21k and controlled 20k→21k outputs.
- **Checkpoint 21,000 PASSED and set another validation best; one narrow output improved:** 21,000
  finite/consecutive rows cover 344,064,000 padded tokens (69.2566%). Train/held-out loss is
  1.5382823/1.4385164, improving 0.0047187 from checkpoint 20k and 0.0397947 from the former 15k
  best (first among 42 reads). P10/median is 3,688.81/3,851.00 tok/s; last-500 loss/gradient norm is
  1.5062109/0.5406150. One finite step-20,825 batch clipped at norm 1.1507/coefficient 0.8690 and
  immediately recovered; no non-finite value followed. All 211 allocator samples remain exactly
  34 slabs/zero overflow. Rows 20,501–21,000 held RSS exactly at 4,418MB and external/ArrayBuffers
  at 2,841–2,843/2,839–2,841MB. Checkpoint/audit/metrics-prefix match remote/mounted at
  `e32b77b...` / `c488613...` / `6bf7908...`; all 57,688,576 parameters are finite/nonzero. Mirror
  proof preceded remote 18k pruning and two-state local deletion SHA `9ef764b...`; both sides retain
  19k/20k/21k. Original PID resumed through 21,100 at 100% GPU; both services active/zero-restart.
  Balance `$18.7239325474`; only Alpha at `$0.303/hr`. The same eight prompts regressed structural
  2→1 and nonempty 3→2, kept EOS 7 and loops zero, and lowered mean repeat 0.0208→0.0054.
  Encouragement worsened and six answers are empty. Casual chat alone improved clearly from `🌟🌟`
  to a relevant greeting/follow-up with EOS; its 862-byte exact pair was the only Discord post, with
  the aggregate regression disclosed. Exact 21k output/summary/post hashes `972068f...` /
  `a39c33a...` / `e9dba56...`. Next: step 21.5k; next text at 22k.
- **Step 21,500 PASSED and set another held-out-loss best; no output/no Discord post:** 21,500
  finite/consecutive rows cover 352,256,000 padded tokens (70.9056%). Train/held-out loss is
  1.6939037/1.4370759, improving 0.0014405 from checkpoint 21k and 0.0412352 from the former 15k
  best (first among 43 reads). P10/median is 3,690.65/3,852.92 tok/s; last-500 loss/gradient norm is
  1.4939640/0.5455652. One finite step-21,494 batch clipped at norm 1.4637/coefficient 0.6832 and
  immediately recovered at step 21,495; no non-finite value followed. All 216 allocator samples
  remain exactly 34 slabs/zero overflow. Rows 21,001–21,500 held RSS exactly at 4,418MB and external/
  ArrayBuffers at 2,841–2,842/2,839–2,840MB. Mounted metrics-prefix SHA is `9a47709...`; original PID
  resumed through 21,550 at 3,970 tok/s. Both host services are active/zero-restart; balance
  `$18.6033369899`; the sole running pod is Alpha at `$0.303/hr` account burn. No text was generated,
  so Discord remained silent. Next: native checkpoint 22k and controlled 21k→22k outputs.
- **Checkpoint 22,000 PASSED mechanically; validation/output wobble, no Discord post:** 22,000
  finite/consecutive rows cover 360,448,000 padded tokens (72.5546%). Train/held-out loss is
  1.2526835/1.4985607; held-out is +0.0614848 from step 21.5k's best and +0.0202496 from the former
  15k best (sixth among 44 reads). P10/median is 3,692.41/3,854.56 tok/s; last-500 loss/gradient norm
  is 1.4767901/0.5452086. Two finite batches clipped at steps 21,586 and 21,887 (norm/coefficient
  1.1266/0.8876 and 2.0639/0.4845), each recovering on the next step. All 221 allocator samples
  remain exactly 34 slabs/zero overflow. Rows 21,501–22,000 held RSS exactly at 4,418MB and external/
  ArrayBuffers at 2,841–2,843/2,839–2,841MB. Checkpoint/audit/metrics-prefix match remote/mounted at
  `e163732...` / `85fff63...` / `dfb7959...`; all 57,688,576 parameters are finite/nonzero. Mirror
  proof preceded remote 19k pruning and two-state local deletion SHA `5f0e6b5...`; both sides retain
  20k/21k/22k. Original PID resumed through 22,100 at 4,031 tok/s and 95% GPU; both services active/
  zero-restart. Balance `$18.3862909342`; the sole running pod is Alpha at `$0.303/hr` account burn.
  Same-prompt structural/nonempty rose 1→2/2→4, but EOS fell 7→6, loops worsened 0→2, and mean
  repetition rose 0.0054→0.2352. Casual regressed to emojis; encouragement/cooking became severe
  loops; the new computer response was not a coherent joke; four answers remained empty. Nothing
  genuinely improved, so Discord remained silent. Exact output/summary hashes `f32c6dc...` /
  `5c73ab1...`. Next: step 22.5k; next controlled text at 23k.
- **Step 22,500 PASSED and recovered the 22k validation wobble; no output/no Discord post:** 22,500
  finite/consecutive rows cover 368,640,000 padded tokens (74.2035%). Train/held-out loss is
  1.5070760/1.4394173, improving 0.0591434 from checkpoint 22k and sitting only 0.0023414 above the
  step-21.5k best (third among 45 reads). P10/median is 3,693.81/3,855.82 tok/s; last-500 loss/
  gradient norm is 1.4739300/0.5463152 with no clipped batch. All 226 allocator samples remain
  exactly 34 slabs/zero overflow. Rows 22,001–22,500 held RSS exactly at 4,418MB and external/
  ArrayBuffers at 2,841–2,842/2,839–2,840MB. Mounted metrics-prefix SHA is `5a6017c...`; original PID
  resumed through 22,550 at 3,853 tok/s. Both host services are active/zero-restart; balance
  `$18.2656717269`; the sole running pod is Alpha at `$0.303/hr` account burn. No text was generated,
  so Discord remained silent. Next: native checkpoint 23k and controlled 22k→23k outputs.
- **Checkpoint 23,000 PASSED mechanically; two bounded response improvements posted:** 23,000
  finite/consecutive rows cover 376,832,000 padded tokens (75.8525%). Train/held-out loss is
  1.4600115/1.5330055; held-out is +0.0935882 from step 22.5k and +0.0959296 from the step-21.5k
  best (eleventh among 46 reads). P10/median is 3,695.41/3,856.97 tok/s; last-500 loss/gradient norm
  is 1.4711545/0.5466474 with no clipped batch. All 231 allocator samples remain exactly 34 slabs/
  zero overflow. Rows 22,501–23,000 held RSS exactly at 4,418MB and external/ArrayBuffers at
  2,841–2,843/2,839–2,841MB. Checkpoint/audit/metrics-prefix match remote/mounted at `4cd2cfe...` /
  `06300c8...` / `a7d1fca...`; all 57,688,576 parameters are finite/nonzero. Mirror proof preceded
  remote 20k pruning and two-state local deletion SHA `eb23e88...`; both sides retain 21k/22k/23k.
  Original PID resumed through 23,150 at 3,935 tok/s and 100% GPU; both services active/zero-restart.
  Balance `$17.9921067897`; Alpha remains `$0.22/hr`, while a separate Wajarri pod raised total account
  burn to `$0.75/hr` and was left untouched. Same-prompt structural/EOS regressed 2→1/6→5,
  nonempty/loops held at 4/2, and mean repetition improved 0.2352→0.1909. Casual changed from
  unrelated emojis to a relevant EOS-terminated invitation; encouragement recovered from a
  `-a-snap` loop to a relevant empathetic response with repetition 0.914→0.129 and no loop flag.
  It still repeats and lacks EOS; four answers remain empty, cooking loops, and the joke became a
  loop. Only those two exact pairs/rationales plus the aggregate boundary went to Discord in one
  1,598-byte message. Exact output/summary/post hashes `6beebe1...` / `0683252...` / `b281d9e...`.
  Next: step 23.5k; next controlled text at 24k.
- **Step 23,500 PASSED mechanically; validation wobble persists, no output/no Discord post:** 23,500
  finite/consecutive rows cover 385,024,000 padded tokens (77.5015%). Train/held-out loss is
  1.7112802/1.5379508; held-out is +0.0049453 from checkpoint 23k and +0.1008749 from the step-21.5k
  best (twelfth among 47 reads). P10/median is 3,696.22/3,857.50 tok/s; last-500 loss/gradient norm
  is 1.4797348/0.5436850 with no clipped batch. All 236 allocator samples remain exactly 34 slabs/
  zero overflow. Rows 23,001–23,500 held RSS exactly at 4,418MB and external/ArrayBuffers at
  2,841–2,842/2,839–2,840MB. Mounted metrics-prefix SHA is `9ec4e58...`; original PID resumed through
  23,550 at 3,755 tok/s. Both services active/zero-restart; balance `$17.7712682989`; only Alpha runs
  at `$0.303/hr` account burn. No text was generated, so Discord remained silent. Next: native
  checkpoint 24k and controlled 23k→24k outputs.
- **Checkpoint 24,000 PASSED and set a decisive validation best; one bounded response improvement
  posted:** 24,000 finite/consecutive rows cover 393,216,000 padded tokens (79.1505%). Train/
  held-out loss is 1.3105124/1.3918040, improving 0.1461468 from step 23.5k and 0.0452719 from the
  former step-21.5k best (first among 48 reads). P10/median is 3,696.69/3,857.58 tok/s; last-500
  loss/gradient norm is 1.4622064/0.5481906 with no clipped batch. All 241 allocator samples remain
  exactly 34 slabs/zero overflow. Rows 23,501–24,000 held RSS exactly at 4,418MB and external/
  ArrayBuffers at 2,841–2,843/2,839–2,841MB. Checkpoint/audit/metrics-prefix match remote/mounted at
  `5dd9ff2...` / `0d1f790...` / `04fcb77...`; all 57,688,576 parameters are finite/nonzero. Mirror
  proof preceded remote 21k pruning and two-state local deletion SHA `e32b77b...`; both sides retain
  22k/23k/24k. Original PID resumed through 24,100 at 3,777 tok/s and 100% GPU; both services active/
  zero-restart. Balance `$17.3491166155`; Alpha remains `$0.22/hr`, while a separate Wajarri pod raised
  total account burn to `$0.75/hr` and was left untouched. Same-prompt structural/EOS improved
  1→2/5→7, loops cleared 2→0, and mean repetition fell 0.1909→0.0328, while nonempty fell 4→3.
  Cooking alone clearly improved from a truncated loop to a finite EOS response, clearing its loop
  flag and reducing repetition 0.516→0.080. It still ignores the supplied ingredients/invents others;
  five answers remain blank. Only that exact pair/rationale/boundary went to Discord in a 1,562-byte
  message. Exact output/summary/post hashes `76b12df...` / `ae1e54b...` / `f32a72a...`. Next: step
  24.5k; next controlled text at 25k.
- **Step 24,500 PASSED and remained in the top validation band; no output/no Discord post:** 24,500
  finite/consecutive rows cover 401,408,000 padded tokens (80.7994%). Train/held-out loss is
  1.2920806/1.4415536; held-out is +0.0497496 from checkpoint 24k's best (fifth among 49 reads).
  P10/median is 3,697.59/3,857.57 tok/s; last-500 loss/gradient norm is 1.4617461/0.5496144 with no
  clipped batch. All 246 allocator samples remain exactly 34 slabs/zero overflow. Rows 24,001–24,500
  held RSS exactly at 4,418MB and external/ArrayBuffers at 2,841–2,842/2,839–2,840MB. Mounted
  metrics-prefix SHA is `2061947...`; original PID resumed through 24,550 at 3,822 tok/s. Both
  services active/zero-restart; balance `$16.994817734`; only Alpha runs at `$0.303/hr` account burn.
  No text was generated, so Discord remained silent. Next: native checkpoint 25k and controlled
  24k→25k outputs.
- **Checkpoint 25,000 PASSED mechanically; validation/output wobble, one bounded response improvement
  posted:** 25,000 finite/consecutive rows cover 409,600,000 padded tokens (82.4484%). Train/
  held-out loss is 1.5305007/1.4674260; held-out is +0.0756220 from checkpoint 24k's best (seventh
  among 50 reads). P10/median is 3,698.41/3,857.69 tok/s; last-500 loss/gradient norm is
  1.4502269/0.5585494. Four finite batches clipped at steps 24,585/24,656/24,893/24,913 (norm
  1.1732–1.8130, coefficient 0.5516–0.8523), each recovering on the next step. All 251 allocator
  samples remain exactly 34 slabs/zero overflow. Rows 24,501–25,000 held RSS exactly at 4,418MB and
  external/ArrayBuffers at 2,841–2,843/2,839–2,841MB. Checkpoint/audit/metrics-prefix match remote/
  mounted at `cee2ec1...` / `f2427f8...` / `935841a...`; all 57,688,576 parameters are finite/nonzero.
  Mirror proof preceded remote 22k pruning and two-state local deletion SHA `e163732...`; both sides
  retain 23k/24k/25k. Original PID resumed through 25,150 at 3,688 tok/s and 100% GPU; both services
  active/zero-restart. Balance `$16.6972858915`; only Alpha runs at `$0.303/hr` account burn. Same-
  prompt structural/EOS regressed 2→1/7→4, loops worsened 0→2, mean repetition rose
  0.0328→0.2258, and nonempty rose 3→5 only because joke/basil became severe loops. Encouragement
  alone improved: it was more directly grounded in being new at the job, removed the inaccurate
  `after a long day` framing, and cut repetition 0.183→0.022. It still starts with stray punctuation,
  becomes generic, and lacks EOS. Only that exact pair/rationale/boundary went to Discord in a
  1,700-byte message. Exact output/summary/post hashes `8277ae1...` / `dd15406...` / `493531d...`.
  Next: step 25.5k; next controlled text at 26k.
- **Step 25,500 PASSED mechanically; validation wobble persists, no output/no Discord post:** 25,500
  finite/consecutive rows cover 417,792,000 padded tokens (84.0974%). Train/held-out loss is
  1.2616466/1.5134359; held-out is +0.0460099 from checkpoint 25k and +0.1216319 from checkpoint
  24k's best (thirteenth among 51 reads). P10/median is 3,698.49/3,857.21 tok/s; last-500 loss/
  gradient norm is 1.4435426/0.5583323. Two finite batches clipped at steps 25,184 and 25,200
  (norm/coefficient 1.4402/0.6943 and 1.0313/0.9697), each recovering on the next step. All 256
  allocator samples remain exactly 34 slabs/zero overflow. Rows 25,001–25,500 held RSS exactly at
  4,418MB and external/ArrayBuffers at 2,841–2,842/2,839–2,840MB. Mounted metrics-prefix SHA is
  `5b42387...`; original PID resumed through 25,550 at 3,863 tok/s. Both services active/zero-restart;
  balance `$16.4295910287`; only Alpha runs at `$0.303/hr` account burn. No text was generated, so
  Discord remained silent. Next: native checkpoint 26k and controlled 25k→26k outputs.
- **Checkpoint 26,000 PASSED mechanically; no qualifying response improvement/no Discord post:**
  26,000 finite/consecutive rows cover 425,984,000 padded tokens (85.7463%). Train/held-out loss is
  1.3613813/1.4856620; held-out improved 0.0277739 from step 25.5k but remains +0.0938580 from
  checkpoint 24k's best (ninth among 52 reads). P10/median is 3,696.80/3,856.05 tok/s; last-500
  loss/gradient norm is 1.4537767/0.5540193. One finite step-25,649 batch clipped at norm/coefficient
  1.0619/0.9417 and recovered on the next step. All 261 allocator samples remain exactly 34 slabs/
  zero overflow. Rows 25,501–26,000 held RSS exactly at 4,418MB and external/ArrayBuffers at
  2,841–2,843/2,839–2,841MB. Checkpoint/audit/metrics-prefix match remote/mounted at `eb66730...` /
  `7f822a9...` / `678c75c...`; all 57,688,576 parameters are finite/nonzero. Mirror proof preceded
  remote 23k pruning and two-state local deletion SHA `4cd2cfe...`; both sides retain 24k/25k/26k.
  Original PID resumed through 26,100 at 3,927 tok/s; both watcher processes remain live. Balance
  `$16.2366328324`; only Alpha runs at `$0.303/hr`. Same-prompt aggregate EOS improved 4→6, loops
  2→1, and mean repetition 0.2258→0.1237, but structural stayed 1 and nonempty fell 5→3. Casual was
  unchanged; cooking regressed to an emoji loop; joke and basil became empty rather than useful; four
  answers were blank; encouragement remained truncated and became less accurately framed. No actual
  response improved, so Discord remained silent. Exact output/summary hashes `05678ed...` /
  `73fca83...`. Next: step 26.5k; next controlled text at 27k.
- **Step 26,500 PASSED and returned to the top validation band; no output/no Discord post:** 26,500
  finite/consecutive rows cover 434,176,000 padded tokens (87.3953%). Train/held-out loss is
  1.2603179/1.4199024; held-out improved 0.0657596 from checkpoint 26k and remains only +0.0280984
  from checkpoint 24k's best (second among 53 reads). P10/median is 3,694.95/3,854.77 tok/s;
  last-500 loss/gradient norm is 1.4369010/0.5571356 with no clipped batch. All 266 allocator samples
  remain exactly 34 slabs/zero overflow. Rows 26,001–26,500 held RSS exactly at 4,418MB and external/
  ArrayBuffers at 2,841–2,842/2,839–2,840MB. Mounted metrics-prefix SHA is `e434598...`; original PID
  resumed through 26,550 at 3,660 tok/s. Both watcher processes remain live; balance
  `$16.0677470953`; only Alpha runs at `$0.303/hr` account burn. No text was generated, so Discord
  remained silent. Next: native checkpoint 27k and controlled 26k→27k outputs.
- **Checkpoint 27,000 PASSED mechanically; difficult-data and validation/output wobble, no Discord
  post:** 27,000 finite/consecutive rows cover 442,368,000 padded tokens (89.0443%). Train/held-out
  loss is 2.1408052/1.5260614; held-out is +0.1061590 from step 26.5k and +0.1342574 from checkpoint
  24k's best (seventeenth among 54 reads). P10/median is 3,692.31/3,852.89 tok/s; last-500 loss/
  gradient norm is 1.8661603/0.6754442. Thirteen finite batches clipped: steps 26,594/26,617, the
  seven-step 26,699–26,705 cluster, and 26,708/26,797/26,908/26,990 (norm 1.0131–1.9668,
  coefficient 0.5084–0.9870). The cluster recovered at 26,706–26,707 and again at 26,709; 27,000
  was unclipped. All 271 allocator samples remain exactly 34 slabs/zero overflow. Rows 26,501–27,000
  held RSS exactly at 4,418MB and external/ArrayBuffers at 2,841–2,843/2,839–2,841MB. Checkpoint/
  audit/metrics-prefix match remote/mounted at `570dd33...` / `4a5b1d7...` / `fba06a1...`; all
  57,688,576 parameters are finite/nonzero. Mirror proof preceded remote 24k pruning and two-state
  local deletion SHA `5dd9ff2...`; both sides retain 25k/26k/27k. Original PID resumed through
  27,100 at 3,717 tok/s; both watcher processes remain live. Balance `$15.8965556804`; only Alpha
  runs at `$0.303/hr`. Same-prompt structural/EOS regressed 1→0/6→5, loops rose 1→3, repetition
  rose 0.1237→0.3374, and only three answers were nonempty. Casual regressed to a Java-fence loop,
  encouragement to punctuation repetition, cooking remained a loop, and five answers were blank.
  Nothing genuinely improved, so Discord remained silent. Exact output/summary hashes `fd6d0ad...` /
  `b60c156...`. Next: step 27.5k; next controlled text at 28k.
- **Step 27,500 PASSED and recovered to the top validation band; no output/no Discord post:** 27,500
  finite/consecutive rows cover 450,560,000 padded tokens (90.6932%). Train/held-out loss is
  2.0169199/1.4241974; held-out improved 0.1018640 from checkpoint 27k and remains only +0.0323934
  from checkpoint 24k's best (third among 55 reads). P10/median is 3,690.04/3,851.09 tok/s;
  last-500 loss/gradient norm is 2.0276268/0.6949074. Seven finite batches clipped at steps
  27,105/27,150/27,192/27,199/27,289/27,364/27,462 (norm 1.0344–1.5347, coefficient
  0.6516–0.9667); 27,150 recovered immediately at 27,151, and 27,500 was unclipped. All 276
  allocator samples remain exactly 34 slabs/zero overflow. Rows 27,001–27,500 held RSS exactly at
  4,418MB and external/ArrayBuffers at 2,841–2,842/2,839–2,840MB. Mounted metrics-prefix SHA is
  `f664410...`; original PID resumed through 27,550 at 3,597 tok/s. Both watcher processes remain
  live; balance `$15.6387599267`; only Alpha runs at `$0.303/hr` account burn. No text was generated,
  so Discord remained silent. Next: native checkpoint 28k and controlled 27k→28k outputs.
- **Checkpoint 28,000 PASSED mechanically; validation/output wobble, no Discord post:** 28,000
  finite/consecutive rows cover 458,752,000 padded tokens (92.3422%). Train/held-out loss is
  1.9920901/1.5399532; held-out is +0.1157558 from step 27.5k and +0.1481493 from checkpoint 24k's
  best (twenty-first among 56 reads). P10/median is 3,688.46/3,849.72 tok/s; last-500 loss/gradient
  norm is 1.9377729/0.6713991. Five finite batches clipped at 27,693/27,697/27,841/27,914/27,980
  (norm 1.0296–1.2598, coefficient 0.7938–0.9713); 28,000 was unclipped. All 281 allocator samples
  remain exactly 34 slabs/zero overflow. Rows 27,501–28,000 held RSS exactly at 4,418MB and external/
  ArrayBuffers at 2,841–2,843/2,839–2,841MB. Checkpoint/audit/metrics-prefix match remote/mounted at
  `b697cef...` / `a52bc1f...` / `f1a97a0...`; all 57,688,576 parameters are finite/nonzero. Mirror
  proof preceded remote 25k pruning and two-state local deletion SHA `cee2ec1...`; both sides retain
  26k/27k/28k. Original PID resumed through 28,075 at 3,698 tok/s; both watcher processes remain live.
  Balance `$15.1785022137`; Alpha remains `$0.22/hr`, while a separate Wajarri pod raised total burn
  to `$0.752/hr` and was left untouched. Same-prompt structural/nonempty improved 0→1/3→4, but the
  finite casual answer was irrelevant Java, the new joke was a loop, encouragement/cooking stayed
  loops, and four answers were blank. Nothing genuinely improved, so Discord remained silent. Exact
  output/summary hashes `6c5c014...` / `0587fdc...`. Next: step 28.5k; text next at 29k.
- **Step 28,500 PASSED and set a decisive new validation best; no output/no Discord post:** 28,500
  finite/consecutive rows cover 466,944,000 padded tokens (93.9912%). Train/held-out loss is
  1.9561090/1.3334526, improving 0.2065006 from checkpoint 28k and 0.0583514 from the former
  checkpoint-24k best (first among 57 reads). P10/median is 3,686.96/3,848.46 tok/s; last-500
  loss/gradient norm is 1.8830708/0.6727490. Seven finite batches clipped at steps
  28,033/28,058/28,119/28,152/28,263/28,383/28,446 (norm 1.0106–1.2619, coefficient
  0.7924–0.9895); 28,500 was unclipped. All 286 allocator samples remain exactly 34 slabs/zero
  overflow. Rows 28,001–28,500 held RSS exactly at 4,418MB and external/ArrayBuffers at
  2,841–2,842/2,839–2,840MB. Mounted metrics-prefix SHA is `4d2b4ef...`; original PID resumed
  through 28,550 at 3,799 tok/s. Both watcher processes remain live. Balance `$14.7481807932`;
  Alpha remains `$0.22/hr`, while the separate Wajarri pod kept total burn at `$0.752/hr` and was
  left untouched. No text was generated, so Discord remained silent. Next: native checkpoint 29k
  and controlled 28k→29k outputs.
- **Checkpoint 29,000 PASSED mechanically; late gradient/validation wobble, one narrow response gain
  posted:** 29,000 finite/consecutive rows cover 475,136,000 padded tokens (95.6401%). Train/held-out
  loss is 2.2414429/1.6412250, +0.3077724 from step 28.5k's best (thirty-seventh among 58 reads).
  P10/median is 3,686.60/3,847.93 tok/s; last-500 loss/gradient norm is 2.1489550/0.9638283.
  Exactly 153 finite batches clipped: ten scattered steps before 28,858, then every step from
  28,858–29,000 (norm 1.0013–2.2073, coefficient 0.4530–0.9987). This late spike regime is real
  and preserved, but every value remained finite and clipping bounded it. All 291 allocator samples
  remain exactly 34 slabs/zero overflow. Rows 28,501–29,000 held RSS exactly at 4,418MB and external/
  ArrayBuffers at 2,841–2,843/2,839–2,841MB. Checkpoint/audit/metrics-prefix match remote/mounted at
  `03eaac3...` / `b71335b...` / `338d89b...`; all 57,688,576 parameters are finite/nonzero. Mirror
  proof preceded remote 26k pruning and two-state local deletion SHA `eb66730...`; both sides retain
  27k/28k/29k. Original PID resumed through 29,125; both watcher processes remain live. Balance
  `$14.3275693782`; only Alpha runs at `$0.303/hr`. Aggregate prompts regressed structural 1→0,
  EOS 5→4, loops 3→4, repetition 0.3347→0.4180. Encouragement alone improved from punctuation-only
  failure to readable supportive wording and cut repetition 0.968→0.731, though it still loops and
  lacks EOS. Casual regressed, cooking became blank, joke/dragon looped, and four answers were blank.
  Only that exact pair/rationale/boundary went to Discord in a 1,236-byte message. Exact output/
  summary/post hashes `ef6c14d...` / `f66ad62...` / `318a7b8...`. Next: step 29.5k; text next at 30k.
- **Step 29,500 remained finite/bounded but confirmed the persistent clipped-gradient and validation
  wobble; no output/no Discord post:** 29,500 finite/consecutive rows cover 483,328,000 padded tokens
  (97.2891%). Train/held-out loss is 1.9044154/1.6889295; held-out is +0.0477045 from checkpoint 29k
  and +0.3554769 from step 28.5k's best (forty-first among 59 reads). P10/median is
  3,687.71/3,848.15 tok/s; last-500 loss/gradient norm is 2.0148914/1.4205415. Every step from
  29,001–29,500 clipped with finite norms 1.1267–2.3803 and coefficients 0.4201–0.8875. This is a
  real persistent regime, but all metrics stayed finite, memory remained bounded, and clipping
  contained it. All 296 allocator samples remain exactly 34 slabs/zero overflow. Rows 29,001–29,500
  held RSS at 4,418–4,419MB and external/ArrayBuffers at 2,841–2,842/2,839–2,840MB. Mounted metrics-
  prefix SHA is `455e631...`; original PID resumed through 29,550 at 3,909 tok/s. Both watcher
  processes remain live. Balance `$13.9783174857`; Alpha remains `$0.22/hr`, while a newly recreated
  Wajarri pod raised total burn to `$0.752/hr` and was left untouched. No text was generated, so
  Discord remained silent. Next: native checkpoint 30k and controlled 29k→30k outputs.
- **Checkpoint 30,000 PASSED mechanically; late validation/output wobble, no Discord post:** 30,000
  finite/consecutive rows cover 491,520,000 padded tokens (98.9381%). Train/held-out loss is
  1.9388337/1.7891528; held-out is +0.1002233 from step 29.5k and +0.4557002 from step 28.5k's best
  (forty-ninth among 60 reads). P10/median is 3,684.66/3,848.57 tok/s; last-500 loss/gradient norm
  is 1.9019102/1.4219356. Every step from 29,501–30,000 clipped with finite norms 1.1840–3.0431
  and coefficients 0.3286–0.8446; all metrics stayed finite and bounded. All 301 allocator samples
  remain exactly 34 slabs/zero overflow. Rows 29,501–30,000 held RSS exactly at 4,419MB and
  external/ArrayBuffers at 2,841–2,843/2,839–2,841MB. Remote/mounted checkpoint/audit/metrics hashes
  are `73db03c...` / `63204e8...` / `dfa9f84...`; all 57,688,576 parameters are finite/nonzero.
  Mirror proof preceded remote 27k pruning and ledgered local deletion SHA `570dd33...`; both sides
  retain 28k/29k/30k. Original PID resumed through 30,050; both watcher processes remain live.
  Balance `$13.4864416169`; Alpha remains `$0.22/hr`, while the separate Wajarri pod keeps total
  burn at `$0.752/hr` and was left untouched. Aggregate mechanics improved, but semantic output did
  not: six immediate-EOS blanks, a code-fence-only joke, and irrelevant looping Python encouragement.
  Nothing genuinely improved, so Discord remained silent. Exact output/summary hashes `1e98c4f...` /
  `540f391...`. Only 322 terminal steps remain before automated finalization.
- **Early ad hoc quality preview remains below the chat bar:** three non-frozen greedy prompts against
  full-run checkpoint 2,000 produced one recognizable personal answer and two obvious repetition loops.
  This is an honest 6.6%-of-epoch diagnostic only; no frozen prompt was inspected or used for tuning,
  and the one-epoch run remains the planned discriminator. The same three prompts at checkpoint 3,000
  remained below bar: an emoji loop, one grammatical but semantically odd visual-art answer, and a
  repetitive rain answer. Mirrored sample log hashes are `d1542164...` (2,000) and `cc0a1b28...`
  (3,000). At checkpoint 4,000, the greeting terminated with EOS but answered `#### 1 The answer is: 1`,
  the fun prompt collapsed into a `fun` loop, and the rain response was grammatical but falsely described
  rain as energy. Its sample log is `df2d1e9e...`. No frozen prompt was exposed, and quality remains below
  the chat bar at 13.2% of the epoch.
- **Fail-closed terminal finalizer is LIVE:** `6d92470` added
  `runpod_sft_terminal_{watch,finalize_remote}.sh`; real-pod one-shot preflight passed exact source,
  frozen/base inputs, analyzers, and isolated Transformers dependencies. Negative tests rejected a
  wrong source at exit 2 and premature finalization at exit 3 without creating artifacts. User service
  `alpha2-flagship-sft-finalizer-20260728.service` is active with zero restarts and proved real progress
  from 3,250 to 3,300 rows while retaining the exact trainer PID. At clean step 30,322 it will run the
  terminal audit/analyzer, sealed 100-chat/200-QA eval, pair gate, HF export, and logit parity; preserve
  machine PASS or FAIL with semantic review still pending; hash-mirror every remote artifact; and only
  then remove scoped pod `gp4m6s8m06bhen`. Any operational failure leaves the pod untouched. It never
  publishes the chat model automatically.
- **Semantic-review handoff is fail-closed and reference-blinded:** `e1df144` adds
  `prepare_frozen_chat_semantic_review.ts`, which will bind the terminal chat checkpoint, canonical
  manifest, exact 100 prompts, summary, and detailed outputs before producing the manual review packet.
  It excludes held-out reference answers, rejects reordered/substituted cases, and predeclares the
  `PASS`/`BORDERLINE`/`FAIL` rubric. `db3d7e2` adds the matching finalizer: all 100 verdicts and rationales
  are mandatory; every sealed input and packet case is re-hashed/reconciled; and semantic PASS requires
  at least 80 `PASS` and zero `FAIL`. Positive and one-gibberish negative proofs pass 2/2 with clean
  package typecheck. Do not run the packet preparer on sealed prompts until terminal generation completes.
- **Chat publication is now explicit and fail-closed:** `1c9d218` adds `publish_hf_chat.py` and its
  release-path test. Read-only preflight is the default; `--publish` requires exact terminal/SFT/D3/
  semantic/parity evidence, one hash-consistent terminal checkpoint, the completed model card, the exact
  six-file zero-custom-code export, and the sole target `ajaxdavis/alpha-60m-chat`. The test exercises
  both preflight and an isolated full publication simulation against the installed Hub API surface. Use
  `/mnt/donto-data/alpha-corpora/.venv/bin/python` after pod teardown; the environment persists and has
  `huggingface_hub`, Transformers, safetensors, and CPU torch. The saved identity is `ajaxdavis`, and an
  authenticated check confirmed the chat repository is not yet present.
- **Frozen base eval COMPLETE and mirrored:** exact terminal base checkpoint `08e14fa9...` ran the
  canonical 100-chat/200-QA suite; remote/local hashes match under
  `/mnt/donto-data/alpha-runs/frozen-eval-base-flagship-20260728`. The honest pre-SFT baseline is
  0/100 structural chat passes, 99 degenerate loops, 0 QA exact, and 1 QA answer-contained. This is
  the before-side of the required base-vs-chat gate, not a passing chat result.
- **Exact base HF export VERIFIED, mirrored, and PUBLIC:** the six-file zero-custom-code export under
  `/mnt/donto-data/alpha-runs/hf-alpha-60m-base-c333bf2-20260728` loaded as stock
  `LlamaForCausalLM` with all 57,688,576 parameters. Alpha `cpu_ref` versus Transformers passed 2/2
  top-1 positions, exact tokenizer parity, and max logit delta `6.771e-05`; stock CPU
  `pipeline("text-generation")` cold-loaded and generated `Hello, I'm a little bit of a`. Public repo
  `https://huggingface.co/ajaxdavis/alpha-60m-base` is live at commit `8693cb4c...`, with the tracked
  Apache-2.0 model card byte-identical on Hub. A second anonymous empty-cache Hub download then loaded
  stock `LlamaForCausalLM` on CPU, re-proved 57,688,576 parameters and exact safetensors SHA
  `d0aa2ccd...`, and completed both plain-text and message-list pipelines without custom code. The first
  Hub verifier attempt is retained because its pipeline auto-selected the training GPU and hit CUDA OOM;
  the corrected verifier hid CUDA and explicitly selected CPU. Sealed publication evidence is at
  `/mnt/donto-data/alpha-runs/hf-base-publication-c333bf2-20260728/` (`hub-cold-load-cpu.log`
  SHA-256 `99736b97...`). Reusable fail-closed `scripts/verify_hf_hub.py` landed in `679ce83`; a second
  anonymous empty-cache run pinned exact Hub revision `8693cb4c...` and passed (`scripted-hub-cold-load.json`
  SHA-256 `1f7fd4c7...`), while an intentional warm-cache rerun was rejected before loading.
- The flagship pretrain guard exited cleanly after its final pull; its retention/provenance artifacts
  remain in the canonical mounted run. Connectivity failures never counted as training stalls. The
  checkpoint filename filter was tightened so native-audit sidecars cannot interrupt retention.
- The stopped original pod `d5m7h1v0kr0zd4` was deleted only after recovery2 caches and fresh GPU
  metrics were proven; it is irrecoverable and no unique data remained on it. Temporary gzip transfer
  copies were also removed after the canonical mounted corpus hashes were reverified.
- **Next gate:** finish/analyze the live full masked SFT, run the frozen chat-side evaluation plus
  base-vs-chat analyzer and human semantic review, then export/verify/publish the chat repo. The base
  repo is already public and cold-load verified.

## Historical pre-interruption flagship record

The chronology below is retained as evidence; its original pod ID and SSH endpoint are no longer usable.

- **Pod `d5m7h1v0kr0zd4`**, RTX 3090 community, **$0.22/hr**. SSH:
  `ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p 8865 root@64.119.209.250`.
- **Flagship pretrain is LIVE** at
  `/workspace/alpha2/runs/flagship-1b-e561f66-20260724`, launched at 10:51 UTC on exact source
  `e561f66c7a88a5294e9cb74a4fc3afd6be167d4f`. Its immutable contract binds selector-report SHA
  `10d39e47…`, selected LR `1e-3`, corpus-manifest SHA `c7ecaf7d…`, tokenizer SHA `c310343a…`,
  57,688,576 parameters, 61,036 steps, and exactly 1,000,013,824 tokens. All three source shards
  passed the launcher's fresh 5,976,889,749-byte hash verification. The two missing shard caches were
  atomically built (exactly 1,029,128,000 new train/validation tokens), and GPU training is now live.
  The first checkpoint gate passed 1,000/1,000 finite/consecutive rows and exactly 16,384,000 tokens:
  train loss 9.4982→4.8432, held-out loss improved 5.4226→4.8698 across steps 500/1,000, and p10/median
  throughput after step 50 is 3,730/3,862 tok/s. All 11 allocator samples report exactly 34 slabs and
  zero overflow; RSS stayed 7,804–8,960MB. The save released all 228 optimizer buffers with GC.
  Remote/mounted checkpoint 1,000 is a byte-identical, native-audited 692,528,815-byte ALPH file at
  SHA-256 `93ddc593…`, with all 57,688,576 parameters finite/nonzero. Metrics at checkpoint 1,000 are
  byte-identical at `bc616a21…`. The next held-out gate also passed: 1,500 finite/consecutive rows and
  24,576,000 tokens, train/held-out loss 4.4025/4.4596, validation improvement 0.4102, p10/median
  throughput 3,725/3,856 tok/s, 16 allocator samples, 34 slabs, and zero overflow. Remote/mounted
  metrics match at `a3860b8b…`. The second checkpoint gate then passed 2,000 finite rows/32,768,000
  tokens, train/held-out loss 4.2562/4.2743, another 0.1853 validation improvement, p10/median
  3,723/3,849 tok/s, 21 allocator samples, 34 slabs, and zero overflow. Checkpoint 2,000 is a
  hash-mirrored/native-audited 692,528,815-byte ALPH file at `7f54b34a…`; metrics match at `01a31962…`.
  Its save released 228 buffers with GC and returned ArrayBuffers 7,072→6,631MB, one MB below the
  first-save plateau, proving no per-checkpoint accumulation. The step-2,500 held-out gate then passed
  2,500 finite rows/40,960,000 tokens: train/held-out loss 4.0449/4.1624, another 0.1119 validation
  improvement, p10/median 3,722/3,850 tok/s, 26 allocator samples, 34 slabs, and zero overflow.
  Remote/mounted metrics match at `44a82dea…`; every one of the 500 rows after checkpoint 2,000 held
  ArrayBuffers exactly at 6,632MB and RSS within 7,883–7,942MB. The checkpoint-3,000 gate then passed
  3,000 finite rows/49,152,000 tokens: train/held-out loss 4.0256/4.0843, another 0.0781 validation
  improvement, p10/median 3,726/3,852 tok/s, 31 allocator samples, 34 slabs, and zero overflow.
  Checkpoint 3,000 is a hash-mirrored/native-audited 692,528,815-byte ALPH file at `a2a56b81…` with
  all 57,688,576 parameters finite/nonzero; the exact metrics prefix matches at `e0139d26…`. Its save
  again released 228 buffers and returned ArrayBuffers 7,072→6,631MB. The step-3,500 held-out gate
  then passed 3,500 finite rows/57,344,000 tokens: train/held-out loss 3.8251/3.9699, another 0.1144
  validation improvement, p10/median 3,723/3,850 tok/s, 36 allocator samples, 34 slabs, and zero
  overflow. Remote/mounted metrics match at `6a3f69cf…`; all 500 rows after checkpoint 3,000 held
  ArrayBuffers exactly at 6,632MB and RSS within 7,885–7,943MB. The checkpoint-4,000 gate then passed
  4,000 finite rows/65,536,000 tokens: train/held-out loss 4.0469/3.8976, another 0.0723 validation
  improvement, p10/median 3,724/3,850 tok/s, 41 allocator samples, 34 slabs, and zero overflow.
  Checkpoint 4,000 is a hash-mirrored/native-audited 692,528,815-byte ALPH file at `25b061b5…` with
  all 57,688,576 parameters finite/nonzero; metrics match at `79c4a1b9…`. Its save again released 228
  buffers and returned ArrayBuffers 7,072→6,631MB. The first live prune passed: only after size/SHA
  verification of the mounted mirror, the guard removed remote checkpoint 1,000, then ledgered and
  removed the identical local `93ddc593…` artifact. The step-4,500 held-out gate then passed 4,500
  finite rows/73,728,000 tokens: train/held-out loss 3.8755/3.8603, another 0.0373 validation
  improvement, p10/median 3,724/3,850 tok/s, 46 allocator samples, 34 slabs, and zero overflow.
  Remote/mounted metrics match at `654595d1…`; all 500 rows after checkpoint 4,000 held ArrayBuffers
  within 6,631–6,632MB and RSS within 7,868–7,933MB. The checkpoint-5,000 gate then passed 5,000
  finite rows/81,920,000 tokens: train/held-out loss 3.8075/3.7961, another 0.0642 validation
  improvement, p10/median 3,723/3,850 tok/s, 51 allocator samples, 34 slabs, and zero overflow.
  Checkpoint 5,000 is a hash-mirrored/native-audited 692,528,815-byte ALPH file at `b9851894…` with
  all 57,688,576 parameters finite/nonzero; metrics match at `34a5e893…`. Its save again released 228
  buffers and returned ArrayBuffers 7,072→6,631MB. The second live prune safely removed remote
  checkpoint 2,000 only after mounted size/SHA proof, followed by ledgered local removal of the same
  `7f54b34a…` artifact. The step-5,500 gate then passed 5,500 finite rows/90,112,000 tokens. Train
  loss was 3.7954; held-out loss 3.8107 was a small +0.0146 wobble from step 5,000, while remaining
  1.6119 below step 500. P10/median throughput was 3,723/3,850 tok/s; all 56 allocator samples report
  34 slabs and zero overflow. Remote/mounted metrics match at `bbc5e153…`; every post-checkpoint row
  held ArrayBuffers exactly at 6,632MB and RSS within 7,860–7,931MB. The checkpoint-6,000 gate then
  passed 6,000 finite rows/98,304,000 tokens: train/held-out loss 3.5498/3.7055, recovering the
  step-5,500 wobble with a new-best 0.1051 validation improvement, p10/median 3,721/3,849 tok/s,
  61 allocator samples, 34 slabs, and zero overflow. Checkpoint 6,000 is a hash-mirrored/native-audited
  692,528,815-byte ALPH file at `6b171970…` with all 57,688,576 parameters finite/nonzero; metrics
  match at `616f5385…`. The save again released 228 buffers and returned ArrayBuffers
  7,072→6,631MB. The third live prune safely removed remote checkpoint 3,000 only after mounted
  size/SHA proof, followed by ledgered local removal of the same `a2a56b81…` artifact. Training
  resumed through step 6,050; both sides retain exactly 4,000/5,000/6,000. The step-6,500 gate then
  passed 6,500 finite rows/106,496,000 tokens. Train loss was 3.6452; held-out loss 3.7176 was a small
  +0.0120 wobble from step 6,000 while remaining 0.0931 below step 5,500. P10/median throughput was
  3,723/3,850 tok/s; all 66 allocator samples report 34 slabs and zero overflow. Remote/mounted
  metrics match at `9d1ff974…`; every post-checkpoint row held ArrayBuffers exactly at 6,632MB and
  RSS within 7,865–7,934MB. The checkpoint-7,000 gate then passed 7,000 finite rows/114,688,000
  tokens: train/held-out loss 3.5480/3.7157, a 0.0019 validation improvement from step 6,500 and
  only 0.0101 above the step-6,000 best. P10/median throughput was 3,722/3,849 tok/s; all 71
  allocator samples report 34 slabs and zero overflow. Checkpoint 7,000 is a
  hash-mirrored/native-audited 692,528,815-byte ALPH file at `b26165fd…` with all 57,688,576
  parameters finite/nonzero; metrics match at `4c835219…`. The save again released 228 buffers and
  returned ArrayBuffers 7,072→6,631MB. The fourth live prune safely removed remote checkpoint 4,000
  only after mounted size/SHA proof, followed by ledgered local removal of the same `25b061b5…`
  artifact. Training resumed through step 7,025; both sides retain exactly 5,000/6,000/7,000. The
  step-7,500 gate then passed 7,500 finite rows/122,880,000 tokens: train/held-out loss
  3.6169/3.6547, a new best by 0.0610 from step 7,000 and 0.0509 below the prior best at step 6,000.
  P10/median throughput was 3,723/3,850 tok/s; all 76 allocator samples report 34 slabs and zero
  overflow. Remote/mounted metrics match at `0dc719fc…`; every post-checkpoint row held ArrayBuffers
  exactly at 6,632MB and RSS within 7,854–7,936MB. The checkpoint-8,000 gate then passed 8,000
  finite rows/131,072,000 tokens: train/held-out loss 3.5756/3.6440, another new best by 0.0106 from
  step 7,500 and 0.0615 below step 6,000. P10/median throughput was 3,724/3,850 tok/s; all 81
  allocator samples report 34 slabs and zero overflow. Checkpoint 8,000 is a
  hash-mirrored/native-audited 692,528,815-byte ALPH file at `e7658b21…` with all 57,688,576
  parameters finite/nonzero; metrics match at `8b1679e0…`. The save again released 228 buffers and
  returned ArrayBuffers 7,072→6,631MB. The fifth live prune safely removed remote checkpoint 5,000
  only after mounted size/SHA proof, followed by ledgered local removal of the same `b9851894…`
  artifact. Training resumed through step 8,050; both sides retain exactly 6,000/7,000/8,000. The
  step-8,500 gate then passed 8,500 finite rows/139,264,000 tokens. Train loss was 3.4844; held-out
  loss 3.7603 was a 0.1163 wobble from the step-8,000 best while remaining 0.0504 below step 5,500
  and 1.6623 below step 500. P10/median throughput was 3,725/3,849 tok/s; all 86 allocator samples
  report 34 slabs and zero overflow. Remote/mounted metrics match at `c301b0b5…`; every
  post-checkpoint row held ArrayBuffers exactly at 6,632MB and RSS within 7,871–7,937MB. Overnight,
  every gate through checkpoint 12,000 passed: held-out loss at steps 9,000/9,500/10,000/10,500/
  11,000/11,500/12,000 was 3.6261/3.6613/3.6328/3.6641/3.6619/3.5723/3.4737, decisively resolving
  the step-8,500 wobble and ending at a new best. All 12,000 rows are finite/consecutive and cover
  196,608,000 tokens; p10/median throughput is 3,721/3,848 tok/s, and all 121 allocator samples
  report 34 slabs and zero overflow. Checkpoints 9,000/10,000/11,000/12,000 at `7ce876e5…`/
  `9352634d…`/`ada7bf46…`/`61eccbe3…` were each hash-mirrored and native-audited with all
  57,688,576 parameters finite/nonzero; the exact 12,000-row metrics prefix matches at `5998c8cf…`.
  The guard safely pruned checkpoints 6,000/7,000/8,000/9,000 only after mounted proof and ledgered
  local deletion, leaving exactly 10,000/11,000/12,000 on both sides. All four saves released 228
  buffers; the step-12,000 immediate 6,694MB ArrayBuffers reading settled to the usual 6,632MB by
  step 12,050. Training is live through step 12,050. Account balance was `$51.1209876189` at
  approximately 01:49 UTC. The next unattended interval also passed cleanly through checkpoint
  22,000. All 22,000 rows are finite/consecutive and cover 360,448,000 tokens (36.0443% of contract);
  p10/median throughput is 3,734/3,865 tok/s, and all 221 allocator samples have the required
  maximum-100-step cadence, exactly 34 slabs, and zero overflow. Held-out loss at steps
  12,500/13,000/13,500/14,000/14,500/15,000/15,500/16,000/16,500/17,000/17,500/18,000/18,500/
  19,000/19,500/20,000/20,500/21,000/21,500/22,000 was 3.5704/3.5529/3.4786/3.5452/3.4795/
  3.5852/3.5613/3.5143/3.5247/3.5558/3.4463/3.4473/3.5148/3.4973/3.4778/**3.4008**/3.4168/
  3.4337/3.4184/3.4330; checkpoint 20,000 is the current best. Exact 22,000-row metrics match at
  `d3dc3886…`. Checkpoints 19,000/20,000/21,000/22,000 are exact remote/mounted matches at
  `c70d86af…`/`bc64cec9…`/`dafaddf7…`/`2b4d4df5…` and each passed the native all-57,688,576-parameter
  finite/nonzero scan. The 19,000 audit JSON was deliberately preserved before retention deleted the
  checkpoint. Earlier 13,000–18,000 checkpoints passed the guard's size/SHA mirror and append-only
  deletion ledger but had already been pruned before a retrospective native scan, so they are not
  overstated as native-audited. Both sides now retain exactly 20,000/21,000/22,000. All rows after
  step 12,000 hold ArrayBuffers within 6,631–6,632MB and RSS within 7,851–7,944MB; every subsequent
  save released 228 buffers directly to 6,631MB. Training resumed beyond 22,000 with the RTX 3090 at
  100% utilization, 24,112/24,576MiB, 67C, and about 7.9GB process RSS. Balance was
  `$47.7200922733` at approximately 13:36 UTC. Step 22,500 subsequently passed 22,500
  finite/consecutive rows and 368,640,000 tokens (36.8635%), p10/median 3,735/3,865 tok/s, all 226
  allocator samples, 34 slabs, zero overflow, and exact remote/mounted metrics hash `d25d85ff…`.
  Held-out loss 3.5532966 is a +0.1202539 single-gate wobble from step 22,000, but every
  numeric/memory/allocator invariant remains green; like the comparable step-8,500 wobble, it needs
  the next aligned checkpoint rather than an intervention. Checkpoint 23,000 is the discriminator.
  Balance was `$47.5512524306` at approximately 14:09 UTC. Checkpoint 23,000 subsequently resolved
  the wobble: held-out loss recovered to 3.4372567, only 0.0042140 above step 22,000. All 23,000
  rows are finite/consecutive and cover 376,832,000 tokens (37.6827%); p10/median is 3,734/3,864
  tok/s; all 231 allocator samples report 34 slabs/zero overflow. Exact metrics `32da13ab…` and
  checkpoint `746e14f4…` match remote/mounted; the checkpoint's native scan passed all 57,688,576
  parameters finite/nonzero, and the save released 228 buffers to 6,631MB. The guard safely pruned
  checkpoint 20,000 only after proof and now retains exactly 21,000/22,000/23,000 on both sides.
  Training resumed through 23,025; balance was `$47.3824188825` at approximately 14:46 UTC.
  Step 23,500 subsequently passed with 23,500 finite/consecutive rows and 385,024,000 tokens
  (38.5019%), p10/median 3,732/3,862 tok/s, all 236 allocator samples present, 34 slabs, zero
  overflow, and exact remote/mounted metrics prefix `82f84baa…`. Held-out loss 3.5132058 is
  +0.0759491 from step 23,000 but 0.0400908 better than step 22,500; advancing five-batch windows
  plus green numeric/memory/allocator invariants make this normal variance pending checkpoint
  24,000. Balance was `$47.2135497066` at approximately 15:21 UTC.
  Checkpoint 24,000 then passed with 24,000 finite/consecutive rows and 393,216,000 tokens
  (39.3211%), p10/median 3,731/3,861 tok/s, all 241 allocator samples present, 34 slabs, and zero
  overflow. Held-out loss improved to 3.4923591. Exact metrics `66e75c19…` and checkpoint
  `1c80ee85…` match remote/mounted; the native scan passed all 57,688,576 parameters finite/nonzero,
  and the save released 228 buffers to 6,631MB. The guard safely pruned checkpoint 21,000 only after
  mirror proof and now retains exactly 22,000/23,000/24,000 on both sides. The RTX 3090 returned to
  100% utilization; balance was `$47.0205838101` at approximately 16:00 UTC.
  Step 24,500 also passed with 24,500 finite/consecutive rows and 401,408,000 tokens (40.1402%),
  p10/median 3,730/3,860 tok/s, all 246 allocator samples present, 34 slabs, and zero overflow.
  Held-out loss was effectively flat at 3.4938740 (+0.0015149 from 24,000), and exact remote/mounted
  metrics match at `f7c2f6a6…`. Post-24k ArrayBuffers stayed exactly 6,632MB and RSS within
  7,878–7,937MB. Balance was `$46.8517209286` at approximately 16:34 UTC.
  Checkpoint 25,000 then passed with 25,000 finite/consecutive rows and 409,600,000 tokens (40.9594%),
  p10/median 3,730/3,859 tok/s, all 251 allocator samples present, 34 slabs, and zero overflow.
  Held-out loss improved to 3.4471820. Exact metrics `78e73346…` and checkpoint `8a86ca42…` match
  remote/mounted; the native scan passed all 57,688,576 parameters finite/nonzero, and the save
  released 228 buffers to 6,631MB. The guard safely pruned checkpoint 22,000 only after mirror proof
  and now retains exactly 23,000/24,000/25,000 on both sides. Balance was `$46.6829211138` at
  approximately 17:10 UTC.
  Step 25,500 also passed with 25,500 finite/consecutive rows and 417,792,000 tokens (41.7786%),
  p10/median 3,729/3,859 tok/s, all 256 allocator samples present, 34 slabs, and zero overflow.
  Held-out loss was effectively flat at 3.4464590, slightly better by 0.0007231 from 25,000. Exact
  remote/mounted metrics match at `b8dcc21a…`; post-25k ArrayBuffers stayed exactly 6,632MB and RSS
  within 7,857–7,937MB. Balance was `$46.5140116656` at approximately 17:45 UTC.
  Checkpoint 26,000 then passed with 26,000 finite/consecutive rows and 425,984,000 tokens (42.5978%),
  p10/median 3,728/3,858 tok/s, all 261 allocator samples present, 34 slabs, and zero overflow.
  Held-out loss improved to 3.4225069, only +0.0217091 from the run best. Exact metrics `c4222263…`
  and checkpoint `28b0050b…` match remote/mounted; the native scan passed all 57,688,576 parameters
  finite/nonzero, and the save released 228 buffers to 6,631MB. The guard safely pruned checkpoint
  23,000 only after mirror proof and now retains exactly 24,000/25,000/26,000 on both sides. Training
  resumed through 26,025; balance was `$46.3211066359` at approximately 18:23 UTC.
  Step 26,500 then established a new run-best held-out loss of 3.3790116, improving 0.0217862 from
  the prior step-20,000 best. All 26,500 rows are finite/consecutive and cover 434,176,000 tokens
  (43.4170%); p10/median is 3,728/3,858 tok/s; all 266 allocator samples report 34 slabs/zero
  overflow. Exact remote/mounted metrics match at `822c37d3…`; post-26k ArrayBuffers stayed exactly
  6,632MB and RSS within 7,854–7,930MB. Balance was `$46.1522706432` at approximately 18:57 UTC.
  Checkpoint 27,000 then extended the run-best held-out loss to 3.3680425, improving 0.0109691 from
  step 26,500 and 0.0327553 from the former step-20,000 best. All 27,000 rows are finite/consecutive
  and cover 442,368,000 tokens (44.2362%); p10/median is 3,728/3,858 tok/s; all 271 allocator samples
  report 34 slabs/zero overflow. Exact metrics `b1fe1b30…` and checkpoint `972902b7…` match remote/
  mounted; the native scan passed all 57,688,576 parameters finite/nonzero, and the save released
  228 buffers to 6,631MB. Safe retention removed checkpoint 24,000 only after proof and now holds
  exactly 25,000/26,000/27,000. Training resumed through 27,050; balance was `$45.983374884`.
  Step 27,500 then passed with 27,500 finite/consecutive rows and 450,560,000 tokens (45.0554%);
  p10/median is 3,728/3,858 tok/s; all 276 allocator samples report 34 slabs/zero overflow. Held-out
  loss 3.3710784 is only +0.0030359 from the step-27,000 run best and remains better than every
  earlier gate. Exact remote/mounted metrics match at `4ca7f954…`; post-27k ArrayBuffers stayed
  exactly 6,632MB and RSS within 7,861–7,931MB. Training resumed through 27,550; balance was
  `$45.8145234914`.
  Checkpoint 28,000 then passed with 28,000 finite/consecutive rows and 458,752,000 tokens (45.8746%),
  p10/median 3,728/3,858 tok/s, all 281 allocator samples present, 34 slabs, and zero overflow.
  Held-out loss 3.4247354 is +0.0566929 from the 27,000 best but only +0.0022285 from checkpoint
  26,000. Exact metrics `a3bb4acf…` and checkpoint `b9f80989…` match remote/mounted; the native scan
  passed all 57,688,576 parameters finite/nonzero. The save released 228 buffers and training
  returned to the 6,632MB ArrayBuffers baseline at step 28,001. Safe retention removed checkpoint
  25,000 only after proof and now holds exactly 26,000/27,000/28,000. Training resumed through
  28,050; balance was `$45.6456990488`.
  Step 28,500 then passed with 28,500 finite/consecutive rows and 466,944,000 tokens (46.6938%);
  p10/median is 3,728/3,857 tok/s; all 286 allocator samples report 34 slabs/zero overflow. Held-out
  loss improved 0.0300772 from checkpoint 28,000 to 3.3946582 and is only +0.0266157 from the
  step-27,000 run best. Exact remote/mounted metrics match at `d4764198…`; post-28k ArrayBuffers
  stayed exactly 6,632MB and RSS within 7,856–7,935MB. Training resumed through 28,525; balance was
  `$45.476881634`.
  PID 101700 remains alive at nice 5. The
  cache-aware matched-retention guard
  `alpha2-flagship-puller-e561f66-cacheaware.service` polls every 60s, permits a 7,200s startup window,
  and retains three size/SHA-verified checkpoints on each side. Local mirror:
  `/mnt/donto-data/alpha-runs/flagship-1b-e561f66-20260724/`; external remote log:
  `/workspace/alpha2-run-logs/flagship-1b-e561f66-20260724.train.log`.
- **Downstream SFT/eval inputs are now staged and hash-verified on the pod.** Under
  `/runpod/data/alpha-sft-v2/`, the 1,262,158,944-byte `sft-v2.txt`, manifest, length audit, mask
  audit, and 12,288-tokenizer match the canonical mounted artifacts at `ffad0a37…`/`e5d034ac…`/
  `1dc89d0f…`/`20c7a45f…`/`c310343a…`. Under `/runpod/data/frozen-eval-v1/`, `MANIFEST.json` and
  final chat/QA inputs match at `bf6e6ea4…`/`6c463deb…`/`bbbeec57…`. Transfer ran under
  nice/ionice with a 20MB/s ceiling while the trainer continued advancing; the pod retained 7.6GB
  free afterward. Do not re-transfer these unless a fresh SHA comparison finds drift.
- **First contracted LR pilot (`1e-3`) is COMPLETE.** Its strict summary passed 6,104/6,104
  consecutive finite rows and exactly 100,007,936 tokens: final train loss 3.6922, last-100 train mean
  3.5989, median post-warmup throughput 3,892 tok/s, and final-three held-out-loss mean 3.6045400
  (3.6270/3.6437/3.5430). All 63 allocator samples are complete with zero overflow. Terminal
  `checkpoint-6104.json` is a hash-mirrored/native-audited 692,528,815-byte ALPH file with all
  57,688,576 parameters finite/nonzero at SHA-256 `e43ce5a9…`; final metrics SHA-256 is
  `8f84060a…`. The guard retained exactly 5,000/6,000/6,104, logged `final pull complete`, and exited.
  Evidence: `/mnt/donto-data/alpha-runs/lr-sweep-llama-100m-lr1e3-e6d9430-20260723/RUN.md`.
- **Second contracted LR pilot (`2e-3`) is COMPLETE** at
  `/workspace/alpha2/runs/lr-sweep-llama-100m-lr2e3-e6d9430-20260723`, started 19:55 UTC on the same
  deliberately pinned `e6d9430` source/data/tokenizer contract. Its strict summary passed 6,104
  consecutive finite rows and exactly 100,007,936 tokens: median post-warmup throughput 3,843 tok/s,
  final train loss 3.7847, last-100 train mean 3.6857, and final-three held-out-loss mean 3.6954683
  (3.7241/3.7361/3.6263). All 63 allocator samples are complete with zero overflow. Terminal
  `checkpoint-6104.json` is a hash-mirrored/native-audited 692,528,815-byte ALPH file at SHA-256
  `ecb79332…`, with all 57,688,576 parameters finite/nonzero; final metrics SHA-256 is `1ed8bd01…`.
  The guard retained exactly 5,000/6,000/6,104, logged `final pull complete`, and exited successfully.
  Its final-three mean is 0.0909283 worse than `1e-3`. Evidence:
  `/mnt/donto-data/alpha-runs/lr-sweep-llama-100m-lr2e3-e6d9430-20260723/RUN.md`.
- **Third contracted LR pilot (`3e-3`) is COMPLETE** on the identical pinned contract: 6,104/6,104
  consecutive finite rows, exactly 100,007,936 tokens, median post-warmup throughput 3,862 tok/s,
  final train loss 4.1647, last-100 train mean 4.0918, and final-three held-out-loss mean 4.1337789
  (4.1705/4.1783/4.0526). All 63 allocator samples are complete with zero overflow. Terminal
  `checkpoint-6104.json` is a hash-mirrored/native-audited 692,528,815-byte ALPH file at SHA-256
  `18cdcec8…`, with all 57,688,576 parameters finite/nonzero; final metrics SHA-256 is `abb47676…`.
  The guard retained exactly 5,000/6,000/6,104, completed its final pull, and exited with status 0.
  Evidence: `/mnt/donto-data/alpha-runs/lr-sweep-llama-100m-lr3e3-e6d9430-20260724/RUN.md`.
- **Contracted LR selection PASS: `1e-3` selected.** All candidates match source `e6d9430`, data,
  tokenizer, model shape, steps, tokens, and allocator contracts. Final-three held-out-loss means rank
  `1e-3` 3.6045400, `2e-3` 3.6954683, `3e-3` 4.1337789. Canonical report:
  `/mnt/donto-data/alpha-runs/lr-sweep-analysis-e6d9430-20260724.json`, SHA-256
  `10d39e4791454ce2a88ee1273b6c6ecdc4d372577b11007e518ad62734b205a9`.
- All three sweep candidates stayed on `e6d9430`. Current-origin `e561f66` then built 19/19 and passed
  the real NVIDIA gate 46/46 with zero failed/skipped/todo. Four consecutive full flagship-size saves
  each released all 228 cloned optimizer buffers, ran host GC, and returned ArrayBuffers to the same
  2,705MB baseline. Every 692,528,809-byte checkpoint independently passed exact-header/payload and
  all-57,688,576-parameter finite/nonzero audits. Evidence:
  `/mnt/donto-data/alpha-runs/{nvidia-gate-e561f66-20260724,checkpoint-reclaim-4cycle-e561f66-20260724}/`.
- **G2 PASSED.** The 5,400-step flagship-shape soak completed cleanly at 20:44 UTC on commit `aca9f97`:
  5,400/5,400 finite rows, 88.47M tokens, literal 6h25m monitoring, p10/median 3,721/3,832 tok/s,
  RSS 681–767MB with negative slope, 34 constant temporary slabs, zero allocator overflow, full 692.5MB
  checkpoint. Every analyzer check is true. Evidence:
  `/mnt/donto-data/alpha-runs/g2-soak-wg64-b16-5400-20260722/{RUN.md,g2-analysis.json}`.
- At the end of G2 the host GPU attachment failed (`nvidia-smi` unknown + `vkCreateInstance` failure).
  Before touching the pod, 6.996GB of previously unmirrored runs plus every root log were copied to the
  mounted drive; two checksum-mode rsync dry runs were exactly empty. A RunPod container restart restored
  NVML and Vulkan. Exact tree `c95f81b` then passed the fail-closed NVIDIA gate: vendor `0x10de`, 46/46
  executed and passed, zero skipped/failed/todo. Evidence:
  `/mnt/donto-data/alpha-runs/nvidia-gate-c95f81b-attempt3/`.
- **G3 Llama COMPLETE.** The exact 100,007,936-token half finished normally at 04:51 UTC with
  6,104/6,104 consecutive finite rows, 57,688,576 parameters, median post-warmup throughput
  3,876 tok/s, final train loss 3.8499150, last-100 mean 3.7737795, and final held-out loss 3.7274671
  (last-three mean 3.7829017). Canonical `summarizePilot` passed all shape/contract/telemetry checks:
  63 allocator samples through terminal step, maximum gap 100, 34 slabs, zero overflow. Terminal
  checkpoint 6,104 is exactly 692,528,815 bytes and hash-identical on pod/mounted drive at SHA-256
  `65b3b1dc5f243746a7ce20dbbae6c97f2d503c37b422ef7bdddd2c7fc0f16b4c`; streaming audit proved
  exact ALPH payload/model and all 57,688,576 parameters finite/nonzero. Final metrics hash is
  `205ad25319245be4c7d82cc143513ab11071e5452103d0f4843a10e5372b3aee` on both sides. The guard
  safely retained exactly checkpoints 5,000/6,000/6,104, logged `final pull complete`, and exited.
  Evidence: `/mnt/donto-data/alpha-runs/g3-llama-100m-lr3e4-c95f81b-20260722/RUN.md`.
- **G3 GPT-2 COMPLETE and canonical pair gate PASS.** The unchanged `c95f81b` control completed at
  6,104/6,104 consecutive finite rows and exactly 100,007,936 tokens: 58,094,592 params, 4,704 tok/s
  median, final/last-100 train loss 4.0688343/3.9916938, final held-out loss 3.9457434, 63 complete
  allocator samples, 34 slabs, and zero overflow. Terminal checkpoint 6,104 is a hash-mirrored,
  native-audited 697,403,761-byte ALPH file with all parameters finite/nonzero at SHA-256
  `de8bc5579755b50235a0a534f7292b98f4ace7fe77383f1c52aa035037a6a553`; final metrics hash is
  `cbcb9ad2a3da4577ffc44d613a3c90cf4c7f526a2000b3985d601410c8daed58`. The guard retained exactly
  5,000/6,000/6,104 on both sides, logged `final pull complete`, and exited. Canonical pair analysis
  passed: contracts match, parameter difference 0.6989%, Llama won all 12 aligned validations, final
  advantage 0.2182763, last-three mean advantage 0.2302373, zero overflow in both runs. Report:
  `/mnt/donto-data/alpha-runs/g3-pair-analysis-c95f81b-20260723.json` (SHA-256 `1c6d26a0…`).
  Full evidence: `/mnt/donto-data/alpha-runs/g3-gpt2-100m-lr3e4-c95f81b-20260723/RUN.md`.
- The exact-pair pin, LR sweep, current-origin NVIDIA gate, and four-cycle checkpoint-reclamation proof
  are all satisfied. The selected `1e-3` flagship is now live on exact source `e561f66`.
- Checkpoint 2,000 in G3 first exposed delayed snapshot reclamation. The `e6d9430` scoping/GC telemetry
  deployment proved the remaining issue precisely at this LR pilot's step 1,000: one full checkpoint
  stayed reachable after GC. `3a7ff9d` now explicitly clears the cloned optimizer snapshot and serializer
  reference list in `finally`; the four-cycle RTX 3090 proof above confirms reclamation before flagship.
- The contracted flagship manifest and all three source shards are now staged under `/runpod/data`.
  Their exact aggregate size is 5,976,889,749 bytes and all remote SHA-256 values match the immutable
  manifest; 13GB remained free afterward. This was a low-priority transfer while the GPU stayed at 100%.
- RunPod balance was **$53.3641259029** at about 18:02 UTC; total account burn was $0.301/hr including
  unrelated stopped volumes. Never delete those unrelated pods. If abandoning this work, terminate this pod with
  `runpodctl remove pod d5m7h1v0kr0zd4`.

## Takeover progress (supersedes stale state later in this file)

- Current deployed/certified functional tree and live flagship source is **`e561f66`**. Its production
  build is 19/19 and the real RTX 3090 gate is **46/46 executed and passed**, zero skipped/failed/todo.
  Root `npm test` is pre-existingly broken
  because Turbo runs Vitest in empty packages; use `npm test -w @alpha/tests`.
- Current origin's latest non-documentation commit is **`08bec45`**. Its production build is green across all 19 buildable packages,
  TypeScript is clean, and the consolidated box suite is **202 pass / 46 GPU-gated skip / 0 fail**.
  `08bec45` repaired required model metadata in the checkpoint-lifecycle test fixture after the fresh
  typecheck exposed that test-only drift; it does not alter training behavior.
- NVIDIA gate work, G1, allocator wiring, and post-slab baseline are done and pushed: 46/46 NVIDIA tests;
  G1 1,000 steps with zero NaN; slab profile WG64/pool512; 57.69M-param baseline improved 3,322→3,790
  tok/s (+14.1%). Relevant commits: `e60391e`, `f595708`, `f7730c6`, `32392a5`, `9d7fbc9`, `aca9f97`.
- G4 data gate **passed**. Canonical SFT v2:
  `/mnt/donto-data/alpha-corpora/sft-text-v2/sft-v2.txt`, 511,428/511,428 clean, SHA `ffad0a37…`, exact
  p50/p95/p99/max 657/978/1,014/1,024 tokens, zero over-bound, SODA 4.828%, real assistant mask green.
  Frozen eval: 49 OASST2 validation + 48 Magpie + 3 everyday prompts, 200 QA, 1,500 validation docs;
  exact SFT audit scanned 205,027,527 13-grams and rejected 658/900; final overlap zero. See
  `docs/SFT_CORPUS.md`, `docs/FROZEN_EVAL.md`, and mounted `RUN.md`/manifests.
- G3 pilot launcher is pushed in `scripts/run_g3_pilot.sh`: equal 100,007,936 tokens, 57.69M Llama vs
  58.09M GPT-2 control. Commit `cc7f450` isolates train/validation RNGs across architectures, seeks all
  loader types on resume, writes an immutable pilot contract, and adds `analyze_g3_pair.ts`; `b97a810`
  enforces full-token corpus coverage, and `5e5b913` parameterizes the contracted LR sweep. `da39e8a`
  decouples frequent validation from full optimizer checkpoint cadence and proves the remote-retention
  guard: old remote copies are pruned only after byte-size + SHA-256 agreement with the mounted-drive
  mirror. `867f016` also pins every paid pilot architecture argument explicitly and makes the analyzer
  reject model-config drift; its 6,104-row synthetic contract proof passed. `58fc691` makes the pilots
  safely resumable: exact original contract required, post-checkpoint metric tails preserved+hashed,
  active metrics atomically realigned, and every attempt recorded in `resume-ledger.jsonl`. Do not start
  either pilot until the G2 soak finishes and its artifacts are archived.
- The LR sweep is now proof-gated too: `analyze_lr_sweep.ts` shares the strict pilot validator and selects
  among exactly `{1e-3,2e-3,3e-3}` by the final-three aligned held-out-loss mean (final loss/lower LR are
  deterministic tie-breaks). Its positive and contract-rejection synthetic tests passed in `61c1edb`;
  `59c62dd` additionally requires complete 100-step allocator telemetry through the final pilot row.
- `run_flagship_pretrain.sh` consumes the analyzer's hash-bound LR-selection report and launches exactly
  1,000,013,824 tokens over the verified three-shard manifest, with the explicit Llama/AdamW profile,
  independent eval/checkpoint cadence, immutable contract, and safe resume ledger. Contract-only positive
  and wrong-tokenizer/report-mutation rejection proofs passed without launching training.
- `analyze_flagship_pretrain.ts` is the terminal 1B-run gate: exact report/manifest/tokenizer/commit,
  architecture, optimizer, data coverage, 61,036 finite rows, 123 aligned validations, complete
  zero-overflow allocator telemetry, ≥3K tok/s p10/median, and a streaming native finite/nonzero audit of
  all 57,688,576 terminal parameters.
- `analyze_flagship_sft.ts` is the matching terminal chat-run gate: exact SFT selector/input/commit,
  30,322 finite rows, 61 aligned validations, complete zero-overflow allocator telemetry, and a
  hash-bound reuse of `verify_flagship_sft_inputs.ts` to scan every terminal chat parameter.
- SFT resume now preserves the fresh run's base `initCheckpointPath` when `config.json` is rewritten,
  records the active `resumePath`, and refuses SFT resume if the existing origin provenance is absent.
- Base→SFT initialization is no longer conflated with resume: `--initCheckpoint` (`55c86db`) validates
  and restores weights only, resets the declared RNG, and starts a fresh optimizer/schedule at step zero;
  it is mutually exclusive with continuation `--resume` and has bit-identical parameter proof at LR zero.
- Token caches are artifact-bound and crash-safe in `45bfe60`: exact tokenizer SHA in the key, checked
  chunked I/O, source mtime+size header, fsync+atomic rename, and automatic truncated-cache recovery.
- Checkpoint compatibility in `6b460e4` covers semantic architecture, not just dimensions: norm type,
  positional encoding, RoPE theta, embedding tying, and soft-cap mismatches all fail closed.
- Flagship SFT is contracted in `7636ad2`: `verify_flagship_sft_inputs.ts` independently streams the
  511,428-row corpus, reconciles both passed audits, derives the exact 485,150/26,278 split, and verifies
  every base-checkpoint parameter byte is present and finite. `run_flagship_sft.sh` admits only the
  `{1e-4,3e-4,1e-3}` sweep, launches exactly one 30,322-step assistant-only epoch, separates weight-only
  initialization from continuation resume, and records immutable hashes. Real corpus + step-100 fixture
  proof passed; wrong-base-step rejection passed; TypeScript and 200/46 consolidated gates stayed green.
- Long-run checkpoint growth is bounded in `99a9116`. Use matching
  `REMOTE_KEEP_CHECKPOINTS=3 LOCAL_KEEP_CHECKPOINTS=3`: remote deletion still requires local byte+SHA
  proof, then local pruning keeps the newest three and fsyncs before/after deletion records (including
  the removed hash) to `checkpoint-prune-ledger.jsonl`. Counts below three or mismatched policies fail
  before SSH. The isolated six-checkpoint fixture retained 4–6, ledgered+removed 1–3, and was idempotent.
- Frozen base-vs-chat evaluation is tamper-evident in `863427f`: v2 summaries hash both detailed JSONL
  outputs and bind EOS/user control IDs; `analyze_frozen_eval_pair.ts` recomputes all 100 chat + 200 QA
  flags/scores, requires exact 61,036/30,322-step checkpoints and identical frozen inputs/case order,
  binds both runs to the canonical final `MANIFEST.json` chat/QA hashes, and enforces the ≥95 structural /
  zero-loop machine bar. Its PASS explicitly leaves conversational
  coherence to separate semantic review. Full synthetic pair passed; altered output hash was rejected.
- Post-G2 NVIDIA regression is fail-closed in `1019b9b`. Run
  `scripts/run_nvidia_gates.sh /workspace/alpha2/runs/nvidia-gate-<commit>` after deploying current
  master; it requires vendor `0x10de` and the exact two files / 46 unique assertions / 46 passed / zero
  skipped-failed-todo, then hashes the Vitest JSON into `gate-summary.json`. The real local all-skipped
  report was rejected, a synthetic 46/46 report passed, and non-NVIDIA preflight stopped before tests.
- SFT LR selection is executable rather than aspirational in `b24c18a`: three sequential
  `run_sft_lr_pilot.sh` runs each consume exactly 2,000 steps / 32,768,000 padded tokens with eight
  aligned validations and the identical verified corpus/audits/tokenizer/base. The selector requires
  complete finite runs, zero allocator overflow, full checkpoints, identical inputs+commit, and ranks
  final-three held-out loss. `run_flagship_sft.sh` now refuses to start without the matching report and
  verifies its selected LR plus every input hash. Positive and mismatch synthetic proofs passed.
- Immediate order: (1) let the healthy guarded Llama run finish all 6,104 rows; (2) verify its final
  checkpoint/mirror and launch the GPT-2 half on the same exact commit/input/LR; (3)
  compare with `analyze_g3_pair.ts`; (4) run the three-way LR sweep; (5) resumable flagship, SFT, frozen
  eval, HF upload. Host disks are unexpectedly full (root 97%, data 87% at 21:10); avoid
  unbounded artifacts and do not destructively clean without resolving exact targets.
  The analyzer also requires the final full model+AdamW checkpoint to be 650–750 MiB (`ddd9bd3`), so a
  nonempty/truncated placeholder cannot satisfy G2.

## Mission in one line

Train a small chatty model **entirely with Alpha's own from-scratch stack** (TS tensor lib, tape
autograd, hand-generated SPIR-V, Helios Vulkan backend — GPU-resident, no PyTorch/CUDA training) on
RunPod, publish to Hugging Face as a **standard zero-custom-code `LlamaForCausalLM`** repo
(`ajaxdavis`, HF auth on box verified WRITE-capable). Operator soul constraint (2026-07-22): every
training FLOP through Alpha's own code. User also directed: **all box-side code work before GPU spend**
— that is now DONE; user has explicitly said to move to GPU ("this box can't handle it" — do NOT run
more CPU training on the box).

## State of the repo (github.com/thomasdavis/alpha2, master, all pushed)

Working tree CLEAN at `84c110c`. Key commits this program (chronological):

- `9524598` GOAL.md + proven RunPod/Vulkan bootstrap (`scripts/runpod_bootstrap.sh`, `docs/RUNPOD.md`)
- `59d79da` **G0 PASSED**: Helios trained on a RunPod 3090 (60 steps, loss 7.28→7.05, 0 NaN, ~40K tok/s
  at 1.33M params, DGC+BDA+coop active). Artifacts: `/mnt/donto-data/alpha-runs/g0-smoke-20260722/`
- `aea174c` deps → latest (TS 7.0.2, vitest 4, Next 16, ai v7, effect 3.22) + 4 known-bug fixes
  (lmHead no-decay name, `--vocabSize` silently ignored, fp16 auto-enable trap, train-nanochat lr
  6e-4→3e-4) + secrets scrub (Discord webhook REVOKED via DELETE 204; `movies/symbio-film/.env`
  untracked — **ElevenLabs key still needs USER dashboard rotation**, it's in public git history)
- `0222fb3` npm audit 5→1-low (overrides sharp/postcss/esbuild; remaining 1 is Windows-only dev-server)
- `9b63685` **Stage-1 gradcheck harness** (see below) + REAL bug fix: `cpu_ref.sum(axis, keepdims=true)`
  was wrong on non-last axes → corrupted broadcast backward grads. Found by the harness day one.
- `fcfa83a` corpus builders (`scripts/build_pretrain_corpus.py`, `scripts/build_sft_corpus.py`)
- `b3ffe90` **Stages 3–4 box-side** (the big one, ~3,200 lines — see below)
- `84c110c` e2e script self-containment fix + final-tree golden numbers

### Test/verification state (all on the FINAL committed tree)

- `nice -n19 npx tsc -b` from root: clean. Full turbo build: 19/19.
- `packages/tests`: **178 passed / 44 GPU-gated skips / 0 failed** (~80–150s wall on the loaded box).
  The 44 skips are `parity-helios.test.ts` (36) + `gpu-perf.test.ts` — they gate on NVIDIA vendorId
  0x10de and have **NEVER run on real NVIDIA hardware**. Running them is the top next step.
- **Golden-token gate (G3 export half) PASSED on final tree**: tiny Llama-form model trained on
  cpu_ref → `alpha export-hf` → loaded by `transformers` (fp32, no trust_remote_code):
  **75/75 top-1 = 100%, max |Δlogit| 1.07e-06** (threshold 1e-3), tokenizer parity 4/4 prompts exact.
  Reproduce: `bash scripts/e2e_hf_export.sh` (self-contained; caches its checkpoint under
  `/mnt/donto-data/alpha-runs/g3-e2e/`). **But do NOT re-run CPU training on the box — user said stop.**
- Byte-BPE exporter cross-verified vs Python `tokenizers` on 9,822 real corpus docs: 100% id agreement.

### What Stages 3–4 added (commit `b3ffe90`)

All config-gated; legacy GPT-2-style configs bit-for-bit unchanged.

- **Arch**: `rmsNorm` (+fused backward) and `rope` ops — cpu_ref + autograd + Helios SPIR-V kernels.
  RoPE is EXACTLY HF `rotate_half` (half-split, `inv_freq=θ^(-2i/D)`) so export needs no permutation;
  backward = rotation by −angle (reuses forward kernel with negated sin). Tied embeddings via
  `lmHead === wte` object identity. `ModelConfig`: `normType`/`posEnc`/`ropeTheta`/`tieEmbeddings`.
  softCap defaults OFF under rope. New domain **`alpha_llama`** = 16L/512d/8H swiglu(1408) rmsnorm rope
  tied, block 1024, tokenizer `bpe-byte-12k` (~60M params — the flagship shape).
- **Tokenizer**: `ByteBpeTokenizer` (`packages/tokenizers/src/byte_bpe.ts`) — 256-byte base (exact GPT-2
  bytes_to_unicode), GPT-2 split regex, lossless decode on anything, atomic chat specials
  `<|user|>` `<|assistant|>` `<|end_of_text|>` (ids 256/257/258). Registry: `bpe-byte-12k`, `bpe-byte-4k`.
  HF exporter (`export_hf.ts` + `alpha tokenizer export-hf`) emits tokenizer.json/tokenizer_config.json/
  chat_template.jinja.
- **SFT loss masking**: `DataBatch.lossMask` + `crossEntropyMasked` (cpu_ref + fused Helios masked-CE
  kernels), SFT loader mode (one conversation per row, assistant-span-only mask, `--sft` on train cmd),
  trainer threads mask through grad-accum + eval.
- **HF export**: `packages/train/src/hf_export.ts` (spec-exact TS safetensors writer; ALPH→Llama state
  dict: wqkv split to q/k/v, fc_gate/up/proj→gate/up/down, NO transposes — `[out,in]` matches nn.Linear;
  omits lm_head when tied) + `alpha export-hf` + `alpha logits` CLI + `scripts/verify_hf_export.py`
  (golden verifier; py deps live in the uv venv at `/mnt/donto-data/alpha-corpora/.venv`: torch-cpu,
  transformers, safetensors, tokenizers, pyarrow).
- **Inference engine** (`packages/inference`): now supports rope/rmsnorm/tied (crash on Llama-form
  checkpoints fixed) + inference-parity tests. This unblocks the HF Space / `alpha sample` fast path.
- **Adversarial-review fixes (measured, not guessed)**:
  - P0: Helios masked-CE kernels had Out/Mask bindings swapped → GPU SFT would have trained on
    garbage SILENTLY (loss exactly 0). Fixed kernel-side (Out = last binding); documented in
    `kernels/nn.ts` next to `ce_fwd_masked`.
  - P1: flash-attention q/k/v used a PLAIN reshape `[B,T,nH*hd]→[B*nH,T,hd]` that scrambles
    (batch,head) rows for nHead>1 — PRE-EXISTING bug affecting all prior multi-head flash training;
    now reshape→transpose(1,2)→reshape head-major (see `gpt.ts` "[defect P1]" comment). RoPE positions
    on the flash path are now correct.
  - P2: shared-memory reduction races in CE/layerNorm/rmsNorm kernels — trailing ControlBarriers added
    (12 sites). Masked by NVIDIA warp lockstep; exposed on relaxed schedulers (rmsNorm dx diverged ~4e3
    on llvmpipe without it).
  - P3 (= the inference-engine fix above).

### Stage-1 harness (commit `9b63685`) — how correctness is enforced

`packages/tests/src/`: `gradcheck-ops.test.ts` (central-difference FD checks for EVERY op the model
uses; reusable `checkGrad`), `gradcheck-model.test.ts` (whole tiny-GPT gradchecks across
swiglu/gelu/universal/kan_spline AND the Llama-form config; top-|grad| element sampling; dead-param +
determinism + checkpoint-bitwise invariants), `optimizer-reference.test.ts` (AdamW vs independent
reference <1e-6), `parity-helios.test.ts` (GPU-gated CPU↔Helios parity: per-op fwd/bwd, tiny-GPT logits/
grads/AdamW-step, 100-step zero-NaN loop, f16 casts, rmsNorm/rope parity, tied-model loop, masked-CE).
**Any new op MUST get: cpu_ref impl + autograd backward + Helios kernel + FD gradcheck + parity test.**
Every check was proven load-bearing by temporary fault injection.

## Training data (READY, on the data disk)

- Pretrain: `/mnt/donto-data/alpha-corpora/pretrain-text/` — 6 shards ≤2GB, 11.7GB, ~3.0B est tokens,
  1.86M docs, `<|end_of_text|>`-delimited. Source: 4 parquet shards of
  `HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled` (kept in
  `premix-shuffled/`; 96 more shards available upstream if more tokens needed). All six outputs are
  sealed and re-verified by `pretrain-text/MANIFEST.sha256`; see the adjacent `RUN.md`. The minimum
  flagship uses `flagship-1b-manifest.json` (first three shards, 5,976,889,749 verified bytes) through
  the deterministic sharded loader in `28c6506`, avoiding both data repetition and giant-buffer limits.
- SFT: `/mnt/donto-data/alpha-corpora/sft-text-v2/sft-v2.txt` — 511,428 structurally clean,
  tokenizer-bounded conversations; SHA-256 `ffad0a376c7eac2e0ec91f0901ec1ff87cba67cc298222828ce3df1a3e60b3fb`.
  The previous unbounded version is preserved under that corpus directory's `history/`.
- Tokenizer: durable canonical artifact
  `/mnt/donto-data/alpha-runs/tokenizers-20260722/g2-bpe-byte-12k.json`; SHA-256
  `c310343a185aecb572b8b6568b55179df248f4adec009d14a9496da354090b24`. It was built on the pod from
  the 128MB pretrain slice, used by G2, then mirrored and local/remote hash-verified. See its `RUN.md`.

## Infra / credentials (all verified working this session)

- **RunPod**: `runpodctl` 2.6.1 configured (`~/.runpod/config.toml`); GraphQL with the same key.
  SSH key `~/.runpod/ssh/runpodctl-ssh-key`. Community RTX 3090 = proven Vulkan host class.
  Prices (community): A5000 $0.16 · 3090 $0.22 · A40 $0.30-0.35 · 4090 $0.34.
- **Vulkan-on-RunPod recipe (PROVEN)**: `scripts/runpod_bootstrap.sh` — driver-matched NVIDIA `.run`
  userspace install (`--no-kernel-modules`, kmod stubs) + **EGL headless ICD** + `VK_ICD_FILENAMES` +
  ctypes probe. Full runbook `docs/RUNPOD.md`. If the probe fails on a host: TERMINATE AND REDEPLOY,
  never debug a bad host. Community-host egress is a lottery: apt (port 80) and github may be dead;
  nodejs.org + download.nvidia.com (443) worked everywhere so far. Deploy code via **rsync from the
  box** if git clone fails (`--exclude=.git --exclude=.next --exclude=.turbo`; sync `packages apps`
  first if in a hurry — node_modules is 1GB and rsyncs alphabetically; full sync ~30-45 min under box
  I/O load).
- **Hugging Face**: `hf` CLI is currently authenticated as `ajaxdavis` (write verified by probe
  create+delete). The canonical mounted verification environment is
  `/mnt/donto-data/alpha-corpora/.venv` with Python 3.11.15, Transformers 5.14.1, CPU Torch 2.13.0,
  Safetensors 0.8.0, and Hugging Face Hub 1.24.0. A fresh
  `pipeline("text-generation", model="/mnt/donto-data/alpha-runs/g3-e2e/hf")` cold-load succeeded
  with zero custom code and generated output. The system Python lacks Transformers; use this venv for
  final Hub cold-load verification rather than installing onto the root disk.
- **Box rules**: shared multi-tenant box — EVERYTHING niced (`nice -n19`, `ionice -c3` for I/O);
  temp files under `$CLAUDE_JOB_DIR/tmp`, NOT bare /tmp; research artifacts under /mnt/donto-data;
  **no more CPU training on the box (user directive)**. lint-staged pre-commit runs a full turbo build
  and TIMES OUT under load — if you've manually verified build+tests, `git commit --no-verify` and say
  so in the message. Commit + push often (user directive; memory `feedback_commit_push_often`).
- Node runtime ONLY for Helios (`node --expose-gc apps/cli/dist/main.js`); the bun compiled binary has
  a known vkCreateInstance failure. Always `--fp16=false` posture (fp16 auto-enable removed, but be
  explicit); `HELIOS_DISABLE_COOP_MAT=1` for training stability per docs.

## NEXT STEPS (in order — the pod is waiting)

1. **Bootstrap the live pod** (`d5m7h1v0kr0zd4`, 64.119.209.250:8865):
   `scp scripts/runpod_bootstrap.sh` → run it → expect `vkCreateInstance OK, 1 device(s)`.
2. **Deploy code**: try `git clone https://github.com/thomasdavis/alpha2 && npm install` on the pod
   (repo is public + fully pushed); if egress is broken, rsync from
   `/mnt/donto-data/workspace/alpha2/` incl. node_modules. Then `nice npm run build` (or
   `npx turbo build --filter=@alpha/cli --filter=@alpha/tests` — much faster, skips the web app)
   and `node packages/helios/native/build.mjs` if the box-built addon doesn't load (`ldd` it).
3. **Run the GPU gates** (never executed on NVIDIA — this is the payoff of all the box work):
   - `cd packages/tests && npx vitest run parity-helios gpu-perf` → **all 44 must pass.**
     Watch specifically: masked-CE parity (P0 fix), rmsNorm/rope parity, the f16 cast tests,
     the tied-model 20-step loop, flash-vs-standard after the P1 relayout.
   - **G1 pilot**: 1,000 steps, ~10M params (e.g. 6L/256d/4H alpha_llama-style, bpe-byte-4k) on a
     pretrain shard slice, f32, helios — **gate: ZERO non-finite gradient steps** (the old 2-7% NaN
     era must be provably over; the SwiGLU/Helios interaction root-cause may surface here — if NaNs
     appear, bisect with the parity suite, do NOT mask with spike-skip).
   - **G2 baseline measurement**: 100-200 steps at the flagship `alpha_llama` shape (16L/512d, block
     1024, batch to fit 24GB) — record tok/s + live-alloc telemetry. Expect ~1K tok/s (allocator-bound).
     This anchors Stage 2.
   - Pull all runs/logs to `/mnt/donto-data/alpha-runs/` (box-side puller loop in docs/RUNPOD.md),
     then TERMINATE the pod. Update GOAL.md gates + ledger + commit.
4. **Stage 2 (the throughput unlock — biggest remaining engineering)**: wire device-local slab
   (TS never passes `temporary=1`; native slab code exists in `helios_vk.c` but is bypassed →
   every device tensor is an individual vkAllocateMemory → GC storms ≥192d). Gate G2:
   **≥3,000 tok/s sustained at flagship shape + 6h soak, zero allocator crashes**. Budget math:
   1B tokens @3K tok/s ≈ 93 GPU-h ≈ $20 on the 3090.
5. **Stage 5 flagship** per GOAL.md: lr sweep {1e-3, 2e-3, 3e-3} at 100M-token pilot scale (the old
   3e-4 lore predates the bug fixes — re-derive), then ~60M pretrain on 1-3B tokens (budget-gated by
   measured tok/s), then masked SFT on the built corpus, frozen evals (GOAL D3 chat bar).
6. **Stage 6 ship**: `alpha export-hf` the flagship → `hf upload ajaxdavis/alpha-60m-base` +
   `-chat` → `pipeline()` cold-load verify → model cards with honest evals + data licenses
   (ODC-BY/Apache-2.0/CC-BY-4.0 attribution). GGUF is a stretch (needs the `get_vocab_base_pre`
   patch, pre name "gpt-2").

## Known gaps / watch-outs

- **flash-attention on GPU**: P1 relayout is committed but flash parity has never run on NVIDIA.
  The parity suite covers it; if flash still diverges from standard on the 3090, train with the
  standard path (flash is a perf optimization) and file it for Stage 2.
- The trainer's in-loop sample subprocess uses cpu_ref — fine (tiny), but `--sampleInterval` large
  keeps pod CPU free.
- `alpha_llama` lr 3e-4 in domains.ts is a PLACEHOLDER — Stage 5 sweeps it.
- Data loader holds the whole tokenized corpus in RAM (Int32 = 4 bytes/token): 1B tokens = 4GB,
  3B = 12GB — check pod RAM at create (3090 hosts vary; ask for ≥32GB vCPU RAM if running 3B).
- `MAX_STRING_BYTES` 30MB / 10MB-chunk tokenization is handled by `loadAndTokenize` — corpus shards
  are already ≤2GB each, fine.
- Eval set (GOAL Stage-4 item) is NOT yet frozen: before the flagship run, build the fixed
  100-chat-prompt + 200-question + repetition/EOS suites (smol-smoltalk test split is reserved for
  this). Don't let benchmark data into training mixes.
- The box `.venv` for python verify work: `/mnt/donto-data/alpha-corpora/.venv` (activate then run
  `scripts/verify_hf_export.py` / `verify_tokenizer_export.py`).
- OUTSTANDING USER ACTIONS (already flagged, don't nag): rotate ElevenLabs key; decide fate of the
  4 stopped migmaq pods.

## The one-paragraph story so far

In one day the project went from a 4-month-dormant repo to: proven Vulkan-on-RunPod (G0: Helios trained
on a $0.22/hr 3090, zero NaN), a fully modernized toolchain (TS7/vitest4/Next16), a fault-injection-
proven gradient-checking harness that immediately caught a real broadcast-gradient bug plus a GPU-SFT-
would-have-been-garbage kernel-binding bug and a scrambled-flash-attention layout bug, a Llama-form
architecture (RoPE/RMSNorm/tied/byte-BPE) whose exports load in stock `transformers` at 100% top-1
agreement, assistant-only loss masking, 3B tokens of pretrain text + 457K SFT conversations staged,
and ~$0.4 of the $70 GPU budget spent. The remaining path to a shipped model is: GPU gates → slab
allocator throughput work → flagship pretrain+SFT → `hf upload`.
