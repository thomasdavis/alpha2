# Alpha chat-model resume dossier

> **New agent entry point (2026-08-03):** read the
> [comprehensive Alpha and Helios handoff](HANDOFF-TO-NEXT-AGENT-2026-08-03.md) first. It is the current
> self-contained transfer package; the rest of this directory supplies its deeper contracts and evidence.

> **HANDOFF (2026-08-03):** the performance research index is
> [alpha-handoff-2026-08-03.md](alpha-handoff-2026-08-03.md), also at
> https://alpha.donto.org/research/alpha-handoff-2026-08-03.html — every document, every result
> with its status, every experiment script, reproduction instructions with data hashes, and the
> ordered next actions. Read it before proposing any performance work.

> **Performance program (2026-08-03):** a standing goal to cut training cost by >=10x is recorded in
> [GOAL-EXTREME-PERFORMANCE-2026-08-03.md](GOAL-EXTREME-PERFORMANCE-2026-08-03.md), with the research at
> `/mnt/donto-data/donto-resources/research/alpha-helios-reimagined/`. Read it before proposing any kernel
> candidate: it measures Helios at **5.74% of the 4090's FP32 peak** (a 12–22x gap), diagnoses the
> unattributed half of the step as **host-bound and unoverlapped**, shows the contracted batch size is
> **2.3x above the gradient noise scale**, and closes four directions by measurement — linear/oscillator
> attention, FMM attention at S=1024, any attention replacement at S=1024, and low-rank weights.

> **Current-project note (2026-08-03):** the original chatty-model goal is active. V11 and V12 remain rejected
> negative results. The foundation LR pilot completed and selected `0.002`; no full foundation run has begun.
> Exact Helios profiling then identified generic GEMM as the immediate bottleneck, and the first portable
> register-blocked kernel raised matched steady throughput from 3,579 to 4,513 tokens/s without changing the
> printed training trajectory. Engine optimization and physical AMD enablement remain active before the
> multi-day run. There is no new behavioral model improvement or publication yet.

- **Corrective run:** complete
- **Selected checkpoint:** step 1,200, SHA `399f776b49acc0c8834ff8a7f2390454e2c5f2d833a264e3f83ff546e973cfec`
- **Development verdict:** structurally chatty, semantically immature
- **Frozen result:** 55/100 structural, 70/100 nonempty, 31 loops, QA 0/200; quality gate FAIL
- **Repair v2:** continuation and clean-base control both rejected; no new selection; sealed final untouched
- **V11:** all-token bridge completed; 615/615 development replies but more loops; blinded reviewer selected NONE
- **BLAH:** V11 mean 0.3625 versus earlier Alpha 0.395833; quality gate FAIL
- **Publication:** V11 has a distinct HF repo, runtime path, and BLAH model ID; older entries were not overwritten

This directory is the shortest safe route into the model program. It preserves the archived failure, the later
corrective experiment, and the evidence required before any further paid run.

## Read order

1. [CURRENT-STATE.md](CURRENT-STATE.md) — current repository, runtime, Hub, BLAH, pod, and authority state.
2. [HELIOS-PROFILER-REGISTER-BLOCKING-EVIDENCE-2026-08-03.md](HELIOS-PROFILER-REGISTER-BLOCKING-EVIDENCE-2026-08-03.md) — exact GPU-time attribution and first portable-kernel gain.
3. [HELIOS-OPTIMIZATION-AND-AMD-PROGRAM-2026-08-03.md](HELIOS-OPTIMIZATION-AND-AMD-PROGRAM-2026-08-03.md) — operation-by-operation published, portable, and novel research lanes.
4. [FOUNDATION-CANDIDATE-FEASIBILITY-2026-08-02.md](FOUNDATION-CANDIDATE-FEASIBILITY-2026-08-02.md) — measured architecture economics and exact LR-pilot contract.
5. [HELIOS-CHAT-THROUGHPUT-SWEEP-OUTCOME-2026-08-02.md](HELIOS-CHAT-THROUGHPUT-SWEEP-OUTCOME-2026-08-02.md) — earlier correctness-gated throughput result and optimization attribution.
6. [CHAT-RECIPE-V12-LR1E3-OUTCOME.md](CHAT-RECIPE-V12-LR1E3-OUTCOME.md) — final same-dataset control rejection.
7. [SAME-DATASET-RECIPE-AUDIT-2026-08-02.md](SAME-DATASET-RECIPE-AUDIT-2026-08-02.md) — correction from public same-corpus and same-Smoltalk training evidence.
8. [CHAT-RECIPE-V12-CONTRACT.md](CHAT-RECIPE-V12-CONTRACT.md) — frozen clean-base packed full-sequence replication contract.
9. [CHAT-FOUNDATIONS-V11-OUTCOME.md](CHAT-FOUNDATIONS-V11-OUTCOME.md) — prior completed training, semantic rejection, versioned publication, and BLAH evidence.
10. [CHAT-REPAIR-V3-LOCAL-PREFLIGHT-2026-08-01.md](CHAT-REPAIR-V3-LOCAL-PREFLIGHT-2026-08-01.md) — earlier local RCR-UL intervention and frozen inputs.
11. [CHAT-REPAIR-V2-2026-07-31.md](CHAT-REPAIR-V2-2026-07-31.md) — rejected continuation and clean-base controls.
12. [CHAT-REPAIR-2026-07-31.md](CHAT-REPAIR-2026-07-31.md) — corrective checkpoint and failed final gate.
13. [SESSION-START.md](SESSION-START.md) — the first-session checklist and hard stops.
14. [FAILURE-ANALYSIS.md](FAILURE-ANALYSIS.md) — why outputs sometimes looked conversational but mostly
   terminated immediately.
15. [CHECKPOINT-CATALOG.md](CHECKPOINT-CATALOG.md) — exact recoverable native checkpoints and hashes.
16. [EVIDENCE-INDEX.md](EVIDENCE-INDEX.md) — canonical reports, samples, manifests, and screenshots.
17. [DECISIONS.md](DECISIONS.md) — binding operator decisions that must not be silently reversed.
18. [EXPERIMENT-BACKLOG.md](EXPERIMENT-BACKLOG.md) — later experiments, not automatic authorization.
19. [ACCEPTANCE-GATES.md](ACCEPTANCE-GATES.md) — proof required before spending, continuing, or publishing.
20. [RUNPOD-RECOVERY.md](RUNPOD-RECOVERY.md) — future recovery only after renewed authorization.
21. [SERVING-OPERATIONS.md](SERVING-OPERATIONS.md) — operation of the public model artifacts.

Then read the repository-level [GOAL.md](../../GOAL.md), [HANDOFF.md](../../HANDOFF.md),
[docs/RUNPOD.md](../RUNPOD.md), and [docs/FROZEN_EVAL.md](../FROZEN_EVAL.md). GOAL.md and HANDOFF.md
contain the complete chronological record; this directory is the compact recovery layer.

## Facts that must survive every handoff

- Alpha trained the model successfully through its own TypeScript, Vulkan, autograd, tokenizer, and
  checkpoint stack.
- The archived terminal checkpoint failed: 92 empty responses, six loops, two unusable fragments, and QA 0/200.
- The later corrective checkpoint 1,200 produced nonempty, EOS-terminated replies on all 48 repair-development
  cases, but the untouched suite fell to 55/100 structural, 70/100 nonempty, and 31 loops.
- Repair v2 made all 96 selector responses nonempty, but both continuation and clean-base training increased
  repetition on exact shared prompts. No v2 checkpoint was selected and the sealed-final suite was not opened.
- Repair v3 is locally ready but unexecuted: its 4,096 rollout cohort, fresh96/panel24/eligible69 evaluation
  contract, paired objective, and selection machinery are frozen; no NVIDIA proof, trained candidate, or quality
  improvement exists.
- Later experiments through V11 did not establish a semantic gain. V11 made every development response nonempty
  and EOS-terminated but increased repetition, lost its blinded comparison to V8, and scored below the earlier
  Alpha on BLAH.
- Every future BLAH publication increments the Alpha version and creates a new model record. Never repoint an
  already evaluated BLAH entry to a new checkpoint or runtime behavior.
- The evaluator and serving prompt also inserted a standalone generation-only space after the assistant marker.
  Commit `cf4ad61` fixes that boundary; historical failed outputs remain preserved.
- The paid Alpha pod was removed after final evaluation was copied, hashed, and recomputed.
- The standard Hugging Face model is inference-only. Future training must use a native ALPH checkpoint
  from the recovery archive because it carries AdamW, RNG, tokenizer, and step state.
- Failed evaluations, abandoned trajectories, and old metrics are evidence. Never overwrite or
  relabel them as a successful release.

## Public artifacts

- Model: https://huggingface.co/ajaxdavis/alpha-60m-chat
- Native recovery archive: https://huggingface.co/ajaxdavis/alpha-60m-training-checkpoints
- Space: https://huggingface.co/spaces/ajaxdavis/alpha-60m-chat
- Exact CPU backend health: https://donto.org/alpha-60m/health
- V11 negative-result model: https://huggingface.co/ajaxdavis/alpha-chat-v11-m300-experimental
- V11 versioned backend health: https://donto.org/alpha-v11-m300/health
- V11 BLAH model: https://evals.blah.dev/models/Mq5PrXS1MUk2yl0eSKUXwA

The Space must remain deliberately honest, expose exact model output, and use no fallback model. Serving the
artifact is not continuation training.
