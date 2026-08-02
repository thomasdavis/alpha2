# Alpha chat-model resume dossier

> **Current-project note (2026-08-02):** the original chatty-model goal is active. V11 completed and was
> rejected as an improvement. Its step-300 checkpoint is public only as a separately versioned negative result.
> Start with the V11 outcome and current state. The next training intervention is a V12 linked contrast-family
> synthetic curriculum; do not continue V11 or run its Phase S.

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

1. [CHAT-FOUNDATIONS-V11-OUTCOME.md](CHAT-FOUNDATIONS-V11-OUTCOME.md) — latest completed training, semantic rejection, versioned publication, and BLAH evidence.
2. [CURRENT-STATE.md](CURRENT-STATE.md) — current repository, runtime, Hub, BLAH, and authority state.
3. [CHAT-FOUNDATIONS-V11-CONTRACT.md](CHAT-FOUNDATIONS-V11-CONTRACT.md) — the predeclared V11 hypothesis and stop gate.
4. [CHAT-FOUNDATIONS-V10-OUTCOME.md](CHAT-FOUNDATIONS-V10-OUTCOME.md) — corpus-generation and assistant-only precursor evidence.
5. [CHAT-REPAIR-V3-LOCAL-PREFLIGHT-2026-08-01.md](CHAT-REPAIR-V3-LOCAL-PREFLIGHT-2026-08-01.md) — earlier local RCR-UL intervention and frozen inputs.
6. [CHAT-REPAIR-V2-2026-07-31.md](CHAT-REPAIR-V2-2026-07-31.md) — rejected continuation and clean-base controls.
7. [CHAT-REPAIR-2026-07-31.md](CHAT-REPAIR-2026-07-31.md) — corrective checkpoint and failed final gate.
8. [SESSION-START.md](SESSION-START.md) — the first-session checklist and hard stops.
9. [FAILURE-ANALYSIS.md](FAILURE-ANALYSIS.md) — why outputs sometimes looked conversational but mostly
   terminated immediately.
10. [CHECKPOINT-CATALOG.md](CHECKPOINT-CATALOG.md) — exact recoverable native checkpoints and hashes.
11. [EVIDENCE-INDEX.md](EVIDENCE-INDEX.md) — canonical reports, samples, manifests, and screenshots.
12. [DECISIONS.md](DECISIONS.md) — binding operator decisions that must not be silently reversed.
13. [EXPERIMENT-BACKLOG.md](EXPERIMENT-BACKLOG.md) — later experiments, not automatic authorization.
14. [ACCEPTANCE-GATES.md](ACCEPTANCE-GATES.md) — proof required before spending, continuing, or publishing.
15. [RUNPOD-RECOVERY.md](RUNPOD-RECOVERY.md) — future recovery only after renewed authorization.
16. [SERVING-OPERATIONS.md](SERVING-OPERATIONS.md) — operation of the public model artifacts.

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
