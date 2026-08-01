# Alpha chat-model resume dossier

> **Current-project note (2026-08-01):** the original chatty-model goal is active again. Start with the v3 local
> preflight below. The synthetic-curriculum program is preserved but paused as a side project.

- **Corrective run:** complete
- **Selected checkpoint:** step 1,200, SHA `399f776b49acc0c8834ff8a7f2390454e2c5f2d833a264e3f83ff546e973cfec`
- **Development verdict:** structurally chatty, semantically immature
- **Frozen result:** 55/100 structural, 70/100 nonempty, 31 loops, QA 0/200; quality gate FAIL
- **Repair v2:** continuation and clean-base control both rejected; no new selection; sealed final untouched
- **Repair v3:** finite RCR-UL experiment implemented and frozen locally; NVIDIA/full-rollout/training gates open
- **Closeout:** best honest model unchanged; v2 recovery archive published; paid Alpha pod removed

This directory is the shortest safe route into the model program. It preserves the archived failure, the later
corrective experiment, and the evidence required before any further paid run.

## Read order

1. [CHAT-REPAIR-V3-LOCAL-PREFLIGHT-2026-08-01.md](CHAT-REPAIR-V3-LOCAL-PREFLIGHT-2026-08-01.md) — current local implementation, frozen inputs, evaluation contract, and open gates.
2. [CHAT-REPAIR-V3-EXPERIMENT-CONTRACT.md](CHAT-REPAIR-V3-EXPERIMENT-CONTRACT.md) — exact finite causal experiment; not execution authorization.
3. [CHAT-REPAIR-V3-GPU-EXECUTION-RUNBOOK.md](CHAT-REPAIR-V3-GPU-EXECUTION-RUNBOOK.md) — exact disposable one-GPU sequence after explicit authorization.
4. [CHAT-REPAIR-V2-2026-07-31.md](CHAT-REPAIR-V2-2026-07-31.md) — latest completed bounded experiment, rejection, diagnosis, and recovery.
5. [CHAT-REPAIR-2026-07-31.md](CHAT-REPAIR-2026-07-31.md) — selected corrective checkpoint and its failed final gate.
6. [SESSION-START.md](SESSION-START.md) — the first-session checklist and hard stops.
7. [CURRENT-STATE.md](CURRENT-STATE.md) — current repository, runtime, Hub, and authority state.
8. [FAILURE-ANALYSIS.md](FAILURE-ANALYSIS.md) — why outputs sometimes looked conversational but mostly
   terminated immediately.
9. [CHECKPOINT-CATALOG.md](CHECKPOINT-CATALOG.md) — exact recoverable native checkpoints and hashes.
10. [EVIDENCE-INDEX.md](EVIDENCE-INDEX.md) — canonical reports, samples, manifests, and screenshots.
11. [DECISIONS.md](DECISIONS.md) — binding operator decisions that must not be silently reversed.
12. [EXPERIMENT-BACKLOG.md](EXPERIMENT-BACKLOG.md) — later experiments, not automatic authorization.
13. [ACCEPTANCE-GATES.md](ACCEPTANCE-GATES.md) — proof required before spending, continuing, or publishing.
14. [RUNPOD-RECOVERY.md](RUNPOD-RECOVERY.md) — future recovery only after renewed authorization.
15. [SERVING-OPERATIONS.md](SERVING-OPERATIONS.md) — operation of the public model artifact.

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

The Space must remain deliberately honest, expose exact model output, and use no fallback model. Serving the
artifact is not continuation training.
