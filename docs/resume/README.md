# Alpha chat-model resume dossier

> **Current-project note (2026-07-31):** the original chatty-model goal is active again. Read
> [CHAT-REPAIR-2026-07-31.md](CHAT-REPAIR-2026-07-31.md) first. The synthetic-curriculum program is preserved
> but paused as a side project.

- **Corrective run:** complete
- **Selected checkpoint:** step 1,200, SHA `399f776b49acc0c8834ff8a7f2390454e2c5f2d833a264e3f83ff546e973cfec`
- **Development verdict:** structurally chatty, semantically immature
- **Closeout:** final frozen evaluation, publication, and pod termination in progress

This directory is the shortest safe route into the model program. It preserves the archived failure, the later
corrective experiment, and the evidence required before any further paid run.

## Read order

1. [CHAT-REPAIR-2026-07-31.md](CHAT-REPAIR-2026-07-31.md) — active corrective run and selected candidate.
2. [SESSION-START.md](SESSION-START.md) — the first-session checklist and hard stops.
3. [CURRENT-STATE.md](CURRENT-STATE.md) — current repository, runtime, Hub, and authority state.
4. [FAILURE-ANALYSIS.md](FAILURE-ANALYSIS.md) — why outputs sometimes looked conversational but mostly
   terminated immediately.
5. [CHECKPOINT-CATALOG.md](CHECKPOINT-CATALOG.md) — exact recoverable native checkpoints and hashes.
6. [EVIDENCE-INDEX.md](EVIDENCE-INDEX.md) — canonical reports, samples, manifests, and screenshots.
7. [DECISIONS.md](DECISIONS.md) — binding operator decisions that must not be silently reversed.
8. [EXPERIMENT-BACKLOG.md](EXPERIMENT-BACKLOG.md) — later experiments, not automatic authorization.
9. [ACCEPTANCE-GATES.md](ACCEPTANCE-GATES.md) — proof required before spending, continuing, or publishing.
10. [RUNPOD-RECOVERY.md](RUNPOD-RECOVERY.md) — future recovery only after renewed authorization.
11. [SERVING-OPERATIONS.md](SERVING-OPERATIONS.md) — operation of the public model artifact.

Then read the repository-level [GOAL.md](../../GOAL.md), [HANDOFF.md](../../HANDOFF.md),
[docs/RUNPOD.md](../RUNPOD.md), and [docs/FROZEN_EVAL.md](../FROZEN_EVAL.md). GOAL.md and HANDOFF.md
contain the complete chronological record; this directory is the compact recovery layer.

## The six facts that must survive every handoff

- Alpha trained the model successfully through its own TypeScript, Vulkan, autograd, tokenizer, and
  checkpoint stack.
- The archived terminal checkpoint failed: 92 empty responses, six loops, two unusable fragments, and QA 0/200.
- The later corrective checkpoint 1,200 produces nonempty, EOS-terminated replies on all 48 repair-development
  cases, but still has five loops and weak semantic answers.
- The evaluator and serving prompt also inserted a standalone generation-only space after the assistant marker.
  Commit `cf4ad61` fixes that boundary; historical failed outputs remain preserved.
- The paid Alpha pod exists only for final evaluation/export and must be removed after verified copy.
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
