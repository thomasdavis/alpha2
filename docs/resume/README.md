# Alpha 60M future-resume dossier

> **Current-project note (2026-07-30):** this directory remains the authoritative archive/recovery dossier for
> the completed Alpha 60M run. Alpha's new planning goal is the documentation suite at
> [../synthetic-curriculum-prd/README.md](../synthetic-curriculum-prd/README.md). The new goal does not authorize
> implementation, generation, training, GPU spend, or continuation of the archived checkpoint.

- **State frozen:** 2026-07-30
- **Training authorization:** none
- **Terminal verdict:** execution PASS, chat-quality D3 FAIL
- **Archive tag:** alpha-60m-archive-20260730

This directory is the shortest safe route back into the Alpha 60M program. It separates what was
proved from what merely looked promising at individual checkpoints, preserves the failed result, and
defines the work required before another paid GPU minute is allowed.

## Read order

1. [SESSION-START.md](SESSION-START.md) — the first-session checklist and hard stops.
2. [CURRENT-STATE.md](CURRENT-STATE.md) — frozen repository, runtime, Hub, and authorization state.
3. [FAILURE-ANALYSIS.md](FAILURE-ANALYSIS.md) — why outputs sometimes looked conversational but mostly
   terminated immediately.
4. [CHECKPOINT-CATALOG.md](CHECKPOINT-CATALOG.md) — exact recoverable native checkpoints and hashes.
5. [EVIDENCE-INDEX.md](EVIDENCE-INDEX.md) — canonical reports, samples, manifests, and screenshots.
6. [DECISIONS.md](DECISIONS.md) — binding operator decisions that must not be silently reversed.
7. [EXPERIMENT-BACKLOG.md](EXPERIMENT-BACKLOG.md) — the ordered technical repair program.
8. [ACCEPTANCE-GATES.md](ACCEPTANCE-GATES.md) — proof required before spending, continuing, or publishing.
9. [RUNPOD-RECOVERY.md](RUNPOD-RECOVERY.md) — future recovery only after renewed authorization.
10. [SERVING-OPERATIONS.md](SERVING-OPERATIONS.md) — operation of the public failed-quality artifact.

Then read the repository-level [GOAL.md](../../GOAL.md), [HANDOFF.md](../../HANDOFF.md),
[docs/RUNPOD.md](../RUNPOD.md), and [docs/FROZEN_EVAL.md](../FROZEN_EVAL.md). GOAL.md and HANDOFF.md
contain the complete chronological record; this directory is the compact recovery layer.

## The five facts that must survive every handoff

- Alpha trained the model successfully through its own TypeScript, Vulkan, autograd, tokenizer, and
  checkpoint stack. The terminal checkpoint is mechanically healthy.
- The model is not chat-ready. The sealed terminal evaluation produced 92 empty responses, six loops,
  two unusable fragments, and closed-book QA 0/200.
- No Alpha GPU pod or training process is live. Do not provision one without a new explicit user
  authorization and a new continuation contract.
- The standard Hugging Face model is inference-only. Future training must use a native ALPH checkpoint
  from the recovery archive because it carries AdamW, RNG, tokenizer, and step state.
- Failed evaluations, abandoned trajectories, and old metrics are evidence. Never overwrite or
  relabel them as a successful release.

## Public artifacts

- Model: https://huggingface.co/ajaxdavis/alpha-60m-chat
- Native recovery archive: https://huggingface.co/ajaxdavis/alpha-60m-training-checkpoints
- Space: https://huggingface.co/spaces/ajaxdavis/alpha-60m-chat
- Exact CPU backend health: https://donto.org/alpha-60m/health

The Space is deliberately honest: it exposes immediate EOS as an empty response and has no fallback
model. Serving the artifact is not continuation training.
