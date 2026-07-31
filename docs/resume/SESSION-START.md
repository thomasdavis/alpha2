# Future session start checklist

Use this checklist when a user later says “resume Alpha,” “continue the model,” or similar. A bare
resume request is not permission to spend money or start training.

## First fifteen minutes

1. Confirm the working directory is /mnt/donto-data/workspace/alpha2.
2. Read, in order:
   - docs/resume/README.md
   - docs/resume/CHAT-REPAIR-2026-07-31.md
   - docs/resume/CURRENT-STATE.md
   - docs/resume/DECISIONS.md
   - HANDOFF.md
   - GOAL.md
3. Inspect repository state without changing it:

       git status --short
       git log -5 --oneline --decorate
       git remote -v

4. Verify the selected corrective checkpoint, then the older recovery bundle if historical reconstruction is
   required:

       sha256sum /mnt/donto-data/alpha-runs/alpha-chat-repair-20260731/full-end2/checkpoint-1200.json

   The selected SHA-256 must be
   `399f776b49acc0c8834ff8a7f2390454e2c5f2d833a264e3f83ff546e973cfec`.

       cd /mnt/donto-data/alpha-runs/alpha-60m-continuation-c333bf2-20260730
       sha256sum -c MANIFEST.sha256

   This reads about 2.1 GB. Run it at low I/O priority if the shared data disk is busy.

5. Verify public metadata and service health using anonymous reads:

       curl -fsS https://huggingface.co/api/models/ajaxdavis/alpha-60m-chat
       curl -fsS https://huggingface.co/api/models/ajaxdavis/alpha-60m-training-checkpoints
       curl -fsS https://huggingface.co/api/spaces/ajaxdavis/alpha-60m-chat
       curl -fsS https://donto.org/alpha-60m/health

6. Check RunPod state before discussing spend:

       runpodctl pod list

   Never stop, remove, signal, or reuse a pod owned by another project. The 2026-07-31 repair pod was
   `ksotbczj60mntk`; the final closeout must verify that exact pod has been removed.

## Authorization checkpoint

Before any later Alpha pod is created, the user must explicitly authorize a new run after being reminded that:

- the archived terminal model failed D3;
- the later selected corrective model is conversational but still loops and answers shallowly;
- the completed corrective contract does not authorize another run;
- another unmodified SFT epoch is not an approved repair;
- native recovery state is already safe locally and on Hugging Face;
- a bounded experiment and maximum spend must be agreed first.

If that authorization is absent, permitted work is read-only analysis, documentation, local unit tests,
and code changes that do not train or generate model outputs.

## State that must not be inferred

- A lower teacher-forced validation loss is not proof of better chat.
- A single attractive sample is not proof that a checkpoint improved globally.
- A process ID or service name is not proof that training is progressing.
- A successful standard Transformers load is not proof that the model is useful.
- The archive tag identifies the terminal program; later documentation commits do not rewrite it.

## Before ending a resumed session

- Update HANDOFF.md and this dossier with exact evidence.
- Preserve raw failures and hashes.
- Commit and push repository changes.
- If a paid pod was authorized and created, either leave a verified guard/finalizer actively advancing
  or terminate the exact Alpha pod. A merely stopped pod can still incur storage cost.
- Post to Discord only when a controlled comparison demonstrates qualitative improvement, and include
  the input, before output, after output, and why it improved.
