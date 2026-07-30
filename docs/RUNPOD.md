# RunPod runbook for Alpha/Helios (Vulkan) — proven 2026-07-22

> **Archived status (2026-07-30):** the Alpha 60M program is closed. No Alpha pod or training process
> is live, and no further run is authorized. The completed SFT checkpoint failed the frozen chat-quality
> gate and was published only as an explicitly labelled research artifact. This document remains the
> proven recovery runbook, not an instruction to provision a pod now. Future work must begin with a new
> operator-approved continuation contract and the immutable recovery archive described below.

## Archived recovery state

The native continuation bundle is public at
[`ajaxdavis/alpha-60m-training-checkpoints`](https://huggingface.co/ajaxdavis/alpha-60m-training-checkpoints),
immutable revision `7198d1a1f094ffe88d06399ea99fecbd78fa8b66`. Its three recovery points are:

| Recovery point | SHA-256 | Intended use |
|---|---|---|
| base step 61,036 | `08e14fa9604bf1b46ebcd5df37933c84d2496c1d05d9e4b32ebad98792cc6049` | canonical completed pretrain |
| SFT step 29,000 | `03eaac3e7be06e8fb5720415a334b36d7ef5019fcff72ca9227636b84011a7f3` | best held-out loss among surviving full SFT checkpoints |
| SFT step 30,322 | `6c279d086d8c0679495e38ebec8a473ac23d16bfb3b93516e144712963fecbc8` | canonical one-epoch continuation state |

These are native ALPH checkpoints with model parameters, AdamW tensors and step, RNG state, and
tokenizer state—not inference-only exports. Contracts, full metrics, audits, corpus/tokenizer manifests,
and the failed evaluation travel with them. The identical local hardlink bundle is
`/mnt/donto-data/alpha-runs/alpha-60m-continuation-c333bf2-20260730/`; run `sha256sum -c MANIFEST.sha256`
there and read `RESUME.md` before any future RunPod transfer. Do not reuse the frozen one-epoch SFT
launcher unchanged: it terminates at step 30,322 by design.

RunPod replaces the dead GCP fleet. Pods are docker containers (no privileged mode, no apt reliability,
sometimes no github egress) with per-second billing. Everything below was executed and verified on a
community RTX 3090 @ $0.22/hr.

## Create a pod
```bash
runpodctl create pod --name alpha-train \
  --gpuType "NVIDIA GeForce RTX 3090" --communityCloud \
  --imageName "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404" \
  --containerDiskSize 40 --volumeSize 20 --startSSH --ports "22/tcp" \
  --env "NVIDIA_DRIVER_CAPABILITIES=all"
```
GPU preference: 3090 $0.22 (proven) → A5000 $0.16 → 4090 $0.34 → A40 $0.30-0.35. The
`NVIDIA_DRIVER_CAPABILITIES` env is set for good measure but RunPod's CDI runtime ignores it — the
bootstrap script below is what actually makes Vulkan work.

SSH endpoint (public IP + mapped port for 22/tcp):
```bash
runpodctl pod get <podId>   # or GraphQL runtime.ports; key: ~/.runpod/ssh/runpodctl-ssh-key
ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p <port> root@<ip>
```

## Bootstrap (Vulkan + Node)
```bash
scp -i ~/.runpod/ssh/runpodctl-ssh-key -P <port> scripts/runpod_bootstrap.sh root@<ip>:/root/
ssh ... root@<ip> 'bash /root/runpod_bootstrap.sh'
```
What it does (and why — full history in GOAL.md §2):
1. Installs the **exact-driver-matched** NVIDIA userspace (`.run` installer, `--no-kernel-modules`,
   kmod-stubs) because RunPod injects only compute libs.
2. Writes the **EGL headless ICD** and exports `VK_ICD_FILENAMES` — the stock GLX ICD fails headless.
3. Probes `vkCreateInstance` via python-ctypes (no apt). **If the probe fails: terminate the pod and
   redeploy — some hosts are bad; per-second billing makes probing ~free. Do not debug a bad host.**
4. Installs Node 22 from the official tarball (ships the `include/node` headers `build.mjs` needs).

## Deploy code (rsync from the box — pods may not reach github)
```bash
rsync -az --partial --exclude=.git --exclude=.next --exclude=.turbo \
  -e "ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p <port>" \
  /mnt/donto-data/workspace/alpha2/ root@<ip>:/workspace/alpha2/
```
The box-built `helios_vk.node` (Ubuntu 24.04 / glibc 2.39) loads on the ubuntu2404 image directly;
if `ldd` complains, rebuild on-pod: `cd /workspace/alpha2 && node packages/helios/native/build.mjs`.

For G3/flagship work, sync the bounded canonical shard and the already-proven tokenizer from their
durable mounted-drive copies, then compare the pod hashes before launching:
```bash
rsync -az --partial -e "ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p <port>" \
  /mnt/donto-data/alpha-corpora/pretrain-text/pretrain-000.txt root@<ip>:/workspace/data/
rsync -az --partial -e "ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p <port>" \
  /mnt/donto-data/alpha-runs/tokenizers-20260722/g2-bpe-byte-12k.json \
  root@<ip>:/workspace/alpha2/artifacts/
```
Expected SHA-256 values: pretrain shard
`d993342b0bb55198c520f1f761bb0aad2812b2d8fb9c6347b4e6f9d622794d9c`; tokenizer
`c310343a185aecb572b8b6568b55179df248f4adec009d14a9496da354090b24`.

## Train (node runtime, NOT the bun binary — bun binary has a known vkCreateInstance failure)
```bash
cd /workspace/alpha2
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd_headless.json \
nohup node --expose-gc apps/cli/dist/main.js train \
  --data=/workspace/data/<corpus>.txt --backend=helios --gpuProfile=none --fp16=false \
  ... > /workspace/train.log 2>&1 &
```
Always `--fp16=false` explicitly (the L4 auto-profile force-enables fp16, which NaNs). Stability env
of record: `HELIOS_DISABLE_COOP_MAT=1`.

### RTX 3090 profile (measured 2026-07-22)

The Stage-2 flagship sweep (57.7M params, f32, block 1024, batch 16) selected:

```bash
export HELIOS_WG_SIZE=64
export HELIOS_MAX_OUTPUT_POOL_ENTRIES=512  # current default; keep explicit in proof runs
export HELIOS_DISABLE_COOP_MAT=1
```

WG64 averaged 3,894 tok/s over the controlled sweep window versus 3,869 at WG128, and passed the
complete NVIDIA gate 46/46. Output-pool caps 256/384 did not improve throughput or the native
VkDeviceMemory count enough to replace the 512 default. Canonical evidence is under
`/mnt/donto-data/alpha-runs/g2-wg-sweep-20260722/`.

After deploying a new source commit, rerun the exact GPU regression through the fail-closed wrapper:

```bash
scripts/run_nvidia_gates.sh /workspace/alpha2/runs/nvidia-gate-<commit>
```

Do not use a bare successful Vitest exit as proof: Vitest exits zero when all GPU cases skip. The wrapper
first requires Vulkan vendor `0x10de`, runs exactly `parity-helios.test.ts` + `gpu-perf.test.ts`, writes
the JSON reporter output, and accepts only 46 unique assertion rows with 46 passed / 0 skipped / 0 failed.
`gate-summary.json` binds the full git SHA, device record, exact filenames/counts, and report SHA-256.

## Get checkpoints OFF the pod continuously (community pods are disposable)
Run the guarded puller on the box; never trust the pod to survive. It logs real remote metric-row
advancement and RSS on every cycle, performs a final sync when training exits, and does not confuse an
SSH outage with a training stall:
```bash
scripts/runpod_run_guard.sh <ip> <port> \
  /workspace/alpha2/runs/<run-name> \
  /mnt/donto-data/alpha-runs/<run-name> 300 1800
```
For a guard that survives the launching shell, run it as a transient user service and verify the first
metric row appears locally:
```bash
systemd-run --user --unit=alpha2-run-puller \
  scripts/runpod_run_guard.sh <ip> <port> \
  /workspace/alpha2/runs/<run-name> /mnt/donto-data/alpha-runs/<run-name> 300 1800
systemctl --user status alpha2-run-puller.service
```
For flagship runs, opt into termination after a verified 30-minute metric stall with
`RUNPOD_POD_ID=<id> TERMINATE_ON_STALL=1`. Without both values the guard exits nonzero and deliberately
leaves the pod running for inspection. The local `puller.log` is part of the run evidence.
Full flagship checkpoints are about 693MB each. Set both `REMOTE_KEEP_CHECKPOINTS=3` and
`LOCAL_KEEP_CHECKPOINTS=3` on long-run guards to retain the newest three on each side. An older remote
checkpoint is removed only after rsync succeeds and its local mirror matches the remote byte size and
SHA-256. Only then may `prune_local_checkpoints.ts` remove the corresponding old local copy: it verifies
the newest three are nonempty, hashes each exact candidate, fsyncs a commit record to
`checkpoint-prune-ledger.jsonl`, deletes it, and fsyncs a completion record. Local pruning is opt-in,
requires the same remote keep count, and rejects counts below three. Omit `LOCAL_KEEP_CHECKPOINTS` to
retain every local checkpoint. `RUNPOD_GUARD_ONCE=1` performs one pull, verification/prune pass, and
status check without entering the monitoring loop; it is useful for an operator check but not as the paid-run watchdog.
Egress is free. `runpodctl send/receive` (croc) also works for one-offs.

For the contracted 30,322-step flagship SFT, pair the live guard with the terminal watcher so a run that
finishes unattended does not leave a GPU billing idle:

```bash
RUNPOD_FINALIZER_ONCE=1 scripts/runpod_sft_terminal_watch.sh \
  <ip> <port> <pod-id> \
  /workspace/alpha2/runs/<sft-run> \
  /mnt/donto-data/alpha-runs/<sft-run> \
  <contracted-source-commit> 60

systemd-run --user --unit=alpha2-sft-terminal-finalizer --collect \
  scripts/runpod_sft_terminal_watch.sh \
  <ip> <port> <pod-id> \
  /workspace/alpha2/runs/<sft-run> \
  /mnt/donto-data/alpha-runs/<sft-run> \
  <contracted-source-commit> 60
```

Always run the one-shot preflight first. The watcher requires the exact terminal row count and checkpoint
with no trainer process before it acts. It then runs the streaming terminal parameter/input audit, strict
SFT analyzer, sealed 100-chat/200-QA evaluation, recomputed base/chat machine gate, standard HF export,
and Alpha-vs-Transformers logit parity. A machine D3 failure is preserved as valid evidence; it is not
turned into a passing publication by the watcher. The 2026-07-30 failed-quality upload used a separate,
explicit operator-override path and remains labelled as failed. The watcher seals every remote artifact in a SHA-256 manifest, rsyncs the run,
verifies every local copy, and only then removes the named pod. Any operational failure exits with the pod
untouched. Semantic review and Hugging Face chat publication remain manual gates.

For a normal release, after terminal artifacts are mirrored, machine D3 passes, and the sealed semantic-review report passes,
finish `docs/MODEL_CARD_CHAT.md` and run the release preflight. Publication is read-only by default and
requires an explicit `--publish`. The archived 2026-07-30 artifact did not satisfy these quality gates;
its publisher additionally required `--experimental-failed-release` so a failed model cannot be confused
with the normal release path:

```bash
nice -n19 ionice -c3 /mnt/donto-data/alpha-corpora/.venv/bin/python scripts/publish_hf_chat.py \
  --export-dir /mnt/donto-data/alpha-runs/FLAGSHIP/hf-alpha-60m-chat \
  --model-card docs/MODEL_CARD_CHAT.md \
  --terminal-status /mnt/donto-data/alpha-runs/FLAGSHIP/terminal-finalizer-status.json \
  --sft-analysis /mnt/donto-data/alpha-runs/FLAGSHIP/flagship-sft-analysis.json \
  --pair-analysis /mnt/donto-data/alpha-runs/FLAGSHIP/frozen-eval-pair-analysis.json \
  --semantic-review /mnt/donto-data/alpha-runs/FLAGSHIP/frozen-chat-semantic-review-report.json \
  --parity-log /mnt/donto-data/alpha-runs/FLAGSHIP/hf-export-parity.log \
  --repo ajaxdavis/alpha-60m-chat \
  --out /mnt/donto-data/alpha-runs/FLAGSHIP/hf-chat-publication-preflight.json
```

The preflight rejects incomplete model-card placeholders, any failed or mismatched terminal/D3 report,
checkpoint-hash disagreement, failed Alpha/Transformers parity, unexpected/custom-code export files,
or the wrong Hub target. Repeat with a fresh `--out` and `--publish` only after inspecting the PASS
report. Then run `scripts/verify_hf_hub.py` against the returned immutable Hub revision from an empty
cache; publication is not complete until that anonymous CPU cold-load proof passes.

The mounted-drive Python environment above is persistent after RunPod teardown and contains the pinned
`huggingface_hub`, Transformers, safetensors, and CPU torch dependencies used for the already-proven base
model publication. Host `python3` is sufficient for read-only preflight, but not for `--publish` or the
anonymous cold-load proof.

G3 pilot recovery uses the same `run_g3_pilot.sh` invocation plus the selected checkpoint as argument
five. The launcher requires the original commit/data/tokenizer/LR contract, and
`prepare_resume_metrics.ts` preserves and hashes any metric tail beyond that checkpoint before atomically
aligning the active stream. Never invoke `alpha train --resume` by hand against an unaligned metrics file.

After G3 and the LR selector pass, launch the minimum flagship only through:
```bash
scripts/run_flagship_pretrain.sh <lr-selection-report> /runpod/data/flagship-1b-manifest.json \
  /workspace/alpha2/artifacts/g2-bpe-byte-12k.json /workspace/alpha2/runs/flagship-1b
```
The launcher derives the LR from the selector's PASS report, verifies its exact three-candidate ranking
and tokenizer hash, binds the report bytes into the run/resume contract, verifies all shard hashes before
GPU initialization, and contracts 61,036 × 16 × 1,024 = 1,000,013,824 tokens. Resume with the same
command plus the checkpoint as argument five. Pair it with the box-side guard using matching
`REMOTE_KEEP_CHECKPOINTS=3 LOCAL_KEEP_CHECKPOINTS=3` retention.

## Terminate
```bash
runpodctl remove pod <podId>   # stopped pods still bill volume storage — terminate, don't stop
```
