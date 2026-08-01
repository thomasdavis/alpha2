# Alpha chat repair v3 — one-GPU execution runbook

**Status:** prepared locally; paid execution still requires explicit operator authorization.

This is the operational companion to
[`CHAT-REPAIR-V3-EXPERIMENT-CONTRACT.md`](CHAT-REPAIR-V3-EXPERIMENT-CONTRACT.md). It does not loosen any admission
gate and does not authorize GPU spend, sealed-final access, checkpoint selection, publication, or modification
of another project's pod.

## 1. Fixed execution order

One Alpha pod performs these stages serially:

1. bootstrap and prove headless NVIDIA Vulkan;
2. deploy one clean Git commit and hash-bound inputs;
3. generate a 24-row CUDA parity prefix and compare every token trajectory with the preserved native reference;
4. generate the complete 4,096-row CUDA rollout ledger;
5. compile and independently audit the immutable RCR-UL masks;
6. run the fail-closed 50/50 NVIDIA test gate;
7. run one selection-ineligible model-sized paired U1 step;
8. only if all prior stages pass, run C0 and U1 for 400 steps each from the same checkpoint and source commit;
9. evaluate I0 once and C0/U1 only at steps 50, 100, 200, and 400;
10. mirror and hash every artifact before terminating the Alpha pod.

The frozen panel remains human-pending. Both sealed finals remain unopened unless a development candidate is
admitted under the experiment contract.

## 2. GPU and account boundary

Prefer one RTX 4090 when available because both the fp32 Transformers rollout and Helios training can use it.
Use the already-proven RTX 3090 as fallback. More VRAM is unnecessary for this model-sized experiment; an A40 or
larger accelerator is useful only if its live price or availability is better.

Immediately before creation:

```bash
runpodctl user -o json
runpodctl pod list -o json
runpodctl gpu list -o json
df -h / /mnt/donto-data
```

Resolve the exact pod ID returned by creation and write it into the run record before deployment. Never reuse,
stop, restart, or remove a pod whose name and ownership record do not identify it as this Alpha v3 run. A stopped
pod still bills storage; after verified mirroring, terminate the exact Alpha pod.

The operator's artifact rule is a pause threshold, not a deletion target: measure project-owned additions and
pause before they exceed 15 GiB. Do not remove checkpoints to disguise growth. The two arms are expected to
produce about 10.3 GiB of optimizer-bearing checkpoints at the declared 50-step cadence.

## 3. Immutable local inputs

The control box is authoritative. Before transfer, verify:

| Input | SHA-256 |
|---|---|
| selected checkpoint 1,200 | `399f776b49acc0c8834ff8a7f2390454e2c5f2d833a264e3f83ff546e973cfec` |
| HF `model.safetensors` | `a5214ebad501b8bd3b09f7552c0db67417d18c3b66432f66f847de0e723dd688` |
| r3 freeze manifest | `976ef6b37949c729a2abad77f50f46c685dcb63269af1a1963dca58428e11231` |
| rollout candidates | `c8df6ccd79c4eb813d87c48eee9d2462837a944d24aeba1263c87515282e670a` |
| positive cohort | `3c9dcc8d44db15491dc94e0167e864da4fc436a49edbdbf9bac6b4b0652377da` |
| development selector | `0133dcda7d6ae3d5d7ed315e528e6cf566f332a355ed6189525f7a9f2b90c683` |
| qualitative panel | `c4c869f6c1dc30a9fa644d5e45782683f200db4f80bc9c54995abf0dd0983000` |
| evaluation contract | `c0270b2fb544fec5e03addb168841c20183ab7b7522a0937e3e0647ae0b509ce` |
| eligible-69 regression prompts | `4ba67c07fea204bbc76d76fb2b9208519bdd0029aa48046bb8143b6bcdedb584` |
| native 24-row rollout reference | `996f6b99f15291efab3d82430f40c66646b79708d16a61ca7d78696ed1433781` |

Canonical roots:

```text
/mnt/donto-data/donto-resources/research/alpha-chat-repair-v3-freeze-r3-20260801/
/mnt/donto-data/donto-resources/research/alpha-chat-repair-v3-evaluation-freeze-r2-canonical-20260801/
/mnt/donto-data/donto-resources/research/alpha-chat-repair-v3-rollout-smoke-r3-20260801/
/mnt/donto-data/alpha-runs/alpha-chat-repair-20260731/full-end2/
```

The tokenizer and teacher-forced development file are the exact paths recorded in the r3 freeze manifest. Do not
substitute a convenient copy without verifying that its hash matches the manifest.

## 4. Deploy a clean source commit

The v3 launchers call Git and reject a dirty tree. The older rsync recipe that excluded `.git` is insufficient.
Pods may not reach GitHub, so deploy through a local Git bundle:

```bash
cd /mnt/donto-data/workspace/alpha2
test -z "$(git status --porcelain)"
execution_commit=$(git rev-parse HEAD)
# A raw object ID is not a bundle ref and produces an empty-bundle failure.
git bundle create /tmp/alpha2-chat-repair-v3.bundle HEAD
git bundle verify /tmp/alpha2-chat-repair-v3.bundle
```

Transfer the bundle, bootstrap script, and immutable inputs. On the pod:

```bash
bash /root/runpod_bootstrap.sh
git clone /workspace/alpha2-chat-repair-v3.bundle /workspace/alpha2
cd /workspace/alpha2
execution_commit=<exact-40-character-commit-recorded-on-the-control-box>
git checkout --detach "$execution_commit"
test "$(git rev-parse HEAD)" = "$execution_commit"
test -z "$(git status --porcelain)"
npm ci
npm run build
node packages/helios/native/build.mjs
```

Create a Python environment using the image's CUDA-enabled Torch and install the same Transformers-family
versions used by the parity-proven local path. Do not replace CUDA Torch with the control box's CPU wheel:

```bash
python3 -m venv --system-site-packages /workspace/alpha-v3-venv
/workspace/alpha-v3-venv/bin/pip install \
  'transformers==5.14.1' 'safetensors==0.8.0' 'tokenizers==0.22.2' 'numpy==2.4.6'
/workspace/alpha-v3-venv/bin/python -c \
  'import torch, transformers; assert torch.cuda.is_available(); print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))'
```

If those packages are incompatible with the image, preserve the failure and use a CUDA image/runtime that can
satisfy them. Do not silently change versions and call the old parity proof applicable; rerun the 24-row
trajectory comparison on the actual runtime.

## 5. CUDA rollout and mask gate

Use separate output directories for the parity prefix and complete ledger. First generate exactly 24 accelerated
rows, then compare them with the native reference:

```bash
py=/workspace/alpha-v3-venv/bin/python
repo=/workspace/alpha2
inputs=/workspace/alpha-v3-inputs
artifacts=/workspace/alpha-v3-artifacts

"$py" "$repo/scripts/generate_chat_repair_v3_rollouts_hf.py" \
  --export-dir "$inputs/hf-alpha-60m-chat-repair-1200" \
  --native-checkpoint "$inputs/checkpoint-1200.json" \
  --expected-native-checkpoint-sha256 399f776b49acc0c8834ff8a7f2390454e2c5f2d833a264e3f83ff546e973cfec \
  --expected-model-sha256 a5214ebad501b8bd3b09f7552c0db67417d18c3b66432f66f847de0e723dd688 \
  --candidates "$inputs/freeze/rollout-candidates.jsonl" \
  --out-dir "$artifacts/rollout-parity24" --batch-size 32 --max-tokens 128 --stop-after 24

"$py" "$repo/scripts/verify_chat_repair_v3_rollout_parity.py" \
  --native "$inputs/native-parity24/raw-rollouts.jsonl" \
  --accelerated "$artifacts/rollout-parity24/raw-rollouts.jsonl" \
  --out "$artifacts/rollout-parity24/native-parity-report.json" --expected-rows 24
```

Only a `PASS` report permits complete generation:

```bash
"$py" "$repo/scripts/generate_chat_repair_v3_rollouts_hf.py" \
  --export-dir "$inputs/hf-alpha-60m-chat-repair-1200" \
  --native-checkpoint "$inputs/checkpoint-1200.json" \
  --expected-native-checkpoint-sha256 399f776b49acc0c8834ff8a7f2390454e2c5f2d833a264e3f83ff546e973cfec \
  --expected-model-sha256 a5214ebad501b8bd3b09f7552c0db67417d18c3b66432f66f847de0e723dd688 \
  --candidates "$inputs/freeze/rollout-candidates.jsonl" \
  --out-dir "$artifacts/rollout-full4096" --batch-size 32 --max-tokens 128

cd "$repo"
npx tsx scripts/compile_chat_repair_v3_rcr_ul.ts \
  --candidates="$inputs/freeze/rollout-candidates.jsonl" \
  --positive-cohort="$inputs/freeze/positive-cohort.txt" \
  --freeze-manifest="$inputs/freeze/freeze-manifest.json" \
  --raw-rollouts="$artifacts/rollout-full4096/raw-rollouts.jsonl" \
  --rollout-manifest="$artifacts/rollout-full4096/rollout-manifest.json" \
  --parity-report="$artifacts/rollout-parity24/native-parity-report.json" \
  --expected-checkpoint-sha256=399f776b49acc0c8834ff8a7f2390454e2c5f2d833a264e3f83ff546e973cfec \
  --out-dir="$artifacts/rcr-ul-cohort"
```

The compiler independently re-reads all 4,096 identities, token trajectories, stop contracts, masks, and hashes.
Its manifest must say `complete-and-immutable`, contain eligible negative rows, and contain nonzero penalty
positions.

## 6. NVIDIA and model-sized probe gates

Run the exact NVIDIA wrapper, never bare Vitest:

```bash
cd /workspace/alpha2
scripts/run_nvidia_gates.sh /workspace/alpha-v3-artifacts/nvidia-gate
```

Admission is exactly 50 passed, zero skipped, zero failed, zero todo on NVIDIA vendor `0x10de`.

Then execute the selection-ineligible one-step probe through the same U1 branch used by the full experiment:

```bash
scripts/run_chat_repair_v3_probe.sh \
  "$inputs/freeze/positive-cohort.txt" \
  "$artifacts/rcr-ul-cohort/negative-cohort.jsonl" \
  "$artifacts/rcr-ul-cohort/rcr-ul-manifest.json" \
  "$inputs/freeze/freeze-manifest.json" \
  "$inputs/dev.txt" "$inputs/tokenizer.json" "$inputs/checkpoint-1200.json" \
  "$artifacts/paired-probe-u1"
```

The probe must record one finite/consecutive optimizer step, execute both positive and negative branches, produce
finite losses, gradients, and parameters, remain within measured GPU memory, and save a nonempty checkpoint. Its
run contract says `executionMode=one-step-paired-probe` and `eligibleForCheckpointSelection=false`; the checkpoint
evaluator rejects it by design.

## 7. Paired arms and development evaluation

Only after all earlier gates pass, run sequentially from the same clean commit and initial checkpoint:

```bash
scripts/run_chat_repair_v3_arm.sh C0 0.0 \
  "$inputs/freeze/positive-cohort.txt" "$artifacts/rcr-ul-cohort/negative-cohort.jsonl" \
  "$artifacts/rcr-ul-cohort/rcr-ul-manifest.json" "$inputs/freeze/freeze-manifest.json" \
  "$inputs/dev.txt" "$inputs/tokenizer.json" "$inputs/checkpoint-1200.json" "$artifacts/C0"

scripts/run_chat_repair_v3_arm.sh U1 0.5 \
  "$inputs/freeze/positive-cohort.txt" "$artifacts/rcr-ul-cohort/negative-cohort.jsonl" \
  "$artifacts/rcr-ul-cohort/rcr-ul-manifest.json" "$inputs/freeze/freeze-manifest.json" \
  "$inputs/dev.txt" "$inputs/tokenizer.json" "$inputs/checkpoint-1200.json" "$artifacts/U1"
```

Preserve all checkpoints at 50, 100, 150, 200, 250, 300, 350, and 400; evaluate only the declared 50, 100, 200,
and 400 steps. Invoke `evaluate_chat_repair_v3_checkpoint.ts` with the canonical evaluation contract, fresh96,
panel24, and eligible69 paths. Evaluate I0 once. Pair only identical steps with
`analyze_chat_repair_v3_pair.ts`.

The analyzer may report `FAIL`, `INCONCLUSIVE_UNDERPOWERED`, or `MECHANICAL_PASS_HUMAN_PENDING`. None selects a
candidate automatically. Only the last state advances to blinded human review. Loss and embedding similarity
remain diagnostics, never checkpoint selectors.

## 8. Mirroring, termination, and publication boundary

Start a box-side guarded puller before the first long command and verify real row or file advancement. Mirror the
input manifest, parity proof, complete rollout and mask artifacts, NVIDIA report, probe, both arms, every
evaluation, and all logs. Compare remote and local byte counts and SHA-256 values before termination.

Do not post routine loss or progress to Discord. Post model outputs only under the operator's qualitative policy
or an explicit sample request.

Do not inspect a sealed final or upload a new Hugging Face checkpoint merely because the GPU run completes. A
development candidate must first pass mechanical admission and blinded conversational review. Publication then
requires the untouched final, export parity, honest model card, anonymous cold load, and live no-fallback serving
proof.
