# Future RunPod recovery procedure

**Do not execute this procedure under the archived authorization.** It is a recovery design for a
future user-approved experiment.

The established low-level Vulkan recipe remains in [docs/RUNPOD.md](../RUNPOD.md). This document
describes the continuation-specific sequence.

## 1. Freeze a new experiment contract locally

Before creating a pod, record:

- source commit;
- starting checkpoint name and SHA-256;
- input corpus/tokenizer/manifests and hashes;
- exact change from the archived recipe;
- maximum steps, tokens, wall time, and spend;
- A3/A4 stop thresholds;
- remote run path and local mirror path;
- guard/finalizer unit names;
- pod ID placeholder that must be filled after creation.

Do not modify the old sft-contract.json or run_flagship_sft.sh to mean something new. Add a new
contract schema and launcher.

## 2. Prefer the correct recovery point

For a repaired SFT recipe, start from:

    checkpoints/base-pretrain-step-61036.alph
    SHA-256 08e14fa9604bf1b46ebcd5df37933c84d2496c1d05d9e4b32ebad98792cc6049

Use terminal SFT only for an experiment that intentionally continues the failed SFT state:

    checkpoints/sft-terminal-step-30322.alph
    SHA-256 6c279d086d8c0679495e38ebec8a473ac23d16bfb3b93516e144712963fecbc8

Verify locally from:

    /mnt/donto-data/alpha-runs/alpha-60m-continuation-c333bf2-20260730/

## 3. Create and identify one scoped pod

Use the proven command in docs/RUNPOD.md, then immediately record:

- pod ID;
- name;
- GPU type;
- hourly price;
- SSH endpoint;
- volume size;
- account balance.

If Vulkan bootstrap fails on the host, terminate that exact pod and recreate rather than debugging a
bad community host indefinitely.

## 4. Bootstrap and deploy

Run scripts/runpod_bootstrap.sh and require vkCreateInstance plus NVIDIA device enumeration.

Deploy the exact intended source tree and compare:

    git rev-parse HEAD

On the pod, use the Node runtime for the proven Vulkan training launcher. The historical Bun standalone
binary had a vkCreateInstance failure on RunPod; do not let the generic compiled-binary preference
override the current platform proof.

Run the canonical NVIDIA gate before checkpoint or corpus transfer is treated as admitted.

## 5. Transfer immutable inputs

Transfer only the selected checkpoint and required inputs from the verified local bundle and canonical
corpus locations. After transfer, hash every file remotely.

Expected SFT locations from the archived contract:

    /runpod/data/alpha-sft-v2/
    /workspace/alpha2/artifacts/g2-bpe-byte-12k.json

A new recipe may choose new paths, but the contract must name and hash them. Do not silently make old
paths point to new content.

## 6. Prepare resume state

If continuing an existing metric stream, use scripts/prepare_resume_metrics.ts. It preserves and hashes
any tail beyond the selected checkpoint before aligning the active stream.

A new repaired-SFT run from the base checkpoint should use a new run directory and a new metrics file.
It must not append to the archived terminal SFT trajectory.

Prove locally and on-pod that:

- checkpoint step matches the launcher start;
- optimizer step and LR schedule match the new contract;
- the first batch identity is deterministic;
- resume produces the expected next batch;
- input hashes match before GPU initialization.

## 7. Install guard and terminal finalizer

The box-side guard must:

- pull at a bounded interval;
- verify real metric-row growth;
- monitor process identity, CPU/RSS, and SSH failures;
- mirror checkpoints before remote pruning;
- compare byte count and SHA-256;
- use an explicit stall deadline;
- scope any termination to the recorded Alpha pod ID.

Run a one-shot guard/finalizer preflight first. After launching the long-lived unit, verify the first
metric row appears locally and then verify a second later row. “Service active” alone is insufficient.

## 8. Launch only the bounded pilot

Use the new launcher. Required environment remains:

    HELIOS_DISABLE_COOP_MAT=1
    HELIOS_NO_FALLBACK=1

Keep f32 unless a separately certified precision program has passed all numerical gates. Do not begin
with another full epoch.

## 9. Observe and stop

Measure:

- actual row delta per wall-clock interval;
- GPU utilization and memory;
- post-warmup throughput;
- loss and gradient finiteness;
- allocator cadence;
- checkpoint memory recovery;
- first-token EOS metrics;
- sealed development-set generation.

On rejection or completion:

1. stop the scoped trainer;
2. perform final mirror;
3. verify manifests locally;
4. preserve failure and decision reports;
5. terminate the exact Alpha pod;
6. re-run runpodctl pod list to confirm only intended workloads remain;
7. update HANDOFF.md and this dossier;
8. commit and push.
