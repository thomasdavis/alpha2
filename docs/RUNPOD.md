# RunPod runbook for Alpha/Helios (Vulkan) — proven 2026-07-22

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

## Get checkpoints OFF the pod continuously (community pods are disposable)
Run a puller loop on the box; never trust the pod to survive:
```bash
while :; do rsync -az -e "ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p <port>" \
  root@<ip>:/workspace/alpha2/runs/ /mnt/donto-data/alpha-runs/<podname>/ ; sleep 300; done
```
Egress is free. `runpodctl send/receive` (croc) also works for one-offs.

## Terminate
```bash
runpodctl remove pod <podId>   # stopped pods still bill volume storage — terminate, don't stop
```
