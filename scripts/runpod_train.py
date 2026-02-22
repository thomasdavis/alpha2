#!/usr/bin/env python3
"""
runpod_train.py — Run Alpha training on RunPod GPU pods.

Provisions a GPU pod, syncs code, runs training, downloads results.
Datasets and runs persist on a RunPod network volume (/workspace).

Usage:
  # Train (creates pod if needed, syncs code, runs, downloads results):
  python scripts/runpod_train.py --data data/concordance.txt --iters 5000 --backend helios

  # Train with specific GPU:
  python scripts/runpod_train.py --data data/concordance.txt --iters 5000 --gpu A100

  # Check pod status:
  python scripts/runpod_train.py --action status

  # Stop pod (volume persists, no GPU charges):
  python scripts/runpod_train.py --action stop

  # Resume stopped pod:
  python scripts/runpod_train.py --action resume

  # Terminate pod completely:
  python scripts/runpod_train.py --action terminate

  # Interactive SSH into pod:
  python scripts/runpod_train.py --action ssh

Requires:
  RUNPOD_API_KEY    — from https://runpod.io/console/user/settings
  RUNPOD_VOLUME_ID  — (optional) network volume ID for persistent storage
  SSH public key added to RunPod account settings
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_URL = "https://api.runpod.io/graphql"
POD_NAME = "alpha-train"
WORKSPACE = "/workspace"

GPU_TYPES = {
    "H100":  "NVIDIA H100 80GB HBM3",
    "H100S": "NVIDIA H100 SXM",
    "A100":  "NVIDIA A100 80GB PCIe",
    "A100S": "NVIDIA A100-SXM4-80GB",
    "A6000": "NVIDIA RTX A6000",
    "4090":  "NVIDIA GeForce RTX 4090",
    "3090":  "NVIDIA GeForce RTX 3090",
}

# Template with CUDA + SSH preinstalled
TEMPLATE_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

# Exclusions for rsync (same as modal_train.py)
RSYNC_EXCLUDES = [
    "node_modules", ".git", ".turbo", ".next", "data", "runs", "outputs",
    "artifacts", ".env.local", ".vercel", "*.node", "*.db", "*.db-wal",
    "*.db-shm", "__pycache__", ".DS_Store",
]

# ---------------------------------------------------------------------------
# RunPod API helpers
# ---------------------------------------------------------------------------

def get_api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        print("Error: RUNPOD_API_KEY not set")
        print("Get your key at https://runpod.io/console/user/settings")
        sys.exit(1)
    return key


def gql(query: str, variables: dict = None) -> dict:
    """Execute GraphQL query against RunPod API."""
    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{API_URL}?api_key={get_api_key()}",
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "alpha-train/1.0",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"API error ({e.code}): {body}")
        sys.exit(1)

    if "errors" in result:
        for err in result["errors"]:
            print(f"GraphQL error: {err.get('message', err)}")
        sys.exit(1)

    return result.get("data", {})


# ---------------------------------------------------------------------------
# Pod management
# ---------------------------------------------------------------------------

POD_FIELDS = """
    id name desiredStatus
    costPerHr gpuCount
    runtime {
        uptimeInSeconds
        ports {
            ip isIpPublic privatePort publicPort type
        }
        gpus {
            id gpuUtilPercent memoryUtilPercent
        }
    }
    machine { gpuDisplayName }
"""


def find_pod() -> dict | None:
    """Find existing alpha-train pod."""
    data = gql(f"""
        query {{
            myself {{
                pods {{
                    {POD_FIELDS}
                }}
            }}
        }}
    """)
    for pod in data.get("myself", {}).get("pods", []):
        if pod["name"] == POD_NAME:
            return pod
    return None


def get_pod(pod_id: str) -> dict | None:
    data = gql(f"""
        query {{
            pod(input: {{podId: "{pod_id}"}}) {{
                {POD_FIELDS}
            }}
        }}
    """)
    return data.get("pod")


def create_pod(gpu: str = "H100", volume_id: str = None) -> dict:
    """Create a new GPU pod."""
    gpu_type_id = GPU_TYPES.get(gpu.upper(), gpu)

    # Inject SSH public key so we can connect without RunPod dashboard setup
    env_vars = []
    for key_file in ["~/.ssh/id_ed25519.pub", "~/.ssh/id_rsa.pub"]:
        path = os.path.expanduser(key_file)
        if os.path.exists(path):
            pubkey = open(path).read().strip()
            env_vars.append({"key": "PUBLIC_KEY", "value": pubkey})
            break

    # Pass alpha remote env vars to the pod if set locally
    for var in ["ALPHA_REMOTE_URL", "ALPHA_REMOTE_SECRET"]:
        val = os.environ.get(var)
        if val:
            env_vars.append({"key": var, "value": val})

    input_vars = {
        "cloudType": "ALL",
        "gpuCount": 1,
        "volumeInGb": 100,
        "containerDiskInGb": 40,
        "minVcpuCount": 4,
        "minMemoryInGb": 32,
        "gpuTypeId": gpu_type_id,
        "name": POD_NAME,
        "imageName": TEMPLATE_IMAGE,
        "ports": "22/tcp",
        "volumeMountPath": WORKSPACE,
        "startSsh": True,
        "env": env_vars,
    }

    if volume_id:
        input_vars["networkVolumeId"] = volume_id

    data = gql("""
        mutation($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id name desiredStatus machineId
                machine { gpuDisplayName }
            }
        }
    """, {"input": input_vars})

    pod = data["podFindAndDeployOnDemand"]
    gpu_name = pod.get("machine", {}).get("gpuDisplayName", "unknown")
    print(f"Created pod {pod['id']} ({gpu_name})")
    return pod


def stop_pod(pod_id: str):
    gql(f'mutation {{ podStop(input: {{podId: "{pod_id}"}}) {{ id desiredStatus }} }}')
    print(f"Stopped pod {pod_id}")


def resume_pod(pod_id: str):
    gql(f'mutation {{ podResume(input: {{podId: "{pod_id}", gpuCount: 1}}) {{ id desiredStatus }} }}')
    print(f"Resuming pod {pod_id}")


def terminate_pod(pod_id: str):
    gql(f'mutation {{ podTerminate(input: {{podId: "{pod_id}"}}) }}')
    print(f"Terminated pod {pod_id}")


def wait_for_pod(pod_id: str, timeout: int = 300) -> dict:
    """Wait for pod to be RUNNING with SSH available."""
    print(f"Waiting for pod {pod_id}...", end="", flush=True)
    start = time.time()

    while time.time() - start < timeout:
        pod = get_pod(pod_id)
        if pod and pod.get("runtime") and pod["runtime"].get("ports"):
            ssh_info = get_ssh_info(pod)
            if ssh_info:
                print(f" ready! ({int(time.time() - start)}s)")
                return pod
        print(".", end="", flush=True)
        time.sleep(5)

    raise TimeoutError(f"Pod did not start within {timeout}s")


def get_ssh_info(pod: dict) -> tuple[str, int] | None:
    """Extract (ip, port) for SSH from pod runtime ports."""
    runtime = pod.get("runtime")
    if not runtime or not runtime.get("ports"):
        return None

    for port in runtime["ports"]:
        if port.get("privatePort") == 22 and port.get("isIpPublic"):
            return (port["ip"], port["publicPort"])

    return None


def print_pod_status(pod: dict):
    """Pretty-print pod status."""
    status = pod["desiredStatus"]
    gpu_name = pod.get("machine", {}).get("gpuDisplayName", "?")
    cost = pod.get("costPerHr", 0)

    print(f"  Pod:    {pod['id']}")
    print(f"  Status: {status}")
    print(f"  GPU:    {gpu_name}")
    print(f"  Cost:   ${cost:.2f}/hr")

    runtime = pod.get("runtime")
    if runtime:
        uptime = runtime.get("uptimeInSeconds", 0)
        hours = uptime / 3600
        print(f"  Uptime: {hours:.1f}h (${cost * hours:.2f} total)")

        gpus = runtime.get("gpus", [])
        for gpu in gpus:
            util = gpu.get("gpuUtilPercent", 0)
            mem = gpu.get("memoryUtilPercent", 0)
            print(f"  GPU util: {util}%  VRAM: {mem}%")

    ssh_info = get_ssh_info(pod)
    if ssh_info:
        ip, port = ssh_info
        print(f"  SSH:    ssh -p {port} root@{ip}")


# ---------------------------------------------------------------------------
# SSH / rsync helpers
# ---------------------------------------------------------------------------

def ssh_opts() -> list[str]:
    return [
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-o", "ConnectTimeout=10",
    ]


def ssh_cmd(ip: str, port: int) -> list[str]:
    return ["ssh"] + ssh_opts() + ["-p", str(port), f"root@{ip}"]


def ssh_run(ip: str, port: int, command: str, stream: bool = False,
            check: bool = True) -> subprocess.CompletedProcess:
    """Run command on pod via SSH."""
    cmd = ssh_cmd(ip, port) + [command]
    if stream:
        r = subprocess.run(cmd)
    else:
        r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode != 0:
        stderr = r.stderr if hasattr(r, "stderr") and r.stderr else ""
        stdout = r.stdout if hasattr(r, "stdout") and r.stdout else ""
        raise RuntimeError(f"SSH command failed ({r.returncode}): {stderr or stdout}")
    return r


def wait_for_ssh(ip: str, port: int, timeout: int = 120):
    """Wait until SSH connection succeeds."""
    print(f"Waiting for SSH ({ip}:{port})...", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = subprocess.run(
                ssh_cmd(ip, port) + ["echo ok"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                print(" connected!")
                return
        except subprocess.TimeoutExpired:
            pass
        print(".", end="", flush=True)
        time.sleep(3)
    raise TimeoutError("SSH connection timed out")


def rsync_to(ip: str, port: int, local_path: str, remote_path: str,
             excludes: list[str] = None):
    """Rsync local → pod."""
    cmd = [
        "rsync", "-az", "--info=progress2", "--delete",
        "-e", f"ssh {' '.join(ssh_opts())} -p {port}",
    ]
    for ex in (excludes or []):
        cmd += ["--exclude", ex]
    # Ensure trailing slash for directory sync
    src = local_path.rstrip("/") + "/"
    cmd += [src, f"root@{ip}:{remote_path}"]
    subprocess.run(cmd, check=True)


def rsync_from(ip: str, port: int, remote_path: str, local_path: str):
    """Rsync pod → local."""
    os.makedirs(local_path, exist_ok=True)
    cmd = [
        "rsync", "-az", "--info=progress2",
        "-e", f"ssh {' '.join(ssh_opts())} -p {port}",
        f"root@{ip}:{remote_path}",
        local_path,
    ]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Pod setup (runs once per pod creation)
# ---------------------------------------------------------------------------

SETUP_SCRIPT = r"""#!/bin/bash
set -e

MARKER="/workspace/.alpha-setup-done-v2"

if [ -f "$MARKER" ]; then
    echo "Environment already set up."
    node --version
    # Ensure Xvfb is running (NVIDIA Vulkan ICD needs X11 display)
    pgrep Xvfb >/dev/null 2>&1 || (Xvfb :99 -screen 0 1024x768x24 &)
    exit 0
fi

echo "============================================================"
echo "Setting up Alpha training environment..."
echo "============================================================"

# Node.js 22
echo "Installing Node.js 22..."
curl -fsSL https://nodejs.org/dist/v22.14.0/node-v22.14.0-linux-x64.tar.xz \
    | tar -xJ -C /usr/local --strip-components=1
echo "Node $(node --version), npm $(npm --version)"

# Add LunarG Vulkan SDK repo for up-to-date Vulkan loader
echo "Adding LunarG Vulkan repo..."
wget -qO /tmp/lunarg-signing-key-pub.asc https://packages.lunarg.com/lunarg-signing-key-pub.asc
apt-key add /tmp/lunarg-signing-key-pub.asc 2>/dev/null
echo 'deb https://packages.lunarg.com/vulkan jammy main' > /etc/apt/sources.list.d/lunarg-vulkan.list

# Vulkan loader (1.4.x for NVIDIA driver 580+ compat) + Xvfb + rsync
echo "Installing Vulkan loader, Xvfb, rsync..."
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libvulkan1 xvfb rsync 2>/dev/null || true

# Start Xvfb virtual display — NVIDIA Vulkan ICD (libGLX_nvidia.so.0)
# requires an X11 display even for headless compute
echo "Starting Xvfb virtual display..."
Xvfb :99 -screen 0 1024x768x24 &
sleep 1

# Verify Vulkan sees the GPU
echo "Verifying Vulkan..."
DISPLAY=:99 vulkaninfo --summary 2>&1 | grep -E 'deviceName|deviceType' || true

touch "$MARKER"
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
"""


def setup_pod(ip: str, port: int):
    """Install Node.js + Vulkan on the pod (idempotent)."""
    print("\n" + "=" * 60)
    print("Setting up pod environment...")
    print("=" * 60)
    # Write and execute setup script
    ssh_run(ip, port, f"cat > /tmp/setup.sh << 'SETUP_EOF'\n{SETUP_SCRIPT}\nSETUP_EOF\nchmod +x /tmp/setup.sh")
    ssh_run(ip, port, "bash /tmp/setup.sh", stream=True)


# ---------------------------------------------------------------------------
# Code sync + build
# ---------------------------------------------------------------------------

def sync_code(ip: str, port: int, project_dir: str):
    """Sync project code to pod (rsync preferred, tar fallback)."""
    print("\n" + "=" * 60)
    print("Syncing project code...")
    print("=" * 60)

    # Check if rsync is available on remote
    r = ssh_run(ip, port, "which rsync", check=False)
    if r.returncode == 0:
        rsync_to(ip, port, project_dir, f"{WORKSPACE}/alpha/", excludes=RSYNC_EXCLUDES)
    else:
        # Fallback: tar over SSH
        print("rsync not available on pod, using tar...")
        ssh_run(ip, port, f"mkdir -p {WORKSPACE}/alpha")
        exclude_args = " ".join(f"--exclude='{e}'" for e in RSYNC_EXCLUDES)
        cmd = (
            f"tar czf - -C {project_dir} {exclude_args} . | "
            f"ssh {' '.join(ssh_opts())} -p {port} root@{ip} "
            f"'tar xzf - -C {WORKSPACE}/alpha'"
        )
        subprocess.run(cmd, shell=True, check=True)


def build_on_pod(ip: str, port: int):
    """npm install + build native addon + TypeScript on pod."""
    print("\n" + "=" * 60)
    print("Building project on pod...")
    print("=" * 60)

    build_cmd = """
        set -e
        cd /workspace/alpha

        echo "Installing npm dependencies..."
        npm install --ignore-scripts 2>&1 | tail -5
        echo "Dependencies installed."

        echo ""
        echo "Building native helios addon..."
        node packages/helios/native/build.mjs 2>&1 || echo "WARNING: native addon build failed"
        echo ""

        echo "Building TypeScript..."
        npx turbo build --filter=@alpha/cli 2>&1 | tail -10
        echo "TypeScript build complete."
    """
    ssh_run(ip, port, build_cmd, stream=True)


# ---------------------------------------------------------------------------
# Dataset management
# ---------------------------------------------------------------------------

def upload_dataset(ip: str, port: int, local_data_path: str) -> str:
    """Upload dataset to pod if not already present. Returns remote path."""
    data_name = os.path.basename(local_data_path)
    remote_data_dir = f"{WORKSPACE}/datasets"
    remote_data_path = f"{remote_data_dir}/{data_name}"

    # Check if already uploaded
    r = ssh_run(ip, port, f"test -f {remote_data_path} && stat -c%s {remote_data_path} || echo MISSING",
                check=False)
    local_size = os.path.getsize(local_data_path)

    if r.returncode == 0 and r.stdout.strip() != "MISSING":
        remote_size = int(r.stdout.strip())
        if remote_size == local_size:
            print(f"Dataset '{data_name}' already on pod ({local_size / 1024 / 1024:.1f} MB) — skipping upload.")
            return remote_data_path

    # Upload
    print(f"Uploading {data_name} ({local_size / 1024 / 1024:.1f} MB)...")
    ssh_run(ip, port, f"mkdir -p {remote_data_dir}")
    cmd = [
        "scp"] + ssh_opts() + ["-P", str(port),
        local_data_path, f"root@{ip}:{remote_data_path}",
    ]
    subprocess.run(cmd, check=True)
    print("Upload complete.")
    return remote_data_path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(ip: str, port: int, remote_data_path: str, train_args: str,
                 run_id: str) -> str:
    """Execute training on the pod. Returns run ID."""
    run_dir = f"{WORKSPACE}/runs/{run_id}"

    # Export ALPHA_REMOTE env vars so metrics stream to the dashboard
    env_exports = "export DISPLAY=:99 && "
    for var in ["ALPHA_REMOTE_URL", "ALPHA_REMOTE_SECRET"]:
        val = os.environ.get(var)
        if val:
            env_exports += f"export {var}='{val}' && "

    cmd = (
        f"cd /workspace/alpha && "
        f"{env_exports}"
        f"pgrep Xvfb >/dev/null 2>&1 || (Xvfb :99 -screen 0 1024x768x24 &) && sleep 0.5 && "
        f"node --expose-gc --max-old-space-size=8192 "
        f"apps/cli/dist/main.js train "
        f"--data={remote_data_path} "
        f"--runDir={run_dir} "
        f"{train_args}"
    )

    print("\n" + "=" * 60)
    print(f"RUN {run_id}")
    print(f"CMD: {cmd}")
    print("=" * 60 + "\n")

    r = ssh_run(ip, port, cmd, stream=True, check=False)

    if r.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {r.returncode}")

    return run_id


def download_results(ip: str, port: int, run_id: str, local_dir: str):
    """Download training results from pod."""
    remote_run = f"{WORKSPACE}/runs/{run_id}/"
    local_run = os.path.join(local_dir, run_id)

    print(f"\nDownloading results to {local_run}/...")
    rsync_from(ip, port, remote_run, local_run + "/")

    # Show what was downloaded
    if os.path.exists(local_run):
        files = sorted(os.listdir(local_run))
        total = sum(os.path.getsize(os.path.join(local_run, f)) for f in files)
        print(f"Downloaded {len(files)} files ({total / 1024 / 1024:.1f} MB)")
        for f in files:
            sz = os.path.getsize(os.path.join(local_run, f))
            print(f"  {f} ({sz / 1024 / 1024:.1f} MB)")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def ensure_pod(gpu: str) -> tuple[dict, str, int]:
    """Find or create pod, wait for SSH. Returns (pod, ip, port)."""
    volume_id = os.environ.get("RUNPOD_VOLUME_ID")

    pod = find_pod()

    if pod:
        status = pod["desiredStatus"]
        print(f"Found existing pod {pod['id']} (status: {status})")

        if status == "EXITED":
            resume_pod(pod["id"])
            pod = wait_for_pod(pod["id"])
        elif status != "RUNNING":
            pod = wait_for_pod(pod["id"])
    else:
        print(f"No existing pod found. Creating with {gpu} GPU...")
        pod = create_pod(gpu=gpu, volume_id=volume_id)
        pod = wait_for_pod(pod["id"])

    ssh_info = get_ssh_info(pod)
    if not ssh_info:
        print("Error: Pod is running but no public SSH port available.")
        print("Ensure the pod has a public IP (some regions don't support this).")
        sys.exit(1)

    ip, port = ssh_info
    wait_for_ssh(ip, port)
    return pod, ip, port


def train_pipeline(args):
    """Full pipeline: create/reuse pod → setup → sync → train → download."""
    # Resolve data path
    data_path = args.data
    if not os.path.exists(data_path):
        data_path = os.path.join("data", args.data)
    if not os.path.exists(data_path):
        data_path = os.path.join("data", os.path.basename(args.data))
    if not os.path.exists(data_path):
        print(f"Error: dataset not found at '{args.data}'")
        sys.exit(1)

    data_size = os.path.getsize(data_path)
    data_name = os.path.basename(data_path)

    # Build train args string
    train_args_parts = [
        f"--iters={args.iters}",
        f"--batch={args.batch}",
        f"--block={args.block}",
        f"--dim={args.dim}",
        f"--heads={args.heads}",
        f"--layers={args.layers}",
        f"--lr={args.lr}",
        f"--backend={args.backend}",
        f"--tokenizer={args.tokenizer}",
    ]
    if args.domain:
        train_args_parts.append(f"--domain={args.domain}")
    if args.trace:
        train_args_parts.append("--trace=true")
    if args.eval_interval:
        train_args_parts.append(f"--evalInterval={args.eval_interval}")
    if args.grad_clip is not None:
        train_args_parts.append(f"--gradClip={args.grad_clip}")
    train_args = " ".join(train_args_parts)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    project_dir = str(Path(__file__).resolve().parent.parent)

    print("=" * 60)
    print("Alpha Training — RunPod GPU")
    print("=" * 60)
    print(f"  Dataset:   {data_name} ({data_size / 1024 / 1024:.1f} MB)")
    print(f"  GPU:       {args.gpu}")
    print(f"  Backend:   {args.backend}")
    print(f"  Iters:     {args.iters}  Batch: {args.batch}  Block: {args.block}")
    print(f"  Model:     {args.dim}d {args.heads}h {args.layers}L")
    print(f"  LR:        {args.lr}  Tokenizer: {args.tokenizer}")
    if args.domain:
        print(f"  Domain:    {args.domain}")
    print("=" * 60)

    # 1. Provision pod
    pod, ip, port = ensure_pod(args.gpu)
    gpu_name = pod.get("machine", {}).get("gpuDisplayName", args.gpu)
    cost = pod.get("costPerHr", 0)
    print(f"\nGPU: {gpu_name} (${cost:.2f}/hr)")

    # 2. Setup environment (idempotent)
    setup_pod(ip, port)

    # 3. Sync code
    sync_code(ip, port, project_dir)

    # 4. Build
    build_on_pod(ip, port)

    # 5. Upload dataset
    remote_data = upload_dataset(ip, port, data_path)

    # 6. GPU check
    r = ssh_run(ip, port,
                "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    print(f"\nGPU: {r.stdout.strip()}")

    # 7. Train
    run_training(ip, port, remote_data, train_args, run_id)

    # 8. Download results
    download_results(ip, port, run_id, args.download_dir)

    # 9. Optionally stop pod
    if args.stop_after:
        stop_pod(pod["id"])
        print("Pod stopped to save costs.")
    else:
        print(f"\nPod still running (${cost:.2f}/hr). Stop with: python scripts/runpod_train.py --action stop")

    print("\n" + "=" * 60)
    print(f"DONE — results in {args.download_dir}/{run_id}/")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Alpha training on RunPod GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Action (non-training commands)
    parser.add_argument("--action", choices=["status", "stop", "resume", "terminate", "ssh"],
                        help="Pod management action (instead of training)")

    # Training params
    parser.add_argument("--data", help="Path to training data file")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backend", default="helios")
    parser.add_argument("--tokenizer", default="bpe")
    parser.add_argument("--domain", default="")
    parser.add_argument("--gpu", default="H100", help="GPU type (H100, A100, A6000, 4090)")
    parser.add_argument("--trace", action="store_true", help="Enable per-step trace timing")
    parser.add_argument("--eval-interval", type=int, default=0, help="Eval/checkpoint interval")
    parser.add_argument("--grad-clip", type=float, default=None, help="Gradient clipping norm")
    parser.add_argument("--download-dir", default="runs", help="Local directory for results")
    parser.add_argument("--stop-after", action="store_true",
                        help="Stop pod after training completes")

    args = parser.parse_args()

    # Handle management actions
    if args.action:
        pod = find_pod()

        if args.action == "status":
            if not pod:
                print("No alpha-train pod found.")
                return
            print_pod_status(pod)

        elif args.action == "stop":
            if not pod:
                print("No pod to stop.")
                return
            stop_pod(pod["id"])

        elif args.action == "resume":
            if not pod:
                print("No pod to resume.")
                return
            resume_pod(pod["id"])

        elif args.action == "terminate":
            if not pod:
                print("No pod to terminate.")
                return
            resp = input(f"Terminate pod {pod['id']}? This is irreversible. [y/N] ")
            if resp.lower() == "y":
                terminate_pod(pod["id"])
            else:
                print("Cancelled.")

        elif args.action == "ssh":
            if not pod:
                print("No pod found. Create one first with a training run.")
                return
            ssh_info = get_ssh_info(pod)
            if not ssh_info:
                print("Pod has no SSH endpoint. Is it running?")
                return
            ip, port = ssh_info
            print(f"Connecting to {ip}:{port}...")
            os.execvp("ssh", ["ssh"] + ssh_opts() + ["-p", str(port), f"root@{ip}"])

        return

    # Training pipeline
    if not args.data:
        parser.error("--data is required for training")

    # Handle Ctrl+C gracefully
    def sigint_handler(sig, frame):
        print("\n\nInterrupted! Pod is still running.")
        print("Stop it with: python scripts/runpod_train.py --action stop")
        sys.exit(1)
    signal.signal(signal.SIGINT, sigint_handler)

    train_pipeline(args)


if __name__ == "__main__":
    main()
