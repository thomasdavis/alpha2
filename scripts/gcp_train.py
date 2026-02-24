#!/usr/bin/env python3
"""
gcp_train.py — Run Alpha training on GCP A100 GPU instances.

Provisions an a2-ultragpu-1g VM (A100 80GB), syncs code, runs training,
downloads results. Boot disk persists across stop/start cycles.
~$1.10/hr on-demand for A100 80GB.

Usage:
  # Train (creates instance if needed, syncs code, runs, downloads results):
  python scripts/gcp_train.py --data data/concordance.txt --iters 5000 --backend helios

  # Train with specific zone:
  python scripts/gcp_train.py --data data/concordance.txt --iters 5000 --zone us-west1-b

  # Check instance status:
  python scripts/gcp_train.py --action status

  # Stop instance (disk persists, no GPU charges):
  python scripts/gcp_train.py --action stop

  # Start stopped instance:
  python scripts/gcp_train.py --action start

  # Delete instance completely:
  python scripts/gcp_train.py --action delete

  # Interactive SSH into instance:
  python scripts/gcp_train.py --action ssh

Requires:
  gcloud CLI installed and authenticated:
    gcloud auth login
    gcloud config set project YOUR_PROJECT_ID
"""

import argparse
import getpass
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INSTANCE_NAME = "alpha-train"
DEFAULT_ZONE = "us-central1-a"
MACHINE_TYPE = "a2-ultragpu-1g"  # A100 80GB included
COST_PER_HR = 1.10  # approximate on-demand $/hr
BOOT_DISK_SIZE_GB = 200
BOOT_DISK_COST_PER_GB_MONTH = 0.17  # pd-ssd $/GB/month

IMAGE_FAMILY = "pytorch-2-7-cu128-ubuntu-2204-nvidia-570"
IMAGE_PROJECT = "deeplearning-platform-release"

SSH_USER = getpass.getuser()
SSH_KEY_PATH = os.path.expanduser("~/.ssh/google_compute_engine")

RECOMMENDED_ZONES = [
    "us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f",
    "us-west1-b", "us-east1-b", "us-east4-a",
    "europe-west4-a", "europe-west4-b",
    "asia-east1-a", "asia-east1-c",
]

# Exclusions for rsync
RSYNC_EXCLUDES = [
    "node_modules", ".git", ".turbo", ".next", "data", "runs", "outputs",
    "artifacts", ".env.local", "*.node", "*.db", "*.db-wal",
    "*.db-shm", "__pycache__", ".DS_Store",
]

# ---------------------------------------------------------------------------
# gcloud CLI helpers
# ---------------------------------------------------------------------------

def check_gcloud():
    """Verify gcloud CLI is installed and a project is configured."""
    try:
        subprocess.run(["gcloud", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("Error: gcloud CLI not found.")
        print("Install it: https://cloud.google.com/sdk/docs/install")
        sys.exit(1)
    except subprocess.CalledProcessError:
        print("Error: gcloud CLI is not working properly.")
        sys.exit(1)

    r = subprocess.run(
        ["gcloud", "config", "get-value", "project"],
        capture_output=True, text=True,
    )
    project = r.stdout.strip()
    if not project or project == "(unset)":
        print("Error: No GCP project configured.")
        print("Set one with: gcloud config set project YOUR_PROJECT_ID")
        sys.exit(1)

    return project


def gcloud(*args, check=True, capture=True) -> subprocess.CompletedProcess:
    """Run a gcloud command."""
    cmd = ["gcloud"] + list(args)
    if capture:
        r = subprocess.run(cmd, capture_output=True, text=True)
    else:
        r = subprocess.run(cmd)
    if check and r.returncode != 0:
        stderr = r.stderr if hasattr(r, "stderr") and r.stderr else ""
        # Detect quota errors
        if "ZONE_RESOURCE_POOL_EXHAUSTED" in stderr or "Quota" in stderr or "quota exceeded" in stderr.lower():
            print(f"Error: Quota/capacity issue in the requested zone.")
            print(f"Request quota increase: https://console.cloud.google.com/iam-admin/quotas")
            print(f"Or try a different zone: {', '.join(RECOMMENDED_ZONES[:5])}")
            sys.exit(1)
        raise RuntimeError(f"gcloud failed ({r.returncode}): {stderr}")
    return r


def gcloud_json(*args, check=True) -> dict | list:
    """Run a gcloud command and parse JSON output."""
    r = gcloud(*args, "--format=json", check=check)
    if r.returncode != 0:
        return {}
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Instance lifecycle
# ---------------------------------------------------------------------------

def find_instance(zone: str) -> dict | None:
    """Find the alpha-train instance."""
    result = gcloud_json(
        "compute", "instances", "describe", INSTANCE_NAME,
        f"--zone={zone}",
        check=False,
    )
    if not result:
        return None
    return result


def get_instance_status(instance: dict) -> str:
    """Get instance status (RUNNING, TERMINATED, STAGING, etc.)."""
    return instance.get("status", "UNKNOWN")


def get_ip(instance: dict) -> str | None:
    """Extract external IP from instance."""
    for iface in instance.get("networkInterfaces", []):
        for access in iface.get("accessConfigs", []):
            ip = access.get("natIP")
            if ip:
                return ip
    return None


def create_instance(zone: str, machine_type: str = MACHINE_TYPE) -> dict:
    """Create a new GPU instance."""
    print(f"Creating instance {INSTANCE_NAME} in {zone}...")
    print(f"  Machine: {machine_type}")
    print(f"  Image:   {IMAGE_FAMILY} ({IMAGE_PROJECT})")
    print(f"  Disk:    {BOOT_DISK_SIZE_GB}GB SSD")

    create_args = [
        "compute", "instances", "create", INSTANCE_NAME,
        f"--zone={zone}",
        f"--machine-type={machine_type}",
        f"--image-family={IMAGE_FAMILY}",
        f"--image-project={IMAGE_PROJECT}",
        f"--boot-disk-size={BOOT_DISK_SIZE_GB}GB",
        "--boot-disk-type=pd-ssd",
        "--maintenance-policy=TERMINATE",
        "--metadata=install-nvidia-driver=True",
        "--quiet",
    ]

    # Non-A2 machine types need an explicit --accelerator flag
    if not machine_type.startswith("a2-") and not machine_type.startswith("a3-"):
        gpu_type_map = {
            "g2-": "nvidia-l4",
            "n1-": "nvidia-tesla-t4",
        }
        for prefix, gpu_type in gpu_type_map.items():
            if machine_type.startswith(prefix):
                create_args += [f"--accelerator=type={gpu_type},count=1"]
                break

    result = gcloud_json(*create_args)

    if isinstance(result, list) and len(result) > 0:
        instance = result[0]
    elif isinstance(result, dict):
        instance = result
    else:
        # Fetch the instance we just created
        instance = find_instance(zone)
        if not instance:
            raise RuntimeError("Instance creation failed — could not find instance after create.")

    print(f"Instance created.")
    return instance


def start_instance(zone: str):
    """Start a stopped instance."""
    print(f"Starting instance {INSTANCE_NAME}...")
    gcloud("compute", "instances", "start", INSTANCE_NAME, f"--zone={zone}", "--quiet")
    print("Start command sent.")


def stop_instance(zone: str):
    """Stop a running instance (disk persists)."""
    gcloud("compute", "instances", "stop", INSTANCE_NAME, f"--zone={zone}", "--quiet")
    print(f"Instance {INSTANCE_NAME} stopped.")
    disk_cost = BOOT_DISK_SIZE_GB * BOOT_DISK_COST_PER_GB_MONTH
    print(f"Disk cost while stopped: ~${disk_cost:.0f}/month ({BOOT_DISK_SIZE_GB}GB SSD)")


def delete_instance(zone: str):
    """Delete instance and its boot disk."""
    gcloud(
        "compute", "instances", "delete", INSTANCE_NAME,
        f"--zone={zone}", "--quiet",
    )
    print(f"Instance {INSTANCE_NAME} deleted.")


def wait_for_instance(zone: str, timeout: int = 300) -> dict:
    """Wait for instance to be RUNNING with an external IP."""
    print(f"Waiting for instance to be ready...", end="", flush=True)
    start = time.time()

    while time.time() - start < timeout:
        instance = find_instance(zone)
        if instance and get_instance_status(instance) == "RUNNING":
            ip = get_ip(instance)
            if ip:
                print(f" ready! ({int(time.time() - start)}s)")
                return instance
        print(".", end="", flush=True)
        time.sleep(5)

    raise TimeoutError(f"Instance did not start within {timeout}s")


def print_instance_status(instance: dict, zone: str):
    """Pretty-print instance status."""
    status = get_instance_status(instance)
    ip = get_ip(instance)
    machine = instance.get("machineType", "").split("/")[-1]

    print(f"  Instance: {INSTANCE_NAME}")
    print(f"  Zone:     {zone}")
    print(f"  Status:   {status}")
    print(f"  Machine:  {machine}")
    print(f"  Cost:     ~${COST_PER_HR:.2f}/hr (when running)")

    if ip:
        print(f"  IP:       {ip}")
        print(f"  SSH:      gcloud compute ssh {INSTANCE_NAME} --zone={zone}")

    disks = instance.get("disks", [])
    for disk in disks:
        size = disk.get("diskSizeGb", "?")
        print(f"  Disk:     {size}GB")


# ---------------------------------------------------------------------------
# SSH / rsync helpers
# ---------------------------------------------------------------------------

def ssh_opts() -> list[str]:
    return [
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=10",
        "-i", SSH_KEY_PATH,
    ]


def ssh_cmd(ip: str) -> list[str]:
    return ["ssh"] + ssh_opts() + [f"{SSH_USER}@{ip}"]


def ssh_run(ip: str, command: str, stream: bool = False,
            check: bool = True) -> subprocess.CompletedProcess:
    """Run command on instance via SSH."""
    cmd = ssh_cmd(ip) + [command]
    if stream:
        r = subprocess.run(cmd)
    else:
        r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode != 0:
        stderr = r.stderr if hasattr(r, "stderr") and r.stderr else ""
        stdout = r.stdout if hasattr(r, "stdout") and r.stdout else ""
        raise RuntimeError(f"SSH command failed ({r.returncode}): {stderr or stdout}")
    return r


def bootstrap_ssh(zone: str):
    """Use gcloud compute ssh to bootstrap SSH keys (first-time setup)."""
    if os.path.exists(SSH_KEY_PATH):
        return

    print("Bootstrapping SSH keys via gcloud...")
    gcloud(
        "compute", "ssh", INSTANCE_NAME,
        f"--zone={zone}",
        "--command=echo ssh-ok",
        "--quiet",
        capture=False,
    )


def wait_for_ssh(ip: str, timeout: int = 180):
    """Wait until SSH connection succeeds."""
    print(f"Waiting for SSH ({ip})...", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = subprocess.run(
                ssh_cmd(ip) + ["echo ok"],
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


def rsync_to(ip: str, local_path: str, remote_path: str,
             excludes: list[str] = None):
    """Rsync local -> instance."""
    cmd = [
        "rsync", "-az", "-v",
        "-e", f"ssh {' '.join(ssh_opts())}",
    ]
    for ex in (excludes or []):
        cmd += ["--exclude", ex]
    src = local_path.rstrip("/") + "/"
    cmd += [src, f"{SSH_USER}@{ip}:{remote_path}"]
    subprocess.run(cmd, check=True)


def rsync_from(ip: str, remote_path: str, local_path: str):
    """Rsync instance -> local."""
    os.makedirs(local_path, exist_ok=True)
    cmd = [
        "rsync", "-az", "-v",
        "-e", f"ssh {' '.join(ssh_opts())}",
        f"{SSH_USER}@{ip}:{remote_path}",
        local_path,
    ]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Instance setup (runs once, cached on boot disk)
# ---------------------------------------------------------------------------

def work_dir() -> str:
    return f"/home/{SSH_USER}/alpha"


SETUP_SCRIPT = r"""#!/bin/bash
set -e

MARKER="$HOME/.alpha-setup-done-v4"

if [ -f "$MARKER" ]; then
    echo "Environment already set up."
    node --version
    # Ensure Xvfb is running
    pgrep Xvfb >/dev/null 2>&1 || (Xvfb :99 -screen 0 1024x768x24 </dev/null >/dev/null 2>&1 &)
    export DISPLAY=:99
    exit 0
fi

echo "============================================================"
echo "Setting up Alpha training environment..."
echo "============================================================"

# Node.js 22
echo "Installing Node.js 22..."
curl -fsSL https://nodejs.org/dist/v22.14.0/node-v22.14.0-linux-x64.tar.xz \
    | sudo tar -xJ -C /usr/local --strip-components=1
echo "Node $(node --version), npm $(npm --version)"

# Vulkan loader + Xvfb + build tools + NVIDIA GL/Vulkan userspace
# The Deep Learning VM ships nvidia-*-server drivers which lack Vulkan ICD.
# libnvidia-gl-570-server provides libGLX_nvidia.so + the Vulkan ICD JSON.
echo "Installing Vulkan, Xvfb, build tools, NVIDIA GL libraries..."
sudo apt-get update -qq
DEBIAN_FRONTEND=noninteractive sudo apt-get install -y -qq \
    libvulkan1 vulkan-tools xvfb rsync gcc make \
    libnvidia-gl-570-server 2>/dev/null || true

# Kill any stale Xvfb before restarting (e.g. from a previous partial setup)
sudo pkill -9 Xvfb 2>/dev/null || true
sleep 1

# Start Xvfb virtual display — NVIDIA Vulkan ICD needs X11 display
echo "Starting Xvfb virtual display..."
(Xvfb :99 -screen 0 1024x768x24 </dev/null >/dev/null 2>&1 &)
sleep 1
export DISPLAY=:99

# Verify GPU + Vulkan
echo "Verifying NVIDIA driver..."
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

echo "Verifying Vulkan..."
DISPLAY=:99 vulkaninfo --summary 2>&1 | grep -E 'deviceName|deviceType' || true

touch "$MARKER"
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
"""


def setup_instance(ip: str):
    """Install Node.js + Vulkan on the instance (idempotent)."""
    print("\n" + "=" * 60)
    print("Setting up instance environment...")
    print("=" * 60)
    ssh_run(ip, f"cat > /tmp/setup.sh << 'SETUP_EOF'\n{SETUP_SCRIPT}\nSETUP_EOF\nchmod +x /tmp/setup.sh")
    ssh_run(ip, "bash /tmp/setup.sh", stream=True)


# ---------------------------------------------------------------------------
# Code sync + build
# ---------------------------------------------------------------------------

def sync_code(ip: str, project_dir: str):
    """Sync project code to instance."""
    print("\n" + "=" * 60)
    print("Syncing project code...")
    print("=" * 60)

    remote_dir = work_dir()
    ssh_run(ip, f"mkdir -p {remote_dir}")
    rsync_to(ip, project_dir, remote_dir + "/", excludes=RSYNC_EXCLUDES)


def build_on_instance(ip: str):
    """npm install + build native addon + TypeScript on instance."""
    print("\n" + "=" * 60)
    print("Building project on instance...")
    print("=" * 60)

    wd = work_dir()
    build_cmd = f"""
        set -e
        cd {wd}

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
    ssh_run(ip, build_cmd, stream=True)


# ---------------------------------------------------------------------------
# Dataset management
# ---------------------------------------------------------------------------

def upload_dataset(ip: str, local_data_path: str) -> str:
    """Upload dataset to instance if not already present. Returns remote path."""
    data_name = os.path.basename(local_data_path)
    remote_data_dir = f"{work_dir()}/datasets"
    remote_data_path = f"{remote_data_dir}/{data_name}"

    # Check if already uploaded
    r = ssh_run(ip, f"test -f {remote_data_path} && stat -c%s {remote_data_path} || echo MISSING",
                check=False)
    local_size = os.path.getsize(local_data_path)

    if r.returncode == 0 and r.stdout.strip() != "MISSING":
        remote_size = int(r.stdout.strip())
        if remote_size == local_size:
            print(f"Dataset '{data_name}' already on instance ({local_size / 1024 / 1024:.1f} MB) — skipping upload.")
            return remote_data_path

    # Upload
    print(f"Uploading {data_name} ({local_size / 1024 / 1024:.1f} MB)...")
    ssh_run(ip, f"mkdir -p {remote_data_dir}")
    cmd = [
        "scp"] + ssh_opts() + [
        local_data_path, f"{SSH_USER}@{ip}:{remote_data_path}",
    ]
    subprocess.run(cmd, check=True)
    print("Upload complete.")
    return remote_data_path


def download_dataset(ip: str, url: str, data_name: str | None = None) -> str:
    """Download a dataset directly on the instance from a URL. Returns remote path."""
    if not data_name:
        data_name = url.rstrip("/").split("/")[-1].split("?")[0]
        if not data_name:
            data_name = "dataset.txt"

    remote_data_dir = f"{work_dir()}/datasets"
    remote_data_path = f"{remote_data_dir}/{data_name}"

    # Check if already downloaded
    r = ssh_run(ip, f"test -f {remote_data_path} && stat -c%s {remote_data_path} || echo MISSING",
                check=False)
    if r.returncode == 0 and r.stdout.strip() != "MISSING":
        size = int(r.stdout.strip())
        print(f"Dataset '{data_name}' already on instance ({size / 1024 / 1024:.1f} MB) — skipping download.")
        return remote_data_path

    print(f"Downloading {data_name} on instance from {url}...")
    ssh_run(ip, f"mkdir -p {remote_data_dir}")
    ssh_run(ip, f"curl -fSL --retry 3 -o {remote_data_path} '{url}'", stream=True)

    # Verify download
    r = ssh_run(ip, f"stat -c%s {remote_data_path}")
    size = int(r.stdout.strip())
    print(f"Downloaded {data_name} ({size / 1024 / 1024:.1f} MB)")
    return remote_data_path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training(ip: str, remote_data_path: str, train_args: str,
                 run_id: str) -> str:
    """Execute training on the instance. Returns run ID."""
    wd = work_dir()
    run_dir = f"{wd}/runs/{run_id}"

    # Export env vars so metrics stream to the dashboard
    env_exports = "export DISPLAY=:99 && "
    for var in ["ALPHA_REMOTE_URL", "ALPHA_REMOTE_SECRET", "DISCORD_WEBHOOK_URL"]:
        val = os.environ.get(var)
        if val:
            env_exports += f"export {var}='{val}' && "

    cmd = (
        f"cd {wd} && "
        f"{env_exports}"
        f"(pgrep Xvfb >/dev/null 2>&1 || nohup Xvfb :99 -screen 0 1024x768x24 >/dev/null 2>&1 &); sleep 0.5 && "
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

    r = ssh_run(ip, cmd, stream=True, check=False)

    if r.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {r.returncode}")

    return run_id


def download_results(ip: str, run_id: str, local_dir: str):
    """Download training results from instance."""
    wd = work_dir()
    remote_run = f"{wd}/runs/{run_id}/"
    local_run = os.path.join(local_dir, run_id)

    print(f"\nDownloading results to {local_run}/...")
    rsync_from(ip, remote_run, local_run + "/")

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

def ensure_instance(zone: str, machine_type: str = MACHINE_TYPE) -> tuple[dict, str]:
    """Find or create instance, wait for SSH. Returns (instance, ip)."""
    instance = find_instance(zone)

    if instance:
        status = get_instance_status(instance)
        print(f"Found existing instance {INSTANCE_NAME} (status: {status})")

        if status == "TERMINATED":
            start_instance(zone)
            instance = wait_for_instance(zone)
        elif status != "RUNNING":
            instance = wait_for_instance(zone)
    else:
        print(f"No existing instance found. Creating {machine_type} in {zone}...")
        instance = create_instance(zone, machine_type)
        instance = wait_for_instance(zone)

    ip = get_ip(instance)
    if not ip:
        print("Error: Instance is running but has no external IP.")
        sys.exit(1)

    # Bootstrap SSH keys via gcloud (first-time only)
    bootstrap_ssh(zone)

    wait_for_ssh(ip)
    return instance, ip


def train_pipeline(args):
    """Full pipeline: create/reuse instance -> setup -> sync -> train -> download."""
    # Resolve data source: either local file (--data) or remote URL (--data-url)
    use_remote_data = bool(args.data_url)
    data_path = None
    data_name = None
    data_size = 0

    if use_remote_data:
        # Extract filename from URL
        data_name = args.data_url.rstrip("/").split("/")[-1].split("?")[0]
        if not data_name:
            data_name = "dataset.txt"
        data_size_str = "remote"
    else:
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
        data_size_str = f"{data_size / 1024 / 1024:.1f} MB"

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
    if args.sample_interval:
        train_args_parts.append(f"--sampleInterval={args.sample_interval}")
    if args.accum_steps:
        train_args_parts.append(f"--accumSteps={args.accum_steps}")
    if args.warmup:
        train_args_parts.append(f"--warmupIters={args.warmup}")
    if args.beta2 is not None:
        train_args_parts.append(f"--beta2={args.beta2}")
    if args.spike_threshold is not None:
        train_args_parts.append(f"--spikeThreshold={args.spike_threshold}")
    train_args = " ".join(train_args_parts)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    project_dir = str(Path(__file__).resolve().parent.parent)
    start_time = time.time()

    machine_type = args.machine_type

    print("=" * 60)
    print("Alpha Training — GCP GPU")
    print("=" * 60)
    print(f"  Dataset:   {data_name} ({data_size_str})")
    print(f"  Zone:      {args.zone}")
    print(f"  Machine:   {machine_type}")
    print(f"  Backend:   {args.backend}")
    print(f"  Iters:     {args.iters}  Batch: {args.batch}  Block: {args.block}")
    print(f"  Model:     {args.dim}d {args.heads}h {args.layers}L")
    print(f"  LR:        {args.lr}  Tokenizer: {args.tokenizer}")
    if args.domain:
        print(f"  Domain:    {args.domain}")
    print("=" * 60)

    # 1. Provision instance
    instance, ip = ensure_instance(args.zone, machine_type)
    print(f"\nMachine: {machine_type} (~${COST_PER_HR:.2f}/hr for A100)")

    # 2. Setup environment (idempotent)
    setup_instance(ip)

    # 3. Sync code
    sync_code(ip, project_dir)

    # 4. Build
    build_on_instance(ip)

    # 5. Get dataset onto instance
    if use_remote_data:
        remote_data = download_dataset(ip, args.data_url, data_name)
    else:
        remote_data = upload_dataset(ip, data_path)

    # 6. GPU check
    r = ssh_run(ip,
                "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    print(f"\nGPU: {r.stdout.strip()}")

    # 7. Train
    run_training(ip, remote_data, train_args, run_id)

    # 8. Download results
    download_results(ip, run_id, args.download_dir)

    # 9. Cost summary + optionally stop
    elapsed_hrs = (time.time() - start_time) / 3600
    est_cost = elapsed_hrs * COST_PER_HR
    disk_cost = BOOT_DISK_SIZE_GB * BOOT_DISK_COST_PER_GB_MONTH

    print(f"\nTraining completed in {elapsed_hrs:.1f} hours. Estimated cost: ~${est_cost:.2f}")

    if args.stop_after:
        stop_instance(args.zone)
        print(f"Instance stopped. Disk cost: ~${disk_cost:.0f}/month ({BOOT_DISK_SIZE_GB}GB SSD).")
        print(f"Delete with: python scripts/gcp_train.py --action delete")
    else:
        print(f"\nInstance still running (~${COST_PER_HR:.2f}/hr). Stop with: python scripts/gcp_train.py --action stop")

    print("\n" + "=" * 60)
    print(f"DONE — results in {args.download_dir}/{run_id}/")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Alpha training on GCP A100 GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Action (non-training commands)
    parser.add_argument("--action", choices=["status", "stop", "start", "delete", "ssh"],
                        help="Instance management action (instead of training)")

    # GCP-specific
    parser.add_argument("--zone", default=DEFAULT_ZONE,
                        help=f"GCP zone (default: {DEFAULT_ZONE})")
    parser.add_argument("--machine-type", default=MACHINE_TYPE,
                        help=f"GCP machine type (default: {MACHINE_TYPE})")

    # Training params
    parser.add_argument("--data", help="Path to local training data file (uploaded via scp)")
    parser.add_argument("--data-url", help="URL to download dataset directly on instance (faster than upload)")
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
    parser.add_argument("--trace", action="store_true", help="Enable per-step trace timing")
    parser.add_argument("--eval-interval", type=int, default=0, help="Eval/checkpoint interval")
    parser.add_argument("--grad-clip", type=float, default=None, help="Gradient clipping norm")
    parser.add_argument("--download-dir", default="runs", help="Local directory for results")
    parser.add_argument("--sample-interval", type=int, default=0, help="Sample generation interval")
    parser.add_argument("--accum-steps", type=int, default=0, help="Gradient accumulation steps")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup iterations")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta2")
    parser.add_argument("--spike-threshold", type=float, default=None, help="Skip optimizer step when grad_norm > threshold × EMA")
    parser.add_argument("--stop-after", action="store_true",
                        help="Stop instance after training completes")

    args = parser.parse_args()

    # Verify gcloud is available
    project = check_gcloud()

    # Handle management actions
    if args.action:
        instance = find_instance(args.zone)

        if args.action == "status":
            if not instance:
                print(f"No {INSTANCE_NAME} instance found in {args.zone}.")
                return
            print_instance_status(instance, args.zone)

        elif args.action == "stop":
            if not instance:
                print("No instance to stop.")
                return
            stop_instance(args.zone)

        elif args.action == "start":
            if not instance:
                print("No instance to start. Run a training job to create one.")
                return
            status = get_instance_status(instance)
            if status == "RUNNING":
                print("Instance is already running.")
                return
            start_instance(args.zone)
            instance = wait_for_instance(args.zone)
            ip = get_ip(instance)
            print(f"Instance running at {ip}")

        elif args.action == "delete":
            if not instance:
                print("No instance to delete.")
                return
            resp = input(f"Delete instance {INSTANCE_NAME} in {args.zone}? This deletes the boot disk too. [y/N] ")
            if resp.lower() == "y":
                delete_instance(args.zone)
            else:
                print("Cancelled.")

        elif args.action == "ssh":
            if not instance:
                print("No instance found. Create one first with a training run.")
                return
            status = get_instance_status(instance)
            if status != "RUNNING":
                print(f"Instance is {status}. Start it first with: python scripts/gcp_train.py --action start")
                return
            print(f"Connecting to {INSTANCE_NAME}...")
            os.execvp("gcloud", [
                "gcloud", "compute", "ssh", INSTANCE_NAME,
                f"--zone={args.zone}",
            ])

        return

    # Training pipeline
    if not args.data and not args.data_url:
        parser.error("--data or --data-url is required for training")

    # Handle Ctrl+C gracefully
    def sigint_handler(sig, frame):
        print("\n\nInterrupted! Instance is still running.")
        print(f"Stop it with: python scripts/gcp_train.py --action stop --zone={args.zone}")
        sys.exit(1)
    signal.signal(signal.SIGINT, sigint_handler)

    train_pipeline(args)


if __name__ == "__main__":
    main()
