"""
modal_train.py — Run Alpha training on Modal.com with NVIDIA H100 GPU.

Datasets persist on a Modal Volume (uploaded once, reused across runs).
Code is synced fresh from local on every invocation.
Run outputs are saved to a Volume and downloaded automatically.
The Modal container shuts down when training finishes — no wasted credits.

Usage:
  # First run (uploads dataset, trains, downloads results):
  modal run scripts/modal_train.py --data data/books_all.txt --iters 5000 --backend helios

  # Subsequent runs (dataset already on volume, skips upload):
  modal run scripts/modal_train.py --data books_all.txt --iters 10000 --backend helios

  # Force re-upload dataset:
  modal run scripts/modal_train.py --data data/books_all.txt --upload true

  # List datasets on volume:
  modal volume ls alpha-datasets

  # List completed runs:
  modal volume ls alpha-runs
"""

import modal
import subprocess
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Modal resources
# ---------------------------------------------------------------------------

app = modal.App("alpha-train")

datasets_vol = modal.Volume.from_name("alpha-datasets", create_if_missing=True)
runs_vol = modal.Volume.from_name("alpha-runs", create_if_missing=True)

# ---------------------------------------------------------------------------
# Container image: Node.js 22 + gcc + Vulkan loader
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "curl", "xz-utils", "gcc", "make", "ca-certificates",
        "libvulkan1", "vulkan-tools",
    )
    .run_commands(
        # Node.js 22 — official tarball (includes headers for native addon compilation)
        "curl -fsSL https://nodejs.org/dist/v22.14.0/node-v22.14.0-linux-x64.tar.xz"
        " | tar -xJ -C /usr/local --strip-components=1",
        "node --version && npm --version",
        # NVIDIA Vulkan ICD — the NVIDIA driver is injected by Modal at
        # runtime; we just need the loader to know where to find it.
        # We write ICDs for both common NVIDIA library names.
        "mkdir -p /etc/vulkan/icd.d",
        """echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.0"}}' > /etc/vulkan/icd.d/nvidia_icd.json""",
        """echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libnvidia-vulkan-producer.so","api_version":"1.3.0"}}' > /etc/vulkan/icd.d/nvidia_icd_alt.json""",
    )
    .env({"NODE_ENV": "production"})
    # Mount project code — synced fresh from local on every run
    .add_local_dir(
        ".",
        remote_path="/app",
        ignore=[
            "node_modules",
            ".git",
            ".turbo",
            ".next",
            "data",
            "runs",
            "outputs",
            "artifacts",
            ".env.local",
            "*.node",        # platform-specific native addons
            "*.db",
            "*.db-wal",
            "*.db-shm",
        ],
    )
)

# ---------------------------------------------------------------------------
# Remote training function (runs on H100)
# ---------------------------------------------------------------------------


@app.function(
    gpu="H100",
    image=image,
    volumes={"/datasets": datasets_vol, "/runs": runs_vol},
    secrets=[modal.Secret.from_name("alpha-remote")],
    timeout=6 * 3600,   # 6 hour max
    memory=32768,        # 32 GB RAM
)
def train(data: str, train_args: str) -> str:
    """Build project and run training on H100. Returns the run ID."""
    os.chdir("/app")

    # ── 0. GPU quick check ──────────────────────────────────────────────
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                       capture_output=True, text=True)
    print(f"GPU: {r.stdout.strip()}")

    # ── 1. Install dependencies ──────────────────────────────────────────
    print("=" * 60)
    print("Installing npm dependencies...")
    print("=" * 60)
    r = subprocess.run(
        ["npm", "install", "--ignore-scripts"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise RuntimeError("npm install failed")
    print("Dependencies installed.")

    # ── 2. Build native addon + TypeScript ───────────────────────────────
    print("\n" + "=" * 60)
    print("Building project (native addon + TypeScript)...")
    print("=" * 60)

    # Build native helios addon first (gcc → .node)
    r = subprocess.run(
        ["node", "packages/helios/native/build.mjs"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print("WARNING: Native addon build failed — helios backend unavailable")
        print(r.stdout[-1000:])
        print(r.stderr[-1000:])
    else:
        print("Native addon built.")

    # Build TypeScript (only CLI + training deps, skip web/server)
    r = subprocess.run(
        ["npx", "turbo", "build", "--filter=@alpha/cli"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print("TypeScript build FAILED:")
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise RuntimeError("turbo build failed")
    print("TypeScript build complete.")

    # ── 3. Verify dataset exists ─────────────────────────────────────────
    data_path = f"/datasets/{data}"
    if not os.path.exists(data_path):
        avail = []
        for f in sorted(os.listdir("/datasets")):
            size = os.path.getsize(f"/datasets/{f}")
            avail.append(f"  {f} ({size / 1024 / 1024:.1f} MB)")
        msg = f"Dataset '{data}' not found on volume.\n"
        msg += "Available:\n" + "\n".join(avail) if avail else "  (volume is empty)"
        raise FileNotFoundError(msg)

    # ── 4. Run training ──────────────────────────────────────────────────
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = f"/runs/{run_id}"

    cmd = [
        "node", "--expose-gc", "--max-old-space-size=8192", "apps/cli/dist/main.js", "train",
        f"--data={data_path}",
        f"--runDir={run_dir}",
        *train_args.split(),
    ]

    print("\n" + "=" * 60)
    print(f"RUN {run_id}")
    print(f"CMD: {' '.join(cmd)}")
    print("=" * 60 + "\n")

    r = subprocess.run(cmd, capture_output=True, text=True)
    # Always print output (stdout may contain training logs)
    if r.returncode != 0:
        # On failure, print only the tail of stdout (skip build noise)
        combined = (r.stdout or "") + "\n" + (r.stderr or "")
        # Find where training actually starts
        train_start = combined.find("── alpha training ──")
        if train_start >= 0:
            print(combined[train_start:])
        else:
            # Fallback: print last 6000 chars
            print(combined[-6000:])
        raise RuntimeError(f"Training failed with exit code {r.returncode} (signal {-r.returncode if r.returncode < 0 else 'none'})")
    else:
        if r.stdout:
            out = r.stdout if len(r.stdout) <= 8000 else "... (truncated) ...\n" + r.stdout[-8000:]
            print(out)

    # ── 5. Persist results ───────────────────────────────────────────────
    runs_vol.commit()

    # Show what was saved
    if os.path.exists(run_dir):
        files = os.listdir(run_dir)
        total = sum(os.path.getsize(f"{run_dir}/{f}") for f in files)
        print(f"\nSaved {len(files)} files ({total / 1024 / 1024:.1f} MB) to volume alpha-runs/{run_id}")
        for f in sorted(files):
            sz = os.path.getsize(f"{run_dir}/{f}")
            print(f"  {f} ({sz / 1024 / 1024:.1f} MB)")

    return run_id


# ---------------------------------------------------------------------------
# Local entrypoint — orchestrates upload → train → download
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    data: str,
    iters: int = 1000,
    batch: int = 64,
    block: int = 128,
    dim: int = 128,
    heads: int = 8,
    layers: int = 6,
    lr: float = 3e-4,
    backend: str = "helios",
    tokenizer: str = "bpe",
    domain: str = "",
    upload: bool = False,
    download_dir: str = "runs",
    extra: str = "",
):
    """
    Train an Alpha model on Modal H100 GPU.

    Handles dataset upload, remote training, and result download automatically.
    """
    data_name = os.path.basename(data)

    # ── Step 1: Sync dataset to Modal volume ─────────────────────────────
    # Check if file already exists on the volume
    need_upload = upload  # forced re-upload
    if not need_upload:
        r = subprocess.run(
            ["modal", "volume", "ls", "alpha-datasets"],
            capture_output=True, text=True,
        )
        if data_name not in r.stdout:
            need_upload = True

    if need_upload:
        # Resolve local path
        local_path = data
        if not os.path.exists(local_path):
            local_path = os.path.join("data", data)
        if not os.path.exists(local_path):
            local_path = os.path.join("data", data_name)
        if not os.path.exists(local_path):
            print(f"Error: cannot find dataset locally at '{data}' or 'data/{data_name}'")
            sys.exit(1)

        size_mb = os.path.getsize(local_path) / 1024 / 1024
        print(f"Uploading {local_path} ({size_mb:.1f} MB) to Modal volume...")
        subprocess.run(
            ["modal", "volume", "put", "alpha-datasets", local_path, f"/{data_name}"],
            check=True,
        )
        print("Upload complete.\n")
    else:
        print(f"Dataset '{data_name}' already on Modal volume — skipping upload.\n")

    # ── Step 2: Build training args ──────────────────────────────────────
    args_parts = [
        f"--iters={iters}",
        f"--batch={batch}",
        f"--block={block}",
        f"--dim={dim}",
        f"--heads={heads}",
        f"--layers={layers}",
        f"--lr={lr}",
        f"--backend={backend}",
        f"--tokenizer={tokenizer}",
    ]
    if domain:
        args_parts.append(f"--domain={domain}")
    if extra:
        args_parts.extend(extra.split())
    train_args = " ".join(args_parts)

    # ── Step 3: Launch training on Modal H100 ────────────────────────────
    print("=" * 60)
    print("Launching training on Modal H100...")
    print(f"  Dataset: {data_name}")
    print(f"  Backend: {backend}")
    print(f"  Iters: {iters}  Batch: {batch}  Block: {block}")
    print(f"  Model: {dim}d {heads}h {layers}L")
    print(f"  LR: {lr}  Tokenizer: {tokenizer}")
    print("=" * 60 + "\n")

    run_id = train.remote(data=data_name, train_args=train_args)

    # ── Step 4: Download results ─────────────────────────────────────────
    # modal volume get puts remote dir into local dir, so download to parent
    os.makedirs(download_dir, exist_ok=True)
    local_dir = os.path.join(download_dir, run_id)

    print(f"\nDownloading run to {local_dir}/...")
    subprocess.run(
        ["modal", "volume", "get", "alpha-runs", f"/{run_id}", f"{download_dir}/"],
        check=True,
    )

    print("\n" + "=" * 60)
    print(f"DONE — results saved to {local_dir}/")
    if os.path.exists(local_dir):
        for f in sorted(os.listdir(local_dir)):
            sz = os.path.getsize(os.path.join(local_dir, f))
            print(f"  {f} ({sz / 1024 / 1024:.1f} MB)")
    print("=" * 60)
