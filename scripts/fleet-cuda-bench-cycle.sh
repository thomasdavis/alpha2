#!/usr/bin/env bash
set -euo pipefail

INSTANCE="alpha-bench-l4-$(date -u +%Y%m%d%H%M%S)"
ZONE="us-central1-b"
MACHINE="g2-standard-4"
IMAGE_FAMILY="pytorch-2-7-cu128-ubuntu-2204-nvidia-570"
IMAGE_PROJECT="deeplearning-platform-release"
SHAPES="1024x1024x1024,2048x2048x2048,3072x3072x3072"
ITERS=12
WARMUP=6
DTYPE="float32"
PYTHON_BIN="python3"
SHUTDOWN_MODE="delete"  # delete|stop|none
KEEP_FLEET_ENTRY=0
PROJECT=""
RUN_SETUP=0
SCRIPT_START_S="$(date +%s)"
T_INSTANCE_READY=0
T_DEPLOY_DONE=0
T_PREREQ_DONE=0
T_BENCH_DONE=0
T_DOWNLOAD_DONE=0

usage() {
  cat <<EOF
Usage: scripts/fleet-cuda-bench-cycle.sh [options]

Options:
  --instance=<name>          Instance name (default: ${INSTANCE})
  --zone=<zone>              GCP zone (default: ${ZONE})
  --machine=<type>           GCP machine type (default: ${MACHINE})
  --project=<id>             GCP project (default: current gcloud config)
  --shapes=<MxKxN,...>       Matmul shapes (default: ${SHAPES})
  --iters=<n>                Timed iterations (default: ${ITERS})
  --warmup=<n>               Warmup iterations (default: ${WARMUP})
  --dtype=<float16|float32|bfloat16> CUDA dtype (default: ${DTYPE})
  --python=<bin>             Python binary on remote (default: ${PYTHON_BIN})
  --shutdown=<delete|stop|none>       Post-run behavior (default: ${SHUTDOWN_MODE})
  --setup                   Run 'fleet setup' before deploy (default: off)
  --keep-fleet-entry         Keep temporary fleet.json instance entry
  --help                     Show this help
EOF
}

for arg in "$@"; do
  case "$arg" in
    --instance=*) INSTANCE="${arg#*=}" ;;
    --zone=*) ZONE="${arg#*=}" ;;
    --machine=*) MACHINE="${arg#*=}" ;;
    --project=*) PROJECT="${arg#*=}" ;;
    --shapes=*) SHAPES="${arg#*=}" ;;
    --iters=*) ITERS="${arg#*=}" ;;
    --warmup=*) WARMUP="${arg#*=}" ;;
    --dtype=*) DTYPE="${arg#*=}" ;;
    --python=*) PYTHON_BIN="${arg#*=}" ;;
    --shutdown=*) SHUTDOWN_MODE="${arg#*=}" ;;
    --setup) RUN_SETUP=1 ;;
    --keep-fleet-entry) KEEP_FLEET_ENTRY=1 ;;
    --help|-h) usage; exit 0 ;;
    *)
      echo "Unknown option: $arg" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "fleet.json" ]]; then
  echo "fleet.json is required at repo root." >&2
  exit 1
fi

case "${DTYPE}" in
  float16|float32|bfloat16) ;;
  *) echo "Invalid --dtype: ${DTYPE}" >&2; exit 1 ;;
esac

case "${SHUTDOWN_MODE}" in
  delete|stop|none) ;;
  *) echo "Invalid --shutdown: ${SHUTDOWN_MODE}" >&2; exit 1 ;;
esac

if [[ -z "${PROJECT}" ]]; then
  PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
fi
if [[ -z "${PROJECT}" || "${PROJECT}" == "(unset)" ]]; then
  echo "No gcloud project configured. Set one with: gcloud config set project <PROJECT_ID>" >&2
  exit 1
fi

GCLOUD=(gcloud --project="${PROJECT}")
FLEET_PATH="$(pwd)/fleet.json"
FLEET_BACKUP="$(mktemp)"
cp "${FLEET_PATH}" "${FLEET_BACKUP}"

INSTANCE_CREATED=0
INSTANCE_STARTED=0
INSTANCE_IP=""

cleanup() {
  local ec=$?
  set +e

  if [[ "${SHUTDOWN_MODE}" != "none" && ( "${INSTANCE_CREATED}" == "1" || "${INSTANCE_STARTED}" == "1" ) ]]; then
    if [[ "${SHUTDOWN_MODE}" == "delete" ]]; then
      echo "Cleaning up: deleting ${INSTANCE}..."
      "${GCLOUD[@]}" compute instances delete "${INSTANCE}" --zone="${ZONE}" --quiet >/dev/null 2>&1 || true
    elif [[ "${SHUTDOWN_MODE}" == "stop" ]]; then
      echo "Cleaning up: stopping ${INSTANCE}..."
      "${GCLOUD[@]}" compute instances stop "${INSTANCE}" --zone="${ZONE}" --quiet >/dev/null 2>&1 || true
    fi
  fi

  if [[ "${KEEP_FLEET_ENTRY}" == "0" ]]; then
    cp "${FLEET_BACKUP}" "${FLEET_PATH}" >/dev/null 2>&1 || true
  fi
  rm -f "${FLEET_BACKUP}" >/dev/null 2>&1 || true
  exit "${ec}"
}
trap cleanup EXIT

echo "Project:  ${PROJECT}"
echo "Instance: ${INSTANCE}"
echo "Zone:     ${ZONE}"
echo "Machine:  ${MACHINE}"
echo "Shutdown: ${SHUTDOWN_MODE}"
echo ""

if "${GCLOUD[@]}" compute instances describe "${INSTANCE}" --zone="${ZONE}" >/dev/null 2>&1; then
  status="$("${GCLOUD[@]}" compute instances describe "${INSTANCE}" --zone="${ZONE}" --format='value(status)')"
  echo "Instance exists (status=${status})."
  if [[ "${status}" != "RUNNING" ]]; then
    echo "Starting instance..."
    "${GCLOUD[@]}" compute instances start "${INSTANCE}" --zone="${ZONE}" --quiet
    INSTANCE_STARTED=1
  fi
else
  echo "Creating new instance..."
  create_args=(
    compute instances create "${INSTANCE}"
    --zone="${ZONE}"
    --machine-type="${MACHINE}"
    --image-family="${IMAGE_FAMILY}"
    --image-project="${IMAGE_PROJECT}"
    --boot-disk-size=200GB
    --boot-disk-type=pd-ssd
    --maintenance-policy=TERMINATE
    --metadata=install-nvidia-driver=True
    --quiet
  )
  if [[ "${MACHINE}" == g2-* ]]; then
    create_args+=(--accelerator=type=nvidia-l4,count=1)
  fi
  "${GCLOUD[@]}" "${create_args[@]}"
  INSTANCE_CREATED=1
fi

echo "Waiting for instance IP..."
for _ in $(seq 1 120); do
  status="$("${GCLOUD[@]}" compute instances describe "${INSTANCE}" --zone="${ZONE}" --format='value(status)')"
  INSTANCE_IP="$("${GCLOUD[@]}" compute instances describe "${INSTANCE}" --zone="${ZONE}" --format='value(networkInterfaces[0].accessConfigs[0].natIP)')"
  if [[ "${status}" == "RUNNING" && -n "${INSTANCE_IP}" ]]; then
    break
  fi
  sleep 5
done

if [[ -z "${INSTANCE_IP}" ]]; then
  echo "Failed to get external IP for ${INSTANCE}" >&2
  exit 1
fi
echo "Instance ready: ${INSTANCE_IP}"
T_INSTANCE_READY="$(date +%s)"
echo ""

echo "Bootstrapping SSH key via gcloud..."
for _ in $(seq 1 30); do
  if "${GCLOUD[@]}" compute ssh "${INSTANCE}" --zone="${ZONE}" --command="echo ssh-ok" --quiet >/dev/null 2>&1; then
    break
  fi
  sleep 5
done
if ! "${GCLOUD[@]}" compute ssh "${INSTANCE}" --zone="${ZONE}" --command="echo ssh-ok" --quiet >/dev/null 2>&1; then
  echo "Failed to establish SSH to ${INSTANCE} after retries." >&2
  exit 1
fi

echo "Injecting temporary fleet entry..."
node - "${FLEET_PATH}" "${INSTANCE}" "${INSTANCE_IP}" "${ZONE}" "${MACHINE}" <<'NODE'
const fs = require("node:fs");
const [,, fleetPath, name, host, zone, machine] = process.argv;
const cfg = JSON.parse(fs.readFileSync(fleetPath, "utf8"));
cfg.instances[name] = {
  host,
  zone,
  machine,
  gpu: "L4",
  role: "ephemeral cuda benchmark",
  setupDone: false,
};
fs.writeFileSync(fleetPath, JSON.stringify(cfg, null, 2) + "\n");
NODE

echo "Building CLI for fleet commands..."
npm run build -w @alpha/cli

if [[ "${RUN_SETUP}" == "1" ]]; then
  echo "Running one-time setup on ${INSTANCE}..."
  npm run fleet -- setup "${INSTANCE}"
else
  echo "Skipping fleet setup (--setup not set)."
fi

echo "Deploying compiled binary to ${INSTANCE}..."
npm run fleet -- deploy "${INSTANCE}"
T_DEPLOY_DONE="$(date +%s)"

DEPLOY_DIR="$(node -e 'const fs=require("node:fs");const cfg=JSON.parse(fs.readFileSync("fleet.json","utf8"));process.stdout.write(cfg.deployDir)')"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_ID="cuda_bench_${STAMP}"

run_remote_script() {
  local script="$1"
  local script_b64
  script_b64="$(printf '%s' "${script}" | base64 | tr -d '\n')"
  npm run fleet -- run "${INSTANCE}" -- "bash -lc 'echo ${script_b64} | base64 -d >/tmp/alpha-fleet-cycle.sh && bash /tmp/alpha-fleet-cycle.sh'"
}

echo "Ensuring Vulkan + CUDA Python prerequisites on ${INSTANCE}..."
PREREQ_SCRIPT="$(cat <<'EOS'
set -euo pipefail

wait_for_apt_lock() {
  for _ in $(seq 1 120); do
    if sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || \
       sudo fuser /var/lib/apt/lists/lock >/dev/null 2>&1 || \
       sudo fuser /var/cache/apt/archives/lock >/dev/null 2>&1; then
      sleep 5
      continue
    fi
    return 0
  done
  return 1
}

install_pkgs() {
  wait_for_apt_lock
  sudo env DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a apt-get update -qq
  wait_for_apt_lock
  sudo env DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a apt-get install -y -qq \
    libvulkan1 vulkan-tools nvidia-utils-570-server libnvidia-gl-570-server
}

if ! command -v vulkaninfo >/dev/null 2>&1 || \
   ! ldconfig -p | grep -q 'libvulkan\.so' || \
   ! ls /usr/share/vulkan/icd.d/*nvidia*json >/dev/null 2>&1 || \
   ! ldconfig -p | grep -q 'libEGL_nvidia\.so\.0'; then
  install_pkgs
fi

sudo mkdir -p /etc/vulkan/icd.d
cat <<'JSON' | sudo tee /etc/vulkan/icd.d/nvidia_icd_headless.json >/dev/null
{
  "file_format_version": "1.0.1",
  "ICD": {
    "library_path": "libEGL_nvidia.so.0",
    "api_version": "1.4.303"
  }
}
JSON

VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd_headless.json \
  vulkaninfo --summary > /tmp/alpha-vulkan-summary.txt 2>&1 || true
grep -E 'deviceName|vendorID|driverName|driverInfo|GPU[0-9]' /tmp/alpha-vulkan-summary.txt | head -20 || true
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || true
EOS
)"
run_remote_script "${PREREQ_SCRIPT}"
T_PREREQ_DONE="$(date +%s)"

echo "Running remote Helios-vs-CUDA benchmark..."
BENCH_SCRIPT="$(cat <<EOS
set -euo pipefail
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd_headless.json
unset DISPLAY
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH:-}
cd ${DEPLOY_DIR}
mkdir -p runs/${RUN_ID}
start_s=\$(date +%s)
{
  echo "date_utc,\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,memory.used --format=csv || true
} > runs/${RUN_ID}/gpu.csv
vulkaninfo --summary > runs/${RUN_ID}/vulkan-summary.txt 2>&1 || true
./alpha bench --suite=cuda --iters=${ITERS} --warmup=${WARMUP} --shapes=${SHAPES} --dtype=${DTYPE} --python=${PYTHON_BIN} --out=runs/${RUN_ID}/cuda-vs-helios.json | tee runs/${RUN_ID}/cuda-vs-helios.log
end_s=\$(date +%s)
echo "bench_elapsed_sec,\$((end_s-start_s))" > runs/${RUN_ID}/timings.csv
EOS
)"
run_remote_script "${BENCH_SCRIPT}"
T_BENCH_DONE="$(date +%s)"

echo "Downloading benchmark artifacts..."
npm run fleet -- download "${INSTANCE}" --run="${RUN_ID}"
T_DOWNLOAD_DONE="$(date +%s)"

LOCAL_RUN_DIR="$(pwd)/runs/${INSTANCE}-${RUN_ID}"
LOCAL_PERF_DIR="$(pwd)/perf/fleet-cuda/${INSTANCE}-${RUN_ID}"
mkdir -p "$(dirname "${LOCAL_PERF_DIR}")"
cp -R "${LOCAL_RUN_DIR}" "${LOCAL_PERF_DIR}"

TIMING_PATH="${LOCAL_RUN_DIR}/cycle-timings.csv"
cat > "${TIMING_PATH}" <<EOF
stage,seconds
to_instance_ready,$((T_INSTANCE_READY - SCRIPT_START_S))
build_and_deploy,$((T_DEPLOY_DONE - T_INSTANCE_READY))
remote_prereq,$((T_PREREQ_DONE - T_DEPLOY_DONE))
remote_benchmark,$((T_BENCH_DONE - T_PREREQ_DONE))
download_artifacts,$((T_DOWNLOAD_DONE - T_BENCH_DONE))
total,$((T_DOWNLOAD_DONE - SCRIPT_START_S))
EOF
cp "${TIMING_PATH}" "${LOCAL_PERF_DIR}/cycle-timings.csv"

echo ""
echo "Benchmark complete."
echo "  run_id: ${RUN_ID}"
echo "  local:  ${LOCAL_RUN_DIR}"
echo "  perf:   ${LOCAL_PERF_DIR}"
echo "  timings:${TIMING_PATH}"
echo ""

if [[ -f "${LOCAL_RUN_DIR}/cuda-vs-helios.json" ]]; then
  node - "${LOCAL_RUN_DIR}/cuda-vs-helios.json" <<'NODE'
const fs = require("node:fs");
const p = process.argv[2];
const data = JSON.parse(fs.readFileSync(p, "utf8"));
console.log("Summary:");
for (const row of data.rows ?? []) {
  const shape = row.shape;
  const h = row.heliosMs;
  const c = row.cudaMs;
  const ratio = (typeof c === "number" && typeof h === "number" && h > 0) ? (c / h) : null;
  const ratioStr = ratio === null ? "n/a" : (ratio >= 1 ? `${ratio.toFixed(2)}x` : `1/${(1/ratio).toFixed(2)}x`);
  console.log(`  ${shape}: helios=${Number(h).toFixed(3)}ms cuda=${c === null ? "n/a" : Number(c).toFixed(3)+"ms"} h_vs_cuda=${ratioStr}`);
}
NODE
  echo ""
fi
