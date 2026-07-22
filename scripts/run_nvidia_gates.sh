#!/usr/bin/env bash
# Run the exact Helios NVIDIA regression gate and reject Vitest's successful-all-skipped state.
# Usage: scripts/run_nvidia_gates.sh /workspace/alpha2/runs/nvidia-gate-<name>

set -euo pipefail

out_dir=${1:?output directory required}
[[ $out_dir == /* && $out_dir != / ]] || { echo "output directory must be a concrete absolute path" >&2; exit 2; }
[[ ! -e $out_dir ]] || { echo "output directory already exists: $out_dir" >&2; exit 1; }
for required in packages/helios/dist/index.js packages/tests/package.json scripts/verify_nvidia_gate_report.ts; do
  [[ -f $required ]] || { echo "required file missing: $required" >&2; exit 1; }
done
[[ -z $(git status --porcelain) ]] || { echo "NVIDIA gate requires a clean worktree" >&2; exit 1; }

export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd_headless.json}
mkdir -p "$out_dir"
source_commit=$(git rev-parse HEAD)
device_tmp="$out_dir/device.json.tmp"
DEVICE_OUT="$device_tmp" SOURCE_COMMIT="$source_commit" nice -n 10 ionice -c 2 -n 7 node --input-type=module -e '
  import { writeFileSync } from "node:fs";
  import { destroyDevice, getDeviceInfo } from "./packages/helios/dist/index.js";
  const info = getDeviceInfo();
  const record = { ...info, sourceCommit: process.env.SOURCE_COMMIT };
  writeFileSync(process.env.DEVICE_OUT, JSON.stringify(record, null, 2) + "\n", { flag: "wx" });
  destroyDevice();
  if (info.vendorId !== 0x10de) throw new Error(`NVIDIA gate requires vendorId 0x10de, found 0x${info.vendorId.toString(16)} (${info.deviceName})`);
'
mv "$device_tmp" "$out_dir/device.json"

report_tmp="$out_dir/vitest.json.tmp"
(
  cd packages/tests
  unset HELIOS_DISABLE_COOP_MAT
  nice -n 5 ionice -c 2 -n 7 npx vitest run \
    src/parity-helios.test.ts src/gpu-perf.test.ts \
    --reporter=json --outputFile="$report_tmp"
)
mv "$report_tmp" "$out_dir/vitest.json"
nice -n 10 ionice -c 2 -n 7 npx tsx scripts/verify_nvidia_gate_report.ts \
  --report "$out_dir/vitest.json" \
  --device "$out_dir/device.json" \
  --sourceCommit "$source_commit" \
  --out "$out_dir/gate-summary.json"
