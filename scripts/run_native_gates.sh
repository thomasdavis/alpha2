#!/usr/bin/env bash
# Fail-closed regression gate for the FROM-SCRATCH backend.
#
# Why this is a separate script from run_nvidia_gates.sh: that one exports
# VK_ICD_FILENAMES and reads device identity out of the Vulkan module. It gates
# the Vulkan path and it should keep doing so — both backends are permanent, and
# a gate that covered only one of them would let the other rot.
#
# What it enforces, which is the same contract:
#   - a real NVIDIA device, vendor 0x10de, reported by the native stack itself
#   - a live channel, so "ran on a GPU" is distinguished from "ran at all"
#   - zero skipped tests, because Vitest exits 0 when everything skips and that
#     is the exact way a suite comes to pass while exercising nothing
#
# Usage: scripts/run_native_gates.sh /abs/path/to/out-dir
set -euo pipefail

out_dir=${1:?output directory required}
[[ $out_dir == /* && $out_dir != / ]] || { echo "output directory must be a concrete absolute path" >&2; exit 2; }
[[ ! -e $out_dir ]] || { echo "output directory already exists: $out_dir" >&2; exit 1; }

for required in packages/helios/dist/index.js packages/helios/native/helios_native.node; do
  [[ -f $required ]] || { echo "required file missing: $required (build with: node packages/helios/native/build-stack.mjs)" >&2; exit 1; }
done

mkdir -p "$out_dir"
# The commit, when there is one. A gate run on a synced tree without .git is
# still a valid gate — it just cannot say which commit it proved.
source_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)

# Device identity, from the native stack rather than from a Vulkan loader.
DEVICE_OUT="$out_dir/device.json" SOURCE_COMMIT="$source_commit" node --input-type=module -e '
  import { writeFileSync } from "node:fs";
  import { NativeHeliosBackend } from "./packages/helios/dist/index.js";
  const info = new NativeHeliosBackend(0).deviceInfo();
  if (!info) { console.error("native gate: no device"); process.exit(1); }
  writeFileSync(process.env.DEVICE_OUT, JSON.stringify({ ...info, sourceCommit: process.env.SOURCE_COMMIT }, null, 2));
  console.log(`native gate: vendor 0x${info.vendorId.toString(16)} minor ${info.minor} channelLive=${info.channelLive}`);
'

vendor=$(node -e 'console.log(require(process.argv[1]).vendorId)' "$out_dir/device.json")
live=$(node -e 'console.log(require(process.argv[1]).channelLive)' "$out_dir/device.json")
[[ $vendor == 4318 ]] || { echo "native gate: vendor $vendor is not 0x10de (4318)" >&2; exit 1; }
[[ $live == true ]] || { echo "native gate: no live channel — the suite would not have touched the GPU" >&2; exit 1; }

# The NATIVE LAYER suites first — C tests that talk to the driver directly.
#
# These were not in the gate, and the gap was not theoretical: a change to the
# matmul emitter faulted two of them while every vitest suite stayed green, and
# the gate said GREEN. The layers below the TypeScript are where the hardware
# actually is; a gate that skips them is gating the wrapper.
echo "native gate: layer suites"
if ! node packages/helios/native/build-stack.mjs > "$out_dir/layers.log" 2>&1; then
  echo "native gate: layer build failed" >&2; tail -20 "$out_dir/layers.log" >&2; exit 1
fi
if ! grep -qE '^[0-9]+ layer suite\(s\), 0 failing' "$out_dir/layers.log"; then
  echo "native gate: layer suites failing" >&2
  grep -E 'FAIL|failing|failures' "$out_dir/layers.log" >&2 || true
  exit 1
fi
grep -E 'layer suite' "$out_dir/layers.log"

# The native suites, with a machine-readable report so skips can be counted.
set +e
npx vitest run --reporter=json --outputFile="$out_dir/report.json" \
  packages/tests/src/known-answer-native.test.ts \
  packages/tests/src/backend-ops-native.test.ts \
  packages/tests/src/parity-native.test.ts \
  packages/tests/src/train-step-native.test.ts \
  packages/tests/src/train-model-native.test.ts > "$out_dir/vitest.log" 2>&1
vitest_status=$?
set -e

node --input-type=module -e '
  import { readFileSync } from "node:fs";
  const r = JSON.parse(readFileSync(process.argv[1], "utf8"));
  const { numPassedTests: pass = 0, numFailedTests: fail = 0, numPendingTests: skip = 0 } = r;
  console.log(`native gate: ${pass} passed, ${fail} failed, ${skip} skipped`);
  /* A bare green exit is not evidence. Vitest reports success when every test
   * skipped, which is precisely how a suite passes while running nothing. */
  if (fail > 0) { console.error("native gate: failures present"); process.exit(1); }
  if (skip > 0) { console.error("native gate: skipped tests are not allowed"); process.exit(1); }
  if (pass < 30) { console.error(`native gate: only ${pass} tests ran; expected the full suite`); process.exit(1); }
' "$out_dir/report.json"

[[ $vitest_status -eq 0 ]] || { echo "native gate: vitest exited $vitest_status" >&2; exit 1; }
echo "native gate: GREEN ($out_dir)"
