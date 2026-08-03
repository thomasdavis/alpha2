#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_dir="${1:-/mnt/donto-data/donto-resources/benchmarks/alpha-helios-coop-accum-${timestamp}}"

mkdir -p "$output_dir"
cd "$repo_root"

git rev-parse HEAD > "$output_dir/SOURCE-COMMIT.txt"
git status --short > "$output_dir/SOURCE-STATUS.txt"
git diff --binary > "$output_dir/SOURCE-DIFF.patch"
sha256sum \
  packages/helios/src/backend.ts \
  packages/helios/src/kernels/matmul-coop.ts \
  packages/helios/native/helios_vk.c \
  packages/train/src/trainer.ts \
  scripts/bench-helios-coop-accum.mjs \
  scripts/run_helios_coop_accum_sweep.sh \
  package-lock.json > "$output_dir/SOURCE-HASHES.sha256"

{
  date -u --iso-8601=seconds
  uname -a
  node --version
  npm --version
  df -h / /workspace
  free -h
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,power.limit,clocks.max.sm,clocks.max.memory --format=csv,noheader
} > "$output_dir/HOST.txt" 2>&1

# Build only the native adapter and the aggregate test graph.  The unrelated
# Next.js application has its own Node/Turbopack compatibility surface and is
# not part of this physical kernel claim.  Force the native addon to bind the
# paid host's compiler/runtime instead of trusting a copied mtime.
HELIOS_NATIVE_FORCE_REBUILD=1 npm run build -w @alpha/helios \
  > "$output_dir/build.log" 2>&1
npx tsc -b packages/tests --pretty false \
  >> "$output_dir/build.log" 2>&1

# This must execute, not skip, on the paid device.  The JSON report is checked
# explicitly so a capability-gated test cannot masquerade as physical proof.
(
  cd packages/tests
  npx vitest run src/gpu-perf.test.ts \
    -t "Cooperative matmul production-pattern oracle" \
    --reporter=json \
    --outputFile="$output_dir/production-oracle.json"
) > "$output_dir/production-oracle.log" 2>&1

jq -e '
  .success == true and
  .numFailedTests == 0 and
  .numPassedTests >= 3 and
  .numPendingTests == 0
' "$output_dir/production-oracle.json" > /dev/null

nice -n 10 node scripts/bench-helios-coop-accum.mjs \
  > "$output_dir/sweep.json" \
  2> "$output_dir/sweep.log"

jq -e '
  .decision.status == "measured" and
  ([.modes[].oracleMaxAbsError] | max) <= 0.000001 and
  ([.modes[].cases[].samples] | min) >= 5
' "$output_dir/sweep.json" > /dev/null

jq -r '
  "# Helios cooperative-matrix accumulation discriminator\n",
  "**Created:** " + .createdAt,
  "**Device:** " + .device.deviceName,
  "**Passes:** " + (.passes | tostring),
  "**Accumulation-rate verdict:** **" + .decision.accumulationRateClass + "**",
  "**Median FP32/F16 accumulator-rate ratio:** " + (.decision.medianF32ToF16AccumRateRatio | tostring),
  "",
  "| Shape | Coop F32-acc TFLOP/s | Coop F16-acc TFLOP/s | Selected FP32 TFLOP/s | F32/F16 acc ratio | F32-acc vs FP32 | Class |",
  "|---|---:|---:|---:|---:|---:|---|",
  (.decision.cases | to_entries[] |
    "| `" + .key + "` | " + (.value.coopF32AccumTflops | tostring) +
    " | " + (.value.coopF16AccumTflops | tostring) +
    " | " + (.value.selectedFp32Tflops | tostring) +
    " | " + (.value.f32ToF16AccumRateRatio | tostring) +
    " | " + (.value.f32AccumVersusSelectedFp32 | tostring) +
    " | " + .value.accumulationRateClass + " |"),
  "",
  "The hardware classification uses the ratio between otherwise identical resident-F16 cooperative kernels. The comparison with selected tiled FP32 and the cast-inclusive mode is practical implementation evidence, not the die-rate classifier. Production-shape cooperative correctness passed before timing began."
' "$output_dir/sweep.json" > "$output_dir/README.md"

(
  cd "$output_dir"
  find . -maxdepth 1 -type f ! -name ARTIFACTS.sha256 -printf '%P\0' \
    | sort -z \
    | xargs -0 sha256sum > ARTIFACTS.sha256
  sha256sum -c ARTIFACTS.sha256
) > "$output_dir/hash-check.log"

echo "$output_dir"
