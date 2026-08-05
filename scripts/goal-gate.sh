#!/usr/bin/env bash
# THE GOAL, as one command.
#
#   native  >= 30,000 tok/s   and   vulkan >= 10,000 tok/s
#   at ~100M parameters, on an RTX 3070, at the same commit.
#
# Both numbers must hold together, so this runs both and fails if either does.
# The shape is fixed here on purpose: a target is only a target if the thing it
# is measured on cannot drift. 18 layers, 640 embd, 10 heads, vocab 12,288,
# 64 tokens a sequence — 104.4M parameters, the shape both backends were last
# measured at.
#
# ONE BACKEND PER PROCESS, deliberately: with the native channel open the same
# Vulkan binary measures a fraction of its real throughput, so a shared process
# cannot produce a comparable pair (X61).
#
# Batch is per-backend because the two peak in different places and the goal is
# about each backend's throughput, not about a shared batch: Vulkan regresses
# above ~768 tokens a step (batch 8 gives 9,126 against batch 6's 10,745) while
# native still climbs. Override with NATIVE_BATCH / VULKAN_BATCH.
#
# NATIVE'S DEFAULT IS ITS MEASURED PEAK, and 24 is where the card runs out
# rather than where the curve turns:
#
#     12   13,474 tok/s   held 3.70 GB
#     16   13,606         held 5.34
#     20   13,622         held 5.41
#     24   14,254         held 5.38
#     32   FAILS — "allocation of 1310720 floats failed" on an 8 GB card
#
# So MEMORY is what caps native's batch, not throughput, and the step's ~275 MB
# per layer of intermediates is therefore a speed constraint as well as a
# footprint one. The curve is also flattening — twice the batch buys ~6% — so
# this is not a route to the target, only to its own last few percent.
#
# Usage: scripts/goal-gate.sh [outdir]
set -uo pipefail

cd "$(dirname "$0")/.."

L=${L:-18} D=${D:-640} H=${H:-10} V=${V:-12288} SEQ=${SEQ:-64}
NATIVE_BATCH=${NATIVE_BATCH:-24}
VULKAN_BATCH=${VULKAN_BATCH:-6}
NATIVE_TARGET=${NATIVE_TARGET:-30000}
VULKAN_TARGET=${VULKAN_TARGET:-10000}
out_dir=${1:-}

# VIDMEM is not a tuning flag — without it the native backend holds a 105M
# model's tensors in host memory and measures 31 tok/s instead of 1,179, a 38x
# error. It is how the backend is meant to run at this size.
export HELIOS_VIDMEM=${HELIOS_VIDMEM:-1}
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd_headless.json}
unset DISPLAY

run() { # backend batch -> prints tok/s on stdout, the full report on stderr
  local backend=$1 batch=$2 log
  log=$(node packages/tests/bench-shape.mjs "$L" "$D" "$H" "$V" "$SEQ" "$batch" "$backend" 2>&1)
  echo "$log" >&2
  [[ -n $out_dir ]] && { mkdir -p "$out_dir"; echo "$log" > "$out_dir/$backend.log"; }
  # The median line, e.g. "    1179 tok/s   median 434.3 ms ..."
  echo "$log" | grep -oE '^ *[0-9]+ tok/s' | head -1 | grep -oE '[0-9]+'
}

echo "=== goal gate — ${L}L ${D}d ${H}h vocab ${V} seq ${SEQ} ==="
echo
native=$(run native "$NATIVE_BATCH")
vulkan=$(run vulkan "$VULKAN_BATCH")
: "${native:=0}" "${vulkan:=0}"

fail=0
verdict() { # name got want
  if (( $2 >= $3 )); then printf '  %-8s %7d tok/s  >= %6d  PASS\n' "$1" "$2" "$3"
  else printf '  %-8s %7d tok/s  <  %6d  FAIL  (%.2fx to go)\n' "$1" "$2" "$3" \
       "$(echo "$3 $2" | awk '{print $1/($2?$2:1)}')"; fail=1; fi
}

echo
echo "=== verdict ==="
verdict native "$native" "$NATIVE_TARGET"
verdict vulkan "$vulkan" "$VULKAN_TARGET"
[[ -n $out_dir ]] && printf '{"native":%d,"vulkan":%d,"nativeTarget":%d,"vulkanTarget":%d,"pass":%s,"commit":"%s"}\n' \
  "$native" "$vulkan" "$NATIVE_TARGET" "$VULKAN_TARGET" \
  "$([[ $fail == 0 ]] && echo true || echo false)" \
  "$(git rev-parse HEAD 2>/dev/null || echo unknown)" > "$out_dir/goal.json"
echo
[[ $fail == 0 ]] && echo "GOAL MET — both targets hold at the same commit." || echo "GOAL NOT MET"
exit $fail
