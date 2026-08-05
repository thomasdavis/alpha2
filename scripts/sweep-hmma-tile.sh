#!/usr/bin/env bash
# Sweep the tensor-core GEMM's tile geometry against the shapes the model runs.
#
# The GEMM is 64% of the step at ~17 TFLOP/s against cuBLAS's 24-32, and every
# cheap structural theory about the gap is spent: occupancy, arithmetic
# intensity, exposed load latency, issue spacing, bank conflicts, barriers,
# ldmatrix, L2 bandwidth, and now f16 accumulate. What is NOT spent is warp-count
# geometry — more warps grows the BLOCK tile, and therefore the operand reuse in
# shared memory, WITHOUT growing per-thread registers, which is what sank the
# three earlier register-tile sweeps.
#
# ⚠️ A tile also decides WHICH SHAPES TAKE THIS PATH AT ALL: pr_hmma_applies
# needs M % blockRows == 0 and N % blockCols == 0, and the attention multiplies
# are m64 n64. A wider tile that drops them onto the scalar kernel can lose more
# than the projections gain, so the gate is the arbiter, not the probe.
#
# ⚠️ AND A BUILD THAT FAILS LEAVES THE PREVIOUS ADDON IN PLACE, so a neutral
# reading and a failed build look identical. Every row here checks the build.
set -uo pipefail
cd "$(dirname "$0")/.."
export HELIOS_VIDMEM=1

run() { # name flags...
  local name=$1; shift
  local out
  if ! out=$(cd packages/helios && HELIOS_CFLAGS="$*" node native/build-stack.mjs 2>&1); then
    printf '  %-22s BUILD FAILED\n' "$name"
    echo "$out" | grep -E 'error|Error' | head -3 | sed 's/^/      /'
    return
  fi
  local fails
  fails=$(echo "$out" | grep -oE '[0-9]+ failing' | head -1)
  if [[ $fails != "0 failing" ]]; then
    printf '  %-22s SUITE %s\n' "$name" "$fails"
    return
  fi
  local rates
  rates=$(M=1536 node packages/tests/probe-gemm-rate.mjs 2>/dev/null \
          | grep -oE '[0-9]+\.[0-9]+ TFLOP/s' | grep -oE '^[0-9.]+' | tr '\n' ' ')
  local mean
  mean=$(echo "$rates" | awk '{s=0;n=0;for(i=1;i<=NF;i++){s+=$i;n++}; if(n)printf "%.2f", s/n}')
  printf '  %-22s mean %s TFLOP/s   [%s]\n' "$name" "$mean" "$rates"
}

echo "GEMM tile sweep — mean TFLOP/s over the eight projection shapes at m1536"
echo
# The REGISTER tile decides how much SHARED MEMORY traffic a fragment buys: a
# warp reads TM A-fragments and TN B-fragments to do TM*TN multiplies, so the
# traffic per FLOP goes as (TM+TN)/(TM*TN) — 0.75 at 2x4, 0.50 at 4x4, 0.625 at
# 2x8. That is the one resource this kernel has never been measured against;
# DRAM, L2, occupancy, issue and barriers are all spent. It trades against
# occupancy, which is why it is a sweep and not a change.
run "2x4 tile, 2x2 warps"  "-DHMMA_TM=2 -DHMMA_TN=4 -DHMMA_WARPS_M=2 -DHMMA_WARPS_N=2"
# SMALLER tiles, for the opposite reason to the bigger ones: BLOCK COUNT.
# Every N=640 shape in the step runs at 10.9-14.8 TFLOP/s while every large-N
# shape runs at 17.7-20.9, and N=640 with a 64-wide block tile is 240 blocks
# against roughly 184 resident — 1.3 waves, so a third of the time most of the
# card is idle. 240/(2*184) = 65%, which is what the slow shapes achieve. These
# halve the tile in one direction to double the blocks; the cost is arithmetic
# intensity per warp, which is why it has to be measured rather than assumed.
run "2x2 tile, 2x2 warps"  "-DHMMA_TM=2 -DHMMA_TN=2 -DHMMA_WARPS_M=2 -DHMMA_WARPS_N=2"
run "1x4 tile, 2x2 warps"  "-DHMMA_TM=1 -DHMMA_TN=4 -DHMMA_WARPS_M=2 -DHMMA_WARPS_N=2"
run "1x8 tile, 2x2 warps"  "-DHMMA_TM=1 -DHMMA_TN=8 -DHMMA_WARPS_M=2 -DHMMA_WARPS_N=2"
run "4x2 tile, 2x2 warps"  "-DHMMA_TM=4 -DHMMA_TN=2 -DHMMA_WARPS_M=2 -DHMMA_WARPS_N=2"
run "4x4 tile, 2x2 warps"  "-DHMMA_TM=4 -DHMMA_TN=4 -DHMMA_WARPS_M=2 -DHMMA_WARPS_N=2"
run "2x8 tile, 2x2 warps"  "-DHMMA_TM=2 -DHMMA_TN=8 -DHMMA_WARPS_M=2 -DHMMA_WARPS_N=2"
echo
echo "restoring the default build"
(cd packages/helios && node native/build-stack.mjs >/dev/null 2>&1) && echo "  done"
