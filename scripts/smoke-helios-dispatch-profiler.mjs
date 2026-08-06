#!/usr/bin/env node

/**
 * Destructive-state-safe smoke test for Helios's in-batch timestamp profiler.
 *
 * The environment flag must be installed before importing @alpha/helios because
 * profiling is intentionally selected once at backend module initialization.
 * This script executes the original graph exactly once; it never replays an
 * optimizer or in-place operation to obtain a timing.
 */
process.env.HELIOS_PROFILE_GPU_TIMESTAMPS = "1";

const { HeliosBackend, destroyDevice, getDeviceInfo } = await import("@alpha/helios");

const gpu = new HeliosBackend();
gpu.setMinGpuSize(1);
gpu.resetStepOps();

const elementCount = Number.parseInt(process.env.HELIOS_PROFILE_SMOKE_ELEMENTS ?? "262144", 10);
if (!Number.isSafeInteger(elementCount) || elementCount < 4) {
  throw new Error("HELIOS_PROFILE_SMOKE_ELEMENTS must be an integer >= 4");
}

const leftValues = new Float32Array(elementCount);
const rightValues = new Float32Array(elementCount);
for (let i = 0; i < elementCount; i++) {
  leftValues[i] = i % 17;
  rightValues[i] = i % 11;
}

const left = gpu.fromArray(leftValues, [elementCount]);
const right = gpu.fromArray(rightValues, [elementCount]);
const sum = gpu.add(left, right);
const half = gpu.scale(sum, 0.5);
const result = gpu.sub(half, left);
const actual = result.data;

for (const index of [0, 1, 17, elementCount - 1]) {
  const expected = 0.5 * (leftValues[index] + rightValues[index]) - leftValues[index];
  if (Math.abs(actual[index] - expected) > 1e-6) {
    throw new Error(`result mismatch at ${index}: actual=${actual[index]} expected=${expected}`);
  }
}

const stats = gpu.getGpuStepStats();
if (!stats.timingEnabled) throw new Error("timestamp profiling was not enabled");
if (stats.operations !== 3) throw new Error(`expected 3 operations, observed ${stats.operations}`);
if (stats.timestampedFlushes !== 1) {
  throw new Error(`expected 1 timestamped flush, observed ${stats.timestampedFlushes}`);
}
if (!(stats.batchGpuTimeUs > 0) || !(stats.dispatchGpuTimeUs > 0)) {
  throw new Error(`invalid GPU timing totals: ${JSON.stringify(stats)}`);
}
if (stats.byKernel.length !== 3 || stats.byKernel.some((row) => !(row.gpuTimeUs > 0))) {
  throw new Error(`missing per-kernel timings: ${JSON.stringify(stats.byKernel)}`);
}

console.log(JSON.stringify({
  schema: "helios-dispatch-profiler-smoke-v1",
  device: getDeviceInfo(),
  elementCount,
  stats,
}, null, 2));

destroyDevice();
