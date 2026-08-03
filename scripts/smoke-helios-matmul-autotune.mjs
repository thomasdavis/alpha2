#!/usr/bin/env node

/**
 * Numerical and capability smoke test for portable FP32 GEMM autotuning.
 *
 * The selector is enabled before @alpha/helios is imported.  Every result is
 * compared with an independent CPU reference, while the emitted decisions
 * prove that the driver timed legal candidates instead of relying on a vendor
 * name or a static size threshold.
 */
process.env.HELIOS_DISABLE_COOP_MAT = "1";
process.env.HELIOS_PROFILE_GPU_OPS = "1";
const requestedVariant = process.env.HELIOS_MATMUL_SMOKE_VARIANT;
const variant = ["reg2x2", "reg4x2", "reg2x2c", "reg4x2c", "reg4x2ca"].includes(requestedVariant)
  ? requestedVariant
  : "autotune";
if (variant === "reg2x2" || variant === "reg2x2c") {
  process.env.HELIOS_MATMUL_REG2X2 = "1";
} else if (variant.startsWith("reg4x2")) {
  process.env.HELIOS_MATMUL_REG4X2 = "1";
  // This smoke validates all three R4x2 shader layouts. Production keeps the
  // transposed-B family independently selectable because the measured Alpha
  // portfolio is faster with R2 for that layout.
  process.env.HELIOS_MATMUL_REG4X2_TRANSPOSED_B = "1";
} else {
  process.env.HELIOS_MATMUL_TILE_AUTOTUNE = "1";
  process.env.HELIOS_MATMUL_TILE_AUTOTUNE_LOG ??= "1";
}
if (variant === "reg2x2c" || variant === "reg4x2c") {
  process.env.HELIOS_MATMUL_TRANSPOSED_B_COALESCED = "1";
}
if (variant === "reg4x2ca") {
  process.env.HELIOS_MATMUL_TRANSPOSED_A_COALESCED = "1";
}

const { HeliosBackend, destroyDevice, getDeviceInfo } = await import("@alpha/helios");

const gpu = new HeliosBackend();
gpu.setMinGpuSize(1);
gpu.resetStepOps();

function values(length, phase) {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    out[i] = Math.sin((i + phase) * 0.017) * 0.25 + Math.cos((i + phase) * 0.031) * 0.125;
  }
  return out;
}

function assertClose(label, actual, expected, tolerance = 2e-4) {
  if (actual.length !== expected.length) {
    throw new Error(`${label}: length mismatch ${actual.length} != ${expected.length}`);
  }
  let maxAbsError = 0;
  let worstIndex = -1;
  for (let i = 0; i < actual.length; i++) {
    const error = Math.abs(actual[i] - expected[i]);
    if (error > maxAbsError) {
      maxAbsError = error;
      worstIndex = i;
    }
  }
  if (maxAbsError > tolerance) {
    throw new Error(`${label}: max error ${maxAbsError} at ${worstIndex} exceeds ${tolerance}`);
  }
  return maxAbsError;
}

function referenceMatmul(a, b, M, N, K) {
  const out = new Float32Array(M * N);
  for (let row = 0; row < M; row++) {
    for (let col = 0; col < N; col++) {
      let sum = 0;
      for (let k = 0; k < K; k++) sum += a[row * K + k] * b[k * N + col];
      out[row * N + col] = sum;
    }
  }
  return out;
}

function referenceMatmulTransposed(a, b, M, N, K) {
  const out = new Float32Array(M * N);
  for (let row = 0; row < M; row++) {
    for (let col = 0; col < N; col++) {
      let sum = 0;
      for (let k = 0; k < K; k++) sum += a[row * K + k] * b[col * K + k];
      out[row * N + col] = sum;
    }
  }
  return out;
}

function referenceMatmulTransposedA(a, b, M, N, K) {
  const out = new Float32Array(K * N);
  for (let row = 0; row < K; row++) {
    for (let col = 0; col < N; col++) {
      let sum = 0;
      for (let k = 0; k < M; k++) sum += a[k * K + row] * b[k * N + col];
      out[row * N + col] = sum;
    }
  }
  return out;
}

// Intentionally cross every 16/32 tile boundary so edge masking is exercised.
const M = 113;
const N = 157;
const K = 93;
const aValues = values(M * K, 1);
const bValues = values(K * N, 2);
const btValues = values(N * K, 3);
const baValues = values(M * N, 4);

const a = gpu.fromArray(aValues, [M, K]);
const b = gpu.fromArray(bValues, [K, N]);
const bt = gpu.fromArray(btValues, [N, K]);
const ba = gpu.fromArray(baValues, [M, N]);

const errors = {
  matmul: assertClose(
    "matmul",
    gpu.matmul(a, b).data,
    referenceMatmul(aValues, bValues, M, N, K),
  ),
  matmulTransposed: assertClose(
    "matmulTransposed",
    gpu.matmulTransposed(a, bt).data,
    referenceMatmulTransposed(aValues, btValues, M, N, K),
  ),
  matmulTransposedA: assertClose(
    "matmulTransposedA",
    gpu.matmulTransposedA(a, ba).data,
    referenceMatmulTransposedA(aValues, baValues, M, N, K),
  ),
};

const decisions = gpu.getMatmulTileAutotuneDecisions();
const stats = gpu.getGpuStepStats();
if (variant === "autotune") {
  const expectedKernels = new Set(["matmul", "matmul_transposed", "matmul_transposed_a"]);
  for (const decision of decisions) expectedKernels.delete(decision.kernel);
  if (expectedKernels.size > 0) {
    throw new Error(`missing autotune decisions for ${[...expectedKernels].join(", ")}`);
  }
  if (decisions.some((decision) => decision.reason !== "measured")) {
    throw new Error(`expected measured decisions: ${JSON.stringify(decisions)}`);
  }
} else {
  const suffix = variant.startsWith("reg4x2") ? "R42" : "R2";
  const transposedSuffix = `${suffix}${variant === "reg2x2c" || variant === "reg4x2c" ? "C" : ""}`;
  const transposedASuffix = `${suffix}${variant === "reg4x2ca" ? "C" : ""}`;
  const expectedKernels = new Set([
    `matmul_${suffix}`,
    `matmul_transposed_${transposedSuffix}`,
    `matmul_transposed_a_${transposedASuffix}`,
  ]);
  for (const row of stats.byKernel) expectedKernels.delete(row.name);
  if (expectedKernels.size > 0) {
    throw new Error(`missing register-blocked dispatches for ${[...expectedKernels].join(", ")}`);
  }
}

console.log(JSON.stringify({
  schema: "helios-matmul-autotune-smoke-v1",
  variant,
  device: getDeviceInfo(),
  shape: { M, N, K },
  errors,
  decisions,
  stats,
}, null, 2));

destroyDevice();
