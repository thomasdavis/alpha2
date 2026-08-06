#!/usr/bin/env node

/**
 * Physical-GPU discriminator for Helios cooperative-matrix accumulation.
 *
 * The driver starts a fresh Node process for each mode because Helios selects
 * cooperative-matrix shader types at module-import time.  Every worker first
 * runs an exact production-pattern oracle and then times the same aligned GEMM
 * shapes with in-batch Vulkan timestamps.  The primary hardware discriminator
 * is resident-F16 FP32-accumulate throughput divided by resident-F16
 * F16-accumulate throughput; the cast-inclusive and tiled-FP32 rows answer the
 * separate question of whether the current implementation is useful in an
 * actual training step.
 */

import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const MODES = [
  "coop_f16acc",
  "coop_f32acc",
  "control_fp32",
  "coop_f32acc_cast",
];

function parsePositiveInt(name, fallback) {
  const value = Number.parseInt(process.env[name] ?? String(fallback), 10);
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new Error(`${name} must be a positive integer`);
  }
  return value;
}

function median(values) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

function percentile(values, fraction) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.round((sorted.length - 1) * fraction)));
  return sorted[index];
}

function summarizeSamples(samples) {
  const gpuUs = samples.map((sample) => sample.dispatchGpuUs);
  const tflops = samples.map((sample) => sample.tflops);
  const wallMs = samples.map((sample) => sample.wallMs);
  return {
    samples: samples.length,
    medianDispatchGpuUs: median(gpuUs),
    minDispatchGpuUs: Math.min(...gpuUs),
    p25DispatchGpuUs: percentile(gpuUs, 0.25),
    p75DispatchGpuUs: percentile(gpuUs, 0.75),
    medianTflops: median(tflops),
    maxTflops: Math.max(...tflops),
    medianWallMs: median(wallMs),
    kernels: [...new Set(samples.flatMap((sample) => sample.kernels))],
  };
}

function configureMode(mode) {
  process.env.HELIOS_PROFILE_GPU_OPS = "1";
  process.env.HELIOS_PROFILE_GPU_TIMESTAMPS = "1";
  process.env.HELIOS_NO_FALLBACK = "1";
  process.env.HELIOS_COOP_SPLIT_K = "0";
  process.env.HELIOS_COOP_DOUBLE_BUF = "0";
  process.env.HELIOS_COOP_KMULTI_ADAPT_MIN_WGS = "0";

  if (mode === "control_fp32") {
    process.env.HELIOS_DISABLE_COOP_MAT = "1";
    process.env.HELIOS_MATMUL_REG4X2 = "1";
    delete process.env.HELIOS_COOP_F16_ACCUM;
    return;
  }
  delete process.env.HELIOS_DISABLE_COOP_MAT;
  delete process.env.HELIOS_MATMUL_REG4X2;
  if (mode === "coop_f16acc") {
    process.env.HELIOS_COOP_F16_ACCUM = "1";
  } else {
    delete process.env.HELIOS_COOP_F16_ACCUM;
  }
}

function isCoopMode(mode) {
  return mode.startsWith("coop_");
}

function usesResidentF16(mode) {
  return mode === "coop_f16acc" || mode === "coop_f32acc";
}

async function runWorker() {
  const mode = process.env.HELIOS_COOP_BENCH_MODE;
  if (!MODES.includes(mode)) throw new Error(`unsupported HELIOS_COOP_BENCH_MODE=${mode}`);
  configureMode(mode);

  const { HeliosBackend, destroyDevice, getDeviceInfo } = await import("@alpha/helios");
  const gpu = new HeliosBackend();
  gpu.setMinGpuSize(1);

  const quick = process.env.HELIOS_COOP_BENCH_QUICK === "1";
  const repetitions = parsePositiveInt("HELIOS_COOP_BENCH_REPETITIONS", quick ? 2 : 9);
  const warmups = parsePositiveInt("HELIOS_COOP_BENCH_WARMUPS", quick ? 1 : 3);
  const device = gpu.getDeviceInfo();
  if (isCoopMode(mode) && !device.coopMatSupported) {
    throw new Error(`mode ${mode} requires cooperative matrices; device=${device.deviceName}`);
  }

  function makeInputs(M, N, K) {
    const a32 = gpu.full([M, K], 1 / 128, "f32");
    const b32 = gpu.full([K, N], -1 / 256, "f32");
    if (!usesResidentF16(mode)) return { a: a32, b: b32, roots: [a32, b32] };
    const a16 = gpu.castDtype(a32, "f16");
    const b16 = gpu.castDtype(b32, "f16");
    gpu.syncGpu();
    return { a: a16, b: b16, roots: [a32, b32, a16, b16] };
  }

  // The values and dot product are binary-exact.  A correct FP32 accumulator
  // must therefore reproduce the closed form without a summation-order caveat.
  const oracleShape = { M: 1024, N: 1408, K: 512 };
  const oracleInputs = makeInputs(oracleShape.M, oracleShape.N, oracleShape.K);
  gpu.resetStepOps();
  const oracleOut = gpu.matmul(oracleInputs.a, oracleInputs.b);
  const oracleValues = oracleOut.data;
  const expected = oracleShape.K * (1 / 128) * (-1 / 256);
  let oracleMaxAbsError = 0;
  let oracleWorstIndex = -1;
  for (let index = 0; index < oracleValues.length; index++) {
    const error = Math.abs(oracleValues[index] - expected);
    if (error > oracleMaxAbsError) {
      oracleMaxAbsError = error;
      oracleWorstIndex = index;
    }
  }
  const oracleStats = gpu.getGpuStepStats();
  const oracleCoop = gpu.getMatmulCoopStats();
  if (oracleMaxAbsError > 1e-6) {
    throw new Error(
      `oracle failed for ${mode}: max_abs_error=${oracleMaxAbsError} index=${oracleWorstIndex}`,
    );
  }
  if (isCoopMode(mode) && oracleCoop.coopDirectDispatches < 1) {
    throw new Error(`oracle did not execute a direct cooperative dispatch: ${JSON.stringify(oracleCoop)}`);
  }
  if (!isCoopMode(mode) && oracleCoop.coopDispatches !== 0) {
    throw new Error(`control unexpectedly executed a cooperative dispatch: ${JSON.stringify(oracleCoop)}`);
  }
  gpu.releaseGpuTensor(oracleOut);
  for (const tensor of oracleInputs.roots) gpu.releaseGpuTensor(tensor);
  gpu.purgeBufferPools();

  const cases = quick
    ? [{ name: "quick", M: 256, N: 256, K: 256, repetitions, warmups }]
    : [
        // Exact foundation-token count and hidden/FFN widths.
        { name: "foundation_ffn_up", M: 24_576, N: 1_728, K: 640, repetitions, warmups },
        // High-arithmetic-intensity reference isolates the die accumulation rate.
        { name: "square_4096", M: 4_096, N: 4_096, K: 4_096, repetitions, warmups },
        // Exact foundation-token count, hidden width, and vocabulary width.
        { name: "foundation_lm_head", M: 24_576, N: 12_288, K: 640, repetitions: Math.max(5, Math.ceil(repetitions / 2)), warmups: Math.max(2, Math.ceil(warmups / 2)) },
      ];

  const results = [];
  for (const bench of cases) {
    const inputs = makeInputs(bench.M, bench.N, bench.K);

    for (let iteration = 0; iteration < bench.warmups; iteration++) {
      gpu.resetStepOps();
      const output = gpu.matmul(inputs.a, inputs.b);
      gpu.syncGpu();
      gpu.releaseGpuTensor(output);
    }

    const samples = [];
    for (let iteration = 0; iteration < bench.repetitions; iteration++) {
      gpu.resetStepOps();
      const started = performance.now();
      const output = gpu.matmul(inputs.a, inputs.b);
      gpu.syncGpu();
      const wallMs = performance.now() - started;
      const stats = gpu.getGpuStepStats();
      const coop = gpu.getMatmulCoopStats();
      if (!stats.timingEnabled || stats.timestampedFlushes < 1 || !(stats.dispatchGpuTimeUs > 0)) {
        throw new Error(`missing GPU timestamps for ${mode}/${bench.name}: ${JSON.stringify(stats)}`);
      }
      const timedKernels = stats.byKernel.map((row) => row.name);
      const timedCoop = timedKernels.some((name) => name.startsWith("matmul_coop_"));
      if (isCoopMode(mode) && (!timedCoop || coop.lastCoopShape === null)) {
        throw new Error(
          `missing cooperative dispatch evidence for ${mode}/${bench.name}: ${JSON.stringify(timedKernels)}`,
        );
      }
      if (!isCoopMode(mode) && timedCoop) {
        throw new Error(`control timed a cooperative kernel for ${bench.name}`);
      }
      const flops = 2 * bench.M * bench.N * bench.K;
      samples.push({
        iteration,
        wallMs,
        dispatchGpuUs: stats.dispatchGpuTimeUs,
        batchGpuUs: stats.batchGpuTimeUs,
        gpuBlockingTimeMs: stats.gpuBlockingTimeMs,
        tflops: flops / (stats.dispatchGpuTimeUs * 1e6),
        kernels: timedKernels,
        coopKernel: coop.lastCoopKernel,
        coopShape: coop.lastCoopShape,
      });
      gpu.releaseGpuTensor(output);
    }

    results.push({
      ...bench,
      flops: 2 * bench.M * bench.N * bench.K,
      summary: summarizeSamples(samples),
      samples,
    });

    for (const tensor of inputs.roots) gpu.releaseGpuTensor(tensor);
    gpu.purgeBufferPools();
  }

  const payload = {
    schema: "alpha-helios-coop-accum-worker-v1",
    mode,
    inputDtype: usesResidentF16(mode) ? "f16_resident" : "f32",
    accumulatorDtype: mode === "coop_f16acc" ? "f16" : "f32",
    includesInputCast: mode === "coop_f32acc_cast",
    quick,
    device,
    oracle: {
      shape: oracleShape,
      expected,
      maxAbsError: oracleMaxAbsError,
      worstIndex: oracleWorstIndex,
      kernels: oracleStats.byKernel.map((row) => row.name),
      coop: oracleCoop,
    },
    cases: results,
  };
  console.log(`ALPHA_COOP_BENCH_JSON=${JSON.stringify(payload)}`);
  destroyDevice();
}

function aggregateWorkerRuns(runs) {
  const grouped = new Map();
  for (const run of runs) {
    if (!grouped.has(run.mode)) grouped.set(run.mode, []);
    grouped.get(run.mode).push(run);
  }
  const modes = {};
  for (const [mode, modeRuns] of grouped) {
    const caseNames = modeRuns[0].cases.map((entry) => entry.name);
    modes[mode] = {
      runs: modeRuns.length,
      oracleMaxAbsError: Math.max(...modeRuns.map((run) => run.oracle.maxAbsError)),
      cases: {},
    };
    for (const caseName of caseNames) {
      const samples = modeRuns.flatMap((run) => run.cases.find((entry) => entry.name === caseName).samples);
      modes[mode].cases[caseName] = summarizeSamples(samples);
    }
  }
  return modes;
}

function deriveDecision(modes) {
  const f32acc = modes.coop_f32acc;
  const f16acc = modes.coop_f16acc;
  const control = modes.control_fp32;
  if (!f32acc || !f16acc || !control) return { status: "incomplete", cases: {} };
  const decisions = {};
  for (const caseName of Object.keys(f32acc.cases)) {
    const f32 = f32acc.cases[caseName].medianTflops;
    const f16 = f16acc.cases[caseName].medianTflops;
    const scalar = control.cases[caseName].medianTflops;
    const ratio = f32 / f16;
    decisions[caseName] = {
      coopF32AccumTflops: f32,
      coopF16AccumTflops: f16,
      selectedFp32Tflops: scalar,
      f32ToF16AccumRateRatio: ratio,
      f32AccumVersusSelectedFp32: f32 / scalar,
      accumulationRateClass: ratio >= 0.8 ? "full_or_near_full" : ratio <= 0.65 ? "half_or_near_half" : "intermediate",
    };
  }
  const ratios = Object.values(decisions).map((entry) => entry.f32ToF16AccumRateRatio);
  return {
    status: "measured",
    medianF32ToF16AccumRateRatio: median(ratios),
    accumulationRateClass: median(ratios) >= 0.8
      ? "full_or_near_full"
      : median(ratios) <= 0.65
        ? "half_or_near_half"
        : "intermediate",
    cases: decisions,
  };
}

function runDriver() {
  const passes = parsePositiveInt("HELIOS_COOP_BENCH_PASSES", 2);
  const filename = fileURLToPath(import.meta.url);
  const forward = MODES;
  const reverse = [...MODES].reverse();
  const runs = [];
  for (let pass = 0; pass < passes; pass++) {
    const order = pass % 2 === 0 ? forward : reverse;
    for (const mode of order) {
      console.error(`[coop-accum] pass=${pass + 1}/${passes} mode=${mode}`);
      const child = spawnSync(process.execPath, [filename, "--worker"], {
        env: { ...process.env, HELIOS_COOP_BENCH_MODE: mode },
        encoding: "utf8",
        maxBuffer: 64 * 1024 * 1024,
      });
      if (child.stderr) process.stderr.write(child.stderr);
      if (child.status !== 0) {
        if (child.stdout) process.stderr.write(child.stdout);
        throw new Error(`worker ${mode} failed with exit status ${child.status}`);
      }
      const marker = child.stdout.split(/\r?\n/).find((line) => line.startsWith("ALPHA_COOP_BENCH_JSON="));
      if (!marker) throw new Error(`worker ${mode} emitted no result marker`);
      runs.push(JSON.parse(marker.slice("ALPHA_COOP_BENCH_JSON=".length)));
    }
  }
  const modes = aggregateWorkerRuns(runs);
  console.log(JSON.stringify({
    schema: "alpha-helios-coop-accum-sweep-v1",
    createdAt: new Date().toISOString(),
    passes,
    quick: process.env.HELIOS_COOP_BENCH_QUICK === "1",
    device: runs[0]?.device ?? null,
    modes,
    decision: deriveDecision(modes),
    rawRuns: runs,
  }, null, 2));
}

if (process.argv.includes("--worker")) {
  await runWorker();
} else {
  runDriver();
}
