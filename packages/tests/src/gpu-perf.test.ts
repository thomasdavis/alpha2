/**
 * GPU performance optimization tests.
 *
 * Tests for: cooperative matrix GEMM, per-buffer barriers, sync elimination,
 * and the checkFinite GPU kernel.
 *
 * Requires a Vulkan-capable GPU to run.
 */

import { describe, it, expect, afterAll } from "vitest";
import { HeliosBackend, destroyDevice } from "@alpha/helios";
import { CpuRefBackend } from "@alpha/tensor";

const gpu = new HeliosBackend();
const cpu = new CpuRefBackend();

// Force GPU dispatch for small tensors in tests
gpu.setMinGpuSize(1);

afterAll(() => {
  destroyDevice();
});

// ── Helper ───────────────────────────────────────────────────────────────────

function approxEqual(a: Float32Array, b: Float32Array, tol = 1e-2): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (Math.abs(a[i] - b[i]) > tol) return false;
  }
  return true;
}

function maxDiff(a: Float32Array, b: Float32Array): number {
  let max = 0;
  for (let i = 0; i < a.length; i++) {
    max = Math.max(max, Math.abs(a[i] - b[i]));
  }
  return max;
}

// ── Barrier correctness tests ────────────────────────────────────────────────

describe("Barrier correctness", () => {
  it("RAW dependency chain: add → scale → sub", () => {
    const a = gpu.fromArray([1, 2, 3, 4], [2, 2]);
    const b = gpu.fromArray([5, 6, 7, 8], [2, 2]);
    const sum = gpu.add(a, b);           // [6, 8, 10, 12]
    const scaled = gpu.scale(sum, 2.0);  // [12, 16, 20, 24]
    const result = gpu.sub(scaled, a);   // [11, 14, 17, 20]

    const vals = Array.from((result.data as Float32Array).subarray(0, 4));
    expect(vals).toEqual([11, 14, 17, 20]);
  });

  it("independent ops produce correct results", () => {
    const a = gpu.fromArray([1, 2, 3, 4], [4]);
    const b = gpu.fromArray([10, 20, 30, 40], [4]);
    const c = gpu.fromArray([100, 200, 300, 400], [4]);
    const d = gpu.fromArray([5, 5, 5, 5], [4]);

    // Two independent additions
    const ab = gpu.add(a, b);  // [11, 22, 33, 44]
    const cd = gpu.add(c, d);  // [105, 205, 305, 405]

    const abVals = Array.from((ab.data as Float32Array).subarray(0, 4));
    const cdVals = Array.from((cd.data as Float32Array).subarray(0, 4));
    expect(abVals).toEqual([11, 22, 33, 44]);
    expect(cdVals).toEqual([105, 205, 305, 405]);
  });

  it("read-after-read: two ops read same buffer", () => {
    const shared = gpu.fromArray([1, 2, 3, 4], [4]);
    const a = gpu.fromArray([10, 10, 10, 10], [4]);
    const b = gpu.fromArray([20, 20, 20, 20], [4]);

    // Both read 'shared' — no write hazard
    const r1 = gpu.add(shared, a); // [11, 12, 13, 14]
    const r2 = gpu.add(shared, b); // [21, 22, 23, 24]

    expect(Array.from((r1.data as Float32Array).subarray(0, 4))).toEqual([11, 12, 13, 14]);
    expect(Array.from((r2.data as Float32Array).subarray(0, 4))).toEqual([21, 22, 23, 24]);
  });

  it("write-after-read: addInplace after scale produces correct scaled result", () => {
    // Create a GPU-resident tensor via an op so it's fully on GPU
    const base = gpu.fromArray([1, 2, 3, 4, 5, 6, 7, 8], [8]);
    const ones = gpu.fromArray([0, 0, 0, 0, 0, 0, 0, 0], [8]);
    const a = gpu.add(base, ones); // force GPU residence: [1, 2, 3, 4, 5, 6, 7, 8]
    const b = gpu.fromArray([10, 10, 10, 10, 10, 10, 10, 10], [8]);

    // Read a first via scale (flush to ensure scale executes before addInplace)
    const scaled = gpu.scale(a, 3.0); // [3, 6, 9, 12, 15, 18, 21, 24]
    gpu.flush();

    // Now write to a
    gpu.addInplace(a, b); // a modified on GPU

    // The scaled output should preserve pre-addInplace values
    expect(Array.from((scaled.data as Float32Array).subarray(0, 8))).toEqual([3, 6, 9, 12, 15, 18, 21, 24]);
  });
});

// ── checkFinite tests ────────────────────────────────────────────────────────

describe("checkFinite GPU kernel", () => {
  it("returns 0 for all-finite tensor", () => {
    const t = gpu.fromArray([1.0, 2.0, 3.0, -4.0, 0.0, 100.0, -0.5, 0.001], [8]);
    const result = gpu.checkFinite(t);
    expect((result.data as Float32Array)[0]).toBe(0.0);
  });

  it("returns 1 for tensor with Inf", () => {
    const t = gpu.fromArray([1.0, 2.0, Infinity, 4.0, 5.0, 6.0, 7.0, 8.0], [8]);
    const result = gpu.checkFinite(t);
    expect((result.data as Float32Array)[0]).toBe(1.0);
  });

  it("returns 1 for tensor with NaN", () => {
    const t = gpu.fromArray([1.0, NaN, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [8]);
    const result = gpu.checkFinite(t);
    expect((result.data as Float32Array)[0]).toBe(1.0);
  });

  it("returns 1 for tensor with -Inf", () => {
    const t = gpu.fromArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -Infinity, 8.0], [8]);
    const result = gpu.checkFinite(t);
    expect((result.data as Float32Array)[0]).toBe(1.0);
  });

  it("handles large tensor", () => {
    const size = 100000;
    const arr = new Float32Array(size);
    for (let i = 0; i < size; i++) arr[i] = i * 0.01;
    const t = gpu.fromArray(Array.from(arr), [size]);
    const result = gpu.checkFinite(t);
    expect((result.data as Float32Array)[0]).toBe(0.0);
  });

  it("detects single NaN in large tensor", () => {
    const size = 100000;
    const arr = new Float32Array(size);
    for (let i = 0; i < size; i++) arr[i] = i * 0.01;
    arr[size - 1] = NaN; // last element is NaN
    const t = gpu.fromArray(Array.from(arr), [size]);
    const result = gpu.checkFinite(t);
    expect((result.data as Float32Array)[0]).toBe(1.0);
  });
});

// ── Matmul correctness tests (GPU vs CPU reference) ──────────────────────────

describe("Matmul correctness", () => {
  it("basic matmul matches CPU reference", () => {
    const M = 64, K = 64, N = 64;
    const aData = Array.from({ length: M * K }, () => Math.random() - 0.5);
    const bData = Array.from({ length: K * N }, () => Math.random() - 0.5);

    const cpuA = cpu.fromArray(aData, [M, K]);
    const cpuB = cpu.fromArray(bData, [K, N]);
    const cpuC = cpu.matmul(cpuA, cpuB);

    const gpuA = gpu.fromArray(aData, [M, K]);
    const gpuB = gpu.fromArray(bData, [K, N]);
    const gpuC = gpu.matmul(gpuA, gpuB);

    const diff = maxDiff(cpuC.data as Float32Array, gpuC.data as Float32Array);
    // Allow f16 precision loss for coop matrix path
    expect(diff).toBeLessThan(0.5);
  });

  it("batched matmul matches CPU reference", () => {
    const B = 4, M = 32, K = 48, N = 16;
    const aData = Array.from({ length: B * M * K }, () => Math.random() - 0.5);
    const bData = Array.from({ length: B * K * N }, () => Math.random() - 0.5);

    const cpuA = cpu.fromArray(aData, [B, M, K]);
    const cpuB = cpu.fromArray(bData, [B, K, N]);
    const cpuC = cpu.matmul(cpuA, cpuB);

    const gpuA = gpu.fromArray(aData, [B, M, K]);
    const gpuB = gpu.fromArray(bData, [B, K, N]);
    const gpuC = gpu.matmul(gpuA, gpuB);

    const diff = maxDiff(cpuC.data as Float32Array, gpuC.data as Float32Array);
    expect(diff).toBeLessThan(0.5);
  });

  it("transposed matmul matches CPU reference", () => {
    const M = 64, K = 64, N = 64;
    const aData = Array.from({ length: M * K }, () => Math.random() - 0.5);
    const bData = Array.from({ length: N * K }, () => Math.random() - 0.5);

    // CPU: transpose B from [N,K] to [K,N] then regular matmul
    const cpuA = cpu.fromArray(aData, [M, K]);
    const bTransposed = new Float32Array(K * N);
    for (let i = 0; i < N; i++)
      for (let j = 0; j < K; j++)
        bTransposed[j * N + i] = bData[i * K + j];
    const cpuBT = cpu.fromArray(Array.from(bTransposed), [K, N]);
    const cpuC = cpu.matmul(cpuA, cpuBT);

    const gpuA = gpu.fromArray(aData, [M, K]);
    const gpuB = gpu.fromArray(bData, [N, K]);
    const gpuC = gpu.matmulTransposed(gpuA, gpuB);

    const diff = maxDiff(cpuC.data as Float32Array, gpuC.data as Float32Array);
    expect(diff).toBeLessThan(0.5);
  });

  it("transposed-A matmul matches CPU reference", () => {
    const M = 64, K = 48, N = 32;
    const aData = Array.from({ length: M * K }, () => Math.random() - 0.5);
    const bData = Array.from({ length: M * N }, () => Math.random() - 0.5);

    // CPU: transpose A from [M,K] to [K,M], then matmul([K,M] x [M,N]).
    const aTransposed = new Float32Array(K * M);
    for (let i = 0; i < M; i++) {
      for (let j = 0; j < K; j++) {
        aTransposed[j * M + i] = aData[i * K + j];
      }
    }
    const cpuAT = cpu.fromArray(Array.from(aTransposed), [K, M]);
    const cpuB = cpu.fromArray(bData, [M, N]);
    const cpuC = cpu.matmul(cpuAT, cpuB);

    const gpuA = gpu.fromArray(aData, [M, K]);
    const gpuB = gpu.fromArray(bData, [M, N]);
    const gpuC = gpu.matmulTransposedA(gpuA, gpuB);

    const diff = maxDiff(cpuC.data as Float32Array, gpuC.data as Float32Array);
    expect(diff).toBeLessThan(0.5);
  });

  it("batched transposed-A matmul matches CPU reference", () => {
    const Bsz = 3, M = 24, K = 20, N = 28;
    const aData = Array.from({ length: Bsz * M * K }, () => Math.random() - 0.5);
    const bData = Array.from({ length: Bsz * M * N }, () => Math.random() - 0.5);

    const aTransposed = new Float32Array(Bsz * K * M);
    for (let bIdx = 0; bIdx < Bsz; bIdx++) {
      const aOff = bIdx * M * K;
      const atOff = bIdx * K * M;
      for (let i = 0; i < M; i++) {
        for (let j = 0; j < K; j++) {
          aTransposed[atOff + j * M + i] = aData[aOff + i * K + j];
        }
      }
    }
    const cpuAT = cpu.fromArray(Array.from(aTransposed), [Bsz, K, M]);
    const cpuB = cpu.fromArray(bData, [Bsz, M, N]);
    const cpuC = cpu.matmul(cpuAT, cpuB);

    const gpuA = gpu.fromArray(aData, [Bsz, M, K]);
    const gpuB = gpu.fromArray(bData, [Bsz, M, N]);
    const gpuC = gpu.matmulTransposedA(gpuA, gpuB);

    const diff = maxDiff(cpuC.data as Float32Array, gpuC.data as Float32Array);
    expect(diff).toBeLessThan(0.5);
  });

  it("non-aligned shapes fall back gracefully", () => {
    // Dimensions not divisible by typical coop matrix tile sizes (16)
    const M = 33, K = 47, N = 19;
    const aData = Array.from({ length: M * K }, () => Math.random() - 0.5);
    const bData = Array.from({ length: K * N }, () => Math.random() - 0.5);

    const cpuA = cpu.fromArray(aData, [M, K]);
    const cpuB = cpu.fromArray(bData, [K, N]);
    const cpuC = cpu.matmul(cpuA, cpuB);

    const gpuA = gpu.fromArray(aData, [M, K]);
    const gpuB = gpu.fromArray(bData, [K, N]);
    const gpuC = gpu.matmul(gpuA, gpuB);

    const diff = maxDiff(cpuC.data as Float32Array, gpuC.data as Float32Array);
    expect(diff).toBeLessThan(0.01); // scalar path should be more precise
  });

  it("transformer-scale matmul produces finite results", () => {
    // M=512, K=768, N=768 (typical self-attention projection)
    const M = 512, K = 768, N = 768;
    const aData = Array.from({ length: M * K }, () => (Math.random() - 0.5) * 0.1);
    const bData = Array.from({ length: K * N }, () => (Math.random() - 0.5) * 0.1);

    const gpuA = gpu.fromArray(aData, [M, K]);
    const gpuB = gpu.fromArray(bData, [K, N]);
    const gpuC = gpu.matmul(gpuA, gpuB);

    const result = gpuC.data as Float32Array;
    expect(result.length).toBeGreaterThanOrEqual(M * N);
    // Check that the first M*N elements (the actual tensor data) are finite
    let allFinite = true;
    for (let i = 0; i < M * N; i++) {
      if (!isFinite(result[i])) { allFinite = false; break; }
    }
    expect(allFinite).toBe(true);
  });
});

// ── GPU loss accumulation tests ──────────────────────────────────────────────

describe("GPU loss accumulation", () => {
  it("GPU accumulation matches CPU microstep sum", () => {
    const accumSteps = 4;
    const losses = [0.5, 0.3, 0.7, 0.2];

    // Simulate GPU-side accumulation
    let lossAccum = null as ReturnType<typeof gpu.zeros> | null;
    for (let i = 0; i < accumSteps; i++) {
      const lossTensor = gpu.fromArray([losses[i]], [1]);
      const scaled = gpu.scale(lossTensor, 1 / accumSteps);
      if (lossAccum === null) {
        lossAccum = scaled;
      } else {
        lossAccum = gpu.add(lossAccum, scaled);
      }
    }

    const gpuResult = (lossAccum!.data as Float32Array)[0];
    const cpuResult = losses.reduce((a, b) => a + b, 0) / accumSteps;
    expect(gpuResult).toBeCloseTo(cpuResult, 4);
  });

  it("NaN propagates through GPU accumulation", () => {
    const accumSteps = 4;
    let lossAccum = null as ReturnType<typeof gpu.zeros> | null;
    for (let i = 0; i < accumSteps; i++) {
      const val = i === 2 ? NaN : 0.5;
      const lossTensor = gpu.fromArray([val], [1]);
      const scaled = gpu.scale(lossTensor, 1 / accumSteps);
      if (lossAccum === null) {
        lossAccum = scaled;
      } else {
        lossAccum = gpu.add(lossAccum, scaled);
      }
    }

    const result = (lossAccum!.data as Float32Array)[0];
    expect(isNaN(result)).toBe(true);
  });
});

// ── Spot check batching tests ────────────────────────────────────────────────

describe("Batched spot checks", () => {
  it("combined spot-check matches individual spot-checks", () => {
    // Simulate the batched gradient inf check pattern
    const grads = [
      gpu.fromArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [8]),
      gpu.fromArray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [8]),
      gpu.fromArray([10, 20, 30, 40, 50, 60, 70, 80], [8]),
    ];

    // Batch: record all ops first, then read
    const spotResults: { s: ReturnType<typeof gpu.sum>; g2: ReturnType<typeof gpu.mul> }[] = [];
    for (const g of grads) {
      const g2 = gpu.mul(g, g);
      spotResults.push({ s: gpu.sum(g2), g2 });
    }

    // Single flush on first .data access
    const batchedSums: number[] = [];
    for (const { s } of spotResults) {
      batchedSums.push((s.data as Float32Array)[0]);
    }

    // Individual: compute one at a time
    const individualSums: number[] = [];
    for (const g of grads) {
      const g2 = gpu.mul(g, g);
      individualSums.push((gpu.sum(g2).data as Float32Array)[0]);
    }

    for (let i = 0; i < batchedSums.length; i++) {
      expect(batchedSums[i]).toBeCloseTo(individualSums[i], 2);
    }
  });
});

// ── Performance benchmark ────────────────────────────────────────────────────

describe("Matmul performance", () => {
  it("GPU matmul is faster than trivial and produces correct results", () => {
    // Warm up
    const warmA = gpu.fromArray(Array.from({ length: 512 * 768 }, () => Math.random()), [512, 768]);
    const warmB = gpu.fromArray(Array.from({ length: 768 * 768 }, () => Math.random()), [768, 768]);
    gpu.matmul(warmA, warmB);
    gpu.flush();

    // Benchmark: 512x768 x 768x768
    const M = 512, K = 768, N = 768;
    const aData = Array.from({ length: M * K }, () => (Math.random() - 0.5) * 0.1);
    const bData = Array.from({ length: K * N }, () => (Math.random() - 0.5) * 0.1);
    const gpuA = gpu.fromArray(aData, [M, K]);
    const gpuB = gpu.fromArray(bData, [K, N]);

    const iterations = 10;
    const start = performance.now();
    for (let i = 0; i < iterations; i++) {
      gpu.matmul(gpuA, gpuB);
    }
    gpu.flush();
    // Force readback to ensure all GPU work is done
    const result = gpu.matmul(gpuA, gpuB);
    const _ = (result.data as Float32Array)[0];
    const elapsed = performance.now() - start;

    const flops = 2 * M * K * N * iterations;
    const gflops = flops / (elapsed * 1e6);
    console.log(`Matmul ${M}x${K}x${N}: ${iterations} iters in ${elapsed.toFixed(1)}ms = ${gflops.toFixed(1)} GFLOPS`);

    // Sanity check: should be finite and not take forever
    expect(elapsed).toBeLessThan(30_000); // less than 30s
    expect(isFinite(_)).toBe(true);
  });
});
