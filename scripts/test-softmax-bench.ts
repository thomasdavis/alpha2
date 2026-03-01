#!/usr/bin/env -S npx tsx
/**
 * Benchmark softmax across training-relevant shapes.
 * Tests both small (L1-cached) and large (DRAM-bound) dimensions.
 */
import { HeliosBackend } from "@alpha/helios";

const b = new HeliosBackend();
const ITERS = 20;
const WARMUP = 5;

async function benchSoftmax(M: number, N: number, label: string) {
  const a = b.randn([M, N]);
  for (let i = 0; i < WARMUP; i++) {
    const r = (b as any).softmax(a, -1);
    b.syncGpu();
    b.releaseGpuTensor(r);
  }
  const times: number[] = [];
  for (let i = 0; i < ITERS; i++) {
    const t0 = performance.now();
    const r = (b as any).softmax(a, -1);
    b.syncGpu();
    const t1 = performance.now();
    b.releaseGpuTensor(r);
    times.push(t1 - t0);
  }
  b.releaseGpuTensor(a);
  times.sort((a, b) => a - b);
  const med = times[Math.floor(times.length / 2)];
  // Softmax reads input + writes output = 2 × M × N × 4 bytes
  const bytes = 2 * M * N * 4;
  const bw = bytes / (med / 1000) / 1e9;
  console.log(`${label.padEnd(35)} [${M}×${N}]  med=${med.toFixed(3).padStart(8)}ms  bw=${bw.toFixed(1).padStart(6)} GB/s`);
}

async function benchAdd(M: number, N: number, label: string) {
  const n = M * N;
  const a = b.randn([n]);
  const c = b.randn([n]);
  for (let i = 0; i < WARMUP; i++) {
    const r = b.add(a, c); b.syncGpu(); b.releaseGpuTensor(r);
  }
  const times: number[] = [];
  for (let i = 0; i < ITERS; i++) {
    const t0 = performance.now();
    const r = b.add(a, c); b.syncGpu();
    const t1 = performance.now();
    b.releaseGpuTensor(r);
    times.push(t1 - t0);
  }
  b.releaseGpuTensor(a); b.releaseGpuTensor(c);
  times.sort((a, b) => a - b);
  const med = times[Math.floor(times.length / 2)];
  const bytes = 3 * n * 4;
  const bw = bytes / (med / 1000) / 1e9;
  console.log(`${label.padEnd(35)} [${n}]  med=${med.toFixed(3).padStart(8)}ms  bw=${bw.toFixed(1).padStart(6)} GB/s`);
}

async function main() {
  console.log("=== Softmax + Bandwidth Benchmarks ===\n");

  // Training-relevant shapes
  console.log("--- Softmax ---");
  await benchSoftmax(1, 1024, "warmup tiny");
  await benchSoftmax(512, 1024, "attn [512,1024]");
  await benchSoftmax(512, 4096, "attn [512,4096]");
  await benchSoftmax(512, 32768, "attn [512,32768]");
  await benchSoftmax(512, 64000, "lm_head [512,64000]");
  await benchSoftmax(256, 64000, "lm_head [256,64000]");
  await benchSoftmax(128, 64000, "lm_head [128,64000]");
  await benchSoftmax(1024, 1024, "large attn [1024,1024]");

  // Bandwidth test: add kernel
  console.log("\n--- Add (bandwidth reference) ---");
  await benchAdd(512, 64000, "add [512×64000]");
  await benchAdd(512, 1024, "add [512×1024]");
  await benchAdd(1024, 1024, "add [1024×1024]");

  console.log("\nDone.");
  process.exit(0);
}

main().catch(e => { console.error(e); process.exit(1); });
