/**
 * GPU stress test — exercises helios Vulkan backend directly on H100.
 * Tests: large matmuls, large element-wise ops, sustained GPU utilization.
 *
 * Run: VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json node --loader ts-node/esm scripts/gpu-stress-test.ts
 */

import { shapeSize } from "@alpha/core";
import type { Backend, TensorData } from "@alpha/core";

async function main() {
  // Dynamically import helios to get HeliosBackend
  const { HeliosBackend } = await import("@alpha/helios");

  const backend = new HeliosBackend() as Backend;
  console.log(`Backend: ${backend.name}`);
  console.log();

  // ── Test 1: Large element-wise ops ────────────────────────────────
  console.log("=== Test 1: Large element-wise ops ===");
  const sizes = [100_000, 1_000_000, 10_000_000, 50_000_000];

  for (const size of sizes) {
    const a = backend.randn([size]);
    const b = backend.randn([size]);

    // Warmup
    const _ = backend.add(a, b);

    const iters = 20;
    const t0 = performance.now();
    let result: TensorData = a;
    for (let i = 0; i < iters; i++) {
      result = backend.add(a, b);
    }
    // Force readback to ensure GPU is done
    const dummy = result.data[0];
    const elapsed = performance.now() - t0;

    const bytesPerOp = size * 4 * 3; // read a, read b, write c
    const throughput = (bytesPerOp * iters) / (elapsed / 1000) / 1e9;
    console.log(`  size=${(size/1e6).toFixed(1)}M | ${(elapsed/iters).toFixed(2)}ms/op | ${throughput.toFixed(1)} GB/s`);
  }

  // ── Test 2: Large matmul ──────────────────────────────────────────
  console.log();
  console.log("=== Test 2: Matmul ===");
  const matSizes: [number,number,number][] = [
    [128, 512, 512],
    [512, 512, 512],
    [1024, 1024, 1024],
    [2048, 2048, 2048],
  ];

  for (const [M, K, N] of matSizes) {
    const a = backend.randn([M, K]);
    const b = backend.randn([K, N]);

    // Warmup
    backend.matmul(a, b);

    const iters = 5;
    const t0 = performance.now();
    let result: TensorData = a;
    for (let i = 0; i < iters; i++) {
      result = backend.matmul(a, b);
    }
    const dummy = result.data[0];
    const elapsed = performance.now() - t0;

    const flops = 2 * M * K * N;
    const gflops = (flops * iters) / (elapsed / 1000) / 1e9;
    console.log(`  ${M}x${K} @ ${K}x${N} | ${(elapsed/iters).toFixed(1)}ms/op | ${gflops.toFixed(1)} GFLOPS`);
  }

  // ── Test 3: Sustained GPU work (simulate training loop) ──────────
  console.log();
  console.log("=== Test 3: Sustained ops (simulated training step) ===");
  const dim = 512;
  const seqLen = 128;
  const batchSz = 4;
  const totalTokens = batchSz * seqLen;  // 512

  // Create tensors that mimic a transformer layer
  const x = backend.randn([totalTokens, dim]);
  const wQ = backend.randn([dim, dim]);
  const wK = backend.randn([dim, dim]);
  const wV = backend.randn([dim, dim]);
  const wO = backend.randn([dim, dim]);
  const wFF1 = backend.randn([dim, dim * 4]);
  const wFF2 = backend.randn([dim * 4, dim]);
  const lnW = backend.ones([dim]);
  const lnB = backend.zeros([dim]);

  console.log(`  Tensors allocated, running transformer-like ops...`);

  const warmupSteps = 2;
  const testSteps = 10;

  for (let step = 0; step < warmupSteps + testSteps; step++) {
    const t0 = performance.now();

    // LayerNorm
    let h = backend.layerNorm(x, lnW, lnB, 1e-5);

    // QKV projections
    const Q = backend.matmul(h, wQ);
    const K = backend.matmul(h, wK);
    const V = backend.matmul(h, wV);

    // Attention: Q @ K^T
    const Kt = backend.transpose(K, 0, 1);
    let attn = backend.matmul(Q, Kt);
    attn = backend.scale(attn, 1 / Math.sqrt(dim));
    attn = backend.softmax(attn, -1);

    // Attention output
    let out = backend.matmul(attn, V);
    out = backend.matmul(out, wO);

    // Residual
    h = backend.add(x, out);

    // FFN
    let ff = backend.layerNorm(h, lnW, lnB, 1e-5);
    ff = backend.matmul(ff, wFF1);
    ff = backend.gelu(ff);
    ff = backend.matmul(ff, wFF2);

    // Residual
    h = backend.add(h, ff);

    // Force GPU sync
    const val = h.data[0];
    const elapsed = performance.now() - t0;

    if (step >= warmupSteps) {
      console.log(`  step ${step - warmupSteps + 1}/${testSteps} | ${elapsed.toFixed(1)}ms | val=${val.toFixed(4)}`);
    }
  }

  // ── Test 4: Memory pressure (allocate lots of VRAM) ───────────────
  console.log();
  console.log("=== Test 4: VRAM pressure ===");
  const bigTensors: TensorData[] = [];
  const targetGB = 4;
  const chunkSize = 50_000_000; // 50M floats = 200MB
  const numChunks = Math.ceil((targetGB * 1e9) / (chunkSize * 4));

  console.log(`  Allocating ${targetGB}GB of GPU memory in ${numChunks} chunks...`);
  for (let i = 0; i < numChunks; i++) {
    bigTensors.push(backend.randn([chunkSize]));
  }

  // Do ops on the big tensors
  const t0 = performance.now();
  for (let i = 0; i < numChunks - 1; i++) {
    const result = backend.add(bigTensors[i], bigTensors[i + 1]);
    bigTensors[i] = result;
  }
  // Force readback
  const val = bigTensors[0].data[0];
  const elapsed = performance.now() - t0;
  console.log(`  ${numChunks - 1} adds on 200MB tensors: ${elapsed.toFixed(0)}ms total`);

  console.log();
  console.log("Done! Check nvidia-smi for GPU memory/utilization.");
}

main().catch(console.error);
