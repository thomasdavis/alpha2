/**
 * Raw GPU test — uses native addon directly to avoid TS overhead.
 * Pre-allocates buffers, does sustained compute, checks utilization.
 *
 * Run: VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json node scripts/gpu-raw-test.mjs
 */

import { getNative, initDevice, type NativeAddon } from "@alpha/helios";
import { getKernelSpirv } from "@alpha/helios";

function main() {
  // Initialize device
  const info = initDevice();
  console.log(`Device: ${info.deviceName}`);
  console.log(`F16: ${info.f16Supported}, AsyncTransfer: ${info.hasAsyncTransfer}`);
  console.log();

  const vk = getNative();
  const WG = 256;

  // Get pipeline for add_vec4 kernel
  const spirv = getKernelSpirv("add_vec4", WG);
  const pipeline = vk.createPipeline(spirv, 3); // 3 bindings: A, B, C

  // ── Test 1: Pre-allocated sustained add loop ─────────────────────
  console.log("=== Test 1: Sustained add (pre-allocated, no readback) ===");

  const testSizes = [1_000_000, 10_000_000, 50_000_000, 100_000_000];

  for (const size of testSizes) {
    const byteSize = size * 4;
    const vec4Size = size >> 2;
    const groups = Math.ceil(vec4Size / WG);

    // Pre-allocate buffers
    const bufA = vk.createBuffer(byteSize, 0);
    const bufB = vk.createBuffer(byteSize, 0);
    const bufC = vk.createBuffer(byteSize, 0);

    // Upload random data
    const dataA = new Float32Array(size);
    const dataB = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      dataA[i] = Math.random();
      dataB[i] = Math.random();
    }
    vk.uploadBuffer(bufA, dataA);
    vk.uploadBuffer(bufB, dataB);

    // Warmup
    const push = new Float32Array([vec4Size, 0]);
    for (let i = 0; i < 3; i++) {
      vk.dispatch(pipeline, [bufA, bufB, bufC], groups, 1, 1, push);
    }

    // Timed loop — synchronous dispatches (each waits for GPU)
    const iters = 50;
    const t0 = performance.now();
    for (let i = 0; i < iters; i++) {
      vk.dispatch(pipeline, [bufA, bufB, bufC], groups, 1, 1, push);
    }
    const elapsed = performance.now() - t0;

    const bytesPerOp = size * 4 * 3; // read a, read b, write c
    const throughput = (bytesPerOp * iters) / (elapsed / 1000) / 1e9;
    console.log(`  size=${(size/1e6).toFixed(0)}M | ${(elapsed/iters).toFixed(2)}ms/op | ${throughput.toFixed(1)} GB/s | total=${elapsed.toFixed(0)}ms`);

    vk.destroyBuffer(bufA);
    vk.destroyBuffer(bufB);
    vk.destroyBuffer(bufC);
  }

  // ── Test 2: Batched dispatches (compute graph style) ─────────────
  console.log();
  console.log("=== Test 2: Batched dispatches (batchBegin/batchDispatch/batchSubmit) ===");

  for (const size of testSizes) {
    const byteSize = size * 4;
    const vec4Size = size >> 2;
    const groups = Math.ceil(vec4Size / WG);

    const bufA = vk.createBuffer(byteSize, 0);
    const bufB = vk.createBuffer(byteSize, 0);
    // Create multiple output buffers for batch
    const numOps = 20;
    const outBufs: number[] = [];
    for (let i = 0; i < numOps; i++) {
      outBufs.push(vk.createBuffer(byteSize, 0));
    }

    const dataA = new Float32Array(size);
    const dataB = new Float32Array(size);
    for (let i = 0; i < size; i++) { dataA[i] = Math.random(); dataB[i] = Math.random(); }
    vk.uploadBuffer(bufA, dataA);
    vk.uploadBuffer(bufB, dataB);

    // Warmup
    const push = new Float32Array([vec4Size, 0]);
    vk.dispatch(pipeline, [bufA, bufB, outBufs[0]], groups, 1, 1, push);

    // Batched: record N ops, submit once
    const batches = 5;
    const t0 = performance.now();
    for (let b = 0; b < batches; b++) {
      vk.batchBegin();
      for (let i = 0; i < numOps; i++) {
        vk.batchDispatch(pipeline, [bufA, bufB, outBufs[i]], groups, 1, 1, push);
      }
      const tv = vk.batchSubmit();
      vk.waitTimeline(tv);
    }
    const elapsed = performance.now() - t0;

    const totalOps = batches * numOps;
    const bytesPerOp = size * 4 * 3;
    const throughput = (bytesPerOp * totalOps) / (elapsed / 1000) / 1e9;
    console.log(`  size=${(size/1e6).toFixed(0)}M | ${numOps} ops/batch × ${batches} batches | ${(elapsed/totalOps).toFixed(2)}ms/op | ${throughput.toFixed(1)} GB/s`);

    vk.destroyBuffer(bufA);
    vk.destroyBuffer(bufB);
    for (const h of outBufs) vk.destroyBuffer(h);
  }

  // ── Test 3: Allocate lots of VRAM ─────────────────────────────────
  console.log();
  console.log("=== Test 3: VRAM allocation capacity ===");
  const handles: number[] = [];
  const chunkBytes = 1024 * 1024 * 1024; // 1 GB chunks
  let totalGB = 0;
  try {
    for (let i = 0; i < 70; i++) { // Try up to 70 GB
      handles.push(vk.createBuffer(chunkBytes, 0));
      totalGB++;
      if (totalGB % 10 === 0) console.log(`  Allocated ${totalGB} GB...`);
    }
  } catch (e: any) {
    console.log(`  Max VRAM allocated: ${totalGB} GB (failed at ${totalGB + 1} GB: ${e.message})`);
  }

  // Do an add on the last two 1GB buffers
  if (handles.length >= 2) {
    const giantPipeline = vk.createPipeline(getKernelSpirv("add_vec4", WG), 3);
    const giantSize = chunkBytes / 4;
    const giantVec4 = giantSize >> 2;
    const giantGroups = Math.ceil(giantVec4 / WG);
    const giantPush = new Float32Array([giantVec4, 0]);
    const outGiant = vk.createBuffer(chunkBytes, 0);

    const t0 = performance.now();
    vk.dispatch(giantPipeline, [handles[0], handles[1], outGiant], giantGroups, 1, 1, giantPush);
    const elapsed = performance.now() - t0;
    const gbps = (chunkBytes * 3) / (elapsed / 1000) / 1e9;
    console.log(`  1GB add dispatch: ${elapsed.toFixed(1)}ms | ${gbps.toFixed(1)} GB/s`);

    vk.destroyBuffer(outGiant);
  }

  for (const h of handles) vk.destroyBuffer(h);

  console.log();
  console.log("All tests complete.");
}

main();
