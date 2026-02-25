/**
 * Sustained GPU load test — runs for 30 seconds to measure nvidia-smi utilization.
 */
import { getNative, initDevice, getKernelSpirv } from "@alpha/helios";

const info = initDevice();
console.log("Device:", info.deviceName);
const vk = getNative();

const WG = 256;
const size = 100_000_000; // 100M elements = 400MB
const byteSize = size * 4;
const vec4Size = size >> 2;
const groups = Math.ceil(vec4Size / WG);

const spirv = getKernelSpirv("add_vec4", WG);
const pipeline = vk.createPipeline(spirv, 3);

const bufA = vk.createBuffer(byteSize, 0);
const bufB = vk.createBuffer(byteSize, 0);
const bufC = vk.createBuffer(byteSize, 0);

// Fill with data
const data = new Float32Array(size);
for (let i = 0; i < size; i++) data[i] = Math.random();
vk.uploadBuffer(bufA, data);
vk.uploadBuffer(bufB, data);

const push = new Float32Array([vec4Size, 0]);

// Warmup
for (let i = 0; i < 5; i++) {
  vk.dispatch(pipeline, [bufA, bufB, bufC], groups, 1, 1, push);
}

console.log("Starting sustained GPU compute for 30 seconds...");
console.log("Run 'nvidia-smi' in another terminal to see utilization.");

const duration = 30_000;
const end = Date.now() + duration;
let count = 0;

while (Date.now() < end) {
  // Batch 100 dispatches per submit
  vk.batchBegin();
  for (let i = 0; i < 100; i++) {
    vk.batchDispatch(pipeline, [bufA, bufB, bufC], groups, 1, 1, push);
  }
  const tv = vk.batchSubmit();
  vk.waitTimeline(tv);
  count += 100;
}

const elapsed = (Date.now() - (end - duration));
console.log(`Done: ${count} dispatches in ${(elapsed/1000).toFixed(1)}s`);
console.log(`${(count/elapsed*1000).toFixed(0)} dispatches/s`);

const bytesPerDispatch = byteSize * 3; // read a + read b + write c
const totalBytes = bytesPerDispatch * count;
console.log(`Throughput: ${(totalBytes / (elapsed/1000) / 1e12).toFixed(2)} TB/s effective`);

vk.destroyBuffer(bufA);
vk.destroyBuffer(bufB);
vk.destroyBuffer(bufC);
