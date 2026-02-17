/**
 * Op-level microbenchmarks.
 */
import type { Backend } from "@alpha/core";

export interface BenchResult {
  name: string;
  shape: string;
  avgMs: number;
  iters: number;
  extra?: string;
}

function run(fn: () => void, iters: number): number {
  for (let i = 0; i < 3; i++) fn(); // warmup
  const start = performance.now();
  for (let i = 0; i < iters; i++) fn();
  return (performance.now() - start) / iters;
}

export function benchMatmul(backend: Backend, n: number, iters = 100): BenchResult {
  const a = backend.randn([n, n]);
  const b = backend.randn([n, n]);
  const avgMs = run(() => backend.matmul(a, b), iters);
  const gflops = (2 * n * n * n) / (avgMs * 1e6);
  return { name: "matmul", shape: `${n}x${n}`, avgMs, iters, extra: `${gflops.toFixed(2)} GFLOP/s` };
}

export function benchSoftmax(backend: Backend, n: number, iters = 100): BenchResult {
  const x = backend.randn([n, n]);
  const avgMs = run(() => backend.softmax(x), iters);
  return { name: "softmax", shape: `${n}x${n}`, avgMs, iters };
}

export function benchGelu(backend: Backend, n: number, iters = 100): BenchResult {
  const x = backend.randn([n, n]);
  const avgMs = run(() => backend.gelu(x), iters);
  return { name: "gelu", shape: `${n}x${n}`, avgMs, iters };
}

export function benchLayerNorm(backend: Backend, n: number, iters = 100): BenchResult {
  const x = backend.randn([n, n]);
  const w = backend.ones([n]);
  const b = backend.zeros([n]);
  const avgMs = run(() => backend.layerNorm(x, w, b, 1e-5), iters);
  return { name: "layernorm", shape: `${n}x${n}`, avgMs, iters };
}

export function benchEmbedding(backend: Backend, vocabSize: number, dim: number, seqLen: number, iters = 100): BenchResult {
  const w = backend.randn([vocabSize, dim]);
  const idx = backend.fromArray(
    Array.from({ length: seqLen }, () => Math.floor(Math.random() * vocabSize)),
    [seqLen],
    "i32",
  );
  const avgMs = run(() => backend.embedding(w, idx), iters);
  return { name: "embedding", shape: `V=${vocabSize} D=${dim} T=${seqLen}`, avgMs, iters };
}

export function runAllBenches(backend: Backend, iters = 100): BenchResult[] {
  const results: BenchResult[] = [];
  for (const n of [64, 128, 256]) {
    results.push(benchMatmul(backend, n, iters));
    results.push(benchSoftmax(backend, n, iters));
    results.push(benchGelu(backend, n, iters));
    results.push(benchLayerNorm(backend, n, iters));
  }
  results.push(benchEmbedding(backend, 256, 64, 512, iters));
  return results;
}
