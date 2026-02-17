/**
 * Command: alpha bench
 */
import { parseKV, strArg, intArg } from "../parse.js";
import { resolveBackend } from "../resolve.js";
import type { Backend, TensorData } from "@alpha/core";

export async function benchCmd(args: string[]): Promise<void> {
  const kv = parseKV(args);
  const suite = strArg(kv, "suite", "ops");
  const backendName = strArg(kv, "backend", "cpu_ref");
  const iters = intArg(kv, "iters", 100);

  const backend = resolveBackend(backendName);
  console.log(`Benchmarking: suite=${suite} backend=${backendName} iters=${iters}\n`);

  if (suite === "ops" || suite === "all") {
    await benchOps(backend, iters);
  }
  if (suite === "e2e" || suite === "all") {
    await benchE2e(backend, iters);
  }
}

async function benchOps(backend: Backend, iters: number): Promise<void> {
  console.log("── Op Benchmarks ──\n");

  const sizes = [64, 128, 256, 512];

  for (const n of sizes) {
    // Matmul
    const a = backend.randn([n, n]);
    const b = backend.randn([n, n]);
    const ms = bench(() => backend.matmul(a, b), iters);
    const gflops = (2 * n * n * n) / (ms * 1e6);
    console.log(`matmul [${n}x${n}]: ${ms.toFixed(2)}ms (${gflops.toFixed(2)} GFLOP/s)`);

    // Softmax
    const x = backend.randn([n, n]);
    const msS = bench(() => backend.softmax(x), iters);
    console.log(`softmax [${n}x${n}]: ${msS.toFixed(2)}ms`);

    // LayerNorm
    const w = backend.ones([n]);
    const bi = backend.zeros([n]);
    const msLn = bench(() => backend.layerNorm(x, w, bi, 1e-5), iters);
    console.log(`layernorm [${n}x${n}]: ${msLn.toFixed(2)}ms`);

    // GELU
    const msG = bench(() => backend.gelu(x), iters);
    console.log(`gelu [${n}x${n}]: ${msG.toFixed(2)}ms`);

    console.log();
  }
}

async function benchE2e(backend: Backend, iters: number): Promise<void> {
  console.log("── E2E Benchmark ──\n");

  // A simple forward-pass-like sequence
  const B = 4, T = 32, D = 64, V = 128;

  const wte = backend.randn([V, D]);
  const wpe = backend.randn([T, D]);
  const wq = backend.randn([D, D]);

  const tokens = backend.fromArray(
    Array.from({ length: B * T }, () => Math.floor(Math.random() * V)),
    [B * T],
    "i32",
  );

  const ms = bench(() => {
    const emb = backend.embedding(wte, tokens);
    const flat = backend.reshape(emb, [B * T, D]);
    const q = backend.matmul(flat, wq);
    backend.gelu(q);
  }, Math.min(iters, 20));

  console.log(`e2e forward (B=${B} T=${T} D=${D}): ${ms.toFixed(2)}ms/iter`);
}

function bench(fn: () => void, iters: number): number {
  // Warmup
  for (let i = 0; i < 3; i++) fn();
  const start = performance.now();
  for (let i = 0; i < iters; i++) fn();
  return (performance.now() - start) / iters;
}
