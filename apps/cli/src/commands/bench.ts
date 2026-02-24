/**
 * Command: alpha bench
 *
 * Suites:
 *   ops   — matmul, softmax, layernorm, gelu (single backend)
 *   e2e   — forward-pass-like sequence (single backend)
 *   gpu   — element-wise ops: helios GPU vs cpu_ref comparison
 *   train — full training iterations: helios vs cpu_ref comparison
 */
import { parseKV, strArg, intArg } from "../parse.js";
import { resolveBackend, resolveTokenizer, resolveOptimizer, resolveRng } from "../resolve.js";
import type { Backend, TensorData } from "@alpha/core";
import { HeliosBackend } from "@alpha/helios";
import type { WebGpuBenchSpec, WebGpuBenchResult } from "./bench-webgpu.js";

export async function benchCmd(args: string[]): Promise<void> {
  const kv = parseKV(args);
  const suite = strArg(kv, "suite", "ops");
  const backendName = strArg(kv, "backend", "cpu_ref");
  const iters = intArg(kv, "iters", 100);

  if (suite === "gpu" || suite === "all") {
    await benchGpu(iters);
  }
  if (suite === "train" || suite === "all") {
    await benchTrain(iters);
  }
  if (suite === "ops" || suite === "all") {
    const backend = resolveBackend(backendName);
    console.log(`Benchmarking: suite=ops backend=${backendName} iters=${iters}\n`);
    await benchOps(backend, iters);
  }
  if (suite === "e2e" || suite === "all") {
    const backend = resolveBackend(backendName);
    console.log(`Benchmarking: suite=e2e backend=${backendName} iters=${iters}\n`);
    await benchE2e(backend, iters);
  }
}

// ── GPU benchmark: helios vs cpu_ref vs webgpu element-wise ops ─────────────

type BenchOp = "add" | "mul" | "scale" | "exp" | "neg";

/**
 * Run WebGPU benchmarks in an isolated child process.
 * Dawn's Vulkan backend conflicts with helios's Vulkan in the same process,
 * so we fork a separate Node.js process for the WebGPU measurements.
 */
async function runWebGpuInChild(
  ops: BenchOp[], sizes: number[], iters: number, checkSize: number,
): Promise<WebGpuBenchResult | null> {
  const { fork } = await import("node:child_process");
  const { fileURLToPath } = await import("node:url");
  const path = await import("node:path");

  // Resolve the compiled bench-webgpu.js path
  const thisFile = fileURLToPath(import.meta.url);
  const workerPath = path.join(path.dirname(thisFile), "bench-webgpu.js");

  return new Promise((resolve) => {
    const child = fork(workerPath, [], { stdio: ["pipe", "pipe", "pipe", "ipc"] });
    let result: WebGpuBenchResult | null = null;

    child.on("message", (msg: WebGpuBenchResult) => {
      result = msg;
    });

    child.on("exit", () => {
      resolve(result);
    });

    child.on("error", () => {
      resolve(null);
    });

    // Send the work spec
    const spec: WebGpuBenchSpec = { ops, sizes, iters, checkSize };
    child.send(spec);
  });
}

async function benchGpu(iters: number): Promise<void> {
  console.log("── GPU Benchmark: helios vs cpu_ref vs webgpu (Dawn) ──\n");

  const cpuBackend = resolveBackend("cpu_ref");
  const gpuBackend = new HeliosBackend();
  gpuBackend.setMinGpuSize(0); // Force GPU path for all sizes

  const ops: { name: BenchOp; run: (b: Backend, a: TensorData, extra: TensorData) => TensorData }[] = [
    { name: "add",   run: (b, a, e) => b.add(a, e) },
    { name: "mul",   run: (b, a, e) => b.mul(a, e) },
    { name: "scale", run: (b, a) => b.scale(a, 2.0) },
    { name: "exp",   run: (b, a) => b.exp(a) },
    { name: "neg",   run: (b, a) => b.neg(a) },
  ];

  const sizes = [1024, 4096, 16_384, 65_536, 262_144, 1_048_576, 4_194_304];
  const opNames = ops.map((o) => o.name);
  const checkSize = 4096;

  // Dispatch overhead test (helios)
  console.log("Helios dispatch overhead (empty round-trip):");
  const tiny = gpuBackend.ones([4]);
  const overheadMs = bench(() => gpuBackend.add(tiny, tiny), Math.min(iters, 50));
  console.log(`  ${overheadMs.toFixed(3)}ms per dispatch\n`);

  // ── Phase 1: CPU + Helios ─────────────────────────────────────────────
  console.log("Benchmarking cpu_ref + helios ...");
  const rows: { op: string; size: number; cpuMs: number; heliosMs: number; webgpuMs: number }[] = [];

  for (const op of ops) {
    for (const size of sizes) {
      const cpuA = cpuBackend.randn([size]);
      const cpuB = cpuBackend.randn([size]);
      const gpuA = gpuBackend.randn([size]);
      const gpuB = gpuBackend.randn([size]);

      const cpuMs = bench(() => op.run(cpuBackend, cpuA, cpuB), iters);
      const heliosMs = bench(() => op.run(gpuBackend, gpuA, gpuB), iters);

      rows.push({ op: op.name, size, cpuMs, heliosMs, webgpuMs: NaN });
    }
  }

  // Helios correctness
  const ca = cpuBackend.randn([checkSize]);
  const ga = gpuBackend.fromArray(Array.from(ca.data), [checkSize]);
  const cb = cpuBackend.randn([checkSize]);
  const gb = gpuBackend.fromArray(Array.from(cb.data), [checkSize]);

  const heliosChecks: { op: string; pass: boolean }[] = [];
  for (const op of ops) {
    const cpuResult = op.run(cpuBackend, ca, cb);
    const heliosResult = op.run(gpuBackend, ga, gb);
    heliosChecks.push({ op: op.name, pass: cpuBackend.allClose(cpuResult, heliosResult, 1e-4) });
  }

  // ── Phase 2: WebGPU (in child process for Vulkan isolation) ───────────
  console.log("Benchmarking webgpu (Dawn) in isolated process ...");
  const wgpuResult = await runWebGpuInChild(opNames, sizes, iters, checkSize);
  const hasWebGpu = wgpuResult !== null && wgpuResult.available;

  if (hasWebGpu && wgpuResult) {
    // Merge webgpu timings into rows
    const wgpuMap = new Map<string, number>();
    for (const wr of wgpuResult.rows) wgpuMap.set(`${wr.op}:${wr.size}`, wr.ms);
    for (const r of rows) {
      const ms = wgpuMap.get(`${r.op}:${r.size}`);
      if (ms !== undefined) r.webgpuMs = ms;
    }
    console.log();
  } else {
    console.log("WebGPU (Dawn): not available — skipping webgpu column\n");
  }

  // ── Print table ───────────────────────────────────────────────────────
  const cols = [
    ["op", 8], ["size", 10],
    ["cpu_ms", 10], ["helios_ms", 11], ...(hasWebGpu ? [["webgpu_ms", 11]] : []),
    ["cpu M/s", 10], ["helios M/s", 12], ...(hasWebGpu ? [["webgpu M/s", 12]] : []),
    ["h/cpu", 8], ...(hasWebGpu ? [["w/cpu", 8], ["h/w", 8]] : []),
  ] as [string, number][];
  const hdr = cols.map(([s, w]) => padR(s, w)).join("");
  console.log(hdr);
  console.log("─".repeat(hdr.length));

  const fmtSpd = (v: number) => isNaN(v) ? "N/A" : v >= 1 ? `${v.toFixed(2)}x` : `1/${(1 / v).toFixed(1)}x`;

  let prevOp = "";
  for (const r of rows) {
    if (prevOp && r.op !== prevOp) console.log();
    prevOp = r.op;

    const cpuTP = r.size / (r.cpuMs * 1e3);
    const heliosTP = r.size / (r.heliosMs * 1e3);
    const hCpu = r.cpuMs / r.heliosMs;

    let row =
      padR(r.op, 8) + padR(fmtSize(r.size), 10) +
      padR(r.cpuMs.toFixed(3), 10) + padR(r.heliosMs.toFixed(3), 11);
    if (hasWebGpu) row += padR(isNaN(r.webgpuMs) ? "N/A" : r.webgpuMs.toFixed(3), 11);
    row += padR(cpuTP.toFixed(1), 10) + padR(heliosTP.toFixed(1), 12);
    if (hasWebGpu) {
      const webgpuTP = r.size / (r.webgpuMs * 1e3);
      row += padR(isNaN(webgpuTP) ? "N/A" : webgpuTP.toFixed(1), 12);
    }
    row += padR(fmtSpd(hCpu), 8);
    if (hasWebGpu) {
      const wCpu = r.cpuMs / r.webgpuMs;
      const hW = r.webgpuMs / r.heliosMs;
      row += padR(fmtSpd(wCpu), 8) + padR(fmtSpd(hW), 8);
    }

    console.log(row);
  }
  console.log();

  // ── Correctness check ─────────────────────────────────────────────────
  console.log("Correctness check (" + checkSize + " elements):");
  for (const hc of heliosChecks) {
    let line = `  ${hc.op}: helios=${hc.pass ? "PASS" : "FAIL"}`;
    if (hasWebGpu && wgpuResult) {
      const wc = wgpuResult.checks.find((c) => c.op === hc.op);
      if (wc) line += ` webgpu=${wc.pass ? "PASS" : "FAIL"}`;
    }
    console.log(line);
  }
  console.log();
}

// ── Training benchmark: helios vs cpu_ref ───────────────────────────────────

async function benchTrain(iters: number): Promise<void> {
  console.log("── Training Benchmark: helios vs cpu_ref ──\n");

  const { train } = await import("@alpha/train");
  const { defaultModelConfig } = await import("@alpha/core");
  const { CharTokenizer } = await import("@alpha/tokenizers");
  const { Effect } = await import("effect");
  const fs = await import("node:fs/promises");
  const os = await import("node:os");
  const path = await import("node:path");

  // Use a small dataset — animals.txt is small and quick to tokenize
  const dataPath = path.join(process.cwd(), "data", "animals.txt");
  let text: string;
  try {
    text = await fs.readFile(dataPath, "utf-8");
  } catch {
    console.log("  Skipping train benchmark: data/animals.txt not found");
    console.log("  (Provide a small text file at data/animals.txt)\n");
    return;
  }

  // Use char tokenizer (instant build, no BPE overhead)
  const tokenizer = new CharTokenizer();
  await Effect.runPromise(tokenizer.build(text));

  // Tiny model config for benchmarking
  const modelConfig = {
    ...defaultModelConfig,
    vocabSize: tokenizer.vocabSize,
    blockSize: 32,
    nLayer: 2,
    nEmbd: 64,
    nHead: 4,
    dropout: 0.0,
  };

  const trainConfig = {
    iters,
    batchSize: 4,
    lr: 3e-4,
    beta1: 0.9,
    beta2: 0.999,
    eps: 1e-8,
    weightDecay: 0.01,
    gradClip: 1.0,
    evalInterval: iters + 1, // no eval during bench
    evalIters: 0,
    seed: 42,
    backend: "cpu_ref",
    tokenizer: "char",
    optimizer: "adamw",
    logLevel: "error" as const,
    trace: false,
    gradAccumSteps: 1,
    lrMin: 0,
    warmupIters: -1, // disable warmup for bench
    sampleInterval: 0, // no samples during bench
    spikeThreshold: 0,
  };

  const backends = ["cpu_ref", "helios"] as const;
  const results: { name: string; totalMs: number; msPerIter: number; tokPerSec: number; finalLoss: number }[] = [];

  for (const backendName of backends) {
    console.log(`  Running ${iters} iterations with ${backendName}...`);

    const backend = resolveBackend(backendName);
    const optimizer = resolveOptimizer("adamw", backend);
    const rng = resolveRng(42);

    const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), `alpha-bench-${backendName}-`));

    const stepMetrics: { loss: number; tokens_per_sec: number; ms_per_iter: number }[] = [];
    const startTime = performance.now();

    await train({
      backend,
      tokenizer,
      optimizer,
      rng,
      modelConfig,
      trainConfig: { ...trainConfig, backend: backendName },
      dataPath,
      runDir: tmpDir,
      onStep: (m) => stepMetrics.push({ loss: m.loss, tokens_per_sec: m.tokens_per_sec, ms_per_iter: m.ms_per_iter }),
    });

    const totalMs = performance.now() - startTime;
    const lastMetric = stepMetrics[stepMetrics.length - 1];
    const avgTokPerSec = stepMetrics.reduce((s, m) => s + m.tokens_per_sec, 0) / stepMetrics.length;
    const avgMsPerIter = stepMetrics.reduce((s, m) => s + m.ms_per_iter, 0) / stepMetrics.length;

    results.push({
      name: backendName,
      totalMs,
      msPerIter: avgMsPerIter,
      tokPerSec: avgTokPerSec,
      finalLoss: lastMetric?.loss ?? NaN,
    });

    // Clean up temp dir
    await fs.rm(tmpDir, { recursive: true, force: true });
  }

  // Print comparison table
  console.log();
  const hdr = padR("backend", 12) + padR("total_ms", 14) + padR("ms/iter", 12) + padR("tok/s", 12) + "final_loss";
  console.log(hdr);
  console.log("─".repeat(hdr.length));

  for (const r of results) {
    console.log(
      padR(r.name, 12) +
      padR(r.totalMs.toFixed(1), 14) +
      padR(r.msPerIter.toFixed(1), 12) +
      padR(r.tokPerSec.toFixed(1), 12) +
      r.finalLoss.toFixed(4)
    );
  }

  // Speedup
  if (results.length === 2) {
    const speedup = results[0].totalMs / results[1].totalMs;
    console.log(`\n  Speedup (helios vs cpu_ref): ${speedup.toFixed(2)}x`);
    const lossDiff = Math.abs(results[0].finalLoss - results[1].finalLoss);
    console.log(`  Loss difference: ${lossDiff.toFixed(6)} ${lossDiff < 0.01 ? "(OK)" : "(DIVERGED)"}`);
  }
  console.log();
}

// ── Original suites (single-backend) ────────────────────────────────────────

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

// ── Helpers ─────────────────────────────────────────────────────────────────

function bench(fn: () => void, iters: number): number {
  // Warmup
  for (let i = 0; i < 3; i++) fn();
  const start = performance.now();
  for (let i = 0; i < iters; i++) fn();
  return (performance.now() - start) / iters;
}

function padR(s: string, w: number): string {
  return s.length >= w ? s + " " : s + " ".repeat(w - s.length);
}

function fmtSize(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
  return `${n}`;
}
