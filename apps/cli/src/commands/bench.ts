/**
 * Command: alpha bench
 *
 * Suites:
 *   ops   — matmul, softmax, layernorm, gelu (single backend)
 *   e2e   — forward-pass-like sequence (single backend)
 *   gpu   — element-wise ops: helios GPU vs cpu_ref comparison
 *   train — full training iterations: helios vs cpu_ref comparison
 *   cuda  — Helios matmul vs PyTorch CUDA matmul comparison
 */
import { parseKV, strArg, intArg } from "../parse.js";
import { resolveBackend, resolveTokenizer, resolveOptimizer, resolveRng } from "../resolve.js";
import type { Backend, TensorData } from "@alpha/core";
import { HeliosBackend } from "@alpha/helios";
import type { WebGpuBenchSpec, WebGpuBenchResult } from "./bench-webgpu.js";
import { spawnSync } from "node:child_process";
import { mkdtemp, writeFile, mkdir } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

interface MatmulShape {
  m: number;
  k: number;
  n: number;
  key: string;
}

interface HeliosCudaRow {
  shape: string;
  heliosMs: number;
  heliosTflops: number;
  cudaMs: number | null;
  cudaTflops: number | null;
}

export async function benchCmd(args: string[]): Promise<void> {
  const kv = parseKV(args);
  const suite = strArg(kv, "suite", "ops");
  const backendName = strArg(kv, "backend", "cpu_ref");
  const iters = intArg(kv, "iters", 100);

  if (suite === "cuda" || suite === "all") {
    await benchCuda(kv, iters);
  }

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
    syncEvery: 1,
    gcEvery: 0,
    packed: false,
    symbio: false,
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

function parseMatmulShape(raw: string): MatmulShape {
  const parts = raw.trim().split("x").map((v) => Number.parseInt(v, 10));
  if (parts.length !== 3 || parts.some((v) => !Number.isFinite(v) || v <= 0)) {
    throw new Error(`Invalid shape "${raw}". Expected MxKxN with positive integers.`);
  }
  const [m, k, n] = parts;
  return { m, k, n, key: `${m}x${k}x${n}` };
}

function parseMatmulShapes(raw: string): MatmulShape[] {
  const items = raw.split(",").map((s) => s.trim()).filter(Boolean);
  if (items.length === 0) throw new Error("No valid shapes provided.");
  return items.map(parseMatmulShape);
}

function ratioStr(r: number | null): string {
  if (r === null || !Number.isFinite(r) || r <= 0) return "n/a";
  if (r >= 1) return `${r.toFixed(2)}x`;
  return `1/${(1 / r).toFixed(2)}x`;
}

function buildCudaReferencePython(): string {
  return String.raw`#!/usr/bin/env python3
import argparse
import json
import time

def parse_shapes(raw):
  out = []
  for part in raw.split(","):
    part = part.strip()
    if not part:
      continue
    bits = part.split("x")
    if len(bits) != 3:
      raise ValueError(f"invalid shape '{part}', expected MxKxN")
    m, k, n = (int(bits[0]), int(bits[1]), int(bits[2]))
    if m <= 0 or k <= 0 or n <= 0:
      raise ValueError(f"invalid shape '{part}', dimensions must be > 0")
    out.append((m, k, n))
  if not out:
    raise ValueError("at least one shape must be provided")
  return out

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--shapes", required=True)
  ap.add_argument("--iters", type=int, default=20)
  ap.add_argument("--warmup", type=int, default=6)
  ap.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float32")
  args = ap.parse_args()

  try:
    import torch
  except Exception as e:
    print(json.dumps({
      "ok": False,
      "error": f"PyTorch import failed: {e}",
      "hint": "Install CUDA PyTorch: pip install --index-url https://download.pytorch.org/whl/cu128 torch"
    }))
    return

  if not torch.cuda.is_available():
    print(json.dumps({
      "ok": False,
      "error": "CUDA is not available in PyTorch.",
      "hint": "Use NVIDIA GPU + CUDA-enabled torch wheel."
    }))
    return

  shapes = parse_shapes(args.shapes)
  dtype_map = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16
  }
  dtype = dtype_map[args.dtype]

  torch.backends.cuda.matmul.allow_tf32 = True
  if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

  rows = []
  for (m, k, n) in shapes:
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((k, n), device="cuda", dtype=dtype)

    for _ in range(max(0, args.warmup)):
      _ = a @ b
    torch.cuda.synchronize()

    start = time.perf_counter()
    c = None
    for _ in range(max(1, args.iters)):
      c = a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed * 1000.0) / max(1, args.iters)
    flops = 2.0 * float(m) * float(k) * float(n)
    tflops = (flops / (avg_ms / 1000.0)) / 1e12
    rows.append({"shape": f"{m}x{k}x{n}", "avg_ms": avg_ms, "tflops": tflops})
    del c

  print(json.dumps({
    "ok": True,
    "framework": "pytorch_cuda",
    "torch_version": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "device_name": torch.cuda.get_device_name(0),
    "device_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
    "dtype": str(dtype).replace("torch.", ""),
    "rows": rows
  }))

if __name__ == "__main__":
  main()
`;
}

async function runCudaReference(pythonBin: string, shapes: MatmulShape[], iters: number, warmup: number, dtype: "float16" | "float32" | "bfloat16"): Promise<any> {
  const tempDir = await mkdtemp(path.join(os.tmpdir(), "alpha-cuda-ref-"));
  const scriptPath = path.join(tempDir, "bench-cuda-ref.py");
  await writeFile(scriptPath, buildCudaReferencePython(), "utf8");

  const shapesArg = shapes.map((s) => s.key).join(",");
  const child = spawnSync(
    pythonBin,
    [scriptPath, `--shapes=${shapesArg}`, `--iters=${iters}`, `--warmup=${warmup}`, `--dtype=${dtype}`],
    { encoding: "utf8", maxBuffer: 4 * 1024 * 1024 },
  );

  if (child.error) {
    return { ok: false, error: `Failed to launch ${pythonBin}: ${child.error.message}` };
  }
  const stdout = (child.stdout ?? "").trim();
  const stderr = (child.stderr ?? "").trim();
  if (!stdout) {
    return {
      ok: false,
      error: `CUDA benchmark returned no stdout (exit=${child.status ?? "unknown"}).`,
      hint: stderr || undefined,
    };
  }
  try {
    return JSON.parse(stdout);
  } catch {
    return {
      ok: false,
      error: "CUDA benchmark returned non-JSON output.",
      hint: stdout.slice(0, 500),
    };
  }
}

async function benchCuda(kv: Record<string, string>, iters: number): Promise<void> {
  const shapesRaw = strArg(kv, "shapes", "1024x1024x1024,2048x2048x2048,3072x3072x3072");
  const warmup = intArg(kv, "warmup", 6);
  const pythonBin = strArg(kv, "python", "python3");
  const dtypeRaw = strArg(kv, "dtype", "float32");
  const outPath = kv["out"];
  const dtype: "float16" | "float32" | "bfloat16" =
    dtypeRaw === "float16" || dtypeRaw === "float32" || dtypeRaw === "bfloat16"
      ? dtypeRaw
      : "float32";

  let shapes: MatmulShape[];
  try {
    shapes = parseMatmulShapes(shapesRaw);
  } catch (e) {
    console.error(`Invalid --shapes: ${(e as Error).message}`);
    process.exit(1);
  }

  console.log("── CUDA Reference Benchmark: Helios vs PyTorch CUDA ──\n");
  console.log(`iters=${iters} warmup=${warmup} dtype=${dtype}`);
  console.log(`shapes=${shapes.map((s) => s.key).join(",")}\n`);

  const helios = new HeliosBackend();
  helios.setMinGpuSize(0);
  const heliosAny = helios as any;
  const info = typeof heliosAny.getDeviceInfo === "function"
    ? heliosAny.getDeviceInfo()
    : {
      deviceName: "unknown",
      vendorId: 0,
      f16Supported: false,
      coopMatSupported: false,
      coopMatM: 0,
      coopMatN: 0,
      coopMatK: 0,
      hasPushDescriptors: false,
    };
  const syncFn: (() => void) | undefined =
    typeof heliosAny.syncGpu === "function" ? heliosAny.syncGpu.bind(heliosAny) : undefined;
  const flushFn: (() => void) | undefined =
    typeof heliosAny.flush === "function" ? heliosAny.flush.bind(heliosAny) : undefined;
  const releaseFn: ((td: TensorData) => void) | undefined =
    typeof heliosAny.releaseGpuTensor === "function" ? heliosAny.releaseGpuTensor.bind(heliosAny) : undefined;

  const rows: HeliosCudaRow[] = [];
  for (const shape of shapes) {
    const a = helios.randn([shape.m, shape.k]);
    const b = helios.randn([shape.k, shape.n]);

    const warmupOuts: TensorData[] = [];
    for (let i = 0; i < warmup; i++) warmupOuts.push(helios.matmul(a, b));
    if (syncFn) syncFn();
    if (flushFn) flushFn();
    if (releaseFn) for (const td of warmupOuts) releaseFn(td);

    const outs: TensorData[] = [];
    const t0 = performance.now();
    for (let i = 0; i < iters; i++) outs.push(helios.matmul(a, b));
    if (syncFn) syncFn();
    else void (outs[outs.length - 1].data as Float32Array)[0];
    const elapsedMs = performance.now() - t0;
    if (flushFn) flushFn();

    if (releaseFn) {
      for (const td of outs) releaseFn(td);
      releaseFn(a);
      releaseFn(b);
    }

    const avgMs = elapsedMs / iters;
    const flops = 2 * shape.m * shape.k * shape.n;
    const tflops = (flops / (avgMs / 1000)) / 1e12;
    rows.push({
      shape: shape.key,
      heliosMs: avgMs,
      heliosTflops: tflops,
      cudaMs: null,
      cudaTflops: null,
    });
  }
  const coopStats = typeof heliosAny.getMatmulCoopStats === "function"
    ? heliosAny.getMatmulCoopStats()
    : null;

  console.log(`helios_device=${info.deviceName} vendor=0x${Number(info.vendorId ?? 0).toString(16)} f16=${!!info.f16Supported}`);
  console.log(
    `helios_coop=${!!info.coopMatSupported} ` +
    `tile=${Number(info.coopMatM ?? 0)}x${Number(info.coopMatN ?? 0)}x${Number(info.coopMatK ?? 0)} ` +
    `push_desc=${!!info.hasPushDescriptors}`,
  );
  if (coopStats) {
    console.log(
      `helios_coop_stats ` +
      `hit_rate=${(Number(coopStats.coopHitRate ?? 0) * 100).toFixed(1)}% ` +
      `direct=${Number(coopStats.coopDirectDispatches ?? 0)} ` +
      `padded2d=${Number(coopStats.coopPadded2DDispatches ?? 0)} ` +
      `padded_batched=${Number(coopStats.coopPaddedBatchedDispatches ?? 0)} ` +
      `total=${Number(coopStats.totalMatmulDispatches ?? 0)}`,
    );
  }
  if (Number(info.vendorId ?? 0) !== 0x10de || /\bllvmpipe\b|\bswiftshader\b/i.test(String(info.deviceName ?? ""))) {
    console.warn("warning: Helios is not running on an NVIDIA Vulkan device (results may reflect CPU/software Vulkan fallback).");
  }
  console.log(`python=${pythonBin}\n`);

  const cudaResult = await runCudaReference(pythonBin, shapes, iters, warmup, dtype);
  if (cudaResult.ok) {
    console.log(
      `cuda_device=${cudaResult.device_name} capability=${cudaResult.device_capability} ` +
      `torch=${cudaResult.torch_version} cuda=${cudaResult.cuda_runtime}`,
    );
  } else {
    console.log(`cuda benchmark unavailable: ${cudaResult.error}`);
    if (cudaResult.hint) console.log(`hint: ${cudaResult.hint}`);
  }
  console.log("");

  const cudaMap = new Map<string, { avg_ms: number; tflops: number }>();
  for (const row of (cudaResult.rows ?? [])) {
    if (row?.shape) cudaMap.set(row.shape, row);
  }

  for (const row of rows) {
    const cuda = cudaMap.get(row.shape);
    if (cuda) {
      row.cudaMs = cuda.avg_ms;
      row.cudaTflops = cuda.tflops;
    }
  }

  const header =
    padR("shape", 18) +
    padR("helios_ms", 12) +
    padR("cuda_ms", 12) +
    padR("helios_tflops", 15) +
    padR("cuda_tflops", 13) +
    "h_vs_cuda";
  console.log(header);
  console.log("─".repeat(header.length));
  for (const row of rows) {
    const ratio = row.cudaMs ? row.cudaMs / row.heliosMs : null;
    console.log(
      padR(row.shape, 18) +
      padR(row.heliosMs.toFixed(3), 12) +
      padR(row.cudaMs ? row.cudaMs.toFixed(3) : "n/a", 12) +
      padR(row.heliosTflops.toFixed(3), 15) +
      padR(row.cudaTflops ? row.cudaTflops.toFixed(3) : "n/a", 13) +
      ratioStr(ratio),
    );
  }
  console.log("");

  const summary = {
    timestamp: new Date().toISOString(),
    iters,
    warmup,
    dtype,
    shapes: shapes.map((s) => s.key),
    helios: {
      backend: helios.name,
      device_name: info.deviceName,
      vendor_id: info.vendorId,
      f16_supported: !!info.f16Supported,
      coop_supported: !!info.coopMatSupported,
      coop_tile_mnk: [Number(info.coopMatM ?? 0), Number(info.coopMatN ?? 0), Number(info.coopMatK ?? 0)],
      has_push_descriptors: !!info.hasPushDescriptors,
      coop_stats: coopStats,
    },
    cuda: cudaResult,
    rows,
  };

  if (outPath) {
    const absOut = path.resolve(process.cwd(), outPath);
    await mkdir(path.dirname(absOut), { recursive: true });
    await writeFile(absOut, JSON.stringify(summary, null, 2) + "\n", "utf8");
    console.log(`saved: ${absOut}\n`);
  }
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
