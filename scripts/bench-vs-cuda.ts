/**
 * Compare Alpha Helios matmul performance against PyTorch CUDA on identical shapes.
 *
 * Usage:
 *   npx tsx scripts/bench-vs-cuda.ts --shapes=1024x1024x1024,2048x2048x2048 --iters=20
 */
import { spawnSync } from "node:child_process";
import path from "node:path";
import { HeliosBackend } from "@alpha/helios";
import type { TensorData } from "@alpha/core";

interface ShapeSpec {
  m: number;
  k: number;
  n: number;
  key: string;
}

interface HeliosRow {
  shape: string;
  avgMs: number;
  tflops: number;
}

interface HeliosResult {
  ok: boolean;
  backend: string;
  deviceName: string;
  vendorId: number;
  f16Supported: boolean;
  iters: number;
  warmup: number;
  rows: HeliosRow[];
  error?: string;
}

interface CudaRow {
  shape: string;
  avg_ms: number;
  tflops: number;
}

interface CudaResult {
  ok: boolean;
  framework?: string;
  torch_version?: string;
  cuda_runtime?: string;
  device_name?: string;
  device_capability?: string;
  dtype?: string;
  rows?: CudaRow[];
  error?: string;
  hint?: string;
}

interface CliOptions {
  shapes: ShapeSpec[];
  iters: number;
  warmup: number;
  python: string;
  dtype: "float16" | "float32" | "bfloat16";
}

function parseShape(entry: string): ShapeSpec {
  const parts = entry.trim().split("x").map((v) => Number.parseInt(v, 10));
  if (parts.length !== 3 || parts.some((v) => !Number.isFinite(v) || v <= 0)) {
    throw new Error(`Invalid shape "${entry}". Expected MxKxN with positive integers.`);
  }
  const [m, k, n] = parts;
  return { m, k, n, key: `${m}x${k}x${n}` };
}

function parseArgs(argv: string[]): CliOptions {
  const kv: Record<string, string> = {};
  for (const arg of argv) {
    if (!arg.startsWith("--")) continue;
    const eq = arg.indexOf("=");
    if (eq > 0) kv[arg.slice(2, eq)] = arg.slice(eq + 1);
    else kv[arg.slice(2)] = "true";
  }

  const rawShapes = kv["shapes"] ?? "1024x1024x1024,2048x2048x2048,3072x3072x3072";
  const shapes = rawShapes.split(",").filter(Boolean).map(parseShape);
  const iters = Math.max(1, Number.parseInt(kv["iters"] ?? "12", 10));
  const warmup = Math.max(0, Number.parseInt(kv["warmup"] ?? "4", 10));
  const python = kv["python"] ?? "python3";
  const dtypeRaw = (kv["dtype"] ?? "float32") as CliOptions["dtype"];
  const dtype = (dtypeRaw === "float16" || dtypeRaw === "float32" || dtypeRaw === "bfloat16")
    ? dtypeRaw
    : "float32";

  return { shapes, iters, warmup, python, dtype };
}

function pad(value: string, width: number): string {
  if (value.length >= width) return `${value} `;
  return `${value}${" ".repeat(width - value.length)}`;
}

function formatRatio(ratio: number): string {
  if (!Number.isFinite(ratio) || ratio <= 0) return "n/a";
  if (ratio >= 1) return `${ratio.toFixed(2)}x`;
  return `1/${(1 / ratio).toFixed(2)}x`;
}

function benchmarkHeliosMatmul(opts: CliOptions): HeliosResult {
  try {
    const backend = new HeliosBackend();
    backend.setMinGpuSize(0);
    const backendAny = backend as any;
    const info = typeof backendAny.getDeviceInfo === "function"
      ? backendAny.getDeviceInfo()
      : { deviceName: "unknown", vendorId: 0, f16Supported: false };
    const syncFn: (() => void) | null =
      typeof backendAny.syncGpu === "function" ? backendAny.syncGpu.bind(backendAny) : null;
    const flushFn: (() => void) | null =
      typeof backendAny.flush === "function" ? backendAny.flush.bind(backendAny) : null;
    const releaseFn: ((td: TensorData) => void) | null =
      typeof backendAny.releaseGpuTensor === "function" ? backendAny.releaseGpuTensor.bind(backendAny) : null;

    const rows: HeliosRow[] = [];
    for (const shape of opts.shapes) {
      const a = backend.randn([shape.m, shape.k]);
      const b = backend.randn([shape.k, shape.n]);

      const warmupOuts: TensorData[] = [];
      for (let i = 0; i < opts.warmup; i++) warmupOuts.push(backend.matmul(a, b));
      if (syncFn) syncFn();
      if (flushFn) flushFn();
      if (releaseFn) for (const td of warmupOuts) releaseFn(td);

      const timedOuts: TensorData[] = [];
      const t0 = performance.now();
      for (let i = 0; i < opts.iters; i++) timedOuts.push(backend.matmul(a, b));
      if (syncFn) syncFn();
      else void (timedOuts[timedOuts.length - 1].data as Float32Array)[0];
      const elapsedMs = performance.now() - t0;

      if (flushFn) flushFn();
      if (releaseFn) for (const td of timedOuts) releaseFn(td);
      if (releaseFn) {
        releaseFn(a);
        releaseFn(b);
      }

      const avgMs = elapsedMs / opts.iters;
      const flops = 2 * shape.m * shape.k * shape.n;
      const tflops = (flops / (avgMs / 1000)) / 1e12;
      rows.push({ shape: shape.key, avgMs, tflops });
    }

    return {
      ok: true,
      backend: backend.name,
      deviceName: info.deviceName,
      vendorId: info.vendorId,
      f16Supported: !!info.f16Supported,
      iters: opts.iters,
      warmup: opts.warmup,
      rows,
    };
  } catch (error) {
    return {
      ok: false,
      backend: "helios",
      deviceName: "unknown",
      vendorId: 0,
      f16Supported: false,
      iters: opts.iters,
      warmup: opts.warmup,
      rows: [],
      error: (error as Error).message,
    };
  }
}

function benchmarkCudaReference(opts: CliOptions): CudaResult {
  const scriptPath = path.resolve(process.cwd(), "scripts", "bench-cuda-reference.py");
  const shapesArg = opts.shapes.map((s) => s.key).join(",");
  const child = spawnSync(
    opts.python,
    [scriptPath, `--shapes=${shapesArg}`, `--iters=${opts.iters}`, `--warmup=${opts.warmup}`, `--dtype=${opts.dtype}`],
    { encoding: "utf-8", maxBuffer: 2 * 1024 * 1024 },
  );

  if (child.error) {
    return { ok: false, error: `Failed to launch ${opts.python}: ${child.error.message}` };
  }
  const out = (child.stdout ?? "").trim();
  if (!out) {
    return {
      ok: false,
      error: `CUDA reference benchmark produced no output (exit=${child.status ?? "unknown"}).`,
      hint: (child.stderr ?? "").trim() || undefined,
    };
  }
  try {
    return JSON.parse(out) as CudaResult;
  } catch {
    return {
      ok: false,
      error: "CUDA reference benchmark returned non-JSON output.",
      hint: out.slice(0, 500),
    };
  }
}

function printResults(helios: HeliosResult, cuda: CudaResult): void {
  console.log("── Alpha Helios vs CUDA (PyTorch) matmul benchmark ──");
  console.log(`iters=${helios.iters} warmup=${helios.warmup}`);

  if (helios.ok) {
    const vendorHex = `0x${helios.vendorId.toString(16)}`;
    console.log(`helios_device=${helios.deviceName} vendor=${vendorHex} f16=${helios.f16Supported}`);
  } else {
    console.log(`helios benchmark failed: ${helios.error}`);
  }

  if (cuda.ok) {
    console.log(`cuda_device=${cuda.device_name} capability=${cuda.device_capability} torch=${cuda.torch_version} cuda=${cuda.cuda_runtime} dtype=${cuda.dtype}`);
  } else {
    console.log(`cuda benchmark unavailable: ${cuda.error}`);
    if (cuda.hint) console.log(`hint: ${cuda.hint}`);
  }

  const cudaMap = new Map<string, CudaRow>();
  for (const row of cuda.rows ?? []) cudaMap.set(row.shape, row);

  console.log("");
  const header =
    pad("shape", 18) +
    pad("helios_ms", 12) +
    pad("cuda_ms", 12) +
    pad("helios_tflops", 15) +
    pad("cuda_tflops", 13) +
    pad("h_vs_cuda", 12);
  console.log(header);
  console.log("-".repeat(header.length));

  for (const h of helios.rows) {
    const c = cudaMap.get(h.shape);
    const heliosMs = h.avgMs.toFixed(3);
    const heliosTf = h.tflops.toFixed(3);
    const cudaMs = c ? c.avg_ms.toFixed(3) : "n/a";
    const cudaTf = c ? c.tflops.toFixed(3) : "n/a";
    const ratio = c ? c.avg_ms / h.avgMs : Number.NaN;

    console.log(
      pad(h.shape, 18) +
      pad(heliosMs, 12) +
      pad(cudaMs, 12) +
      pad(heliosTf, 15) +
      pad(cudaTf, 13) +
      pad(formatRatio(ratio), 12),
    );
  }
  console.log("");
}

function printUsage(): void {
  console.log(`Usage: npx tsx scripts/bench-vs-cuda.ts [options]

Options:
  --shapes=1024x1024x1024,2048x2048x2048
  --iters=12
  --warmup=4
  --dtype=float32            (for CUDA reference: float16|float32|bfloat16)
  --python=python3
`);
}

function main(): void {
  if (process.argv.includes("--help") || process.argv.includes("-h")) {
    printUsage();
    return;
  }

  const opts = parseArgs(process.argv.slice(2));
  const helios = benchmarkHeliosMatmul(opts);
  const cuda = benchmarkCudaReference(opts);
  printResults(helios, cuda);

  if (!helios.ok || !cuda.ok) process.exitCode = 1;
}

main();
