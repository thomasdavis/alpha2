/**
 * bench-flash-repeat.ts
 *
 * Decision-grade flash microbench for the L4 coop2 race.
 *
 * Usage:
 *   npx tsx scripts/bench-flash-repeat.ts --repeats=7 --iters=40 --warmup=6
 *   npx tsx scripts/bench-flash-repeat.ts --repeats=7 --iters=40 --warmup=6 --gpu-time
 *
 * --gpu-time: Use Vulkan timestamp queries (GPU-only time, no CPU/queue overhead)
 *
 * Env toggles (examples):
 *   HELIOS_FLASH_COOP2_QT=2
 *   HELIOS_FLASH_COOP2_SCOPE=workgroup
 *   HELIOS_FLASH_COOP2_LS=128
 *   HELIOS_FLASH_COOP2_F16_INPUT=1
 *   HELIOS_FLASH_COOP2_SKIP_LSE_WRITE=0
 */
import { HeliosBackend } from "@alpha/helios";
import type { TensorData } from "@alpha/core";

interface CliOpts {
  repeats: number;
  iters: number;
  warmup: number;
  order: "default-first" | "direct-first" | "alternate";
  gpuTime: boolean;
}

interface RepeatRow {
  msDefault: number;
  msDirect: number;
  msQK: number;
  msPV: number;
  msKVOnly: number;
  // GPU timestamp versions (only when --gpu-time)
  gpuFull?: number;
  gpuQK?: number;
  gpuPV?: number;
  gpuKVOnly?: number;
  debugDefault?: string;
  debugDirect?: string;
  debugQK?: string;
  debugPV?: string;
  debugKVOnly?: string;
}

function parseArgs(): CliOpts {
  const kv: Record<string, string> = {};
  for (const arg of process.argv.slice(2)) {
    if (!arg.startsWith("--")) continue;
    const eq = arg.indexOf("=");
    if (eq > 0) kv[arg.slice(2, eq)] = arg.slice(eq + 1);
    else kv[arg.slice(2)] = "true";
  }
  return {
    repeats: Math.max(1, parseInt(kv.repeats ?? "7", 10)),
    iters: Math.max(1, parseInt(kv.iters ?? "40", 10)),
    warmup: Math.max(0, parseInt(kv.warmup ?? "6", 10)),
    order:
      kv.order === "default-first" || kv.order === "direct-first" || kv.order === "alternate"
        ? kv.order
        : "alternate",
    gpuTime: kv["gpu-time"] === "true",
  };
}

function median(xs: number[]): number {
  const arr = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(arr.length / 2);
  return arr.length % 2 === 0 ? 0.5 * (arr[mid - 1] + arr[mid]) : arr[mid];
}

function percentile(xs: number[], p: number): number {
  if (xs.length === 0) return NaN;
  const arr = [...xs].sort((a, b) => a - b);
  const idx = Math.min(arr.length - 1, Math.max(0, Math.ceil((p / 100) * arr.length) - 1));
  return arr[idx];
}

function trimmedMean(xs: number[]): number {
  if (xs.length <= 2) return median(xs);
  const arr = [...xs].sort((a, b) => a - b).slice(1, -1);
  let sum = 0;
  for (const x of arr) sum += x;
  return sum / arr.length;
}

function makeBackend(): HeliosBackend {
  const b = new HeliosBackend();
  (b as any).setMinGpuSize?.(0);
  return b;
}

function runOneRepeat(
  bAny: any,
  q: TensorData,
  k: TensorData,
  v: TensorData,
  iters: number,
  warmup: number,
  order: "default-first" | "direct-first",
  useGpuTime: boolean,
): RepeatRow {
  const sync: (() => void) = typeof bAny.syncGpu === "function" ? bAny.syncGpu.bind(bAny) : (() => {});
  const release: ((td: TensorData) => void) =
    typeof bAny.releaseGpuTensor === "function" ? bAny.releaseGpuTensor.bind(bAny) : (() => {});
  const BH = 16, T = 512, D = 64;
  const scale = 1.0 / Math.sqrt(D);

  function bench(fn: () => TensorData[]): number {
    for (let i = 0; i < warmup; i++) {
      const rs = fn();
      sync();
      for (const r of rs) release(r);
    }
    const times: number[] = [];
    for (let i = 0; i < iters; i++) {
      const t0 = performance.now();
      const rs = fn();
      sync();
      times.push(performance.now() - t0);
      for (const r of rs) release(r);
    }
    return median(times);
  }

  // GPU timestamp bench: uses vkCmdWriteTimestamp for true GPU kernel time
  function benchGpuTime(mode: "full" | "qk" | "pv" | "kv_only"): number {
    const hasMethod = typeof bAny.flashAttentionCoop2GpuTime === "function";
    if (!hasMethod) return NaN;
    // Warmup
    for (let i = 0; i < warmup; i++) {
      const out = bAny.flashAttentionCoop2GpuTime(mode, q, k, v, T, scale, 30);
      release(out.output);
      release(out.lse);
    }
    // Timed runs — gpuTimeUs is already GPU-only microseconds
    const times: number[] = [];
    for (let i = 0; i < iters; i++) {
      const out = bAny.flashAttentionCoop2GpuTime(mode, q, k, v, T, scale, 30);
      times.push(out.gpuTimeUs / 1000); // convert µs → ms for consistency
      release(out.output);
      release(out.lse);
    }
    return median(times);
  }

  const runDefault = () => bench(() => {
    const out = bAny.flashAttention(q, k, v, T, scale, 30);
    return [out.output, out.lse];
  });
  const runDirect = () => bench(() => {
    const out = bAny.flashAttentionCoop2(q, k, v, T, scale, 30);
    return [out.output, out.lse];
  });
  let msDefault = 0;
  let msDirect = 0;
  let debugDefault: string | undefined;
  let debugDirect: string | undefined;
  const mkDbg = () => {
    const d = typeof bAny.getLastFlashDispatchDebug === "function" ? bAny.getLastFlashDispatchDebug() : null;
    if (!d) return undefined;
    return `${d.path}:${d.kernelName}${d.scope ? `:${d.scope}` : ""}`;
  };
  if (order === "default-first") {
    msDefault = runDefault();
    debugDefault = mkDbg();
    msDirect = runDirect();
    debugDirect = mkDbg();
  } else {
    msDirect = runDirect();
    debugDirect = mkDbg();
    msDefault = runDefault();
    debugDefault = mkDbg();
  }
  const msQK = bench(() => {
    const out = bAny.flashAttentionCoop2Probe("qk", q, k, v, T, scale, 30);
    return [out.output, out.lse];
  });
  const debugQK = mkDbg();
  const msPV = bench(() => {
    const out = bAny.flashAttentionCoop2Probe("pv", q, k, v, T, scale, 30);
    return [out.output, out.lse];
  });
  const debugPV = mkDbg();
  const msKVOnly = bench(() => {
    const out = bAny.flashAttentionCoop2Probe("kv_only", q, k, v, T, scale, 30);
    return [out.output, out.lse];
  });
  const debugKVOnly = mkDbg();

  // GPU timestamp measurements
  let gpuFull: number | undefined;
  let gpuQK: number | undefined;
  let gpuPV: number | undefined;
  let gpuKVOnly: number | undefined;
  if (useGpuTime) {
    gpuFull = benchGpuTime("full");
    gpuQK = benchGpuTime("qk");
    gpuPV = benchGpuTime("pv");
    gpuKVOnly = benchGpuTime("kv_only");
  }

  return {
    msDefault, msDirect, msQK, msPV, msKVOnly,
    gpuFull, gpuQK, gpuPV, gpuKVOnly,
    debugDefault, debugDirect, debugQK, debugPV, debugKVOnly,
  };
}

function summarize(name: string, values: number[]): string {
  const valid = values.filter(v => !isNaN(v));
  if (valid.length === 0) return `${name}: (no data)`;
  const med = median(valid);
  const p90 = percentile(valid, 90);
  const tmean = trimmedMean(valid);
  const best = Math.min(...valid);
  const worst = Math.max(...valid);
  return `${name}: med=${med.toFixed(6)} p90=${p90.toFixed(6)} tmean=${tmean.toFixed(6)} best=${best.toFixed(6)} worst=${worst.toFixed(6)}`;
}

function main(): void {
  const opts = parseArgs();
  const b = makeBackend();
  const bAny = b as any;
  const release: ((td: TensorData) => void) =
    typeof bAny.releaseGpuTensor === "function" ? bAny.releaseGpuTensor.bind(bAny) : (() => {});
  const BH = 16, T = 512, D = 64;
  const q = b.randn([BH, T, D]);
  const k = b.randn([BH, T, D]);
  const v = b.randn([BH, T, D]);

  if (opts.gpuTime) {
    console.error("GPU timestamp mode enabled (Vulkan vkCmdWriteTimestamp)");
  }

  const rows: RepeatRow[] = [];
  for (let r = 0; r < opts.repeats; r++) {
    const orderForRepeat =
      opts.order === "alternate"
        ? ((r & 1) === 0 ? "default-first" : "direct-first")
        : opts.order;
    const row = runOneRepeat(bAny, q, k, v, opts.iters, opts.warmup, orderForRepeat, opts.gpuTime);
    rows.push(row);
    let line =
      `[repeat ${r + 1}/${opts.repeats}] order=${orderForRepeat} default=${row.msDefault.toFixed(6)} ` +
      `direct=${row.msDirect.toFixed(6)} qk=${row.msQK.toFixed(6)} pv=${row.msPV.toFixed(6)} kv_only=${row.msKVOnly.toFixed(6)}`;
    if (row.gpuFull !== undefined) {
      line += ` | GPU: full=${row.gpuFull.toFixed(6)} qk=${row.gpuQK!.toFixed(6)} pv=${row.gpuPV!.toFixed(6)} kv_only=${row.gpuKVOnly!.toFixed(6)}`;
    }
    line += `${row.debugDefault ? ` defaultKernel=${row.debugDefault}` : ""}`;
    line += `${row.debugDirect ? ` directKernel=${row.debugDirect}` : ""}`;
    console.error(line);
  }

  const defaults = rows.map(r => r.msDefault);
  const directs = rows.map(r => r.msDirect);
  const qks = rows.map(r => r.msQK);
  const pvs = rows.map(r => r.msPV);
  const kvonlys = rows.map(r => r.msKVOnly);
  const deltas = rows.map(r => r.msDefault - r.msDirect);

  console.log(`repeats=${opts.repeats} iters=${opts.iters} warmup=${opts.warmup} order=${opts.order} gpuTime=${opts.gpuTime}`);
  console.log("");
  console.log("=== Wall-clock (performance.now + syncGpu) ===");
  console.log(summarize("flashAttention", defaults));
  console.log(summarize("flashAttentionCoop2", directs));
  console.log(summarize("probe_qk", qks));
  console.log(summarize("probe_pv", pvs));
  console.log(summarize("probe_kv_only", kvonlys));
  console.log(summarize("wrapper_delta(default-direct)", deltas));

  if (opts.gpuTime) {
    const gpuFulls = rows.map(r => r.gpuFull!);
    const gpuQKs = rows.map(r => r.gpuQK!);
    const gpuPVs = rows.map(r => r.gpuPV!);
    const gpuKVOnlys = rows.map(r => r.gpuKVOnly!);
    const gpuCompute = rows.map(r => r.gpuFull! - r.gpuKVOnly!);

    console.log("");
    console.log("=== GPU timestamp (vkCmdWriteTimestamp, kernel-only) ===");
    console.log(summarize("gpu_full", gpuFulls));
    console.log(summarize("gpu_qk", gpuQKs));
    console.log(summarize("gpu_pv", gpuPVs));
    console.log(summarize("gpu_kv_only", gpuKVOnlys));
    console.log(summarize("gpu_compute_delta(full-kv_only)", gpuCompute));

    // Compare wall-clock vs GPU time
    const wallMed = median(directs);
    const gpuMed = median(gpuFulls);
    const overhead = wallMed - gpuMed;
    console.log("");
    console.log(`wall-clock median: ${wallMed.toFixed(6)} ms`);
    console.log(`gpu-time median:   ${gpuMed.toFixed(6)} ms`);
    console.log(`overhead (wall-gpu): ${overhead.toFixed(6)} ms (${((overhead / wallMed) * 100).toFixed(1)}%)`);
  }

  console.log("");
  console.log(JSON.stringify({ rows }, null, 2));

  release(q);
  release(k);
  release(v);
}

main();
