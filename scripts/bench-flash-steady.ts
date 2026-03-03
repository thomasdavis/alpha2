/**
 * bench-flash-steady.ts
 *
 * Steady-state GPU timestamp benchmark for flash attention.
 * Measures true per-kernel GPU time using multi-dispatch command buffers
 * with warmup, eliminating cold-cache artifacts.
 *
 * Usage:
 *   npx tsx scripts/bench-flash-steady.ts --iters=100 --warmup=10 --repeats=5
 *
 * Env toggles (examples):
 *   HELIOS_FLASH_COOP2_QT=2
 *   HELIOS_FLASH_COOP2_SCOPE=workgroup
 *   HELIOS_FLASH_COOP2_LS=128
 *   HELIOS_FLASH_COOP2_F16_INPUT=1
 */
import { HeliosBackend } from "@alpha/helios";
import type { TensorData } from "@alpha/core";

interface CliOpts {
  repeats: number;
  iters: number;
  warmup: number;
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
    repeats: Math.max(1, parseInt(kv.repeats ?? "5", 10)),
    iters: Math.max(1, parseInt(kv.iters ?? "100", 10)),
    warmup: Math.max(0, parseInt(kv.warmup ?? "10", 10)),
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

type ColdRow = {
  scalarUs: number;
  coop2Us: number;
  coop2QKUs: number;
  coop2QKSoftmaxUs: number;
  coop2PVUs: number;
  coop2KVOnlyUs: number;
  coop2KVSynthUs: number;
  coop2PerElemOnlyUs: number;
};

type SteadyRow = ColdRow; // same shape, different measurement regime

function makeBackend(): HeliosBackend {
  const b = new HeliosBackend();
  (b as any).setMinGpuSize?.(0);
  return b;
}

function measureGpuTime(
  bAny: any,
  release: (td: TensorData) => void,
  q: TensorData, k: TensorData, v: TensorData,
  T: number, scale: number, softCap: number,
  iters: number, warmup: number,
): ColdRow {
  // Scalar full
  const scalarOut = bAny.flashAttentionGpuTime(q, k, v, T, scale, softCap, iters, warmup);
  const scalarUs = scalarOut.gpuTimeUs;
  release(scalarOut.output);
  release(scalarOut.lse);

  // Coop2 full
  const coop2Out = bAny.flashAttentionCoop2GpuTime("full", q, k, v, T, scale, softCap, iters, warmup);
  const coop2Us = coop2Out.gpuTimeUs;
  release(coop2Out.output);
  release(coop2Out.lse);

  // Coop2 probes
  const probes = ["qk", "qk_softmax", "pv", "kv_only", "kv_synth", "per_elem_only"] as const;
  const probeUs: Record<string, number> = {};
  for (const mode of probes) {
    const out = bAny.flashAttentionCoop2GpuTime(mode, q, k, v, T, scale, softCap, iters, warmup);
    probeUs[mode] = out.gpuTimeUs;
    release(out.output);
    release(out.lse);
  }

  return {
    scalarUs,
    coop2Us,
    coop2QKUs: probeUs.qk,
    coop2QKSoftmaxUs: probeUs.qk_softmax,
    coop2PVUs: probeUs.pv,
    coop2KVOnlyUs: probeUs.kv_only,
    coop2KVSynthUs: probeUs.kv_synth,
    coop2PerElemOnlyUs: probeUs.per_elem_only,
  };
}

function fmtUs(us: number): string {
  return us.toFixed(1).padStart(8);
}

function fmtMs(us: number): string {
  return (us / 1000).toFixed(4).padStart(9);
}

function summarize(label: string, values: number[]): string {
  const valid = values.filter(v => !isNaN(v) && isFinite(v));
  if (valid.length === 0) return `${label}: (no data)`;
  const med = median(valid);
  const p10 = percentile(valid, 10);
  const p90 = percentile(valid, 90);
  return `${label.padEnd(28)} med=${fmtUs(med)} us  p10=${fmtUs(p10)} us  p90=${fmtUs(p90)} us  (${fmtMs(med)} ms)`;
}

function main(): void {
  const opts = parseArgs();
  const b = makeBackend();
  const bAny = b as any;
  const release: (td: TensorData) => void =
    typeof bAny.releaseGpuTensor === "function" ? bAny.releaseGpuTensor.bind(bAny) : (() => {});

  // Check methods exist
  if (typeof bAny.flashAttentionGpuTime !== "function") {
    console.error("ERROR: flashAttentionGpuTime not found on backend (rebuild native addon?)");
    process.exit(1);
  }
  if (typeof bAny.flashAttentionCoop2GpuTime !== "function") {
    console.error("ERROR: flashAttentionCoop2GpuTime not found on backend");
    process.exit(1);
  }

  const BH = 16, T = 512, D = 64;
  const scale = 1.0 / Math.sqrt(D);
  const softCap = 30;
  const q = b.randn([BH, T, D]);
  const k = b.randn([BH, T, D]);
  const v = b.randn([BH, T, D]);

  console.error(`bench-flash-steady: BH=${BH} T=${T} D=${D} softCap=${softCap}`);
  console.error(`  repeats=${opts.repeats} iters=${opts.iters} warmup=${opts.warmup}`);
  console.error("");

  // ── Cold measurements (iters=1, warmup=0) ──────────────────────────────
  console.error("=== Measuring cold (iters=1, warmup=0) ===");
  const coldRows: ColdRow[] = [];
  for (let r = 0; r < opts.repeats; r++) {
    const row = measureGpuTime(bAny, release, q, k, v, T, scale, softCap, 1, 0);
    coldRows.push(row);
    console.error(`  [cold ${r + 1}/${opts.repeats}] scalar=${fmtUs(row.scalarUs)} coop2=${fmtUs(row.coop2Us)} kv_only=${fmtUs(row.coop2KVOnlyUs)} kv_synth=${fmtUs(row.coop2KVSynthUs)} per_elem=${fmtUs(row.coop2PerElemOnlyUs)}`);
  }

  // ── Steady-state measurements (multi-dispatch with warmup) ─────────────
  console.error("");
  console.error(`=== Measuring steady (iters=${opts.iters}, warmup=${opts.warmup}) ===`);
  const steadyRows: SteadyRow[] = [];
  for (let r = 0; r < opts.repeats; r++) {
    const row = measureGpuTime(bAny, release, q, k, v, T, scale, softCap, opts.iters, opts.warmup);
    steadyRows.push(row);
    console.error(`  [steady ${r + 1}/${opts.repeats}] scalar=${fmtUs(row.scalarUs)} coop2=${fmtUs(row.coop2Us)} kv_only=${fmtUs(row.coop2KVOnlyUs)} kv_synth=${fmtUs(row.coop2KVSynthUs)} per_elem=${fmtUs(row.coop2PerElemOnlyUs)}`);
  }

  // ── Summary ────────────────────────────────────────────────────────────
  console.log("");
  console.log(`bench-flash-steady: BH=${BH} T=${T} D=${D} softCap=${softCap}  repeats=${opts.repeats} iters=${opts.iters} warmup=${opts.warmup}`);

  console.log("");
  console.log("=== COLD (iters=1, warmup=0) — single-dispatch GPU timestamps ===");
  console.log(summarize("scalar_full", coldRows.map(r => r.scalarUs)));
  console.log(summarize("coop2_full", coldRows.map(r => r.coop2Us)));
  console.log(summarize("coop2_qk", coldRows.map(r => r.coop2QKUs)));
  console.log(summarize("coop2_qk_softmax", coldRows.map(r => r.coop2QKSoftmaxUs)));
  console.log(summarize("coop2_pv", coldRows.map(r => r.coop2PVUs)));
  console.log(summarize("coop2_kv_only", coldRows.map(r => r.coop2KVOnlyUs)));
  console.log(summarize("coop2_kv_synth", coldRows.map(r => r.coop2KVSynthUs)));
  console.log(summarize("coop2_per_elem_only", coldRows.map(r => r.coop2PerElemOnlyUs)));

  console.log("");
  console.log(`=== STEADY (iters=${opts.iters}, warmup=${opts.warmup}) — per-kernel GPU time ===`);
  console.log(summarize("scalar_full", steadyRows.map(r => r.scalarUs)));
  console.log(summarize("coop2_full", steadyRows.map(r => r.coop2Us)));
  console.log(summarize("coop2_qk", steadyRows.map(r => r.coop2QKUs)));
  console.log(summarize("coop2_qk_softmax", steadyRows.map(r => r.coop2QKSoftmaxUs)));
  console.log(summarize("coop2_pv", steadyRows.map(r => r.coop2PVUs)));
  console.log(summarize("coop2_kv_only", steadyRows.map(r => r.coop2KVOnlyUs)));
  console.log(summarize("coop2_kv_synth", steadyRows.map(r => r.coop2KVSynthUs)));
  console.log(summarize("coop2_per_elem_only", steadyRows.map(r => r.coop2PerElemOnlyUs)));

  // ── Decomposition ──────────────────────────────────────────────────────
  console.log("");
  console.log("=== STEADY-STATE DECOMPOSITION ===");
  const sMed = (rows: SteadyRow[], fn: (r: SteadyRow) => number) => median(rows.map(fn));
  const sScalar = sMed(steadyRows, r => r.scalarUs);
  const sCoop2 = sMed(steadyRows, r => r.coop2Us);
  const sKVOnly = sMed(steadyRows, r => r.coop2KVOnlyUs);
  const sKVSynth = sMed(steadyRows, r => r.coop2KVSynthUs);
  const sPerElem = sMed(steadyRows, r => r.coop2PerElemOnlyUs);
  const sQK = sMed(steadyRows, r => r.coop2QKUs);
  const sQKSoftmax = sMed(steadyRows, r => r.coop2QKSoftmaxUs);
  const sPV = sMed(steadyRows, r => r.coop2PVUs);

  const computeDelta = sCoop2 - sKVOnly;
  const perElemDelta = sPerElem - sKVOnly;
  const qkMMA = sQK - sKVOnly;
  const softmaxOnly = sQKSoftmax - sQK;
  const pvMMA = sPV - sKVOnly;
  const kvGlobalReadCost = sCoop2 - sKVSynth;

  console.log(`  scalar_full:               ${fmtUs(sScalar)} us`);
  console.log(`  coop2_full:                ${fmtUs(sCoop2)} us`);
  console.log(`  coop2_kv_only (mem base):  ${fmtUs(sKVOnly)} us`);
  console.log(`  coop2_kv_synth (no reads): ${fmtUs(sKVSynth)} us`);
  console.log(`  coop2_compute (full-kv):   ${fmtUs(computeDelta)} us`);
  console.log(`  ---`);
  console.log(`  kv_global_read (full-synth):${fmtUs(kvGlobalReadCost)} us  (global KV read cost)`);
  console.log(`  coop2_qk_mma (qk-kv):     ${fmtUs(qkMMA)} us`);
  console.log(`  coop2_softmax (qksoft-qk): ${fmtUs(softmaxOnly)} us`);
  console.log(`  coop2_pv_mma (pv-kv):      ${fmtUs(pvMMA)} us`);
  console.log(`  ---`);
  console.log(`  per_elem_only:             ${fmtUs(sPerElem)} us`);
  console.log(`  per_elem_delta (pe-kv):    ${fmtUs(perElemDelta)} us  (PerElementOp/Reduce overhead without MMA)`);
  console.log(`  ---`);
  console.log(`  scalar vs coop2:           ${sScalar < sCoop2 ? "SCALAR WINS" : "COOP2 WINS"} by ${fmtUs(Math.abs(sScalar - sCoop2))} us (${((Math.abs(sScalar - sCoop2) / Math.max(sScalar, sCoop2)) * 100).toFixed(1)}%)`);

  // ── Cold vs Steady comparison ──────────────────────────────────────────
  console.log("");
  console.log("=== COLD vs STEADY ===");
  const cScalar = sMed(coldRows as any, (r: any) => r.scalarUs);
  const cCoop2 = sMed(coldRows as any, (r: any) => r.coop2Us);
  console.log(`  scalar: cold=${fmtUs(cScalar)} us  steady=${fmtUs(sScalar)} us  ratio=${(cScalar / sScalar).toFixed(2)}x`);
  console.log(`  coop2:  cold=${fmtUs(cCoop2)} us  steady=${fmtUs(sCoop2)} us  ratio=${(cCoop2 / sCoop2).toFixed(2)}x`);

  // ── JSON output ────────────────────────────────────────────────────────
  console.log("");
  console.log(JSON.stringify({ opts, coldRows, steadyRows }, null, 2));

  release(q);
  release(k);
  release(v);
}

main();
