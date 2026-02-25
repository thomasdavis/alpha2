/**
 * Compare autograd gradients between helios and cpu_ref backends
 * using the ACTUAL GPT model forward+backward pass.
 */
import { Variable, Tape } from "@alpha/autograd";
import { shapeSize, SeededRng } from "@alpha/core";
import type { Backend, TensorData } from "@alpha/core";
import { initGPT, gptForward, collectParams, type GPTParams } from "@alpha/model";
import { HeliosBackend } from "@alpha/helios";
import { backendRegistry } from "@alpha/tensor";

function gradNorm(g: TensorData | null): number {
  if (!g) return 0;
  const d = g.data as Float32Array;
  let s = 0;
  for (let i = 0; i < d.length; i++) s += d[i] * d[i];
  return Math.sqrt(s);
}

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let mx = 0;
  for (let i = 0; i < a.length; i++) mx = Math.max(mx, Math.abs(a[i] - b[i]));
  return mx;
}

function relDiff(a: number, b: number): string {
  if (b === 0 && a === 0) return "0";
  return ((a - b) / Math.max(Math.abs(a), Math.abs(b))).toExponential(3);
}

async function main() {
  const helios = new HeliosBackend();
  const cpu = backendRegistry.get("cpu_ref") as Backend;
  if (!cpu) throw new Error("cpu_ref backend not found");

  // Exact training config from previous session
  const config = {
    vocabSize: 256,
    blockSize: 256,
    nLayer: 4,
    nEmbd: 128,
    nHead: 4,
  };

  const B = 4;  // batch size
  const T = 64; // sequence length

  // Create random input tokens and targets
  const rngData = new SeededRng(42);
  const tokens: TensorData = {
    shape: [B, T], dtype: "i32",
    data: Int32Array.from({ length: B * T }, () => Math.floor(Math.random() * config.vocabSize)),
  };
  const targets: TensorData = {
    shape: [B, T], dtype: "i32",
    data: Int32Array.from({ length: B * T }, () => Math.floor(Math.random() * config.vocabSize)),
  };

  // ─── Run with helios ────────────────────────────────────────────────
  console.log("=== Helios forward+backward ===");
  const paramsH = initGPT(config as any, helios, new SeededRng(1337));
  const tapeH = new Tape();
  const resultH = gptForward(config as any, paramsH, helios, tapeH, tokens, targets);
  const lossH = (resultH.loss!.data.data as Float32Array)[0];
  console.log(`  loss: ${lossH.toFixed(6)}`);

  tapeH.backward(resultH.loss!, helios);

  const paramMapH = collectParams(paramsH);
  const heliosGrads = new Map<string, { norm: number; data: Float32Array }>();
  for (const [name, variable] of paramMapH) {
    const g = variable.grad;
    const norm = gradNorm(g);
    heliosGrads.set(name, { norm, data: g ? Float32Array.from(g.data as Float32Array) : new Float32Array(0) });
  }

  // ─── Run with cpu_ref ───────────────────────────────────────────────
  console.log("\n=== CPU_ref forward+backward ===");
  const paramsC = initGPT(config as any, cpu, new SeededRng(1337));
  const tapeC = new Tape();
  const resultC = gptForward(config as any, paramsC, cpu, tapeC, tokens, targets);
  const lossC = (resultC.loss!.data.data as Float32Array)[0];
  console.log(`  loss: ${lossC.toFixed(6)}`);

  tapeC.backward(resultC.loss!, cpu);

  const paramMapC = collectParams(paramsC);
  const cpuGrads = new Map<string, { norm: number; data: Float32Array }>();
  for (const [name, variable] of paramMapC) {
    const g = variable.grad;
    const norm = gradNorm(g);
    cpuGrads.set(name, { norm, data: g ? Float32Array.from(g.data as Float32Array) : new Float32Array(0) });
  }

  // ─── Compare ────────────────────────────────────────────────────────
  console.log("\n=== Comparison ===");
  console.log(`  loss diff: ${Math.abs(lossH - lossC).toExponential(3)}`);
  console.log("");

  // Compute total grad norm
  let totalH = 0, totalC = 0;
  for (const [name] of paramMapH) {
    const h = heliosGrads.get(name)!;
    const c = cpuGrads.get(name)!;
    totalH += h.norm * h.norm;
    totalC += c.norm * c.norm;
  }
  console.log(`  total grad_norm: helios=${Math.sqrt(totalH).toFixed(6)}, cpu=${Math.sqrt(totalC).toFixed(6)}, rel_diff=${relDiff(Math.sqrt(totalH), Math.sqrt(totalC))}`);
  console.log("");

  // Per-parameter comparison
  console.log("  Per-parameter gradient comparison:");
  console.log(`  ${"name".padEnd(30)} ${"helios_norm".padStart(14)} ${"cpu_norm".padStart(14)} ${"rel_diff".padStart(12)} ${"max_abs_diff".padStart(14)}`);
  console.log("  " + "-".repeat(88));

  for (const [name] of paramMapH) {
    const h = heliosGrads.get(name)!;
    const c = cpuGrads.get(name)!;
    const rd = relDiff(h.norm, c.norm);
    const mad = h.data.length > 0 && c.data.length > 0 ? maxAbsDiff(h.data, c.data).toExponential(3) : "N/A";
    const flag = Math.abs(h.norm - c.norm) / Math.max(h.norm, c.norm, 1e-10) > 0.01 ? " <<<" : "";
    console.log(`  ${name.padEnd(30)} ${h.norm.toFixed(6).padStart(14)} ${c.norm.toFixed(6).padStart(14)} ${rd.padStart(12)} ${String(mad).padStart(14)}${flag}`);
  }

  console.log("\nDone.");
}

main().catch(console.error);
