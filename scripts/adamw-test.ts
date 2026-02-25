/**
 * Test: does helios AdamW actually update parameters?
 * Runs one forward+backward, applies AdamW, checks if weights changed.
 */
import { Variable, Tape } from "@alpha/autograd";
import * as ag from "@alpha/autograd";
import { shapeSize, SeededRng } from "@alpha/core";
import type { Backend, TensorData } from "@alpha/core";
import { HeliosBackend } from "@alpha/helios";
import { backendRegistry } from "@alpha/tensor";

function norm(d: Float32Array): number {
  let s = 0;
  for (let i = 0; i < d.length; i++) s += d[i] * d[i];
  return Math.sqrt(s);
}

function maxDiff(a: Float32Array, b: Float32Array): number {
  let mx = 0;
  for (let i = 0; i < a.length; i++) mx = Math.max(mx, Math.abs(a[i] - b[i]));
  return mx;
}

async function main() {
  const helios = new HeliosBackend();
  const cpu = backendRegistry.get("cpu_ref") as Backend;

  const rng = new SeededRng(42);
  const wData = new Float32Array(128 * 128);
  for (let i = 0; i < wData.length; i++) wData[i] = rng.nextGauss() * 0.02;

  const xData = new Float32Array(64 * 128);
  for (let i = 0; i < xData.length; i++) xData[i] = rng.nextGauss() * 0.1;

  console.log("=== Testing parameter update ===");
  for (const [name, B] of [["helios", helios], ["cpu", cpu]] as [string, Backend][]) {
    // Create parameter
    const wTD: TensorData = { shape: [128, 128], dtype: "f32", data: Float32Array.from(wData) };
    const W = new Variable(wTD, true);
    const wBefore = Float32Array.from(wTD.data as Float32Array);

    // Forward
    const tape = new Tape();
    const ctx = { tape, backend: B };
    const x = new Variable({ shape: [64, 128], dtype: "f32", data: Float32Array.from(xData) }, false);
    const y = ag.matmul(ctx, x, W);
    const loss = ag.sum(ctx, y);

    // Backward
    tape.backward(loss, B);
    const g = W.grad!;
    console.log(`  [${name}] grad_norm = ${norm(g.data as Float32Array).toFixed(6)}`);

    // AdamW step (manual, like the training loop does)
    const lr = 3e-4;
    const beta1 = 0.9, beta2 = 0.999, eps = 1e-8, wd = 0.01;
    const bc1 = 1 - beta1, bc2 = 1 - beta2;
    const paramData = W.data;
    const gradData = g;
    const m = B.zeros([128, 128]);
    const v = B.zeros([128, 128]);

    if (B.adamwStep) {
      console.log(`  [${name}] Using GPU adamwStep`);
      B.adamwStep(paramData, gradData, m, v, lr, beta1, beta2, eps, wd, bc1, bc2);
      // Force flush if available
      if ("flush" in B) (B as any).flush();
    } else {
      console.log(`  [${name}] Using CPU adamwStep fallback`);
      const pArr = paramData.data as Float32Array;
      const gArr = gradData.data as Float32Array;
      const mArr = m.data as Float32Array;
      const vArr = v.data as Float32Array;
      for (let i = 0; i < pArr.length; i++) {
        pArr[i] -= lr * wd * pArr[i];
        mArr[i] = beta1 * mArr[i] + (1 - beta1) * gArr[i];
        vArr[i] = beta2 * vArr[i] + (1 - beta2) * gArr[i] * gArr[i];
        const mHat = mArr[i] / bc1;
        const vHat = vArr[i] / bc2;
        pArr[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
      }
    }

    // Check if weights changed
    const wAfter = Float32Array.from(paramData.data as Float32Array);
    const diff = maxDiff(wBefore, wAfter);
    const wNormBefore = norm(wBefore);
    const wNormAfter = norm(wAfter);
    console.log(`  [${name}] W norm before: ${wNormBefore.toFixed(6)}, after: ${wNormAfter.toFixed(6)}`);
    console.log(`  [${name}] max param change: ${diff.toExponential(4)}`);
    console.log(`  [${name}] W[0] before: ${wBefore[0].toFixed(8)}, after: ${wAfter[0].toFixed(8)}`);
    console.log();
  }

  // Now test TWO consecutive forward+backward+AdamW steps with helios
  console.log("=== Testing 2 consecutive steps (helios) ===");
  {
    const B = helios;
    const wTD: TensorData = { shape: [128, 128], dtype: "f32", data: Float32Array.from(wData) };
    const W = new Variable(wTD, true);
    const m = B.zeros([128, 128]);
    const v = B.zeros([128, 128]);
    const lr = 3e-4;
    const beta1 = 0.9, beta2 = 0.999, eps = 1e-8, wd = 0.01;

    for (let step = 0; step < 5; step++) {
      const bc1 = 1 - Math.pow(beta1, step + 1);
      const bc2 = 1 - Math.pow(beta2, step + 1);

      const wBefore = Float32Array.from(W.data.data as Float32Array);

      const tape = new Tape();
      const ctx = { tape, backend: B };
      const x = new Variable({ shape: [64, 128], dtype: "f32", data: Float32Array.from(xData) }, false);
      const y = ag.matmul(ctx, x, W);
      const loss = ag.sum(ctx, y);
      tape.backward(loss, B);

      const g = W.grad!;
      const gNorm = norm(g.data as Float32Array);

      // AdamW
      B.adamwStep!(W.data, g, m, v, lr, beta1, beta2, eps, wd, bc1, bc2);
      (B as any).flush();

      // Check weight change
      const wAfter = Float32Array.from(W.data.data as Float32Array);
      const diff = maxDiff(wBefore, wAfter);

      console.log(`  step ${step + 1}: loss=${((loss.data.data as Float32Array)[0]).toFixed(6)}, grad_norm=${gNorm.toFixed(4)}, param_change=${diff.toExponential(4)}, W[0]=${wAfter[0].toFixed(8)}`);

      // Clear for next step
      W.grad = null;
      tape.clear();
    }
  }

  console.log("\nDone.");
}

main().catch(console.error);
