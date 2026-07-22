/**
 * optimizer-reference — AdamW checked against an independent inline reference.
 *
 * The reference is a plain, short re-implementation of decoupled-weight-decay
 * AdamW with bias correction, operating on Float32Array (so f32 rounding
 * matches the backend under test). Over 10 steps of seeded random gradients
 * the two must agree to < 1e-6.
 *
 * Also verifies:
 *   - a no-decay parameter is stepped exactly as if weightDecay were 0, while
 *     a decaying parameter genuinely differs (decay is actually skipped);
 *   - the CLI's no-decay name set matches real parameter names — specifically
 *     that collectParamEntries yields "lmHead" and NOT "lmHead.weight" (the
 *     historical bug where the head silently received weight decay).
 */
import { describe, it, expect } from "vitest";
import { CpuRefBackend } from "@alpha/tensor";
import { AdamW } from "@alpha/train";
import { SeededRng, type ModelConfig, type TensorData } from "@alpha/core";
import { initGPT, collectParamEntries } from "@alpha/model";

const B = new CpuRefBackend();

interface AdamCfg { lr: number; beta1: number; beta2: number; eps: number; weightDecay: number }

/** Independent inline AdamW reference (decoupled weight decay + bias correction).
 *  Runs on Float32Array so f32 rounding matches the optimizer under test. */
function refAdamW(
  p: Float32Array, grads: Float32Array[], cfg: AdamCfg, useDecay: boolean,
): Float32Array {
  const { lr, beta1, beta2, eps, weightDecay } = cfg;
  const param = Float32Array.from(p);
  const m = new Float32Array(param.length);
  const v = new Float32Array(param.length);
  let b1p = 1, b2p = 1;
  for (const g of grads) {
    b1p *= beta1;
    b2p *= beta2;
    const bc1 = 1 - b1p;
    const bc2 = 1 - b2p;
    const wd = useDecay ? weightDecay : 0;
    for (let i = 0; i < param.length; i++) {
      const gi = g[i];
      if (wd > 0) param[i] -= lr * wd * param[i];
      m[i] = beta1 * m[i] + (1 - beta1) * gi;
      v[i] = beta2 * v[i] + (1 - beta2) * gi * gi;
      const mHat = m[i] / bc1;
      const vHat = v[i] / bc2;
      param[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
    }
  }
  return param;
}

function seededGradSeq(seed: number, n: number, steps: number): Float32Array[] {
  const r = new SeededRng(seed);
  const seq: Float32Array[] = [];
  for (let s = 0; s < steps; s++) {
    const g = new Float32Array(n);
    for (let i = 0; i < n; i++) g[i] = (r.next() * 2 - 1) * 0.5;
    seq.push(g);
  }
  return seq;
}

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

describe("AdamW vs inline reference", () => {
  const cfg: AdamCfg = { lr: 3e-4, beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0.1 };
  const STEPS = 10;
  const N = 6;

  it("10 steps of seeded grads: max abs diff < 1e-6 (with decay)", () => {
    const p0 = Float32Array.from(seededGradSeq(1, N, 1)[0].map((x) => x * 3 + 0.1));
    const grads = seededGradSeq(2, N, STEPS);

    const opt = new AdamW(B, cfg);
    const param: TensorData = { shape: [N], dtype: "f32", data: Float32Array.from(p0) };
    const params = new Map<string, TensorData>([["w", param]]);
    for (const g of grads) {
      opt.step(params, new Map([["w", { shape: [N], dtype: "f32", data: Float32Array.from(g) }]]));
    }

    const ref = refAdamW(p0, grads, cfg, true);
    expect(maxAbsDiff(param.data as Float32Array, ref)).toBeLessThan(1e-6);
  });

  it("noDecayNames: skipped param == weightDecay-0; decaying param differs", () => {
    const p0 = Float32Array.from(seededGradSeq(3, N, 1)[0].map((x) => x * 3 + 0.5));
    const grads = seededGradSeq(4, N, STEPS);

    // Optimizer with weight decay ON but "b" excluded from decay.
    const opt = new AdamW(B, { ...cfg, noDecayNames: new Set(["b"]) });
    const decayParam: TensorData = { shape: [N], dtype: "f32", data: Float32Array.from(p0) };
    const noDecayParam: TensorData = { shape: [N], dtype: "f32", data: Float32Array.from(p0) };
    const params = new Map<string, TensorData>([["a", decayParam], ["b", noDecayParam]]);
    for (const g of grads) {
      const gm = new Map([
        ["a", { shape: [N], dtype: "f32", data: Float32Array.from(g) } as TensorData],
        ["b", { shape: [N], dtype: "f32", data: Float32Array.from(g) } as TensorData],
      ]);
      opt.step(params, gm);
    }

    const refDecay = refAdamW(p0, grads, cfg, true);
    const refNoDecay = refAdamW(p0, grads, cfg, false);

    // No-decay param must match the weightDecay-0 reference exactly.
    expect(maxAbsDiff(noDecayParam.data as Float32Array, refNoDecay)).toBeLessThan(1e-6);
    // Decay param matches the with-decay reference.
    expect(maxAbsDiff(decayParam.data as Float32Array, refDecay)).toBeLessThan(1e-6);
    // And decay genuinely changed something: the two references differ.
    expect(maxAbsDiff(refDecay, refNoDecay)).toBeGreaterThan(1e-6);
  });
});

describe("no-decay name set matches real parameter names", () => {
  const config: ModelConfig = {
    vocabSize: 17, blockSize: 8, nLayer: 2, nEmbd: 16, nHead: 2, dropout: 0, ffnActivation: "swiglu",
  };

  it("collectParamEntries has 'lmHead' and not 'lmHead.weight'", () => {
    const params = initGPT(config, B, new SeededRng(1));
    const names = new Set(collectParamEntries(params).map(([n]) => n));
    expect(names.has("lmHead")).toBe(true);
    expect(names.has("lmHead.weight")).toBe(false);
  });

  it("every CLI no-decay name is an actual parameter name", () => {
    const params = initGPT(config, B, new SeededRng(1));
    const names = new Set(collectParamEntries(params).map(([n]) => n));

    // Replicate the CLI's no-decay construction WITHOUT importing from apps/.
    // (apps/cli/src/commands/train.ts adds these names; if any name doesn't
    //  exist in collectParamEntries it silently gets weight decay — the bug.)
    const noDecay = new Set<string>();
    noDecay.add("wte");
    noDecay.add("wpe");
    noDecay.add("lnF.weight");
    noDecay.add("lnF.bias");
    noDecay.add("lmHead");
    for (let i = 0; i < config.nLayer; i++) {
      noDecay.add(`layer.${i}.ln1.weight`);
      noDecay.add(`layer.${i}.ln1.bias`);
      noDecay.add(`layer.${i}.ln2.weight`);
      noDecay.add(`layer.${i}.ln2.bias`);
    }

    for (const n of noDecay) {
      expect(names.has(n), `no-decay name "${n}" is not a real parameter`).toBe(true);
    }
  });
});
