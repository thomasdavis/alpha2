import { describe, it, expect } from "vitest";
import { CpuRefBackend } from "@alpha/tensor";
import { AdamW, Lion, Adafactor, SGD } from "@alpha/train";
import type { TensorData } from "@alpha/core";

function cloneTensor(td: TensorData): TensorData {
  return {
    shape: [...td.shape],
    dtype: td.dtype,
    data: new Float32Array(td.data as Float32Array),
  };
}

function maxDiff(a: Float32Array, b: Float32Array): number {
  let max = 0;
  for (let i = 0; i < a.length; i++) {
    max = Math.max(max, Math.abs(a[i] - b[i]));
  }
  return max;
}

describe("Optimizer gradScale", () => {
  it("AdamW gradScale matches pre-scaled gradients", () => {
    const B = new CpuRefBackend();
    const gradScale = 0.125;

    const paramInit = B.fromArray([1.0, -2.0, 0.5, 3.0], [4]);
    const gradRaw = B.fromArray([0.4, -0.2, 0.8, -0.5], [4]);

    const paramsA = new Map<string, TensorData>([["w", cloneTensor(paramInit)]]);
    const gradsA = new Map<string, TensorData>([["w", cloneTensor(gradRaw)]]);
    const optA = new AdamW(B, { lr: 3e-4, weightDecay: 0.01 });
    optA.step(paramsA, gradsA, gradScale);

    const paramsB = new Map<string, TensorData>([["w", cloneTensor(paramInit)]]);
    const gradsB = new Map<string, TensorData>([
      ["w", B.scale(cloneTensor(gradRaw), gradScale)],
    ]);
    const optB = new AdamW(B, { lr: 3e-4, weightDecay: 0.01 });
    optB.step(paramsB, gradsB);

    const pA = paramsA.get("w")!.data as Float32Array;
    const pB = paramsB.get("w")!.data as Float32Array;
    expect(maxDiff(pA, pB)).toBeLessThan(1e-8);
  });

  it("SGD gradScale matches pre-scaled gradients", () => {
    const B = new CpuRefBackend();
    const gradScale = 0.25;

    const paramInit = B.fromArray([2.0, -1.0, 0.75], [3]);
    const gradRaw = B.fromArray([0.5, 0.25, -1.0], [3]);

    const paramsA = new Map<string, TensorData>([["w", cloneTensor(paramInit)]]);
    const gradsA = new Map<string, TensorData>([["w", cloneTensor(gradRaw)]]);
    const sgdA = new SGD(B, 0.1);
    sgdA.step(paramsA, gradsA, gradScale);

    const paramsB = new Map<string, TensorData>([["w", cloneTensor(paramInit)]]);
    const gradsB = new Map<string, TensorData>([
      ["w", B.scale(cloneTensor(gradRaw), gradScale)],
    ]);
    const sgdB = new SGD(B, 0.1);
    sgdB.step(paramsB, gradsB);

    const pA = paramsA.get("w")!.data as Float32Array;
    const pB = paramsB.get("w")!.data as Float32Array;
    expect(maxDiff(pA, pB)).toBeLessThan(1e-8);
  });

  it("AdamW stepParamEntries matches map step", () => {
    const B = new CpuRefBackend();
    const gradScale = 0.5;

    const paramInit = B.fromArray([1.0, -2.0, 0.5, 3.0], [4]);
    const gradRaw = B.fromArray([0.4, -0.2, 0.8, -0.5], [4]);

    const paramsA = new Map<string, TensorData>([["w", cloneTensor(paramInit)]]);
    const gradsA = new Map<string, TensorData>([["w", cloneTensor(gradRaw)]]);
    const optA = new AdamW(B, { lr: 3e-4, weightDecay: 0.01 });
    optA.step(paramsA, gradsA, gradScale);

    const paramsB = new Map<string, TensorData>([["w", cloneTensor(paramInit)]]);
    const varEntries: readonly [string, { data: TensorData; grad: TensorData | null }][] = [
      ["w", { data: paramsB.get("w")!, grad: cloneTensor(gradRaw) }],
    ];
    const optB = new AdamW(B, { lr: 3e-4, weightDecay: 0.01 });
    (optB as any).stepParamEntries(varEntries, gradScale);

    const pA = paramsA.get("w")!.data as Float32Array;
    const pB = paramsB.get("w")!.data as Float32Array;
    expect(maxDiff(pA, pB)).toBeLessThan(1e-8);
  });

  it("SGD stepParamEntries matches map step", () => {
    const B = new CpuRefBackend();
    const gradScale = 0.3;

    const paramInit = B.fromArray([2.0, -1.0, 0.75], [3]);
    const gradRaw = B.fromArray([0.5, 0.25, -1.0], [3]);

    const paramsA = new Map<string, TensorData>([["w", cloneTensor(paramInit)]]);
    const gradsA = new Map<string, TensorData>([["w", cloneTensor(gradRaw)]]);
    const sgdA = new SGD(B, 0.1);
    sgdA.step(paramsA, gradsA, gradScale);

    const paramsB = new Map<string, TensorData>([["w", cloneTensor(paramInit)]]);
    const varEntries: readonly [string, { data: TensorData; grad: TensorData | null }][] = [
      ["w", { data: paramsB.get("w")!, grad: cloneTensor(gradRaw) }],
    ];
    const sgdB = new SGD(B, 0.1);
    (sgdB as any).stepParamEntries(varEntries, gradScale);

    const pA = paramsA.get("w")!.data as Float32Array;
    const pB = paramsB.get("w")!.data as Float32Array;
    expect(maxDiff(pA, pB)).toBeLessThan(1e-8);
  });

  it("Lion gradScale matches pre-scaled gradients", () => {
    const B = new CpuRefBackend();
    const gradScale = 0.2;

    const paramInit = B.fromArray([1.5, -0.75, 0.25, -2.0], [4]);
    const gradRaw = B.fromArray([0.3, -0.6, 0.9, -0.1], [4]);

    const paramsA = new Map<string, TensorData>([["w", cloneTensor(paramInit)]]);
    const gradsA = new Map<string, TensorData>([["w", cloneTensor(gradRaw)]]);
    const lionA = new Lion(B, { lr: 1e-4, beta1: 0.9, beta2: 0.99, weightDecay: 0.01 });
    lionA.step(paramsA, gradsA, gradScale);

    const paramsB = new Map<string, TensorData>([["w", cloneTensor(paramInit)]]);
    const gradsB = new Map<string, TensorData>([
      ["w", B.scale(cloneTensor(gradRaw), gradScale)],
    ]);
    const lionB = new Lion(B, { lr: 1e-4, beta1: 0.9, beta2: 0.99, weightDecay: 0.01 });
    lionB.step(paramsB, gradsB);

    const pA = paramsA.get("w")!.data as Float32Array;
    const pB = paramsB.get("w")!.data as Float32Array;
    expect(maxDiff(pA, pB)).toBeLessThan(1e-8);
  });

  it("Lion stepParamEntries matches map step", () => {
    const B = new CpuRefBackend();
    const gradScale = 0.4;

    const paramInit = B.fromArray([2.0, -1.0, 0.75], [3]);
    const gradRaw = B.fromArray([0.5, 0.25, -1.0], [3]);

    const paramsA = new Map<string, TensorData>([["w", cloneTensor(paramInit)]]);
    const gradsA = new Map<string, TensorData>([["w", cloneTensor(gradRaw)]]);
    const lionA = new Lion(B, { lr: 1e-4, beta1: 0.9, beta2: 0.99, weightDecay: 0.01 });
    lionA.step(paramsA, gradsA, gradScale);

    const paramsB = new Map<string, TensorData>([["w", cloneTensor(paramInit)]]);
    const varEntries: readonly [string, { data: TensorData; grad: TensorData | null }][] = [
      ["w", { data: paramsB.get("w")!, grad: cloneTensor(gradRaw) }],
    ];
    const lionB = new Lion(B, { lr: 1e-4, beta1: 0.9, beta2: 0.99, weightDecay: 0.01 });
    (lionB as any).stepParamEntries(varEntries, gradScale);

    const pA = paramsA.get("w")!.data as Float32Array;
    const pB = paramsB.get("w")!.data as Float32Array;
    expect(maxDiff(pA, pB)).toBeLessThan(1e-8);
  });

  it("Adafactor gradScale matches pre-scaled gradients", () => {
    const B = new CpuRefBackend();
    const gradScale = 0.3;

    // 2D tensor exercises factored Adafactor path.
    const paramInit = B.fromArray([
      1.0, -2.0, 0.5,
      3.0, -1.0, 2.0,
    ], [2, 3]);
    const gradRaw = B.fromArray([
      0.4, -0.2, 0.8,
      -0.5, 0.1, 0.3,
    ], [2, 3]);

    const cfg = { lr: 1e-3, beta2: 0.999, eps: 1e-30, clipThreshold: 1e9, weightDecay: 0.01 };

    const paramsA = new Map<string, TensorData>([["w", cloneTensor(paramInit)]]);
    const gradsA = new Map<string, TensorData>([["w", cloneTensor(gradRaw)]]);
    const optA = new Adafactor(B, cfg);
    optA.step(paramsA, gradsA, gradScale);

    const paramsB = new Map<string, TensorData>([["w", cloneTensor(paramInit)]]);
    const gradsB = new Map<string, TensorData>([
      ["w", B.scale(cloneTensor(gradRaw), gradScale)],
    ]);
    const optB = new Adafactor(B, cfg);
    optB.step(paramsB, gradsB);

    const pA = paramsA.get("w")!.data as Float32Array;
    const pB = paramsB.get("w")!.data as Float32Array;
    expect(maxDiff(pA, pB)).toBeLessThan(1e-8);
  });

  it("Adafactor stepParamEntries matches map step", () => {
    const B = new CpuRefBackend();
    const gradScale = 0.45;

    const paramInit = B.fromArray([
      1.0, -2.0, 0.5,
      3.0, -1.0, 2.0,
    ], [2, 3]);
    const gradRaw = B.fromArray([
      0.4, -0.2, 0.8,
      -0.5, 0.1, 0.3,
    ], [2, 3]);
    const cfg = { lr: 1e-3, beta2: 0.999, eps: 1e-30, clipThreshold: 1e9, weightDecay: 0.01 };

    const paramsA = new Map<string, TensorData>([["w", cloneTensor(paramInit)]]);
    const gradsA = new Map<string, TensorData>([["w", cloneTensor(gradRaw)]]);
    const optA = new Adafactor(B, cfg);
    optA.step(paramsA, gradsA, gradScale);

    const paramsB = new Map<string, TensorData>([["w", cloneTensor(paramInit)]]);
    const varEntries: readonly [string, { data: TensorData; grad: TensorData | null }][] = [
      ["w", { data: paramsB.get("w")!, grad: cloneTensor(gradRaw) }],
    ];
    const optB = new Adafactor(B, cfg);
    (optB as any).stepParamEntries(varEntries, gradScale);

    const pA = paramsA.get("w")!.data as Float32Array;
    const pB = paramsB.get("w")!.data as Float32Array;
    expect(maxDiff(pA, pB)).toBeLessThan(1e-8);
  });
});
