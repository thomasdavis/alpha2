import { describe, it, expect } from "vitest";
import { CpuRefBackend } from "@alpha/tensor";
import { Variable, Tape, add, mul, matmul, sum, softmax, crossEntropy, relu, gelu } from "@alpha/autograd";

describe("Autograd", () => {
  const B = new CpuRefBackend();

  function makeVar(data: number[], shape: number[]): Variable {
    return new Variable(B.fromArray(data, shape), true);
  }

  it("add backward", () => {
    const tape = new Tape();
    const ctx = { tape, backend: B };

    const a = makeVar([1, 2, 3], [3]);
    const b = makeVar([4, 5, 6], [3]);
    const c = add(ctx, a, b);
    const loss = sum(ctx, c);

    tape.backward(loss, B);

    expect(a.grad).not.toBeNull();
    const ga = Array.from(a.grad!.data);
    expect(ga).toEqual([1, 1, 1]);
  });

  it("mul backward", () => {
    const tape = new Tape();
    const ctx = { tape, backend: B };

    const a = makeVar([2, 3], [2]);
    const b = makeVar([4, 5], [2]);
    const c = mul(ctx, a, b);
    const loss = sum(ctx, c);

    tape.backward(loss, B);

    // d(a*b)/da = b, d(a*b)/db = a
    expect(Array.from(a.grad!.data).map(v => Math.round(v))).toEqual([4, 5]);
    expect(Array.from(b.grad!.data).map(v => Math.round(v))).toEqual([2, 3]);
  });

  it("matmul backward", () => {
    const tape = new Tape();
    const ctx = { tape, backend: B };

    const a = makeVar([1, 2, 3, 4], [2, 2]);
    const b = makeVar([5, 6, 7, 8], [2, 2]);
    const c = matmul(ctx, a, b);
    const loss = sum(ctx, c);

    tape.backward(loss, B);

    // dL/dA = ones @ B^T, dL/dB = A^T @ ones
    expect(a.grad).not.toBeNull();
    expect(b.grad).not.toBeNull();
  });

  it("numerical gradient check for add", () => {
    const eps = 1e-4;

    const fn = (x: number, y: number): number => {
      const tape = new Tape();
      const ctx = { tape, backend: B };
      const a = makeVar([x], [1]);
      const b = makeVar([y], [1]);
      const c = add(ctx, a, b);
      const loss = sum(ctx, c);
      return (loss.data.data as Float32Array)[0];
    };

    const x0 = 3.0, y0 = 4.0;
    const numGradX = (fn(x0 + eps, y0) - fn(x0 - eps, y0)) / (2 * eps);
    const numGradY = (fn(x0, y0 + eps) - fn(x0, y0 - eps)) / (2 * eps);

    // Analytic: both should be 1.0
    expect(numGradX).toBeCloseTo(1.0, 2);
    expect(numGradY).toBeCloseTo(1.0, 2);
  });

  it("numerical gradient check for mul", () => {
    const eps = 1e-4;

    const fn = (x: number, y: number): number => {
      const tape = new Tape();
      const ctx = { tape, backend: B };
      const a = makeVar([x], [1]);
      const b = makeVar([y], [1]);
      const c = mul(ctx, a, b);
      const loss = sum(ctx, c);
      return (loss.data.data as Float32Array)[0];
    };

    const x0 = 3.0, y0 = 4.0;
    const numGradX = (fn(x0 + eps, y0) - fn(x0 - eps, y0)) / (2 * eps);
    const numGradY = (fn(x0, y0 + eps) - fn(x0, y0 - eps)) / (2 * eps);

    // d(x*y)/dx = y = 4, d(x*y)/dy = x = 3
    expect(numGradX).toBeCloseTo(4.0, 2);
    expect(numGradY).toBeCloseTo(3.0, 2);
  });

  it("cross entropy backward produces gradients", () => {
    const tape = new Tape();
    const ctx = { tape, backend: B };

    const logits = makeVar([2, 1, 0.1, 0.1, 2, 0.1], [2, 3]);
    const targets = B.fromArray([0, 1], [2], "i32");
    const loss = crossEntropy(ctx, logits, targets);

    tape.backward(loss, B);

    expect(logits.grad).not.toBeNull();
    // Gradient shape should match logits
    expect(logits.grad!.shape).toEqual([2, 3]);
  });
});
