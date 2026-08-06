import { describe, it, expect } from "vitest";
import type { TensorData } from "@alpha/core";
import { CpuRefBackend } from "@alpha/tensor";
import {
  Variable, Tape, add, mul, matmul, sum, mean, softmax, crossEntropy, relu,
  gelu, checkpoint, scale, rmsNorm, residualDropoutAddRmsNorm, qkvHeadMajorRope,
} from "@alpha/autograd";

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

  it("moves single-owner gradients and clones only genuine aliases", () => {
    class CountingBackend extends CpuRefBackend {
      cloneCalls = 0;
      override clone(a: TensorData): TensorData {
        this.cloneCalls++;
        return super.clone(a);
      }
    }

    const backend = new CountingBackend();
    const tape = new Tape();
    const ctx = { tape, backend };
    const a = new Variable(backend.fromArray([1, 2, 3], [3]), true);
    const b = new Variable(backend.fromArray([4, 5, 6], [3]), true);
    const scaled = scale(ctx, add(ctx, a, b), 2);
    const loss = sum(ctx, scaled);

    tape.backward(loss, backend);

    // sum and scale each have one gradient consumer, so their buffers move.
    // add aliases its upstream gradient across two inputs: one clone remains
    // necessary and the last consumer takes ownership of the original.
    expect(backend.cloneCalls).toBe(1);
    expect(a.grad).not.toBe(b.grad);
    expect(Array.from(a.grad!.data)).toEqual([2, 2, 2]);
    expect(Array.from(b.grad!.data)).toEqual([2, 2, 2]);
  });

  it("fused residual-add RMSNorm preserves both outputs and their combined gradients", () => {
    class FusedReferenceBackend extends CpuRefBackend {
      fusedCalls = 0;

      residualAddRmsNorm(
        residual: TensorData,
        projected: TensorData,
        weight: TensorData,
        eps: number,
      ): { residual: TensorData; normalized: TensorData } {
        this.fusedCalls++;
        const residualOut = super.add(residual, projected);
        return {
          residual: residualOut,
          normalized: super.rmsNorm(residualOut, weight, eps),
        };
      }
    }

    const inputs = {
      residual: [0.2, -0.7, 1.3, 0.4, -0.2, 0.9],
      projected: [-0.1, 0.5, 0.2, -0.6, 0.8, -0.3],
      weight: [0.8, 1.1, -0.4],
    };

    const run = (backend: CpuRefBackend, fused: boolean) => {
      const tape = new Tape();
      const ctx = { tape, backend };
      const residual = new Variable(backend.fromArray(inputs.residual, [2, 3]), true);
      const projected = new Variable(backend.fromArray(inputs.projected, [2, 3]), true);
      const weight = new Variable(backend.fromArray(inputs.weight, [3]), true);
      let residualOut: Variable;
      let normalized: Variable;
      if (fused) {
        ({ residual: residualOut, normalized } = residualDropoutAddRmsNorm(
          ctx, residual, projected, weight, 1e-5, 0, true,
        ));
      } else {
        residualOut = add(ctx, residual, projected);
        normalized = rmsNorm(ctx, residualOut, weight, 1e-5);
      }
      // Both returned values contribute with different coefficients. This
      // verifies that the residual graph node receives and combines its direct
      // path and RMSNorm path before propagating to both original inputs.
      const loss = sum(ctx, add(ctx, scale(ctx, residualOut, 0.3), scale(ctx, normalized, -0.7)));
      const residualValues = Array.from(residualOut.data.data as Float32Array);
      const normalizedValues = Array.from(normalized.data.data as Float32Array);
      tape.backward(loss, backend);
      return {
        residualValues,
        normalizedValues,
        residualGrad: Array.from(residual.grad!.data as Float32Array),
        projectedGrad: Array.from(projected.grad!.data as Float32Array),
        weightGrad: Array.from(weight.grad!.data as Float32Array),
      };
    };

    const baseline = run(new CpuRefBackend(), false);
    const fusedBackend = new FusedReferenceBackend();
    const actual = run(fusedBackend, true);
    expect(fusedBackend.fusedCalls).toBe(1);
    for (const key of Object.keys(baseline) as Array<keyof typeof baseline>) {
      expect(actual[key]).toHaveLength(baseline[key].length);
      actual[key].forEach((value, index) => {
        expect(value).toBeCloseTo(baseline[key][index], 6);
      });
    }
  });

  it("fused QKV head-major RoPE preserves layout, values, and all branch gradients", () => {
    class FusedQkvReferenceBackend extends CpuRefBackend {
      forwardCalls = 0;
      backwardCalls = 0;

      qkvHeadMajorRope(
        qkv: TensorData,
        cos: TensorData,
        sin: TensorData,
        batch: number,
        sequence: number,
        heads: number,
        headDim: number,
      ): [TensorData, TensorData, TensorData] {
        this.forwardCalls++;
        const modelDim = heads * headDim;
        const half = headDim / 2;
        const outShape = [batch * heads, sequence, headDim];
        const size = batch * sequence * modelDim;
        const source = qkv.data as Float32Array;
        const cData = cos.data as Float32Array;
        const sData = sin.data as Float32Array;
        const outputs = [new Float32Array(size), new Float32Array(size), new Float32Array(size)];
        for (let b = 0; b < batch; b++) for (let h = 0; h < heads; h++) {
          for (let t = 0; t < sequence; t++) {
            const src = (b * sequence + t) * 3 * modelDim + h * headDim;
            const dst = ((b * heads + h) * sequence + t) * headDim;
            for (let i = 0; i < half; i++) {
              const c = cData[t * half + i];
              const s = sData[t * half + i];
              for (let branch = 0; branch < 3; branch++) {
                const a = source[src + branch * modelDim + i];
                const bb = source[src + branch * modelDim + i + half];
                if (branch < 2) {
                  outputs[branch][dst + i] = a * c - bb * s;
                  outputs[branch][dst + i + half] = bb * c + a * s;
                } else {
                  outputs[branch][dst + i] = a;
                  outputs[branch][dst + i + half] = bb;
                }
              }
            }
          }
        }
        return [
          { shape: outShape, dtype: "f32", data: outputs[0] },
          { shape: outShape, dtype: "f32", data: outputs[1] },
          { shape: outShape, dtype: "f32", data: outputs[2] },
        ];
      }

      qkvHeadMajorRopeBackward(
        grad: TensorData,
        cos: TensorData,
        inverseSin: TensorData,
        batch: number,
        sequence: number,
        heads: number,
        headDim: number,
        which: 0 | 1 | 2,
      ): TensorData {
        this.backwardCalls++;
        const modelDim = heads * headDim;
        const half = headDim / 2;
        const out = new Float32Array(batch * sequence * 3 * modelDim);
        const g = grad.data as Float32Array;
        const cData = cos.data as Float32Array;
        const sData = inverseSin.data as Float32Array;
        for (let b = 0; b < batch; b++) for (let h = 0; h < heads; h++) {
          for (let t = 0; t < sequence; t++) {
            const src = ((b * heads + h) * sequence + t) * headDim;
            const dst = (b * sequence + t) * 3 * modelDim + which * modelDim + h * headDim;
            for (let i = 0; i < half; i++) {
              const a = g[src + i];
              const bb = g[src + i + half];
              if (which < 2) {
                const c = cData[t * half + i];
                const s = sData[t * half + i];
                out[dst + i] = a * c - bb * s;
                out[dst + i + half] = bb * c + a * s;
              } else {
                out[dst + i] = a;
                out[dst + i + half] = bb;
              }
            }
          }
        }
        return { shape: [batch * sequence, 3 * modelDim], dtype: "f32", data: out };
      }
    }

    const batch = 2, sequence = 3, heads = 2, headDim = 4;
    const modelDim = heads * headDim;
    const input = Array.from(
      { length: batch * sequence * 3 * modelDim },
      (_, i) => Math.sin(i * 0.37) * 1.7,
    );
    const run = (backend: CpuRefBackend) => {
      const tape = new Tape();
      const ctx = { tape, backend };
      const qkv = new Variable(backend.fromArray(input, [batch * sequence, 3 * modelDim]), true);
      const [q, k, v] = qkvHeadMajorRope(
        ctx, qkv, batch, sequence, heads, headDim, 10000,
      );
      const loss = sum(ctx, add(ctx, add(ctx, scale(ctx, q, 0.3), scale(ctx, k, -0.7)), scale(ctx, v, 1.1)));
      const values = [q, k, v].map((x) => Array.from(x.data.data as Float32Array));
      tape.backward(loss, backend);
      return { values, grad: Array.from(qkv.grad!.data as Float32Array) };
    };

    const baseline = run(new CpuRefBackend());
    const fusedBackend = new FusedQkvReferenceBackend();
    const actual = run(fusedBackend);
    expect(fusedBackend.forwardCalls).toBe(1);
    expect(fusedBackend.backwardCalls).toBe(3);
    for (let branch = 0; branch < 3; branch++) {
      actual.values[branch].forEach((value, index) => {
        expect(value).toBeCloseTo(baseline.values[branch][index], 6);
      });
    }
    actual.grad.forEach((value, index) => {
      expect(value).toBeCloseTo(baseline.grad[index], 6);
    });
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

  it("broadcast [3,1] + [1,4] produces correct result and gradients", () => {
    const tape = new Tape();
    const ctx = { tape, backend: B };

    // a: [3,1] = [[1],[2],[3]], b: [1,4] = [[10,20,30,40]]
    const a = makeVar([1, 2, 3], [3, 1]);
    const b = makeVar([10, 20, 30, 40], [1, 4]);
    const c = add(ctx, a, b); // [3,4]
    const loss = sum(ctx, c);

    tape.backward(loss, B);

    // c = [[11,21,31,41],[12,22,32,42],[13,23,33,43]]
    // sum = 11+21+31+41 + 12+22+32+42 + 13+23+33+43 = 324
    expect((loss.data.data as Float32Array)[0]).toBeCloseTo(324, 2);

    // da/d(a[i,0]) = 4 (broadcast along dim 1, 4 elements)
    expect(Array.from(a.grad!.data)).toEqual([4, 4, 4]);
    // db/d(b[0,j]) = 3 (broadcast along dim 0, 3 elements)
    expect(Array.from(b.grad!.data)).toEqual([3, 3, 3, 3]);
  });

  it("broadcast [3,1] → [3,4] in mean backward", () => {
    const tape = new Tape();
    const ctx = { tape, backend: B };

    // a: [3,4], mean over axis=1 keepdims → [3,1], then broadcast back
    const a = makeVar([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [3, 4]);
    const m = mean(ctx, a, 1, true); // [3,1]
    const loss = sum(ctx, m);

    tape.backward(loss, B);

    // Each element of a gets grad 1/4 (mean over 4 elements, then sum of 3)
    const ga = Array.from(a.grad!.data as Float32Array);
    for (const g of ga) expect(g).toBeCloseTo(0.25, 5);
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

  it("checkpoint matches non-checkpoint gradients", () => {
    const run = (useCheckpoint: boolean) => {
      const tape = new Tape();
      const ctx = { tape, backend: B };

      const x = makeVar([1, -2, 3, 0.5], [2, 2]);
      const w = makeVar([0.5, -1.5, 2, 1], [2, 2]);

      const forward = (innerCtx: any, input: Variable): Variable => {
        const y = mul(innerCtx, input, w);
        const z = relu(innerCtx, y);
        return add(innerCtx, z, w);
      };

      const out = useCheckpoint ? checkpoint(ctx as any, forward, x) : forward(ctx, x);
      const loss = sum(ctx, out);
      tape.backward(loss, B);

      return {
        xGrad: Array.from(x.grad!.data as Float32Array),
        wGrad: Array.from(w.grad!.data as Float32Array),
        loss: (loss.data.data as Float32Array)[0],
      };
    };

    const base = run(false);
    const chk = run(true);

    expect(chk.loss).toBeCloseTo(base.loss, 6);
    for (let i = 0; i < base.xGrad.length; i++) expect(chk.xGrad[i]).toBeCloseTo(base.xGrad[i], 6);
    for (let i = 0; i < base.wGrad.length; i++) expect(chk.wGrad[i]).toBeCloseTo(base.wGrad[i], 6);
  });
});
