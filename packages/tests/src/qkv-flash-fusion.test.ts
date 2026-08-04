import { describe, expect, it } from "vitest";
import type { ModelConfig, TensorData } from "@alpha/core";
import { SeededRng } from "@alpha/core";
import { CpuRefBackend } from "@alpha/tensor";
import {
  Tape,
  Variable,
  flashAttention,
  mul,
  qkvFlashAttention,
  qkvFlashAttentionTokenMajor,
  qkvHeadMajorRope,
  sum,
} from "@alpha/autograd";
import { collectParamEntries, gptForward, initGPT } from "@alpha/model";

/**
 * CPU reference for the physical Helios contracts. This deliberately avoids
 * constructing HeliosBackend, so the local correctness test cannot initialize
 * Vulkan or consume accelerator time.
 */
class ReferenceFlashBackend extends CpuRefBackend {
  qkvForwardCalls = 0;
  qkvBranchBackwardCalls = 0;
  qkvCombinedBackwardCalls = 0;
  tokenMajorForwardCalls = 0;
  tokenMajorBackwardCalls = 0;

  qkvHeadMajorRope(
    qkv: TensorData,
    cos: TensorData,
    sin: TensorData,
    batch: number,
    sequence: number,
    heads: number,
    headDim: number,
  ): [TensorData, TensorData, TensorData] {
    this.qkvForwardCalls++;
    const modelDim = heads * headDim;
    const half = headDim / 2;
    const shape = [batch * heads, sequence, headDim];
    const source = qkv.data as Float32Array;
    const cosData = cos.data as Float32Array;
    const sinData = sin.data as Float32Array;
    const outputs = [
      new Float32Array(batch * sequence * modelDim),
      new Float32Array(batch * sequence * modelDim),
      new Float32Array(batch * sequence * modelDim),
    ];
    for (let b = 0; b < batch; b++) for (let h = 0; h < heads; h++) {
      for (let t = 0; t < sequence; t++) {
        const sourceBase = (b * sequence + t) * 3 * modelDim + h * headDim;
        const outputBase = ((b * heads + h) * sequence + t) * headDim;
        for (let pair = 0; pair < half; pair++) {
          const c = cosData[t * half + pair];
          const s = sinData[t * half + pair];
          for (let branch = 0; branch < 3; branch++) {
            const a = source[sourceBase + branch * modelDim + pair];
            const bb = source[sourceBase + branch * modelDim + pair + half];
            if (branch < 2) {
              outputs[branch][outputBase + pair] = a * c - bb * s;
              outputs[branch][outputBase + pair + half] = bb * c + a * s;
            } else {
              outputs[branch][outputBase + pair] = a;
              outputs[branch][outputBase + pair + half] = bb;
            }
          }
        }
      }
    }
    return [
      { shape, dtype: "f32", data: outputs[0] },
      { shape, dtype: "f32", data: outputs[1] },
      { shape, dtype: "f32", data: outputs[2] },
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
    this.qkvBranchBackwardCalls++;
    const modelDim = heads * headDim;
    const half = headDim / 2;
    const output = new Float32Array(batch * sequence * 3 * modelDim);
    const gradData = grad.data as Float32Array;
    const cosData = cos.data as Float32Array;
    const inverseSinData = inverseSin.data as Float32Array;
    for (let b = 0; b < batch; b++) for (let h = 0; h < heads; h++) {
      for (let t = 0; t < sequence; t++) {
        const sourceBase = ((b * heads + h) * sequence + t) * headDim;
        const outputBase = (b * sequence + t) * 3 * modelDim
          + which * modelDim + h * headDim;
        for (let pair = 0; pair < half; pair++) {
          const a = gradData[sourceBase + pair];
          const bb = gradData[sourceBase + pair + half];
          if (which < 2) {
            const c = cosData[t * half + pair];
            const s = inverseSinData[t * half + pair];
            output[outputBase + pair] = a * c - bb * s;
            output[outputBase + pair + half] = bb * c + a * s;
          } else {
            output[outputBase + pair] = a;
            output[outputBase + pair + half] = bb;
          }
        }
      }
    }
    return { shape: [batch * sequence, 3 * modelDim], dtype: "f32", data: output };
  }

  qkvHeadMajorRopeBackwardCombined(
    qGrad: TensorData,
    kGrad: TensorData,
    vGrad: TensorData,
    cos: TensorData,
    inverseSin: TensorData,
    batch: number,
    sequence: number,
    heads: number,
    headDim: number,
  ): TensorData {
    this.qkvCombinedBackwardCalls++;
    const branches = [qGrad, kGrad, vGrad] as const;
    const modelDim = heads * headDim;
    const half = headDim / 2;
    const output = new Float32Array(batch * sequence * 3 * modelDim);
    const cosData = cos.data as Float32Array;
    const inverseSinData = inverseSin.data as Float32Array;
    for (let branch = 0; branch < 3; branch++) {
      const gradData = branches[branch].data as Float32Array;
      for (let b = 0; b < batch; b++) for (let h = 0; h < heads; h++) {
        for (let t = 0; t < sequence; t++) {
          const sourceBase = ((b * heads + h) * sequence + t) * headDim;
          const outputBase = (b * sequence + t) * 3 * modelDim
            + branch * modelDim + h * headDim;
          for (let pair = 0; pair < half; pair++) {
            const a = gradData[sourceBase + pair];
            const bb = gradData[sourceBase + pair + half];
            if (branch < 2) {
              const c = cosData[t * half + pair];
              const s = inverseSinData[t * half + pair];
              output[outputBase + pair] = a * c - bb * s;
              output[outputBase + pair + half] = bb * c + a * s;
            } else {
              output[outputBase + pair] = a;
              output[outputBase + pair + half] = bb;
            }
          }
        }
      }
    }
    return { shape: [batch * sequence, 3 * modelDim], dtype: "f32", data: output };
  }

  flashAttention(
    q: TensorData,
    k: TensorData,
    v: TensorData,
    sequence: number,
    attentionScale: number,
    softCap: number,
  ): { output: TensorData; lse: TensorData } {
    if (softCap !== 0) throw new Error("reference Flash Attention only supports softCap=0");
    const batchHeads = q.shape[0];
    const headDim = q.shape[2];
    const qd = q.data as Float32Array;
    const kd = k.data as Float32Array;
    const vd = v.data as Float32Array;
    const output = new Float32Array(batchHeads * sequence * headDim);
    const lse = new Float32Array(batchHeads * sequence);

    for (let bh = 0; bh < batchHeads; bh++) {
      for (let row = 0; row < sequence; row++) {
        const scores = new Float64Array(row + 1);
        let maxScore = -Infinity;
        for (let col = 0; col <= row; col++) {
          let dot = 0;
          for (let d = 0; d < headDim; d++) {
            dot += qd[(bh * sequence + row) * headDim + d]
              * kd[(bh * sequence + col) * headDim + d];
          }
          const score = dot * attentionScale;
          scores[col] = score;
          if (score > maxScore) maxScore = score;
        }
        let denominator = 0;
        for (let col = 0; col <= row; col++) {
          scores[col] = Math.exp(scores[col] - maxScore);
          denominator += scores[col];
        }
        lse[bh * sequence + row] = maxScore + Math.log(denominator);
        for (let col = 0; col <= row; col++) {
          const probability = scores[col] / denominator;
          for (let d = 0; d < headDim; d++) {
            output[(bh * sequence + row) * headDim + d] +=
              probability * vd[(bh * sequence + col) * headDim + d];
          }
        }
      }
    }
    return {
      output: { shape: [batchHeads, sequence, headDim], dtype: "f32", data: output },
      lse: { shape: [batchHeads, sequence], dtype: "f32", data: lse },
    };
  }

  flashAttentionBackward(
    q: TensorData,
    k: TensorData,
    v: TensorData,
    _output: TensorData,
    outputGrad: TensorData,
    _lse: TensorData,
    sequence: number,
    attentionScale: number,
    softCap: number,
  ): { dQ: TensorData; dK: TensorData; dV: TensorData } {
    if (softCap !== 0) throw new Error("reference Flash Attention only supports softCap=0");
    const batchHeads = q.shape[0];
    const headDim = q.shape[2];
    const qd = q.data as Float32Array;
    const kd = k.data as Float32Array;
    const vd = v.data as Float32Array;
    const gd = outputGrad.data as Float32Array;
    const dQ = new Float32Array(qd.length);
    const dK = new Float32Array(kd.length);
    const dV = new Float32Array(vd.length);

    for (let bh = 0; bh < batchHeads; bh++) {
      for (let row = 0; row < sequence; row++) {
        const probabilities = new Float64Array(row + 1);
        const probabilityGrad = new Float64Array(row + 1);
        let maxScore = -Infinity;
        for (let col = 0; col <= row; col++) {
          let dot = 0;
          for (let d = 0; d < headDim; d++) {
            dot += qd[(bh * sequence + row) * headDim + d]
              * kd[(bh * sequence + col) * headDim + d];
          }
          probabilities[col] = dot * attentionScale;
          if (probabilities[col] > maxScore) maxScore = probabilities[col];
        }
        let denominator = 0;
        for (let col = 0; col <= row; col++) {
          probabilities[col] = Math.exp(probabilities[col] - maxScore);
          denominator += probabilities[col];
        }
        let probabilityMean = 0;
        for (let col = 0; col <= row; col++) {
          probabilities[col] /= denominator;
          let grad = 0;
          for (let d = 0; d < headDim; d++) {
            const gradIndex = (bh * sequence + row) * headDim + d;
            const valueIndex = (bh * sequence + col) * headDim + d;
            grad += gd[gradIndex] * vd[valueIndex];
            dV[valueIndex] += probabilities[col] * gd[gradIndex];
          }
          probabilityGrad[col] = grad;
          probabilityMean += probabilities[col] * grad;
        }
        for (let col = 0; col <= row; col++) {
          const scoreGrad = probabilities[col] * (probabilityGrad[col] - probabilityMean);
          for (let d = 0; d < headDim; d++) {
            const queryIndex = (bh * sequence + row) * headDim + d;
            const keyIndex = (bh * sequence + col) * headDim + d;
            dQ[queryIndex] += scoreGrad * attentionScale * kd[keyIndex];
            dK[keyIndex] += scoreGrad * attentionScale * qd[queryIndex];
          }
        }
      }
    }
    const shape = [batchHeads, sequence, headDim];
    return {
      dQ: { shape, dtype: "f32", data: dQ },
      dK: { shape, dtype: "f32", data: dK },
      dV: { shape, dtype: "f32", data: dV },
    };
  }

  flashAttentionTokenMajor(
    q: TensorData,
    k: TensorData,
    v: TensorData,
    sequence: number,
    batch: number,
    heads: number,
    attentionScale: number,
    softCap: number,
  ): { output: TensorData; lse: TensorData } {
    this.tokenMajorForwardCalls++;
    const { output: headMajor, lse } = this.flashAttention(
      q, k, v, sequence, attentionScale, softCap,
    );
    const headDim = q.shape[2];
    const source = headMajor.data as Float32Array;
    const output = new Float32Array(source.length);
    for (let b = 0; b < batch; b++) for (let t = 0; t < sequence; t++) {
      for (let h = 0; h < heads; h++) for (let d = 0; d < headDim; d++) {
        output[(b * sequence + t) * heads * headDim + h * headDim + d] =
          source[((b * heads + h) * sequence + t) * headDim + d];
      }
    }
    return {
      output: { shape: [batch * sequence, heads * headDim], dtype: "f32", data: output },
      lse,
    };
  }

  flashAttentionBackwardTokenMajor(
    q: TensorData,
    k: TensorData,
    v: TensorData,
    output: TensorData,
    outputGrad: TensorData,
    lse: TensorData,
    sequence: number,
    batch: number,
    heads: number,
    attentionScale: number,
    softCap: number,
  ): { dQ: TensorData; dK: TensorData; dV: TensorData } {
    this.tokenMajorBackwardCalls++;
    const headDim = q.shape[2];
    const toHeadMajor = (value: TensorData): TensorData => {
      const source = value.data as Float32Array;
      const converted = new Float32Array(source.length);
      for (let b = 0; b < batch; b++) for (let t = 0; t < sequence; t++) {
        for (let h = 0; h < heads; h++) for (let d = 0; d < headDim; d++) {
          converted[((b * heads + h) * sequence + t) * headDim + d] =
            source[(b * sequence + t) * heads * headDim + h * headDim + d];
        }
      }
      return { shape: [batch * heads, sequence, headDim], dtype: "f32", data: converted };
    };
    return this.flashAttentionBackward(
      q, k, v, toHeadMajor(output), toHeadMajor(outputGrad), lse,
      sequence, attentionScale, softCap,
    );
  }
}

function expectArraysClose(actual: number[], expected: number[], digits = 5): void {
  expect(actual).toHaveLength(expected.length);
  actual.forEach((value, index) => expect(value).toBeCloseTo(expected[index], digits));
}

describe("one-tape grouped-QKV Flash Attention", () => {
  it("writes token-major output and preserves the complete grouped gradient", () => {
    const batch = 2;
    const sequence = 3;
    const heads = 2;
    const headDim = 4;
    const modelDim = heads * headDim;
    const input = Array.from(
      { length: batch * sequence * 3 * modelDim },
      (_, index) => Math.sin(index * 0.11) - Math.cos(index * 0.17) * 0.2,
    );
    const tokenWeights = Array.from(
      { length: batch * sequence * modelDim }, (_, index) => (index % 7 - 3) / 5,
    );
    const headWeights = new Array<number>(tokenWeights.length);
    for (let b = 0; b < batch; b++) for (let t = 0; t < sequence; t++) {
      for (let h = 0; h < heads; h++) for (let d = 0; d < headDim; d++) {
        headWeights[((b * heads + h) * sequence + t) * headDim + d] =
          tokenWeights[(b * sequence + t) * modelDim + h * headDim + d];
      }
    }

    const run = (tokenMajor: boolean) => {
      const backend = new ReferenceFlashBackend();
      const tape = new Tape();
      const ctx = { tape, backend };
      const qkv = new Variable(
        backend.fromArray(input, [batch * sequence, 3 * modelDim]), true,
      );
      const output = tokenMajor
        ? qkvFlashAttentionTokenMajor(
          ctx, qkv, batch, sequence, heads, headDim, 10000, 1 / 2, 0,
        )
        : qkvFlashAttention(
          ctx, qkv, batch, sequence, heads, headDim, 10000, 1 / 2, 0,
        );
      const weights = new Variable(
        backend.fromArray(
          tokenMajor ? tokenWeights : headWeights,
          output.data.shape,
        ),
        false,
      );
      const loss = sum(ctx, mul(ctx, output, weights));
      const values = Array.from(output.data.data as Float32Array);
      tape.backward(loss, backend);
      return {
        backend,
        loss: (loss.data.data as Float32Array)[0],
        values,
        gradient: Array.from(qkv.grad!.data as Float32Array),
      };
    };

    const headMajor = run(false);
    const tokenMajor = run(true);
    const expectedTokenMajor = new Array<number>(headMajor.values.length);
    for (let b = 0; b < batch; b++) for (let t = 0; t < sequence; t++) {
      for (let h = 0; h < heads; h++) for (let d = 0; d < headDim; d++) {
        expectedTokenMajor[(b * sequence + t) * modelDim + h * headDim + d] =
          headMajor.values[((b * heads + h) * sequence + t) * headDim + d];
      }
    }
    expectArraysClose(tokenMajor.values, expectedTokenMajor, 6);
    expect(tokenMajor.loss).toBeCloseTo(headMajor.loss, 6);
    expectArraysClose(tokenMajor.gradient, headMajor.gradient, 6);
    expect(tokenMajor.backend.tokenMajorForwardCalls).toBe(1);
    expect(tokenMajor.backend.tokenMajorBackwardCalls).toBe(1);
    expect(tokenMajor.backend.qkvCombinedBackwardCalls).toBe(1);
  });

  it("matches the compositional output and complete grouped gradient", () => {
    const batch = 2;
    const sequence = 3;
    const heads = 2;
    const headDim = 4;
    const modelDim = heads * headDim;
    const input = Array.from(
      { length: batch * sequence * 3 * modelDim },
      (_, index) => Math.sin(index * 0.19) + Math.cos(index * 0.07) * 0.3,
    );

    const run = (combined: boolean) => {
      const backend = new ReferenceFlashBackend();
      const tape = new Tape();
      const ctx = { tape, backend };
      const qkv = new Variable(
        backend.fromArray(input, [batch * sequence, 3 * modelDim]),
        true,
      );
      let output: Variable;
      if (combined) {
        output = qkvFlashAttention(
          ctx, qkv, batch, sequence, heads, headDim, 10000, 1 / 2, 0,
        );
      } else {
        const [q, k, v] = qkvHeadMajorRope(
          ctx, qkv, batch, sequence, heads, headDim, 10000,
        );
        output = flashAttention(ctx, q, k, v, sequence, 1 / 2, 0);
      }
      const weights = new Variable(
        backend.fromArray(
          Array.from({ length: output.data.data.length }, (_, index) => (index % 9 - 4) / 7),
          output.data.shape,
        ),
        false,
      );
      const loss = sum(ctx, mul(ctx, output, weights));
      const values = Array.from(output.data.data as Float32Array);
      tape.backward(loss, backend);
      return {
        backend,
        values,
        gradient: Array.from(qkv.grad!.data as Float32Array),
        tapeEntries: tape.size,
      };
    };

    const baseline = run(false);
    const fused = run(true);
    expectArraysClose(fused.values, baseline.values, 6);
    expectArraysClose(fused.gradient, baseline.gradient, 6);
    expect(baseline.backend.qkvBranchBackwardCalls).toBe(3);
    expect(baseline.backend.qkvCombinedBackwardCalls).toBe(0);
    expect(fused.backend.qkvBranchBackwardCalls).toBe(0);
    expect(fused.backend.qkvCombinedBackwardCalls).toBe(1);
    expect(fused.tapeEntries).toBe(baseline.tapeEntries - 3);
  });

  it("preserves tiny Llama-form loss and every parameter gradient", () => {
    const config: ModelConfig = {
      vocabSize: 17,
      blockSize: 4,
      nLayer: 2,
      nEmbd: 16,
      nHead: 2,
      dropout: 0,
      ffnActivation: "swiglu",
      normType: "rmsnorm",
      posEnc: "rope",
      ropeTheta: 10000,
      tieEmbeddings: true,
    };
    const tokens: TensorData = {
      shape: [2, 4],
      dtype: "i32",
      data: Int32Array.from([1, 4, 9, 3, 2, 8, 5, 11]),
    };
    const targets: TensorData = {
      shape: [2, 4],
      dtype: "i32",
      data: Int32Array.from([4, 9, 3, 7, 8, 5, 11, 6]),
    };
    const run = (backend: CpuRefBackend) => {
      const params = initGPT(config, backend, new SeededRng(20260804));
      const tape = new Tape();
      const result = gptForward(config, params, backend, tape, tokens, targets, false);
      const loss = (result.loss!.data.data as Float32Array)[0];
      tape.backward(result.loss!, backend);
      return {
        loss,
        params: collectParamEntries(params).map(([name, variable]) => ({
          name,
          gradient: Array.from(variable.grad!.data as Float32Array),
        })),
      };
    };

    const baseline = run(new CpuRefBackend());
    const backend = new ReferenceFlashBackend();
    const fused = run(backend);
    expect(fused.loss).toBeCloseTo(baseline.loss, 6);
    expect(fused.params.map(({ name }) => name)).toEqual(baseline.params.map(({ name }) => name));
    for (let tensor = 0; tensor < baseline.params.length; tensor++) {
      expectArraysClose(fused.params[tensor].gradient, baseline.params[tensor].gradient, 4);
    }
    expect(backend.qkvForwardCalls).toBe(config.nLayer);
    expect(backend.qkvCombinedBackwardCalls).toBe(config.nLayer);
    expect(backend.qkvBranchBackwardCalls).toBe(0);
    expect(backend.tokenMajorForwardCalls).toBe(config.nLayer);
    expect(backend.tokenMajorBackwardCalls).toBe(config.nLayer);
  });
});
