import { describe, expect, it } from "vitest";
import { SeededRng, type ModelConfig, type TensorData } from "@alpha/core";
import { CpuRefBackend } from "@alpha/tensor";
import { Tape, Variable } from "@alpha/autograd";
import { blockPostSiteId, collectParamEntries, exactFinalDecode, gptForward, initGPT } from "@alpha/model";
import { createSession, decodeStep, prefill, prepareInferenceWeights } from "@alpha/inference";

const backend = new CpuRefBackend();
const config: ModelConfig = {
  vocabSize: 19,
  blockSize: 6,
  nLayer: 2,
  nEmbd: 8,
  nHead: 2,
  dropout: 0,
  ffnActivation: "swiglu",
  ffnDim: 16,
  normType: "rmsnorm",
  posEnc: "rope",
  tieEmbeddings: true,
};

function tokens(): TensorData {
  return { shape: [1, 4], dtype: "i32", data: new Int32Array([1, 4, 7, 3]) };
}

describe("native lens model seams", () => {
  it("captures ordered post-block sites and exact final decode is identical", () => {
    const params = initGPT(config, backend, new SeededRng(17));
    const tape = new Tape();
    const sites = new Set([blockPostSiteId(0), blockPostSiteId(1)]);
    const forward = gptForward(config, params, backend, tape, tokens(), undefined, false, false, false,
      undefined, undefined, undefined, { kind: "cross_entropy" },
      { requestedSites: sites, captureTarget: true });

    expect([...forward.sites!.keys()]).toEqual(["block.000.post", "block.001.post"]);
    expect(forward.sites!.get("block.000.post")!.data.shape).toEqual([1, 4, 8]);
    expect(forward.target).toBe(forward.sites!.get("block.001.post"));

    const decodeTape = new Tape();
    const decoded = exactFinalDecode(
      config,
      params,
      backend,
      decodeTape,
      new Variable(backend.clone(forward.target!.data), false),
    );
    expect(backend.allClose(decoded.data, forward.logits.data, 0, 0)).toBe(true);
  });

  it("native VJP agrees with a central finite difference at a post-block site", () => {
    const params = initGPT(config, backend, new SeededRng(23));
    const siteId = blockPostSiteId(0);
    const tape = new Tape();
    const forward = gptForward(config, params, backend, tape, tokens(), undefined, false, false, false,
      undefined, undefined, undefined, { kind: "cross_entropy" },
      { requestedSites: new Set([siteId]), captureTarget: true });
    const source = forward.sites!.get(siteId)!;
    const direction = new Float32Array(1 * 4 * 8);
    for (let i = 0; i < direction.length; i++) direction[i] = Math.sin(i + 1) * 0.1;
    const targetCotangent = new Float32Array(1 * 4 * 8);
    for (let i = 0; i < targetCotangent.length; i++) targetCotangent[i] = Math.cos(i + 0.5) * 0.07;
    let sourceGradient: Float32Array | undefined;
    tape.backward(
      forward.target!,
      backend,
      undefined,
      { shape: [1, 4, 8], dtype: "f32", data: targetCotangent },
      {
        onGradient: (variable, gradient) => {
          if (variable.id === source.id) sourceGradient = new Float32Array(gradient.data as Float32Array);
        },
      },
    );
    expect(sourceGradient).toBeDefined();
    const analytic = dot(sourceGradient!, direction);

    const scalarAt = (scale: number): number => {
      const perturbation = Float32Array.from(direction, (value) => value * scale);
      const localTape = new Tape();
      const result = gptForward(config, params, backend, localTape, tokens(), undefined, false, false, false,
        undefined, undefined, undefined, { kind: "cross_entropy" },
        {
          requestedSites: new Set([siteId]),
          captureTarget: true,
          sitePerturbations: new Map([[siteId, { shape: [1, 4, 8], dtype: "f32", data: perturbation }]]),
        });
      return dot(result.target!.data.data as Float32Array, targetCotangent);
    };
    const epsilon = 1e-3;
    const numeric = (scalarAt(epsilon) - scalarAt(-epsilon)) / (2 * epsilon);
    expect(analytic).toBeCloseTo(numeric, 3);
  });

  it("KV-cache inference captures the same post-block sites without recomputing the prefix", () => {
    const params = initGPT(config, backend, new SeededRng(31));
    const checkpointParams: Record<string, { shape: number[]; data: Float32Array }> = {};
    for (const [name, variable] of collectParamEntries(params)) {
      checkpointParams[name] = { shape: [...variable.data.shape], data: variable.data.data as Float32Array };
    }
    const requested = new Set([blockPostSiteId(0), blockPostSiteId(1)]);
    const nativeTape = new Tape();
    const native = gptForward(config, params, backend, nativeTape, tokens(), undefined, false, false, false,
      undefined, undefined, undefined, { kind: "cross_entropy" }, { requestedSites: requested, captureTarget: true });

    const weights = prepareInferenceWeights(config, checkpointParams);
    const session = createSession(weights);
    const prefillCapture = { requestedSites: requested, sites: new Map<string, Float32Array>() };
    const promptIds = tokens().data as Int32Array;
    const logits = prefill(weights, session, promptIds, prefillCapture);
    for (const siteId of requested) {
      const expected = native.sites!.get(siteId)!.data.data as Float32Array;
      const observed = prefillCapture.sites.get(siteId)!;
      expect(observed.length).toBe(expected.length);
      for (let index = 0; index < observed.length; index++) expect(observed[index]).toBeCloseTo(expected[index], 4);
    }
    const next = logits.indexOf(Math.max(...logits));
    const stepCapture = { requestedSites: requested, sites: new Map<string, Float32Array>() };
    decodeStep(weights, session, next, promptIds.length, stepCapture);

    const extendedIds = new Int32Array([...promptIds, next]);
    const extendedTape = new Tape();
    const extended = gptForward(
      { ...config, blockSize: 6 },
      params,
      backend,
      extendedTape,
      { shape: [1, extendedIds.length], dtype: "i32", data: extendedIds },
      undefined,
      false,
      false,
      false,
      undefined,
      undefined,
      undefined,
      { kind: "cross_entropy" },
      { requestedSites: requested, captureTarget: true },
    );
    for (const siteId of requested) {
      const full = extended.sites!.get(siteId)!.data.data as Float32Array;
      const expectedLast = full.subarray((extendedIds.length - 1) * config.nEmbd);
      const observed = stepCapture.sites.get(siteId)!;
      for (let index = 0; index < observed.length; index++) expect(observed[index]).toBeCloseTo(expectedLast[index], 4);
    }
  });
});

function dot(a: Float32Array, b: Float32Array): number {
  let total = 0;
  for (let i = 0; i < a.length; i++) total += a[i] * b[i];
  return total;
}
