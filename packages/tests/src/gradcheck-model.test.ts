/**
 * gradcheck-model — full tiny-GPT gradient check on cpu_ref.
 *
 * A whole-model finite-difference check: build a tiny GPT, get analytic
 * gradients for every parameter via the tape, then perturb a handful of
 * seeded-random elements per parameter tensor and confirm the numeric
 * gradient of the scalar loss matches. Runs for both the SwiGLU and GELU
 * FFN variants.
 *
 * Also guards three whole-model invariants:
 *   - no parameter has an all-zero gradient (catches dead paths such as the
 *     historical untied-lmHead bug where the head received no gradient),
 *   - the loss is finite,
 *   - init + forward + backward is deterministic (bit-identical across two
 *     runs with the same seed).
 */
import { describe, it, expect } from "vitest";
import { CpuRefBackend } from "@alpha/tensor";
import { SeededRng, type ModelConfig, type TensorData } from "@alpha/core";
import { Tape, DropoutRng } from "@alpha/autograd";
import { initGPT, gptForward, collectParamEntries } from "@alpha/model";

const B = new CpuRefBackend();

// Whole-model finite differences accumulate more f32 round-off than a single
// op, so the relative tolerance is a little looser than the per-op harness
// (2e-2/2e-3). The ABSOLUTE floor, however, must sit just above the measured
// FD noise (~7e-5 on this model): tiny-GPT parameter gradients are small
// (fc1 max|g| ≈ 1.5e-2, LN params ≈ 1e-3), so a floor of 3e-3 would swallow a
// 20% backward error outright — proven by fault injection (a ×1.2 fault in
// matmulTransposedGelu backward passed at 3e-3, fails at 3e-4).
const REL_TOL = 3e-2;
const ABS_TOL = 3e-4;
const EPS = 2e-3;
// Per parameter tensor: check the TOP_ELEMS largest-|analytic| elements (max
// relative-tolerance headroom, so proportional backward errors are visible)
// plus RND_ELEMS seeded-random extras for unbiased coverage.
const TOP_ELEMS = 3;
const RND_ELEMS = 2;

function makeBatch(config: ModelConfig, seed: number): { tokens: TensorData; targets: TensorData } {
  const [Bsz, T] = [2, config.blockSize];
  const r = new SeededRng(seed);
  const tok = new Int32Array(Bsz * T);
  const tgt = new Int32Array(Bsz * T);
  for (let i = 0; i < Bsz * T; i++) {
    tok[i] = Math.floor(r.next() * config.vocabSize);
    tgt[i] = Math.floor(r.next() * config.vocabSize);
  }
  return {
    tokens: { shape: [Bsz, T], dtype: "i32", data: tok },
    targets: { shape: [Bsz, T], dtype: "i32", data: tgt },
  };
}

function lossOf(config: ModelConfig, params: ReturnType<typeof initGPT>, batch: { tokens: TensorData; targets: TensorData }): number {
  const tape = new Tape();
  const res = gptForward(config, params, B, tape, batch.tokens, batch.targets, /*training*/ false);
  return (res.loss!.data.data as Float32Array)[0];
}

function runGradcheck(config: ModelConfig) {
  const params = initGPT(config, B, new SeededRng(20260722));
  const batch = makeBatch(config, 99);

  // Analytic gradients for every parameter.
  const tape = new Tape();
  const res = gptForward(config, params, B, tape, batch.tokens, batch.targets, false);
  const loss = res.loss!;
  expect(Number.isFinite((loss.data.data as Float32Array)[0])).toBe(true);
  tape.backward(loss, B);

  const entries = collectParamEntries(params);

  // Invariant: no dead parameter (all-zero gradient).
  for (const [name, v] of entries) {
    expect(v.grad, `${name} has null gradient`).not.toBeNull();
    const g = v.grad!.data as Float32Array;
    let anyNonZero = false;
    for (let i = 0; i < g.length; i++) {
      if (!Number.isFinite(g[i])) throw new Error(`${name}[${i}] gradient is non-finite`);
      if (Math.abs(g[i]) > 1e-9) { anyNonZero = true; break; }
    }
    expect(anyNonZero, `${name} has an all-zero gradient (dead path)`).toBe(true);
  }

  // Numeric check: top-|g| elements + seeded-random extras per tensor.
  let worst = "";
  for (let e = 0; e < entries.length; e++) {
    const [name, v] = entries[e];
    const data = v.data.data as Float32Array;
    const analytic = v.grad!.data as Float32Array;
    const size = data.length;
    const order = Array.from(analytic.keys())
      .sort((p, q) => Math.abs(analytic[q]) - Math.abs(analytic[p]));
    const picks = new Set<number>(order.slice(0, Math.min(TOP_ELEMS, size)));
    const idxRng = new SeededRng(1000 + e);
    while (picks.size < Math.min(TOP_ELEMS + RND_ELEMS, size)) {
      picks.add(Math.floor(idxRng.next() * size));
    }
    for (const j of picks) {
      const x0 = data[j];
      const h = EPS * Math.max(1, Math.abs(x0));
      data[j] = x0 + h;
      const lp = lossOf(config, params, batch);
      data[j] = x0 - h;
      const lm = lossOf(config, params, batch);
      data[j] = x0;

      const numeric = (lp - lm) / (2 * h);
      const a = analytic[j];
      const absErr = Math.abs(numeric - a);
      const denom = Math.max(Math.abs(numeric), Math.abs(a));
      const allowed = ABS_TOL + REL_TOL * denom;
      if (absErr > allowed) {
        worst = `${name}[${j}] analytic=${a.toExponential(4)} numeric=${numeric.toExponential(4)} `
          + `absErr=${absErr.toExponential(3)} (allowed ${allowed.toExponential(3)})`;
      }
    }
  }
  if (worst) throw new Error(`model gradcheck FAILED ${worst}`);
}

describe("gradcheck: tiny GPT (SwiGLU)", () => {
  const config: ModelConfig = {
    vocabSize: 17, blockSize: 8, nLayer: 2, nEmbd: 16, nHead: 2, dropout: 0, ffnActivation: "swiglu",
  };

  it("analytic == numeric for all parameters", () => {
    runGradcheck(config);
  });

  it("init + forward + backward is deterministic (bit-identical)", () => {
    const batch = makeBatch(config, 99);

    const run = () => {
      const params = initGPT(config, B, new SeededRng(20260722));
      const tape = new Tape();
      const res = gptForward(config, params, B, tape, batch.tokens, batch.targets, false);
      tape.backward(res.loss!, B);
      const entries = collectParamEntries(params);
      return {
        loss: (res.loss!.data.data as Float32Array)[0],
        grads: entries.map(([, v]) => Array.from(v.grad!.data as Float32Array)),
      };
    };

    const a = run();
    const b = run();
    expect(b.loss).toBe(a.loss); // exact
    for (let i = 0; i < a.grads.length; i++) {
      expect(b.grads[i]).toEqual(a.grads[i]); // exact
    }
  });
});

describe("gradcheck: tiny GPT (GELU)", () => {
  const config: ModelConfig = {
    vocabSize: 17, blockSize: 8, nLayer: 2, nEmbd: 16, nHead: 2, dropout: 0, ffnActivation: "gelu",
  };

  it("analytic == numeric for all parameters", () => {
    runGradcheck(config);
  });
});

// The universal / kan_spline FFN variants are the composed consumers of the
// [1,ffnDim]-broadcast backward (the impact area of the historical
// sum-keepdims bug), so they get their own whole-model checks — including the
// dead-param invariant over act_gate/act_skip/kan_c* coefficients.
describe("gradcheck: tiny GPT (universal)", () => {
  const config: ModelConfig = {
    vocabSize: 17, blockSize: 8, nLayer: 2, nEmbd: 16, nHead: 2, dropout: 0, ffnActivation: "universal",
  };

  it("analytic == numeric for all parameters", () => {
    runGradcheck(config);
  });
});

describe("gradcheck: tiny GPT (kan_spline)", () => {
  const config: ModelConfig = {
    vocabSize: 17, blockSize: 8, nLayer: 2, nEmbd: 16, nHead: 2, dropout: 0, ffnActivation: "kan_spline",
  };

  it("analytic == numeric for all parameters", () => {
    runGradcheck(config);
  });
});

// ── Activation checkpointing + dropout RNG replay + mixed-precision wiring ──
//
// gptForward's checkpoint path saves the dropout RNG counter before each block
// and restores it inside the checkpointed fn so the backward RECOMPUTATION
// replays bit-identical masks. If that save/restore breaks, checkpointed
// gradients silently diverge from plain gradients — so the test is exact
// equality between the two paths with dropout ON.
describe("model paths: checkpointing + dropout + mixed precision", () => {
  const config: ModelConfig = {
    vocabSize: 17, blockSize: 8, nLayer: 2, nEmbd: 16, nHead: 2, dropout: 0.2, ffnActivation: "gelu",
  };

  function trainRun(activationCheckpointing: boolean, mixedPrecision: boolean) {
    const params = initGPT(config, B, new SeededRng(20260722));
    const batch = makeBatch(config, 99);
    const tape = new Tape();
    const rng = new DropoutRng(42);
    const res = gptForward(
      config, params, B, tape, batch.tokens, batch.targets,
      /*training*/ true, activationCheckpointing, mixedPrecision, rng,
    );
    tape.backward(res.loss!, B);
    const entries = collectParamEntries(params);
    return {
      loss: (res.loss!.data.data as Float32Array)[0],
      names: entries.map(([n]) => n),
      grads: entries.map(([, v]) => Array.from(v.grad!.data as Float32Array)),
    };
  }

  it("training + dropout: checkpointed == plain (loss and every gradient, exact)", () => {
    const plain = trainRun(false, false);
    const chk = trainRun(true, false);
    expect(Number.isFinite(plain.loss)).toBe(true);
    expect(chk.loss).toBe(plain.loss); // bit-identical forward (mask replay)
    for (let i = 0; i < plain.grads.length; i++) {
      expect(chk.grads[i], `grad mismatch for ${plain.names[i]}`).toEqual(plain.grads[i]);
    }
    // Dropout actually engaged: a training run must differ from an eval run.
    const params = initGPT(config, B, new SeededRng(20260722));
    const batch = makeBatch(config, 99);
    const evalRes = gptForward(config, params, B, new Tape(), batch.tokens, batch.targets, false);
    expect((evalRes.loss!.data.data as Float32Array)[0]).not.toBe(plain.loss);
  });

  it("mixedPrecision=true wiring executes (cpu_ref: casts are pass-through no-ops)", () => {
    // CpuRefBackend has no castDtype, so castToF16/castToF32 are documented
    // no-ops here — this covers the gptForward branch wiring (both the
    // checkpointed and non-checkpointed cast sites) and asserts it changes
    // nothing on a castless backend. Real f16 quantization parity is asserted
    // GPU-side in parity-helios.test.ts.
    const plain = trainRun(false, false);
    const mp = trainRun(false, true);
    const mpChk = trainRun(true, true);
    expect(mp.loss).toBe(plain.loss);
    expect(mpChk.loss).toBe(plain.loss);
    for (let i = 0; i < plain.grads.length; i++) {
      expect(mp.grads[i], `mp grad mismatch for ${plain.names[i]}`).toEqual(plain.grads[i]);
      expect(mpChk.grads[i], `mp+ckpt grad mismatch for ${plain.names[i]}`).toEqual(plain.grads[i]);
    }
  });
});
