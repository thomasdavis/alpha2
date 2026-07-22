/**
 * parity-helios — CPU (cpu_ref) vs GPU (Helios) numerical parity.
 *
 * The whole suite lives inside `describeGpu` and SKIPS unless a real NVIDIA
 * GPU is present (same detection as gpu-perf.test.ts), so it is a no-op on the
 * dev box and runs for real on the GPU pods.
 *
 * Coverage, in order of trust:
 *   1. per-op forward parity (matmul/softmax/layerNorm/gelu/silu/siluMul/
 *      crossEntropy/embedding/softCap),
 *   2. tiny-GPT forward logits + loss parity,
 *   3. full backward: every parameter gradient parity,
 *   4. one AdamW step parity,
 *   4.5 mixed-precision f16 casts (castDtype roundtrip, castToF16/castToF32
 *      autograd grad pass-through, mixedPrecision=true tiny-GPT fwd/bwd) —
 *      Helios-only functionality, untestable on cpu_ref (no castDtype),
 *   5. a 100-step training loop on Helios: ZERO non-finite losses/grads and a
 *      final loss within 5% of the cpu_ref run of the same seeded config.
 *
 * All tolerances are constants below so a drift shows up as one edit.
 */
import { describe, it, expect, afterAll } from "vitest";
import { HeliosBackend, destroyDevice, getDeviceInfo } from "@alpha/helios";
import { CpuRefBackend } from "@alpha/tensor";
import { SeededRng, type ModelConfig, type TensorData, type Backend } from "@alpha/core";
import { Tape, Variable, castToF16, castToF32, sum } from "@alpha/autograd";
import { initGPT, gptForward, collectParamEntries } from "@alpha/model";
import { AdamW } from "@alpha/train";

// ── Tolerances (edit here) ───────────────────────────────────────────────────
const FWD_REL_TOL = 1e-3;
const FWD_ABS_TOL = 1e-4;
const LOGITS_REL_TOL = 1e-3;
const LOGITS_ABS_TOL = 2e-3;
const LOSS_REL_TOL = 2e-3;
const GRAD_REL_TOL = 1e-2;
const GRAD_ABS_TOL = 2e-3;
const ADAMW_REL_TOL = 1e-3;
const ADAMW_ABS_TOL = 1e-5;
const TRAIN_STEPS = 100;
const TRAIN_LOSS_PCT = 0.05; // final helios loss must be within 5% of cpu_ref

// ── GPU detection (identical to gpu-perf.test.ts) ────────────────────────────
let hasNvidiaGpu = false;
try {
  hasNvidiaGpu = getDeviceInfo().vendorId === 0x10de;
} catch {
  hasNvidiaGpu = false;
}
if (!hasNvidiaGpu) {
  console.log("[parity-helios] no NVIDIA GPU detected — suite skipped");
}

const gpu = hasNvidiaGpu ? new HeliosBackend() : (null as unknown as HeliosBackend);
const cpu = new CpuRefBackend();
if (hasNvidiaGpu) gpu.setMinGpuSize(1);

afterAll(() => {
  if (hasNvidiaGpu) destroyDevice();
});

const describeGpu = describe.skipIf(!hasNvidiaGpu);

// ── Helpers ──────────────────────────────────────────────────────────────────

function rnd(seed: number, n: number, lo = -1, hi = 1): number[] {
  const r = new SeededRng(seed);
  return Array.from({ length: n }, () => lo + (hi - lo) * r.next());
}

/** Fail if any element is outside abs+rel tolerance; message names the worst. */
function assertClose(name: string, a: Float32Array, b: Float32Array, relTol: number, absTol: number): void {
  expect(a.length, `${name}: length mismatch`).toBe(b.length);
  let worst = -1;
  let worstAbs = 0;
  for (let i = 0; i < a.length; i++) {
    if (!Number.isFinite(a[i]) || !Number.isFinite(b[i])) {
      throw new Error(`${name}: non-finite at ${i} (gpu=${a[i]} cpu=${b[i]})`);
    }
    const absErr = Math.abs(a[i] - b[i]);
    const denom = Math.max(Math.abs(a[i]), Math.abs(b[i]));
    if (absErr > absTol + relTol * denom && absErr > worstAbs) {
      worst = i;
      worstAbs = absErr;
    }
  }
  expect(worst, `${name}: elem ${worst} gpu=${a[worst]} cpu=${b[worst]} absErr=${worstAbs.toExponential(3)}`).toBe(-1);
}

function relCloseScalar(name: string, a: number, b: number, relTol: number): void {
  expect(Number.isFinite(a) && Number.isFinite(b), `${name}: non-finite (gpu=${a} cpu=${b})`).toBe(true);
  const denom = Math.max(Math.abs(a), Math.abs(b), 1e-6);
  expect(Math.abs(a - b) / denom, `${name}: gpu=${a} cpu=${b}`).toBeLessThan(relTol);
}

const f32 = (t: TensorData): Float32Array => t.data as Float32Array;

// ── 1. Per-op forward parity ─────────────────────────────────────────────────

describeGpu("per-op forward parity", () => {
  it("matmul [8,12] x [12,10]", () => {
    const aData = rnd(1, 96, -0.5, 0.5), bData = rnd(2, 120, -0.5, 0.5);
    const c = cpu.matmul(cpu.fromArray(aData, [8, 12]), cpu.fromArray(bData, [12, 10]));
    const g = gpu.matmul(gpu.fromArray(aData, [8, 12]), gpu.fromArray(bData, [12, 10]));
    assertClose("matmul", f32(g), f32(c), FWD_REL_TOL, 5e-3); // coop-matmul f16 accum → looser abs
  });

  it("softmax axis=-1 [6,10]", () => {
    const x = rnd(3, 60, -3, 3);
    const c = cpu.softmax(cpu.fromArray(x, [6, 10]), -1);
    const g = gpu.softmax(gpu.fromArray(x, [6, 10]), -1);
    assertClose("softmax", f32(g), f32(c), FWD_REL_TOL, FWD_ABS_TOL);
  });

  it("layerNorm [6,16]", () => {
    const x = rnd(4, 96, -2, 2), w = rnd(5, 16, 0.5, 1.5), b = rnd(6, 16, -0.5, 0.5);
    const c = cpu.layerNorm(cpu.fromArray(x, [6, 16]), cpu.fromArray(w, [16]), cpu.fromArray(b, [16]), 1e-5);
    const g = gpu.layerNorm(gpu.fromArray(x, [6, 16]), gpu.fromArray(w, [16]), gpu.fromArray(b, [16]), 1e-5);
    assertClose("layerNorm", f32(g), f32(c), FWD_REL_TOL, FWD_ABS_TOL);
  });

  it("gelu [64]", () => {
    const x = rnd(7, 64, -4, 4);
    assertClose("gelu", f32(gpu.gelu(gpu.fromArray(x, [64]))), f32(cpu.gelu(cpu.fromArray(x, [64]))), FWD_REL_TOL, FWD_ABS_TOL);
  });

  it("silu [64]", () => {
    const x = rnd(8, 64, -4, 4);
    assertClose("silu", f32(gpu.silu(gpu.fromArray(x, [64]))), f32(cpu.silu(cpu.fromArray(x, [64]))), FWD_REL_TOL, FWD_ABS_TOL);
  });

  it("siluMul [64] (SwiGLU core)", () => {
    const x = rnd(9, 64, -4, 4), y = rnd(10, 64, -2, 2);
    // cpu_ref has no fused siluMul — reference is silu(x) * y.
    const cRef = cpu.mul(cpu.silu(cpu.fromArray(x, [64])), cpu.fromArray(y, [64]));
    const g = gpu.siluMul!(gpu.fromArray(x, [64]), gpu.fromArray(y, [64]));
    assertClose("siluMul", f32(g), f32(cRef), FWD_REL_TOL, FWD_ABS_TOL);
  });

  it("crossEntropy [6,10]", () => {
    const logits = rnd(11, 60, -3, 3);
    const targets: TensorData = { shape: [6], dtype: "i32", data: Int32Array.from([1, 3, 0, 9, 4, 7]) };
    const c = cpu.crossEntropy(cpu.fromArray(logits, [6, 10]), targets);
    const g = gpu.crossEntropy(gpu.fromArray(logits, [6, 10]), targets);
    relCloseScalar("crossEntropy", f32(g)[0], f32(c)[0], LOSS_REL_TOL);
  });

  it("embedding [12,8] gather 6 rows", () => {
    const table = rnd(12, 96, -1, 1);
    const indices: TensorData = { shape: [6], dtype: "i32", data: Int32Array.from([0, 11, 5, 2, 7, 3]) };
    const c = cpu.embedding(cpu.fromArray(table, [12, 8]), indices);
    const g = gpu.embedding(gpu.fromArray(table, [12, 8]), indices);
    assertClose("embedding", f32(g), f32(c), FWD_REL_TOL, FWD_ABS_TOL);
  });

  it("softCap cap=30 [64]", () => {
    const cap = 30;
    const x = rnd(13, 64, -50, 50);
    // cpu_ref has no softCap — reference is cap*tanh(x/cap).
    const ref = Float32Array.from(x, (v) => cap * Math.tanh(v / cap));
    const g = gpu.softCap!(gpu.fromArray(x, [64]), cap);
    assertClose("softCap", f32(g), ref, FWD_REL_TOL, FWD_ABS_TOL);
  });
});

// ── Tiny-GPT config shared by 2–5 ────────────────────────────────────────────
const GPT_CONFIG: ModelConfig = {
  vocabSize: 32, blockSize: 16, nLayer: 2, nEmbd: 32, nHead: 4, dropout: 0, ffnActivation: "swiglu",
};
const GPT_SEED = 1337;
const BATCH = 2;

function makeBatch(config: ModelConfig, seed: number): { tokens: TensorData; targets: TensorData } {
  const T = config.blockSize;
  const r = new SeededRng(seed);
  const tok = new Int32Array(BATCH * T);
  const tgt = new Int32Array(BATCH * T);
  for (let i = 0; i < BATCH * T; i++) {
    tok[i] = Math.floor(r.next() * config.vocabSize);
    tgt[i] = Math.floor(r.next() * config.vocabSize);
  }
  return {
    tokens: { shape: [BATCH, T], dtype: "i32", data: tok },
    targets: { shape: [BATCH, T], dtype: "i32", data: tgt },
  };
}

const releaser = (b: Backend): ((td: TensorData) => void) | undefined => {
  const any = b as unknown as { releaseGpuTensor?: (td: TensorData) => void };
  return typeof any.releaseGpuTensor === "function" ? (td) => any.releaseGpuTensor!(td) : undefined;
};

// ── 2. Tiny-GPT forward parity ───────────────────────────────────────────────

describeGpu("tiny-GPT forward parity", () => {
  it("logits + loss match cpu_ref", () => {
    const batch = makeBatch(GPT_CONFIG, 99);
    const paramsC = initGPT(GPT_CONFIG, cpu, new SeededRng(GPT_SEED));
    const paramsG = initGPT(GPT_CONFIG, gpu, new SeededRng(GPT_SEED));

    const tapeC = new Tape();
    const resC = gptForward(GPT_CONFIG, paramsC, cpu, tapeC, batch.tokens, batch.targets, false);
    const tapeG = new Tape();
    const resG = gptForward(GPT_CONFIG, paramsG, gpu, tapeG, batch.tokens, batch.targets, false);

    assertClose("logits", f32(resG.logits.data), f32(resC.logits.data), LOGITS_REL_TOL, LOGITS_ABS_TOL);
    relCloseScalar("loss", f32(resG.loss!.data)[0], f32(resC.loss!.data)[0], LOSS_REL_TOL);
  });
});

// ── 3. Full backward gradient parity ─────────────────────────────────────────

describeGpu("tiny-GPT backward parity", () => {
  it("every parameter gradient matches cpu_ref", () => {
    const batch = makeBatch(GPT_CONFIG, 99);

    const paramsC = initGPT(GPT_CONFIG, cpu, new SeededRng(GPT_SEED));
    const tapeC = new Tape();
    const resC = gptForward(GPT_CONFIG, paramsC, cpu, tapeC, batch.tokens, batch.targets, false);
    tapeC.backward(resC.loss!, cpu);

    const paramsG = initGPT(GPT_CONFIG, gpu, new SeededRng(GPT_SEED));
    const tapeG = new Tape();
    const resG = gptForward(GPT_CONFIG, paramsG, gpu, tapeG, batch.tokens, batch.targets, false);
    tapeG.backward(resG.loss!, gpu);

    const gc = new Map(collectParamEntries(paramsC));
    const gg = new Map(collectParamEntries(paramsG));
    for (const [name, vC] of gc) {
      const vG = gg.get(name)!;
      expect(vC.grad, `${name}: cpu grad null`).not.toBeNull();
      expect(vG.grad, `${name}: gpu grad null`).not.toBeNull();
      assertClose(`grad ${name}`, f32(vG.grad!), f32(vC.grad!), GRAD_REL_TOL, GRAD_ABS_TOL);
    }
  });
});

// ── 4. One AdamW step parity ─────────────────────────────────────────────────

describeGpu("AdamW step parity", () => {
  it("one step matches cpu_ref", () => {
    const batch = makeBatch(GPT_CONFIG, 99);
    const cfg = { lr: 3e-4, beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0.1 };

    const stepOnce = (backend: Backend) => {
      const params = initGPT(GPT_CONFIG, backend, new SeededRng(GPT_SEED));
      const tape = new Tape();
      const res = gptForward(GPT_CONFIG, params, backend, tape, batch.tokens, batch.targets, false);
      tape.backward(res.loss!, backend);
      const opt = new AdamW(backend, cfg);
      const pmap = new Map<string, TensorData>();
      const gmap = new Map<string, TensorData>();
      for (const [name, v] of collectParamEntries(params)) {
        pmap.set(name, v.data);
        if (v.grad) gmap.set(name, v.grad);
      }
      opt.step(pmap, gmap);
      return new Map([...pmap].map(([n, t]) => [n, Float32Array.from(f32(t))]));
    };

    const after = stepOnce(gpu);
    const afterC = stepOnce(cpu);
    for (const [name, g] of after) {
      assertClose(`adamw ${name}`, g, afterC.get(name)!, ADAMW_REL_TOL, ADAMW_ABS_TOL);
    }
  });
});

// ── 4.5 Mixed-precision f16 casts (Helios-only: cpu_ref has no castDtype) ────

describeGpu("mixed precision (f16 casts)", () => {
  it("castDtype f32 → f16 → f32 roundtrip within f16 epsilon", () => {
    const x = rnd(21, 64, -4, 4);
    const h = gpu.castDtype!(gpu.fromArray(x, [8, 8]), "f16");
    expect(h.dtype).toBe("f16");
    const back = gpu.castDtype!(h, "f32");
    const b = f32(back);
    for (let i = 0; i < x.length; i++) {
      // f16 has 10 mantissa bits → round-to-nearest rel error ≤ 2^-11 ≈ 4.9e-4.
      const allowed = Math.abs(x[i]) * 6e-4 + 1e-6;
      expect(Math.abs(b[i] - x[i]), `elem ${i}: ${x[i]} → ${b[i]}`).toBeLessThanOrEqual(allowed);
    }
  });

  it("castToF16 → castToF32 autograd roundtrip: value ≈ input, gradient passes through", () => {
    const tape = new Tape();
    const ctx = { tape, backend: gpu as unknown as Backend };
    const xData = rnd(22, 32, -2, 2);
    const x = new Variable(gpu.fromArray(xData, [4, 8]), true);
    const y = castToF32(ctx, castToF16(ctx, x));
    expect(y.data.dtype).toBe("f32");
    const yArr = f32(y.data);
    for (let i = 0; i < xData.length; i++) {
      expect(Math.abs(yArr[i] - xData[i]), `value ${i}`).toBeLessThanOrEqual(Math.abs(xData[i]) * 6e-4 + 1e-6);
    }
    const loss = sum(ctx, y);
    tape.backward(loss, gpu);
    const g = f32(x.grad!);
    for (let i = 0; i < g.length; i++) {
      // d(sum)/dx = 1 and 1.0 is exactly representable in f16, so the grad
      // must survive the backward f32→f16→f32 chain (near-)exactly.
      expect(Math.abs(g[i] - 1), `grad[${i}]=${g[i]}`).toBeLessThanOrEqual(1e-3);
    }
  });

  it("tiny-GPT fwd/bwd with mixedPrecision=true: all finite, loss within 5% of fp32", () => {
    const batch = makeBatch(GPT_CONFIG, 99);
    const run = (mp: boolean): number => {
      const params = initGPT(GPT_CONFIG, gpu, new SeededRng(GPT_SEED));
      const tape = new Tape();
      // training=true so the mixed-precision casts actually engage (dropout=0,
      // so no RNG is needed and the run stays deterministic).
      const res = gptForward(GPT_CONFIG, params, gpu, tape, batch.tokens, batch.targets, true, false, mp);
      const lossVal = f32(res.loss!.data)[0];
      tape.backward(res.loss!, gpu, releaser(gpu));
      for (const [name, v] of collectParamEntries(params)) {
        expect(v.grad, `${name}: grad null (mp=${mp})`).not.toBeNull();
        const gd = f32(v.grad!);
        for (let i = 0; i < gd.length; i++) {
          if (!Number.isFinite(gd[i])) throw new Error(`${name}[${i}] grad non-finite (mp=${mp})`);
        }
      }
      return lossVal;
    };
    const mpLoss = run(true);
    const fpLoss = run(false);
    expect(Number.isFinite(mpLoss), `mp loss non-finite (${mpLoss})`).toBe(true);
    expect(Math.abs(mpLoss - fpLoss) / Math.max(Math.abs(fpLoss), 1e-6),
      `mp loss ${mpLoss} vs fp32 ${fpLoss}`).toBeLessThan(0.05);
  });
});

// ── 5. 100-step Helios training loop ─────────────────────────────────────────

describeGpu("Helios training loop", () => {
  it("100 steps: no non-finite; final loss within 5% of cpu_ref", () => {
    const batch = makeBatch(GPT_CONFIG, 99); // fixed batch every step
    const cfg = { lr: 1e-3, beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0.1 };

    const runLoop = (backend: Backend): number[] => {
      const release = releaser(backend);
      const params = initGPT(GPT_CONFIG, backend, new SeededRng(GPT_SEED));
      const opt = new AdamW(backend, cfg);
      const entries = collectParamEntries(params);
      const losses: number[] = [];

      for (let step = 0; step < TRAIN_STEPS; step++) {
        const tape = new Tape();
        const res = gptForward(GPT_CONFIG, params, backend, tape, batch.tokens, batch.targets, false);
        const lossVal = f32(res.loss!.data)[0];
        losses.push(lossVal);

        tape.backward(res.loss!, backend, release);

        // Every gradient must be finite before we step.
        for (const [name, v] of entries) {
          const g = v.grad;
          if (!g) throw new Error(`step ${step}: ${name} has null grad`);
          const gd = g.data as Float32Array;
          for (let i = 0; i < gd.length; i++) {
            if (!Number.isFinite(gd[i])) throw new Error(`step ${step}: ${name}[${i}] grad non-finite`);
          }
        }

        opt.stepParamEntries(entries as any);

        // Zero grads + reclaim buffers for the next step.
        for (const [, v] of entries) {
          if (v.grad && release) release(v.grad);
          v.grad = null;
        }
        tape.clear(release);
        (backend as unknown as { flush?: () => void }).flush?.();
      }
      return losses;
    };

    const gpuLosses = runLoop(gpu);
    for (let i = 0; i < gpuLosses.length; i++) {
      expect(Number.isFinite(gpuLosses[i]), `helios step ${i} loss non-finite (${gpuLosses[i]})`).toBe(true);
    }

    const cpuLosses = runLoop(cpu);
    const gpuFinal = gpuLosses[gpuLosses.length - 1];
    const cpuFinal = cpuLosses[cpuLosses.length - 1];
    expect(gpuFinal).toBeLessThan(gpuLosses[0]); // it actually learned
    expect(Math.abs(gpuFinal - cpuFinal) / cpuFinal,
      `helios final ${gpuFinal} vs cpu ${cpuFinal}`).toBeLessThan(TRAIN_LOSS_PCT);
  });
});
