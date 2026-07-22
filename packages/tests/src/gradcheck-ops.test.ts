/**
 * gradcheck-ops — finite-difference gradient checks for every op the GPT
 * forward/backward path uses.
 *
 * The heart of the file is `checkGrad`, a reusable central-difference checker:
 * it builds a computation on the cpu_ref backend, reduces the output to a
 * scalar with a fixed random projection, runs tape backward to get analytic
 * gradients, then perturbs each input element and re-runs the forward to get
 * numeric gradients. Analytic vs numeric are compared per element with a
 * combined absolute+relative tolerance.
 *
 * Central differences on f32 are used with h ≈ 1e-3 * scale; the default
 * comparison is relTol 2e-2 with an absolute floor 2e-3. A handful of ops
 * loosen this and each says why.
 *
 * Every op is exercised at a non-trivial shape (and a broadcast shape where
 * broadcasting is on the backward path). Inputs are kept ≤ 64 elements so the
 * whole file stays fast on CPU.
 */
import { describe, it, expect } from "vitest";
import { CpuRefBackend } from "@alpha/tensor";
import { shapeSize, SeededRng } from "@alpha/core";
import type { TensorData } from "@alpha/core";
import {
  Variable, Tape, DropoutRng,
  add, sub, mul, div, scale, neg,
  matmul, matmulTransposed, matmulTransposedGelu,
  sum, mean, exp, log, sqrt, relu, gelu, silu, siluMul, clamp, softCap,
  softmax, layerNorm, embedding, crossEntropy,
  reshape, transpose, slice, sliceQkv, dropout, residualDropoutAdd,
  checkpoint,
} from "@alpha/autograd";

// ── checkGrad harness ────────────────────────────────────────────────────────

/** Factory the build closure calls to materialise each differentiable input,
 *  in a fixed order. The harness owns the data so it can perturb one element. */
type MakeInput = (data: ArrayLike<number>, shape: number[]) => Variable;
type BuildCtx = { tape: Tape; backend: CpuRefBackend; dropoutRng?: DropoutRng };
type BuildFn = (ctx: BuildCtx, mk: MakeInput) => Variable;

interface CheckGradOptions {
  /** Relative step-size base: h = eps * max(1, |x|). Default 1e-3. */
  eps?: number;
  /** Relative tolerance. Default 2e-2. */
  relTol?: number;
  /** Absolute tolerance floor. Default 2e-3. */
  absTol?: number;
  /** Seed for the fixed output projection. Default 1234. */
  seed?: number;
  /** Seed for the (deterministic) dropout RNG. Default 7. */
  dropoutSeed?: number;
  /** Max elements per input (guard: keeps the CPU sweep cheap). Default 64. */
  maxElems?: number;
  label?: string;
}

interface CheckGradResult {
  maxAbsErr: number;
  maxRelErr: number;
  nChecked: number;
}

function checkGrad(build: BuildFn, opts: CheckGradOptions = {}): CheckGradResult {
  const B = new CpuRefBackend();
  const epsBase = opts.eps ?? 1e-3;
  const relTol = opts.relTol ?? 2e-2;
  const absTol = opts.absTol ?? 2e-3;
  const maxElems = opts.maxElems ?? 64;
  const projSeed = opts.seed ?? 1234;
  const dropoutSeed = opts.dropoutSeed ?? 7;
  const label = opts.label ?? "op";

  const base: Float32Array[] = [];
  const shapes: number[][] = [];
  let proj: Float32Array | null = null;

  // Reduce a (possibly non-scalar) output to a scalar loss with a fixed
  // random projection so that no op has an identically-constant loss
  // (e.g. sum(softmax) ≡ 1 would give a zero gradient).
  const reduceLoss = (ctx: BuildCtx, out: Variable): Variable => {
    const size = shapeSize(out.data.shape);
    if (size === 1 || !proj) return sum(ctx, out);
    const projVar = new Variable(
      { shape: [...out.data.shape], dtype: "f32", data: proj } as TensorData,
      false,
    );
    return sum(ctx, mul(ctx, out, projVar));
  };

  // Analytic + recording pass. mk captures the base data on the way through.
  const analyticGrads: Float32Array[] = [];
  {
    const ctx: BuildCtx = { tape: new Tape(), backend: B, dropoutRng: new DropoutRng(dropoutSeed) };
    let k = 0;
    const wrt: Variable[] = [];
    const mk: MakeInput = (data, shape) => {
      const d = Float32Array.from(data as ArrayLike<number>);
      base[k] = Float32Array.from(d);
      shapes[k] = [...shape];
      if (d.length > maxElems) {
        throw new Error(`[${label}] input #${k} has ${d.length} elems > maxElems ${maxElems}`);
      }
      k++;
      const v = new Variable({ shape: [...shape], dtype: "f32", data: d } as TensorData, true);
      wrt.push(v);
      return v;
    };
    const out = build(ctx, mk);
    const outSize = shapeSize(out.data.shape);
    if (outSize > 1) {
      const r = new SeededRng(projSeed);
      proj = new Float32Array(outSize);
      for (let i = 0; i < outSize; i++) proj[i] = r.next() * 2 - 1;
    }
    const loss = reduceLoss(ctx, out);
    ctx.tape.backward(loss, B);
    for (let i = 0; i < wrt.length; i++) {
      const g = wrt[i].grad;
      analyticGrads[i] = g ? Float32Array.from(g.data as Float32Array) : new Float32Array(base[i].length);
    }
  }

  // Forward-only loss evaluator with a substituted set of input arrays.
  const lossAt = (arrays: Float32Array[]): number => {
    const ctx: BuildCtx = { tape: new Tape(), backend: B, dropoutRng: new DropoutRng(dropoutSeed) };
    let k = 0;
    const mk: MakeInput = (_data, shape) => {
      const v = new Variable(
        { shape: [...shape], dtype: "f32", data: Float32Array.from(arrays[k]) } as TensorData,
        true,
      );
      k++;
      return v;
    };
    const out = build(ctx, mk);
    const loss = reduceLoss(ctx, out);
    return (loss.data.data as Float32Array)[0];
  };

  // Numeric central differences, element by element.
  let maxAbsErr = 0;
  let maxRelErr = 0;
  let nChecked = 0;
  let worst = "";
  for (let i = 0; i < base.length; i++) {
    const xi = base[i];
    for (let j = 0; j < xi.length; j++) {
      const x0 = xi[j];
      const h = epsBase * Math.max(1, Math.abs(x0));

      const plus = base.map((a, idx) => (idx === i ? Float32Array.from(a) : a));
      plus[i][j] = x0 + h;
      const lp = lossAt(plus);

      const minus = base.map((a, idx) => (idx === i ? Float32Array.from(a) : a));
      minus[i][j] = x0 - h;
      const lm = lossAt(minus);

      const numeric = (lp - lm) / (2 * h);
      const analytic = analyticGrads[i][j];

      const absErr = Math.abs(numeric - analytic);
      const denom = Math.max(Math.abs(numeric), Math.abs(analytic));
      const relErr = denom > 0 ? absErr / denom : 0;
      const allowed = absTol + relTol * denom;

      if (absErr > maxAbsErr) maxAbsErr = absErr;
      if (relErr > maxRelErr && denom > absTol) maxRelErr = relErr;
      nChecked++;

      if (absErr > allowed) {
        worst = `[${label}] input#${i}[${j}] analytic=${analytic.toExponential(4)} `
          + `numeric=${numeric.toExponential(4)} absErr=${absErr.toExponential(3)} `
          + `relErr=${relErr.toExponential(3)} (allowed ${allowed.toExponential(3)})`;
      }
    }
  }

  if (worst) throw new Error(`gradcheck FAILED ${worst}`);
  return { maxAbsErr, maxRelErr, nChecked };
}

/** Deterministic seeded value array in [lo, hi). */
function rnd(seed: number, n: number, lo = -1, hi = 1): number[] {
  const r = new SeededRng(seed);
  return Array.from({ length: n }, () => lo + (hi - lo) * r.next());
}

// ── Arithmetic ───────────────────────────────────────────────────────────────

describe("gradcheck: arithmetic", () => {
  it("add [3,4]", () => {
    checkGrad((ctx, mk) => add(ctx, mk(rnd(1, 12), [3, 4]), mk(rnd(2, 12), [3, 4])), { label: "add" });
  });

  it("add broadcast [3,4] + [1,4]", () => {
    checkGrad((ctx, mk) => add(ctx, mk(rnd(1, 12), [3, 4]), mk(rnd(2, 4), [1, 4])), { label: "add-bcast" });
  });

  it("sub broadcast [3,1] - [3,4]", () => {
    checkGrad((ctx, mk) => sub(ctx, mk(rnd(3, 3), [3, 1]), mk(rnd(4, 12), [3, 4])), { label: "sub-bcast" });
  });

  it("mul [2,3]", () => {
    checkGrad((ctx, mk) => mul(ctx, mk(rnd(5, 6), [2, 3]), mk(rnd(6, 6), [2, 3])), { label: "mul" });
  });

  it("mul broadcast [2,3] * [1,3] (act_gate shape)", () => {
    checkGrad((ctx, mk) => mul(ctx, mk(rnd(5, 6), [2, 3]), mk(rnd(6, 3), [1, 3])), { label: "mul-bcast" });
  });

  it("div [2,3] (denominator away from 0)", () => {
    checkGrad((ctx, mk) => div(ctx, mk(rnd(7, 6, -1, 1), [2, 3]), mk(rnd(8, 6, 1.5, 3), [2, 3])), { label: "div" });
  });

  it("div broadcast [2,3] / [2,1]", () => {
    checkGrad((ctx, mk) => div(ctx, mk(rnd(7, 6, -1, 1), [2, 3]), mk(rnd(8, 2, 1.5, 3), [2, 1])), { label: "div-bcast" });
  });

  it("scale [2,3] by 2.5", () => {
    checkGrad((ctx, mk) => scale(ctx, mk(rnd(9, 6), [2, 3]), 2.5), { label: "scale" });
  });

  it("neg [2,3]", () => {
    checkGrad((ctx, mk) => neg(ctx, mk(rnd(10, 6), [2, 3])), { label: "neg" });
  });
});

// ── Matmul ───────────────────────────────────────────────────────────────────

describe("gradcheck: matmul", () => {
  it("matmul 2D [3,4] x [4,5]", () => {
    checkGrad((ctx, mk) => matmul(ctx, mk(rnd(1, 12), [3, 4]), mk(rnd(2, 20), [4, 5])), { label: "matmul" });
  });

  it("matmul batched [2,3,4] x [2,4,3]", () => {
    checkGrad((ctx, mk) => matmul(ctx, mk(rnd(3, 24), [2, 3, 4]), mk(rnd(4, 24), [2, 4, 3])), { label: "matmul-batched" });
  });

  it("matmulTransposed [3,4] x [5,4] -> [3,5]", () => {
    checkGrad((ctx, mk) => matmulTransposed(ctx, mk(rnd(5, 12), [3, 4]), mk(rnd(6, 20), [5, 4])), { label: "matmulT" });
  });

  it("matmulTransposedGelu (fused, model's default GELU path) [3,4] x [5,4]", () => {
    // The fused op carries its OWN duplicated gelu-derivative in its backward
    // closure, so the standalone gelu gradcheck does not protect it.
    checkGrad((ctx, mk) => matmulTransposedGelu(ctx, mk(rnd(5, 12), [3, 4]), mk(rnd(6, 20), [5, 4])), { label: "matmulTGelu" });
  });
});

// ── Reductions ───────────────────────────────────────────────────────────────

describe("gradcheck: reductions", () => {
  it("sum all -> scalar", () => {
    checkGrad((ctx, mk) => sum(ctx, mk(rnd(1, 12), [3, 4])), { label: "sum-all" });
  });

  it("sum axis=1 keepdims", () => {
    checkGrad((ctx, mk) => sum(ctx, mk(rnd(2, 12), [3, 4]), 1, true), { label: "sum-axis" });
  });

  it("mean all -> scalar", () => {
    checkGrad((ctx, mk) => mean(ctx, mk(rnd(3, 12), [3, 4])), { label: "mean-all" });
  });

  it("mean axis=1 keepdims", () => {
    checkGrad((ctx, mk) => mean(ctx, mk(rnd(4, 12), [3, 4]), 1, true), { label: "mean-axis" });
  });
});

// ── Element-wise ─────────────────────────────────────────────────────────────

describe("gradcheck: element-wise", () => {
  it("exp [2,3] (moderate range)", () => {
    checkGrad((ctx, mk) => exp(ctx, mk(rnd(1, 6, -1.2, 1.2), [2, 3])), { label: "exp" });
  });

  it("log [2,3] (positive input)", () => {
    checkGrad((ctx, mk) => log(ctx, mk(rnd(2, 6, 0.3, 3), [2, 3])), { label: "log" });
  });

  it("sqrt [2,3] (positive input)", () => {
    checkGrad((ctx, mk) => sqrt(ctx, mk(rnd(3, 6, 0.3, 3), [2, 3])), { label: "sqrt" });
  });

  it("relu [2,4] (inputs away from the kink)", () => {
    // Kept away from 0 so the finite-difference step never straddles the kink.
    checkGrad((ctx, mk) => relu(ctx, mk([-2, -0.5, 0.6, 2, 1.3, -1.7, 0.9, -0.4], [2, 4])), { label: "relu" });
  });

  it("gelu [2,3]", () => {
    checkGrad((ctx, mk) => gelu(ctx, mk(rnd(5, 6, -2, 2), [2, 3])), { label: "gelu" });
  });

  it("silu [2,3]", () => {
    checkGrad((ctx, mk) => silu(ctx, mk(rnd(6, 6, -2, 2), [2, 3])), { label: "silu" });
  });

  it("siluMul (SwiGLU core) [2,3]", () => {
    checkGrad((ctx, mk) => siluMul(ctx, mk(rnd(7, 6, -2, 2), [2, 3]), mk(rnd(8, 6, -2, 2), [2, 3])), { label: "siluMul" });
  });

  it("clamp [-1,1] [2,4] (inputs away from boundaries)", () => {
    checkGrad((ctx, mk) => clamp(ctx, mk([-2, -0.5, 0.4, 2, 0.9, -1.8, 0.2, -0.9], [2, 4]), -1, 1), { label: "clamp" });
  });

  it("softCap cap=4 [2,3]", () => {
    // tanh curvature makes central-difference error a touch larger than the
    // linear ops; still comfortably inside the default tolerance.
    checkGrad((ctx, mk) => softCap(ctx, mk(rnd(9, 6, -5, 5), [2, 3]), 4), { label: "softCap" });
  });
});

// ── NN ops ───────────────────────────────────────────────────────────────────

describe("gradcheck: nn", () => {
  it("softmax axis=-1 [3,4]", () => {
    checkGrad((ctx, mk) => softmax(ctx, mk(rnd(1, 12, -2, 2), [3, 4]), -1), { label: "softmax" });
  });

  it("layerNorm [3,4] (input, weight, bias grads)", () => {
    checkGrad((ctx, mk) => {
      const x = mk(rnd(1, 12, -2, 2), [3, 4]);
      const w = mk(rnd(2, 4, 0.5, 1.5), [4]);
      const b = mk(rnd(3, 4, -0.5, 0.5), [4]);
      return layerNorm(ctx, x, w, b, 1e-5);
    }, { label: "layerNorm" });
  });

  it("embedding (table grads; indices fixed)", () => {
    const indices: TensorData = { shape: [4], dtype: "i32", data: Int32Array.from([0, 2, 1, 2]) };
    checkGrad((ctx, mk) => embedding(ctx, mk(rnd(4, 15), [5, 3]), indices), { label: "embedding" });
  });

  it("crossEntropy [3,4] (finite-difference on the scalar loss)", () => {
    const targets: TensorData = { shape: [3], dtype: "i32", data: Int32Array.from([1, 3, 0]) };
    checkGrad((ctx, mk) => crossEntropy(ctx, mk(rnd(5, 12, -2, 2), [3, 4]), targets), { label: "crossEntropy" });
  });

  it("crossEntropy analytic grad == (softmax - onehot)/N", () => {
    const B = new CpuRefBackend();
    const N = 3, C = 4;
    const logitData = rnd(5, N * C, -2, 2);
    const targetIdx = [1, 3, 0];
    const targets: TensorData = { shape: [N], dtype: "i32", data: Int32Array.from(targetIdx) };

    const tape = new Tape();
    const ctx = { tape, backend: B };
    const logits = new Variable(B.fromArray(logitData, [N, C]), true);
    const loss = crossEntropy(ctx, logits, targets);
    tape.backward(loss, B);

    const probs = B.softmax(B.fromArray(logitData, [N, C]), -1).data as Float32Array;
    const ref = new Float32Array(N * C);
    for (let i = 0; i < N; i++) {
      for (let c = 0; c < C; c++) {
        ref[i * C + c] = (probs[i * C + c] - (c === targetIdx[i] ? 1 : 0)) / N;
      }
    }
    const g = logits.grad!.data as Float32Array;
    for (let i = 0; i < N * C; i++) expect(g[i]).toBeCloseTo(ref[i], 6);
  });
});

// ── Reshape / view ───────────────────────────────────────────────────────────

describe("gradcheck: reshape/view", () => {
  it("reshape [2,6] -> [3,4]", () => {
    checkGrad((ctx, mk) => reshape(ctx, mk(rnd(1, 12), [2, 6]), [3, 4]), { label: "reshape" });
  });

  it("transpose [2,3] (0,1)", () => {
    checkGrad((ctx, mk) => transpose(ctx, mk(rnd(2, 6), [2, 3]), 0, 1), { label: "transpose" });
  });

  it("slice [4,5] -> [1:3, 1:4]", () => {
    checkGrad((ctx, mk) => slice(ctx, mk(rnd(3, 20), [4, 5]), [1, 1], [3, 4]), { label: "slice" });
  });

  it("sliceQkv [2, 3*4] combined with distinct scales", () => {
    // Distinct scales on q/k/v so the gradient into each third is distinguishable.
    checkGrad((ctx, mk) => {
      const a = mk(rnd(4, 24), [2, 12]);
      const [q, k, v] = sliceQkv(ctx, a);
      return add(ctx, add(ctx, scale(ctx, q, 1), scale(ctx, k, 2)), scale(ctx, v, 3));
    }, { label: "sliceQkv" });
  });
});

// ── Dropout ──────────────────────────────────────────────────────────────────

describe("gradcheck: dropout", () => {
  it("dropout p=0.3 gradient == mask (fixed seed)", () => {
    checkGrad((ctx, mk) => dropout(ctx, mk(rnd(1, 24, -2, 2), [4, 6]), 0.3, true), { label: "dropout", dropoutSeed: 11 });
  });

  it("dropout mask is deterministic across forwards with the same seed", () => {
    const B = new CpuRefBackend();
    const run = () => {
      const ctx = { tape: new Tape(), backend: B, dropoutRng: new DropoutRng(11) };
      const a = new Variable(B.fromArray(rnd(1, 24, 1, 2), [4, 6]), true);
      const out = dropout(ctx, a, 0.3, true);
      return Float32Array.from(out.data.data as Float32Array);
    };
    const o1 = run();
    const o2 = run();
    expect(Array.from(o1)).toEqual(Array.from(o2));
  });

  it("dropout kept elements are scaled by 1/(1-p)", () => {
    const B = new CpuRefBackend();
    const p = 0.3;
    const inData = rnd(1, 24, 1, 2); // strictly positive so kept/dropped is unambiguous
    const ctx = { tape: new Tape(), backend: B, dropoutRng: new DropoutRng(11) };
    const a = new Variable(B.fromArray(inData, [4, 6]), true);
    const out = dropout(ctx, a, p, true) as Variable;
    const o = out.data.data as Float32Array;
    const scaleVal = 1 / (1 - p);
    for (let i = 0; i < o.length; i++) {
      const ratio = o[i] / inData[i];
      const ok = Math.abs(ratio) < 1e-6 || Math.abs(ratio - scaleVal) < 1e-4;
      expect(ok, `element ${i} ratio ${ratio} is neither 0 nor ${scaleVal}`).toBe(true);
    }
  });

  it("dropout p=0 / training=false is identity (grad passes through)", () => {
    checkGrad((ctx, mk) => dropout(ctx, mk(rnd(1, 12, -2, 2), [3, 4]), 0.0, true), { label: "dropout-p0" });
  });

  it("residualDropoutAdd p=0.3 (residual pass-through + masked projection)", () => {
    checkGrad((ctx, mk) => {
      const res = mk(rnd(1, 24, -2, 2), [4, 6]);
      const proj = mk(rnd(2, 24, -2, 2), [4, 6]);
      return residualDropoutAdd(ctx, res, proj, 0.3, true);
    }, { label: "residualDropoutAdd", dropoutSeed: 13 });
  });
});

// ── Checkpoint ───────────────────────────────────────────────────────────────

describe("gradcheck: checkpoint", () => {
  it("2-block stack: gradients equal WITH vs WITHOUT checkpoint", () => {
    const B = new CpuRefBackend();

    // Two parameter matrices shared by both variants; a small 2-layer MLP-ish
    // stack (matmul → gelu → layerNorm-ish) exercised through the residual add.
    const w1Data = rnd(1, 16, -0.5, 0.5);
    const w2Data = rnd(2, 16, -0.5, 0.5);

    const run = (useCheckpoint: boolean) => {
      const tape = new Tape();
      const ctx = { tape, backend: B };

      const x = new Variable(B.fromArray(rnd(3, 8, -1, 1), [2, 4]), true);
      const w1 = new Variable(B.fromArray(w1Data, [4, 4]), true);
      const w2 = new Variable(B.fromArray(w2Data, [4, 4]), true);

      const block = (innerCtx: typeof ctx, input: Variable, w: Variable): Variable => {
        const h = matmul(innerCtx, input, w);   // [2,4]
        const a = gelu(innerCtx, h);
        return add(innerCtx, a, input);          // residual
      };

      let y: Variable = x;
      for (const w of [w1, w2]) {
        y = useCheckpoint
          ? checkpoint(ctx as any, (ic, inp) => block(ic as typeof ctx, inp, w), y)
          : block(ctx, y, w);
      }
      const loss = sum(ctx, y);
      tape.backward(loss, B);

      return {
        loss: (loss.data.data as Float32Array)[0],
        xGrad: Array.from(x.grad!.data as Float32Array),
        w1Grad: Array.from(w1.grad!.data as Float32Array),
        w2Grad: Array.from(w2.grad!.data as Float32Array),
      };
    };

    const plain = run(false);
    const chk = run(true);

    expect(chk.loss).toBeCloseTo(plain.loss, 6);
    for (let i = 0; i < plain.xGrad.length; i++) expect(chk.xGrad[i]).toBeCloseTo(plain.xGrad[i], 6);
    for (let i = 0; i < plain.w1Grad.length; i++) expect(chk.w1Grad[i]).toBeCloseTo(plain.w1Grad[i], 6);
    for (let i = 0; i < plain.w2Grad.length; i++) expect(chk.w2Grad[i]).toBeCloseTo(plain.w2Grad[i], 6);
  });

  it("checkpointed block gradcheck (finite difference through recomputation)", () => {
    // The checkpoint recomputes forward during backward; the numeric check
    // confirms that recomputed backward matches the true gradient.
    checkGrad((ctx, mk) => {
      const x = mk(rnd(3, 8, -1, 1), [2, 4]);
      const w = mk(rnd(1, 16, -0.5, 0.5), [4, 4]);
      return checkpoint(ctx as any, (ic, inp) => {
        const h = matmul(ic as BuildCtx, inp, w);
        const a = gelu(ic as BuildCtx, h);
        return add(ic as BuildCtx, a, inp);
      }, x);
    }, { label: "checkpoint" });
  });

  it("checkpointed block containing dropout (RNG counter save/restore replay)", () => {
    // Mirrors the gpt.ts activation-checkpointing pattern: save the dropout
    // RNG counter before the block and restore it inside fn, so the backward
    // RECOMPUTATION replays the identical mask. The FD harness re-seeds the
    // dropout RNG per eval, so numeric == analytic holds iff the replayed
    // mask is bit-identical to the forward mask.
    checkGrad((ctx, mk) => {
      const x = mk(rnd(31, 8, -1, 1), [2, 4]);
      const w = mk(rnd(32, 16, -0.5, 0.5), [4, 4]);
      const rng = ctx.dropoutRng!;
      const saved = rng.saveCounter();
      return checkpoint(ctx as any, (ic, inp) => {
        rng.restoreCounter(saved);
        const icd: BuildCtx = { ...(ic as BuildCtx), dropoutRng: rng };
        const h = matmul(icd, inp, w);
        const d = dropout(icd, h, 0.3, true);
        return add(icd, d, inp);
      }, x);
    }, { label: "checkpoint-dropout", dropoutSeed: 21 });
  });
});
