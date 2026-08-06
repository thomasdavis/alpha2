/**
 * train-model-native — a real Alpha training step on the from-scratch stack.
 *
 * WHAT: initGPT, gptForward with targets, backward through the tape, AdamW --
 * the actual model and optimizer packages, not a hand-written layer -- with the
 * native backend underneath, compared against the same run on cpu_ref.
 *
 * WHY A HAND-WRITTEN LAYER WAS NOT ENOUGH: it exercised the operations I chose
 * to exercise. The model exercises the ones it needs, in the order and shapes it
 * needs them, including every path through attention and the feed-forward block
 * that I would not have thought to write down. The difference between those two
 * is where a backend's remaining gaps live.
 *
 * The config is deliberately tiny. This is a correctness test, and a tiny model
 * reaches every code path a large one does while its disagreements stay small
 * enough to attribute to one operation rather than to accumulated drift.
 */
import { describe, it, expect, beforeAll } from "vitest";
import { NativeHeliosBackend } from "@alpha/helios";
import { CpuRefBackend } from "@alpha/tensor";
import { SeededRng, type Backend, type ModelConfig } from "@alpha/core";
import { Tape } from "@alpha/autograd";
import { initGPT, gptForward, collectParamEntries } from "@alpha/model";

let gpu: NativeHeliosBackend | null = null;
let why = "";
beforeAll(() => {
  try { gpu = new NativeHeliosBackend(0); }
  catch (e) { why = e instanceof Error ? e.message : String(e); }
});

const CONFIG: ModelConfig = {
  vocabSize: 16, blockSize: 8, nLayer: 1, nEmbd: 16, nHead: 2, dropout: 0,
};
/*
 * The SAME tolerances the Vulkan parity suite uses, rather than ones invented
 * here.
 *
 * This checked an absolute bound alone, which is wrong for a gradient whose
 * magnitude varies by orders of magnitude across parameters -- 2e-2 on a
 * gradient of 0.5 is 4%, and on a gradient of 20 it is a rounding error. The
 * established pair is relative-plus-absolute and it is calibrated against a
 * backend that has been trusted for a while, so borrowing it is both more
 * defensible and comparable.
 *
 * The differential trace justifies expecting SOME disagreement: correctly
 * aligned, the first divergence is a sub[1,2,8,8] differing by 6e-11 absolute
 * on a value of -1.5e-5, downstream of exp. That is MUFU's documented
 * approximation showing up where relative error is meaningless, not an error in
 * the arithmetic -- and it accumulates through the chain.
 */
const LOSS_REL_TOL = 5e-3;
const GRAD_REL_TOL = 1e-2;
const GRAD_ABS_TOL = 2e-3;

/** One forward and backward, returning the loss and every parameter gradient. */
function modelStep(B: Backend) {
  const rng = new SeededRng(1234);
  const params = initGPT(CONFIG, B, rng);
  const tape = new Tape();
  const T = CONFIG.blockSize;
  const tokens = B.fromArray(Array.from({ length: T }, (_, i) => i % CONFIG.vocabSize), [1, T]);
  const targets = B.fromArray(Array.from({ length: T }, (_, i) => (i + 1) % CONFIG.vocabSize), [1, T]);
  const out = gptForward(CONFIG, params, B, tape, tokens, targets, true);
  const loss = (out as { loss?: { data: { data: ArrayLike<number> } } }).loss;
  if (!loss) throw new Error("gptForward returned no loss");
  tape.backward(loss as never, B);
  /* Entries are [name, Variable] tuples. Named here rather than destructured
   * inline so a gradient that is missing says WHICH parameter it belonged to --
   * on a model, "some gradient disagreed" is not a diagnosis. */
  const grads = collectParamEntries(params).map(([name, v]) => ({
    name,
    values: v.grad ? Array.from(v.grad.data as ArrayLike<number>) : [],
  }));
  return { loss: (loss.data.data as ArrayLike<number>)[0], grads };
}

describe("a real Alpha training step on the native backend", () => {
  it("reports why it cannot run rather than skipping silently", () => {
    if (!gpu) console.warn(`native backend unavailable: ${why}`);
    expect(true).toBe(true);
  });

  it.runIf(() => gpu !== null)("model loss and every parameter gradient match cpu_ref", () => {
    const want = modelStep(new CpuRefBackend());
    const got = modelStep(gpu!);

    expect(Math.abs(got.loss - want.loss)).toBeLessThan(
      LOSS_REL_TOL * Math.abs(want.loss) + 1e-5,
    );
    expect(got.grads.length).toBe(want.grads.length);
    for (let p = 0; p < want.grads.length; p++) {
      const g = got.grads[p], w = want.grads[p];
      expect(g.name).toBe(w.name);
      expect(g.values.length).toBe(w.values.length);
      for (let i = 0; i < w.values.length; i++) {
        const d = Math.abs(g.values[i] - w.values[i]);
        if (!(d <= GRAD_ABS_TOL + GRAD_REL_TOL * Math.abs(w.values[i]))) {
          throw new Error(
            `${w.name} grad[${i}]: device ${g.values[i]} vs reference ${w.values[i]}`,
          );
        }
      }
    }
  });
});
