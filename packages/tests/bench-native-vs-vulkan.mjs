/* Tokens/sec, measured well enough to decide things by.
 *
 * The previous version took one mean of 20 steps and reported three digits.
 * The same Vulkan binary measured 613, 563 and 247 across runs -- a 2.5x
 * spread -- so it could not distinguish a real 3% win from thermal drift.
 *
 * This warms up until the numbers settle, then reports the MEDIAN of many
 * samples along with the spread, so a difference smaller than the spread is
 * visibly not a difference. */
import { NativeHeliosBackend, HeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";
import { CpuRefBackend } from "/workspace/alpha2/packages/tensor/dist/index.js";
import { SeededRng } from "/workspace/alpha2/packages/core/dist/index.js";
import { Tape } from "/workspace/alpha2/packages/autograd/dist/index.js";
import { initGPT, gptForward } from "/workspace/alpha2/packages/model/dist/index.js";

const C = { vocabSize: 64, blockSize: 32, nLayer: 2, nEmbd: 64, nHead: 4, dropout: 0 };
const TOKENS = C.blockSize;
/*
 * Warm up by TIME, not by step count.
 *
 * This card idles at 210 MHz and runs at 2100 -- a 10x clock swing -- and
 * nvidia-smi cannot lock clocks inside the container, so the only defence is to
 * hold the GPU busy until it has ramped. Ten steps was 0.85 s and not enough:
 * the same Vulkan binary measured 127 tok/s in a cold process and 628 in a warm
 * one, which is a 4.9x error, far larger than any change worth arguing about.
 *
 * Three seconds is where the medians stopped moving. The step floor keeps the
 * warmup meaningful for a backend fast enough to do 3 s in very few steps.
 */
const WARMUP_MS = 3000, WARMUP_MIN = 10, SAMPLES = 25;

function step(B, params) {
  const tape = new Tape();
  const tok = B.fromArray(Array.from({ length: TOKENS }, (_, i) => i % C.vocabSize), [1, TOKENS]);
  const tgt = B.fromArray(Array.from({ length: TOKENS }, (_, i) => (i + 1) % C.vocabSize), [1, TOKENS]);
  const out = gptForward(C, params, B, tape, tok, tgt, true);
  tape.backward(out.loss, B);
  return out.loss.data.data[0];
}

function bench(name, make) {
  let B;
  try { B = make(); } catch (e) { console.log(`${name.padEnd(15)} unavailable`); return null; }
  const params = initGPT(C, B, new SeededRng(7));
  let loss = 0;
  const warmStart = process.hrtime.bigint();
  let warmSteps = 0;
  while (warmSteps < WARMUP_MIN ||
         Number(process.hrtime.bigint() - warmStart) / 1e6 < WARMUP_MS) {
    loss = step(B, params);
    warmSteps++;
  }

  const ms = [];
  for (let i = 0; i < SAMPLES; i++) {
    const t = process.hrtime.bigint();
    loss = step(B, params);
    ms.push(Number(process.hrtime.bigint() - t) / 1e6);
  }
  ms.sort((a, b) => a - b);
  const med = ms[Math.floor(ms.length / 2)];
  const lo = ms[0], hi = ms[ms.length - 1];
  /* The spread is reported because it is what says whether a comparison means
   * anything: two medians closer together than either spread are the same. */
  console.log(
    `${name.padEnd(15)} ${(TOKENS / (med / 1000)).toFixed(0).padStart(6)} tok/s   ` +
    `median ${med.toFixed(1)} ms  [${lo.toFixed(1)}-${hi.toFixed(1)}]  loss ${loss.toFixed(4)}` +
    `  (${warmSteps} warmup steps)`);
  return TOKENS / (med / 1000);
}

/*
 * ONE BACKEND PER PROCESS, when it matters.
 *
 * Running all three in one process is convenient and it is not safe: the
 * native stack holds an open channel and pinned slabs for the rest of the
 * process, and with those held the SAME Vulkan binary measured 142 tok/s where
 * it had measured 627 -- a 4.4x swing in code that had not changed. Whichever
 * direction that interference runs, a comparison drawn from it is worthless.
 *
 * So `node bench-native-vs-vulkan.mjs native|vulkan|cpu` runs exactly one, and
 * the caller compares numbers from separate processes. With no argument it
 * still runs all three, which is fine for a quick look and labelled as such.
 */
const which = process.argv[2];
console.log(`${C.nLayer}L ${C.nEmbd}d ${C.nHead}h, ${TOKENS} tok/step, ${WARMUP_MS}ms warmup + ${SAMPLES} samples` +
            (which ? ` — ${which} only` : " — ALL THREE IN ONE PROCESS, they interfere") + "\n");
const want = (name) => !which || which === name;
const n = want("native") ? bench("helios-native", () => new NativeHeliosBackend(0)) : null;
const v = want("vulkan") ? bench("helios-vulkan", () => new HeliosBackend()) : null;
if (want("cpu")) bench("cpu_ref", () => new CpuRefBackend());
if (n && v) console.log(`\nnative / vulkan = ${(n / v).toFixed(2)}x  ${n > v ? "AHEAD" : "behind"}`);
