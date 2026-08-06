/* Who frees the [1,4,32,32] gradient that `add` then reads?
 *
 * With release wired to tape.backward the step dies with
 *   "add was given RELEASED operand(s) #0 [1,4,32,32]"
 * -- the attention-scores gradient accumulator, freed while still referenced.
 * The tape guards its own double-releases with a Set, so it cannot be releasing
 * the same TENSOR twice; something is releasing the same BUFFER twice through
 * two different tensors, or the backend is freeing one internally.
 *
 * So watch the buffer rather than the tensor. Every release of a tensor of that
 * shape is logged with the stack that asked for it, and the reference count
 * after. The last line before the failure is the culprit, and the count says
 * whether it was one release too many or one retain too few. */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";
import { SeededRng } from "/workspace/alpha2/packages/core/dist/index.js";
import { Tape } from "/workspace/alpha2/packages/autograd/dist/index.js";
import { initGPT, gptForward } from "/workspace/alpha2/packages/model/dist/index.js";

const C = { vocabSize: 64, blockSize: 32, nLayer: 2, nEmbd: 64, nHead: 4, dropout: 0 };
const TOKENS = C.blockSize;
const WATCH = "[1,4,32,32]";

const B = new NativeHeliosBackend(0);
const params = initGPT(C, B, new SeededRng(7));

const kept = new Set();
(function walk(v, d) {
  if (!v || typeof v !== "object" || d > 6) return;
  if (v.buffer && v.shape) { kept.add(v); return; }
  if (v.data) kept.add(v.data);
  for (const x of Array.isArray(v) ? v : Object.values(v)) walk(x, d + 1);
})(params, 0);

/* Every release of a watched shape, with who asked and what the count became.
 * `rc` is private, so it is read back through the only thing that exposes it:
 * whether the buffer now reports itself released. */
const log = [];
const origRelease = B.release.bind(B);
B.release = (t) => {
  const watched = t && t.shape && t.buffer && JSON.stringify(t.shape) === WATCH;
  const before = watched ? t.buffer.released : null;
  origRelease(t);
  if (watched) {
    const site = (new Error().stack ?? "").split("\n").slice(2, 5)
      .map((l) => l.trim().replace(/^at /, "").replace(/.*\/packages\//, ""))
      .join(" <- ");
    log.push(`${before ? "ALREADY-DEAD" : t.buffer.released ? "freed now  " : "still live "}  ${site}`);
  }
};

/* Also watch the backend handing the same BUFFER out under a new tensor, which
 * is what reshape does and the only way one buffer gets two owners. */
const origReshape = B.reshape.bind(B);
B.reshape = (a, shape) => {
  const r = origReshape(a, shape);
  if (JSON.stringify(a.shape) === WATCH || JSON.stringify(shape) === WATCH)
    log.push(`RESHAPE ${JSON.stringify(a.shape)} -> ${JSON.stringify(shape)} (buffer now shared)`);
  return r;
};

const release = (td) => { if (td && !kept.has(td)) B.release(td); };

try {
  for (let i = 0; i < 3; i++) {
    log.length = 0;
    const tape = new Tape();
    const tok = B.fromArray(Array.from({ length: TOKENS }, (_, j) => j % C.vocabSize), [1, TOKENS]);
    const tgt = B.fromArray(Array.from({ length: TOKENS }, (_, j) => (j + 1) % C.vocabSize), [1, TOKENS]);
    const out = gptForward(C, params, B, tape, tok, tgt, true);
    tape.backward(out.loss, B, release);
    console.log(`step ${i} ok, ${log.length} events on ${WATCH}`);
  }
} catch (e) {
  console.log(`\nFAILED: ${e.message}\n`);
  console.log(`last ${Math.min(log.length, 14)} events on a ${WATCH} buffer:\n`);
  for (const l of log.slice(-14)) console.log("  " + l);
}
