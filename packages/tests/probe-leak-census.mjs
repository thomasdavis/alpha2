/* WHERE is each surviving buffer allocated? The census, for the native pool.
 *
 * probe-leak-by-class says HOW MUCH leaks and at what size; it cannot say from
 * which line, and at a fixed model shape several call sites share a size. This
 * records a JavaScript stack per allocation and drops it on release, then
 * reports what is still held after a step boundary — the same trick the Vulkan
 * backend's HELIOS_LEAK_CENSUS used, where it "named all 203 at once".
 *
 * It is a MEASUREMENT TOOL and not production behaviour: holding a stack per
 * buffer costs memory and defeats the finalizer, which is fine for twenty steps
 * and wrong for a training run.
 *
 * Method note that matters: the pool is legitimately allowed to grow for the
 * first steps while it fills. So the census is RESET after a warmup and read
 * after several more steps — anything appearing in that window survived a step
 * boundary it should not have.
 *
 * Usage: node probe-leak-census.mjs <nLayer> <nEmbd> <nHead> <vocab> <seq> <batch> [steps]
 */
import { NativeHeliosBackend } from "/workspace/alpha2/packages/helios/dist/index.js";
import { NativeBuffer } from "/workspace/alpha2/packages/helios/dist/nativeDevice.js";
import { SeededRng } from "/workspace/alpha2/packages/core/dist/index.js";
import { Tape } from "/workspace/alpha2/packages/autograd/dist/index.js";
import { initGPT, gptForward } from "/workspace/alpha2/packages/model/dist/index.js";

const C = {
  nLayer: Number(process.argv[2] ?? 4),
  nEmbd: Number(process.argv[3] ?? 640),
  nHead: Number(process.argv[4] ?? 10),
  vocabSize: Number(process.argv[5] ?? 12288),
  blockSize: Number(process.argv[6] ?? 64),
  dropout: 0,
};
const BATCH = Number(process.argv[7] ?? 8);
const STEPS = Number(process.argv[8] ?? 8);

/* The census. Keyed by the buffer OBJECT, which the map holds alive on
 * purpose — a WeakMap cannot be walked, and being walkable is the whole
 * point. */
const live = new Map();
let recording = false;

const origAlloc = NativeBuffer.alloc.bind(NativeBuffer);
NativeBuffer.alloc = (hl, elements) => {
  const b = origAlloc(hl, elements);
  if (recording) {
    const st = (new Error().stack ?? "").split("\n").slice(2, 7)
      .map((l) => l.trim().replace(/^at /, "").replace(/.*\/packages\//, "packages/"))
      .filter((l) => !l.includes("probe-leak-census"))
      .join("\n            <- ");
    live.set(b, { elements, st });
  }
  return b;
};
const origRelease = NativeBuffer.prototype.release;
NativeBuffer.prototype.release = function (hl) {
  live.delete(this);
  return origRelease.call(this, hl);
};

const B = new NativeHeliosBackend(0);
const params = initGPT(C, B, new SeededRng(7));

const kept = new Set();
(function walk(v, d) {
  if (!v || typeof v !== "object" || d > 6) return;
  if (v.buffer && v.shape) { kept.add(v); return; }
  if (v.data) kept.add(v.data);
  for (const x of Array.isArray(v) ? v : Object.values(v)) walk(x, d + 1);
})(params, 0);
const rel = (td) => { if (td && !kept.has(td)) B.releaseGpuTensor(td); };

const N = BATCH * C.blockSize;
function step() {
  const tape = new Tape();
  const tok = B.fromArray(Array.from({ length: N }, (_, i) => i % C.vocabSize), [BATCH, C.blockSize]);
  const tgt = B.fromArray(Array.from({ length: N }, (_, i) => (i + 1) % C.vocabSize), [BATCH, C.blockSize]);
  const out = gptForward(C, params, B, tape, tok, tgt, true, false, false, undefined, rel);
  const loss = out.loss.data.data[0];
  tape.backward(out.loss, B, rel);
  B.finishStepOps();
  return loss;
}

/*
 * TWO QUESTIONS, and they have different answers.
 *
 * "survivors" reports what is still held after a step boundary — a LEAK, growth
 * across steps.
 *
 * "all" reports everything allocated during the window whether or not it was
 * freed — the PEAK, which is the quantity that actually bounds the batch here.
 * A release only MARKS in this allocator: the buffer stays valid to any queued
 * launch and to any reader until helios_end_step, so a step's memory is total
 * bytes ALLOCATED and not the live set. A path can leak nothing and still
 * exhaust the card, which is exactly what the fused layerNorm backward does.
 */
const MODE = process.env.CENSUS_MODE ?? "survivors";
const all = [];
if (MODE === "all") {
  const wrapped = NativeBuffer.alloc;
  NativeBuffer.alloc = (hl, elements) => {
    const b = wrapped(hl, elements);
    if (recording) all.push(live.get(b) ?? { elements, st: "(unrecorded)" });
    return b;
  };
}

for (let i = 0; i < 5; i++) step();
recording = true;
live.clear();
for (let i = 0; i < STEPS; i++) step();
recording = false;

/* Group by site, because one leaking line produces hundreds of buffers. */
const bySite = new Map();
for (const { elements, st } of (MODE === "all" ? all : live.values())) {
  const e = bySite.get(st) ?? { n: 0, elements: 0 };
  e.n++; e.elements += elements;
  bySite.set(st, e);
}
const rows = [...bySite.entries()].sort((a, b) => b[1].elements - a[1].elements);

console.log(`${C.nLayer}L ${C.nEmbd}d batch ${BATCH} — ` + (MODE === "all" ? `ALL allocations over ${STEPS} steps` : `buffers surviving ${STEPS} step boundaries`) + "\n");
let totalMiB = 0;
for (const [st, e] of rows.slice(0, 12)) {
  const miB = e.elements * 4 / (1 << 20);
  totalMiB += miB;
  console.log(`  ${e.n} buffers  ${miB.toFixed(2)} MiB  (${(miB / STEPS).toFixed(2)} MiB/step)`);
  console.log(`            ${st}\n`);
}
for (const [, e] of rows.slice(12)) totalMiB += e.elements * 4 / (1 << 20);
console.log(`  ${live.size} buffers held, ${totalMiB.toFixed(1)} MiB total, ` +
            `${(totalMiB / STEPS).toFixed(2)} MiB per step`);
