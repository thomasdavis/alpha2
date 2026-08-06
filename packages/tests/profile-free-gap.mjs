/* Which allocations never see hl.free?
 *
 * Usage: HELIOS_VIDMEM=1 node profile-free-gap.mjs
 *
 * READ BOTH NUMBERS TOGETHER or this instrument will lie to you. It hooks the
 * allocator and hl.free and diffs them, AND hooks release() to see whether the
 * free actually ran. At 18 layers the answer is 216 class-11 allocations a
 * step, of which 108 are released and freed correctly and 108 never have
 * release() called at all. Reading only the first hook makes it look as though
 * an explicit release is failing; it is not — half the allocations at those
 * sites simply never reach one.
 *
 * The pool says 216 class-11 slots a step are never freed. The site census says
 * they come from lines that DO call release. Both cannot be right, so hook the
 * two calls that matter — the allocator and the free — and diff them by size. */
import { NativeHeliosBackend, NativeBuffer } from "/workspace/alpha2/packages/helios/dist/index.js";
import { SeededRng } from "/workspace/alpha2/packages/core/dist/index.js";
import { Tape } from "/workspace/alpha2/packages/autograd/dist/index.js";
import { initGPT, gptForward } from "/workspace/alpha2/packages/model/dist/index.js";
const L = 18, SEQ = 64, V = 12288, D = 640, H = 10;
const C = { vocabSize: V, blockSize: SEQ, nLayer: L, nEmbd: D, nHead: H, dropout: 0 };
const B = new NativeHeliosBackend(0);
const hl = B.hl;

const alloc = new Map();   /* handle -> [elements, stack] */
let tracking = false;
const origAlloc = NativeBuffer.alloc.bind(NativeBuffer);
NativeBuffer.alloc = (h, n) => {
  const b = origAlloc(h, n);
  if (tracking && n * 4 > 4 * 1048576 && n * 4 <= 8 * 1048576) {
    const st = (new Error().stack ?? "").split("\n").slice(2, 6)
      .map((l) => l.trim().replace(/^at /, "").replace(/.*\/packages\//, "packages/").replace(/ \(.*/, ""))
      .filter((l) => !l.includes("probe-freegap")).join(" <- ");
    alloc.set(b.handle, [n, st]);
  }
  return b;
};
/* What is the refcount when release() is called on a class-11 buffer? If it is
 * above one, something retained it and the release only decrements. */
const rcSeen = new Map();
const origRel = NativeBuffer.prototype.release;
let releasedButNotFreed = 0, releasedAndFreed = 0, sameAddon = 0, otherAddon = 0;
NativeBuffer.prototype.release = function (h) {
  const handle = this.handle;
  const watched = tracking && alloc.has(handle);
  const rcBefore = this.rc;
  const r = origRel.call(this, h);
  if (watched) {
    rcSeen.set(rcBefore, (rcSeen.get(rcBefore) ?? 0) + 1);
    if (h === hl) sameAddon++; else otherAddon++;
    /* The decisive question: release returned — is the handle gone from the
     * map, i.e. did hl.free actually run for it? */
    if (alloc.has(handle)) releasedButNotFreed++; else releasedAndFreed++;
  }
  return r;
};
const origFree = hl.free.bind(hl);
hl.free = (handle) => { if (tracking) alloc.delete(handle); return origFree(handle); };

const P = initGPT(C, B, new SeededRng(7));
const kept = new Set();
(function walk(v, d) { if (!v || typeof v !== "object" || d > 6) return;
  if (v.buffer && v.shape) { kept.add(v); return; }
  if (v.data) kept.add(v.data);
  for (const x of Array.isArray(v) ? v : Object.values(v)) walk(x, d + 1); })(P, 0);
const rel = (td) => { if (td && !kept.has(td)) B.releaseGpuTensor(td); };
const pv = [];
(function w2(v, d) { if (!v || typeof v !== "object" || d > 6) return;
  if (v.requiresGrad !== undefined && v.data) { pv.push(v); return; }
  for (const x of Array.isArray(v) ? v : Object.values(v)) w2(x, d + 1); })(P, 0);

function step() {
  const tape = new Tape();
  const tok = B.fromArray(Array.from({length:SEQ},(_,i)=>i%V),[1,SEQ]);
  const tgt = B.fromArray(Array.from({length:SEQ},(_,i)=>(i+1)%V),[1,SEQ]);
  const out = gptForward(C, P, B, tape, tok, tgt, true, false, false, undefined, rel);
  tape.backward(out.loss, B, rel);
  for (const v of pv) { if (v.grad) { rel(v.grad); v.grad = null; } }
  tape.clear(rel); B.finishStepOps?.();
}
step();
tracking = true; alloc.clear();
step();
tracking = false;

const bySite = new Map();
for (const [, [n, st]] of alloc) {
  const e = bySite.get(st) ?? { n: 0, mb: 0 };
  e.n++; e.mb += n * 4 / 1048576; bySite.set(st, e);
}
console.log(`class-11 allocations in one step that never saw hl.free: ${alloc.size}`);
console.log(`release() on class-11: refcounts ${JSON.stringify([...rcSeen])}`);
console.log(`  after release returned: freed ${releasedAndFreed}, STILL UNFREED ${releasedButNotFreed}`);
console.log(`  addon identity: same ${sameAddon}, different ${otherAddon}\n`);
for (const [st, v] of [...bySite.entries()].sort((a,b) => b[1].n - a[1].n).slice(0, 6))
  console.log(`${String(v.n).padStart(4)}  ${v.mb.toFixed(1).padStart(7)} MB   ${st}`);
