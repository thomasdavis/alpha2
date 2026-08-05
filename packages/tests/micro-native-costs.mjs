/* The fixed cost of an operation, decomposed.
 *
 * The step profile shows ~500-900 us per backend call REGARDLESS of the work:
 * reshape launches nothing and costs 496 us, zeros costs 869. A cost that does
 * not vary with the work is not the work. This prices each candidate
 * separately, so the next change is aimed at a measured number rather than at
 * the most plausible-sounding line of code.
 *
 * Everything is a median of many samples: the first allocation of a size class
 * pays for page mapping the rest inherit, and a mean would hide that. */
import { nativeAddon, NativeBuffer } from "/workspace/alpha2/packages/helios/dist/index.js";

const hl = nativeAddon(0);

function med(f, n) {
  const xs = [];
  for (let i = 0; i < n; i++) {
    const t = process.hrtime.bigint();
    f(i);
    xs.push(Number(process.hrtime.bigint() - t) / 1000); /* us */
  }
  xs.sort((a, b) => a - b);
  return { med: xs[Math.floor(n / 2)], lo: xs[0], hi: xs[n - 1] };
}
const show = (name, r) =>
  console.log(`${name.padEnd(34)} ${r.med.toFixed(1).padStart(8)} us  [${r.lo.toFixed(1)}-${r.hi.toFixed(1)}]`);

/* 1. A FRESH allocation: no free buffer of the class exists, so it goes to the
 *    driver -- gaia_alloc, map_gpu, map_host. */
const keep = [];
show("alloc 4KB (fresh, never freed)", med(() => keep.push(NativeBuffer.alloc(hl, 1024)), 200));

/* 2. A RECYCLED allocation: free first so the class has a buffer waiting. The
 *    flush is what retires it onto the free list. */
hl.flush();
for (const b of keep.splice(0, 150)) b.release(hl);
hl.flush();
show("alloc 4KB (recycled from pool)", med(() => {
  const b = NativeBuffer.alloc(hl, 1024);
  b.release(hl);
  hl.flush();
}, 100));

/* 3. free + flush alone, to separate the retire from the allocate above. */
const pool = [];
for (let i = 0; i < 100; i++) pool.push(NativeBuffer.alloc(hl, 1024));
hl.flush();
show("free + flush (empty queue)", med((i) => { pool[i].release(hl); hl.flush(); }, 100));

/* 4. A flush with NOTHING pending: this is what a tensor's `data` getter costs
 *    on a tensor whose kernel has already run, and reshape pays it. */
show("flush (nothing pending)", med(() => hl.flush(), 500));

/* 5. hl.view: the napi call that hands back the ArrayBuffer. */
const h = NativeBuffer.alloc(hl, 1024);
show("view(handle)", med(() => hl.view(h.handle), 500));

/* 6. One kernel enqueued and drained -- the true round trip. */
const a = NativeBuffer.alloc(hl, 1024), b = NativeBuffer.alloc(hl, 1024), o = NativeBuffer.alloc(hl, 1024);
show("elementwise 1024 + flush", med(() => {
  hl.elementwise(hl.op.add, o.handle, a.handle, b.handle, 1024, 0, 0, 0, 0, 0, 0, 0);
  hl.flush();
}, 200));

/* 7. The same kernel ENQUEUED ONLY, amortising one flush over 8. If enqueue is
 *    cheap and flush is dear, batching is worth something; the step profile
 *    says 3.15 enqueues per flush today. */
show("elementwise 1024, 8 enq + 1 flush", med(() => {
  for (let i = 0; i < 8; i++)
    hl.elementwise(hl.op.add, o.handle, a.handle, b.handle, 1024, 0, 0, 0, 0, 0, 0, 0);
  hl.flush();
}, 200));

/* 8. Enqueue with no flush at all, to isolate the enqueue itself. The ring is
 *    HELIOS_RING_SLOTS deep and flushes itself when full, so this is an
 *    average over the ring rather than a pure enqueue -- stated, not hidden. */
show("elementwise 1024, enqueue only", med(() => {
  hl.elementwise(hl.op.add, o.handle, a.handle, b.handle, 1024, 0, 0, 0, 0, 0, 0, 0);
}, 500));
hl.flush();

const s = hl.stats();
console.log(`\nlive ${s.live}  pooled ${s.pooled}  allocations ${s.allocations}  enqueued ${s.enqueued}  flushes ${s.flushes}`);
