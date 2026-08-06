/* How fast can the host read and write tensor memory?
 *
 * The step is 85 ms and its kernels are 7 ms of it, so the backend is
 * host-bound and the host rows are the ones to explain. layerNorm is the
 * clearest: 5 calls, 12.1 ms, against 6.6 us of GPU work per call. It launches
 * one normalize and then multiplies by a weight and adds a bias, and both of
 * those broadcast -- which expand() does on the HOST, in a JavaScript loop over
 * 2048 elements. 2.4 ms for 2048 elements is roughly a microsecond an element,
 * which no amount of index arithmetic explains.
 *
 * What would explain it is WHERE the memory is. gaia_alloc asks for system
 * memory with ATTR_COHERENCY_WRITE_COMBINE. Write-combined memory is meant for
 * streaming CPU writes that the GPU will read; the CPU READING it bypasses the
 * cache entirely, one transaction per access. Every host-side loop in this
 * backend -- expand, slice, cat, permuteSwap, zeros, fromArray, and every CPU
 * fallback in autograd -- walks exactly this memory.
 *
 * So: the same loops over tensor memory and over an ordinary JavaScript array.
 * If tensor memory is dramatically slower, the host cost is the mapping, not
 * the code, and it is fixable in the allocator rather than in fifteen call
 * sites. */
import { nativeAddon, NativeBuffer } from "/workspace/alpha2/packages/helios/dist/index.js";

const hl = nativeAddon(0);
const N = 2048;
const dev = NativeBuffer.alloc(hl, N);
const host = new Float32Array(N);
const devB = NativeBuffer.alloc(hl, N);
const hostB = new Float32Array(N);

function med(f, reps = 200) {
  for (let i = 0; i < 20; i++) f();
  const xs = [];
  for (let i = 0; i < reps; i++) {
    const t = process.hrtime.bigint();
    f();
    xs.push(Number(process.hrtime.bigint() - t) / 1000);
  }
  xs.sort((a, b) => a - b);
  return xs[Math.floor(reps / 2)];
}
const row = (name, us) =>
  console.log(`${name.padEnd(40)} ${us.toFixed(1).padStart(9)} us   ${(N / us).toFixed(1).padStart(8)} elem/us`);

console.log(`\n${N} elements per operation\n`);

row("read  device -> sum", med(() => { let s = 0; const a = dev.floats; for (let i = 0; i < N; i++) s += a[i]; return s; }));
row("read  plain  -> sum", med(() => { let s = 0; for (let i = 0; i < N; i++) s += host[i]; return s; }));
row("write device", med(() => { const a = dev.floats; for (let i = 0; i < N; i++) a[i] = i; }));
row("write plain", med(() => { for (let i = 0; i < N; i++) host[i] = i; }));
row("copy  device -> device", med(() => { const a = dev.floats, b = devB.floats; for (let i = 0; i < N; i++) b[i] = a[i]; }));
row("copy  plain  -> plain", med(() => { for (let i = 0; i < N; i++) hostB[i] = host[i]; }));
row("fill  device (.fill)", med(() => dev.floats.fill(0, 0, N)));
row("fill  plain  (.fill)", med(() => host.fill(0, 0, N)));
row("set   device <- plain (.set)", med(() => dev.floats.set(host.subarray(0, N), 0)));
row("set   plain  <- plain (.set)", med(() => hostB.set(host.subarray(0, N), 0)));

/* The broadcast expand() actually performs for layerNorm's weight: [64] tiled
 * across [1,32,64], written the way expand writes it. */
console.log();
row("expand [64] -> [1,32,64] on device", med(() => {
  const src = dev.floats, out = devB.floats;
  for (let i = 0; i < N; i++) out[i] = src[i % 64];
}));
row("expand [64] -> [1,32,64] on plain", med(() => {
  for (let i = 0; i < N; i++) hostB[i] = host[i % 64];
}));
