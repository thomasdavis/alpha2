#!/usr/bin/env node
/**
 * X58 — does the collapsed gradient norm equal the tree it replaces?
 *
 * X50 found `totalSumOfSquares` spends 127 `add` dispatches summing 128 scalars,
 * because each per-tensor partial lives in its own buffer. X56/X57 built and
 * validated a buffer-device-address reduction with an output-slot offset, which
 * lets each tensor's partial land in slot i of one buffer so a single reduction
 * finishes the job.
 *
 * This checks the only thing that matters before that becomes a default: the
 * collapsed path and the tree must agree, across tensor counts and sizes.
 *
 * The comparison is agreement between the two GPU paths AND against a CPU
 * reference, because the two GPU paths could agree with each other while both
 * being wrong.
 *
 * Usage:  VK_ICD_FILENAMES=... node scripts/x58-sumsq-slot-collapse.mjs
 */

import { HeliosBackend } from "../packages/helios/dist/backend.js";

const TOL = 2e-5;   // f32 reductions in different orders

function makeTensors(counts, size, seed) {
  const out = [];
  let s = seed;
  for (let t = 0; t < counts; t++) {
    const a = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      a[i] = ((s / 0x7fffffff) - 0.5) * 2.0;
    }
    out.push(a);
  }
  return out;
}

function cpuSumSq(arrays) {
  // Pairwise per array, then pairwise across arrays — closer to the GPU's tree
  // shape than a flat sequential accumulation.
  const per = arrays.map((a) => {
    const rec = (lo, hi) => {
      const n = hi - lo;
      if (n < 128) { let s = 0; for (let i = lo; i < hi; i++) s += a[i] * a[i]; return s; }
      const mid = lo + ((n / 2) | 0);
      return rec(lo, mid) + rec(mid, hi);
    };
    return rec(0, a.length);
  });
  const rec2 = (lo, hi) => {
    const n = hi - lo;
    if (n === 1) return per[lo];
    const mid = lo + ((n / 2) | 0);
    return rec2(lo, mid) + rec2(mid, hi);
  };
  return rec2(0, per.length);
}

function run(backend, arrays, size) {
  // fromArray is typed for number[] but does Ctor.from(data), so a Float32Array
  // passes through without materialising a multi-million-element JS array.
  const tds = arrays.map((a) => backend.fromArray(a, [size]));
  const t = backend.totalSumOfSquares(tds);
  return Number(t.data[0]);   // .data materialises the lazy GPU tensor
}

async function main() {
  const cases = [
    { n: 4,   size: 70000, label: "4 tensors x 70K (stride path)" },
    { n: 16,  size: 70000, label: "16 tensors x 70K" },
    { n: 128, size: 70000, label: "128 tensors x 70K (the real shape)" },
    { n: 17,  size: 99991, label: "17 tensors x 99991 (odd count + odd size)" },
  ];

  let failures = 0;
  for (const c of cases) {
    const arrays = makeTensors(c.n, c.size, 12345 + c.n);
    const want = cpuSumSq(arrays);

    process.env.HELIOS_SUMSQ_SLOT_REDUCE = "0";
    const treeBackend = new HeliosBackend();
    const tree = run(treeBackend, arrays, c.size);

    process.env.HELIOS_SUMSQ_SLOT_REDUCE = "1";
    const slotBackend = new HeliosBackend();
    const slot = run(slotBackend, arrays, c.size);

    const relTree = Math.abs(tree - want) / Math.abs(want);
    const relSlot = Math.abs(slot - want) / Math.abs(want);
    const relPair = Math.abs(slot - tree) / Math.abs(tree);
    const ok = relSlot <= TOL && relPair <= TOL;
    if (!ok) failures++;

    console.log(`  ${ok ? "PASS" : "FAIL"}  ${c.label}`);
    console.log(`        cpu ${want.toFixed(4)}  tree ${tree.toFixed(4)} (rel ${relTree.toExponential(2)})` +
                `  slot ${slot.toFixed(4)} (rel ${relSlot.toExponential(2)})  tree-vs-slot ${relPair.toExponential(2)}`);
  }

  console.log(failures === 0
    ? "\nresult: collapsed gradient norm agrees with the tree and with CPU"
    : `\nresult: ${failures} disagreement(s)`);
  process.exit(failures === 0 ? 0 : 1);
}

main();
