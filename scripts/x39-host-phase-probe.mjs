#!/usr/bin/env node
/**
 * X39 — native host-phase probe.
 *
 * Purpose
 * -------
 * X38 rejected JS field packing as the location of the host interval: a 3.57x
 * faster encoder saved 228.36 us, 0.0663% of the measured 344.55 ms. The
 * remaining interval had to be located rather than assumed. This probe drives
 * the real `batchExecuteAll` native path and reports the disjoint phase
 * breakdown accumulated inside it.
 *
 * Why a probe rather than a training run
 * --------------------------------------
 * The trainer refuses to run on llvmpipe by design: the capability guard
 * rejects non-GPU device types and subgroup size 8 against 32-lane kernel
 * layouts. That guard is correct and is not weakened here. But the host
 * interval being localized is command recording, descriptor work, and
 * submission — none of which depends on kernel arithmetic being correct or on
 * the device being a real GPU. So the probe exercises the identical native
 * dispatch path with a synthetic dispatch stream.
 *
 * What transfers and what does not
 * --------------------------------
 * TRANSFERS: the phase decomposition, the per-dispatch call counts, whether a
 * phase is per-batch or per-dispatch, and the relative cost of descriptor
 * allocation versus push descriptors.
 * DOES NOT TRANSFER: absolute microseconds. Mesa's host-side driver cost is not
 * NVIDIA's. Any absolute claim requires a physical run under Phase B.
 *
 * Usage:
 *   HELIOS_HOST_TIMING=1 node scripts/x39-host-phase-probe.mjs [dispatches] [batches]
 */

import { createRequire } from "node:module";
import { getNative, initDevice, getDeviceInfo } from "../packages/helios/dist/device.js";
import { getKernelSpirv } from "../packages/helios/dist/kernels.js";

const DISPATCHES = Number(process.argv[2] ?? 1703); // the real per-step operation count
const BATCHES = Number(process.argv[3] ?? 20);
const BUFS_PER_DISPATCH = 3; // binary op: two inputs, one output — the common case
const ELEMS = 4096;

if (process.env.HELIOS_HOST_TIMING !== "1") {
  console.error("refusing to run: set HELIOS_HOST_TIMING=1 so the native accumulator is active");
  process.exit(2);
}

initDevice();
const vk = getNative();
const info = getDeviceInfo();

if (typeof vk.getHostTiming !== "function") {
  console.error("native addon lacks getHostTiming; rebuild packages/helios (npm run build:native)");
  process.exit(2);
}

// A plain elementwise add: three storage bindings, no push constants beyond the
// element count. This mirrors the most common operation shape in the real graph.
const spirv = getKernelSpirv("add");
const pipeline = vk.createPipeline(spirv, BUFS_PER_DISPATCH, 4);

const bufs = [];
for (let i = 0; i < 8; i++) bufs.push(vk.createBuffer(ELEMS * 4));

// Pack in exactly the layout napi_batchExecuteAllImpl decodes:
//   pipeline i32 | bufCount u16 | flags u16 | groupsX u32 | writeMask u32
//   | buffers i32[bufCount] | pushConstants
const OP_BYTES = 4 + 2 + 2 + 4 + 4 + BUFS_PER_DISPATCH * 4 + 4;
const packed = new ArrayBuffer(OP_BYTES * DISPATCHES);
const dv = new DataView(packed);
let off = 0;
for (let d = 0; d < DISPATCHES; d++) {
  dv.setInt32(off, pipeline, true); off += 4;
  dv.setUint16(off, BUFS_PER_DISPATCH, true); off += 2;
  dv.setUint16(off, 1 << 1, true); off += 2;          // groupsY = 1, no groupsZ
  dv.setUint32(off, Math.ceil(ELEMS / 128), true); off += 4;
  dv.setUint32(off, 0b100, true); off += 4;            // third binding is the write
  // Rotate buffers so the write-tracking and barrier logic behaves realistically
  dv.setInt32(off, bufs[d % 6], true); off += 4;
  dv.setInt32(off, bufs[(d + 1) % 6], true); off += 4;
  dv.setInt32(off, bufs[(d + 2) % 6], true); off += 4;
  dv.setUint32(off, ELEMS, true); off += 4;            // push constant
}

// Warm up so first-touch allocation and pipeline binding are not charged to the
// measured window.
vk.batchExecuteAll(packed, DISPATCHES);
vk.waitIdle();
vk.resetHostTiming();

const wall0 = process.hrtime.bigint();
for (let b = 0; b < BATCHES; b++) vk.batchExecuteAll(packed, DISPATCHES);
vk.waitIdle();
const wallMs = Number(process.hrtime.bigint() - wall0) / 1e6;

const t = vk.getHostTiming();
const rows = Object.entries(t.phases)
  .filter(([, v]) => v.calls > 0)
  .sort((a, b) => b[1].us - a[1].us);
const totalUs = rows.reduce((s, [, v]) => s + v.us, 0);
const ringWaitUs = t.phases.ring_wait?.us ?? 0;
const hostUs = totalUs - ringWaitUs;

// Measure the instrumentation's own cost so it can be subtracted rather than
// assumed negligible.
const CAL = 2_000_000;
const c0 = process.hrtime.bigint();
for (let i = 0; i < CAL; i++) process.hrtime.bigint();
const perReadNs = Number(process.hrtime.bigint() - c0) / CAL;
const overheadUs = (t.clockReads * perReadNs) / 1000;

const out = {
  device: info?.deviceName ?? "unknown",
  dispatchesPerBatch: DISPATCHES,
  batches: BATCHES,
  wallMs,
  nativeBatches: t.batches,
  nativeDispatches: t.dispatches,
  clockReads: t.clockReads,
  estimatedInstrumentationUs: overheadUs,
  totalMeasuredUs: totalUs,
  ringWaitUs,
  hostUsExcludingRingWait: hostUs,
  phases: Object.fromEntries(rows.map(([k, v]) => [k, {
    us: v.us,
    calls: v.calls,
    usPerCall: v.us / v.calls,
    shareOfHost: k === "ring_wait" ? null : v.us / hostUs,
  }])),
};

console.log(`device: ${out.device}`);
console.log(`${BATCHES} batches x ${DISPATCHES} dispatches, wall ${wallMs.toFixed(1)} ms`);
console.log(`native saw ${t.batches} batches / ${t.dispatches} dispatches`);
console.log(`instrumentation cost ~${overheadUs.toFixed(0)} us of ${totalUs.toFixed(0)} us measured ` +
  `(${(100 * overheadUs / Math.max(totalUs, 1)).toFixed(2)}%)`);
console.log("");
console.log("phase            total_us    calls   us/call   %host");
for (const [k, v] of rows) {
  const share = k === "ring_wait" ? "   —  " : `${(100 * v.us / hostUs).toFixed(1)}%`.padStart(6);
  console.log(
    `${k.padEnd(14)} ${v.us.toFixed(0).padStart(10)} ${String(v.calls).padStart(8)} ` +
    `${(v.us / v.calls).toFixed(3).padStart(9)} ${share}`,
  );
}
console.log("");
console.log(`host total excluding ring_wait: ${hostUs.toFixed(0)} us ` +
  `(${(hostUs / (BATCHES * DISPATCHES)).toFixed(3)} us per dispatch)`);
console.log(`ring_wait (GPU completion, NOT host work): ${ringWaitUs.toFixed(0)} us`);

if (process.env.X39_JSON) {
  const require = createRequire(import.meta.url);
  require("node:fs").writeFileSync(process.env.X39_JSON, JSON.stringify(out, null, 2));
  console.log(`\nwrote ${process.env.X39_JSON}`);
}
