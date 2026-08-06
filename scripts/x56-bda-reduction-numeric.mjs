#!/usr/bin/env node
/**
 * X56 — numerical validation of the buffer-device-address sum reduction.
 *
 * The companion script only proves the SPIR-V is valid: pipeline creation runs
 * the driver's validator, which says nothing about whether the kernel computes
 * the right number. This runs it for real and compares against a CPU sum.
 *
 * Three things are under test, and the third is the one that matters most:
 *
 *   1. BDA addressing — the kernel reaches its operands through 64-bit
 *      addresses in push constants rather than a descriptor set. That is what
 *      makes it DGC-eligible and command-buffer replayable (X56).
 *   2. The reduction arithmetic itself, against a CPU reference.
 *   3. The `outOffset` parameter — writing the result into slot i of a shared
 *      buffer. This is the mechanism X50 needs to collapse the 127-dispatch
 *      gradient-norm tree into a single reduction, and it is the reason this
 *      kernel resolves the X49/X50 tension instead of trading one for the other.
 *
 * Tolerance: the GPU reduces in a different order than a sequential CPU sum, so
 * results are compared as relative error against a pairwise CPU reference
 * rather than for bit-equality.
 *
 * Usage:  node scripts/x56-bda-reduction-numeric.mjs
 */

import { getNative, initDevice } from "../packages/helios/dist/device.js";
import { getKernelSpirv } from "../packages/helios/dist/kernels.js";

const WG = 256;
const TOL = 1e-5;

/** Pairwise CPU sum — the reference, matching the GPU's tree shape more closely
 *  than a sequential accumulation would. */
function pairwiseSum(a, lo = 0, hi = a.length) {
  const n = hi - lo;
  if (n === 0) return 0;
  if (n < 128) {
    let s = 0;
    for (let i = lo; i < hi; i++) s += a[i];
    return s;
  }
  const mid = (lo + ((n / 2) | 0));
  return pairwiseSum(a, lo, mid) + pairwiseSum(a, mid, hi);
}

function pushBlock(addrA, addrC, len, outOffset) {
  const buf = new ArrayBuffer(24);
  const dv = new DataView(buf);
  dv.setUint32(0, addrA[0], true); dv.setUint32(4, addrA[1], true);   // u64 A
  dv.setUint32(8, addrC[0], true); dv.setUint32(12, addrC[1], true);  // u64 C
  dv.setUint32(16, len, true);
  dv.setUint32(20, outOffset, true);
  return new Float32Array(buf);   // reinterpreted; the native side memcpy's bytes
}

function main() {
  const vk = getNative();
  initDevice(vk);

  const spirv = getKernelSpirv("sum_reduce_bda", WG);
  const pipe = vk.createPipeline(spirv, 0, 24);
  console.log(`pipeline slot ${pipe} (0 bindings, 24B push)`);

  // ---- Route constraint, discovered the hard way (SIGSEGV) -----------------
  // `dispatch` unconditionally allocates and writes a descriptor set from the
  // pipeline's layout. A BDA kernel declares zero bindings, so that layout has
  // nothing to write into and the native path faults. BDA kernels are therefore
  // reachable ONLY through the DGC executor.
  //
  // And DGC is bound to a single pipeline: napi_dgcSetup stores one global
  // `dgcPipeSlot` and passes it via VkGeneratedCommandsPipelineInfoEXT with
  // indirectExecutionSet = VK_NULL_HANDLE ("simpler for fixed-pipeline DGC").
  // That slot is already taken by `add_bda`.
  //
  // So numerically exercising this kernel needs one of:
  //   (a) VkIndirectExecutionSetEXT, so DGC can switch pipelines mid-stream —
  //       this is the real unlock for the whole BDA tranche; or
  //   (b) a descriptor-skip branch in napi_dispatch when bindingCount == 0.
  //
  // Both are native changes. Until one lands, running the tests below faults
  // the process rather than reporting a failure, so they are gated off by
  // default. Set X56_FORCE_DISPATCH=1 to reproduce the fault deliberately.
  if (process.env.X56_FORCE_DISPATCH !== "1") {
    console.log("\nSKIP: no route to execute a 0-binding pipeline.");
    console.log("  dispatch()  -> writes descriptors from an empty layout (faults)");
    console.log("  dgcExecute() -> fixed to one pipeline, already bound to add_bda");
    console.log("\nBlocked on: VkIndirectExecutionSetEXT, or a bindingCount==0");
    console.log("branch in napi_dispatch. The module itself is valid (see");
    console.log("x56-bda-reduction-validate.mjs); only its execution route is missing.");
    process.exit(0);
  }

  let failures = 0;
  const check = (name, got, want, extra = "") => {
    const rel = Math.abs(got - want) / Math.max(Math.abs(want), 1e-30);
    const ok = rel <= TOL;
    if (!ok) failures++;
    console.log(`  ${ok ? "PASS" : "FAIL"}  ${name.padEnd(34)} got ${got.toFixed(6)} ` +
                `want ${want.toFixed(6)} rel ${rel.toExponential(2)} ${extra}`);
  };

  // ---- Test 1: single-workgroup reduction, outOffset = 0 -------------------
  {
    const n = WG;
    const host = new Float32Array(n);
    for (let i = 0; i < n; i++) host[i] = Math.sin(i * 0.37) * 3.0;

    const inBuf = vk.createBuffer(n * 4);
    const outBuf = vk.createBuffer(4 * 8);
    vk.uploadBuffer(inBuf, host);
    vk.uploadBuffer(outBuf, new Float32Array(8));

    const aAddr = vk.dgcGetBufferAddress(inBuf);
    const cAddr = vk.dgcGetBufferAddress(outBuf);
    vk.dispatch(pipe, [inBuf, outBuf], 1, 1, 1, pushBlock(aAddr, cAddr, n, 0));
    vk.waitIdle();

    const out = vk.readBuffer(outBuf);
    check("single WG, offset 0", out[0], pairwiseSum(host));
    vk.destroyBuffer(inBuf); vk.destroyBuffer(outBuf);
  }

  // ---- Test 2: the outOffset mechanism — X50's slot write ------------------
  // Four independent inputs each reduce into a different slot of ONE buffer,
  // which is exactly how the gradient-norm tree would be collapsed.
  {
    const SLOTS = 4, n = WG;
    const outBuf = vk.createBuffer(SLOTS * 4);
    vk.uploadBuffer(outBuf, new Float32Array(SLOTS));
    const cAddr = vk.dgcGetBufferAddress(outBuf);

    const wants = [];
    const inBufs = [];
    for (let s = 0; s < SLOTS; s++) {
      const host = new Float32Array(n);
      for (let i = 0; i < n; i++) host[i] = Math.cos(i * 0.11 + s) * (s + 1);
      wants.push(pairwiseSum(host));
      const b = vk.createBuffer(n * 4);
      vk.uploadBuffer(b, host);
      inBufs.push(b);
      vk.dispatch(pipe, [b, outBuf], 1, 1, 1,
                  pushBlock(vk.dgcGetBufferAddress(b), cAddr, n, s));
    }
    vk.waitIdle();

    const out = vk.readBuffer(outBuf);
    for (let s = 0; s < SLOTS; s++) {
      check(`outOffset slot ${s}`, out[s], wants[s]);
    }
    for (const b of inBufs) vk.destroyBuffer(b);
    vk.destroyBuffer(outBuf);
  }

  // ---- Test 3: partial workgroup (bounds check / zero padding) -------------
  {
    const n = 100;                        // < WG, exercises the OOB phi path
    const host = new Float32Array(n);
    for (let i = 0; i < n; i++) host[i] = (i % 7) - 3;

    const inBuf = vk.createBuffer(WG * 4);
    const padded = new Float32Array(WG);
    padded.set(host);
    padded.fill(999999, n);               // poison beyond len: must be ignored
    vk.uploadBuffer(inBuf, padded);

    const outBuf = vk.createBuffer(4);
    vk.uploadBuffer(outBuf, new Float32Array(1));
    vk.dispatch(pipe, [inBuf, outBuf], 1, 1, 1,
                pushBlock(vk.dgcGetBufferAddress(inBuf),
                          vk.dgcGetBufferAddress(outBuf), n, 0));
    vk.waitIdle();

    const out = vk.readBuffer(outBuf);
    check("partial WG (len 100, poisoned tail)", out[0], pairwiseSum(host));
    vk.destroyBuffer(inBuf); vk.destroyBuffer(outBuf);
  }

  console.log(failures === 0
    ? "\nresult: BDA reduction is numerically correct, including outOffset"
    : `\nresult: ${failures} failure(s)`);
  process.exit(failures === 0 ? 0 : 1);
}

main();
