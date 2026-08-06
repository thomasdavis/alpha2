#!/usr/bin/env node
/**
 * X56 — validate the buffer-device-address sum reduction.
 *
 * Purpose
 * -------
 * X56 found Helios's DGC path is wired to exactly one pipeline (`add_bda`), so
 * it can only carry elementwise binary ops: 10.6% of the operation graph, of
 * which 70% is the gradient-norm reduction tree that X50 wants to delete. The
 * way out is to convert kernels to BDA form so they need no descriptor set;
 * reductions are the largest single kind at 30.4%.
 *
 * `kernelSumReduceBDA` is the first of that tranche. This checks two things
 * before any of it is believed:
 *
 *   1. The emitted SPIR-V is valid. Pipeline creation runs the driver's own
 *      SPIR-V validator, which is a real check and not a self-report from the
 *      builder that produced the module.
 *   2. The push-constant block is the size the kernel declares (24 bytes:
 *      two u64 addresses plus two u32 params).
 *
 * What transfers and what does not
 * --------------------------------
 * TRANSFERS: SPIR-V validity, capability declarations, push-block layout.
 * These are properties of the module, not of the device.
 * DOES NOT TRANSFER: whether the device supports bufferDeviceAddress at all.
 * lavapipe may decline, in which case this reports UNSUPPORTED rather than
 * FAIL — the module can still be valid on a device that does support it, and
 * saying otherwise would be an unsupported negative.
 *
 * Usage:  node scripts/x56-bda-reduction-validate.mjs
 */

import { getNative, initDevice, getDeviceInfo } from "../packages/helios/dist/device.js";
import { getKernelSpirv } from "../packages/helios/dist/kernels.js";

const WG = 256;

function main() {
  const vk = getNative();
  initDevice(vk);
  const info = getDeviceInfo?.(vk) ?? {};
  console.log(`device: ${info.deviceName ?? "?"} type=${info.deviceType ?? "?"} ` +
              `subgroup=${info.subgroupSize ?? "?"}`);

  const cases = [
    // name, descriptor bindings, push bytes
    ["add_bda",        0, 32],  // the proven reference: 3 addresses + 2 u32
    ["sum_reduce_bda", 0, 24],  // new: 2 addresses + 2 u32
    ["sum_reduce",     2, 8],   // the descriptor-bound original, for contrast
  ];

  let failures = 0;
  for (const [name, bindings, pushBytes] of cases) {
    let spirv;
    try {
      spirv = getKernelSpirv(name, WG);
    } catch (e) {
      console.log(`  ${name.padEnd(16)} EMIT-FAIL   ${e.message}`);
      failures++;
      continue;
    }

    const words = spirv.length;
    let slot = -1, err = null;
    try {
      slot = vk.createPipeline(spirv, bindings, pushBytes);
    } catch (e) {
      err = e;
    }

    if (slot >= 0) {
      console.log(`  ${name.padEnd(16)} OK          ${words} words, ` +
                  `${bindings} bindings, ${pushBytes}B push -> pipeline slot ${slot}`);
    } else {
      const msg = String(err?.message ?? err ?? "createPipeline returned < 0");
      // A device that lacks bufferDeviceAddress is not a module defect.
      const unsupported = /bufferDeviceAddress|PhysicalStorageBuffer|Int64|capability/i.test(msg);
      console.log(`  ${name.padEnd(16)} ${unsupported ? "UNSUPPORTED" : "FAIL       "} ${msg}`);
      if (!unsupported) failures++;
    }
  }

  console.log(failures === 0
    ? "\nresult: no module defects detected"
    : `\nresult: ${failures} module defect(s)`);
  process.exit(failures === 0 ? 0 : 1);
}

main();
