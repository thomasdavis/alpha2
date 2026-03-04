#!/usr/bin/env npx tsx
/**
 * Dump flash attention coop2 SPIR-V for validation with spirv-val.
 */
import { kernelFlashAttentionCoop2Forward } from "../packages/helios/src/kernels/attention-coop2.js";
import { writeFileSync } from "fs";

const modes = ["full", "kv_only", "kv_synth", "per_elem_only"] as const;
for (const mode of modes) {
  try {
    const spirv = kernelFlashAttentionCoop2Forward(
      16, 16, 64,
      mode,
      "workgroup",
      true,   // useSoftCap
      30,     // softCapConst
      true,   // useF16Input
      false,  // skipLseWrite
      2,      // qTilesPerWG
      128,    // localSize
      false,  // doubleBuf
    );
    const filename = `/tmp/flash_coop2_${mode}.spv`;
    writeFileSync(filename, Buffer.from(spirv.buffer));
    console.log(`${mode}: wrote ${filename} (${spirv.byteLength} bytes)`);
  } catch (e: any) {
    console.error(`${mode}: GENERATION FAILED: ${e.message}`);
  }
}
