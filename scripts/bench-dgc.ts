/**
 * bench-dgc.ts — Benchmark VK_EXT_device_generated_commands dispatch overhead.
 *
 * Tests the DGC path with a BDA-aware add kernel:
 * - Creates a PhysicalStorageBuffer SPIR-V kernel that reads buffer addresses from push constants
 * - Sets up DGC infrastructure (indirect commands layout, execution set, preprocess buffer)
 * - Writes N dispatch sequences into the command stream buffer
 * - Executes with vkCmdExecuteGeneratedCommandsEXT
 * - Compares latency vs regular batchExecuteAll for N tiny dispatches
 */

import { createRequire } from "node:module";
import { join } from "node:path";

const require = createRequire(import.meta.url);

// Load SPIR-V builder from compiled dist (avoids ts-node ESM issues)
const spirvMod = require(join(process.cwd(), "packages/helios/dist/spirv.js"));
const { SpirVBuilder, Op, Capability, AddressingModel, MemoryModel,
  ExecutionModel, ExecutionMode, StorageClass, Decoration, BuiltIn } = spirvMod;

// Load native Vulkan module
const vk: any = require(join(process.cwd(), "packages/helios/native/helios_vk.node"));

// ── Build BDA Add Kernel ─────────────────────────────────────────────────────
//
// Push constants layout (32 bytes):
//   uint64 addrA  (offset 0)
//   uint64 addrB  (offset 8)
//   uint64 addrC  (offset 16)
//   uint32 count  (offset 24)
//   uint32 _pad   (offset 28)
//
// Kernel: C[gid] = A[gid] + B[gid]

function buildBDAAddKernel(): Uint32Array {
  const b = new SpirVBuilder();

  // Capabilities
  b.addCapability(Capability.Shader);
  b.addCapability(Capability.Int64);
  b.addCapability(Capability.PhysicalStorageBufferAddresses);

  // Extension
  b.addExtension("SPV_KHR_physical_storage_buffer");

  // Memory model: PhysicalStorageBuffer64 + GLSL450
  b.setMemoryModel(AddressingModel.PhysicalStorageBuffer64, MemoryModel.GLSL450);

  // Types
  const tVoid = b.id();
  const tF32  = b.id();
  const tU32  = b.id();
  const tU64  = b.id();
  const tBool = b.id();
  const tVec3U32 = b.id();
  const tFnVoid  = b.id();

  b.typeVoid(tVoid);
  b.typeFloat(tF32, 32);
  b.typeInt(tU32, 32, 0);
  b.typeInt(tU64, 64, 0);
  b.typeBool(tBool);
  b.typeVector(tVec3U32, tU32, 3);
  b.typeFunction(tFnVoid, tVoid);

  // RuntimeArray<f32> for buffer reference
  const tRuntimeArrayF32 = b.id();
  b.typeRuntimeArray(tRuntimeArrayF32, tF32);
  b.addDecorate(tRuntimeArrayF32, Decoration.ArrayStride, 4);

  // Buffer struct (for PhysicalStorageBuffer reference)
  const tBufferStruct = b.id();
  b.typeStruct(tBufferStruct, [tRuntimeArrayF32]);
  b.addDecorate(tBufferStruct, Decoration.Block);
  b.addMemberDecorate(tBufferStruct, 0, Decoration.Offset, 0);

  // Pointer to buffer struct in PhysicalStorageBuffer storage class
  const tPtrPSB = b.id();
  b.typePointer(tPtrPSB, StorageClass.PhysicalStorageBuffer, tBufferStruct);

  // Pointer to f32 element in PhysicalStorageBuffer
  const tPtrPSBF32 = b.id();
  b.typePointer(tPtrPSBF32, StorageClass.PhysicalStorageBuffer, tF32);

  // Push constant struct: { u64 addrA, u64 addrB, u64 addrC, u32 count, u32 _pad }
  const tPCStruct = b.id();
  b.typeStruct(tPCStruct, [tU64, tU64, tU64, tU32, tU32]);
  b.addDecorate(tPCStruct, Decoration.Block);
  b.addMemberDecorate(tPCStruct, 0, Decoration.Offset, 0);
  b.addMemberDecorate(tPCStruct, 1, Decoration.Offset, 8);
  b.addMemberDecorate(tPCStruct, 2, Decoration.Offset, 16);
  b.addMemberDecorate(tPCStruct, 3, Decoration.Offset, 24);
  b.addMemberDecorate(tPCStruct, 4, Decoration.Offset, 28);

  // PC pointers
  const tPtrPC = b.id();
  b.typePointer(tPtrPC, StorageClass.PushConstant, tPCStruct);
  const tPtrPCU64 = b.id();
  b.typePointer(tPtrPCU64, StorageClass.PushConstant, tU64);
  const tPtrPCU32 = b.id();
  b.typePointer(tPtrPCU32, StorageClass.PushConstant, tU32);

  // PC variable
  const vPC = b.id();
  b.variable(tPtrPC, vPC, StorageClass.PushConstant);

  // Built-in: GlobalInvocationId
  const tPtrInputVec3 = b.id();
  b.typePointer(tPtrInputVec3, StorageClass.Input, tVec3U32);
  const vGlobalId = b.id();
  b.variable(tPtrInputVec3, vGlobalId, StorageClass.Input);
  b.addDecorate(vGlobalId, Decoration.BuiltIn, BuiltIn.GlobalInvocationId);

  // Constants
  const const0u = b.id();
  b.constant(tU32, const0u, 0);
  const const1u = b.id();
  b.constant(tU32, const1u, 1);
  const const2u = b.id();
  b.constant(tU32, const2u, 2);
  const const3u = b.id();
  b.constant(tU32, const3u, 3);

  // Entry point
  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, 64, 1, 1);

  // Function body
  b.emit(Op.Function, [tVoid, fnMain, 0, tFnVoid]);
  const labelEntry = b.id();
  b.emit(Op.Label, [labelEntry]);

  // Load global ID
  const gidVec = b.id();
  b.emit(Op.Load, [tVec3U32, gidVec, vGlobalId]);
  const gid = b.id();
  b.emit(Op.CompositeExtract, [tU32, gid, gidVec, 0]);

  // Load count from push constants
  const ptrCount = b.id();
  b.emit(Op.AccessChain, [tPtrPCU32, ptrCount, vPC, const3u]);
  const count = b.id();
  b.emit(Op.Load, [tU32, count, ptrCount]);

  // Bounds check: if (gid >= count) return
  const oob = b.id();
  b.emit(Op.UGreaterThanEqual, [tBool, oob, gid, count]);
  const labelBody = b.id();
  const labelEnd = b.id();
  b.emit(Op.SelectionMerge, [labelEnd, 0]);
  b.emit(Op.BranchConditional, [oob, labelEnd, labelBody]);

  b.emit(Op.Label, [labelBody]);

  // Load buffer addresses from push constants
  const ptrAddrA = b.id();
  b.emit(Op.AccessChain, [tPtrPCU64, ptrAddrA, vPC, const0u]);
  const addrA = b.id();
  b.emit(Op.Load, [tU64, addrA, ptrAddrA]);

  const ptrAddrB = b.id();
  b.emit(Op.AccessChain, [tPtrPCU64, ptrAddrB, vPC, const1u]);
  const addrB = b.id();
  b.emit(Op.Load, [tU64, addrB, ptrAddrB]);

  const ptrAddrC = b.id();
  b.emit(Op.AccessChain, [tPtrPCU64, ptrAddrC, vPC, const2u]);
  const addrC = b.id();
  b.emit(Op.Load, [tU64, addrC, ptrAddrC]);

  // Convert addresses to PhysicalStorageBuffer pointers
  const bufPtrA = b.id();
  b.emit(Op.ConvertUToPtr, [tPtrPSB, bufPtrA, addrA]);
  const bufPtrB = b.id();
  b.emit(Op.ConvertUToPtr, [tPtrPSB, bufPtrB, addrB]);
  const bufPtrC = b.id();
  b.emit(Op.ConvertUToPtr, [tPtrPSB, bufPtrC, addrC]);

  // Load A[gid]
  const ptrA = b.id();
  b.emit(Op.AccessChain, [tPtrPSBF32, ptrA, bufPtrA, const0u, gid]);
  const valA = b.id();
  // Aligned load with Aligned operand (4 bytes) — required for PSB
  b.emit(Op.Load, [tF32, valA, ptrA, 2 /*Aligned*/, 4]);

  // Load B[gid]
  const ptrB = b.id();
  b.emit(Op.AccessChain, [tPtrPSBF32, ptrB, bufPtrB, const0u, gid]);
  const valB = b.id();
  b.emit(Op.Load, [tF32, valB, ptrB, 2 /*Aligned*/, 4]);

  // C[gid] = A[gid] + B[gid]
  const sum = b.id();
  b.emit(Op.FAdd, [tF32, sum, valA, valB]);

  const ptrC = b.id();
  b.emit(Op.AccessChain, [tPtrPSBF32, ptrC, bufPtrC, const0u, gid]);
  b.emit(Op.Store, [ptrC, sum, 2 /*Aligned*/, 4]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

// ── Main benchmark ──────────────────────────────────────────────────────────

async function main() {
  const info = vk.initDevice();
  console.log(`Device: ${info.deviceName}`);
  console.log(`BDA: ${info.hasBDA}, DGC: ${info.hasDGC}`);

  if (!info.hasBDA) {
    console.error("BDA not supported, cannot test DGC");
    vk.destroy();
    process.exit(1);
  }

  const dgcInfo = vk.dgcInfo();
  console.log(`DGC properties:`, JSON.stringify(dgcInfo, null, 2));

  // Build BDA add kernel
  console.log("\nBuilding BDA add kernel...");
  const spirv = buildBDAAddKernel();
  console.log(`SPIR-V: ${spirv.length} words`);

  // Create pipeline: 0 descriptor bindings, 32-byte push constant
  let pipeSlot: number;
  try {
    pipeSlot = vk.createPipeline(spirv, 0, 32);
    console.log(`Pipeline created: slot ${pipeSlot}`);
  } catch (e: any) {
    console.error(`Pipeline creation failed: ${e.message}`);
    vk.destroy();
    process.exit(1);
  }

  // Create test buffers (64 elements = 256 bytes each)
  const N = 64;
  const bytes = N * 4;
  const bufA = vk.createBuffer(bytes, 0); // device-local
  const bufB = vk.createBuffer(bytes, 0);
  const bufC = vk.createBuffer(bytes, 0);

  // Upload test data
  const dataA = new Float32Array(N);
  const dataB = new Float32Array(N);
  for (let i = 0; i < N; i++) { dataA[i] = i; dataB[i] = 100; }
  vk.uploadBuffer(bufA, dataA);
  vk.uploadBuffer(bufB, dataB);

  // Get device addresses
  const addrA = vk.dgcGetBufferAddress(bufA);
  const addrB = vk.dgcGetBufferAddress(bufB);
  const addrC = vk.dgcGetBufferAddress(bufC);
  console.log(`Buffer addresses: A=[${addrA}] B=[${addrB}] C=[${addrC}]`);

  if (!info.hasDGC) {
    console.log("\nDGC not available, skipping DGC benchmark");
    vk.destroy();
    process.exit(0);
  }

  // Set up DGC: pushSize=32 bytes, maxSequences=256
  console.log("\nSetting up DGC...");
  try {
    vk.dgcSetup(pipeSlot, 32, 256);
  } catch (e: any) {
    console.error(`dgcSetup failed: ${e.message}`);
    vk.destroy();
    process.exit(1);
  }

  const dgcInfo2 = vk.dgcInfo();
  console.log(`DGC state: stride=${dgcInfo2.stride} maxSeq=${dgcInfo2.maxSequences}`);

  // Get the command buffer for writing sequences
  const cmdBuf = vk.dgcGetCommandBuffer();
  const cmdView = new DataView(cmdBuf);
  const stride = dgcInfo2.stride; // should be 32 + 12 = 44

  // Write a single test sequence: add A + B -> C with N elements
  // Push constants: addrA(u64), addrB(u64), addrC(u64), count(u32), pad(u32)
  // Dispatch: groupsX=ceil(N/64), groupsY=1, groupsZ=1
  function writeSequence(idx: number, addrA: number[], addrB: number[], addrC: number[], count: number) {
    const off = idx * stride;
    // u64 addrA (lo, hi as little-endian)
    cmdView.setUint32(off + 0, addrA[0], true);
    cmdView.setUint32(off + 4, addrA[1], true);
    // u64 addrB
    cmdView.setUint32(off + 8, addrB[0], true);
    cmdView.setUint32(off + 12, addrB[1], true);
    // u64 addrC
    cmdView.setUint32(off + 16, addrC[0], true);
    cmdView.setUint32(off + 20, addrC[1], true);
    // u32 count
    cmdView.setUint32(off + 24, count, true);
    // u32 pad
    cmdView.setUint32(off + 28, 0, true);
    // Dispatch: groupsX, groupsY, groupsZ
    cmdView.setUint32(off + 32, Math.ceil(count / 64), true);
    cmdView.setUint32(off + 36, 1, true);
    cmdView.setUint32(off + 40, 1, true);
  }

  // Write 1 sequence and test correctness
  writeSequence(0, addrA, addrB, addrC, N);
  console.log("\nExecuting 1 DGC sequence...");
  const tv = vk.dgcExecute(1);
  vk.waitTimeline(tv);

  // Read back result
  const result = vk.readBuffer(bufC, bytes);
  let correct = true;
  for (let i = 0; i < Math.min(10, N); i++) {
    if (Math.abs(result[i] - (i + 100)) > 0.01) {
      console.error(`FAIL: C[${i}] = ${result[i]}, expected ${i + 100}`);
      correct = false;
    }
  }
  console.log(correct ? "Correctness: PASS" : "Correctness: FAIL");

  // ── Benchmark: DGC N dispatches vs regular N dispatches ──
  const numDispatches = 200;
  const iters = 50;
  const warmup = 5;

  console.log(`\n--- Benchmark: ${numDispatches} sequential adds (${iters} iters, ${warmup} warmup) ---`);

  // Write N sequences for DGC benchmark
  for (let i = 0; i < numDispatches; i++) {
    writeSequence(i, addrA, addrB, addrC, N);
  }

  // Warmup DGC
  for (let i = 0; i < warmup; i++) {
    const t = vk.dgcExecute(numDispatches);
    vk.waitTimeline(t);
  }

  // Benchmark DGC
  const dgcTimes: number[] = [];
  for (let i = 0; i < iters; i++) {
    const start = performance.now();
    const t = vk.dgcExecute(numDispatches);
    vk.waitTimeline(t);
    dgcTimes.push(performance.now() - start);
  }

  // Sort and take median
  dgcTimes.sort((a, b) => a - b);
  const dgcMedian = dgcTimes[Math.floor(dgcTimes.length / 2)];
  const dgcMin = dgcTimes[0];
  console.log(`DGC:     median=${dgcMedian.toFixed(3)}ms  min=${dgcMin.toFixed(3)}ms  per-dispatch=${(dgcMedian / numDispatches * 1000).toFixed(1)}µs`);

  // Now benchmark regular dispatch path for comparison
  // We need the regular add kernel for this... skip if no descriptor-set pipeline available
  console.log(`\n(Regular dispatch comparison would need descriptor-set pipeline)`);
  console.log(`For reference: launch_overhead_200_adds in bench-ops is ~47ms (0.234ms/add)`);
  console.log(`CUDA reference: ~1.8ms (0.009ms/add)`);

  vk.destroy();
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
