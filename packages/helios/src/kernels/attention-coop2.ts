/**
 * kernels/attention-coop2.ts — Cooperative matrix flash attention v2 using
 * VK_NV_cooperative_matrix2 (SPV_NV_cooperative_matrix2).
 *
 * Eliminates shmem round-trips for softmax by using:
 *   - OpCooperativeMatrixReduceNV for row max / row sum
 *   - OpCooperativeMatrixPerElementOpNV for scale+mask, exp
 *   - OpFConvert (with CooperativeMatrixConversionsNV) for f32 Acc → f16 MatrixA
 *   - Standard FMul/FAdd/FDiv on cooperative matrices for arithmetic
 *
 * Architecture:
 *   - Br=16, Bc=16, D=64, WG configurable (default 64; 2 subgroups of 32)
 *   - S = Q @ K^T as coop accumulator (f32, 16x16)
 *   - Softmax entirely in cooperative matrix registers (no sS, sP, sDeltaO)
 *   - O accumulated as 4 coop accumulators (16x16 f32 each, covering D=64)
 *   - m, l tracked as coop accumulators (16x16 f32, broadcast across columns)
 *   - Only 2 barriers in hot loop: after Q load, after K/V load per kBlock
 *
 * Requires SPIR-V 1.6 and VK_NV_cooperative_matrix2.
 *
 * Bindings (same as attention-coop.ts):
 *   0: Q [B*H, T, D] f32 (readonly)
 *   1: K [B*H, T, D] f32 (readonly)
 *   2: V [B*H, T, D] f32 (readonly)
 *   3: O [B*H, T, D] f32 (write)
 *   4: LSE [B*H, T]  f32 (write)
 *
 * Push constants: { T, scale, _pad0, _pad1 }
 * Dispatch: (ceil(T/Br), B*H, 1)
 */

import {
  SpirVBuilder, Op, Capability, CooperativeMatrixUse, CooperativeMatrixReduce,
  VERSION_1_6, AddressingModel, MemoryModel as MemModelConst, ExecutionModel,
  ExecutionMode, StorageClass, Decoration, BuiltIn, FunctionControl, Scope,
  MemorySemantics, GLSLstd450, declareParamsPushConstant,
} from "./helpers.js";

function declareF32Ssbo(
  b: SpirVBuilder, tF32: number, set: number, binding: number,
  readonly_: boolean, writeonly_ = false,
): { varId: number; tPtr: number } {
  const tRuntimeArr = b.id();
  b.typeRuntimeArray(tRuntimeArr, tF32);
  b.addDecorate(tRuntimeArr, Decoration.ArrayStride, 4);
  const tStruct = b.id();
  b.typeStruct(tStruct, [tRuntimeArr]);
  b.addDecorate(tStruct, Decoration.Block);
  b.addMemberDecorate(tStruct, 0, Decoration.Offset, 0);
  if (readonly_) b.addMemberDecorate(tStruct, 0, Decoration.NonWritable);
  if (writeonly_) b.addMemberDecorate(tStruct, 0, Decoration.NonReadable);
  const tPtrStruct = b.id();
  b.typePointer(tPtrStruct, StorageClass.StorageBuffer, tStruct);
  const tPtrF32 = b.id();
  b.typePointer(tPtrF32, StorageClass.StorageBuffer, tF32);
  const varId = b.id();
  b.variable(tPtrStruct, varId, StorageClass.StorageBuffer);
  b.addDecorate(varId, Decoration.DescriptorSet, set);
  b.addDecorate(varId, Decoration.Binding, binding);
  return { varId, tPtr: tPtrF32 };
}

function declareF16Ssbo(
  b: SpirVBuilder, tF16: number, set: number, binding: number,
  readonly_: boolean, writeonly_ = false,
): { varId: number; tPtr: number } {
  const tRuntimeArr = b.id();
  b.typeRuntimeArray(tRuntimeArr, tF16);
  b.addDecorate(tRuntimeArr, Decoration.ArrayStride, 2);
  const tStruct = b.id();
  b.typeStruct(tStruct, [tRuntimeArr]);
  b.addDecorate(tStruct, Decoration.Block);
  b.addMemberDecorate(tStruct, 0, Decoration.Offset, 0);
  if (readonly_) b.addMemberDecorate(tStruct, 0, Decoration.NonWritable);
  if (writeonly_) b.addMemberDecorate(tStruct, 0, Decoration.NonReadable);
  const tPtrStruct = b.id();
  b.typePointer(tPtrStruct, StorageClass.StorageBuffer, tStruct);
  const tPtrF16 = b.id();
  b.typePointer(tPtrF16, StorageClass.StorageBuffer, tF16);
  const varId = b.id();
  b.variable(tPtrStruct, varId, StorageClass.StorageBuffer);
  b.addDecorate(varId, Decoration.DescriptorSet, set);
  b.addDecorate(varId, Decoration.Binding, binding);
  return { varId, tPtr: tPtrF16 };
}

export type FlashAttentionCoop2Mode = "full" | "qk" | "qk_mask" | "qk_softmax" | "pv" | "kv_only" | "kv_synth" | "per_elem_only";
export type FlashAttentionCoop2Scope = "subgroup" | "workgroup";

export function kernelFlashAttentionCoop2Forward(
  Br: number,
  Bc: number,
  D: number,
  mode: FlashAttentionCoop2Mode = "full",
  scopeMode: FlashAttentionCoop2Scope = "workgroup",
  useSoftCap = false,
  softCapConst: number | null = null,
  useF16Input = false,
  skipLseWrite = false,
  qTilesPerWG = 1,
  localSize = 64,
  doubleBuf = false,
): Uint32Array {
  const WG = localSize;
  const qTiles = Math.max(1, Math.trunc(qTilesPerWG));
  const coopM = 16, coopN = 16, coopK = 16;
  const kSteps = D / coopK;  // D=64 → 4
  const pvTiles = D / coopN; // D=64 → 4
  const useWorkgroupScope = scopeMode === "workgroup";
  const isFullMode = mode === "full";
  const isQKOnlyMode = mode === "qk";
  const isQKMaskMode = mode === "qk_mask";
  const isQKSoftmaxMode = mode === "qk_softmax";
  const isPVOnlyMode = mode === "pv";
  const isKVOnlyMode = mode === "kv_only";
  const isKVSynthMode = mode === "kv_synth";
  const isPerElemOnlyMode = mode === "per_elem_only";
  const useSoftCapConst = useSoftCap && softCapConst !== null && Number.isFinite(softCapConst);
  const useManualSoftmax = false;

  // ── Compile-time shift/mask for power-of-2 index math ──────────────────
  function log2PoT(n: number): number {
    if (n <= 0 || (n & (n - 1)) !== 0) return -1;
    return 31 - Math.clz32(n);
  }
  const log2D = log2PoT(D);
  const log2Bc = log2PoT(Bc);
  const log2ThreadsPerRow = log2PoT(Math.max(1, Math.floor(localSize / Br)));
  const log2QElemsPerThread = log2PoT((Br * D) / localSize);
  const log2KVElemsPerThread = log2PoT((Bc * D) / localSize);

  const b = new SpirVBuilder(VERSION_1_6);

  // ── Capabilities & extensions ────────────────────────────────────────────
  b.addCapability(Capability.Shader);
  b.addCapability(Capability.Float16);
  b.addCapability(Capability.VulkanMemoryModel);
  b.addCapability(Capability.CooperativeMatrixKHR);
  b.addCapability(Capability.StorageBufferStorageClass);
  b.addCapability(Capability.GroupNonUniform);
  b.addCapability(Capability.GroupNonUniformArithmetic);
  b.addCapability(Capability.GroupNonUniformClustered);
  b.addCapability(Capability.CooperativeMatrixReductionsNV);
  b.addCapability(Capability.CooperativeMatrixConversionsNV);
  b.addCapability(Capability.CooperativeMatrixPerElementOperationsNV);

  b.addExtension("SPV_KHR_cooperative_matrix");
  b.addExtension("SPV_KHR_vulkan_memory_model");
  b.addExtension("SPV_KHR_storage_buffer_storage_class");
  b.addExtension("SPV_NV_cooperative_matrix2");

  const glslStd = b.id();
  b.addExtInstImport(glslStd, "GLSL.std.450");
  b.setMemoryModel(AddressingModel.Logical, MemModelConst.Vulkan);

  // ── Types ────────────────────────────────────────────────────────────────
  const tVoid = b.id(); b.typeVoid(tVoid);
  const tF32  = b.id(); b.typeFloat(tF32, 32);
  const tF16  = b.id(); b.typeFloat(tF16, 16);
  const tU32  = b.id(); b.typeInt(tU32, 32, 0);
  const tBool = b.id(); b.typeBool(tBool);
  const tVec3U32 = b.id(); b.typeVector(tVec3U32, tU32, 3);

  // ── Constants ────────────────────────────────────────────────────────────
  const const0u = b.id(); b.constant(tU32, const0u, 0);
  const const1u = b.id(); b.constant(tU32, const1u, 1);
  if (WG <= 0) throw new Error(`flash_attn_coop2: WG=${WG} must be > 0`);
  if ((Br * Bc) % WG !== 0) throw new Error(`flash_attn_coop2: WG=${WG} must divide Br*Bc=${Br * Bc}`);
  if ((Br * D) % WG !== 0) throw new Error(`flash_attn_coop2: WG=${WG} must divide Br*D=${Br * D}`);
  if ((Bc * D) % WG !== 0) throw new Error(`flash_attn_coop2: WG=${WG} must divide Bc*D=${Bc * D}`);
  const threadsPerRow = Math.max(1, Math.floor(WG / Br));
  if ((WG % Br) !== 0) throw new Error(`flash_attn_coop2: WG=${WG} must be divisible by Br=${Br}`);
  if ((D % threadsPerRow) !== 0) throw new Error(`flash_attn_coop2: D=${D} must be divisible by threadsPerRow=${threadsPerRow}`);
  const constThreadsPerRow = b.id(); b.constant(tU32, constThreadsPerRow, threadsPerRow);

  // ── Shift/mask constants for power-of-2 index math ─────────────────────
  // Replace UDiv/UMod with ShiftRightLogical/BitwiseAnd in hot loops.
  const constLog2D = log2D >= 0 ? (() => { const id = b.id(); b.constant(tU32, id, log2D); return id; })() : null;
  const constMaskD = log2D >= 0 ? (() => { const id = b.id(); b.constant(tU32, id, D - 1); return id; })() : null;
  const constLog2Bc = log2Bc >= 0 ? (() => { const id = b.id(); b.constant(tU32, id, log2Bc); return id; })() : null;
  const constLog2TPR = log2ThreadsPerRow >= 0 ? (() => { const id = b.id(); b.constant(tU32, id, log2ThreadsPerRow); return id; })() : null;
  const constMaskTPR = log2ThreadsPerRow >= 0 ? (() => { const id = b.id(); b.constant(tU32, id, threadsPerRow - 1); return id; })() : null;
  const constLog2QEpt = log2QElemsPerThread >= 0 ? (() => { const id = b.id(); b.constant(tU32, id, log2QElemsPerThread); return id; })() : null;
  const constLog2KVEpt = log2KVElemsPerThread >= 0 ? (() => { const id = b.id(); b.constant(tU32, id, log2KVElemsPerThread); return id; })() : null;

  const const0f = b.id(); b.constantF32(tF32, const0f, 0.0);
  const const0h = b.id(); b.constant(tF16, const0h, 0);
  const const05h = isKVSynthMode ? (() => { const id = b.id(); b.constant(tF16, id, 0x3800); return id; })() : 0;
  const const1f = b.id(); b.constantF32(tF32, const1f, 1.0);
  const constNegInf = b.id(); b.constantF32(tF32, constNegInf, -Infinity);
  const constLog2e = b.id(); b.constantF32(tF32, constLog2e, Math.LOG2E);
  const constLn2 = b.id(); b.constantF32(tF32, constLn2, Math.LN2);
  const constInvBc = b.id(); b.constantF32(tF32, constInvBc, 1.0 / Bc);
  const constSoftCapValue = useSoftCapConst ? (softCapConst as number) : 1.0;
  const constSoftCap = b.id(); b.constantF32(tF32, constSoftCap, constSoftCapValue);
  const constInvSoftCap = b.id(); b.constantF32(tF32, constInvSoftCap, 1.0 / constSoftCapValue);

  const constBr = b.id(); b.constant(tU32, constBr, Br);
  const constBc = b.id(); b.constant(tU32, constBc, Bc);
  const constD  = b.id(); b.constant(tU32, constD, D);
  const constCluster8 = b.id(); b.constant(tU32, constCluster8, 8);

  // ── Cooperative matrix types ─────────────────────────────────────────────
  const scopeSubgroup = b.id(); b.constant(tU32, scopeSubgroup, Scope.Subgroup);
  const scopeWg = b.id(); b.constant(tU32, scopeWg, Scope.Workgroup);
  const coopScope = useWorkgroupScope ? scopeWg : scopeSubgroup;

  const constCoopM = b.id(); b.constant(tU32, constCoopM, coopM);
  const constCoopN = b.id(); b.constant(tU32, constCoopN, coopN);
  const constCoopK = b.id(); b.constant(tU32, constCoopK, coopK);
  const constUseA   = b.id(); b.constant(tU32, constUseA, CooperativeMatrixUse.MatrixA);
  const constUseB   = b.id(); b.constant(tU32, constUseB, CooperativeMatrixUse.MatrixB);
  const constUseAcc = b.id(); b.constant(tU32, constUseAcc, CooperativeMatrixUse.MatrixAccumulator);

  // A: f16, 16x16, MatrixA (for P after convert)
  const tCoopA = b.id();
  b.typeCooperativeMatrixKHR(tCoopA, tF16, coopScope, constCoopM, constCoopK, constUseA);
  // B: f16, 16x16, MatrixB
  const tCoopB = b.id();
  b.typeCooperativeMatrixKHR(tCoopB, tF16, coopScope, constCoopK, constCoopN, constUseB);
  // Accumulator: f32, 16x16
  const tCoopAcc = b.id();
  b.typeCooperativeMatrixKHR(tCoopAcc, tF32, coopScope, constCoopM, constCoopN, constUseAcc);

  const constRowMajor = const0u;
  const constColMajor = (() => { const id = b.id(); b.constant(tU32, id, 1); return id; })();
  const constNullAcc = b.id(); b.constantNull(tCoopAcc, constNullAcc);

  // ── Function types ───────────────────────────────────────────────────────
  const tFnVoid = b.id(); b.typeFunction(tFnVoid, tVoid);
  // CombineFunc: (f32, f32) → f32
  const tFnCombine = b.id(); b.typeFunction(tFnCombine, tF32, [tF32, tF32]);
  // PerElementOp 1-matrix: (u32, u32, f32) → f32
  const tFnPE1 = b.id(); b.typeFunction(tFnPE1, tF32, [tU32, tU32, tF32]);
  // PerElementOp 1-matrix + scalar state: (u32, u32, f32, f32, u32, u32, u32) → f32
  const tFnPE1ScaleMask = b.id();
  b.typeFunction(tFnPE1ScaleMask, tF32, [tU32, tU32, tF32, tF32, tU32, tU32, tU32]);
  // PerElementOp 1-matrix + scalar state (causal only, no oob check):
  // (u32, u32, f32, f32, u32, u32) → f32
  const tFnPE1ScaleMaskCausal = b.id();
  b.typeFunction(tFnPE1ScaleMaskCausal, tF32, [tU32, tU32, tF32, tF32, tU32, tU32]);
  // PerElementOp 1-matrix + scale + softcap: (u32, u32, f32, f32, f32) → f32
  const tFnPE1ScaleSoftCap = b.id();
  b.typeFunction(tFnPE1ScaleSoftCap, tF32, [tU32, tU32, tF32, tF32, tF32]);
  // PerElementOp 1-matrix + scalar state + softcap:
  // (u32, u32, f32, f32, f32, u32, u32, u32) → f32
  const tFnPE1ScaleMaskSoftCap = b.id();
  b.typeFunction(tFnPE1ScaleMaskSoftCap, tF32, [tU32, tU32, tF32, tF32, tF32, tU32, tU32, tU32]);
  // PerElementOp 1-matrix + scalar state + softcap (causal only, no oob check):
  // (u32, u32, f32, f32, f32, u32, u32) → f32
  const tFnPE1ScaleMaskSoftCapCausal = b.id();
  b.typeFunction(tFnPE1ScaleMaskSoftCapCausal, tF32, [tU32, tU32, tF32, tF32, tF32, tU32, tU32]);
  // PerElementOp 2-matrix: (u32, u32, f32, f32) → f32
  const tFnPE2 = b.id(); b.typeFunction(tFnPE2, tF32, [tU32, tU32, tF32, tF32]);

  // ── Pointer types ────────────────────────────────────────────────────────
  const tPtrFnF32 = b.id(); b.typePointer(tPtrFnF32, StorageClass.Function, tF32);
  const tPtrFnU32 = b.id(); b.typePointer(tPtrFnU32, StorageClass.Function, tU32);
  const tPtrFnCoopAcc = b.id(); b.typePointer(tPtrFnCoopAcc, StorageClass.Function, tCoopAcc);
  const tPtrSharedF16 = b.id(); b.typePointer(tPtrSharedF16, StorageClass.Workgroup, tF16);
  const tPtrSharedF32 = b.id(); b.typePointer(tPtrSharedF32, StorageClass.Workgroup, tF32);

  // ── Storage buffers ──────────────────────────────────────────────────────
  const bufQ = useF16Input ? declareF16Ssbo(b, tF16, 0, 0, true) : declareF32Ssbo(b, tF32, 0, 0, true);
  const bufK = useF16Input ? declareF16Ssbo(b, tF16, 0, 1, true) : declareF32Ssbo(b, tF32, 0, 1, true);
  const bufV = useF16Input ? declareF16Ssbo(b, tF16, 0, 2, true) : declareF32Ssbo(b, tF32, 0, 2, true);
  const bufO   = declareF32Ssbo(b, tF32, 0, 3, false, true);
  const bufLSE = declareF32Ssbo(b, tF32, 0, 4, false, true);

  // Push constants: { T, scale, softCap, _pad1 }
  const pc = declareParamsPushConstant(b, tF32, 4);

  // ── Shared memory ────────────────────────────────────────────────────────
  // sQ[qTiles * Br * D] f16 — query blocks for this workgroup
  const constSQSize = b.id(); b.constant(tU32, constSQSize, qTiles * Br * D);
  const tArrSQ = b.id(); b.typeArray(tArrSQ, tF16, constSQSize);
  const tPtrArrSQ = b.id(); b.typePointer(tPtrArrSQ, StorageClass.Workgroup, tArrSQ);
  const sQ = b.id(); b.variable(tPtrArrSQ, sQ, StorageClass.Workgroup);

  // sK[Bc * D * mul] f16 — K row-major [Bc, D], optionally double-buffered.
  // CoopMatrixLoadKHR with ColumnMajor+strideD gives K^T for QK^T computation.
  const skSize = Bc * D * (doubleBuf ? 2 : 1);
  const constSKSize = b.id(); b.constant(tU32, constSKSize, skSize);
  const tArrSK = b.id(); b.typeArray(tArrSK, tF16, constSKSize);
  const tPtrArrSK = b.id(); b.typePointer(tPtrArrSK, StorageClass.Workgroup, tArrSK);
  const sK = b.id(); b.variable(tPtrArrSK, sK, StorageClass.Workgroup);

  // sV[Bc * D * mul] f16 — V row-major, optionally double-buffered.
  const svSize = Bc * D * (doubleBuf ? 2 : 1);
  const constSVSize = b.id(); b.constant(tU32, constSVSize, svSize);
  const tArrSV = b.id(); b.typeArray(tArrSV, tF16, constSVSize);
  const tPtrArrSV = b.id(); b.typePointer(tPtrArrSV, StorageClass.Workgroup, tArrSV);
  const sV = b.id(); b.variable(tPtrArrSV, sV, StorageClass.Workgroup);

  // sScratch[Br * D] f32 — for m/l extraction and O writeback
  const scratchSize = Math.max(Br * D, Br * Bc);
  const constScratchSize = b.id(); b.constant(tU32, constScratchSize, scratchSize);
  const tArrScratch = b.id(); b.typeArray(tArrScratch, tF32, constScratchSize);
  const tPtrArrScratch = b.id(); b.typePointer(tPtrArrScratch, StorageClass.Workgroup, tArrScratch);
  const sScratch = b.id(); b.variable(tPtrArrScratch, sScratch, StorageClass.Workgroup);

  // ── Built-in variables ───────────────────────────────────────────────────
  const tPtrInputVec3 = b.id(); b.typePointer(tPtrInputVec3, StorageClass.Input, tVec3U32);
  const vWorkgroupId = b.id(); b.variable(tPtrInputVec3, vWorkgroupId, StorageClass.Input);
  b.addDecorate(vWorkgroupId, Decoration.BuiltIn, BuiltIn.WorkgroupId);
  const vLocalId = b.id(); b.variable(tPtrInputVec3, vLocalId, StorageClass.Input);
  b.addDecorate(vLocalId, Decoration.BuiltIn, BuiltIn.LocalInvocationId);

  const semAcqRelWg = b.id();
  b.constant(tU32, semAcqRelWg, MemorySemantics.AcquireRelease | MemorySemantics.WorkgroupMemory);

  // ════════════════════════════════════════════════════════════════════════
  // CALLBACK FUNCTIONS (all declared before main)
  // ════════════════════════════════════════════════════════════════════════

  // --- CombineFunc: fmaxCombine(a, b) → max(a, b) ---
  const fnFMaxCombine = b.id();
  b.emit(Op.Function, [tF32, fnFMaxCombine, FunctionControl.None, tFnCombine]);
  const fmcP1 = b.id(); b.emit(Op.FunctionParameter, [tF32, fmcP1]);
  const fmcP2 = b.id(); b.emit(Op.FunctionParameter, [tF32, fmcP2]);
  const fmcLabel = b.id(); b.emit(Op.Label, [fmcLabel]);
  const fmcResult = b.id(); b.emit(Op.ExtInst, [tF32, fmcResult, glslStd, GLSLstd450.FMax, fmcP1, fmcP2]);
  b.emit(Op.ReturnValue, [fmcResult]);
  b.emit(Op.FunctionEnd, []);

  // --- PerElementOp: scaleSoftCap ---
  const fnScaleSoftCap = b.id();
  if (useSoftCapConst) {
    // Constant-softcap fast path: avoid dynamic softcap operand in hot tiles.
    b.emit(Op.Function, [tF32, fnScaleSoftCap, FunctionControl.None, tFnPE2]);
    const sscRow = b.id(); b.emit(Op.FunctionParameter, [tU32, sscRow]);
    const sscCol = b.id(); b.emit(Op.FunctionParameter, [tU32, sscCol]);
    const sscVal = b.id(); b.emit(Op.FunctionParameter, [tF32, sscVal]);
    // sscScaleInvSoftCap is precomputed once as (scale * invSoftCapConst).
    const sscScaleInvSoftCap = b.id(); b.emit(Op.FunctionParameter, [tF32, sscScaleInvSoftCap]);
    const sscLabel = b.id(); b.emit(Op.Label, [sscLabel]);
    {
      const scaledInv = b.id(); b.emit(Op.FMul, [tF32, scaledInv, sscVal, sscScaleInvSoftCap]);
      const tanhVal = b.id(); b.emit(Op.ExtInst, [tF32, tanhVal, glslStd, GLSLstd450.Tanh, scaledInv]);
      const capped = b.id(); b.emit(Op.FMul, [tF32, capped, tanhVal, constSoftCap]);
      b.emit(Op.ReturnValue, [capped]);
    }
    b.emit(Op.FunctionEnd, []);
  } else {
    b.emit(Op.Function, [tF32, fnScaleSoftCap, FunctionControl.None, tFnPE1ScaleSoftCap]);
    const sscRow = b.id(); b.emit(Op.FunctionParameter, [tU32, sscRow]);
    const sscCol = b.id(); b.emit(Op.FunctionParameter, [tU32, sscCol]);
    const sscVal = b.id(); b.emit(Op.FunctionParameter, [tF32, sscVal]);
    const sscScale = b.id(); b.emit(Op.FunctionParameter, [tF32, sscScale]);
    const sscSoftCap = b.id(); b.emit(Op.FunctionParameter, [tF32, sscSoftCap]);
    const sscLabel = b.id(); b.emit(Op.Label, [sscLabel]);
    {
      const scaled = b.id(); b.emit(Op.FMul, [tF32, scaled, sscVal, sscScale]);
      const invSoftCap = b.id(); b.emit(Op.FDiv, [tF32, invSoftCap, const1f, sscSoftCap]);
      const scaledInv = b.id(); b.emit(Op.FMul, [tF32, scaledInv, scaled, invSoftCap]);
      const tanhVal = b.id(); b.emit(Op.ExtInst, [tF32, tanhVal, glslStd, GLSLstd450.Tanh, scaledInv]);
      const capped = b.id(); b.emit(Op.FMul, [tF32, capped, tanhVal, sscSoftCap]);
      b.emit(Op.ReturnValue, [capped]);
    }
    b.emit(Op.FunctionEnd, []);
  }

  // --- CombineFunc: faddCombine(a, b) → a + b ---
  const fnFAddCombine = b.id();
  b.emit(Op.Function, [tF32, fnFAddCombine, FunctionControl.None, tFnCombine]);
  const facP1 = b.id(); b.emit(Op.FunctionParameter, [tF32, facP1]);
  const facP2 = b.id(); b.emit(Op.FunctionParameter, [tF32, facP2]);
  const facLabel = b.id(); b.emit(Op.Label, [facLabel]);
  const facResult = b.id(); b.emit(Op.FAdd, [tF32, facResult, facP1, facP2]);
  b.emit(Op.ReturnValue, [facResult]);
  b.emit(Op.FunctionEnd, []);

  // --- PerElementOp: scaleMaskFunc(row, col, val, scale[, softCap], qBase, kBase, T) ---
  const fnScaleMask = b.id();
  b.emit(Op.Function, [
    tF32,
    fnScaleMask,
    FunctionControl.None,
    useSoftCap && !useSoftCapConst ? tFnPE1ScaleMaskSoftCap : tFnPE1ScaleMask,
  ]);
  const smRow = b.id(); b.emit(Op.FunctionParameter, [tU32, smRow]);
  const smCol = b.id(); b.emit(Op.FunctionParameter, [tU32, smCol]);
  const smVal = b.id(); b.emit(Op.FunctionParameter, [tF32, smVal]);
  const smScale = b.id(); b.emit(Op.FunctionParameter, [tF32, smScale]);
  const smSoftCap = useSoftCap && !useSoftCapConst ? b.id() : 0;
  if (useSoftCap && !useSoftCapConst) b.emit(Op.FunctionParameter, [tF32, smSoftCap]);
  const smQBlockBase = b.id(); b.emit(Op.FunctionParameter, [tU32, smQBlockBase]);
  const smKBlockBase = b.id(); b.emit(Op.FunctionParameter, [tU32, smKBlockBase]);
  const smT = b.id(); b.emit(Op.FunctionParameter, [tU32, smT]);
    const smLabel = b.id(); b.emit(Op.Label, [smLabel]);
  {
    let score = b.id();
    if (useSoftCap) {
      let scaledInv = b.id();
      if (useSoftCapConst) {
        // smScale carries (scale * invSoftCapConst) for constant-softcap variants.
        b.emit(Op.FMul, [tF32, scaledInv, smVal, smScale]);
      } else {
        const scaled = b.id(); b.emit(Op.FMul, [tF32, scaled, smVal, smScale]);
        const invSoftCap = b.id(); b.emit(Op.FDiv, [tF32, invSoftCap, const1f, smSoftCap]);
        scaledInv = b.id(); b.emit(Op.FMul, [tF32, scaledInv, scaled, invSoftCap]);
      }
      const tanhVal = b.id(); b.emit(Op.ExtInst, [tF32, tanhVal, glslStd, GLSLstd450.Tanh, scaledInv]);
      const capVal = useSoftCapConst ? constSoftCap : smSoftCap;
      const capped = b.id(); b.emit(Op.FMul, [tF32, capped, tanhVal, capVal]);
      score = capped;
    } else {
      const scaled = b.id(); b.emit(Op.FMul, [tF32, scaled, smVal, smScale]);
      score = scaled;
    }
    const qRow = b.id(); b.emit(Op.IAdd, [tU32, qRow, smQBlockBase, smRow]);
    const kCol = b.id(); b.emit(Op.IAdd, [tU32, kCol, smKBlockBase, smCol]);
    const causal = b.id(); b.emit(Op.ULessThan, [tBool, causal, qRow, kCol]);
    const oob = b.id(); b.emit(Op.UGreaterThanEqual, [tBool, oob, kCol, smT]);
    const masked = b.id(); b.emit(Op.LogicalOr, [tBool, masked, causal, oob]);
    const result = b.id(); b.emit(Op.Select, [tF32, result, masked, constNegInf, score]);
    b.emit(Op.ReturnValue, [result]);
  }
  b.emit(Op.FunctionEnd, []);

  // --- PerElementOp: scaleMaskCausalNoOob(row, col, val, scale[, softCap], qBase, kBase) ---
  const fnScaleMaskCausalNoOob = b.id();
  b.emit(Op.Function, [
    tF32,
    fnScaleMaskCausalNoOob,
    FunctionControl.None,
    useSoftCap && !useSoftCapConst ? tFnPE1ScaleMaskSoftCapCausal : tFnPE1ScaleMaskCausal,
  ]);
  const smcRow = b.id(); b.emit(Op.FunctionParameter, [tU32, smcRow]);
  const smcCol = b.id(); b.emit(Op.FunctionParameter, [tU32, smcCol]);
  const smcVal = b.id(); b.emit(Op.FunctionParameter, [tF32, smcVal]);
  const smcScale = b.id(); b.emit(Op.FunctionParameter, [tF32, smcScale]);
  const smcSoftCap = useSoftCap && !useSoftCapConst ? b.id() : 0;
  if (useSoftCap && !useSoftCapConst) b.emit(Op.FunctionParameter, [tF32, smcSoftCap]);
  const smcQBlockBase = b.id(); b.emit(Op.FunctionParameter, [tU32, smcQBlockBase]);
  const smcKBlockBase = b.id(); b.emit(Op.FunctionParameter, [tU32, smcKBlockBase]);
  const smcLabel = b.id(); b.emit(Op.Label, [smcLabel]);
  {
    let score = b.id();
    if (useSoftCap) {
      let scaledInv = b.id();
      if (useSoftCapConst) {
        // smcScale carries (scale * invSoftCapConst) for constant-softcap variants.
        b.emit(Op.FMul, [tF32, scaledInv, smcVal, smcScale]);
      } else {
        const scaled = b.id(); b.emit(Op.FMul, [tF32, scaled, smcVal, smcScale]);
        const invSoftCap = b.id(); b.emit(Op.FDiv, [tF32, invSoftCap, const1f, smcSoftCap]);
        scaledInv = b.id(); b.emit(Op.FMul, [tF32, scaledInv, scaled, invSoftCap]);
      }
      const tanhVal = b.id(); b.emit(Op.ExtInst, [tF32, tanhVal, glslStd, GLSLstd450.Tanh, scaledInv]);
      const capVal = useSoftCapConst ? constSoftCap : smcSoftCap;
      const capped = b.id(); b.emit(Op.FMul, [tF32, capped, tanhVal, capVal]);
      score = capped;
    } else {
      const scaled = b.id(); b.emit(Op.FMul, [tF32, scaled, smcVal, smcScale]);
      score = scaled;
    }
    const qRow = b.id(); b.emit(Op.IAdd, [tU32, qRow, smcQBlockBase, smcRow]);
    const kCol = b.id(); b.emit(Op.IAdd, [tU32, kCol, smcKBlockBase, smcCol]);
    const causal = b.id(); b.emit(Op.ULessThan, [tBool, causal, qRow, kCol]);
    const result = b.id(); b.emit(Op.Select, [tF32, result, causal, constNegInf, score]);
    b.emit(Op.ReturnValue, [result]);
  }
  b.emit(Op.FunctionEnd, []);

  // --- PerElementOp: expSubFunc(row, col, val, maxVal) → exp2((val - maxVal) * LOG2E) ---
  const fnExpSub = b.id();
  b.emit(Op.Function, [tF32, fnExpSub, FunctionControl.None, tFnPE2]);
  const esRow = b.id(); b.emit(Op.FunctionParameter, [tU32, esRow]);
  const esCol = b.id(); b.emit(Op.FunctionParameter, [tU32, esCol]);
  const esVal = b.id(); b.emit(Op.FunctionParameter, [tF32, esVal]);
  const esMax = b.id(); b.emit(Op.FunctionParameter, [tF32, esMax]);
  const esLabel = b.id(); b.emit(Op.Label, [esLabel]);
  {
    const diff = b.id(); b.emit(Op.FSub, [tF32, diff, esVal, esMax]);
    const diffLog2 = b.id(); b.emit(Op.FMul, [tF32, diffLog2, diff, constLog2e]);
    const result = b.id(); b.emit(Op.ExtInst, [tF32, result, glslStd, GLSLstd450.Exp2, diffLog2]);
    b.emit(Op.ReturnValue, [result]);
  }
  b.emit(Op.FunctionEnd, []);

  // --- PerElementOp: max2Func(row, col, a, b) → max(a, b) ---
  const fnMax2 = b.id();
  b.emit(Op.Function, [tF32, fnMax2, FunctionControl.None, tFnPE2]);
  const m2Row = b.id(); b.emit(Op.FunctionParameter, [tU32, m2Row]);
  const m2Col = b.id(); b.emit(Op.FunctionParameter, [tU32, m2Col]);
  const m2A = b.id(); b.emit(Op.FunctionParameter, [tF32, m2A]);
  const m2B = b.id(); b.emit(Op.FunctionParameter, [tF32, m2B]);
  const m2Label = b.id(); b.emit(Op.Label, [m2Label]);
  {
    const result = b.id(); b.emit(Op.ExtInst, [tF32, result, glslStd, GLSLstd450.FMax, m2A, m2B]);
    b.emit(Op.ReturnValue, [result]);
  }
  b.emit(Op.FunctionEnd, []);

  // --- PerElementOp: constOne(row, col, val) → 1.0 ---
  const fnConstOne = b.id();
  b.emit(Op.Function, [tF32, fnConstOne, FunctionControl.None, tFnPE1]);
  const c1Row = b.id(); b.emit(Op.FunctionParameter, [tU32, c1Row]);
  const c1Col = b.id(); b.emit(Op.FunctionParameter, [tU32, c1Col]);
  const c1Val = b.id(); b.emit(Op.FunctionParameter, [tF32, c1Val]);
  const c1Label = b.id(); b.emit(Op.Label, [c1Label]);
  b.emit(Op.ReturnValue, [const1f]);
  b.emit(Op.FunctionEnd, []);

  // --- PerElementOp: constNegInf(row, col, val) → -inf ---
  const fnConstNegInf = b.id();
  b.emit(Op.Function, [tF32, fnConstNegInf, FunctionControl.None, tFnPE1]);
  const cniRow = b.id(); b.emit(Op.FunctionParameter, [tU32, cniRow]);
  const cniCol = b.id(); b.emit(Op.FunctionParameter, [tU32, cniCol]);
  const cniVal = b.id(); b.emit(Op.FunctionParameter, [tF32, cniVal]);
  const cniLabel = b.id(); b.emit(Op.Label, [cniLabel]);
  b.emit(Op.ReturnValue, [constNegInf]);
  b.emit(Op.FunctionEnd, []);

  // --- PerElementOp: mulScalar(row, col, val, scalar) → val * scalar ---
  const fnMulScalar = b.id();
  b.emit(Op.Function, [tF32, fnMulScalar, FunctionControl.None, tFnPE2]);
  const msRow = b.id(); b.emit(Op.FunctionParameter, [tU32, msRow]);
  const msCol = b.id(); b.emit(Op.FunctionParameter, [tU32, msCol]);
  const msVal = b.id(); b.emit(Op.FunctionParameter, [tF32, msVal]);
  const msScale = b.id(); b.emit(Op.FunctionParameter, [tF32, msScale]);
  const msLabel = b.id(); b.emit(Op.Label, [msLabel]);
  {
    const result = b.id(); b.emit(Op.FMul, [tF32, result, msVal, msScale]);
    b.emit(Op.ReturnValue, [result]);
  }
  b.emit(Op.FunctionEnd, []);

  // ════════════════════════════════════════════════════════════════════════
  // MAIN FUNCTION
  // ════════════════════════════════════════════════════════════════════════
  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main",
    [vWorkgroupId, vLocalId, sQ, sK, sV, sScratch,
     bufQ.varId, bufK.varId, bufV.varId, bufO.varId, bufLSE.varId, pc.varId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, WG, 1, 1);
  b.emit(Op.Function, [tVoid, fnMain, FunctionControl.None, tFnVoid]);
  const labelEntry = b.id(); b.emit(Op.Label, [labelEntry]);

  // ── Function variables ───────────────────────────────────────────────────
  const varKBlock = b.id(); b.emit(Op.Variable, [tPtrFnU32, varKBlock, StorageClass.Function]);
  // Per-thread scalar m/l for manual softmax (one per qTile).
  const varMyMArr: number[] = [];
  const varMyLArr: number[] = [];
  if (useManualSoftmax) {
    for (let qti = 0; qti < qTiles; qti++) {
      varMyMArr[qti] = b.id(); b.emit(Op.Variable, [tPtrFnF32, varMyMArr[qti], StorageClass.Function]);
      varMyLArr[qti] = b.id(); b.emit(Op.Variable, [tPtrFnF32, varMyLArr[qti], StorageClass.Function]);
    }
  }

  // O accumulators: qTiles × pvTiles cooperative matrix tiles.
  const varOArr: number[][] = [];
  for (let qti = 0; qti < qTiles; qti++) {
    varOArr[qti] = [];
    for (let t = 0; t < pvTiles; t++) {
      varOArr[qti][t] = b.id(); b.emit(Op.Variable, [tPtrFnCoopAcc, varOArr[qti][t], StorageClass.Function]);
    }
  }
  // m (row max) and l (row sum) per query tile.
  const varMArr: number[] = [];
  const varLArr: number[] = [];
  for (let qti = 0; qti < qTiles; qti++) {
    varMArr[qti] = b.id(); b.emit(Op.Variable, [tPtrFnCoopAcc, varMArr[qti], StorageClass.Function]);
    varLArr[qti] = b.id(); b.emit(Op.Variable, [tPtrFnCoopAcc, varLArr[qti], StorageClass.Function]);
  }
  // Active scalar aliases while qTiles>1 compute path is still under construction.
  const varO = varOArr[0];
  const varM = varMArr[0];
  const varL = varLArr[0];

  // ── Load IDs ─────────────────────────────────────────────────────────────
  const lidVec = b.id(); b.emit(Op.Load, [tVec3U32, lidVec, vLocalId]);
  const tid = b.id(); b.emit(Op.CompositeExtract, [tU32, tid, lidVec, 0]);
  const wgIdVec = b.id(); b.emit(Op.Load, [tVec3U32, wgIdVec, vWorkgroupId]);
  const qBlockIdx = b.id(); b.emit(Op.CompositeExtract, [tU32, qBlockIdx, wgIdVec, 0]);
  const bhIdx = b.id(); b.emit(Op.CompositeExtract, [tU32, bhIdx, wgIdVec, 1]);

  // Push constants
  const ptrTpc = b.id(); b.emit(Op.AccessChain, [pc.tPtrF32, ptrTpc, pc.varId, const0u]);
  const TF = b.id(); b.emit(Op.Load, [tF32, TF, ptrTpc]);
  const T = b.id(); b.emit(Op.ConvertFToU, [tU32, T, TF]);
  const ptrScale = b.id(); b.emit(Op.AccessChain, [pc.tPtrF32, ptrScale, pc.varId, const1u]);
  const scale = b.id(); b.emit(Op.Load, [tF32, scale, ptrScale]);
  let scaleInvSoftCap = 0;
  if (useSoftCap && useSoftCapConst) {
    scaleInvSoftCap = b.id();
    b.emit(Op.FMul, [tF32, scaleInvSoftCap, scale, constInvSoftCap]);
  }
  let softCap = 0;
  if (useSoftCap && !useSoftCapConst) {
    const const2u = b.id(); b.constant(tU32, const2u, 2);
    const ptrSoftCap = b.id(); b.emit(Op.AccessChain, [pc.tPtrF32, ptrSoftCap, pc.varId, const2u]);
    softCap = b.id(); b.emit(Op.Load, [tF32, softCap, ptrSoftCap]);
  }

  // ── Thread mapping for manual softmax (and output writeback) ─────────────
  // 128 threads → 16 rows × 8 threads/row, each handling 2 adjacent columns.
  const myRow = b.id();
  if (constLog2TPR) b.emit(Op.ShiftRightLogical, [tU32, myRow, tid, constLog2TPR]);
  else b.emit(Op.UDiv, [tU32, myRow, tid, constThreadsPerRow]);
  const myColInRow = b.id();
  if (constMaskTPR) b.emit(Op.BitwiseAnd, [tU32, myColInRow, tid, constMaskTPR]);
  else b.emit(Op.UMod, [tU32, myColInRow, tid, constThreadsPerRow]);
  // For manual softmax: myCol0 = myColInRow * 2, myCol1 = myCol0 + 1
  const myCol0 = b.id(); b.emit(Op.ShiftLeftLogical, [tU32, myCol0, myColInRow, const1u]);
  const myCol1 = b.id(); b.emit(Op.IAdd, [tU32, myCol1, myCol0, const1u]);
  // Scratch index: myRow * Bc + myCol0 (contiguous pair at myCol0, myCol0+1)
  const myScrBase = b.id();
  if (constLog2Bc) b.emit(Op.ShiftLeftLogical, [tU32, myScrBase, myRow, constLog2Bc]);
  else b.emit(Op.IMul, [tU32, myScrBase, myRow, constBc]);
  const myScrIdx0 = b.id(); b.emit(Op.IAdd, [tU32, myScrIdx0, myScrBase, myCol0]);
  const myScrIdx1 = b.id(); b.emit(Op.IAdd, [tU32, myScrIdx1, myScrBase, myCol1]);

  // Base offsets
  const TD = b.id(); b.emit(Op.IMul, [tU32, TD, T, constD]);
  const baseOff = b.id(); b.emit(Op.IMul, [tU32, baseOff, bhIdx, TD]);
  const lseBaseOff = b.id(); b.emit(Op.IMul, [tU32, lseBaseOff, bhIdx, T]);
  const constBrQTiles = b.id(); b.constant(tU32, constBrQTiles, Br * qTiles);
  const qBlockGroupBase = b.id(); b.emit(Op.IMul, [tU32, qBlockGroupBase, qBlockIdx, constBrQTiles]);
  const qBlockBaseArr: number[] = [];
  for (let qti = 0; qti < qTiles; qti++) {
    const constQOff = b.id(); b.constant(tU32, constQOff, qti * Br);
    qBlockBaseArr[qti] = b.id(); b.emit(Op.IAdd, [tU32, qBlockBaseArr[qti], qBlockGroupBase, constQOff]);
  }
  const qBlockBase = qBlockBaseArr[0];

  // ── Initialize O, m, l ────────────────────────────────────────────────
  for (let qti = 0; qti < qTiles; qti++) {
    for (let t = 0; t < pvTiles; t++) {
      b.emit(Op.Store, [varOArr[qti][t], constNullAcc]);
    }
    b.emit(Op.Store, [varLArr[qti], constNullAcc]); // l = 0
  }

  // Initialize m = -inf directly in cooperative-matrix space.
  if (!useManualSoftmax) {
    const mInit = b.id();
    b.emit(Op.OpCooperativeMatrixPerElementOpNV, [tCoopAcc, mInit, constNullAcc, fnConstNegInf]);
    for (let qti = 0; qti < qTiles; qti++) {
      b.emit(Op.Store, [varMArr[qti], mInit]);
    }
  }
  // Initialize per-thread scalar m/l for manual softmax.
  if (useManualSoftmax) {
    for (let qti = 0; qti < qTiles; qti++) {
      b.emit(Op.Store, [varMyMArr[qti], constNegInf]);
      b.emit(Op.Store, [varMyLArr[qti], const0f]);
    }
  }

  // ── Load Q block into sQ as f16 ─────────────────────────────────────────
  {
    for (let qti = 0; qti < qTiles; qti++) {
      const qBlockBaseTile = qBlockBaseArr[qti];
      const constSQBase = b.id(); b.constant(tU32, constSQBase, qti * Br * D);
      const qBlockEnd = b.id(); b.emit(Op.IAdd, [tU32, qBlockEnd, qBlockBaseTile, constBr]);
      const qBlockFullInBounds = b.id(); b.emit(Op.UGreaterThanEqual, [tBool, qBlockFullInBounds, T, qBlockEnd]);
      const labelQLoadFast = b.id();
      const labelQLoadSlow = b.id();
      const labelQLoadMerge = b.id();
      b.emit(Op.SelectionMerge, [labelQLoadMerge, 0]);
      b.emit(Op.BranchConditional, [qBlockFullInBounds, labelQLoadFast, labelQLoadSlow]);

      b.emit(Op.Label, [labelQLoadFast]);
      {
        const elemsPerThread = (Br * D) / WG;
        // Precompute Q global base: baseOff + qBlockBaseTile * D
        const qBlockBaseTileD = b.id();
        if (constLog2D) b.emit(Op.ShiftLeftLogical, [tU32, qBlockBaseTileD, qBlockBaseTile, constLog2D]);
        else { const cDv = b.id(); b.constant(tU32, cDv, D); b.emit(Op.IMul, [tU32, qBlockBaseTileD, qBlockBaseTile, cDv]); }
        const qGlobalBase = b.id(); b.emit(Op.IAdd, [tU32, qGlobalBase, baseOff, qBlockBaseTileD]);
        // Per-thread-consecutive: flatIdx = tid * E + i → 8 consecutive f16 loads
        // per thread. NVIDIA driver vectorizes to 1×16-byte load (8× fewer instructions).
        const constQEpt = b.id(); b.constant(tU32, constQEpt, elemsPerThread);
        const qTidBase = b.id(); b.emit(Op.IMul, [tU32, qTidBase, tid, constQEpt]);
        for (let i = 0; i < elemsPerThread; i++) {
          const constI = b.id(); b.constant(tU32, constI, i);
          const sharedIdx = b.id(); b.emit(Op.IAdd, [tU32, sharedIdx, qTidBase, constI]);
          const gIdx = b.id(); b.emit(Op.IAdd, [tU32, gIdx, qGlobalBase, sharedIdx]);
          const ptrG = b.id(); b.emit(Op.AccessChain, [bufQ.tPtr, ptrG, bufQ.varId, const0u, gIdx]);
          let valF16: number;
          if (useF16Input) {
            valF16 = b.id(); b.emit(Op.Load, [tF16, valF16, ptrG]);
          } else {
            const valF32 = b.id(); b.emit(Op.Load, [tF32, valF32, ptrG]);
            valF16 = b.id(); b.emit(Op.FConvert, [tF16, valF16, valF32]);
          }
          const sqIdx = b.id(); b.emit(Op.IAdd, [tU32, sqIdx, constSQBase, sharedIdx]);
          const ptrSQ = b.id(); b.emit(Op.AccessChain, [tPtrSharedF16, ptrSQ, sQ, sqIdx]);
          b.emit(Op.Store, [ptrSQ, valF16]);
        }
      }
      b.emit(Op.Branch, [labelQLoadMerge]);

      b.emit(Op.Label, [labelQLoadSlow]);
      {
        const elemsPerThread = (Br * D) / WG;
        // Precompute Q global base (same as fast path)
        const qBlockBaseTileD2 = b.id();
        if (constLog2D) b.emit(Op.ShiftLeftLogical, [tU32, qBlockBaseTileD2, qBlockBaseTile, constLog2D]);
        else { const cDv = b.id(); b.constant(tU32, cDv, D); b.emit(Op.IMul, [tU32, qBlockBaseTileD2, qBlockBaseTile, cDv]); }
        const qGlobalBase2 = b.id(); b.emit(Op.IAdd, [tU32, qGlobalBase2, baseOff, qBlockBaseTileD2]);
        const constQEpt2 = b.id(); b.constant(tU32, constQEpt2, elemsPerThread);
        const qTidBase2 = b.id(); b.emit(Op.IMul, [tU32, qTidBase2, tid, constQEpt2]);
        for (let i = 0; i < elemsPerThread; i++) {
          const constI = b.id(); b.constant(tU32, constI, i);
          const sharedIdx = b.id(); b.emit(Op.IAdd, [tU32, sharedIdx, qTidBase2, constI]);
          // Row for OOB check
          const row = b.id();
          if (constLog2D) b.emit(Op.ShiftRightLogical, [tU32, row, sharedIdx, constLog2D]);
          else { const cDv = b.id(); b.constant(tU32, cDv, D); b.emit(Op.UDiv, [tU32, row, sharedIdx, cDv]); }
          const qRow = b.id(); b.emit(Op.IAdd, [tU32, qRow, qBlockBaseTile, row]);
          const inBounds = b.id(); b.emit(Op.ULessThan, [tBool, inBounds, qRow, T]);
          const gIdx = b.id(); b.emit(Op.IAdd, [tU32, gIdx, qGlobalBase2, sharedIdx]);
          const ptrG = b.id(); b.emit(Op.AccessChain, [bufQ.tPtr, ptrG, bufQ.varId, const0u, gIdx]);
          let valF16Masked: number;
          if (useF16Input) {
            const valF16 = b.id(); b.emit(Op.Load, [tF16, valF16, ptrG]);
            valF16Masked = b.id(); b.emit(Op.Select, [tF16, valF16Masked, inBounds, valF16, const0h]);
          } else {
            const valF32 = b.id(); b.emit(Op.Load, [tF32, valF32, ptrG]);
            const maskedF32 = b.id(); b.emit(Op.Select, [tF32, maskedF32, inBounds, valF32, const0f]);
            valF16Masked = b.id(); b.emit(Op.FConvert, [tF16, valF16Masked, maskedF32]);
          }
          const sqIdx = b.id(); b.emit(Op.IAdd, [tU32, sqIdx, constSQBase, sharedIdx]);
          const ptrSQ = b.id(); b.emit(Op.AccessChain, [tPtrSharedF16, ptrSQ, sQ, sqIdx]);
          b.emit(Op.Store, [ptrSQ, valF16Masked]);
        }
      }
      b.emit(Op.Branch, [labelQLoadMerge]);
      b.emit(Op.Label, [labelQLoadMerge]);
    }
  }
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]); // barrier: Q loaded

  // ── Double-buffer KV staging infrastructure ───────────────────────────
  let dbVarBufIdx = 0; // SPIR-V ID for buffer index variable (0 = doubleBuf disabled)
  let dbConstBufSize = 0; // SPIR-V ID: Bc * D (buffer size for one KV set)
  let curKOff = const0u; // SPIR-V ID: current buffer offset for sK reads
  let curVOff = const0u;  // SPIR-V ID: current buffer offset for sV reads
  if (doubleBuf) {
    dbConstBufSize = b.id(); b.constant(tU32, dbConstBufSize, Bc * D);
    dbVarBufIdx = b.id(); b.emit(Op.Variable, [tPtrFnU32, dbVarBufIdx, StorageClass.Function]);
    b.emit(Op.Store, [dbVarBufIdx, const0u]);
  }

  // Helper: emit SPIR-V to load K and V from global into shared memory (non-transposed).
  // kBlockBaseP/fullTBlockP are SPIR-V IDs. kOff/vOff are buffer offsets into sK/sV.
  function emitKVLoad(kBlockBaseP: number, fullTBlockP: number, kOff: number, vOff: number) {
    const labelKVFast = b.id();
    const labelKVSlow = b.id();
    const labelKVMerge = b.id();
    b.emit(Op.SelectionMerge, [labelKVMerge, 0]);
    b.emit(Op.BranchConditional, [fullTBlockP, labelKVFast, labelKVSlow]);

    b.emit(Op.Label, [labelKVFast]);
    {
      const elemsPerThread = (Bc * D) / WG;
      // Precompute global base: baseOff + kBlockBase * D.
      // Since K/V are [BH,T,D] contiguous and shmem is non-transposed [Bc,D],
      // the global address for flatIdx is simply globalBase + flatIdx.
      let globalBase: number | undefined;
      if (!isKVSynthMode) {
        const kBlockBaseD = b.id();
        if (constLog2D) b.emit(Op.ShiftLeftLogical, [tU32, kBlockBaseD, kBlockBaseP, constLog2D]);
        else { const cDv = b.id(); b.constant(tU32, cDv, D); b.emit(Op.IMul, [tU32, kBlockBaseD, kBlockBaseP, cDv]); }
        globalBase = b.id(); b.emit(Op.IAdd, [tU32, globalBase, baseOff, kBlockBaseD]);
      }
      // Per-thread-consecutive: flatIdx = tid * E + i → consecutive f16 loads per
      // thread enable NVIDIA driver vectorization (1×16-byte load vs 8×2-byte).
      const constKVEpt = b.id(); b.constant(tU32, constKVEpt, elemsPerThread);
      const kvTidBase = b.id(); b.emit(Op.IMul, [tU32, kvTidBase, tid, constKVEpt]);
      for (let i = 0; i < elemsPerThread; i++) {
        const constI = b.id(); b.constant(tU32, constI, i);
        const flatIdx = b.id(); b.emit(Op.IAdd, [tU32, flatIdx, kvTidBase, constI]);
        if (!isKVSynthMode) {
          const gIdx = b.id(); b.emit(Op.IAdd, [tU32, gIdx, globalBase!, flatIdx]);
          // K: store non-transposed at sK[flatIdx + kOff] (same layout as V)
          if (!isPVOnlyMode) {
            const pGK = b.id(); b.emit(Op.AccessChain, [bufK.tPtr, pGK, bufK.varId, const0u, gIdx]);
            let kF16: number;
            if (useF16Input) { kF16 = b.id(); b.emit(Op.Load, [tF16, kF16, pGK]); }
            else { const kF32 = b.id(); b.emit(Op.Load, [tF32, kF32, pGK]); kF16 = b.id(); b.emit(Op.FConvert, [tF16, kF16, kF32]); }
            const kIB = b.id(); b.emit(Op.IAdd, [tU32, kIB, flatIdx, kOff]);
            const pSK = b.id(); b.emit(Op.AccessChain, [tPtrSharedF16, pSK, sK, kIB]);
            b.emit(Op.Store, [pSK, kF16]);
          }
          // V: store at sV[flatIdx + vOff]
          const pGV = b.id(); b.emit(Op.AccessChain, [bufV.tPtr, pGV, bufV.varId, const0u, gIdx]);
          let vF16: number;
          if (useF16Input) { vF16 = b.id(); b.emit(Op.Load, [tF16, vF16, pGV]); }
          else { const vF32 = b.id(); b.emit(Op.Load, [tF32, vF32, pGV]); vF16 = b.id(); b.emit(Op.FConvert, [tF16, vF16, vF32]); }
          const vIB = b.id(); b.emit(Op.IAdd, [tU32, vIB, flatIdx, vOff]);
          const pSV = b.id(); b.emit(Op.AccessChain, [tPtrSharedF16, pSV, sV, vIB]);
          b.emit(Op.Store, [pSV, vF16]);
        } else {
          // kv_synth: write constant 0.5h to shmem (preserves shmem write cost, no global reads)
          if (!isPVOnlyMode) {
            const kIB = b.id(); b.emit(Op.IAdd, [tU32, kIB, flatIdx, kOff]);
            const pSK = b.id(); b.emit(Op.AccessChain, [tPtrSharedF16, pSK, sK, kIB]);
            b.emit(Op.Store, [pSK, const05h]);
          }
          const vIB = b.id(); b.emit(Op.IAdd, [tU32, vIB, flatIdx, vOff]);
          const pSV = b.id(); b.emit(Op.AccessChain, [tPtrSharedF16, pSV, sV, vIB]);
          b.emit(Op.Store, [pSV, const05h]);
        }
      }
    }
    b.emit(Op.Branch, [labelKVMerge]);

    b.emit(Op.Label, [labelKVSlow]);
    {
      const elemsPerThread = (Bc * D) / WG;
      // Precompute global base (same as fast path). OOB check uses j = flatIdx / D.
      let globalBase2: number | undefined;
      if (!isKVSynthMode) {
        const kBlockBaseD = b.id();
        if (constLog2D) b.emit(Op.ShiftLeftLogical, [tU32, kBlockBaseD, kBlockBaseP, constLog2D]);
        else { const cDv = b.id(); b.constant(tU32, cDv, D); b.emit(Op.IMul, [tU32, kBlockBaseD, kBlockBaseP, cDv]); }
        globalBase2 = b.id(); b.emit(Op.IAdd, [tU32, globalBase2, baseOff, kBlockBaseD]);
      }
      const constKVEpt2 = b.id(); b.constant(tU32, constKVEpt2, elemsPerThread);
      const kvTidBase2 = b.id(); b.emit(Op.IMul, [tU32, kvTidBase2, tid, constKVEpt2]);
      for (let i = 0; i < elemsPerThread; i++) {
        const constI = b.id(); b.constant(tU32, constI, i);
        const flatIdx = b.id(); b.emit(Op.IAdd, [tU32, flatIdx, kvTidBase2, constI]);
        if (!isKVSynthMode) {
          // j = flatIdx / D for OOB check
          const j = b.id();
          if (constLog2D) b.emit(Op.ShiftRightLogical, [tU32, j, flatIdx, constLog2D]);
          else { const cDv = b.id(); b.constant(tU32, cDv, D); b.emit(Op.UDiv, [tU32, j, flatIdx, cDv]); }
          const kRow = b.id(); b.emit(Op.IAdd, [tU32, kRow, kBlockBaseP, j]);
          const inB = b.id(); b.emit(Op.ULessThan, [tBool, inB, kRow, T]);
          const gIdx = b.id(); b.emit(Op.IAdd, [tU32, gIdx, globalBase2!, flatIdx]);
          // K: store non-transposed at sK[flatIdx + kOff] with OOB masking
          if (!isPVOnlyMode) {
            const pGK = b.id(); b.emit(Op.AccessChain, [bufK.tPtr, pGK, bufK.varId, const0u, gIdx]);
            let kF16M: number;
            if (useF16Input) { const kF = b.id(); b.emit(Op.Load, [tF16, kF, pGK]); kF16M = b.id(); b.emit(Op.Select, [tF16, kF16M, inB, kF, const0h]); }
            else { const kF = b.id(); b.emit(Op.Load, [tF32, kF, pGK]); const km = b.id(); b.emit(Op.Select, [tF32, km, inB, kF, const0f]); kF16M = b.id(); b.emit(Op.FConvert, [tF16, kF16M, km]); }
            const kIB = b.id(); b.emit(Op.IAdd, [tU32, kIB, flatIdx, kOff]);
            const pSK = b.id(); b.emit(Op.AccessChain, [tPtrSharedF16, pSK, sK, kIB]);
            b.emit(Op.Store, [pSK, kF16M]);
          }
          // V: store at sV[flatIdx + vOff] with OOB masking
          const pGV = b.id(); b.emit(Op.AccessChain, [bufV.tPtr, pGV, bufV.varId, const0u, gIdx]);
          let vF16M: number;
          if (useF16Input) { const vF = b.id(); b.emit(Op.Load, [tF16, vF, pGV]); vF16M = b.id(); b.emit(Op.Select, [tF16, vF16M, inB, vF, const0h]); }
          else { const vF = b.id(); b.emit(Op.Load, [tF32, vF, pGV]); const vm = b.id(); b.emit(Op.Select, [tF32, vm, inB, vF, const0f]); vF16M = b.id(); b.emit(Op.FConvert, [tF16, vF16M, vm]); }
          const vIB = b.id(); b.emit(Op.IAdd, [tU32, vIB, flatIdx, vOff]);
          const pSV = b.id(); b.emit(Op.AccessChain, [tPtrSharedF16, pSV, sV, vIB]);
          b.emit(Op.Store, [pSV, vF16M]);
        } else {
          // kv_synth: write constant 0.5h to shmem (no global reads, no OOB checks)
          if (!isPVOnlyMode) {
            const kIB = b.id(); b.emit(Op.IAdd, [tU32, kIB, flatIdx, kOff]);
            const pSK = b.id(); b.emit(Op.AccessChain, [tPtrSharedF16, pSK, sK, kIB]);
            b.emit(Op.Store, [pSK, const05h]);
          }
          const vIB = b.id(); b.emit(Op.IAdd, [tU32, vIB, flatIdx, vOff]);
          const pSV = b.id(); b.emit(Op.AccessChain, [tPtrSharedF16, pSV, sV, vIB]);
          b.emit(Op.Store, [pSV, const05h]);
        }
      }
    }
    b.emit(Op.Branch, [labelKVMerge]);
    b.emit(Op.Label, [labelKVMerge]);
  }

  // ── kBlock loop ──────────────────────────────────────────────────────────
  const constBrMinus1 = b.id(); b.constant(tU32, constBrMinus1, Br - 1);
  const constBcMinus1 = b.id(); b.constant(tU32, constBcMinus1, Bc - 1);
  const numKBlocks = b.id();
  const TplusBcm1 = b.id(); b.emit(Op.IAdd, [tU32, TplusBcm1, T, constBcMinus1]);
  if (constLog2Bc) b.emit(Op.ShiftRightLogical, [tU32, numKBlocks, TplusBcm1, constLog2Bc]);
  else b.emit(Op.UDiv, [tU32, numKBlocks, TplusBcm1, constBc]);

  // ── Double-buffer prologue: load KV[0] into buf[0] ───────────────────
  if (doubleBuf) {
    const kbEnd0 = b.id(); b.emit(Op.IAdd, [tU32, kbEnd0, const0u, constBc]);
    const ft0 = b.id(); b.emit(Op.UGreaterThanEqual, [tBool, ft0, T, kbEnd0]);
    emitKVLoad(const0u, ft0, const0u, const0u);
    b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]); // barrier: KV[0] loaded
  }

  b.emit(Op.Store, [varKBlock, const0u]);
  const labelLoopHead = b.id(); const labelLoopBody = b.id();
  const labelLoopMerge = b.id(); const labelLoopCont = b.id();
  const labelLoopSkip = b.id(); const labelLoopCompute = b.id();
  b.emit(Op.Branch, [labelLoopHead]);
  b.emit(Op.Label, [labelLoopHead]);
  const kBlock = b.id(); b.emit(Op.Load, [tU32, kBlock, varKBlock]);
  const loopCmp = b.id(); b.emit(Op.ULessThan, [tBool, loopCmp, kBlock, numKBlocks]);
  b.emit(Op.LoopMerge, [labelLoopMerge, labelLoopCont, 0]);
  b.emit(Op.BranchConditional, [loopCmp, labelLoopBody, labelLoopMerge]);
  b.emit(Op.Label, [labelLoopBody]);

  const kBlockBase = b.id(); b.emit(Op.IMul, [tU32, kBlockBase, kBlock, constBc]);
  // Causal fast-path: skip blocks that are entirely in the future for all qTiles.
  // If kBlockBase > min(qBlockBase[last] + Br - 1, T - 1), all scores are masked.
  const qBlockBaseFirst = qBlockBaseArr[0];
  const qBlockBaseLastTile = qBlockBaseArr[qTiles - 1];
  const hasAnyQRows = b.id(); b.emit(Op.ULessThan, [tBool, hasAnyQRows, qBlockBaseFirst, T]);
  const qBlockLastRaw = b.id(); b.emit(Op.IAdd, [tU32, qBlockLastRaw, qBlockBaseLastTile, constBrMinus1]);
  const tMinus1 = b.id(); b.emit(Op.ISub, [tU32, tMinus1, T, const1u]);
  const qBlockLastInRange = b.id(); b.emit(Op.ULessThan, [tBool, qBlockLastInRange, qBlockLastRaw, T]);
  const qBlockLast = b.id(); b.emit(Op.Select, [tU32, qBlockLast, qBlockLastInRange, qBlockLastRaw, tMinus1]);
  const fullFutureBlock = b.id(); b.emit(Op.ULessThan, [tBool, fullFutureBlock, qBlockLast, kBlockBase]);
  const noValidQRows = b.id(); b.emit(Op.LogicalNot, [tBool, noValidQRows, hasAnyQRows]);
  const skipBlock = b.id(); b.emit(Op.LogicalOr, [tBool, skipBlock, noValidQRows, fullFutureBlock]);
  b.emit(Op.BranchConditional, [skipBlock, labelLoopSkip, labelLoopCompute]);
  b.emit(Op.Label, [labelLoopCompute]);

  const kBlockEnd = b.id(); b.emit(Op.IAdd, [tU32, kBlockEnd, kBlockBase, constBc]);
  const fullTBlock = b.id(); b.emit(Op.UGreaterThanEqual, [tBool, fullTBlock, T, kBlockEnd]);

  // ── KV load / double-buffer prefetch ────────────────────────────────
  if (doubleBuf) {
    // Compute buffer offsets from current buffer index
    const bufIdx = b.id(); b.emit(Op.Load, [tU32, bufIdx, dbVarBufIdx]);
    curKOff = b.id(); b.emit(Op.IMul, [tU32, curKOff, bufIdx, dbConstBufSize]);
    curVOff = b.id(); b.emit(Op.IMul, [tU32, curVOff, bufIdx, dbConstBufSize]);
    // Next buffer offset: (1 - bufIdx) * bufSize
    const oneMinusBuf = b.id(); b.emit(Op.ISub, [tU32, oneMinusBuf, const1u, bufIdx]);
    const nextKOff = b.id(); b.emit(Op.IMul, [tU32, nextKOff, oneMinusBuf, dbConstBufSize]);
    const nextVOff = b.id(); b.emit(Op.IMul, [tU32, nextVOff, oneMinusBuf, dbConstBufSize]);
    // Prefetch KV[k+1] into next buffer (if there is a next kBlock)
    const nextK = b.id(); b.emit(Op.IAdd, [tU32, nextK, kBlock, const1u]);
    const hasNext = b.id(); b.emit(Op.ULessThan, [tBool, hasNext, nextK, numKBlocks]);
    const labelPrefetch = b.id(); const labelPrefetchEnd = b.id();
    b.emit(Op.SelectionMerge, [labelPrefetchEnd, 0]);
    b.emit(Op.BranchConditional, [hasNext, labelPrefetch, labelPrefetchEnd]);
    b.emit(Op.Label, [labelPrefetch]);
    {
      const nextKBase = b.id(); b.emit(Op.IMul, [tU32, nextKBase, nextK, constBc]);
      const nextKEnd = b.id(); b.emit(Op.IAdd, [tU32, nextKEnd, nextKBase, constBc]);
      const nextFT = b.id(); b.emit(Op.UGreaterThanEqual, [tBool, nextFT, T, nextKEnd]);
      emitKVLoad(nextKBase, nextFT, nextKOff, nextVOff);
    }
    b.emit(Op.Branch, [labelPrefetchEnd]);
    b.emit(Op.Label, [labelPrefetchEnd]);
  } else {
    curKOff = const0u;
    curVOff = const0u;
    emitKVLoad(kBlockBase, fullTBlock, const0u, const0u);
    b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]); // barrier: K/V loaded
  }

  if (!isKVOnlyMode)
  for (let qti = 0; qti < qTiles; qti++) {
    const qBlockBase = qBlockBaseArr[qti];
    const varO = varOArr[qti];
    const varM = varMArr[qti];
    const varL = varLArr[qti];

    let probeMatrix: number | null = null;
    let pForPV: number | null = null;

    if (!isPVOnlyMode) {
    // ── Step 1: QK^T via cooperative matrix (or synthetic S for per_elem_only) ─
    let accS: number;
    if (isPerElemOnlyMode) {
      // per_elem_only: fill accumulator with 1.0 (synthetic S, same memory layout)
      const ones = b.id();
      b.emit(Op.OpCooperativeMatrixPerElementOpNV, [tCoopAcc, ones, constNullAcc, fnConstOne]);
      accS = ones;
    } else {
    // S[16x16] = Q[16xD] @ K^T[Dx16]
    {
      const varAcc = b.id(); b.emit(Op.Variable, [tPtrFnCoopAcc, varAcc, StorageClass.Function]);
      b.emit(Op.Store, [varAcc, constNullAcc]);

      for (let kStep = 0; kStep < kSteps; kStep++) {
        const constAOff = b.id(); b.constant(tU32, constAOff, kStep * coopK);
        const ptrA = b.id(); b.emit(Op.AccessChain, [tPtrSharedF16, ptrA, sQ, constAOff]);
        const constStrideD = b.id(); b.constant(tU32, constStrideD, D);
        const coopA = b.id();
        b.emit(Op.OpCooperativeMatrixLoadKHR, [tCoopA, coopA, ptrA, constRowMajor, constStrideD]);

        // Load B (K^T) from sK with ColumnMajor layout.
        // sK stores K[Bc, D] row-major. ColumnMajor+strideD reads K^T[D, Bc]:
        //   element[i, j] = *(ptr + j*D + i) = sK[j*D + kStep*coopK + i]
        const constBOff = b.id(); b.constant(tU32, constBOff, kStep * coopK);
        const bOffBuf = b.id(); b.emit(Op.IAdd, [tU32, bOffBuf, constBOff, curKOff]);
        const ptrB = b.id(); b.emit(Op.AccessChain, [tPtrSharedF16, ptrB, sK, bOffBuf]);
        const coopB = b.id();
        b.emit(Op.OpCooperativeMatrixLoadKHR, [tCoopB, coopB, ptrB, constColMajor, constStrideD]);

        const prevAcc = b.id(); b.emit(Op.Load, [tCoopAcc, prevAcc, varAcc]);
        const newAcc = b.id();
        b.emit(Op.OpCooperativeMatrixMulAddKHR, [tCoopAcc, newAcc, coopA, coopB, prevAcc]);
        b.emit(Op.Store, [varAcc, newAcc]);
      }
      accS = b.id(); b.emit(Op.Load, [tCoopAcc, accS, varAcc]);
    }
    }

    if (isQKOnlyMode) {
      probeMatrix = accS;
    } else if (useManualSoftmax) {
      // ════════════════════════════════════════════════════════════════════════
      // MANUAL SOFTMAX: scalar ops + subgroup clustered reduce
      // Replaces PerElementOpNV/ReduceNV with explicit shmem round-trips.
      // ════════════════════════════════════════════════════════════════════════

      // Store S to sScratch[0..255] as f32 row-major, stride=Bc
      const ptrScrS = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrScrS, sScratch, const0u]);
      b.emit(Op.OpCooperativeMatrixStoreKHR, [ptrScrS, accS, constRowMajor, constBc]);
      b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]); // S visible in sScratch

      // Each thread reads its 2 elements: s0 = S[myRow, myCol0], s1 = S[myRow, myCol1]
      const ptrS0 = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrS0, sScratch, myScrIdx0]);
      let s0 = b.id(); b.emit(Op.Load, [tF32, s0, ptrS0]);
      const ptrS1 = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrS1, sScratch, myScrIdx1]);
      let s1 = b.id(); b.emit(Op.Load, [tF32, s1, ptrS1]);

      // Scale
      const s0s = b.id(); b.emit(Op.FMul, [tF32, s0s, s0, scale]);
      const s1s = b.id(); b.emit(Op.FMul, [tF32, s1s, s1, scale]);
      s0 = s0s; s1 = s1s;

      // SoftCap: s = softCap * tanh(s * invSoftCap)
      if (useSoftCap) {
        const inv = useSoftCapConst ? constInvSoftCap : b.id();
        if (!useSoftCapConst) b.emit(Op.FDiv, [tF32, inv, const1f, softCap]);
        const cap = useSoftCapConst ? constSoftCap : softCap;
        const si0 = b.id(); b.emit(Op.FMul, [tF32, si0, s0, inv]);
        const si1 = b.id(); b.emit(Op.FMul, [tF32, si1, s1, inv]);
        const th0 = b.id(); b.emit(Op.ExtInst, [tF32, th0, glslStd, GLSLstd450.Tanh, si0]);
        const th1 = b.id(); b.emit(Op.ExtInst, [tF32, th1, glslStd, GLSLstd450.Tanh, si1]);
        const c0 = b.id(); b.emit(Op.FMul, [tF32, c0, th0, cap]);
        const c1 = b.id(); b.emit(Op.FMul, [tF32, c1, th1, cap]);
        s0 = c0; s1 = c1;
      }

      // Causal + OOB mask: if (qRow < kCol || kCol >= T) s = -inf
      const qRowMS = b.id(); b.emit(Op.IAdd, [tU32, qRowMS, qBlockBase, myRow]);
      const kCol0MS = b.id(); b.emit(Op.IAdd, [tU32, kCol0MS, kBlockBase, myCol0]);
      const kCol1MS = b.id(); b.emit(Op.IAdd, [tU32, kCol1MS, kBlockBase, myCol1]);
      const causal0 = b.id(); b.emit(Op.ULessThan, [tBool, causal0, qRowMS, kCol0MS]);
      const oob0 = b.id(); b.emit(Op.UGreaterThanEqual, [tBool, oob0, kCol0MS, T]);
      const mask0 = b.id(); b.emit(Op.LogicalOr, [tBool, mask0, causal0, oob0]);
      const ms0 = b.id(); b.emit(Op.Select, [tF32, ms0, mask0, constNegInf, s0]);
      const causal1 = b.id(); b.emit(Op.ULessThan, [tBool, causal1, qRowMS, kCol1MS]);
      const oob1 = b.id(); b.emit(Op.UGreaterThanEqual, [tBool, oob1, kCol1MS, T]);
      const mask1 = b.id(); b.emit(Op.LogicalOr, [tBool, mask1, causal1, oob1]);
      const ms1 = b.id(); b.emit(Op.Select, [tF32, ms1, mask1, constNegInf, s1]);
      s0 = ms0; s1 = ms1;

      // Row max via subgroup clustered reduce
      const localMax = b.id(); b.emit(Op.ExtInst, [tF32, localMax, glslStd, GLSLstd450.FMax, s0, s1]);
      const rowMax = b.id();
      b.emit(Op.GroupNonUniformFMax, [tF32, rowMax, scopeSubgroup, 3 /* ClusteredReduce */, localMax, constCluster8]);

      // Update m: mNew = max(mPrev, rowMax)
      const mPrevS = b.id(); b.emit(Op.Load, [tF32, mPrevS, varMyMArr[qti]]);
      const mNewS = b.id(); b.emit(Op.ExtInst, [tF32, mNewS, glslStd, GLSLstd450.FMax, mPrevS, rowMax]);
      b.emit(Op.Store, [varMyMArr[qti], mNewS]);

      // blockAlpha = exp2((mPrev - mNew) * LOG2E)
      const mDiff = b.id(); b.emit(Op.FSub, [tF32, mDiff, mPrevS, mNewS]);
      const mDiffL2 = b.id(); b.emit(Op.FMul, [tF32, mDiffL2, mDiff, constLog2e]);
      const blockAlphaS = b.id(); b.emit(Op.ExtInst, [tF32, blockAlphaS, glslStd, GLSLstd450.Exp2, mDiffL2]);

      // P = exp2((s - mNew) * LOG2E)
      const sd0 = b.id(); b.emit(Op.FSub, [tF32, sd0, s0, mNewS]);
      const sd0l = b.id(); b.emit(Op.FMul, [tF32, sd0l, sd0, constLog2e]);
      const p0 = b.id(); b.emit(Op.ExtInst, [tF32, p0, glslStd, GLSLstd450.Exp2, sd0l]);
      const sd1 = b.id(); b.emit(Op.FSub, [tF32, sd1, s1, mNewS]);
      const sd1l = b.id(); b.emit(Op.FMul, [tF32, sd1l, sd1, constLog2e]);
      const p1 = b.id(); b.emit(Op.ExtInst, [tF32, p1, glslStd, GLSLstd450.Exp2, sd1l]);

      // Row sum via subgroup clustered reduce
      const localSum = b.id(); b.emit(Op.FAdd, [tF32, localSum, p0, p1]);
      const rowSum = b.id();
      b.emit(Op.GroupNonUniformFAdd, [tF32, rowSum, scopeSubgroup, 3 /* ClusteredReduce */, localSum, constCluster8]);

      // Update l: lNew = lPrev * blockAlpha + rowSum
      const lPrevS = b.id(); b.emit(Op.Load, [tF32, lPrevS, varMyLArr[qti]]);
      const lScaledS = b.id(); b.emit(Op.FMul, [tF32, lScaledS, lPrevS, blockAlphaS]);
      const lNewS = b.id(); b.emit(Op.FAdd, [tF32, lNewS, lScaledS, rowSum]);
      b.emit(Op.Store, [varMyLArr[qti], lNewS]);

      // Write P and blockAlpha to sScratch for loading as coop matrices.
      // P at [0..255], blockAlpha at [256..511] (broadcast across columns).
      const ptrP0 = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrP0, sScratch, myScrIdx0]);
      b.emit(Op.Store, [ptrP0, p0]);
      const ptrP1 = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrP1, sScratch, myScrIdx1]);
      b.emit(Op.Store, [ptrP1, p1]);
      const constScrOff256 = b.id(); b.constant(tU32, constScrOff256, Br * Bc);
      const aIdx0 = b.id(); b.emit(Op.IAdd, [tU32, aIdx0, constScrOff256, myScrIdx0]);
      const ptrA0 = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrA0, sScratch, aIdx0]);
      b.emit(Op.Store, [ptrA0, blockAlphaS]);
      const aIdx1 = b.id(); b.emit(Op.IAdd, [tU32, aIdx1, constScrOff256, myScrIdx1]);
      const ptrA1 = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrA1, sScratch, aIdx1]);
      b.emit(Op.Store, [ptrA1, blockAlphaS]);

      b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]); // P and blockAlpha visible

      // Load P as coop accumulator f32 → FConvert to f16 MatrixA
      const ptrPMat = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrPMat, sScratch, const0u]);
      const pAcc = b.id();
      b.emit(Op.OpCooperativeMatrixLoadKHR, [tCoopAcc, pAcc, ptrPMat, constRowMajor, constBc]);
      const pF16A = b.id(); b.emit(Op.FConvert, [tCoopA, pF16A, pAcc]);

      // Load blockAlpha as coop accumulator
      const ptrAMat = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrAMat, sScratch, constScrOff256]);
      const blockAlphaMat = b.id();
      b.emit(Op.OpCooperativeMatrixLoadKHR, [tCoopAcc, blockAlphaMat, ptrAMat, constRowMajor, constBc]);

      // Rescale O by blockAlpha
      for (let t = 0; t < pvTiles; t++) {
        const oOld = b.id(); b.emit(Op.Load, [tCoopAcc, oOld, varO[t]]);
        const oScaled = b.id(); b.emit(Op.FMul, [tCoopAcc, oScaled, oOld, blockAlphaMat]);
        b.emit(Op.Store, [varO[t], oScaled]);
      }

      // PV matmul: O += P @ V
      for (let rn = 0; rn < pvTiles; rn++) {
        const constVOff = b.id(); b.constant(tU32, constVOff, rn * coopN);
        const vOffBuf = b.id(); b.emit(Op.IAdd, [tU32, vOffBuf, constVOff, curVOff]);
        const ptrV = b.id(); b.emit(Op.AccessChain, [tPtrSharedF16, ptrV, sV, vOffBuf]);
        const constStrideVD = b.id(); b.constant(tU32, constStrideVD, D);
        const coopV = b.id();
        b.emit(Op.OpCooperativeMatrixLoadKHR, [tCoopB, coopV, ptrV, constRowMajor, constStrideVD]);

        const oOld = b.id(); b.emit(Op.Load, [tCoopAcc, oOld, varO[rn]]);
        const oNew = b.id();
        b.emit(Op.OpCooperativeMatrixMulAddKHR, [tCoopAcc, oNew, pF16A, coopV, oOld]);
        b.emit(Op.Store, [varO[rn], oNew]);
      }

    } else {
      // ════════════════════════════════════════════════════════════════════════
      // LEGACY SOFTMAX: PerElementOpNV / ReduceNV path (probe modes)
      // ════════════════════════════════════════════════════════════════════════
      const kBlockLast = b.id(); b.emit(Op.IAdd, [tU32, kBlockLast, kBlockBase, constBcMinus1]);
      const fullCausalBlock = b.id(); b.emit(Op.UGreaterThanEqual, [tBool, fullCausalBlock, qBlockBase, kBlockLast]);
      const canUseScaleOnly = b.id(); b.emit(Op.LogicalAnd, [tBool, canUseScaleOnly, fullCausalBlock, fullTBlock]);

      const varScaledS = b.id(); b.emit(Op.Variable, [tPtrFnCoopAcc, varScaledS, StorageClass.Function]);
      const labelScaleFast = b.id();
      const labelScaleMaskSelect = b.id();
      const labelScaleMaskNoOob = b.id();
      const labelScaleMaskWithOob = b.id();
      const labelScaleMaskMerge = b.id();
      const labelScaleMerge = b.id();
      b.emit(Op.SelectionMerge, [labelScaleMerge, 0]);
      b.emit(Op.BranchConditional, [canUseScaleOnly, labelScaleFast, labelScaleMaskSelect]);

      b.emit(Op.Label, [labelScaleFast]);
      {
        const scaledFast = b.id();
        if (useSoftCap) {
          if (useSoftCapConst) {
            b.emit(Op.OpCooperativeMatrixPerElementOpNV, [tCoopAcc, scaledFast, accS, fnScaleSoftCap, scaleInvSoftCap]);
          } else {
            b.emit(Op.OpCooperativeMatrixPerElementOpNV, [tCoopAcc, scaledFast, accS, fnScaleSoftCap, scale, softCap]);
          }
        } else {
          b.emit(Op.OpCooperativeMatrixPerElementOpNV, [tCoopAcc, scaledFast, accS, fnMulScalar, scale]);
        }
        b.emit(Op.Store, [varScaledS, scaledFast]);
      }
      b.emit(Op.Branch, [labelScaleMerge]);

      b.emit(Op.Label, [labelScaleMaskSelect]);
      b.emit(Op.SelectionMerge, [labelScaleMaskMerge, 0]);
      b.emit(Op.BranchConditional, [fullTBlock, labelScaleMaskNoOob, labelScaleMaskWithOob]);

      b.emit(Op.Label, [labelScaleMaskNoOob]);
      {
        const scaledMasked = b.id();
        if (useSoftCap) {
          if (useSoftCapConst) {
            b.emit(Op.OpCooperativeMatrixPerElementOpNV, [tCoopAcc, scaledMasked, accS, fnScaleMaskCausalNoOob, scaleInvSoftCap, qBlockBase, kBlockBase]);
          } else {
            b.emit(Op.OpCooperativeMatrixPerElementOpNV, [tCoopAcc, scaledMasked, accS, fnScaleMaskCausalNoOob, scale, softCap, qBlockBase, kBlockBase]);
          }
        } else {
          b.emit(Op.OpCooperativeMatrixPerElementOpNV, [tCoopAcc, scaledMasked, accS, fnScaleMaskCausalNoOob, scale, qBlockBase, kBlockBase]);
        }
        b.emit(Op.Store, [varScaledS, scaledMasked]);
      }
      b.emit(Op.Branch, [labelScaleMaskMerge]);

      b.emit(Op.Label, [labelScaleMaskWithOob]);
      {
        const scaledMasked = b.id();
        if (useSoftCap) {
          if (useSoftCapConst) {
            b.emit(Op.OpCooperativeMatrixPerElementOpNV, [tCoopAcc, scaledMasked, accS, fnScaleMask, scaleInvSoftCap, qBlockBase, kBlockBase, T]);
          } else {
            b.emit(Op.OpCooperativeMatrixPerElementOpNV, [tCoopAcc, scaledMasked, accS, fnScaleMask, scale, softCap, qBlockBase, kBlockBase, T]);
          }
        } else {
          b.emit(Op.OpCooperativeMatrixPerElementOpNV, [tCoopAcc, scaledMasked, accS, fnScaleMask, scale, qBlockBase, kBlockBase, T]);
        }
        b.emit(Op.Store, [varScaledS, scaledMasked]);
      }
      b.emit(Op.Branch, [labelScaleMaskMerge]);

      b.emit(Op.Label, [labelScaleMaskMerge]);
      b.emit(Op.Branch, [labelScaleMerge]);

      b.emit(Op.Label, [labelScaleMerge]);
      const scaledS = b.id(); b.emit(Op.Load, [tCoopAcc, scaledS, varScaledS]);

      if (isQKMaskMode) {
        probeMatrix = scaledS;
      } else {
        const rowMaxS = b.id();
        b.emit(Op.OpCooperativeMatrixReduceNV, [tCoopAcc, rowMaxS, scaledS, CooperativeMatrixReduce.Row, fnFMaxCombine]);
        const mPrev = b.id(); b.emit(Op.Load, [tCoopAcc, mPrev, varM]);
        const mNew = b.id();
        b.emit(Op.OpCooperativeMatrixPerElementOpNV, [tCoopAcc, mNew, mPrev, fnMax2, rowMaxS]);
        const blockAlpha = b.id();
        b.emit(Op.OpCooperativeMatrixPerElementOpNV, [tCoopAcc, blockAlpha, mPrev, fnExpSub, mNew]);
        const P = b.id();
        b.emit(Op.OpCooperativeMatrixPerElementOpNV, [tCoopAcc, P, scaledS, fnExpSub, mNew]);
        const rowSumP = b.id();
        b.emit(Op.OpCooperativeMatrixReduceNV, [tCoopAcc, rowSumP, P, CooperativeMatrixReduce.Row, fnFAddCombine]);

        if (isQKSoftmaxMode || isPerElemOnlyMode) {
          const oOld = b.id(); b.emit(Op.Load, [tCoopAcc, oOld, varO[0]]);
          const oScaled = b.id(); b.emit(Op.FMul, [tCoopAcc, oScaled, oOld, blockAlpha]);
          const oNew = b.id(); b.emit(Op.FAdd, [tCoopAcc, oNew, oScaled, P]);
          b.emit(Op.Store, [varO[0], oNew]);
          const lOld = b.id(); b.emit(Op.Load, [tCoopAcc, lOld, varL]);
          const lScaled = b.id(); b.emit(Op.FMul, [tCoopAcc, lScaled, lOld, blockAlpha]);
          const lNew = b.id(); b.emit(Op.FAdd, [tCoopAcc, lNew, lScaled, rowSumP]);
          b.emit(Op.Store, [varL, lNew]);
          b.emit(Op.Store, [varM, mNew]);
        } else if (isFullMode || isKVSynthMode) {
          // ── Step 8: Rescale O and update l/m ──────────────────────────────
          for (let t = 0; t < pvTiles; t++) {
            const oOld = b.id(); b.emit(Op.Load, [tCoopAcc, oOld, varO[t]]);
            const oScaled = b.id(); b.emit(Op.FMul, [tCoopAcc, oScaled, oOld, blockAlpha]);
            b.emit(Op.Store, [varO[t], oScaled]);
          }
          const lOld = b.id(); b.emit(Op.Load, [tCoopAcc, lOld, varL]);
          const lScaled = b.id(); b.emit(Op.FMul, [tCoopAcc, lScaled, lOld, blockAlpha]);
          const lNew = b.id(); b.emit(Op.FAdd, [tCoopAcc, lNew, lScaled, rowSumP]);
          b.emit(Op.Store, [varL, lNew]);
          b.emit(Op.Store, [varM, mNew]);
        }
        pForPV = P;
      }
    }
    } else {
      // PV-only probe: use synthetic uniform probabilities P = 1/Bc.
      const ones = b.id();
      b.emit(Op.OpCooperativeMatrixPerElementOpNV, [tCoopAcc, ones, constNullAcc, fnConstOne]);
      const pSynthetic = b.id();
      b.emit(Op.OpCooperativeMatrixPerElementOpNV, [tCoopAcc, pSynthetic, ones, fnMulScalar, constInvBc]);
      pForPV = pSynthetic;
    }

    if (probeMatrix !== null) {
      const oOld = b.id(); b.emit(Op.Load, [tCoopAcc, oOld, varO[0]]);
      const oNew = b.id(); b.emit(Op.FAdd, [tCoopAcc, oNew, oOld, probeMatrix]);
      b.emit(Op.Store, [varO[0], oNew]);
    }

    if (pForPV !== null) {
      // ── Step 9: PV matmul ──────────────────────────────────────────────────
      const pF16A = b.id(); b.emit(Op.FConvert, [tCoopA, pF16A, pForPV]);

      for (let rn = 0; rn < pvTiles; rn++) {
        const constVOff = b.id(); b.constant(tU32, constVOff, rn * coopN);
        const vOffBuf = b.id(); b.emit(Op.IAdd, [tU32, vOffBuf, constVOff, curVOff]);
        const ptrV = b.id(); b.emit(Op.AccessChain, [tPtrSharedF16, ptrV, sV, vOffBuf]);
        const constStrideVD = b.id(); b.constant(tU32, constStrideVD, D);
        const coopV = b.id();
        b.emit(Op.OpCooperativeMatrixLoadKHR, [tCoopB, coopV, ptrV, constRowMajor, constStrideVD]);

        const oOld = b.id(); b.emit(Op.Load, [tCoopAcc, oOld, varO[rn]]);
        const oNew = b.id();
        b.emit(Op.OpCooperativeMatrixMulAddKHR, [tCoopAcc, oNew, pF16A, coopV, oOld]);
        b.emit(Op.Store, [varO[rn], oNew]);
      }
    }
  }

  // ── Double-buffer: barrier after compute + flip buffer index ──────────
  if (doubleBuf) {
    b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]); // barrier: prefetch visible + compute done
    const curBuf = b.id(); b.emit(Op.Load, [tU32, curBuf, dbVarBufIdx]);
    const flipped = b.id(); b.emit(Op.ISub, [tU32, flipped, const1u, curBuf]);
    b.emit(Op.Store, [dbVarBufIdx, flipped]);
  }

  // ── kBlock loop continuation ─────────────────────────────────────────────
  b.emit(Op.Branch, [labelLoopCont]);
  b.emit(Op.Label, [labelLoopSkip]);
  b.emit(Op.Branch, [labelLoopCont]);
  b.emit(Op.Label, [labelLoopCont]);
  const nextKBlock = b.id(); b.emit(Op.Load, [tU32, nextKBlock, varKBlock]);
  const incKBlock = b.id(); b.emit(Op.IAdd, [tU32, incKBlock, nextKBlock, const1u]);
  b.emit(Op.Store, [varKBlock, incKBlock]);
  b.emit(Op.Branch, [labelLoopHead]);
  b.emit(Op.Label, [labelLoopMerge]);

  // ════════════════════════════════════════════════════════════════════════
  // OUTPUT
  // ════════════════════════════════════════════════════════════════════════
  if (isFullMode || isKVSynthMode) {
    for (let qti = 0; qti < qTiles; qti++) {
      const qBlockBase = qBlockBaseArr[qti];
      const varO = varOArr[qti];

    const dimsPerThread = D / threadsPerRow;

    // Normalize O by l
    if (useManualSoftmax) {
      // Manual softmax: broadcast per-thread scalar l to scratch → coop matrix
      const myLFinal = b.id(); b.emit(Op.Load, [tF32, myLFinal, varMyLArr[qti]]);
      const ptrLScr0 = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrLScr0, sScratch, myScrIdx0]);
      b.emit(Op.Store, [ptrLScr0, myLFinal]);
      const ptrLScr1 = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrLScr1, sScratch, myScrIdx1]);
      b.emit(Op.Store, [ptrLScr1, myLFinal]);
      b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);
      const ptrLMat = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrLMat, sScratch, const0u]);
      const lMat = b.id();
      b.emit(Op.OpCooperativeMatrixLoadKHR, [tCoopAcc, lMat, ptrLMat, constRowMajor, constBc]);
      const onesForInv = b.id();
      b.emit(Op.OpCooperativeMatrixPerElementOpNV, [tCoopAcc, onesForInv, constNullAcc, fnConstOne]);
      const lInv = b.id(); b.emit(Op.FDiv, [tCoopAcc, lInv, onesForInv, lMat]);
      for (let rn = 0; rn < pvTiles; rn++) {
        const oRaw = b.id(); b.emit(Op.Load, [tCoopAcc, oRaw, varO[rn]]);
        const oNorm = b.id(); b.emit(Op.FMul, [tCoopAcc, oNorm, oRaw, lInv]);
        b.emit(Op.Store, [varO[rn], oNorm]);
      }
    } else {
      // Legacy NV: use coop matrix l directly
      const varL = varLArr[qti];
      const lFinal = b.id(); b.emit(Op.Load, [tCoopAcc, lFinal, varL]);
      const onesForInv = b.id();
      b.emit(Op.OpCooperativeMatrixPerElementOpNV, [tCoopAcc, onesForInv, constNullAcc, fnConstOne]);
      const lInv = b.id(); b.emit(Op.FDiv, [tCoopAcc, lInv, onesForInv, lFinal]);
      for (let rn = 0; rn < pvTiles; rn++) {
        const oRaw = b.id(); b.emit(Op.Load, [tCoopAcc, oRaw, varO[rn]]);
        const oNorm = b.id(); b.emit(Op.FMul, [tCoopAcc, oNorm, oRaw, lInv]);
        b.emit(Op.Store, [varO[rn], oNorm]);
      }
    }

    const qBlockEndOut = b.id(); b.emit(Op.IAdd, [tU32, qBlockEndOut, qBlockBase, constBr]);
    const qBlockFullInBoundsOut = b.id(); b.emit(Op.UGreaterThanEqual, [tBool, qBlockFullInBoundsOut, T, qBlockEndOut]);
    const labelOWriteDirect = b.id();
    const labelOWriteSlow = b.id();
    const labelOWriteMerge = b.id();
    b.emit(Op.SelectionMerge, [labelOWriteMerge, 0]);
    b.emit(Op.BranchConditional, [qBlockFullInBoundsOut, labelOWriteDirect, labelOWriteSlow]);

    b.emit(Op.Label, [labelOWriteDirect]);
    {
      const qBlockBaseD = b.id(); b.emit(Op.IMul, [tU32, qBlockBaseD, qBlockBase, constD]);
      const oGlobalTileBase = b.id(); b.emit(Op.IAdd, [tU32, oGlobalTileBase, baseOff, qBlockBaseD]);
      const constStrideD = b.id(); b.constant(tU32, constStrideD, D);
      for (let rn = 0; rn < pvTiles; rn++) {
        const constColOff = b.id(); b.constant(tU32, constColOff, rn * coopN);
        const tileBaseOff = b.id(); b.emit(Op.IAdd, [tU32, tileBaseOff, oGlobalTileBase, constColOff]);
        const ptrOTile = b.id(); b.emit(Op.AccessChain, [bufO.tPtr, ptrOTile, bufO.varId, const0u, tileBaseOff]);
        const oTile = b.id(); b.emit(Op.Load, [tCoopAcc, oTile, varO[rn]]);
        b.emit(Op.OpCooperativeMatrixStoreKHR, [ptrOTile, oTile, constRowMajor, constStrideD]);
      }
    }
    b.emit(Op.Branch, [labelOWriteMerge]);

    b.emit(Op.Label, [labelOWriteSlow]);
    {
      b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]); // barrier: protect scratch reuse

      // Store all O tiles to scratch
      for (let rn = 0; rn < pvTiles; rn++) {
        const constColOff = b.id(); b.constant(tU32, constColOff, rn * coopN);
        const ptrOScr = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrOScr, sScratch, constColOff]);
        const constStrideD = b.id(); b.constant(tU32, constStrideD, D);
        const oTile = b.id(); b.emit(Op.Load, [tCoopAcc, oTile, varO[rn]]);
        b.emit(Op.OpCooperativeMatrixStoreKHR, [ptrOScr, oTile, constRowMajor, constStrideD]);
      }

      b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]); // barrier: O stored

      // Write O to global
      const constDimsPerThread = b.id(); b.constant(tU32, constDimsPerThread, dimsPerThread);
      const myDimStart = b.id(); b.emit(Op.IMul, [tU32, myDimStart, myColInRow, constDimsPerThread]);

      const qRowO = b.id(); b.emit(Op.IAdd, [tU32, qRowO, qBlockBase, myRow]);
      const oInBounds = b.id(); b.emit(Op.ULessThan, [tBool, oInBounds, qRowO, T]);
      const labelOWriteTail = b.id(); const labelOWriteTailEnd = b.id();
      b.emit(Op.SelectionMerge, [labelOWriteTailEnd, 0]);
      b.emit(Op.BranchConditional, [oInBounds, labelOWriteTail, labelOWriteTailEnd]);
      b.emit(Op.Label, [labelOWriteTail]);

      const qRowOD = b.id(); b.emit(Op.IMul, [tU32, qRowOD, qRowO, constD]);
      const oGlobalBase = b.id(); b.emit(Op.IAdd, [tU32, oGlobalBase, baseOff, qRowOD]);

      for (let d = 0; d < dimsPerThread; d++) {
        const constDIdx = b.id(); b.constant(tU32, constDIdx, d);
        const rowTimesD = b.id(); b.emit(Op.IMul, [tU32, rowTimesD, myRow, constD]);
        const dimOff = b.id(); b.emit(Op.IAdd, [tU32, dimOff, myDimStart, constDIdx]);
        const scrIdx = b.id(); b.emit(Op.IAdd, [tU32, scrIdx, rowTimesD, dimOff]);
        const ptrScr = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrScr, sScratch, scrIdx]);
        const oVal = b.id(); b.emit(Op.Load, [tF32, oVal, ptrScr]);

        const gDimOff = b.id(); b.emit(Op.IAdd, [tU32, gDimOff, oGlobalBase, dimOff]);
        const ptrO = b.id(); b.emit(Op.AccessChain, [bufO.tPtr, ptrO, bufO.varId, const0u, gDimOff]);
        b.emit(Op.Store, [ptrO, oVal]);
      }

      b.emit(Op.Branch, [labelOWriteTailEnd]);
      b.emit(Op.Label, [labelOWriteTailEnd]);
    }
    b.emit(Op.Branch, [labelOWriteMerge]);
    b.emit(Op.Label, [labelOWriteMerge]);

    if (!skipLseWrite) {
      const qRow = b.id(); b.emit(Op.IAdd, [tU32, qRow, qBlockBase, myRow]);
      const qRowInBounds = b.id(); b.emit(Op.ULessThan, [tBool, qRowInBounds, qRow, T]);
      const isLSEWriter = b.id(); b.emit(Op.IEqual, [tBool, isLSEWriter, myColInRow, const0u]);
      const writeLSE = b.id(); b.emit(Op.LogicalAnd, [tBool, writeLSE, qRowInBounds, isLSEWriter]);
      const labelLSE = b.id(); const labelLSEEnd = b.id();
      if (useManualSoftmax) {
        // Per-thread scalar m/l — no coop matrix extraction needed.
        const myMFinal = b.id(); b.emit(Op.Load, [tF32, myMFinal, varMyMArr[qti]]);
        const myLFinal2 = b.id(); b.emit(Op.Load, [tF32, myLFinal2, varMyLArr[qti]]);
        b.emit(Op.SelectionMerge, [labelLSEEnd, 0]);
        b.emit(Op.BranchConditional, [writeLSE, labelLSE, labelLSEEnd]);
        b.emit(Op.Label, [labelLSE]);
        {
          const log2L = b.id(); b.emit(Op.ExtInst, [tF32, log2L, glslStd, GLSLstd450.Log2, myLFinal2]);
          const logL = b.id(); b.emit(Op.FMul, [tF32, logL, log2L, constLn2]);
          const lseVal = b.id(); b.emit(Op.FAdd, [tF32, lseVal, myMFinal, logL]);
          const lseIdx = b.id(); b.emit(Op.IAdd, [tU32, lseIdx, lseBaseOff, qRow]);
          const ptrLSE = b.id(); b.emit(Op.AccessChain, [bufLSE.tPtr, ptrLSE, bufLSE.varId, const0u, lseIdx]);
          b.emit(Op.Store, [ptrLSE, lseVal]);
        }
      } else {
        // Extract m/l from coop matrices to scratch for scalar LSE computation.
        const varM = varMArr[qti];
        const varL = varLArr[qti];
        const mFinal = b.id(); b.emit(Op.Load, [tCoopAcc, mFinal, varM]);
        const ptrScratchM = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrScratchM, sScratch, const0u]);
        b.emit(Op.OpCooperativeMatrixStoreKHR, [ptrScratchM, mFinal, constRowMajor, constBc]);
        const const256u = b.id(); b.constant(tU32, const256u, Br * Bc);
        const lFinalNV = b.id(); b.emit(Op.Load, [tCoopAcc, lFinalNV, varL]);
        const ptrScratchL = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrScratchL, sScratch, const256u]);
        b.emit(Op.OpCooperativeMatrixStoreKHR, [ptrScratchL, lFinalNV, constRowMajor, constBc]);
        b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);
        b.emit(Op.SelectionMerge, [labelLSEEnd, 0]);
        b.emit(Op.BranchConditional, [writeLSE, labelLSE, labelLSEEnd]);
        b.emit(Op.Label, [labelLSE]);
        {
          const mScrIdx = b.id(); b.emit(Op.IMul, [tU32, mScrIdx, myRow, constBc]);
          const ptrMV = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrMV, sScratch, mScrIdx]);
          const mVal = b.id(); b.emit(Op.Load, [tF32, mVal, ptrMV]);
          const lScrIdx = b.id(); b.emit(Op.IAdd, [tU32, lScrIdx, const256u, mScrIdx]);
          const ptrLV = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrLV, sScratch, lScrIdx]);
          const lVal = b.id(); b.emit(Op.Load, [tF32, lVal, ptrLV]);
          const log2L = b.id(); b.emit(Op.ExtInst, [tF32, log2L, glslStd, GLSLstd450.Log2, lVal]);
          const logL = b.id(); b.emit(Op.FMul, [tF32, logL, log2L, constLn2]);
          const lseVal = b.id(); b.emit(Op.FAdd, [tF32, lseVal, mVal, logL]);
          const lseIdx = b.id(); b.emit(Op.IAdd, [tU32, lseIdx, lseBaseOff, qRow]);
          const ptrLSE = b.id(); b.emit(Op.AccessChain, [bufLSE.tPtr, ptrLSE, bufLSE.varId, const0u, lseIdx]);
          b.emit(Op.Store, [ptrLSE, lseVal]);
        }
      }
      b.emit(Op.Branch, [labelLSEEnd]);
      b.emit(Op.Label, [labelLSEEnd]);
    }
    }
  } else {
    for (let qti = 0; qti < qTiles; qti++) {
      const qBlockBase = qBlockBaseArr[qti];
      const varO = varOArr[qti];

    // Probe modes: write zero LSE and raw accumulator output (no final normalization).
    const myRow = b.id();
    if (constLog2TPR) b.emit(Op.ShiftRightLogical, [tU32, myRow, tid, constLog2TPR]);
    else b.emit(Op.UDiv, [tU32, myRow, tid, constThreadsPerRow]);
    const myColInRow = b.id();
    if (constMaskTPR) b.emit(Op.BitwiseAnd, [tU32, myColInRow, tid, constMaskTPR]);
    else b.emit(Op.UMod, [tU32, myColInRow, tid, constThreadsPerRow]);
    const dimsPerThread = D / threadsPerRow;

    const qRow = b.id(); b.emit(Op.IAdd, [tU32, qRow, qBlockBase, myRow]);
    const qRowInBounds = b.id(); b.emit(Op.ULessThan, [tBool, qRowInBounds, qRow, T]);
    const isLSEWriter = b.id(); b.emit(Op.IEqual, [tBool, isLSEWriter, myColInRow, const0u]);
    const writeLSE = b.id(); b.emit(Op.LogicalAnd, [tBool, writeLSE, qRowInBounds, isLSEWriter]);
    const labelLSE = b.id(); const labelLSEEnd = b.id();
    b.emit(Op.SelectionMerge, [labelLSEEnd, 0]);
    b.emit(Op.BranchConditional, [writeLSE, labelLSE, labelLSEEnd]);
    b.emit(Op.Label, [labelLSE]);
    {
      const lseIdx = b.id(); b.emit(Op.IAdd, [tU32, lseIdx, lseBaseOff, qRow]);
      const ptrLSE = b.id(); b.emit(Op.AccessChain, [bufLSE.tPtr, ptrLSE, bufLSE.varId, const0u, lseIdx]);
      b.emit(Op.Store, [ptrLSE, const0f]);
    }
    b.emit(Op.Branch, [labelLSEEnd]);
    b.emit(Op.Label, [labelLSEEnd]);

    const qBlockEndOut = b.id(); b.emit(Op.IAdd, [tU32, qBlockEndOut, qBlockBase, constBr]);
    const qBlockFullInBoundsOut = b.id(); b.emit(Op.UGreaterThanEqual, [tBool, qBlockFullInBoundsOut, T, qBlockEndOut]);
    const labelProbeWriteDirect = b.id();
    const labelProbeWriteSlow = b.id();
    const labelProbeWriteMerge = b.id();
    b.emit(Op.SelectionMerge, [labelProbeWriteMerge, 0]);
    b.emit(Op.BranchConditional, [qBlockFullInBoundsOut, labelProbeWriteDirect, labelProbeWriteSlow]);

    b.emit(Op.Label, [labelProbeWriteDirect]);
    {
      const qBlockBaseD = b.id(); b.emit(Op.IMul, [tU32, qBlockBaseD, qBlockBase, constD]);
      const oGlobalTileBase = b.id(); b.emit(Op.IAdd, [tU32, oGlobalTileBase, baseOff, qBlockBaseD]);
      const constStrideD = b.id(); b.constant(tU32, constStrideD, D);
      for (let rn = 0; rn < pvTiles; rn++) {
        const constColOff = b.id(); b.constant(tU32, constColOff, rn * coopN);
        const tileBaseOff = b.id(); b.emit(Op.IAdd, [tU32, tileBaseOff, oGlobalTileBase, constColOff]);
        const ptrOTile = b.id(); b.emit(Op.AccessChain, [bufO.tPtr, ptrOTile, bufO.varId, const0u, tileBaseOff]);
        const oTile = b.id(); b.emit(Op.Load, [tCoopAcc, oTile, varO[rn]]);
        b.emit(Op.OpCooperativeMatrixStoreKHR, [ptrOTile, oTile, constRowMajor, constStrideD]);
      }
    }
    b.emit(Op.Branch, [labelProbeWriteMerge]);

    b.emit(Op.Label, [labelProbeWriteSlow]);
    {
      b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]); // barrier: protect scratch reuse

      for (let rn = 0; rn < pvTiles; rn++) {
        const constColOff = b.id(); b.constant(tU32, constColOff, rn * coopN);
        const ptrOScr = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrOScr, sScratch, constColOff]);
        const constStrideD = b.id(); b.constant(tU32, constStrideD, D);
        const oTile = b.id(); b.emit(Op.Load, [tCoopAcc, oTile, varO[rn]]);
        b.emit(Op.OpCooperativeMatrixStoreKHR, [ptrOScr, oTile, constRowMajor, constStrideD]);
      }

      b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]); // barrier: O stored

      const constDimsPerThread = b.id(); b.constant(tU32, constDimsPerThread, dimsPerThread);
      const myDimStart = b.id(); b.emit(Op.IMul, [tU32, myDimStart, myColInRow, constDimsPerThread]);

      const qRowO = b.id(); b.emit(Op.IAdd, [tU32, qRowO, qBlockBase, myRow]);
      const oInBounds = b.id(); b.emit(Op.ULessThan, [tBool, oInBounds, qRowO, T]);
      const labelProbeWriteTail = b.id(); const labelProbeWriteTailEnd = b.id();
      b.emit(Op.SelectionMerge, [labelProbeWriteTailEnd, 0]);
      b.emit(Op.BranchConditional, [oInBounds, labelProbeWriteTail, labelProbeWriteTailEnd]);
      b.emit(Op.Label, [labelProbeWriteTail]);

      const qRowOD = b.id(); b.emit(Op.IMul, [tU32, qRowOD, qRowO, constD]);
      const oGlobalBase = b.id(); b.emit(Op.IAdd, [tU32, oGlobalBase, baseOff, qRowOD]);

      for (let d = 0; d < dimsPerThread; d++) {
        const constDIdx = b.id(); b.constant(tU32, constDIdx, d);
        const rowTimesD = b.id(); b.emit(Op.IMul, [tU32, rowTimesD, myRow, constD]);
        const dimOff = b.id(); b.emit(Op.IAdd, [tU32, dimOff, myDimStart, constDIdx]);
        const scrIdx = b.id(); b.emit(Op.IAdd, [tU32, scrIdx, rowTimesD, dimOff]);
        const ptrScr = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrScr, sScratch, scrIdx]);
        const oVal = b.id(); b.emit(Op.Load, [tF32, oVal, ptrScr]);

        const gDimOff = b.id(); b.emit(Op.IAdd, [tU32, gDimOff, oGlobalBase, dimOff]);
        const ptrO = b.id(); b.emit(Op.AccessChain, [bufO.tPtr, ptrO, bufO.varId, const0u, gDimOff]);
        b.emit(Op.Store, [ptrO, oVal]);
      }

      b.emit(Op.Branch, [labelProbeWriteTailEnd]);
      b.emit(Op.Label, [labelProbeWriteTailEnd]);
    }
    b.emit(Op.Branch, [labelProbeWriteMerge]);
    b.emit(Op.Label, [labelProbeWriteMerge]);
    }
  }

  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);
  return b.build();
}
