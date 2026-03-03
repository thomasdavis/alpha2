/**
 * kernels/attention-coop.ts — Cooperative matrix flash attention forward kernel.
 *
 * Uses VK_KHR_cooperative_matrix for hardware-accelerated QK^T and PV matmuls.
 * FP16 inputs → FP32 accumulator for the matmuls. Online softmax + O accumulation
 * remain scalar (per-row, in FP32 registers).
 *
 * Architecture (per ChatGPT consultation rounds 1-3):
 *   - Br=16, Bc=16, WG=64 (2 subgroups of 32)
 *   - Step 1: QK^T via coop matrix → sS[Br*Bc] in shared (FP32)
 *   - Step 2: Two-pass softmax (max scan + P computation) → sP[Br*Bc] in shared (FP16)
 *   - Step 3: PV via coop matrix → sDeltaO[Br*D] in shared (FP32)
 *   - Step 4: O update: regO = regO * blockAlpha + sDeltaO (scalar, threads 0..Br-1)
 *   - Block-alpha deferred O rescaling (exp(m_prev - m_final) applied once per kBlock)
 *
 * Bindings:
 *   0: Q [B*H, T, D] f32 (readonly)
 *   1: K [B*H, T, D] f32 (readonly)
 *   2: V [B*H, T, D] f32 (readonly)
 *   3: O [B*H, T, D] f32 (write)
 *   4: LSE [B*H, T]  f32 (write)
 *
 * Push constants (16 bytes = 4 x f32):
 *   member 0: T (sequence length, as float)
 *   member 1: scale (1/sqrt(D))
 *   member 2: _pad0
 *   member 3: _pad1
 *
 * Dispatch: (ceil(T/Br), B*H, 1)
 * Workgroup: (WG, 1, 1) where WG = 64
 */

import {
  SpirVBuilder, Op, Capability, CooperativeMatrixUse, GroupOperation,
  AddressingModel, MemoryModel as MemModelConst, ExecutionModel, ExecutionMode,
  StorageClass, Decoration, BuiltIn, FunctionControl, Scope, MemorySemantics,
  GLSLstd450, declareParamsPushConstant,
} from "./helpers.js";

// Declare f32 SSBO (same pattern as matmul-coop.ts)
function declareF32Ssbo(
  b: SpirVBuilder, tF32: number, set: number, binding: number,
  readonly_: boolean, writeonly_ = false,
): { varId: number; tPtrF32: number } {
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
  return { varId, tPtrF32 };
}

export function kernelFlashAttentionCoopForward(Br: number, Bc: number, D: number): Uint32Array {
  const WG = 64; // 2 subgroups of 32
  const coopM = 16, coopN = 16, coopK = 16;
  const kSteps = D / coopK;  // D=64 → 4
  const pvTiles = D / coopN; // D=64 → 4

  const b = new SpirVBuilder();

  // ── Capabilities & extensions ────────────────────────────────────────────
  b.addCapability(Capability.Shader);
  b.addCapability(Capability.Float16);
  b.addCapability(Capability.VulkanMemoryModel);
  b.addCapability(Capability.CooperativeMatrixKHR);
  b.addCapability(Capability.StorageBufferStorageClass);
  b.addCapability(Capability.GroupNonUniform);
  b.addCapability(Capability.GroupNonUniformArithmetic);
  b.addCapability(Capability.GroupNonUniformClustered);

  b.addExtension("SPV_KHR_cooperative_matrix");
  b.addExtension("SPV_KHR_vulkan_memory_model");
  b.addExtension("SPV_KHR_storage_buffer_storage_class");

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
  const tFnVoid  = b.id(); b.typeFunction(tFnVoid, tVoid);

  // ── Constants ────────────────────────────────────────────────────────────
  const const0u = b.id(); b.constant(tU32, const0u, 0);
  const const1u = b.id(); b.constant(tU32, const1u, 1);
  const const2u = b.id(); b.constant(tU32, const2u, 2);
  const const0f = b.id(); b.constantF32(tF32, const0f, 0.0);
  const const1f = b.id(); b.constantF32(tF32, const1f, 1.0);
  const constNegInf = b.id(); b.constant(tF32, constNegInf, 0xFF800000);
  const constLog2e = b.id(); b.constantF32(tF32, constLog2e, Math.LOG2E);

  const constBr = b.id(); b.constant(tU32, constBr, Br);
  const constBc = b.id(); b.constant(tU32, constBc, Bc);
  const constD  = b.id(); b.constant(tU32, constD, D);
  const constWG = b.id(); b.constant(tU32, constWG, WG);
  const constBrMinus1 = b.id(); b.constant(tU32, constBrMinus1, Br - 1);

  // ── Cooperative matrix types ─────────────────────────────────────────────
  const scopeSubgroup = b.id(); b.constant(tU32, scopeSubgroup, Scope.Subgroup);
  const scopeWg = b.id(); b.constant(tU32, scopeWg, Scope.Workgroup);

  const constCoopM = b.id(); b.constant(tU32, constCoopM, coopM);
  const constCoopN = b.id(); b.constant(tU32, constCoopN, coopN);
  const constCoopK = b.id(); b.constant(tU32, constCoopK, coopK);
  const constUseA   = b.id(); b.constant(tU32, constUseA, CooperativeMatrixUse.MatrixA);
  const constUseB   = b.id(); b.constant(tU32, constUseB, CooperativeMatrixUse.MatrixB);
  const constUseAcc = b.id(); b.constant(tU32, constUseAcc, CooperativeMatrixUse.MatrixAccumulator);

  // A: f16, M(=Br=16) x K(=coopK=16), MatrixA
  const tCoopA = b.id();
  b.typeCooperativeMatrixKHR(tCoopA, tF16, scopeSubgroup, constCoopM, constCoopK, constUseA);
  // B: f16, K(=coopK=16) x N(=coopN=16), MatrixB
  const tCoopB = b.id();
  b.typeCooperativeMatrixKHR(tCoopB, tF16, scopeSubgroup, constCoopK, constCoopN, constUseB);
  // Accumulator: f32, M(=16) x N(=16)
  const tCoopAcc = b.id();
  b.typeCooperativeMatrixKHR(tCoopAcc, tF32, scopeSubgroup, constCoopM, constCoopN, constUseAcc);

  const constRowMajor = const0u;  // RowMajorKHR = 0
  const constColMajor = const1u;  // ColumnMajorKHR = 1
  const constNullAcc = b.id(); b.constantNull(tCoopAcc, constNullAcc);

  // ── Pointer types ────────────────────────────────────────────────────────
  const tPtrFnF32 = b.id(); b.typePointer(tPtrFnF32, StorageClass.Function, tF32);
  const tPtrFnU32 = b.id(); b.typePointer(tPtrFnU32, StorageClass.Function, tU32);
  const tPtrSharedF16 = b.id(); b.typePointer(tPtrSharedF16, StorageClass.Workgroup, tF16);
  const tPtrSharedF32 = b.id(); b.typePointer(tPtrSharedF32, StorageClass.Workgroup, tF32);
  const tPtrFnCoopAcc = b.id(); b.typePointer(tPtrFnCoopAcc, StorageClass.Function, tCoopAcc);

  // ── Storage buffers ──────────────────────────────────────────────────────
  const bufQ   = declareF32Ssbo(b, tF32, 0, 0, true);
  const bufK   = declareF32Ssbo(b, tF32, 0, 1, true);
  const bufV   = declareF32Ssbo(b, tF32, 0, 2, true);
  const bufO   = declareF32Ssbo(b, tF32, 0, 3, false, true);
  const bufLSE = declareF32Ssbo(b, tF32, 0, 4, false, true);

  // Push constants: { T, scale, _pad0, _pad1 }
  const pc = declareParamsPushConstant(b, tF32, 4);

  // ── Shared memory ────────────────────────────────────────────────────────
  // sQ[Br * D] f16 — query block (loaded once)
  const constSQSize = b.id(); b.constant(tU32, constSQSize, Br * D);
  const tArrSQ = b.id(); b.typeArray(tArrSQ, tF16, constSQSize);
  const tPtrArrSQ = b.id(); b.typePointer(tPtrArrSQ, StorageClass.Workgroup, tArrSQ);
  const sQ = b.id(); b.variable(tPtrArrSQ, sQ, StorageClass.Workgroup);

  // sKT[D * Bc] f16 — key block, stored transposed as K^T[D, Bc]
  const constSKTSize = b.id(); b.constant(tU32, constSKTSize, D * Bc);
  const tArrSKT = b.id(); b.typeArray(tArrSKT, tF16, constSKTSize);
  const tPtrArrSKT = b.id(); b.typePointer(tPtrArrSKT, StorageClass.Workgroup, tArrSKT);
  const sKT = b.id(); b.variable(tPtrArrSKT, sKT, StorageClass.Workgroup);

  // sV[Bc * D] f16 — value block, row-major
  const constSVSize = b.id(); b.constant(tU32, constSVSize, Bc * D);
  const tArrSV = b.id(); b.typeArray(tArrSV, tF16, constSVSize);
  const tPtrArrSV = b.id(); b.typePointer(tPtrArrSV, StorageClass.Workgroup, tArrSV);
  const sV = b.id(); b.variable(tPtrArrSV, sV, StorageClass.Workgroup);

  // sS[Br * Bc] f32 — QK^T scores
  const constSSSize = b.id(); b.constant(tU32, constSSSize, Br * Bc);
  const tArrSS = b.id(); b.typeArray(tArrSS, tF32, constSSSize);
  const tPtrArrSS = b.id(); b.typePointer(tPtrArrSS, StorageClass.Workgroup, tArrSS);
  const sS = b.id(); b.variable(tPtrArrSS, sS, StorageClass.Workgroup);

  // sP[Br * Bc] f16 — attention weights for PV
  const constSPSize = b.id(); b.constant(tU32, constSPSize, Br * Bc);
  const tArrSP = b.id(); b.typeArray(tArrSP, tF16, constSPSize);
  const tPtrArrSP = b.id(); b.typePointer(tPtrArrSP, StorageClass.Workgroup, tArrSP);
  const sP = b.id(); b.variable(tPtrArrSP, sP, StorageClass.Workgroup);

  // sDeltaO[Br * D] f32 — PV output
  const constSDOSize = b.id(); b.constant(tU32, constSDOSize, Br * D);
  const tArrSDO = b.id(); b.typeArray(tArrSDO, tF32, constSDOSize);
  const tPtrArrSDO = b.id(); b.typePointer(tPtrArrSDO, StorageClass.Workgroup, tArrSDO);
  const sDeltaO = b.id(); b.variable(tPtrArrSDO, sDeltaO, StorageClass.Workgroup);

  // ── Built-in variables ───────────────────────────────────────────────────
  const tPtrInputVec3 = b.id(); b.typePointer(tPtrInputVec3, StorageClass.Input, tVec3U32);
  const vWorkgroupId = b.id(); b.variable(tPtrInputVec3, vWorkgroupId, StorageClass.Input);
  b.addDecorate(vWorkgroupId, Decoration.BuiltIn, BuiltIn.WorkgroupId);
  const vLocalId = b.id(); b.variable(tPtrInputVec3, vLocalId, StorageClass.Input);
  b.addDecorate(vLocalId, Decoration.BuiltIn, BuiltIn.LocalInvocationId);

  const semAcqRelWg = b.id();
  b.constant(tU32, semAcqRelWg, MemorySemantics.AcquireRelease | MemorySemantics.WorkgroupMemory);

  // ── Function: main ───────────────────────────────────────────────────────
  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [vWorkgroupId, vLocalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, WG, 1, 1);
  b.emit(Op.Function, [tVoid, fnMain, FunctionControl.None, tFnVoid]);
  const labelEntry = b.id(); b.emit(Op.Label, [labelEntry]);

  // ── Function variables ───────────────────────────────────────────────────
  // regO[D/4]: per-thread output accumulator. 4 threads per row, each owns D/4 dims.
  const dimsPerThread = D / 4; // 16
  const colsPerThread = Bc / 4; // 4
  const regO: number[] = [];
  for (let d = 0; d < dimsPerThread; d++) {
    regO[d] = b.id();
    b.emit(Op.Variable, [tPtrFnF32, regO[d], StorageClass.Function]);
  }
  const varM = b.id(); b.emit(Op.Variable, [tPtrFnF32, varM, StorageClass.Function]);
  const varL = b.id(); b.emit(Op.Variable, [tPtrFnF32, varL, StorageClass.Function]);
  const varKBlock = b.id(); b.emit(Op.Variable, [tPtrFnU32, varKBlock, StorageClass.Function]);
  // Softmax scratch variables (must be declared in first block)
  const varLocalMax = b.id(); b.emit(Op.Variable, [tPtrFnF32, varLocalMax, StorageClass.Function]);
  const varLocalSum = b.id(); b.emit(Op.Variable, [tPtrFnF32, varLocalSum, StorageClass.Function]);

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

  // Base offsets
  const TD = b.id(); b.emit(Op.IMul, [tU32, TD, T, constD]);
  const baseOff = b.id(); b.emit(Op.IMul, [tU32, baseOff, bhIdx, TD]); // bh * T * D
  const lseBaseOff = b.id(); b.emit(Op.IMul, [tU32, lseBaseOff, bhIdx, T]); // bh * T

  const qBlockBase = b.id(); b.emit(Op.IMul, [tU32, qBlockBase, qBlockIdx, constBr]);

  // ── Thread role: 4 threads per row ────────────────────────────────────────
  // myRow = tid / 4 (0..15), myColInRow = tid % 4 (0..3)
  const const4u = b.id(); b.constant(tU32, const4u, 4);
  const myRow = b.id(); b.emit(Op.UDiv, [tU32, myRow, tid, const4u]);
  const myColInRow = b.id(); b.emit(Op.UMod, [tU32, myColInRow, tid, const4u]);
  // Column start for softmax: myColStart = myColInRow * colsPerThread
  const constColsPerThread = b.id(); b.constant(tU32, constColsPerThread, colsPerThread);
  const myColStart = b.id(); b.emit(Op.IMul, [tU32, myColStart, myColInRow, constColsPerThread]);
  // Dim start for regO: myDimStart = myColInRow * dimsPerThread
  const constDimsPerThread = b.id(); b.constant(tU32, constDimsPerThread, dimsPerThread);
  const myDimStart = b.id(); b.emit(Op.IMul, [tU32, myDimStart, myColInRow, constDimsPerThread]);

  // ── Initialize regO, m, l ────────────────────────────────────────────────
  for (let d = 0; d < dimsPerThread; d++) {
    b.emit(Op.Store, [regO[d], const0f]);
  }
  b.emit(Op.Store, [varM, constNegInf]);
  b.emit(Op.Store, [varL, const0f]);

  // ── Load Q block into sQ as f16 (cooperative, all 64 threads) ────────────
  // Q[bh, qBlockBase+row, col] → sQ[row * D + col] as f16
  // Total elements: Br * D = 16 * 64 = 1024. Each of 64 threads loads 16.
  {
    const elemsPerThread = (Br * D) / WG; // 16
    for (let i = 0; i < elemsPerThread; i++) {
      const constI = b.id(); b.constant(tU32, constI, i);
      const constE = b.id(); b.constant(tU32, constE, elemsPerThread);
      // sharedIdx = tid * elemsPerThread + i
      const tidTimesE = b.id(); b.emit(Op.IMul, [tU32, tidTimesE, tid, constE]);
      const sharedIdx = b.id(); b.emit(Op.IAdd, [tU32, sharedIdx, tidTimesE, constI]);
      // row = sharedIdx / D, col = sharedIdx % D
      const constDval = b.id(); b.constant(tU32, constDval, D);
      const row = b.id(); b.emit(Op.UDiv, [tU32, row, sharedIdx, constDval]);
      const col = b.id(); b.emit(Op.UMod, [tU32, col, sharedIdx, constDval]);
      const qRow = b.id(); b.emit(Op.IAdd, [tU32, qRow, qBlockBase, row]);
      // bounds check
      const inBounds = b.id(); b.emit(Op.ULessThan, [tBool, inBounds, qRow, T]);
      // global index: baseOff + qRow * D + col
      const qRowD = b.id(); b.emit(Op.IMul, [tU32, qRowD, qRow, constDval]);
      const gIdx = b.id(); b.emit(Op.IAdd, [tU32, gIdx, baseOff, qRowD]);
      const gIdx2 = b.id(); b.emit(Op.IAdd, [tU32, gIdx2, gIdx, col]);
      // load f32 from global
      const ptrG = b.id(); b.emit(Op.AccessChain, [bufQ.tPtrF32, ptrG, bufQ.varId, const0u, gIdx2]);
      const valF32 = b.id(); b.emit(Op.Load, [tF32, valF32, ptrG]);
      const maskedF32 = b.id(); b.emit(Op.Select, [tF32, maskedF32, inBounds, valF32, const0f]);
      // convert to f16 and store to shared
      const valF16 = b.id(); b.emit(Op.FConvert, [tF16, valF16, maskedF32]);
      const ptrSQ = b.id(); b.emit(Op.AccessChain, [tPtrSharedF16, ptrSQ, sQ, sharedIdx]);
      b.emit(Op.Store, [ptrSQ, valF16]);
    }
  }
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  // ── kBlock loop: kBlock = 0 .. numKBlocks-1 ─────────────────────────────
  const constBcMinus1 = b.id(); b.constant(tU32, constBcMinus1, Bc - 1);
  const numKBlocks = b.id();
  const TplusBcm1 = b.id(); b.emit(Op.IAdd, [tU32, TplusBcm1, T, constBcMinus1]);
  b.emit(Op.UDiv, [tU32, numKBlocks, TplusBcm1, constBc]);

  b.emit(Op.Store, [varKBlock, const0u]);
  const labelLoopHead = b.id(); const labelLoopBody = b.id();
  const labelLoopMerge = b.id(); const labelLoopCont = b.id();
  b.emit(Op.Branch, [labelLoopHead]);
  b.emit(Op.Label, [labelLoopHead]);
  const kBlock = b.id(); b.emit(Op.Load, [tU32, kBlock, varKBlock]);
  const loopCmp = b.id(); b.emit(Op.ULessThan, [tBool, loopCmp, kBlock, numKBlocks]);
  b.emit(Op.LoopMerge, [labelLoopMerge, labelLoopCont, 0]);
  b.emit(Op.BranchConditional, [loopCmp, labelLoopBody, labelLoopMerge]);
  b.emit(Op.Label, [labelLoopBody]);

  const kBlockBase = b.id(); b.emit(Op.IMul, [tU32, kBlockBase, kBlock, constBc]);

  // ── Step 0: Cooperative load K (transposed) and V into shared ────────────
  // K[bh, kBlockBase+j, d] → sKT[d * Bc + j] (transposed layout)
  // V[bh, kBlockBase+j, d] → sV[j * D + d] (row-major layout)
  // Total K elements: Bc * D = 1024, V elements: Bc * D = 1024
  // 2048 total loads across 64 threads = 32 per thread
  {
    const elemsPerThread = (Bc * D) / WG; // 16
    for (let i = 0; i < elemsPerThread; i++) {
      const constI = b.id(); b.constant(tU32, constI, i);
      const constE = b.id(); b.constant(tU32, constE, elemsPerThread);
      const tidTimesE = b.id(); b.emit(Op.IMul, [tU32, tidTimesE, tid, constE]);
      const flatIdx = b.id(); b.emit(Op.IAdd, [tU32, flatIdx, tidTimesE, constI]);
      // j = flatIdx / D, d = flatIdx % D
      const constDval = b.id(); b.constant(tU32, constDval, D);
      const j = b.id(); b.emit(Op.UDiv, [tU32, j, flatIdx, constDval]);
      const d = b.id(); b.emit(Op.UMod, [tU32, d, flatIdx, constDval]);
      const kRow = b.id(); b.emit(Op.IAdd, [tU32, kRow, kBlockBase, j]);
      const inBounds = b.id(); b.emit(Op.ULessThan, [tBool, inBounds, kRow, T]);
      // global index: baseOff + kRow * D + d
      const kRowD = b.id(); b.emit(Op.IMul, [tU32, kRowD, kRow, constDval]);
      const gIdx = b.id(); b.emit(Op.IAdd, [tU32, gIdx, baseOff, kRowD]);
      const gIdx2 = b.id(); b.emit(Op.IAdd, [tU32, gIdx2, gIdx, d]);

      // Load K and store transposed: sKT[d * Bc + j]
      const ptrGK = b.id(); b.emit(Op.AccessChain, [bufK.tPtrF32, ptrGK, bufK.varId, const0u, gIdx2]);
      const kF32 = b.id(); b.emit(Op.Load, [tF32, kF32, ptrGK]);
      const kMasked = b.id(); b.emit(Op.Select, [tF32, kMasked, inBounds, kF32, const0f]);
      const kF16 = b.id(); b.emit(Op.FConvert, [tF16, kF16, kMasked]);
      const constBcVal = b.id(); b.constant(tU32, constBcVal, Bc);
      const dTimesBc = b.id(); b.emit(Op.IMul, [tU32, dTimesBc, d, constBcVal]);
      const ktIdx = b.id(); b.emit(Op.IAdd, [tU32, ktIdx, dTimesBc, j]);
      const ptrSKT = b.id(); b.emit(Op.AccessChain, [tPtrSharedF16, ptrSKT, sKT, ktIdx]);
      b.emit(Op.Store, [ptrSKT, kF16]);

      // Load V and store row-major: sV[j * D + d]
      const ptrGV = b.id(); b.emit(Op.AccessChain, [bufV.tPtrF32, ptrGV, bufV.varId, const0u, gIdx2]);
      const vF32 = b.id(); b.emit(Op.Load, [tF32, vF32, ptrGV]);
      const vMasked = b.id(); b.emit(Op.Select, [tF32, vMasked, inBounds, vF32, const0f]);
      const vF16 = b.id(); b.emit(Op.FConvert, [tF16, vF16, vMasked]);
      // flatIdx = j * D + d, which is the same as the row-major index
      const ptrSV = b.id(); b.emit(Op.AccessChain, [tPtrSharedF16, ptrSV, sV, flatIdx]);
      b.emit(Op.Store, [ptrSV, vF16]);
    }
  }
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]); // barrier 1

  // ── Step 1: QK^T via cooperative matrix ──────────────────────────────────
  // S[16x16] = Q[16x64] @ K^T[64x16]
  // K^T stored as sKT[D, Bc], row-major with stride Bc
  // Q stored as sQ[Br, D], row-major with stride D
  // 4 kSteps: for each kStep, load Q[16x16] and K^T[16x16], accumulate
  {
    // Initialize accumulator
    const varAcc = b.id(); b.emit(Op.Variable, [tPtrFnCoopAcc, varAcc, StorageClass.Function]);
    b.emit(Op.Store, [varAcc, constNullAcc]);

    for (let kStep = 0; kStep < kSteps; kStep++) {
      // Load A tile from sQ: row=0, col=kStep*coopK, stride=D
      const aOffset = kStep * coopK; // column offset in sQ
      // In row-major, element at (r,c) is r*D+c. CoopMatLoad at (0, kStep*16) with stride D
      // pointer = &sQ[aOffset]
      const constAOff = b.id(); b.constant(tU32, constAOff, aOffset);
      const ptrA = b.id(); b.emit(Op.AccessChain, [tPtrSharedF16, ptrA, sQ, constAOff]);
      const constStrideD = b.id(); b.constant(tU32, constStrideD, D);
      const coopA = b.id();
      b.emit(Op.OpCooperativeMatrixLoadKHR, [tCoopA, coopA, ptrA, constRowMajor, constStrideD]);

      // Load B tile from sKT: row=kStep*coopK, col=0, stride=Bc
      const bOffset = kStep * coopK * Bc; // row offset in sKT (which is [D,Bc])
      const constBOff = b.id(); b.constant(tU32, constBOff, bOffset);
      const ptrB = b.id(); b.emit(Op.AccessChain, [tPtrSharedF16, ptrB, sKT, constBOff]);
      const constStrideBc = b.id(); b.constant(tU32, constStrideBc, Bc);
      const coopB = b.id();
      b.emit(Op.OpCooperativeMatrixLoadKHR, [tCoopB, coopB, ptrB, constRowMajor, constStrideBc]);

      // MMA: acc += A * B
      const prevAcc = b.id(); b.emit(Op.Load, [tCoopAcc, prevAcc, varAcc]);
      const newAcc = b.id();
      b.emit(Op.OpCooperativeMatrixMulAddKHR, [tCoopAcc, newAcc, coopA, coopB, prevAcc]);
      b.emit(Op.Store, [varAcc, newAcc]);
    }

    // Store accumulator to sS[Br * Bc] (f32)
    const ptrSS = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrSS, sS, const0u]);
    const finalAcc = b.id(); b.emit(Op.Load, [tCoopAcc, finalAcc, varAcc]);
    const constStrideBcF32 = b.id(); b.constant(tU32, constStrideBcF32, Bc);
    b.emit(Op.OpCooperativeMatrixStoreKHR, [ptrSS, finalAcc, constRowMajor, constStrideBcF32]);
  }
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]); // barrier 2

  // ── Step 2: Warp-packed softmax (ALL 64 threads) ────────────────────────
  // 4 threads per row: myRow = tid/4, myColInRow = tid%4
  // Each thread handles colsPerThread=4 columns in the Bc dimension.
  // ClusteredReduce(4) gives cross-thread row max/sum.
  // All 4 threads per row maintain identical m, l (synced via clustered reduce).
  {
    const qRow = b.id(); b.emit(Op.IAdd, [tU32, qRow, qBlockBase, myRow]);

    // Save m_prev for block-alpha
    const mPrev = b.id(); b.emit(Op.Load, [tF32, mPrev, varM]);

    // Pass 1: each thread scans its colsPerThread columns for local max
    b.emit(Op.Store, [varLocalMax, constNegInf]);
    for (let jLocal = 0; jLocal < colsPerThread; jLocal++) {
      const constJLocal = b.id(); b.constant(tU32, constJLocal, jLocal);
      // j = myColStart + jLocal
      const j = b.id(); b.emit(Op.IAdd, [tU32, j, myColStart, constJLocal]);
      // sS[myRow * Bc + j]
      const rowTimesBc = b.id(); b.emit(Op.IMul, [tU32, rowTimesBc, myRow, constBc]);
      const ssIdx = b.id(); b.emit(Op.IAdd, [tU32, ssIdx, rowTimesBc, j]);
      const ptrSSij = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrSSij, sS, ssIdx]);
      const rawScore = b.id(); b.emit(Op.Load, [tF32, rawScore, ptrSSij]);
      const scorePrescaled = b.id(); b.emit(Op.FMul, [tF32, scorePrescaled, rawScore, scale]);
      // causal mask: kCol = kBlockBase + j
      const kCol = b.id(); b.emit(Op.IAdd, [tU32, kCol, kBlockBase, j]);
      const causal = b.id(); b.emit(Op.ULessThan, [tBool, causal, qRow, kCol]);
      const oob = b.id(); b.emit(Op.UGreaterThanEqual, [tBool, oob, kCol, T]);
      const masked = b.id(); b.emit(Op.LogicalOr, [tBool, masked, causal, oob]);
      const score = b.id(); b.emit(Op.Select, [tF32, score, masked, constNegInf, scorePrescaled]);
      // localMax = max(localMax, score)
      const lmOld = b.id(); b.emit(Op.Load, [tF32, lmOld, varLocalMax]);
      const lmNew = b.id(); b.emit(Op.ExtInst, [tF32, lmNew, glslStd, GLSLstd450.FMax, lmOld, score]);
      b.emit(Op.Store, [varLocalMax, lmNew]);
    }

    // ClusteredReduce(FMax, clusterSize=4) → row max across 4 threads
    const localMaxVal = b.id(); b.emit(Op.Load, [tF32, localMaxVal, varLocalMax]);
    const rowMax = b.id();
    b.emit(Op.GroupNonUniformFMax, [tF32, rowMax, scopeSubgroup, GroupOperation.ClusteredReduce, localMaxVal, const4u]);

    // Update m = max(m, rowMax)
    const mOld = b.id(); b.emit(Op.Load, [tF32, mOld, varM]);
    const mNew = b.id(); b.emit(Op.ExtInst, [tF32, mNew, glslStd, GLSLstd450.FMax, mOld, rowMax]);
    b.emit(Op.Store, [varM, mNew]);

    // Block alpha = exp2((m_prev - m_new) * LOG2E)
    const mDiff = b.id(); b.emit(Op.FSub, [tF32, mDiff, mPrev, mNew]);
    const mDiffLog2 = b.id(); b.emit(Op.FMul, [tF32, mDiffLog2, mDiff, constLog2e]);
    const blockAlpha = b.id(); b.emit(Op.ExtInst, [tF32, blockAlpha, glslStd, GLSLstd450.Exp2, mDiffLog2]);

    // Rescale l
    const lOld = b.id(); b.emit(Op.Load, [tF32, lOld, varL]);
    const lRescaled = b.id(); b.emit(Op.FMul, [tF32, lRescaled, lOld, blockAlpha]);
    b.emit(Op.Store, [varL, lRescaled]);

    // Pass 2: compute P for each thread's columns, accumulate local sum
    b.emit(Op.Store, [varLocalSum, const0f]);
    const mCur = b.id(); b.emit(Op.Load, [tF32, mCur, varM]);
    for (let jLocal = 0; jLocal < colsPerThread; jLocal++) {
      const constJLocal = b.id(); b.constant(tU32, constJLocal, jLocal);
      const j = b.id(); b.emit(Op.IAdd, [tU32, j, myColStart, constJLocal]);
      const rowTimesBc = b.id(); b.emit(Op.IMul, [tU32, rowTimesBc, myRow, constBc]);
      const ssIdx = b.id(); b.emit(Op.IAdd, [tU32, ssIdx, rowTimesBc, j]);
      const ptrSSij = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrSSij, sS, ssIdx]);
      const rawScore = b.id(); b.emit(Op.Load, [tF32, rawScore, ptrSSij]);
      const scorePrescaled = b.id(); b.emit(Op.FMul, [tF32, scorePrescaled, rawScore, scale]);
      // causal mask
      const kCol = b.id(); b.emit(Op.IAdd, [tU32, kCol, kBlockBase, j]);
      const causal = b.id(); b.emit(Op.ULessThan, [tBool, causal, qRow, kCol]);
      const oob = b.id(); b.emit(Op.UGreaterThanEqual, [tBool, oob, kCol, T]);
      const masked = b.id(); b.emit(Op.LogicalOr, [tBool, masked, causal, oob]);
      const score = b.id(); b.emit(Op.Select, [tF32, score, masked, constNegInf, scorePrescaled]);
      // p = exp2((score - m) * LOG2E)
      const diff = b.id(); b.emit(Op.FSub, [tF32, diff, score, mCur]);
      const diffLog2 = b.id(); b.emit(Op.FMul, [tF32, diffLog2, diff, constLog2e]);
      const p = b.id(); b.emit(Op.ExtInst, [tF32, p, glslStd, GLSLstd450.Exp2, diffLog2]);
      // store p as f16 to sP[myRow * Bc + j]
      const pF16 = b.id(); b.emit(Op.FConvert, [tF16, pF16, p]);
      const spIdx = b.id(); b.emit(Op.IAdd, [tU32, spIdx, rowTimesBc, j]);
      const ptrSP = b.id(); b.emit(Op.AccessChain, [tPtrSharedF16, ptrSP, sP, spIdx]);
      b.emit(Op.Store, [ptrSP, pF16]);
      // localSum += p
      const lsCur = b.id(); b.emit(Op.Load, [tF32, lsCur, varLocalSum]);
      const lsNew = b.id(); b.emit(Op.FAdd, [tF32, lsNew, lsCur, p]);
      b.emit(Op.Store, [varLocalSum, lsNew]);
    }

    // ClusteredReduce(FAdd, clusterSize=4) → row sum across 4 threads
    const localSumVal = b.id(); b.emit(Op.Load, [tF32, localSumVal, varLocalSum]);
    const rowSum = b.id();
    b.emit(Op.GroupNonUniformFAdd, [tF32, rowSum, scopeSubgroup, GroupOperation.ClusteredReduce, localSumVal, const4u]);

    // l += rowSum
    const lCur = b.id(); b.emit(Op.Load, [tF32, lCur, varL]);
    const lNew = b.id(); b.emit(Op.FAdd, [tF32, lNew, lCur, rowSum]);
    b.emit(Op.Store, [varL, lNew]);

    // Rescale this thread's regO dims by block alpha (deferred rescale)
    for (let d = 0; d < dimsPerThread; d++) {
      const oOld = b.id(); b.emit(Op.Load, [tF32, oOld, regO[d]]);
      const oScaled = b.id(); b.emit(Op.FMul, [tF32, oScaled, oOld, blockAlpha]);
      b.emit(Op.Store, [regO[d], oScaled]);
    }
  }
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]); // barrier 3

  // ── Step 3: PV via cooperative matrix ────────────────────────────────────
  // deltaO[16x64] = P[16x16] @ V[16x64]
  // P in sP[Br, Bc] f16, row-major, stride=Bc
  // V in sV[Bc, D] f16, row-major, stride=D
  // 4 tiles across D (each 16x16)
  {
    for (let rn = 0; rn < pvTiles; rn++) {
      // Load P tile (same for all rn)
      const ptrP = b.id(); b.emit(Op.AccessChain, [tPtrSharedF16, ptrP, sP, const0u]);
      const constStridePBc = b.id(); b.constant(tU32, constStridePBc, Bc);
      const coopP = b.id();
      b.emit(Op.OpCooperativeMatrixLoadKHR, [tCoopA, coopP, ptrP, constRowMajor, constStridePBc]);

      // Load V tile: column offset rn*coopN, stride=D
      const vColOff = rn * coopN;
      const constVOff = b.id(); b.constant(tU32, constVOff, vColOff);
      const ptrV = b.id(); b.emit(Op.AccessChain, [tPtrSharedF16, ptrV, sV, constVOff]);
      const constStrideVD = b.id(); b.constant(tU32, constStrideVD, D);
      const coopV = b.id();
      b.emit(Op.OpCooperativeMatrixLoadKHR, [tCoopB, coopV, ptrV, constRowMajor, constStrideVD]);

      // deltaO tile = P @ V (fresh accumulator, not accumulating across tiles)
      const deltaOTile = b.id();
      b.emit(Op.OpCooperativeMatrixMulAddKHR, [tCoopAcc, deltaOTile, coopP, coopV, constNullAcc]);

      // Store deltaO tile to sDeltaO at column offset rn*16
      const constDOOff = b.id(); b.constant(tU32, constDOOff, rn * coopN);
      const ptrDO = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrDO, sDeltaO, constDOOff]);
      const constStrideDOD = b.id(); b.constant(tU32, constStrideDOD, D);
      b.emit(Op.OpCooperativeMatrixStoreKHR, [ptrDO, deltaOTile, constRowMajor, constStrideDOD]);
    }
  }
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]); // barrier 4

  // ── Step 4: Accumulate deltaO into regO (ALL 64 threads) ───────────────
  // Each thread reads its dimsPerThread dims from sDeltaO[myRow, myDimStart+d]
  {
    for (let d = 0; d < dimsPerThread; d++) {
      const constDIdx = b.id(); b.constant(tU32, constDIdx, d);
      // sDeltaO[myRow * D + myDimStart + d]
      const rowTimesD = b.id(); b.emit(Op.IMul, [tU32, rowTimesD, myRow, constD]);
      const dimOff = b.id(); b.emit(Op.IAdd, [tU32, dimOff, myDimStart, constDIdx]);
      const doIdx = b.id(); b.emit(Op.IAdd, [tU32, doIdx, rowTimesD, dimOff]);
      const ptrDO = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrDO, sDeltaO, doIdx]);
      const deltaO = b.id(); b.emit(Op.Load, [tF32, deltaO, ptrDO]);
      // regO[d] += deltaO (already rescaled in Step 2)
      const oOld = b.id(); b.emit(Op.Load, [tF32, oOld, regO[d]]);
      const oNew = b.id(); b.emit(Op.FAdd, [tF32, oNew, oOld, deltaO]);
      b.emit(Op.Store, [regO[d], oNew]);
    }
  }

  // ── kBlock loop continuation ─────────────────────────────────────────────
  b.emit(Op.Branch, [labelLoopCont]);
  b.emit(Op.Label, [labelLoopCont]);
  const nextKBlock = b.id(); b.emit(Op.Load, [tU32, nextKBlock, varKBlock]);
  const incKBlock = b.id(); b.emit(Op.IAdd, [tU32, incKBlock, nextKBlock, const1u]);
  b.emit(Op.Store, [varKBlock, incKBlock]);
  b.emit(Op.Branch, [labelLoopHead]);
  b.emit(Op.Label, [labelLoopMerge]);

  // ── Final: write O and LSE to global (ALL 64 threads) ───────────────────
  // Each thread writes its dimsPerThread dims. Only myColInRow==0 writes LSE.
  {
    const qRow = b.id(); b.emit(Op.IAdd, [tU32, qRow, qBlockBase, myRow]);
    const qRowInBounds = b.id(); b.emit(Op.ULessThan, [tBool, qRowInBounds, qRow, T]);
    const labelWrite = b.id(); const labelWriteEnd = b.id();
    b.emit(Op.SelectionMerge, [labelWriteEnd, 0]);
    b.emit(Op.BranchConditional, [qRowInBounds, labelWrite, labelWriteEnd]);
    b.emit(Op.Label, [labelWrite]);

    // O[bh, qRow, myDimStart + d] = regO[d] / l
    const lFinal = b.id(); b.emit(Op.Load, [tF32, lFinal, varL]);
    const invL = b.id(); b.emit(Op.FDiv, [tF32, invL, const1f, lFinal]);
    const qRowD = b.id(); b.emit(Op.IMul, [tU32, qRowD, qRow, constD]);
    const oGlobalBase = b.id(); b.emit(Op.IAdd, [tU32, oGlobalBase, baseOff, qRowD]);

    for (let d = 0; d < dimsPerThread; d++) {
      const constDIdx = b.id(); b.constant(tU32, constDIdx, d);
      const oVal = b.id(); b.emit(Op.Load, [tF32, oVal, regO[d]]);
      const oNorm = b.id(); b.emit(Op.FMul, [tF32, oNorm, oVal, invL]);
      // global dim index = myDimStart + d
      const dimOff = b.id(); b.emit(Op.IAdd, [tU32, dimOff, myDimStart, constDIdx]);
      const gIdx = b.id(); b.emit(Op.IAdd, [tU32, gIdx, oGlobalBase, dimOff]);
      const ptrO = b.id(); b.emit(Op.AccessChain, [bufO.tPtrF32, ptrO, bufO.varId, const0u, gIdx]);
      b.emit(Op.Store, [ptrO, oNorm]);
    }

    // LSE[bh, qRow] = m + log(l) — only one thread per row writes this
    const isLSEWriter = b.id(); b.emit(Op.IEqual, [tBool, isLSEWriter, myColInRow, const0u]);
    const labelLSE = b.id(); const labelLSEEnd = b.id();
    b.emit(Op.SelectionMerge, [labelLSEEnd, 0]);
    b.emit(Op.BranchConditional, [isLSEWriter, labelLSE, labelLSEEnd]);
    b.emit(Op.Label, [labelLSE]);

    const mFinal = b.id(); b.emit(Op.Load, [tF32, mFinal, varM]);
    const lFinalV = b.id(); b.emit(Op.Load, [tF32, lFinalV, varL]);
    const logL = b.id(); b.emit(Op.ExtInst, [tF32, logL, glslStd, GLSLstd450.Log, lFinalV]);
    const lseVal = b.id(); b.emit(Op.FAdd, [tF32, lseVal, mFinal, logL]);
    const lseIdx = b.id(); b.emit(Op.IAdd, [tU32, lseIdx, lseBaseOff, qRow]);
    const ptrLSE = b.id(); b.emit(Op.AccessChain, [bufLSE.tPtrF32, ptrLSE, bufLSE.varId, const0u, lseIdx]);
    b.emit(Op.Store, [ptrLSE, lseVal]);

    b.emit(Op.Branch, [labelLSEEnd]);
    b.emit(Op.Label, [labelLSEEnd]);
    b.emit(Op.Branch, [labelWriteEnd]);
    b.emit(Op.Label, [labelWriteEnd]);
  }

  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);
  return b.build();
}
