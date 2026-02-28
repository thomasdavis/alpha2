/**
 * kernels/matmul-coop.ts — Cooperative matrix (tensor core) GEMM kernels.
 *
 * Uses VK_KHR_cooperative_matrix for hardware-accelerated matrix multiply-add.
 * A/B are loaded as f16, accumulated in f32, output f32.
 *
 * Four variants: basic, batched, transposed, transposed+batched.
 *
 * Each workgroup (32 threads = one subgroup) computes one coopM x coopN output tile.
 * The K dimension is tiled: loop k=0..K step coopK, loading tiles cooperatively
 * from global memory into shared memory, then using OpCooperativeMatrixMulAddKHR.
 *
 * Push constants: { M: f32, N: f32, K: f32, _pad: f32 } — 16 bytes.
 * Bindings: 0=A(in), 1=B(in), 2=C(out)
 * Dispatch: (ceil(N/coopN), ceil(M/coopM), batchSize)
 */

import {
  SpirVBuilder, Op, Capability, CooperativeMatrixUse,
  AddressingModel, MemoryModel as MemModelConst, ExecutionModel, ExecutionMode,
  StorageClass, Decoration, BuiltIn, FunctionControl, Scope, MemorySemantics,
  declareParamsPushConstant,
} from "./helpers.js";

function isPowerOfTwo(v: number): boolean {
  return v > 0 && (v & (v - 1)) === 0;
}

function log2Pow2(v: number): number {
  return 31 - Math.clz32(v);
}

function declareStorageBufferF32Ssbo(
  b: SpirVBuilder,
  tF32: number,
  set: number,
  binding: number,
  readonly_: boolean,
  writeonly_?: boolean,
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

function declareStorageBufferF16Ssbo(
  b: SpirVBuilder,
  tF16: number,
  set: number,
  binding: number,
  readonly_: boolean,
): { varId: number; tPtrF16: number } {
  const tRuntimeArr = b.id();
  b.typeRuntimeArray(tRuntimeArr, tF16);
  b.addDecorate(tRuntimeArr, Decoration.ArrayStride, 2);

  const tStruct = b.id();
  b.typeStruct(tStruct, [tRuntimeArr]);
  b.addDecorate(tStruct, Decoration.Block);
  b.addMemberDecorate(tStruct, 0, Decoration.Offset, 0);
  if (readonly_) b.addMemberDecorate(tStruct, 0, Decoration.NonWritable);

  const tPtrStruct = b.id();
  b.typePointer(tPtrStruct, StorageClass.StorageBuffer, tStruct);
  const tPtrF16 = b.id();
  b.typePointer(tPtrF16, StorageClass.StorageBuffer, tF16);

  const varId = b.id();
  b.variable(tPtrStruct, varId, StorageClass.StorageBuffer);
  b.addDecorate(varId, Decoration.DescriptorSet, set);
  b.addDecorate(varId, Decoration.Binding, binding);

  return { varId, tPtrF16 };
}

/**
 * Emit cooperative matrix loads from shared memory + MMA operations.
 * Used by both single-buffer and double-buffer paths, and for the
 * double-buffer epilogue MMA after the K-loop.
 */
function emitCoopMatrixMMA(
  b: SpirVBuilder,
  tF16: number, tU32: number,
  tCoopA: number, tCoopB: number, tCoopCAcc: number,
  tPtrSharedF16: number,
  tileA: number, tileB: number,
  tileABase: number, tileBBase: number,
  constRowMajor: number,
  coopM: number, coopN: number, coopK: number,
  regTilesM: number, regTilesN: number,
  bTileStride: number,
  varAccs: number[][],
  shmemPadA = 0,
  shmemPadB = 0,
): void {
  const strideA = coopK + shmemPadA;
  const strideB = bTileStride + shmemPadB;
  const constStrideA = b.id(); b.constant(tU32, constStrideA, strideA);
  // Load A tiles: regTilesM tiles at offsets 0, coopM*strideA, 2*coopM*strideA, ...
  const coopAMats: number[] = [];
  for (let rm = 0; rm < regTilesM; rm++) {
    let aOffset: number;
    if (rm === 0) {
      aOffset = tileABase;
    } else {
      const constAOff = b.id(); b.constant(tU32, constAOff, rm * coopM * strideA);
      aOffset = b.id();
      b.emit(Op.IAdd, [tU32, aOffset, tileABase, constAOff]);
    }
    const ptrA = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF16, ptrA, tileA, aOffset]);
    coopAMats[rm] = b.id();
    b.emit(Op.OpCooperativeMatrixLoadKHR, [tCoopA, coopAMats[rm], ptrA, constRowMajor, constStrideA]);
  }
  // Load B tiles: regTilesN tiles at column offsets 0, coopN, 2*coopN, ...
  const constBTileStrideVal = b.id(); b.constant(tU32, constBTileStrideVal, strideB);
  const coopBMats: number[] = [];
  for (let rn = 0; rn < regTilesN; rn++) {
    let bOffset: number;
    if (rn === 0) {
      bOffset = tileBBase;
    } else {
      const constBOff = b.id(); b.constant(tU32, constBOff, rn * coopN);
      bOffset = b.id();
      b.emit(Op.IAdd, [tU32, bOffset, tileBBase, constBOff]);
    }
    const ptrB = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF16, ptrB, tileB, bOffset]);
    coopBMats[rn] = b.id();
    b.emit(Op.OpCooperativeMatrixLoadKHR, [tCoopB, coopBMats[rn], ptrB, constRowMajor, constBTileStrideVal]);
  }
  // MMA: C += A * B (register-tiled)
  for (let rm = 0; rm < regTilesM; rm++) {
    for (let rn = 0; rn < regTilesN; rn++) {
      const prevAcc = b.id();
      b.emit(Op.Load, [tCoopCAcc, prevAcc, varAccs[rm][rn]]);
      const newAcc = b.id();
      b.emit(Op.OpCooperativeMatrixMulAddKHR, [tCoopCAcc, newAcc, coopAMats[rm], coopBMats[rn], prevAcc]);
      b.emit(Op.Store, [varAccs[rm][rn], newAcc]);
    }
  }
}

/**
 * Build cooperative matrix matmul kernel.
 *
 * @param coopM - Cooperative matrix M tile size (rows of C)
 * @param coopN - Cooperative matrix N tile size (cols of C)
 * @param coopK - Cooperative matrix K tile size (inner dimension)
 * @param batched - If true, dispatch z = batch index
 * @param transposedB - If true, B is stored as [N,K] (row-major for each row of B^T)
 * @param transposedA - If true, A is stored as [K,M] while logical multiply uses A^T view
 */
function buildCoopMatmul(
  coopM: number,
  coopN: number,
  coopK: number,
  batched: boolean,
  transposedB: boolean,
  transposedA: boolean,
  inputF16: boolean,
  accumF16: boolean,
  subgroupTilesX: number,
  subgroupTilesY: number,
  regTilesM = 1,
  regTilesN = 1,
  doubleBuf = false,
): Uint32Array {
  const b = new SpirVBuilder();
  const coopDebugMode = process.env.HELIOS_COOP_DEBUG_MODE ?? "";
  const useDirectF16Load = inputF16 && process.env.HELIOS_COOP_DIRECT_LOAD === "1";
  // Double buffering only applies to the shared memory path
  const useDoubleBuf = doubleBuf && !useDirectF16Load;
  // Super-tile swizzle for L2 cache reuse (0=disabled, 2/4/8=super-tile width)
  const swizzleSize = parseInt(process.env.HELIOS_COOP_SWIZZLE ?? "0", 10);
  const useSwizzle = swizzleSize >= 2 && isPowerOfTwo(swizzleSize);
  // Shared memory padding per row to avoid bank conflicts (0=disabled, 8=NVIDIA default for f16)
  const shmemPad = useDirectF16Load ? 0 : parseInt(process.env.HELIOS_COOP_SHMEM_PAD ?? "8", 10);
  // K-tile multiplier: load kMulti K-steps at once, reducing barrier overhead (1/2/4)
  const kMulti = parseInt(process.env.HELIOS_COOP_K_MULTI ?? "4", 10);
  const kTileK = coopK * kMulti;

  // Capabilities
  b.addCapability(Capability.Shader);
  b.addCapability(Capability.Float16);
  b.addCapability(Capability.VulkanMemoryModel);
  b.addCapability(Capability.CooperativeMatrixKHR);
  b.addCapability(Capability.StorageBufferStorageClass);
  if (inputF16 || accumF16) {
    b.addCapability(Capability.StorageBuffer16BitAccess);
  }

  // Extension
  b.addExtension("SPV_KHR_cooperative_matrix");
  b.addExtension("SPV_KHR_vulkan_memory_model");
  b.addExtension("SPV_KHR_storage_buffer_storage_class");

  const glslStd = b.id();
  b.addExtInstImport(glslStd, "GLSL.std.450");
  b.setMemoryModel(AddressingModel.Logical, MemModelConst.Vulkan);

  // Types
  const tVoid = b.id(); b.typeVoid(tVoid);
  const tF32 = b.id();  b.typeFloat(tF32, 32);
  const tF16 = b.id();  b.typeFloat(tF16, 16);
  const tU32 = b.id();  b.typeInt(tU32, 32, 0);
  const tBool = b.id(); b.typeBool(tBool);
  const tVec3U32 = b.id(); b.typeVector(tVec3U32, tU32, 3);
  const tFnVoid = b.id(); b.typeFunction(tFnVoid, tVoid);

  // Scope constants (must be constants for cooperative matrix types)
  const scopeSubgroup = b.id();
  b.constant(tU32, scopeSubgroup, Scope.Subgroup);
  const scopeWg = b.id();
  b.constant(tU32, scopeWg, Scope.Workgroup);

  // Dimension constants for cooperative matrix types
  const constCoopM = b.id(); b.constant(tU32, constCoopM, coopM);
  const constCoopN = b.id(); b.constant(tU32, constCoopN, coopN);
  const constCoopK = b.id(); b.constant(tU32, constCoopK, coopK);
  const constUseA = b.id(); b.constant(tU32, constUseA, CooperativeMatrixUse.MatrixA);
  const constUseB = b.id(); b.constant(tU32, constUseB, CooperativeMatrixUse.MatrixB);
  const constUseAcc = b.id(); b.constant(tU32, constUseAcc, CooperativeMatrixUse.MatrixAccumulator);

  // Cooperative matrix types
  // A: f16, M x K, MatrixA
  const tCoopA = b.id();
  b.typeCooperativeMatrixKHR(tCoopA, tF16, scopeSubgroup, constCoopM, constCoopK, constUseA);
  // B: f16, K x N, MatrixB
  const tCoopB = b.id();
  b.typeCooperativeMatrixKHR(tCoopB, tF16, scopeSubgroup, constCoopK, constCoopN, constUseB);
  // C accumulator tile type (f32 default, optional f16)
  const tCoopCAcc = b.id();
  b.typeCooperativeMatrixKHR(
    tCoopCAcc,
    accumF16 ? tF16 : tF32,
    scopeSubgroup,
    constCoopM,
    constCoopN,
    constUseAcc,
  );
  // Output tile type is always f32 (we keep model-visible outputs in f32).
  const tCoopCOut = accumF16 ? b.id() : tCoopCAcc;
  if (accumF16) {
    b.typeCooperativeMatrixKHR(
      tCoopCOut,
      tF32,
      scopeSubgroup,
      constCoopM,
      constCoopN,
      constUseAcc,
    );
  }

  // Pointer types for cooperative matrices (Function storage class)
  const tPtrFnCoopC = b.id();
  b.typePointer(tPtrFnCoopC, StorageClass.Function, tCoopCAcc);

  // Common constants
  const const0u = b.id(); b.constant(tU32, const0u, 0);
  const const1u = b.id(); b.constant(tU32, const1u, 1);
  const const2u = b.id(); b.constant(tU32, const2u, 2);
  const const3u = b.id(); b.constant(tU32, const3u, 3);
  const const0f = b.id(); b.constantF32(tF32, const0f, 0.0);

  // Cooperative matrix layout constant: RowMajorKHR = 0
  const constRowMajor = const0u;
  // Cooperative matrix layout constant: ColumnMajorKHR = 1
  const constColumnMajor = const1u;
  const coopKPow2 = isPowerOfTwo(coopK);
  // B tile stride: with register tiling, each subgroup covers coopN*regTilesN columns
  const bTileStride = coopN * regTilesN;
  const bTileStridePow2 = isPowerOfTwo(bTileStride);
  // Padded strides for shared memory (avoids bank conflicts when shmemPad > 0)
  // A rows are kTileK wide (= coopK * kMulti) to hold multiple K-steps
  const paddedStrideA = kTileK + shmemPad;
  const paddedStrideB = bTileStride + shmemPad;

  let constCoopKShift: number | undefined;
  let constCoopKMask: number | undefined;
  if (coopKPow2) {
    constCoopKShift = b.id();
    b.constant(tU32, constCoopKShift, log2Pow2(coopK));
    constCoopKMask = b.id();
    b.constant(tU32, constCoopKMask, coopK - 1);
  }

  // B tile stride shift/mask for shared memory load coordinate decomposition
  let constBTileStrideShift: number | undefined;
  let constBTileStrideMask: number | undefined;
  if (bTileStridePow2) {
    constBTileStrideShift = b.id();
    b.constant(tU32, constBTileStrideShift, log2Pow2(bTileStride));
    constBTileStrideMask = b.id();
    b.constant(tU32, constBTileStrideMask, bTileStride - 1);
  }

  // K-tile multiplier: kTileK shift/mask for A tile coordinate decomposition
  const kTileKPow2 = isPowerOfTwo(kTileK);
  let constKTileKShift: number | undefined;
  let constKTileKMask: number | undefined;
  if (kMulti > 1 && kTileKPow2) {
    constKTileKShift = b.id();
    b.constant(tU32, constKTileKShift, log2Pow2(kTileK));
    constKTileKMask = b.id();
    b.constant(tU32, constKTileKMask, kTileK - 1);
  } else {
    constKTileKShift = constCoopKShift;
    constKTileKMask = constCoopKMask;
  }

  // K-loop step constant (= coopK * kMulti)
  let constKStep: number;
  if (kMulti > 1) {
    constKStep = b.id(); b.constant(tU32, constKStep, kTileK);
  } else {
    constKStep = constCoopK;
  }

  // Shared memory for loading tiles from global memory
  // Tile A: (coopM * regTilesM) rows × kTileK cols, per subgroup-Y section
  // Tile B: kTileK rows × (coopN * regTilesN) cols, per subgroup-X section
  // In multi-subgroup mode, A tiles are shared by subgroup rows (Y),
  // B tiles are shared by subgroup columns (X). This improves tile reuse.
  // Double buffering: 2× size for ping-pong shared memory.
  const dbMul = useDoubleBuf ? 2 : 1;
  const halfSizeA = coopM * regTilesM * paddedStrideA * subgroupTilesY;
  const halfSizeB = kTileK * paddedStrideB * subgroupTilesX;
  const constTileASize = b.id(); b.constant(tU32, constTileASize, halfSizeA * dbMul);
  const tArrayTileA = b.id(); b.typeArray(tArrayTileA, tF16, constTileASize);
  const tPtrSharedArrA = b.id(); b.typePointer(tPtrSharedArrA, StorageClass.Workgroup, tArrayTileA);
  const tileA = b.id(); b.variable(tPtrSharedArrA, tileA, StorageClass.Workgroup);

  const constTileBSize = b.id(); b.constant(tU32, constTileBSize, halfSizeB * dbMul);
  const tArrayTileB = b.id(); b.typeArray(tArrayTileB, tF16, constTileBSize);
  const tPtrSharedArrB = b.id(); b.typePointer(tPtrSharedArrB, StorageClass.Workgroup, tArrayTileB);
  const tileB = b.id(); b.variable(tPtrSharedArrB, tileB, StorageClass.Workgroup);

  const tPtrSharedF16 = b.id();
  b.typePointer(tPtrSharedF16, StorageClass.Workgroup, tF16);

  // Storage buffers:
  // - A/B: f32 source buffers or pre-cast f16 source buffers
  // - C:   f32 output buffer
  const bufA: { varId: number; tPtrF32?: number; tPtrF16?: number } = inputF16
    ? declareStorageBufferF16Ssbo(b, tF16, 0, 0, true)
    : declareStorageBufferF32Ssbo(b, tF32, 0, 0, true);
  const bufB: { varId: number; tPtrF32?: number; tPtrF16?: number } = inputF16
    ? declareStorageBufferF16Ssbo(b, tF16, 0, 1, true)
    : declareStorageBufferF32Ssbo(b, tF32, 0, 1, true);
  const bufC = declareStorageBufferF32Ssbo(b, tF32, 0, 2, false, true);

  // Push constants: { M, N, K, _pad }
  const pc = declareParamsPushConstant(b, tF32, 4);

  // Built-in variables
  const tPtrInputVec3 = b.id();
  b.typePointer(tPtrInputVec3, StorageClass.Input, tVec3U32);

  const vGlobalId = b.id();
  b.variable(tPtrInputVec3, vGlobalId, StorageClass.Input);
  b.addDecorate(vGlobalId, Decoration.BuiltIn, BuiltIn.GlobalInvocationId);

  const vWorkgroupId = b.id();
  b.variable(tPtrInputVec3, vWorkgroupId, StorageClass.Input);
  b.addDecorate(vWorkgroupId, Decoration.BuiltIn, BuiltIn.WorkgroupId);

  const vLocalId = b.id();
  b.variable(tPtrInputVec3, vLocalId, StorageClass.Input);
  b.addDecorate(vLocalId, Decoration.BuiltIn, BuiltIn.LocalInvocationId);

  // Barrier constants
  const semAcqRelWg = b.id();
  b.constant(tU32, semAcqRelWg, MemorySemantics.AcquireRelease | MemorySemantics.WorkgroupMemory);

  // Function-local pointer types
  const tPtrFnU32 = b.id();
  b.typePointer(tPtrFnU32, StorageClass.Function, tU32);

  // Entry point
  // - baseline: one subgroup (32 threads) computes one output tile
  // - s2x2 mode: 4 subgroups (128 threads) compute 2x2 tiles per workgroup
  const WG_SIZE = 32 * subgroupTilesX * subgroupTilesY;
  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main",
    [vGlobalId, vWorkgroupId, vLocalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, WG_SIZE, 1, 1);

  // ── Function body ──────────────────────────────────────────────────
  b.emit(Op.Function, [tVoid, fnMain, FunctionControl.None, tFnVoid]);
  const labelEntry = b.id();
  b.emit(Op.Label, [labelEntry]);

  if (coopDebugMode === "type_only") {
    b.emit(Op.Return, []);
    b.emit(Op.FunctionEnd, []);
    return b.build();
  }

  // Function-local variables
  const varK = b.id();
  b.emit(Op.Variable, [tPtrFnU32, varK, StorageClass.Function]);
  // Register tiling: regTilesM × regTilesN accumulator tiles per subgroup
  const varAccs: number[][] = [];
  for (let rm = 0; rm < regTilesM; rm++) {
    varAccs[rm] = [];
    for (let rn = 0; rn < regTilesN; rn++) {
      varAccs[rm][rn] = b.id();
      b.emit(Op.Variable, [tPtrFnCoopC, varAccs[rm][rn], StorageClass.Function]);
    }
  }

  // Load workgroup ID
  const wgIdVec = b.id();
  b.emit(Op.Load, [tVec3U32, wgIdVec, vWorkgroupId]);
  let wgTileCol = b.id();
  b.emit(Op.CompositeExtract, [tU32, wgTileCol, wgIdVec, 0]);
  let wgTileRow = b.id();
  b.emit(Op.CompositeExtract, [tU32, wgTileRow, wgIdVec, 1]);

  // Early-load M from push constant [0] (needed by GROUP_M swizzle before the main load block)
  const ptrM_early = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrM_early, pc.varId, const0u]);
  const MF_early = b.id();
  b.emit(Op.Load, [tF32, MF_early, ptrM_early]);
  const M_early = b.id();
  b.emit(Op.ConvertFToU, [tU32, M_early, MF_early]);

  // GROUP_M swizzle (Triton-style): remap WG indices so consecutive WGs cycle
  // through GROUP_M rows before advancing to the next column. This ensures
  // GROUP_M WGs share the same B-column data (L2 cache hit), and WGs across
  // all columns share the same GROUP_M A-rows (L2 reuse across columns).
  // gridX is passed as push constant [3]. 0 = swizzle disabled.
  if (useSwizzle) {
    const GROUP_M = swizzleSize;
    const constGroupM = b.id(); b.constant(tU32, constGroupM, GROUP_M);

    // Load gridX from push constant [3] (0 = swizzle disabled)
    const ptrGridX = b.id();
    b.emit(Op.AccessChain, [pc.tPtrF32, ptrGridX, pc.varId, const3u]);
    const gridXF = b.id();
    b.emit(Op.Load, [tF32, gridXF, ptrGridX]);
    const gridX = b.id();
    b.emit(Op.ConvertFToU, [tU32, gridX, gridXF]);

    // Runtime guard: only swizzle when gridX > 0
    const swizzleOk = b.id();
    b.emit(Op.ULessThan, [tBool, swizzleOk, const0u, gridX]);
    const labelSwizzle = b.id();
    const labelNoSwizzle = b.id();
    const labelSwizzleMerge = b.id();
    b.emit(Op.SelectionMerge, [labelSwizzleMerge, 0]);
    b.emit(Op.BranchConditional, [swizzleOk, labelSwizzle, labelNoSwizzle]);

    // Swizzled path — Triton GROUP_M pattern
    b.emit(Op.Label, [labelSwizzle]);

    // linear = wgTileRow * gridX + wgTileCol
    const rowTimesGX = b.id();
    b.emit(Op.IMul, [tU32, rowTimesGX, wgTileRow, gridX]);
    const linear = b.id();
    b.emit(Op.IAdd, [tU32, linear, wgTileCol, rowTimesGX]);

    // Compute gridY = ceil(M / tileM) where tileM is compile-time constant
    const wgTileM = coopM * regTilesM * subgroupTilesY; // output rows per WG
    const constTileM = b.id(); b.constant(tU32, constTileM, wgTileM);
    const constTileMM1 = b.id(); b.constant(tU32, constTileMM1, wgTileM - 1);
    const mPlusTM1 = b.id(); b.emit(Op.IAdd, [tU32, mPlusTM1, M_early, constTileMM1]);
    const gridY = b.id(); b.emit(Op.UDiv, [tU32, gridY, mPlusTM1, constTileM]);

    // groupSizeTotal = GROUP_M * gridX
    const groupSizeTotal = b.id();
    b.emit(Op.IMul, [tU32, groupSizeTotal, constGroupM, gridX]);

    // groupId = linear / groupSizeTotal
    const groupId = b.id();
    b.emit(Op.UDiv, [tU32, groupId, linear, groupSizeTotal]);

    // firstPidM = groupId * GROUP_M
    const firstPidM = b.id();
    b.emit(Op.IMul, [tU32, firstPidM, groupId, constGroupM]);

    // remainingM = gridY - firstPidM
    const remainingM = b.id();
    b.emit(Op.ISub, [tU32, remainingM, gridY, firstPidM]);

    // groupSizeM = min(remainingM, GROUP_M)
    const cmpLt = b.id();
    b.emit(Op.ULessThan, [tBool, cmpLt, remainingM, constGroupM]);
    const groupSizeM = b.id();
    b.emit(Op.Select, [tU32, groupSizeM, cmpLt, remainingM, constGroupM]);

    // groupSizeMTimesGX = groupSizeM * gridX
    const groupSizeMTimesGX = b.id();
    b.emit(Op.IMul, [tU32, groupSizeMTimesGX, groupSizeM, gridX]);

    // within = linear % groupSizeMTimesGX
    const within = b.id();
    b.emit(Op.UMod, [tU32, within, linear, groupSizeMTimesGX]);

    // swTileRow = firstPidM + (within % groupSizeM)
    const withinModGSM = b.id();
    b.emit(Op.UMod, [tU32, withinModGSM, within, groupSizeM]);
    const swTileRow = b.id();
    b.emit(Op.IAdd, [tU32, swTileRow, firstPidM, withinModGSM]);

    // swTileCol = within / groupSizeM
    const swTileCol = b.id();
    b.emit(Op.UDiv, [tU32, swTileCol, within, groupSizeM]);

    b.emit(Op.Branch, [labelSwizzleMerge]);

    // Non-swizzled path (passthrough)
    b.emit(Op.Label, [labelNoSwizzle]);
    b.emit(Op.Branch, [labelSwizzleMerge]);

    // Merge: Phi selects between swizzled and original coords
    b.emit(Op.Label, [labelSwizzleMerge]);
    const mergedCol = b.id();
    b.emit(Op.Phi, [tU32, mergedCol, swTileCol, labelSwizzle, wgTileCol, labelNoSwizzle]);
    const mergedRow = b.id();
    b.emit(Op.Phi, [tU32, mergedRow, swTileRow, labelSwizzle, wgTileRow, labelNoSwizzle]);
    wgTileCol = mergedCol;
    wgTileRow = mergedRow;
  }

  // Load local ID (thread index within workgroup)
  const lidVec = b.id();
  b.emit(Op.Load, [tVec3U32, lidVec, vLocalId]);
  const tid = b.id();
  b.emit(Op.CompositeExtract, [tU32, tid, lidVec, 0]);

  const subgroupMode = subgroupTilesX > 1 || subgroupTilesY > 1;
  let subgroupIdVal: number | undefined;
  let laneIdVal: number | undefined;
  let subgroupXVal: number | undefined;
  let subgroupYVal: number | undefined;

  let tileCol: number;
  let tileRow: number;
  if (subgroupMode) {
    const constSubgroupSize = b.id();
    b.constant(tU32, constSubgroupSize, 32);
    const constSubgroupTilesX = b.id();
    b.constant(tU32, constSubgroupTilesX, subgroupTilesX);
    const constSubgroupTilesY = b.id();
    b.constant(tU32, constSubgroupTilesY, subgroupTilesY);

    const subgroupId = b.id();
    b.emit(Op.UDiv, [tU32, subgroupId, tid, constSubgroupSize]);
    subgroupIdVal = subgroupId;
    const laneId = b.id();
    b.emit(Op.UMod, [tU32, laneId, tid, constSubgroupSize]);
    laneIdVal = laneId;
    const subgroupX = b.id();
    b.emit(Op.UMod, [tU32, subgroupX, subgroupId, constSubgroupTilesX]);
    subgroupXVal = subgroupX;
    const subgroupY = b.id();
    b.emit(Op.UDiv, [tU32, subgroupY, subgroupId, constSubgroupTilesX]);
    subgroupYVal = subgroupY;

    const wgBaseCol = b.id();
    b.emit(Op.IMul, [tU32, wgBaseCol, wgTileCol, constSubgroupTilesX]);
    tileCol = b.id();
    b.emit(Op.IAdd, [tU32, tileCol, wgBaseCol, subgroupX]);

    const wgBaseRow = b.id();
    b.emit(Op.IMul, [tU32, wgBaseRow, wgTileRow, constSubgroupTilesY]);
    tileRow = b.id();
    b.emit(Op.IAdd, [tU32, tileRow, wgBaseRow, subgroupY]);
  } else {
    tileCol = wgTileCol;
    tileRow = wgTileRow;
  }

  // Batch index (z dimension of workgroup ID)
  let batchIdx: number | undefined;
  if (batched) {
    batchIdx = b.id();
    b.emit(Op.CompositeExtract, [tU32, batchIdx, wgIdVec, 2]);
  }

  // Load M, N, K from push constants
  const ptrM = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrM, pc.varId, const0u]);
  const MF = b.id();
  b.emit(Op.Load, [tF32, MF, ptrM]);
  const M = b.id();
  b.emit(Op.ConvertFToU, [tU32, M, MF]);

  const ptrN = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrN, pc.varId, const1u]);
  const NF = b.id();
  b.emit(Op.Load, [tF32, NF, ptrN]);
  const N = b.id();
  b.emit(Op.ConvertFToU, [tU32, N, NF]);

  const ptrK = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrK, pc.varId, const2u]);
  const KF = b.id();
  b.emit(Op.Load, [tF32, KF, ptrK]);
  const K = b.id();
  b.emit(Op.ConvertFToU, [tU32, K, KF]);

  // Compute global row/col offsets for this tile
  // With register tiling, each subgroup covers coopM*regTilesM rows and coopN*regTilesN cols
  let globalRowBase: number;
  if (regTilesM > 1) {
    const constEffM = b.id(); b.constant(tU32, constEffM, coopM * regTilesM);
    globalRowBase = b.id();
    b.emit(Op.IMul, [tU32, globalRowBase, tileRow, constEffM]);
  } else {
    globalRowBase = b.id();
    b.emit(Op.IMul, [tU32, globalRowBase, tileRow, constCoopM]);
  }
  let globalColBase: number;
  if (regTilesN > 1) {
    const constEffN = b.id(); b.constant(tU32, constEffN, coopN * regTilesN);
    globalColBase = b.id();
    b.emit(Op.IMul, [tU32, globalColBase, tileCol, constEffN]);
  } else {
    globalColBase = b.id();
    b.emit(Op.IMul, [tU32, globalColBase, tileCol, constCoopN]);
  }

  // Batch offset for A and B
  const MK = b.id(); // M * K
  b.emit(Op.IMul, [tU32, MK, M, K]);
  const MN = b.id(); // M * N  (or N*K for B when not transposed)
  b.emit(Op.IMul, [tU32, MN, M, N]);

  let NK: number | undefined;
  if (!transposedB) {
    NK = b.id(); // K * N for B stride
    b.emit(Op.IMul, [tU32, NK, K, N]);
  }

  let batchOffsetA: number | undefined;
  let batchOffsetB: number | undefined;
  if (batched) {
    batchOffsetA = b.id();
    b.emit(Op.IMul, [tU32, batchOffsetA, batchIdx!, MK]);
    batchOffsetB = b.id();
    if (transposedB) {
      // B is [N, K], batch stride is N*K
      const NKb = b.id();
      b.emit(Op.IMul, [tU32, NKb, N, K]);
      b.emit(Op.IMul, [tU32, batchOffsetB, batchIdx!, NKb]);
    } else {
      b.emit(Op.IMul, [tU32, batchOffsetB, batchIdx!, NK!]);
    }
  }

  // ── Tile loading constants (for address hoisting fast path) ─────────
  const totalA = coopM * regTilesM * coopK;
  const totalB = coopK * coopN * regTilesN;
  const loadWidthA = subgroupMode ? (32 * subgroupTilesX) : WG_SIZE;
  const loadWidthB = subgroupMode ? (32 * subgroupTilesY) : WG_SIZE;
  const elemsPerThreadA = Math.ceil(totalA / loadWidthA);
  const elemsPerThreadB = Math.ceil(totalB / loadWidthB);
  const canUseStridedCoordsA = (loadWidthA % coopK) === 0;
  const canUseStridedCoordsB = (loadWidthB % bTileStride) === 0;
  const rowStrideA = canUseStridedCoordsA ? (loadWidthA / coopK) : 0;
  const rowStrideB = canUseStridedCoordsB ? (loadWidthB / bTileStride) : 0;
  const useFastPath = false; // disabled — address hoisting regresses perf

  // Initialize accumulator to zero
  // We need to create a constant zero cooperative matrix — not directly supported.
  // Instead, we'll use OpConstant for f32 zero and composite-construct won't work for coop matrices.
  // The correct approach: use a variable and store zeros via a loop, or
  // better — we initialize by doing the first MulAdd with a zero acc.
  // Actually, OpConstantNull works for cooperative matrix types.
  const constNullC = b.id();
  b.constantNull(tCoopCAcc, constNullC);

  for (let rm = 0; rm < regTilesM; rm++)
    for (let rn = 0; rn < regTilesN; rn++)
      b.emit(Op.Store, [varAccs[rm][rn], constNullC]);

  // Double buffer phase variable: alternates 0/1 to select shared memory half
  let varDbPhase: number | undefined;
  if (useDoubleBuf) {
    varDbPhase = b.id();
    b.emit(Op.Variable, [tPtrFnU32, varDbPhase, StorageClass.Function]);
    b.emit(Op.Store, [varDbPhase, const0u]);
  }

  // ── Address hoisting: precompute K-invariant address bases ───────────
  // Only aBase and bBase survive across the K-loop (2 extra registers).
  // All other values (loadThreadBase, subgroupBase, strides) are
  // recomputed inside the loop body to avoid register pressure.
  let fastABase: number | undefined;
  let fastBBase: number | undefined;

  if (useFastPath) {
    // Temporary: compute loadThreadBase + localRow/Col (dies after aBase/bBase)
    let tmpLTBA = tid;
    let tmpLTBB = tid;
    if (subgroupMode) {
      const cSgSz = b.id(); b.constant(tU32, cSgSz, 32);
      const sgXB = b.id();
      b.emit(Op.IMul, [tU32, sgXB, subgroupXVal!, cSgSz]);
      tmpLTBA = b.id();
      b.emit(Op.IAdd, [tU32, tmpLTBA, sgXB, laneIdVal!]);
      const sgYB = b.id();
      b.emit(Op.IMul, [tU32, sgYB, subgroupYVal!, cSgSz]);
      tmpLTBB = b.id();
      b.emit(Op.IAdd, [tU32, tmpLTBB, sgYB, laneIdVal!]);
    }

    // A: localRow/Col from loadThreadBase
    const fpRowA = b.id();
    const fpColA = b.id();
    if (constCoopKShift !== undefined && constCoopKMask !== undefined) {
      b.emit(Op.ShiftRightLogical, [tU32, fpRowA, tmpLTBA, constCoopKShift]);
      b.emit(Op.BitwiseAnd, [tU32, fpColA, tmpLTBA, constCoopKMask]);
    } else {
      b.emit(Op.UDiv, [tU32, fpRowA, tmpLTBA, constCoopK]);
      b.emit(Op.UMod, [tU32, fpColA, tmpLTBA, constCoopK]);
    }

    // aBase: K-invariant global address for element 0
    const gRowA = b.id();
    b.emit(Op.IAdd, [tU32, gRowA, globalRowBase, fpRowA]);
    if (transposedA) {
      const colM = b.id();
      b.emit(Op.IMul, [tU32, colM, fpColA, M]);
      fastABase = b.id();
      b.emit(Op.IAdd, [tU32, fastABase, colM, gRowA]);
    } else {
      const rowK = b.id();
      b.emit(Op.IMul, [tU32, rowK, gRowA, K]);
      fastABase = b.id();
      b.emit(Op.IAdd, [tU32, fastABase, rowK, fpColA]);
    }
    if (batched) {
      const ab = b.id();
      b.emit(Op.IAdd, [tU32, ab, fastABase, batchOffsetA!]);
      fastABase = ab;
    }

    // B: localRow/Col from loadThreadBase
    const fpRowB = b.id();
    const fpColB = b.id();
    if (constBTileStrideShift !== undefined && constBTileStrideMask !== undefined) {
      b.emit(Op.ShiftRightLogical, [tU32, fpRowB, tmpLTBB, constBTileStrideShift]);
      b.emit(Op.BitwiseAnd, [tU32, fpColB, tmpLTBB, constBTileStrideMask]);
    } else {
      const cBSt = b.id(); b.constant(tU32, cBSt, bTileStride);
      b.emit(Op.UDiv, [tU32, fpRowB, tmpLTBB, cBSt]);
      b.emit(Op.UMod, [tU32, fpColB, tmpLTBB, cBSt]);
    }

    // bBase: K-invariant global address for element 0
    const nCol = b.id();
    b.emit(Op.IAdd, [tU32, nCol, globalColBase, fpColB]);
    if (transposedB) {
      const nK = b.id();
      b.emit(Op.IMul, [tU32, nK, nCol, K]);
      fastBBase = b.id();
      b.emit(Op.IAdd, [tU32, fastBBase, nK, fpRowB]);
    } else {
      const rN = b.id();
      b.emit(Op.IMul, [tU32, rN, fpRowB, N]);
      fastBBase = b.id();
      b.emit(Op.IAdd, [tU32, fastBBase, rN, nCol]);
    }
    if (batched) {
      const bb = b.id();
      b.emit(Op.IAdd, [tU32, bb, fastBBase, batchOffsetB!]);
      fastBBase = bb;
    }
  }

  // ── K-tile loop: for k = 0; k < K; k += kTileK ─────────────────────
  // Pre-allocate forward reference for Phi
  const kNext = b.id();

  const labelLoopHeader = b.id();
  const labelLoopBody = b.id();
  const labelLoopContinue = b.id();
  const labelLoopEnd = b.id();

  // Capture pre-loop block label for Phi incoming edge
  const labelPreLoop = b.id();
  b.emit(Op.Branch, [labelPreLoop]);
  b.emit(Op.Label, [labelPreLoop]);
  b.emit(Op.Branch, [labelLoopHeader]);

  b.emit(Op.Label, [labelLoopHeader]);

  // SSA Phi: kVal = 0 on first entry, kNext on loop back-edge
  const kVal = b.id();
  b.emit(Op.Phi, [tU32, kVal, const0u, labelPreLoop, kNext, labelLoopContinue]);
  const kLtK = b.id();
  b.emit(Op.ULessThan, [tBool, kLtK, kVal, K]);

  b.emit(Op.LoopMerge, [labelLoopEnd, labelLoopContinue, 1]); // Unroll hint
  b.emit(Op.BranchConditional, [kLtK, labelLoopBody, labelLoopEnd]);

  b.emit(Op.Label, [labelLoopBody]);

  let coopAMat: number;
  let coopBMat: number;
  let coopAMats: number[] = [];
  let coopBMats: number[] = [];
  // Hoisted for double-buffer epilogue access
  let subgroupBaseA_h = const0u;
  let subgroupBaseB_h = const0u;
  if (useDirectF16Load) {
    // Direct cooperative loads from global f16 buffers — NO shared memory, NO barriers.
    // Each subgroup independently loads its own A/B tiles from global memory.
    // L2 cache handles cross-subgroup data sharing.
    // With kMulti>1, unroll kMulti load+MMA sub-steps per K-loop iteration.

    for (let km = 0; km < kMulti; km++) {
      // kOff = kVal + km * coopK (K offset for this sub-step)
      let kOff = kVal;
      if (km > 0) {
        const constKmOff = b.id(); b.constant(tU32, constKmOff, km * coopK);
        kOff = b.id();
        b.emit(Op.IAdd, [tU32, kOff, kVal, constKmOff]);
      }

      // Load regTilesM A matrices
      const directAMats: number[] = [];
      for (let rm = 0; rm < regTilesM; rm++) {
        let aBase = b.id();
        if (transposedA) {
          const kTimesM = b.id();
          b.emit(Op.IMul, [tU32, kTimesM, kOff, M]);
          if (rm === 0) {
            b.emit(Op.IAdd, [tU32, aBase, kTimesM, globalRowBase]);
          } else {
            const rowOff = b.id();
            const cRM = b.id(); b.constant(tU32, cRM, rm * coopM);
            b.emit(Op.IAdd, [tU32, rowOff, globalRowBase, cRM]);
            b.emit(Op.IAdd, [tU32, aBase, kTimesM, rowOff]);
          }
        } else {
          const rowOff = rm === 0 ? globalRowBase : (() => {
            const cRM = b.id(); b.constant(tU32, cRM, rm * coopM);
            const r = b.id();
            b.emit(Op.IAdd, [tU32, r, globalRowBase, cRM]);
            return r;
          })();
          const rowTimesK = b.id();
          b.emit(Op.IMul, [tU32, rowTimesK, rowOff, K]);
          b.emit(Op.IAdd, [tU32, aBase, rowTimesK, kOff]);
        }
        if (batched) {
          const aWithBatch = b.id();
          b.emit(Op.IAdd, [tU32, aWithBatch, aBase, batchOffsetA!]);
          aBase = aWithBatch;
        }
        const ptrA = b.id();
        b.emit(Op.AccessChain, [bufA.tPtrF16!, ptrA, bufA.varId, const0u, aBase]);
        directAMats[rm] = b.id();
        b.emit(Op.OpCooperativeMatrixLoadKHR, [
          tCoopA, directAMats[rm], ptrA,
          transposedA ? constColumnMajor : constRowMajor,
          transposedA ? M : K,
        ]);
      }

      // Load regTilesN B matrices
      const directBMats: number[] = [];
      for (let rn = 0; rn < regTilesN; rn++) {
        let bBase = b.id();
        if (transposedB) {
          const colOff = rn === 0 ? globalColBase : (() => {
            const cRN = b.id(); b.constant(tU32, cRN, rn * coopN);
            const r = b.id();
            b.emit(Op.IAdd, [tU32, r, globalColBase, cRN]);
            return r;
          })();
          const nTimesK = b.id();
          b.emit(Op.IMul, [tU32, nTimesK, colOff, K]);
          b.emit(Op.IAdd, [tU32, bBase, nTimesK, kOff]);
        } else {
          const kTimesN = b.id();
          b.emit(Op.IMul, [tU32, kTimesN, kOff, N]);
          if (rn === 0) {
            b.emit(Op.IAdd, [tU32, bBase, kTimesN, globalColBase]);
          } else {
            const colOff = b.id();
            const cRN = b.id(); b.constant(tU32, cRN, rn * coopN);
            b.emit(Op.IAdd, [tU32, colOff, globalColBase, cRN]);
            b.emit(Op.IAdd, [tU32, bBase, kTimesN, colOff]);
          }
        }
        if (batched) {
          const bWithBatch = b.id();
          b.emit(Op.IAdd, [tU32, bWithBatch, bBase, batchOffsetB!]);
          bBase = bWithBatch;
        }
        const ptrB = b.id();
        b.emit(Op.AccessChain, [bufB.tPtrF16!, ptrB, bufB.varId, const0u, bBase]);
        directBMats[rn] = b.id();
        b.emit(Op.OpCooperativeMatrixLoadKHR, [
          tCoopB, directBMats[rn], ptrB,
          transposedB ? constColumnMajor : constRowMajor,
          transposedB ? K : N,
        ]);
      }

      // MMA: C[rm][rn] += A[rm] * B[rn] — no barriers needed!
      for (let rm = 0; rm < regTilesM; rm++) {
        for (let rn = 0; rn < regTilesN; rn++) {
          const prevAcc = b.id();
          b.emit(Op.Load, [tCoopCAcc, prevAcc, varAccs[rm][rn]]);
          const newAcc = b.id();
          b.emit(Op.OpCooperativeMatrixMulAddKHR, [tCoopCAcc, newAcc, directAMats[rm], directBMats[rn], prevAcc]);
          b.emit(Op.Store, [varAccs[rm][rn], newAcc]);
        }
      }
    }

    coopAMat = 0 as any;
    coopBMat = 0 as any;
  } else if (useFastPath) {
    // ── Fast path: address hoisting (only aBase/bBase precomputed) ────
    // Recompute loadThreadBase, subgroupBase, strides inside loop body
    // to keep register pressure low (only 2 extra live regs: aBase, bBase).

    // loadThreadBase (recomputed each K-step — same values, but no cross-loop pressure)
    let fpLTBA = tid;
    let fpLTBB = tid;
    if (subgroupMode) {
      const cSgSz = b.id(); b.constant(tU32, cSgSz, 32);
      const sgXB = b.id();
      b.emit(Op.IMul, [tU32, sgXB, subgroupXVal!, cSgSz]);
      fpLTBA = b.id();
      b.emit(Op.IAdd, [tU32, fpLTBA, sgXB, laneIdVal!]);
      const sgYB = b.id();
      b.emit(Op.IMul, [tU32, sgYB, subgroupYVal!, cSgSz]);
      fpLTBB = b.id();
      b.emit(Op.IAdd, [tU32, fpLTBB, sgYB, laneIdVal!]);
    }

    // subgroupBase (recomputed)
    let fpSgBaseA = const0u;
    let fpSgBaseB = const0u;
    if (subgroupMode) {
      const cTotA = b.id(); b.constant(tU32, cTotA, totalA);
      fpSgBaseA = b.id();
      b.emit(Op.IMul, [tU32, fpSgBaseA, subgroupYVal!, cTotA]);
      const cTotB = b.id(); b.constant(tU32, cTotB, totalB);
      fpSgBaseB = b.id();
      b.emit(Op.IMul, [tU32, fpSgBaseB, subgroupXVal!, cTotB]);
    }

    // sharedBase: first shared offset
    let fpShrBaseA: number;
    let fpShrBaseB: number;
    if (subgroupMode) {
      fpShrBaseA = b.id();
      b.emit(Op.IAdd, [tU32, fpShrBaseA, fpSgBaseA, fpLTBA]);
      fpShrBaseB = b.id();
      b.emit(Op.IAdd, [tU32, fpShrBaseB, fpSgBaseB, fpLTBB]);
    } else {
      fpShrBaseA = fpLTBA;
      fpShrBaseB = fpLTBB;
    }

    // A stride: rowStrideA * K (non-transposed) or rowStrideA (transposed)
    let fpAStride: number;
    if (transposedA) {
      fpAStride = b.id(); b.constant(tU32, fpAStride, rowStrideA);
    } else {
      const cRSA = b.id(); b.constant(tU32, cRSA, rowStrideA);
      fpAStride = b.id();
      b.emit(Op.IMul, [tU32, fpAStride, cRSA, K]);
    }

    // B stride: rowStrideB * N (non-transposed) or rowStrideB (transposed)
    let fpBStride: number;
    if (transposedB) {
      fpBStride = b.id(); b.constant(tU32, fpBStride, rowStrideB);
    } else {
      const cRSB = b.id(); b.constant(tU32, cRSB, rowStrideB);
      fpBStride = b.id();
      b.emit(Op.IMul, [tU32, fpBStride, cRSB, N]);
    }

    // Per-K-step multiply (shared across all elements):
    let kTermA: number;
    if (transposedA) {
      kTermA = b.id();
      b.emit(Op.IMul, [tU32, kTermA, kVal, M]);
    } else {
      kTermA = kVal;
    }
    let kTermB: number;
    if (!transposedB) {
      kTermB = b.id();
      b.emit(Op.IMul, [tU32, kTermB, kVal, N]);
    } else {
      kTermB = kVal;
    }

    // ── A loads: addr = kTermA + aBase, then += aStride per element ───
    let addrA = b.id();
    b.emit(Op.IAdd, [tU32, addrA, kTermA, fastABase!]);
    let shrA = fpShrBaseA;
    const cLdWA = b.id(); b.constant(tU32, cLdWA, loadWidthA);

    for (let e = 0; e < elemsPerThreadA; e++) {
      if (e * loadWidthA + loadWidthA > totalA) {
        const eOff = b.id();
        if (e === 0) {
          b.emit(Op.CopyObject, [tU32, eOff, fpLTBA]);
        } else {
          const cE = b.id(); b.constant(tU32, cE, e * loadWidthA);
          b.emit(Op.IAdd, [tU32, eOff, fpLTBA, cE]);
        }
        const cTot = b.id(); b.constant(tU32, cTot, totalA);
        const inB = b.id();
        b.emit(Op.ULessThan, [tBool, inB, eOff, cTot]);
        const lblL = b.id(); const lblS = b.id();
        b.emit(Op.SelectionMerge, [lblS, 0]);
        b.emit(Op.BranchConditional, [inB, lblL, lblS]);
        b.emit(Op.Label, [lblL]);
        emitFastLoad(b, tF32, tF16, tU32, tPtrSharedF16,
          bufA, inputF16, addrA, shrA, tileA, const0u);
        b.emit(Op.Branch, [lblS]);
        b.emit(Op.Label, [lblS]);
      } else {
        emitFastLoad(b, tF32, tF16, tU32, tPtrSharedF16,
          bufA, inputF16, addrA, shrA, tileA, const0u);
      }
      if (e < elemsPerThreadA - 1) {
        const nA = b.id();
        b.emit(Op.IAdd, [tU32, nA, addrA, fpAStride]);
        addrA = nA;
        const nS = b.id();
        b.emit(Op.IAdd, [tU32, nS, shrA, cLdWA]);
        shrA = nS;
      }
    }

    // ── B loads: addr = kTermB + bBase, then += bStride per element ───
    let addrB = b.id();
    b.emit(Op.IAdd, [tU32, addrB, kTermB, fastBBase!]);
    let shrB = fpShrBaseB;
    const cLdWB = b.id(); b.constant(tU32, cLdWB, loadWidthB);

    for (let e = 0; e < elemsPerThreadB; e++) {
      if (e * loadWidthB + loadWidthB > totalB) {
        const eOff = b.id();
        if (e === 0) {
          b.emit(Op.CopyObject, [tU32, eOff, fpLTBB]);
        } else {
          const cE = b.id(); b.constant(tU32, cE, e * loadWidthB);
          b.emit(Op.IAdd, [tU32, eOff, fpLTBB, cE]);
        }
        const cTot = b.id(); b.constant(tU32, cTot, totalB);
        const inB = b.id();
        b.emit(Op.ULessThan, [tBool, inB, eOff, cTot]);
        const lblL = b.id(); const lblS = b.id();
        b.emit(Op.SelectionMerge, [lblS, 0]);
        b.emit(Op.BranchConditional, [inB, lblL, lblS]);
        b.emit(Op.Label, [lblL]);
        emitFastLoad(b, tF32, tF16, tU32, tPtrSharedF16,
          bufB, inputF16, addrB, shrB, tileB, const0u);
        b.emit(Op.Branch, [lblS]);
        b.emit(Op.Label, [lblS]);
      } else {
        emitFastLoad(b, tF32, tF16, tU32, tPtrSharedF16,
          bufB, inputF16, addrB, shrB, tileB, const0u);
      }
      if (e < elemsPerThreadB - 1) {
        const nB = b.id();
        b.emit(Op.IAdd, [tU32, nB, addrB, fpBStride]);
        addrB = nB;
        const nS = b.id();
        b.emit(Op.IAdd, [tU32, nS, shrB, cLdWB]);
        shrB = nS;
      }
    }

    // ── Barrier → MMA → Barrier ──────────────────────────────────────
    b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

    const tileABaseFP = subgroupMode ? fpSgBaseA : const0u;
    const tileBBaseFP = subgroupMode ? fpSgBaseB : const0u;
    emitCoopMatrixMMA(b, tF16, tU32, tCoopA, tCoopB, tCoopCAcc,
      tPtrSharedF16, tileA, tileB, tileABaseFP, tileBBaseFP,
      constRowMajor, coopM, coopN, coopK,
      regTilesM, regTilesN, bTileStride, varAccs, shmemPad, shmemPad);

    b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

    coopAMat = 0 as any;
    coopBMat = 0 as any;
  } else {
    // ── Load tile A from global to shared ───────────────────────────────
    // With register tiling: each subgroup section covers (coopM*regTilesM) × kTileK
    // With register tiling: each B section covers kTileK × (coopN*regTilesN)
    const totalA = coopM * regTilesM * kTileK;
    const totalB = kTileK * coopN * regTilesN;
    const loadWidthA = subgroupMode ? (32 * subgroupTilesX) : WG_SIZE;
    const loadWidthB = subgroupMode ? (32 * subgroupTilesY) : WG_SIZE;
    const elemsPerThreadA = Math.ceil(totalA / loadWidthA);
    const elemsPerThreadB = Math.ceil(totalB / loadWidthB);
    let loadThreadBaseA = tid;
    let loadThreadBaseB = tid;
    if (subgroupMode) {
      const constSubgroupSize = b.id();
      b.constant(tU32, constSubgroupSize, 32);

      const subgroupXBase = b.id();
      b.emit(Op.IMul, [tU32, subgroupXBase, subgroupXVal!, constSubgroupSize]);
      loadThreadBaseA = b.id();
      b.emit(Op.IAdd, [tU32, loadThreadBaseA, subgroupXBase, laneIdVal!]);

      const subgroupYBase = b.id();
      b.emit(Op.IMul, [tU32, subgroupYBase, subgroupYVal!, constSubgroupSize]);
      loadThreadBaseB = b.id();
      b.emit(Op.IAdd, [tU32, loadThreadBaseB, subgroupYBase, laneIdVal!]);
    }
    // A tile stride is kTileK = coopK * kMulti (wider rows when kMulti > 1)
    const canUseStridedCoordsA = (loadWidthA % kTileK) === 0;
    // B tile stride is bTileStride = coopN * regTilesN (contiguous columns)
    const canUseStridedCoordsB = (loadWidthB % bTileStride) === 0;
    const rowStrideA = canUseStridedCoordsA ? (loadWidthA / kTileK) : 0;
    const rowStrideB = canUseStridedCoordsB ? (loadWidthB / bTileStride) : 0;

    let baseLocalRowA: number | undefined;
    let baseLocalColA: number | undefined;
    if (canUseStridedCoordsA) {
      baseLocalRowA = b.id();
      baseLocalColA = b.id();
      if (constKTileKShift !== undefined && constKTileKMask !== undefined) {
        b.emit(Op.ShiftRightLogical, [tU32, baseLocalRowA, loadThreadBaseA, constKTileKShift]);
        b.emit(Op.BitwiseAnd, [tU32, baseLocalColA, loadThreadBaseA, constKTileKMask]);
      } else {
        const ck = b.id(); b.constant(tU32, ck, kTileK);
        b.emit(Op.UDiv, [tU32, baseLocalRowA, loadThreadBaseA, ck]);
        b.emit(Op.UMod, [tU32, baseLocalColA, loadThreadBaseA, ck]);
      }
    }

    let baseLocalRowB: number | undefined;
    let baseLocalColB: number | undefined;
    if (canUseStridedCoordsB) {
      baseLocalRowB = b.id();
      baseLocalColB = b.id();
      if (constBTileStrideShift !== undefined && constBTileStrideMask !== undefined) {
        b.emit(Op.ShiftRightLogical, [tU32, baseLocalRowB, loadThreadBaseB, constBTileStrideShift]);
        b.emit(Op.BitwiseAnd, [tU32, baseLocalColB, loadThreadBaseB, constBTileStrideMask]);
      } else {
        const constBStride = b.id(); b.constant(tU32, constBStride, bTileStride);
        b.emit(Op.UDiv, [tU32, baseLocalRowB, loadThreadBaseB, constBStride]);
        b.emit(Op.UMod, [tU32, baseLocalColB, loadThreadBaseB, constBStride]);
      }
    }

    let subgroupBaseA = const0u;
    let subgroupBaseB = const0u;
    // Use padded section sizes so subgroup offsets account for shared memory padding
    const paddedSectionA = coopM * regTilesM * paddedStrideA;
    const paddedSectionB = kTileK * paddedStrideB;
    if (subgroupMode) {
      const constTotalA = b.id();
      b.constant(tU32, constTotalA, paddedSectionA);
      subgroupBaseA = b.id();
      b.emit(Op.IMul, [tU32, subgroupBaseA, subgroupYVal!, constTotalA]);

      const constTotalB = b.id();
      b.constant(tU32, constTotalB, paddedSectionB);
      subgroupBaseB = b.id();
      b.emit(Op.IMul, [tU32, subgroupBaseB, subgroupXVal!, constTotalB]);
    }
    subgroupBaseA_h = subgroupBaseA;
    subgroupBaseB_h = subgroupBaseB;

    // Double buffer: compute write/read offsets from phase variable
    let writeOffA = const0u;
    let writeOffB = const0u;
    if (useDoubleBuf) {
      const phase = b.id();
      b.emit(Op.Load, [tU32, phase, varDbPhase!]);
      const constHalfA = b.id(); b.constant(tU32, constHalfA, halfSizeA);
      const constHalfB = b.id(); b.constant(tU32, constHalfB, halfSizeB);
      writeOffA = b.id(); b.emit(Op.IMul, [tU32, writeOffA, phase, constHalfA]);
      writeOffB = b.id(); b.emit(Op.IMul, [tU32, writeOffB, phase, constHalfB]);
    }

    // ── Load tile A: incremental stride addressing ────────────────────
    // Instead of recomputing row/col/global-address from scratch for each element,
    // precompute base addresses and add constant strides per element.
    // Saves ~6 instructions per element (50% reduction in load instructions).
    if (canUseStridedCoordsA) {
      // Global base address: A[globalRowBase + localRow0, kVal + localCol0]
      const gRow0A = b.id();
      b.emit(Op.IAdd, [tU32, gRow0A, globalRowBase, baseLocalRowA!]);
      let globalBaseA: number;
      if (transposedA) {
        // A[K,M]: index = (kVal + localCol) * M + globalRow
        const kCol0 = b.id();
        b.emit(Op.IAdd, [tU32, kCol0, kVal, baseLocalColA!]);
        const kCol0M = b.id();
        b.emit(Op.IMul, [tU32, kCol0M, kCol0, M]);
        globalBaseA = b.id();
        b.emit(Op.IAdd, [tU32, globalBaseA, kCol0M, gRow0A]);
      } else {
        // A[M,K]: index = globalRow * K + kVal + localCol
        const gRow0K = b.id();
        b.emit(Op.IMul, [tU32, gRow0K, gRow0A, K]);
        const gRow0Kk = b.id();
        b.emit(Op.IAdd, [tU32, gRow0Kk, gRow0K, kVal]);
        globalBaseA = b.id();
        b.emit(Op.IAdd, [tU32, globalBaseA, gRow0Kk, baseLocalColA!]);
      }
      if (batched) {
        const gba = b.id();
        b.emit(Op.IAdd, [tU32, gba, globalBaseA, batchOffsetA!]);
        globalBaseA = gba;
      }

      // Global stride: rowStrideA rows apart in the major dimension
      let globalStrideA: number;
      if (transposedA) {
        // transposed: increment in M dimension = rowStrideA
        if (rowStrideA === 1) {
          globalStrideA = const1u;
        } else {
          globalStrideA = b.id(); b.constant(tU32, globalStrideA, rowStrideA);
        }
      } else {
        // non-transposed: increment in row dimension = rowStrideA * K
        if (rowStrideA === 1) {
          globalStrideA = K;
        } else {
          const cRSA = b.id(); b.constant(tU32, cRSA, rowStrideA);
          globalStrideA = b.id();
          b.emit(Op.IMul, [tU32, globalStrideA, cRSA, K]);
        }
      }

      // Shared base: subgroupBaseA + localRow0 * paddedStrideA + localCol0
      const constPaddedStrA = b.id(); b.constant(tU32, constPaddedStrA, paddedStrideA);
      const shrRowOff0 = b.id();
      b.emit(Op.IMul, [tU32, shrRowOff0, baseLocalRowA!, constPaddedStrA]);
      let sharedBaseA = b.id();
      b.emit(Op.IAdd, [tU32, sharedBaseA, shrRowOff0, baseLocalColA!]);
      if (subgroupMode) {
        const sba = b.id();
        b.emit(Op.IAdd, [tU32, sba, subgroupBaseA, sharedBaseA]);
        sharedBaseA = sba;
      }
      if (useDoubleBuf) {
        const dba = b.id();
        b.emit(Op.IAdd, [tU32, dba, sharedBaseA, writeOffA]);
        sharedBaseA = dba;
      }

      // Shared stride: rowStrideA * paddedStrideA (compile-time constant)
      const sharedStrideA = rowStrideA * paddedStrideA;
      const constSharedStrA = b.id(); b.constant(tU32, constSharedStrA, sharedStrideA);

      // Element loop: base + stride increments
      let curGlobalA = globalBaseA;
      let curSharedA = sharedBaseA;
      for (let e = 0; e < elemsPerThreadA; e++) {
        if (e > 0) {
          const nextG = b.id();
          b.emit(Op.IAdd, [tU32, nextG, curGlobalA, globalStrideA]);
          curGlobalA = nextG;
          const nextS = b.id();
          b.emit(Op.IAdd, [tU32, nextS, curSharedA, constSharedStrA]);
          curSharedA = nextS;
        }

        if (e * loadWidthA + loadWidthA > totalA) {
          // Bounds check for partial last elements
          const elemOff = b.id();
          if (e === 0) {
            b.emit(Op.CopyObject, [tU32, elemOff, loadThreadBaseA]);
          } else {
            const cE = b.id(); b.constant(tU32, cE, e * loadWidthA);
            b.emit(Op.IAdd, [tU32, elemOff, loadThreadBaseA, cE]);
          }
          const cTot = b.id(); b.constant(tU32, cTot, totalA);
          const inB = b.id();
          b.emit(Op.ULessThan, [tBool, inB, elemOff, cTot]);
          const lblL = b.id(); const lblS = b.id();
          b.emit(Op.SelectionMerge, [lblS, 0]);
          b.emit(Op.BranchConditional, [inB, lblL, lblS]);
          b.emit(Op.Label, [lblL]);
          emitFastLoad(b, tF32, tF16, tU32, tPtrSharedF16,
            bufA, inputF16, curGlobalA, curSharedA, tileA, const0u);
          b.emit(Op.Branch, [lblS]);
          b.emit(Op.Label, [lblS]);
        } else {
          emitFastLoad(b, tF32, tF16, tU32, tPtrSharedF16,
            bufA, inputF16, curGlobalA, curSharedA, tileA, const0u);
        }
      }
    } else {
      // Fallback: per-element decomposition for non-strided configs
      let constShmemPadA: number | undefined;
      if (shmemPad > 0) {
        constShmemPadA = b.id(); b.constant(tU32, constShmemPadA, shmemPad);
      }
      for (let e = 0; e < elemsPerThreadA; e++) {
        const elemOffset = b.id();
        if (e === 0) {
          b.emit(Op.CopyObject, [tU32, elemOffset, loadThreadBaseA]);
        } else {
          const constE = b.id(); b.constant(tU32, constE, e * loadWidthA);
          b.emit(Op.IAdd, [tU32, elemOffset, loadThreadBaseA, constE]);
        }
        let effectiveOffset = elemOffset;
        if (shmemPad > 0) {
          const localRow = b.id();
          if (constKTileKShift !== undefined) {
            b.emit(Op.ShiftRightLogical, [tU32, localRow, elemOffset, constKTileKShift]);
          } else {
            const ck = b.id(); b.constant(tU32, ck, kTileK);
            b.emit(Op.UDiv, [tU32, localRow, elemOffset, ck]);
          }
          const padExtra = b.id();
          b.emit(Op.IMul, [tU32, padExtra, localRow, constShmemPadA!]);
          effectiveOffset = b.id();
          b.emit(Op.IAdd, [tU32, effectiveOffset, elemOffset, padExtra]);
        }
        let sharedOffset: number;
        if (subgroupMode) {
          sharedOffset = b.id();
          b.emit(Op.IAdd, [tU32, sharedOffset, subgroupBaseA, effectiveOffset]);
        } else {
          sharedOffset = effectiveOffset;
        }
        if (useDoubleBuf) {
          const dbOff = b.id();
          b.emit(Op.IAdd, [tU32, dbOff, sharedOffset, writeOffA]);
          sharedOffset = dbOff;
        }
        emitLoadTileA(b, tF32, tF16, tU32, tPtrSharedF16, bufA, inputF16,
          elemOffset, sharedOffset, kVal, globalRowBase, M, K, tileA,
          const0u, kTileK, constKTileKShift, constKTileKMask,
          batchOffsetA, batched, transposedA);
      }
    }

    // ── Load tile B: incremental stride addressing ────────────────────
    if (canUseStridedCoordsB) {
      // Global base address
      const kRow0B = b.id();
      b.emit(Op.IAdd, [tU32, kRow0B, kVal, baseLocalRowB!]);
      const nCol0B = b.id();
      b.emit(Op.IAdd, [tU32, nCol0B, globalColBase, baseLocalColB!]);
      let globalBaseB: number;
      if (transposedB) {
        // B[N,K]: index = nCol * K + kRow
        const nK = b.id();
        b.emit(Op.IMul, [tU32, nK, nCol0B, K]);
        globalBaseB = b.id();
        b.emit(Op.IAdd, [tU32, globalBaseB, nK, kRow0B]);
      } else {
        // B[K,N]: index = kRow * N + nCol
        const kN = b.id();
        b.emit(Op.IMul, [tU32, kN, kRow0B, N]);
        globalBaseB = b.id();
        b.emit(Op.IAdd, [tU32, globalBaseB, kN, nCol0B]);
      }
      if (batched) {
        const gbb = b.id();
        b.emit(Op.IAdd, [tU32, gbb, globalBaseB, batchOffsetB!]);
        globalBaseB = gbb;
      }

      // Global stride
      let globalStrideB: number;
      if (transposedB) {
        // transposed: increment in K dimension = rowStrideB
        if (rowStrideB === 1) {
          globalStrideB = const1u;
        } else {
          globalStrideB = b.id(); b.constant(tU32, globalStrideB, rowStrideB);
        }
      } else {
        // non-transposed: increment in K row dimension = rowStrideB * N
        if (rowStrideB === 1) {
          globalStrideB = N;
        } else {
          const cRSB = b.id(); b.constant(tU32, cRSB, rowStrideB);
          globalStrideB = b.id();
          b.emit(Op.IMul, [tU32, globalStrideB, cRSB, N]);
        }
      }

      // Shared base: subgroupBaseB + localRow0 * paddedStrideB + localCol0
      const constPaddedStrB = b.id(); b.constant(tU32, constPaddedStrB, paddedStrideB);
      const shrRowOff0B = b.id();
      b.emit(Op.IMul, [tU32, shrRowOff0B, baseLocalRowB!, constPaddedStrB]);
      let sharedBaseB = b.id();
      b.emit(Op.IAdd, [tU32, sharedBaseB, shrRowOff0B, baseLocalColB!]);
      if (subgroupMode) {
        const sbb = b.id();
        b.emit(Op.IAdd, [tU32, sbb, subgroupBaseB, sharedBaseB]);
        sharedBaseB = sbb;
      }
      if (useDoubleBuf) {
        const dbb = b.id();
        b.emit(Op.IAdd, [tU32, dbb, sharedBaseB, writeOffB]);
        sharedBaseB = dbb;
      }

      // Shared stride: rowStrideB * paddedStrideB (compile-time constant)
      const sharedStrideB = rowStrideB * paddedStrideB;
      const constSharedStrB = b.id(); b.constant(tU32, constSharedStrB, sharedStrideB);

      // Element loop
      let curGlobalB = globalBaseB;
      let curSharedB = sharedBaseB;
      for (let e = 0; e < elemsPerThreadB; e++) {
        if (e > 0) {
          const nextG = b.id();
          b.emit(Op.IAdd, [tU32, nextG, curGlobalB, globalStrideB]);
          curGlobalB = nextG;
          const nextS = b.id();
          b.emit(Op.IAdd, [tU32, nextS, curSharedB, constSharedStrB]);
          curSharedB = nextS;
        }

        if (e * loadWidthB + loadWidthB > totalB) {
          const elemOff = b.id();
          if (e === 0) {
            b.emit(Op.CopyObject, [tU32, elemOff, loadThreadBaseB]);
          } else {
            const cE = b.id(); b.constant(tU32, cE, e * loadWidthB);
            b.emit(Op.IAdd, [tU32, elemOff, loadThreadBaseB, cE]);
          }
          const cTot = b.id(); b.constant(tU32, cTot, totalB);
          const inB = b.id();
          b.emit(Op.ULessThan, [tBool, inB, elemOff, cTot]);
          const lblL = b.id(); const lblS = b.id();
          b.emit(Op.SelectionMerge, [lblS, 0]);
          b.emit(Op.BranchConditional, [inB, lblL, lblS]);
          b.emit(Op.Label, [lblL]);
          emitFastLoad(b, tF32, tF16, tU32, tPtrSharedF16,
            bufB, inputF16, curGlobalB, curSharedB, tileB, const0u);
          b.emit(Op.Branch, [lblS]);
          b.emit(Op.Label, [lblS]);
        } else {
          emitFastLoad(b, tF32, tF16, tU32, tPtrSharedF16,
            bufB, inputF16, curGlobalB, curSharedB, tileB, const0u);
        }
      }
    } else {
      // Fallback: per-element decomposition for non-strided configs
      let constShmemPadB: number | undefined;
      if (shmemPad > 0) {
        constShmemPadB = b.id(); b.constant(tU32, constShmemPadB, shmemPad);
      }
      for (let e = 0; e < elemsPerThreadB; e++) {
        const elemOffset = b.id();
        if (e === 0) {
          b.emit(Op.CopyObject, [tU32, elemOffset, loadThreadBaseB]);
        } else {
          const constE = b.id(); b.constant(tU32, constE, e * loadWidthB);
          b.emit(Op.IAdd, [tU32, elemOffset, loadThreadBaseB, constE]);
        }
        let effectiveOffset = elemOffset;
        if (shmemPad > 0) {
          const localRow = b.id();
          if (constBTileStrideShift !== undefined) {
            b.emit(Op.ShiftRightLogical, [tU32, localRow, elemOffset, constBTileStrideShift]);
          } else {
            const cbs = b.id(); b.constant(tU32, cbs, bTileStride);
            b.emit(Op.UDiv, [tU32, localRow, elemOffset, cbs]);
          }
          const padExtra = b.id();
          b.emit(Op.IMul, [tU32, padExtra, localRow, constShmemPadB!]);
          effectiveOffset = b.id();
          b.emit(Op.IAdd, [tU32, effectiveOffset, elemOffset, padExtra]);
        }
        let sharedOffset: number;
        if (subgroupMode) {
          sharedOffset = b.id();
          b.emit(Op.IAdd, [tU32, sharedOffset, subgroupBaseB, effectiveOffset]);
        } else {
          sharedOffset = effectiveOffset;
        }
        if (useDoubleBuf) {
          const dbOff = b.id();
          b.emit(Op.IAdd, [tU32, dbOff, sharedOffset, writeOffB]);
          sharedOffset = dbOff;
        }
        emitLoadTileB(b, tF32, tF16, tU32, tPtrSharedF16, bufB, inputF16,
          elemOffset, sharedOffset, kVal, globalColBase, N, K, tileB,
          const0u, bTileStride, constBTileStrideShift, constBTileStrideMask,
          batchOffsetB, batched, transposedB);
      }
    }

    if (useDoubleBuf) {
      // ── Double buffer: MMA from PREVIOUS tile, overlapped with loads ────
      // On the first iteration (k==0), skip MMA (no previous data yet).
      const isNotFirst = b.id();
      b.emit(Op.ULessThan, [tBool, isNotFirst, const0u, kVal]);
      const labelDoMMA = b.id();
      const labelSkipMMA = b.id();
      b.emit(Op.SelectionMerge, [labelSkipMMA, 0]);
      b.emit(Op.BranchConditional, [isNotFirst, labelDoMMA, labelSkipMMA]);
      b.emit(Op.Label, [labelDoMMA]);

      // Compute read offset (opposite half from write offset)
      const phase2 = b.id();
      b.emit(Op.Load, [tU32, phase2, varDbPhase!]);
      const oneMinusPhase = b.id();
      b.emit(Op.ISub, [tU32, oneMinusPhase, const1u, phase2]);
      const constHalfA2 = b.id(); b.constant(tU32, constHalfA2, halfSizeA);
      const constHalfB2 = b.id(); b.constant(tU32, constHalfB2, halfSizeB);
      const readOffA = b.id(); b.emit(Op.IMul, [tU32, readOffA, oneMinusPhase, constHalfA2]);
      const readOffB = b.id(); b.emit(Op.IMul, [tU32, readOffB, oneMinusPhase, constHalfB2]);

      let readBaseA: number;
      if (subgroupMode) {
        readBaseA = b.id();
        b.emit(Op.IAdd, [tU32, readBaseA, subgroupBaseA, readOffA]);
      } else {
        readBaseA = readOffA;
      }
      let readBaseB: number;
      if (subgroupMode) {
        readBaseB = b.id();
        b.emit(Op.IAdd, [tU32, readBaseB, subgroupBaseB, readOffB]);
      } else {
        readBaseB = readOffB;
      }

      // kMulti MMA steps from previous tile (same pattern as single buffer)
      const effectivePadADb = coopK * (kMulti - 1) + shmemPad;
      for (let km = 0; km < kMulti; km++) {
        let adjABase = readBaseA;
        if (km > 0) {
          const constKmA = b.id(); b.constant(tU32, constKmA, km * coopK);
          adjABase = b.id();
          b.emit(Op.IAdd, [tU32, adjABase, readBaseA, constKmA]);
        }
        let adjBBase = readBaseB;
        if (km > 0) {
          const constKmB = b.id(); b.constant(tU32, constKmB, km * coopK * paddedStrideB);
          adjBBase = b.id();
          b.emit(Op.IAdd, [tU32, adjBBase, readBaseB, constKmB]);
        }
        emitCoopMatrixMMA(b, tF16, tU32, tCoopA, tCoopB, tCoopCAcc,
          tPtrSharedF16, tileA, tileB, adjABase, adjBBase,
          constRowMajor, coopM, coopN, coopK,
          regTilesM, regTilesN, bTileStride, varAccs, effectivePadADb, shmemPad);
      }

      b.emit(Op.Branch, [labelSkipMMA]);
      b.emit(Op.Label, [labelSkipMMA]);

      // Single barrier: wait for current tile's loads to complete
      b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

      // Toggle phase: 0→1, 1→0
      const prevPhase = b.id();
      b.emit(Op.Load, [tU32, prevPhase, varDbPhase!]);
      const nextPhase = b.id();
      b.emit(Op.ISub, [tU32, nextPhase, const1u, prevPhase]);
      b.emit(Op.Store, [varDbPhase!, nextPhase]);
    } else {
      // ── Single buffer: barrier → kMulti × MMA → barrier ────────────────
      // Barrier — wait for all shared memory writes
      b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

      const tileABase = subgroupMode ? subgroupBaseA : const0u;
      const tileBBase = subgroupMode ? subgroupBaseB : const0u;
      // Effective shmemPadA for emitCoopMatrixMMA: strideA = coopK + shmemPadA
      // We need strideA = paddedStrideA = kTileK + shmemPad = coopK*kMulti + shmemPad
      // So shmemPadA = coopK*(kMulti-1) + shmemPad
      const effectivePadA = coopK * (kMulti - 1) + shmemPad;
      for (let km = 0; km < kMulti; km++) {
        let adjABase = tileABase;
        if (km > 0) {
          const constKmA = b.id(); b.constant(tU32, constKmA, km * coopK);
          adjABase = b.id();
          b.emit(Op.IAdd, [tU32, adjABase, tileABase, constKmA]);
        }
        let adjBBase = tileBBase;
        if (km > 0) {
          const constKmB = b.id(); b.constant(tU32, constKmB, km * coopK * paddedStrideB);
          adjBBase = b.id();
          b.emit(Op.IAdd, [tU32, adjBBase, tileBBase, constKmB]);
        }
        emitCoopMatrixMMA(b, tF16, tU32, tCoopA, tCoopB, tCoopCAcc,
          tPtrSharedF16, tileA, tileB, adjABase, adjBBase,
          constRowMajor, coopM, coopN, coopK,
          regTilesM, regTilesN, bTileStride, varAccs, effectivePadA, shmemPad);
      }

      // Barrier before next tile load
      b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);
    }
    coopAMat = 0 as any; // unused in refactored path
    coopBMat = 0 as any;
  }

  // ── Loop continue: k += kTileK (= coopK * kMulti) ──────────────────
  b.emit(Op.Branch, [labelLoopContinue]);
  b.emit(Op.Label, [labelLoopContinue]);

  // kNext is pre-allocated; Phi in header resolves it as the back-edge value
  b.emit(Op.IAdd, [tU32, kNext, kVal, constKStep]);
  b.emit(Op.Branch, [labelLoopHeader]);

  // ── Loop end: store results to global C (register-tiled) ───────────
  b.emit(Op.Label, [labelLoopEnd]);

  // Double buffer epilogue: MMA the final K-tile that was loaded but not yet consumed.
  // After the last iteration, phase was toggled, so the data we need is at (1-phase)*halfSize.
  if (useDoubleBuf) {
    // Barrier to ensure all stores from the last iteration are visible
    b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

    const epilogPhase = b.id();
    b.emit(Op.Load, [tU32, epilogPhase, varDbPhase!]);
    const epilogOneMinusPhase = b.id();
    b.emit(Op.ISub, [tU32, epilogOneMinusPhase, const1u, epilogPhase]);
    const constHalfAEpi = b.id(); b.constant(tU32, constHalfAEpi, halfSizeA);
    const constHalfBEpi = b.id(); b.constant(tU32, constHalfBEpi, halfSizeB);
    const readOffAEpi = b.id(); b.emit(Op.IMul, [tU32, readOffAEpi, epilogOneMinusPhase, constHalfAEpi]);
    const readOffBEpi = b.id(); b.emit(Op.IMul, [tU32, readOffBEpi, epilogOneMinusPhase, constHalfBEpi]);

    let readBaseAEpi: number;
    if (subgroupMode) {
      readBaseAEpi = b.id();
      b.emit(Op.IAdd, [tU32, readBaseAEpi, subgroupBaseA_h, readOffAEpi]);
    } else {
      readBaseAEpi = readOffAEpi;
    }
    let readBaseBEpi: number;
    if (subgroupMode) {
      readBaseBEpi = b.id();
      b.emit(Op.IAdd, [tU32, readBaseBEpi, subgroupBaseB_h, readOffBEpi]);
    } else {
      readBaseBEpi = readOffBEpi;
    }

    // kMulti MMA steps for final tile (same pattern as loop body)
    const effectivePadAEpi = coopK * (kMulti - 1) + shmemPad;
    for (let km = 0; km < kMulti; km++) {
      let adjABaseEpi = readBaseAEpi;
      if (km > 0) {
        const constKmA = b.id(); b.constant(tU32, constKmA, km * coopK);
        adjABaseEpi = b.id();
        b.emit(Op.IAdd, [tU32, adjABaseEpi, readBaseAEpi, constKmA]);
      }
      let adjBBaseEpi = readBaseBEpi;
      if (km > 0) {
        const constKmB = b.id(); b.constant(tU32, constKmB, km * coopK * paddedStrideB);
        adjBBaseEpi = b.id();
        b.emit(Op.IAdd, [tU32, adjBBaseEpi, readBaseBEpi, constKmB]);
      }
      emitCoopMatrixMMA(b, tF16, tU32, tCoopA, tCoopB, tCoopCAcc,
        tPtrSharedF16, tileA, tileB, adjABaseEpi, adjBBaseEpi,
        constRowMajor, coopM, coopN, coopK,
        regTilesM, regTilesN, bTileStride, varAccs, effectivePadAEpi, shmemPad);
    }
  }

  for (let rm = 0; rm < regTilesM; rm++) {
    for (let rn = 0; rn < regTilesN; rn++) {
      // Output address for tile (rm, rn):
      //   row = globalRowBase + rm * coopM
      //   col = globalColBase + rn * coopN
      //   addr = batchOffset + row * N + col
      let tileRowBase: number;
      if (rm === 0) {
        tileRowBase = globalRowBase;
      } else {
        const constRM = b.id(); b.constant(tU32, constRM, rm * coopM);
        tileRowBase = b.id();
        b.emit(Op.IAdd, [tU32, tileRowBase, globalRowBase, constRM]);
      }
      let tileColBase: number;
      if (rn === 0) {
        tileColBase = globalColBase;
      } else {
        const constRN = b.id(); b.constant(tU32, constRN, rn * coopN);
        tileColBase = b.id();
        b.emit(Op.IAdd, [tU32, tileColBase, globalColBase, constRN]);
      }

      const rowTimesN = b.id();
      b.emit(Op.IMul, [tU32, rowTimesN, tileRowBase, N]);
      let outBase = b.id();
      b.emit(Op.IAdd, [tU32, outBase, rowTimesN, tileColBase]);

      if (batched) {
        const batchOffsetC = b.id();
        b.emit(Op.IMul, [tU32, batchOffsetC, batchIdx!, MN]);
        const outWithBatch = b.id();
        b.emit(Op.IAdd, [tU32, outWithBatch, outBase, batchOffsetC]);
        outBase = outWithBatch;
      }

      const ptrCOut = b.id();
      b.emit(Op.AccessChain, [bufC.tPtrF32, ptrCOut, bufC.varId, const0u, outBase]);

      let finalAcc = b.id();
      b.emit(Op.Load, [tCoopCAcc, finalAcc, varAccs[rm][rn]]);
      if (accumF16) {
        const finalAccF32 = b.id();
        b.emit(Op.FConvert, [tCoopCOut, finalAccF32, finalAcc]);
        finalAcc = finalAccF32;
      }
      b.emit(Op.OpCooperativeMatrixStoreKHR, [ptrCOut, finalAcc, constRowMajor, N]);
    }
  }

  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

/**
 * Emit a single global→shared load with a precomputed global address.
 * Used by the address-hoisting fast path.
 */
function emitFastLoad(
  b: SpirVBuilder,
  tF32: number, tF16: number, tU32: number, tPtrSharedF16: number,
  buf: { varId: number; tPtrF32?: number; tPtrF16?: number },
  inputF16: boolean,
  globalAddr: number, sharedAddr: number,
  tile: number, const0u: number,
): void {
  let valF16: number;
  if (inputF16) {
    const ptr = b.id();
    b.emit(Op.AccessChain, [buf.tPtrF16!, ptr, buf.varId, const0u, globalAddr]);
    valF16 = b.id();
    b.emit(Op.Load, [tF16, valF16, ptr]);
  } else {
    const ptr = b.id();
    b.emit(Op.AccessChain, [buf.tPtrF32!, ptr, buf.varId, const0u, globalAddr]);
    const valF32 = b.id();
    b.emit(Op.Load, [tF32, valF32, ptr]);
    valF16 = b.id();
    b.emit(Op.FConvert, [tF16, valF16, valF32]);
  }
  const ptrShared = b.id();
  b.emit(Op.AccessChain, [tPtrSharedF16, ptrShared, tile, sharedAddr]);
  b.emit(Op.Store, [ptrShared, valF16]);
}

/**
 * Emit code to load one A element from global memory into a Function-scope variable (register).
 * Returns the variable ID holding the f16 value.
 * Used by software pipelining to separate global load from shared store.
 */
function emitGlobalLoadA(
  b: SpirVBuilder,
  tF32: number, tF16: number, tU32: number,
  bufA: { varId: number; tPtrF32?: number; tPtrF16?: number },
  inputF16: boolean,
  elemOffset: number,
  kVal: number,
  globalRowBase: number,
  M: number, K: number,
  const0u: number, coopK: number,
  constCoopKShift: number | undefined,
  constCoopKMask: number | undefined,
  batchOffsetA: number | undefined, batched: boolean,
  transposedA: boolean,
  precomputedLocalRow?: number,
  precomputedLocalCol?: number,
): number {
  const constCoopK = b.id();
  b.constant(tU32, constCoopK, coopK);

  let localRow = precomputedLocalRow;
  let localCol = precomputedLocalCol;
  if (localRow === undefined || localCol === undefined) {
    localRow = b.id();
    localCol = b.id();
    if (constCoopKShift !== undefined && constCoopKMask !== undefined) {
      b.emit(Op.ShiftRightLogical, [tU32, localRow, elemOffset, constCoopKShift]);
      b.emit(Op.BitwiseAnd, [tU32, localCol, elemOffset, constCoopKMask]);
    } else {
      b.emit(Op.UDiv, [tU32, localRow, elemOffset, constCoopK]);
      b.emit(Op.UMod, [tU32, localCol, elemOffset, constCoopK]);
    }
  }

  const globalRow = b.id();
  b.emit(Op.IAdd, [tU32, globalRow, globalRowBase, localRow!]);
  const globalCol = b.id();
  b.emit(Op.IAdd, [tU32, globalCol, kVal, localCol!]);

  let aIdx: number;
  if (transposedA) {
    const colTimesM = b.id();
    b.emit(Op.IMul, [tU32, colTimesM, globalCol, M]);
    aIdx = b.id();
    b.emit(Op.IAdd, [tU32, aIdx, colTimesM, globalRow]);
  } else {
    const rowTimesK = b.id();
    b.emit(Op.IMul, [tU32, rowTimesK, globalRow, K]);
    aIdx = b.id();
    b.emit(Op.IAdd, [tU32, aIdx, rowTimesK, globalCol]);
  }

  if (batched) {
    const aIdxBatch = b.id();
    b.emit(Op.IAdd, [tU32, aIdxBatch, aIdx, batchOffsetA!]);
    aIdx = aIdxBatch;
  }

  let valF16: number;
  if (inputF16) {
    const ptrA = b.id();
    b.emit(Op.AccessChain, [bufA.tPtrF16!, ptrA, bufA.varId, const0u, aIdx]);
    valF16 = b.id();
    b.emit(Op.Load, [tF16, valF16, ptrA]);
  } else {
    const ptrA = b.id();
    b.emit(Op.AccessChain, [bufA.tPtrF32!, ptrA, bufA.varId, const0u, aIdx]);
    const valF32 = b.id();
    b.emit(Op.Load, [tF32, valF32, ptrA]);
    valF16 = b.id();
    b.emit(Op.FConvert, [tF16, valF16, valF32]);
  }

  return valF16;
}

/**
 * Emit code to load one B element from global memory into a register.
 * Returns the SSA ID of the f16 value.
 */
function emitGlobalLoadB(
  b: SpirVBuilder,
  tF32: number, tF16: number, tU32: number,
  bufB: { varId: number; tPtrF32?: number; tPtrF16?: number },
  inputF16: boolean,
  elemOffset: number,
  kVal: number,
  globalColBase: number,
  N: number, K: number,
  const0u: number, bTileStride: number,
  constBTileStrideShift: number | undefined,
  constBTileStrideMask: number | undefined,
  batchOffsetB: number | undefined, batched: boolean,
  transposed: boolean,
  precomputedLocalRow?: number,
  precomputedLocalCol?: number,
): number {
  const constCoopN = b.id();
  b.constant(tU32, constCoopN, bTileStride);

  let localRow = precomputedLocalRow;
  let localCol = precomputedLocalCol;
  if (localRow === undefined || localCol === undefined) {
    localRow = b.id();
    localCol = b.id();
    if (constBTileStrideShift !== undefined && constBTileStrideMask !== undefined) {
      b.emit(Op.ShiftRightLogical, [tU32, localRow, elemOffset, constBTileStrideShift]);
      b.emit(Op.BitwiseAnd, [tU32, localCol, elemOffset, constBTileStrideMask]);
    } else {
      b.emit(Op.UDiv, [tU32, localRow, elemOffset, constCoopN]);
      b.emit(Op.UMod, [tU32, localCol, elemOffset, constCoopN]);
    }
  }

  const kRow = b.id();
  b.emit(Op.IAdd, [tU32, kRow, kVal, localRow!]);
  const nCol = b.id();
  b.emit(Op.IAdd, [tU32, nCol, globalColBase, localCol!]);

  let bIdx: number;
  if (transposed) {
    const nTimesK = b.id();
    b.emit(Op.IMul, [tU32, nTimesK, nCol, K]);
    bIdx = b.id();
    b.emit(Op.IAdd, [tU32, bIdx, nTimesK, kRow]);
  } else {
    const kTimesN = b.id();
    b.emit(Op.IMul, [tU32, kTimesN, kRow, N]);
    bIdx = b.id();
    b.emit(Op.IAdd, [tU32, bIdx, kTimesN, nCol]);
  }

  if (batched) {
    const bIdxBatch = b.id();
    b.emit(Op.IAdd, [tU32, bIdxBatch, bIdx, batchOffsetB!]);
    bIdx = bIdxBatch;
  }

  let valF16: number;
  if (inputF16) {
    const ptrB = b.id();
    b.emit(Op.AccessChain, [bufB.tPtrF16!, ptrB, bufB.varId, const0u, bIdx]);
    valF16 = b.id();
    b.emit(Op.Load, [tF16, valF16, ptrB]);
  } else {
    const ptrB = b.id();
    b.emit(Op.AccessChain, [bufB.tPtrF32!, ptrB, bufB.varId, const0u, bIdx]);
    const valF32 = b.id();
    b.emit(Op.Load, [tF32, valF32, ptrB]);
    valF16 = b.id();
    b.emit(Op.FConvert, [tF16, valF16, valF32]);
  }

  return valF16;
}

/**
 * Emit code to load one element from A[global] into shared tileA[offset].
 * A is row-major [M, K], element at (row, col) = A[row * K + col].
 * For transposedA path, A is interpreted as [K, M] and loaded as A[col, row].
 * Tile element offset maps to (localRow, localCol) within the coopM x coopK tile.
 */
function emitLoadTileA(
  b: SpirVBuilder,
  tF32: number, tF16: number, tU32: number, tPtrSharedF16: number,
  bufA: { varId: number; tPtrF32?: number; tPtrF16?: number },
  inputF16: boolean,
  elemOffset: number,
  sharedOffset: number,
  kVal: number,
  globalRowBase: number,
  M: number, K: number,
  tileA: number,
  const0u: number, coopK: number,
  constCoopKShift: number | undefined,
  constCoopKMask: number | undefined,
  batchOffsetA: number | undefined, batched: boolean,
  transposedA: boolean,
  precomputedLocalRow?: number,
  precomputedLocalCol?: number,
): void {
  const constCoopK = b.id();
  b.constant(tU32, constCoopK, coopK);

  // localRow = elemOffset / coopK, localCol = elemOffset % coopK
  let localRow = precomputedLocalRow;
  let localCol = precomputedLocalCol;
  if (localRow === undefined || localCol === undefined) {
    localRow = b.id();
    localCol = b.id();
    if (constCoopKShift !== undefined && constCoopKMask !== undefined) {
      b.emit(Op.ShiftRightLogical, [tU32, localRow, elemOffset, constCoopKShift]);
      b.emit(Op.BitwiseAnd, [tU32, localCol, elemOffset, constCoopKMask]);
    } else {
      b.emit(Op.UDiv, [tU32, localRow, elemOffset, constCoopK]);
      b.emit(Op.UMod, [tU32, localCol, elemOffset, constCoopK]);
    }
  }
  const localRowVal = localRow!;
  const localColVal = localCol!;

  // globalRow = globalRowBase + localRow
  const globalRow = b.id();
  b.emit(Op.IAdd, [tU32, globalRow, globalRowBase, localRowVal]);

  // globalCol = kVal + localCol
  const globalCol = b.id();
  b.emit(Op.IAdd, [tU32, globalCol, kVal, localColVal]);

  // A index:
  //   regular:     A[globalRow, globalCol] => globalRow * K + globalCol
  //   transposedA: A[globalCol, globalRow] => globalCol * M + globalRow
  let aIdx: number;
  if (transposedA) {
    const colTimesM = b.id();
    b.emit(Op.IMul, [tU32, colTimesM, globalCol, M]);
    aIdx = b.id();
    b.emit(Op.IAdd, [tU32, aIdx, colTimesM, globalRow]);
  } else {
    const rowTimesK = b.id();
    b.emit(Op.IMul, [tU32, rowTimesK, globalRow, K]);
    aIdx = b.id();
    b.emit(Op.IAdd, [tU32, aIdx, rowTimesK, globalCol]);
  }

  if (batched) {
    const aIdxBatch = b.id();
    b.emit(Op.IAdd, [tU32, aIdxBatch, aIdx, batchOffsetA!]);
    aIdx = aIdxBatch;
  }

  let valF16: number;
  if (inputF16) {
    // Load f16 directly from pre-cast storage buffer.
    const ptrA = b.id();
    b.emit(Op.AccessChain, [bufA.tPtrF16!, ptrA, bufA.varId, const0u, aIdx]);
    valF16 = b.id();
    b.emit(Op.Load, [tF16, valF16, ptrA]);
  } else {
    // Load f32 then convert to f16 for cooperative matrix tile.
    const ptrA = b.id();
    b.emit(Op.AccessChain, [bufA.tPtrF32!, ptrA, bufA.varId, const0u, aIdx]);
    const valF32 = b.id();
    b.emit(Op.Load, [tF32, valF32, ptrA]);
    valF16 = b.id();
    b.emit(Op.FConvert, [tF16, valF16, valF32]);
  }

  // Store to shared memory
  const ptrShared = b.id();
  b.emit(Op.AccessChain, [tPtrSharedF16, ptrShared, tileA, sharedOffset]);
  b.emit(Op.Store, [ptrShared, valF16]);
}

/**
 * Emit code to load one element from B[global] into shared tileB[offset].
 * Non-transposed: B is row-major [K, N], element at (row, col) = B[row * N + col].
 * Transposed: B is stored as [N, K], element at (row, col) = B[col * K + row].
 * Tile element maps to (localRow, localCol) within the coopK x coopN tile.
 */
function emitLoadTileB(
  b: SpirVBuilder,
  tF32: number, tF16: number, tU32: number, tPtrSharedF16: number,
  bufB: { varId: number; tPtrF32?: number; tPtrF16?: number },
  inputF16: boolean,
  elemOffset: number,
  sharedOffset: number,
  kVal: number,
  globalColBase: number,
  N: number, K: number,
  tileB: number,
  const0u: number, coopN: number,
  constCoopNShift: number | undefined,
  constCoopNMask: number | undefined,
  batchOffsetB: number | undefined, batched: boolean,
  transposed: boolean,
  precomputedLocalRow?: number,
  precomputedLocalCol?: number,
): void {
  const constCoopN = b.id();
  b.constant(tU32, constCoopN, coopN);

  // localRow = elemOffset / coopN, localCol = elemOffset % coopN
  let localRow = precomputedLocalRow;
  let localCol = precomputedLocalCol;
  if (localRow === undefined || localCol === undefined) {
    localRow = b.id();
    localCol = b.id();
    if (constCoopNShift !== undefined && constCoopNMask !== undefined) {
      b.emit(Op.ShiftRightLogical, [tU32, localRow, elemOffset, constCoopNShift]);
      b.emit(Op.BitwiseAnd, [tU32, localCol, elemOffset, constCoopNMask]);
    } else {
      b.emit(Op.UDiv, [tU32, localRow, elemOffset, constCoopN]);
      b.emit(Op.UMod, [tU32, localCol, elemOffset, constCoopN]);
    }
  }
  const localRowVal = localRow!;
  const localColVal = localCol!;

  // kRow = kVal + localRow (row within K dimension)
  const kRow = b.id();
  b.emit(Op.IAdd, [tU32, kRow, kVal, localRowVal]);

  // nCol = globalColBase + localCol
  const nCol = b.id();
  b.emit(Op.IAdd, [tU32, nCol, globalColBase, localColVal]);

  let bIdx: number;
  if (transposed) {
    // B stored as [N, K]: B[nCol * K + kRow]
    const nTimesK = b.id();
    b.emit(Op.IMul, [tU32, nTimesK, nCol, K]);
    bIdx = b.id();
    b.emit(Op.IAdd, [tU32, bIdx, nTimesK, kRow]);
  } else {
    // B stored as [K, N]: B[kRow * N + nCol]
    const kTimesN = b.id();
    b.emit(Op.IMul, [tU32, kTimesN, kRow, N]);
    bIdx = b.id();
    b.emit(Op.IAdd, [tU32, bIdx, kTimesN, nCol]);
  }

  if (batched) {
    const bIdxBatch = b.id();
    b.emit(Op.IAdd, [tU32, bIdxBatch, bIdx, batchOffsetB!]);
    bIdx = bIdxBatch;
  }

  let valF16: number;
  if (inputF16) {
    // Load f16 directly from pre-cast storage buffer.
    const ptrB = b.id();
    b.emit(Op.AccessChain, [bufB.tPtrF16!, ptrB, bufB.varId, const0u, bIdx]);
    valF16 = b.id();
    b.emit(Op.Load, [tF16, valF16, ptrB]);
  } else {
    // Load f32 then convert to f16 for cooperative matrix tile.
    const ptrB = b.id();
    b.emit(Op.AccessChain, [bufB.tPtrF32!, ptrB, bufB.varId, const0u, bIdx]);
    const valF32 = b.id();
    b.emit(Op.Load, [tF32, valF32, ptrB]);
    valF16 = b.id();
    b.emit(Op.FConvert, [tF16, valF16, valF32]);
  }

  // Store to shared memory
  const ptrShared = b.id();
  b.emit(Op.AccessChain, [tPtrSharedF16, ptrShared, tileB, sharedOffset]);
  b.emit(Op.Store, [ptrShared, valF16]);
}

// ── Exported kernel generators ──────────────────────────────────────────────

export function kernelCoopMatmulBasic(
  coopM: number, coopN: number, coopK: number,
  inputF16 = false, accumF16 = false,
  subgroupTilesX = 1, subgroupTilesY = 1,
  regTilesM = 1, regTilesN = 1,
  doubleBuf = false,
): Uint32Array {
  return buildCoopMatmul(coopM, coopN, coopK, false, false, false, inputF16, accumF16, subgroupTilesX, subgroupTilesY, regTilesM, regTilesN, doubleBuf);
}

export function kernelCoopMatmulBatched(
  coopM: number, coopN: number, coopK: number,
  inputF16 = false, accumF16 = false,
  subgroupTilesX = 1, subgroupTilesY = 1,
  regTilesM = 1, regTilesN = 1,
  doubleBuf = false,
): Uint32Array {
  return buildCoopMatmul(coopM, coopN, coopK, true, false, false, inputF16, accumF16, subgroupTilesX, subgroupTilesY, regTilesM, regTilesN, doubleBuf);
}

export function kernelCoopMatmulTransposed(
  coopM: number, coopN: number, coopK: number,
  inputF16 = false, accumF16 = false,
  subgroupTilesX = 1, subgroupTilesY = 1,
  regTilesM = 1, regTilesN = 1,
  doubleBuf = false,
): Uint32Array {
  return buildCoopMatmul(coopM, coopN, coopK, false, true, false, inputF16, accumF16, subgroupTilesX, subgroupTilesY, regTilesM, regTilesN, doubleBuf);
}

export function kernelCoopMatmulTransposedBatched(
  coopM: number, coopN: number, coopK: number,
  inputF16 = false, accumF16 = false,
  subgroupTilesX = 1, subgroupTilesY = 1,
  regTilesM = 1, regTilesN = 1,
  doubleBuf = false,
): Uint32Array {
  return buildCoopMatmul(coopM, coopN, coopK, true, true, false, inputF16, accumF16, subgroupTilesX, subgroupTilesY, regTilesM, regTilesN, doubleBuf);
}

export function kernelCoopMatmulTransposedA(
  coopM: number, coopN: number, coopK: number,
  inputF16 = false, accumF16 = false,
  subgroupTilesX = 1, subgroupTilesY = 1,
  regTilesM = 1, regTilesN = 1,
  doubleBuf = false,
): Uint32Array {
  return buildCoopMatmul(coopM, coopN, coopK, false, false, true, inputF16, accumF16, subgroupTilesX, subgroupTilesY, regTilesM, regTilesN, doubleBuf);
}

export function kernelCoopMatmulTransposedABatched(
  coopM: number, coopN: number, coopK: number,
  inputF16 = false, accumF16 = false,
  subgroupTilesX = 1, subgroupTilesY = 1,
  regTilesM = 1, regTilesN = 1,
  doubleBuf = false,
): Uint32Array {
  return buildCoopMatmul(coopM, coopN, coopK, true, false, true, inputF16, accumF16, subgroupTilesX, subgroupTilesY, regTilesM, regTilesN, doubleBuf);
}
