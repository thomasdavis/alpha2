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
  declareStorageBuffer, declareStorageBufferF16, declareParamsPushConstant,
} from "./helpers.js";

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
): Uint32Array {
  const b = new SpirVBuilder();
  const coopDebugMode = process.env.HELIOS_COOP_DEBUG_MODE ?? "";
  // Multi-subgroup tiles need direct cooperative global loads for correctness:
  // the shared-memory path currently uses one tile buffer per workgroup.
  const forceDirectForSubgroupTiles = subgroupTilesX > 1 || subgroupTilesY > 1;
  const useDirectF16Load =
    inputF16 && (process.env.HELIOS_COOP_DIRECT_LOAD === "1" || forceDirectForSubgroupTiles);

  // Capabilities
  b.addCapability(Capability.Shader);
  b.addCapability(Capability.Float16);
  b.addCapability(Capability.VulkanMemoryModel);
  b.addCapability(Capability.CooperativeMatrixKHR);
  if (inputF16 || accumF16) {
    b.addCapability(Capability.StorageBuffer16BitAccess);
  }

  // Extension
  b.addExtension("SPV_KHR_cooperative_matrix");
  b.addExtension("SPV_KHR_vulkan_memory_model");

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
  const const0f = b.id(); b.constantF32(tF32, const0f, 0.0);

  // Cooperative matrix layout constant: RowMajorKHR = 0
  const constRowMajor = const0u;
  // Cooperative matrix layout constant: ColumnMajorKHR = 1
  const constColumnMajor = const1u;

  // Shared memory for loading tiles from global memory
  // Tile A: coopM * coopK f16 elements = coopM * coopK * 2 bytes
  // Tile B: coopK * coopN f16 elements = coopK * coopN * 2 bytes
  // We store them as f16 arrays in shared memory
  const constTileASize = b.id(); b.constant(tU32, constTileASize, coopM * coopK);
  const tArrayTileA = b.id(); b.typeArray(tArrayTileA, tF16, constTileASize);
  const tPtrSharedArrA = b.id(); b.typePointer(tPtrSharedArrA, StorageClass.Workgroup, tArrayTileA);
  const tileA = b.id(); b.variable(tPtrSharedArrA, tileA, StorageClass.Workgroup);

  const constTileBSize = b.id(); b.constant(tU32, constTileBSize, coopK * coopN);
  const tArrayTileB = b.id(); b.typeArray(tArrayTileB, tF16, constTileBSize);
  const tPtrSharedArrB = b.id(); b.typePointer(tPtrSharedArrB, StorageClass.Workgroup, tArrayTileB);
  const tileB = b.id(); b.variable(tPtrSharedArrB, tileB, StorageClass.Workgroup);

  const tPtrSharedF16 = b.id();
  b.typePointer(tPtrSharedF16, StorageClass.Workgroup, tF16);

  // Storage buffers:
  // - A/B: f32 source buffers or pre-cast f16 source buffers
  // - C:   f32 output buffer
  const bufA: { varId: number; tPtrF32?: number; tPtrF16?: number } = inputF16
    ? declareStorageBufferF16(b, tF16, tU32, 0, 0, true)
    : declareStorageBuffer(b, tF32, tU32, 0, 0, true);
  const bufB: { varId: number; tPtrF32?: number; tPtrF16?: number } = inputF16
    ? declareStorageBufferF16(b, tF16, tU32, 0, 1, true)
    : declareStorageBuffer(b, tF32, tU32, 0, 1, true);
  const bufC = declareStorageBuffer(b, tF32, tU32, 0, 2, false);

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
  const varAcc = b.id();
  b.emit(Op.Variable, [tPtrFnCoopC, varAcc, StorageClass.Function]);

  // Load workgroup ID
  const wgIdVec = b.id();
  b.emit(Op.Load, [tVec3U32, wgIdVec, vWorkgroupId]);
  const wgTileCol = b.id();
  b.emit(Op.CompositeExtract, [tU32, wgTileCol, wgIdVec, 0]);
  const wgTileRow = b.id();
  b.emit(Op.CompositeExtract, [tU32, wgTileRow, wgIdVec, 1]);

  // Load local ID (thread index within workgroup)
  const lidVec = b.id();
  b.emit(Op.Load, [tVec3U32, lidVec, vLocalId]);
  const tid = b.id();
  b.emit(Op.CompositeExtract, [tU32, tid, lidVec, 0]);

  let tileCol: number;
  let tileRow: number;
  if (subgroupTilesX > 1 || subgroupTilesY > 1) {
    const constSubgroupSize = b.id();
    b.constant(tU32, constSubgroupSize, 32);
    const constSubgroupTilesX = b.id();
    b.constant(tU32, constSubgroupTilesX, subgroupTilesX);
    const constSubgroupTilesY = b.id();
    b.constant(tU32, constSubgroupTilesY, subgroupTilesY);

    const subgroupId = b.id();
    b.emit(Op.UDiv, [tU32, subgroupId, tid, constSubgroupSize]);
    const subgroupX = b.id();
    b.emit(Op.UMod, [tU32, subgroupX, subgroupId, constSubgroupTilesX]);
    const subgroupY = b.id();
    b.emit(Op.UDiv, [tU32, subgroupY, subgroupId, constSubgroupTilesX]);

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
  const globalRowBase = b.id(); // tileRow * coopM
  b.emit(Op.IMul, [tU32, globalRowBase, tileRow, constCoopM]);
  const globalColBase = b.id(); // tileCol * coopN
  b.emit(Op.IMul, [tU32, globalColBase, tileCol, constCoopN]);

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

  // Initialize accumulator to zero
  // We need to create a constant zero cooperative matrix — not directly supported.
  // Instead, we'll use OpConstant for f32 zero and composite-construct won't work for coop matrices.
  // The correct approach: use a variable and store zeros via a loop, or
  // better — we initialize by doing the first MulAdd with a zero acc.
  // Actually, OpConstantNull works for cooperative matrix types.
  const constNullC = b.id();
  b.constantNull(tCoopCAcc, constNullC);

  b.emit(Op.Store, [varAcc, constNullC]);

  // ── K-tile loop: for k = 0; k < K; k += coopK ──────────────────────
  b.emit(Op.Store, [varK, const0u]);

  const labelLoopHeader = b.id();
  const labelLoopBody = b.id();
  const labelLoopContinue = b.id();
  const labelLoopEnd = b.id();

  b.emit(Op.Branch, [labelLoopHeader]);
  b.emit(Op.Label, [labelLoopHeader]);

  const kVal = b.id();
  b.emit(Op.Load, [tU32, kVal, varK]);
  const kLtK = b.id();
  b.emit(Op.ULessThan, [tBool, kLtK, kVal, K]);

  b.emit(Op.LoopMerge, [labelLoopEnd, labelLoopContinue, 0]);
  b.emit(Op.BranchConditional, [kLtK, labelLoopBody, labelLoopEnd]);

  b.emit(Op.Label, [labelLoopBody]);

  let coopAMat: number;
  let coopBMat: number;
  if (useDirectF16Load) {
    // Direct cooperative loads from global f16 buffers.
    // This avoids shared-memory staging and per-element conversion overhead.
    let aBase = b.id();
    if (transposedA) {
      // A^T view of A[M,K] -> load with ColumnMajor layout from base (k, row)
      const kTimesM = b.id();
      b.emit(Op.IMul, [tU32, kTimesM, kVal, M]);
      b.emit(Op.IAdd, [tU32, aBase, kTimesM, globalRowBase]);
    } else {
      const rowTimesK = b.id();
      b.emit(Op.IMul, [tU32, rowTimesK, globalRowBase, K]);
      b.emit(Op.IAdd, [tU32, aBase, rowTimesK, kVal]);
    }
    if (batched) {
      const aWithBatch = b.id();
      b.emit(Op.IAdd, [tU32, aWithBatch, aBase, batchOffsetA!]);
      aBase = aWithBatch;
    }
    const ptrAStart = b.id();
    b.emit(Op.AccessChain, [bufA.tPtrF16!, ptrAStart, bufA.varId, const0u, aBase]);
    coopAMat = b.id();
    b.emit(Op.OpCooperativeMatrixLoadKHR, [
      tCoopA,
      coopAMat,
      ptrAStart,
      transposedA ? constColumnMajor : constRowMajor,
      transposedA ? M : K,
    ]);

    let bBase = b.id();
    if (transposedB) {
      // B^T view of B[N,K] -> load with ColumnMajor layout from base (n, k)
      const nTimesK = b.id();
      b.emit(Op.IMul, [tU32, nTimesK, globalColBase, K]);
      b.emit(Op.IAdd, [tU32, bBase, nTimesK, kVal]);
    } else {
      const kTimesN = b.id();
      b.emit(Op.IMul, [tU32, kTimesN, kVal, N]);
      b.emit(Op.IAdd, [tU32, bBase, kTimesN, globalColBase]);
    }
    if (batched) {
      const bWithBatch = b.id();
      b.emit(Op.IAdd, [tU32, bWithBatch, bBase, batchOffsetB!]);
      bBase = bWithBatch;
    }
    const ptrBStart = b.id();
    b.emit(Op.AccessChain, [bufB.tPtrF16!, ptrBStart, bufB.varId, const0u, bBase]);
    coopBMat = b.id();
    b.emit(Op.OpCooperativeMatrixLoadKHR, [
      tCoopB,
      coopBMat,
      ptrBStart,
      transposedB ? constColumnMajor : constRowMajor,
      transposedB ? K : N,
    ]);
  } else {
    // ── Load tile A from global to shared ───────────────────────────────
    // Each of 32 threads loads multiple elements to fill coopM*coopK
    // Total elements = coopM * coopK. Each thread loads ceil(total/32) elements.
    const totalA = coopM * coopK;
    const elemsPerThreadA = Math.ceil(totalA / WG_SIZE);

    for (let e = 0; e < elemsPerThreadA; e++) {
      const elemOffset = b.id();
      if (e === 0) {
        b.emit(Op.CopyObject, [tU32, elemOffset, tid]);
      } else {
        const constE = b.id();
        b.constant(tU32, constE, e * WG_SIZE);
        b.emit(Op.IAdd, [tU32, elemOffset, tid, constE]);
      }

      if (e * WG_SIZE + WG_SIZE > totalA) {
        const constTotal = b.id();
        b.constant(tU32, constTotal, totalA);
        const inBounds = b.id();
        b.emit(Op.ULessThan, [tBool, inBounds, elemOffset, constTotal]);
        const labelLoad = b.id();
        const labelSkip = b.id();
        b.emit(Op.SelectionMerge, [labelSkip, 0]);
        b.emit(Op.BranchConditional, [inBounds, labelLoad, labelSkip]);
        b.emit(Op.Label, [labelLoad]);

        emitLoadTileA(b, tF32, tF16, tU32, tPtrSharedF16, bufA, inputF16,
          elemOffset, kVal, globalRowBase, M, K, tileA,
          const0u, coopK, batchOffsetA, batched, transposedA);

        b.emit(Op.Branch, [labelSkip]);
        b.emit(Op.Label, [labelSkip]);
      } else {
        emitLoadTileA(b, tF32, tF16, tU32, tPtrSharedF16, bufA, inputF16,
          elemOffset, kVal, globalRowBase, M, K, tileA,
          const0u, coopK, batchOffsetA, batched, transposedA);
      }
    }

    // ── Load tile B from global to shared ───────────────────────────────
    const totalB = coopK * coopN;
    const elemsPerThreadB = Math.ceil(totalB / WG_SIZE);

    for (let e = 0; e < elemsPerThreadB; e++) {
      const elemOffset = b.id();
      if (e === 0) {
        b.emit(Op.CopyObject, [tU32, elemOffset, tid]);
      } else {
        const constE = b.id();
        b.constant(tU32, constE, e * WG_SIZE);
        b.emit(Op.IAdd, [tU32, elemOffset, tid, constE]);
      }

      if (e * WG_SIZE + WG_SIZE > totalB) {
        const constTotal = b.id();
        b.constant(tU32, constTotal, totalB);
        const inBounds = b.id();
        b.emit(Op.ULessThan, [tBool, inBounds, elemOffset, constTotal]);
        const labelLoad = b.id();
        const labelSkip = b.id();
        b.emit(Op.SelectionMerge, [labelSkip, 0]);
        b.emit(Op.BranchConditional, [inBounds, labelLoad, labelSkip]);
        b.emit(Op.Label, [labelLoad]);

        emitLoadTileB(b, tF32, tF16, tU32, tPtrSharedF16, bufB, inputF16,
          elemOffset, kVal, globalColBase, N, K, tileB,
          const0u, coopN, batchOffsetB, batched, transposedB);

        b.emit(Op.Branch, [labelSkip]);
        b.emit(Op.Label, [labelSkip]);
      } else {
        emitLoadTileB(b, tF32, tF16, tU32, tPtrSharedF16, bufB, inputF16,
          elemOffset, kVal, globalColBase, N, K, tileB,
          const0u, coopN, batchOffsetB, batched, transposedB);
      }
    }

    // Barrier — wait for all shared memory writes
    b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

    // ── Cooperative matrix load from shared memory ──────────────────────
    const ptrTileAStart = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF16, ptrTileAStart, tileA, const0u]);
    coopAMat = b.id();
    b.emit(Op.OpCooperativeMatrixLoadKHR, [tCoopA, coopAMat, ptrTileAStart, constRowMajor, constCoopK]);

    const ptrTileBStart = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF16, ptrTileBStart, tileB, const0u]);
    coopBMat = b.id();
    b.emit(Op.OpCooperativeMatrixLoadKHR, [tCoopB, coopBMat, ptrTileBStart, constRowMajor, constCoopN]);
  }

  // ── MulAdd: C += A * B ──────────────────────────────────────────────
  const prevAcc = b.id();
  b.emit(Op.Load, [tCoopCAcc, prevAcc, varAcc]);
  const newAcc = b.id();
  b.emit(Op.OpCooperativeMatrixMulAddKHR, [tCoopCAcc, newAcc, coopAMat, coopBMat, prevAcc]);
  b.emit(Op.Store, [varAcc, newAcc]);

  if (!useDirectF16Load) {
    // Barrier before next tile load (shared-memory path only)
    b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);
  }

  // ── Loop continue: k += coopK ──────────────────────────────────────
  b.emit(Op.Branch, [labelLoopContinue]);
  b.emit(Op.Label, [labelLoopContinue]);

  const kNext = b.id();
  b.emit(Op.IAdd, [tU32, kNext, kVal, constCoopK]);
  b.emit(Op.Store, [varK, kNext]);
  b.emit(Op.Branch, [labelLoopHeader]);

  // ── Loop end: store result to global C ──────────────────────────────
  b.emit(Op.Label, [labelLoopEnd]);

  // Store accumulator to global output
  // Output address = batchOffset + globalRowBase * N + globalColBase
  const rowTimesN = b.id();
  b.emit(Op.IMul, [tU32, rowTimesN, globalRowBase, N]);
  let outBase = b.id();
  b.emit(Op.IAdd, [tU32, outBase, rowTimesN, globalColBase]);

  if (batched) {
    const batchOffsetC = b.id();
    b.emit(Op.IMul, [tU32, batchOffsetC, batchIdx!, MN]);
    const outWithBatch = b.id();
    b.emit(Op.IAdd, [tU32, outWithBatch, outBase, batchOffsetC]);
    outBase = outWithBatch;
  }

  // Pointer to C[outBase] in global memory
  const ptrCOut = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrF32, ptrCOut, bufC.varId, const0u, outBase]);

  // Store cooperative matrix to global memory
  // OpCooperativeMatrixStoreKHR: pointer, object, memoryLayout, stride
  let finalAcc = b.id();
  b.emit(Op.Load, [tCoopCAcc, finalAcc, varAcc]);
  if (accumF16) {
    const finalAccF32 = b.id();
    b.emit(Op.FConvert, [tCoopCOut, finalAccF32, finalAcc]);
    finalAcc = finalAccF32;
  }
  b.emit(Op.OpCooperativeMatrixStoreKHR, [ptrCOut, finalAcc, constRowMajor, N]);

  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
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
  kVal: number,
  globalRowBase: number,
  M: number, K: number,
  tileA: number,
  const0u: number, coopK: number,
  batchOffsetA: number | undefined, batched: boolean,
  transposedA: boolean,
): void {
  const constCoopK = b.id();
  b.constant(tU32, constCoopK, coopK);

  // localRow = elemOffset / coopK, localCol = elemOffset % coopK
  const localRow = b.id();
  b.emit(Op.UDiv, [tU32, localRow, elemOffset, constCoopK]);
  const localCol = b.id();
  b.emit(Op.UMod, [tU32, localCol, elemOffset, constCoopK]);

  // globalRow = globalRowBase + localRow
  const globalRow = b.id();
  b.emit(Op.IAdd, [tU32, globalRow, globalRowBase, localRow]);

  // globalCol = kVal + localCol
  const globalCol = b.id();
  b.emit(Op.IAdd, [tU32, globalCol, kVal, localCol]);

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
  b.emit(Op.AccessChain, [tPtrSharedF16, ptrShared, tileA, elemOffset]);
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
  kVal: number,
  globalColBase: number,
  N: number, K: number,
  tileB: number,
  const0u: number, coopN: number,
  batchOffsetB: number | undefined, batched: boolean,
  transposed: boolean,
): void {
  const constCoopN = b.id();
  b.constant(tU32, constCoopN, coopN);

  // localRow = elemOffset / coopN, localCol = elemOffset % coopN
  const localRow = b.id();
  b.emit(Op.UDiv, [tU32, localRow, elemOffset, constCoopN]);
  const localCol = b.id();
  b.emit(Op.UMod, [tU32, localCol, elemOffset, constCoopN]);

  // kRow = kVal + localRow (row within K dimension)
  const kRow = b.id();
  b.emit(Op.IAdd, [tU32, kRow, kVal, localRow]);

  // nCol = globalColBase + localCol
  const nCol = b.id();
  b.emit(Op.IAdd, [tU32, nCol, globalColBase, localCol]);

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
  b.emit(Op.AccessChain, [tPtrSharedF16, ptrShared, tileB, elemOffset]);
  b.emit(Op.Store, [ptrShared, valF16]);
}

// ── Exported kernel generators ──────────────────────────────────────────────

export function kernelCoopMatmulBasic(
  coopM: number,
  coopN: number,
  coopK: number,
  inputF16 = false,
  accumF16 = false,
  subgroupTilesX = 1,
  subgroupTilesY = 1,
): Uint32Array {
  return buildCoopMatmul(coopM, coopN, coopK, false, false, false, inputF16, accumF16, subgroupTilesX, subgroupTilesY);
}

export function kernelCoopMatmulBatched(
  coopM: number,
  coopN: number,
  coopK: number,
  inputF16 = false,
  accumF16 = false,
  subgroupTilesX = 1,
  subgroupTilesY = 1,
): Uint32Array {
  return buildCoopMatmul(coopM, coopN, coopK, true, false, false, inputF16, accumF16, subgroupTilesX, subgroupTilesY);
}

export function kernelCoopMatmulTransposed(
  coopM: number,
  coopN: number,
  coopK: number,
  inputF16 = false,
  accumF16 = false,
  subgroupTilesX = 1,
  subgroupTilesY = 1,
): Uint32Array {
  return buildCoopMatmul(coopM, coopN, coopK, false, true, false, inputF16, accumF16, subgroupTilesX, subgroupTilesY);
}

export function kernelCoopMatmulTransposedBatched(
  coopM: number,
  coopN: number,
  coopK: number,
  inputF16 = false,
  accumF16 = false,
  subgroupTilesX = 1,
  subgroupTilesY = 1,
): Uint32Array {
  return buildCoopMatmul(coopM, coopN, coopK, true, true, false, inputF16, accumF16, subgroupTilesX, subgroupTilesY);
}

export function kernelCoopMatmulTransposedA(
  coopM: number,
  coopN: number,
  coopK: number,
  inputF16 = false,
  accumF16 = false,
  subgroupTilesX = 1,
  subgroupTilesY = 1,
): Uint32Array {
  return buildCoopMatmul(coopM, coopN, coopK, false, false, true, inputF16, accumF16, subgroupTilesX, subgroupTilesY);
}

export function kernelCoopMatmulTransposedABatched(
  coopM: number,
  coopN: number,
  coopK: number,
  inputF16 = false,
  accumF16 = false,
  subgroupTilesX = 1,
  subgroupTilesY = 1,
): Uint32Array {
  return buildCoopMatmul(coopM, coopN, coopK, true, false, true, inputF16, accumF16, subgroupTilesX, subgroupTilesY);
}
