/**
 * kernels/matmul.ts — Tiled matrix multiplication GPU kernels.
 *
 * Shared-memory tiled matmul with 16x16 workgroups.
 * Variants: basic, batched, transposed, transposed+batched.
 */

import {
  SpirVBuilder, Op, ExecutionModel, ExecutionMode, StorageClass, Decoration,
  BuiltIn, FunctionControl, Scope, MemorySemantics,
  preamble, declareStorageBuffer, declareParamsPushConstant,
  loadPushLen, loadPushScalar,
} from "./helpers.js";

// ── Kernel: Tiled Matrix Multiply (shared memory) ───────────────────────────

/**
 * C = A @ B  (M×K × K×N → M×N)
 * Uses shared memory tiling for cache efficiency.
 *
 * Push constants: { M: f32, N: f32 }  (K is derived from buffer sizes or passed separately)
 * Actually, we need M, N, K. Let's use: push constants = { M_f32, N_f32 }
 * and pass K as a separate push constant. We have 8 bytes... not enough for 3 values.
 * Let's expand push constants to 16 bytes for matmul: { M, N, K, _pad }
 *
 * Bindings: 0=A(in, M×K), 1=B(in, K×N), 2=C(out, M×N)
 * Push constants: { M: f32, N: f32, K: f32, _pad: f32 }
 * Dispatch: (ceil(N/TILE), ceil(M/TILE), 1) workgroups
 * Each workgroup computes a TILE×TILE block of output.
 */
const DEFAULT_TILE = 16;

export function kernelMatmul(wgSize = DEFAULT_TILE * DEFAULT_TILE, tileSize = DEFAULT_TILE): Uint32Array {
  const TILE_SIZE = tileSize;
  const b = new SpirVBuilder();
  // workgroup is 2D: TILE_SIZE × TILE_SIZE
  const p = preamble(b, TILE_SIZE, TILE_SIZE, 1);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufB = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, true);
  const bufC = declareStorageBuffer(b, p.tF32, p.tU32, 0, 2, false);
  // 4 push constant floats = 16 bytes: {M, N, K, _pad}
  const pc = declareParamsPushConstant(b, p.tF32, 4);

  // Shared memory: 2 tiles of TILE_SIZE × TILE_SIZE floats
  const constTileSize = b.id();
  b.constant(p.tU32, constTileSize, TILE_SIZE);
  const constTileSizeSq = b.id();
  b.constant(p.tU32, constTileSizeSq, TILE_SIZE * TILE_SIZE);
  const tArrayTile = b.id();
  b.typeArray(tArrayTile, p.tF32, constTileSizeSq);
  const tPtrSharedArr = b.id();
  b.typePointer(tPtrSharedArr, StorageClass.Workgroup, tArrayTile);
  const tPtrSharedF32 = b.id();
  b.typePointer(tPtrSharedF32, StorageClass.Workgroup, p.tF32);
  const tileA = b.id();
  b.variable(tPtrSharedArr, tileA, StorageClass.Workgroup);
  const tileB = b.id();
  b.variable(tPtrSharedArr, tileB, StorageClass.Workgroup);

  // Built-in variables
  const tPtrInputVec3 = b.id();
  b.typePointer(tPtrInputVec3, StorageClass.Input, p.tVec3U32);
  const vWorkgroupId = b.id();
  b.variable(tPtrInputVec3, vWorkgroupId, StorageClass.Input);
  b.addDecorate(vWorkgroupId, Decoration.BuiltIn, BuiltIn.WorkgroupId);
  const vLocalId = b.id();
  b.variable(tPtrInputVec3, vLocalId, StorageClass.Input);
  b.addDecorate(vLocalId, Decoration.BuiltIn, BuiltIn.LocalInvocationId);

  const scopeWg = b.id();
  b.constant(p.tU32, scopeWg, Scope.Workgroup);
  const semAcqRelWg = b.id();
  b.constant(p.tU32, semAcqRelWg, MemorySemantics.AcquireRelease | MemorySemantics.WorkgroupMemory);

  const tPtrFnU32 = b.id();
  b.typePointer(tPtrFnU32, StorageClass.Function, p.tU32);
  const tPtrFnF32 = b.id();
  b.typePointer(tPtrFnF32, StorageClass.Function, p.tF32);

  // Push constant accessors for members 2 and 3
  const const3u = b.id();
  b.constant(p.tU32, const3u, 3);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId, vWorkgroupId, vLocalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, TILE_SIZE, TILE_SIZE, 1);

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  const labelEntry = b.id();
  b.emit(Op.Label, [labelEntry]);

  const varT = b.id();
  b.emit(Op.Variable, [tPtrFnU32, varT, StorageClass.Function]);
  const varAcc = b.id();
  b.emit(Op.Variable, [tPtrFnF32, varAcc, StorageClass.Function]);

  // Local thread coords
  const lidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, lidVec, vLocalId]);
  const tx = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, tx, lidVec, 0]); // column within tile
  const ty = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, ty, lidVec, 1]); // row within tile

  // Workgroup coords
  const wgIdVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, wgIdVec, vWorkgroupId]);
  const bx = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, bx, wgIdVec, 0]); // tile column
  const by = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, by, wgIdVec, 1]); // tile row

  // Load M, N, K from push constants
  const MF = loadPushLen(b, p, pc);       // member 0 = M
  const NF = loadPushScalar(b, p, pc);    // member 1 = N
  // member 2 = K
  const ptrK = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrK, pc.varId, p.const2u]);
  const KF = b.id();
  b.emit(Op.Load, [p.tF32, KF, ptrK]);

  const M = b.id(); b.emit(Op.ConvertFToU, [p.tU32, M, MF]);
  const N = b.id(); b.emit(Op.ConvertFToU, [p.tU32, N, NF]);
  const K = b.id(); b.emit(Op.ConvertFToU, [p.tU32, K, KF]);

  // Global output row/col
  const globalRow = b.id();
  const byTimesT = b.id();
  b.emit(Op.IMul, [p.tU32, byTimesT, by, constTileSize]);
  b.emit(Op.IAdd, [p.tU32, globalRow, byTimesT, ty]);
  const globalCol = b.id();
  const bxTimesT = b.id();
  b.emit(Op.IMul, [p.tU32, bxTimesT, bx, constTileSize]);
  b.emit(Op.IAdd, [p.tU32, globalCol, bxTimesT, tx]);

  // acc = 0.0
  b.emit(Op.Store, [varAcc, p.const0f]);

  // Tile index within shared memory: ty * TILE_SIZE + tx
  const localTileIdx = b.id();
  const tyTimesT = b.id();
  b.emit(Op.IMul, [p.tU32, tyTimesT, ty, constTileSize]);
  b.emit(Op.IAdd, [p.tU32, localTileIdx, tyTimesT, tx]);

  // Number of tiles along K dimension
  // Loop: for (t = 0; t < K; t += TILE_SIZE)
  b.emit(Op.Store, [varT, p.const0u]);

  const labelHead = b.id();
  const labelBody = b.id();
  const labelMerge = b.id();
  const labelCont = b.id();

  b.emit(Op.Branch, [labelHead]);
  b.emit(Op.Label, [labelHead]);
  const t = b.id();
  b.emit(Op.Load, [p.tU32, t, varT]);
  const cmp = b.id();
  b.emit(Op.ULessThan, [p.tBool, cmp, t, K]);
  b.emit(Op.LoopMerge, [labelMerge, labelCont, 0]);
  b.emit(Op.BranchConditional, [cmp, labelBody, labelMerge]);

  b.emit(Op.Label, [labelBody]);

  // Load tile of A: A[globalRow, t + tx]
  // row check: globalRow < M, col check: (t + tx) < K
  const aCol = b.id();
  b.emit(Op.IAdd, [p.tU32, aCol, t, tx]);
  const aInBoundsR = b.id();
  b.emit(Op.ULessThan, [p.tBool, aInBoundsR, globalRow, M]);
  const aInBoundsC = b.id();
  b.emit(Op.ULessThan, [p.tBool, aInBoundsC, aCol, K]);
  const aInBounds = b.id();
  b.emit(Op.LogicalAnd, [p.tBool, aInBounds, aInBoundsR, aInBoundsC]);
  // A[globalRow * K + aCol] or 0
  const aLinear = b.id();
  const grTimesK = b.id();
  b.emit(Op.IMul, [p.tU32, grTimesK, globalRow, K]);
  b.emit(Op.IAdd, [p.tU32, aLinear, grTimesK, aCol]);
  const ptrAElem = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrF32, ptrAElem, bufA.varId, p.const0u, aLinear]);
  const aRaw = b.id();
  b.emit(Op.Load, [p.tF32, aRaw, ptrAElem]);
  const aVal = b.id();
  b.emit(Op.Select, [p.tF32, aVal, aInBounds, aRaw, p.const0f]);
  // Store to tileA[ty * TILE_SIZE + tx]
  const ptrTileA = b.id();
  b.emit(Op.AccessChain, [tPtrSharedF32, ptrTileA, tileA, localTileIdx]);
  b.emit(Op.Store, [ptrTileA, aVal]);

  // Load tile of B: B[t + ty, globalCol]
  const bRow = b.id();
  b.emit(Op.IAdd, [p.tU32, bRow, t, ty]);
  const bInBoundsR = b.id();
  b.emit(Op.ULessThan, [p.tBool, bInBoundsR, bRow, K]);
  const bInBoundsC = b.id();
  b.emit(Op.ULessThan, [p.tBool, bInBoundsC, globalCol, N]);
  const bInBounds = b.id();
  b.emit(Op.LogicalAnd, [p.tBool, bInBounds, bInBoundsR, bInBoundsC]);
  const bLinear = b.id();
  const brTimesN = b.id();
  b.emit(Op.IMul, [p.tU32, brTimesN, bRow, N]);
  b.emit(Op.IAdd, [p.tU32, bLinear, brTimesN, globalCol]);
  const ptrBElem = b.id();
  b.emit(Op.AccessChain, [bufB.tPtrF32, ptrBElem, bufB.varId, p.const0u, bLinear]);
  const bRaw = b.id();
  b.emit(Op.Load, [p.tF32, bRaw, ptrBElem]);
  const bVal = b.id();
  b.emit(Op.Select, [p.tF32, bVal, bInBounds, bRaw, p.const0f]);
  const ptrTileB = b.id();
  b.emit(Op.AccessChain, [tPtrSharedF32, ptrTileB, tileB, localTileIdx]);
  b.emit(Op.Store, [ptrTileB, bVal]);

  // Barrier — all threads have loaded their tile elements
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  // Accumulate: for k = 0..TILE_SIZE-1: acc += tileA[ty][k] * tileB[k][tx]
  for (let k = 0; k < TILE_SIZE; k++) {
    const kConst = b.id();
    b.constant(p.tU32, kConst, k);
    // tileA[ty * TILE_SIZE + k]
    const aIdx = b.id();
    const tyT = b.id();
    b.emit(Op.IMul, [p.tU32, tyT, ty, constTileSize]);
    b.emit(Op.IAdd, [p.tU32, aIdx, tyT, kConst]);
    const pA = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF32, pA, tileA, aIdx]);
    const aV = b.id();
    b.emit(Op.Load, [p.tF32, aV, pA]);
    // tileB[k * TILE_SIZE + tx]
    const bIdx = b.id();
    const kT = b.id();
    b.emit(Op.IMul, [p.tU32, kT, kConst, constTileSize]);
    b.emit(Op.IAdd, [p.tU32, bIdx, kT, tx]);
    const pB = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF32, pB, tileB, bIdx]);
    const bV = b.id();
    b.emit(Op.Load, [p.tF32, bV, pB]);
    // acc += aV * bV
    const curAcc = b.id();
    b.emit(Op.Load, [p.tF32, curAcc, varAcc]);
    const prod = b.id();
    b.emit(Op.FMul, [p.tF32, prod, aV, bV]);
    const newAcc = b.id();
    b.emit(Op.FAdd, [p.tF32, newAcc, curAcc, prod]);
    b.emit(Op.Store, [varAcc, newAcc]);
  }

  // Barrier before next tile load
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  b.emit(Op.Branch, [labelCont]);
  b.emit(Op.Label, [labelCont]);
  const nextT = b.id();
  b.emit(Op.Load, [p.tU32, nextT, varT]);
  const incT = b.id();
  b.emit(Op.IAdd, [p.tU32, incT, nextT, constTileSize]);
  b.emit(Op.Store, [varT, incT]);
  b.emit(Op.Branch, [labelHead]);

  b.emit(Op.Label, [labelMerge]);

  // Write output: C[globalRow * N + globalCol] = acc (if in bounds)
  const outInBoundsR = b.id();
  b.emit(Op.ULessThan, [p.tBool, outInBoundsR, globalRow, M]);
  const outInBoundsC = b.id();
  b.emit(Op.ULessThan, [p.tBool, outInBoundsC, globalCol, N]);
  const outInBounds = b.id();
  b.emit(Op.LogicalAnd, [p.tBool, outInBounds, outInBoundsR, outInBoundsC]);
  const labelWrite = b.id();
  const labelEnd = b.id();
  b.emit(Op.SelectionMerge, [labelEnd, 0]);
  b.emit(Op.BranchConditional, [outInBounds, labelWrite, labelEnd]);

  b.emit(Op.Label, [labelWrite]);
  const outLinear = b.id();
  const grTimesN = b.id();
  b.emit(Op.IMul, [p.tU32, grTimesN, globalRow, N]);
  b.emit(Op.IAdd, [p.tU32, outLinear, grTimesN, globalCol]);
  const ptrOut = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrF32, ptrOut, bufC.varId, p.const0u, outLinear]);
  const finalAcc = b.id();
  b.emit(Op.Load, [p.tF32, finalAcc, varAcc]);
  b.emit(Op.Store, [ptrOut, finalAcc]);
  b.emit(Op.Branch, [labelEnd]);

  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

/**
 * Batched tiled matmul: C[b] = A[b] × B[b] for each batch b.
 *
 * Same tiling strategy as kernelMatmul, but uses WorkgroupId.z as batch index.
 * Each batch element is a contiguous M×K / K×N / M×N slice in the flat buffers.
 *
 * Push constants: { M: f32, N: f32, K: f32, _pad: f32 } — 16 bytes
 * Bindings: 0=A(in), 1=B(in), 2=C(out)
 * Dispatch: (ceil(N/TILE), ceil(M/TILE), batchCount)
 */
export function kernelMatmulBatched(wgSize = DEFAULT_TILE * DEFAULT_TILE, tileSize = DEFAULT_TILE): Uint32Array {
  const TILE_SIZE = tileSize;
  const b = new SpirVBuilder();
  const p = preamble(b, TILE_SIZE, TILE_SIZE, 1);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufB = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, true);
  const bufC = declareStorageBuffer(b, p.tF32, p.tU32, 0, 2, false);
  const pc = declareParamsPushConstant(b, p.tF32, 4);

  // Shared memory tiles
  const constTileSize = b.id();
  b.constant(p.tU32, constTileSize, TILE_SIZE);
  const constTileSizeSq = b.id();
  b.constant(p.tU32, constTileSizeSq, TILE_SIZE * TILE_SIZE);
  const tArrayTile = b.id();
  b.typeArray(tArrayTile, p.tF32, constTileSizeSq);
  const tPtrSharedArr = b.id();
  b.typePointer(tPtrSharedArr, StorageClass.Workgroup, tArrayTile);
  const tPtrSharedF32 = b.id();
  b.typePointer(tPtrSharedF32, StorageClass.Workgroup, p.tF32);
  const tileA = b.id();
  b.variable(tPtrSharedArr, tileA, StorageClass.Workgroup);
  const tileB = b.id();
  b.variable(tPtrSharedArr, tileB, StorageClass.Workgroup);

  // Built-in variables
  const tPtrInputVec3 = b.id();
  b.typePointer(tPtrInputVec3, StorageClass.Input, p.tVec3U32);
  const vWorkgroupId = b.id();
  b.variable(tPtrInputVec3, vWorkgroupId, StorageClass.Input);
  b.addDecorate(vWorkgroupId, Decoration.BuiltIn, BuiltIn.WorkgroupId);
  const vLocalId = b.id();
  b.variable(tPtrInputVec3, vLocalId, StorageClass.Input);
  b.addDecorate(vLocalId, Decoration.BuiltIn, BuiltIn.LocalInvocationId);

  const scopeWg = b.id();
  b.constant(p.tU32, scopeWg, Scope.Workgroup);
  const semAcqRelWg = b.id();
  b.constant(p.tU32, semAcqRelWg, MemorySemantics.AcquireRelease | MemorySemantics.WorkgroupMemory);

  const tPtrFnU32 = b.id();
  b.typePointer(tPtrFnU32, StorageClass.Function, p.tU32);
  const tPtrFnF32 = b.id();
  b.typePointer(tPtrFnF32, StorageClass.Function, p.tF32);

  const const3u = b.id();
  b.constant(p.tU32, const3u, 3);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId, vWorkgroupId, vLocalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, TILE_SIZE, TILE_SIZE, 1);

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  const labelEntry = b.id();
  b.emit(Op.Label, [labelEntry]);

  const varT = b.id();
  b.emit(Op.Variable, [tPtrFnU32, varT, StorageClass.Function]);
  const varAcc = b.id();
  b.emit(Op.Variable, [tPtrFnF32, varAcc, StorageClass.Function]);

  // Local thread coords
  const lidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, lidVec, vLocalId]);
  const tx = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, tx, lidVec, 0]);
  const ty = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, ty, lidVec, 1]);

  // Workgroup coords — x=tile col, y=tile row, z=batch index
  const wgIdVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, wgIdVec, vWorkgroupId]);
  const bx = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, bx, wgIdVec, 0]);
  const by = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, by, wgIdVec, 1]);
  const batchIdx = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, batchIdx, wgIdVec, 2]);

  // Load M, N, K from push constants
  const MF = loadPushLen(b, p, pc);
  const NF = loadPushScalar(b, p, pc);
  const ptrK = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrK, pc.varId, p.const2u]);
  const KF = b.id();
  b.emit(Op.Load, [p.tF32, KF, ptrK]);

  const M = b.id(); b.emit(Op.ConvertFToU, [p.tU32, M, MF]);
  const N = b.id(); b.emit(Op.ConvertFToU, [p.tU32, N, NF]);
  const K = b.id(); b.emit(Op.ConvertFToU, [p.tU32, K, KF]);

  // Batch offsets: A_off = batchIdx * M * K, B_off = batchIdx * K * N, C_off = batchIdx * M * N
  const MK = b.id(); b.emit(Op.IMul, [p.tU32, MK, M, K]);
  const KN = b.id(); b.emit(Op.IMul, [p.tU32, KN, K, N]);
  const MN = b.id(); b.emit(Op.IMul, [p.tU32, MN, M, N]);
  const aOff = b.id(); b.emit(Op.IMul, [p.tU32, aOff, batchIdx, MK]);
  const bOff = b.id(); b.emit(Op.IMul, [p.tU32, bOff, batchIdx, KN]);
  const cOff = b.id(); b.emit(Op.IMul, [p.tU32, cOff, batchIdx, MN]);

  // Global output row/col
  const globalRow = b.id();
  const byTimesT = b.id();
  b.emit(Op.IMul, [p.tU32, byTimesT, by, constTileSize]);
  b.emit(Op.IAdd, [p.tU32, globalRow, byTimesT, ty]);
  const globalCol = b.id();
  const bxTimesT = b.id();
  b.emit(Op.IMul, [p.tU32, bxTimesT, bx, constTileSize]);
  b.emit(Op.IAdd, [p.tU32, globalCol, bxTimesT, tx]);

  // acc = 0.0
  b.emit(Op.Store, [varAcc, p.const0f]);

  // Tile index within shared memory
  const localTileIdx = b.id();
  const tyTimesT = b.id();
  b.emit(Op.IMul, [p.tU32, tyTimesT, ty, constTileSize]);
  b.emit(Op.IAdd, [p.tU32, localTileIdx, tyTimesT, tx]);

  // Loop: for (t = 0; t < K; t += TILE_SIZE)
  b.emit(Op.Store, [varT, p.const0u]);

  const labelHead = b.id();
  const labelBody = b.id();
  const labelMerge = b.id();
  const labelCont = b.id();

  b.emit(Op.Branch, [labelHead]);
  b.emit(Op.Label, [labelHead]);
  const t = b.id();
  b.emit(Op.Load, [p.tU32, t, varT]);
  const cmp = b.id();
  b.emit(Op.ULessThan, [p.tBool, cmp, t, K]);
  b.emit(Op.LoopMerge, [labelMerge, labelCont, 0]);
  b.emit(Op.BranchConditional, [cmp, labelBody, labelMerge]);

  b.emit(Op.Label, [labelBody]);

  // Load tile of A: A[aOff + globalRow * K + t + tx]
  const aCol = b.id();
  b.emit(Op.IAdd, [p.tU32, aCol, t, tx]);
  const aInBoundsR = b.id();
  b.emit(Op.ULessThan, [p.tBool, aInBoundsR, globalRow, M]);
  const aInBoundsC = b.id();
  b.emit(Op.ULessThan, [p.tBool, aInBoundsC, aCol, K]);
  const aInBounds = b.id();
  b.emit(Op.LogicalAnd, [p.tBool, aInBounds, aInBoundsR, aInBoundsC]);
  const aLinear = b.id();
  const grTimesK = b.id();
  b.emit(Op.IMul, [p.tU32, grTimesK, globalRow, K]);
  b.emit(Op.IAdd, [p.tU32, aLinear, grTimesK, aCol]);
  const aIdx = b.id();
  b.emit(Op.IAdd, [p.tU32, aIdx, aOff, aLinear]);  // + batch offset
  const ptrAElem = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrF32, ptrAElem, bufA.varId, p.const0u, aIdx]);
  const aRaw = b.id();
  b.emit(Op.Load, [p.tF32, aRaw, ptrAElem]);
  const aVal = b.id();
  b.emit(Op.Select, [p.tF32, aVal, aInBounds, aRaw, p.const0f]);
  const ptrTileA = b.id();
  b.emit(Op.AccessChain, [tPtrSharedF32, ptrTileA, tileA, localTileIdx]);
  b.emit(Op.Store, [ptrTileA, aVal]);

  // Load tile of B: B[bOff + (t + ty) * N + globalCol]
  const bRow = b.id();
  b.emit(Op.IAdd, [p.tU32, bRow, t, ty]);
  const bInBoundsR = b.id();
  b.emit(Op.ULessThan, [p.tBool, bInBoundsR, bRow, K]);
  const bInBoundsC = b.id();
  b.emit(Op.ULessThan, [p.tBool, bInBoundsC, globalCol, N]);
  const bInBounds = b.id();
  b.emit(Op.LogicalAnd, [p.tBool, bInBounds, bInBoundsR, bInBoundsC]);
  const bLinear = b.id();
  const brTimesN = b.id();
  b.emit(Op.IMul, [p.tU32, brTimesN, bRow, N]);
  b.emit(Op.IAdd, [p.tU32, bLinear, brTimesN, globalCol]);
  const bIdx = b.id();
  b.emit(Op.IAdd, [p.tU32, bIdx, bOff, bLinear]);  // + batch offset
  const ptrBElem = b.id();
  b.emit(Op.AccessChain, [bufB.tPtrF32, ptrBElem, bufB.varId, p.const0u, bIdx]);
  const bRaw = b.id();
  b.emit(Op.Load, [p.tF32, bRaw, ptrBElem]);
  const bVal = b.id();
  b.emit(Op.Select, [p.tF32, bVal, bInBounds, bRaw, p.const0f]);
  const ptrTileB = b.id();
  b.emit(Op.AccessChain, [tPtrSharedF32, ptrTileB, tileB, localTileIdx]);
  b.emit(Op.Store, [ptrTileB, bVal]);

  // Barrier
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  // Accumulate: for k = 0..TILE_SIZE-1: acc += tileA[ty][k] * tileB[k][tx]
  for (let k = 0; k < TILE_SIZE; k++) {
    const kConst = b.id();
    b.constant(p.tU32, kConst, k);
    const aI = b.id();
    const tyT = b.id();
    b.emit(Op.IMul, [p.tU32, tyT, ty, constTileSize]);
    b.emit(Op.IAdd, [p.tU32, aI, tyT, kConst]);
    const pA = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF32, pA, tileA, aI]);
    const aV = b.id();
    b.emit(Op.Load, [p.tF32, aV, pA]);
    const bI = b.id();
    const kT = b.id();
    b.emit(Op.IMul, [p.tU32, kT, kConst, constTileSize]);
    b.emit(Op.IAdd, [p.tU32, bI, kT, tx]);
    const pB = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF32, pB, tileB, bI]);
    const bV = b.id();
    b.emit(Op.Load, [p.tF32, bV, pB]);
    const curAcc = b.id();
    b.emit(Op.Load, [p.tF32, curAcc, varAcc]);
    const prod = b.id();
    b.emit(Op.FMul, [p.tF32, prod, aV, bV]);
    const newAcc = b.id();
    b.emit(Op.FAdd, [p.tF32, newAcc, curAcc, prod]);
    b.emit(Op.Store, [varAcc, newAcc]);
  }

  // Barrier
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  b.emit(Op.Branch, [labelCont]);
  b.emit(Op.Label, [labelCont]);
  const nextT = b.id();
  b.emit(Op.Load, [p.tU32, nextT, varT]);
  const incT = b.id();
  b.emit(Op.IAdd, [p.tU32, incT, nextT, constTileSize]);
  b.emit(Op.Store, [varT, incT]);
  b.emit(Op.Branch, [labelHead]);

  b.emit(Op.Label, [labelMerge]);

  // Write output: C[cOff + globalRow * N + globalCol] = acc (if in bounds)
  const outInBoundsR = b.id();
  b.emit(Op.ULessThan, [p.tBool, outInBoundsR, globalRow, M]);
  const outInBoundsC = b.id();
  b.emit(Op.ULessThan, [p.tBool, outInBoundsC, globalCol, N]);
  const outInBounds = b.id();
  b.emit(Op.LogicalAnd, [p.tBool, outInBounds, outInBoundsR, outInBoundsC]);
  const labelWrite = b.id();
  const labelEnd = b.id();
  b.emit(Op.SelectionMerge, [labelEnd, 0]);
  b.emit(Op.BranchConditional, [outInBounds, labelWrite, labelEnd]);

  b.emit(Op.Label, [labelWrite]);
  const outLinear = b.id();
  const grTimesN = b.id();
  b.emit(Op.IMul, [p.tU32, grTimesN, globalRow, N]);
  b.emit(Op.IAdd, [p.tU32, outLinear, grTimesN, globalCol]);
  const outIdx = b.id();
  b.emit(Op.IAdd, [p.tU32, outIdx, cOff, outLinear]);  // + batch offset
  const ptrOut = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrF32, ptrOut, bufC.varId, p.const0u, outIdx]);
  const finalAcc = b.id();
  b.emit(Op.Load, [p.tF32, finalAcc, varAcc]);
  b.emit(Op.Store, [ptrOut, finalAcc]);
  b.emit(Op.Branch, [labelEnd]);

  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

// ── Kernel: matmul_transposed ────────────────────────────────────────────────

/**
 * Tiled matmul with B transposed: C = A × B^T
 *
 * A is [M, K], B is stored as [N, K] (row-major), used as B^T = [K, N].
 * Result C is [M, N].
 *
 * Same tiling strategy as kernelMatmul. The only difference is how B is loaded
 * into shared memory tiles: instead of B[bRow * N + globalCol], we read
 * B[globalCol * K + bRow] to effect the transpose.
 *
 * Push constants: { M: f32, N: f32, K: f32, _pad: f32 } — 16 bytes
 * Bindings: 0=A(in), 1=B(in), 2=C(out)
 * Dispatch: (ceil(N/TILE), ceil(M/TILE), 1)
 */
export function kernelMatmulTransposed(wgSize = DEFAULT_TILE * DEFAULT_TILE, tileSize = DEFAULT_TILE): Uint32Array {
  const TILE_SIZE = tileSize;
  const b = new SpirVBuilder();
  const p = preamble(b, TILE_SIZE, TILE_SIZE, 1);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufB = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, true);
  const bufC = declareStorageBuffer(b, p.tF32, p.tU32, 0, 2, false);
  const pc = declareParamsPushConstant(b, p.tF32, 4);

  // Shared memory: 2 tiles of TILE_SIZE × TILE_SIZE floats
  const constTileSize = b.id();
  b.constant(p.tU32, constTileSize, TILE_SIZE);
  const constTileSizeSq = b.id();
  b.constant(p.tU32, constTileSizeSq, TILE_SIZE * TILE_SIZE);
  const tArrayTile = b.id();
  b.typeArray(tArrayTile, p.tF32, constTileSizeSq);
  const tPtrSharedArr = b.id();
  b.typePointer(tPtrSharedArr, StorageClass.Workgroup, tArrayTile);
  const tPtrSharedF32 = b.id();
  b.typePointer(tPtrSharedF32, StorageClass.Workgroup, p.tF32);
  const tileA = b.id();
  b.variable(tPtrSharedArr, tileA, StorageClass.Workgroup);
  const tileB = b.id();
  b.variable(tPtrSharedArr, tileB, StorageClass.Workgroup);

  // Built-in variables
  const tPtrInputVec3 = b.id();
  b.typePointer(tPtrInputVec3, StorageClass.Input, p.tVec3U32);
  const vWorkgroupId = b.id();
  b.variable(tPtrInputVec3, vWorkgroupId, StorageClass.Input);
  b.addDecorate(vWorkgroupId, Decoration.BuiltIn, BuiltIn.WorkgroupId);
  const vLocalId = b.id();
  b.variable(tPtrInputVec3, vLocalId, StorageClass.Input);
  b.addDecorate(vLocalId, Decoration.BuiltIn, BuiltIn.LocalInvocationId);

  const scopeWg = b.id();
  b.constant(p.tU32, scopeWg, Scope.Workgroup);
  const semAcqRelWg = b.id();
  b.constant(p.tU32, semAcqRelWg, MemorySemantics.AcquireRelease | MemorySemantics.WorkgroupMemory);

  const tPtrFnU32 = b.id();
  b.typePointer(tPtrFnU32, StorageClass.Function, p.tU32);
  const tPtrFnF32 = b.id();
  b.typePointer(tPtrFnF32, StorageClass.Function, p.tF32);

  const const3u = b.id();
  b.constant(p.tU32, const3u, 3);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId, vWorkgroupId, vLocalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, TILE_SIZE, TILE_SIZE, 1);

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  const labelEntry = b.id();
  b.emit(Op.Label, [labelEntry]);

  const varT = b.id();
  b.emit(Op.Variable, [tPtrFnU32, varT, StorageClass.Function]);
  const varAcc = b.id();
  b.emit(Op.Variable, [tPtrFnF32, varAcc, StorageClass.Function]);

  // Local thread coords
  const lidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, lidVec, vLocalId]);
  const tx = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, tx, lidVec, 0]);
  const ty = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, ty, lidVec, 1]);

  // Workgroup coords
  const wgIdVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, wgIdVec, vWorkgroupId]);
  const bx = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, bx, wgIdVec, 0]);
  const by = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, by, wgIdVec, 1]);

  // Load M, N, K from push constants
  const MF = loadPushLen(b, p, pc);
  const NF = loadPushScalar(b, p, pc);
  const ptrK = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrK, pc.varId, p.const2u]);
  const KF = b.id();
  b.emit(Op.Load, [p.tF32, KF, ptrK]);

  const M = b.id(); b.emit(Op.ConvertFToU, [p.tU32, M, MF]);
  const N = b.id(); b.emit(Op.ConvertFToU, [p.tU32, N, NF]);
  const K = b.id(); b.emit(Op.ConvertFToU, [p.tU32, K, KF]);

  // Global output row/col
  const globalRow = b.id();
  const byTimesT = b.id();
  b.emit(Op.IMul, [p.tU32, byTimesT, by, constTileSize]);
  b.emit(Op.IAdd, [p.tU32, globalRow, byTimesT, ty]);
  const globalCol = b.id();
  const bxTimesT = b.id();
  b.emit(Op.IMul, [p.tU32, bxTimesT, bx, constTileSize]);
  b.emit(Op.IAdd, [p.tU32, globalCol, bxTimesT, tx]);

  // acc = 0.0
  b.emit(Op.Store, [varAcc, p.const0f]);

  // Tile index within shared memory: ty * TILE_SIZE + tx
  const localTileIdx = b.id();
  const tyTimesT = b.id();
  b.emit(Op.IMul, [p.tU32, tyTimesT, ty, constTileSize]);
  b.emit(Op.IAdd, [p.tU32, localTileIdx, tyTimesT, tx]);

  // Loop: for (t = 0; t < K; t += TILE_SIZE)
  b.emit(Op.Store, [varT, p.const0u]);

  const labelHead = b.id();
  const labelBody = b.id();
  const labelMerge = b.id();
  const labelCont = b.id();

  b.emit(Op.Branch, [labelHead]);
  b.emit(Op.Label, [labelHead]);
  const t = b.id();
  b.emit(Op.Load, [p.tU32, t, varT]);
  const cmp = b.id();
  b.emit(Op.ULessThan, [p.tBool, cmp, t, K]);
  b.emit(Op.LoopMerge, [labelMerge, labelCont, 0]);
  b.emit(Op.BranchConditional, [cmp, labelBody, labelMerge]);

  b.emit(Op.Label, [labelBody]);

  // Load tile of A: A[globalRow, t + tx]
  const aCol = b.id();
  b.emit(Op.IAdd, [p.tU32, aCol, t, tx]);
  const aInBoundsR = b.id();
  b.emit(Op.ULessThan, [p.tBool, aInBoundsR, globalRow, M]);
  const aInBoundsC = b.id();
  b.emit(Op.ULessThan, [p.tBool, aInBoundsC, aCol, K]);
  const aInBounds = b.id();
  b.emit(Op.LogicalAnd, [p.tBool, aInBounds, aInBoundsR, aInBoundsC]);
  const aLinear = b.id();
  const grTimesK = b.id();
  b.emit(Op.IMul, [p.tU32, grTimesK, globalRow, K]);
  b.emit(Op.IAdd, [p.tU32, aLinear, grTimesK, aCol]);
  const ptrAElem = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrF32, ptrAElem, bufA.varId, p.const0u, aLinear]);
  const aRaw = b.id();
  b.emit(Op.Load, [p.tF32, aRaw, ptrAElem]);
  const aVal = b.id();
  b.emit(Op.Select, [p.tF32, aVal, aInBounds, aRaw, p.const0f]);
  const ptrTileA = b.id();
  b.emit(Op.AccessChain, [tPtrSharedF32, ptrTileA, tileA, localTileIdx]);
  b.emit(Op.Store, [ptrTileA, aVal]);

  // Load tile of B transposed: B^T[t + ty, globalCol] = B[globalCol, t + ty]
  // B is stored as [N, K], so B[globalCol, t+ty] = B[globalCol * K + (t + ty)]
  const bRow = b.id();
  b.emit(Op.IAdd, [p.tU32, bRow, t, ty]);
  const bInBoundsR = b.id();
  b.emit(Op.ULessThan, [p.tBool, bInBoundsR, bRow, K]);
  const bInBoundsC = b.id();
  b.emit(Op.ULessThan, [p.tBool, bInBoundsC, globalCol, N]);
  const bInBounds = b.id();
  b.emit(Op.LogicalAnd, [p.tBool, bInBounds, bInBoundsR, bInBoundsC]);
  // B[globalCol * K + bRow] (transposed access)
  const bLinear = b.id();
  const gcTimesK = b.id();
  b.emit(Op.IMul, [p.tU32, gcTimesK, globalCol, K]);
  b.emit(Op.IAdd, [p.tU32, bLinear, gcTimesK, bRow]);
  const ptrBElem = b.id();
  b.emit(Op.AccessChain, [bufB.tPtrF32, ptrBElem, bufB.varId, p.const0u, bLinear]);
  const bRaw = b.id();
  b.emit(Op.Load, [p.tF32, bRaw, ptrBElem]);
  const bVal = b.id();
  b.emit(Op.Select, [p.tF32, bVal, bInBounds, bRaw, p.const0f]);
  const ptrTileB = b.id();
  b.emit(Op.AccessChain, [tPtrSharedF32, ptrTileB, tileB, localTileIdx]);
  b.emit(Op.Store, [ptrTileB, bVal]);

  // Barrier
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  // Accumulate: for k = 0..TILE_SIZE-1: acc += tileA[ty][k] * tileB[k][tx]
  for (let k = 0; k < TILE_SIZE; k++) {
    const kConst = b.id();
    b.constant(p.tU32, kConst, k);
    const aIdx = b.id();
    const tyT = b.id();
    b.emit(Op.IMul, [p.tU32, tyT, ty, constTileSize]);
    b.emit(Op.IAdd, [p.tU32, aIdx, tyT, kConst]);
    const pA = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF32, pA, tileA, aIdx]);
    const aV = b.id();
    b.emit(Op.Load, [p.tF32, aV, pA]);
    const bIdx = b.id();
    const kT = b.id();
    b.emit(Op.IMul, [p.tU32, kT, kConst, constTileSize]);
    b.emit(Op.IAdd, [p.tU32, bIdx, kT, tx]);
    const pB = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF32, pB, tileB, bIdx]);
    const bV = b.id();
    b.emit(Op.Load, [p.tF32, bV, pB]);
    const curAcc = b.id();
    b.emit(Op.Load, [p.tF32, curAcc, varAcc]);
    const prod = b.id();
    b.emit(Op.FMul, [p.tF32, prod, aV, bV]);
    const newAcc = b.id();
    b.emit(Op.FAdd, [p.tF32, newAcc, curAcc, prod]);
    b.emit(Op.Store, [varAcc, newAcc]);
  }

  // Barrier before next tile load
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  b.emit(Op.Branch, [labelCont]);
  b.emit(Op.Label, [labelCont]);
  const nextT = b.id();
  b.emit(Op.Load, [p.tU32, nextT, varT]);
  const incT = b.id();
  b.emit(Op.IAdd, [p.tU32, incT, nextT, constTileSize]);
  b.emit(Op.Store, [varT, incT]);
  b.emit(Op.Branch, [labelHead]);

  b.emit(Op.Label, [labelMerge]);

  // Write output: C[globalRow * N + globalCol] = acc (if in bounds)
  const outInBoundsR = b.id();
  b.emit(Op.ULessThan, [p.tBool, outInBoundsR, globalRow, M]);
  const outInBoundsC = b.id();
  b.emit(Op.ULessThan, [p.tBool, outInBoundsC, globalCol, N]);
  const outInBounds = b.id();
  b.emit(Op.LogicalAnd, [p.tBool, outInBounds, outInBoundsR, outInBoundsC]);
  const labelWrite = b.id();
  const labelEnd = b.id();
  b.emit(Op.SelectionMerge, [labelEnd, 0]);
  b.emit(Op.BranchConditional, [outInBounds, labelWrite, labelEnd]);

  b.emit(Op.Label, [labelWrite]);
  const outLinear = b.id();
  const grTimesN = b.id();
  b.emit(Op.IMul, [p.tU32, grTimesN, globalRow, N]);
  b.emit(Op.IAdd, [p.tU32, outLinear, grTimesN, globalCol]);
  const ptrOut = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrF32, ptrOut, bufC.varId, p.const0u, outLinear]);
  const finalAcc = b.id();
  b.emit(Op.Load, [p.tF32, finalAcc, varAcc]);
  b.emit(Op.Store, [ptrOut, finalAcc]);
  b.emit(Op.Branch, [labelEnd]);

  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

/**
 * Batched tiled matmul with B transposed: C[b] = A[b] × B[b]^T for each batch b.
 *
 * A is [batch, M, K], B is stored as [batch, N, K], used as B^T = [batch, K, N].
 * Result C is [batch, M, N].
 *
 * Push constants: { M: f32, N: f32, K: f32, _pad: f32 } — 16 bytes
 * Bindings: 0=A(in), 1=B(in), 2=C(out)
 * Dispatch: (ceil(N/TILE), ceil(M/TILE), batchCount)
 */
export function kernelMatmulTransposedBatched(wgSize = DEFAULT_TILE * DEFAULT_TILE, tileSize = DEFAULT_TILE): Uint32Array {
  const TILE_SIZE = tileSize;
  const b = new SpirVBuilder();
  const p = preamble(b, TILE_SIZE, TILE_SIZE, 1);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufB = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, true);
  const bufC = declareStorageBuffer(b, p.tF32, p.tU32, 0, 2, false);
  const pc = declareParamsPushConstant(b, p.tF32, 4);

  // Shared memory tiles
  const constTileSize = b.id();
  b.constant(p.tU32, constTileSize, TILE_SIZE);
  const constTileSizeSq = b.id();
  b.constant(p.tU32, constTileSizeSq, TILE_SIZE * TILE_SIZE);
  const tArrayTile = b.id();
  b.typeArray(tArrayTile, p.tF32, constTileSizeSq);
  const tPtrSharedArr = b.id();
  b.typePointer(tPtrSharedArr, StorageClass.Workgroup, tArrayTile);
  const tPtrSharedF32 = b.id();
  b.typePointer(tPtrSharedF32, StorageClass.Workgroup, p.tF32);
  const tileA = b.id();
  b.variable(tPtrSharedArr, tileA, StorageClass.Workgroup);
  const tileB = b.id();
  b.variable(tPtrSharedArr, tileB, StorageClass.Workgroup);

  // Built-in variables
  const tPtrInputVec3 = b.id();
  b.typePointer(tPtrInputVec3, StorageClass.Input, p.tVec3U32);
  const vWorkgroupId = b.id();
  b.variable(tPtrInputVec3, vWorkgroupId, StorageClass.Input);
  b.addDecorate(vWorkgroupId, Decoration.BuiltIn, BuiltIn.WorkgroupId);
  const vLocalId = b.id();
  b.variable(tPtrInputVec3, vLocalId, StorageClass.Input);
  b.addDecorate(vLocalId, Decoration.BuiltIn, BuiltIn.LocalInvocationId);

  const scopeWg = b.id();
  b.constant(p.tU32, scopeWg, Scope.Workgroup);
  const semAcqRelWg = b.id();
  b.constant(p.tU32, semAcqRelWg, MemorySemantics.AcquireRelease | MemorySemantics.WorkgroupMemory);

  const tPtrFnU32 = b.id();
  b.typePointer(tPtrFnU32, StorageClass.Function, p.tU32);
  const tPtrFnF32 = b.id();
  b.typePointer(tPtrFnF32, StorageClass.Function, p.tF32);

  const const3u = b.id();
  b.constant(p.tU32, const3u, 3);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId, vWorkgroupId, vLocalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, TILE_SIZE, TILE_SIZE, 1);

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  const labelEntry = b.id();
  b.emit(Op.Label, [labelEntry]);

  const varT = b.id();
  b.emit(Op.Variable, [tPtrFnU32, varT, StorageClass.Function]);
  const varAcc = b.id();
  b.emit(Op.Variable, [tPtrFnF32, varAcc, StorageClass.Function]);

  // Local thread coords
  const lidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, lidVec, vLocalId]);
  const tx = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, tx, lidVec, 0]);
  const ty = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, ty, lidVec, 1]);

  // Workgroup coords — x=tile col, y=tile row, z=batch index
  const wgIdVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, wgIdVec, vWorkgroupId]);
  const bx = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, bx, wgIdVec, 0]);
  const by = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, by, wgIdVec, 1]);
  const batchIdx = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, batchIdx, wgIdVec, 2]);

  // Load M, N, K from push constants
  const MF = loadPushLen(b, p, pc);
  const NF = loadPushScalar(b, p, pc);
  const ptrK = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrK, pc.varId, p.const2u]);
  const KF = b.id();
  b.emit(Op.Load, [p.tF32, KF, ptrK]);

  const M = b.id(); b.emit(Op.ConvertFToU, [p.tU32, M, MF]);
  const N = b.id(); b.emit(Op.ConvertFToU, [p.tU32, N, NF]);
  const K = b.id(); b.emit(Op.ConvertFToU, [p.tU32, K, KF]);

  // Batch offsets: A_off = batchIdx * M * K, B_off = batchIdx * N * K, C_off = batchIdx * M * N
  const MK = b.id(); b.emit(Op.IMul, [p.tU32, MK, M, K]);
  const NK = b.id(); b.emit(Op.IMul, [p.tU32, NK, N, K]);
  const MN = b.id(); b.emit(Op.IMul, [p.tU32, MN, M, N]);
  const aOff = b.id(); b.emit(Op.IMul, [p.tU32, aOff, batchIdx, MK]);
  const bOff = b.id(); b.emit(Op.IMul, [p.tU32, bOff, batchIdx, NK]);
  const cOff = b.id(); b.emit(Op.IMul, [p.tU32, cOff, batchIdx, MN]);

  // Global output row/col
  const globalRow = b.id();
  const byTimesT = b.id();
  b.emit(Op.IMul, [p.tU32, byTimesT, by, constTileSize]);
  b.emit(Op.IAdd, [p.tU32, globalRow, byTimesT, ty]);
  const globalCol = b.id();
  const bxTimesT = b.id();
  b.emit(Op.IMul, [p.tU32, bxTimesT, bx, constTileSize]);
  b.emit(Op.IAdd, [p.tU32, globalCol, bxTimesT, tx]);

  // acc = 0.0
  b.emit(Op.Store, [varAcc, p.const0f]);

  // Tile index within shared memory
  const localTileIdx = b.id();
  const tyTimesT = b.id();
  b.emit(Op.IMul, [p.tU32, tyTimesT, ty, constTileSize]);
  b.emit(Op.IAdd, [p.tU32, localTileIdx, tyTimesT, tx]);

  // Loop: for (t = 0; t < K; t += TILE_SIZE)
  b.emit(Op.Store, [varT, p.const0u]);

  const labelHead = b.id();
  const labelBody = b.id();
  const labelMerge = b.id();
  const labelCont = b.id();

  b.emit(Op.Branch, [labelHead]);
  b.emit(Op.Label, [labelHead]);
  const t = b.id();
  b.emit(Op.Load, [p.tU32, t, varT]);
  const cmp = b.id();
  b.emit(Op.ULessThan, [p.tBool, cmp, t, K]);
  b.emit(Op.LoopMerge, [labelMerge, labelCont, 0]);
  b.emit(Op.BranchConditional, [cmp, labelBody, labelMerge]);

  b.emit(Op.Label, [labelBody]);

  // Load tile of A: A[aOff + globalRow * K + t + tx]
  const aCol = b.id();
  b.emit(Op.IAdd, [p.tU32, aCol, t, tx]);
  const aInBoundsR = b.id();
  b.emit(Op.ULessThan, [p.tBool, aInBoundsR, globalRow, M]);
  const aInBoundsC = b.id();
  b.emit(Op.ULessThan, [p.tBool, aInBoundsC, aCol, K]);
  const aInBounds = b.id();
  b.emit(Op.LogicalAnd, [p.tBool, aInBounds, aInBoundsR, aInBoundsC]);
  const aLinear = b.id();
  const grTimesK = b.id();
  b.emit(Op.IMul, [p.tU32, grTimesK, globalRow, K]);
  b.emit(Op.IAdd, [p.tU32, aLinear, grTimesK, aCol]);
  const aIdx = b.id();
  b.emit(Op.IAdd, [p.tU32, aIdx, aOff, aLinear]);
  const ptrAElem = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrF32, ptrAElem, bufA.varId, p.const0u, aIdx]);
  const aRaw = b.id();
  b.emit(Op.Load, [p.tF32, aRaw, ptrAElem]);
  const aVal = b.id();
  b.emit(Op.Select, [p.tF32, aVal, aInBounds, aRaw, p.const0f]);
  const ptrTileA = b.id();
  b.emit(Op.AccessChain, [tPtrSharedF32, ptrTileA, tileA, localTileIdx]);
  b.emit(Op.Store, [ptrTileA, aVal]);

  // Load tile of B transposed: B^T[t + ty, globalCol] = B[globalCol, t + ty]
  // B is stored as [N, K] per batch, so B[globalCol, t+ty] = B[bOff + globalCol * K + (t + ty)]
  const bRow = b.id();
  b.emit(Op.IAdd, [p.tU32, bRow, t, ty]);
  const bInBoundsR = b.id();
  b.emit(Op.ULessThan, [p.tBool, bInBoundsR, bRow, K]);
  const bInBoundsC = b.id();
  b.emit(Op.ULessThan, [p.tBool, bInBoundsC, globalCol, N]);
  const bInBounds = b.id();
  b.emit(Op.LogicalAnd, [p.tBool, bInBounds, bInBoundsR, bInBoundsC]);
  const bLinear = b.id();
  const gcTimesK = b.id();
  b.emit(Op.IMul, [p.tU32, gcTimesK, globalCol, K]);
  b.emit(Op.IAdd, [p.tU32, bLinear, gcTimesK, bRow]);
  const bIdx = b.id();
  b.emit(Op.IAdd, [p.tU32, bIdx, bOff, bLinear]);
  const ptrBElem = b.id();
  b.emit(Op.AccessChain, [bufB.tPtrF32, ptrBElem, bufB.varId, p.const0u, bIdx]);
  const bRaw = b.id();
  b.emit(Op.Load, [p.tF32, bRaw, ptrBElem]);
  const bVal = b.id();
  b.emit(Op.Select, [p.tF32, bVal, bInBounds, bRaw, p.const0f]);
  const ptrTileB = b.id();
  b.emit(Op.AccessChain, [tPtrSharedF32, ptrTileB, tileB, localTileIdx]);
  b.emit(Op.Store, [ptrTileB, bVal]);

  // Barrier
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  // Accumulate: for k = 0..TILE_SIZE-1: acc += tileA[ty][k] * tileB[k][tx]
  for (let k = 0; k < TILE_SIZE; k++) {
    const kConst = b.id();
    b.constant(p.tU32, kConst, k);
    const aI = b.id();
    const tyT = b.id();
    b.emit(Op.IMul, [p.tU32, tyT, ty, constTileSize]);
    b.emit(Op.IAdd, [p.tU32, aI, tyT, kConst]);
    const pA = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF32, pA, tileA, aI]);
    const aV = b.id();
    b.emit(Op.Load, [p.tF32, aV, pA]);
    const bI = b.id();
    const kT = b.id();
    b.emit(Op.IMul, [p.tU32, kT, kConst, constTileSize]);
    b.emit(Op.IAdd, [p.tU32, bI, kT, tx]);
    const pB = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF32, pB, tileB, bI]);
    const bV = b.id();
    b.emit(Op.Load, [p.tF32, bV, pB]);
    const curAcc = b.id();
    b.emit(Op.Load, [p.tF32, curAcc, varAcc]);
    const prod = b.id();
    b.emit(Op.FMul, [p.tF32, prod, aV, bV]);
    const newAcc = b.id();
    b.emit(Op.FAdd, [p.tF32, newAcc, curAcc, prod]);
    b.emit(Op.Store, [varAcc, newAcc]);
  }

  // Barrier
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  b.emit(Op.Branch, [labelCont]);
  b.emit(Op.Label, [labelCont]);
  const nextT = b.id();
  b.emit(Op.Load, [p.tU32, nextT, varT]);
  const incT = b.id();
  b.emit(Op.IAdd, [p.tU32, incT, nextT, constTileSize]);
  b.emit(Op.Store, [varT, incT]);
  b.emit(Op.Branch, [labelHead]);

  b.emit(Op.Label, [labelMerge]);

  // Write output: C[cOff + globalRow * N + globalCol] = acc (if in bounds)
  const outInBoundsR = b.id();
  b.emit(Op.ULessThan, [p.tBool, outInBoundsR, globalRow, M]);
  const outInBoundsC = b.id();
  b.emit(Op.ULessThan, [p.tBool, outInBoundsC, globalCol, N]);
  const outInBounds = b.id();
  b.emit(Op.LogicalAnd, [p.tBool, outInBounds, outInBoundsR, outInBoundsC]);
  const labelWrite = b.id();
  const labelEnd = b.id();
  b.emit(Op.SelectionMerge, [labelEnd, 0]);
  b.emit(Op.BranchConditional, [outInBounds, labelWrite, labelEnd]);

  b.emit(Op.Label, [labelWrite]);
  const outLinear = b.id();
  const grTimesN = b.id();
  b.emit(Op.IMul, [p.tU32, grTimesN, globalRow, N]);
  b.emit(Op.IAdd, [p.tU32, outLinear, grTimesN, globalCol]);
  const outIdx = b.id();
  b.emit(Op.IAdd, [p.tU32, outIdx, cOff, outLinear]);
  const ptrOut = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrF32, ptrOut, bufC.varId, p.const0u, outIdx]);
  const finalAcc = b.id();
  b.emit(Op.Load, [p.tF32, finalAcc, varAcc]);
  b.emit(Op.Store, [ptrOut, finalAcc]);
  b.emit(Op.Branch, [labelEnd]);

  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}
