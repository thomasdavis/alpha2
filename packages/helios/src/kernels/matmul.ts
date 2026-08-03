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

// ── Kernel: matmul_transposed_a ──────────────────────────────────────────────

/**
 * Tiled matmul with A transposed: C = A^T × B
 *
 * A is stored as [M, K] (row-major), used as A^T = [K, M].
 * B is stored as [M, N].
 * Result C is [K, N].
 *
 * Push constants encode generic matmul dims for the transposed-A view:
 *   { M: f32, N: f32, K: f32, _pad: f32 }
 * where:
 *   M = output rows (= original K),
 *   N = output cols,
 *   K = reduction dim (= original M).
 *
 * Bindings: 0=A(in), 1=B(in), 2=C(out)
 * Dispatch: (ceil(N/TILE), ceil(M/TILE), 1)
 */
export function kernelMatmulTransposedA(wgSize = DEFAULT_TILE * DEFAULT_TILE, tileSize = DEFAULT_TILE): Uint32Array {
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

  // Load generic matmul dims from push constants:
  // M = output rows, N = output cols, K = reduction dim
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

  // acc = 0
  b.emit(Op.Store, [varAcc, p.const0f]);

  // Tile index in shared memory
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

  // Load tile of A transposed:
  // A^T[globalRow, t+tx] = A[t+tx, globalRow], A stored as [K, M]
  const aRow = b.id();
  b.emit(Op.IAdd, [p.tU32, aRow, t, tx]);
  const aInBoundsR = b.id();
  b.emit(Op.ULessThan, [p.tBool, aInBoundsR, aRow, K]);
  const aInBoundsC = b.id();
  b.emit(Op.ULessThan, [p.tBool, aInBoundsC, globalRow, M]);
  const aInBounds = b.id();
  b.emit(Op.LogicalAnd, [p.tBool, aInBounds, aInBoundsR, aInBoundsC]);
  const aLinear = b.id();
  const arTimesM = b.id();
  b.emit(Op.IMul, [p.tU32, arTimesM, aRow, M]);
  b.emit(Op.IAdd, [p.tU32, aLinear, arTimesM, globalRow]);
  const ptrAElem = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrF32, ptrAElem, bufA.varId, p.const0u, aLinear]);
  const aRaw = b.id();
  b.emit(Op.Load, [p.tF32, aRaw, ptrAElem]);
  const aVal = b.id();
  b.emit(Op.Select, [p.tF32, aVal, aInBounds, aRaw, p.const0f]);
  const ptrTileA = b.id();
  b.emit(Op.AccessChain, [tPtrSharedF32, ptrTileA, tileA, localTileIdx]);
  b.emit(Op.Store, [ptrTileA, aVal]);

  // Load tile of B: B[t+ty, globalCol], B stored as [K, N]
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

  // Barrier
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  // Accumulate: acc += tileA[ty][k] * tileB[k][tx]
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

  // Barrier before next tile
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

  // Write output C[globalRow, globalCol]
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
 * Batched tiled matmul with A transposed: C[b] = A[b]^T × B[b].
 *
 * A is [batch, M, K] (stored row-major), used as [batch, K, M].
 * B is [batch, M, N].
 * Result C is [batch, K, N].
 *
 * Push constants use transposed-A generic dims:
 *   M = output rows (= original K), N = output cols, K = reduction (= original M)
 */
export function kernelMatmulTransposedABatched(wgSize = DEFAULT_TILE * DEFAULT_TILE, tileSize = DEFAULT_TILE): Uint32Array {
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

  // Built-ins
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
  const batchIdx = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, batchIdx, wgIdVec, 2]);

  // Load generic dims: M=output rows, N=output cols, K=reduction
  const MF = loadPushLen(b, p, pc);
  const NF = loadPushScalar(b, p, pc);
  const ptrK = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrK, pc.varId, p.const2u]);
  const KF = b.id();
  b.emit(Op.Load, [p.tF32, KF, ptrK]);

  const M = b.id(); b.emit(Op.ConvertFToU, [p.tU32, M, MF]);
  const N = b.id(); b.emit(Op.ConvertFToU, [p.tU32, N, NF]);
  const K = b.id(); b.emit(Op.ConvertFToU, [p.tU32, K, KF]);

  // Batch offsets:
  // A [batch, K, M], B [batch, K, N], C [batch, M, N]
  const KM = b.id(); b.emit(Op.IMul, [p.tU32, KM, K, M]);
  const KN = b.id(); b.emit(Op.IMul, [p.tU32, KN, K, N]);
  const MN = b.id(); b.emit(Op.IMul, [p.tU32, MN, M, N]);
  const aOff = b.id(); b.emit(Op.IMul, [p.tU32, aOff, batchIdx, KM]);
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

  // acc = 0
  b.emit(Op.Store, [varAcc, p.const0f]);

  // Tile index in shared memory
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

  // Load tile of A transposed from stored [K, M]:
  // A^T[globalRow, t+tx] = A[t+tx, globalRow]
  const aRow = b.id();
  b.emit(Op.IAdd, [p.tU32, aRow, t, tx]);
  const aInBoundsR = b.id();
  b.emit(Op.ULessThan, [p.tBool, aInBoundsR, aRow, K]);
  const aInBoundsC = b.id();
  b.emit(Op.ULessThan, [p.tBool, aInBoundsC, globalRow, M]);
  const aInBounds = b.id();
  b.emit(Op.LogicalAnd, [p.tBool, aInBounds, aInBoundsR, aInBoundsC]);
  const aLinear = b.id();
  const arTimesM = b.id();
  b.emit(Op.IMul, [p.tU32, arTimesM, aRow, M]);
  b.emit(Op.IAdd, [p.tU32, aLinear, arTimesM, globalRow]);
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

  // Load tile of B: B[bOff + (t+ty) * N + globalCol], B stored [K, N]
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

  // Accumulate
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

  // Write output
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

// ── Register-blocked 2×2 output kernels ────────────────────────────────────

type RegisterBlockMode = "basic" | "transposed-b" | "transposed-a";

/**
 * Portable register-blocked FP32 GEMM.
 *
 * A 16×16 workgroup computes a 32×32 output tile.  Each invocation owns four
 * accumulators and therefore reuses every shared-memory value twice in each
 * output dimension.  Compared with the historical tile-32 kernel this keeps
 * the larger output tile while reducing the workgroup from 1024 to 256
 * invocations, which is legal on a much wider range of Vulkan hardware.
 *
 * This deliberately remains scalar FP32 and subgroup-independent.  It is a
 * portable baseline for NVIDIA and AMD; cooperative-matrix kernels are a
 * separate capability-selected family.
 */
function kernelMatmulReg2x2Impl(mode: RegisterBlockMode): Uint32Array {
  const WG = 16;
  const OUT_TILE = 32;
  const b = new SpirVBuilder();
  const p = preamble(b, WG, WG, 1);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufB = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, true);
  const bufC = declareStorageBuffer(b, p.tF32, p.tU32, 0, 2, false);
  const pc = declareParamsPushConstant(b, p.tF32, 4);

  const c16 = b.id(); b.constant(p.tU32, c16, WG);
  const c32 = b.id(); b.constant(p.tU32, c32, OUT_TILE);
  const c256 = b.id(); b.constant(p.tU32, c256, WG * WG);
  const c512 = b.id(); b.constant(p.tU32, c512, OUT_TILE * WG);

  // A tile is [32,16], B tile is [16,32].
  const tSharedArray = b.id();
  b.typeArray(tSharedArray, p.tF32, c512);
  const tPtrSharedArray = b.id();
  b.typePointer(tPtrSharedArray, StorageClass.Workgroup, tSharedArray);
  const tPtrSharedF32 = b.id();
  b.typePointer(tPtrSharedF32, StorageClass.Workgroup, p.tF32);
  const tileA = b.id(); b.variable(tPtrSharedArray, tileA, StorageClass.Workgroup);
  const tileB = b.id(); b.variable(tPtrSharedArray, tileB, StorageClass.Workgroup);

  const tPtrInputVec3 = b.id();
  b.typePointer(tPtrInputVec3, StorageClass.Input, p.tVec3U32);
  const vWorkgroupId = b.id();
  b.variable(tPtrInputVec3, vWorkgroupId, StorageClass.Input);
  b.addDecorate(vWorkgroupId, Decoration.BuiltIn, BuiltIn.WorkgroupId);
  const vLocalId = b.id();
  b.variable(tPtrInputVec3, vLocalId, StorageClass.Input);
  b.addDecorate(vLocalId, Decoration.BuiltIn, BuiltIn.LocalInvocationId);

  const scopeWg = b.id(); b.constant(p.tU32, scopeWg, Scope.Workgroup);
  const semAcqRelWg = b.id();
  b.constant(
    p.tU32,
    semAcqRelWg,
    MemorySemantics.AcquireRelease | MemorySemantics.WorkgroupMemory,
  );

  const tPtrFnU32 = b.id(); b.typePointer(tPtrFnU32, StorageClass.Function, p.tU32);
  const tPtrFnF32 = b.id(); b.typePointer(tPtrFnF32, StorageClass.Function, p.tF32);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId, vWorkgroupId, vLocalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, WG, WG, 1);
  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  const entry = b.id(); b.emit(Op.Label, [entry]);

  const varT = b.id(); b.emit(Op.Variable, [tPtrFnU32, varT, StorageClass.Function]);
  const acc00 = b.id(); b.emit(Op.Variable, [tPtrFnF32, acc00, StorageClass.Function]);
  const acc01 = b.id(); b.emit(Op.Variable, [tPtrFnF32, acc01, StorageClass.Function]);
  const acc10 = b.id(); b.emit(Op.Variable, [tPtrFnF32, acc10, StorageClass.Function]);
  const acc11 = b.id(); b.emit(Op.Variable, [tPtrFnF32, acc11, StorageClass.Function]);
  for (const acc of [acc00, acc01, acc10, acc11]) b.emit(Op.Store, [acc, p.const0f]);

  const lid = b.id(); b.emit(Op.Load, [p.tVec3U32, lid, vLocalId]);
  const tx = b.id(); b.emit(Op.CompositeExtract, [p.tU32, tx, lid, 0]);
  const ty = b.id(); b.emit(Op.CompositeExtract, [p.tU32, ty, lid, 1]);
  const wgId = b.id(); b.emit(Op.Load, [p.tVec3U32, wgId, vWorkgroupId]);
  const bx = b.id(); b.emit(Op.CompositeExtract, [p.tU32, bx, wgId, 0]);
  const by = b.id(); b.emit(Op.CompositeExtract, [p.tU32, by, wgId, 1]);

  const MF = loadPushLen(b, p, pc);
  const NF = loadPushScalar(b, p, pc);
  const ptrK = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrK, pc.varId, p.const2u]);
  const KF = b.id(); b.emit(Op.Load, [p.tF32, KF, ptrK]);
  const M = b.id(); b.emit(Op.ConvertFToU, [p.tU32, M, MF]);
  const N = b.id(); b.emit(Op.ConvertFToU, [p.tU32, N, NF]);
  const K = b.id(); b.emit(Op.ConvertFToU, [p.tU32, K, KF]);

  const bx32 = b.id(); b.emit(Op.IMul, [p.tU32, bx32, bx, c32]);
  const by32 = b.id(); b.emit(Op.IMul, [p.tU32, by32, by, c32]);
  const col0 = b.id(); b.emit(Op.IAdd, [p.tU32, col0, bx32, tx]);
  const col1 = b.id(); b.emit(Op.IAdd, [p.tU32, col1, col0, c16]);
  const row0 = b.id(); b.emit(Op.IAdd, [p.tU32, row0, by32, ty]);
  const row1 = b.id(); b.emit(Op.IAdd, [p.tU32, row1, row0, c16]);

  const ty16 = b.id(); b.emit(Op.IMul, [p.tU32, ty16, ty, c16]);
  const localA0 = b.id(); b.emit(Op.IAdd, [p.tU32, localA0, ty16, tx]);
  const localA1 = b.id(); b.emit(Op.IAdd, [p.tU32, localA1, localA0, c256]);
  const ty32 = b.id(); b.emit(Op.IMul, [p.tU32, ty32, ty, c32]);
  const localB0 = b.id(); b.emit(Op.IAdd, [p.tU32, localB0, ty32, tx]);
  const localB1 = b.id(); b.emit(Op.IAdd, [p.tU32, localB1, localB0, c16]);

  const loadA = (row: number, reductionCol: number): number => {
    const rowOk = b.id(); b.emit(Op.ULessThan, [p.tBool, rowOk, row, M]);
    const colOk = b.id(); b.emit(Op.ULessThan, [p.tBool, colOk, reductionCol, K]);
    const inBounds = b.id(); b.emit(Op.LogicalAnd, [p.tBool, inBounds, rowOk, colOk]);
    const major = b.id();
    const linear = b.id();
    if (mode === "transposed-a") {
      b.emit(Op.IMul, [p.tU32, major, reductionCol, M]);
      b.emit(Op.IAdd, [p.tU32, linear, major, row]);
    } else {
      b.emit(Op.IMul, [p.tU32, major, row, K]);
      b.emit(Op.IAdd, [p.tU32, linear, major, reductionCol]);
    }
    const ptr = b.id();
    b.emit(Op.AccessChain, [bufA.tPtrF32, ptr, bufA.varId, p.const0u, linear]);
    const raw = b.id(); b.emit(Op.Load, [p.tF32, raw, ptr]);
    const value = b.id(); b.emit(Op.Select, [p.tF32, value, inBounds, raw, p.const0f]);
    return value;
  };

  const loadB = (reductionRow: number, col: number): number => {
    const rowOk = b.id(); b.emit(Op.ULessThan, [p.tBool, rowOk, reductionRow, K]);
    const colOk = b.id(); b.emit(Op.ULessThan, [p.tBool, colOk, col, N]);
    const inBounds = b.id(); b.emit(Op.LogicalAnd, [p.tBool, inBounds, rowOk, colOk]);
    const major = b.id();
    const linear = b.id();
    if (mode === "transposed-b") {
      b.emit(Op.IMul, [p.tU32, major, col, K]);
      b.emit(Op.IAdd, [p.tU32, linear, major, reductionRow]);
    } else {
      b.emit(Op.IMul, [p.tU32, major, reductionRow, N]);
      b.emit(Op.IAdd, [p.tU32, linear, major, col]);
    }
    const ptr = b.id();
    b.emit(Op.AccessChain, [bufB.tPtrF32, ptr, bufB.varId, p.const0u, linear]);
    const raw = b.id(); b.emit(Op.Load, [p.tF32, raw, ptr]);
    const value = b.id(); b.emit(Op.Select, [p.tF32, value, inBounds, raw, p.const0f]);
    return value;
  };

  const storeShared = (tile: number, index: number, value: number): void => {
    const ptr = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF32, ptr, tile, index]);
    b.emit(Op.Store, [ptr, value]);
  };

  b.emit(Op.Store, [varT, p.const0u]);
  const loopHead = b.id();
  const loopBody = b.id();
  const loopMerge = b.id();
  const loopContinue = b.id();
  b.emit(Op.Branch, [loopHead]);
  b.emit(Op.Label, [loopHead]);
  const t = b.id(); b.emit(Op.Load, [p.tU32, t, varT]);
  const keepGoing = b.id(); b.emit(Op.ULessThan, [p.tBool, keepGoing, t, K]);
  b.emit(Op.LoopMerge, [loopMerge, loopContinue, 0]);
  b.emit(Op.BranchConditional, [keepGoing, loopBody, loopMerge]);
  b.emit(Op.Label, [loopBody]);

  const aCol = b.id(); b.emit(Op.IAdd, [p.tU32, aCol, t, tx]);
  const bRow = b.id(); b.emit(Op.IAdd, [p.tU32, bRow, t, ty]);
  storeShared(tileA, localA0, loadA(row0, aCol));
  storeShared(tileA, localA1, loadA(row1, aCol));
  storeShared(tileB, localB0, loadB(bRow, col0));
  storeShared(tileB, localB1, loadB(bRow, col1));
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  const accumulate = (acc: number, av: number, bv: number): void => {
    const current = b.id(); b.emit(Op.Load, [p.tF32, current, acc]);
    const product = b.id(); b.emit(Op.FMul, [p.tF32, product, av, bv]);
    const next = b.id(); b.emit(Op.FAdd, [p.tF32, next, current, product]);
    b.emit(Op.Store, [acc, next]);
  };

  for (let k = 0; k < WG; k++) {
    const ck = b.id(); b.constant(p.tU32, ck, k);
    const a0Idx = b.id(); b.emit(Op.IAdd, [p.tU32, a0Idx, ty16, ck]);
    const a1Idx = b.id(); b.emit(Op.IAdd, [p.tU32, a1Idx, a0Idx, c256]);
    const k32 = b.id(); b.emit(Op.IMul, [p.tU32, k32, ck, c32]);
    const b0Idx = b.id(); b.emit(Op.IAdd, [p.tU32, b0Idx, k32, tx]);
    const b1Idx = b.id(); b.emit(Op.IAdd, [p.tU32, b1Idx, b0Idx, c16]);

    const pA0 = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, pA0, tileA, a0Idx]);
    const pA1 = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, pA1, tileA, a1Idx]);
    const pB0 = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, pB0, tileB, b0Idx]);
    const pB1 = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, pB1, tileB, b1Idx]);
    const a0 = b.id(); b.emit(Op.Load, [p.tF32, a0, pA0]);
    const a1 = b.id(); b.emit(Op.Load, [p.tF32, a1, pA1]);
    const bv0 = b.id(); b.emit(Op.Load, [p.tF32, bv0, pB0]);
    const bv1 = b.id(); b.emit(Op.Load, [p.tF32, bv1, pB1]);
    accumulate(acc00, a0, bv0);
    accumulate(acc01, a0, bv1);
    accumulate(acc10, a1, bv0);
    accumulate(acc11, a1, bv1);
  }

  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);
  b.emit(Op.Branch, [loopContinue]);
  b.emit(Op.Label, [loopContinue]);
  const oldT = b.id(); b.emit(Op.Load, [p.tU32, oldT, varT]);
  const nextT = b.id(); b.emit(Op.IAdd, [p.tU32, nextT, oldT, c16]);
  b.emit(Op.Store, [varT, nextT]);
  b.emit(Op.Branch, [loopHead]);
  b.emit(Op.Label, [loopMerge]);

  const writeOutput = (row: number, col: number, acc: number): void => {
    const rowOk = b.id(); b.emit(Op.ULessThan, [p.tBool, rowOk, row, M]);
    const colOk = b.id(); b.emit(Op.ULessThan, [p.tBool, colOk, col, N]);
    const inBounds = b.id(); b.emit(Op.LogicalAnd, [p.tBool, inBounds, rowOk, colOk]);
    const writeLabel = b.id();
    const endLabel = b.id();
    b.emit(Op.SelectionMerge, [endLabel, 0]);
    b.emit(Op.BranchConditional, [inBounds, writeLabel, endLabel]);
    b.emit(Op.Label, [writeLabel]);
    const rowBase = b.id(); b.emit(Op.IMul, [p.tU32, rowBase, row, N]);
    const linear = b.id(); b.emit(Op.IAdd, [p.tU32, linear, rowBase, col]);
    const ptr = b.id();
    b.emit(Op.AccessChain, [bufC.tPtrF32, ptr, bufC.varId, p.const0u, linear]);
    const value = b.id(); b.emit(Op.Load, [p.tF32, value, acc]);
    b.emit(Op.Store, [ptr, value]);
    b.emit(Op.Branch, [endLabel]);
    b.emit(Op.Label, [endLabel]);
  };

  writeOutput(row0, col0, acc00);
  writeOutput(row0, col1, acc01);
  writeOutput(row1, col0, acc10);
  writeOutput(row1, col1, acc11);

  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);
  return b.build();
}

export function kernelMatmulReg2x2(): Uint32Array {
  return kernelMatmulReg2x2Impl("basic");
}

export function kernelMatmulTransposedReg2x2(): Uint32Array {
  return kernelMatmulReg2x2Impl("transposed-b");
}

export function kernelMatmulTransposedAReg2x2(): Uint32Array {
  return kernelMatmulReg2x2Impl("transposed-a");
}

/**
 * Experimental portable 4x2 register-blocked FP32 GEMM.
 *
 * A 16x8 workgroup still computes a 32x32 output tile, but each invocation
 * owns four rows and two columns.  Relative to R2 this halves the invocation
 * count, doubles A-value reuse per thread, and retains the same 4 KiB shared
 * tile.  The exact Alpha shape profile decides whether the added register
 * pressure is worthwhile; this generator makes no vendor assumption.
 */
function kernelMatmulReg4x2Impl(mode: RegisterBlockMode): Uint32Array {
  const WG_X = 16;
  const WG_Y = 8;
  const OUT_TILE = 32;
  const b = new SpirVBuilder();
  const p = preamble(b, WG_X, WG_Y, 1);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufB = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, true);
  const bufC = declareStorageBuffer(b, p.tF32, p.tU32, 0, 2, false);
  const pc = declareParamsPushConstant(b, p.tF32, 4);

  const c8 = b.id(); b.constant(p.tU32, c8, 8);
  const c16 = b.id(); b.constant(p.tU32, c16, 16);
  const c32 = b.id(); b.constant(p.tU32, c32, OUT_TILE);
  const c128 = b.id(); b.constant(p.tU32, c128, WG_X * WG_Y);
  const c512 = b.id(); b.constant(p.tU32, c512, OUT_TILE * WG_X);

  const tSharedArray = b.id();
  b.typeArray(tSharedArray, p.tF32, c512);
  const tPtrSharedArray = b.id();
  b.typePointer(tPtrSharedArray, StorageClass.Workgroup, tSharedArray);
  const tPtrSharedF32 = b.id();
  b.typePointer(tPtrSharedF32, StorageClass.Workgroup, p.tF32);
  const tileA = b.id(); b.variable(tPtrSharedArray, tileA, StorageClass.Workgroup);
  const tileB = b.id(); b.variable(tPtrSharedArray, tileB, StorageClass.Workgroup);

  const tPtrInputVec3 = b.id();
  b.typePointer(tPtrInputVec3, StorageClass.Input, p.tVec3U32);
  const vWorkgroupId = b.id();
  b.variable(tPtrInputVec3, vWorkgroupId, StorageClass.Input);
  b.addDecorate(vWorkgroupId, Decoration.BuiltIn, BuiltIn.WorkgroupId);
  const vLocalId = b.id();
  b.variable(tPtrInputVec3, vLocalId, StorageClass.Input);
  b.addDecorate(vLocalId, Decoration.BuiltIn, BuiltIn.LocalInvocationId);

  const scopeWg = b.id(); b.constant(p.tU32, scopeWg, Scope.Workgroup);
  const semAcqRelWg = b.id();
  b.constant(
    p.tU32,
    semAcqRelWg,
    MemorySemantics.AcquireRelease | MemorySemantics.WorkgroupMemory,
  );

  const tPtrFnU32 = b.id(); b.typePointer(tPtrFnU32, StorageClass.Function, p.tU32);
  const tPtrFnF32 = b.id(); b.typePointer(tPtrFnF32, StorageClass.Function, p.tF32);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId, vWorkgroupId, vLocalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, WG_X, WG_Y, 1);
  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  const entry = b.id(); b.emit(Op.Label, [entry]);

  const varT = b.id(); b.emit(Op.Variable, [tPtrFnU32, varT, StorageClass.Function]);
  const accumulators: number[][] = [];
  for (let r = 0; r < 4; r++) {
    const row: number[] = [];
    for (let c = 0; c < 2; c++) {
      const acc = b.id();
      b.emit(Op.Variable, [tPtrFnF32, acc, StorageClass.Function]);
      b.emit(Op.Store, [acc, p.const0f]);
      row.push(acc);
    }
    accumulators.push(row);
  }

  const lid = b.id(); b.emit(Op.Load, [p.tVec3U32, lid, vLocalId]);
  const tx = b.id(); b.emit(Op.CompositeExtract, [p.tU32, tx, lid, 0]);
  const ty = b.id(); b.emit(Op.CompositeExtract, [p.tU32, ty, lid, 1]);
  const wgId = b.id(); b.emit(Op.Load, [p.tVec3U32, wgId, vWorkgroupId]);
  const bx = b.id(); b.emit(Op.CompositeExtract, [p.tU32, bx, wgId, 0]);
  const by = b.id(); b.emit(Op.CompositeExtract, [p.tU32, by, wgId, 1]);

  const MF = loadPushLen(b, p, pc);
  const NF = loadPushScalar(b, p, pc);
  const ptrK = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrK, pc.varId, p.const2u]);
  const KF = b.id(); b.emit(Op.Load, [p.tF32, KF, ptrK]);
  const M = b.id(); b.emit(Op.ConvertFToU, [p.tU32, M, MF]);
  const N = b.id(); b.emit(Op.ConvertFToU, [p.tU32, N, NF]);
  const K = b.id(); b.emit(Op.ConvertFToU, [p.tU32, K, KF]);

  const bx32 = b.id(); b.emit(Op.IMul, [p.tU32, bx32, bx, c32]);
  const by32 = b.id(); b.emit(Op.IMul, [p.tU32, by32, by, c32]);
  const col0 = b.id(); b.emit(Op.IAdd, [p.tU32, col0, bx32, tx]);
  const col1 = b.id(); b.emit(Op.IAdd, [p.tU32, col1, col0, c16]);
  const rows: number[] = [];
  const row0 = b.id(); b.emit(Op.IAdd, [p.tU32, row0, by32, ty]);
  rows.push(row0);
  for (let r = 1; r < 4; r++) {
    const row = b.id();
    const offset = b.id(); b.constant(p.tU32, offset, r * WG_Y);
    b.emit(Op.IAdd, [p.tU32, row, row0, offset]);
    rows.push(row);
  }

  const ty16 = b.id(); b.emit(Op.IMul, [p.tU32, ty16, ty, c16]);
  const localLinear = b.id(); b.emit(Op.IAdd, [p.tU32, localLinear, ty16, tx]);

  const loadA = (row: number, reductionCol: number): number => {
    const rowOk = b.id(); b.emit(Op.ULessThan, [p.tBool, rowOk, row, M]);
    const colOk = b.id(); b.emit(Op.ULessThan, [p.tBool, colOk, reductionCol, K]);
    const inBounds = b.id(); b.emit(Op.LogicalAnd, [p.tBool, inBounds, rowOk, colOk]);
    const major = b.id();
    const linear = b.id();
    if (mode === "transposed-a") {
      b.emit(Op.IMul, [p.tU32, major, reductionCol, M]);
      b.emit(Op.IAdd, [p.tU32, linear, major, row]);
    } else {
      b.emit(Op.IMul, [p.tU32, major, row, K]);
      b.emit(Op.IAdd, [p.tU32, linear, major, reductionCol]);
    }
    const ptr = b.id();
    b.emit(Op.AccessChain, [bufA.tPtrF32, ptr, bufA.varId, p.const0u, linear]);
    const raw = b.id(); b.emit(Op.Load, [p.tF32, raw, ptr]);
    const value = b.id(); b.emit(Op.Select, [p.tF32, value, inBounds, raw, p.const0f]);
    return value;
  };

  const loadB = (reductionRow: number, col: number): number => {
    const rowOk = b.id(); b.emit(Op.ULessThan, [p.tBool, rowOk, reductionRow, K]);
    const colOk = b.id(); b.emit(Op.ULessThan, [p.tBool, colOk, col, N]);
    const inBounds = b.id(); b.emit(Op.LogicalAnd, [p.tBool, inBounds, rowOk, colOk]);
    const major = b.id();
    const linear = b.id();
    if (mode === "transposed-b") {
      b.emit(Op.IMul, [p.tU32, major, col, K]);
      b.emit(Op.IAdd, [p.tU32, linear, major, reductionRow]);
    } else {
      b.emit(Op.IMul, [p.tU32, major, reductionRow, N]);
      b.emit(Op.IAdd, [p.tU32, linear, major, col]);
    }
    const ptr = b.id();
    b.emit(Op.AccessChain, [bufB.tPtrF32, ptr, bufB.varId, p.const0u, linear]);
    const raw = b.id(); b.emit(Op.Load, [p.tF32, raw, ptr]);
    const value = b.id(); b.emit(Op.Select, [p.tF32, value, inBounds, raw, p.const0f]);
    return value;
  };

  const storeShared = (tile: number, index: number, value: number): void => {
    const ptr = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF32, ptr, tile, index]);
    b.emit(Op.Store, [ptr, value]);
  };

  b.emit(Op.Store, [varT, p.const0u]);
  const loopHead = b.id();
  const loopBody = b.id();
  const loopMerge = b.id();
  const loopContinue = b.id();
  b.emit(Op.Branch, [loopHead]);
  b.emit(Op.Label, [loopHead]);
  const t = b.id(); b.emit(Op.Load, [p.tU32, t, varT]);
  const keepGoing = b.id(); b.emit(Op.ULessThan, [p.tBool, keepGoing, t, K]);
  b.emit(Op.LoopMerge, [loopMerge, loopContinue, 0]);
  b.emit(Op.BranchConditional, [keepGoing, loopBody, loopMerge]);
  b.emit(Op.Label, [loopBody]);

  const aCol = b.id(); b.emit(Op.IAdd, [p.tU32, aCol, t, tx]);
  for (let r = 0; r < 4; r++) {
    const sharedIndex = b.id();
    const offset = b.id(); b.constant(p.tU32, offset, r * WG_X * WG_Y);
    b.emit(Op.IAdd, [p.tU32, sharedIndex, localLinear, offset]);
    storeShared(tileA, sharedIndex, loadA(rows[r], aCol));
  }
  for (let i = 0; i < 4; i++) {
    const sharedIndex = b.id();
    const offset = b.id(); b.constant(p.tU32, offset, i * WG_X * WG_Y);
    b.emit(Op.IAdd, [p.tU32, sharedIndex, localLinear, offset]);
    const tileRow = b.id(); b.emit(Op.UDiv, [p.tU32, tileRow, sharedIndex, c32]);
    const tileCol = b.id(); b.emit(Op.UMod, [p.tU32, tileCol, sharedIndex, c32]);
    const reductionRow = b.id(); b.emit(Op.IAdd, [p.tU32, reductionRow, t, tileRow]);
    const globalCol = b.id(); b.emit(Op.IAdd, [p.tU32, globalCol, bx32, tileCol]);
    storeShared(tileB, sharedIndex, loadB(reductionRow, globalCol));
  }
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  const accumulate = (acc: number, av: number, bv: number): void => {
    const current = b.id(); b.emit(Op.Load, [p.tF32, current, acc]);
    const product = b.id(); b.emit(Op.FMul, [p.tF32, product, av, bv]);
    const next = b.id(); b.emit(Op.FAdd, [p.tF32, next, current, product]);
    b.emit(Op.Store, [acc, next]);
  };

  for (let k = 0; k < WG_X; k++) {
    const ck = b.id(); b.constant(p.tU32, ck, k);
    const aValues: number[] = [];
    for (let r = 0; r < 4; r++) {
      const rowBase = b.id();
      const rowOffset = b.id(); b.constant(p.tU32, rowOffset, r * WG_Y * WG_X);
      b.emit(Op.IAdd, [p.tU32, rowBase, ty16, rowOffset]);
      const aIndex = b.id(); b.emit(Op.IAdd, [p.tU32, aIndex, rowBase, ck]);
      const ptr = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptr, tileA, aIndex]);
      const value = b.id(); b.emit(Op.Load, [p.tF32, value, ptr]);
      aValues.push(value);
    }
    const k32 = b.id(); b.emit(Op.IMul, [p.tU32, k32, ck, c32]);
    const b0Index = b.id(); b.emit(Op.IAdd, [p.tU32, b0Index, k32, tx]);
    const b1Index = b.id(); b.emit(Op.IAdd, [p.tU32, b1Index, b0Index, c16]);
    const ptrB0 = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrB0, tileB, b0Index]);
    const ptrB1 = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrB1, tileB, b1Index]);
    const b0 = b.id(); b.emit(Op.Load, [p.tF32, b0, ptrB0]);
    const b1 = b.id(); b.emit(Op.Load, [p.tF32, b1, ptrB1]);
    for (let r = 0; r < 4; r++) {
      accumulate(accumulators[r][0], aValues[r], b0);
      accumulate(accumulators[r][1], aValues[r], b1);
    }
  }

  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);
  b.emit(Op.Branch, [loopContinue]);
  b.emit(Op.Label, [loopContinue]);
  const oldT = b.id(); b.emit(Op.Load, [p.tU32, oldT, varT]);
  const nextT = b.id(); b.emit(Op.IAdd, [p.tU32, nextT, oldT, c16]);
  b.emit(Op.Store, [varT, nextT]);
  b.emit(Op.Branch, [loopHead]);
  b.emit(Op.Label, [loopMerge]);

  const writeOutput = (row: number, col: number, acc: number): void => {
    const rowOk = b.id(); b.emit(Op.ULessThan, [p.tBool, rowOk, row, M]);
    const colOk = b.id(); b.emit(Op.ULessThan, [p.tBool, colOk, col, N]);
    const inBounds = b.id(); b.emit(Op.LogicalAnd, [p.tBool, inBounds, rowOk, colOk]);
    const writeLabel = b.id();
    const endLabel = b.id();
    b.emit(Op.SelectionMerge, [endLabel, 0]);
    b.emit(Op.BranchConditional, [inBounds, writeLabel, endLabel]);
    b.emit(Op.Label, [writeLabel]);
    const rowBase = b.id(); b.emit(Op.IMul, [p.tU32, rowBase, row, N]);
    const linear = b.id(); b.emit(Op.IAdd, [p.tU32, linear, rowBase, col]);
    const ptr = b.id();
    b.emit(Op.AccessChain, [bufC.tPtrF32, ptr, bufC.varId, p.const0u, linear]);
    const value = b.id(); b.emit(Op.Load, [p.tF32, value, acc]);
    b.emit(Op.Store, [ptr, value]);
    b.emit(Op.Branch, [endLabel]);
    b.emit(Op.Label, [endLabel]);
  };

  for (let r = 0; r < 4; r++) {
    writeOutput(rows[r], col0, accumulators[r][0]);
    writeOutput(rows[r], col1, accumulators[r][1]);
  }

  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);
  return b.build();
}

export function kernelMatmulReg4x2(): Uint32Array {
  return kernelMatmulReg4x2Impl("basic");
}

export function kernelMatmulTransposedReg4x2(): Uint32Array {
  return kernelMatmulReg4x2Impl("transposed-b");
}

export function kernelMatmulTransposedAReg4x2(): Uint32Array {
  return kernelMatmulReg4x2Impl("transposed-a");
}
