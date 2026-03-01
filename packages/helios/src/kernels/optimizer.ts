/**
 * kernels/optimizer.ts — Optimizer and utility GPU kernels.
 *
 * AdamW optimizer step, transpose, add_inplace.
 */

import {
  SpirVBuilder, Op, ExecutionModel, ExecutionMode, StorageClass, Decoration,
  BuiltIn, FunctionControl, Scope, MemorySemantics, GLSLstd450,
  preamble, declareStorageBuffer, declareParamsPushConstant,
  loadPushLen, loadPushScalar, emitBoundsCheck,
} from "./helpers.js";

// ── Kernel: AdamW optimizer step (in-place) ─────────────────────────────────

/**
 * In-place AdamW update on GPU. One thread per parameter element.
 * Bindings: 0=params(rw), 1=grads(r), 2=m(rw), 3=v(rw)
 * Push constants: { len, lr, beta1, beta2, eps, weightDecay, bc1, bc2, gradScale } (9 x f32)
 *
 * For each element i:
 *   params[i] -= lr * weightDecay * params[i]          // decoupled weight decay
 *   m[i] = beta1 * m[i] + (1 - beta1) * grads[i]      // first moment
 *   v[i] = beta2 * v[i] + (1 - beta2) * grads[i]^2    // second moment
 *   mHat = m[i] / bc1                                   // bias correction
 *   vHat = v[i] / bc2
 *   params[i] -= lr * mHat / (sqrt(vHat) + eps)        // parameter update
 */
export function kernelAdamW(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  // 4 buffers: params(rw), grads(r), m(rw), v(rw)
  const bufParams = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, false);
  const bufGrads  = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, true);
  const bufM      = declareStorageBuffer(b, p.tF32, p.tU32, 0, 2, false);
  const bufV      = declareStorageBuffer(b, p.tF32, p.tU32, 0, 3, false);

  // 9 push constants: len, lr, beta1, beta2, eps, weightDecay, bc1, bc2, gradScale
  const pc = declareParamsPushConstant(b, p.tF32, 9);

  // Constants
  const const1f = b.id(); b.constantF32(p.tF32, const1f, 1.0);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  const labelEntry = b.id();
  const labelEnd = b.id();

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  b.emit(Op.Label, [labelEntry]);

  // Load global ID
  const gidVec = b.id(); b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gidX = b.id(); b.emit(Op.CompositeExtract, [p.tU32, gidX, gidVec, 0]);

  // Load all push constants
  const lenF = loadPushLen(b, p, pc);
  // pc members: 0=len, 1=lr, 2=beta1, 3=beta2, 4=eps, 5=weightDecay, 6=bc1, 7=bc2
  const ptrLr = b.id(); b.emit(Op.AccessChain, [pc.tPtrF32, ptrLr, pc.varId, p.const1u]);
  const lr = b.id(); b.emit(Op.Load, [p.tF32, lr, ptrLr]);

  const idx2 = b.id(); b.constant(p.tU32, idx2, 2);
  const idx3 = b.id(); b.constant(p.tU32, idx3, 3);
  const idx4 = b.id(); b.constant(p.tU32, idx4, 4);
  const idx5 = b.id(); b.constant(p.tU32, idx5, 5);
  const idx6 = b.id(); b.constant(p.tU32, idx6, 6);
  const idx7 = b.id(); b.constant(p.tU32, idx7, 7);
  const idx8 = b.id(); b.constant(p.tU32, idx8, 8);

  const ptrB1 = b.id(); b.emit(Op.AccessChain, [pc.tPtrF32, ptrB1, pc.varId, idx2]);
  const beta1 = b.id(); b.emit(Op.Load, [p.tF32, beta1, ptrB1]);

  const ptrB2 = b.id(); b.emit(Op.AccessChain, [pc.tPtrF32, ptrB2, pc.varId, idx3]);
  const beta2 = b.id(); b.emit(Op.Load, [p.tF32, beta2, ptrB2]);

  const ptrEps = b.id(); b.emit(Op.AccessChain, [pc.tPtrF32, ptrEps, pc.varId, idx4]);
  const eps = b.id(); b.emit(Op.Load, [p.tF32, eps, ptrEps]);

  const ptrWD = b.id(); b.emit(Op.AccessChain, [pc.tPtrF32, ptrWD, pc.varId, idx5]);
  const wd = b.id(); b.emit(Op.Load, [p.tF32, wd, ptrWD]);

  const ptrBc1 = b.id(); b.emit(Op.AccessChain, [pc.tPtrF32, ptrBc1, pc.varId, idx6]);
  const bc1 = b.id(); b.emit(Op.Load, [p.tF32, bc1, ptrBc1]);

  const ptrBc2 = b.id(); b.emit(Op.AccessChain, [pc.tPtrF32, ptrBc2, pc.varId, idx7]);
  const bc2 = b.id(); b.emit(Op.Load, [p.tF32, bc2, ptrBc2]);

  const ptrGradScale = b.id(); b.emit(Op.AccessChain, [pc.tPtrF32, ptrGradScale, pc.varId, idx8]);
  const gradScale = b.id(); b.emit(Op.Load, [p.tF32, gradScale, ptrGradScale]);

  // Bounds check
  emitBoundsCheck(b, p, lenF, gidX, labelEnd);

  // (1 - beta1), (1 - beta2)
  const oneMinusB1 = b.id(); b.emit(Op.FSub, [p.tF32, oneMinusB1, const1f, beta1]);
  const oneMinusB2 = b.id(); b.emit(Op.FSub, [p.tF32, oneMinusB2, const1f, beta2]);

  // Load params[i]
  const ptrP = b.id(); b.emit(Op.AccessChain, [bufParams.tPtrF32, ptrP, bufParams.varId, p.const0u, gidX]);
  const paramVal = b.id(); b.emit(Op.Load, [p.tF32, paramVal, ptrP]);

  // Load grads[i]
  const ptrG = b.id(); b.emit(Op.AccessChain, [bufGrads.tPtrF32, ptrG, bufGrads.varId, p.const0u, gidX]);
  const gradVal = b.id(); b.emit(Op.Load, [p.tF32, gradVal, ptrG]);
  const gradScaled = b.id(); b.emit(Op.FMul, [p.tF32, gradScaled, gradVal, gradScale]);

  // Load m[i], v[i]
  const ptrM = b.id(); b.emit(Op.AccessChain, [bufM.tPtrF32, ptrM, bufM.varId, p.const0u, gidX]);
  const mVal = b.id(); b.emit(Op.Load, [p.tF32, mVal, ptrM]);

  const ptrV = b.id(); b.emit(Op.AccessChain, [bufV.tPtrF32, ptrV, bufV.varId, p.const0u, gidX]);
  const vVal = b.id(); b.emit(Op.Load, [p.tF32, vVal, ptrV]);

  // Weight decay: params[i] -= lr * weightDecay * params[i]
  const lrWd = b.id(); b.emit(Op.FMul, [p.tF32, lrWd, lr, wd]);
  const decay = b.id(); b.emit(Op.FMul, [p.tF32, decay, lrWd, paramVal]);
  const p1 = b.id(); b.emit(Op.FSub, [p.tF32, p1, paramVal, decay]);

  // m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
  const mBeta = b.id(); b.emit(Op.FMul, [p.tF32, mBeta, beta1, mVal]);
  const mGrad = b.id(); b.emit(Op.FMul, [p.tF32, mGrad, oneMinusB1, gradScaled]);
  const mNew = b.id(); b.emit(Op.FAdd, [p.tF32, mNew, mBeta, mGrad]);

  // v[i] = beta2 * v[i] + (1 - beta2) * grads[i]^2
  const g2 = b.id(); b.emit(Op.FMul, [p.tF32, g2, gradScaled, gradScaled]);
  const vBeta = b.id(); b.emit(Op.FMul, [p.tF32, vBeta, beta2, vVal]);
  const vGrad = b.id(); b.emit(Op.FMul, [p.tF32, vGrad, oneMinusB2, g2]);
  const vNew = b.id(); b.emit(Op.FAdd, [p.tF32, vNew, vBeta, vGrad]);

  // Bias-corrected: mHat = mNew / bc1, vHat = vNew / bc2
  const mHat = b.id(); b.emit(Op.FDiv, [p.tF32, mHat, mNew, bc1]);
  const vHat = b.id(); b.emit(Op.FDiv, [p.tF32, vHat, vNew, bc2]);

  // sqrt(vHat) + eps
  const sqrtV = b.id(); b.emit(Op.ExtInst, [p.tF32, sqrtV, p.glslStd, GLSLstd450.Sqrt, vHat]);
  const denom = b.id(); b.emit(Op.FAdd, [p.tF32, denom, sqrtV, eps]);

  // lr * mHat / denom
  const lrMhat = b.id(); b.emit(Op.FMul, [p.tF32, lrMhat, lr, mHat]);
  const update = b.id(); b.emit(Op.FDiv, [p.tF32, update, lrMhat, denom]);

  // params[i] -= update (apply to p1 which already has weight decay applied)
  const pFinal = b.id(); b.emit(Op.FSub, [p.tF32, pFinal, p1, update]);

  // Store params[i], m[i], v[i]
  b.emit(Op.Store, [ptrP, pFinal]);
  b.emit(Op.Store, [ptrM, mNew]);
  b.emit(Op.Store, [ptrV, vNew]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);
  return b.build();
}

// ── Kernel: Transpose (stride-based 4D) ────────────────────────────────────

/**
 * B[out_idx] = A[i]  — general transpose via stride remapping.
 *
 * All shapes are padded to 4D. Push constants encode input/output strides
 * so dimension swapping is implicit in the stride layout.
 * Uses u32 push constants + u32 arithmetic for full 32-bit precision.
 *
 * Bindings: 0=A(in), 1=B(out)
 * Push constants (u32): [len, in_s0, in_s1, in_s2, in_s3, out_s0, out_s1, out_s2, out_s3]
 */
export function kernelTranspose(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufB = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, false);

  // Push constants as u32 (9 members) — custom declaration
  const numPC = 9;
  const pcMemberTypes = Array(numPC).fill(p.tU32) as number[];
  const tPCStruct = b.id();
  b.typeStruct(tPCStruct, pcMemberTypes);
  b.addDecorate(tPCStruct, Decoration.Block);
  for (let i = 0; i < numPC; i++) {
    b.addMemberDecorate(tPCStruct, i, Decoration.Offset, i * 4);
  }
  const tPtrPCStruct = b.id();
  b.typePointer(tPtrPCStruct, StorageClass.PushConstant, tPCStruct);
  const tPtrU32PC = b.id();
  b.typePointer(tPtrU32PC, StorageClass.PushConstant, p.tU32);
  const pcVar = b.id();
  b.variable(tPtrPCStruct, pcVar, StorageClass.PushConstant);

  // Index constants for accessing push constant members 3-8
  const idx3 = b.id(); b.constant(p.tU32, idx3, 3);
  const idx4 = b.id(); b.constant(p.tU32, idx4, 4);
  const idx5 = b.id(); b.constant(p.tU32, idx5, 5);
  const idx6 = b.id(); b.constant(p.tU32, idx6, 6);
  const idx7 = b.id(); b.constant(p.tU32, idx7, 7);
  const idx8 = b.id(); b.constant(p.tU32, idx8, 8);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  const labelEntry = b.id();
  const labelBody  = b.id();
  const labelEnd   = b.id();

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  b.emit(Op.Label, [labelEntry]);

  // Load global ID
  const gidVec = b.id(); b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gidX = b.id(); b.emit(Op.CompositeExtract, [p.tU32, gidX, gidVec, 0]);

  // Load len (u32)
  const ptrLen = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrLen, pcVar, p.const0u]);
  const lenU = b.id(); b.emit(Op.Load, [p.tU32, lenU, ptrLen]);

  // Bounds check in u32: if (gidX >= len) skip
  const cmp = b.id(); b.emit(Op.UGreaterThanEqual, [p.tBool, cmp, gidX, lenU]);
  b.emit(Op.SelectionMerge, [labelEnd, 0]);
  b.emit(Op.BranchConditional, [cmp, labelEnd, labelBody]);
  b.emit(Op.Label, [labelBody]);

  // Load input strides (u32)
  const ptrIS0 = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrIS0, pcVar, p.const1u]);
  const inS0 = b.id(); b.emit(Op.Load, [p.tU32, inS0, ptrIS0]);
  const ptrIS1 = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrIS1, pcVar, p.const2u]);
  const inS1 = b.id(); b.emit(Op.Load, [p.tU32, inS1, ptrIS1]);
  const ptrIS2 = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrIS2, pcVar, idx3]);
  const inS2 = b.id(); b.emit(Op.Load, [p.tU32, inS2, ptrIS2]);
  const ptrIS3 = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrIS3, pcVar, idx4]);
  const inS3 = b.id(); b.emit(Op.Load, [p.tU32, inS3, ptrIS3]);

  // Load output strides (u32)
  const ptrOS0 = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrOS0, pcVar, idx5]);
  const outS0 = b.id(); b.emit(Op.Load, [p.tU32, outS0, ptrOS0]);
  const ptrOS1 = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrOS1, pcVar, idx6]);
  const outS1 = b.id(); b.emit(Op.Load, [p.tU32, outS1, ptrOS1]);
  const ptrOS2 = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrOS2, pcVar, idx7]);
  const outS2 = b.id(); b.emit(Op.Load, [p.tU32, outS2, ptrOS2]);
  const ptrOS3 = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrOS3, pcVar, idx8]);
  const outS3 = b.id(); b.emit(Op.Load, [p.tU32, outS3, ptrOS3]);

  // Decompose flat input index (gidX) into 4D coords using input strides
  // c0 = gidX / inS0
  const c0 = b.id(); b.emit(Op.UDiv, [p.tU32, c0, gidX, inS0]);
  const c0s = b.id(); b.emit(Op.IMul, [p.tU32, c0s, c0, inS0]);
  const rem0 = b.id(); b.emit(Op.ISub, [p.tU32, rem0, gidX, c0s]);
  // c1 = rem0 / inS1
  const c1 = b.id(); b.emit(Op.UDiv, [p.tU32, c1, rem0, inS1]);
  const c1s = b.id(); b.emit(Op.IMul, [p.tU32, c1s, c1, inS1]);
  const rem1 = b.id(); b.emit(Op.ISub, [p.tU32, rem1, rem0, c1s]);
  // c2 = rem1 / inS2
  const c2 = b.id(); b.emit(Op.UDiv, [p.tU32, c2, rem1, inS2]);
  const c2s = b.id(); b.emit(Op.IMul, [p.tU32, c2s, c2, inS2]);
  const c3 = b.id(); b.emit(Op.ISub, [p.tU32, c3, rem1, c2s]);

  // Compute output flat index: c0*outS0 + c1*outS1 + c2*outS2 + c3*outS3
  const t0 = b.id(); b.emit(Op.IMul, [p.tU32, t0, c0, outS0]);
  const t1 = b.id(); b.emit(Op.IMul, [p.tU32, t1, c1, outS1]);
  const t2 = b.id(); b.emit(Op.IMul, [p.tU32, t2, c2, outS2]);
  const t3 = b.id(); b.emit(Op.IMul, [p.tU32, t3, c3, outS3]);
  const s01 = b.id(); b.emit(Op.IAdd, [p.tU32, s01, t0, t1]);
  const s012 = b.id(); b.emit(Op.IAdd, [p.tU32, s012, s01, t2]);
  const outIdx = b.id(); b.emit(Op.IAdd, [p.tU32, outIdx, s012, t3]);

  // B[outIdx] = A[gidX]
  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrF32, ptrA, bufA.varId, p.const0u, gidX]);
  const valA = b.id();
  b.emit(Op.Load, [p.tF32, valA, ptrA]);
  const ptrB = b.id();
  b.emit(Op.AccessChain, [bufB.tPtrF32, ptrB, bufB.varId, p.const0u, outIdx]);
  b.emit(Op.Store, [ptrB, valA]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

// ── Kernel: Tiled 2D Transpose (shared memory, last-2-dim swap) ──────────────

/**
 * B[batch][col][row] = A[batch][row][col] — tiled transpose for last 2 dims.
 *
 * Uses 32×32 shared memory tiles with +1 padding to avoid bank conflicts.
 * Each workgroup (256 threads) processes one tile, 4 rows per thread.
 * Coalesced reads AND writes (classic shared-memory transpose trick).
 *
 * Bindings: 0=A(in), 1=B(out)
 * Push constants (u32): [rows, cols]
 * Dispatch: [ceil(cols/32), ceil(rows/32), batch]
 */
export function kernelTranspose2DTiled(wgSize = 256): Uint32Array {
  const TILE = 32;
  const TILE_P1 = 33;
  const THREAD_ROWS = wgSize / TILE; // = 8
  const ELEMS_PER_THREAD = (TILE * TILE) / wgSize; // = 4

  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufB = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, false);

  // Push constants: {rows: u32, cols: u32}
  const tPCStruct = b.id();
  b.typeStruct(tPCStruct, [p.tU32, p.tU32]);
  b.addDecorate(tPCStruct, Decoration.Block);
  b.addMemberDecorate(tPCStruct, 0, Decoration.Offset, 0);
  b.addMemberDecorate(tPCStruct, 1, Decoration.Offset, 4);
  const tPtrPCStruct = b.id();
  b.typePointer(tPtrPCStruct, StorageClass.PushConstant, tPCStruct);
  const tPtrU32PC = b.id();
  b.typePointer(tPtrU32PC, StorageClass.PushConstant, p.tU32);
  const pcVar = b.id();
  b.variable(tPtrPCStruct, pcVar, StorageClass.PushConstant);

  // Shared memory: TILE * (TILE+1) f32
  const constSharedSize = b.id();
  b.constant(p.tU32, constSharedSize, TILE * TILE_P1);
  const tArrayShared = b.id();
  b.typeArray(tArrayShared, p.tF32, constSharedSize);
  const tPtrShared = b.id();
  b.typePointer(tPtrShared, StorageClass.Workgroup, tArrayShared);
  const tPtrSharedF32 = b.id();
  b.typePointer(tPtrSharedF32, StorageClass.Workgroup, p.tF32);
  const sharedMem = b.id();
  b.variable(tPtrShared, sharedMem, StorageClass.Workgroup);

  // Constants
  const constTile = b.id(); b.constant(p.tU32, constTile, TILE);
  const constTileP1 = b.id(); b.constant(p.tU32, constTileP1, TILE_P1);
  const const5u = b.id(); b.constant(p.tU32, const5u, 5);
  const const31u = b.id(); b.constant(p.tU32, const31u, 31);

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

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId, vWorkgroupId, vLocalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  const labelEntry = b.id();
  b.emit(Op.Label, [labelEntry]);

  // Load workgroup and local IDs
  const wgIdVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, wgIdVec, vWorkgroupId]);
  const tileColIdx = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, tileColIdx, wgIdVec, 0]);
  const tileRowIdx = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, tileRowIdx, wgIdVec, 1]);
  const batchIdx = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, batchIdx, wgIdVec, 2]);

  const lidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, lidVec, vLocalId]);
  const localIdx = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, localIdx, lidVec, 0]);

  // Thread position: tx = localIdx & 31, ty = localIdx >> 5
  const tx = b.id();
  b.emit(Op.BitwiseAnd, [p.tU32, tx, localIdx, const31u]);
  const ty = b.id();
  b.emit(Op.ShiftRightLogical, [p.tU32, ty, localIdx, const5u]);

  // Load push constants
  const ptrRows = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrRows, pcVar, p.const0u]);
  const rowsU = b.id(); b.emit(Op.Load, [p.tU32, rowsU, ptrRows]);
  const ptrCols = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrCols, pcVar, p.const1u]);
  const colsU = b.id(); b.emit(Op.Load, [p.tU32, colsU, ptrCols]);

  // Tile base positions
  const tileRowBase = b.id();
  b.emit(Op.IMul, [p.tU32, tileRowBase, tileRowIdx, constTile]);
  const tileColBase = b.id();
  b.emit(Op.IMul, [p.tU32, tileColBase, tileColIdx, constTile]);

  // Batch offset = batchIdx * rows * cols
  const batchStride = b.id();
  b.emit(Op.IMul, [p.tU32, batchStride, rowsU, colsU]);
  const batchOffset = b.id();
  b.emit(Op.IMul, [p.tU32, batchOffset, batchIdx, batchStride]);

  // ── Load phase: read input tile into shared memory (coalesced reads) ──
  for (let k = 0; k < ELEMS_PER_THREAD; k++) {
    const kOff = b.id(); b.constant(p.tU32, kOff, k * THREAD_ROWS);
    const rowInTile = b.id(); b.emit(Op.IAdd, [p.tU32, rowInTile, ty, kOff]);
    const globalRow = b.id(); b.emit(Op.IAdd, [p.tU32, globalRow, tileRowBase, rowInTile]);
    const globalCol = b.id(); b.emit(Op.IAdd, [p.tU32, globalCol, tileColBase, tx]);

    const cmpR = b.id(); b.emit(Op.ULessThan, [p.tBool, cmpR, globalRow, rowsU]);
    const cmpC = b.id(); b.emit(Op.ULessThan, [p.tBool, cmpC, globalCol, colsU]);
    const inBounds = b.id(); b.emit(Op.LogicalAnd, [p.tBool, inBounds, cmpR, cmpC]);

    const lblLoad = b.id();
    const lblAfterLoad = b.id();
    b.emit(Op.SelectionMerge, [lblAfterLoad, 0]);
    b.emit(Op.BranchConditional, [inBounds, lblLoad, lblAfterLoad]);
    b.emit(Op.Label, [lblLoad]);

    // input[batchOffset + globalRow * cols + globalCol]
    const rowTimesC = b.id(); b.emit(Op.IMul, [p.tU32, rowTimesC, globalRow, colsU]);
    const idx1 = b.id(); b.emit(Op.IAdd, [p.tU32, idx1, batchOffset, rowTimesC]);
    const inIdx = b.id(); b.emit(Op.IAdd, [p.tU32, inIdx, idx1, globalCol]);
    const ptrA = b.id(); b.emit(Op.AccessChain, [bufA.tPtrF32, ptrA, bufA.varId, p.const0u, inIdx]);
    const val = b.id(); b.emit(Op.Load, [p.tF32, val, ptrA]);

    // shared[rowInTile * TILE_P1 + tx]
    const shRow = b.id(); b.emit(Op.IMul, [p.tU32, shRow, rowInTile, constTileP1]);
    const shIdx = b.id(); b.emit(Op.IAdd, [p.tU32, shIdx, shRow, tx]);
    const ptrS = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrS, sharedMem, shIdx]);
    b.emit(Op.Store, [ptrS, val]);

    b.emit(Op.Branch, [lblAfterLoad]);
    b.emit(Op.Label, [lblAfterLoad]);
  }

  // Barrier: ensure all loads complete before transposed reads
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  // ── Store phase: write transposed data from shared to output (coalesced writes) ──
  for (let k = 0; k < ELEMS_PER_THREAD; k++) {
    const kOff = b.id(); b.constant(p.tU32, kOff, k * THREAD_ROWS);
    const rowInTile = b.id(); b.emit(Op.IAdd, [p.tU32, rowInTile, ty, kOff]);

    // Transposed output: outR = tileColBase + rowInTile, outC = tileRowBase + tx
    const outR = b.id(); b.emit(Op.IAdd, [p.tU32, outR, tileColBase, rowInTile]);
    const outC = b.id(); b.emit(Op.IAdd, [p.tU32, outC, tileRowBase, tx]);

    const cmpR = b.id(); b.emit(Op.ULessThan, [p.tBool, cmpR, outR, colsU]);
    const cmpC = b.id(); b.emit(Op.ULessThan, [p.tBool, cmpC, outC, rowsU]);
    const inBounds = b.id(); b.emit(Op.LogicalAnd, [p.tBool, inBounds, cmpR, cmpC]);

    const lblStore = b.id();
    const lblAfterStore = b.id();
    b.emit(Op.SelectionMerge, [lblAfterStore, 0]);
    b.emit(Op.BranchConditional, [inBounds, lblStore, lblAfterStore]);
    b.emit(Op.Label, [lblStore]);

    // Read transposed from shared: shared[tx * TILE_P1 + rowInTile]
    const shCol = b.id(); b.emit(Op.IMul, [p.tU32, shCol, tx, constTileP1]);
    const shIdx = b.id(); b.emit(Op.IAdd, [p.tU32, shIdx, shCol, rowInTile]);
    const ptrS = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrS, sharedMem, shIdx]);
    const val = b.id(); b.emit(Op.Load, [p.tF32, val, ptrS]);

    // output[batchOffset + outR * rows + outC]
    const outRTimesRows = b.id(); b.emit(Op.IMul, [p.tU32, outRTimesRows, outR, rowsU]);
    const idx1 = b.id(); b.emit(Op.IAdd, [p.tU32, idx1, batchOffset, outRTimesRows]);
    const outIdx = b.id(); b.emit(Op.IAdd, [p.tU32, outIdx, idx1, outC]);
    const ptrB = b.id(); b.emit(Op.AccessChain, [bufB.tPtrF32, ptrB, bufB.varId, p.const0u, outIdx]);
    b.emit(Op.Store, [ptrB, val]);

    b.emit(Op.Branch, [lblAfterStore]);
    b.emit(Op.Label, [lblAfterStore]);
  }

  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

// ── Kernel: add_inplace ─────────────────────────────────────────────────────

/**
 * In-place add: A[i] += B[i]
 *
 * Single binding for A (read-write), one for B (read-only).
 * Push: [len]
 * Bindings: 0=A(rw), 1=B(in)
 */
export function kernelAddInplace(wgSize: number): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, false); // read-write
  const bufB = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, true);  // read-only
  const pc = declareParamsPushConstant(b, p.tF32, 2);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  const labelEntry = b.id();
  b.emit(Op.Label, [labelEntry]);

  // gid = GlobalInvocationId.x
  const gidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gid = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, gid, gidVec, 0]);

  // Bounds check
  const lenF = loadPushLen(b, p, pc);
  const len = b.id();
  b.emit(Op.ConvertFToU, [p.tU32, len, lenF]);
  const inBounds = b.id();
  b.emit(Op.ULessThan, [p.tBool, inBounds, gid, len]);
  const labelDo = b.id();
  const labelEnd = b.id();
  b.emit(Op.SelectionMerge, [labelEnd, 0]);
  b.emit(Op.BranchConditional, [inBounds, labelDo, labelEnd]);

  b.emit(Op.Label, [labelDo]);
  // A[i] += B[i]
  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrF32, ptrA, bufA.varId, p.const0u, gid]);
  const aVal = b.id();
  b.emit(Op.Load, [p.tF32, aVal, ptrA]);
  const ptrB = b.id();
  b.emit(Op.AccessChain, [bufB.tPtrF32, ptrB, bufB.varId, p.const0u, gid]);
  const bVal = b.id();
  b.emit(Op.Load, [p.tF32, bVal, ptrB]);
  const sum = b.id();
  b.emit(Op.FAdd, [p.tF32, sum, aVal, bVal]);
  b.emit(Op.Store, [ptrA, sum]);
  b.emit(Op.Branch, [labelEnd]);

  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

/**
 * In-place add vec4: A[i:i+4] += B[i:i+4]
 *
 * Processes 4 elements per thread via vec4 loads/stores.
 * Push: [len] (len = total elements / 4)
 * Bindings: 0=A(rw), 1=B(in)
 */
export function kernelAddInplaceVec4(wgSize: number): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  // vec4 types
  const tVec4F32 = b.id();
  b.typeVector(tVec4F32, p.tF32, 4);

  const bufA = declareStorageBuffer(b, tVec4F32, p.tU32, 0, 0, false);
  const bufB = declareStorageBuffer(b, tVec4F32, p.tU32, 0, 1, true);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

  // Pointer types for vec4
  const tPtrBufVec4A = b.id();
  b.typePointer(tPtrBufVec4A, StorageClass.Uniform, tVec4F32);
  const tPtrBufVec4B = b.id();
  b.typePointer(tPtrBufVec4B, StorageClass.Uniform, tVec4F32);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  const labelEntry = b.id();
  b.emit(Op.Label, [labelEntry]);

  const gidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gid = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, gid, gidVec, 0]);

  const lenF = loadPushLen(b, p, pc);
  const len = b.id();
  b.emit(Op.ConvertFToU, [p.tU32, len, lenF]);
  const inBounds = b.id();
  b.emit(Op.ULessThan, [p.tBool, inBounds, gid, len]);
  const labelDo = b.id();
  const labelEnd = b.id();
  b.emit(Op.SelectionMerge, [labelEnd, 0]);
  b.emit(Op.BranchConditional, [inBounds, labelDo, labelEnd]);

  b.emit(Op.Label, [labelDo]);
  const ptrA = b.id();
  b.emit(Op.AccessChain, [tPtrBufVec4A, ptrA, bufA.varId, p.const0u, gid]);
  const aVal = b.id();
  b.emit(Op.Load, [tVec4F32, aVal, ptrA]);
  const ptrB = b.id();
  b.emit(Op.AccessChain, [tPtrBufVec4B, ptrB, bufB.varId, p.const0u, gid]);
  const bVal = b.id();
  b.emit(Op.Load, [tVec4F32, bVal, ptrB]);
  const sum = b.id();
  b.emit(Op.FAdd, [tVec4F32, sum, aVal, bVal]);
  b.emit(Op.Store, [ptrA, sum]);
  b.emit(Op.Branch, [labelEnd]);

  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

// ── Kernel: scale_inplace ───────────────────────────────────────────────────

/**
 * In-place scale: A[i] *= scalar
 *
 * Push: [len, scalar]
 * Bindings: 0=A(rw)
 */
export function kernelScaleInplace(wgSize: number): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, false); // read-write
  const pc = declareParamsPushConstant(b, p.tF32, 2);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  const labelEntry = b.id();
  b.emit(Op.Label, [labelEntry]);

  const gidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gid = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, gid, gidVec, 0]);

  const lenF = loadPushLen(b, p, pc);
  const len = b.id();
  b.emit(Op.ConvertFToU, [p.tU32, len, lenF]);
  const scalar = loadPushScalar(b, p, pc);

  const inBounds = b.id();
  b.emit(Op.ULessThan, [p.tBool, inBounds, gid, len]);
  const labelDo = b.id();
  const labelEnd = b.id();
  b.emit(Op.SelectionMerge, [labelEnd, 0]);
  b.emit(Op.BranchConditional, [inBounds, labelDo, labelEnd]);

  b.emit(Op.Label, [labelDo]);
  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrF32, ptrA, bufA.varId, p.const0u, gid]);
  const aVal = b.id();
  b.emit(Op.Load, [p.tF32, aVal, ptrA]);
  const scaled = b.id();
  b.emit(Op.FMul, [p.tF32, scaled, aVal, scalar]);
  b.emit(Op.Store, [ptrA, scaled]);
  b.emit(Op.Branch, [labelEnd]);

  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

/**
 * In-place scale vec4: A[i:i+4] *= scalar
 *
 * Push: [lenVec4, scalar]
 * Bindings: 0=A(rw)
 */
export function kernelScaleInplaceVec4(wgSize: number): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const tVec4F32 = b.id();
  b.typeVector(tVec4F32, p.tF32, 4);
  const bufA = declareStorageBuffer(b, tVec4F32, p.tU32, 0, 0, false);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

  const tPtrBufVec4A = b.id();
  b.typePointer(tPtrBufVec4A, StorageClass.Uniform, tVec4F32);

  const tVec4Scalar = b.id();
  b.typeVector(tVec4Scalar, p.tF32, 4);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  const labelEntry = b.id();
  b.emit(Op.Label, [labelEntry]);

  const gidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gid = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, gid, gidVec, 0]);

  const lenF = loadPushLen(b, p, pc);
  const len = b.id();
  b.emit(Op.ConvertFToU, [p.tU32, len, lenF]);
  const scalar = loadPushScalar(b, p, pc);

  const inBounds = b.id();
  b.emit(Op.ULessThan, [p.tBool, inBounds, gid, len]);
  const labelDo = b.id();
  const labelEnd = b.id();
  b.emit(Op.SelectionMerge, [labelEnd, 0]);
  b.emit(Op.BranchConditional, [inBounds, labelDo, labelEnd]);

  b.emit(Op.Label, [labelDo]);
  const ptrA = b.id();
  b.emit(Op.AccessChain, [tPtrBufVec4A, ptrA, bufA.varId, p.const0u, gid]);
  const aVal = b.id();
  b.emit(Op.Load, [tVec4F32, aVal, ptrA]);

  const vecScalar = b.id();
  b.emit(Op.CompositeConstruct, [tVec4Scalar, vecScalar, scalar, scalar, scalar, scalar]);
  const scaled = b.id();
  b.emit(Op.FMul, [tVec4F32, scaled, aVal, vecScalar]);
  b.emit(Op.Store, [ptrA, scaled]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}
