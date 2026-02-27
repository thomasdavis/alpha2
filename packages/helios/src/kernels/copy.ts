/**
 * kernels/copy.ts — Slice and scatter-slice GPU kernels.
 *
 * Used by grouped QKV projection and other 2D slicing operations.
 * Keeps data on GPU instead of falling back to CPU readback + re-upload.
 */

import {
  SpirVBuilder, Op, ExecutionModel, ExecutionMode, StorageClass, Decoration,
  FunctionControl,
  preamble, declareStorageBuffer,
} from "./helpers.js";

// ── Kernel: slice2D ──────────────────────────────────────────────────────────

/**
 * 2D slice: out[i] = src[srcRow * srcCols + srcCol]
 * where srcRow = outRow + startRow, srcCol = outCol + startCol.
 *
 * One thread per output element.
 * Bindings: 0=src(in), 1=out(out)
 * Push constants (u32): [totalElements, outCols, srcCols, startRow, startCol]
 */
export function kernelSlice2D(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufSrc = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufOut = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, false);

  // Push constants as u32 (5 members)
  const numPC = 5;
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

  // Index constants for push constant members 3, 4
  const idx3 = b.id(); b.constant(p.tU32, idx3, 3);
  const idx4 = b.id(); b.constant(p.tU32, idx4, 4);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  const labelEntry = b.id();
  const labelBody = b.id();
  const labelEnd = b.id();

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  b.emit(Op.Label, [labelEntry]);

  // Load global ID
  const gidVec = b.id(); b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gidX = b.id(); b.emit(Op.CompositeExtract, [p.tU32, gidX, gidVec, 0]);

  // Load len (u32)
  const ptrLen = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrLen, pcVar, p.const0u]);
  const lenU = b.id(); b.emit(Op.Load, [p.tU32, lenU, ptrLen]);

  // Bounds check: if (gidX >= len) skip
  const cmp = b.id(); b.emit(Op.UGreaterThanEqual, [p.tBool, cmp, gidX, lenU]);
  b.emit(Op.SelectionMerge, [labelEnd, 0]);
  b.emit(Op.BranchConditional, [cmp, labelEnd, labelBody]);
  b.emit(Op.Label, [labelBody]);

  // Load push constants
  const ptrOutCols = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrOutCols, pcVar, p.const1u]);
  const outCols = b.id(); b.emit(Op.Load, [p.tU32, outCols, ptrOutCols]);

  const ptrSrcCols = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrSrcCols, pcVar, p.const2u]);
  const srcCols = b.id(); b.emit(Op.Load, [p.tU32, srcCols, ptrSrcCols]);

  const ptrStartRow = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrStartRow, pcVar, idx3]);
  const startRow = b.id(); b.emit(Op.Load, [p.tU32, startRow, ptrStartRow]);

  const ptrStartCol = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrStartCol, pcVar, idx4]);
  const startCol = b.id(); b.emit(Op.Load, [p.tU32, startCol, ptrStartCol]);

  // Decompose gidX into output row, col
  // outRow = gidX / outCols, outCol = gidX - outRow * outCols
  const outRow = b.id(); b.emit(Op.UDiv, [p.tU32, outRow, gidX, outCols]);
  const outRowTimesC = b.id(); b.emit(Op.IMul, [p.tU32, outRowTimesC, outRow, outCols]);
  const outCol = b.id(); b.emit(Op.ISub, [p.tU32, outCol, gidX, outRowTimesC]);

  // srcIdx = (outRow + startRow) * srcCols + (outCol + startCol)
  const srcRow = b.id(); b.emit(Op.IAdd, [p.tU32, srcRow, outRow, startRow]);
  const srcCol = b.id(); b.emit(Op.IAdd, [p.tU32, srcCol, outCol, startCol]);
  const srcRowBase = b.id(); b.emit(Op.IMul, [p.tU32, srcRowBase, srcRow, srcCols]);
  const srcIdx = b.id(); b.emit(Op.IAdd, [p.tU32, srcIdx, srcRowBase, srcCol]);

  // out[gidX] = src[srcIdx]
  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufSrc.tPtrF32, ptrA, bufSrc.varId, p.const0u, srcIdx]);
  const valA = b.id();
  b.emit(Op.Load, [p.tF32, valA, ptrA]);

  const ptrB = b.id();
  b.emit(Op.AccessChain, [bufOut.tPtrF32, ptrB, bufOut.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrB, valA]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

// ── Kernel: slice3D ──────────────────────────────────────────────────────────

/**
 * 3D slice on contiguous tensors.
 *
 * Push constants (u32):
 * [totalElements, outD1, outD2, srcD1, srcD2, start0, start1, start2]
 */
export function kernelSlice3D(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufSrc = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufOut = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, false);

  const numPC = 8;
  const pcMemberTypes = Array(numPC).fill(p.tU32) as number[];
  const tPCStruct = b.id();
  b.typeStruct(tPCStruct, pcMemberTypes);
  b.addDecorate(tPCStruct, Decoration.Block);
  for (let i = 0; i < numPC; i++) b.addMemberDecorate(tPCStruct, i, Decoration.Offset, i * 4);
  const tPtrPCStruct = b.id(); b.typePointer(tPtrPCStruct, StorageClass.PushConstant, tPCStruct);
  const tPtrU32PC = b.id(); b.typePointer(tPtrU32PC, StorageClass.PushConstant, p.tU32);
  const pcVar = b.id(); b.variable(tPtrPCStruct, pcVar, StorageClass.PushConstant);

  const idx3 = b.id(); b.constant(p.tU32, idx3, 3);
  const idx4 = b.id(); b.constant(p.tU32, idx4, 4);
  const idx5 = b.id(); b.constant(p.tU32, idx5, 5);
  const idx6 = b.id(); b.constant(p.tU32, idx6, 6);
  const idx7 = b.id(); b.constant(p.tU32, idx7, 7);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  const labelEntry = b.id();
  const labelBody = b.id();
  const labelEnd = b.id();

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  b.emit(Op.Label, [labelEntry]);

  const gidVec = b.id(); b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gidX = b.id(); b.emit(Op.CompositeExtract, [p.tU32, gidX, gidVec, 0]);

  const ptrLen = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrLen, pcVar, p.const0u]);
  const lenU = b.id(); b.emit(Op.Load, [p.tU32, lenU, ptrLen]);

  const cmp = b.id(); b.emit(Op.UGreaterThanEqual, [p.tBool, cmp, gidX, lenU]);
  b.emit(Op.SelectionMerge, [labelEnd, 0]);
  b.emit(Op.BranchConditional, [cmp, labelEnd, labelBody]);
  b.emit(Op.Label, [labelBody]);

  const ptrOutD1 = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrOutD1, pcVar, p.const1u]);
  const outD1 = b.id(); b.emit(Op.Load, [p.tU32, outD1, ptrOutD1]);
  const ptrOutD2 = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrOutD2, pcVar, p.const2u]);
  const outD2 = b.id(); b.emit(Op.Load, [p.tU32, outD2, ptrOutD2]);
  const ptrSrcD1 = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrSrcD1, pcVar, idx3]);
  const srcD1 = b.id(); b.emit(Op.Load, [p.tU32, srcD1, ptrSrcD1]);
  const ptrSrcD2 = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrSrcD2, pcVar, idx4]);
  const srcD2 = b.id(); b.emit(Op.Load, [p.tU32, srcD2, ptrSrcD2]);
  const ptrStart0 = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrStart0, pcVar, idx5]);
  const start0 = b.id(); b.emit(Op.Load, [p.tU32, start0, ptrStart0]);
  const ptrStart1 = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrStart1, pcVar, idx6]);
  const start1 = b.id(); b.emit(Op.Load, [p.tU32, start1, ptrStart1]);
  const ptrStart2 = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrStart2, pcVar, idx7]);
  const start2 = b.id(); b.emit(Op.Load, [p.tU32, start2, ptrStart2]);

  const outD1D2 = b.id(); b.emit(Op.IMul, [p.tU32, outD1D2, outD1, outD2]);
  const out0 = b.id(); b.emit(Op.UDiv, [p.tU32, out0, gidX, outD1D2]);
  const out0Base = b.id(); b.emit(Op.IMul, [p.tU32, out0Base, out0, outD1D2]);
  const rem0 = b.id(); b.emit(Op.ISub, [p.tU32, rem0, gidX, out0Base]);
  const out1 = b.id(); b.emit(Op.UDiv, [p.tU32, out1, rem0, outD2]);
  const out1Base = b.id(); b.emit(Op.IMul, [p.tU32, out1Base, out1, outD2]);
  const out2 = b.id(); b.emit(Op.ISub, [p.tU32, out2, rem0, out1Base]);

  const src0 = b.id(); b.emit(Op.IAdd, [p.tU32, src0, out0, start0]);
  const src1 = b.id(); b.emit(Op.IAdd, [p.tU32, src1, out1, start1]);
  const src2 = b.id(); b.emit(Op.IAdd, [p.tU32, src2, out2, start2]);
  const src0Base = b.id();
  const srcD1D2 = b.id(); b.emit(Op.IMul, [p.tU32, srcD1D2, srcD1, srcD2]);
  b.emit(Op.IMul, [p.tU32, src0Base, src0, srcD1D2]);
  const src1Base = b.id(); b.emit(Op.IMul, [p.tU32, src1Base, src1, srcD2]);
  const src01 = b.id(); b.emit(Op.IAdd, [p.tU32, src01, src0Base, src1Base]);
  const srcIdx = b.id(); b.emit(Op.IAdd, [p.tU32, srcIdx, src01, src2]);

  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufSrc.tPtrF32, ptrA, bufSrc.varId, p.const0u, srcIdx]);
  const valA = b.id();
  b.emit(Op.Load, [p.tF32, valA, ptrA]);

  const ptrB = b.id();
  b.emit(Op.AccessChain, [bufOut.tPtrF32, ptrB, bufOut.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrB, valA]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);
  return b.build();
}

// ── Kernel: scatterSlice2D ───────────────────────────────────────────────────

/**
 * Backward of 2D slice: write gradient into a zeroed output at the slice position.
 *
 * For each output element at (row, col) in the full [totalRows, totalCols] output:
 *   if (row, col) falls within the slice region [startRow..startRow+sliceRows, startCol..startCol+sliceCols]:
 *     out[row * totalCols + col] = grad[(row-startRow) * sliceCols + (col-startCol)]
 *   else:
 *     out[row * totalCols + col] = 0.0
 *
 * One thread per output element.
 * Bindings: 0=grad(in), 1=out(out)
 * Push constants (u32): [totalElements, totalCols, sliceCols, startRow, startCol, sliceRows]
 */
export function kernelScatterSlice2D(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufGrad = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufOut = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, false);

  // Push constants as u32 (6 members)
  const numPC = 6;
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

  // Index constants for push constant members 3, 4, 5
  const idx3 = b.id(); b.constant(p.tU32, idx3, 3);
  const idx4 = b.id(); b.constant(p.tU32, idx4, 4);
  const idx5 = b.id(); b.constant(p.tU32, idx5, 5);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  const labelEntry = b.id();
  const labelBody = b.id();
  const labelEnd = b.id();

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  b.emit(Op.Label, [labelEntry]);

  // Load global ID
  const gidVec = b.id(); b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gidX = b.id(); b.emit(Op.CompositeExtract, [p.tU32, gidX, gidVec, 0]);

  // Load len (u32)
  const ptrLen = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrLen, pcVar, p.const0u]);
  const lenU = b.id(); b.emit(Op.Load, [p.tU32, lenU, ptrLen]);

  // Bounds check: if (gidX >= len) skip
  const cmpBounds = b.id(); b.emit(Op.UGreaterThanEqual, [p.tBool, cmpBounds, gidX, lenU]);
  b.emit(Op.SelectionMerge, [labelEnd, 0]);
  b.emit(Op.BranchConditional, [cmpBounds, labelEnd, labelBody]);
  b.emit(Op.Label, [labelBody]);

  // Load push constants
  const ptrTotalCols = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrTotalCols, pcVar, p.const1u]);
  const totalCols = b.id(); b.emit(Op.Load, [p.tU32, totalCols, ptrTotalCols]);

  const ptrSliceCols = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrSliceCols, pcVar, p.const2u]);
  const sliceCols = b.id(); b.emit(Op.Load, [p.tU32, sliceCols, ptrSliceCols]);

  const ptrStartRow = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrStartRow, pcVar, idx3]);
  const startRow = b.id(); b.emit(Op.Load, [p.tU32, startRow, ptrStartRow]);

  const ptrStartCol = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrStartCol, pcVar, idx4]);
  const startCol = b.id(); b.emit(Op.Load, [p.tU32, startCol, ptrStartCol]);

  const ptrSliceRows = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrSliceRows, pcVar, idx5]);
  const sliceRows = b.id(); b.emit(Op.Load, [p.tU32, sliceRows, ptrSliceRows]);

  // Decompose gidX into output row, col
  const row = b.id(); b.emit(Op.UDiv, [p.tU32, row, gidX, totalCols]);
  const rowTimesC = b.id(); b.emit(Op.IMul, [p.tU32, rowTimesC, row, totalCols]);
  const col = b.id(); b.emit(Op.ISub, [p.tU32, col, gidX, rowTimesC]);

  // Compute local coords (unsigned subtraction — wraps if out of range, caught by ULessThan)
  const localRow = b.id(); b.emit(Op.ISub, [p.tU32, localRow, row, startRow]);
  const localCol = b.id(); b.emit(Op.ISub, [p.tU32, localCol, col, startCol]);

  // Check bounds using unsigned comparison (handles wrap-around naturally)
  const inRowBounds = b.id(); b.emit(Op.ULessThan, [p.tBool, inRowBounds, localRow, sliceRows]);
  const inColBounds = b.id(); b.emit(Op.ULessThan, [p.tBool, inColBounds, localCol, sliceCols]);
  const inSlice = b.id(); b.emit(Op.LogicalAnd, [p.tBool, inSlice, inRowBounds, inColBounds]);

  // Compute grad index (safe even if out of bounds — clamped by Select below)
  const gradRowBase = b.id(); b.emit(Op.IMul, [p.tU32, gradRowBase, localRow, sliceCols]);
  const gradIdx = b.id(); b.emit(Op.IAdd, [p.tU32, gradIdx, gradRowBase, localCol]);

  // Clamp gradIdx to 0 if out of bounds (prevents buffer overread)
  const safeIdx = b.id(); b.emit(Op.Select, [p.tU32, safeIdx, inSlice, gradIdx, p.const0u]);

  // Load grad value
  const ptrGrad = b.id();
  b.emit(Op.AccessChain, [bufGrad.tPtrF32, ptrGrad, bufGrad.varId, p.const0u, safeIdx]);
  const gradVal = b.id();
  b.emit(Op.Load, [p.tF32, gradVal, ptrGrad]);

  // Select: inSlice ? gradVal : 0.0
  const finalVal = b.id();
  b.emit(Op.Select, [p.tF32, finalVal, inSlice, gradVal, p.const0f]);

  // Store output
  const ptrOut = b.id();
  b.emit(Op.AccessChain, [bufOut.tPtrF32, ptrOut, bufOut.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrOut, finalVal]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

// ── Kernel: scatterSlice3D ───────────────────────────────────────────────────

/**
 * Backward of 3D slice: scatter grad into a zeroed output.
 *
 * Push constants (u32):
 * [totalElements, totalD1, totalD2, sliceD0, sliceD1, sliceD2, start0, start1, start2]
 */
export function kernelScatterSlice3D(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufGrad = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufOut = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, false);

  const numPC = 9;
  const pcMemberTypes = Array(numPC).fill(p.tU32) as number[];
  const tPCStruct = b.id();
  b.typeStruct(tPCStruct, pcMemberTypes);
  b.addDecorate(tPCStruct, Decoration.Block);
  for (let i = 0; i < numPC; i++) b.addMemberDecorate(tPCStruct, i, Decoration.Offset, i * 4);
  const tPtrPCStruct = b.id(); b.typePointer(tPtrPCStruct, StorageClass.PushConstant, tPCStruct);
  const tPtrU32PC = b.id(); b.typePointer(tPtrU32PC, StorageClass.PushConstant, p.tU32);
  const pcVar = b.id(); b.variable(tPtrPCStruct, pcVar, StorageClass.PushConstant);

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
  const labelBody = b.id();
  const labelEnd = b.id();

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  b.emit(Op.Label, [labelEntry]);

  const gidVec = b.id(); b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gidX = b.id(); b.emit(Op.CompositeExtract, [p.tU32, gidX, gidVec, 0]);

  const ptrLen = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrLen, pcVar, p.const0u]);
  const lenU = b.id(); b.emit(Op.Load, [p.tU32, lenU, ptrLen]);

  const cmp = b.id(); b.emit(Op.UGreaterThanEqual, [p.tBool, cmp, gidX, lenU]);
  b.emit(Op.SelectionMerge, [labelEnd, 0]);
  b.emit(Op.BranchConditional, [cmp, labelEnd, labelBody]);
  b.emit(Op.Label, [labelBody]);

  const ptrTotalD1 = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrTotalD1, pcVar, p.const1u]);
  const totalD1 = b.id(); b.emit(Op.Load, [p.tU32, totalD1, ptrTotalD1]);
  const ptrTotalD2 = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrTotalD2, pcVar, p.const2u]);
  const totalD2 = b.id(); b.emit(Op.Load, [p.tU32, totalD2, ptrTotalD2]);

  const ptrSliceD0 = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrSliceD0, pcVar, idx3]);
  const sliceD0 = b.id(); b.emit(Op.Load, [p.tU32, sliceD0, ptrSliceD0]);
  const ptrSliceD1 = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrSliceD1, pcVar, idx4]);
  const sliceD1 = b.id(); b.emit(Op.Load, [p.tU32, sliceD1, ptrSliceD1]);
  const ptrSliceD2 = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrSliceD2, pcVar, idx5]);
  const sliceD2 = b.id(); b.emit(Op.Load, [p.tU32, sliceD2, ptrSliceD2]);
  const ptrStart0 = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrStart0, pcVar, idx6]);
  const start0 = b.id(); b.emit(Op.Load, [p.tU32, start0, ptrStart0]);
  const ptrStart1 = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrStart1, pcVar, idx7]);
  const start1 = b.id(); b.emit(Op.Load, [p.tU32, start1, ptrStart1]);
  const ptrStart2 = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrStart2, pcVar, idx8]);
  const start2 = b.id(); b.emit(Op.Load, [p.tU32, start2, ptrStart2]);

  const totalD1D2 = b.id(); b.emit(Op.IMul, [p.tU32, totalD1D2, totalD1, totalD2]);
  const out0 = b.id(); b.emit(Op.UDiv, [p.tU32, out0, gidX, totalD1D2]);
  const out0Base = b.id(); b.emit(Op.IMul, [p.tU32, out0Base, out0, totalD1D2]);
  const rem0 = b.id(); b.emit(Op.ISub, [p.tU32, rem0, gidX, out0Base]);
  const out1 = b.id(); b.emit(Op.UDiv, [p.tU32, out1, rem0, totalD2]);
  const out1Base = b.id(); b.emit(Op.IMul, [p.tU32, out1Base, out1, totalD2]);
  const out2 = b.id(); b.emit(Op.ISub, [p.tU32, out2, rem0, out1Base]);

  const local0 = b.id(); b.emit(Op.ISub, [p.tU32, local0, out0, start0]);
  const local1 = b.id(); b.emit(Op.ISub, [p.tU32, local1, out1, start1]);
  const local2 = b.id(); b.emit(Op.ISub, [p.tU32, local2, out2, start2]);

  const in0 = b.id(); b.emit(Op.ULessThan, [p.tBool, in0, local0, sliceD0]);
  const in1 = b.id(); b.emit(Op.ULessThan, [p.tBool, in1, local1, sliceD1]);
  const in2 = b.id(); b.emit(Op.ULessThan, [p.tBool, in2, local2, sliceD2]);
  const in01 = b.id(); b.emit(Op.LogicalAnd, [p.tBool, in01, in0, in1]);
  const inSlice = b.id(); b.emit(Op.LogicalAnd, [p.tBool, inSlice, in01, in2]);

  const grad0Base = b.id();
  const sliceD1D2 = b.id(); b.emit(Op.IMul, [p.tU32, sliceD1D2, sliceD1, sliceD2]);
  b.emit(Op.IMul, [p.tU32, grad0Base, local0, sliceD1D2]);
  const grad1Base = b.id(); b.emit(Op.IMul, [p.tU32, grad1Base, local1, sliceD2]);
  const grad01 = b.id(); b.emit(Op.IAdd, [p.tU32, grad01, grad0Base, grad1Base]);
  const gradIdx = b.id(); b.emit(Op.IAdd, [p.tU32, gradIdx, grad01, local2]);
  const safeIdx = b.id(); b.emit(Op.Select, [p.tU32, safeIdx, inSlice, gradIdx, p.const0u]);

  const ptrGrad = b.id();
  b.emit(Op.AccessChain, [bufGrad.tPtrF32, ptrGrad, bufGrad.varId, p.const0u, safeIdx]);
  const gradVal = b.id();
  b.emit(Op.Load, [p.tF32, gradVal, ptrGrad]);

  const finalVal = b.id();
  b.emit(Op.Select, [p.tF32, finalVal, inSlice, gradVal, p.const0f]);

  const ptrOut = b.id();
  b.emit(Op.AccessChain, [bufOut.tPtrF32, ptrOut, bufOut.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrOut, finalVal]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);
  return b.build();
}
