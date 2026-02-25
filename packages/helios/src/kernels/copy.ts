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
