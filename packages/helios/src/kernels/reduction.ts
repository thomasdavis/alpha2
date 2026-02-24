/**
 * kernels/reduction.ts — Reduction GPU kernels.
 *
 * Sum reduce, max reduce, column sum, axis-specific sum reduction.
 */

import {
  SpirVBuilder, Op, ExecutionModel, ExecutionMode, StorageClass, Decoration,
  BuiltIn, FunctionControl, GLSLstd450, Scope, MemorySemantics,
  preamble, declareStorageBuffer, declareParamsPushConstant,
  loadPushLen, loadPushScalar, emitBoundsCheck,
} from "./helpers.js";

// ── Kernel: GPU Sum Reduction (Phase 1) ─────────────────────────────────────

/**
 * Parallel sum reduction using shared memory.
 * Each workgroup reduces WG_SIZE elements down to 1 partial sum.
 *
 * Bindings: 0=A(in), 1=C(out, partial sums)
 * Push constants: { totalLen: f32, _unused: f32 }
 *
 * Each thread loads one element (or 0 if out of bounds).
 * Tree reduction in shared memory with workgroup barriers.
 * Thread 0 of each workgroup writes the partial sum.
 */
export function kernelSumReduce(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufC = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, false);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

  // Shared memory: array of WG_SIZE floats
  const constWgSize = b.id();
  b.constant(p.tU32, constWgSize, wgSize);
  const tArrayShared = b.id();
  b.typeArray(tArrayShared, p.tF32, constWgSize);
  const tPtrShared = b.id();
  b.typePointer(tPtrShared, StorageClass.Workgroup, tArrayShared);
  const tPtrSharedF32 = b.id();
  b.typePointer(tPtrSharedF32, StorageClass.Workgroup, p.tF32);
  const sharedMem = b.id();
  b.variable(tPtrShared, sharedMem, StorageClass.Workgroup);

  // Pointer for Function-scope variable (loop counter)
  const tPtrFnU32 = b.id();
  b.typePointer(tPtrFnU32, StorageClass.Function, p.tU32);

  // WorkgroupId built-in
  const tPtrInputVec3 = b.id();
  b.typePointer(tPtrInputVec3, StorageClass.Input, p.tVec3U32);
  const vWorkgroupId = b.id();
  b.variable(tPtrInputVec3, vWorkgroupId, StorageClass.Input);
  b.addDecorate(vWorkgroupId, Decoration.BuiltIn, BuiltIn.WorkgroupId);

  // LocalInvocationId built-in
  const vLocalId = b.id();
  b.variable(tPtrInputVec3, vLocalId, StorageClass.Input);
  b.addDecorate(vLocalId, Decoration.BuiltIn, BuiltIn.LocalInvocationId);

  // Scope/semantics constants for barrier
  const scopeWg = b.id();
  b.constant(p.tU32, scopeWg, Scope.Workgroup);
  const semAcqRelWg = b.id();
  b.constant(p.tU32, semAcqRelWg, MemorySemantics.AcquireRelease | MemorySemantics.WorkgroupMemory);

  // Additional int constants
  const const1u_extra = b.id();
  b.constant(p.tU32, const1u_extra, 1);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId, vWorkgroupId, vLocalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  const labelEntry = b.id();
  b.emit(Op.Label, [labelEntry]);

  // gidX = GlobalInvocationId.x
  const gidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gidX = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, gidX, gidVec, 0]);

  // localIdx = LocalInvocationId.x
  const lidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, lidVec, vLocalId]);
  const localIdx = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, localIdx, lidVec, 0]);

  // wgId = WorkgroupId.x
  const wgIdVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, wgIdVec, vWorkgroupId]);
  const wgId = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, wgId, wgIdVec, 0]);

  // Load total length from push constants
  const lenF = loadPushLen(b, p, pc);

  // Load value: val = (gidX < len) ? A[gidX] : 0.0
  const gidF = b.id();
  b.emit(Op.ConvertUToF, [p.tF32, gidF, gidX]);
  const inBounds = b.id();
  b.emit(Op.FOrdLessThan, [p.tBool, inBounds, gidF, lenF]);
  const labelLoad = b.id();
  const labelAfterLoad = b.id();
  const labelOOB = b.id();
  b.emit(Op.SelectionMerge, [labelAfterLoad, 0]);
  b.emit(Op.BranchConditional, [inBounds, labelLoad, labelOOB]);

  b.emit(Op.Label, [labelLoad]);
  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrF32, ptrA, bufA.varId, p.const0u, gidX]);
  const loadedVal = b.id();
  b.emit(Op.Load, [p.tF32, loadedVal, ptrA]);
  b.emit(Op.Branch, [labelAfterLoad]);

  b.emit(Op.Label, [labelOOB]);
  b.emit(Op.Branch, [labelAfterLoad]);

  b.emit(Op.Label, [labelAfterLoad]);
  const val = b.id();
  b.emit(Op.Phi, [p.tF32, val, loadedVal, labelLoad, p.const0f, labelOOB]);

  // Store to shared memory: shared[localIdx] = val
  const ptrSharedLocal = b.id();
  b.emit(Op.AccessChain, [tPtrSharedF32, ptrSharedLocal, sharedMem, localIdx]);
  b.emit(Op.Store, [ptrSharedLocal, val]);

  // Workgroup barrier
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  // Tree reduction: for (stride = wgSize/2; stride > 0; stride >>= 1)
  // Unroll the loop since wgSize is known at compile time
  let stride = wgSize >> 1;
  while (stride > 0) {
    const strideConst = b.id();
    b.constant(p.tU32, strideConst, stride);

    const cmp = b.id();
    b.emit(Op.ULessThan, [p.tBool, cmp, localIdx, strideConst]);
    const labelReduce = b.id();
    const labelAfterReduce = b.id();
    b.emit(Op.SelectionMerge, [labelAfterReduce, 0]);
    b.emit(Op.BranchConditional, [cmp, labelReduce, labelAfterReduce]);

    b.emit(Op.Label, [labelReduce]);
    // shared[localIdx] += shared[localIdx + stride]
    const otherIdx = b.id();
    b.emit(Op.IAdd, [p.tU32, otherIdx, localIdx, strideConst]);
    const ptrMe = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF32, ptrMe, sharedMem, localIdx]);
    const myVal = b.id();
    b.emit(Op.Load, [p.tF32, myVal, ptrMe]);
    const ptrOther = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF32, ptrOther, sharedMem, otherIdx]);
    const otherVal = b.id();
    b.emit(Op.Load, [p.tF32, otherVal, ptrOther]);
    const sum = b.id();
    b.emit(Op.FAdd, [p.tF32, sum, myVal, otherVal]);
    b.emit(Op.Store, [ptrMe, sum]);
    b.emit(Op.Branch, [labelAfterReduce]);

    b.emit(Op.Label, [labelAfterReduce]);
    b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

    stride >>= 1;
  }

  // Thread 0 writes the partial sum to output
  const isZero = b.id();
  b.emit(Op.ULessThan, [p.tBool, isZero, localIdx, const1u_extra]);
  const labelWrite = b.id();
  const labelEnd = b.id();
  b.emit(Op.SelectionMerge, [labelEnd, 0]);
  b.emit(Op.BranchConditional, [isZero, labelWrite, labelEnd]);

  b.emit(Op.Label, [labelWrite]);
  const ptrShared0 = b.id();
  b.emit(Op.AccessChain, [tPtrSharedF32, ptrShared0, sharedMem, p.const0u]);
  const partialSum = b.id();
  b.emit(Op.Load, [p.tF32, partialSum, ptrShared0]);
  const ptrC = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrF32, ptrC, bufC.varId, p.const0u, wgId]);
  b.emit(Op.Store, [ptrC, partialSum]);
  b.emit(Op.Branch, [labelEnd]);

  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

/**
 * Max reduction kernel (same structure as sum, but uses FMax instead of FAdd).
 * Identity element: -inf (instead of 0).
 */
export function kernelMaxReduce(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufC = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, false);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

  // -Infinity constant
  const constNegInf = b.id();
  // IEEE 754: -infinity = 0xFF800000
  b.constant(p.tF32, constNegInf, 0xFF800000);

  // Shared memory
  const constWgSize = b.id();
  b.constant(p.tU32, constWgSize, wgSize);
  const tArrayShared = b.id();
  b.typeArray(tArrayShared, p.tF32, constWgSize);
  const tPtrShared = b.id();
  b.typePointer(tPtrShared, StorageClass.Workgroup, tArrayShared);
  const tPtrSharedF32 = b.id();
  b.typePointer(tPtrSharedF32, StorageClass.Workgroup, p.tF32);
  const sharedMem = b.id();
  b.variable(tPtrShared, sharedMem, StorageClass.Workgroup);

  const tPtrFnU32 = b.id();
  b.typePointer(tPtrFnU32, StorageClass.Function, p.tU32);

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
  const const1u_extra = b.id();
  b.constant(p.tU32, const1u_extra, 1);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId, vWorkgroupId, vLocalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  const labelEntry = b.id();
  b.emit(Op.Label, [labelEntry]);

  const gidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gidX = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, gidX, gidVec, 0]);
  const lidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, lidVec, vLocalId]);
  const localIdx = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, localIdx, lidVec, 0]);
  const wgIdVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, wgIdVec, vWorkgroupId]);
  const wgId = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, wgId, wgIdVec, 0]);

  const lenF = loadPushLen(b, p, pc);

  // Load: val = (gidX < len) ? A[gidX] : -inf
  const gidF = b.id();
  b.emit(Op.ConvertUToF, [p.tF32, gidF, gidX]);
  const inBounds = b.id();
  b.emit(Op.FOrdLessThan, [p.tBool, inBounds, gidF, lenF]);
  const labelLoad = b.id();
  const labelAfterLoad = b.id();
  const labelOOB = b.id();
  b.emit(Op.SelectionMerge, [labelAfterLoad, 0]);
  b.emit(Op.BranchConditional, [inBounds, labelLoad, labelOOB]);

  b.emit(Op.Label, [labelLoad]);
  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrF32, ptrA, bufA.varId, p.const0u, gidX]);
  const loadedVal = b.id();
  b.emit(Op.Load, [p.tF32, loadedVal, ptrA]);
  b.emit(Op.Branch, [labelAfterLoad]);

  b.emit(Op.Label, [labelOOB]);
  b.emit(Op.Branch, [labelAfterLoad]);

  b.emit(Op.Label, [labelAfterLoad]);
  const val = b.id();
  b.emit(Op.Phi, [p.tF32, val, loadedVal, labelLoad, constNegInf, labelOOB]);

  // Store to shared memory
  const ptrSharedLocal = b.id();
  b.emit(Op.AccessChain, [tPtrSharedF32, ptrSharedLocal, sharedMem, localIdx]);
  b.emit(Op.Store, [ptrSharedLocal, val]);
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  // Tree reduction using FMax
  let stride = wgSize >> 1;
  while (stride > 0) {
    const strideConst = b.id();
    b.constant(p.tU32, strideConst, stride);
    const cmp = b.id();
    b.emit(Op.ULessThan, [p.tBool, cmp, localIdx, strideConst]);
    const labelReduce = b.id();
    const labelAfterReduce = b.id();
    b.emit(Op.SelectionMerge, [labelAfterReduce, 0]);
    b.emit(Op.BranchConditional, [cmp, labelReduce, labelAfterReduce]);

    b.emit(Op.Label, [labelReduce]);
    const otherIdx = b.id();
    b.emit(Op.IAdd, [p.tU32, otherIdx, localIdx, strideConst]);
    const ptrMe = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF32, ptrMe, sharedMem, localIdx]);
    const myVal = b.id();
    b.emit(Op.Load, [p.tF32, myVal, ptrMe]);
    const ptrOther = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF32, ptrOther, sharedMem, otherIdx]);
    const otherVal = b.id();
    b.emit(Op.Load, [p.tF32, otherVal, ptrOther]);
    const maxVal = b.id();
    b.emit(Op.ExtInst, [p.tF32, maxVal, p.glslStd, GLSLstd450.FMax, myVal, otherVal]);
    b.emit(Op.Store, [ptrMe, maxVal]);
    b.emit(Op.Branch, [labelAfterReduce]);

    b.emit(Op.Label, [labelAfterReduce]);
    b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);
    stride >>= 1;
  }

  // Thread 0 writes
  const isZero = b.id();
  b.emit(Op.ULessThan, [p.tBool, isZero, localIdx, const1u_extra]);
  const labelWrite = b.id();
  const labelEnd = b.id();
  b.emit(Op.SelectionMerge, [labelEnd, 0]);
  b.emit(Op.BranchConditional, [isZero, labelWrite, labelEnd]);

  b.emit(Op.Label, [labelWrite]);
  const ptrShared0 = b.id();
  b.emit(Op.AccessChain, [tPtrSharedF32, ptrShared0, sharedMem, p.const0u]);
  const partialMax = b.id();
  b.emit(Op.Load, [p.tF32, partialMax, ptrShared0]);
  const ptrC = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrF32, ptrC, bufC.varId, p.const0u, wgId]);
  b.emit(Op.Store, [ptrC, partialMax]);
  b.emit(Op.Branch, [labelEnd]);

  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

// ── Kernel: Column sum (reduce axis 0 of a 2D buffer) ────────────────────────

/**
 * C[j] = sum_i(A[i*dim + j]) for j = 0..dim-1
 * Each thread handles one column, sums over all rows.
 *
 * Bindings: 0=A(in), 1=C(out)
 * Push constants: { dim: f32, numRows: f32 }
 * Dispatch: (ceil(dim/wgSize), 1, 1)
 */
export function kernelColumnSum(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufC = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, false);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

  const tPtrFnU32 = b.id(); b.typePointer(tPtrFnU32, StorageClass.Function, p.tU32);
  const tPtrFnF32 = b.id(); b.typePointer(tPtrFnF32, StorageClass.Function, p.tF32);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  const labelEntry = b.id();
  b.emit(Op.Label, [labelEntry]);

  const varRow = b.id(); b.emit(Op.Variable, [tPtrFnU32, varRow, StorageClass.Function]);
  const varAcc = b.id(); b.emit(Op.Variable, [tPtrFnF32, varAcc, StorageClass.Function]);

  const gidVec = b.id(); b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gidX = b.id(); b.emit(Op.CompositeExtract, [p.tU32, gidX, gidVec, 0]);

  const dimF = loadPushLen(b, p, pc);
  const dimU = b.id(); b.emit(Op.ConvertFToU, [p.tU32, dimU, dimF]);
  const numRowsF = loadPushScalar(b, p, pc);
  const numRowsU = b.id(); b.emit(Op.ConvertFToU, [p.tU32, numRowsU, numRowsF]);

  const labelEnd = b.id();
  emitBoundsCheck(b, p, dimF, gidX, labelEnd);

  b.emit(Op.Store, [varAcc, p.const0f]);
  b.emit(Op.Store, [varRow, p.const0u]);

  const lH = b.id(), lB = b.id(), lM = b.id(), lC = b.id();
  b.emit(Op.Branch, [lH]);
  b.emit(Op.Label, [lH]);
  const curRow = b.id(); b.emit(Op.Load, [p.tU32, curRow, varRow]);
  const cmp = b.id(); b.emit(Op.ULessThan, [p.tBool, cmp, curRow, numRowsU]);
  b.emit(Op.LoopMerge, [lM, lC, 0]);
  b.emit(Op.BranchConditional, [cmp, lB, lM]);
  b.emit(Op.Label, [lB]);

  const roff = b.id(); b.emit(Op.IMul, [p.tU32, roff, curRow, dimU]);
  const idx = b.id(); b.emit(Op.IAdd, [p.tU32, idx, roff, gidX]);
  const ptr = b.id(); b.emit(Op.AccessChain, [bufA.tPtrF32, ptr, bufA.varId, p.const0u, idx]);
  const val = b.id(); b.emit(Op.Load, [p.tF32, val, ptr]);
  const acc = b.id(); b.emit(Op.Load, [p.tF32, acc, varAcc]);
  const nAcc = b.id(); b.emit(Op.FAdd, [p.tF32, nAcc, acc, val]);
  b.emit(Op.Store, [varAcc, nAcc]);
  b.emit(Op.Branch, [lC]);

  b.emit(Op.Label, [lC]);
  const nr = b.id(); b.emit(Op.Load, [p.tU32, nr, varRow]);
  const ir = b.id(); b.emit(Op.IAdd, [p.tU32, ir, nr, p.const1u]);
  b.emit(Op.Store, [varRow, ir]);
  b.emit(Op.Branch, [lH]);

  b.emit(Op.Label, [lM]);
  const fAcc = b.id(); b.emit(Op.Load, [p.tF32, fAcc, varAcc]);
  const ptrC = b.id(); b.emit(Op.AccessChain, [bufC.tPtrF32, ptrC, bufC.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrC, fAcc]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);
  return b.build();
}

// ── Kernel: Axis-specific sum reduction ─────────────────────────────────────

/**
 * B[i] = sum over axis dimension of A.
 *
 * Tensor is viewed as [outerSize, axisSize, innerSize] where:
 *   outerSize = product of dims before axis
 *   innerSize = product of dims after axis
 * Each thread computes one output element by summing axisSize values.
 *
 * Bindings: 0=A(in), 1=B(out)
 * Push constants (u32): [totalOutput, axisSize, innerSize]
 */
export function kernelSumAxis(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufB = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, false);

  // Push constants as u32 (3 members)
  const numPC = 3;
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

  // Load push constants
  const ptrPC0 = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrPC0, pcVar, p.const0u]);
  const totalOutput = b.id(); b.emit(Op.Load, [p.tU32, totalOutput, ptrPC0]);
  const ptrPC1 = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrPC1, pcVar, p.const1u]);
  const axisSize = b.id(); b.emit(Op.Load, [p.tU32, axisSize, ptrPC1]);
  const ptrPC2 = b.id(); b.emit(Op.AccessChain, [tPtrU32PC, ptrPC2, pcVar, p.const2u]);
  const innerSize = b.id(); b.emit(Op.Load, [p.tU32, innerSize, ptrPC2]);

  // Bounds check: if (gidX >= totalOutput) skip
  const cmp = b.id(); b.emit(Op.UGreaterThanEqual, [p.tBool, cmp, gidX, totalOutput]);
  b.emit(Op.SelectionMerge, [labelEnd, 0]);
  b.emit(Op.BranchConditional, [cmp, labelEnd, labelBody]);
  b.emit(Op.Label, [labelBody]);

  // outer = gidX / innerSize
  const outer = b.id(); b.emit(Op.UDiv, [p.tU32, outer, gidX, innerSize]);
  // inner = gidX - outer * innerSize
  const outerTimesInner = b.id(); b.emit(Op.IMul, [p.tU32, outerTimesInner, outer, innerSize]);
  const inner = b.id(); b.emit(Op.ISub, [p.tU32, inner, gidX, outerTimesInner]);
  // strideAx = axisSize * innerSize
  const strideAx = b.id(); b.emit(Op.IMul, [p.tU32, strideAx, axisSize, innerSize]);
  // base = outer * strideAx + inner
  const outerTimesStride = b.id(); b.emit(Op.IMul, [p.tU32, outerTimesStride, outer, strideAx]);
  const base = b.id(); b.emit(Op.IAdd, [p.tU32, base, outerTimesStride, inner]);

  // Loop: sum = 0; for j = 0..axisSize-1: sum += A[base + j * innerSize]
  // Pre-allocate forward reference IDs for Phi operands
  const jNext = b.id();
  const sumNext = b.id();

  const labelLoopHeader = b.id();
  const labelLoopBody = b.id();
  const labelLoopContinue = b.id();
  const labelLoopEnd = b.id();

  b.emit(Op.Branch, [labelLoopHeader]);
  b.emit(Op.Label, [labelLoopHeader]);

  // Phi nodes: j (u32), sum (f32) — forward refs to jNext/sumNext
  const phiJ = b.id();
  b.emit(Op.Phi, [p.tU32, phiJ, p.const0u, labelBody, jNext, labelLoopContinue]);
  const phiSum = b.id();
  b.emit(Op.Phi, [p.tF32, phiSum, p.const0f, labelBody, sumNext, labelLoopContinue]);

  // Loop condition: j < axisSize
  const loopCond = b.id(); b.emit(Op.ULessThan, [p.tBool, loopCond, phiJ, axisSize]);
  b.emit(Op.LoopMerge, [labelLoopEnd, labelLoopContinue, 0]);
  b.emit(Op.BranchConditional, [loopCond, labelLoopBody, labelLoopEnd]);

  b.emit(Op.Label, [labelLoopBody]);

  // idx = base + j * innerSize
  const jTimesInner = b.id(); b.emit(Op.IMul, [p.tU32, jTimesInner, phiJ, innerSize]);
  const idx = b.id(); b.emit(Op.IAdd, [p.tU32, idx, base, jTimesInner]);
  // Load A[idx], accumulate sum
  const ptrA = b.id(); b.emit(Op.AccessChain, [bufA.tPtrF32, ptrA, bufA.varId, p.const0u, idx]);
  const valA = b.id(); b.emit(Op.Load, [p.tF32, valA, ptrA]);
  b.emit(Op.FAdd, [p.tF32, sumNext, phiSum, valA]);

  b.emit(Op.Branch, [labelLoopContinue]);
  b.emit(Op.Label, [labelLoopContinue]);

  // j++
  b.emit(Op.IAdd, [p.tU32, jNext, phiJ, p.const1u]);

  b.emit(Op.Branch, [labelLoopHeader]);
  b.emit(Op.Label, [labelLoopEnd]);

  // Store result: B[gidX] = sum
  const ptrB = b.id(); b.emit(Op.AccessChain, [bufB.tPtrF32, ptrB, bufB.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrB, phiSum]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}
