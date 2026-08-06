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
  preambleBDA, declareBDAPushConstants, loadBDABuffer, loadBDAElement, storeBDAElement,
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
 * Buffer-device-address sum reduction, with an output slot offset.
 *
 * X56 found that Helios's device-generated-commands path is wired to exactly one
 * pipeline (`add_bda`), so it can only carry elementwise binary ops — 10.6% of the
 * operation graph, of which 70% is the gradient-norm reduction tree. Reductions are
 * the single largest kind at 30.4% and were excluded purely because they bind
 * descriptor sets. This is the first of that tranche: same algorithm as
 * `kernelSumReduce`, but addresses arrive through push constants so the kernel needs
 * no descriptor set, which is what makes it DGC-eligible and command-buffer
 * replayable.
 *
 * The `outOffset` parameter is the second half of the design. X50 showed the
 * gradient norm spends 127 dispatches summing 128 scalars, purely because each
 * partial lives in its own buffer and the tree exists to bring them together. With
 * an output offset each tensor's reduction writes directly into slot i of one shared
 * partials buffer, and a single reduction finishes the job. That collapses the tree
 * *without* shrinking DGC's coverage — the tension X56 identified between the X49
 * and X50 fixes — because both are satisfied by the same kernel.
 *
 * Push constants: { u64 A, u64 C, u32 len, u32 outOffset }  (24 bytes)
 * No descriptor bindings.
 */
export function kernelSumReduceBDA(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preambleBDA(b, wgSize, 1, 1);

  // 2 buffer addresses + 2 u32 params (len, outOffset)
  const pc = declareBDAPushConstants(b, p.tU64, p.tU32, 2, 2);

  // Shared memory: array of wgSize floats
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

  // WorkgroupId / LocalInvocationId built-ins
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

  // len = push member 2, outOffset = push member 3
  const const2idx = b.id();
  b.constant(p.tU32, const2idx, 2);
  const ptrLen = b.id();
  b.emit(Op.AccessChain, [pc.tPtrU32, ptrLen, pc.varId, const2idx]);
  const lenU = b.id();
  b.emit(Op.Load, [p.tU32, lenU, ptrLen]);

  const const3idx = b.id();
  b.constant(p.tU32, const3idx, 3);
  const ptrOutOff = b.id();
  b.emit(Op.AccessChain, [pc.tPtrU32, ptrOutOff, pc.varId, const3idx]);
  const outOffset = b.id();
  b.emit(Op.Load, [p.tU32, outOffset, ptrOutOff]);

  const bufPtrA = loadBDABuffer(b, p, pc, 0);
  const bufPtrC = loadBDABuffer(b, p, pc, 1);

  // val = (gidX < len) ? A[gidX] : 0.0
  const inBounds = b.id();
  b.emit(Op.ULessThan, [p.tBool, inBounds, gidX, lenU]);
  const labelLoad = b.id();
  const labelAfterLoad = b.id();
  const labelOOB = b.id();
  b.emit(Op.SelectionMerge, [labelAfterLoad, 0]);
  b.emit(Op.BranchConditional, [inBounds, labelLoad, labelOOB]);

  b.emit(Op.Label, [labelLoad]);
  const loadedVal = loadBDAElement(b, p, bufPtrA, gidX);
  b.emit(Op.Branch, [labelAfterLoad]);

  b.emit(Op.Label, [labelOOB]);
  b.emit(Op.Branch, [labelAfterLoad]);

  b.emit(Op.Label, [labelAfterLoad]);
  const val = b.id();
  b.emit(Op.Phi, [p.tF32, val, loadedVal, labelLoad, p.const0f, labelOOB]);

  const ptrSharedLocal = b.id();
  b.emit(Op.AccessChain, [tPtrSharedF32, ptrSharedLocal, sharedMem, localIdx]);
  b.emit(Op.Store, [ptrSharedLocal, val]);
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  // Tree reduction in shared memory, unrolled (wgSize known at build time)
  let strideBda = wgSize >> 1;
  while (strideBda > 0) {
    const strideConst = b.id();
    b.constant(p.tU32, strideConst, strideBda);

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
    const sumBda = b.id();
    b.emit(Op.FAdd, [p.tF32, sumBda, myVal, otherVal]);
    b.emit(Op.Store, [ptrMe, sumBda]);
    b.emit(Op.Branch, [labelAfterReduce]);

    b.emit(Op.Label, [labelAfterReduce]);
    b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

    strideBda >>= 1;
  }

  // Thread 0 writes the partial sum to C[outOffset + wgId]
  const isZero = b.id();
  b.emit(Op.ULessThan, [p.tBool, isZero, localIdx, const1u_extra]);
  const labelWrite = b.id();
  const labelEndBda = b.id();
  b.emit(Op.SelectionMerge, [labelEndBda, 0]);
  b.emit(Op.BranchConditional, [isZero, labelWrite, labelEndBda]);

  b.emit(Op.Label, [labelWrite]);
  const ptrShared0 = b.id();
  b.emit(Op.AccessChain, [tPtrSharedF32, ptrShared0, sharedMem, p.const0u]);
  const partialSum = b.id();
  b.emit(Op.Load, [p.tF32, partialSum, ptrShared0]);
  const outIdx = b.id();
  b.emit(Op.IAdd, [p.tU32, outIdx, wgId, outOffset]);
  storeBDAElement(b, p, bufPtrC, outIdx, partialSum);
  b.emit(Op.Branch, [labelEndBda]);

  b.emit(Op.Label, [labelEndBda]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

/**
 * Max reduction kernel (same structure as sum, but uses FMax instead of FAdd).
 * Identity element: -inf (instead of 0).
 */
export function kernelMaxReduce(wgSize = 256, absolute = false): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufC = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, false);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

  // -Infinity constant
  const constNegInf = b.id();
  // IEEE 754: -infinity = 0xFF800000
  b.constant(p.tF32, constNegInf, 0xFF800000);
  const constPosInf = b.id();
  // IEEE 754: +infinity = 0x7F800000.  The max-absolute calibration variant
  // maps every NaN/Inf input to +Inf so a non-finite operand cannot be hidden
  // by GLSL FMax's implementation-defined NaN choice.
  b.constant(p.tF32, constPosInf, 0x7F800000);

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
  let reducedVal = loadedVal;
  if (absolute) {
    const absVal = b.id();
    b.emit(Op.ExtInst, [p.tF32, absVal, p.glslStd, GLSLstd450.FAbs, loadedVal]);
    const isNan = b.id();
    b.emit(Op.IsNan, [p.tBool, isNan, loadedVal]);
    const isInf = b.id();
    b.emit(Op.IsInf, [p.tBool, isInf, loadedVal]);
    const isNotFinite = b.id();
    b.emit(Op.LogicalOr, [p.tBool, isNotFinite, isNan, isInf]);
    reducedVal = b.id();
    b.emit(Op.Select, [p.tF32, reducedVal, isNotFinite, constPosInf, absVal]);
  }
  b.emit(Op.Branch, [labelAfterLoad]);

  b.emit(Op.Label, [labelOOB]);
  b.emit(Op.Branch, [labelAfterLoad]);

  b.emit(Op.Label, [labelAfterLoad]);
  const val = b.id();
  b.emit(Op.Phi, [p.tF32, val, reducedVal, labelLoad, constNegInf, labelOOB]);

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

// ── Kernel: GPU Sum-of-Squares Reduction (Phase 1) ───────────────────────────

/**
 * Parallel sum-of-squares reduction using shared memory.
 * Each workgroup reduces WG_SIZE elements down to 1 partial sum of squares.
 *
 * Identical to kernelSumReduce but squares each value before accumulating:
 *   val² = val * val
 *
 * Bindings: 0=A(in), 1=C(out, partial sums)
 * Push constants: { totalLen: f32, _unused: f32 }
 *
 * Each thread loads one element (or 0 if out of bounds), squares it.
 * Tree reduction in shared memory with workgroup barriers.
 * Thread 0 of each workgroup writes the partial sum of squares.
 */
export function kernelSumOfSquares(wgSize = 256): Uint32Array {
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
  // Square the loaded value: squaredVal = loadedVal * loadedVal
  const squaredVal = b.id();
  b.emit(Op.FMul, [p.tF32, squaredVal, loadedVal, loadedVal]);
  b.emit(Op.Branch, [labelAfterLoad]);

  b.emit(Op.Label, [labelOOB]);
  b.emit(Op.Branch, [labelAfterLoad]);

  b.emit(Op.Label, [labelAfterLoad]);
  const val = b.id();
  b.emit(Op.Phi, [p.tF32, val, squaredVal, labelLoad, p.const0f, labelOOB]);

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

// ── Kernel: Grid-stride sum-of-squares (fewer WGs, more work per thread) ─────

/**
 * Grid-stride sum-of-squares: each thread loops over many elements.
 * Uses N workgroups instead of N/wgSize, reducing multi-pass overhead.
 *
 * Each thread:
 *   acc = 0
 *   for i = globalId; i < totalLen; i += gridStride:
 *     acc += A[i] * A[i]
 *   shared[localId] = acc
 *   // tree reduction
 *   C[wgId] = shared[0]
 *
 * Bindings: 0=A(in), 1=C(out, partial sums)
 * Push constants: { totalLen: f32, _unused: f32 }
 */
export function kernelSumOfSquaresStride(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufC = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, false);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

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

  // Function-scope variable for loop counter
  const tPtrFnU32 = b.id();
  b.typePointer(tPtrFnU32, StorageClass.Function, p.tU32);
  const tPtrFnF32 = b.id();
  b.typePointer(tPtrFnF32, StorageClass.Function, p.tF32);

  // Built-ins
  const tPtrInputVec3 = b.id();
  b.typePointer(tPtrInputVec3, StorageClass.Input, p.tVec3U32);
  const vWorkgroupId = b.id();
  b.variable(tPtrInputVec3, vWorkgroupId, StorageClass.Input);
  b.addDecorate(vWorkgroupId, Decoration.BuiltIn, BuiltIn.WorkgroupId);
  const vLocalId = b.id();
  b.variable(tPtrInputVec3, vLocalId, StorageClass.Input);
  b.addDecorate(vLocalId, Decoration.BuiltIn, BuiltIn.LocalInvocationId);
  const vNumWorkgroups = b.id();
  b.variable(tPtrInputVec3, vNumWorkgroups, StorageClass.Input);
  b.addDecorate(vNumWorkgroups, Decoration.BuiltIn, BuiltIn.NumWorkgroups);

  // Barrier constants
  const scopeWg = b.id();
  b.constant(p.tU32, scopeWg, Scope.Workgroup);
  const semAcqRelWg = b.id();
  b.constant(p.tU32, semAcqRelWg, MemorySemantics.AcquireRelease | MemorySemantics.WorkgroupMemory);

  const const1u_extra = b.id();
  b.constant(p.tU32, const1u_extra, 1);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId, vWorkgroupId, vLocalId, vNumWorkgroups]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  const labelEntry = b.id();
  b.emit(Op.Label, [labelEntry]);

  // Function-scope variables MUST be first in the entry block (SPIR-V spec)
  const varIdx = b.id();
  b.emit(Op.Variable, [tPtrFnU32, varIdx, StorageClass.Function]);
  const varAcc = b.id();
  b.emit(Op.Variable, [tPtrFnF32, varAcc, StorageClass.Function]);

  // globalId = GlobalInvocationId.x
  const gidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const globalId = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, globalId, gidVec, 0]);

  // localIdx
  const lidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, lidVec, vLocalId]);
  const localIdx = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, localIdx, lidVec, 0]);

  // wgId
  const wgIdVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, wgIdVec, vWorkgroupId]);
  const wgId = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, wgId, wgIdVec, 0]);

  // numWgs = NumWorkgroups.x
  const nwgVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, nwgVec, vNumWorkgroups]);
  const numWgs = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, numWgs, nwgVec, 0]);

  // gridStride = numWgs * wgSize
  const gridStride = b.id();
  b.emit(Op.IMul, [p.tU32, gridStride, numWgs, constWgSize]);

  // totalLen (u32) from push constant
  const lenF = loadPushLen(b, p, pc);
  const totalLen = b.id();
  b.emit(Op.ConvertFToU, [p.tU32, totalLen, lenF]);

  // Initialize loop variables
  b.emit(Op.Store, [varIdx, globalId]);
  b.emit(Op.Store, [varAcc, p.const0f]);

  const labelLoopHead = b.id();
  const labelLoopBody = b.id();
  const labelLoopContinue = b.id();
  const labelLoopEnd = b.id();

  b.emit(Op.Branch, [labelLoopHead]);
  b.emit(Op.Label, [labelLoopHead]);
  const curIdx = b.id();
  b.emit(Op.Load, [p.tU32, curIdx, varIdx]);
  const loopCond = b.id();
  b.emit(Op.ULessThan, [p.tBool, loopCond, curIdx, totalLen]);
  b.emit(Op.LoopMerge, [labelLoopEnd, labelLoopContinue, 0]);
  b.emit(Op.BranchConditional, [loopCond, labelLoopBody, labelLoopEnd]);

  b.emit(Op.Label, [labelLoopBody]);
  // Load A[idx], square, add to acc
  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrF32, ptrA, bufA.varId, p.const0u, curIdx]);
  const aVal = b.id();
  b.emit(Op.Load, [p.tF32, aVal, ptrA]);
  const sq = b.id();
  b.emit(Op.FMul, [p.tF32, sq, aVal, aVal]);
  const curAcc = b.id();
  b.emit(Op.Load, [p.tF32, curAcc, varAcc]);
  const newAcc = b.id();
  b.emit(Op.FAdd, [p.tF32, newAcc, curAcc, sq]);
  b.emit(Op.Store, [varAcc, newAcc]);
  b.emit(Op.Branch, [labelLoopContinue]);

  // Continue block: advance idx and branch back to header
  b.emit(Op.Label, [labelLoopContinue]);
  const nextIdx = b.id();
  b.emit(Op.IAdd, [p.tU32, nextIdx, curIdx, gridStride]);
  b.emit(Op.Store, [varIdx, nextIdx]);
  b.emit(Op.Branch, [labelLoopHead]);

  b.emit(Op.Label, [labelLoopEnd]);

  // Store accumulated value to shared memory
  const finalAcc = b.id();
  b.emit(Op.Load, [p.tF32, finalAcc, varAcc]);
  const ptrSharedLocal = b.id();
  b.emit(Op.AccessChain, [tPtrSharedF32, ptrSharedLocal, sharedMem, localIdx]);
  b.emit(Op.Store, [ptrSharedLocal, finalAcc]);

  // Barrier
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  // Tree reduction in shared memory (unrolled)
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
    const sum = b.id();
    b.emit(Op.FAdd, [p.tF32, sum, myVal, otherVal]);
    b.emit(Op.Store, [ptrMe, sum]);
    b.emit(Op.Branch, [labelAfterReduce]);

    b.emit(Op.Label, [labelAfterReduce]);
    b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);
    stride >>= 1;
  }

  // Thread 0 writes partial sum
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

// ── Kernel: Fused dual column sum (for LayerNorm backward dw + db) ──────────

/**
 * Fused dual column sum: reduces two [numRows, dim] partial buffers to [dim] each.
 *   C[j] = sum_i(A[i*dim + j])
 *   D[j] = sum_i(B[i*dim + j])
 *
 * Saves one dispatch + barrier vs two separate column_sum calls.
 * Both buffers share the same index pattern, improving cache reuse.
 *
 * Bindings: 0=A(in), 1=B(in), 2=C(out), 3=D(out)
 * Push constants: { dim: f32, numRows: f32 }
 * Dispatch: (ceil(dim/wgSize), 1, 1)
 */
export function kernelColumnSumDual(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufB = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, true);
  const bufC = declareStorageBuffer(b, p.tF32, p.tU32, 0, 2, false);
  const bufD = declareStorageBuffer(b, p.tF32, p.tU32, 0, 3, false);
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
  const varAccA = b.id(); b.emit(Op.Variable, [tPtrFnF32, varAccA, StorageClass.Function]);
  const varAccB = b.id(); b.emit(Op.Variable, [tPtrFnF32, varAccB, StorageClass.Function]);

  const gidVec = b.id(); b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gidX = b.id(); b.emit(Op.CompositeExtract, [p.tU32, gidX, gidVec, 0]);

  const dimF = loadPushLen(b, p, pc);
  const dimU = b.id(); b.emit(Op.ConvertFToU, [p.tU32, dimU, dimF]);
  const numRowsF = loadPushScalar(b, p, pc);
  const numRowsU = b.id(); b.emit(Op.ConvertFToU, [p.tU32, numRowsU, numRowsF]);

  const labelEnd = b.id();
  emitBoundsCheck(b, p, dimF, gidX, labelEnd);

  b.emit(Op.Store, [varAccA, p.const0f]);
  b.emit(Op.Store, [varAccB, p.const0f]);
  b.emit(Op.Store, [varRow, p.const0u]);

  // Loop over rows, accumulating both A and B
  const lH = b.id(), lBd = b.id(), lM = b.id(), lC = b.id();
  b.emit(Op.Branch, [lH]);
  b.emit(Op.Label, [lH]);
  const curRow = b.id(); b.emit(Op.Load, [p.tU32, curRow, varRow]);
  const cmp = b.id(); b.emit(Op.ULessThan, [p.tBool, cmp, curRow, numRowsU]);
  b.emit(Op.LoopMerge, [lM, lC, 0]);
  b.emit(Op.BranchConditional, [cmp, lBd, lM]);
  b.emit(Op.Label, [lBd]);

  const roff = b.id(); b.emit(Op.IMul, [p.tU32, roff, curRow, dimU]);
  const idx = b.id(); b.emit(Op.IAdd, [p.tU32, idx, roff, gidX]);

  // Load and accumulate A
  const ptrA = b.id(); b.emit(Op.AccessChain, [bufA.tPtrF32, ptrA, bufA.varId, p.const0u, idx]);
  const valA = b.id(); b.emit(Op.Load, [p.tF32, valA, ptrA]);
  const accA = b.id(); b.emit(Op.Load, [p.tF32, accA, varAccA]);
  const nAccA = b.id(); b.emit(Op.FAdd, [p.tF32, nAccA, accA, valA]);
  b.emit(Op.Store, [varAccA, nAccA]);

  // Load and accumulate B
  const ptrB = b.id(); b.emit(Op.AccessChain, [bufB.tPtrF32, ptrB, bufB.varId, p.const0u, idx]);
  const valB = b.id(); b.emit(Op.Load, [p.tF32, valB, ptrB]);
  const accB = b.id(); b.emit(Op.Load, [p.tF32, accB, varAccB]);
  const nAccB = b.id(); b.emit(Op.FAdd, [p.tF32, nAccB, accB, valB]);
  b.emit(Op.Store, [varAccB, nAccB]);

  b.emit(Op.Branch, [lC]);
  b.emit(Op.Label, [lC]);
  const nr = b.id(); b.emit(Op.Load, [p.tU32, nr, varRow]);
  const ir = b.id(); b.emit(Op.IAdd, [p.tU32, ir, nr, p.const1u]);
  b.emit(Op.Store, [varRow, ir]);
  b.emit(Op.Branch, [lH]);

  b.emit(Op.Label, [lM]);

  // Store both results
  const fAccA = b.id(); b.emit(Op.Load, [p.tF32, fAccA, varAccA]);
  const ptrC = b.id(); b.emit(Op.AccessChain, [bufC.tPtrF32, ptrC, bufC.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrC, fAccA]);

  const fAccB = b.id(); b.emit(Op.Load, [p.tF32, fAccB, varAccB]);
  const ptrD = b.id(); b.emit(Op.AccessChain, [bufD.tPtrF32, ptrD, bufD.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrD, fAccB]);

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

/**
 * Row-parallel column sum for tall RMSNorm partial-gradient matrices.
 *
 * The original column_sum assigns one thread to one column. At Alpha's
 * [24576, 512] shape that exposes only 512 useful threads while each walks all
 * rows. This kernel assigns eight row lanes to each of 32 adjacent columns,
 * preserving coalesced reads while exposing eight times more row parallelism.
 * Shared-memory reduction combines the row-lane partials without atomics or a
 * subgroup-size assumption, so the algorithm remains wave32/wave64 portable.
 *
 * Bindings: 0=A(in), 1=C(out)
 * Push constants: { dim: f32, numRows: f32 }
 * Dispatch: (ceil(dim/32), 1, 1), local size [32,8,1]
 */
export function kernelColumnSumRowLanes(
  columnsPerGroup = 32,
  rowLanes = 8,
): Uint32Array {
  if (rowLanes <= 0 || (rowLanes & (rowLanes - 1)) !== 0) {
    throw new Error("column-sum rowLanes must be a positive power of two");
  }

  const b = new SpirVBuilder();
  const p = preamble(b, columnsPerGroup, rowLanes, 1);
  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufC = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, false);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

  const constColumns = b.id(); b.constant(p.tU32, constColumns, columnsPerGroup);
  const constRowLanes = b.id(); b.constant(p.tU32, constRowLanes, rowLanes);
  const constSharedSize = b.id();
  b.constant(p.tU32, constSharedSize, columnsPerGroup * rowLanes);
  const tArrayShared = b.id(); b.typeArray(tArrayShared, p.tF32, constSharedSize);
  const tPtrShared = b.id();
  b.typePointer(tPtrShared, StorageClass.Workgroup, tArrayShared);
  const tPtrSharedF32 = b.id();
  b.typePointer(tPtrSharedF32, StorageClass.Workgroup, p.tF32);
  const shared = b.id(); b.variable(tPtrShared, shared, StorageClass.Workgroup);

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
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [vWorkgroupId, vLocalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, columnsPerGroup, rowLanes, 1);
  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  const labelEntry = b.id(); b.emit(Op.Label, [labelEntry]);

  const varRow = b.id(); b.emit(Op.Variable, [tPtrFnU32, varRow, StorageClass.Function]);
  const varAcc = b.id(); b.emit(Op.Variable, [tPtrFnF32, varAcc, StorageClass.Function]);

  const localIdVec = b.id(); b.emit(Op.Load, [p.tVec3U32, localIdVec, vLocalId]);
  const columnLane = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, columnLane, localIdVec, 0]);
  const rowLane = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, rowLane, localIdVec, 1]);
  const workgroupIdVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, workgroupIdVec, vWorkgroupId]);
  const workgroupX = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, workgroupX, workgroupIdVec, 0]);
  const groupColumnBase = b.id();
  b.emit(Op.IMul, [p.tU32, groupColumnBase, workgroupX, constColumns]);
  const column = b.id();
  b.emit(Op.IAdd, [p.tU32, column, groupColumnBase, columnLane]);

  const dimF = loadPushLen(b, p, pc);
  const dim = b.id(); b.emit(Op.ConvertFToU, [p.tU32, dim, dimF]);
  const numRowsF = loadPushScalar(b, p, pc);
  const numRows = b.id(); b.emit(Op.ConvertFToU, [p.tU32, numRows, numRowsF]);
  const columnInBounds = b.id();
  b.emit(Op.ULessThan, [p.tBool, columnInBounds, column, dim]);

  b.emit(Op.Store, [varAcc, p.const0f]);
  b.emit(Op.Store, [varRow, rowLane]);
  const loopHead = b.id(), loopBody = b.id(), loopMerge = b.id(), loopContinue = b.id();
  b.emit(Op.Branch, [loopHead]);
  b.emit(Op.Label, [loopHead]);
  const currentRow = b.id(); b.emit(Op.Load, [p.tU32, currentRow, varRow]);
  const rowInBounds = b.id();
  b.emit(Op.ULessThan, [p.tBool, rowInBounds, currentRow, numRows]);
  const loadInBounds = b.id();
  b.emit(Op.LogicalAnd, [p.tBool, loadInBounds, rowInBounds, columnInBounds]);
  b.emit(Op.LoopMerge, [loopMerge, loopContinue, 0]);
  b.emit(Op.BranchConditional, [loadInBounds, loopBody, loopMerge]);

  b.emit(Op.Label, [loopBody]);
  const rowOffset = b.id(); b.emit(Op.IMul, [p.tU32, rowOffset, currentRow, dim]);
  const inputIndex = b.id(); b.emit(Op.IAdd, [p.tU32, inputIndex, rowOffset, column]);
  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrF32, ptrA, bufA.varId, p.const0u, inputIndex]);
  const value = b.id(); b.emit(Op.Load, [p.tF32, value, ptrA]);
  const accumulator = b.id(); b.emit(Op.Load, [p.tF32, accumulator, varAcc]);
  const nextAccumulator = b.id();
  b.emit(Op.FAdd, [p.tF32, nextAccumulator, accumulator, value]);
  b.emit(Op.Store, [varAcc, nextAccumulator]);
  b.emit(Op.Branch, [loopContinue]);

  b.emit(Op.Label, [loopContinue]);
  const previousRow = b.id(); b.emit(Op.Load, [p.tU32, previousRow, varRow]);
  const nextRow = b.id();
  b.emit(Op.IAdd, [p.tU32, nextRow, previousRow, constRowLanes]);
  b.emit(Op.Store, [varRow, nextRow]);
  b.emit(Op.Branch, [loopHead]);

  b.emit(Op.Label, [loopMerge]);
  const rowSharedBase = b.id();
  b.emit(Op.IMul, [p.tU32, rowSharedBase, rowLane, constColumns]);
  const sharedIndex = b.id();
  b.emit(Op.IAdd, [p.tU32, sharedIndex, rowSharedBase, columnLane]);
  const ptrShared = b.id();
  b.emit(Op.AccessChain, [tPtrSharedF32, ptrShared, shared, sharedIndex]);
  const finalAccumulator = b.id(); b.emit(Op.Load, [p.tF32, finalAccumulator, varAcc]);
  b.emit(Op.Store, [ptrShared, finalAccumulator]);
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  // Reduce along the row-lane dimension. Every invocation participates in
  // every barrier, including out-of-bounds columns in the final workgroup.
  for (let stride = rowLanes >> 1; stride > 0; stride >>= 1) {
    const constStride = b.id(); b.constant(p.tU32, constStride, stride);
    const isReducingLane = b.id();
    b.emit(Op.ULessThan, [p.tBool, isReducingLane, rowLane, constStride]);
    const reduceLabel = b.id(), reduceMerge = b.id();
    b.emit(Op.SelectionMerge, [reduceMerge, 0]);
    b.emit(Op.BranchConditional, [isReducingLane, reduceLabel, reduceMerge]);
    b.emit(Op.Label, [reduceLabel]);
    const otherLane = b.id();
    b.emit(Op.IAdd, [p.tU32, otherLane, rowLane, constStride]);
    const otherBase = b.id();
    b.emit(Op.IMul, [p.tU32, otherBase, otherLane, constColumns]);
    const otherIndex = b.id();
    b.emit(Op.IAdd, [p.tU32, otherIndex, otherBase, columnLane]);
    const ptrMine = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF32, ptrMine, shared, sharedIndex]);
    const ptrOther = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF32, ptrOther, shared, otherIndex]);
    const mine = b.id(); b.emit(Op.Load, [p.tF32, mine, ptrMine]);
    const other = b.id(); b.emit(Op.Load, [p.tF32, other, ptrOther]);
    const combined = b.id(); b.emit(Op.FAdd, [p.tF32, combined, mine, other]);
    b.emit(Op.Store, [ptrMine, combined]);
    b.emit(Op.Branch, [reduceMerge]);
    b.emit(Op.Label, [reduceMerge]);
    b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);
  }

  const rowLaneIsZero = b.id();
  b.emit(Op.IEqual, [p.tBool, rowLaneIsZero, rowLane, p.const0u]);
  const shouldWrite = b.id();
  b.emit(Op.LogicalAnd, [p.tBool, shouldWrite, rowLaneIsZero, columnInBounds]);
  const writeLabel = b.id(), endLabel = b.id();
  b.emit(Op.SelectionMerge, [endLabel, 0]);
  b.emit(Op.BranchConditional, [shouldWrite, writeLabel, endLabel]);
  b.emit(Op.Label, [writeLabel]);
  const ptrResult = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrF32, ptrResult, bufC.varId, p.const0u, column]);
  const reduced = b.id(); b.emit(Op.Load, [p.tF32, reduced, ptrShared]);
  b.emit(Op.Store, [ptrResult, reduced]);
  b.emit(Op.Branch, [endLabel]);
  b.emit(Op.Label, [endLabel]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);
  return b.build();
}
