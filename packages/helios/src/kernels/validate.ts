/**
 * kernels/validate.ts — GPU validation kernels.
 *
 * checkFinite: parallel reduction to detect any Inf/NaN values in a buffer.
 * Output: scalar f32, 0.0 = all finite, 1.0 = contains Inf/NaN.
 */

import {
  SpirVBuilder, Op, ExecutionModel, ExecutionMode, StorageClass, Decoration,
  BuiltIn, FunctionControl, Scope, MemorySemantics,
  preamble, declareStorageBuffer, declareParamsPushConstant,
  loadPushLen,
} from "./helpers.js";

/**
 * checkFinite kernel: checks if any element in input is Inf or NaN.
 *
 * Uses OpIsNan (156) and OpIsInf (157) SPIR-V instructions.
 * Each workgroup reduces to a single flag via shared memory.
 * Output: scalar f32, 0.0 = all finite, 1.0 = contains Inf/NaN.
 *
 * Bindings: 0=input(readonly), 1=output(f32 scalar)
 * Push constants: { len: f32, _pad: f32 }
 * Dispatch: (ceil(len/WG_SIZE), 1, 1)
 */
export function kernelCheckFinite(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufIn  = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufOut = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, false);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

  // Shared memory for workgroup reduction: array of u32[WG_SIZE]
  const constWgSize = b.id();
  b.constant(p.tU32, constWgSize, wgSize);
  const tArrayWg = b.id();
  b.typeArray(tArrayWg, p.tU32, constWgSize);
  const tPtrSharedArr = b.id();
  b.typePointer(tPtrSharedArr, StorageClass.Workgroup, tArrayWg);
  const tPtrSharedU32 = b.id();
  b.typePointer(tPtrSharedU32, StorageClass.Workgroup, p.tU32);
  const sharedFlags = b.id();
  b.variable(tPtrSharedArr, sharedFlags, StorageClass.Workgroup);

  // Built-ins
  const tPtrInputVec3 = b.id();
  b.typePointer(tPtrInputVec3, StorageClass.Input, p.tVec3U32);
  const vLocalId = b.id();
  b.variable(tPtrInputVec3, vLocalId, StorageClass.Input);
  b.addDecorate(vLocalId, Decoration.BuiltIn, BuiltIn.LocalInvocationId);
  const vWorkgroupId = b.id();
  b.variable(tPtrInputVec3, vWorkgroupId, StorageClass.Input);
  b.addDecorate(vWorkgroupId, Decoration.BuiltIn, BuiltIn.WorkgroupId);

  // Barrier constants
  const scopeWg = b.id();
  b.constant(p.tU32, scopeWg, Scope.Workgroup);
  const semAcqRelWg = b.id();
  b.constant(p.tU32, semAcqRelWg, MemorySemantics.AcquireRelease | MemorySemantics.WorkgroupMemory);

  const const1f = b.id();
  b.constantF32(p.tF32, const1f, 1.0);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId, vLocalId, vWorkgroupId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  const labelEntry = b.id();
  b.emit(Op.Label, [labelEntry]);

  // Load global ID and local ID
  const gidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gidX = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, gidX, gidVec, 0]);

  const lidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, lidVec, vLocalId]);
  const lid = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, lid, lidVec, 0]);

  const wgIdVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, wgIdVec, vWorkgroupId]);
  const wgId = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, wgId, wgIdVec, 0]);

  // Load len
  const lenF = loadPushLen(b, p, pc);
  const lenU = b.id();
  b.emit(Op.ConvertFToU, [p.tU32, lenU, lenF]);

  // Check if this thread is in bounds
  const inBounds = b.id();
  b.emit(Op.ULessThan, [p.tBool, inBounds, gidX, lenU]);

  // Per-thread flag: 1 if this element is not finite, 0 otherwise
  const labelInBounds = b.id();
  const labelOutBounds = b.id();
  const labelMerge = b.id();
  b.emit(Op.SelectionMerge, [labelMerge, 0]);
  b.emit(Op.BranchConditional, [inBounds, labelInBounds, labelOutBounds]);

  // In bounds: load element, check IsNan and IsInf
  b.emit(Op.Label, [labelInBounds]);
  const ptrElem = b.id();
  b.emit(Op.AccessChain, [bufIn.tPtrF32, ptrElem, bufIn.varId, p.const0u, gidX]);
  const val = b.id();
  b.emit(Op.Load, [p.tF32, val, ptrElem]);
  const isNan = b.id();
  b.emit(Op.IsNan, [p.tBool, isNan, val]);
  const isInf = b.id();
  b.emit(Op.IsInf, [p.tBool, isInf, val]);
  const isNotFinite = b.id();
  b.emit(Op.LogicalOr, [p.tBool, isNotFinite, isNan, isInf]);
  const flagInBounds = b.id();
  b.emit(Op.Select, [p.tU32, flagInBounds, isNotFinite, p.const1u, p.const0u]);
  b.emit(Op.Branch, [labelMerge]);

  // Out of bounds: flag = 0
  b.emit(Op.Label, [labelOutBounds]);
  b.emit(Op.Branch, [labelMerge]);

  // Merge: phi to get the flag
  b.emit(Op.Label, [labelMerge]);
  const threadFlag = b.id();
  b.emit(Op.Phi, [p.tU32, threadFlag, flagInBounds, labelInBounds, p.const0u, labelOutBounds]);

  // Store to shared memory
  const ptrShared = b.id();
  b.emit(Op.AccessChain, [tPtrSharedU32, ptrShared, sharedFlags, lid]);
  b.emit(Op.Store, [ptrShared, threadFlag]);

  // Barrier
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  // Parallel reduction in shared memory: fold with BitwiseOr
  // We do log2(WG_SIZE) steps
  let stride = wgSize >> 1;
  while (stride > 0) {
    const cmpR = b.id();
    const constStride = b.id();
    b.constant(p.tU32, constStride, stride);
    b.emit(Op.ULessThan, [p.tBool, cmpR, lid, constStride]);

    const labelBody = b.id();
    const labelEnd = b.id();
    b.emit(Op.SelectionMerge, [labelEnd, 0]);
    b.emit(Op.BranchConditional, [cmpR, labelBody, labelEnd]);

    b.emit(Op.Label, [labelBody]);
    // shared[lid] |= shared[lid + stride]
    const otherIdx = b.id();
    b.emit(Op.IAdd, [p.tU32, otherIdx, lid, constStride]);
    const ptrA = b.id();
    b.emit(Op.AccessChain, [tPtrSharedU32, ptrA, sharedFlags, lid]);
    const ptrB = b.id();
    b.emit(Op.AccessChain, [tPtrSharedU32, ptrB, sharedFlags, otherIdx]);
    const valA = b.id();
    b.emit(Op.Load, [p.tU32, valA, ptrA]);
    const valB = b.id();
    b.emit(Op.Load, [p.tU32, valB, ptrB]);
    const merged = b.id();
    b.emit(Op.BitwiseOr, [p.tU32, merged, valA, valB]);
    b.emit(Op.Store, [ptrA, merged]);
    b.emit(Op.Branch, [labelEnd]);

    b.emit(Op.Label, [labelEnd]);

    if (stride > 1) {
      b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);
    }
    stride >>= 1;
  }

  // Thread 0 writes result to output buffer using atomicOr
  // Output[0] |= shared[0] (so any workgroup finding non-finite sets the flag)
  const isThread0 = b.id();
  b.emit(Op.IEqual, [p.tBool, isThread0, lid, p.const0u]);
  const labelWrite = b.id();
  const labelDone = b.id();
  b.emit(Op.SelectionMerge, [labelDone, 0]);
  b.emit(Op.BranchConditional, [isThread0, labelWrite, labelDone]);

  b.emit(Op.Label, [labelWrite]);
  const ptrResult = b.id();
  b.emit(Op.AccessChain, [tPtrSharedU32, ptrResult, sharedFlags, p.const0u]);
  const resultU = b.id();
  b.emit(Op.Load, [p.tU32, resultU, ptrResult]);
  // Convert to f32: 0 or 1
  const resultF = b.id();
  b.emit(Op.ConvertUToF, [p.tF32, resultF, resultU]);
  // If any workgroup found non-finite, we need to "max" with existing output.
  // Since output is initialized to 0.0 and we write 1.0, just check if > 0
  // and write via plain store (we only need to find *any* non-finite).
  // For multi-workgroup: read existing, take max, write back.
  // Simpler: write 1.0 if result != 0, otherwise don't touch output.
  const hasNonFinite = b.id();
  b.emit(Op.FOrdGreaterThan, [p.tBool, hasNonFinite, resultF, p.const0f]);
  const labelStore = b.id();
  const labelSkip = b.id();
  b.emit(Op.SelectionMerge, [labelSkip, 0]);
  b.emit(Op.BranchConditional, [hasNonFinite, labelStore, labelSkip]);

  b.emit(Op.Label, [labelStore]);
  const ptrOut = b.id();
  b.emit(Op.AccessChain, [bufOut.tPtrF32, ptrOut, bufOut.varId, p.const0u, p.const0u]);
  b.emit(Op.Store, [ptrOut, const1f]);
  b.emit(Op.Branch, [labelSkip]);

  b.emit(Op.Label, [labelSkip]);
  b.emit(Op.Branch, [labelDone]);

  b.emit(Op.Label, [labelDone]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}
