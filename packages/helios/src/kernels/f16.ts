/**
 * kernels/f16.ts — F16 storage variant kernels and dtype cast kernels.
 *
 * Compute in f32, load/store f16. Requires Float16 + StorageBuffer16BitAccess.
 */

import {
  SpirVBuilder, Op, ExecutionModel, ExecutionMode, StorageClass, Decoration,
  BuiltIn, FunctionControl, GLSLstd450, Capability,
  preamble, declareStorageBuffer, declareStorageBufferF16, declareParamsPushConstant,
  loadPushLen, emitBoundsCheck,
} from "./helpers.js";

/**
 * Generate element-wise binary op kernel with f16 storage:
 * Load f16 → convert to f32 → compute → convert to f16 → store.
 *
 * Bindings: 0=A(f16), 1=B(f16), 2=C(f16)
 * Push constants: { len: f32, _unused: f32 }
 */
export function kernelBinaryOpF16(opcode: number, wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  // Extra capabilities for f16
  b.addCapability(Capability.Float16);
  b.addCapability(Capability.StorageBuffer16BitAccess);

  // f16 type
  const tF16 = b.id();
  b.typeFloat(tF16, 16);

  const bufA = declareStorageBufferF16(b, tF16, p.tU32, 0, 0, true);
  const bufB = declareStorageBufferF16(b, tF16, p.tU32, 0, 1, true);
  const bufC = declareStorageBufferF16(b, tF16, p.tU32, 0, 2, false);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

  // Entry point
  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  const label0 = b.id();
  b.emit(Op.Label, [label0]);

  const ptrGid = b.id();
  b.emit(Op.AccessChain, [p.tPtrInputVec3, ptrGid, p.vGlobalId]);
  const gid = b.id();
  b.emit(Op.Load, [p.tVec3U32, gid, ptrGid]);
  const gidX = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, gidX, gid, 0]);

  const lenF = loadPushLen(b, p, pc);
  const labelEnd = b.id();
  emitBoundsCheck(b, p, lenF, gidX, labelEnd);

  // Load f16 values
  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrF16, ptrA, bufA.varId, p.const0u, gidX]);
  const valAf16 = b.id();
  b.emit(Op.Load, [tF16, valAf16, ptrA]);

  const ptrB = b.id();
  b.emit(Op.AccessChain, [bufB.tPtrF16, ptrB, bufB.varId, p.const0u, gidX]);
  const valBf16 = b.id();
  b.emit(Op.Load, [tF16, valBf16, ptrB]);

  // Convert f16 → f32
  const valA = b.id();
  b.emit(Op.FConvert, [p.tF32, valA, valAf16]);
  const valB = b.id();
  b.emit(Op.FConvert, [p.tF32, valB, valBf16]);

  // Compute in f32
  const result = b.id();
  b.emit(opcode, [p.tF32, result, valA, valB]);

  // Convert f32 → f16 and store
  const resultF16 = b.id();
  b.emit(Op.FConvert, [tF16, resultF16, result]);
  const ptrC = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrF16, ptrC, bufC.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrC, resultF16]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

/**
 * Cast f32 → f16: Read f32, FConvert to f16, store f16.
 * Bindings: 0=A(f32, readonly), 1=C(f16, writeonly)
 * Push constants: { len: f32, _unused: f32 }
 */
export function kernelCastF32ToF16(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  b.addCapability(Capability.Float16);
  b.addCapability(Capability.StorageBuffer16BitAccess);

  const tF16 = b.id();
  b.typeFloat(tF16, 16);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufC = declareStorageBufferF16(b, tF16, p.tU32, 0, 1, false);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  const label0 = b.id();
  b.emit(Op.Label, [label0]);

  const ptrGid = b.id();
  b.emit(Op.AccessChain, [p.tPtrInputVec3, ptrGid, p.vGlobalId]);
  const gid = b.id();
  b.emit(Op.Load, [p.tVec3U32, gid, ptrGid]);
  const gidX = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, gidX, gid, 0]);

  const lenF = loadPushLen(b, p, pc);
  const labelEnd = b.id();
  emitBoundsCheck(b, p, lenF, gidX, labelEnd);

  // Load f32
  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrF32, ptrA, bufA.varId, p.const0u, gidX]);
  const valF32 = b.id();
  b.emit(Op.Load, [p.tF32, valF32, ptrA]);

  // Convert f32 → f16
  const valF16 = b.id();
  b.emit(Op.FConvert, [tF16, valF16, valF32]);

  // Store f16
  const ptrC = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrF16, ptrC, bufC.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrC, valF16]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

/**
 * Cast f16 → f32: Read f16, FConvert to f32, store f32.
 * Bindings: 0=A(f16, readonly), 1=C(f32, writeonly)
 * Push constants: { len: f32, _unused: f32 }
 */
export function kernelCastF16ToF32(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  b.addCapability(Capability.Float16);
  b.addCapability(Capability.StorageBuffer16BitAccess);

  const tF16 = b.id();
  b.typeFloat(tF16, 16);

  const bufA = declareStorageBufferF16(b, tF16, p.tU32, 0, 0, true);
  const bufC = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, false);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  const label0 = b.id();
  b.emit(Op.Label, [label0]);

  const ptrGid = b.id();
  b.emit(Op.AccessChain, [p.tPtrInputVec3, ptrGid, p.vGlobalId]);
  const gid = b.id();
  b.emit(Op.Load, [p.tVec3U32, gid, ptrGid]);
  const gidX = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, gidX, gid, 0]);

  const lenF = loadPushLen(b, p, pc);
  const labelEnd = b.id();
  emitBoundsCheck(b, p, lenF, gidX, labelEnd);

  // Load f16
  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrF16, ptrA, bufA.varId, p.const0u, gidX]);
  const valF16 = b.id();
  b.emit(Op.Load, [tF16, valF16, ptrA]);

  // Convert f16 → f32
  const valF32 = b.id();
  b.emit(Op.FConvert, [p.tF32, valF32, valF16]);

  // Store f32
  const ptrC = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrF32, ptrC, bufC.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrC, valF32]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

/**
 * Generate element-wise unary op kernel with f16 storage.
 */
export function kernelUnaryOpF16(glslOp: number | null, wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  b.addCapability(Capability.Float16);
  b.addCapability(Capability.StorageBuffer16BitAccess);

  const tF16 = b.id();
  b.typeFloat(tF16, 16);

  const bufA = declareStorageBufferF16(b, tF16, p.tU32, 0, 0, true);
  const bufC = declareStorageBufferF16(b, tF16, p.tU32, 0, 1, false);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  const label0 = b.id();
  b.emit(Op.Label, [label0]);

  const ptrGid = b.id();
  b.emit(Op.AccessChain, [p.tPtrInputVec3, ptrGid, p.vGlobalId]);
  const gid = b.id();
  b.emit(Op.Load, [p.tVec3U32, gid, ptrGid]);
  const gidX = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, gidX, gid, 0]);

  const lenF = loadPushLen(b, p, pc);
  const labelEnd = b.id();
  emitBoundsCheck(b, p, lenF, gidX, labelEnd);

  // Load f16 → f32
  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrF16, ptrA, bufA.varId, p.const0u, gidX]);
  const valAf16 = b.id();
  b.emit(Op.Load, [tF16, valAf16, ptrA]);
  const valA = b.id();
  b.emit(Op.FConvert, [p.tF32, valA, valAf16]);

  // Apply operation (GLSL.std.450 or custom)
  let result: number;
  if (glslOp !== null) {
    result = b.id();
    b.emit(Op.ExtInst, [p.tF32, result, p.glslStd, glslOp, valA]);
  } else {
    // neg: FNegate
    result = b.id();
    b.emit(Op.FNegate, [p.tF32, result, valA]);
  }

  // Convert f32 → f16 and store
  const resultF16 = b.id();
  b.emit(Op.FConvert, [tF16, resultF16, result]);
  const ptrC = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrF16, ptrC, bufC.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrC, resultF16]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}
