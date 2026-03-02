/**
 * kernels/helpers.ts — Shared preamble, buffer declarations, and SPIR-V re-exports.
 *
 * Every kernel file imports from this module.
 */

// Re-export all SPIR-V symbols so kernel files only import from helpers
export {
  SpirVBuilder,
  Op,
  Capability,
  CooperativeMatrixUse,
  GroupOperation,
  AddressingModel,
  MemoryModel,
  ExecutionModel,
  ExecutionMode,
  StorageClass,
  Decoration,
  BuiltIn,
  GLSLstd450,
  FunctionControl,
  Scope,
  MemorySemantics,
} from "../spirv.js";

import {
  SpirVBuilder,
  Op,
  Capability,
  CooperativeMatrixUse,
  GroupOperation,
  AddressingModel,
  MemoryModel,
  ExecutionModel,
  ExecutionMode,
  StorageClass,
  Decoration,
  BuiltIn,
  GLSLstd450,
  FunctionControl,
  Scope,
  MemorySemantics,
} from "../spirv.js";

// ── Helpers ─────────────────────────────────────────────────────────────────

/**
 * Set up the common preamble for a compute shader:
 *   - Capability, memory model
 *   - GLSL.std.450 import
 *   - Common types (void, f32, u32, vec3<u32>)
 *   - GlobalInvocationId built-in
 *
 * Returns an object with all the IDs you need.
 */
export function preamble(b: SpirVBuilder, wgX: number, wgY: number, wgZ: number) {
  b.addCapability(Capability.Shader);

  const glslStd = b.id();
  b.addExtInstImport(glslStd, "GLSL.std.450");

  b.setMemoryModel(AddressingModel.Logical, MemoryModel.GLSL450);

  // Types
  const tVoid = b.id();
  const tF32  = b.id();
  const tU32  = b.id();
  const tBool = b.id();
  const tVec3U32 = b.id();
  const tFnVoid  = b.id();

  b.typeVoid(tVoid);
  b.typeFloat(tF32, 32);
  b.typeInt(tU32, 32, 0); // unsigned
  b.typeBool(tBool);
  b.typeVector(tVec3U32, tU32, 3);
  b.typeFunction(tFnVoid, tVoid);

  // Pointer to vec3<u32> for built-in input
  const tPtrInputVec3 = b.id();
  b.typePointer(tPtrInputVec3, StorageClass.Input, tVec3U32);

  // GlobalInvocationId variable
  const vGlobalId = b.id();
  b.variable(tPtrInputVec3, vGlobalId, StorageClass.Input);
  b.addDecorate(vGlobalId, Decoration.BuiltIn, BuiltIn.GlobalInvocationId);

  // Common constants
  const const0u = b.id();
  const const1u = b.id();
  const const2u = b.id();
  b.constant(tU32, const0u, 0);
  b.constant(tU32, const1u, 1);
  b.constant(tU32, const2u, 2);

  const const0f = b.id();
  b.constantF32(tF32, const0f, 0.0);

  return {
    glslStd, tVoid, tF32, tU32, tBool, tVec3U32, tFnVoid,
    tPtrInputVec3, vGlobalId,
    const0u, const1u, const2u, const0f,
    wgX, wgY, wgZ,
  };
}

/**
 * Declare a storage buffer binding (runtime array of f32 wrapped in a struct).
 *
 * Uses Uniform storage class + BufferBlock decoration (classic SPIR-V pattern).
 * This is compatible with all Vulkan implementations including Intel Mesa ANV,
 * which may reject the newer StorageBuffer storage class.
 *
 * Returns { varId, tPtrF32 } for accessing elements.
 */
export function declareStorageBuffer(
  b: SpirVBuilder,
  tF32: number,
  _tU32: number,
  set: number,
  binding: number,
  readonly_: boolean = false,
  writeonly_: boolean = false,
) {
  // RuntimeArray<f32>
  const tRuntimeArr = b.id();
  b.typeRuntimeArray(tRuntimeArr, tF32);
  b.addDecorate(tRuntimeArr, Decoration.ArrayStride, 4);

  // Struct { RuntimeArray<f32> } with BufferBlock decoration
  const tStruct = b.id();
  b.typeStruct(tStruct, [tRuntimeArr]);
  b.addDecorate(tStruct, Decoration.BufferBlock);
  b.addMemberDecorate(tStruct, 0, Decoration.Offset, 0);

  // NonWritable/NonReadable go on the struct member, not the variable
  if (readonly_) {
    b.addMemberDecorate(tStruct, 0, Decoration.NonWritable);
  }
  if (writeonly_) {
    b.addMemberDecorate(tStruct, 0, Decoration.NonReadable);
  }

  // Pointer to struct in Uniform storage class
  const tPtrStruct = b.id();
  b.typePointer(tPtrStruct, StorageClass.Uniform, tStruct);

  // Pointer to f32 in Uniform (for AccessChain into the array)
  const tPtrF32 = b.id();
  b.typePointer(tPtrF32, StorageClass.Uniform, tF32);

  // Variable
  const varId = b.id();
  b.variable(tPtrStruct, varId, StorageClass.Uniform);
  b.addDecorate(varId, Decoration.DescriptorSet, set);
  b.addDecorate(varId, Decoration.Binding, binding);

  return { varId, tPtrF32, tPtrStruct };
}

/**
 * Declare push constant params block.
 *
 * Uses PushConstant storage class + Block decoration.
 * Members are f32 at 4-byte offsets.
 * No descriptor set/binding needed — data comes from vkCmdPushConstants.
 */
export function declareParamsPushConstant(
  b: SpirVBuilder,
  tF32: number,
  numMembers: number,
) {
  const memberTypes = Array(numMembers).fill(tF32) as number[];
  const tStruct = b.id();
  b.typeStruct(tStruct, memberTypes);
  b.addDecorate(tStruct, Decoration.Block);
  for (let i = 0; i < numMembers; i++) {
    b.addMemberDecorate(tStruct, i, Decoration.Offset, i * 4);
  }

  const tPtrStruct = b.id();
  b.typePointer(tPtrStruct, StorageClass.PushConstant, tStruct);

  const tPtrF32 = b.id();
  b.typePointer(tPtrF32, StorageClass.PushConstant, tF32);

  const varId = b.id();
  b.variable(tPtrStruct, varId, StorageClass.PushConstant);

  return { varId, tPtrF32 };
}

// ── Bounds check helper ─────────────────────────────────────────────────────

/**
 * Emit bounds check: if (gidX >= len) skip to labelEnd.
 *
 * lenF is an already-loaded f32 value representing the element count.
 * We convert gidX to f32 and compare in float space. This avoids the
 * denorm-flush issue where GPU hardware flushes small uint32 values
 * (stored as f32 bits via bitcast) to zero.
 * f32 exactly represents integers up to 2^24 (~16M elements), which is plenty.
 */
export function emitBoundsCheck(
  b: SpirVBuilder,
  p: ReturnType<typeof preamble>,
  lenF: number,
  gidX: number,
  labelEnd: number,
): number {
  // gidF = float(gidX)
  const gidF = b.id();
  b.emit(Op.ConvertUToF, [p.tF32, gidF, gidX]);

  // if (gidF >= lenF) skip
  const cmp = b.id();
  b.emit(Op.FOrdGreaterThanEqual, [p.tBool, cmp, gidF, lenF]);
  const labelBody = b.id();
  b.emit(Op.SelectionMerge, [labelEnd, 0]); // 0 = None
  b.emit(Op.BranchConditional, [cmp, labelEnd, labelBody]);

  b.emit(Op.Label, [labelBody]);
  return labelBody;
}

/**
 * Load len (member 0) from push constant block.
 */
export function loadPushLen(
  b: SpirVBuilder,
  p: ReturnType<typeof preamble>,
  pc: { varId: number; tPtrF32: number },
): number {
  const ptrLen = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrLen, pc.varId, p.const0u]);
  const lenF = b.id();
  b.emit(Op.Load, [p.tF32, lenF, ptrLen]);
  return lenF;
}

/**
 * Load scalar (member 1) from push constant block.
 */
export function loadPushScalar(
  b: SpirVBuilder,
  p: ReturnType<typeof preamble>,
  pc: { varId: number; tPtrF32: number },
): number {
  const ptrScalar = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrScalar, pc.varId, p.const1u]);
  const scalar = b.id();
  b.emit(Op.Load, [p.tF32, scalar, ptrScalar]);
  return scalar;
}

// ── f16 storage helpers ─────────────────────────────────────────────────────

/**
 * Declare a storage buffer binding with f16 elements (runtime array of f16).
 * Requires Float16 + StorageBuffer16BitAccess capabilities.
 * Returns { varId, tPtrF16, tF16 } for load/store + conversion.
 */
export function declareStorageBufferF16(
  b: SpirVBuilder,
  tF16: number,
  _tU32: number,
  set: number,
  binding: number,
  readonly_: boolean = false,
) {
  const tRuntimeArr = b.id();
  b.typeRuntimeArray(tRuntimeArr, tF16);
  b.addDecorate(tRuntimeArr, Decoration.ArrayStride, 2); // f16 = 2 bytes

  const tStruct = b.id();
  b.typeStruct(tStruct, [tRuntimeArr]);
  b.addDecorate(tStruct, Decoration.BufferBlock);
  b.addMemberDecorate(tStruct, 0, Decoration.Offset, 0);

  if (readonly_) {
    b.addMemberDecorate(tStruct, 0, Decoration.NonWritable);
  }

  const tPtrStruct = b.id();
  b.typePointer(tPtrStruct, StorageClass.Uniform, tStruct);

  const tPtrF16 = b.id();
  b.typePointer(tPtrF16, StorageClass.Uniform, tF16);

  const varId = b.id();
  b.variable(tPtrStruct, varId, StorageClass.Uniform);
  b.addDecorate(varId, Decoration.DescriptorSet, set);
  b.addDecorate(varId, Decoration.Binding, binding);

  return { varId, tPtrF16 };
}

// ── BDA (Buffer Device Address) helpers ─────────────────────────────────────
//
// For DGC (device-generated commands), kernels use PhysicalStorageBuffer
// addressing instead of descriptor sets. Buffer addresses are passed as u64
// push constants. This eliminates per-dispatch descriptor allocation.

/**
 * Set up the BDA preamble (PhysicalStorageBuffer addressing model).
 * Like preamble() but adds Int64 + PhysicalStorageBufferAddresses capabilities
 * and uses PhysicalStorageBuffer64 addressing model.
 *
 * Returns all IDs from preamble() plus BDA-specific types.
 */
export function preambleBDA(b: SpirVBuilder, wgX: number, wgY: number, wgZ: number) {
  b.addCapability(Capability.Shader);
  b.addCapability(Capability.Int64);
  b.addCapability(Capability.PhysicalStorageBufferAddresses);
  b.addExtension("SPV_KHR_physical_storage_buffer");

  const glslStd = b.id();
  b.addExtInstImport(glslStd, "GLSL.std.450");

  b.setMemoryModel(AddressingModel.PhysicalStorageBuffer64, MemoryModel.GLSL450);

  // Standard types
  const tVoid = b.id();
  const tF32  = b.id();
  const tU32  = b.id();
  const tBool = b.id();
  const tVec3U32 = b.id();
  const tFnVoid  = b.id();

  b.typeVoid(tVoid);
  b.typeFloat(tF32, 32);
  b.typeInt(tU32, 32, 0);
  b.typeBool(tBool);
  b.typeVector(tVec3U32, tU32, 3);
  b.typeFunction(tFnVoid, tVoid);

  // BDA-specific types
  const tU64 = b.id();
  b.typeInt(tU64, 64, 0);

  // RuntimeArray<f32> for buffer references
  const tRuntimeArrayF32 = b.id();
  b.typeRuntimeArray(tRuntimeArrayF32, tF32);
  b.addDecorate(tRuntimeArrayF32, Decoration.ArrayStride, 4);

  // Buffer struct (PhysicalStorageBuffer reference)
  const tBufferStruct = b.id();
  b.typeStruct(tBufferStruct, [tRuntimeArrayF32]);
  b.addDecorate(tBufferStruct, Decoration.Block);
  b.addMemberDecorate(tBufferStruct, 0, Decoration.Offset, 0);

  // Pointer types for PSB
  const tPtrPSB = b.id();
  b.typePointer(tPtrPSB, StorageClass.PhysicalStorageBuffer, tBufferStruct);
  const tPtrPSBF32 = b.id();
  b.typePointer(tPtrPSBF32, StorageClass.PhysicalStorageBuffer, tF32);

  // Built-in: GlobalInvocationId
  const tPtrInputVec3 = b.id();
  b.typePointer(tPtrInputVec3, StorageClass.Input, tVec3U32);
  const vGlobalId = b.id();
  b.variable(tPtrInputVec3, vGlobalId, StorageClass.Input);
  b.addDecorate(vGlobalId, Decoration.BuiltIn, BuiltIn.GlobalInvocationId);

  // Constants
  const const0u = b.id();
  const const1u = b.id();
  const const2u = b.id();
  b.constant(tU32, const0u, 0);
  b.constant(tU32, const1u, 1);
  b.constant(tU32, const2u, 2);

  const const0f = b.id();
  b.constantF32(tF32, const0f, 0.0);

  return {
    glslStd, tVoid, tF32, tU32, tU64, tBool, tVec3U32, tFnVoid,
    tPtrInputVec3, vGlobalId,
    tRuntimeArrayF32, tBufferStruct, tPtrPSB, tPtrPSBF32,
    const0u, const1u, const2u, const0f,
    wgX, wgY, wgZ,
  };
}

/**
 * Declare a BDA push constant block with N u64 buffer addresses + M u32 params.
 *
 * Layout:
 *   u64 addr[0]   (offset 0)
 *   u64 addr[1]   (offset 8)
 *   ...
 *   u64 addr[N-1] (offset (N-1)*8)
 *   u32 param[0]  (offset N*8)
 *   u32 param[1]  (offset N*8+4)
 *   ...
 *
 * Returns { varId, tPtrU64, tPtrU32 } for accessing members.
 */
export function declareBDAPushConstants(
  b: SpirVBuilder,
  tU64: number,
  tU32: number,
  numBuffers: number,
  numU32Params: number,
) {
  const memberTypes: number[] = [];
  for (let i = 0; i < numBuffers; i++) memberTypes.push(tU64);
  for (let i = 0; i < numU32Params; i++) memberTypes.push(tU32);

  const tStruct = b.id();
  b.typeStruct(tStruct, memberTypes);
  b.addDecorate(tStruct, Decoration.Block);

  let offset = 0;
  for (let i = 0; i < numBuffers; i++) {
    b.addMemberDecorate(tStruct, i, Decoration.Offset, offset);
    offset += 8; // u64
  }
  for (let i = 0; i < numU32Params; i++) {
    b.addMemberDecorate(tStruct, numBuffers + i, Decoration.Offset, offset);
    offset += 4; // u32
  }

  const tPtrStruct = b.id();
  b.typePointer(tPtrStruct, StorageClass.PushConstant, tStruct);
  const tPtrU64 = b.id();
  b.typePointer(tPtrU64, StorageClass.PushConstant, tU64);
  const tPtrU32 = b.id();
  b.typePointer(tPtrU32, StorageClass.PushConstant, tU32);

  const varId = b.id();
  b.variable(tPtrStruct, varId, StorageClass.PushConstant);

  return { varId, tPtrU64, tPtrU32, totalSize: offset };
}

/**
 * Load a buffer address from BDA push constants and convert to PSB pointer.
 * memberIndex is the push constant member index (0, 1, 2, ...).
 */
export function loadBDABuffer(
  b: SpirVBuilder,
  p: ReturnType<typeof preambleBDA>,
  pc: ReturnType<typeof declareBDAPushConstants>,
  memberIndex: number,
): number {
  const constIdx = b.id();
  b.constant(p.tU32, constIdx, memberIndex);

  const ptrAddr = b.id();
  b.emit(Op.AccessChain, [pc.tPtrU64, ptrAddr, pc.varId, constIdx]);
  const addr = b.id();
  b.emit(Op.Load, [p.tU64, addr, ptrAddr]);

  // Convert u64 → PhysicalStorageBuffer pointer
  const bufPtr = b.id();
  b.emit(Op.ConvertUToPtr, [p.tPtrPSB, bufPtr, addr]);
  return bufPtr;
}

/**
 * Load f32 from a BDA buffer at index gid.
 * bufPtr is the result of loadBDABuffer().
 * Returns the loaded f32 value ID.
 */
export function loadBDAElement(
  b: SpirVBuilder,
  p: ReturnType<typeof preambleBDA>,
  bufPtr: number,
  gid: number,
): number {
  const ptr = b.id();
  b.emit(Op.AccessChain, [p.tPtrPSBF32, ptr, bufPtr, p.const0u, gid]);
  const val = b.id();
  b.emit(Op.Load, [p.tF32, val, ptr, 2 /*Aligned*/, 4]);
  return val;
}

/**
 * Store f32 to a BDA buffer at index gid.
 * bufPtr is the result of loadBDABuffer().
 */
export function storeBDAElement(
  b: SpirVBuilder,
  p: ReturnType<typeof preambleBDA>,
  bufPtr: number,
  gid: number,
  value: number,
): void {
  const ptr = b.id();
  b.emit(Op.AccessChain, [p.tPtrPSBF32, ptr, bufPtr, p.const0u, gid]);
  b.emit(Op.Store, [ptr, value, 2 /*Aligned*/, 4]);
}

// ── Vec4 helpers ────────────────────────────────────────────────────────────

/**
 * Declare a vec4 storage buffer binding (runtime array of vec4<f32> in a struct).
 * Same pattern as declareStorageBuffer but with 16-byte stride for vec4.
 */
export function declareStorageBufferVec4(
  b: SpirVBuilder,
  tVec4F32: number,
  set: number,
  binding: number,
  readonly_: boolean = false,
  writeonly_: boolean = false,
) {
  const tRuntimeArr = b.id();
  b.typeRuntimeArray(tRuntimeArr, tVec4F32);
  b.addDecorate(tRuntimeArr, Decoration.ArrayStride, 16);

  const tStruct = b.id();
  b.typeStruct(tStruct, [tRuntimeArr]);
  b.addDecorate(tStruct, Decoration.BufferBlock);
  b.addMemberDecorate(tStruct, 0, Decoration.Offset, 0);
  if (readonly_) b.addMemberDecorate(tStruct, 0, Decoration.NonWritable);
  if (writeonly_) b.addMemberDecorate(tStruct, 0, Decoration.NonReadable);

  const tPtrStruct = b.id();
  b.typePointer(tPtrStruct, StorageClass.Uniform, tStruct);

  const tPtrVec4 = b.id();
  b.typePointer(tPtrVec4, StorageClass.Uniform, tVec4F32);

  const varId = b.id();
  b.variable(tPtrStruct, varId, StorageClass.Uniform);
  b.addDecorate(varId, Decoration.DescriptorSet, set);
  b.addDecorate(varId, Decoration.Binding, binding);

  return { varId, tPtrVec4 };
}

