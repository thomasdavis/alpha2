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

  // NonWritable goes on the struct member, not the variable
  if (readonly_) {
    b.addMemberDecorate(tStruct, 0, Decoration.NonWritable);
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
) {
  const tRuntimeArr = b.id();
  b.typeRuntimeArray(tRuntimeArr, tVec4F32);
  b.addDecorate(tRuntimeArr, Decoration.ArrayStride, 16);

  const tStruct = b.id();
  b.typeStruct(tStruct, [tRuntimeArr]);
  b.addDecorate(tStruct, Decoration.BufferBlock);
  b.addMemberDecorate(tStruct, 0, Decoration.Offset, 0);
  if (readonly_) b.addMemberDecorate(tStruct, 0, Decoration.NonWritable);

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

