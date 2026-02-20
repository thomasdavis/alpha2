/**
 * kernels.ts — SPIR-V compute kernel generators.
 *
 * Each function builds a complete SPIR-V module for a specific GPU operation.
 * Generated entirely from TypeScript — no external shader compiler needed.
 *
 * Pattern for each kernel:
 *   - binding 0..N-1 = storage buffers (f32 runtime arrays)
 *   - last binding    = params buffer (u32/f32 values)
 *   - workgroup size  = specified per kernel
 *   - entry point     = "main"
 */

import {
  SpirVBuilder,
  Op,
  Capability,
  AddressingModel,
  MemoryModel,
  ExecutionModel,
  ExecutionMode,
  StorageClass,
  Decoration,
  BuiltIn,
  GLSLstd450,
  FunctionControl,
} from "./spirv.js";

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
function preamble(b: SpirVBuilder, wgX: number, wgY: number, wgZ: number) {
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
function declareStorageBuffer(
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

// ── Bounds check helper ─────────────────────────────────────────────────────

/**
 * Emit bounds check: if (gidX >= len) skip to labelEnd.
 *
 * Length is stored as f32 in params[0]. We convert gidX to f32 and compare
 * in float space. This avoids the denorm-flush issue where GPU hardware
 * flushes small uint32 values (stored as f32 bits via bitcast) to zero.
 * f32 exactly represents integers up to 2^24 (~16M elements), which is plenty.
 */
function emitBoundsCheck(
  b: SpirVBuilder,
  p: ReturnType<typeof preamble>,
  bufParams: { varId: number; tPtrF32: number },
  gidX: number,
  labelEnd: number,
): number {
  // len = params[0] (stored as f32 directly)
  const ptrLen = b.id();
  b.emit(Op.AccessChain, [bufParams.tPtrF32, ptrLen, bufParams.varId, p.const0u, p.const0u]);
  const lenF = b.id();
  b.emit(Op.Load, [p.tF32, lenF, ptrLen]);

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

// ── Kernel: Element-wise binary ops ─────────────────────────────────────────

/**
 * Generate element-wise binary op kernel: C[i] = op(A[i], B[i])
 *
 * Bindings:
 *   0 = A (storage buffer, readonly)
 *   1 = B (storage buffer, readonly)
 *   2 = C (storage buffer, writeonly)
 *   3 = params { len: f32 }
 */
function kernelBinaryOp(opcode: number, wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufB = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, true);
  const bufC = declareStorageBuffer(b, p.tF32, p.tU32, 0, 2, false);
  const bufParams = declareStorageBuffer(b, p.tF32, p.tU32, 0, 3, true);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  const labelEntry = b.id();
  const labelEnd   = b.id();

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  b.emit(Op.Label, [labelEntry]);

  const gidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gidX = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, gidX, gidVec, 0]);

  emitBoundsCheck(b, p, bufParams, gidX, labelEnd);

  // a = A[gidX], b = B[gidX]
  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrF32, ptrA, bufA.varId, p.const0u, gidX]);
  const valA = b.id();
  b.emit(Op.Load, [p.tF32, valA, ptrA]);

  const ptrB = b.id();
  b.emit(Op.AccessChain, [bufB.tPtrF32, ptrB, bufB.varId, p.const0u, gidX]);
  const valB = b.id();
  b.emit(Op.Load, [p.tF32, valB, ptrB]);

  // c = op(a, b)
  const valC = b.id();
  b.emit(opcode, [p.tF32, valC, valA, valB]);

  // C[gidX] = c
  const ptrC = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrF32, ptrC, bufC.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrC, valC]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

export function kernelAdd(wgSize = 256): Uint32Array { return kernelBinaryOp(Op.FAdd, wgSize); }
export function kernelSub(wgSize = 256): Uint32Array { return kernelBinaryOp(Op.FSub, wgSize); }
export function kernelMul(wgSize = 256): Uint32Array { return kernelBinaryOp(Op.FMul, wgSize); }
export function kernelDiv(wgSize = 256): Uint32Array { return kernelBinaryOp(Op.FDiv, wgSize); }

// ── Kernel: Element-wise scale ──────────────────────────────────────────────

/**
 * C[i] = A[i] * scalar
 * Bindings: 0=A(in), 1=C(out), 2=params{len:f32, scalar:f32}
 */
export function kernelScale(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufC = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, false);
  const bufParams = declareStorageBuffer(b, p.tF32, p.tU32, 0, 2, true);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  const labelEntry = b.id();
  const labelEnd   = b.id();

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  b.emit(Op.Label, [labelEntry]);

  const gidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gidX = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, gidX, gidVec, 0]);

  // scalar = params[1]
  const ptrScalar = b.id();
  b.emit(Op.AccessChain, [bufParams.tPtrF32, ptrScalar, bufParams.varId, p.const0u, p.const1u]);
  const scalar = b.id();
  b.emit(Op.Load, [p.tF32, scalar, ptrScalar]);

  emitBoundsCheck(b, p, bufParams, gidX, labelEnd);

  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrF32, ptrA, bufA.varId, p.const0u, gidX]);
  const valA = b.id();
  b.emit(Op.Load, [p.tF32, valA, ptrA]);

  const valC = b.id();
  b.emit(Op.FMul, [p.tF32, valC, valA, scalar]);

  const ptrC = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrF32, ptrC, bufC.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrC, valC]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

// ── Kernel: Neg ─────────────────────────────────────────────────────────────

/**
 * C[i] = -A[i]
 * Bindings: 0=A(in), 1=C(out), 2=params{len:f32}
 */
export function kernelNeg(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufC = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, false);
  const bufParams = declareStorageBuffer(b, p.tF32, p.tU32, 0, 2, true);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  const labelEntry = b.id();
  const labelEnd   = b.id();

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  b.emit(Op.Label, [labelEntry]);

  const gidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gidX = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, gidX, gidVec, 0]);

  emitBoundsCheck(b, p, bufParams, gidX, labelEnd);

  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrF32, ptrA, bufA.varId, p.const0u, gidX]);
  const valA = b.id();
  b.emit(Op.Load, [p.tF32, valA, ptrA]);

  const valC = b.id();
  b.emit(Op.FNegate, [p.tF32, valC, valA]);

  const ptrC = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrF32, ptrC, bufC.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrC, valC]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

// ── Kernel: Unary math (exp, log, sqrt) via GLSL.std.450 ───────────────────

function kernelUnaryGlsl(glslOp: number, wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufC = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, false);
  const bufParams = declareStorageBuffer(b, p.tF32, p.tU32, 0, 2, true);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  const labelEntry = b.id();
  const labelEnd   = b.id();

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  b.emit(Op.Label, [labelEntry]);

  const gidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gidX = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, gidX, gidVec, 0]);

  emitBoundsCheck(b, p, bufParams, gidX, labelEnd);

  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrF32, ptrA, bufA.varId, p.const0u, gidX]);
  const valA = b.id();
  b.emit(Op.Load, [p.tF32, valA, ptrA]);

  // result = glslOp(valA)
  const valC = b.id();
  b.emit(Op.ExtInst, [p.tF32, valC, p.glslStd, glslOp, valA]);

  const ptrC = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrF32, ptrC, bufC.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrC, valC]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

export function kernelExp(wgSize = 256): Uint32Array { return kernelUnaryGlsl(GLSLstd450.Exp, wgSize); }
export function kernelLog(wgSize = 256): Uint32Array { return kernelUnaryGlsl(GLSLstd450.Log, wgSize); }
export function kernelSqrt(wgSize = 256): Uint32Array { return kernelUnaryGlsl(GLSLstd450.Sqrt, wgSize); }

// ── Kernel cache ────────────────────────────────────────────────────────────

const spirvCache = new Map<string, Uint32Array>();

/** Get a cached SPIR-V binary, generating it on first use. */
export function getKernelSpirv(name: string, wgSize = 256): Uint32Array {
  const key = `${name}:${wgSize}`;
  let spirv = spirvCache.get(key);
  if (spirv) return spirv;

  switch (name) {
    case "add":   spirv = kernelAdd(wgSize); break;
    case "sub":   spirv = kernelSub(wgSize); break;
    case "mul":   spirv = kernelMul(wgSize); break;
    case "div":   spirv = kernelDiv(wgSize); break;
    case "neg":   spirv = kernelNeg(wgSize); break;
    case "scale": spirv = kernelScale(wgSize); break;
    case "exp":   spirv = kernelExp(wgSize); break;
    case "log":   spirv = kernelLog(wgSize); break;
    case "sqrt":  spirv = kernelSqrt(wgSize); break;
    default:
      throw new Error(`Helios: unknown kernel "${name}"`);
  }

  spirvCache.set(key, spirv);
  return spirv;
}
