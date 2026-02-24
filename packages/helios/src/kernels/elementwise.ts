/**
 * kernels/elementwise.ts — Element-wise binary and unary GPU kernels.
 *
 * Binary: add, sub, mul, div (scalar + vec4)
 * Unary: neg, scale, exp, log, sqrt, relu, gelu, clamp, clampMin, softcap
 * Backward: gelu_backward, relu_backward, clamp_backward, softcap_backward
 * Vec4 variants of most operations.
 */

import {
  SpirVBuilder, Op, ExecutionModel, ExecutionMode, FunctionControl, GLSLstd450,
  preamble, declareStorageBuffer, declareStorageBufferVec4, declareParamsPushConstant,
  loadPushLen, loadPushScalar, emitBoundsCheck,
} from "./helpers.js";

// ── Kernel: Element-wise binary ops ─────────────────────────────────────────

/**
 * Generate element-wise binary op kernel: C[i] = op(A[i], B[i])
 *
 * Bindings:
 *   0 = A (storage buffer, readonly)
 *   1 = B (storage buffer, readonly)
 *   2 = C (storage buffer, writeonly)
 * Push constants: { len: f32, _unused: f32 }
 */
function kernelBinaryOp(opcode: number, wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufB = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, true);
  const bufC = declareStorageBuffer(b, p.tF32, p.tU32, 0, 2, false);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

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

  const lenF = loadPushLen(b, p, pc);
  emitBoundsCheck(b, p, lenF, gidX, labelEnd);

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
 * Bindings: 0=A(in), 1=C(out)
 * Push constants: { len: f32, scalar: f32 }
 */
export function kernelScale(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufC = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, false);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

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

  // Load scalar before bounds check
  const scalar = loadPushScalar(b, p, pc);

  const lenF = loadPushLen(b, p, pc);
  emitBoundsCheck(b, p, lenF, gidX, labelEnd);

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
 * Bindings: 0=A(in), 1=C(out)
 * Push constants: { len: f32, _unused: f32 }
 */
export function kernelNeg(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufC = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, false);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

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

  const lenF = loadPushLen(b, p, pc);
  emitBoundsCheck(b, p, lenF, gidX, labelEnd);

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
  const pc = declareParamsPushConstant(b, p.tF32, 2);

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

  const lenF = loadPushLen(b, p, pc);
  emitBoundsCheck(b, p, lenF, gidX, labelEnd);

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

// ── Kernel: ReLU ────────────────────────────────────────────────────────────

/**
 * C[i] = max(A[i], 0.0)
 * Bindings: 0=A(in), 1=C(out)
 * Push constants: { len: f32, _unused: f32 }
 */
export function kernelRelu(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufC = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, false);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

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

  const lenF = loadPushLen(b, p, pc);
  emitBoundsCheck(b, p, lenF, gidX, labelEnd);

  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrF32, ptrA, bufA.varId, p.const0u, gidX]);
  const valA = b.id();
  b.emit(Op.Load, [p.tF32, valA, ptrA]);

  // max(valA, 0.0)
  const valC = b.id();
  b.emit(Op.ExtInst, [p.tF32, valC, p.glslStd, GLSLstd450.FMax, valA, p.const0f]);

  const ptrC = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrF32, ptrC, bufC.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrC, valC]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

// ── Kernel: GELU (fused tanh approximation) ─────────────────────────────────

/**
 * C[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 * Fully fused into a single kernel — no intermediate buffers.
 * Bindings: 0=A(in), 1=C(out)
 * Push constants: { len: f32, _unused: f32 }
 */
export function kernelGelu(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufC = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, false);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

  // Constants for GELU
  const constHalf = b.id();
  b.constantF32(p.tF32, constHalf, 0.5);
  const constOne = b.id();
  b.constantF32(p.tF32, constOne, 1.0);
  const constCoeff = b.id();
  b.constantF32(p.tF32, constCoeff, 0.044715);
  const constSqrt2OverPi = b.id();
  b.constantF32(p.tF32, constSqrt2OverPi, Math.sqrt(2.0 / Math.PI));

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

  const lenF = loadPushLen(b, p, pc);
  emitBoundsCheck(b, p, lenF, gidX, labelEnd);

  // x = A[gidX]
  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrF32, ptrA, bufA.varId, p.const0u, gidX]);
  const x = b.id();
  b.emit(Op.Load, [p.tF32, x, ptrA]);

  // x3 = x * x * x
  const x2 = b.id();
  b.emit(Op.FMul, [p.tF32, x2, x, x]);
  const x3 = b.id();
  b.emit(Op.FMul, [p.tF32, x3, x2, x]);

  // inner = x + 0.044715 * x^3
  const cx3 = b.id();
  b.emit(Op.FMul, [p.tF32, cx3, constCoeff, x3]);
  const inner = b.id();
  b.emit(Op.FAdd, [p.tF32, inner, x, cx3]);

  // tanhArg = sqrt(2/pi) * inner
  const tanhArg = b.id();
  b.emit(Op.FMul, [p.tF32, tanhArg, constSqrt2OverPi, inner]);

  // t = tanh(tanhArg)
  const t = b.id();
  b.emit(Op.ExtInst, [p.tF32, t, p.glslStd, GLSLstd450.Tanh, tanhArg]);

  // result = 0.5 * x * (1 + t)
  const onePlusT = b.id();
  b.emit(Op.FAdd, [p.tF32, onePlusT, constOne, t]);
  const halfX = b.id();
  b.emit(Op.FMul, [p.tF32, halfX, constHalf, x]);
  const valC = b.id();
  b.emit(Op.FMul, [p.tF32, valC, halfX, onePlusT]);

  const ptrC = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrF32, ptrC, bufC.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrC, valC]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

// ── Kernel: GELU backward ───────────────────────────────────────────────────

/**
 * GELU backward: C[i] = B[i] * gelu_grad(A[i])
 * gelu_grad(x) = 0.5*(1+tanh(s)) + 0.5*x*(1-tanh²(s))*s'
 * where s = sqrt(2/pi)*(x + 0.044715*x³), s' = sqrt(2/pi)*(1 + 3*0.044715*x²)
 *
 * Bindings: 0=A(input), 1=B(gradOutput), 2=C(gradInput)
 * Push constants: { len: f32, _unused: f32 }
 */
export function kernelGeluBackward(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufB = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, true);
  const bufC = declareStorageBuffer(b, p.tF32, p.tU32, 0, 2, false);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

  const constHalf = b.id(); b.constantF32(p.tF32, constHalf, 0.5);
  const constOne = b.id(); b.constantF32(p.tF32, constOne, 1.0);
  const constThree = b.id(); b.constantF32(p.tF32, constThree, 3.0);
  const constCoeff = b.id(); b.constantF32(p.tF32, constCoeff, 0.044715);
  const constSqrt2OverPi = b.id(); b.constantF32(p.tF32, constSqrt2OverPi, Math.sqrt(2.0 / Math.PI));

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  const labelEntry = b.id();
  const labelEnd = b.id();

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  b.emit(Op.Label, [labelEntry]);

  const gidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gidX = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, gidX, gidVec, 0]);

  const lenF = loadPushLen(b, p, pc);
  emitBoundsCheck(b, p, lenF, gidX, labelEnd);

  // x = A[gidX]
  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrF32, ptrA, bufA.varId, p.const0u, gidX]);
  const x = b.id();
  b.emit(Op.Load, [p.tF32, x, ptrA]);

  // dout = B[gidX]
  const ptrB = b.id();
  b.emit(Op.AccessChain, [bufB.tPtrF32, ptrB, bufB.varId, p.const0u, gidX]);
  const dout = b.id();
  b.emit(Op.Load, [p.tF32, dout, ptrB]);

  // x² and x³
  const x2 = b.id(); b.emit(Op.FMul, [p.tF32, x2, x, x]);
  const x3 = b.id(); b.emit(Op.FMul, [p.tF32, x3, x2, x]);

  // s = sqrt(2/pi) * (x + 0.044715*x³)
  const cx3 = b.id(); b.emit(Op.FMul, [p.tF32, cx3, constCoeff, x3]);
  const xPcx3 = b.id(); b.emit(Op.FAdd, [p.tF32, xPcx3, x, cx3]);
  const s = b.id(); b.emit(Op.FMul, [p.tF32, s, constSqrt2OverPi, xPcx3]);

  // t = tanh(s), sech² = 1 - t²
  const t = b.id(); b.emit(Op.ExtInst, [p.tF32, t, p.glslStd, GLSLstd450.Tanh, s]);
  const t2 = b.id(); b.emit(Op.FMul, [p.tF32, t2, t, t]);
  const sech2 = b.id(); b.emit(Op.FSub, [p.tF32, sech2, constOne, t2]);

  // ds = sqrt(2/pi) * (1 + 3*0.044715*x²)
  const c3 = b.id(); b.emit(Op.FMul, [p.tF32, c3, constThree, constCoeff]);
  const c3x2 = b.id(); b.emit(Op.FMul, [p.tF32, c3x2, c3, x2]);
  const onePC3x2 = b.id(); b.emit(Op.FAdd, [p.tF32, onePC3x2, constOne, c3x2]);
  const ds = b.id(); b.emit(Op.FMul, [p.tF32, ds, constSqrt2OverPi, onePC3x2]);

  // gelu_grad = 0.5*(1+t) + 0.5*x*sech²*ds
  const onePt = b.id(); b.emit(Op.FAdd, [p.tF32, onePt, constOne, t]);
  const halfOnePt = b.id(); b.emit(Op.FMul, [p.tF32, halfOnePt, constHalf, onePt]);
  const xSech2 = b.id(); b.emit(Op.FMul, [p.tF32, xSech2, x, sech2]);
  const xSech2Ds = b.id(); b.emit(Op.FMul, [p.tF32, xSech2Ds, xSech2, ds]);
  const halfXSD = b.id(); b.emit(Op.FMul, [p.tF32, halfXSD, constHalf, xSech2Ds]);
  const geluGrad = b.id(); b.emit(Op.FAdd, [p.tF32, geluGrad, halfOnePt, halfXSD]);

  // result = dout * gelu_grad
  const result = b.id(); b.emit(Op.FMul, [p.tF32, result, dout, geluGrad]);
  const ptrC = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrF32, ptrC, bufC.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrC, result]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);
  return b.build();
}

// ── Kernel: ReLU backward ───────────────────────────────────────────────────

/**
 * ReLU backward: C[i] = A[i] > 0 ? B[i] : 0
 * Bindings: 0=A(input), 1=B(gradOutput), 2=C(gradInput)
 * Push constants: { len: f32, _unused: f32 }
 */
export function kernelReluBackward(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufB = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, true);
  const bufC = declareStorageBuffer(b, p.tF32, p.tU32, 0, 2, false);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  const labelEntry = b.id();
  const labelEnd = b.id();

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  b.emit(Op.Label, [labelEntry]);

  const gidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gidX = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, gidX, gidVec, 0]);

  const lenF = loadPushLen(b, p, pc);
  emitBoundsCheck(b, p, lenF, gidX, labelEnd);

  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrF32, ptrA, bufA.varId, p.const0u, gidX]);
  const x = b.id();
  b.emit(Op.Load, [p.tF32, x, ptrA]);

  const ptrB = b.id();
  b.emit(Op.AccessChain, [bufB.tPtrF32, ptrB, bufB.varId, p.const0u, gidX]);
  const dout = b.id();
  b.emit(Op.Load, [p.tF32, dout, ptrB]);

  // result = x > 0 ? dout : 0
  const cmp = b.id();
  b.emit(Op.FOrdGreaterThan, [p.tBool, cmp, x, p.const0f]);
  const result = b.id();
  b.emit(Op.Select, [p.tF32, result, cmp, dout, p.const0f]);

  const ptrC = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrF32, ptrC, bufC.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrC, result]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);
  return b.build();
}

/**
 * Clamp backward: result[i] = (input[i] > lo && input[i] < hi) ? gradOut[i] : 0.0
 * Push constants: { len: f32, lo: f32, hi: f32 } — 12 bytes
 * Bindings: 0=input (readonly), 1=gradOutput (readonly), 2=output
 */
export function kernelClampBackward(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufInput = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufGrad = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, true);
  const bufOut = declareStorageBuffer(b, p.tF32, p.tU32, 0, 2, false);
  const pc = declareParamsPushConstant(b, p.tF32, 3);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  const labelEntry = b.id();
  const labelEnd = b.id();

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  b.emit(Op.Label, [labelEntry]);

  const gidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gidX = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, gidX, gidVec, 0]);

  // Load lo and hi from push constants
  const ptrLo = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrLo, pc.varId, p.const1u]);
  const lo = b.id();
  b.emit(Op.Load, [p.tF32, lo, ptrLo]);

  const ptrHi = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrHi, pc.varId, p.const2u]);
  const hi = b.id();
  b.emit(Op.Load, [p.tF32, hi, ptrHi]);

  const lenF = loadPushLen(b, p, pc);
  emitBoundsCheck(b, p, lenF, gidX, labelEnd);

  // Load input value
  const ptrIn = b.id();
  b.emit(Op.AccessChain, [bufInput.tPtrF32, ptrIn, bufInput.varId, p.const0u, gidX]);
  const x = b.id();
  b.emit(Op.Load, [p.tF32, x, ptrIn]);

  // Load gradient
  const ptrG = b.id();
  b.emit(Op.AccessChain, [bufGrad.tPtrF32, ptrG, bufGrad.varId, p.const0u, gidX]);
  const dout = b.id();
  b.emit(Op.Load, [p.tF32, dout, ptrG]);

  // result = (x > lo && x < hi) ? dout : 0.0
  const cmpLo = b.id();
  b.emit(Op.FOrdGreaterThan, [p.tBool, cmpLo, x, lo]);
  const cmpHi = b.id();
  b.emit(Op.FOrdLessThan, [p.tBool, cmpHi, x, hi]);
  const inRange = b.id();
  b.emit(Op.LogicalAnd, [p.tBool, inRange, cmpLo, cmpHi]);
  const result = b.id();
  b.emit(Op.Select, [p.tF32, result, inRange, dout, p.const0f]);

  const ptrOut = b.id();
  b.emit(Op.AccessChain, [bufOut.tPtrF32, ptrOut, bufOut.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrOut, result]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);
  return b.build();
}

// ── Kernel: softCap forward ──────────────────────────────────────────────────

/**
 * softCap forward: output[i] = tanh(input[i] / cap) * cap
 * Smooth logit capping (PaLM/Gemma technique).
 *
 * Bindings: 0=input(f32, readonly), 1=output(f32)
 * Push constants: { len: f32, cap: f32 }
 */
export function kernelSoftCapForward(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufC = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, false);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

  // Constants for overflow protection
  const constNeg80 = b.id(); b.constantF32(p.tF32, constNeg80, -80.0);
  const constPos80 = b.id(); b.constantF32(p.tF32, constPos80, 80.0);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  const labelEntry = b.id();
  const labelEnd = b.id();

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  b.emit(Op.Label, [labelEntry]);

  const gidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gidX = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, gidX, gidVec, 0]);

  const lenF = loadPushLen(b, p, pc);
  const cap = loadPushScalar(b, p, pc);
  emitBoundsCheck(b, p, lenF, gidX, labelEnd);

  // x = input[gidX]
  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrF32, ptrA, bufA.varId, p.const0u, gidX]);
  const x = b.id();
  b.emit(Op.Load, [p.tF32, x, ptrA]);

  // scaled = x / cap
  const scaled = b.id();
  b.emit(Op.FDiv, [p.tF32, scaled, x, cap]);

  // clamped = clamp(scaled, -80, 80) — prevent tanh overflow
  const clamped = b.id();
  b.emit(Op.ExtInst, [p.tF32, clamped, p.glslStd, GLSLstd450.FClamp, scaled, constNeg80, constPos80]);

  // t = tanh(clamped)
  const t = b.id();
  b.emit(Op.ExtInst, [p.tF32, t, p.glslStd, GLSLstd450.Tanh, clamped]);

  // result = t * cap
  const result = b.id();
  b.emit(Op.FMul, [p.tF32, result, t, cap]);

  const ptrC = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrF32, ptrC, bufC.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrC, result]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

// ── Kernel: softCap backward ─────────────────────────────────────────────────

/**
 * softCap backward: gradInput[i] = gradOutput[i] * (1 - tanh(input[i]/cap)^2)
 * Derivative of tanh(x/cap)*cap w.r.t. x is (1 - tanh(x/cap)^2).
 *
 * Bindings: 0=gradOutput(f32, readonly), 1=input(f32, readonly), 2=gradInput(f32)
 * Push constants: { len: f32, cap: f32 }
 */
export function kernelSoftCapBackward(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufGrad = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufInput = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, true);
  const bufOut = declareStorageBuffer(b, p.tF32, p.tU32, 0, 2, false);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

  // Constants
  const constOne = b.id(); b.constantF32(p.tF32, constOne, 1.0);
  const constNeg80 = b.id(); b.constantF32(p.tF32, constNeg80, -80.0);
  const constPos80 = b.id(); b.constantF32(p.tF32, constPos80, 80.0);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  const labelEntry = b.id();
  const labelEnd = b.id();

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  b.emit(Op.Label, [labelEntry]);

  const gidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gidX = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, gidX, gidVec, 0]);

  const lenF = loadPushLen(b, p, pc);
  const cap = loadPushScalar(b, p, pc);
  emitBoundsCheck(b, p, lenF, gidX, labelEnd);

  // dout = gradOutput[gidX]
  const ptrG = b.id();
  b.emit(Op.AccessChain, [bufGrad.tPtrF32, ptrG, bufGrad.varId, p.const0u, gidX]);
  const dout = b.id();
  b.emit(Op.Load, [p.tF32, dout, ptrG]);

  // x = input[gidX]
  const ptrIn = b.id();
  b.emit(Op.AccessChain, [bufInput.tPtrF32, ptrIn, bufInput.varId, p.const0u, gidX]);
  const x = b.id();
  b.emit(Op.Load, [p.tF32, x, ptrIn]);

  // scaled = x / cap
  const scaled = b.id();
  b.emit(Op.FDiv, [p.tF32, scaled, x, cap]);

  // clamped = clamp(scaled, -80, 80)
  const clamped = b.id();
  b.emit(Op.ExtInst, [p.tF32, clamped, p.glslStd, GLSLstd450.FClamp, scaled, constNeg80, constPos80]);

  // t = tanh(clamped)
  const t = b.id();
  b.emit(Op.ExtInst, [p.tF32, t, p.glslStd, GLSLstd450.Tanh, clamped]);

  // t2 = t * t
  const t2 = b.id();
  b.emit(Op.FMul, [p.tF32, t2, t, t]);

  // deriv = 1 - t2
  const deriv = b.id();
  b.emit(Op.FSub, [p.tF32, deriv, constOne, t2]);

  // result = dout * deriv
  const result = b.id();
  b.emit(Op.FMul, [p.tF32, result, dout, deriv]);

  const ptrOut = b.id();
  b.emit(Op.AccessChain, [bufOut.tPtrF32, ptrOut, bufOut.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrOut, result]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

// ── Kernel: softCap forward vec4 ─────────────────────────────────────────────

/**
 * Vec4 softCap forward: out[i] = tanh(clamp(in[i]/cap, -80, 80)) * cap
 * Processes 4 elements per thread for 4x throughput.
 *
 * Bindings: 0=input(vec4,ro), 1=output(vec4,wo)
 * Push constants: { vec4Count: f32, cap: f32 }
 */
export function kernelSoftCapForwardVec4(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const tVec4F32 = b.id();
  b.typeVector(tVec4F32, p.tF32, 4);

  const bufA = declareStorageBufferVec4(b, tVec4F32, 0, 0, true);
  const bufC = declareStorageBufferVec4(b, tVec4F32, 0, 1, false);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

  // Constants for overflow protection
  const constNeg80 = b.id(); b.constantF32(p.tF32, constNeg80, -80.0);
  const constPos80 = b.id(); b.constantF32(p.tF32, constPos80, 80.0);
  const neg80Vec = b.id(); b.constantComposite(tVec4F32, neg80Vec, [constNeg80, constNeg80, constNeg80, constNeg80]);
  const pos80Vec = b.id(); b.constantComposite(tVec4F32, pos80Vec, [constPos80, constPos80, constPos80, constPos80]);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  const labelEntry = b.id();
  const labelEnd = b.id();

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  b.emit(Op.Label, [labelEntry]);

  const gidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gidX = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, gidX, gidVec, 0]);

  const lenF = loadPushLen(b, p, pc);
  const cap = loadPushScalar(b, p, pc);
  emitBoundsCheck(b, p, lenF, gidX, labelEnd);

  // Splat cap to vec4
  const capVec = b.id();
  b.emit(Op.CompositeConstruct, [tVec4F32, capVec, cap, cap, cap, cap]);

  // Load vec4 input
  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrVec4, ptrA, bufA.varId, p.const0u, gidX]);
  const valA = b.id();
  b.emit(Op.Load, [tVec4F32, valA, ptrA]);

  // scaled = input / cap
  const scaled = b.id();
  b.emit(Op.FDiv, [tVec4F32, scaled, valA, capVec]);

  // clamped = clamp(scaled, -80, 80)
  const clamped = b.id();
  b.emit(Op.ExtInst, [tVec4F32, clamped, p.glslStd, GLSLstd450.FClamp, scaled, neg80Vec, pos80Vec]);

  // t = tanh(clamped)
  const t = b.id();
  b.emit(Op.ExtInst, [tVec4F32, t, p.glslStd, GLSLstd450.Tanh, clamped]);

  // result = t * cap
  const result = b.id();
  b.emit(Op.FMul, [tVec4F32, result, t, capVec]);

  const ptrC = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrVec4, ptrC, bufC.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrC, result]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

// ── Kernel: softCap backward vec4 ────────────────────────────────────────────

/**
 * Vec4 softCap backward: gradInput[i] = gradOutput[i] * (1 - tanh(input[i]/cap)^2)
 * Processes 4 elements per thread for 4x throughput.
 *
 * Bindings: 0=gradOutput(vec4,ro), 1=input(vec4,ro), 2=gradInput(vec4,wo)
 * Push constants: { vec4Count: f32, cap: f32 }
 */
export function kernelSoftCapBackwardVec4(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const tVec4F32 = b.id();
  b.typeVector(tVec4F32, p.tF32, 4);

  const bufGrad = declareStorageBufferVec4(b, tVec4F32, 0, 0, true);
  const bufInput = declareStorageBufferVec4(b, tVec4F32, 0, 1, true);
  const bufOut = declareStorageBufferVec4(b, tVec4F32, 0, 2, false);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

  // Constants
  const constOne = b.id(); b.constantF32(p.tF32, constOne, 1.0);
  const constNeg80 = b.id(); b.constantF32(p.tF32, constNeg80, -80.0);
  const constPos80 = b.id(); b.constantF32(p.tF32, constPos80, 80.0);
  const oneVec = b.id(); b.constantComposite(tVec4F32, oneVec, [constOne, constOne, constOne, constOne]);
  const neg80Vec = b.id(); b.constantComposite(tVec4F32, neg80Vec, [constNeg80, constNeg80, constNeg80, constNeg80]);
  const pos80Vec = b.id(); b.constantComposite(tVec4F32, pos80Vec, [constPos80, constPos80, constPos80, constPos80]);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  const labelEntry = b.id();
  const labelEnd = b.id();

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  b.emit(Op.Label, [labelEntry]);

  const gidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gidX = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, gidX, gidVec, 0]);

  const lenF = loadPushLen(b, p, pc);
  const cap = loadPushScalar(b, p, pc);
  emitBoundsCheck(b, p, lenF, gidX, labelEnd);

  // Splat cap to vec4
  const capVec = b.id();
  b.emit(Op.CompositeConstruct, [tVec4F32, capVec, cap, cap, cap, cap]);

  // dout = gradOutput[gidX]
  const ptrG = b.id();
  b.emit(Op.AccessChain, [bufGrad.tPtrVec4, ptrG, bufGrad.varId, p.const0u, gidX]);
  const dout = b.id();
  b.emit(Op.Load, [tVec4F32, dout, ptrG]);

  // x = input[gidX]
  const ptrIn = b.id();
  b.emit(Op.AccessChain, [bufInput.tPtrVec4, ptrIn, bufInput.varId, p.const0u, gidX]);
  const x = b.id();
  b.emit(Op.Load, [tVec4F32, x, ptrIn]);

  // scaled = x / cap
  const scaled = b.id();
  b.emit(Op.FDiv, [tVec4F32, scaled, x, capVec]);

  // clamped = clamp(scaled, -80, 80)
  const clamped = b.id();
  b.emit(Op.ExtInst, [tVec4F32, clamped, p.glslStd, GLSLstd450.FClamp, scaled, neg80Vec, pos80Vec]);

  // t = tanh(clamped)
  const t = b.id();
  b.emit(Op.ExtInst, [tVec4F32, t, p.glslStd, GLSLstd450.Tanh, clamped]);

  // t2 = t * t
  const t2 = b.id();
  b.emit(Op.FMul, [tVec4F32, t2, t, t]);

  // deriv = 1 - t2
  const deriv = b.id();
  b.emit(Op.FSub, [tVec4F32, deriv, oneVec, t2]);

  // result = dout * deriv
  const result = b.id();
  b.emit(Op.FMul, [tVec4F32, result, dout, deriv]);

  const ptrOut = b.id();
  b.emit(Op.AccessChain, [bufOut.tPtrVec4, ptrOut, bufOut.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrOut, result]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

// ── Vec4 Kernels: 4 elements per thread, 128-bit loads/stores ───────────────

/**
 * Vec4 binary op: C[i] = op(A[i], B[i]) where each index is a vec4.
 * Push constants: { vec4Count: f32, _unused: f32 }
 * Bindings: 0=A(vec4,ro), 1=B(vec4,ro), 2=C(vec4,wo)
 */
function kernelBinaryOpVec4(opcode: number, wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const tVec4F32 = b.id();
  b.typeVector(tVec4F32, p.tF32, 4);

  const bufA = declareStorageBufferVec4(b, tVec4F32, 0, 0, true);
  const bufB = declareStorageBufferVec4(b, tVec4F32, 0, 1, true);
  const bufC = declareStorageBufferVec4(b, tVec4F32, 0, 2, false);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  const labelEntry = b.id();
  const labelEnd = b.id();

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  b.emit(Op.Label, [labelEntry]);

  const gidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gidX = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, gidX, gidVec, 0]);

  const lenF = loadPushLen(b, p, pc);
  emitBoundsCheck(b, p, lenF, gidX, labelEnd);

  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrVec4, ptrA, bufA.varId, p.const0u, gidX]);
  const valA = b.id();
  b.emit(Op.Load, [tVec4F32, valA, ptrA]);

  const ptrB = b.id();
  b.emit(Op.AccessChain, [bufB.tPtrVec4, ptrB, bufB.varId, p.const0u, gidX]);
  const valB = b.id();
  b.emit(Op.Load, [tVec4F32, valB, ptrB]);

  const valC = b.id();
  b.emit(opcode, [tVec4F32, valC, valA, valB]);

  const ptrC = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrVec4, ptrC, bufC.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrC, valC]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

export function kernelAddVec4(wgSize = 256): Uint32Array { return kernelBinaryOpVec4(Op.FAdd, wgSize); }
export function kernelSubVec4(wgSize = 256): Uint32Array { return kernelBinaryOpVec4(Op.FSub, wgSize); }
export function kernelMulVec4(wgSize = 256): Uint32Array { return kernelBinaryOpVec4(Op.FMul, wgSize); }
export function kernelDivVec4(wgSize = 256): Uint32Array { return kernelBinaryOpVec4(Op.FDiv, wgSize); }

/**
 * Vec4 scale: C[i] = A[i] * scalar (vec4 × scalar)
 * Push constants: { vec4Count: f32, scalar: f32 }
 * Bindings: 0=A(vec4,ro), 1=C(vec4,wo)
 */
export function kernelScaleVec4(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const tVec4F32 = b.id();
  b.typeVector(tVec4F32, p.tF32, 4);

  const bufA = declareStorageBufferVec4(b, tVec4F32, 0, 0, true);
  const bufC = declareStorageBufferVec4(b, tVec4F32, 0, 1, false);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  const labelEntry = b.id();
  const labelEnd = b.id();

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  b.emit(Op.Label, [labelEntry]);

  const gidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gidX = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, gidX, gidVec, 0]);

  // Load scalar before bounds check
  const scalar = loadPushScalar(b, p, pc);

  const lenF = loadPushLen(b, p, pc);
  emitBoundsCheck(b, p, lenF, gidX, labelEnd);

  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrVec4, ptrA, bufA.varId, p.const0u, gidX]);
  const valA = b.id();
  b.emit(Op.Load, [tVec4F32, valA, ptrA]);

  // OpVectorTimesScalar: vec4 * f32
  const valC = b.id();
  b.emit(Op.VectorTimesScalar, [tVec4F32, valC, valA, scalar]);

  const ptrC = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrVec4, ptrC, bufC.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrC, valC]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

/**
 * Vec4 neg: C[i] = -A[i]
 * Push constants: { vec4Count: f32, _unused: f32 }
 * Bindings: 0=A(vec4,ro), 1=C(vec4,wo)
 */
export function kernelNegVec4(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const tVec4F32 = b.id();
  b.typeVector(tVec4F32, p.tF32, 4);

  const bufA = declareStorageBufferVec4(b, tVec4F32, 0, 0, true);
  const bufC = declareStorageBufferVec4(b, tVec4F32, 0, 1, false);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  const labelEntry = b.id();
  const labelEnd = b.id();

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  b.emit(Op.Label, [labelEntry]);

  const gidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gidX = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, gidX, gidVec, 0]);

  const lenF = loadPushLen(b, p, pc);
  emitBoundsCheck(b, p, lenF, gidX, labelEnd);

  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrVec4, ptrA, bufA.varId, p.const0u, gidX]);
  const valA = b.id();
  b.emit(Op.Load, [tVec4F32, valA, ptrA]);

  const valC = b.id();
  b.emit(Op.FNegate, [tVec4F32, valC, valA]);

  const ptrC = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrVec4, ptrC, bufC.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrC, valC]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

/**
 * Vec4 unary GLSL op (exp, log, sqrt): C[i] = op(A[i]) on vec4
 * Push constants: { vec4Count: f32, _unused: f32 }
 * Bindings: 0=A(vec4,ro), 1=C(vec4,wo)
 */
function kernelUnaryGlslVec4(glslOp: number, wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const tVec4F32 = b.id();
  b.typeVector(tVec4F32, p.tF32, 4);

  const bufA = declareStorageBufferVec4(b, tVec4F32, 0, 0, true);
  const bufC = declareStorageBufferVec4(b, tVec4F32, 0, 1, false);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  const labelEntry = b.id();
  const labelEnd = b.id();

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  b.emit(Op.Label, [labelEntry]);

  const gidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gidX = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, gidX, gidVec, 0]);

  const lenF = loadPushLen(b, p, pc);
  emitBoundsCheck(b, p, lenF, gidX, labelEnd);

  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrVec4, ptrA, bufA.varId, p.const0u, gidX]);
  const valA = b.id();
  b.emit(Op.Load, [tVec4F32, valA, ptrA]);

  const valC = b.id();
  b.emit(Op.ExtInst, [tVec4F32, valC, p.glslStd, glslOp, valA]);

  const ptrC = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrVec4, ptrC, bufC.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrC, valC]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

export function kernelExpVec4(wgSize = 256): Uint32Array { return kernelUnaryGlslVec4(GLSLstd450.Exp, wgSize); }
export function kernelLogVec4(wgSize = 256): Uint32Array { return kernelUnaryGlslVec4(GLSLstd450.Log, wgSize); }
export function kernelSqrtVec4(wgSize = 256): Uint32Array { return kernelUnaryGlslVec4(GLSLstd450.Sqrt, wgSize); }

/**
 * Vec4 ReLU: C[i] = max(A[i], vec4(0))
 * Push constants: { vec4Count: f32, _unused: f32 }
 * Bindings: 0=A(vec4,ro), 1=C(vec4,wo)
 */
export function kernelReluVec4(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const tVec4F32 = b.id();
  b.typeVector(tVec4F32, p.tF32, 4);

  const bufA = declareStorageBufferVec4(b, tVec4F32, 0, 0, true);
  const bufC = declareStorageBufferVec4(b, tVec4F32, 0, 1, false);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

  // vec4(0, 0, 0, 0)
  const zeroVec4 = b.id();
  b.constantComposite(tVec4F32, zeroVec4, [p.const0f, p.const0f, p.const0f, p.const0f]);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  const labelEntry = b.id();
  const labelEnd = b.id();

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  b.emit(Op.Label, [labelEntry]);

  const gidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gidX = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, gidX, gidVec, 0]);

  const lenF = loadPushLen(b, p, pc);
  emitBoundsCheck(b, p, lenF, gidX, labelEnd);

  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrVec4, ptrA, bufA.varId, p.const0u, gidX]);
  const valA = b.id();
  b.emit(Op.Load, [tVec4F32, valA, ptrA]);

  // max(valA, vec4(0))
  const valC = b.id();
  b.emit(Op.ExtInst, [tVec4F32, valC, p.glslStd, GLSLstd450.FMax, valA, zeroVec4]);

  const ptrC = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrVec4, ptrC, bufC.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrC, valC]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

// ── Kernel: ClampMin ─────────────────────────────────────────────────────────

/**
 * C[i] = max(A[i], scalar)
 * Like ReLU but with a configurable floor instead of 0.
 * Used to prevent log(0) = -Inf in cross-entropy.
 * Bindings: 0=A(in), 1=C(out)
 * Push constants: { len: f32, scalar: f32 }
 */
export function kernelClampMin(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufC = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, false);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

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

  // Load min-value scalar before bounds check
  const scalar = loadPushScalar(b, p, pc);

  const lenF = loadPushLen(b, p, pc);
  emitBoundsCheck(b, p, lenF, gidX, labelEnd);

  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrF32, ptrA, bufA.varId, p.const0u, gidX]);
  const valA = b.id();
  b.emit(Op.Load, [p.tF32, valA, ptrA]);

  // max(valA, scalar) — clamp to minimum value
  const valC = b.id();
  b.emit(Op.ExtInst, [p.tF32, valC, p.glslStd, GLSLstd450.FMax, valA, scalar]);

  const ptrC = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrF32, ptrC, bufC.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrC, valC]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

export function kernelClampMinVec4(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const tVec4F32 = b.id();
  b.typeVector(tVec4F32, p.tF32, 4);

  const bufA = declareStorageBufferVec4(b, tVec4F32, 0, 0, true);
  const bufC = declareStorageBufferVec4(b, tVec4F32, 0, 1, false);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  const labelEntry = b.id();
  const labelEnd = b.id();

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  b.emit(Op.Label, [labelEntry]);

  const gidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gidX = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, gidX, gidVec, 0]);

  // Load min-value scalar and splat to vec4
  const scalar = loadPushScalar(b, p, pc);
  const scalarVec4 = b.id();
  b.emit(Op.CompositeConstruct, [tVec4F32, scalarVec4, scalar, scalar, scalar, scalar]);

  const lenF = loadPushLen(b, p, pc);
  emitBoundsCheck(b, p, lenF, gidX, labelEnd);

  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrVec4, ptrA, bufA.varId, p.const0u, gidX]);
  const valA = b.id();
  b.emit(Op.Load, [tVec4F32, valA, ptrA]);

  // max(valA, vec4(scalar)) — clamp each component to minimum
  const valC = b.id();
  b.emit(Op.ExtInst, [tVec4F32, valC, p.glslStd, GLSLstd450.FMax, valA, scalarVec4]);

  const ptrC = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrVec4, ptrC, bufC.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrC, valC]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

/**
 * Clamp kernel: C[i] = clamp(A[i], lo, hi)
 * Push constants: { len: f32, lo: f32, hi: f32 } — 12 bytes
 */
export function kernelClamp(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const bufA = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufC = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, false);
  const pc = declareParamsPushConstant(b, p.tF32, 3);

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

  // Load lo (member 1) and hi (member 2) scalars
  const ptrLo = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrLo, pc.varId, p.const1u]);
  const lo = b.id();
  b.emit(Op.Load, [p.tF32, lo, ptrLo]);

  const ptrHi = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrHi, pc.varId, p.const2u]);
  const hi = b.id();
  b.emit(Op.Load, [p.tF32, hi, ptrHi]);

  const lenF = loadPushLen(b, p, pc);
  emitBoundsCheck(b, p, lenF, gidX, labelEnd);

  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrF32, ptrA, bufA.varId, p.const0u, gidX]);
  const valA = b.id();
  b.emit(Op.Load, [p.tF32, valA, ptrA]);

  // clamp(valA, lo, hi) — single GLSL instruction
  const valC = b.id();
  b.emit(Op.ExtInst, [p.tF32, valC, p.glslStd, GLSLstd450.FClamp, valA, lo, hi]);

  const ptrC = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrF32, ptrC, bufC.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrC, valC]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

export function kernelClampVec4(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const tVec4F32 = b.id();
  b.typeVector(tVec4F32, p.tF32, 4);

  const bufA = declareStorageBufferVec4(b, tVec4F32, 0, 0, true);
  const bufC = declareStorageBufferVec4(b, tVec4F32, 0, 1, false);
  const pc = declareParamsPushConstant(b, p.tF32, 3);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  const labelEntry = b.id();
  const labelEnd = b.id();

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  b.emit(Op.Label, [labelEntry]);

  const gidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gidX = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, gidX, gidVec, 0]);

  // Load lo and hi scalars, splat to vec4
  const ptrLo = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrLo, pc.varId, p.const1u]);
  const lo = b.id();
  b.emit(Op.Load, [p.tF32, lo, ptrLo]);
  const loVec4 = b.id();
  b.emit(Op.CompositeConstruct, [tVec4F32, loVec4, lo, lo, lo, lo]);

  const ptrHi = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrHi, pc.varId, p.const2u]);
  const hi = b.id();
  b.emit(Op.Load, [p.tF32, hi, ptrHi]);
  const hiVec4 = b.id();
  b.emit(Op.CompositeConstruct, [tVec4F32, hiVec4, hi, hi, hi, hi]);

  const lenF = loadPushLen(b, p, pc);
  emitBoundsCheck(b, p, lenF, gidX, labelEnd);

  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrVec4, ptrA, bufA.varId, p.const0u, gidX]);
  const valA = b.id();
  b.emit(Op.Load, [tVec4F32, valA, ptrA]);

  // clamp(valA, loVec4, hiVec4)
  const valC = b.id();
  b.emit(Op.ExtInst, [tVec4F32, valC, p.glslStd, GLSLstd450.FClamp, valA, loVec4, hiVec4]);

  const ptrC = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrVec4, ptrC, bufC.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrC, valC]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

/**
 * Vec4 GELU (fused tanh approximation):
 * C[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 * Push constants: { vec4Count: f32, _unused: f32 }
 * Bindings: 0=A(vec4,ro), 1=C(vec4,wo)
 */
export function kernelGeluVec4(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);

  const tVec4F32 = b.id();
  b.typeVector(tVec4F32, p.tF32, 4);

  const bufA = declareStorageBufferVec4(b, tVec4F32, 0, 0, true);
  const bufC = declareStorageBufferVec4(b, tVec4F32, 0, 1, false);
  const pc = declareParamsPushConstant(b, p.tF32, 2);

  // Scalar constants
  const constHalf = b.id();
  b.constantF32(p.tF32, constHalf, 0.5);
  const constOne = b.id();
  b.constantF32(p.tF32, constOne, 1.0);
  const constCoeff = b.id();
  b.constantF32(p.tF32, constCoeff, 0.044715);
  const constSqrt2OverPi = b.id();
  b.constantF32(p.tF32, constSqrt2OverPi, Math.sqrt(2.0 / Math.PI));

  // Splat to vec4 constants
  const halfVec = b.id();
  b.constantComposite(tVec4F32, halfVec, [constHalf, constHalf, constHalf, constHalf]);
  const oneVec = b.id();
  b.constantComposite(tVec4F32, oneVec, [constOne, constOne, constOne, constOne]);
  const coeffVec = b.id();
  b.constantComposite(tVec4F32, coeffVec, [constCoeff, constCoeff, constCoeff, constCoeff]);
  const sqrt2piVec = b.id();
  b.constantComposite(tVec4F32, sqrt2piVec, [constSqrt2OverPi, constSqrt2OverPi, constSqrt2OverPi, constSqrt2OverPi]);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, wgSize, 1, 1);

  const labelEntry = b.id();
  const labelEnd = b.id();

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  b.emit(Op.Label, [labelEntry]);

  const gidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gidX = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, gidX, gidVec, 0]);

  const lenF = loadPushLen(b, p, pc);
  emitBoundsCheck(b, p, lenF, gidX, labelEnd);

  const ptrA = b.id();
  b.emit(Op.AccessChain, [bufA.tPtrVec4, ptrA, bufA.varId, p.const0u, gidX]);
  const x = b.id();
  b.emit(Op.Load, [tVec4F32, x, ptrA]);

  // x^2 = x * x, x^3 = x^2 * x
  const x2 = b.id();
  b.emit(Op.FMul, [tVec4F32, x2, x, x]);
  const x3 = b.id();
  b.emit(Op.FMul, [tVec4F32, x3, x2, x]);

  // inner = x + coeff * x^3
  const cx3 = b.id();
  b.emit(Op.FMul, [tVec4F32, cx3, coeffVec, x3]);
  const inner = b.id();
  b.emit(Op.FAdd, [tVec4F32, inner, x, cx3]);

  // tanhArg = sqrt(2/pi) * inner
  const tanhArg = b.id();
  b.emit(Op.FMul, [tVec4F32, tanhArg, sqrt2piVec, inner]);

  // t = tanh(tanhArg)
  const t = b.id();
  b.emit(Op.ExtInst, [tVec4F32, t, p.glslStd, GLSLstd450.Tanh, tanhArg]);

  // result = 0.5 * x * (1 + t)
  const onePlusT = b.id();
  b.emit(Op.FAdd, [tVec4F32, onePlusT, oneVec, t]);
  const halfX = b.id();
  b.emit(Op.FMul, [tVec4F32, halfX, halfVec, x]);
  const valC = b.id();
  b.emit(Op.FMul, [tVec4F32, valC, halfX, onePlusT]);

  const ptrC = b.id();
  b.emit(Op.AccessChain, [bufC.tPtrVec4, ptrC, bufC.varId, p.const0u, gidX]);
  b.emit(Op.Store, [ptrC, valC]);

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}
