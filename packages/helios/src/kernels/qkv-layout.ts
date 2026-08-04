/**
 * Fused grouped-QKV layout and RoPE kernels.
 *
 * The projection writes token-major [B*T, 3*H*D]. Flash attention consumes
 * head-major [B*H, T, D]. Q and K additionally need the HF-Llama half-split
 * rotary transform. These kernels cross that boundary once instead of
 * materialising slice, transpose, and RoPE intermediates.
 */

import {
  SpirVBuilder, Op, ExecutionModel, ExecutionMode, StorageClass, Decoration,
  FunctionControl, preamble, declareStorageBuffer,
} from "./helpers.js";

function declareU32PushConstants(
  b: SpirVBuilder,
  tU32: number,
  count: number,
): { variable: number; pointer: number; indices: number[] } {
  const memberTypes = Array(count).fill(tU32) as number[];
  const struct = b.id();
  b.typeStruct(struct, memberTypes);
  b.addDecorate(struct, Decoration.Block);
  for (let i = 0; i < count; i++) b.addMemberDecorate(struct, i, Decoration.Offset, i * 4);
  const structPointer = b.id();
  b.typePointer(structPointer, StorageClass.PushConstant, struct);
  const scalarPointer = b.id();
  b.typePointer(scalarPointer, StorageClass.PushConstant, tU32);
  const variable = b.id();
  b.variable(structPointer, variable, StorageClass.PushConstant);
  const indices: number[] = [];
  for (let i = 0; i < count; i++) {
    const index = b.id();
    b.constant(tU32, index, i);
    indices.push(index);
  }
  return { variable, pointer: scalarPointer, indices };
}

function loadPushU32(
  b: SpirVBuilder,
  tU32: number,
  pc: { variable: number; pointer: number; indices: number[] },
  index: number,
): number {
  const pointer = b.id();
  b.emit(Op.AccessChain, [pc.pointer, pointer, pc.variable, pc.indices[index]]);
  const value = b.id();
  b.emit(Op.Load, [tU32, value, pointer]);
  return value;
}

function loadF32(
  b: SpirVBuilder,
  tF32: number,
  const0u: number,
  buffer: { varId: number; tPtrF32: number },
  index: number,
): number {
  const pointer = b.id();
  b.emit(Op.AccessChain, [buffer.tPtrF32, pointer, buffer.varId, const0u, index]);
  const value = b.id();
  b.emit(Op.Load, [tF32, value, pointer]);
  return value;
}

function storeF32(
  b: SpirVBuilder,
  const0u: number,
  buffer: { varId: number; tPtrF32: number },
  index: number,
  value: number,
): void {
  const pointer = b.id();
  b.emit(Op.AccessChain, [buffer.tPtrF32, pointer, buffer.varId, const0u, index]);
  b.emit(Op.Store, [pointer, value]);
}

/**
 * Bindings: 0=qkv, 1=cos, 2=sin, 3=q, 4=k, 5=v.
 * Push u32: [totalPairs, T, H, headDim, modelDim].
 * One invocation transforms one half-split pair for all three outputs.
 */
export function kernelQkvHeadMajorRope(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);
  const qkv = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const cos = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, true);
  const sin = declareStorageBuffer(b, p.tF32, p.tU32, 0, 2, true);
  const qOut = declareStorageBuffer(b, p.tF32, p.tU32, 0, 3, false, true);
  const kOut = declareStorageBuffer(b, p.tF32, p.tU32, 0, 4, false, true);
  const vOut = declareStorageBuffer(b, p.tF32, p.tU32, 0, 5, false, true);
  const pc = declareU32PushConstants(b, p.tU32, 5);
  const three = b.id(); b.constant(p.tU32, three, 3);

  const main = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, main, "main", [p.vGlobalId]);
  b.addExecutionMode(main, ExecutionMode.LocalSize, wgSize, 1, 1);
  const entry = b.id();
  const body = b.id();
  const end = b.id();
  b.emit(Op.Function, [p.tVoid, main, FunctionControl.None, p.tFnVoid]);
  b.emit(Op.Label, [entry]);

  const gidVec = b.id(); b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gid = b.id(); b.emit(Op.CompositeExtract, [p.tU32, gid, gidVec, 0]);
  const totalPairs = loadPushU32(b, p.tU32, pc, 0);
  const outOfBounds = b.id();
  b.emit(Op.UGreaterThanEqual, [p.tBool, outOfBounds, gid, totalPairs]);
  b.emit(Op.SelectionMerge, [end, 0]);
  b.emit(Op.BranchConditional, [outOfBounds, end, body]);
  b.emit(Op.Label, [body]);

  const sequence = loadPushU32(b, p.tU32, pc, 1);
  const heads = loadPushU32(b, p.tU32, pc, 2);
  const headDim = loadPushU32(b, p.tU32, pc, 3);
  const modelDim = loadPushU32(b, p.tU32, pc, 4);
  const half = b.id(); b.emit(Op.UDiv, [p.tU32, half, headDim, p.const2u]);

  // gid = (((batch * H + head) * T + token) * half) + pair
  const pair = b.id(); b.emit(Op.UMod, [p.tU32, pair, gid, half]);
  const row = b.id(); b.emit(Op.UDiv, [p.tU32, row, gid, half]);
  const token = b.id(); b.emit(Op.UMod, [p.tU32, token, row, sequence]);
  const batchHead = b.id(); b.emit(Op.UDiv, [p.tU32, batchHead, row, sequence]);
  const head = b.id(); b.emit(Op.UMod, [p.tU32, head, batchHead, heads]);
  const batch = b.id(); b.emit(Op.UDiv, [p.tU32, batch, batchHead, heads]);

  const batchTimesT = b.id(); b.emit(Op.IMul, [p.tU32, batchTimesT, batch, sequence]);
  const sourceRow = b.id(); b.emit(Op.IAdd, [p.tU32, sourceRow, batchTimesT, token]);
  const qkvRowWidth = b.id(); b.emit(Op.IMul, [p.tU32, qkvRowWidth, modelDim, three]);
  const sourceRowBase = b.id(); b.emit(Op.IMul, [p.tU32, sourceRowBase, sourceRow, qkvRowWidth]);
  const headOffset = b.id(); b.emit(Op.IMul, [p.tU32, headOffset, head, headDim]);
  const sourceHeadBase = b.id(); b.emit(Op.IAdd, [p.tU32, sourceHeadBase, sourceRowBase, headOffset]);
  const sourceA = b.id(); b.emit(Op.IAdd, [p.tU32, sourceA, sourceHeadBase, pair]);
  const sourceB = b.id(); b.emit(Op.IAdd, [p.tU32, sourceB, sourceA, half]);
  const kSourceA = b.id(); b.emit(Op.IAdd, [p.tU32, kSourceA, sourceA, modelDim]);
  const kSourceB = b.id(); b.emit(Op.IAdd, [p.tU32, kSourceB, sourceB, modelDim]);
  const twoModelDim = b.id(); b.emit(Op.IMul, [p.tU32, twoModelDim, modelDim, p.const2u]);
  const vSourceA = b.id(); b.emit(Op.IAdd, [p.tU32, vSourceA, sourceA, twoModelDim]);
  const vSourceB = b.id(); b.emit(Op.IAdd, [p.tU32, vSourceB, sourceB, twoModelDim]);

  const tableRow = b.id(); b.emit(Op.IMul, [p.tU32, tableRow, token, half]);
  const tableIndex = b.id(); b.emit(Op.IAdd, [p.tU32, tableIndex, tableRow, pair]);
  const c = loadF32(b, p.tF32, p.const0u, cos, tableIndex);
  const s = loadF32(b, p.tF32, p.const0u, sin, tableIndex);
  const qA = loadF32(b, p.tF32, p.const0u, qkv, sourceA);
  const qB = loadF32(b, p.tF32, p.const0u, qkv, sourceB);
  const kA = loadF32(b, p.tF32, p.const0u, qkv, kSourceA);
  const kB = loadF32(b, p.tF32, p.const0u, qkv, kSourceB);
  const vA = loadF32(b, p.tF32, p.const0u, qkv, vSourceA);
  const vB = loadF32(b, p.tF32, p.const0u, qkv, vSourceB);

  const qAc = b.id(); b.emit(Op.FMul, [p.tF32, qAc, qA, c]);
  const qBs = b.id(); b.emit(Op.FMul, [p.tF32, qBs, qB, s]);
  const qRotA = b.id(); b.emit(Op.FSub, [p.tF32, qRotA, qAc, qBs]);
  const qBc = b.id(); b.emit(Op.FMul, [p.tF32, qBc, qB, c]);
  const qAs = b.id(); b.emit(Op.FMul, [p.tF32, qAs, qA, s]);
  const qRotB = b.id(); b.emit(Op.FAdd, [p.tF32, qRotB, qBc, qAs]);
  const kAc = b.id(); b.emit(Op.FMul, [p.tF32, kAc, kA, c]);
  const kBs = b.id(); b.emit(Op.FMul, [p.tF32, kBs, kB, s]);
  const kRotA = b.id(); b.emit(Op.FSub, [p.tF32, kRotA, kAc, kBs]);
  const kBc = b.id(); b.emit(Op.FMul, [p.tF32, kBc, kB, c]);
  const kAs = b.id(); b.emit(Op.FMul, [p.tF32, kAs, kA, s]);
  const kRotB = b.id(); b.emit(Op.FAdd, [p.tF32, kRotB, kBc, kAs]);

  const outputBase = b.id(); b.emit(Op.IMul, [p.tU32, outputBase, row, headDim]);
  const outputA = b.id(); b.emit(Op.IAdd, [p.tU32, outputA, outputBase, pair]);
  const outputB = b.id(); b.emit(Op.IAdd, [p.tU32, outputB, outputA, half]);
  storeF32(b, p.const0u, qOut, outputA, qRotA);
  storeF32(b, p.const0u, qOut, outputB, qRotB);
  storeF32(b, p.const0u, kOut, outputA, kRotA);
  storeF32(b, p.const0u, kOut, outputB, kRotB);
  storeF32(b, p.const0u, vOut, outputA, vA);
  storeF32(b, p.const0u, vOut, outputB, vB);

  b.emit(Op.Branch, [end]);
  b.emit(Op.Label, [end]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);
  return b.build();
}

/**
 * Bindings: 0=branch grad [B*H,T,D], 1=cos, 2=inverseSin, 3=grouped grad.
 * Push u32: [totalGroupedElements, T, H, headDim, modelDim, which].
 * All grouped elements are written. The selected Q/K/V segment receives the
 * inverse layout/rotation; the other two segments receive exact zero.
 */
export function kernelQkvHeadMajorRopeBackward(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);
  const grad = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const cos = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, true);
  const inverseSin = declareStorageBuffer(b, p.tF32, p.tU32, 0, 2, true);
  const out = declareStorageBuffer(b, p.tF32, p.tU32, 0, 3, false, true);
  const pc = declareU32PushConstants(b, p.tU32, 6);
  const three = b.id(); b.constant(p.tU32, three, 3);

  const main = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, main, "main", [p.vGlobalId]);
  b.addExecutionMode(main, ExecutionMode.LocalSize, wgSize, 1, 1);
  const entry = b.id();
  const body = b.id();
  const end = b.id();
  b.emit(Op.Function, [p.tVoid, main, FunctionControl.None, p.tFnVoid]);
  b.emit(Op.Label, [entry]);

  const gidVec = b.id(); b.emit(Op.Load, [p.tVec3U32, gidVec, p.vGlobalId]);
  const gid = b.id(); b.emit(Op.CompositeExtract, [p.tU32, gid, gidVec, 0]);
  const total = loadPushU32(b, p.tU32, pc, 0);
  const outOfBounds = b.id();
  b.emit(Op.UGreaterThanEqual, [p.tBool, outOfBounds, gid, total]);
  b.emit(Op.SelectionMerge, [end, 0]);
  b.emit(Op.BranchConditional, [outOfBounds, end, body]);
  b.emit(Op.Label, [body]);

  const sequence = loadPushU32(b, p.tU32, pc, 1);
  const heads = loadPushU32(b, p.tU32, pc, 2);
  const headDim = loadPushU32(b, p.tU32, pc, 3);
  const modelDim = loadPushU32(b, p.tU32, pc, 4);
  const which = loadPushU32(b, p.tU32, pc, 5);
  const groupedWidth = b.id(); b.emit(Op.IMul, [p.tU32, groupedWidth, modelDim, three]);
  const tokenRow = b.id(); b.emit(Op.UDiv, [p.tU32, tokenRow, gid, groupedWidth]);
  const groupedCol = b.id(); b.emit(Op.UMod, [p.tU32, groupedCol, gid, groupedWidth]);
  const segment = b.id(); b.emit(Op.UDiv, [p.tU32, segment, groupedCol, modelDim]);
  const isSelectedSegment = b.id(); b.emit(Op.IEqual, [p.tBool, isSelectedSegment, segment, which]);
  const selectedBody = b.id();
  const zeroBody = b.id();
  const branchEnd = b.id();
  b.emit(Op.SelectionMerge, [branchEnd, 0]);
  b.emit(Op.BranchConditional, [isSelectedSegment, selectedBody, zeroBody]);

  // The branch writes all non-selected Q/K/V segments as zero without reading
  // the incoming head-major gradient or RoPE tables. This keeps each branch's
  // traffic equivalent to one scatter plus one inverse-layout transform.
  b.emit(Op.Label, [zeroBody]);
  storeF32(b, p.const0u, out, gid, p.const0f);
  b.emit(Op.Branch, [branchEnd]);

  b.emit(Op.Label, [selectedBody]);
  const local = b.id(); b.emit(Op.UMod, [p.tU32, local, groupedCol, modelDim]);
  const head = b.id(); b.emit(Op.UDiv, [p.tU32, head, local, headDim]);
  const dimension = b.id(); b.emit(Op.UMod, [p.tU32, dimension, local, headDim]);
  const batch = b.id(); b.emit(Op.UDiv, [p.tU32, batch, tokenRow, sequence]);
  const token = b.id(); b.emit(Op.UMod, [p.tU32, token, tokenRow, sequence]);
  const half = b.id(); b.emit(Op.UDiv, [p.tU32, half, headDim, p.const2u]);
  const pair = b.id(); b.emit(Op.UMod, [p.tU32, pair, dimension, half]);

  const batchTimesH = b.id(); b.emit(Op.IMul, [p.tU32, batchTimesH, batch, heads]);
  const batchHead = b.id(); b.emit(Op.IAdd, [p.tU32, batchHead, batchTimesH, head]);
  const bhTimesT = b.id(); b.emit(Op.IMul, [p.tU32, bhTimesT, batchHead, sequence]);
  const headMajorRow = b.id(); b.emit(Op.IAdd, [p.tU32, headMajorRow, bhTimesT, token]);
  const gradBase = b.id(); b.emit(Op.IMul, [p.tU32, gradBase, headMajorRow, headDim]);
  const gradAIndex = b.id(); b.emit(Op.IAdd, [p.tU32, gradAIndex, gradBase, pair]);
  const gradBIndex = b.id(); b.emit(Op.IAdd, [p.tU32, gradBIndex, gradAIndex, half]);
  const gA = loadF32(b, p.tF32, p.const0u, grad, gradAIndex);
  const gB = loadF32(b, p.tF32, p.const0u, grad, gradBIndex);
  const secondHalf = b.id();
  b.emit(Op.UGreaterThanEqual, [p.tBool, secondHalf, dimension, half]);
  const isRotatedBranch = b.id(); b.emit(Op.ULessThan, [p.tBool, isRotatedBranch, which, p.const2u]);
  const rotateBody = b.id();
  const rawBody = b.id();
  const selectedEnd = b.id();
  b.emit(Op.SelectionMerge, [selectedEnd, 0]);
  b.emit(Op.BranchConditional, [isRotatedBranch, rotateBody, rawBody]);

  b.emit(Op.Label, [rotateBody]);
  const tableBase = b.id(); b.emit(Op.IMul, [p.tU32, tableBase, token, half]);
  const tableIndex = b.id(); b.emit(Op.IAdd, [p.tU32, tableIndex, tableBase, pair]);
  const c = loadF32(b, p.tF32, p.const0u, cos, tableIndex);
  const sInv = loadF32(b, p.tF32, p.const0u, inverseSin, tableIndex);

  // Apply the same rotation formula with sin negated: this is R(theta)^T.
  const aCos = b.id(); b.emit(Op.FMul, [p.tF32, aCos, gA, c]);
  const bSin = b.id(); b.emit(Op.FMul, [p.tF32, bSin, gB, sInv]);
  const inverseA = b.id(); b.emit(Op.FSub, [p.tF32, inverseA, aCos, bSin]);
  const bCos = b.id(); b.emit(Op.FMul, [p.tF32, bCos, gB, c]);
  const aSin = b.id(); b.emit(Op.FMul, [p.tF32, aSin, gA, sInv]);
  const inverseB = b.id(); b.emit(Op.FAdd, [p.tF32, inverseB, bCos, aSin]);
  const rotated = b.id(); b.emit(Op.Select, [p.tF32, rotated, secondHalf, inverseB, inverseA]);
  storeF32(b, p.const0u, out, gid, rotated);
  b.emit(Op.Branch, [selectedEnd]);

  // V has no rotary transform. Select the already-loaded pair member and do
  // not touch the cosine/sine tables at all.
  b.emit(Op.Label, [rawBody]);
  const raw = b.id(); b.emit(Op.Select, [p.tF32, raw, secondHalf, gB, gA]);
  storeF32(b, p.const0u, out, gid, raw);
  b.emit(Op.Branch, [selectedEnd]);

  b.emit(Op.Label, [selectedEnd]);
  b.emit(Op.Branch, [branchEnd]);

  b.emit(Op.Label, [branchEnd]);
  b.emit(Op.Branch, [end]);
  b.emit(Op.Label, [end]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);
  return b.build();
}
