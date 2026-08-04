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
 * Push u32: [totalGroupedPairs, T, H, headDim, modelDim, which].
 * One invocation writes a half-split pair. The selected Q/K/V segment receives
 * the inverse layout/rotation; the other two segments receive exact zero.
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
  const half = b.id(); b.emit(Op.UDiv, [p.tU32, half, headDim, p.const2u]);
  const groupedWidth = b.id(); b.emit(Op.IMul, [p.tU32, groupedWidth, modelDim, three]);
  const groupedPairWidth = b.id(); b.emit(Op.UDiv, [p.tU32, groupedPairWidth, groupedWidth, p.const2u]);
  const segmentPairWidth = b.id(); b.emit(Op.UDiv, [p.tU32, segmentPairWidth, modelDim, p.const2u]);
  const tokenRow = b.id(); b.emit(Op.UDiv, [p.tU32, tokenRow, gid, groupedPairWidth]);
  const groupedPairCol = b.id(); b.emit(Op.UMod, [p.tU32, groupedPairCol, gid, groupedPairWidth]);
  const segment = b.id(); b.emit(Op.UDiv, [p.tU32, segment, groupedPairCol, segmentPairWidth]);
  const localPair = b.id(); b.emit(Op.UMod, [p.tU32, localPair, groupedPairCol, segmentPairWidth]);
  const head = b.id(); b.emit(Op.UDiv, [p.tU32, head, localPair, half]);
  const pair = b.id(); b.emit(Op.UMod, [p.tU32, pair, localPair, half]);
  const tokenRowBase = b.id(); b.emit(Op.IMul, [p.tU32, tokenRowBase, tokenRow, groupedWidth]);
  const segmentBase = b.id(); b.emit(Op.IMul, [p.tU32, segmentBase, segment, modelDim]);
  const headBase = b.id(); b.emit(Op.IMul, [p.tU32, headBase, head, headDim]);
  const rowSegmentBase = b.id(); b.emit(Op.IAdd, [p.tU32, rowSegmentBase, tokenRowBase, segmentBase]);
  const outputHeadBase = b.id(); b.emit(Op.IAdd, [p.tU32, outputHeadBase, rowSegmentBase, headBase]);
  const outputA = b.id(); b.emit(Op.IAdd, [p.tU32, outputA, outputHeadBase, pair]);
  const outputB = b.id(); b.emit(Op.IAdd, [p.tU32, outputB, outputA, half]);
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
  storeF32(b, p.const0u, out, outputA, p.const0f);
  storeF32(b, p.const0u, out, outputB, p.const0f);
  b.emit(Op.Branch, [branchEnd]);

  b.emit(Op.Label, [selectedBody]);
  const batch = b.id(); b.emit(Op.UDiv, [p.tU32, batch, tokenRow, sequence]);
  const token = b.id(); b.emit(Op.UMod, [p.tU32, token, tokenRow, sequence]);

  const batchTimesH = b.id(); b.emit(Op.IMul, [p.tU32, batchTimesH, batch, heads]);
  const batchHead = b.id(); b.emit(Op.IAdd, [p.tU32, batchHead, batchTimesH, head]);
  const bhTimesT = b.id(); b.emit(Op.IMul, [p.tU32, bhTimesT, batchHead, sequence]);
  const headMajorRow = b.id(); b.emit(Op.IAdd, [p.tU32, headMajorRow, bhTimesT, token]);
  const gradBase = b.id(); b.emit(Op.IMul, [p.tU32, gradBase, headMajorRow, headDim]);
  const gradAIndex = b.id(); b.emit(Op.IAdd, [p.tU32, gradAIndex, gradBase, pair]);
  const gradBIndex = b.id(); b.emit(Op.IAdd, [p.tU32, gradBIndex, gradAIndex, half]);
  const gA = loadF32(b, p.tF32, p.const0u, grad, gradAIndex);
  const gB = loadF32(b, p.tF32, p.const0u, grad, gradBIndex);
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
  storeF32(b, p.const0u, out, outputA, inverseA);
  storeF32(b, p.const0u, out, outputB, inverseB);
  b.emit(Op.Branch, [selectedEnd]);

  // V has no rotary transform. Write the already-loaded pair and do not touch
  // the cosine/sine tables at all.
  b.emit(Op.Label, [rawBody]);
  storeF32(b, p.const0u, out, outputA, gA);
  storeF32(b, p.const0u, out, outputB, gB);
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

/**
 * Combined Q/K/V inverse-layout backward.
 *
 * Bindings: 0=dQ, 1=dK, 2=dV, 3=cos, 4=inverseSin, 5=grouped grad.
 * Push u32: [totalGroupedPairs, T, H, headDim, modelDim].
 * One invocation writes one complete pair in its final Q/K/V segment, so no
 * zero-padded branch tensors or later grouped-gradient additions are needed.
 */
export function kernelQkvHeadMajorRopeBackwardCombined(wgSize = 256): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, wgSize, 1, 1);
  const qGrad = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const kGrad = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, true);
  const vGrad = declareStorageBuffer(b, p.tF32, p.tU32, 0, 2, true);
  const cos = declareStorageBuffer(b, p.tF32, p.tU32, 0, 3, true);
  const inverseSin = declareStorageBuffer(b, p.tF32, p.tU32, 0, 4, true);
  const out = declareStorageBuffer(b, p.tF32, p.tU32, 0, 5, false, true);
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
  const half = b.id(); b.emit(Op.UDiv, [p.tU32, half, headDim, p.const2u]);
  const groupedWidth = b.id(); b.emit(Op.IMul, [p.tU32, groupedWidth, modelDim, three]);
  const groupedPairWidth = b.id(); b.emit(Op.UDiv, [p.tU32, groupedPairWidth, groupedWidth, p.const2u]);
  const segmentPairWidth = b.id(); b.emit(Op.UDiv, [p.tU32, segmentPairWidth, modelDim, p.const2u]);
  const tokenRow = b.id(); b.emit(Op.UDiv, [p.tU32, tokenRow, gid, groupedPairWidth]);
  const groupedPairCol = b.id(); b.emit(Op.UMod, [p.tU32, groupedPairCol, gid, groupedPairWidth]);
  const segment = b.id(); b.emit(Op.UDiv, [p.tU32, segment, groupedPairCol, segmentPairWidth]);
  const localPair = b.id(); b.emit(Op.UMod, [p.tU32, localPair, groupedPairCol, segmentPairWidth]);
  const head = b.id(); b.emit(Op.UDiv, [p.tU32, head, localPair, half]);
  const pair = b.id(); b.emit(Op.UMod, [p.tU32, pair, localPair, half]);

  const tokenRowBase = b.id(); b.emit(Op.IMul, [p.tU32, tokenRowBase, tokenRow, groupedWidth]);
  const segmentBase = b.id(); b.emit(Op.IMul, [p.tU32, segmentBase, segment, modelDim]);
  const headOffset = b.id(); b.emit(Op.IMul, [p.tU32, headOffset, head, headDim]);
  const rowSegmentBase = b.id(); b.emit(Op.IAdd, [p.tU32, rowSegmentBase, tokenRowBase, segmentBase]);
  const outputHeadBase = b.id(); b.emit(Op.IAdd, [p.tU32, outputHeadBase, rowSegmentBase, headOffset]);
  const outputA = b.id(); b.emit(Op.IAdd, [p.tU32, outputA, outputHeadBase, pair]);
  const outputB = b.id(); b.emit(Op.IAdd, [p.tU32, outputB, outputA, half]);

  const batch = b.id(); b.emit(Op.UDiv, [p.tU32, batch, tokenRow, sequence]);
  const token = b.id(); b.emit(Op.UMod, [p.tU32, token, tokenRow, sequence]);
  const batchTimesH = b.id(); b.emit(Op.IMul, [p.tU32, batchTimesH, batch, heads]);
  const batchHead = b.id(); b.emit(Op.IAdd, [p.tU32, batchHead, batchTimesH, head]);
  const bhTimesT = b.id(); b.emit(Op.IMul, [p.tU32, bhTimesT, batchHead, sequence]);
  const headMajorRow = b.id(); b.emit(Op.IAdd, [p.tU32, headMajorRow, bhTimesT, token]);
  const gradBase = b.id(); b.emit(Op.IMul, [p.tU32, gradBase, headMajorRow, headDim]);
  const gradAIndex = b.id(); b.emit(Op.IAdd, [p.tU32, gradAIndex, gradBase, pair]);
  const gradBIndex = b.id(); b.emit(Op.IAdd, [p.tU32, gradBIndex, gradAIndex, half]);
  const tableBase = b.id(); b.emit(Op.IMul, [p.tU32, tableBase, token, half]);
  const tableIndex = b.id(); b.emit(Op.IAdd, [p.tU32, tableIndex, tableBase, pair]);

  const isQ = b.id(); b.emit(Op.IEqual, [p.tBool, isQ, segment, p.const0u]);
  const qBody = b.id();
  const notQBody = b.id();
  const segmentEnd = b.id();
  b.emit(Op.SelectionMerge, [segmentEnd, 0]);
  b.emit(Op.BranchConditional, [isQ, qBody, notQBody]);

  b.emit(Op.Label, [qBody]);
  const qA = loadF32(b, p.tF32, p.const0u, qGrad, gradAIndex);
  const qB = loadF32(b, p.tF32, p.const0u, qGrad, gradBIndex);
  const qC = loadF32(b, p.tF32, p.const0u, cos, tableIndex);
  const qS = loadF32(b, p.tF32, p.const0u, inverseSin, tableIndex);
  const qAc = b.id(); b.emit(Op.FMul, [p.tF32, qAc, qA, qC]);
  const qBs = b.id(); b.emit(Op.FMul, [p.tF32, qBs, qB, qS]);
  const qOutA = b.id(); b.emit(Op.FSub, [p.tF32, qOutA, qAc, qBs]);
  const qBc = b.id(); b.emit(Op.FMul, [p.tF32, qBc, qB, qC]);
  const qAs = b.id(); b.emit(Op.FMul, [p.tF32, qAs, qA, qS]);
  const qOutB = b.id(); b.emit(Op.FAdd, [p.tF32, qOutB, qBc, qAs]);
  storeF32(b, p.const0u, out, outputA, qOutA);
  storeF32(b, p.const0u, out, outputB, qOutB);
  b.emit(Op.Branch, [segmentEnd]);

  b.emit(Op.Label, [notQBody]);
  const isK = b.id(); b.emit(Op.IEqual, [p.tBool, isK, segment, p.const1u]);
  const kBody = b.id();
  const vBody = b.id();
  const notQEnd = b.id();
  b.emit(Op.SelectionMerge, [notQEnd, 0]);
  b.emit(Op.BranchConditional, [isK, kBody, vBody]);

  b.emit(Op.Label, [kBody]);
  const kA = loadF32(b, p.tF32, p.const0u, kGrad, gradAIndex);
  const kB = loadF32(b, p.tF32, p.const0u, kGrad, gradBIndex);
  const kC = loadF32(b, p.tF32, p.const0u, cos, tableIndex);
  const kS = loadF32(b, p.tF32, p.const0u, inverseSin, tableIndex);
  const kAc = b.id(); b.emit(Op.FMul, [p.tF32, kAc, kA, kC]);
  const kBs = b.id(); b.emit(Op.FMul, [p.tF32, kBs, kB, kS]);
  const kOutA = b.id(); b.emit(Op.FSub, [p.tF32, kOutA, kAc, kBs]);
  const kBc = b.id(); b.emit(Op.FMul, [p.tF32, kBc, kB, kC]);
  const kAs = b.id(); b.emit(Op.FMul, [p.tF32, kAs, kA, kS]);
  const kOutB = b.id(); b.emit(Op.FAdd, [p.tF32, kOutB, kBc, kAs]);
  storeF32(b, p.const0u, out, outputA, kOutA);
  storeF32(b, p.const0u, out, outputB, kOutB);
  b.emit(Op.Branch, [notQEnd]);

  b.emit(Op.Label, [vBody]);
  const vA = loadF32(b, p.tF32, p.const0u, vGrad, gradAIndex);
  const vB = loadF32(b, p.tF32, p.const0u, vGrad, gradBIndex);
  storeF32(b, p.const0u, out, outputA, vA);
  storeF32(b, p.const0u, out, outputB, vB);
  b.emit(Op.Branch, [notQEnd]);

  b.emit(Op.Label, [notQEnd]);
  b.emit(Op.Branch, [segmentEnd]);
  b.emit(Op.Label, [segmentEnd]);
  b.emit(Op.Branch, [end]);
  b.emit(Op.Label, [end]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);
  return b.build();
}
