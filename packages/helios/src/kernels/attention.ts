/**
 * kernels/attention.ts — Flash Attention forward GPU kernel.
 *
 * FlashAttention-2 with online softmax, causal masking, and optional soft-capping.
 * Each workgroup processes one query block (Br rows). Each thread handles one
 * query row within the block. Key/value blocks are cooperatively loaded into
 * shared memory, then attention scores are computed using the online softmax
 * algorithm (single pass per key block, no separate max-finding pass).
 *
 * Compile-time parameters: Br (query block size), Bc (key block size), D (head dim).
 * The inner loops over D and Bc are fully unrolled at code generation time.
 * Only the outer loop over key blocks is a runtime SPIR-V loop.
 *
 * Bindings:
 *   0: Q [B*H, T, D] (readonly)
 *   1: K [B*H, T, D] (readonly)
 *   2: V [B*H, T, D] (readonly)
 *   3: O [B*H, T, D] (write)
 *   4: LSE [B*H, T]  (write, log-sum-exp for backward)
 *
 * Push constants (16 bytes = 4 x f32):
 *   member 0: T (sequence length, as float)
 *   member 1: scale (1/sqrt(D))
 *   member 2: softCapValue (tanh capping, 0 = disabled)
 *   member 3: _pad
 *
 * Dispatch: (ceil(T/Br), B*H, 1)
 * Workgroup: (Br, 1, 1)
 */

import {
  SpirVBuilder, Op, ExecutionModel, ExecutionMode, StorageClass, Decoration,
  BuiltIn, FunctionControl, Scope, MemorySemantics, GLSLstd450,
  preamble, declareStorageBuffer, declareStorageBufferVec4, declareParamsPushConstant,
} from "./helpers.js";

// ── Kernel: Flash Attention Forward ──────────────────────────────────────────

export function kernelFlashAttentionForward(Br: number, Bc: number, D: number): Uint32Array {
  const D4 = D >>> 2; // D must be divisible by 4
  const b = new SpirVBuilder();
  const p = preamble(b, Br, 1, 1);

  // ── Vec4 type ────────────────────────────────────────────────────────────
  const tVec4F32 = b.id();
  b.typeVector(tVec4F32, p.tF32, 4);

  // Storage buffers: Q, K, V (readonly vec4), O (write vec4), LSE (write f32)
  const bufQ   = declareStorageBufferVec4(b, tVec4F32, 0, 0, true);
  const bufK   = declareStorageBufferVec4(b, tVec4F32, 0, 1, true);
  const bufV   = declareStorageBufferVec4(b, tVec4F32, 0, 2, true);
  const bufO   = declareStorageBufferVec4(b, tVec4F32, 0, 3, false, true);
  const bufLSE = declareStorageBuffer(b, p.tF32, p.tU32, 0, 4, false, true);

  // Push constants: { T, scale, softCapValue, _pad }
  const pc = declareParamsPushConstant(b, p.tF32, 4);

  // ── Constants ──────────────────────────────────────────────────────────────
  const constBr = b.id(); b.constant(p.tU32, constBr, Br);
  const constBc = b.id(); b.constant(p.tU32, constBc, Bc);
  const constD4 = b.id(); b.constant(p.tU32, constD4, D4);

  const constNegInf = b.id(); b.constant(p.tF32, constNegInf, 0xFF800000);
  const const1f = b.id(); b.constantF32(p.tF32, const1f, 1.0);
  const vec4Zero = b.id(); b.constantNull(tVec4F32, vec4Zero);

  // Pre-generate d4 index constants (0..D4-1)
  const constD4Idx: number[] = [];
  for (let d4 = 0; d4 < D4; d4++) {
    const cd4 = b.id();
    b.constant(p.tU32, cd4, d4);
    constD4Idx.push(cd4);
  }

  // ── Shared memory: sK[Bc * D4] and sV[Bc * D4] of vec4 ─────────────────
  const constSharedSize = b.id();
  b.constant(p.tU32, constSharedSize, Bc * D4);
  const tArrayShared = b.id();
  b.typeArray(tArrayShared, tVec4F32, constSharedSize);
  const tPtrSharedArr = b.id();
  b.typePointer(tPtrSharedArr, StorageClass.Workgroup, tArrayShared);
  const tPtrSharedVec4 = b.id();
  b.typePointer(tPtrSharedVec4, StorageClass.Workgroup, tVec4F32);
  const sK = b.id();
  b.variable(tPtrSharedArr, sK, StorageClass.Workgroup);
  const sV = b.id();
  b.variable(tPtrSharedArr, sV, StorageClass.Workgroup);

  // ── Function-scope array types: regQ[D4] and regO[D4] of vec4 ──────────
  const constD4Arr = b.id();
  b.constant(p.tU32, constD4Arr, D4);
  const tArrayD4 = b.id();
  b.typeArray(tArrayD4, tVec4F32, constD4Arr);
  const tPtrFnArr = b.id();
  b.typePointer(tPtrFnArr, StorageClass.Function, tArrayD4);
  const tPtrFnVec4 = b.id();
  b.typePointer(tPtrFnVec4, StorageClass.Function, tVec4F32);
  const tPtrFnF32 = b.id();
  b.typePointer(tPtrFnF32, StorageClass.Function, p.tF32);
  const tPtrFnU32 = b.id();
  b.typePointer(tPtrFnU32, StorageClass.Function, p.tU32);

  // ── Built-in variables ────────────────────────────────────────────────────
  const tPtrInputVec3 = b.id();
  b.typePointer(tPtrInputVec3, StorageClass.Input, p.tVec3U32);
  const vWorkgroupId = b.id();
  b.variable(tPtrInputVec3, vWorkgroupId, StorageClass.Input);
  b.addDecorate(vWorkgroupId, Decoration.BuiltIn, BuiltIn.WorkgroupId);
  const vLocalId = b.id();
  b.variable(tPtrInputVec3, vLocalId, StorageClass.Input);
  b.addDecorate(vLocalId, Decoration.BuiltIn, BuiltIn.LocalInvocationId);

  // ── Barrier constants ─────────────────────────────────────────────────────
  const scopeWg = b.id();
  b.constant(p.tU32, scopeWg, Scope.Workgroup);
  const semAcqRelWg = b.id();
  b.constant(p.tU32, semAcqRelWg, MemorySemantics.AcquireRelease | MemorySemantics.WorkgroupMemory);

  // ── Entry point ───────────────────────────────────────────────────────────
  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId, vWorkgroupId, vLocalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, Br, 1, 1);

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  const labelEntry = b.id();
  b.emit(Op.Label, [labelEntry]);

  // ── Variable declarations (must be in entry block) ──────────────────────
  const regQ = b.id();
  b.emit(Op.Variable, [tPtrFnArr, regQ, StorageClass.Function]);
  const regO = b.id();
  b.emit(Op.Variable, [tPtrFnArr, regO, StorageClass.Function]);
  const varM = b.id();
  b.emit(Op.Variable, [tPtrFnF32, varM, StorageClass.Function]);
  const varL = b.id();
  b.emit(Op.Variable, [tPtrFnF32, varL, StorageClass.Function]);
  const varKBlockIdx = b.id();
  b.emit(Op.Variable, [tPtrFnU32, varKBlockIdx, StorageClass.Function]);
  const varJ = b.id();
  b.emit(Op.Variable, [tPtrFnU32, varJ, StorageClass.Function]);

  // ── Load thread/workgroup IDs ─────────────────────────────────────────────
  const lidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, lidVec, vLocalId]);
  const threadIdx = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, threadIdx, lidVec, 0]);

  const wgIdVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, wgIdVec, vWorkgroupId]);
  const qBlockIdx = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, qBlockIdx, wgIdVec, 0]);
  const bhIdx = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, bhIdx, wgIdVec, 1]);

  // qRow = qBlockIdx * Br + threadIdx
  const qBlockOff = b.id();
  b.emit(Op.IMul, [p.tU32, qBlockOff, qBlockIdx, constBr]);
  const qRow = b.id();
  b.emit(Op.IAdd, [p.tU32, qRow, qBlockOff, threadIdx]);

  // ── Load push constants ─────────────────────────────────────────────────
  const ptrTpc = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrTpc, pc.varId, p.const0u]);
  const TF = b.id();
  b.emit(Op.Load, [p.tF32, TF, ptrTpc]);
  const T = b.id();
  b.emit(Op.ConvertFToU, [p.tU32, T, TF]);

  const ptrScale = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrScale, pc.varId, p.const1u]);
  const scale = b.id();
  b.emit(Op.Load, [p.tF32, scale, ptrScale]);

  const ptrSoftCap = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrSoftCap, pc.varId, p.const2u]);
  const softCapValue = b.id();
  b.emit(Op.Load, [p.tF32, softCapValue, ptrSoftCap]);

  const softCapEnabled = b.id();
  b.emit(Op.FOrdGreaterThan, [p.tBool, softCapEnabled, softCapValue, p.const0f]);
  const invSoftCap = b.id();
  b.emit(Op.FDiv, [p.tF32, invSoftCap, const1f, softCapValue]);

  // ── Compute vec4 base offset: baseOff4 = bhIdx * T * D4 ────────────────
  const TD4 = b.id();
  b.emit(Op.IMul, [p.tU32, TD4, T, constD4]);
  const baseOff4 = b.id();
  b.emit(Op.IMul, [p.tU32, baseOff4, bhIdx, TD4]);

  // ── Bounds check ────────────────────────────────────────────────────────
  const qRowOob = b.id();
  b.emit(Op.UGreaterThanEqual, [p.tBool, qRowOob, qRow, T]);
  const labelMain = b.id();
  const labelEnd = b.id();
  b.emit(Op.SelectionMerge, [labelEnd, 0]);
  b.emit(Op.BranchConditional, [qRowOob, labelEnd, labelMain]);

  b.emit(Op.Label, [labelMain]);

  // ── Load Q[qRow] into regQ as vec4, pre-scaled by `scale` ──────────────
  const qRowD4 = b.id();
  b.emit(Op.IMul, [p.tU32, qRowD4, qRow, constD4]);
  const qBase4 = b.id();
  b.emit(Op.IAdd, [p.tU32, qBase4, baseOff4, qRowD4]);

  for (let d4 = 0; d4 < D4; d4++) {
    const qIdx = b.id();
    b.emit(Op.IAdd, [p.tU32, qIdx, qBase4, constD4Idx[d4]]);
    const ptrQElem = b.id();
    b.emit(Op.AccessChain, [bufQ.tPtrVec4, ptrQElem, bufQ.varId, p.const0u, qIdx]);
    const qVec = b.id();
    b.emit(Op.Load, [tVec4F32, qVec, ptrQElem]);
    const qScaled = b.id();
    b.emit(Op.VectorTimesScalar, [tVec4F32, qScaled, qVec, scale]);
    const ptrRegQ = b.id();
    b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegQ, regQ, constD4Idx[d4]]);
    b.emit(Op.Store, [ptrRegQ, qScaled]);
  }

  // ── Initialize regO to vec4(0) ──────────────────────────────────────────
  for (let d4 = 0; d4 < D4; d4++) {
    const ptrRegO = b.id();
    b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegO, regO, constD4Idx[d4]]);
    b.emit(Op.Store, [ptrRegO, vec4Zero]);
  }

  // m = -inf, l = 0.0
  b.emit(Op.Store, [varM, constNegInf]);
  b.emit(Op.Store, [varL, p.const0f]);

  // effectiveKBlocks = (qBlockIdx + 1) * kBlocksPerQBlock (causal)
  // When Br > Bc, each Q block spans multiple K blocks
  const kBlocksPerQBlock = Br / Bc; // compile-time integer (Br must be multiple of Bc)
  const qPlus1 = b.id();
  b.emit(Op.IAdd, [p.tU32, qPlus1, qBlockIdx, p.const1u]);
  let effectiveKBlocks: number;
  if (kBlocksPerQBlock === 1) {
    effectiveKBlocks = qPlus1;
  } else {
    const constKBPQB = b.id(); b.constant(p.tU32, constKBPQB, kBlocksPerQBlock);
    effectiveKBlocks = b.id();
    b.emit(Op.IMul, [p.tU32, effectiveKBlocks, qPlus1, constKBPQB]);
  }

  // ── Outer loop: kBlockIdx = 0..effectiveKBlocks ─────────────────────────
  b.emit(Op.Store, [varKBlockIdx, p.const0u]);

  const labelLoopHead = b.id();
  const labelLoopBody = b.id();
  const labelLoopMerge = b.id();
  const labelLoopCont = b.id();

  b.emit(Op.Branch, [labelLoopHead]);
  b.emit(Op.Label, [labelLoopHead]);
  const kBlockIdx = b.id();
  b.emit(Op.Load, [p.tU32, kBlockIdx, varKBlockIdx]);
  const loopCmp = b.id();
  b.emit(Op.ULessThan, [p.tBool, loopCmp, kBlockIdx, effectiveKBlocks]);
  b.emit(Op.LoopMerge, [labelLoopMerge, labelLoopCont, 0]);
  b.emit(Op.BranchConditional, [loopCmp, labelLoopBody, labelLoopMerge]);

  b.emit(Op.Label, [labelLoopBody]);

  // ── Cooperative load: K/V into shared memory as vec4 ────────────────────
  // Distribute Bc*D4 elements across all Br threads for optimal coalescing.
  // Each pass, 32 threads load 32 consecutive vec4s from K and V.
  // linearIdx = row * D4 + d4, mapped via: row = linearIdx >> log2(D4), d4 = linearIdx & (D4-1)
  const kBlockBase = b.id();
  b.emit(Op.IMul, [p.tU32, kBlockBase, kBlockIdx, constBc]);

  // kBlockGlobalBase4 = baseOff4 + kBlockBase * D4 (starting vec4 index for this K block)
  const kBlockBaseTD4 = b.id();
  b.emit(Op.IMul, [p.tU32, kBlockBaseTD4, kBlockBase, constD4]);
  const kBlockGlobal4 = b.id();
  b.emit(Op.IAdd, [p.tU32, kBlockGlobal4, baseOff4, kBlockBaseTD4]);

  const totalLoadElems = Bc * D4; // compile-time
  const elemsPerThread = Math.ceil(totalLoadElems / Br); // compile-time
  const log2D4 = Math.log2(D4); // compile-time, integer since D4 is power of 2
  const constLog2D4 = b.id(); b.constant(p.tU32, constLog2D4, log2D4);

  for (let pass = 0; pass < elemsPerThread; pass++) {
    // linearIdx = threadIdx + pass * Br
    let linearIdx: number;
    if (pass === 0) {
      linearIdx = threadIdx;
    } else {
      const constPassOff = b.id(); b.constant(p.tU32, constPassOff, pass * Br);
      linearIdx = b.id();
      b.emit(Op.IAdd, [p.tU32, linearIdx, threadIdx, constPassOff]);
    }

    // Bounds check: only needed if this pass might exceed totalLoadElems
    const needCheck = (pass + 1) * Br > totalLoadElems;
    let labelPassBody: number | undefined;
    let labelPassEnd: number | undefined;
    if (needCheck) {
      const constTotalElems = b.id(); b.constant(p.tU32, constTotalElems, totalLoadElems);
      const inRange = b.id();
      b.emit(Op.ULessThan, [p.tBool, inRange, linearIdx, constTotalElems]);
      labelPassBody = b.id();
      labelPassEnd = b.id();
      b.emit(Op.SelectionMerge, [labelPassEnd, 0]);
      b.emit(Op.BranchConditional, [inRange, labelPassBody, labelPassEnd]);
      b.emit(Op.Label, [labelPassBody]);
    }

    // row = linearIdx >> log2(D4)
    const row = b.id();
    b.emit(Op.ShiftRightLogical, [p.tU32, row, linearIdx, constLog2D4]);

    // kRow = kBlockBase + row
    const kRow = b.id();
    b.emit(Op.IAdd, [p.tU32, kRow, kBlockBase, row]);
    const kRowInBounds = b.id();
    b.emit(Op.ULessThan, [p.tBool, kRowInBounds, kRow, T]);
    const inBoundsF = b.id();
    b.emit(Op.Select, [p.tF32, inBoundsF, kRowInBounds, const1f, p.const0f]);

    // gIdx = kBlockGlobal4 + linearIdx (global vec4 address for K[kRow, d4])
    const gIdx = b.id();
    b.emit(Op.IAdd, [p.tU32, gIdx, kBlockGlobal4, linearIdx]);

    // Load K vec4 and mask OOB to zero
    const ptrKElem = b.id();
    b.emit(Op.AccessChain, [bufK.tPtrVec4, ptrKElem, bufK.varId, p.const0u, gIdx]);
    const kRaw = b.id();
    b.emit(Op.Load, [tVec4F32, kRaw, ptrKElem]);
    const kVal = b.id();
    b.emit(Op.VectorTimesScalar, [tVec4F32, kVal, kRaw, inBoundsF]);

    // Store to sK[linearIdx]
    const ptrSK = b.id();
    b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSK, sK, linearIdx]);
    b.emit(Op.Store, [ptrSK, kVal]);

    // Load V vec4 and mask OOB to zero
    const ptrVElem = b.id();
    b.emit(Op.AccessChain, [bufV.tPtrVec4, ptrVElem, bufV.varId, p.const0u, gIdx]);
    const vRaw = b.id();
    b.emit(Op.Load, [tVec4F32, vRaw, ptrVElem]);
    const vVal = b.id();
    b.emit(Op.VectorTimesScalar, [tVec4F32, vVal, vRaw, inBoundsF]);

    // Store to sV[linearIdx]
    const ptrSV = b.id();
    b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSV, sV, linearIdx]);
    b.emit(Op.Store, [ptrSV, vVal]);

    if (needCheck) {
      b.emit(Op.Branch, [labelPassEnd!]);
      b.emit(Op.Label, [labelPassEnd!]);
    }
  }

  // Barrier — all threads have loaded their K/V rows
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  // ── Inner loop: j = 0..Bc (runtime SPIR-V loop, not compile-time unrolled) ─
  // This reduces shader binary from ~33K to ~300 SPIR-V instructions.
  // The D dimension is still compile-time unrolled via vec4 (D/4 iterations).
  b.emit(Op.Store, [varJ, p.const0u]);

  const labelJHead = b.id();
  const labelJBody = b.id();
  const labelJMerge = b.id();
  const labelJCont = b.id();

  b.emit(Op.Branch, [labelJHead]);
  b.emit(Op.Label, [labelJHead]);
  const j = b.id();
  b.emit(Op.Load, [p.tU32, j, varJ]);
  const jCmp = b.id();
  b.emit(Op.ULessThan, [p.tBool, jCmp, j, constBc]);
  b.emit(Op.LoopMerge, [labelJMerge, labelJCont, 0]);
  b.emit(Op.BranchConditional, [jCmp, labelJBody, labelJMerge]);

  b.emit(Op.Label, [labelJBody]);

  // kPos = kBlockBase + j
  const kPos = b.id();
  b.emit(Op.IAdd, [p.tU32, kPos, kBlockBase, j]);

  // jD4 = j * D4 (shared memory offset for key j)
  const jD4 = b.id();
  b.emit(Op.IMul, [p.tU32, jD4, j, constD4]);

  // ── Vec4 dot product: regQ · sK[j] using OpDot ─────────────────────────
  // Q is pre-scaled, so result is already scaled

  // First vec4 pair (d4 = 0) — use jD4 directly as sK index
  const ptrRegQ0 = b.id();
  b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegQ0, regQ, constD4Idx[0]]);
  const qVec0 = b.id();
  b.emit(Op.Load, [tVec4F32, qVec0, ptrRegQ0]);
  const ptrSK0 = b.id();
  b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSK0, sK, jD4]);
  const kVec0 = b.id();
  b.emit(Op.Load, [tVec4F32, kVec0, ptrSK0]);
  let dotAcc = b.id();
  b.emit(Op.Dot, [p.tF32, dotAcc, qVec0, kVec0]);

  // Remaining vec4 pairs (d4 = 1..D4-1)
  for (let d4 = 1; d4 < D4; d4++) {
    const ptrRegQd4 = b.id();
    b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegQd4, regQ, constD4Idx[d4]]);
    const qVecD4 = b.id();
    b.emit(Op.Load, [tVec4F32, qVecD4, ptrRegQd4]);

    const sKIdxD4 = b.id();
    b.emit(Op.IAdd, [p.tU32, sKIdxD4, jD4, constD4Idx[d4]]);
    const ptrSKd4 = b.id();
    b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSKd4, sK, sKIdxD4]);
    const kVecD4 = b.id();
    b.emit(Op.Load, [tVec4F32, kVecD4, ptrSKd4]);

    const partial = b.id();
    b.emit(Op.Dot, [p.tF32, partial, qVecD4, kVecD4]);
    const newDotAcc = b.id();
    b.emit(Op.FAdd, [p.tF32, newDotAcc, dotAcc, partial]);
    dotAcc = newDotAcc;
  }

  // ── Soft capping: tanh(dot / cap) * cap if enabled ────────────────────
  const dotDivCap = b.id();
  b.emit(Op.FMul, [p.tF32, dotDivCap, dotAcc, invSoftCap]);
  const tanhVal = b.id();
  b.emit(Op.ExtInst, [p.tF32, tanhVal, p.glslStd, GLSLstd450.Tanh, dotDivCap]);
  const dotCapped = b.id();
  b.emit(Op.FMul, [p.tF32, dotCapped, tanhVal, softCapValue]);
  const dotAfterCap = b.id();
  b.emit(Op.Select, [p.tF32, dotAfterCap, softCapEnabled, dotCapped, dotAcc]);

  // ── Causal mask + out-of-bounds ─────────────────────────────────────────
  const oob = b.id();
  b.emit(Op.UGreaterThanEqual, [p.tBool, oob, kPos, T]);
  const causal = b.id();
  b.emit(Op.ULessThan, [p.tBool, causal, qRow, kPos]);
  const masked = b.id();
  b.emit(Op.LogicalOr, [p.tBool, masked, oob, causal]);
  const dot = b.id();
  b.emit(Op.Select, [p.tF32, dot, masked, constNegInf, dotAfterCap]);

  // ── Online softmax update ───────────────────────────────────────────────
  const mOld = b.id();
  b.emit(Op.Load, [p.tF32, mOld, varM]);
  const mNew = b.id();
  b.emit(Op.ExtInst, [p.tF32, mNew, p.glslStd, GLSLstd450.FMax, mOld, dot]);

  const mDiff = b.id();
  b.emit(Op.FSub, [p.tF32, mDiff, mOld, mNew]);
  const alpha = b.id();
  b.emit(Op.ExtInst, [p.tF32, alpha, p.glslStd, GLSLstd450.Exp, mDiff]);

  const dotDiff = b.id();
  b.emit(Op.FSub, [p.tF32, dotDiff, dot, mNew]);
  const pj = b.id();
  b.emit(Op.ExtInst, [p.tF32, pj, p.glslStd, GLSLstd450.Exp, dotDiff]);

  const lOld = b.id();
  b.emit(Op.Load, [p.tF32, lOld, varL]);
  const lScaled = b.id();
  b.emit(Op.FMul, [p.tF32, lScaled, lOld, alpha]);
  const lNew = b.id();
  b.emit(Op.FAdd, [p.tF32, lNew, lScaled, pj]);
  b.emit(Op.Store, [varL, lNew]);
  b.emit(Op.Store, [varM, mNew]);

  // ── V accumulation: regO[d4] = regO[d4] * alpha + pj * sV[j*D4+d4] ────
  for (let d4 = 0; d4 < D4; d4++) {
    const ptrO = b.id();
    b.emit(Op.AccessChain, [tPtrFnVec4, ptrO, regO, constD4Idx[d4]]);
    const oOld = b.id();
    b.emit(Op.Load, [tVec4F32, oOld, ptrO]);
    const oScaled = b.id();
    b.emit(Op.VectorTimesScalar, [tVec4F32, oScaled, oOld, alpha]);

    const sVIdx = b.id();
    b.emit(Op.IAdd, [p.tU32, sVIdx, jD4, constD4Idx[d4]]);
    const ptrSV = b.id();
    b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSV, sV, sVIdx]);
    const vv = b.id();
    b.emit(Op.Load, [tVec4F32, vv, ptrSV]);
    const pvv = b.id();
    b.emit(Op.VectorTimesScalar, [tVec4F32, pvv, vv, pj]);
    const oNew = b.id();
    b.emit(Op.FAdd, [tVec4F32, oNew, oScaled, pvv]);
    b.emit(Op.Store, [ptrO, oNew]);
  }

  // ── j loop continuation ─────────────────────────────────────────────────
  b.emit(Op.Branch, [labelJCont]);
  b.emit(Op.Label, [labelJCont]);
  const nextJ = b.id();
  b.emit(Op.Load, [p.tU32, nextJ, varJ]);
  const incJ = b.id();
  b.emit(Op.IAdd, [p.tU32, incJ, nextJ, p.const1u]);
  b.emit(Op.Store, [varJ, incJ]);
  b.emit(Op.Branch, [labelJHead]);

  b.emit(Op.Label, [labelJMerge]);

  // Barrier before next K block
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  // ── kBlock loop continuation ────────────────────────────────────────────
  b.emit(Op.Branch, [labelLoopCont]);
  b.emit(Op.Label, [labelLoopCont]);
  const nextKBlock = b.id();
  b.emit(Op.Load, [p.tU32, nextKBlock, varKBlockIdx]);
  const incKBlock = b.id();
  b.emit(Op.IAdd, [p.tU32, incKBlock, nextKBlock, p.const1u]);
  b.emit(Op.Store, [varKBlockIdx, incKBlock]);
  b.emit(Op.Branch, [labelLoopHead]);

  b.emit(Op.Label, [labelLoopMerge]);

  // ── Normalize output and write as vec4 ──────────────────────────────────
  const finalL = b.id();
  b.emit(Op.Load, [p.tF32, finalL, varL]);
  const invL = b.id();
  b.emit(Op.FDiv, [p.tF32, invL, const1f, finalL]);

  const oBase4 = b.id();
  b.emit(Op.IAdd, [p.tU32, oBase4, baseOff4, qRowD4]);

  for (let d4 = 0; d4 < D4; d4++) {
    const ptrRegOd4 = b.id();
    b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegOd4, regO, constD4Idx[d4]]);
    const regOVec = b.id();
    b.emit(Op.Load, [tVec4F32, regOVec, ptrRegOd4]);
    const oNorm = b.id();
    b.emit(Op.VectorTimesScalar, [tVec4F32, oNorm, regOVec, invL]);

    const oIdx = b.id();
    b.emit(Op.IAdd, [p.tU32, oIdx, oBase4, constD4Idx[d4]]);
    const ptrOut = b.id();
    b.emit(Op.AccessChain, [bufO.tPtrVec4, ptrOut, bufO.varId, p.const0u, oIdx]);
    b.emit(Op.Store, [ptrOut, oNorm]);
  }

  // ── Store LSE: LSE[bhIdx * T + qRow] = m + log(l) ──────────────────────
  const finalM = b.id();
  b.emit(Op.Load, [p.tF32, finalM, varM]);
  const logL = b.id();
  b.emit(Op.ExtInst, [p.tF32, logL, p.glslStd, GLSLstd450.Log, finalL]);
  const lseVal = b.id();
  b.emit(Op.FAdd, [p.tF32, lseVal, finalM, logL]);

  const bhT = b.id();
  b.emit(Op.IMul, [p.tU32, bhT, bhIdx, T]);
  const lseIdx = b.id();
  b.emit(Op.IAdd, [p.tU32, lseIdx, bhT, qRow]);
  const ptrLSE = b.id();
  b.emit(Op.AccessChain, [bufLSE.tPtrF32, ptrLSE, bufLSE.varId, p.const0u, lseIdx]);
  b.emit(Op.Store, [ptrLSE, lseVal]);

  b.emit(Op.Branch, [labelEnd]);

  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}


// ── Kernel: Flash Attention Forward V2 (compile-time j-unroll + batched softmax) ─
//
// Same algorithm as V1 but with the inner j-loop fully unrolled at code-gen time
// in batches of BJ. Each batch computes BJ dot products independently (ILP),
// then performs a single softmax update (1 alpha-rescaling per BJ instead of per-j).
// This reduces O rescalings by BJ× and allows the SPIR-V→SASS compiler to overlap
// shared memory loads with computation across j values within each batch.

export function kernelFlashAttentionForwardV2(
  Br: number, Bc: number, D: number, BJ: number = 4
): Uint32Array {
  if (Bc % BJ !== 0) throw new Error(`Bc=${Bc} must be divisible by BJ=${BJ}`);
  const D4 = D >>> 2;
  const b = new SpirVBuilder();
  const p = preamble(b, Br, 1, 1);

  // ── Vec4 type ────────────────────────────────────────────────────────────
  const tVec4F32 = b.id();
  b.typeVector(tVec4F32, p.tF32, 4);

  // Storage buffers
  const bufQ   = declareStorageBufferVec4(b, tVec4F32, 0, 0, true);
  const bufK   = declareStorageBufferVec4(b, tVec4F32, 0, 1, true);
  const bufV   = declareStorageBufferVec4(b, tVec4F32, 0, 2, true);
  const bufO   = declareStorageBufferVec4(b, tVec4F32, 0, 3, false, true);
  const bufLSE = declareStorageBuffer(b, p.tF32, p.tU32, 0, 4, false, true);

  // Push constants: { T, scale, softCapValue, _pad }
  const pc = declareParamsPushConstant(b, p.tF32, 4);

  // ── Constants ──────────────────────────────────────────────────────────────
  const constBr = b.id(); b.constant(p.tU32, constBr, Br);
  const constBc = b.id(); b.constant(p.tU32, constBc, Bc);
  const constD4 = b.id(); b.constant(p.tU32, constD4, D4);

  const constNegInf = b.id(); b.constant(p.tF32, constNegInf, 0xFF800000);
  const const1f = b.id(); b.constantF32(p.tF32, const1f, 1.0);
  const vec4Zero = b.id(); b.constantNull(tVec4F32, vec4Zero);

  // Pre-generate d4 index constants (0..D4-1)
  const constD4Idx: number[] = [];
  for (let d4 = 0; d4 < D4; d4++) {
    constD4Idx.push(b.id());
    b.constant(p.tU32, constD4Idx[d4], d4);
  }

  // Pre-generate j index constants (0..Bc-1) and j*D4 constants
  const constJIdx: number[] = [];
  const constJD4: number[] = [];
  for (let j = 0; j < Bc; j++) {
    constJIdx.push(b.id());
    b.constant(p.tU32, constJIdx[j], j);
    constJD4.push(b.id());
    b.constant(p.tU32, constJD4[j], j * D4);
  }

  // ── Shared memory: sK[Bc * D4] and sV[Bc * D4] of vec4 ─────────────────
  const constSharedSize = b.id();
  b.constant(p.tU32, constSharedSize, Bc * D4);
  const tArrayShared = b.id();
  b.typeArray(tArrayShared, tVec4F32, constSharedSize);
  const tPtrSharedArr = b.id();
  b.typePointer(tPtrSharedArr, StorageClass.Workgroup, tArrayShared);
  const tPtrSharedVec4 = b.id();
  b.typePointer(tPtrSharedVec4, StorageClass.Workgroup, tVec4F32);
  const sK = b.id();
  b.variable(tPtrSharedArr, sK, StorageClass.Workgroup);
  const sV = b.id();
  b.variable(tPtrSharedArr, sV, StorageClass.Workgroup);

  // ── Function-scope array types: regQ[D4] and regO[D4] of vec4 ──────────
  const constD4Arr = b.id();
  b.constant(p.tU32, constD4Arr, D4);
  const tArrayD4 = b.id();
  b.typeArray(tArrayD4, tVec4F32, constD4Arr);
  const tPtrFnArr = b.id();
  b.typePointer(tPtrFnArr, StorageClass.Function, tArrayD4);
  const tPtrFnVec4 = b.id();
  b.typePointer(tPtrFnVec4, StorageClass.Function, tVec4F32);
  const tPtrFnF32 = b.id();
  b.typePointer(tPtrFnF32, StorageClass.Function, p.tF32);
  const tPtrFnU32 = b.id();
  b.typePointer(tPtrFnU32, StorageClass.Function, p.tU32);

  // ── Built-in variables ────────────────────────────────────────────────────
  const tPtrInputVec3 = b.id();
  b.typePointer(tPtrInputVec3, StorageClass.Input, p.tVec3U32);
  const vWorkgroupId = b.id();
  b.variable(tPtrInputVec3, vWorkgroupId, StorageClass.Input);
  b.addDecorate(vWorkgroupId, Decoration.BuiltIn, BuiltIn.WorkgroupId);
  const vLocalId = b.id();
  b.variable(tPtrInputVec3, vLocalId, StorageClass.Input);
  b.addDecorate(vLocalId, Decoration.BuiltIn, BuiltIn.LocalInvocationId);

  // ── Barrier constants ─────────────────────────────────────────────────────
  const scopeWg = b.id();
  b.constant(p.tU32, scopeWg, Scope.Workgroup);
  const semAcqRelWg = b.id();
  b.constant(p.tU32, semAcqRelWg, MemorySemantics.AcquireRelease | MemorySemantics.WorkgroupMemory);

  // ── Entry point ───────────────────────────────────────────────────────────
  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId, vWorkgroupId, vLocalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, Br, 1, 1);

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  const labelEntry = b.id();
  b.emit(Op.Label, [labelEntry]);

  // ── Variable declarations ────────────────────────────────────────────────
  // regQ uses pure SSA variables (loaded once, never modified) — avoids
  // Function-scope array AccessChain+Load overhead in inner dot product loop.
  const regQVecs: number[] = []; // populated after Q load below
  const regO = b.id();
  b.emit(Op.Variable, [tPtrFnArr, regO, StorageClass.Function]);
  const varM = b.id();
  b.emit(Op.Variable, [tPtrFnF32, varM, StorageClass.Function]);
  const varL = b.id();
  b.emit(Op.Variable, [tPtrFnF32, varL, StorageClass.Function]);
  const varKBlockIdx = b.id();
  b.emit(Op.Variable, [tPtrFnU32, varKBlockIdx, StorageClass.Function]);

  // ── Load thread/workgroup IDs ─────────────────────────────────────────────
  const lidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, lidVec, vLocalId]);
  const threadIdx = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, threadIdx, lidVec, 0]);

  const wgIdVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, wgIdVec, vWorkgroupId]);
  const qBlockIdx = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, qBlockIdx, wgIdVec, 0]);
  const bhIdx = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, bhIdx, wgIdVec, 1]);

  // qRow = qBlockIdx * Br + threadIdx
  const qBlockOff = b.id();
  b.emit(Op.IMul, [p.tU32, qBlockOff, qBlockIdx, constBr]);
  const qRow = b.id();
  b.emit(Op.IAdd, [p.tU32, qRow, qBlockOff, threadIdx]);

  // ── Load push constants ─────────────────────────────────────────────────
  const ptrTpc = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrTpc, pc.varId, p.const0u]);
  const TF = b.id();
  b.emit(Op.Load, [p.tF32, TF, ptrTpc]);
  const T = b.id();
  b.emit(Op.ConvertFToU, [p.tU32, T, TF]);

  const ptrScale = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrScale, pc.varId, p.const1u]);
  const scale = b.id();
  b.emit(Op.Load, [p.tF32, scale, ptrScale]);

  const ptrSoftCap = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrSoftCap, pc.varId, p.const2u]);
  const softCapValue = b.id();
  b.emit(Op.Load, [p.tF32, softCapValue, ptrSoftCap]);

  const softCapEnabled = b.id();
  b.emit(Op.FOrdGreaterThan, [p.tBool, softCapEnabled, softCapValue, p.const0f]);
  const invSoftCap = b.id();
  b.emit(Op.FDiv, [p.tF32, invSoftCap, const1f, softCapValue]);

  // ── Compute vec4 base offset: baseOff4 = bhIdx * T * D4 ────────────────
  const TD4 = b.id();
  b.emit(Op.IMul, [p.tU32, TD4, T, constD4]);
  const baseOff4 = b.id();
  b.emit(Op.IMul, [p.tU32, baseOff4, bhIdx, TD4]);

  // ── Bounds check ────────────────────────────────────────────────────────
  const qRowOob = b.id();
  b.emit(Op.UGreaterThanEqual, [p.tBool, qRowOob, qRow, T]);
  const labelMain = b.id();
  const labelEnd = b.id();
  b.emit(Op.SelectionMerge, [labelEnd, 0]);
  b.emit(Op.BranchConditional, [qRowOob, labelEnd, labelMain]);

  b.emit(Op.Label, [labelMain]);

  // ── Load Q[qRow] into SSA registers, pre-scaled by `scale` ─────────────
  const qRowD4 = b.id();
  b.emit(Op.IMul, [p.tU32, qRowD4, qRow, constD4]);
  const qBase4 = b.id();
  b.emit(Op.IAdd, [p.tU32, qBase4, baseOff4, qRowD4]);

  for (let d4 = 0; d4 < D4; d4++) {
    const qIdx = b.id();
    b.emit(Op.IAdd, [p.tU32, qIdx, qBase4, constD4Idx[d4]]);
    const ptrQElem = b.id();
    b.emit(Op.AccessChain, [bufQ.tPtrVec4, ptrQElem, bufQ.varId, p.const0u, qIdx]);
    const qVec = b.id();
    b.emit(Op.Load, [tVec4F32, qVec, ptrQElem]);
    const qScaled = b.id();
    b.emit(Op.VectorTimesScalar, [tVec4F32, qScaled, qVec, scale]);
    regQVecs.push(qScaled); // pure SSA — stays in GPU registers
  }

  // ── Initialize regO to vec4(0) ──────────────────────────────────────────
  for (let d4 = 0; d4 < D4; d4++) {
    const ptrRegO = b.id();
    b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegO, regO, constD4Idx[d4]]);
    b.emit(Op.Store, [ptrRegO, vec4Zero]);
  }

  // m = -inf, l = 0.0
  b.emit(Op.Store, [varM, constNegInf]);
  b.emit(Op.Store, [varL, p.const0f]);

  // effectiveKBlocks = qBlockIdx + 1 (causal: Br == Bc)
  const effectiveKBlocks = b.id();
  b.emit(Op.IAdd, [p.tU32, effectiveKBlocks, qBlockIdx, p.const1u]);

  // ── Outer loop: kBlockIdx = 0..effectiveKBlocks ─────────────────────────
  b.emit(Op.Store, [varKBlockIdx, p.const0u]);

  const labelLoopHead = b.id();
  const labelLoopBody = b.id();
  const labelLoopMerge = b.id();
  const labelLoopCont = b.id();

  b.emit(Op.Branch, [labelLoopHead]);
  b.emit(Op.Label, [labelLoopHead]);
  const kBlockIdx = b.id();
  b.emit(Op.Load, [p.tU32, kBlockIdx, varKBlockIdx]);
  const loopCmp = b.id();
  b.emit(Op.ULessThan, [p.tBool, loopCmp, kBlockIdx, effectiveKBlocks]);
  b.emit(Op.LoopMerge, [labelLoopMerge, labelLoopCont, 0]);
  b.emit(Op.BranchConditional, [loopCmp, labelLoopBody, labelLoopMerge]);

  b.emit(Op.Label, [labelLoopBody]);

  // ── Cooperative load: K/V into shared memory (same as V1) ─────────────
  const kBlockBase = b.id();
  b.emit(Op.IMul, [p.tU32, kBlockBase, kBlockIdx, constBc]);
  const kRow = b.id();
  b.emit(Op.IAdd, [p.tU32, kRow, kBlockBase, threadIdx]);

  const kRowInBounds = b.id();
  b.emit(Op.ULessThan, [p.tBool, kRowInBounds, kRow, T]);
  const inBoundsF = b.id();
  b.emit(Op.Select, [p.tF32, inBoundsF, kRowInBounds, const1f, p.const0f]);

  const kRowD4 = b.id();
  b.emit(Op.IMul, [p.tU32, kRowD4, kRow, constD4]);
  const kGlobalBase4 = b.id();
  b.emit(Op.IAdd, [p.tU32, kGlobalBase4, baseOff4, kRowD4]);

  const sharedRowOff4 = b.id();
  b.emit(Op.IMul, [p.tU32, sharedRowOff4, threadIdx, constD4]);

  for (let d4 = 0; d4 < D4; d4++) {
    const gIdx = b.id();
    b.emit(Op.IAdd, [p.tU32, gIdx, kGlobalBase4, constD4Idx[d4]]);
    const ptrKElem = b.id();
    b.emit(Op.AccessChain, [bufK.tPtrVec4, ptrKElem, bufK.varId, p.const0u, gIdx]);
    const kRaw = b.id();
    b.emit(Op.Load, [tVec4F32, kRaw, ptrKElem]);
    const kVal = b.id();
    b.emit(Op.VectorTimesScalar, [tVec4F32, kVal, kRaw, inBoundsF]);
    const sKIdx = b.id();
    b.emit(Op.IAdd, [p.tU32, sKIdx, sharedRowOff4, constD4Idx[d4]]);
    const ptrSK = b.id();
    b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSK, sK, sKIdx]);
    b.emit(Op.Store, [ptrSK, kVal]);
    const ptrVElem = b.id();
    b.emit(Op.AccessChain, [bufV.tPtrVec4, ptrVElem, bufV.varId, p.const0u, gIdx]);
    const vRaw = b.id();
    b.emit(Op.Load, [tVec4F32, vRaw, ptrVElem]);
    const vVal = b.id();
    b.emit(Op.VectorTimesScalar, [tVec4F32, vVal, vRaw, inBoundsF]);
    const ptrSV = b.id();
    b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSV, sV, sKIdx]);
    b.emit(Op.Store, [ptrSV, vVal]);
  }

  // Barrier
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  // ── Compile-time unrolled j-loop in batches of BJ ─────────────────────
  // For each batch of BJ j-values:
  //   1. Compute BJ dot products (independent, good ILP)
  //   2. Batch softmax: max of BJ dots, single alpha rescaling
  //   3. V accumulation with single alpha rescaling

  for (let jBatch = 0; jBatch < Bc; jBatch += BJ) {
    // ── Phase 1: Compute BJ dot products ─────────────────────────────────
    const dots: number[] = []; // SSA IDs for dot results (after masking)

    for (let jj = 0; jj < BJ; jj++) {
      const j = jBatch + jj;

      // kPos = kBlockBase + j
      const kPos = b.id();
      b.emit(Op.IAdd, [p.tU32, kPos, kBlockBase, constJIdx[j]]);

      // Vec4 dot product: regQ · sK[j] — regQ is pure SSA (no array load)
      const ptrSK0 = b.id();
      b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSK0, sK, constJD4[j]]);
      const kVec0 = b.id();
      b.emit(Op.Load, [tVec4F32, kVec0, ptrSK0]);
      let dotAcc = b.id();
      b.emit(Op.Dot, [p.tF32, dotAcc, regQVecs[0], kVec0]);

      for (let d4 = 1; d4 < D4; d4++) {
        const sKIdxD4 = b.id();
        b.emit(Op.IAdd, [p.tU32, sKIdxD4, constJD4[j], constD4Idx[d4]]);
        const ptrSKd4 = b.id();
        b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSKd4, sK, sKIdxD4]);
        const kVecD4 = b.id();
        b.emit(Op.Load, [tVec4F32, kVecD4, ptrSKd4]);
        const partial = b.id();
        b.emit(Op.Dot, [p.tF32, partial, regQVecs[d4], kVecD4]);
        const newDotAcc = b.id();
        b.emit(Op.FAdd, [p.tF32, newDotAcc, dotAcc, partial]);
        dotAcc = newDotAcc;
      }

      // Soft capping
      const dotDivCap = b.id();
      b.emit(Op.FMul, [p.tF32, dotDivCap, dotAcc, invSoftCap]);
      const tanhVal = b.id();
      b.emit(Op.ExtInst, [p.tF32, tanhVal, p.glslStd, GLSLstd450.Tanh, dotDivCap]);
      const dotCapped = b.id();
      b.emit(Op.FMul, [p.tF32, dotCapped, tanhVal, softCapValue]);
      const dotAfterCap = b.id();
      b.emit(Op.Select, [p.tF32, dotAfterCap, softCapEnabled, dotCapped, dotAcc]);

      // Causal mask + OOB
      const oob = b.id();
      b.emit(Op.UGreaterThanEqual, [p.tBool, oob, kPos, T]);
      const causal = b.id();
      b.emit(Op.ULessThan, [p.tBool, causal, qRow, kPos]);
      const masked = b.id();
      b.emit(Op.LogicalOr, [p.tBool, masked, oob, causal]);
      const dot = b.id();
      b.emit(Op.Select, [p.tF32, dot, masked, constNegInf, dotAfterCap]);

      dots.push(dot);
    }

    // ── Phase 2: Batched softmax ───────────────────────────────────────────
    // Find max of BJ dots
    let mBatch = dots[0];
    for (let jj = 1; jj < BJ; jj++) {
      const newMax = b.id();
      b.emit(Op.ExtInst, [p.tF32, newMax, p.glslStd, GLSLstd450.FMax, mBatch, dots[jj]]);
      mBatch = newMax;
    }

    // mNew = max(mOld, mBatch)
    const mOld = b.id();
    b.emit(Op.Load, [p.tF32, mOld, varM]);
    const mNew = b.id();
    b.emit(Op.ExtInst, [p.tF32, mNew, p.glslStd, GLSLstd450.FMax, mOld, mBatch]);

    // alpha = exp(mOld - mNew)
    const mDiff = b.id();
    b.emit(Op.FSub, [p.tF32, mDiff, mOld, mNew]);
    const alpha = b.id();
    b.emit(Op.ExtInst, [p.tF32, alpha, p.glslStd, GLSLstd450.Exp, mDiff]);

    // p[jj] = exp(dot[jj] - mNew) for each jj
    const pVals: number[] = [];
    let lBlock = b.id();
    {
      // First p value
      const dotDiff0 = b.id();
      b.emit(Op.FSub, [p.tF32, dotDiff0, dots[0], mNew]);
      const p0 = b.id();
      b.emit(Op.ExtInst, [p.tF32, p0, p.glslStd, GLSLstd450.Exp, dotDiff0]);
      pVals.push(p0);
      // Initialize lBlock with p0
      lBlock = p0;
    }
    for (let jj = 1; jj < BJ; jj++) {
      const dotDiffJ = b.id();
      b.emit(Op.FSub, [p.tF32, dotDiffJ, dots[jj], mNew]);
      const pj = b.id();
      b.emit(Op.ExtInst, [p.tF32, pj, p.glslStd, GLSLstd450.Exp, dotDiffJ]);
      pVals.push(pj);
      const newLBlock = b.id();
      b.emit(Op.FAdd, [p.tF32, newLBlock, lBlock, pj]);
      lBlock = newLBlock;
    }

    // lNew = lOld * alpha + lBlock
    const lOld = b.id();
    b.emit(Op.Load, [p.tF32, lOld, varL]);
    const lScaled = b.id();
    b.emit(Op.FMul, [p.tF32, lScaled, lOld, alpha]);
    const lNew = b.id();
    b.emit(Op.FAdd, [p.tF32, lNew, lScaled, lBlock]);
    b.emit(Op.Store, [varL, lNew]);
    b.emit(Op.Store, [varM, mNew]);

    // ── Phase 3: Rescale O once + accumulate BJ V terms ──────────────────
    for (let d4 = 0; d4 < D4; d4++) {
      const ptrO = b.id();
      b.emit(Op.AccessChain, [tPtrFnVec4, ptrO, regO, constD4Idx[d4]]);
      let oVal = b.id();
      b.emit(Op.Load, [tVec4F32, oVal, ptrO]);

      // Rescale by alpha (once per batch!)
      const oScaled = b.id();
      b.emit(Op.VectorTimesScalar, [tVec4F32, oScaled, oVal, alpha]);
      oVal = oScaled;

      // Accumulate BJ V terms
      for (let jj = 0; jj < BJ; jj++) {
        const j = jBatch + jj;
        const sVIdx = b.id();
        b.emit(Op.IAdd, [p.tU32, sVIdx, constJD4[j], constD4Idx[d4]]);
        const ptrSV = b.id();
        b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSV, sV, sVIdx]);
        const vv = b.id();
        b.emit(Op.Load, [tVec4F32, vv, ptrSV]);
        const pvv = b.id();
        b.emit(Op.VectorTimesScalar, [tVec4F32, pvv, vv, pVals[jj]]);
        const oNew = b.id();
        b.emit(Op.FAdd, [tVec4F32, oNew, oVal, pvv]);
        oVal = oNew;
      }

      b.emit(Op.Store, [ptrO, oVal]);
    }
  } // end jBatch loop

  // Barrier before next K block
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  // ── kBlock loop continuation ────────────────────────────────────────────
  b.emit(Op.Branch, [labelLoopCont]);
  b.emit(Op.Label, [labelLoopCont]);
  const nextKBlock = b.id();
  b.emit(Op.Load, [p.tU32, nextKBlock, varKBlockIdx]);
  const incKBlock = b.id();
  b.emit(Op.IAdd, [p.tU32, incKBlock, nextKBlock, p.const1u]);
  b.emit(Op.Store, [varKBlockIdx, incKBlock]);
  b.emit(Op.Branch, [labelLoopHead]);

  b.emit(Op.Label, [labelLoopMerge]);

  // ── Normalize output and write as vec4 ──────────────────────────────────
  const finalL = b.id();
  b.emit(Op.Load, [p.tF32, finalL, varL]);
  const invL = b.id();
  b.emit(Op.FDiv, [p.tF32, invL, const1f, finalL]);

  const oBase4 = b.id();
  b.emit(Op.IAdd, [p.tU32, oBase4, baseOff4, qRowD4]);

  for (let d4 = 0; d4 < D4; d4++) {
    const ptrRegOd4 = b.id();
    b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegOd4, regO, constD4Idx[d4]]);
    const regOVec = b.id();
    b.emit(Op.Load, [tVec4F32, regOVec, ptrRegOd4]);
    const oNorm = b.id();
    b.emit(Op.VectorTimesScalar, [tVec4F32, oNorm, regOVec, invL]);

    const oIdx = b.id();
    b.emit(Op.IAdd, [p.tU32, oIdx, oBase4, constD4Idx[d4]]);
    const ptrOut = b.id();
    b.emit(Op.AccessChain, [bufO.tPtrVec4, ptrOut, bufO.varId, p.const0u, oIdx]);
    b.emit(Op.Store, [ptrOut, oNorm]);
  }

  // ── Store LSE: LSE[bhIdx * T + qRow] = m + log(l) ──────────────────────
  const finalM = b.id();
  b.emit(Op.Load, [p.tF32, finalM, varM]);
  const logL = b.id();
  b.emit(Op.ExtInst, [p.tF32, logL, p.glslStd, GLSLstd450.Log, finalL]);
  const lseVal = b.id();
  b.emit(Op.FAdd, [p.tF32, lseVal, finalM, logL]);

  const bhT = b.id();
  b.emit(Op.IMul, [p.tU32, bhT, bhIdx, T]);
  const lseIdx = b.id();
  b.emit(Op.IAdd, [p.tU32, lseIdx, bhT, qRow]);
  const ptrLSE = b.id();
  b.emit(Op.AccessChain, [bufLSE.tPtrF32, ptrLSE, bufLSE.varId, p.const0u, lseIdx]);
  b.emit(Op.Store, [ptrLSE, lseVal]);

  b.emit(Op.Branch, [labelEnd]);

  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}


// ── Kernel: Flash Attention Backward dQ (vec4 + runtime loop) ────────────────
//
// Computes dQ using recomputed attention weights. Vec4 buffers + runtime j loop.
//
// Bindings:
//   0: Q [B*H, T, D] (readonly, vec4)  1: K  2: V  3: dO
//   4: LSE [B*H, T] (readonly, f32)  5: D_precomp  6: dQ (write, vec4)
// Push constants: { T, scale, softCapValue, _pad }
// Dispatch: (ceil(T/Br), B*H, 1)  Workgroup: (Br, 1, 1)

export function kernelFlashAttentionBackwardDQ(Br: number, Bc: number, D: number): Uint32Array {
  const D4 = D >>> 2;
  const b = new SpirVBuilder();
  const p = preamble(b, Br, 1, 1);

  const tVec4F32 = b.id();
  b.typeVector(tVec4F32, p.tF32, 4);

  // Bindings: Q(0), K(1), V(2), dO(3), O(4), LSE(5), Dpre(6,wo), dQ(7,wo)
  // D_precomp is computed inline: Di = dot(dO[i,:], O[i,:]) — eliminates 2 dispatch calls
  const bufQ    = declareStorageBufferVec4(b, tVec4F32, 0, 0, true);
  const bufK    = declareStorageBufferVec4(b, tVec4F32, 0, 1, true);
  const bufV    = declareStorageBufferVec4(b, tVec4F32, 0, 2, true);
  const bufDO   = declareStorageBufferVec4(b, tVec4F32, 0, 3, true);
  const bufO    = declareStorageBufferVec4(b, tVec4F32, 0, 4, true);
  const bufLSE  = declareStorageBuffer(b, p.tF32, p.tU32, 0, 5, true);
  const bufDpre = declareStorageBuffer(b, p.tF32, p.tU32, 0, 6, false, true);
  const bufDQ   = declareStorageBufferVec4(b, tVec4F32, 0, 7, false, true);

  const pc = declareParamsPushConstant(b, p.tF32, 4);

  const constBr = b.id(); b.constant(p.tU32, constBr, Br);
  const constBc = b.id(); b.constant(p.tU32, constBc, Bc);
  const constD4 = b.id(); b.constant(p.tU32, constD4, D4);
  const constNegInf = b.id(); b.constant(p.tF32, constNegInf, 0xFF800000);
  const const1f = b.id(); b.constantF32(p.tF32, const1f, 1.0);
  const vec4Zero = b.id(); b.constantNull(tVec4F32, vec4Zero);

  const constD4Idx: number[] = [];
  for (let d4 = 0; d4 < D4; d4++) {
    const cd4 = b.id(); b.constant(p.tU32, cd4, d4); constD4Idx.push(cd4);
  }

  // Shared memory: sK[Bc*D4], sV[Bc*D4] of vec4
  const constSharedSize = b.id(); b.constant(p.tU32, constSharedSize, Bc * D4);
  const tArrayShared = b.id(); b.typeArray(tArrayShared, tVec4F32, constSharedSize);
  const tPtrSharedArr = b.id(); b.typePointer(tPtrSharedArr, StorageClass.Workgroup, tArrayShared);
  const tPtrSharedVec4 = b.id(); b.typePointer(tPtrSharedVec4, StorageClass.Workgroup, tVec4F32);
  const sK = b.id(); b.variable(tPtrSharedArr, sK, StorageClass.Workgroup);
  const sV = b.id(); b.variable(tPtrSharedArr, sV, StorageClass.Workgroup);

  // SSA arrays for Q and dO (read-only, loaded once into registers)
  const regQVecs: number[] = [];
  const regDOVecs: number[] = [];

  // Function array for regDQ only (accumulated in loop)
  const constD4Arr = b.id(); b.constant(p.tU32, constD4Arr, D4);
  const tArrayD4 = b.id(); b.typeArray(tArrayD4, tVec4F32, constD4Arr);
  const tPtrFnArr = b.id(); b.typePointer(tPtrFnArr, StorageClass.Function, tArrayD4);
  const tPtrFnVec4 = b.id(); b.typePointer(tPtrFnVec4, StorageClass.Function, tVec4F32);
  const tPtrFnF32 = b.id(); b.typePointer(tPtrFnF32, StorageClass.Function, p.tF32);
  const tPtrFnU32 = b.id(); b.typePointer(tPtrFnU32, StorageClass.Function, p.tU32);

  const tPtrInputVec3 = b.id(); b.typePointer(tPtrInputVec3, StorageClass.Input, p.tVec3U32);
  const vWorkgroupId = b.id(); b.variable(tPtrInputVec3, vWorkgroupId, StorageClass.Input);
  b.addDecorate(vWorkgroupId, Decoration.BuiltIn, BuiltIn.WorkgroupId);
  const vLocalId = b.id(); b.variable(tPtrInputVec3, vLocalId, StorageClass.Input);
  b.addDecorate(vLocalId, Decoration.BuiltIn, BuiltIn.LocalInvocationId);

  const scopeWg = b.id(); b.constant(p.tU32, scopeWg, Scope.Workgroup);
  const semAcqRelWg = b.id(); b.constant(p.tU32, semAcqRelWg, MemorySemantics.AcquireRelease | MemorySemantics.WorkgroupMemory);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId, vWorkgroupId, vLocalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, Br, 1, 1);
  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  const labelEntry = b.id(); b.emit(Op.Label, [labelEntry]);

  // Variable declarations (entry block) — regQ and regDO are SSA, only regDQ needs array
  const regDQ = b.id(); b.emit(Op.Variable, [tPtrFnArr, regDQ, StorageClass.Function]);
  const varKBlockIdx = b.id(); b.emit(Op.Variable, [tPtrFnU32, varKBlockIdx, StorageClass.Function]);
  const varJ = b.id(); b.emit(Op.Variable, [tPtrFnU32, varJ, StorageClass.Function]);

  // Load IDs
  const lidVec = b.id(); b.emit(Op.Load, [p.tVec3U32, lidVec, vLocalId]);
  const threadIdx = b.id(); b.emit(Op.CompositeExtract, [p.tU32, threadIdx, lidVec, 0]);
  const wgIdVec = b.id(); b.emit(Op.Load, [p.tVec3U32, wgIdVec, vWorkgroupId]);
  const qBlockIdx = b.id(); b.emit(Op.CompositeExtract, [p.tU32, qBlockIdx, wgIdVec, 0]);
  const bhIdx = b.id(); b.emit(Op.CompositeExtract, [p.tU32, bhIdx, wgIdVec, 1]);

  const qBlockOff = b.id(); b.emit(Op.IMul, [p.tU32, qBlockOff, qBlockIdx, constBr]);
  const qRow = b.id(); b.emit(Op.IAdd, [p.tU32, qRow, qBlockOff, threadIdx]);

  // Push constants
  const ptrTpc = b.id(); b.emit(Op.AccessChain, [pc.tPtrF32, ptrTpc, pc.varId, p.const0u]);
  const TF = b.id(); b.emit(Op.Load, [p.tF32, TF, ptrTpc]);
  const T = b.id(); b.emit(Op.ConvertFToU, [p.tU32, T, TF]);
  const ptrScale = b.id(); b.emit(Op.AccessChain, [pc.tPtrF32, ptrScale, pc.varId, p.const1u]);
  const scale = b.id(); b.emit(Op.Load, [p.tF32, scale, ptrScale]);
  const ptrSoftCap = b.id(); b.emit(Op.AccessChain, [pc.tPtrF32, ptrSoftCap, pc.varId, p.const2u]);
  const softCapValue = b.id(); b.emit(Op.Load, [p.tF32, softCapValue, ptrSoftCap]);
  const softCapEnabled = b.id(); b.emit(Op.FOrdGreaterThan, [p.tBool, softCapEnabled, softCapValue, p.const0f]);
  const invSoftCap = b.id(); b.emit(Op.FDiv, [p.tF32, invSoftCap, const1f, softCapValue]);

  // Base offsets
  const TD4 = b.id(); b.emit(Op.IMul, [p.tU32, TD4, T, constD4]);
  const baseOff4 = b.id(); b.emit(Op.IMul, [p.tU32, baseOff4, bhIdx, TD4]);
  const lseBaseOff = b.id(); b.emit(Op.IMul, [p.tU32, lseBaseOff, bhIdx, T]);

  // Bounds check
  const qRowOob = b.id(); b.emit(Op.UGreaterThanEqual, [p.tBool, qRowOob, qRow, T]);
  const labelMain = b.id(); const labelEnd = b.id();
  b.emit(Op.SelectionMerge, [labelEnd, 0]);
  b.emit(Op.BranchConditional, [qRowOob, labelEnd, labelMain]);
  b.emit(Op.Label, [labelMain]);

  // Load Q[qRow] pre-scaled, dO[qRow]
  const qRowD4 = b.id(); b.emit(Op.IMul, [p.tU32, qRowD4, qRow, constD4]);
  const qBase4 = b.id(); b.emit(Op.IAdd, [p.tU32, qBase4, baseOff4, qRowD4]);

  // Load Q[qRow] pre-scaled into SSA registers (stays in GPU registers)
  for (let d4 = 0; d4 < D4; d4++) {
    const qIdx = b.id(); b.emit(Op.IAdd, [p.tU32, qIdx, qBase4, constD4Idx[d4]]);
    const ptrQElem = b.id(); b.emit(Op.AccessChain, [bufQ.tPtrVec4, ptrQElem, bufQ.varId, p.const0u, qIdx]);
    const qVec = b.id(); b.emit(Op.Load, [tVec4F32, qVec, ptrQElem]);
    const qScaled = b.id(); b.emit(Op.VectorTimesScalar, [tVec4F32, qScaled, qVec, scale]);
    regQVecs.push(qScaled);
  }

  // Load dO[qRow] into SSA registers
  for (let d4 = 0; d4 < D4; d4++) {
    const doIdx = b.id(); b.emit(Op.IAdd, [p.tU32, doIdx, qBase4, constD4Idx[d4]]);
    const ptrDOElem = b.id(); b.emit(Op.AccessChain, [bufDO.tPtrVec4, ptrDOElem, bufDO.varId, p.const0u, doIdx]);
    const doVec = b.id(); b.emit(Op.Load, [tVec4F32, doVec, ptrDOElem]);
    regDOVecs.push(doVec);
  }

  // Load LSE[qRow]
  const lseQRowIdx = b.id(); b.emit(Op.IAdd, [p.tU32, lseQRowIdx, lseBaseOff, qRow]);
  const ptrLSEi = b.id(); b.emit(Op.AccessChain, [bufLSE.tPtrF32, ptrLSEi, bufLSE.varId, p.const0u, lseQRowIdx]);
  const lse_i = b.id(); b.emit(Op.Load, [p.tF32, lse_i, ptrLSEi]);

  // Inline D_precomp: Di = sum_d(dO[i,d] * O[i,d]) — dot product via Op.Dot on vec4s
  // Load O[qRow] and compute dot product with already-loaded dO[qRow]
  const oIdx0 = b.id(); b.emit(Op.IAdd, [p.tU32, oIdx0, qBase4, constD4Idx[0]]);
  const ptrO0 = b.id(); b.emit(Op.AccessChain, [bufO.tPtrVec4, ptrO0, bufO.varId, p.const0u, oIdx0]);
  const oVec0 = b.id(); b.emit(Op.Load, [tVec4F32, oVec0, ptrO0]);
  let Di = b.id(); b.emit(Op.Dot, [p.tF32, Di, regDOVecs[0], oVec0]);
  for (let d4 = 1; d4 < D4; d4++) {
    const oIdx = b.id(); b.emit(Op.IAdd, [p.tU32, oIdx, qBase4, constD4Idx[d4]]);
    const ptrO = b.id(); b.emit(Op.AccessChain, [bufO.tPtrVec4, ptrO, bufO.varId, p.const0u, oIdx]);
    const oVec = b.id(); b.emit(Op.Load, [tVec4F32, oVec, ptrO]);
    const partialDi = b.id(); b.emit(Op.Dot, [p.tF32, partialDi, regDOVecs[d4], oVec]);
    const newDi = b.id(); b.emit(Op.FAdd, [p.tF32, newDi, Di, partialDi]);
    Di = newDi;
  }
  // Write Di to D_precomp output buffer (for dKV kernel to read)
  const ptrDiOut = b.id(); b.emit(Op.AccessChain, [bufDpre.tPtrF32, ptrDiOut, bufDpre.varId, p.const0u, lseQRowIdx]);
  b.emit(Op.Store, [ptrDiOut, Di]);

  // Initialize regDQ to vec4(0)
  for (let d4 = 0; d4 < D4; d4++) {
    const ptrRegDQ = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegDQ, regDQ, constD4Idx[d4]]);
    b.emit(Op.Store, [ptrRegDQ, vec4Zero]);
  }

  // effectiveKBlocks = (qBlockIdx + 1) * kBlocksPerQBlock (causal)
  // When Br > Bc, each Q block spans multiple K blocks
  const kBlocksPerQBlock_dq = Br / Bc; // compile-time integer
  const qPlus1_dq = b.id();
  b.emit(Op.IAdd, [p.tU32, qPlus1_dq, qBlockIdx, p.const1u]);
  let effectiveKBlocks: number;
  if (kBlocksPerQBlock_dq === 1) {
    effectiveKBlocks = qPlus1_dq;
  } else {
    const constKBPQB_dq = b.id(); b.constant(p.tU32, constKBPQB_dq, kBlocksPerQBlock_dq);
    effectiveKBlocks = b.id();
    b.emit(Op.IMul, [p.tU32, effectiveKBlocks, qPlus1_dq, constKBPQB_dq]);
  }

  // ── Outer loop: kBlockIdx = 0..effectiveKBlocks ───────────────────────────
  b.emit(Op.Store, [varKBlockIdx, p.const0u]);
  const labelLoopHead = b.id(); const labelLoopBody = b.id();
  const labelLoopMerge = b.id(); const labelLoopCont = b.id();

  b.emit(Op.Branch, [labelLoopHead]);
  b.emit(Op.Label, [labelLoopHead]);
  const kBlockIdx = b.id(); b.emit(Op.Load, [p.tU32, kBlockIdx, varKBlockIdx]);
  const loopCmp = b.id(); b.emit(Op.ULessThan, [p.tBool, loopCmp, kBlockIdx, effectiveKBlocks]);
  b.emit(Op.LoopMerge, [labelLoopMerge, labelLoopCont, 0]);
  b.emit(Op.BranchConditional, [loopCmp, labelLoopBody, labelLoopMerge]);
  b.emit(Op.Label, [labelLoopBody]);

  // Cooperative load K, V into shared as vec4
  // Distribute Bc*D4 elements across all Br threads for optimal coalescing.
  const kBlockBase = b.id(); b.emit(Op.IMul, [p.tU32, kBlockBase, kBlockIdx, constBc]);

  const kBlockBaseTD4_dq = b.id();
  b.emit(Op.IMul, [p.tU32, kBlockBaseTD4_dq, kBlockBase, constD4]);
  const kBlockGlobal4_dq = b.id();
  b.emit(Op.IAdd, [p.tU32, kBlockGlobal4_dq, baseOff4, kBlockBaseTD4_dq]);

  const totalLoadElems_dq = Bc * D4; // compile-time
  const elemsPerThread_dq = Math.ceil(totalLoadElems_dq / Br); // compile-time
  const log2D4_dq = Math.log2(D4); // compile-time
  const constLog2D4_dq = b.id(); b.constant(p.tU32, constLog2D4_dq, log2D4_dq);

  for (let pass = 0; pass < elemsPerThread_dq; pass++) {
    let linearIdx: number;
    if (pass === 0) {
      linearIdx = threadIdx;
    } else {
      const constPassOff = b.id(); b.constant(p.tU32, constPassOff, pass * Br);
      linearIdx = b.id();
      b.emit(Op.IAdd, [p.tU32, linearIdx, threadIdx, constPassOff]);
    }

    const needCheck = (pass + 1) * Br > totalLoadElems_dq;
    let labelPassBody: number | undefined;
    let labelPassEnd: number | undefined;
    if (needCheck) {
      const constTotalElems = b.id(); b.constant(p.tU32, constTotalElems, totalLoadElems_dq);
      const inRange = b.id();
      b.emit(Op.ULessThan, [p.tBool, inRange, linearIdx, constTotalElems]);
      labelPassBody = b.id();
      labelPassEnd = b.id();
      b.emit(Op.SelectionMerge, [labelPassEnd!, 0]);
      b.emit(Op.BranchConditional, [inRange, labelPassBody!, labelPassEnd!]);
      b.emit(Op.Label, [labelPassBody!]);
    }

    const row = b.id();
    b.emit(Op.ShiftRightLogical, [p.tU32, row, linearIdx, constLog2D4_dq]);
    const kRowL = b.id();
    b.emit(Op.IAdd, [p.tU32, kRowL, kBlockBase, row]);
    const kRowInBounds = b.id();
    b.emit(Op.ULessThan, [p.tBool, kRowInBounds, kRowL, T]);
    const inBoundsF = b.id();
    b.emit(Op.Select, [p.tF32, inBoundsF, kRowInBounds, const1f, p.const0f]);

    const gIdx = b.id();
    b.emit(Op.IAdd, [p.tU32, gIdx, kBlockGlobal4_dq, linearIdx]);

    const ptrKElem = b.id(); b.emit(Op.AccessChain, [bufK.tPtrVec4, ptrKElem, bufK.varId, p.const0u, gIdx]);
    const kRaw = b.id(); b.emit(Op.Load, [tVec4F32, kRaw, ptrKElem]);
    const kVal = b.id(); b.emit(Op.VectorTimesScalar, [tVec4F32, kVal, kRaw, inBoundsF]);
    const ptrSK = b.id(); b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSK, sK, linearIdx]);
    b.emit(Op.Store, [ptrSK, kVal]);

    const ptrVElem = b.id(); b.emit(Op.AccessChain, [bufV.tPtrVec4, ptrVElem, bufV.varId, p.const0u, gIdx]);
    const vRaw = b.id(); b.emit(Op.Load, [tVec4F32, vRaw, ptrVElem]);
    const vVal = b.id(); b.emit(Op.VectorTimesScalar, [tVec4F32, vVal, vRaw, inBoundsF]);
    const ptrSV = b.id(); b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSV, sV, linearIdx]);
    b.emit(Op.Store, [ptrSV, vVal]);

    if (needCheck) {
      b.emit(Op.Branch, [labelPassEnd!]);
      b.emit(Op.Label, [labelPassEnd!]);
    }
  }

  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  // ── Inner loop: j = 0..Bc (runtime) ───────────────────────────────────────
  b.emit(Op.Store, [varJ, p.const0u]);
  const labelJHead = b.id(); const labelJBody = b.id();
  const labelJMerge = b.id(); const labelJCont = b.id();

  b.emit(Op.Branch, [labelJHead]);
  b.emit(Op.Label, [labelJHead]);
  const j = b.id(); b.emit(Op.Load, [p.tU32, j, varJ]);
  const jCmp = b.id(); b.emit(Op.ULessThan, [p.tBool, jCmp, j, constBc]);
  b.emit(Op.LoopMerge, [labelJMerge, labelJCont, 0]);
  b.emit(Op.BranchConditional, [jCmp, labelJBody, labelJMerge]);
  b.emit(Op.Label, [labelJBody]);

  const kPos = b.id(); b.emit(Op.IAdd, [p.tU32, kPos, kBlockBase, j]);
  const jD4 = b.id(); b.emit(Op.IMul, [p.tU32, jD4, j, constD4]);

  // Cache K[j] from shared memory (loaded once, reused for Q·K dot + dQ accumulation)
  const kCache: number[] = [];
  for (let d4 = 0; d4 < D4; d4++) {
    let sKIdx: number;
    if (d4 === 0) { sKIdx = jD4; }
    else { sKIdx = b.id(); b.emit(Op.IAdd, [p.tU32, sKIdx, jD4, constD4Idx[d4]]); }
    const ptrSK = b.id(); b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSK, sK, sKIdx]);
    const kVec = b.id(); b.emit(Op.Load, [tVec4F32, kVec, ptrSK]);
    kCache.push(kVec);
  }

  // Dot product: SSA regQ · cached K[j] (Q pre-scaled)
  let dotAcc = b.id(); b.emit(Op.Dot, [p.tF32, dotAcc, regQVecs[0], kCache[0]]);

  for (let d4 = 1; d4 < D4; d4++) {
    const partial = b.id(); b.emit(Op.Dot, [p.tF32, partial, regQVecs[d4], kCache[d4]]);
    const newDotAcc = b.id(); b.emit(Op.FAdd, [p.tF32, newDotAcc, dotAcc, partial]);
    dotAcc = newDotAcc;
  }

  // Soft capping
  const dotDivCap = b.id(); b.emit(Op.FMul, [p.tF32, dotDivCap, dotAcc, invSoftCap]);
  const tanhVal = b.id(); b.emit(Op.ExtInst, [p.tF32, tanhVal, p.glslStd, GLSLstd450.Tanh, dotDivCap]);
  const dotCapped = b.id(); b.emit(Op.FMul, [p.tF32, dotCapped, tanhVal, softCapValue]);
  const dotAfterCap = b.id(); b.emit(Op.Select, [p.tF32, dotAfterCap, softCapEnabled, dotCapped, dotAcc]);

  // Causal mask + OOB
  const oob = b.id(); b.emit(Op.UGreaterThanEqual, [p.tBool, oob, kPos, T]);
  const causal = b.id(); b.emit(Op.ULessThan, [p.tBool, causal, qRow, kPos]);
  const masked = b.id(); b.emit(Op.LogicalOr, [p.tBool, masked, oob, causal]);
  const dot = b.id(); b.emit(Op.Select, [p.tF32, dot, masked, constNegInf, dotAfterCap]);

  // p_ij = exp(dot - lse_i)
  const dotMinusLSE = b.id(); b.emit(Op.FSub, [p.tF32, dotMinusLSE, dot, lse_i]);
  const p_ij = b.id(); b.emit(Op.ExtInst, [p.tF32, p_ij, p.glslStd, GLSLstd450.Exp, dotMinusLSE]);

  // dotDOV = SSA regDO · sV[j]
  const ptrSV0 = b.id(); b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSV0, sV, jD4]);
  const vVec0 = b.id(); b.emit(Op.Load, [tVec4F32, vVec0, ptrSV0]);
  let dovAcc = b.id(); b.emit(Op.Dot, [p.tF32, dovAcc, regDOVecs[0], vVec0]);

  for (let d4 = 1; d4 < D4; d4++) {
    const sVIdxD4 = b.id(); b.emit(Op.IAdd, [p.tU32, sVIdxD4, jD4, constD4Idx[d4]]);
    const ptrSVd4 = b.id(); b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSVd4, sV, sVIdxD4]);
    const vVecD4 = b.id(); b.emit(Op.Load, [tVec4F32, vVecD4, ptrSVd4]);
    const partialDOV = b.id(); b.emit(Op.Dot, [p.tF32, partialDOV, regDOVecs[d4], vVecD4]);
    const newDovAcc = b.id(); b.emit(Op.FAdd, [p.tF32, newDovAcc, dovAcc, partialDOV]);
    dovAcc = newDovAcc;
  }

  // dS_ij = p_ij * (dotDOV - Di)
  const dovMinusDi = b.id(); b.emit(Op.FSub, [p.tF32, dovMinusDi, dovAcc, Di]);
  const dS_ij = b.id(); b.emit(Op.FMul, [p.tF32, dS_ij, p_ij, dovMinusDi]);

  // dScore_ij with softcap derivative
  const tanhSq = b.id(); b.emit(Op.FMul, [p.tF32, tanhSq, tanhVal, tanhVal]);
  const deriv = b.id(); b.emit(Op.FSub, [p.tF32, deriv, const1f, tanhSq]);
  const dSscale = b.id(); b.emit(Op.FMul, [p.tF32, dSscale, dS_ij, scale]);
  const dScore_capped = b.id(); b.emit(Op.FMul, [p.tF32, dScore_capped, dSscale, deriv]);
  const dScore_uncapped = b.id(); b.emit(Op.FMul, [p.tF32, dScore_uncapped, dS_ij, scale]);
  const dScore_ij = b.id(); b.emit(Op.Select, [p.tF32, dScore_ij, softCapEnabled, dScore_capped, dScore_uncapped]);

  // dQ accumulation using cached K (no redundant shared memory reload)
  for (let d4 = 0; d4 < D4; d4++) {
    const ptrDQ = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrDQ, regDQ, constD4Idx[d4]]);
    const dqOld = b.id(); b.emit(Op.Load, [tVec4F32, dqOld, ptrDQ]);
    const dScoreK = b.id(); b.emit(Op.VectorTimesScalar, [tVec4F32, dScoreK, kCache[d4], dScore_ij]);
    const dqNew = b.id(); b.emit(Op.FAdd, [tVec4F32, dqNew, dqOld, dScoreK]);
    b.emit(Op.Store, [ptrDQ, dqNew]);
  }

  // j loop continuation
  b.emit(Op.Branch, [labelJCont]);
  b.emit(Op.Label, [labelJCont]);
  const nextJ = b.id(); b.emit(Op.Load, [p.tU32, nextJ, varJ]);
  const incJ = b.id(); b.emit(Op.IAdd, [p.tU32, incJ, nextJ, p.const1u]);
  b.emit(Op.Store, [varJ, incJ]);
  b.emit(Op.Branch, [labelJHead]);
  b.emit(Op.Label, [labelJMerge]);

  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  // kBlock loop continuation
  b.emit(Op.Branch, [labelLoopCont]);
  b.emit(Op.Label, [labelLoopCont]);
  const nextKBlock = b.id(); b.emit(Op.Load, [p.tU32, nextKBlock, varKBlockIdx]);
  const incKBlock = b.id(); b.emit(Op.IAdd, [p.tU32, incKBlock, nextKBlock, p.const1u]);
  b.emit(Op.Store, [varKBlockIdx, incKBlock]);
  b.emit(Op.Branch, [labelLoopHead]);
  b.emit(Op.Label, [labelLoopMerge]);

  // Write regDQ to output
  const dqBase4 = b.id(); b.emit(Op.IAdd, [p.tU32, dqBase4, baseOff4, qRowD4]);
  for (let d4 = 0; d4 < D4; d4++) {
    const ptrRegDQd4 = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegDQd4, regDQ, constD4Idx[d4]]);
    const regDQVec = b.id(); b.emit(Op.Load, [tVec4F32, regDQVec, ptrRegDQd4]);
    const dqIdx = b.id(); b.emit(Op.IAdd, [p.tU32, dqIdx, dqBase4, constD4Idx[d4]]);
    const ptrOut = b.id(); b.emit(Op.AccessChain, [bufDQ.tPtrVec4, ptrOut, bufDQ.varId, p.const0u, dqIdx]);
    b.emit(Op.Store, [ptrOut, regDQVec]);
  }

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);
  return b.build();
}

// ── Kernel: Flash Attention Backward dKV (vec4 + runtime loop) ───────────────
//
// Computes dK and dV using recomputed attention weights. Vec4 buffers + runtime i loop.
//
// Bindings:
//   0: Q [B*H, T, D] (readonly, vec4)  1: K  2: V  3: dO
//   4: LSE [B*H, T] (readonly, f32)  5: D_precomp
//   6: dK (write, vec4)  7: dV (write, vec4)
// Push constants: { T, scale, softCapValue, _pad }
// Dispatch: (ceil(T/Bc), B*H, 1)  Workgroup: (Bc, 1, 1)

export function kernelFlashAttentionBackwardDKV(Br: number, Bc: number, D: number): Uint32Array {
  const D4 = D >>> 2;
  const b = new SpirVBuilder();
  const p = preamble(b, Bc, 1, 1);

  const tVec4F32 = b.id();
  b.typeVector(tVec4F32, p.tF32, 4);

  const bufQ    = declareStorageBufferVec4(b, tVec4F32, 0, 0, true);
  const bufK    = declareStorageBufferVec4(b, tVec4F32, 0, 1, true);
  const bufV    = declareStorageBufferVec4(b, tVec4F32, 0, 2, true);
  const bufDO   = declareStorageBufferVec4(b, tVec4F32, 0, 3, true);
  const bufLSE  = declareStorageBuffer(b, p.tF32, p.tU32, 0, 4, true);
  const bufDpre = declareStorageBuffer(b, p.tF32, p.tU32, 0, 5, true);
  const bufDK   = declareStorageBufferVec4(b, tVec4F32, 0, 6, false, true);
  const bufDV   = declareStorageBufferVec4(b, tVec4F32, 0, 7, false, true);

  const pc = declareParamsPushConstant(b, p.tF32, 4);

  const constBr = b.id(); b.constant(p.tU32, constBr, Br);
  const constBc = b.id(); b.constant(p.tU32, constBc, Bc);
  const constD4 = b.id(); b.constant(p.tU32, constD4, D4);
  const constBrMinus1 = b.id(); b.constant(p.tU32, constBrMinus1, Br - 1);
  const constNegInf = b.id(); b.constant(p.tF32, constNegInf, 0xFF800000);
  const const1f = b.id(); b.constantF32(p.tF32, const1f, 1.0);
  const vec4Zero = b.id(); b.constantNull(tVec4F32, vec4Zero);

  const constD4Idx: number[] = [];
  for (let d4 = 0; d4 < D4; d4++) {
    const cd4 = b.id(); b.constant(p.tU32, cd4, d4); constD4Idx.push(cd4);
  }

  // Shared memory: sQ[Br*D4], sDO[Br*D4] of vec4, sLSE[Br], sDpre[Br] of f32
  const constSharedVec4Size = b.id(); b.constant(p.tU32, constSharedVec4Size, Br * D4);
  const tArraySharedVec4 = b.id(); b.typeArray(tArraySharedVec4, tVec4F32, constSharedVec4Size);
  const tPtrSharedArrVec4 = b.id(); b.typePointer(tPtrSharedArrVec4, StorageClass.Workgroup, tArraySharedVec4);
  const tPtrSharedVec4 = b.id(); b.typePointer(tPtrSharedVec4, StorageClass.Workgroup, tVec4F32);
  const sQ = b.id(); b.variable(tPtrSharedArrVec4, sQ, StorageClass.Workgroup);
  const sDO = b.id(); b.variable(tPtrSharedArrVec4, sDO, StorageClass.Workgroup);

  const constSharedSmallSize = b.id(); b.constant(p.tU32, constSharedSmallSize, Br);
  const tArraySharedSmall = b.id(); b.typeArray(tArraySharedSmall, p.tF32, constSharedSmallSize);
  const tPtrSharedSmallArr = b.id(); b.typePointer(tPtrSharedSmallArr, StorageClass.Workgroup, tArraySharedSmall);
  const tPtrSharedF32 = b.id(); b.typePointer(tPtrSharedF32, StorageClass.Workgroup, p.tF32);
  const sLSE = b.id(); b.variable(tPtrSharedSmallArr, sLSE, StorageClass.Workgroup);
  const sDpre = b.id(); b.variable(tPtrSharedSmallArr, sDpre, StorageClass.Workgroup);

  // Function arrays: regK[D4], regV[D4], regDK[D4], regDV[D4] of vec4
  const constD4Arr = b.id(); b.constant(p.tU32, constD4Arr, D4);
  const tArrayD4 = b.id(); b.typeArray(tArrayD4, tVec4F32, constD4Arr);
  const tPtrFnArr = b.id(); b.typePointer(tPtrFnArr, StorageClass.Function, tArrayD4);
  const tPtrFnVec4 = b.id(); b.typePointer(tPtrFnVec4, StorageClass.Function, tVec4F32);
  const tPtrFnF32 = b.id(); b.typePointer(tPtrFnF32, StorageClass.Function, p.tF32);
  const tPtrFnU32 = b.id(); b.typePointer(tPtrFnU32, StorageClass.Function, p.tU32);

  const tPtrInputVec3 = b.id(); b.typePointer(tPtrInputVec3, StorageClass.Input, p.tVec3U32);
  const vWorkgroupId = b.id(); b.variable(tPtrInputVec3, vWorkgroupId, StorageClass.Input);
  b.addDecorate(vWorkgroupId, Decoration.BuiltIn, BuiltIn.WorkgroupId);
  const vLocalId = b.id(); b.variable(tPtrInputVec3, vLocalId, StorageClass.Input);
  b.addDecorate(vLocalId, Decoration.BuiltIn, BuiltIn.LocalInvocationId);

  const scopeWg = b.id(); b.constant(p.tU32, scopeWg, Scope.Workgroup);
  const semAcqRelWg = b.id(); b.constant(p.tU32, semAcqRelWg, MemorySemantics.AcquireRelease | MemorySemantics.WorkgroupMemory);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId, vWorkgroupId, vLocalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, Bc, 1, 1);
  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  const labelEntry = b.id(); b.emit(Op.Label, [labelEntry]);

  // Variable declarations
  const regK = b.id(); b.emit(Op.Variable, [tPtrFnArr, regK, StorageClass.Function]);
  const regV = b.id(); b.emit(Op.Variable, [tPtrFnArr, regV, StorageClass.Function]);
  const regDK = b.id(); b.emit(Op.Variable, [tPtrFnArr, regDK, StorageClass.Function]);
  const regDV = b.id(); b.emit(Op.Variable, [tPtrFnArr, regDV, StorageClass.Function]);
  const varQBlockIdx = b.id(); b.emit(Op.Variable, [tPtrFnU32, varQBlockIdx, StorageClass.Function]);
  const varI = b.id(); b.emit(Op.Variable, [tPtrFnU32, varI, StorageClass.Function]);

  // Load IDs
  const lidVec = b.id(); b.emit(Op.Load, [p.tVec3U32, lidVec, vLocalId]);
  const threadIdx = b.id(); b.emit(Op.CompositeExtract, [p.tU32, threadIdx, lidVec, 0]);
  const wgIdVec = b.id(); b.emit(Op.Load, [p.tVec3U32, wgIdVec, vWorkgroupId]);
  const kBlockIdx = b.id(); b.emit(Op.CompositeExtract, [p.tU32, kBlockIdx, wgIdVec, 0]);
  const bhIdx = b.id(); b.emit(Op.CompositeExtract, [p.tU32, bhIdx, wgIdVec, 1]);

  const kBlockOff = b.id(); b.emit(Op.IMul, [p.tU32, kBlockOff, kBlockIdx, constBc]);
  const kRow = b.id(); b.emit(Op.IAdd, [p.tU32, kRow, kBlockOff, threadIdx]);

  // Push constants
  const ptrTpc = b.id(); b.emit(Op.AccessChain, [pc.tPtrF32, ptrTpc, pc.varId, p.const0u]);
  const TF = b.id(); b.emit(Op.Load, [p.tF32, TF, ptrTpc]);
  const T = b.id(); b.emit(Op.ConvertFToU, [p.tU32, T, TF]);
  const ptrScale = b.id(); b.emit(Op.AccessChain, [pc.tPtrF32, ptrScale, pc.varId, p.const1u]);
  const scale = b.id(); b.emit(Op.Load, [p.tF32, scale, ptrScale]);
  const ptrSoftCap = b.id(); b.emit(Op.AccessChain, [pc.tPtrF32, ptrSoftCap, pc.varId, p.const2u]);
  const softCapValue = b.id(); b.emit(Op.Load, [p.tF32, softCapValue, ptrSoftCap]);
  const softCapEnabled = b.id(); b.emit(Op.FOrdGreaterThan, [p.tBool, softCapEnabled, softCapValue, p.const0f]);
  const invSoftCap = b.id(); b.emit(Op.FDiv, [p.tF32, invSoftCap, const1f, softCapValue]);

  // Base offsets
  const TD4 = b.id(); b.emit(Op.IMul, [p.tU32, TD4, T, constD4]);
  const baseOff4 = b.id(); b.emit(Op.IMul, [p.tU32, baseOff4, bhIdx, TD4]);
  const lseBaseOff = b.id(); b.emit(Op.IMul, [p.tU32, lseBaseOff, bhIdx, T]);

  // Bounds check
  const kRowOob = b.id(); b.emit(Op.UGreaterThanEqual, [p.tBool, kRowOob, kRow, T]);
  const labelMain = b.id(); const labelEnd = b.id();
  b.emit(Op.SelectionMerge, [labelEnd, 0]);
  b.emit(Op.BranchConditional, [kRowOob, labelEnd, labelMain]);
  b.emit(Op.Label, [labelMain]);

  // Load K[kRow] pre-scaled, V[kRow]
  const kRowD4 = b.id(); b.emit(Op.IMul, [p.tU32, kRowD4, kRow, constD4]);
  const kBase4 = b.id(); b.emit(Op.IAdd, [p.tU32, kBase4, baseOff4, kRowD4]);

  for (let d4 = 0; d4 < D4; d4++) {
    const kIdx = b.id(); b.emit(Op.IAdd, [p.tU32, kIdx, kBase4, constD4Idx[d4]]);
    const ptrKElem = b.id(); b.emit(Op.AccessChain, [bufK.tPtrVec4, ptrKElem, bufK.varId, p.const0u, kIdx]);
    const kVec = b.id(); b.emit(Op.Load, [tVec4F32, kVec, ptrKElem]);
    const kScaled = b.id(); b.emit(Op.VectorTimesScalar, [tVec4F32, kScaled, kVec, scale]);
    const ptrRegK = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegK, regK, constD4Idx[d4]]);
    b.emit(Op.Store, [ptrRegK, kScaled]);
  }

  for (let d4 = 0; d4 < D4; d4++) {
    const vIdx = b.id(); b.emit(Op.IAdd, [p.tU32, vIdx, kBase4, constD4Idx[d4]]);
    const ptrVElem = b.id(); b.emit(Op.AccessChain, [bufV.tPtrVec4, ptrVElem, bufV.varId, p.const0u, vIdx]);
    const vVec = b.id(); b.emit(Op.Load, [tVec4F32, vVec, ptrVElem]);
    const ptrRegV = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegV, regV, constD4Idx[d4]]);
    b.emit(Op.Store, [ptrRegV, vVec]);
  }

  // Initialize regDK, regDV to vec4(0)
  for (let d4 = 0; d4 < D4; d4++) {
    const ptrRegDK = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegDK, regDK, constD4Idx[d4]]);
    b.emit(Op.Store, [ptrRegDK, vec4Zero]);
    const ptrRegDV = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegDV, regDV, constD4Idx[d4]]);
    b.emit(Op.Store, [ptrRegDV, vec4Zero]);
  }

  // numQBlocks = ceil(T / Br) = (T + Br - 1) / Br
  const TplusBrm1 = b.id(); b.emit(Op.IAdd, [p.tU32, TplusBrm1, T, constBrMinus1]);
  const numQBlocks = b.id(); b.emit(Op.UDiv, [p.tU32, numQBlocks, TplusBrm1, constBr]);

  // ── Outer loop: qBlockIdx = kBlockIdx..numQBlocks ─────────────────────────
  b.emit(Op.Store, [varQBlockIdx, kBlockIdx]);
  const labelLoopHead = b.id(); const labelLoopBody = b.id();
  const labelLoopMerge = b.id(); const labelLoopCont = b.id();

  b.emit(Op.Branch, [labelLoopHead]);
  b.emit(Op.Label, [labelLoopHead]);
  const qBlockIdx = b.id(); b.emit(Op.Load, [p.tU32, qBlockIdx, varQBlockIdx]);
  const loopCmp = b.id(); b.emit(Op.ULessThan, [p.tBool, loopCmp, qBlockIdx, numQBlocks]);
  b.emit(Op.LoopMerge, [labelLoopMerge, labelLoopCont, 0]);
  b.emit(Op.BranchConditional, [loopCmp, labelLoopBody, labelLoopMerge]);
  b.emit(Op.Label, [labelLoopBody]);

  // Cooperative load Q, dO into shared as vec4 + LSE, D_precomp as scalar
  const qBlockBase = b.id(); b.emit(Op.IMul, [p.tU32, qBlockBase, qBlockIdx, constBr]);
  const qRow = b.id(); b.emit(Op.IAdd, [p.tU32, qRow, qBlockBase, threadIdx]);
  const qRowInBounds = b.id(); b.emit(Op.ULessThan, [p.tBool, qRowInBounds, qRow, T]);
  const inBoundsF = b.id(); b.emit(Op.Select, [p.tF32, inBoundsF, qRowInBounds, const1f, p.const0f]);
  const qRowD4 = b.id(); b.emit(Op.IMul, [p.tU32, qRowD4, qRow, constD4]);
  const qGlobalBase4 = b.id(); b.emit(Op.IAdd, [p.tU32, qGlobalBase4, baseOff4, qRowD4]);
  const sharedRowOff4 = b.id(); b.emit(Op.IMul, [p.tU32, sharedRowOff4, threadIdx, constD4]);

  for (let d4 = 0; d4 < D4; d4++) {
    const gIdx = b.id(); b.emit(Op.IAdd, [p.tU32, gIdx, qGlobalBase4, constD4Idx[d4]]);
    const ptrQElem = b.id(); b.emit(Op.AccessChain, [bufQ.tPtrVec4, ptrQElem, bufQ.varId, p.const0u, gIdx]);
    const qRaw = b.id(); b.emit(Op.Load, [tVec4F32, qRaw, ptrQElem]);
    const qVal = b.id(); b.emit(Op.VectorTimesScalar, [tVec4F32, qVal, qRaw, inBoundsF]);
    const sQIdx = b.id(); b.emit(Op.IAdd, [p.tU32, sQIdx, sharedRowOff4, constD4Idx[d4]]);
    const ptrSQ = b.id(); b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSQ, sQ, sQIdx]);
    b.emit(Op.Store, [ptrSQ, qVal]);

    const ptrDOElem = b.id(); b.emit(Op.AccessChain, [bufDO.tPtrVec4, ptrDOElem, bufDO.varId, p.const0u, gIdx]);
    const doRaw = b.id(); b.emit(Op.Load, [tVec4F32, doRaw, ptrDOElem]);
    const doVal = b.id(); b.emit(Op.VectorTimesScalar, [tVec4F32, doVal, doRaw, inBoundsF]);
    const ptrSDO = b.id(); b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSDO, sDO, sQIdx]);
    b.emit(Op.Store, [ptrSDO, doVal]);
  }

  // Load LSE and D_precomp for this thread's query row
  const lseQRowIdx = b.id(); b.emit(Op.IAdd, [p.tU32, lseQRowIdx, lseBaseOff, qRow]);
  const ptrLSEi = b.id(); b.emit(Op.AccessChain, [bufLSE.tPtrF32, ptrLSEi, bufLSE.varId, p.const0u, lseQRowIdx]);
  const lseRaw = b.id(); b.emit(Op.Load, [p.tF32, lseRaw, ptrLSEi]);
  const lseVal = b.id(); b.emit(Op.Select, [p.tF32, lseVal, qRowInBounds, lseRaw, p.const0f]);
  const ptrSLSE = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrSLSE, sLSE, threadIdx]);
  b.emit(Op.Store, [ptrSLSE, lseVal]);

  const ptrDpreI = b.id(); b.emit(Op.AccessChain, [bufDpre.tPtrF32, ptrDpreI, bufDpre.varId, p.const0u, lseQRowIdx]);
  const dpreRaw = b.id(); b.emit(Op.Load, [p.tF32, dpreRaw, ptrDpreI]);
  const dpreVal = b.id(); b.emit(Op.Select, [p.tF32, dpreVal, qRowInBounds, dpreRaw, p.const0f]);
  const ptrSDpre = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrSDpre, sDpre, threadIdx]);
  b.emit(Op.Store, [ptrSDpre, dpreVal]);

  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  // ── Inner loop: i = 0..Br (runtime) ───────────────────────────────────────
  b.emit(Op.Store, [varI, p.const0u]);
  const labelIHead = b.id(); const labelIBody = b.id();
  const labelIMerge = b.id(); const labelICont = b.id();

  b.emit(Op.Branch, [labelIHead]);
  b.emit(Op.Label, [labelIHead]);
  const i = b.id(); b.emit(Op.Load, [p.tU32, i, varI]);
  const iCmp = b.id(); b.emit(Op.ULessThan, [p.tBool, iCmp, i, constBr]);
  b.emit(Op.LoopMerge, [labelIMerge, labelICont, 0]);
  b.emit(Op.BranchConditional, [iCmp, labelIBody, labelIMerge]);
  b.emit(Op.Label, [labelIBody]);

  const qPos = b.id(); b.emit(Op.IAdd, [p.tU32, qPos, qBlockBase, i]);
  const iD4 = b.id(); b.emit(Op.IMul, [p.tU32, iD4, i, constD4]);

  // Cache Q[i] from shared memory (loaded once, reused for K·Q dot + dK accumulation)
  const qCache: number[] = [];
  for (let d4 = 0; d4 < D4; d4++) {
    let sQIdx: number;
    if (d4 === 0) { sQIdx = iD4; }
    else { sQIdx = b.id(); b.emit(Op.IAdd, [p.tU32, sQIdx, iD4, constD4Idx[d4]]); }
    const ptrSQ = b.id(); b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSQ, sQ, sQIdx]);
    const qVec = b.id(); b.emit(Op.Load, [tVec4F32, qVec, ptrSQ]);
    qCache.push(qVec);
  }

  // Dot product: regK · cached Q[i] (K pre-scaled)
  const ptrRegK0 = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegK0, regK, constD4Idx[0]]);
  const kVec0 = b.id(); b.emit(Op.Load, [tVec4F32, kVec0, ptrRegK0]);
  let dotAcc = b.id(); b.emit(Op.Dot, [p.tF32, dotAcc, kVec0, qCache[0]]);

  for (let d4 = 1; d4 < D4; d4++) {
    const ptrRegKd4 = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegKd4, regK, constD4Idx[d4]]);
    const kVecD4 = b.id(); b.emit(Op.Load, [tVec4F32, kVecD4, ptrRegKd4]);
    const partial = b.id(); b.emit(Op.Dot, [p.tF32, partial, kVecD4, qCache[d4]]);
    const newDotAcc = b.id(); b.emit(Op.FAdd, [p.tF32, newDotAcc, dotAcc, partial]);
    dotAcc = newDotAcc;
  }

  // Soft capping
  const dotDivCap = b.id(); b.emit(Op.FMul, [p.tF32, dotDivCap, dotAcc, invSoftCap]);
  const tanhVal = b.id(); b.emit(Op.ExtInst, [p.tF32, tanhVal, p.glslStd, GLSLstd450.Tanh, dotDivCap]);
  const dotCapped = b.id(); b.emit(Op.FMul, [p.tF32, dotCapped, tanhVal, softCapValue]);
  const dotAfterCap = b.id(); b.emit(Op.Select, [p.tF32, dotAfterCap, softCapEnabled, dotCapped, dotAcc]);

  // Causal mask: qPos < kRow or qPos >= T → -inf
  const oob = b.id(); b.emit(Op.UGreaterThanEqual, [p.tBool, oob, qPos, T]);
  const causal = b.id(); b.emit(Op.ULessThan, [p.tBool, causal, qPos, kRow]);
  const masked = b.id(); b.emit(Op.LogicalOr, [p.tBool, masked, oob, causal]);
  const dot = b.id(); b.emit(Op.Select, [p.tF32, dot, masked, constNegInf, dotAfterCap]);

  // p_ij = exp(dot - sLSE[i])
  const ptrSLSEi = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrSLSEi, sLSE, i]);
  const lse_i = b.id(); b.emit(Op.Load, [p.tF32, lse_i, ptrSLSEi]);
  const dotMinusLSE = b.id(); b.emit(Op.FSub, [p.tF32, dotMinusLSE, dot, lse_i]);
  const p_ij = b.id(); b.emit(Op.ExtInst, [p.tF32, p_ij, p.glslStd, GLSLstd450.Exp, dotMinusLSE]);

  // dV accumulation + dotDOV computation (interleaved, sharing sDO loads)
  let dovAcc = p.const0f;
  for (let d4 = 0; d4 < D4; d4++) {
    // Load sDO[i*D4+d4]
    const sDOIdx = b.id();
    if (d4 === 0) { b.emit(Op.CopyObject, [p.tU32, sDOIdx, iD4]); }
    else { b.emit(Op.IAdd, [p.tU32, sDOIdx, iD4, constD4Idx[d4]]); }
    const ptrSDOd = b.id(); b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSDOd, sDO, sDOIdx]);
    const doVec = b.id(); b.emit(Op.Load, [tVec4F32, doVec, ptrSDOd]);

    // dV[d4] += p_ij * doVec
    const ptrDV = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrDV, regDV, constD4Idx[d4]]);
    const dvOld = b.id(); b.emit(Op.Load, [tVec4F32, dvOld, ptrDV]);
    const pDO = b.id(); b.emit(Op.VectorTimesScalar, [tVec4F32, pDO, doVec, p_ij]);
    const dvNew = b.id(); b.emit(Op.FAdd, [tVec4F32, dvNew, dvOld, pDO]);
    b.emit(Op.Store, [ptrDV, dvNew]);

    // dotDOV partial: OpDot(doVec, regV[d4])
    const ptrRegVd4 = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegVd4, regV, constD4Idx[d4]]);
    const vVec = b.id(); b.emit(Op.Load, [tVec4F32, vVec, ptrRegVd4]);
    const partialDOV = b.id(); b.emit(Op.Dot, [p.tF32, partialDOV, doVec, vVec]);
    const newDovAcc = b.id(); b.emit(Op.FAdd, [p.tF32, newDovAcc, dovAcc, partialDOV]);
    dovAcc = newDovAcc;
  }

  // dS_ij = p_ij * (dotDOV - sDpre[i])
  const ptrSDpreI = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrSDpreI, sDpre, i]);
  const dpreI = b.id(); b.emit(Op.Load, [p.tF32, dpreI, ptrSDpreI]);
  const dovMinusDi = b.id(); b.emit(Op.FSub, [p.tF32, dovMinusDi, dovAcc, dpreI]);
  const dS_ij = b.id(); b.emit(Op.FMul, [p.tF32, dS_ij, p_ij, dovMinusDi]);

  // dScore_ij with softcap derivative
  const tanhSq = b.id(); b.emit(Op.FMul, [p.tF32, tanhSq, tanhVal, tanhVal]);
  const deriv = b.id(); b.emit(Op.FSub, [p.tF32, deriv, const1f, tanhSq]);
  const dSscale = b.id(); b.emit(Op.FMul, [p.tF32, dSscale, dS_ij, scale]);
  const dScore_capped = b.id(); b.emit(Op.FMul, [p.tF32, dScore_capped, dSscale, deriv]);
  const dScore_uncapped = b.id(); b.emit(Op.FMul, [p.tF32, dScore_uncapped, dS_ij, scale]);
  const dScore_ij = b.id(); b.emit(Op.Select, [p.tF32, dScore_ij, softCapEnabled, dScore_capped, dScore_uncapped]);

  // dK accumulation using cached Q (no redundant shared memory reload)
  for (let d4 = 0; d4 < D4; d4++) {
    const ptrDK = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrDK, regDK, constD4Idx[d4]]);
    const dkOld = b.id(); b.emit(Op.Load, [tVec4F32, dkOld, ptrDK]);
    const dScoreQ = b.id(); b.emit(Op.VectorTimesScalar, [tVec4F32, dScoreQ, qCache[d4], dScore_ij]);
    const dkNew = b.id(); b.emit(Op.FAdd, [tVec4F32, dkNew, dkOld, dScoreQ]);
    b.emit(Op.Store, [ptrDK, dkNew]);
  }

  // i loop continuation
  b.emit(Op.Branch, [labelICont]);
  b.emit(Op.Label, [labelICont]);
  const nextI = b.id(); b.emit(Op.Load, [p.tU32, nextI, varI]);
  const incI = b.id(); b.emit(Op.IAdd, [p.tU32, incI, nextI, p.const1u]);
  b.emit(Op.Store, [varI, incI]);
  b.emit(Op.Branch, [labelIHead]);
  b.emit(Op.Label, [labelIMerge]);

  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  // qBlock loop continuation
  b.emit(Op.Branch, [labelLoopCont]);
  b.emit(Op.Label, [labelLoopCont]);
  const nextQBlock = b.id(); b.emit(Op.Load, [p.tU32, nextQBlock, varQBlockIdx]);
  const incQBlock = b.id(); b.emit(Op.IAdd, [p.tU32, incQBlock, nextQBlock, p.const1u]);
  b.emit(Op.Store, [varQBlockIdx, incQBlock]);
  b.emit(Op.Branch, [labelLoopHead]);
  b.emit(Op.Label, [labelLoopMerge]);

  // Write regDK, regDV to output
  const dkBase4 = b.id(); b.emit(Op.IAdd, [p.tU32, dkBase4, baseOff4, kRowD4]);
  for (let d4 = 0; d4 < D4; d4++) {
    const ptrRegDKd4 = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegDKd4, regDK, constD4Idx[d4]]);
    const regDKVec = b.id(); b.emit(Op.Load, [tVec4F32, regDKVec, ptrRegDKd4]);
    const dkIdx = b.id(); b.emit(Op.IAdd, [p.tU32, dkIdx, dkBase4, constD4Idx[d4]]);
    const ptrOutDK = b.id(); b.emit(Op.AccessChain, [bufDK.tPtrVec4, ptrOutDK, bufDK.varId, p.const0u, dkIdx]);
    b.emit(Op.Store, [ptrOutDK, regDKVec]);

    const ptrRegDVd4 = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegDVd4, regDV, constD4Idx[d4]]);
    const regDVVec = b.id(); b.emit(Op.Load, [tVec4F32, regDVVec, ptrRegDVd4]);
    const dvIdx = b.id(); b.emit(Op.IAdd, [p.tU32, dvIdx, dkBase4, constD4Idx[d4]]);
    const ptrOutDV = b.id(); b.emit(Op.AccessChain, [bufDV.tPtrVec4, ptrOutDV, bufDV.varId, p.const0u, dvIdx]);
    b.emit(Op.Store, [ptrOutDV, regDVVec]);
  }

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);
  return b.build();
}


// ── Kernel: Flash Attention Backward dQ V2 (batched j-unroll for ILP) ────────
export function kernelFlashAttentionBackwardDQV2(
  Br: number, Bc: number, D: number, BJ: number = 4
): Uint32Array {
  if (Bc % BJ !== 0) throw new Error(`Bc=${Bc} must be divisible by BJ=${BJ}`);
  const D4 = D >>> 2;
  const b = new SpirVBuilder();
  const p = preamble(b, Br, 1, 1);

  const tVec4F32 = b.id(); b.typeVector(tVec4F32, p.tF32, 4);
  const bufQ    = declareStorageBufferVec4(b, tVec4F32, 0, 0, true);
  const bufK    = declareStorageBufferVec4(b, tVec4F32, 0, 1, true);
  const bufV    = declareStorageBufferVec4(b, tVec4F32, 0, 2, true);
  const bufDO   = declareStorageBufferVec4(b, tVec4F32, 0, 3, true);
  const bufLSE  = declareStorageBuffer(b, p.tF32, p.tU32, 0, 4, true);
  const bufDpre = declareStorageBuffer(b, p.tF32, p.tU32, 0, 5, true);
  const bufDQ   = declareStorageBufferVec4(b, tVec4F32, 0, 6, false, true);
  const pc = declareParamsPushConstant(b, p.tF32, 4);

  const constBr = b.id(); b.constant(p.tU32, constBr, Br);
  const constBc = b.id(); b.constant(p.tU32, constBc, Bc);
  const constD4 = b.id(); b.constant(p.tU32, constD4, D4);
  const constNegInf = b.id(); b.constant(p.tF32, constNegInf, 0xFF800000);
  const const1f = b.id(); b.constantF32(p.tF32, const1f, 1.0);
  const vec4Zero = b.id(); b.constantNull(tVec4F32, vec4Zero);

  const constD4Idx: number[] = [];
  for (let d4 = 0; d4 < D4; d4++) {
    const cd4 = b.id(); b.constant(p.tU32, cd4, d4); constD4Idx.push(cd4);
  }
  const constJIdx: number[] = []; const constJD4: number[] = [];
  for (let j = 0; j < Bc; j++) {
    constJIdx.push(b.id()); b.constant(p.tU32, constJIdx[j], j);
    constJD4.push(b.id()); b.constant(p.tU32, constJD4[j], j * D4);
  }

  const constSharedSize = b.id(); b.constant(p.tU32, constSharedSize, Bc * D4);
  const tArrayShared = b.id(); b.typeArray(tArrayShared, tVec4F32, constSharedSize);
  const tPtrSharedArr = b.id(); b.typePointer(tPtrSharedArr, StorageClass.Workgroup, tArrayShared);
  const tPtrSharedVec4 = b.id(); b.typePointer(tPtrSharedVec4, StorageClass.Workgroup, tVec4F32);
  const sK = b.id(); b.variable(tPtrSharedArr, sK, StorageClass.Workgroup);
  const sV = b.id(); b.variable(tPtrSharedArr, sV, StorageClass.Workgroup);

  const constD4Arr = b.id(); b.constant(p.tU32, constD4Arr, D4);
  const tArrayD4 = b.id(); b.typeArray(tArrayD4, tVec4F32, constD4Arr);
  const tPtrFnArr = b.id(); b.typePointer(tPtrFnArr, StorageClass.Function, tArrayD4);
  const tPtrFnVec4 = b.id(); b.typePointer(tPtrFnVec4, StorageClass.Function, tVec4F32);
  const tPtrFnU32 = b.id(); b.typePointer(tPtrFnU32, StorageClass.Function, p.tU32);

  const tPtrInputVec3 = b.id(); b.typePointer(tPtrInputVec3, StorageClass.Input, p.tVec3U32);
  const vWorkgroupId = b.id(); b.variable(tPtrInputVec3, vWorkgroupId, StorageClass.Input);
  b.addDecorate(vWorkgroupId, Decoration.BuiltIn, BuiltIn.WorkgroupId);
  const vLocalId = b.id(); b.variable(tPtrInputVec3, vLocalId, StorageClass.Input);
  b.addDecorate(vLocalId, Decoration.BuiltIn, BuiltIn.LocalInvocationId);

  const scopeWg = b.id(); b.constant(p.tU32, scopeWg, Scope.Workgroup);
  const semAcqRelWg = b.id(); b.constant(p.tU32, semAcqRelWg, MemorySemantics.AcquireRelease | MemorySemantics.WorkgroupMemory);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId, vWorkgroupId, vLocalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, Br, 1, 1);
  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  const labelEntry = b.id(); b.emit(Op.Label, [labelEntry]);

  const regQ = b.id(); b.emit(Op.Variable, [tPtrFnArr, regQ, StorageClass.Function]);
  const regDO = b.id(); b.emit(Op.Variable, [tPtrFnArr, regDO, StorageClass.Function]);
  const regDQ = b.id(); b.emit(Op.Variable, [tPtrFnArr, regDQ, StorageClass.Function]);
  const varKBlockIdx = b.id(); b.emit(Op.Variable, [tPtrFnU32, varKBlockIdx, StorageClass.Function]);

  const lidVec = b.id(); b.emit(Op.Load, [p.tVec3U32, lidVec, vLocalId]);
  const threadIdx = b.id(); b.emit(Op.CompositeExtract, [p.tU32, threadIdx, lidVec, 0]);
  const wgIdVec = b.id(); b.emit(Op.Load, [p.tVec3U32, wgIdVec, vWorkgroupId]);
  const qBlockIdx = b.id(); b.emit(Op.CompositeExtract, [p.tU32, qBlockIdx, wgIdVec, 0]);
  const bhIdx = b.id(); b.emit(Op.CompositeExtract, [p.tU32, bhIdx, wgIdVec, 1]);

  const qBlockOff = b.id(); b.emit(Op.IMul, [p.tU32, qBlockOff, qBlockIdx, constBr]);
  const qRow = b.id(); b.emit(Op.IAdd, [p.tU32, qRow, qBlockOff, threadIdx]);

  const ptrTpc = b.id(); b.emit(Op.AccessChain, [pc.tPtrF32, ptrTpc, pc.varId, p.const0u]);
  const TF = b.id(); b.emit(Op.Load, [p.tF32, TF, ptrTpc]);
  const T = b.id(); b.emit(Op.ConvertFToU, [p.tU32, T, TF]);
  const ptrScale = b.id(); b.emit(Op.AccessChain, [pc.tPtrF32, ptrScale, pc.varId, p.const1u]);
  const scale = b.id(); b.emit(Op.Load, [p.tF32, scale, ptrScale]);
  const ptrSoftCap = b.id(); b.emit(Op.AccessChain, [pc.tPtrF32, ptrSoftCap, pc.varId, p.const2u]);
  const softCapValue = b.id(); b.emit(Op.Load, [p.tF32, softCapValue, ptrSoftCap]);
  const softCapEnabled = b.id(); b.emit(Op.FOrdGreaterThan, [p.tBool, softCapEnabled, softCapValue, p.const0f]);
  const invSoftCap = b.id(); b.emit(Op.FDiv, [p.tF32, invSoftCap, const1f, softCapValue]);

  const TD4 = b.id(); b.emit(Op.IMul, [p.tU32, TD4, T, constD4]);
  const baseOff4 = b.id(); b.emit(Op.IMul, [p.tU32, baseOff4, bhIdx, TD4]);
  const lseBaseOff = b.id(); b.emit(Op.IMul, [p.tU32, lseBaseOff, bhIdx, T]);

  const qRowOob = b.id(); b.emit(Op.UGreaterThanEqual, [p.tBool, qRowOob, qRow, T]);
  const labelMain = b.id(); const labelEnd = b.id();
  b.emit(Op.SelectionMerge, [labelEnd, 0]);
  b.emit(Op.BranchConditional, [qRowOob, labelEnd, labelMain]);
  b.emit(Op.Label, [labelMain]);

  const qRowD4 = b.id(); b.emit(Op.IMul, [p.tU32, qRowD4, qRow, constD4]);
  const qBase4 = b.id(); b.emit(Op.IAdd, [p.tU32, qBase4, baseOff4, qRowD4]);

  for (let d4 = 0; d4 < D4; d4++) {
    const qIdx = b.id(); b.emit(Op.IAdd, [p.tU32, qIdx, qBase4, constD4Idx[d4]]);
    const ptrQElem = b.id(); b.emit(Op.AccessChain, [bufQ.tPtrVec4, ptrQElem, bufQ.varId, p.const0u, qIdx]);
    const qVec = b.id(); b.emit(Op.Load, [tVec4F32, qVec, ptrQElem]);
    const qScaled = b.id(); b.emit(Op.VectorTimesScalar, [tVec4F32, qScaled, qVec, scale]);
    const ptrRegQ = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegQ, regQ, constD4Idx[d4]]);
    b.emit(Op.Store, [ptrRegQ, qScaled]);
  }
  for (let d4 = 0; d4 < D4; d4++) {
    const doIdx = b.id(); b.emit(Op.IAdd, [p.tU32, doIdx, qBase4, constD4Idx[d4]]);
    const ptrDOElem = b.id(); b.emit(Op.AccessChain, [bufDO.tPtrVec4, ptrDOElem, bufDO.varId, p.const0u, doIdx]);
    const doVec = b.id(); b.emit(Op.Load, [tVec4F32, doVec, ptrDOElem]);
    const ptrRegDO = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegDO, regDO, constD4Idx[d4]]);
    b.emit(Op.Store, [ptrRegDO, doVec]);
  }

  const lseQRowIdx = b.id(); b.emit(Op.IAdd, [p.tU32, lseQRowIdx, lseBaseOff, qRow]);
  const ptrLSEi = b.id(); b.emit(Op.AccessChain, [bufLSE.tPtrF32, ptrLSEi, bufLSE.varId, p.const0u, lseQRowIdx]);
  const lse_i = b.id(); b.emit(Op.Load, [p.tF32, lse_i, ptrLSEi]);
  const ptrDi = b.id(); b.emit(Op.AccessChain, [bufDpre.tPtrF32, ptrDi, bufDpre.varId, p.const0u, lseQRowIdx]);
  const Di = b.id(); b.emit(Op.Load, [p.tF32, Di, ptrDi]);

  for (let d4 = 0; d4 < D4; d4++) {
    const ptrRegDQ = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegDQ, regDQ, constD4Idx[d4]]);
    b.emit(Op.Store, [ptrRegDQ, vec4Zero]);
  }

  const kBlocksPerQBlock = Br / Bc;
  const qPlus1 = b.id(); b.emit(Op.IAdd, [p.tU32, qPlus1, qBlockIdx, p.const1u]);
  let effectiveKBlocks: number;
  if (kBlocksPerQBlock === 1) {
    effectiveKBlocks = qPlus1;
  } else {
    const constKBPQB = b.id(); b.constant(p.tU32, constKBPQB, kBlocksPerQBlock);
    effectiveKBlocks = b.id();
    b.emit(Op.IMul, [p.tU32, effectiveKBlocks, qPlus1, constKBPQB]);
  }

  // ── Outer kBlock loop ──────────────────────────────────────────────────
  b.emit(Op.Store, [varKBlockIdx, p.const0u]);
  const labelLoopHead = b.id(); const labelLoopBody = b.id();
  const labelLoopMerge = b.id(); const labelLoopCont = b.id();
  b.emit(Op.Branch, [labelLoopHead]);
  b.emit(Op.Label, [labelLoopHead]);
  const kBlockIdx = b.id(); b.emit(Op.Load, [p.tU32, kBlockIdx, varKBlockIdx]);
  const loopCmp = b.id(); b.emit(Op.ULessThan, [p.tBool, loopCmp, kBlockIdx, effectiveKBlocks]);
  b.emit(Op.LoopMerge, [labelLoopMerge, labelLoopCont, 0]);
  b.emit(Op.BranchConditional, [loopCmp, labelLoopBody, labelLoopMerge]);
  b.emit(Op.Label, [labelLoopBody]);

  // Cooperative load K, V into shared
  const kBlockBase = b.id(); b.emit(Op.IMul, [p.tU32, kBlockBase, kBlockIdx, constBc]);
  const kBlockBaseTD4 = b.id(); b.emit(Op.IMul, [p.tU32, kBlockBaseTD4, kBlockBase, constD4]);
  const kBlockGlobal4 = b.id(); b.emit(Op.IAdd, [p.tU32, kBlockGlobal4, baseOff4, kBlockBaseTD4]);
  const totalLoadElems = Bc * D4;
  const elemsPerThread = Math.ceil(totalLoadElems / Br);
  const log2D4 = Math.log2(D4);
  const constLog2D4 = b.id(); b.constant(p.tU32, constLog2D4, log2D4);

  for (let pass = 0; pass < elemsPerThread; pass++) {
    let linearIdx: number;
    if (pass === 0) { linearIdx = threadIdx; }
    else {
      const constPassOff = b.id(); b.constant(p.tU32, constPassOff, pass * Br);
      linearIdx = b.id(); b.emit(Op.IAdd, [p.tU32, linearIdx, threadIdx, constPassOff]);
    }
    const needCheck = (pass + 1) * Br > totalLoadElems;
    let labelPassEnd: number | undefined;
    if (needCheck) {
      const constTotalElems = b.id(); b.constant(p.tU32, constTotalElems, totalLoadElems);
      const inRange = b.id(); b.emit(Op.ULessThan, [p.tBool, inRange, linearIdx, constTotalElems]);
      const labelPassBody = b.id(); labelPassEnd = b.id();
      b.emit(Op.SelectionMerge, [labelPassEnd, 0]);
      b.emit(Op.BranchConditional, [inRange, labelPassBody, labelPassEnd]);
      b.emit(Op.Label, [labelPassBody]);
    }
    const row = b.id(); b.emit(Op.ShiftRightLogical, [p.tU32, row, linearIdx, constLog2D4]);
    const kRowL = b.id(); b.emit(Op.IAdd, [p.tU32, kRowL, kBlockBase, row]);
    const kRowInBounds = b.id(); b.emit(Op.ULessThan, [p.tBool, kRowInBounds, kRowL, T]);
    const inBoundsF = b.id(); b.emit(Op.Select, [p.tF32, inBoundsF, kRowInBounds, const1f, p.const0f]);
    const gIdx = b.id(); b.emit(Op.IAdd, [p.tU32, gIdx, kBlockGlobal4, linearIdx]);
    const ptrKElem = b.id(); b.emit(Op.AccessChain, [bufK.tPtrVec4, ptrKElem, bufK.varId, p.const0u, gIdx]);
    const kRaw = b.id(); b.emit(Op.Load, [tVec4F32, kRaw, ptrKElem]);
    const kVal = b.id(); b.emit(Op.VectorTimesScalar, [tVec4F32, kVal, kRaw, inBoundsF]);
    const ptrSK = b.id(); b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSK, sK, linearIdx]);
    b.emit(Op.Store, [ptrSK, kVal]);
    const ptrVElem = b.id(); b.emit(Op.AccessChain, [bufV.tPtrVec4, ptrVElem, bufV.varId, p.const0u, gIdx]);
    const vRaw = b.id(); b.emit(Op.Load, [tVec4F32, vRaw, ptrVElem]);
    const vVal = b.id(); b.emit(Op.VectorTimesScalar, [tVec4F32, vVal, vRaw, inBoundsF]);
    const ptrSV = b.id(); b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSV, sV, linearIdx]);
    b.emit(Op.Store, [ptrSV, vVal]);
    if (needCheck) { b.emit(Op.Branch, [labelPassEnd!]); b.emit(Op.Label, [labelPassEnd!]); }
  }
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  // ── Compile-time unrolled j-loop in batches of BJ ─────────────────────
  for (let jBatch = 0; jBatch < Bc; jBatch += BJ) {
    const dots: number[] = []; const tanhVals: number[] = [];
    for (let jj = 0; jj < BJ; jj++) {
      const j = jBatch + jj;
      const kPos = b.id(); b.emit(Op.IAdd, [p.tU32, kPos, kBlockBase, constJIdx[j]]);
      const ptrRegQ0 = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegQ0, regQ, constD4Idx[0]]);
      const qVec0 = b.id(); b.emit(Op.Load, [tVec4F32, qVec0, ptrRegQ0]);
      const ptrSK0 = b.id(); b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSK0, sK, constJD4[j]]);
      const kVec0 = b.id(); b.emit(Op.Load, [tVec4F32, kVec0, ptrSK0]);
      let dotAcc = b.id(); b.emit(Op.Dot, [p.tF32, dotAcc, qVec0, kVec0]);
      for (let d4 = 1; d4 < D4; d4++) {
        const ptrRegQd4 = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegQd4, regQ, constD4Idx[d4]]);
        const qVecD4 = b.id(); b.emit(Op.Load, [tVec4F32, qVecD4, ptrRegQd4]);
        const sKIdx = b.id(); b.emit(Op.IAdd, [p.tU32, sKIdx, constJD4[j], constD4Idx[d4]]);
        const ptrSKd4 = b.id(); b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSKd4, sK, sKIdx]);
        const kVecD4 = b.id(); b.emit(Op.Load, [tVec4F32, kVecD4, ptrSKd4]);
        const partial = b.id(); b.emit(Op.Dot, [p.tF32, partial, qVecD4, kVecD4]);
        const newDot = b.id(); b.emit(Op.FAdd, [p.tF32, newDot, dotAcc, partial]);
        dotAcc = newDot;
      }
      const dotDivCap = b.id(); b.emit(Op.FMul, [p.tF32, dotDivCap, dotAcc, invSoftCap]);
      const tanhVal = b.id(); b.emit(Op.ExtInst, [p.tF32, tanhVal, p.glslStd, GLSLstd450.Tanh, dotDivCap]);
      const dotCapped = b.id(); b.emit(Op.FMul, [p.tF32, dotCapped, tanhVal, softCapValue]);
      const dotAfterCap = b.id(); b.emit(Op.Select, [p.tF32, dotAfterCap, softCapEnabled, dotCapped, dotAcc]);
      tanhVals.push(tanhVal);
      const oob = b.id(); b.emit(Op.UGreaterThanEqual, [p.tBool, oob, kPos, T]);
      const causal = b.id(); b.emit(Op.ULessThan, [p.tBool, causal, qRow, kPos]);
      const masked = b.id(); b.emit(Op.LogicalOr, [p.tBool, masked, oob, causal]);
      const dot = b.id(); b.emit(Op.Select, [p.tF32, dot, masked, constNegInf, dotAfterCap]);
      dots.push(dot);
    }
    const pVals: number[] = [];
    for (let jj = 0; jj < BJ; jj++) {
      const sub = b.id(); b.emit(Op.FSub, [p.tF32, sub, dots[jj], lse_i]);
      const p_ij = b.id(); b.emit(Op.ExtInst, [p.tF32, p_ij, p.glslStd, GLSLstd450.Exp, sub]);
      pVals.push(p_ij);
    }
    const dovAccs: number[] = [];
    for (let jj = 0; jj < BJ; jj++) {
      const j = jBatch + jj;
      const ptrDO0 = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrDO0, regDO, constD4Idx[0]]);
      const doVec0 = b.id(); b.emit(Op.Load, [tVec4F32, doVec0, ptrDO0]);
      const ptrSV0 = b.id(); b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSV0, sV, constJD4[j]]);
      const vVec0 = b.id(); b.emit(Op.Load, [tVec4F32, vVec0, ptrSV0]);
      let dovAcc = b.id(); b.emit(Op.Dot, [p.tF32, dovAcc, doVec0, vVec0]);
      for (let d4 = 1; d4 < D4; d4++) {
        const ptrDOd4 = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrDOd4, regDO, constD4Idx[d4]]);
        const doVecD4 = b.id(); b.emit(Op.Load, [tVec4F32, doVecD4, ptrDOd4]);
        const sVIdx = b.id(); b.emit(Op.IAdd, [p.tU32, sVIdx, constJD4[j], constD4Idx[d4]]);
        const ptrSVd4 = b.id(); b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSVd4, sV, sVIdx]);
        const vVecD4 = b.id(); b.emit(Op.Load, [tVec4F32, vVecD4, ptrSVd4]);
        const partialDOV = b.id(); b.emit(Op.Dot, [p.tF32, partialDOV, doVecD4, vVecD4]);
        const newDov = b.id(); b.emit(Op.FAdd, [p.tF32, newDov, dovAcc, partialDOV]);
        dovAcc = newDov;
      }
      dovAccs.push(dovAcc);
    }
    const dScores: number[] = [];
    for (let jj = 0; jj < BJ; jj++) {
      const sub2 = b.id(); b.emit(Op.FSub, [p.tF32, sub2, dovAccs[jj], Di]);
      const dS = b.id(); b.emit(Op.FMul, [p.tF32, dS, pVals[jj], sub2]);
      const tSq = b.id(); b.emit(Op.FMul, [p.tF32, tSq, tanhVals[jj], tanhVals[jj]]);
      const deriv = b.id(); b.emit(Op.FSub, [p.tF32, deriv, const1f, tSq]);
      const dSs = b.id(); b.emit(Op.FMul, [p.tF32, dSs, dS, scale]);
      const dSc = b.id(); b.emit(Op.FMul, [p.tF32, dSc, dSs, deriv]);
      const dSu = b.id(); b.emit(Op.FMul, [p.tF32, dSu, dS, scale]);
      const dScore = b.id(); b.emit(Op.Select, [p.tF32, dScore, softCapEnabled, dSc, dSu]);
      dScores.push(dScore);
    }
    for (let d4 = 0; d4 < D4; d4++) {
      const ptrDQ = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrDQ, regDQ, constD4Idx[d4]]);
      let dqVal = b.id(); b.emit(Op.Load, [tVec4F32, dqVal, ptrDQ]);
      for (let jj = 0; jj < BJ; jj++) {
        const j = jBatch + jj;
        const sKIdx = b.id(); b.emit(Op.IAdd, [p.tU32, sKIdx, constJD4[j], constD4Idx[d4]]);
        const ptrSKdq = b.id(); b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSKdq, sK, sKIdx]);
        const kVecDQ = b.id(); b.emit(Op.Load, [tVec4F32, kVecDQ, ptrSKdq]);
        const contrib = b.id(); b.emit(Op.VectorTimesScalar, [tVec4F32, contrib, kVecDQ, dScores[jj]]);
        const dqNew = b.id(); b.emit(Op.FAdd, [tVec4F32, dqNew, dqVal, contrib]);
        dqVal = dqNew;
      }
      b.emit(Op.Store, [ptrDQ, dqVal]);
    }
  }

  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);
  b.emit(Op.Branch, [labelLoopCont]);
  b.emit(Op.Label, [labelLoopCont]);
  const nextKBlock = b.id(); b.emit(Op.Load, [p.tU32, nextKBlock, varKBlockIdx]);
  const incKBlock = b.id(); b.emit(Op.IAdd, [p.tU32, incKBlock, nextKBlock, p.const1u]);
  b.emit(Op.Store, [varKBlockIdx, incKBlock]);
  b.emit(Op.Branch, [labelLoopHead]);
  b.emit(Op.Label, [labelLoopMerge]);

  const dqBase4 = b.id(); b.emit(Op.IAdd, [p.tU32, dqBase4, baseOff4, qRowD4]);
  for (let d4 = 0; d4 < D4; d4++) {
    const ptrRegDQd4 = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegDQd4, regDQ, constD4Idx[d4]]);
    const regDQVec = b.id(); b.emit(Op.Load, [tVec4F32, regDQVec, ptrRegDQd4]);
    const dqIdx = b.id(); b.emit(Op.IAdd, [p.tU32, dqIdx, dqBase4, constD4Idx[d4]]);
    const ptrOut = b.id(); b.emit(Op.AccessChain, [bufDQ.tPtrVec4, ptrOut, bufDQ.varId, p.const0u, dqIdx]);
    b.emit(Op.Store, [ptrOut, regDQVec]);
  }

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);
  return b.build();
}


// ── Kernel: Flash Attention Backward dKV V2 (batched i-unroll for ILP) ───────
export function kernelFlashAttentionBackwardDKVV2(
  Br: number, Bc: number, D: number, BI: number = 4
): Uint32Array {
  if (Br % BI !== 0) throw new Error(`Br=${Br} must be divisible by BI=${BI}`);
  const D4 = D >>> 2;
  const b = new SpirVBuilder();
  const p = preamble(b, Bc, 1, 1);

  const tVec4F32 = b.id(); b.typeVector(tVec4F32, p.tF32, 4);
  const bufQ    = declareStorageBufferVec4(b, tVec4F32, 0, 0, true);
  const bufK    = declareStorageBufferVec4(b, tVec4F32, 0, 1, true);
  const bufV    = declareStorageBufferVec4(b, tVec4F32, 0, 2, true);
  const bufDO   = declareStorageBufferVec4(b, tVec4F32, 0, 3, true);
  const bufLSE  = declareStorageBuffer(b, p.tF32, p.tU32, 0, 4, true);
  const bufDpre = declareStorageBuffer(b, p.tF32, p.tU32, 0, 5, true);
  const bufDK   = declareStorageBufferVec4(b, tVec4F32, 0, 6, false, true);
  const bufDV   = declareStorageBufferVec4(b, tVec4F32, 0, 7, false, true);
  const pc = declareParamsPushConstant(b, p.tF32, 4);

  const constBr = b.id(); b.constant(p.tU32, constBr, Br);
  const constBc = b.id(); b.constant(p.tU32, constBc, Bc);
  const constD4 = b.id(); b.constant(p.tU32, constD4, D4);
  const constBrMinus1 = b.id(); b.constant(p.tU32, constBrMinus1, Br - 1);
  const constNegInf = b.id(); b.constant(p.tF32, constNegInf, 0xFF800000);
  const const1f = b.id(); b.constantF32(p.tF32, const1f, 1.0);
  const vec4Zero = b.id(); b.constantNull(tVec4F32, vec4Zero);

  const constD4Idx: number[] = [];
  for (let d4 = 0; d4 < D4; d4++) {
    const cd4 = b.id(); b.constant(p.tU32, cd4, d4); constD4Idx.push(cd4);
  }
  const constIIdx: number[] = []; const constID4: number[] = [];
  for (let i = 0; i < Br; i++) {
    constIIdx.push(b.id()); b.constant(p.tU32, constIIdx[i], i);
    constID4.push(b.id()); b.constant(p.tU32, constID4[i], i * D4);
  }

  const constSharedVec4Size = b.id(); b.constant(p.tU32, constSharedVec4Size, Br * D4);
  const tArraySharedVec4 = b.id(); b.typeArray(tArraySharedVec4, tVec4F32, constSharedVec4Size);
  const tPtrSharedArrVec4 = b.id(); b.typePointer(tPtrSharedArrVec4, StorageClass.Workgroup, tArraySharedVec4);
  const tPtrSharedVec4 = b.id(); b.typePointer(tPtrSharedVec4, StorageClass.Workgroup, tVec4F32);
  const sQ = b.id(); b.variable(tPtrSharedArrVec4, sQ, StorageClass.Workgroup);
  const sDO = b.id(); b.variable(tPtrSharedArrVec4, sDO, StorageClass.Workgroup);

  const constSharedSmallSize = b.id(); b.constant(p.tU32, constSharedSmallSize, Br);
  const tArraySharedSmall = b.id(); b.typeArray(tArraySharedSmall, p.tF32, constSharedSmallSize);
  const tPtrSharedSmallArr = b.id(); b.typePointer(tPtrSharedSmallArr, StorageClass.Workgroup, tArraySharedSmall);
  const tPtrSharedF32 = b.id(); b.typePointer(tPtrSharedF32, StorageClass.Workgroup, p.tF32);
  const sLSE = b.id(); b.variable(tPtrSharedSmallArr, sLSE, StorageClass.Workgroup);
  const sDpre = b.id(); b.variable(tPtrSharedSmallArr, sDpre, StorageClass.Workgroup);

  const constD4Arr = b.id(); b.constant(p.tU32, constD4Arr, D4);
  const tArrayD4 = b.id(); b.typeArray(tArrayD4, tVec4F32, constD4Arr);
  const tPtrFnArr = b.id(); b.typePointer(tPtrFnArr, StorageClass.Function, tArrayD4);
  const tPtrFnVec4 = b.id(); b.typePointer(tPtrFnVec4, StorageClass.Function, tVec4F32);
  const tPtrFnU32 = b.id(); b.typePointer(tPtrFnU32, StorageClass.Function, p.tU32);

  const tPtrInputVec3 = b.id(); b.typePointer(tPtrInputVec3, StorageClass.Input, p.tVec3U32);
  const vWorkgroupId = b.id(); b.variable(tPtrInputVec3, vWorkgroupId, StorageClass.Input);
  b.addDecorate(vWorkgroupId, Decoration.BuiltIn, BuiltIn.WorkgroupId);
  const vLocalId = b.id(); b.variable(tPtrInputVec3, vLocalId, StorageClass.Input);
  b.addDecorate(vLocalId, Decoration.BuiltIn, BuiltIn.LocalInvocationId);

  const scopeWg = b.id(); b.constant(p.tU32, scopeWg, Scope.Workgroup);
  const semAcqRelWg = b.id(); b.constant(p.tU32, semAcqRelWg, MemorySemantics.AcquireRelease | MemorySemantics.WorkgroupMemory);

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId, vWorkgroupId, vLocalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, Bc, 1, 1);
  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  const labelEntry = b.id(); b.emit(Op.Label, [labelEntry]);

  const regK = b.id(); b.emit(Op.Variable, [tPtrFnArr, regK, StorageClass.Function]);
  const regV = b.id(); b.emit(Op.Variable, [tPtrFnArr, regV, StorageClass.Function]);
  const regDK = b.id(); b.emit(Op.Variable, [tPtrFnArr, regDK, StorageClass.Function]);
  const regDV = b.id(); b.emit(Op.Variable, [tPtrFnArr, regDV, StorageClass.Function]);
  const varQBlockIdx = b.id(); b.emit(Op.Variable, [tPtrFnU32, varQBlockIdx, StorageClass.Function]);

  const lidVec = b.id(); b.emit(Op.Load, [p.tVec3U32, lidVec, vLocalId]);
  const threadIdx = b.id(); b.emit(Op.CompositeExtract, [p.tU32, threadIdx, lidVec, 0]);
  const wgIdVec = b.id(); b.emit(Op.Load, [p.tVec3U32, wgIdVec, vWorkgroupId]);
  const kBlockIdx = b.id(); b.emit(Op.CompositeExtract, [p.tU32, kBlockIdx, wgIdVec, 0]);
  const bhIdx = b.id(); b.emit(Op.CompositeExtract, [p.tU32, bhIdx, wgIdVec, 1]);

  const kBlockOff = b.id(); b.emit(Op.IMul, [p.tU32, kBlockOff, kBlockIdx, constBc]);
  const kRow = b.id(); b.emit(Op.IAdd, [p.tU32, kRow, kBlockOff, threadIdx]);

  const ptrTpc = b.id(); b.emit(Op.AccessChain, [pc.tPtrF32, ptrTpc, pc.varId, p.const0u]);
  const TF = b.id(); b.emit(Op.Load, [p.tF32, TF, ptrTpc]);
  const T = b.id(); b.emit(Op.ConvertFToU, [p.tU32, T, TF]);
  const ptrScale = b.id(); b.emit(Op.AccessChain, [pc.tPtrF32, ptrScale, pc.varId, p.const1u]);
  const scale = b.id(); b.emit(Op.Load, [p.tF32, scale, ptrScale]);
  const ptrSoftCap = b.id(); b.emit(Op.AccessChain, [pc.tPtrF32, ptrSoftCap, pc.varId, p.const2u]);
  const softCapValue = b.id(); b.emit(Op.Load, [p.tF32, softCapValue, ptrSoftCap]);
  const softCapEnabled = b.id(); b.emit(Op.FOrdGreaterThan, [p.tBool, softCapEnabled, softCapValue, p.const0f]);
  const invSoftCap = b.id(); b.emit(Op.FDiv, [p.tF32, invSoftCap, const1f, softCapValue]);

  const TD4 = b.id(); b.emit(Op.IMul, [p.tU32, TD4, T, constD4]);
  const baseOff4 = b.id(); b.emit(Op.IMul, [p.tU32, baseOff4, bhIdx, TD4]);
  const lseBaseOff = b.id(); b.emit(Op.IMul, [p.tU32, lseBaseOff, bhIdx, T]);

  const kRowOob = b.id(); b.emit(Op.UGreaterThanEqual, [p.tBool, kRowOob, kRow, T]);
  const labelMain = b.id(); const labelEnd = b.id();
  b.emit(Op.SelectionMerge, [labelEnd, 0]);
  b.emit(Op.BranchConditional, [kRowOob, labelEnd, labelMain]);
  b.emit(Op.Label, [labelMain]);

  const kRowD4 = b.id(); b.emit(Op.IMul, [p.tU32, kRowD4, kRow, constD4]);
  const kBase4 = b.id(); b.emit(Op.IAdd, [p.tU32, kBase4, baseOff4, kRowD4]);

  for (let d4 = 0; d4 < D4; d4++) {
    const kIdx = b.id(); b.emit(Op.IAdd, [p.tU32, kIdx, kBase4, constD4Idx[d4]]);
    const ptrKElem = b.id(); b.emit(Op.AccessChain, [bufK.tPtrVec4, ptrKElem, bufK.varId, p.const0u, kIdx]);
    const kVec = b.id(); b.emit(Op.Load, [tVec4F32, kVec, ptrKElem]);
    const kScaled = b.id(); b.emit(Op.VectorTimesScalar, [tVec4F32, kScaled, kVec, scale]);
    const ptrRegK = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegK, regK, constD4Idx[d4]]);
    b.emit(Op.Store, [ptrRegK, kScaled]);
  }
  for (let d4 = 0; d4 < D4; d4++) {
    const vIdx = b.id(); b.emit(Op.IAdd, [p.tU32, vIdx, kBase4, constD4Idx[d4]]);
    const ptrVElem = b.id(); b.emit(Op.AccessChain, [bufV.tPtrVec4, ptrVElem, bufV.varId, p.const0u, vIdx]);
    const vVec = b.id(); b.emit(Op.Load, [tVec4F32, vVec, ptrVElem]);
    const ptrRegV = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegV, regV, constD4Idx[d4]]);
    b.emit(Op.Store, [ptrRegV, vVec]);
  }
  for (let d4 = 0; d4 < D4; d4++) {
    const ptrRegDK = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegDK, regDK, constD4Idx[d4]]);
    b.emit(Op.Store, [ptrRegDK, vec4Zero]);
    const ptrRegDV = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegDV, regDV, constD4Idx[d4]]);
    b.emit(Op.Store, [ptrRegDV, vec4Zero]);
  }

  const TplusBrm1 = b.id(); b.emit(Op.IAdd, [p.tU32, TplusBrm1, T, constBrMinus1]);
  const numQBlocks = b.id(); b.emit(Op.UDiv, [p.tU32, numQBlocks, TplusBrm1, constBr]);

  // ── Outer qBlock loop ─────────────────────────────────────────────────
  b.emit(Op.Store, [varQBlockIdx, kBlockIdx]);
  const labelLoopHead = b.id(); const labelLoopBody = b.id();
  const labelLoopMerge = b.id(); const labelLoopCont = b.id();
  b.emit(Op.Branch, [labelLoopHead]);
  b.emit(Op.Label, [labelLoopHead]);
  const qBlockIdx = b.id(); b.emit(Op.Load, [p.tU32, qBlockIdx, varQBlockIdx]);
  const loopCmp = b.id(); b.emit(Op.ULessThan, [p.tBool, loopCmp, qBlockIdx, numQBlocks]);
  b.emit(Op.LoopMerge, [labelLoopMerge, labelLoopCont, 0]);
  b.emit(Op.BranchConditional, [loopCmp, labelLoopBody, labelLoopMerge]);
  b.emit(Op.Label, [labelLoopBody]);

  // Load Q, dO, LSE, Dpre into shared (one row per thread)
  const qBlockBase = b.id(); b.emit(Op.IMul, [p.tU32, qBlockBase, qBlockIdx, constBr]);
  const qRow = b.id(); b.emit(Op.IAdd, [p.tU32, qRow, qBlockBase, threadIdx]);
  const qRowInBounds = b.id(); b.emit(Op.ULessThan, [p.tBool, qRowInBounds, qRow, T]);
  const inBoundsF = b.id(); b.emit(Op.Select, [p.tF32, inBoundsF, qRowInBounds, const1f, p.const0f]);
  const qRowD4 = b.id(); b.emit(Op.IMul, [p.tU32, qRowD4, qRow, constD4]);
  const qGlobalBase4 = b.id(); b.emit(Op.IAdd, [p.tU32, qGlobalBase4, baseOff4, qRowD4]);
  const sharedRowOff4 = b.id(); b.emit(Op.IMul, [p.tU32, sharedRowOff4, threadIdx, constD4]);

  for (let d4 = 0; d4 < D4; d4++) {
    const gIdx = b.id(); b.emit(Op.IAdd, [p.tU32, gIdx, qGlobalBase4, constD4Idx[d4]]);
    const ptrQElem = b.id(); b.emit(Op.AccessChain, [bufQ.tPtrVec4, ptrQElem, bufQ.varId, p.const0u, gIdx]);
    const qRaw = b.id(); b.emit(Op.Load, [tVec4F32, qRaw, ptrQElem]);
    const qVal = b.id(); b.emit(Op.VectorTimesScalar, [tVec4F32, qVal, qRaw, inBoundsF]);
    const sQIdx = b.id(); b.emit(Op.IAdd, [p.tU32, sQIdx, sharedRowOff4, constD4Idx[d4]]);
    const ptrSQ = b.id(); b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSQ, sQ, sQIdx]);
    b.emit(Op.Store, [ptrSQ, qVal]);
    const ptrDOElem = b.id(); b.emit(Op.AccessChain, [bufDO.tPtrVec4, ptrDOElem, bufDO.varId, p.const0u, gIdx]);
    const doRaw = b.id(); b.emit(Op.Load, [tVec4F32, doRaw, ptrDOElem]);
    const doVal = b.id(); b.emit(Op.VectorTimesScalar, [tVec4F32, doVal, doRaw, inBoundsF]);
    const ptrSDO = b.id(); b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSDO, sDO, sQIdx]);
    b.emit(Op.Store, [ptrSDO, doVal]);
  }
  const lseQRowIdx = b.id(); b.emit(Op.IAdd, [p.tU32, lseQRowIdx, lseBaseOff, qRow]);
  const ptrLSEi = b.id(); b.emit(Op.AccessChain, [bufLSE.tPtrF32, ptrLSEi, bufLSE.varId, p.const0u, lseQRowIdx]);
  const lseRaw = b.id(); b.emit(Op.Load, [p.tF32, lseRaw, ptrLSEi]);
  const lseVal = b.id(); b.emit(Op.Select, [p.tF32, lseVal, qRowInBounds, lseRaw, p.const0f]);
  const ptrSLSE = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrSLSE, sLSE, threadIdx]);
  b.emit(Op.Store, [ptrSLSE, lseVal]);
  const ptrDpreI = b.id(); b.emit(Op.AccessChain, [bufDpre.tPtrF32, ptrDpreI, bufDpre.varId, p.const0u, lseQRowIdx]);
  const dpreRaw = b.id(); b.emit(Op.Load, [p.tF32, dpreRaw, ptrDpreI]);
  const dpreVal = b.id(); b.emit(Op.Select, [p.tF32, dpreVal, qRowInBounds, dpreRaw, p.const0f]);
  const ptrSDpre = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrSDpre, sDpre, threadIdx]);
  b.emit(Op.Store, [ptrSDpre, dpreVal]);

  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  // ── Compile-time unrolled i-loop in batches of BI ─────────────────────
  for (let iBatch = 0; iBatch < Br; iBatch += BI) {
    const dots: number[] = []; const tanhVals: number[] = [];
    for (let ii = 0; ii < BI; ii++) {
      const i = iBatch + ii;
      const qPos = b.id(); b.emit(Op.IAdd, [p.tU32, qPos, qBlockBase, constIIdx[i]]);
      const ptrRegK0 = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegK0, regK, constD4Idx[0]]);
      const kVec0 = b.id(); b.emit(Op.Load, [tVec4F32, kVec0, ptrRegK0]);
      const ptrSQ0 = b.id(); b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSQ0, sQ, constID4[i]]);
      const qVec0 = b.id(); b.emit(Op.Load, [tVec4F32, qVec0, ptrSQ0]);
      let dotAcc = b.id(); b.emit(Op.Dot, [p.tF32, dotAcc, kVec0, qVec0]);
      for (let d4 = 1; d4 < D4; d4++) {
        const ptrRegKd4 = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegKd4, regK, constD4Idx[d4]]);
        const kVecD4 = b.id(); b.emit(Op.Load, [tVec4F32, kVecD4, ptrRegKd4]);
        const sQIdx = b.id(); b.emit(Op.IAdd, [p.tU32, sQIdx, constID4[i], constD4Idx[d4]]);
        const ptrSQd4 = b.id(); b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSQd4, sQ, sQIdx]);
        const qVecD4 = b.id(); b.emit(Op.Load, [tVec4F32, qVecD4, ptrSQd4]);
        const partial = b.id(); b.emit(Op.Dot, [p.tF32, partial, kVecD4, qVecD4]);
        const newDot = b.id(); b.emit(Op.FAdd, [p.tF32, newDot, dotAcc, partial]);
        dotAcc = newDot;
      }
      const dotDivCap = b.id(); b.emit(Op.FMul, [p.tF32, dotDivCap, dotAcc, invSoftCap]);
      const tanhVal = b.id(); b.emit(Op.ExtInst, [p.tF32, tanhVal, p.glslStd, GLSLstd450.Tanh, dotDivCap]);
      const dotCapped = b.id(); b.emit(Op.FMul, [p.tF32, dotCapped, tanhVal, softCapValue]);
      const dotAfterCap = b.id(); b.emit(Op.Select, [p.tF32, dotAfterCap, softCapEnabled, dotCapped, dotAcc]);
      tanhVals.push(tanhVal);
      const oob = b.id(); b.emit(Op.UGreaterThanEqual, [p.tBool, oob, qPos, T]);
      const causal = b.id(); b.emit(Op.ULessThan, [p.tBool, causal, qPos, kRow]);
      const masked = b.id(); b.emit(Op.LogicalOr, [p.tBool, masked, oob, causal]);
      const dot = b.id(); b.emit(Op.Select, [p.tF32, dot, masked, constNegInf, dotAfterCap]);
      dots.push(dot);
    }
    const pVals: number[] = [];
    for (let ii = 0; ii < BI; ii++) {
      const i = iBatch + ii;
      const ptrSLSEi = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrSLSEi, sLSE, constIIdx[i]]);
      const lse_i = b.id(); b.emit(Op.Load, [p.tF32, lse_i, ptrSLSEi]);
      const sub = b.id(); b.emit(Op.FSub, [p.tF32, sub, dots[ii], lse_i]);
      const p_ij = b.id(); b.emit(Op.ExtInst, [p.tF32, p_ij, p.glslStd, GLSLstd450.Exp, sub]);
      pVals.push(p_ij);
    }
    const dovAccs: number[] = new Array(BI).fill(p.const0f);
    for (let d4 = 0; d4 < D4; d4++) {
      const ptrDV = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrDV, regDV, constD4Idx[d4]]);
      let dvVal = b.id(); b.emit(Op.Load, [tVec4F32, dvVal, ptrDV]);
      const ptrRegVd4 = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegVd4, regV, constD4Idx[d4]]);
      const vVec = b.id(); b.emit(Op.Load, [tVec4F32, vVec, ptrRegVd4]);
      for (let ii = 0; ii < BI; ii++) {
        const i = iBatch + ii;
        const sDOIdx = b.id(); b.emit(Op.IAdd, [p.tU32, sDOIdx, constID4[i], constD4Idx[d4]]);
        const ptrSDOd = b.id(); b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSDOd, sDO, sDOIdx]);
        const doVec = b.id(); b.emit(Op.Load, [tVec4F32, doVec, ptrSDOd]);
        const pDO = b.id(); b.emit(Op.VectorTimesScalar, [tVec4F32, pDO, doVec, pVals[ii]]);
        const dvNew = b.id(); b.emit(Op.FAdd, [tVec4F32, dvNew, dvVal, pDO]);
        dvVal = dvNew;
        const partialDOV = b.id(); b.emit(Op.Dot, [p.tF32, partialDOV, doVec, vVec]);
        const newDov = b.id(); b.emit(Op.FAdd, [p.tF32, newDov, dovAccs[ii], partialDOV]);
        dovAccs[ii] = newDov;
      }
      b.emit(Op.Store, [ptrDV, dvVal]);
    }
    const dScores: number[] = [];
    for (let ii = 0; ii < BI; ii++) {
      const i = iBatch + ii;
      const ptrSDpreI = b.id(); b.emit(Op.AccessChain, [tPtrSharedF32, ptrSDpreI, sDpre, constIIdx[i]]);
      const dpreI = b.id(); b.emit(Op.Load, [p.tF32, dpreI, ptrSDpreI]);
      const sub2 = b.id(); b.emit(Op.FSub, [p.tF32, sub2, dovAccs[ii], dpreI]);
      const dS = b.id(); b.emit(Op.FMul, [p.tF32, dS, pVals[ii], sub2]);
      const tSq = b.id(); b.emit(Op.FMul, [p.tF32, tSq, tanhVals[ii], tanhVals[ii]]);
      const deriv = b.id(); b.emit(Op.FSub, [p.tF32, deriv, const1f, tSq]);
      const dSs = b.id(); b.emit(Op.FMul, [p.tF32, dSs, dS, scale]);
      const dSc = b.id(); b.emit(Op.FMul, [p.tF32, dSc, dSs, deriv]);
      const dSu = b.id(); b.emit(Op.FMul, [p.tF32, dSu, dS, scale]);
      const dScore = b.id(); b.emit(Op.Select, [p.tF32, dScore, softCapEnabled, dSc, dSu]);
      dScores.push(dScore);
    }
    for (let d4 = 0; d4 < D4; d4++) {
      const ptrDK = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrDK, regDK, constD4Idx[d4]]);
      let dkVal = b.id(); b.emit(Op.Load, [tVec4F32, dkVal, ptrDK]);
      for (let ii = 0; ii < BI; ii++) {
        const i = iBatch + ii;
        const sQIdx = b.id(); b.emit(Op.IAdd, [p.tU32, sQIdx, constID4[i], constD4Idx[d4]]);
        const ptrSQdq = b.id(); b.emit(Op.AccessChain, [tPtrSharedVec4, ptrSQdq, sQ, sQIdx]);
        const qVecDK = b.id(); b.emit(Op.Load, [tVec4F32, qVecDK, ptrSQdq]);
        const contrib = b.id(); b.emit(Op.VectorTimesScalar, [tVec4F32, contrib, qVecDK, dScores[ii]]);
        const dkNew = b.id(); b.emit(Op.FAdd, [tVec4F32, dkNew, dkVal, contrib]);
        dkVal = dkNew;
      }
      b.emit(Op.Store, [ptrDK, dkVal]);
    }
  }

  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);
  b.emit(Op.Branch, [labelLoopCont]);
  b.emit(Op.Label, [labelLoopCont]);
  const nextQBlock = b.id(); b.emit(Op.Load, [p.tU32, nextQBlock, varQBlockIdx]);
  const incQBlock = b.id(); b.emit(Op.IAdd, [p.tU32, incQBlock, nextQBlock, p.const1u]);
  b.emit(Op.Store, [varQBlockIdx, incQBlock]);
  b.emit(Op.Branch, [labelLoopHead]);
  b.emit(Op.Label, [labelLoopMerge]);

  const dkBase4 = b.id(); b.emit(Op.IAdd, [p.tU32, dkBase4, baseOff4, kRowD4]);
  for (let d4 = 0; d4 < D4; d4++) {
    const ptrRegDKd4 = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegDKd4, regDK, constD4Idx[d4]]);
    const regDKVec = b.id(); b.emit(Op.Load, [tVec4F32, regDKVec, ptrRegDKd4]);
    const dkIdx = b.id(); b.emit(Op.IAdd, [p.tU32, dkIdx, dkBase4, constD4Idx[d4]]);
    const ptrOutDK = b.id(); b.emit(Op.AccessChain, [bufDK.tPtrVec4, ptrOutDK, bufDK.varId, p.const0u, dkIdx]);
    b.emit(Op.Store, [ptrOutDK, regDKVec]);
    const ptrRegDVd4 = b.id(); b.emit(Op.AccessChain, [tPtrFnVec4, ptrRegDVd4, regDV, constD4Idx[d4]]);
    const regDVVec = b.id(); b.emit(Op.Load, [tVec4F32, regDVVec, ptrRegDVd4]);
    const dvIdx = b.id(); b.emit(Op.IAdd, [p.tU32, dvIdx, dkBase4, constD4Idx[d4]]);
    const ptrOutDV = b.id(); b.emit(Op.AccessChain, [bufDV.tPtrVec4, ptrOutDV, bufDV.varId, p.const0u, dvIdx]);
    b.emit(Op.Store, [ptrOutDV, regDVVec]);
  }

  b.emit(Op.Branch, [labelEnd]);
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);
  return b.build();
}

