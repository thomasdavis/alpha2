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
  preamble, declareStorageBuffer, declareParamsPushConstant,
} from "./helpers.js";

// ── Kernel: Flash Attention Forward ──────────────────────────────────────────

export function kernelFlashAttentionForward(Br: number, Bc: number, D: number): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, Br, 1, 1);

  // Storage buffers: Q, K, V (readonly), O (write), LSE (write)
  const bufQ   = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufK   = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, true);
  const bufV   = declareStorageBuffer(b, p.tF32, p.tU32, 0, 2, true);
  const bufO   = declareStorageBuffer(b, p.tF32, p.tU32, 0, 3, false);
  const bufLSE = declareStorageBuffer(b, p.tF32, p.tU32, 0, 4, false);

  // Push constants: { T, scale, softCapValue, _pad }
  const pc = declareParamsPushConstant(b, p.tF32, 4);

  // ── Constants ──────────────────────────────────────────────────────────────

  const constBr = b.id();
  b.constant(p.tU32, constBr, Br);
  const constBc = b.id();
  b.constant(p.tU32, constBc, Bc);
  const constD = b.id();
  b.constant(p.tU32, constD, D);
  const constBcD = b.id();
  b.constant(p.tU32, constBcD, Bc * D);

  const constNegInf = b.id();
  b.constant(p.tF32, constNegInf, 0xFF800000); // -infinity (IEEE 754 raw bits)

  const const1f = b.id();
  b.constantF32(p.tF32, const1f, 1.0);

  // Pre-generate uint constants for unrolled j indices
  const constJ: number[] = [];
  for (let j = 0; j < Bc; j++) {
    const cj = b.id();
    b.constant(p.tU32, cj, j);
    constJ.push(cj);
  }

  // Pre-generate uint constants for j * D offsets into shared memory
  const constJD: number[] = [];
  for (let j = 0; j < Bc; j++) {
    const cjd = b.id();
    b.constant(p.tU32, cjd, j * D);
    constJD.push(cjd);
  }

  // Pre-generate uint constants for d indices
  const constDIdx: number[] = [];
  for (let d = 0; d < D; d++) {
    const cd = b.id();
    b.constant(p.tU32, cd, d);
    constDIdx.push(cd);
  }

  // ── Shared memory: sK[Bc * D] and sV[Bc * D] ─────────────────────────────

  const constSharedSize = b.id();
  b.constant(p.tU32, constSharedSize, Bc * D);
  const tArrayShared = b.id();
  b.typeArray(tArrayShared, p.tF32, constSharedSize);
  const tPtrSharedArr = b.id();
  b.typePointer(tPtrSharedArr, StorageClass.Workgroup, tArrayShared);
  const tPtrSharedF32 = b.id();
  b.typePointer(tPtrSharedF32, StorageClass.Workgroup, p.tF32);
  const sK = b.id();
  b.variable(tPtrSharedArr, sK, StorageClass.Workgroup);
  const sV = b.id();
  b.variable(tPtrSharedArr, sV, StorageClass.Workgroup);

  // ── Function-scope array types: regQ[D] and regO[D] ──────────────────────

  const constDArr = b.id();
  b.constant(p.tU32, constDArr, D);
  const tArrayD = b.id();
  b.typeArray(tArrayD, p.tF32, constDArr);
  const tPtrFnArr = b.id();
  b.typePointer(tPtrFnArr, StorageClass.Function, tArrayD);
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

  // Push constant member index constants
  const const3u = b.id();
  b.constant(p.tU32, const3u, 3);

  // ── Entry point ───────────────────────────────────────────────────────────

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId, vWorkgroupId, vLocalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, Br, 1, 1);

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  const labelEntry = b.id();
  b.emit(Op.Label, [labelEntry]);

  // ── All Variable declarations must be in the entry block ──────────────────

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

  // ── Load push constants ───────────────────────────────────────────────────

  // T (sequence length)
  const ptrTpc = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrTpc, pc.varId, p.const0u]);
  const TF = b.id();
  b.emit(Op.Load, [p.tF32, TF, ptrTpc]);
  const T = b.id();
  b.emit(Op.ConvertFToU, [p.tU32, T, TF]);

  // scale (1/sqrt(D))
  const ptrScale = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrScale, pc.varId, p.const1u]);
  const scale = b.id();
  b.emit(Op.Load, [p.tF32, scale, ptrScale]);

  // softCapValue
  const ptrSoftCap = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrSoftCap, pc.varId, p.const2u]);
  const softCapValue = b.id();
  b.emit(Op.Load, [p.tF32, softCapValue, ptrSoftCap]);

  // Check if softcap is enabled (softCapValue > 0)
  const softCapEnabled = b.id();
  b.emit(Op.FOrdGreaterThan, [p.tBool, softCapEnabled, softCapValue, p.const0f]);

  // Precompute 1/softCapValue for softcap (safe: if 0, we never use it)
  const invSoftCap = b.id();
  b.emit(Op.FDiv, [p.tF32, invSoftCap, const1f, softCapValue]);

  // ── Compute base offset: baseOff = bhIdx * T * D ──────────────────────────

  const TD = b.id();
  b.emit(Op.IMul, [p.tU32, TD, T, constD]);
  const baseOff = b.id();
  b.emit(Op.IMul, [p.tU32, baseOff, bhIdx, TD]);

  // ── Bounds check: if qRow >= T, skip to end ───────────────────────────────

  const qRowOob = b.id();
  b.emit(Op.UGreaterThanEqual, [p.tBool, qRowOob, qRow, T]);
  const labelMain = b.id();
  const labelEnd = b.id();
  b.emit(Op.SelectionMerge, [labelEnd, 0]);
  b.emit(Op.BranchConditional, [qRowOob, labelEnd, labelMain]);

  b.emit(Op.Label, [labelMain]);

  // ── Load Q[qRow] into regQ ────────────────────────────────────────────────
  // Q[baseOff + qRow * D + d] for d = 0..D-1

  const qRowOff = b.id();
  b.emit(Op.IMul, [p.tU32, qRowOff, qRow, constD]);
  const qBase = b.id();
  b.emit(Op.IAdd, [p.tU32, qBase, baseOff, qRowOff]);

  for (let d = 0; d < D; d++) {
    const qIdx = b.id();
    b.emit(Op.IAdd, [p.tU32, qIdx, qBase, constDIdx[d]]);
    const ptrQElem = b.id();
    b.emit(Op.AccessChain, [bufQ.tPtrF32, ptrQElem, bufQ.varId, p.const0u, qIdx]);
    const qVal = b.id();
    b.emit(Op.Load, [p.tF32, qVal, ptrQElem]);
    const ptrRegQ = b.id();
    b.emit(Op.AccessChain, [tPtrFnF32, ptrRegQ, regQ, constDIdx[d]]);
    b.emit(Op.Store, [ptrRegQ, qVal]);
  }

  // ── Initialize regO to 0.0 ────────────────────────────────────────────────

  for (let d = 0; d < D; d++) {
    const ptrRegO = b.id();
    b.emit(Op.AccessChain, [tPtrFnF32, ptrRegO, regO, constDIdx[d]]);
    b.emit(Op.Store, [ptrRegO, p.const0f]);
  }

  // m = -inf, l = 0.0
  b.emit(Op.Store, [varM, constNegInf]);
  b.emit(Op.Store, [varL, p.const0f]);

  // ── Compute effective number of key blocks (causal) ───────────────────────
  // effectiveKBlocks = qBlockIdx + 1 (since Br == Bc, causal masking)

  const effectiveKBlocks = b.id();
  b.emit(Op.IAdd, [p.tU32, effectiveKBlocks, qBlockIdx, p.const1u]);

  // ── Outer loop: for kBlockIdx = 0; kBlockIdx < effectiveKBlocks; kBlockIdx++ ──

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

  // ── Cooperative load: each thread loads one row of K and V into shared mem ─
  // kRow = kBlockIdx * Bc + threadIdx

  const kBlockBase = b.id();
  b.emit(Op.IMul, [p.tU32, kBlockBase, kBlockIdx, constBc]);
  const kRow = b.id();
  b.emit(Op.IAdd, [p.tU32, kRow, kBlockBase, threadIdx]);

  // kRowInBounds = kRow < T
  const kRowInBounds = b.id();
  b.emit(Op.ULessThan, [p.tBool, kRowInBounds, kRow, T]);

  // kGlobalBase = baseOff + kRow * D
  const kRowOff = b.id();
  b.emit(Op.IMul, [p.tU32, kRowOff, kRow, constD]);
  const kGlobalBase = b.id();
  b.emit(Op.IAdd, [p.tU32, kGlobalBase, baseOff, kRowOff]);

  // threadIdx * D = offset into shared memory for this thread's row
  const sharedRowOff = b.id();
  b.emit(Op.IMul, [p.tU32, sharedRowOff, threadIdx, constD]);

  // Load D elements for K and V
  for (let d = 0; d < D; d++) {
    // Global index into K/V buffer
    const gIdx = b.id();
    b.emit(Op.IAdd, [p.tU32, gIdx, kGlobalBase, constDIdx[d]]);

    // Load K element (0 if out of bounds)
    const ptrKElem = b.id();
    b.emit(Op.AccessChain, [bufK.tPtrF32, ptrKElem, bufK.varId, p.const0u, gIdx]);
    const kRaw = b.id();
    b.emit(Op.Load, [p.tF32, kRaw, ptrKElem]);
    const kVal = b.id();
    b.emit(Op.Select, [p.tF32, kVal, kRowInBounds, kRaw, p.const0f]);

    // Store to sK[threadIdx * D + d]
    const sKIdx = b.id();
    b.emit(Op.IAdd, [p.tU32, sKIdx, sharedRowOff, constDIdx[d]]);
    const ptrSK = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF32, ptrSK, sK, sKIdx]);
    b.emit(Op.Store, [ptrSK, kVal]);

    // Load V element (0 if out of bounds)
    const ptrVElem = b.id();
    b.emit(Op.AccessChain, [bufV.tPtrF32, ptrVElem, bufV.varId, p.const0u, gIdx]);
    const vRaw = b.id();
    b.emit(Op.Load, [p.tF32, vRaw, ptrVElem]);
    const vVal = b.id();
    b.emit(Op.Select, [p.tF32, vVal, kRowInBounds, vRaw, p.const0f]);

    // Store to sV[threadIdx * D + d]
    const ptrSV = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF32, ptrSV, sV, sKIdx]); // same index
    b.emit(Op.Store, [ptrSV, vVal]);
  }

  // Barrier — all threads have loaded their K/V rows
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  // ── Unrolled loop over Bc keys: online softmax + accumulate ───────────────
  // For each j = 0..Bc-1:
  //   1. Compute dot = Q[qRow] · K[kPos] * scale
  //   2. Apply softcap if enabled
  //   3. Apply causal mask (kPos > qRow or kPos >= T → -inf)
  //   4. Online softmax update: rescale old accumulators, accumulate new

  for (let j = 0; j < Bc; j++) {
    // kPos = kBlockBase + j
    const kPos = b.id();
    b.emit(Op.IAdd, [p.tU32, kPos, kBlockBase, constJ[j]]);

    // ── Dot product: Q[qRow] · sK[j] ──────────────────────────────────────
    // Unrolled over D with FMA-style accumulation

    // Start with first element
    const ptrRegQ0 = b.id();
    b.emit(Op.AccessChain, [tPtrFnF32, ptrRegQ0, regQ, constDIdx[0]]);
    const qv0 = b.id();
    b.emit(Op.Load, [p.tF32, qv0, ptrRegQ0]);

    const sKIdx0 = b.id();
    b.emit(Op.IAdd, [p.tU32, sKIdx0, constJD[j], constDIdx[0]]);
    const ptrSK0 = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF32, ptrSK0, sK, sKIdx0]);
    const kv0 = b.id();
    b.emit(Op.Load, [p.tF32, kv0, ptrSK0]);
    let dotAcc = b.id();
    b.emit(Op.FMul, [p.tF32, dotAcc, qv0, kv0]);

    // Remaining D-1 elements
    for (let d = 1; d < D; d++) {
      const ptrRegQd = b.id();
      b.emit(Op.AccessChain, [tPtrFnF32, ptrRegQd, regQ, constDIdx[d]]);
      const qvd = b.id();
      b.emit(Op.Load, [p.tF32, qvd, ptrRegQd]);

      const sKIdxD = b.id();
      b.emit(Op.IAdd, [p.tU32, sKIdxD, constJD[j], constDIdx[d]]);
      const ptrSKd = b.id();
      b.emit(Op.AccessChain, [tPtrSharedF32, ptrSKd, sK, sKIdxD]);
      const kvd = b.id();
      b.emit(Op.Load, [p.tF32, kvd, ptrSKd]);

      const prod = b.id();
      b.emit(Op.FMul, [p.tF32, prod, qvd, kvd]);
      const newDotAcc = b.id();
      b.emit(Op.FAdd, [p.tF32, newDotAcc, dotAcc, prod]);
      dotAcc = newDotAcc;
    }

    // dot *= scale
    const dotScaled = b.id();
    b.emit(Op.FMul, [p.tF32, dotScaled, dotAcc, scale]);

    // ── Soft capping: tanh(dot / cap) * cap if enabled ──────────────────────
    // Branchless: compute both, select based on softCapEnabled

    const dotDivCap = b.id();
    b.emit(Op.FMul, [p.tF32, dotDivCap, dotScaled, invSoftCap]);
    const tanhVal = b.id();
    b.emit(Op.ExtInst, [p.tF32, tanhVal, p.glslStd, GLSLstd450.Tanh, dotDivCap]);
    const dotCapped = b.id();
    b.emit(Op.FMul, [p.tF32, dotCapped, tanhVal, softCapValue]);
    const dotAfterCap = b.id();
    b.emit(Op.Select, [p.tF32, dotAfterCap, softCapEnabled, dotCapped, dotScaled]);

    // ── Causal mask + out-of-bounds ─────────────────────────────────────────
    // masked = (kPos >= T) || (qRow < kPos)  → set dot to -inf
    const oob = b.id();
    b.emit(Op.UGreaterThanEqual, [p.tBool, oob, kPos, T]);
    const causal = b.id();
    b.emit(Op.ULessThan, [p.tBool, causal, qRow, kPos]); // qRow < kPos ≡ kPos > qRow
    const masked = b.id();
    b.emit(Op.LogicalOr, [p.tBool, masked, oob, causal]);
    const dot = b.id();
    b.emit(Op.Select, [p.tF32, dot, masked, constNegInf, dotAfterCap]);

    // ── Online softmax update ───────────────────────────────────────────────
    // mOld = load(varM)
    const mOld = b.id();
    b.emit(Op.Load, [p.tF32, mOld, varM]);

    // mNew = max(mOld, dot)
    const mNew = b.id();
    b.emit(Op.ExtInst, [p.tF32, mNew, p.glslStd, GLSLstd450.FMax, mOld, dot]);

    // alpha = exp(mOld - mNew)  (rescale factor for old accumulators)
    const mDiff = b.id();
    b.emit(Op.FSub, [p.tF32, mDiff, mOld, mNew]);
    const alpha = b.id();
    b.emit(Op.ExtInst, [p.tF32, alpha, p.glslStd, GLSLstd450.Exp, mDiff]);

    // pj = exp(dot - mNew)  (new attention weight)
    const dotDiff = b.id();
    b.emit(Op.FSub, [p.tF32, dotDiff, dot, mNew]);
    const pj = b.id();
    b.emit(Op.ExtInst, [p.tF32, pj, p.glslStd, GLSLstd450.Exp, dotDiff]);

    // l = l * alpha + pj
    const lOld = b.id();
    b.emit(Op.Load, [p.tF32, lOld, varL]);
    const lScaled = b.id();
    b.emit(Op.FMul, [p.tF32, lScaled, lOld, alpha]);
    const lNew = b.id();
    b.emit(Op.FAdd, [p.tF32, lNew, lScaled, pj]);
    b.emit(Op.Store, [varL, lNew]);

    // m = mNew
    b.emit(Op.Store, [varM, mNew]);

    // regO[d] = regO[d] * alpha + pj * sV[j * D + d]  for d = 0..D-1
    for (let d = 0; d < D; d++) {
      // Load regO[d]
      const ptrO = b.id();
      b.emit(Op.AccessChain, [tPtrFnF32, ptrO, regO, constDIdx[d]]);
      const oOld = b.id();
      b.emit(Op.Load, [p.tF32, oOld, ptrO]);

      // regO[d] * alpha
      const oScaled = b.id();
      b.emit(Op.FMul, [p.tF32, oScaled, oOld, alpha]);

      // Load sV[j * D + d]
      const sVIdx = b.id();
      b.emit(Op.IAdd, [p.tU32, sVIdx, constJD[j], constDIdx[d]]);
      const ptrSV = b.id();
      b.emit(Op.AccessChain, [tPtrSharedF32, ptrSV, sV, sVIdx]);
      const vv = b.id();
      b.emit(Op.Load, [p.tF32, vv, ptrSV]);

      // pj * sV[j * D + d]
      const pvv = b.id();
      b.emit(Op.FMul, [p.tF32, pvv, pj, vv]);

      // regO[d] = oScaled + pvv
      const oNew = b.id();
      b.emit(Op.FAdd, [p.tF32, oNew, oScaled, pvv]);
      b.emit(Op.Store, [ptrO, oNew]);
    }
  }

  // Barrier before next key block loads new sK/sV
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  // ── Loop continuation ─────────────────────────────────────────────────────

  b.emit(Op.Branch, [labelLoopCont]);
  b.emit(Op.Label, [labelLoopCont]);
  const nextKBlock = b.id();
  b.emit(Op.Load, [p.tU32, nextKBlock, varKBlockIdx]);
  const incKBlock = b.id();
  b.emit(Op.IAdd, [p.tU32, incKBlock, nextKBlock, p.const1u]);
  b.emit(Op.Store, [varKBlockIdx, incKBlock]);
  b.emit(Op.Branch, [labelLoopHead]);

  b.emit(Op.Label, [labelLoopMerge]);

  // ── Normalize output and write ────────────────────────────────────────────
  // O[baseOff + qRow * D + d] = regO[d] / l

  const finalL = b.id();
  b.emit(Op.Load, [p.tF32, finalL, varL]);
  const invL = b.id();
  b.emit(Op.FDiv, [p.tF32, invL, const1f, finalL]);

  const oBase = b.id();
  b.emit(Op.IAdd, [p.tU32, oBase, baseOff, qRowOff]);

  for (let d = 0; d < D; d++) {
    // Load regO[d]
    const ptrRegOd = b.id();
    b.emit(Op.AccessChain, [tPtrFnF32, ptrRegOd, regO, constDIdx[d]]);
    const regOVal = b.id();
    b.emit(Op.Load, [p.tF32, regOVal, ptrRegOd]);

    // Normalize
    const oNorm = b.id();
    b.emit(Op.FMul, [p.tF32, oNorm, regOVal, invL]);

    // Write to output
    const oIdx = b.id();
    b.emit(Op.IAdd, [p.tU32, oIdx, oBase, constDIdx[d]]);
    const ptrOut = b.id();
    b.emit(Op.AccessChain, [bufO.tPtrF32, ptrOut, bufO.varId, p.const0u, oIdx]);
    b.emit(Op.Store, [ptrOut, oNorm]);
  }

  // ── Store LSE for backward: LSE[bhIdx * T + qRow] = m + log(l) ───────────

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

  // Branch to end
  b.emit(Op.Branch, [labelEnd]);

  // ── End label (early exit for OOB threads also lands here) ────────────────
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

// ── Kernel: Flash Attention Backward dQ ─────────────────────────────────────
//
// Computes dQ (gradient w.r.t. queries) using recomputed attention weights.
// Each workgroup processes one query block (Br rows). Each thread handles one
// query row within the block. Key/value blocks are cooperatively loaded into
// shared memory, then attention gradients are accumulated.
//
// Bindings:
//   0: Q [B*H, T, D]       (readonly)
//   1: K [B*H, T, D]       (readonly)
//   2: V [B*H, T, D]       (readonly)
//   3: dO [B*H, T, D]      (readonly) — gradient of output
//   4: LSE [B*H, T]        (readonly) — log-sum-exp from forward
//   5: D_precomp [B*H, T]  (readonly) — D[i] = sum_d(dO[i,d] * O[i,d])
//   6: dQ [B*H, T, D]      (write)   — output gradient
//
// Push constants (16 bytes = 4 x f32): { T, scale, softCapValue, _pad }
//
// Dispatch: (ceil(T/Br), B*H, 1)
// Workgroup: (Br, 1, 1)

export function kernelFlashAttentionBackwardDQ(Br: number, Bc: number, D: number): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, Br, 1, 1);

  // Storage buffers: Q, K, V, dO, LSE, D_precomp (readonly), dQ (write)
  const bufQ      = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufK      = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, true);
  const bufV      = declareStorageBuffer(b, p.tF32, p.tU32, 0, 2, true);
  const bufDO     = declareStorageBuffer(b, p.tF32, p.tU32, 0, 3, true);
  const bufLSE    = declareStorageBuffer(b, p.tF32, p.tU32, 0, 4, true);
  const bufDpre   = declareStorageBuffer(b, p.tF32, p.tU32, 0, 5, true);
  const bufDQ     = declareStorageBuffer(b, p.tF32, p.tU32, 0, 6, false);

  // Push constants: { T, scale, softCapValue, _pad }
  const pc = declareParamsPushConstant(b, p.tF32, 4);

  // ── Constants ──────────────────────────────────────────────────────────────

  const constBr = b.id();
  b.constant(p.tU32, constBr, Br);
  const constBc = b.id();
  b.constant(p.tU32, constBc, Bc);
  const constD = b.id();
  b.constant(p.tU32, constD, D);
  const constBcD = b.id();
  b.constant(p.tU32, constBcD, Bc * D);

  const constNegInf = b.id();
  b.constant(p.tF32, constNegInf, 0xFF800000); // -infinity (IEEE 754 raw bits)

  const const1f = b.id();
  b.constantF32(p.tF32, const1f, 1.0);

  // Pre-generate uint constants for unrolled j indices
  const constJ: number[] = [];
  for (let j = 0; j < Bc; j++) {
    const cj = b.id();
    b.constant(p.tU32, cj, j);
    constJ.push(cj);
  }

  // Pre-generate uint constants for j * D offsets into shared memory
  const constJD: number[] = [];
  for (let j = 0; j < Bc; j++) {
    const cjd = b.id();
    b.constant(p.tU32, cjd, j * D);
    constJD.push(cjd);
  }

  // Pre-generate uint constants for d indices
  const constDIdx: number[] = [];
  for (let d = 0; d < D; d++) {
    const cd = b.id();
    b.constant(p.tU32, cd, d);
    constDIdx.push(cd);
  }

  // ── Shared memory: sK[Bc * D] and sV[Bc * D] ─────────────────────────────

  const constSharedSize = b.id();
  b.constant(p.tU32, constSharedSize, Bc * D);
  const tArrayShared = b.id();
  b.typeArray(tArrayShared, p.tF32, constSharedSize);
  const tPtrSharedArr = b.id();
  b.typePointer(tPtrSharedArr, StorageClass.Workgroup, tArrayShared);
  const tPtrSharedF32 = b.id();
  b.typePointer(tPtrSharedF32, StorageClass.Workgroup, p.tF32);
  const sK = b.id();
  b.variable(tPtrSharedArr, sK, StorageClass.Workgroup);
  const sV = b.id();
  b.variable(tPtrSharedArr, sV, StorageClass.Workgroup);

  // ── Function-scope array types: regQ[D], regDO[D], regDQ[D] ──────────────

  const constDArr = b.id();
  b.constant(p.tU32, constDArr, D);
  const tArrayD = b.id();
  b.typeArray(tArrayD, p.tF32, constDArr);
  const tPtrFnArr = b.id();
  b.typePointer(tPtrFnArr, StorageClass.Function, tArrayD);
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

  // Push constant member index constants
  const const3u = b.id();
  b.constant(p.tU32, const3u, 3);

  // ── Entry point ───────────────────────────────────────────────────────────

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId, vWorkgroupId, vLocalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, Br, 1, 1);

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  const labelEntry = b.id();
  b.emit(Op.Label, [labelEntry]);

  // ── All Variable declarations must be in the entry block ──────────────────

  const regQ = b.id();
  b.emit(Op.Variable, [tPtrFnArr, regQ, StorageClass.Function]);
  const regDO = b.id();
  b.emit(Op.Variable, [tPtrFnArr, regDO, StorageClass.Function]);
  const regDQ = b.id();
  b.emit(Op.Variable, [tPtrFnArr, regDQ, StorageClass.Function]);
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

  // ── Load push constants ───────────────────────────────────────────────────

  // T (sequence length)
  const ptrTpc = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrTpc, pc.varId, p.const0u]);
  const TF = b.id();
  b.emit(Op.Load, [p.tF32, TF, ptrTpc]);
  const T = b.id();
  b.emit(Op.ConvertFToU, [p.tU32, T, TF]);

  // scale (1/sqrt(D))
  const ptrScale = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrScale, pc.varId, p.const1u]);
  const scale = b.id();
  b.emit(Op.Load, [p.tF32, scale, ptrScale]);

  // softCapValue
  const ptrSoftCap = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrSoftCap, pc.varId, p.const2u]);
  const softCapValue = b.id();
  b.emit(Op.Load, [p.tF32, softCapValue, ptrSoftCap]);

  // Check if softcap is enabled (softCapValue > 0)
  const softCapEnabled = b.id();
  b.emit(Op.FOrdGreaterThan, [p.tBool, softCapEnabled, softCapValue, p.const0f]);

  // Precompute 1/softCapValue for softcap (safe: if 0, we never use it)
  const invSoftCap = b.id();
  b.emit(Op.FDiv, [p.tF32, invSoftCap, const1f, softCapValue]);

  // ── Compute base offsets ──────────────────────────────────────────────────
  // baseOff = bhIdx * T * D  (for Q/K/V/dO/dQ indexing)
  // lseBaseOff = bhIdx * T   (for LSE/D_precomp indexing)

  const TD = b.id();
  b.emit(Op.IMul, [p.tU32, TD, T, constD]);
  const baseOff = b.id();
  b.emit(Op.IMul, [p.tU32, baseOff, bhIdx, TD]);

  const lseBaseOff = b.id();
  b.emit(Op.IMul, [p.tU32, lseBaseOff, bhIdx, T]);

  // ── Bounds check: if qRow >= T, skip to end ───────────────────────────────

  const qRowOob = b.id();
  b.emit(Op.UGreaterThanEqual, [p.tBool, qRowOob, qRow, T]);
  const labelMain = b.id();
  const labelEnd = b.id();
  b.emit(Op.SelectionMerge, [labelEnd, 0]);
  b.emit(Op.BranchConditional, [qRowOob, labelEnd, labelMain]);

  b.emit(Op.Label, [labelMain]);

  // ── Load Q[qRow] into regQ ────────────────────────────────────────────────

  const qRowOff = b.id();
  b.emit(Op.IMul, [p.tU32, qRowOff, qRow, constD]);
  const qBase = b.id();
  b.emit(Op.IAdd, [p.tU32, qBase, baseOff, qRowOff]);

  for (let d = 0; d < D; d++) {
    const qIdx = b.id();
    b.emit(Op.IAdd, [p.tU32, qIdx, qBase, constDIdx[d]]);
    const ptrQElem = b.id();
    b.emit(Op.AccessChain, [bufQ.tPtrF32, ptrQElem, bufQ.varId, p.const0u, qIdx]);
    const qVal = b.id();
    b.emit(Op.Load, [p.tF32, qVal, ptrQElem]);
    const ptrRegQ = b.id();
    b.emit(Op.AccessChain, [tPtrFnF32, ptrRegQ, regQ, constDIdx[d]]);
    b.emit(Op.Store, [ptrRegQ, qVal]);
  }

  // ── Load dO[qRow] into regDO ──────────────────────────────────────────────

  for (let d = 0; d < D; d++) {
    const doIdx = b.id();
    b.emit(Op.IAdd, [p.tU32, doIdx, qBase, constDIdx[d]]);
    const ptrDOElem = b.id();
    b.emit(Op.AccessChain, [bufDO.tPtrF32, ptrDOElem, bufDO.varId, p.const0u, doIdx]);
    const doVal = b.id();
    b.emit(Op.Load, [p.tF32, doVal, ptrDOElem]);
    const ptrRegDO = b.id();
    b.emit(Op.AccessChain, [tPtrFnF32, ptrRegDO, regDO, constDIdx[d]]);
    b.emit(Op.Store, [ptrRegDO, doVal]);
  }

  // ── Load LSE[qRow] and D_precomp[qRow] ───────────────────────────────────

  const lseQRowIdx = b.id();
  b.emit(Op.IAdd, [p.tU32, lseQRowIdx, lseBaseOff, qRow]);

  const ptrLSEi = b.id();
  b.emit(Op.AccessChain, [bufLSE.tPtrF32, ptrLSEi, bufLSE.varId, p.const0u, lseQRowIdx]);
  const lse_i = b.id();
  b.emit(Op.Load, [p.tF32, lse_i, ptrLSEi]);

  const ptrDi = b.id();
  b.emit(Op.AccessChain, [bufDpre.tPtrF32, ptrDi, bufDpre.varId, p.const0u, lseQRowIdx]);
  const Di = b.id();
  b.emit(Op.Load, [p.tF32, Di, ptrDi]);

  // ── Initialize regDQ to 0.0 ───────────────────────────────────────────────

  for (let d = 0; d < D; d++) {
    const ptrRegDQ = b.id();
    b.emit(Op.AccessChain, [tPtrFnF32, ptrRegDQ, regDQ, constDIdx[d]]);
    b.emit(Op.Store, [ptrRegDQ, p.const0f]);
  }

  // ── Compute effective number of key blocks (causal) ───────────────────────
  // effectiveKBlocks = qBlockIdx + 1

  const effectiveKBlocks = b.id();
  b.emit(Op.IAdd, [p.tU32, effectiveKBlocks, qBlockIdx, p.const1u]);

  // ── Outer loop: for kBlockIdx = 0; kBlockIdx < effectiveKBlocks; kBlockIdx++ ──

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

  // ── Cooperative load: each thread loads one row of K and V into shared mem ─

  const kBlockBase = b.id();
  b.emit(Op.IMul, [p.tU32, kBlockBase, kBlockIdx, constBc]);
  const kRow = b.id();
  b.emit(Op.IAdd, [p.tU32, kRow, kBlockBase, threadIdx]);

  // kRowInBounds = kRow < T
  const kRowInBounds = b.id();
  b.emit(Op.ULessThan, [p.tBool, kRowInBounds, kRow, T]);

  // kGlobalBase = baseOff + kRow * D
  const kRowOff = b.id();
  b.emit(Op.IMul, [p.tU32, kRowOff, kRow, constD]);
  const kGlobalBase = b.id();
  b.emit(Op.IAdd, [p.tU32, kGlobalBase, baseOff, kRowOff]);

  // threadIdx * D = offset into shared memory for this thread's row
  const sharedRowOff = b.id();
  b.emit(Op.IMul, [p.tU32, sharedRowOff, threadIdx, constD]);

  // Load D elements for K and V
  for (let d = 0; d < D; d++) {
    // Global index into K/V buffer
    const gIdx = b.id();
    b.emit(Op.IAdd, [p.tU32, gIdx, kGlobalBase, constDIdx[d]]);

    // Load K element (0 if out of bounds)
    const ptrKElem = b.id();
    b.emit(Op.AccessChain, [bufK.tPtrF32, ptrKElem, bufK.varId, p.const0u, gIdx]);
    const kRaw = b.id();
    b.emit(Op.Load, [p.tF32, kRaw, ptrKElem]);
    const kVal = b.id();
    b.emit(Op.Select, [p.tF32, kVal, kRowInBounds, kRaw, p.const0f]);

    // Store to sK[threadIdx * D + d]
    const sKIdx = b.id();
    b.emit(Op.IAdd, [p.tU32, sKIdx, sharedRowOff, constDIdx[d]]);
    const ptrSK = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF32, ptrSK, sK, sKIdx]);
    b.emit(Op.Store, [ptrSK, kVal]);

    // Load V element (0 if out of bounds)
    const ptrVElem = b.id();
    b.emit(Op.AccessChain, [bufV.tPtrF32, ptrVElem, bufV.varId, p.const0u, gIdx]);
    const vRaw = b.id();
    b.emit(Op.Load, [p.tF32, vRaw, ptrVElem]);
    const vVal = b.id();
    b.emit(Op.Select, [p.tF32, vVal, kRowInBounds, vRaw, p.const0f]);

    // Store to sV[threadIdx * D + d]
    const ptrSV = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF32, ptrSV, sV, sKIdx]); // same index
    b.emit(Op.Store, [ptrSV, vVal]);
  }

  // Barrier — all threads have loaded their K/V rows
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  // ── Unrolled loop over Bc keys: recompute attention + accumulate dQ ───────
  // For each j = 0..Bc-1:
  //   1. Compute raw dot = Q[qRow] · K[kPos] * scale (before masking)
  //   2. Apply softcap to get capped value
  //   3. Apply causal mask
  //   4. Recompute p_ij = exp(dot - lse_i)
  //   5. dS_ij = p_ij * (sum_d(dO * V[j]) - Di)
  //   6. dScore_ij = dS_ij * scale [* softcap_deriv]
  //   7. dQ += dScore_ij * K[j]

  for (let j = 0; j < Bc; j++) {
    // kPos = kBlockBase + j
    const kPos = b.id();
    b.emit(Op.IAdd, [p.tU32, kPos, kBlockBase, constJ[j]]);

    // ── Dot product: Q[qRow] · sK[j] (raw, before masking) ───────────────
    // Unrolled over D with FMA-style accumulation

    // Start with first element
    const ptrRegQ0 = b.id();
    b.emit(Op.AccessChain, [tPtrFnF32, ptrRegQ0, regQ, constDIdx[0]]);
    const qv0 = b.id();
    b.emit(Op.Load, [p.tF32, qv0, ptrRegQ0]);

    const sKIdx0 = b.id();
    b.emit(Op.IAdd, [p.tU32, sKIdx0, constJD[j], constDIdx[0]]);
    const ptrSK0 = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF32, ptrSK0, sK, sKIdx0]);
    const kv0 = b.id();
    b.emit(Op.Load, [p.tF32, kv0, ptrSK0]);
    let dotAcc = b.id();
    b.emit(Op.FMul, [p.tF32, dotAcc, qv0, kv0]);

    // Remaining D-1 elements
    for (let d = 1; d < D; d++) {
      const ptrRegQd = b.id();
      b.emit(Op.AccessChain, [tPtrFnF32, ptrRegQd, regQ, constDIdx[d]]);
      const qvd = b.id();
      b.emit(Op.Load, [p.tF32, qvd, ptrRegQd]);

      const sKIdxD = b.id();
      b.emit(Op.IAdd, [p.tU32, sKIdxD, constJD[j], constDIdx[d]]);
      const ptrSKd = b.id();
      b.emit(Op.AccessChain, [tPtrSharedF32, ptrSKd, sK, sKIdxD]);
      const kvd = b.id();
      b.emit(Op.Load, [p.tF32, kvd, ptrSKd]);

      const prod = b.id();
      b.emit(Op.FMul, [p.tF32, prod, qvd, kvd]);
      const newDotAcc = b.id();
      b.emit(Op.FAdd, [p.tF32, newDotAcc, dotAcc, prod]);
      dotAcc = newDotAcc;
    }

    // rawDotScaled = dot * scale (keep raw, before any masking)
    const rawDotScaled = b.id();
    b.emit(Op.FMul, [p.tF32, rawDotScaled, dotAcc, scale]);

    // ── Soft capping: tanh(dot / cap) * cap if enabled ────────────────────
    // Branchless: compute both, select based on softCapEnabled

    const dotDivCap = b.id();
    b.emit(Op.FMul, [p.tF32, dotDivCap, rawDotScaled, invSoftCap]);
    const tanhVal = b.id();
    b.emit(Op.ExtInst, [p.tF32, tanhVal, p.glslStd, GLSLstd450.Tanh, dotDivCap]);
    const dotCapped = b.id();
    b.emit(Op.FMul, [p.tF32, dotCapped, tanhVal, softCapValue]);
    const dotAfterCap = b.id();
    b.emit(Op.Select, [p.tF32, dotAfterCap, softCapEnabled, dotCapped, rawDotScaled]);

    // ── Causal mask + out-of-bounds ───────────────────────────────────────
    const oob = b.id();
    b.emit(Op.UGreaterThanEqual, [p.tBool, oob, kPos, T]);
    const causal = b.id();
    b.emit(Op.ULessThan, [p.tBool, causal, qRow, kPos]);
    const masked = b.id();
    b.emit(Op.LogicalOr, [p.tBool, masked, oob, causal]);
    const dot = b.id();
    b.emit(Op.Select, [p.tF32, dot, masked, constNegInf, dotAfterCap]);

    // ── Recompute attention weight from LSE ───────────────────────────────
    // p_ij = exp(dot - lse_i)
    const dotMinusLSE = b.id();
    b.emit(Op.FSub, [p.tF32, dotMinusLSE, dot, lse_i]);
    const p_ij = b.id();
    b.emit(Op.ExtInst, [p.tF32, p_ij, p.glslStd, GLSLstd450.Exp, dotMinusLSE]);

    // ── Compute dotDOV = sum_d(dO[d] * V[j,d]) ──────────────────────────

    // Start with first element
    const ptrRegDO0 = b.id();
    b.emit(Op.AccessChain, [tPtrFnF32, ptrRegDO0, regDO, constDIdx[0]]);
    const dov0 = b.id();
    b.emit(Op.Load, [p.tF32, dov0, ptrRegDO0]);

    const sVIdx0 = b.id();
    b.emit(Op.IAdd, [p.tU32, sVIdx0, constJD[j], constDIdx[0]]);
    const ptrSV0 = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF32, ptrSV0, sV, sVIdx0]);
    const vv0 = b.id();
    b.emit(Op.Load, [p.tF32, vv0, ptrSV0]);
    let dovAcc = b.id();
    b.emit(Op.FMul, [p.tF32, dovAcc, dov0, vv0]);

    // Remaining D-1 elements
    for (let d = 1; d < D; d++) {
      const ptrRegDOd = b.id();
      b.emit(Op.AccessChain, [tPtrFnF32, ptrRegDOd, regDO, constDIdx[d]]);
      const dovd = b.id();
      b.emit(Op.Load, [p.tF32, dovd, ptrRegDOd]);

      const sVIdxD = b.id();
      b.emit(Op.IAdd, [p.tU32, sVIdxD, constJD[j], constDIdx[d]]);
      const ptrSVd = b.id();
      b.emit(Op.AccessChain, [tPtrSharedF32, ptrSVd, sV, sVIdxD]);
      const vvd = b.id();
      b.emit(Op.Load, [p.tF32, vvd, ptrSVd]);

      const prodDOV = b.id();
      b.emit(Op.FMul, [p.tF32, prodDOV, dovd, vvd]);
      const newDovAcc = b.id();
      b.emit(Op.FAdd, [p.tF32, newDovAcc, dovAcc, prodDOV]);
      dovAcc = newDovAcc;
    }

    // ── dS_ij = p_ij * (dotDOV - Di) ─────────────────────────────────────
    const dovMinusDi = b.id();
    b.emit(Op.FSub, [p.tF32, dovMinusDi, dovAcc, Di]);
    const dS_ij = b.id();
    b.emit(Op.FMul, [p.tF32, dS_ij, p_ij, dovMinusDi]);

    // ── dScore_ij: apply scale and softcap derivative ─────────────────────
    // With softcap: dScore = dS * scale * (1 - tanh(raw/cap)^2)
    // Without:      dScore = dS * scale
    // Use raw (pre-mask) dot for softcap derivative to avoid NaN.
    // tanhVal was computed above from the raw dot, so reuse it.

    // tanhSq = tanhVal * tanhVal
    const tanhSq = b.id();
    b.emit(Op.FMul, [p.tF32, tanhSq, tanhVal, tanhVal]);
    // deriv = 1 - tanhSq
    const deriv = b.id();
    b.emit(Op.FSub, [p.tF32, deriv, const1f, tanhSq]);
    // dScore_capped = dS_ij * scale * deriv
    const dSscale = b.id();
    b.emit(Op.FMul, [p.tF32, dSscale, dS_ij, scale]);
    const dScore_capped = b.id();
    b.emit(Op.FMul, [p.tF32, dScore_capped, dSscale, deriv]);
    // dScore_uncapped = dS_ij * scale
    const dScore_uncapped = b.id();
    b.emit(Op.FMul, [p.tF32, dScore_uncapped, dS_ij, scale]);
    // dScore_ij = softCapEnabled ? dScore_capped : dScore_uncapped
    const dScore_ij = b.id();
    b.emit(Op.Select, [p.tF32, dScore_ij, softCapEnabled, dScore_capped, dScore_uncapped]);

    // ── Accumulate: dQ[d] += dScore_ij * K[j,d] ──────────────────────────
    for (let d = 0; d < D; d++) {
      // Load regDQ[d]
      const ptrDQ = b.id();
      b.emit(Op.AccessChain, [tPtrFnF32, ptrDQ, regDQ, constDIdx[d]]);
      const dqOld = b.id();
      b.emit(Op.Load, [p.tF32, dqOld, ptrDQ]);

      // Load sK[j * D + d]
      const sKIdx = b.id();
      b.emit(Op.IAdd, [p.tU32, sKIdx, constJD[j], constDIdx[d]]);
      const ptrSK = b.id();
      b.emit(Op.AccessChain, [tPtrSharedF32, ptrSK, sK, sKIdx]);
      const kk = b.id();
      b.emit(Op.Load, [p.tF32, kk, ptrSK]);

      // dScore_ij * K[j,d]
      const dScoreK = b.id();
      b.emit(Op.FMul, [p.tF32, dScoreK, dScore_ij, kk]);

      // regDQ[d] = dqOld + dScoreK
      const dqNew = b.id();
      b.emit(Op.FAdd, [p.tF32, dqNew, dqOld, dScoreK]);
      b.emit(Op.Store, [ptrDQ, dqNew]);
    }
  }

  // Barrier before next key block loads new sK/sV
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  // ── Loop continuation ─────────────────────────────────────────────────────

  b.emit(Op.Branch, [labelLoopCont]);
  b.emit(Op.Label, [labelLoopCont]);
  const nextKBlock = b.id();
  b.emit(Op.Load, [p.tU32, nextKBlock, varKBlockIdx]);
  const incKBlock = b.id();
  b.emit(Op.IAdd, [p.tU32, incKBlock, nextKBlock, p.const1u]);
  b.emit(Op.Store, [varKBlockIdx, incKBlock]);
  b.emit(Op.Branch, [labelLoopHead]);

  b.emit(Op.Label, [labelLoopMerge]);

  // ── Write dQ to output ─────────────────────────────────────────────────────
  // dQ[baseOff + qRow * D + d] = regDQ[d]

  const dqBase = b.id();
  b.emit(Op.IAdd, [p.tU32, dqBase, baseOff, qRowOff]);

  for (let d = 0; d < D; d++) {
    // Load regDQ[d]
    const ptrRegDQd = b.id();
    b.emit(Op.AccessChain, [tPtrFnF32, ptrRegDQd, regDQ, constDIdx[d]]);
    const regDQVal = b.id();
    b.emit(Op.Load, [p.tF32, regDQVal, ptrRegDQd]);

    // Write to output
    const dqIdx = b.id();
    b.emit(Op.IAdd, [p.tU32, dqIdx, dqBase, constDIdx[d]]);
    const ptrOut = b.id();
    b.emit(Op.AccessChain, [bufDQ.tPtrF32, ptrOut, bufDQ.varId, p.const0u, dqIdx]);
    b.emit(Op.Store, [ptrOut, regDQVal]);
  }

  // Branch to end
  b.emit(Op.Branch, [labelEnd]);

  // ── End label (early exit for OOB threads also lands here) ────────────────
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}

// ── Kernel: Flash Attention Backward dKV ────────────────────────────────────
//
// Computes dK and dV (gradients w.r.t. keys and values) using recomputed
// attention weights. Each workgroup processes one KEY block (Bc rows). Each
// thread handles one key row within the block. Query blocks are cooperatively
// loaded into shared memory, then attention gradients are accumulated.
//
// Bindings:
//   0: Q [B*H, T, D]       (readonly)
//   1: K [B*H, T, D]       (readonly)
//   2: V [B*H, T, D]       (readonly)
//   3: dO [B*H, T, D]      (readonly) — gradient of output
//   4: LSE [B*H, T]        (readonly) — log-sum-exp from forward
//   5: D_precomp [B*H, T]  (readonly) — D[i] = sum_d(dO[i,d] * O[i,d])
//   6: dK [B*H, T, D]      (write)   — output gradient for keys
//   7: dV [B*H, T, D]      (write)   — output gradient for values
//
// Push constants (16 bytes = 4 x f32): { T, scale, softCapValue, _pad }
//
// Dispatch: (ceil(T/Bc), B*H, 1)
// Workgroup: (Bc, 1, 1)

export function kernelFlashAttentionBackwardDKV(Br: number, Bc: number, D: number): Uint32Array {
  const b = new SpirVBuilder();
  const p = preamble(b, Bc, 1, 1);

  // Storage buffers: Q, K, V, dO, LSE, D_precomp (readonly), dK, dV (write)
  const bufQ      = declareStorageBuffer(b, p.tF32, p.tU32, 0, 0, true);
  const bufK      = declareStorageBuffer(b, p.tF32, p.tU32, 0, 1, true);
  const bufV      = declareStorageBuffer(b, p.tF32, p.tU32, 0, 2, true);
  const bufDO     = declareStorageBuffer(b, p.tF32, p.tU32, 0, 3, true);
  const bufLSE    = declareStorageBuffer(b, p.tF32, p.tU32, 0, 4, true);
  const bufDpre   = declareStorageBuffer(b, p.tF32, p.tU32, 0, 5, true);
  const bufDK     = declareStorageBuffer(b, p.tF32, p.tU32, 0, 6, false);
  const bufDV     = declareStorageBuffer(b, p.tF32, p.tU32, 0, 7, false);

  // Push constants: { T, scale, softCapValue, _pad }
  const pc = declareParamsPushConstant(b, p.tF32, 4);

  // ── Constants ──────────────────────────────────────────────────────────────

  const constBr = b.id();
  b.constant(p.tU32, constBr, Br);
  const constBc = b.id();
  b.constant(p.tU32, constBc, Bc);
  const constD = b.id();
  b.constant(p.tU32, constD, D);
  const constBrD = b.id();
  b.constant(p.tU32, constBrD, Br * D);
  const constBrMinus1 = b.id();
  b.constant(p.tU32, constBrMinus1, Br - 1);

  const constNegInf = b.id();
  b.constant(p.tF32, constNegInf, 0xFF800000); // -infinity (IEEE 754 raw bits)

  const const1f = b.id();
  b.constantF32(p.tF32, const1f, 1.0);

  // Pre-generate uint constants for unrolled i indices (over Br query positions)
  const constI: number[] = [];
  for (let i = 0; i < Br; i++) {
    const ci = b.id();
    b.constant(p.tU32, ci, i);
    constI.push(ci);
  }

  // Pre-generate uint constants for i * D offsets into shared memory
  const constID: number[] = [];
  for (let i = 0; i < Br; i++) {
    const cid = b.id();
    b.constant(p.tU32, cid, i * D);
    constID.push(cid);
  }

  // Pre-generate uint constants for d indices
  const constDIdx: number[] = [];
  for (let d = 0; d < D; d++) {
    const cd = b.id();
    b.constant(p.tU32, cd, d);
    constDIdx.push(cd);
  }

  // ── Shared memory: sQ[Br * D], sDO[Br * D], sLSE[Br], sDpre[Br] ─────────

  const constSharedSize = b.id();
  b.constant(p.tU32, constSharedSize, Br * D);
  const tArrayShared = b.id();
  b.typeArray(tArrayShared, p.tF32, constSharedSize);
  const tPtrSharedArr = b.id();
  b.typePointer(tPtrSharedArr, StorageClass.Workgroup, tArrayShared);
  const tPtrSharedF32 = b.id();
  b.typePointer(tPtrSharedF32, StorageClass.Workgroup, p.tF32);
  const sQ = b.id();
  b.variable(tPtrSharedArr, sQ, StorageClass.Workgroup);
  const sDO = b.id();
  b.variable(tPtrSharedArr, sDO, StorageClass.Workgroup);

  // Small shared arrays: sLSE[Br] and sDpre[Br]
  const constSharedSmallSize = b.id();
  b.constant(p.tU32, constSharedSmallSize, Br);
  const tArraySharedSmall = b.id();
  b.typeArray(tArraySharedSmall, p.tF32, constSharedSmallSize);
  const tPtrSharedSmallArr = b.id();
  b.typePointer(tPtrSharedSmallArr, StorageClass.Workgroup, tArraySharedSmall);
  const sLSE = b.id();
  b.variable(tPtrSharedSmallArr, sLSE, StorageClass.Workgroup);
  const sDpre = b.id();
  b.variable(tPtrSharedSmallArr, sDpre, StorageClass.Workgroup);

  // ── Function-scope array types: regK[D], regV[D], regDK[D], regDV[D] ─────

  const constDArr = b.id();
  b.constant(p.tU32, constDArr, D);
  const tArrayD = b.id();
  b.typeArray(tArrayD, p.tF32, constDArr);
  const tPtrFnArr = b.id();
  b.typePointer(tPtrFnArr, StorageClass.Function, tArrayD);
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

  // Push constant member index constants
  const const3u = b.id();
  b.constant(p.tU32, const3u, 3);

  // ── Entry point ───────────────────────────────────────────────────────────

  const fnMain = b.id();
  b.addEntryPoint(ExecutionModel.GLCompute, fnMain, "main", [p.vGlobalId, vWorkgroupId, vLocalId]);
  b.addExecutionMode(fnMain, ExecutionMode.LocalSize, Bc, 1, 1);

  b.emit(Op.Function, [p.tVoid, fnMain, FunctionControl.None, p.tFnVoid]);
  const labelEntry = b.id();
  b.emit(Op.Label, [labelEntry]);

  // ── All Variable declarations must be in the entry block ──────────────────

  const regK = b.id();
  b.emit(Op.Variable, [tPtrFnArr, regK, StorageClass.Function]);
  const regV = b.id();
  b.emit(Op.Variable, [tPtrFnArr, regV, StorageClass.Function]);
  const regDK = b.id();
  b.emit(Op.Variable, [tPtrFnArr, regDK, StorageClass.Function]);
  const regDV = b.id();
  b.emit(Op.Variable, [tPtrFnArr, regDV, StorageClass.Function]);
  const varQBlockIdx = b.id();
  b.emit(Op.Variable, [tPtrFnU32, varQBlockIdx, StorageClass.Function]);

  // ── Load thread/workgroup IDs ─────────────────────────────────────────────

  const lidVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, lidVec, vLocalId]);
  const threadIdx = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, threadIdx, lidVec, 0]);

  const wgIdVec = b.id();
  b.emit(Op.Load, [p.tVec3U32, wgIdVec, vWorkgroupId]);
  const kBlockIdx = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, kBlockIdx, wgIdVec, 0]);
  const bhIdx = b.id();
  b.emit(Op.CompositeExtract, [p.tU32, bhIdx, wgIdVec, 1]);

  // kRow = kBlockIdx * Bc + threadIdx
  const kBlockOff = b.id();
  b.emit(Op.IMul, [p.tU32, kBlockOff, kBlockIdx, constBc]);
  const kRow = b.id();
  b.emit(Op.IAdd, [p.tU32, kRow, kBlockOff, threadIdx]);

  // ── Load push constants ───────────────────────────────────────────────────

  // T (sequence length)
  const ptrTpc = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrTpc, pc.varId, p.const0u]);
  const TF = b.id();
  b.emit(Op.Load, [p.tF32, TF, ptrTpc]);
  const T = b.id();
  b.emit(Op.ConvertFToU, [p.tU32, T, TF]);

  // scale (1/sqrt(D))
  const ptrScale = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrScale, pc.varId, p.const1u]);
  const scale = b.id();
  b.emit(Op.Load, [p.tF32, scale, ptrScale]);

  // softCapValue
  const ptrSoftCap = b.id();
  b.emit(Op.AccessChain, [pc.tPtrF32, ptrSoftCap, pc.varId, p.const2u]);
  const softCapValue = b.id();
  b.emit(Op.Load, [p.tF32, softCapValue, ptrSoftCap]);

  // Check if softcap is enabled (softCapValue > 0)
  const softCapEnabled = b.id();
  b.emit(Op.FOrdGreaterThan, [p.tBool, softCapEnabled, softCapValue, p.const0f]);

  // Precompute 1/softCapValue for softcap (safe: if 0, we never use it)
  const invSoftCap = b.id();
  b.emit(Op.FDiv, [p.tF32, invSoftCap, const1f, softCapValue]);

  // ── Compute base offsets ──────────────────────────────────────────────────
  // baseOff = bhIdx * T * D  (for Q/K/V/dO/dK/dV indexing)
  // lseBaseOff = bhIdx * T   (for LSE/D_precomp indexing)

  const TD = b.id();
  b.emit(Op.IMul, [p.tU32, TD, T, constD]);
  const baseOff = b.id();
  b.emit(Op.IMul, [p.tU32, baseOff, bhIdx, TD]);

  const lseBaseOff = b.id();
  b.emit(Op.IMul, [p.tU32, lseBaseOff, bhIdx, T]);

  // ── Bounds check: if kRow >= T, skip to end ───────────────────────────────

  const kRowOob = b.id();
  b.emit(Op.UGreaterThanEqual, [p.tBool, kRowOob, kRow, T]);
  const labelMain = b.id();
  const labelEnd = b.id();
  b.emit(Op.SelectionMerge, [labelEnd, 0]);
  b.emit(Op.BranchConditional, [kRowOob, labelEnd, labelMain]);

  b.emit(Op.Label, [labelMain]);

  // ── Load K[kRow] into regK ────────────────────────────────────────────────

  const kRowOff = b.id();
  b.emit(Op.IMul, [p.tU32, kRowOff, kRow, constD]);
  const kBase = b.id();
  b.emit(Op.IAdd, [p.tU32, kBase, baseOff, kRowOff]);

  for (let d = 0; d < D; d++) {
    const kIdx = b.id();
    b.emit(Op.IAdd, [p.tU32, kIdx, kBase, constDIdx[d]]);
    const ptrKElem = b.id();
    b.emit(Op.AccessChain, [bufK.tPtrF32, ptrKElem, bufK.varId, p.const0u, kIdx]);
    const kVal = b.id();
    b.emit(Op.Load, [p.tF32, kVal, ptrKElem]);
    const ptrRegK = b.id();
    b.emit(Op.AccessChain, [tPtrFnF32, ptrRegK, regK, constDIdx[d]]);
    b.emit(Op.Store, [ptrRegK, kVal]);
  }

  // ── Load V[kRow] into regV ────────────────────────────────────────────────

  for (let d = 0; d < D; d++) {
    const vIdx = b.id();
    b.emit(Op.IAdd, [p.tU32, vIdx, kBase, constDIdx[d]]);
    const ptrVElem = b.id();
    b.emit(Op.AccessChain, [bufV.tPtrF32, ptrVElem, bufV.varId, p.const0u, vIdx]);
    const vVal = b.id();
    b.emit(Op.Load, [p.tF32, vVal, ptrVElem]);
    const ptrRegV = b.id();
    b.emit(Op.AccessChain, [tPtrFnF32, ptrRegV, regV, constDIdx[d]]);
    b.emit(Op.Store, [ptrRegV, vVal]);
  }

  // ── Initialize regDK and regDV to 0.0 ─────────────────────────────────────

  for (let d = 0; d < D; d++) {
    const ptrRegDK = b.id();
    b.emit(Op.AccessChain, [tPtrFnF32, ptrRegDK, regDK, constDIdx[d]]);
    b.emit(Op.Store, [ptrRegDK, p.const0f]);
    const ptrRegDV = b.id();
    b.emit(Op.AccessChain, [tPtrFnF32, ptrRegDV, regDV, constDIdx[d]]);
    b.emit(Op.Store, [ptrRegDV, p.const0f]);
  }

  // ── Compute numQBlocks = ceil(T / Br) ─────────────────────────────────────
  // numQBlocks = (T + Br - 1) / Br

  const TplusBrm1 = b.id();
  b.emit(Op.IAdd, [p.tU32, TplusBrm1, T, constBrMinus1]);
  const numQBlocks = b.id();
  b.emit(Op.UDiv, [p.tU32, numQBlocks, TplusBrm1, constBr]);

  // ── Outer loop: for qBlockIdx = kBlockIdx; qBlockIdx < numQBlocks; qBlockIdx++ ──

  b.emit(Op.Store, [varQBlockIdx, kBlockIdx]);

  const labelLoopHead = b.id();
  const labelLoopBody = b.id();
  const labelLoopMerge = b.id();
  const labelLoopCont = b.id();

  b.emit(Op.Branch, [labelLoopHead]);
  b.emit(Op.Label, [labelLoopHead]);
  const qBlockIdx = b.id();
  b.emit(Op.Load, [p.tU32, qBlockIdx, varQBlockIdx]);
  const loopCmp = b.id();
  b.emit(Op.ULessThan, [p.tBool, loopCmp, qBlockIdx, numQBlocks]);
  b.emit(Op.LoopMerge, [labelLoopMerge, labelLoopCont, 0]);
  b.emit(Op.BranchConditional, [loopCmp, labelLoopBody, labelLoopMerge]);

  b.emit(Op.Label, [labelLoopBody]);

  // ── Cooperative load: each thread loads one row of Q, dO, LSE, D_precomp ──
  // qRow = qBlockIdx * Br + threadIdx

  const qBlockBase = b.id();
  b.emit(Op.IMul, [p.tU32, qBlockBase, qBlockIdx, constBr]);
  const qRow = b.id();
  b.emit(Op.IAdd, [p.tU32, qRow, qBlockBase, threadIdx]);

  // qRowInBounds = qRow < T
  const qRowInBounds = b.id();
  b.emit(Op.ULessThan, [p.tBool, qRowInBounds, qRow, T]);

  // qGlobalBase = baseOff + qRow * D
  const qRowOff = b.id();
  b.emit(Op.IMul, [p.tU32, qRowOff, qRow, constD]);
  const qGlobalBase = b.id();
  b.emit(Op.IAdd, [p.tU32, qGlobalBase, baseOff, qRowOff]);

  // threadIdx * D = offset into shared memory for this thread's row
  const sharedRowOff = b.id();
  b.emit(Op.IMul, [p.tU32, sharedRowOff, threadIdx, constD]);

  // Load D elements for Q and dO
  for (let d = 0; d < D; d++) {
    // Global index into Q/dO buffer
    const gIdx = b.id();
    b.emit(Op.IAdd, [p.tU32, gIdx, qGlobalBase, constDIdx[d]]);

    // Load Q element (0 if out of bounds)
    const ptrQElem = b.id();
    b.emit(Op.AccessChain, [bufQ.tPtrF32, ptrQElem, bufQ.varId, p.const0u, gIdx]);
    const qRaw = b.id();
    b.emit(Op.Load, [p.tF32, qRaw, ptrQElem]);
    const qVal = b.id();
    b.emit(Op.Select, [p.tF32, qVal, qRowInBounds, qRaw, p.const0f]);

    // Store to sQ[threadIdx * D + d]
    const sQIdx = b.id();
    b.emit(Op.IAdd, [p.tU32, sQIdx, sharedRowOff, constDIdx[d]]);
    const ptrSQ = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF32, ptrSQ, sQ, sQIdx]);
    b.emit(Op.Store, [ptrSQ, qVal]);

    // Load dO element (0 if out of bounds)
    const ptrDOElem = b.id();
    b.emit(Op.AccessChain, [bufDO.tPtrF32, ptrDOElem, bufDO.varId, p.const0u, gIdx]);
    const doRaw = b.id();
    b.emit(Op.Load, [p.tF32, doRaw, ptrDOElem]);
    const doVal = b.id();
    b.emit(Op.Select, [p.tF32, doVal, qRowInBounds, doRaw, p.const0f]);

    // Store to sDO[threadIdx * D + d]
    const ptrSDO = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF32, ptrSDO, sDO, sQIdx]); // same index
    b.emit(Op.Store, [ptrSDO, doVal]);
  }

  // Load LSE and D_precomp for this thread's query row
  const lseQRowIdx = b.id();
  b.emit(Op.IAdd, [p.tU32, lseQRowIdx, lseBaseOff, qRow]);

  const ptrLSEi = b.id();
  b.emit(Op.AccessChain, [bufLSE.tPtrF32, ptrLSEi, bufLSE.varId, p.const0u, lseQRowIdx]);
  const lseRaw = b.id();
  b.emit(Op.Load, [p.tF32, lseRaw, ptrLSEi]);
  const lseVal = b.id();
  b.emit(Op.Select, [p.tF32, lseVal, qRowInBounds, lseRaw, p.const0f]);
  const ptrSLSE = b.id();
  b.emit(Op.AccessChain, [tPtrSharedF32, ptrSLSE, sLSE, threadIdx]);
  b.emit(Op.Store, [ptrSLSE, lseVal]);

  const ptrDpreI = b.id();
  b.emit(Op.AccessChain, [bufDpre.tPtrF32, ptrDpreI, bufDpre.varId, p.const0u, lseQRowIdx]);
  const dpreRaw = b.id();
  b.emit(Op.Load, [p.tF32, dpreRaw, ptrDpreI]);
  const dpreVal = b.id();
  b.emit(Op.Select, [p.tF32, dpreVal, qRowInBounds, dpreRaw, p.const0f]);
  const ptrSDpre = b.id();
  b.emit(Op.AccessChain, [tPtrSharedF32, ptrSDpre, sDpre, threadIdx]);
  b.emit(Op.Store, [ptrSDpre, dpreVal]);

  // Barrier — all threads have loaded their Q/dO/LSE/D_precomp rows
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  // ── Unrolled loop over Br queries: recompute attention + accumulate dK/dV ─
  // For each i = 0..Br-1:
  //   1. Compute raw dot = K[kRow] · Q[qPos] * scale (before masking)
  //   2. Apply softcap to get capped value
  //   3. Apply causal mask (qPos < kRow or qPos >= T → -inf)
  //   4. Recompute p_ij = exp(dot - sLSE[i])
  //   5. dV += p_ij * dO[i]
  //   6. dS_ij = p_ij * (sum_d(dO[i,d] * V[kRow,d]) - sDpre[i])
  //   7. dScore_ij = dS_ij * scale [* softcap_deriv]
  //   8. dK += dScore_ij * Q[i]

  for (let i = 0; i < Br; i++) {
    // qPos = qBlockBase + i
    const qPos = b.id();
    b.emit(Op.IAdd, [p.tU32, qPos, qBlockBase, constI[i]]);

    // ── Dot product: K[kRow] · sQ[i] (raw, before masking) ───────────────
    // Unrolled over D with FMA-style accumulation

    // Start with first element
    const ptrRegK0 = b.id();
    b.emit(Op.AccessChain, [tPtrFnF32, ptrRegK0, regK, constDIdx[0]]);
    const kv0 = b.id();
    b.emit(Op.Load, [p.tF32, kv0, ptrRegK0]);

    const sQIdx0 = b.id();
    b.emit(Op.IAdd, [p.tU32, sQIdx0, constID[i], constDIdx[0]]);
    const ptrSQ0 = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF32, ptrSQ0, sQ, sQIdx0]);
    const qv0 = b.id();
    b.emit(Op.Load, [p.tF32, qv0, ptrSQ0]);
    let dotAcc = b.id();
    b.emit(Op.FMul, [p.tF32, dotAcc, kv0, qv0]);

    // Remaining D-1 elements
    for (let d = 1; d < D; d++) {
      const ptrRegKd = b.id();
      b.emit(Op.AccessChain, [tPtrFnF32, ptrRegKd, regK, constDIdx[d]]);
      const kvd = b.id();
      b.emit(Op.Load, [p.tF32, kvd, ptrRegKd]);

      const sQIdxD = b.id();
      b.emit(Op.IAdd, [p.tU32, sQIdxD, constID[i], constDIdx[d]]);
      const ptrSQd = b.id();
      b.emit(Op.AccessChain, [tPtrSharedF32, ptrSQd, sQ, sQIdxD]);
      const qvd = b.id();
      b.emit(Op.Load, [p.tF32, qvd, ptrSQd]);

      const prod = b.id();
      b.emit(Op.FMul, [p.tF32, prod, kvd, qvd]);
      const newDotAcc = b.id();
      b.emit(Op.FAdd, [p.tF32, newDotAcc, dotAcc, prod]);
      dotAcc = newDotAcc;
    }

    // rawDotScaled = dot * scale (keep raw, before any masking)
    const rawDotScaled = b.id();
    b.emit(Op.FMul, [p.tF32, rawDotScaled, dotAcc, scale]);

    // ── Soft capping: tanh(dot / cap) * cap if enabled ────────────────────
    // Branchless: compute both, select based on softCapEnabled

    const dotDivCap = b.id();
    b.emit(Op.FMul, [p.tF32, dotDivCap, rawDotScaled, invSoftCap]);
    const tanhVal = b.id();
    b.emit(Op.ExtInst, [p.tF32, tanhVal, p.glslStd, GLSLstd450.Tanh, dotDivCap]);
    const dotCapped = b.id();
    b.emit(Op.FMul, [p.tF32, dotCapped, tanhVal, softCapValue]);
    const dotAfterCap = b.id();
    b.emit(Op.Select, [p.tF32, dotAfterCap, softCapEnabled, dotCapped, rawDotScaled]);

    // ── Causal mask + out-of-bounds ───────────────────────────────────────
    // masked = (qPos >= T) || (qPos < kRow)  → set dot to -inf
    const oob = b.id();
    b.emit(Op.UGreaterThanEqual, [p.tBool, oob, qPos, T]);
    const causal = b.id();
    b.emit(Op.ULessThan, [p.tBool, causal, qPos, kRow]); // qPos < kRow ≡ key after query
    const masked = b.id();
    b.emit(Op.LogicalOr, [p.tBool, masked, oob, causal]);
    const dot = b.id();
    b.emit(Op.Select, [p.tF32, dot, masked, constNegInf, dotAfterCap]);

    // ── Recompute attention weight from LSE ─────────────────────────────
    // p_ij = exp(dot - sLSE[i])
    const ptrSLSEi = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF32, ptrSLSEi, sLSE, constI[i]]);
    const lse_i = b.id();
    b.emit(Op.Load, [p.tF32, lse_i, ptrSLSEi]);
    const dotMinusLSE = b.id();
    b.emit(Op.FSub, [p.tF32, dotMinusLSE, dot, lse_i]);
    const p_ij = b.id();
    b.emit(Op.ExtInst, [p.tF32, p_ij, p.glslStd, GLSLstd450.Exp, dotMinusLSE]);

    // ── dV accumulation: dV += p_ij * dO[i] ─────────────────────────────
    for (let d = 0; d < D; d++) {
      // Load regDV[d]
      const ptrDV = b.id();
      b.emit(Op.AccessChain, [tPtrFnF32, ptrDV, regDV, constDIdx[d]]);
      const dvOld = b.id();
      b.emit(Op.Load, [p.tF32, dvOld, ptrDV]);

      // Load sDO[i * D + d]
      const sDOIdx = b.id();
      b.emit(Op.IAdd, [p.tU32, sDOIdx, constID[i], constDIdx[d]]);
      const ptrSDOd = b.id();
      b.emit(Op.AccessChain, [tPtrSharedF32, ptrSDOd, sDO, sDOIdx]);
      const doVal = b.id();
      b.emit(Op.Load, [p.tF32, doVal, ptrSDOd]);

      // p_ij * dO[i,d]
      const pDO = b.id();
      b.emit(Op.FMul, [p.tF32, pDO, p_ij, doVal]);

      // regDV[d] = dvOld + pDO
      const dvNew = b.id();
      b.emit(Op.FAdd, [p.tF32, dvNew, dvOld, pDO]);
      b.emit(Op.Store, [ptrDV, dvNew]);
    }

    // ── Compute dotDOV = sum_d(dO[i,d] * V[kRow,d]) ────────────────────

    // Start with first element
    const sDOIdx0 = b.id();
    b.emit(Op.IAdd, [p.tU32, sDOIdx0, constID[i], constDIdx[0]]);
    const ptrSDO0 = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF32, ptrSDO0, sDO, sDOIdx0]);
    const dov0 = b.id();
    b.emit(Op.Load, [p.tF32, dov0, ptrSDO0]);

    const ptrRegV0 = b.id();
    b.emit(Op.AccessChain, [tPtrFnF32, ptrRegV0, regV, constDIdx[0]]);
    const vv0 = b.id();
    b.emit(Op.Load, [p.tF32, vv0, ptrRegV0]);
    let dovAcc = b.id();
    b.emit(Op.FMul, [p.tF32, dovAcc, dov0, vv0]);

    // Remaining D-1 elements
    for (let d = 1; d < D; d++) {
      const sDOIdxD = b.id();
      b.emit(Op.IAdd, [p.tU32, sDOIdxD, constID[i], constDIdx[d]]);
      const ptrSDOd = b.id();
      b.emit(Op.AccessChain, [tPtrSharedF32, ptrSDOd, sDO, sDOIdxD]);
      const dovd = b.id();
      b.emit(Op.Load, [p.tF32, dovd, ptrSDOd]);

      const ptrRegVd = b.id();
      b.emit(Op.AccessChain, [tPtrFnF32, ptrRegVd, regV, constDIdx[d]]);
      const vvd = b.id();
      b.emit(Op.Load, [p.tF32, vvd, ptrRegVd]);

      const prodDOV = b.id();
      b.emit(Op.FMul, [p.tF32, prodDOV, dovd, vvd]);
      const newDovAcc = b.id();
      b.emit(Op.FAdd, [p.tF32, newDovAcc, dovAcc, prodDOV]);
      dovAcc = newDovAcc;
    }

    // ── dS_ij = p_ij * (dotDOV - sDpre[i]) ─────────────────────────────
    const ptrSDpreI = b.id();
    b.emit(Op.AccessChain, [tPtrSharedF32, ptrSDpreI, sDpre, constI[i]]);
    const dpreI = b.id();
    b.emit(Op.Load, [p.tF32, dpreI, ptrSDpreI]);
    const dovMinusDi = b.id();
    b.emit(Op.FSub, [p.tF32, dovMinusDi, dovAcc, dpreI]);
    const dS_ij = b.id();
    b.emit(Op.FMul, [p.tF32, dS_ij, p_ij, dovMinusDi]);

    // ── dScore_ij: apply scale and softcap derivative ─────────────────────
    // With softcap: dScore = dS * scale * (1 - tanh(raw/cap)^2)
    // Without:      dScore = dS * scale
    // Use raw (pre-mask) dot for softcap derivative to avoid NaN.
    // tanhVal was computed above from the raw dot, so reuse it.

    // tanhSq = tanhVal * tanhVal
    const tanhSq = b.id();
    b.emit(Op.FMul, [p.tF32, tanhSq, tanhVal, tanhVal]);
    // deriv = 1 - tanhSq
    const deriv = b.id();
    b.emit(Op.FSub, [p.tF32, deriv, const1f, tanhSq]);
    // dScore_capped = dS_ij * scale * deriv
    const dSscale = b.id();
    b.emit(Op.FMul, [p.tF32, dSscale, dS_ij, scale]);
    const dScore_capped = b.id();
    b.emit(Op.FMul, [p.tF32, dScore_capped, dSscale, deriv]);
    // dScore_uncapped = dS_ij * scale
    const dScore_uncapped = b.id();
    b.emit(Op.FMul, [p.tF32, dScore_uncapped, dS_ij, scale]);
    // dScore_ij = softCapEnabled ? dScore_capped : dScore_uncapped
    const dScore_ij = b.id();
    b.emit(Op.Select, [p.tF32, dScore_ij, softCapEnabled, dScore_capped, dScore_uncapped]);

    // ── Accumulate: dK[d] += dScore_ij * Q[i,d] ──────────────────────────
    for (let d = 0; d < D; d++) {
      // Load regDK[d]
      const ptrDK = b.id();
      b.emit(Op.AccessChain, [tPtrFnF32, ptrDK, regDK, constDIdx[d]]);
      const dkOld = b.id();
      b.emit(Op.Load, [p.tF32, dkOld, ptrDK]);

      // Load sQ[i * D + d]
      const sQIdx = b.id();
      b.emit(Op.IAdd, [p.tU32, sQIdx, constID[i], constDIdx[d]]);
      const ptrSQ = b.id();
      b.emit(Op.AccessChain, [tPtrSharedF32, ptrSQ, sQ, sQIdx]);
      const qq = b.id();
      b.emit(Op.Load, [p.tF32, qq, ptrSQ]);

      // dScore_ij * Q[i,d]
      const dScoreQ = b.id();
      b.emit(Op.FMul, [p.tF32, dScoreQ, dScore_ij, qq]);

      // regDK[d] = dkOld + dScoreQ
      const dkNew = b.id();
      b.emit(Op.FAdd, [p.tF32, dkNew, dkOld, dScoreQ]);
      b.emit(Op.Store, [ptrDK, dkNew]);
    }
  }

  // Barrier before next query block loads new sQ/sDO/sLSE/sDpre
  b.emit(Op.ControlBarrier, [scopeWg, scopeWg, semAcqRelWg]);

  // ── Loop continuation ─────────────────────────────────────────────────────

  b.emit(Op.Branch, [labelLoopCont]);
  b.emit(Op.Label, [labelLoopCont]);
  const nextQBlock = b.id();
  b.emit(Op.Load, [p.tU32, nextQBlock, varQBlockIdx]);
  const incQBlock = b.id();
  b.emit(Op.IAdd, [p.tU32, incQBlock, nextQBlock, p.const1u]);
  b.emit(Op.Store, [varQBlockIdx, incQBlock]);
  b.emit(Op.Branch, [labelLoopHead]);

  b.emit(Op.Label, [labelLoopMerge]);

  // ── Write dK and dV to output ─────────────────────────────────────────────
  // dK[baseOff + kRow * D + d] = regDK[d]
  // dV[baseOff + kRow * D + d] = regDV[d]

  const dkBase = b.id();
  b.emit(Op.IAdd, [p.tU32, dkBase, baseOff, kRowOff]);

  for (let d = 0; d < D; d++) {
    // Load regDK[d]
    const ptrRegDKd = b.id();
    b.emit(Op.AccessChain, [tPtrFnF32, ptrRegDKd, regDK, constDIdx[d]]);
    const regDKVal = b.id();
    b.emit(Op.Load, [p.tF32, regDKVal, ptrRegDKd]);

    // Write to dK output
    const dkIdx = b.id();
    b.emit(Op.IAdd, [p.tU32, dkIdx, dkBase, constDIdx[d]]);
    const ptrOutDK = b.id();
    b.emit(Op.AccessChain, [bufDK.tPtrF32, ptrOutDK, bufDK.varId, p.const0u, dkIdx]);
    b.emit(Op.Store, [ptrOutDK, regDKVal]);

    // Load regDV[d]
    const ptrRegDVd = b.id();
    b.emit(Op.AccessChain, [tPtrFnF32, ptrRegDVd, regDV, constDIdx[d]]);
    const regDVVal = b.id();
    b.emit(Op.Load, [p.tF32, regDVVal, ptrRegDVd]);

    // Write to dV output
    const dvIdx = b.id();
    b.emit(Op.IAdd, [p.tU32, dvIdx, dkBase, constDIdx[d]]);
    const ptrOutDV = b.id();
    b.emit(Op.AccessChain, [bufDV.tPtrF32, ptrOutDV, bufDV.varId, p.const0u, dvIdx]);
    b.emit(Op.Store, [ptrOutDV, regDVVal]);
  }

  // Branch to end
  b.emit(Op.Branch, [labelEnd]);

  // ── End label (early exit for OOB threads also lands here) ────────────────
  b.emit(Op.Label, [labelEnd]);
  b.emit(Op.Return, []);
  b.emit(Op.FunctionEnd, []);

  return b.build();
}
