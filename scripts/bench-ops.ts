/**
 * Comprehensive Helios vs CUDA benchmark for ALL Alpha training operations.
 *
 * Tests every operation type at the exact shapes used in the 300M GPT training
 * pipeline. Compares Helios (Vulkan) against PyTorch CUDA on each.
 *
 * Usage:
 *   npx tsx scripts/bench-ops.ts [--iters=30] [--warmup=8] [--python=python3] [--cuda-json=path]
 *
 * Use --cuda-json to load pre-computed CUDA results (avoids GPU memory conflicts).
 * Generate CUDA JSON: python3 scripts/bench-ops-cuda.py --iters=30 --warmup=8 > cuda.json
 */
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import path from "node:path";
import { HeliosBackend } from "@alpha/helios";
import type { TensorData } from "@alpha/core";

// ── Config ──────────────────────────────────────────────────────────────────

interface CliOpts {
  iters: number;
  warmup: number;
  python: string;
  cudaJson: string;
}

function parseArgs(): CliOpts {
  const kv: Record<string, string> = {};
  for (const arg of process.argv.slice(2)) {
    if (!arg.startsWith("--")) continue;
    const eq = arg.indexOf("=");
    if (eq > 0) kv[arg.slice(2, eq)] = arg.slice(eq + 1);
    else kv[arg.slice(2)] = "true";
  }
  return {
    iters: Math.max(1, parseInt(kv.iters ?? "30", 10)),
    warmup: Math.max(0, parseInt(kv.warmup ?? "8", 10)),
    python: kv.python ?? "python3",
    cudaJson: kv["cuda-json"] ?? "",
  };
}

// ── Timing helpers ──────────────────────────────────────────────────────────

interface OpResult {
  ms: number;
  tflops?: number;
  gbps?: number;
  note?: string;
}

function median(arr: number[]): number {
  const s = arr.slice().sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

// ── Main ────────────────────────────────────────────────────────────────────

function main(): void {
  const opts = parseArgs();
  const b = new HeliosBackend();
  b.setMinGpuSize(0);

  const bAny = b as any;
  const syncFn: (() => void) | null =
    typeof bAny.syncGpu === "function" ? bAny.syncGpu.bind(bAny) : null;
  const flushFn: (() => void) | null =
    typeof bAny.flush === "function" ? bAny.flush.bind(bAny) : null;
  const releaseFn: ((td: TensorData) => void) | null =
    typeof bAny.releaseGpuTensor === "function" ? bAny.releaseGpuTensor.bind(bAny) : null;

  const info = typeof bAny.getDeviceInfo === "function" ? bAny.getDeviceInfo() : {};

  function sync(): void {
    if (syncFn) syncFn();
  }
  function flush(): void {
    if (flushFn) flushFn();
  }
  function release(td: TensorData): void {
    if (releaseFn) releaseFn(td);
  }

  // Training shapes: 300M model (batch=1, block=512, dim=1024, heads=16, ffn=2752, vocab=64000)
  const B = 1, T = 512, D = 1024, H = 16, Dh = 64;
  const FFN = 2752;
  const V = 64000;
  const BT = B * T;
  const BH = B * H;

  const results: Record<string, OpResult> = {};

  function record(name: string, ms: number, opts?: { flops?: number; bytes?: number; note?: string }): void {
    const r: OpResult = { ms: Math.round(ms * 10000) / 10000 };
    if (opts?.flops && opts.flops > 0) r.tflops = Math.round((opts.flops / (ms / 1000)) / 1e12 * 1000) / 1000;
    if (opts?.bytes && opts.bytes > 0) r.gbps = Math.round((opts.bytes / (ms / 1000)) / 1e9 * 10) / 10;
    if (opts?.note) r.note = opts.note;
    results[name] = r;
  }

  /** Bench a single-dispatch op. Returns median ms. */
  function benchOp(fn: () => TensorData): number {
    // Warmup
    for (let i = 0; i < opts.warmup; i++) {
      const r = fn();
      sync();
      release(r);
    }
    // Timed (per-op sync for accurate median)
    const times: number[] = [];
    for (let i = 0; i < opts.iters; i++) {
      const t0 = performance.now();
      const r = fn();
      sync();
      times.push(performance.now() - t0);
      release(r);
    }
    return median(times);
  }

  /** Bench a function that does its own GPU work, returns cleanup list. */
  function benchCustom(fn: () => TensorData[]): number {
    for (let i = 0; i < opts.warmup; i++) {
      const rs = fn();
      sync();
      for (const r of rs) release(r);
    }
    const times: number[] = [];
    for (let i = 0; i < opts.iters; i++) {
      const t0 = performance.now();
      const rs = fn();
      sync();
      times.push(performance.now() - t0);
      for (const r of rs) release(r);
    }
    return median(times);
  }

  console.error(`[bench] Helios device: ${info.deviceName ?? "unknown"} coopMat=${info.coopMatSupported ?? false}`);
  console.error(`[bench] iters=${opts.iters} warmup=${opts.warmup}`);

  // ── 1. MATMUL (training shapes: matmulTransposed = A @ B^T) ─────────────

  const matmulShapes: [string, number, number, number, string][] = [
    ["matmul_qkv",       BT, 3072, 1024, "QKV projection"],
    ["matmul_attn_out",  BT, 1024, 1024, "attention output"],
    ["matmul_swiglu_g",  BT, 2752, 1024, "SwiGLU gate"],
    ["matmul_swiglu_u",  BT, 2752, 1024, "SwiGLU up"],
    ["matmul_swiglu_p",  BT, 1024, 2752, "SwiGLU proj"],
    ["matmul_lm_head",   BT, V,    1024, "LM head"],
  ];

  const hasMatmulTransposed = typeof (b as any).matmulTransposed === "function";
  const hasMatmulTransposedA = typeof (b as any).matmulTransposedA === "function";

  for (const [name, M, N, K, note] of matmulShapes) {
    const a = b.randn([M, K]);
    const w = b.randn([N, K]); // stored transposed, as in training
    // Use native matmulTransposed (A @ W^T) — what training actually calls
    const ms = hasMatmulTransposed
      ? benchOp(() => (b as any).matmulTransposed(a, w))
      : benchOp(() => b.matmul(a, (b as any).transpose(w, 0, 1)));
    record(name, ms, { flops: 2 * M * N * K, note });
    release(a);
    release(w);
  }

  // Backward matmuls (weight grads: A^T @ dOut → matmulTransposedA)
  const bwdShapes: [string, number, number, number, string][] = [
    ["matmul_bwd_qkv",      1024, 3072, BT, "QKV weight grad"],
    ["matmul_bwd_attn_out",  1024, 1024, BT, "attn out weight grad"],
    ["matmul_bwd_swiglu_g",  1024, 2752, BT, "SwiGLU gate weight grad"],
    ["matmul_bwd_lm_head",   1024, V,    BT, "LM head weight grad"],
  ];

  for (const [name, M, N, K, note] of bwdShapes) {
    // A^T @ B: A is [K, M], B is [K, N] → result [M, N]
    const a = b.randn([K, M]);
    const dout = b.randn([K, N]);
    const ms = hasMatmulTransposedA
      ? benchOp(() => (b as any).matmulTransposedA(a, dout))
      : benchOp(() => {
          const at = (b as any).transpose(a, 0, 1);
          const r = b.matmul(at, dout);
          release(at);
          return r;
        });
    record(name, ms, { flops: 2 * M * N * K, note });
    release(a);
    release(dout);
  }

  // Square matmuls for reference
  for (const sz of [1024, 2048, 3072, 4096]) {
    const a = b.randn([sz, sz]);
    const bm = b.randn([sz, sz]);
    const ms = benchOp(() => b.matmul(a, bm));
    record(`matmul_${sz}sq`, ms, { flops: 2 * sz ** 3, note: `${sz}x${sz}x${sz}` });
    release(a);
    release(bm);
  }

  // ── 2. ELEMENT-WISE OPS ────────────────────────────────────────────────────

  {
    const x = b.randn([BT, D]);
    const y = b.randn([BT, D]);
    const x2 = b.randn([BT, FFN]);
    const sz1024 = BT * D;
    const sz2752 = BT * FFN;

    let ms = benchOp(() => b.add(x, y));
    record("add_512x1024", ms, { bytes: sz1024 * 4 * 3, note: "a+b" });

    ms = benchOp(() => b.mul(x, y));
    record("mul_512x1024", ms, { bytes: sz1024 * 4 * 3, note: "a*b" });

    ms = benchOp(() => (b as any).gelu(x));
    record("gelu_512x1024", ms, { bytes: sz1024 * 4 * 2, note: "GELU activation" });

    ms = benchOp(() => (b as any).silu(x2));
    record("silu_512x2752", ms, { bytes: sz2752 * 4 * 2, note: "SiLU activation" });

    ms = benchOp(() => (b as any).scale(x, 0.125));
    record("scale_512x1024", ms, { bytes: sz1024 * 4 * 2, note: "scalar multiply" });

    ms = benchOp(() => (b as any).neg(x));
    record("neg_512x1024", ms, { bytes: sz1024 * 4 * 2, note: "negate" });

    ms = benchOp(() => (b as any).exp(x));
    record("exp_512x1024", ms, { bytes: sz1024 * 4 * 2, note: "exp" });

    // Activation backward ops (fused kernels)
    if (typeof (b as any).geluBackward === "function") {
      ms = benchOp(() => (b as any).geluBackward(x, y));
      record("gelu_bwd_512x1024", ms, { bytes: sz1024 * 4 * 3, note: "GELU backward (fused)" });
    }

    // SiLU backward (used in SwiGLU)
    const xSilu = b.randn([512, 2752]);
    const ySilu = b.randn([512, 2752]);
    if (typeof (b as any).siluBackward === "function") {
      ms = benchOp(() => (b as any).siluBackward(xSilu, ySilu));
      record("silu_bwd_512x2752", ms, { bytes: sz2752 * 4 * 3, note: "SiLU backward (fused)" });
    }
    release(xSilu);
    release(ySilu);

    release(x);
    release(y);
    release(x2);
  }

  // Large tensor bandwidth test
  {
    const big = b.randn([4096, 4096]);
    const big2 = b.randn([4096, 4096]);
    const sz = 4096 * 4096;

    let ms = benchOp(() => b.add(big, big2));
    record("add_4096x4096", ms, { bytes: sz * 4 * 3, note: "large add bandwidth test" });

    ms = benchOp(() => (b as any).scale(big, 2.0));
    record("scale_4096x4096", ms, { bytes: sz * 4 * 2, note: "large scale bandwidth test" });

    release(big);
    release(big2);
  }

  // LM head weight-sized element-wise (tests sustained bandwidth on 256MB)
  {
    const lmW1 = b.randn([1024, V]);
    const lmW2 = b.randn([1024, V]);
    const lmSize = 1024 * V;

    let ms = benchOp(() => b.add(lmW1, lmW2));
    record("add_lm_head_1024x64000", ms, { bytes: lmSize * 4 * 3, note: "LM head-sized add (256MB)" });

    release(lmW1);
    release(lmW2);
  }

  // ── 3. LAYERNORM ──────────────────────────────────────────────────────────

  {
    const x = b.randn([BT, D]);
    const w = b.randn([D]);
    const bias = b.randn([D]);

    const ms = benchOp(() => (b as any).layerNorm(x, w, bias, 1e-5));
    record("layernorm_512x1024", ms, { bytes: BT * D * 4 * 3, note: "LayerNorm fwd" });

    // Backward
    const gradOut = b.randn([BT, D]);
    if (typeof (b as any).layerNormBackward === "function") {
      const msBwd = benchCustom(() => {
        const result = (b as any).layerNormBackward(x, w, gradOut, 1e-5);
        return [result.dx, result.dw, result.db];
      });
      record("layernorm_bwd_512x1024", msBwd, { note: "LayerNorm backward" });
      release(gradOut);
    }

    release(x);
    release(w);
    release(bias);
  }

  // ── 4. SOFTMAX ────────────────────────────────────────────────────────────

  {
    const attnScores = b.randn([BH, T, T]);
    let ms = benchOp(() => (b as any).softmax(attnScores, -1));
    record("softmax_attn_16x512x512", ms, {
      bytes: BH * T * T * 4 * 2, note: "attention softmax",
    });
    release(attnScores);

    const logits = b.randn([BT, V]);
    ms = benchOp(() => (b as any).softmax(logits, -1));
    record("softmax_logits_512x64000", ms, {
      bytes: BT * V * 4 * 2, note: "output softmax",
    });
    release(logits);
  }

  // ── 5. CROSS-ENTROPY ──────────────────────────────────────────────────────

  if (typeof (b as any).crossEntropy === "function") {
    const logits = b.randn([BT, V]);
    const targets: TensorData = {
      data: new Int32Array(BT).map(() => Math.floor(Math.random() * V)),
      shape: [BT],
      dtype: "i32",
    };

    const ms = benchOp(() => (b as any).crossEntropy(logits, targets));
    record("cross_entropy_fwd_512x64000", ms, { note: "CE loss forward" });

    // CE backward
    if (typeof (b as any).crossEntropyBackward === "function") {
      const gradOut: TensorData = { data: new Float32Array([1.0]), shape: [1], dtype: "f32" };
      const msBwd = benchOp(() => (b as any).crossEntropyBackward(logits, targets, gradOut));
      record("cross_entropy_bwd_512x64000", msBwd, {
        bytes: BT * V * 4 * 2,
        note: "CE loss backward",
      });
    }

    release(logits);
  }

  // ── 6. FLASH ATTENTION ────────────────────────────────────────────────────

  if (typeof (b as any).flashAttention === "function") {
    const q = b.randn([BH, T, Dh]);
    const k = b.randn([BH, T, Dh]);
    const v = b.randn([BH, T, Dh]);

    const ms = benchCustom(() => {
      const result = (b as any).flashAttention(q, k, v, T, 1.0 / Math.sqrt(Dh), 30);
      // flashAttention returns { output, lse }
      if (result.output && result.lse) return [result.output, result.lse];
      return [result];
    });
    record("flash_attn_fwd_b1_h16_t512_d64", ms, {
      flops: 2 * BH * T * T * Dh * 2, note: "Flash Attention fwd",
    });

    // Flash attention backward
    if (typeof (b as any).flashAttentionBackward === "function") {
      const fwdResult = (b as any).flashAttention(q, k, v, T, 1.0 / Math.sqrt(Dh), 30);
      const O = fwdResult.output ?? fwdResult;
      const lse = fwdResult.lse;
      const dO = b.randn([BH, T, Dh]);

      if (lse) {
        const msBwd = benchCustom(() => {
          const result = (b as any).flashAttentionBackward(q, k, v, O, dO, lse, T, 1.0 / Math.sqrt(Dh), 30);
          return [result.dQ, result.dK, result.dV];
        });
        record("flash_attn_bwd_b1_h16_t512_d64", msBwd, {
          flops: 2 * BH * T * T * Dh * 4, note: "Flash Attention bwd",
        });
      }
      release(dO);
    }

    release(q);
    release(k);
    release(v);
  }

  // ── 7. EMBEDDING ──────────────────────────────────────────────────────────

  if (typeof (b as any).embedding === "function") {
    const embW = b.randn([V, D]);
    const indices: TensorData = {
      data: new Int32Array(BT).map(() => Math.floor(Math.random() * V)),
      shape: [BT],
      dtype: "i32",
    };

    const ms = benchOp(() => (b as any).embedding(embW, indices));
    record("embedding_fwd_64000x1024", ms, {
      bytes: BT * D * 4, note: "embedding lookup",
    });

    // Embedding backward (scatter-add gradients back to vocab table)
    if (typeof (b as any).embeddingBackward === "function") {
      const gradOut = b.randn([BT, D]);
      const msBwd = benchOp(() => (b as any).embeddingBackward(indices, gradOut, V));
      record("embedding_bwd_64000x1024", msBwd, {
        bytes: BT * D * 4 + V * D * 4, note: "embedding backward (scatter-add)",
      });
      release(gradOut);
    }

    release(embW);
  }

  // ── 8. ADAMW STEP ─────────────────────────────────────────────────────────

  if (typeof (b as any).adamwStep === "function") {
    // One layer's worth of params (~8.5M)
    const pSize = 1024 * 3072 + 1024 * 1024 + 1024 * 2752 * 2 + 2752 * 1024;
    const param = b.randn([pSize]);
    const grad = b.randn([pSize]);
    const m = b.randn([pSize]);
    const vState = b.randn([pSize]);

    const ms = benchCustom(() => {
      (b as any).adamwStep(param, grad, m, vState, 3e-4, 0.9, 0.999, 1e-8, 0.1, 0.1, 0.001, 1.0);
      return []; // in-place, nothing to release
    });
    record("adamw_step_8.5M", ms, {
      bytes: pSize * 4 * 7, note: "AdamW for 1 layer (~8.5M params)",
    });

    release(param);
    release(grad);
    release(m);
    release(vState);
  }

  // ── 8b. GRADIENT ACCUMULATION (add_inplace, scale_inplace) ───────────────

  if (typeof (b as any).addInplace === "function") {
    // LM head weight grad accumulation: [1024, 64000] = 64M elements = 256MB
    const lmGrad = b.randn([1024, V]);
    const lmAcc = b.randn([1024, V]);
    const lmSize = 1024 * V;
    const ms = benchCustom(() => {
      (b as any).addInplace(lmAcc, lmGrad);
      return [];
    });
    record("grad_accum_lm_head", ms, {
      bytes: lmSize * 4 * 3, note: "gradient accumulation (add_inplace)",
    });

    // Scale grad for clipping
    if (typeof (b as any).scaleInplace === "function") {
      const msScale = benchCustom(() => {
        (b as any).scaleInplace(lmAcc, 0.5);
        return [];
      });
      record("grad_scale_lm_head", msScale, {
        bytes: lmSize * 4 * 2, note: "gradient clipping scale (scale_inplace)",
      });
    }

    release(lmGrad);
    release(lmAcc);
  }

  // ── 8b2. DROPOUT MASK GENERATION ──────────────────────────────────────────

  if (typeof (b as any).dropoutMask === "function") {
    const msDropout = benchOp(() => (b as any).dropoutMask([BT, D], 42, 0, 0.1));
    record("dropout_mask_512x1024", msDropout, {
      bytes: BT * D * 4, note: "GPU-side dropout mask (PhiloxRNG)",
    });
  }

  // ── 8c. GRADIENT NORM (sum of squares) ─────────────────────────────────────

  if (typeof (b as any).sumOfSquares === "function") {
    // One layer's worth of params (~8.5M) — used for gradient clipping norm calc
    const pSize = 1024 * 3072 + 1024 * 1024 + 1024 * 2752 * 2 + 2752 * 1024;
    const gradTensor = b.randn([pSize]);
    const msNorm = benchOp(() => (b as any).sumOfSquares(gradTensor));
    record("grad_norm_8.5M", msNorm, {
      bytes: pSize * 4, note: "sum of squares for gradient norm (fused)",
    });
    release(gradTensor);
  }

  // ── 9. FUSED OPS ──────────────────────────────────────────────────────────

  if (typeof (b as any).residualDropoutAdd === "function") {
    const residual = b.randn([BT, D]);
    const projected = b.randn([BT, D]);
    // Generate a dropout mask (0 or 1/0.9 scaled)
    const maskData = new Float32Array(BT * D);
    for (let i = 0; i < maskData.length; i++) maskData[i] = Math.random() > 0.1 ? 1.0 / 0.9 : 0;
    const mask: TensorData = { data: maskData, shape: [BT, D], dtype: "f32" };

    const ms = benchOp(() => (b as any).residualDropoutAdd(residual, projected, mask));
    record("residual_dropout_add_512x1024", ms, {
      bytes: BT * D * 4 * 4, note: "fused residual+dropout+add",
    });

    release(residual);
    release(projected);
  }

  // ── 10. KERNEL LAUNCH OVERHEAD ────────────────────────────────────────────

  {
    const small = b.randn([64, 64]);
    const small2 = b.randn([64, 64]);

    // 200 sequential small adds (tests dispatch batching efficiency)
    // Warmup
    for (let w = 0; w < opts.warmup; w++) {
      let r = small;
      for (let j = 0; j < 200; j++) {
        const prev = r;
        r = b.add(r, small2);
        if (prev !== small) release(prev);
      }
      sync();
      if (r !== small) release(r);
    }

    const times: number[] = [];
    for (let i = 0; i < opts.iters; i++) {
      const t0 = performance.now();
      let r = small;
      for (let j = 0; j < 200; j++) {
        const prev = r;
        r = b.add(r, small2);
        if (prev !== small) release(prev);
      }
      sync();
      times.push(performance.now() - t0);
      if (r !== small) release(r);
    }
    const totalMs = median(times);
    const perAdd = totalMs / 200;
    record("launch_overhead_200_adds", totalMs, {
      note: `200 sequential 64x64 adds, ${perAdd.toFixed(3)}ms/add`,
    });

    release(small);
    release(small2);
  }

  // ── 11. TRANSPOSE ─────────────────────────────────────────────────────────

  if (typeof (b as any).transpose === "function") {
    const t = b.randn([BH, T, Dh]);
    const ms = benchOp(() => (b as any).transpose(t, 1, 2));
    record("transpose_16x512x64", ms, {
      bytes: BH * T * Dh * 4 * 2, note: "transpose",
    });
    release(t);
  }

  // ── 12. SLICE ─────────────────────────────────────────────────────────────

  if (typeof (b as any).slice === "function") {
    const qkv = b.randn([BT, 3 * D]);
    const ms = benchCustom(() => {
      const q = (b as any).slice(qkv, [0, 0], [BT, D]);
      const k = (b as any).slice(qkv, [0, D], [BT, 2 * D]);
      const v = (b as any).slice(qkv, [0, 2 * D], [BT, 3 * D]);
      return [q, k, v];
    });
    record("slice_qkv_512x3072", ms, { note: "3-way slice for Q,K,V" });
    release(qkv);
  }

  // ── Load CUDA reference ─────────────────────────────────────────────────

  let cudaOps: Record<string, OpResult> = {};
  let cudaInfo = "";

  if (opts.cudaJson) {
    // Load from pre-computed JSON file
    console.error(`[bench] Loading CUDA reference from ${opts.cudaJson}...`);
    try {
      const parsed = JSON.parse(readFileSync(opts.cudaJson, "utf-8"));
      if (parsed.ok) {
        cudaOps = parsed.ops ?? {};
        cudaInfo = `${parsed.device} torch=${parsed.torch_version} cuda=${parsed.cuda_version}`;
      } else {
        console.error(`[bench] CUDA error: ${parsed.error}`);
      }
    } catch (e: any) {
      console.error(`[bench] Failed to load CUDA JSON: ${e.message}`);
    }
  } else {
    // Spawn Python CUDA benchmark (may OOM if Helios holds GPU memory)
    console.error("[bench] Running CUDA reference (consider --cuda-json for separate runs)...");
    const cudaScript = path.resolve(process.cwd(), "scripts", "bench-ops-cuda.py");
    const child = spawnSync(
      opts.python,
      [cudaScript, `--iters=${opts.iters}`, `--warmup=${opts.warmup}`],
      { encoding: "utf-8", maxBuffer: 4 * 1024 * 1024, timeout: 300_000 },
    );

    if (child.error) {
      console.error(`[bench] CUDA reference failed: ${child.error.message}`);
    } else {
      const out = (child.stdout ?? "").trim();
      try {
        const parsed = JSON.parse(out);
        if (parsed.ok) {
          cudaOps = parsed.ops ?? {};
          cudaInfo = `${parsed.device} torch=${parsed.torch_version} cuda=${parsed.cuda_version}`;
        } else {
          console.error(`[bench] CUDA error: ${parsed.error}`);
        }
      } catch {
        console.error("[bench] CUDA reference returned non-JSON");
        if (child.stderr) console.error(child.stderr.slice(0, 500));
      }
    }
  }

  // ── Print comparison table ────────────────────────────────────────────────

  console.log("\n══ Alpha Helios vs CUDA — Comprehensive Training Op Benchmark ══");
  console.log(`Helios: ${info.deviceName ?? "unknown"} coopMat=${info.coopMatSupported ?? false}`);
  if (cudaInfo) console.log(`CUDA:   ${cudaInfo}`);
  console.log(`Config: iters=${opts.iters} warmup=${opts.warmup}`);
  console.log("");

  function pad(s: string, w: number): string {
    return s.length >= w ? s + " " : s + " ".repeat(w - s.length);
  }

  const hdr =
    pad("operation", 38) +
    pad("helios_ms", 11) +
    pad("cuda_ms", 11) +
    pad("h_tflops", 10) +
    pad("c_tflops", 10) +
    pad("h_gbps", 9) +
    pad("c_gbps", 9) +
    pad("winner", 10) +
    "note";
  console.log(hdr);
  console.log("─".repeat(hdr.length));

  let heliosWins = 0;
  let cudaWins = 0;
  let ties = 0;

  const allOps = new Set([...Object.keys(results), ...Object.keys(cudaOps)]);
  // Sort by category
  const sortedOps = [...allOps].sort();

  for (const op of sortedOps) {
    const h = results[op];
    const c = cudaOps[op];
    if (!h && !c) continue;

    const hMs = h ? h.ms.toFixed(3) : "n/a";
    const cMs = c ? c.ms.toFixed(3) : "n/a";
    const hTf = h?.tflops ? h.tflops.toFixed(2) : "";
    const cTf = c?.tflops ? c.tflops.toFixed(2) : "";
    const hGb = h?.gbps ? h.gbps.toFixed(0) : "";
    const cGb = c?.gbps ? c.gbps.toFixed(0) : "";

    let winner = "";
    if (h && c && h.ms > 0 && c.ms > 0) {
      const ratio = c.ms / h.ms;
      if (ratio > 1.05) { winner = `H ${ratio.toFixed(2)}×`; heliosWins++; }
      else if (ratio < 0.95) { winner = `C ${(1/ratio).toFixed(2)}×`; cudaWins++; }
      else { winner = "tie"; ties++; }
    }

    const note = h?.note ?? c?.note ?? "";

    console.log(
      pad(op, 38) +
      pad(hMs, 11) +
      pad(cMs, 11) +
      pad(hTf, 10) +
      pad(cTf, 10) +
      pad(hGb, 9) +
      pad(cGb, 9) +
      pad(winner, 10) +
      note,
    );
  }

  console.log("─".repeat(hdr.length));
  console.log(`\nScore: Helios wins ${heliosWins} / CUDA wins ${cudaWins} / ties ${ties}`);
  console.log("");
}

main();
