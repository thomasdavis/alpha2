/**
 * Comprehensive Helios vs CUDA benchmark for ALL Alpha training operations.
 *
 * Tests every operation type at the exact shapes used in the 300M GPT training
 * pipeline. Compares Helios (Vulkan) against PyTorch CUDA on each.
 *
 * Usage:
 *   npx tsx scripts/bench-ops.ts [--iters=30] [--warmup=8] [--python=python3] [--cuda-json=path] [--only=matmul_bwd,softmax]
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
  only: string;  // comma-separated substrings to filter ops
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
    warmup: Math.max(0, parseInt(kv.warmup ?? "20", 10)),
    python: kv.python ?? "python3",
    cudaJson: kv["cuda-json"] ?? "",
    only: kv.only ?? "",
  };
}

// ── Hardware specs (NVIDIA L4) ─────────────────────────────────────────────

const L4_MEM_BW_BYTES = 300e9;   // ~300 GB/s GDDR6
const L4_FP16_TC_FLOPS = 120e12; // 120 TFLOP/s f16 tensor cores (dense)
const L4_FP32_FLOPS = 30.3e12;   // 30.3 TFLOP/s fp32 (non-tensor)

// ── Timing helpers ──────────────────────────────────────────────────────────

interface OpResult {
  ms: number;
  tflops?: number;
  gbps?: number;
  peakMs?: number;  // theoretical minimum time (roofline)
  note?: string;
}

interface FlashDispatchDebug {
  requestedOp: "flashAttention" | "flashAttentionCoop2" | "flashAttentionCoop2Probe";
  executedPath: "scalar" | "coop2";
  mode: "full" | "qk" | "qk_mask" | "qk_softmax" | "pv";
  softCap: number;
  BH: number;
  T: number;
  D: number;
  kernelName: string;
  pipelineKey: string;
  pipelineHandle: number;
  pipelineCreated: boolean;
  scope?: "wg" | "sg";
  fallbackReason?: string;
}

interface FlashBenchTrace {
  firstTimed: FlashDispatchDebug | null;
  pipelineCreatedWarmup: boolean;
  pipelineCreatedTimed: boolean;
  warmupWaitCalls: number;
  timedWaitCalls: number;
  kernels: string[];
  pipelineKeys: string[];
  paths: string[];
  scopes: string[];
  fallbackReasons: string[];
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
  const getFlashDispatchDebugFn: (() => FlashDispatchDebug | null) | null =
    typeof bAny.getLastFlashDispatchDebug === "function" ? bAny.getLastFlashDispatchDebug.bind(bAny) : null;
  const getWaitTimelineCountFn: (() => number) | null =
    typeof bAny.getWaitTimelineCount === "function" ? bAny.getWaitTimelineCount.bind(bAny) : null;

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
  const onlyFilters = opts.only ? opts.only.split(",").map(s => s.trim()) : [];
  function shouldRun(name: string): boolean {
    if (onlyFilters.length === 0) return true;
    return onlyFilters.some(f => name.includes(f));
  }

  function record(name: string, ms: number, opts?: { flops?: number; bytes?: number; note?: string; tensorCore?: boolean }): void {
    const r: OpResult = { ms: Math.round(ms * 10000) / 10000 };
    if (opts?.flops && opts.flops > 0) r.tflops = Math.round((opts.flops / (ms / 1000)) / 1e12 * 1000) / 1000;
    if (opts?.bytes && opts.bytes > 0) r.gbps = Math.round((opts.bytes / (ms / 1000)) / 1e9 * 10) / 10;
    // Roofline: theoretical min time = max(bandwidth_time, compute_time)
    const bwMs = (opts?.bytes && opts.bytes > 0) ? (opts.bytes / L4_MEM_BW_BYTES) * 1000 : 0;
    const peakFlops = opts?.tensorCore ? L4_FP16_TC_FLOPS : L4_FP32_FLOPS;
    const computeMs = (opts?.flops && opts.flops > 0) ? (opts.flops / peakFlops) * 1000 : 0;
    const peak = Math.max(bwMs, computeMs);
    if (peak > 0) r.peakMs = Math.round(peak * 10000) / 10000;
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

  function benchCustomWithFlashTrace(fn: () => TensorData[]): { ms: number; trace: FlashBenchTrace } {
    const traceEveryIter = (process.env.HELIOS_FLASH_BENCH_TRACE_EVERY_ITER ?? "0") === "1";
    const kernels = new Set<string>();
    const pipelineKeys = new Set<string>();
    const paths = new Set<string>();
    const scopes = new Set<string>();
    const fallbackReasons = new Set<string>();
    let firstTimed: FlashDispatchDebug | null = null;
    let pipelineCreatedWarmup = false;
    let pipelineCreatedTimed = false;

    const waitBeforeWarmup = getWaitTimelineCountFn ? getWaitTimelineCountFn() : 0;
    for (let i = 0; i < opts.warmup; i++) {
      const rs = fn();
      sync();
      for (const r of rs) release(r);
      if (traceEveryIter || i === opts.warmup - 1) {
        const debug = getFlashDispatchDebugFn ? getFlashDispatchDebugFn() : null;
        if (debug?.pipelineCreated) pipelineCreatedWarmup = true;
      }
    }
    const waitAfterWarmup = getWaitTimelineCountFn ? getWaitTimelineCountFn() : waitBeforeWarmup;
    const warmupWaitCalls = waitAfterWarmup - waitBeforeWarmup;

    const times: number[] = [];
    for (let i = 0; i < opts.iters; i++) {
      const t0 = performance.now();
      const rs = fn();
      sync();
      const elapsed = performance.now() - t0;
      times.push(elapsed);
      for (const r of rs) release(r);

      if (traceEveryIter || i === 0) {
        const debug = getFlashDispatchDebugFn ? getFlashDispatchDebugFn() : null;
        if (!debug) continue;
        if (!firstTimed) firstTimed = debug;
        kernels.add(debug.kernelName);
        pipelineKeys.add(debug.pipelineKey);
        paths.add(debug.executedPath);
        if (debug.scope) scopes.add(debug.scope);
        if (debug.fallbackReason) fallbackReasons.add(debug.fallbackReason);
        if (debug.pipelineCreated) pipelineCreatedTimed = true;
      }
    }
    const waitAfterTimed = getWaitTimelineCountFn ? getWaitTimelineCountFn() : waitAfterWarmup;
    const timedWaitCalls = waitAfterTimed - waitAfterWarmup;

    return {
      ms: median(times),
      trace: {
        firstTimed,
        pipelineCreatedWarmup,
        pipelineCreatedTimed,
        warmupWaitCalls,
        timedWaitCalls,
        kernels: Array.from(kernels),
        pipelineKeys: Array.from(pipelineKeys),
        paths: Array.from(paths),
        scopes: Array.from(scopes),
        fallbackReasons: Array.from(fallbackReasons),
      },
    };
  }

  function logFlashTrace(opName: string, trace: FlashBenchTrace): void {
    const first = trace.firstTimed;
    const pipelineKey = first?.pipelineKey ?? (trace.pipelineKeys[0] ?? "n/a");
    const kernelName = first?.kernelName ?? (trace.kernels[0] ?? "n/a");
    const path = first?.executedPath ?? (trace.paths[0] ?? "n/a");
    const scope = first?.scope ?? (trace.scopes[0] ?? "n/a");
    const fallback = first?.fallbackReason ?? (trace.fallbackReasons[0] ?? "none");
    const timedWaitAny = trace.timedWaitCalls > 0 ? "yes" : "no";
    const timedExpected = opts.iters;
    const timedExtraWaits = trace.timedWaitCalls - timedExpected;
    console.error(
      `[flash-debug] op=${opName} path=${path} kernel=${kernelName} scope=${scope} ` +
      `pipelineKey=${pipelineKey} pipelineCreatedWarmup=${trace.pipelineCreatedWarmup} ` +
      `pipelineCreatedTimed=${trace.pipelineCreatedTimed} queueWaitTimed=${timedWaitAny} ` +
      `timedWaitCalls=${trace.timedWaitCalls}/${timedExpected} timedExtraWaits=${timedExtraWaits} ` +
      `fallback=${fallback}`,
    );
  }

  console.error(`[bench] Helios device: ${info.deviceName ?? "unknown"} coopMat=${info.coopMatSupported ?? false}`);
  console.error(`[bench] iters=${opts.iters} warmup=${opts.warmup}${opts.only ? ` only=${opts.only}` : ""}`);

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
    if (!shouldRun(name)) continue;
    const a = b.randn([M, K]);
    const w = b.randn([N, K]); // stored transposed, as in training
    // Use native matmulTransposed (A @ W^T) — what training actually calls
    const ms = hasMatmulTransposed
      ? benchOp(() => (b as any).matmulTransposed(a, w))
      : benchOp(() => b.matmul(a, (b as any).transpose(w, 0, 1)));
    record(name, ms, { flops: 2 * M * N * K, bytes: (M * K + K * N + M * N) * 4, tensorCore: true, note });
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
    if (!shouldRun(name)) continue;
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
    record(name, ms, { flops: 2 * M * N * K, bytes: (M * K + K * N + M * N) * 4, tensorCore: true, note });
    release(a);
    release(dout);
  }

  // Square matmuls for reference
  for (const sz of [1024, 2048, 3072, 4096]) {
    const sqName = `matmul_${sz}sq`;
    if (!shouldRun(sqName)) continue;
    const a = b.randn([sz, sz]);
    const bm = b.randn([sz, sz]);
    const ms = benchOp(() => b.matmul(a, bm));
    record(sqName, ms, { flops: 2 * sz ** 3, bytes: 3 * sz * sz * 4, tensorCore: true, note: `${sz}x${sz}x${sz}` });
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

    if (shouldRun("add_512x1024")) { const ms = benchOp(() => b.add(x, y)); record("add_512x1024", ms, { bytes: sz1024 * 4 * 3, note: "a+b" }); }
    if (shouldRun("mul_512x1024")) { const ms = benchOp(() => b.mul(x, y)); record("mul_512x1024", ms, { bytes: sz1024 * 4 * 3, note: "a*b" }); }
    if (shouldRun("gelu_512x1024")) { const ms = benchOp(() => (b as any).gelu(x)); record("gelu_512x1024", ms, { bytes: sz1024 * 4 * 2, note: "GELU activation" }); }
    if (shouldRun("silu_512x2752")) { const ms = benchOp(() => (b as any).silu(x2)); record("silu_512x2752", ms, { bytes: sz2752 * 4 * 2, note: "SiLU activation" }); }
    if (shouldRun("scale_512x1024")) { const ms = benchOp(() => (b as any).scale(x, 0.125)); record("scale_512x1024", ms, { bytes: sz1024 * 4 * 2, note: "scalar multiply" }); }
    if (shouldRun("neg_512x1024")) { const ms = benchOp(() => (b as any).neg(x)); record("neg_512x1024", ms, { bytes: sz1024 * 4 * 2, note: "negate" }); }
    if (shouldRun("exp_512x1024")) { const ms = benchOp(() => (b as any).exp(x)); record("exp_512x1024", ms, { bytes: sz1024 * 4 * 2, note: "exp" }); }

    // Activation backward ops (fused kernels)
    if (shouldRun("gelu_bwd_512x1024") && typeof (b as any).geluBackward === "function") {
      const ms = benchOp(() => (b as any).geluBackward(x, y));
      record("gelu_bwd_512x1024", ms, { bytes: sz1024 * 4 * 3, note: "GELU backward (fused)" });
    }

    // SiLU backward (used in SwiGLU)
    if (shouldRun("silu_bwd_512x2752") && typeof (b as any).siluBackward === "function") {
      const xSilu = b.randn([512, 2752]);
      const ySilu = b.randn([512, 2752]);
      const ms = benchOp(() => (b as any).siluBackward(xSilu, ySilu));
      record("silu_bwd_512x2752", ms, { bytes: sz2752 * 4 * 3, note: "SiLU backward (fused)" });
      release(xSilu);
      release(ySilu);
    }

    // Fused SiLU-Mul (SwiGLU forward: silu(a)*b in 1 dispatch instead of 2)
    if (shouldRun("silu_mul_512x2752") && typeof (b as any).siluMul === "function") {
      const smA = b.randn([512, 2752]);
      const smB = b.randn([512, 2752]);
      const ms = benchOp(() => (b as any).siluMul(smA, smB));
      record("silu_mul_512x2752", ms, { bytes: sz2752 * 4 * 3, note: "fused SiLU*Mul (SwiGLU fwd)" });
      release(smA);
      release(smB);
    }

    // Fused SiLU-Mul Backward (SwiGLU backward: both da,db in 1 dispatch instead of 3)
    if (shouldRun("silu_mul_bwd_512x2752") && typeof (b as any).siluMulBackward === "function") {
      const smA = b.randn([512, 2752]);
      const smB = b.randn([512, 2752]);
      const smG = b.randn([512, 2752]);
      // benchOp expects single TensorData; wrapper releases both outputs
      const ms = benchOp(() => {
        const [da, db] = (b as any).siluMulBackward(smA, smB, smG);
        release(db);
        return da;
      });
      record("silu_mul_bwd_512x2752", ms, { bytes: sz2752 * 4 * 5, note: "fused SiLU*Mul backward (SwiGLU bwd)" });
      release(smA);
      release(smB);
      release(smG);
    }

    // SoftCap backward (complex fused: tanh + exp derivatives)
    if (shouldRun("softcap_bwd_512x2752") && typeof (b as any).softCapBackward === "function") {
      const scGrad = b.randn([512, 2752]);
      const scInput = b.randn([512, 2752]);
      const ms = benchOp(() => (b as any).softCapBackward(scGrad, scInput, 30.0));
      record("softcap_bwd_512x2752", ms, { bytes: sz2752 * 4 * 3, note: "softcap backward (fused)" });
      release(scGrad);
      release(scInput);
    }

    // ReLU backward (simple fused: x > 0 ? grad : 0)
    if (shouldRun("relu_bwd_512x1024") && typeof (b as any).reluBackward === "function") {
      const ms = benchOp(() => (b as any).reluBackward(x, y));
      record("relu_bwd_512x1024", ms, { bytes: sz1024 * 4 * 3, note: "ReLU backward (fused)" });
    }

    // Clamp backward (fused: lo < x < hi ? grad : 0)
    if (shouldRun("clamp_bwd_512x1024") && typeof (b as any).clampBackward === "function") {
      const ms = benchOp(() => (b as any).clampBackward(x, y, -1.0, 1.0));
      record("clamp_bwd_512x1024", ms, { bytes: sz1024 * 4 * 3, note: "clamp backward (fused)" });
    }

    release(x);
    release(y);
    release(x2);
  }

  // Large tensor backward ops (Helios amortizes dispatch overhead on big tensors)
  {
    const bigX = b.randn([4096, 4096]);
    const bigG = b.randn([4096, 4096]);
    const bigSize = 4096 * 4096;

    if (shouldRun("gelu_bwd_4096sq") && typeof (b as any).geluBackward === "function") {
      const ms = benchOp(() => (b as any).geluBackward(bigX, bigG));
      record("gelu_bwd_4096sq", ms, { bytes: bigSize * 4 * 3, note: "GELU backward large (16M)" });
    }
    if (shouldRun("silu_bwd_4096sq") && typeof (b as any).siluBackward === "function") {
      const ms = benchOp(() => (b as any).siluBackward(bigX, bigG));
      record("silu_bwd_4096sq", ms, { bytes: bigSize * 4 * 3, note: "SiLU backward large (16M)" });
    }
    if (shouldRun("softcap_bwd_4096sq") && typeof (b as any).softCapBackward === "function") {
      const ms = benchOp(() => (b as any).softCapBackward(bigG, bigX, 30.0));
      record("softcap_bwd_4096sq", ms, { bytes: bigSize * 4 * 3, note: "softcap backward large (16M)" });
    }
    if (shouldRun("relu_bwd_4096sq") && typeof (b as any).reluBackward === "function") {
      const ms = benchOp(() => (b as any).reluBackward(bigX, bigG));
      record("relu_bwd_4096sq", ms, { bytes: bigSize * 4 * 3, note: "ReLU backward large (16M)" });
    }
    if (shouldRun("clamp_bwd_4096sq") && typeof (b as any).clampBackward === "function") {
      const ms = benchOp(() => (b as any).clampBackward(bigX, bigG, -1.0, 1.0));
      record("clamp_bwd_4096sq", ms, { bytes: bigSize * 4 * 3, note: "clamp backward large (16M)" });
    }

    // Large forward activations (Helios fused unary kernels)
    if (shouldRun("gelu_fwd_4096sq")) {
      const ms = benchOp(() => b.gelu(bigX));
      record("gelu_fwd_4096sq", ms, { bytes: bigSize * 4 * 2, note: "GELU forward large (16M)" });
    }
    if (shouldRun("silu_fwd_4096sq")) {
      const ms = benchOp(() => (b as any).silu(bigX));
      record("silu_fwd_4096sq", ms, { bytes: bigSize * 4 * 2, note: "SiLU forward large (16M)" });
    }
    if (shouldRun("relu_fwd_4096sq") && typeof (b as any).relu === "function") {
      const ms = benchOp(() => (b as any).relu(bigX));
      record("relu_fwd_4096sq", ms, { bytes: bigSize * 4 * 2, note: "ReLU forward large (16M)" });
    }

    release(bigX);
    release(bigG);
  }

  // Large tensor bandwidth test
  {
    const big = b.randn([4096, 4096]);
    const big2 = b.randn([4096, 4096]);
    const sz = 4096 * 4096;

    if (shouldRun("add_4096x4096")) { const ms = benchOp(() => b.add(big, big2)); record("add_4096x4096", ms, { bytes: sz * 4 * 3, note: "large add bandwidth test" }); }
    if (shouldRun("scale_4096x4096")) { const ms = benchOp(() => (b as any).scale(big, 2.0)); record("scale_4096x4096", ms, { bytes: sz * 4 * 2, note: "large scale bandwidth test" }); }

    release(big);
    release(big2);
  }

  // LM head weight-sized element-wise (tests sustained bandwidth on 256MB)
  {
    const lmW1 = b.randn([1024, V]);
    const lmW2 = b.randn([1024, V]);
    const lmSize = 1024 * V;

    if (shouldRun("add_lm_head_1024x64000")) { const ms = benchOp(() => b.add(lmW1, lmW2)); record("add_lm_head_1024x64000", ms, { bytes: lmSize * 4 * 3, note: "LM head-sized add (256MB)" }); }

    release(lmW1);
    release(lmW2);
  }

  // ── 3. LAYERNORM ──────────────────────────────────────────────────────────

  {
    const x = b.randn([BT, D]);
    const w = b.randn([D]);
    const bias = b.randn([D]);

    if (shouldRun("layernorm_512x1024")) {
      const ms = benchOp(() => (b as any).layerNorm(x, w, bias, 1e-5));
      record("layernorm_512x1024", ms, { bytes: BT * D * 4 * 3, note: "LayerNorm fwd" });
    }

    // Backward
    if (shouldRun("layernorm_bwd_512x1024") && typeof (b as any).layerNormBackward === "function") {
      const gradOut = b.randn([BT, D]);
      const msBwd = benchCustom(() => {
        const result = (b as any).layerNormBackward(x, w, gradOut, 1e-5);
        return [result.dx, result.dw, result.db];
      });
      // bwd reads: x, w, gradOut, writes: dx, dw, db  → ~5 full passes of [BT,D]
      record("layernorm_bwd_512x1024", msBwd, { bytes: BT * D * 4 * 5, note: "LayerNorm backward" });
      release(gradOut);
    }

    release(x);
    release(w);
    release(bias);
  }

  // ── 4. SOFTMAX ────────────────────────────────────────────────────────────

  {
    if (shouldRun("softmax_attn_16x512x512")) {
      const attnScores = b.randn([BH, T, T]);
      const ms = benchOp(() => (b as any).softmax(attnScores, -1));
      record("softmax_attn_16x512x512", ms, { bytes: BH * T * T * 4 * 2, note: "attention softmax" });
      release(attnScores);
    }

    // Cache warmth experiment: cold vs hot vs producer-hot
    if (shouldRun("softmax_cache_test")) {
      const smBytes = BH * T * T * 4 * 2;

      // Test 1: "hot" — run softmax 2x back-to-back in same batch, time total/2
      // The 2nd softmax reads input that was just read by the 1st → should be L2-hot
      {
        const attn = b.randn([BH, T, T]);
        const ms = benchCustom(() => {
          const r1 = (b as any).softmax(attn, -1);
          const r2 = (b as any).softmax(attn, -1);
          return [r1, r2];
        });
        record("softmax_attn_hot_2x", ms / 2, { bytes: smBytes, note: "L2-hot (2nd of 2 back-to-back)" });
        release(attn);
      }

      // Test 2: "producer-hot" — add kernel writes output, then softmax reads it
      // Simulates real attention: matmul writes scores, softmax reads them
      {
        const a1 = b.randn([BH, T, T]);
        const a2 = b.randn([BH, T, T]);
        const ms = benchCustom(() => {
          const scores = b.add(a1, a2);  // producer: writes [BH,T,T] to GPU → L2-hot
          const sm = (b as any).softmax(scores, -1);  // consumer reads L2-hot data
          return [scores, sm];
        });
        // Subtract standalone add time to isolate softmax-only
        const addMs = benchOp(() => b.add(a1, a2));
        record("softmax_attn_producer_hot", ms - addMs, { bytes: smBytes, note: "producer→consumer L2 reuse" });
        record("softmax_attn_producer_total", ms, { bytes: smBytes, note: "add+softmax fused" });
        release(a1); release(a2);
      }

      // Test 3: "cold" — allocate fresh buffer each iteration to defeat L2
      // Fill with 128MB of noise first to flush L2 (L4 has 48MB L2)
      {
        const flushBuf = b.randn([32 * 1024 * 1024]);  // 128MB to flush L2
        const attn = b.randn([BH, T, T]);
        const ms = benchCustom(() => {
          // Read flushBuf to evict attn from L2
          const trash = b.add(flushBuf, flushBuf);
          const sm = (b as any).softmax(attn, -1);
          return [trash, sm];
        });
        const flushMs = benchOp(() => b.add(flushBuf, flushBuf));
        record("softmax_attn_cold", ms - flushMs, { bytes: smBytes, note: "L2-cold (after 128MB flush)" });
        release(flushBuf); release(attn);
      }

      console.error("[cache-test] softmax_attn standard / hot_2x / producer_hot / cold measured");
    }

    if (shouldRun("softmax_logits_512x64000")) {
      const logits = b.randn([BT, V]);
      const ms = benchOp(() => (b as any).softmax(logits, -1));
      record("softmax_logits_512x64000", ms, { bytes: BT * V * 4 * 2, note: "output softmax" });
      release(logits);
    }
  }

  // ── 5. CROSS-ENTROPY ──────────────────────────────────────────────────────

  if (typeof (b as any).crossEntropy === "function" && (shouldRun("cross_entropy_fwd") || shouldRun("cross_entropy_bwd"))) {
    const logits = b.randn([BT, V]);
    const targets: TensorData = {
      data: new Int32Array(BT).map(() => Math.floor(Math.random() * V)),
      shape: [BT],
      dtype: "i32",
    };

    if (shouldRun("cross_entropy_fwd_512x64000")) {
      const ms = benchOp(() => (b as any).crossEntropy(logits, targets));
      // CE fwd: internally does softmax (2 passes of [BT,V]) + NLL scatter-read + reduction
      record("cross_entropy_fwd_512x64000", ms, { bytes: BT * V * 4 * 4, note: "CE loss forward" });
    }

    // CE backward
    if (shouldRun("cross_entropy_bwd_512x64000") && typeof (b as any).crossEntropyBackward === "function") {
      const gradOut: TensorData = { data: new Float32Array([1.0]), shape: [1], dtype: "f32" };
      const msBwd = benchOp(() => (b as any).crossEntropyBackward(logits, targets, gradOut));
      record("cross_entropy_bwd_512x64000", msBwd, { bytes: BT * V * 4 * 2, note: "CE loss backward" });
    }

    release(logits);
  }

  // ── 6. FLASH ATTENTION ────────────────────────────────────────────────────

  const runFlashSection =
    shouldRun("flash_attn") ||
    shouldRun("flash_attn_fwd") ||
    shouldRun("flash_attn_bwd") ||
    shouldRun("flash_attn_coop") ||
    shouldRun("flash_attn_coop2") ||
    shouldRun("flash_attn_coop2_probe");
  if (runFlashSection) {
    const q = b.randn([BH, T, Dh]);
    const k = b.randn([BH, T, Dh]);
    const v = b.randn([BH, T, Dh]);

    if (shouldRun("flash_attn_fwd_b1_h16_t512_d64")) {
      const traced = benchCustomWithFlashTrace(() => {
        const result = (b as any).flashAttention(q, k, v, T, 1.0 / Math.sqrt(Dh), 30);
        if (result.output && result.lse) return [result.output, result.lse];
        return [result];
      });
      const ms = traced.ms;
      // Flash attn: reads Q,K,V [BH,T,Dh], writes O [BH,T,Dh] + LSE [BH,T]
      // Multi-pass tiled so actual bandwidth >> single pass, but roofline = single pass minimum
      record("flash_attn_fwd_b1_h16_t512_d64", ms, {
        flops: 2 * BH * T * T * Dh * 2,
        bytes: BH * T * Dh * 4 * 4 + BH * T * 4,
        note: "Flash Attention fwd",
      });
      logFlashTrace("flash_attn_fwd_b1_h16_t512_d64", traced.trace);
    }

    // Flash attention backward
    if (shouldRun("flash_attn_bwd_b1_h16_t512_d64") && typeof (b as any).flashAttentionBackward === "function") {
      const fwdResult = (b as any).flashAttention(q, k, v, T, 1.0 / Math.sqrt(Dh), 30);
      const O = fwdResult.output ?? fwdResult;
      const lse = fwdResult.lse;
      const dO = b.randn([BH, T, Dh]);

      if (lse) {
        const msBwd = benchCustom(() => {
          const result = (b as any).flashAttentionBackward(q, k, v, O, dO, lse, T, 1.0 / Math.sqrt(Dh), 30);
          return [result.dQ, result.dK, result.dV];
        });
        // Flash bwd: reads Q,K,V,O,dO,LSE, writes dQ,dK,dV
        record("flash_attn_bwd_b1_h16_t512_d64", msBwd, {
          flops: 2 * BH * T * T * Dh * 4,
          bytes: BH * T * Dh * 4 * 8 + BH * T * 4,
          note: "Flash Attention bwd",
        });
      }
      release(dO);
    }

    // softcap=0 variants (compile-time specialized — no tanh code in SPIR-V)
    if (shouldRun("flash_attn_fwd_nosc_b1_h16_t512_d64")) {
      const traced = benchCustomWithFlashTrace(() => {
        const result = (b as any).flashAttention(q, k, v, T, 1.0 / Math.sqrt(Dh), 0);
        if (result.output && result.lse) return [result.output, result.lse];
        return [result];
      });
      const ms = traced.ms;
      record("flash_attn_fwd_nosc_b1_h16_t512_d64", ms, {
        flops: 2 * BH * T * T * Dh * 2,
        bytes: BH * T * Dh * 4 * 4 + BH * T * 4,
        note: "Flash Attention fwd (no softcap)",
      });
      logFlashTrace("flash_attn_fwd_nosc_b1_h16_t512_d64", traced.trace);
    }

    if (shouldRun("flash_attn_bwd_nosc_b1_h16_t512_d64") && typeof (b as any).flashAttentionBackward === "function") {
      const fwdNoSC = (b as any).flashAttention(q, k, v, T, 1.0 / Math.sqrt(Dh), 0);
      const oNoSC = fwdNoSC.output ?? fwdNoSC;
      const lseNoSC = fwdNoSC.lse;
      const dONoSC = b.randn([BH, T, Dh]);

      if (lseNoSC) {
        const msBwd = benchCustom(() => {
          const result = (b as any).flashAttentionBackward(q, k, v, oNoSC, dONoSC, lseNoSC, T, 1.0 / Math.sqrt(Dh), 0);
          return [result.dQ, result.dK, result.dV];
        });
        record("flash_attn_bwd_nosc_b1_h16_t512_d64", msBwd, {
          flops: 2 * BH * T * T * Dh * 4,
          bytes: BH * T * Dh * 4 * 8 + BH * T * 4,
          note: "Flash Attention bwd (no softcap)",
        });
      }
      release(dONoSC);
    }

    // Cooperative matrix flash attention forward (tensor core accelerated)
    if (shouldRun("flash_attn_coop_fwd_b1_h16_t512_d64") && typeof (b as any).flashAttentionCoop === "function") {
      const ms = benchCustom(() => {
        const result = (b as any).flashAttentionCoop(q, k, v, T, 1.0 / Math.sqrt(Dh));
        if (result.output && result.lse) return [result.output, result.lse];
        return [result];
      });
      record("flash_attn_coop_fwd_b1_h16_t512_d64", ms, {
        flops: 2 * BH * T * T * Dh * 2,
        bytes: BH * T * Dh * 4 * 4 + BH * T * 4,
        note: "Flash Attention fwd (coop matrix, tensor core)",
      });
    }

    // Cooperative matrix 2 flash attention forward (NV coop matrix 2: reduce, per-element ops)
    if (shouldRun("flash_attn_coop2_fwd_b1_h16_t512_d64") && typeof (b as any).flashAttentionCoop2 === "function") {
      const traced = benchCustomWithFlashTrace(() => {
        const result = (b as any).flashAttentionCoop2(q, k, v, T, 1.0 / Math.sqrt(Dh));
        if (result.output && result.lse) return [result.output, result.lse];
        return [result];
      });
      const ms = traced.ms;
      record("flash_attn_coop2_fwd_b1_h16_t512_d64", ms, {
        flops: 2 * BH * T * T * Dh * 2,
        bytes: BH * T * Dh * 4 * 4 + BH * T * 4,
        note: "Flash Attention fwd (coop matrix 2, NV reduce+perElemOp)",
      });
      logFlashTrace("flash_attn_coop2_fwd_b1_h16_t512_d64", traced.trace);
    }

    if (shouldRun("flash_attn_coop2_fwd_sc_b1_h16_t512_d64") && typeof (b as any).flashAttentionCoop2 === "function") {
      const traced = benchCustomWithFlashTrace(() => {
        const result = (b as any).flashAttentionCoop2(q, k, v, T, 1.0 / Math.sqrt(Dh), 30);
        if (result.output && result.lse) return [result.output, result.lse];
        return [result];
      });
      const ms = traced.ms;
      record("flash_attn_coop2_fwd_sc_b1_h16_t512_d64", ms, {
        flops: 2 * BH * T * T * Dh * 2,
        bytes: BH * T * Dh * 4 * 4 + BH * T * 4,
        note: "Flash Attention fwd (coop matrix 2 + softcap=30)",
      });
      logFlashTrace("flash_attn_coop2_fwd_sc_b1_h16_t512_d64", traced.trace);
    }

    // Coop2 stage probes (driver 590+): isolate where the fused pipeline spends time.
    {
      const runAnyProbe =
        shouldRun("flash_attn_coop2_probe") ||
        shouldRun("flash_attn_coop2_probe_qk_b1_h16_t512_d64") ||
        shouldRun("flash_attn_coop2_probe_qk_mask_b1_h16_t512_d64") ||
        shouldRun("flash_attn_coop2_probe_qk_softmax_b1_h16_t512_d64") ||
        shouldRun("flash_attn_coop2_probe_pv_b1_h16_t512_d64");
      if (runAnyProbe && typeof (b as any).flashAttentionCoop2Probe === "function") {
        const probeSpecs: Array<{ mode: "qk" | "qk_mask" | "qk_softmax" | "pv"; key: string; note: string; flops: number }> = [
          {
            mode: "qk",
            key: "flash_attn_coop2_probe_qk_b1_h16_t512_d64",
            note: "Probe: QK MMA only (no mask/softmax/PV)",
            flops: 2 * BH * T * T * Dh,
          },
          {
            mode: "qk_mask",
            key: "flash_attn_coop2_probe_qk_mask_b1_h16_t512_d64",
            note: "Probe: QK MMA + scale/mask (no softmax/PV)",
            flops: 2 * BH * T * T * Dh,
          },
          {
            mode: "qk_softmax",
            key: "flash_attn_coop2_probe_qk_softmax_b1_h16_t512_d64",
            note: "Probe: QK + scale/mask + coop2 softmax (no PV)",
            flops: 2 * BH * T * T * Dh,
          },
          {
            mode: "pv",
            key: "flash_attn_coop2_probe_pv_b1_h16_t512_d64",
            note: "Probe: PV MMA only (synthetic P, no QK/softmax)",
            flops: 2 * BH * T * T * Dh,
          },
        ];

        for (const spec of probeSpecs) {
          if (!shouldRun(spec.key) && !shouldRun("flash_attn_coop2_probe")) continue;
          const traced = benchCustomWithFlashTrace(() => {
            const result = (b as any).flashAttentionCoop2Probe(spec.mode, q, k, v, T, 1.0 / Math.sqrt(Dh));
            return [result.output, result.lse];
          });
          const ms = traced.ms;
          record(spec.key, ms, {
            flops: spec.flops,
            bytes: BH * T * Dh * 4 * 4 + BH * T * 4,
            note: spec.note,
          });
          logFlashTrace(spec.key, traced.trace);
        }
      }
    }

    release(q);
    release(k);
    release(v);
  }

  // ── 7. EMBEDDING ──────────────────────────────────────────────────────────

  if (typeof (b as any).embedding === "function" && (shouldRun("embedding_fwd") || shouldRun("embedding_bwd"))) {
    const embW = b.randn([V, D]);
    const indices: TensorData = {
      data: new Int32Array(BT).map(() => Math.floor(Math.random() * V)),
      shape: [BT],
      dtype: "i32",
    };

    if (shouldRun("embedding_fwd_64000x1024")) {
      const ms = benchOp(() => (b as any).embedding(embW, indices));
      record("embedding_fwd_64000x1024", ms, { bytes: BT * D * 4, note: "embedding lookup" });
    }

    if (shouldRun("embedding_bwd_64000x1024") && typeof (b as any).embeddingBackward === "function") {
      const gradOut = b.randn([BT, D]);
      const msBwd = benchOp(() => (b as any).embeddingBackward(indices, gradOut, V));
      record("embedding_bwd_64000x1024", msBwd, { bytes: BT * D * 4 + V * D * 4, note: "embedding backward (scatter-add)" });
      release(gradOut);
    }

    release(embW);
  }

  // ── 8. ADAMW STEP ─────────────────────────────────────────────────────────

  if (typeof (b as any).adamwStep === "function") {
    const pSize = 1024 * 3072 + 1024 * 1024 + 1024 * 2752 * 2 + 2752 * 1024;
    if (shouldRun("adamw_step_8.5M")) {
      const param = b.randn([pSize]);
      const grad = b.randn([pSize]);
      const m = b.randn([pSize]);
      const vState = b.randn([pSize]);
      const ms = benchCustom(() => { (b as any).adamwStep(param, grad, m, vState, 3e-4, 0.9, 0.999, 1e-8, 0.1, 0.1, 0.001, 1.0); return []; });
      record("adamw_step_8.5M", ms, { bytes: pSize * 4 * 7, note: "AdamW for 1 layer (~8.5M params)" });
      release(param); release(grad); release(m); release(vState);
    }

    if (shouldRun("adamw_step_34M")) {
      const pSize4 = pSize * 4;
      const param4 = b.randn([pSize4]);
      const grad4 = b.randn([pSize4]);
      const m4 = b.randn([pSize4]);
      const vState4 = b.randn([pSize4]);
      const ms4 = benchCustom(() => { (b as any).adamwStep(param4, grad4, m4, vState4, 3e-4, 0.9, 0.999, 1e-8, 0.1, 0.1, 0.001, 1.0); return []; });
      record("adamw_step_34M", ms4, { bytes: pSize4 * 4 * 7, note: "AdamW 4 layers (~34M params)" });
      release(param4); release(grad4); release(m4); release(vState4);
    }
  }

  // ── 8b. GRADIENT ACCUMULATION (add_inplace, scale_inplace) ───────────────

  if (typeof (b as any).addInplace === "function" && (shouldRun("grad_accum_lm_head") || shouldRun("grad_scale_lm_head"))) {
    const lmGrad = b.randn([1024, V]);
    const lmAcc = b.randn([1024, V]);
    const lmSize = 1024 * V;
    if (shouldRun("grad_accum_lm_head")) {
      const ms = benchCustom(() => { (b as any).addInplace(lmAcc, lmGrad); return []; });
      record("grad_accum_lm_head", ms, { bytes: lmSize * 4 * 3, note: "gradient accumulation (add_inplace)" });
    }
    if (shouldRun("grad_scale_lm_head") && typeof (b as any).scaleInplace === "function") {
      const msScale = benchCustom(() => { (b as any).scaleInplace(lmAcc, 0.5); return []; });
      record("grad_scale_lm_head", msScale, { bytes: lmSize * 4 * 2, note: "gradient clipping scale (scale_inplace)" });
    }
    release(lmGrad);
    release(lmAcc);
  }

  // ── 8b1.5. WEIGHT DECAY + FULL-MODEL IN-PLACE OPS ───────────────────────

  if (typeof (b as any).scaleInplace === "function") {
    const pSizeWd = (1024 * 3072 + 1024 * 1024 + 1024 * 2752 * 2 + 2752 * 1024) * 4;
    if (shouldRun("weight_decay_34M")) {
      const wdParams = b.randn([pSizeWd]);
      const msWd = benchCustom(() => { (b as any).scaleInplace(wdParams, 0.9997); return []; });
      record("weight_decay_34M", msWd, { bytes: pSizeWd * 4 * 2, note: "full-model weight decay (scale_inplace 34M)" });
      release(wdParams);
    }

    if (typeof (b as any).addInplace === "function" && (shouldRun("grad_accum_34M") || shouldRun("grad_scale_34M"))) {
      const accGrad = b.randn([pSizeWd]);
      const accAcc = b.randn([pSizeWd]);
      if (shouldRun("grad_accum_34M")) {
        const msAcc34 = benchCustom(() => { (b as any).addInplace(accAcc, accGrad); return []; });
        record("grad_accum_34M", msAcc34, { bytes: pSizeWd * 4 * 3, note: "full-model gradient accumulation (add_inplace 34M)" });
      }
      if (shouldRun("grad_scale_34M")) {
        const msScale34 = benchCustom(() => { (b as any).scaleInplace(accAcc, 0.5); return []; });
        record("grad_scale_34M", msScale34, { bytes: pSizeWd * 4 * 2, note: "full-model gradient scale (scale_inplace 34M)" });
      }
      release(accGrad);
      release(accAcc);
    }
  }

  // ── 8b1.6. LARGE ELEMENTWISE OPS (full-model sized) ─────────────────────

  {
    // Full model residual add: ~34M elements (same size as AdamW/weight_decay)
    const elSize = (1024 * 3072 + 1024 * 1024 + 1024 * 2752 * 2 + 2752 * 1024) * 4;
    const elA = b.randn([elSize]);
    const elB = b.randn([elSize]);

    if (shouldRun("add_34M")) { const ms = benchOp(() => b.add(elA, elB)); record("add_34M", ms, { bytes: elSize * 4 * 3, note: "full-model add (34M elements)" }); }
    if (shouldRun("scale_34M")) { const ms = benchOp(() => (b as any).scale(elA, 0.5)); record("scale_34M", ms, { bytes: elSize * 4 * 2, note: "full-model scale (34M elements)" }); }
    if (shouldRun("sub_34M")) { const ms = benchOp(() => b.sub(elA, elB)); record("sub_34M", ms, { bytes: elSize * 4 * 3, note: "full-model sub (34M elements)" }); }
    if (shouldRun("mul_34M")) { const ms = benchOp(() => b.mul(elA, elB)); record("mul_34M", ms, { bytes: elSize * 4 * 3, note: "full-model mul (34M elements)" }); }

    release(elA);
    release(elB);
  }

  // ── 8b2. DROPOUT MASK GENERATION ──────────────────────────────────────────

  if (shouldRun("dropout_mask_512x1024") && typeof (b as any).dropoutMask === "function") {
    const msDropout = benchOp(() => (b as any).dropoutMask([BT, D], 42, 0, 0.1));
    record("dropout_mask_512x1024", msDropout, { bytes: BT * D * 4, note: "GPU-side dropout mask (PhiloxRNG)" });
  }

  // ── 8c. GRADIENT NORM (sum of squares) ─────────────────────────────────────

  if (typeof (b as any).sumOfSquares === "function") {
    const pSize = 1024 * 3072 + 1024 * 1024 + 1024 * 2752 * 2 + 2752 * 1024;
    if (shouldRun("grad_norm_8.5M")) {
      const gradTensor = b.randn([pSize]);
      const msNorm = benchOp(() => (b as any).sumOfSquares(gradTensor));
      record("grad_norm_8.5M", msNorm, { bytes: pSize * 4, note: "sum of squares for gradient norm (fused)" });
      release(gradTensor);
    }
    if (shouldRun("grad_norm_34M")) {
      const gradTensor34 = b.randn([pSize * 4]);
      const msNorm34 = benchOp(() => (b as any).sumOfSquares(gradTensor34));
      record("grad_norm_34M", msNorm34, { bytes: pSize * 4 * 4, note: "full-model sum of squares (34M params)" });
      release(gradTensor34);
    }
  }

  // ── 9. FUSED OPS ──────────────────────────────────────────────────────────

  if (shouldRun("residual_dropout_add_512x1024") && typeof (b as any).residualDropoutAdd === "function") {
    const residual = b.randn([BT, D]);
    const projected = b.randn([BT, D]);
    const maskData = new Float32Array(BT * D);
    for (let i = 0; i < maskData.length; i++) maskData[i] = Math.random() > 0.1 ? 1.0 / 0.9 : 0;
    const mask: TensorData = { data: maskData, shape: [BT, D], dtype: "f32" };
    const ms = benchOp(() => (b as any).residualDropoutAdd(residual, projected, mask));
    record("residual_dropout_add_512x1024", ms, { bytes: BT * D * 4 * 4, note: "fused residual+dropout+add" });
    release(residual);
    release(projected);
  }

  // ── 10. KERNEL LAUNCH OVERHEAD ────────────────────────────────────────────

  if (shouldRun("launch_overhead_200_adds")) {
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

  if (shouldRun("transpose_16x512x64") && typeof (b as any).transpose === "function") {
    const t = b.randn([BH, T, Dh]);
    const ms = benchOp(() => (b as any).transpose(t, 1, 2));
    record("transpose_16x512x64", ms, { bytes: BH * T * Dh * 4 * 2, note: "transpose" });
    release(t);
  }

  // ── 11b. GRADIENT CLIP PIPELINE ──────────────────────────────────────────
  // Full gradient clipping: sumOfSquares → sqrt → max(1, norm/maxNorm) → scaleInplace
  // This tests sustained throughput of combined reduction + in-place ops
  if (shouldRun("grad_clip_pipeline_34M") && typeof (b as any).sumOfSquares === "function" && typeof (b as any).scaleInplace === "function") {
    const totalParams = 1024 * 3072 + 1024 * 1024 + 1024 * 2752 * 2 + 2752 * 1024;
    const allGrads: TensorData[] = [];
    for (let l = 0; l < 4; l++) allGrads.push(b.randn([totalParams]));

    const msClip = benchCustom(() => {
      // 1. Compute total grad norm across all layers
      let totalNorm = 0;
      for (const g of allGrads) {
        const ss = (b as any).sumOfSquares(g);
        sync();
        totalNorm += (ss.data as Float32Array)[0];
        release(ss);
      }
      // 2. Clip
      const norm = Math.sqrt(totalNorm);
      const maxNorm = 1.0;
      if (norm > maxNorm) {
        const scale = maxNorm / norm;
        for (const g of allGrads) (b as any).scaleInplace(g, scale);
      }
      return [];
    });
    record("grad_clip_pipeline_34M", msClip, {
      bytes: totalParams * 4 * 4 * 2, note: "4-layer gradient norm + clip pipeline",
    });

    for (const g of allGrads) release(g);
  }

  // ── 12. SLICE ─────────────────────────────────────────────────────────────

  if (shouldRun("slice_qkv_512x3072") && typeof (b as any).slice === "function") {
    const qkv = b.randn([BT, 3 * D]);
    const hasFused = typeof (b as any).sliceQkv === "function";
    const ms = benchCustom(() => {
      if (hasFused) {
        return (b as any).sliceQkv(qkv);
      }
      const q = (b as any).slice(qkv, [0, 0], [BT, D]);
      const k = (b as any).slice(qkv, [0, D], [BT, 2 * D]);
      const v = (b as any).slice(qkv, [0, 2 * D], [BT, 3 * D]);
      return [q, k, v];
    });
    // Reads full QKV [BT,3D], writes Q,K,V [BT,D] each
    record("slice_qkv_512x3072", ms, { bytes: BT * 3 * D * 4 * 2, note: hasFused ? "fused 3-way slice" : "3-way slice for Q,K,V" });
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
    pad("peak_ms", 9) +
    pad("h_tflops", 10) +
    pad("c_tflops", 10) +
    pad("h_gbps", 9) +
    pad("c_gbps", 9) +
    pad("h_eff%", 7) +
    pad("c_eff%", 7) +
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
    const peakMs = h?.peakMs ? h.peakMs.toFixed(3) : "";
    const hTf = h?.tflops ? h.tflops.toFixed(2) : "";
    const cTf = c?.tflops ? c.tflops.toFixed(2) : "";
    const hGb = h?.gbps ? h.gbps.toFixed(0) : "";
    const cGb = c?.gbps ? c.gbps.toFixed(0) : "";
    // Efficiency: peak_ms / actual_ms × 100%
    const hEff = (h?.peakMs && h.ms > 0) ? Math.round(h.peakMs / h.ms * 100).toString() : "";
    const cEff = (h?.peakMs && c && c.ms > 0) ? Math.round(h.peakMs / c.ms * 100).toString() : "";

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
      pad(peakMs, 9) +
      pad(hTf, 10) +
      pad(cTf, 10) +
      pad(hGb, 9) +
      pad(cGb, 9) +
      pad(hEff, 7) +
      pad(cEff, 7) +
      pad(winner, 10) +
      note,
    );
  }

  console.log("─".repeat(hdr.length));
  console.log(`\nScore: Helios wins ${heliosWins} / CUDA wins ${cudaWins} / ties ${ties}`);
  console.log("");
}

main();
