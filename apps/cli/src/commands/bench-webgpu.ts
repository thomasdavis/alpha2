/**
 * bench-webgpu.ts — WebGPU (Dawn) compute benchmark runner.
 *
 * Runs in an isolated child process to avoid Vulkan contention between
 * our helios native addon and Dawn's Vulkan backend. The parent (bench.ts)
 * spawns this as a worker via `fork()` and communicates via IPC messages.
 *
 * Can also be imported directly for standalone use.
 */

// ── WGSL Shaders ─────────────────────────────────────────────────────────────

const binaryShader = (op: string) => `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i < arrayLength(&a)) {
    out[i] = a[i] ${op} b[i];
  }
}
`;

const unaryShader = (expr: string) => `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i < arrayLength(&a)) {
    out[i] = ${expr};
  }
}
`;

const SHADERS: Record<string, string> = {
  add:   binaryShader("+"),
  mul:   binaryShader("*"),
  scale: unaryShader("a[i] * 2.0"),
  exp:   unaryShader("exp(a[i])"),
  neg:   unaryShader("-a[i]"),
};

const WG_SIZE = 256;

// ── Types ────────────────────────────────────────────────────────────────────

type WebGpuOp = "add" | "mul" | "scale" | "exp" | "neg";

interface BenchResources {
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  bufA: GPUBuffer;
  bufB: GPUBuffer | null;
  bufOut: GPUBuffer;
  readBuf: GPUBuffer;
  size: number;
}

/** Spec sent from parent to child process */
export interface WebGpuBenchSpec {
  ops: WebGpuOp[];
  sizes: number[];
  iters: number;
  checkSize: number;
}

/** Result row sent back from child */
export interface WebGpuBenchRow {
  op: string;
  size: number;
  ms: number;
}

/** Correctness result sent back from child */
export interface WebGpuCheckResult {
  op: string;
  pass: boolean;
}

/** Full result message from child */
export interface WebGpuBenchResult {
  available: boolean;
  rows: WebGpuBenchRow[];
  checks: WebGpuCheckResult[];
}

// ── GPU helpers ──────────────────────────────────────────────────────────────

function isBinaryOp(op: WebGpuOp): boolean {
  return op === "add" || op === "mul";
}

function createResources(
  device: GPUDevice,
  op: WebGpuOp,
  size: number,
  inputA: Float32Array,
  inputB: Float32Array | null,
): BenchResources {
  const code = SHADERS[op];
  const module = device.createShaderModule({ code });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });

  const byteLen = size * 4;

  const bufA = device.createBuffer({
    size: byteLen,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(bufA, 0, inputA);

  let bufB: GPUBuffer | null = null;
  if (isBinaryOp(op) && inputB) {
    bufB = device.createBuffer({
      size: byteLen,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(bufB, 0, inputB);
  }

  const bufOut = device.createBuffer({
    size: byteLen,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const readBuf = device.createBuffer({
    size: byteLen,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const entries: GPUBindGroupEntry[] = isBinaryOp(op)
    ? [
        { binding: 0, resource: { buffer: bufA } },
        { binding: 1, resource: { buffer: bufB! } },
        { binding: 2, resource: { buffer: bufOut } },
      ]
    : [
        { binding: 0, resource: { buffer: bufA } },
        { binding: 1, resource: { buffer: bufOut } },
      ];

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  return { pipeline, bindGroup, bufA, bufB, bufOut, readBuf, size };
}

async function dispatchCompute(device: GPUDevice, res: BenchResources): Promise<void> {
  const workgroups = Math.ceil(res.size / WG_SIZE);
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(res.pipeline);
  pass.setBindGroup(0, res.bindGroup);
  pass.dispatchWorkgroups(workgroups);
  pass.end();
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
}

async function dispatchWithCopy(device: GPUDevice, res: BenchResources): Promise<void> {
  const workgroups = Math.ceil(res.size / WG_SIZE);
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(res.pipeline);
  pass.setBindGroup(0, res.bindGroup);
  pass.dispatchWorkgroups(workgroups);
  pass.end();
  enc.copyBufferToBuffer(res.bufOut, 0, res.readBuf, 0, res.size * 4);
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
}

async function readback(res: BenchResources): Promise<Float32Array> {
  await res.readBuf.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(res.readBuf.getMappedRange().slice(0));
  res.readBuf.unmap();
  return data;
}

function destroyResources(res: BenchResources): void {
  res.bufA.destroy();
  if (res.bufB) res.bufB.destroy();
  res.bufOut.destroy();
  res.readBuf.destroy();
}

// ── Seeded RNG (matches @alpha/core SeededRng for reproducible data) ─────────

function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function generateF32(size: number, seed: number): Float32Array {
  const rand = mulberry32(seed);
  const out = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    // Box-Muller for normal distribution
    const u1 = rand() || 1e-10;
    const u2 = rand();
    out[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
  return out;
}

// ── CPU reference for correctness checking ──────────────────────────────────

function cpuRef(op: WebGpuOp, a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) {
    switch (op) {
      case "add":   out[i] = a[i] + b[i]; break;
      case "mul":   out[i] = a[i] * b[i]; break;
      case "scale": out[i] = a[i] * 2.0; break;
      case "exp":   out[i] = Math.exp(a[i]); break;
      case "neg":   out[i] = -a[i]; break;
    }
  }
  return out;
}

function allClose(a: Float32Array, b: Float32Array, atol = 1e-4): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (Math.abs(a[i] - b[i]) > atol + 1e-8 * Math.abs(b[i])) return false;
  }
  return true;
}

// ── Child process entry point ───────────────────────────────────────────────

async function runAsChild(spec: WebGpuBenchSpec): Promise<void> {
  const { create, globals } = await import("webgpu");
  Object.assign(globalThis, globals);
  const gpu: GPU = create([]);
  const adapter = await gpu.requestAdapter();
  if (!adapter) {
    process.send!({ available: false, rows: [], checks: [] } satisfies WebGpuBenchResult);
    return;
  }

  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });

  const rows: WebGpuBenchRow[] = [];
  let seedCounter = 1000;

  for (const op of spec.ops) {
    for (const size of spec.sizes) {
      const inputA = generateF32(size, seedCounter++);
      const inputB = isBinaryOp(op) ? generateF32(size, seedCounter++) : null;

      try {
        const res = createResources(device, op, size, inputA, inputB);
        // Warmup
        for (let i = 0; i < 3; i++) await dispatchCompute(device, res);
        // Timed
        const start = performance.now();
        for (let i = 0; i < spec.iters; i++) await dispatchCompute(device, res);
        const ms = (performance.now() - start) / spec.iters;
        destroyResources(res);
        rows.push({ op, size, ms });
      } catch {
        rows.push({ op, size, ms: NaN });
      }
    }
  }

  // Correctness checks
  const checks: WebGpuCheckResult[] = [];
  const checkA = generateF32(spec.checkSize, 42);
  const checkB = generateF32(spec.checkSize, 43);

  for (const op of spec.ops) {
    try {
      const res = createResources(device, op, spec.checkSize, checkA, isBinaryOp(op) ? checkB : null);
      await dispatchWithCopy(device, res);
      const gpuOut = await readback(res);
      destroyResources(res);
      const cpuOut = cpuRef(op, checkA, checkB);
      checks.push({ op, pass: allClose(gpuOut, cpuOut) });
    } catch {
      checks.push({ op, pass: false });
    }
  }

  process.send!({ available: true, rows, checks } satisfies WebGpuBenchResult);
}

// ── If run as child process, start immediately ──────────────────────────────

if (process.send) {
  process.on("message", (msg: WebGpuBenchSpec) => {
    runAsChild(msg).then(() => process.exit(0)).catch(() => process.exit(1));
  });
}
