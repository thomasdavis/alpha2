/**
 * Checkpoint save/load.
 *
 * Binary format (v2): compact binary with raw Float32 tensor data.
 * JSON format (v1): legacy, for small models or backward compat.
 *
 * Binary layout:
 *   [4 bytes: magic "ALPH"]
 *   [4 bytes: uint32 LE header JSON byte length]
 *   [N bytes: header JSON (UTF-8)]
 *   [remaining: concatenated raw Float32 tensor data]
 */
import { Effect } from "effect";
import type { CheckpointState, Checkpoint, TensorData, OptimizerState, TokenizerArtifacts } from "@alpha/core";
import { CheckpointError } from "@alpha/core";
import type { GPTParams } from "@alpha/model";
import { collectParams } from "@alpha/model";
import type { Optimizer } from "@alpha/core";

const MAGIC = Buffer.from("ALPH");

// ── Binary save ──────────────────────────────────────────────────────────────

interface TensorEntry {
  name: string;
  shape: number[];
  elements: number;
}

function saveBinary(
  path: string,
  state: CheckpointState,
): Promise<void> {
  // Collect all tensors in order: params first, then optimizer buffers
  const tensors: TensorEntry[] = [];
  const f32Arrays: Float32Array[] = [];

  for (const [name, p] of Object.entries(state.params)) {
    const data = p.data;
    const f32 = data instanceof Float32Array ? data : new Float32Array(data);
    tensors.push({ name: `p.${name}`, shape: p.shape, elements: f32.length });
    f32Arrays.push(f32);
  }

  if (state.optimizerState?.buffers) {
    for (const [name, td] of state.optimizerState.buffers) {
      const data = td.data;
      const f32 = data instanceof Float32Array ? data : new Float32Array(data as any);
      tensors.push({ name: `o.${name}`, shape: [...td.shape], elements: f32.length });
      f32Arrays.push(f32);
    }
  }

  const header = JSON.stringify({
    modelConfig: state.modelConfig,
    configHash: state.configHash,
    rngState: state.rngState,
    step: state.step,
    tokenizerArtifacts: state.tokenizerArtifacts,
    optimizerStep: state.optimizerState?.step ?? 0,
    tensors,
  });

  const headerBuf = Buffer.from(header, "utf-8");

  // Stream writes: magic(4) + headerLen(4) + header + tensor data
  // Avoids allocating a single buffer for the entire file (can be 200MB+).
  return import("node:fs/promises").then(async (fs) => {
    const fspath = await import("node:path");
    await fs.mkdir(fspath.dirname(path), { recursive: true });

    const handle = await fs.open(path, "w");
    try {
      // Write magic
      await handle.write(MAGIC);
      // Write header length (uint32 LE)
      const lenBuf = Buffer.alloc(4);
      lenBuf.writeUInt32LE(headerBuf.length, 0);
      await handle.write(lenBuf);
      // Write header JSON
      await handle.write(headerBuf);
      // Write each tensor chunk sequentially
      for (const f32 of f32Arrays) {
        await handle.write(Buffer.from(f32.buffer, f32.byteOffset, f32.byteLength));
      }
    } finally {
      await handle.close();
    }
  });
}

// ── Binary load ──────────────────────────────────────────────────────────────

function loadBinary(data: Buffer): CheckpointState {
  let offset = 4; // skip magic
  const headerLen = data.readUInt32LE(offset); offset += 4;
  const header = JSON.parse(data.subarray(offset, offset + headerLen).toString("utf-8"));
  offset += headerLen;

  const params: Record<string, { shape: number[]; data: number[] }> = {};
  const optBuffers = new Map<string, TensorData>();

  for (const t of header.tensors as TensorEntry[]) {
    const byteLen = t.elements * 4;
    const f32 = new Float32Array(data.buffer.slice(data.byteOffset + offset, data.byteOffset + offset + byteLen));
    offset += byteLen;

    if (t.name.startsWith("p.")) {
      // Pass Float32Array directly — restoreParams indexes numerically so this works
      params[t.name.slice(2)] = { shape: t.shape, data: f32 as any };
    } else if (t.name.startsWith("o.")) {
      optBuffers.set(t.name.slice(2), { shape: t.shape, dtype: "f32", data: f32 });
    }
  }

  return {
    modelConfig: header.modelConfig,
    params,
    optimizerState: { step: header.optimizerStep ?? 0, buffers: optBuffers },
    tokenizerArtifacts: header.tokenizerArtifacts,
    rngState: header.rngState,
    configHash: header.configHash,
    step: header.step,
  };
}

// ── Legacy JSON helpers ──────────────────────────────────────────────────────

function serializeOptimizerState(state: OptimizerState): {
  step: number;
  buffers: Record<string, { shape: number[]; data: number[] }>;
} {
  const buffers: Record<string, { shape: number[]; data: number[] }> = {};
  for (const [k, v] of state.buffers) {
    buffers[k] = { shape: [...v.shape], data: Array.from(v.data) };
  }
  return { step: state.step, buffers };
}

// ── FileCheckpoint ─────────────────────────────────────────────────────────

export class FileCheckpoint implements Checkpoint {
  save(path: string, state: CheckpointState): Effect.Effect<void, CheckpointError> {
    return Effect.tryPromise({
      try: () => saveBinary(path, state),
      catch: (e) => new CheckpointError({ message: `Failed to save checkpoint: ${e}`, cause: e }),
    });
  }

  load(path: string): Effect.Effect<CheckpointState, CheckpointError> {
    return Effect.tryPromise({
      try: async () => {
        const fs = await import("node:fs/promises");
        const raw = await fs.readFile(path);

        // Detect format: binary starts with "ALPH", JSON starts with "{"
        if (raw.length >= 4 && raw[0] === 0x41 && raw[1] === 0x4c && raw[2] === 0x50 && raw[3] === 0x48) {
          return loadBinary(raw as unknown as Buffer);
        }

        // Legacy JSON format
        const parsed = JSON.parse(raw.toString("utf-8"));
        const buffers = new Map<string, TensorData>();
        if (parsed.optimizerState?.buffers) {
          for (const [k, v] of Object.entries(parsed.optimizerState.buffers) as [string, any][]) {
            buffers.set(k, {
              shape: v.shape,
              dtype: "f32",
              data: new Float32Array(v.data),
            });
          }
        }
        return {
          modelConfig: parsed.modelConfig,
          params: parsed.params,
          optimizerState: { step: parsed.optimizerState?.step ?? 0, buffers },
          tokenizerArtifacts: parsed.tokenizerArtifacts,
          rngState: parsed.rngState,
          configHash: parsed.configHash,
          step: parsed.step,
        } satisfies CheckpointState;
      },
      catch: (e) => new CheckpointError({ message: `Failed to load checkpoint: ${e}`, cause: e }),
    });
  }
}

/** Save a full training state to checkpoint. */
export function buildCheckpointState(
  gptParams: GPTParams,
  optimizer: Optimizer,
  rngState: number,
  configHash: string,
  step: number,
  modelConfig: any,
  tokenizerArtifacts?: TokenizerArtifacts,
): CheckpointState {
  const paramMap = collectParams(gptParams);
  const params: Record<string, { shape: number[]; data: number[] }> = {};
  for (const [name, v] of paramMap) {
    // Pass Float32Array directly — saveBinary handles it, cast for interface
    params[name] = { shape: [...v.data.shape], data: v.data.data as any };
  }
  return {
    modelConfig,
    params,
    optimizerState: optimizer.stateDict(),
    tokenizerArtifacts,
    rngState,
    configHash,
    step,
  };
}

/** Restore parameters from checkpoint state into existing Variables. */
export function restoreParams(
  gptParams: GPTParams,
  checkpointParams: Record<string, { shape: number[]; data: number[] }>,
): void {
  const paramMap = collectParams(gptParams);
  for (const [name, variable] of paramMap) {
    const saved = checkpointParams[name];
    if (saved) {
      const arr = variable.data.data as Float32Array;
      for (let i = 0; i < arr.length; i++) {
        arr[i] = saved.data[i];
      }
      continue;
    }
    // Backward compat: old checkpoints have separate wq/wk/wv instead of wqkv
    if (name.endsWith(".attn.wqkv")) {
      const prefix = name.replace(".attn.wqkv", "");
      const wq = checkpointParams[`${prefix}.attn.wq`];
      const wk = checkpointParams[`${prefix}.attn.wk`];
      const wv = checkpointParams[`${prefix}.attn.wv`];
      if (wq && wk && wv) {
        const arr = variable.data.data as Float32Array;
        let offset = 0;
        for (const src of [wq, wk, wv]) {
          for (let i = 0; i < src.data.length; i++) {
            arr[offset++] = src.data[i];
          }
        }
      }
    }
  }
}
