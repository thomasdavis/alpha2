/**
 * Checkpoint save/load.
 *
 * Serializes model weights + optimizer state + RNG state + config metadata.
 */
import { Effect } from "effect";
import type { CheckpointState, Checkpoint, TensorData, OptimizerState, TokenizerArtifacts } from "@alpha/core";
import { CheckpointError } from "@alpha/core";
import type { GPTParams } from "@alpha/model";
import { collectParams } from "@alpha/model";
import type { Optimizer } from "@alpha/core";

// ── Serialization helpers ──────────────────────────────────────────────────

function serializeParams(params: Map<string, TensorData>): Record<string, { shape: number[]; data: number[] }> {
  const out: Record<string, { shape: number[]; data: number[] }> = {};
  for (const [name, td] of params) {
    out[name] = { shape: [...td.shape], data: Array.from(td.data) };
  }
  return out;
}

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
      try: async () => {
        const fs = await import("node:fs/promises");
        const fspath = await import("node:path");
        await fs.mkdir(fspath.dirname(path), { recursive: true });
        const serialized = {
          modelConfig: state.modelConfig,
          params: state.params,
          optimizerState: serializeOptimizerState(state.optimizerState),
          tokenizerArtifacts: state.tokenizerArtifacts,
          rngState: state.rngState,
          configHash: state.configHash,
          step: state.step,
        };
        await fs.writeFile(path, JSON.stringify(serialized));
      },
      catch: (e) => new CheckpointError({ message: `Failed to save checkpoint: ${e}`, cause: e }),
    });
  }

  load(path: string): Effect.Effect<CheckpointState, CheckpointError> {
    return Effect.tryPromise({
      try: async () => {
        const fs = await import("node:fs/promises");
        const raw = await fs.readFile(path, "utf-8");
        const parsed = JSON.parse(raw);
        // Reconstruct optimizer state with Map
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
    params[name] = { shape: [...v.data.shape], data: Array.from(v.data.data) };
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
    if (!saved) continue;
    const arr = variable.data.data as Float32Array;
    for (let i = 0; i < arr.length; i++) {
      arr[i] = saved.data[i];
    }
  }
}
