/**
 * Effect layers for dependency injection.
 *
 * Each service gets a Layer that constructs it from config.
 */
import { Layer, Effect, Context } from "effect";
import {
  BackendService, TokenizerService, OptimizerService, CheckpointService, RngService,
  type Backend, type Tokenizer, type Optimizer, type Checkpoint, type Rng,
  SeededRng,
  type TrainConfig, type ModelConfig,
} from "@alpha/core";

// ── RNG Layer ──────────────────────────────────────────────────────────────

export const RngLive = (seed: number) =>
  Layer.succeed(RngService, new SeededRng(seed) as Rng);

// ── Backend Layer ──────────────────────────────────────────────────────────

export const BackendLive = (backendName: string) =>
  Layer.effect(
    BackendService,
    Effect.sync(() => {
      // Dynamic import would be ideal but we keep it simple for now
      // The CLI resolves backends before constructing layers
      throw new Error(`BackendLive: use BackendFromRegistry instead`);
    }),
  );

export const BackendFrom = (backend: Backend) =>
  Layer.succeed(BackendService, backend);

// ── Tokenizer Layer ────────────────────────────────────────────────────────

export const TokenizerFrom = (tokenizer: Tokenizer) =>
  Layer.succeed(TokenizerService, tokenizer);

// ── Optimizer Layer ────────────────────────────────────────────────────────

export const OptimizerFrom = (optimizer: Optimizer) =>
  Layer.succeed(OptimizerService, optimizer);

// ── Checkpoint Layer ───────────────────────────────────────────────────────

export const CheckpointFrom = (checkpoint: Checkpoint) =>
  Layer.succeed(CheckpointService, checkpoint);
