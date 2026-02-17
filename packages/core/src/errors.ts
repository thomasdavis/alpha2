/**
 * Typed error classes for every subsystem.
 */
import { Data } from "effect";

export class TokenizerError extends Data.TaggedError("TokenizerError")<{
  readonly message: string;
  readonly cause?: unknown;
}> {}

export class BackendError extends Data.TaggedError("BackendError")<{
  readonly message: string;
  readonly cause?: unknown;
}> {}

export class AutogradError extends Data.TaggedError("AutogradError")<{
  readonly message: string;
  readonly cause?: unknown;
}> {}

export class OptimizerError extends Data.TaggedError("OptimizerError")<{
  readonly message: string;
  readonly cause?: unknown;
}> {}

export class CheckpointError extends Data.TaggedError("CheckpointError")<{
  readonly message: string;
  readonly cause?: unknown;
}> {}

export class ModelError extends Data.TaggedError("ModelError")<{
  readonly message: string;
  readonly cause?: unknown;
}> {}

export class TrainError extends Data.TaggedError("TrainError")<{
  readonly message: string;
  readonly cause?: unknown;
}> {}

export class ConfigError extends Data.TaggedError("ConfigError")<{
  readonly message: string;
  readonly cause?: unknown;
}> {}
