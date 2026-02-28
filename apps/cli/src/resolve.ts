/**
 * Resolve pluggable implementations from CLI args.
 */
import { backendRegistry } from "@alpha/tensor";
import { heliosRegistry } from "@alpha/helios";
import { tokenizerRegistry } from "@alpha/tokenizers";
import { AdamW, Lion, Adafactor, createOptimizerRegistry } from "@alpha/train";
import type { AdamWConfig, LionConfig, AdafactorConfig } from "@alpha/train";
import type { Backend, Tokenizer, Optimizer, Rng } from "@alpha/core";
import { SeededRng } from "@alpha/core";

// Register helios GPU backend alongside CPU backends
for (const name of heliosRegistry.list()) {
  backendRegistry.register(name, () => heliosRegistry.get(name));
}

export function resolveBackend(name: string): Backend {
  return backendRegistry.get(name);
}

export function resolveTokenizer(name: string): Tokenizer {
  return tokenizerRegistry.get(name);
}

export function resolveOptimizer(name: string, backend: Backend, config?: Partial<AdamWConfig & LionConfig & AdafactorConfig>): Optimizer {
  if (name === "adamw") {
    return new AdamW(backend, config);
  }
  if (name === "lion") {
    return new Lion(backend, config);
  }
  if (name === "adafactor") {
    return new Adafactor(backend, config);
  }
  return createOptimizerRegistry(backend).get(name);
}

export function resolveRng(seed: number): Rng {
  return new SeededRng(seed);
}

export function listImplementations(): string {
  return [
    `Tokenizers: ${tokenizerRegistry.list().join(", ")}`,
    `Backends:   ${backendRegistry.list().join(", ")}`,
    `Optimizers: adamw, lion, adafactor, sgd`,
  ].join("\n");
}
