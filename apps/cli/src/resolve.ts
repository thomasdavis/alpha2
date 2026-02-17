/**
 * Resolve pluggable implementations from CLI args.
 */
import { backendRegistry } from "@alpha/tensor";
import { tokenizerRegistry } from "@alpha/tokenizers";
import { createOptimizerRegistry } from "@alpha/train";
import type { Backend, Tokenizer, Optimizer, Rng } from "@alpha/core";
import { SeededRng } from "@alpha/core";

export function resolveBackend(name: string): Backend {
  return backendRegistry.get(name);
}

export function resolveTokenizer(name: string): Tokenizer {
  return tokenizerRegistry.get(name);
}

export function resolveOptimizer(name: string, backend: Backend): Optimizer {
  return createOptimizerRegistry(backend).get(name);
}

export function resolveRng(seed: number): Rng {
  return new SeededRng(seed);
}

export function listImplementations(): string {
  return [
    `Tokenizers: ${tokenizerRegistry.list().join(", ")}`,
    `Backends:   ${backendRegistry.list().join(", ")}`,
    `Optimizers: adamw, sgd`,
  ].join("\n");
}
