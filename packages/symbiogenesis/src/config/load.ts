/**
 * Load and validate SymbioConfig from file, merge with defaults.
 */
import { defaultSymbioConfig, type SymbioConfig } from "./schema.js";

/** Load a SymbioConfig from a JSON file path, merging with defaults. */
export async function loadSymbioConfig(path?: string): Promise<SymbioConfig> {
  if (!path) return { ...defaultSymbioConfig };

  const fs = await import("node:fs/promises");
  const raw = await fs.readFile(path, "utf-8");
  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new Error(`Failed to parse symbio config at ${path}: invalid JSON`);
  }

  const config = { ...defaultSymbioConfig, ...parsed } as SymbioConfig;
  validateSymbioConfig(config);
  return config;
}

/** Validate a SymbioConfig, throwing on invalid values. */
export function validateSymbioConfig(config: SymbioConfig): void {
  if (config.cusumSensitivity <= 0) {
    throw new Error(`cusumSensitivity must be > 0, got ${config.cusumSensitivity}`);
  }
  if (config.cusumBaselineWindow < 2) {
    throw new Error(`cusumBaselineWindow must be >= 2, got ${config.cusumBaselineWindow}`);
  }
  if (config.metricsInterval < 1) {
    throw new Error(`metricsInterval must be >= 1, got ${config.metricsInterval}`);
  }
  if (config.populationAdaptation) {
    if (config.populationScaleMin <= 0) {
      throw new Error(`populationScaleMin must be > 0, got ${config.populationScaleMin}`);
    }
    if (config.populationScaleMax < config.populationScaleMin) {
      throw new Error(`populationScaleMax (${config.populationScaleMax}) must be >= populationScaleMin (${config.populationScaleMin})`);
    }
    if (config.populationScaleStep <= 0) {
      throw new Error(`populationScaleStep must be > 0, got ${config.populationScaleStep}`);
    }
    if (config.populationAdaptationCooldown < 0) {
      throw new Error(`populationAdaptationCooldown must be >= 0, got ${config.populationAdaptationCooldown}`);
    }
    if (config.mutationRateMin < 0 || config.mutationRateMin > 1) {
      throw new Error(`mutationRateMin must be in [0,1], got ${config.mutationRateMin}`);
    }
    if (config.mutationRateMax < config.mutationRateMin || config.mutationRateMax > 1) {
      throw new Error(`mutationRateMax must be in [mutationRateMin,1], got ${config.mutationRateMax}`);
    }
  }
  if (config.batchMin < 1) {
    throw new Error(`batchMin must be >= 1, got ${config.batchMin}`);
  }
  if (config.batchMax < config.batchMin) {
    throw new Error(`batchMax (${config.batchMax}) must be >= batchMin (${config.batchMin})`);
  }
  if (config.batchStep < 1) {
    throw new Error(`batchStep must be >= 1, got ${config.batchStep}`);
  }
  if (config.populationSize < 2) {
    throw new Error(`populationSize must be >= 2, got ${config.populationSize}`);
  }
  if (config.generations < 1) {
    throw new Error(`generations must be >= 1, got ${config.generations}`);
  }
  if (config.stepsPerCandidate < 1) {
    throw new Error(`stepsPerCandidate must be >= 1, got ${config.stepsPerCandidate}`);
  }
  if (config.fusionShadowEma < 0 || config.fusionShadowEma > 1) {
    throw new Error(`fusionShadowEma must be in [0,1], got ${config.fusionShadowEma}`);
  }
  if (config.fusionBaseStrength < 0 || config.fusionBaseStrength > 1) {
    throw new Error(`fusionBaseStrength must be in [0,1], got ${config.fusionBaseStrength}`);
  }
  if (config.fusionMaxStrength < config.fusionBaseStrength || config.fusionMaxStrength > 1) {
    throw new Error(`fusionMaxStrength must be in [fusionBaseStrength,1], got ${config.fusionMaxStrength}`);
  }
  if (config.kuramotoCoupling < 0) {
    throw new Error(`kuramotoCoupling must be >= 0, got ${config.kuramotoCoupling}`);
  }
  if (config.kuramotoDt <= 0) {
    throw new Error(`kuramotoDt must be > 0, got ${config.kuramotoDt}`);
  }
  if (config.kuramotoDamping < 0 || config.kuramotoDamping >= 1) {
    throw new Error(`kuramotoDamping must be in [0,1), got ${config.kuramotoDamping}`);
  }
  if (config.searchMode === "composed-activation-search") {
    // Composed mode — validate basis pool
    const validBases = new Set(["silu", "relu", "gelu", "identity", "square"]);
    for (const b of config.basisPool ?? []) {
      if (!validBases.has(b)) {
        throw new Error(`Invalid basis in pool: "${b}". Valid: ${[...validBases].join(", ")}`);
      }
    }
  } else {
    // Fixed-activation mode — validate activation pool
    const validActivations = new Set(["gelu", "relu", "silu", "swiglu", "universal", "kan_spline"]);
    for (const act of config.activationPool) {
      if (!validActivations.has(act)) {
        throw new Error(`Invalid activation in pool: "${act}". Valid: ${[...validActivations].join(", ")}`);
      }
    }
  }
}
