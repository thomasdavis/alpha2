/**
 * Symbio preset overrides for ModelConfig and TrainConfig.
 * Applied when --symbio is enabled, before explicit CLI overrides.
 */
import type { ModelConfig, TrainConfig } from "@alpha/core";

/** Apply symbio preset to ModelConfig. Returns a new config with symbio defaults. */
export function applySymbioModelPreset(config: ModelConfig): ModelConfig {
  const ffnDim = Math.ceil((8 / 3) * config.nEmbd / 64) * 64;
  return {
    ...config,
    ffnActivation: "swiglu",
    ffnDim: ffnDim,
  };
}

/** Apply symbio preset to TrainConfig. Returns a new config. */
export function applySymbioTrainPreset(config: TrainConfig): TrainConfig {
  return {
    ...config,
    lr: config.lr === 3e-4 ? 5e-5 : config.lr, // only override if at default
    gradClip: config.gradClip === 1.0 ? 5.0 : config.gradClip,
    warmupIters: config.warmupIters <= 0 ? 500 : config.warmupIters,
    spikeThreshold: config.spikeThreshold === 0 ? 10.0 : config.spikeThreshold,
  };
}
