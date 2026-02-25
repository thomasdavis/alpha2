import type { SymbioConfig } from "../config/schema.js";

export interface PopulationDynamicsInput {
  step: number;
  loss: number;
  lr: number;
  cusumAlerts: number;
}

export interface PopulationDynamicsState {
  populationScale: number;
  effectivePopulationSize: number;
  mutationRate: number;
  explorePressure: number;
  convergePressure: number;
  plateauPressure: number;
  cusumPressure: number;
  lossSlope: number;
  changed: boolean;
  reason?: string;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function bitCount32(v: number): number {
  let x = v | 0;
  let c = 0;
  while (x) {
    c += x & 1;
    x >>>= 1;
  }
  return c;
}

/**
 * Adaptive population/mutation controller.
 *
 * Uses loss frontier movement, current LR, and CUSUM alerts to shift between
 * exploration (larger population / higher mutation) and exploitation
 * (smaller population / lower mutation).
 */
export class PopulationDynamicsController {
  private readonly basePopulation: number;
  private readonly baseMutationRate: number;
  private readonly cfg: SymbioConfig;

  private populationScale = 1.0;
  private maxLrSeen = 0;
  private lossFastEma: number | null = null;
  private lossSlowEma: number | null = null;
  private runningBest: number | null = null;
  private lastAdjustStep = -1_000_000;

  constructor(basePopulation: number, baseMutationRate: number, cfg: SymbioConfig) {
    this.basePopulation = Math.max(2, basePopulation);
    this.baseMutationRate = clamp(baseMutationRate, 0, 1);
    this.cfg = cfg;
  }

  update(input: PopulationDynamicsInput): PopulationDynamicsState {
    this.maxLrSeen = Math.max(this.maxLrSeen, input.lr);
    const lrNorm = this.maxLrSeen > 0 ? clamp(input.lr / this.maxLrSeen, 0, 1) : 0;

    if (this.lossFastEma == null) {
      this.lossFastEma = input.loss;
      this.lossSlowEma = input.loss;
      this.runningBest = input.loss;
    } else {
      const fastA = 0.2;
      const slowA = 0.03;
      this.lossFastEma = fastA * input.loss + (1 - fastA) * this.lossFastEma;
      this.lossSlowEma = slowA * input.loss + (1 - slowA) * this.lossSlowEma!;
      this.runningBest = Math.min(this.runningBest ?? input.loss, input.loss);
    }

    const slow = this.lossSlowEma ?? input.loss;
    const fast = this.lossFastEma ?? input.loss;
    const denom = Math.max(1e-6, Math.abs(slow));
    // Positive = improving (fast below slow), Negative = worsening
    const lossSlope = clamp((slow - fast) / denom, -1, 1);
    const lossSlopeScore = clamp(lossSlope * 100, 0, 1);

    const best = this.runningBest ?? input.loss;
    const plateauGap = clamp((input.loss - best) / Math.max(1e-6, Math.abs(best)), 0, 1);
    const plateauPressure = clamp(plateauGap * 4, 0, 1);

    const cusumPressure = clamp(bitCount32(input.cusumAlerts) / 4, 0, 1);

    const explorePressure = clamp(
      0.45 * plateauPressure +
      0.35 * cusumPressure +
      0.20 * lrNorm,
      0,
      1,
    );
    const convergePressure = clamp(
      0.55 * lossSlopeScore +
      0.25 * (1 - cusumPressure) +
      0.20 * (1 - lrNorm),
      0,
      1,
    );

    let changed = false;
    let reason: string | undefined;
    const drive = explorePressure - convergePressure;
    const canAdjust = (input.step - this.lastAdjustStep) >= this.cfg.populationAdaptationCooldown;
    if (this.cfg.populationAdaptation && canAdjust && Math.abs(drive) > 0.08) {
      const step = this.cfg.populationScaleStep;
      const nextScale = clamp(
        this.populationScale + Math.sign(drive) * step,
        this.cfg.populationScaleMin,
        this.cfg.populationScaleMax,
      );
      if (Math.abs(nextScale - this.populationScale) > 1e-9) {
        this.populationScale = nextScale;
        this.lastAdjustStep = input.step;
        changed = true;
        reason = drive > 0
          ? "explore↑ (plateau/cusum/lr pressure)"
          : "converge↑ (frontier motion/stability)";
      }
    }

    const effectivePopulationSize = Math.max(
      2,
      Math.round(this.basePopulation * this.populationScale),
    );

    const driveNorm = clamp((drive + 1) / 2, 0, 1);
    const targetMutation = this.baseMutationRate + (driveNorm - 0.5) * 0.5;
    const mutationRate = clamp(
      targetMutation,
      this.cfg.mutationRateMin,
      this.cfg.mutationRateMax,
    );

    return {
      populationScale: this.populationScale,
      effectivePopulationSize,
      mutationRate,
      explorePressure,
      convergePressure,
      plateauPressure,
      cusumPressure,
      lossSlope,
      changed,
      reason,
    };
  }
}
