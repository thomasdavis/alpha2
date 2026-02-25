import type { SymbioConfig } from "../config/schema.js";

export interface KuramotoFusionInput {
  loss: number;
  lr: number;
  cusumAlerts: number;
  switchedCandidate?: boolean;
}

export interface KuramotoFusionState {
  fusionAlpha: number;
  phaseGap: number;
  sync: number;
  order: number;
  thetaModel: number;
  thetaConsensus: number;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function wrapAngle(theta: number): number {
  let t = theta;
  while (t > Math.PI) t -= 2 * Math.PI;
  while (t < -Math.PI) t += 2 * Math.PI;
  return t;
}

function angularDiff(a: number, b: number): number {
  return wrapAngle(a - b);
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
 * Kuramoto-inspired 2-oscillator synchronizer (model vs consensus shadow).
 *
 * We use it to modulate step-wise fusion strength:
 * - in-phase => stronger fusion (stabilize and consolidate)
 * - out-of-phase => weaker fusion (allow exploration / transient divergence)
 */
export class KuramotoFusionController {
  private readonly cfg: SymbioConfig;
  private thetaModel = 0;
  private thetaConsensus = 0;
  private lastLoss: number | null = null;
  private maxLrSeen = 0;

  constructor(cfg: SymbioConfig) {
    this.cfg = cfg;
  }

  update(input: KuramotoFusionInput): KuramotoFusionState {
    this.maxLrSeen = Math.max(this.maxLrSeen, input.lr);
    const lrNorm = this.maxLrSeen > 0 ? clamp(input.lr / this.maxLrSeen, 0, 1) : 0;
    const cusumPressure = clamp(bitCount32(input.cusumAlerts) / 4, 0, 1);

    const prevLoss = this.lastLoss ?? input.loss;
    const lossDelta = input.loss - prevLoss;
    this.lastLoss = input.loss;

    // Improving loss -> slightly lower natural frequency (more settling).
    const normDelta = clamp(lossDelta / Math.max(1e-6, Math.abs(prevLoss)), -1, 1);
    const omegaModel = (0.4 * lrNorm) + (normDelta * 12);
    const omegaConsensus = 0.05 * (1 - lrNorm);

    const K = this.cfg.kuramotoCoupling * (1 + 0.5 * cusumPressure);
    const dt = this.cfg.kuramotoDt;
    const damping = 1 - this.cfg.kuramotoDamping;

    // Candidate switch perturbs model phase so re-synchronization becomes visible.
    if (input.switchedCandidate) {
      this.thetaModel = wrapAngle(this.thetaModel + Math.PI / 6);
    }

    const dm = omegaModel + K * Math.sin(this.thetaConsensus - this.thetaModel);
    const dc = omegaConsensus + K * 0.5 * Math.sin(this.thetaModel - this.thetaConsensus);

    this.thetaModel = wrapAngle((this.thetaModel + dt * dm) * damping);
    this.thetaConsensus = wrapAngle((this.thetaConsensus + dt * dc) * damping);

    const gap = Math.abs(angularDiff(this.thetaModel, this.thetaConsensus));
    const sync = clamp((1 + Math.cos(gap)) * 0.5, 0, 1); // coherence proxy

    // Order parameter proxy for 2 oscillators.
    const order = Math.sqrt(
      Math.pow((Math.cos(this.thetaModel) + Math.cos(this.thetaConsensus)) / 2, 2) +
      Math.pow((Math.sin(this.thetaModel) + Math.sin(this.thetaConsensus)) / 2, 2),
    );

    let fusionAlpha = this.cfg.fusionBaseStrength + sync * (this.cfg.fusionMaxStrength - this.cfg.fusionBaseStrength);
    // Increase stabilization under alerts; reduce if heavily desynchronized.
    fusionAlpha *= (1 + 0.35 * cusumPressure) * (0.65 + 0.35 * order);
    fusionAlpha = clamp(fusionAlpha, 0, this.cfg.fusionMaxStrength);

    return {
      fusionAlpha,
      phaseGap: gap,
      sync,
      order,
      thetaModel: this.thetaModel,
      thetaConsensus: this.thetaConsensus,
    };
  }
}

