/**
 * CusumDashboard: orchestrates 4 CUSUM monitors (gradNorm, clipPct, tokensPerSec, valLoss).
 * Produces per-step CUSUM statistics and alert bitmask.
 */
import { CusumMonitor } from "./cusum.js";
import { CUSUM_GRAD, CUSUM_CLIP, CUSUM_TPS, CUSUM_VAL, type TrainerStepInfo } from "../types.js";

export interface CusumResult {
  cusum_grad: number;
  cusum_clip: number;
  cusum_tps: number;
  cusum_val: number;
  cusum_alerts: number;
  cusum_alert_reason?: string;
}

export class CusumDashboard {
  private grad: CusumMonitor;
  private clip: CusumMonitor;
  private tps: CusumMonitor;
  private val: CusumMonitor;

  constructor(sensitivity: number, baselineWindow: number) {
    this.grad = new CusumMonitor("gradNorm", sensitivity, baselineWindow);
    this.clip = new CusumMonitor("clipPct", sensitivity, baselineWindow);
    this.tps = new CusumMonitor("tokensPerSec", sensitivity, baselineWindow);
    this.val = new CusumMonitor("valLoss", sensitivity, baselineWindow);
  }

  /** Update all monitors with the current step's metrics. */
  update(info: TrainerStepInfo): CusumResult {
    const cusum_grad = this.grad.update(info.gradNorm);
    const cusum_clip = this.clip.update(info.clipPct);
    const cusum_tps = this.tps.update(info.tokensPerSec);
    const cusum_val = info.valLoss !== undefined ? this.val.update(info.valLoss) : this.val.value;

    let alerts = 0;
    const reasons: string[] = [];

    if (this.grad.alerted) {
      alerts |= CUSUM_GRAD;
      reasons.push("grad_norm regime shift");
    }
    if (this.clip.alerted) {
      alerts |= CUSUM_CLIP;
      reasons.push("persistent clipping onset");
    }
    if (this.tps.alerted) {
      alerts |= CUSUM_TPS;
      reasons.push("throughput collapse");
    }
    if (this.val.alerted) {
      alerts |= CUSUM_VAL;
      reasons.push("validation loss divergence");
    }

    return {
      cusum_grad,
      cusum_clip,
      cusum_tps,
      cusum_val,
      cusum_alerts: alerts,
      cusum_alert_reason: reasons.length > 0 ? reasons.join("; ") : undefined,
    };
  }

  /** Get the current alert bitmask without feeding new data. */
  get alertMask(): number {
    let mask = 0;
    if (this.grad.alerted) mask |= CUSUM_GRAD;
    if (this.clip.alerted) mask |= CUSUM_CLIP;
    if (this.tps.alerted) mask |= CUSUM_TPS;
    if (this.val.alerted) mask |= CUSUM_VAL;
    return mask;
  }
}
