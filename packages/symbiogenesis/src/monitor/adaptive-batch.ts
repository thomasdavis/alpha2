/**
 * Adaptive batch sizing responding to CUSUM alerts.
 *
 * - CUSUM grad alert (0x01): reduce batch — smaller batches recover faster from instability
 * - CUSUM clip alert (0x02): increase batch — larger batches smooth gradients
 * - CUSUM throughput alert (0x04): reduce batch — GPU may be under memory pressure
 * - No alerts for calmStepsBeforeRestore steps: restore toward original batch by batchStep
 */
import { CUSUM_GRAD, CUSUM_CLIP, CUSUM_TPS } from "../types.js";
import type { SymbioConfig } from "../config/schema.js";

export class AdaptiveBatch {
  private currentBatch: number;
  private readonly originalBatch: number;
  private readonly min: number;
  private readonly max: number;
  private readonly step: number;
  private readonly calmThreshold: number;
  private calmSteps = 0;
  private lastChangeReason?: string;

  constructor(initialBatch: number, config: SymbioConfig) {
    this.currentBatch = initialBatch;
    this.originalBatch = initialBatch;
    this.min = config.batchMin;
    this.max = config.batchMax;
    this.step = config.batchStep;
    this.calmThreshold = config.calmStepsBeforeRestore;
  }

  /** Process a CUSUM alert bitmask. Returns the new batch size. */
  onAlert(alertMask: number): number {
    if (alertMask === 0) {
      this.calmSteps++;
      // Restore toward original batch after calm period
      if (this.calmSteps >= this.calmThreshold && this.currentBatch !== this.originalBatch) {
        if (this.currentBatch < this.originalBatch) {
          this.currentBatch = Math.min(this.currentBatch + this.step, this.originalBatch);
          this.lastChangeReason = "restore";
        } else if (this.currentBatch > this.originalBatch) {
          this.currentBatch = Math.max(this.currentBatch - this.step, this.originalBatch);
          this.lastChangeReason = "restore";
        }
        this.calmSteps = 0;
      }
      return this.currentBatch;
    }

    this.calmSteps = 0;

    if (alertMask & CUSUM_GRAD) {
      this.currentBatch = Math.max(this.min, this.currentBatch - this.step);
      this.lastChangeReason = "cusum_grad";
    }
    if (alertMask & CUSUM_TPS) {
      this.currentBatch = Math.max(this.min, this.currentBatch - this.step);
      this.lastChangeReason = "cusum_tps";
    }
    if (alertMask & CUSUM_CLIP) {
      this.currentBatch = Math.min(this.max, this.currentBatch + this.step);
      this.lastChangeReason = "cusum_clip";
    }

    return this.currentBatch;
  }

  get batchSize(): number {
    return this.currentBatch;
  }

  get changeReason(): string | undefined {
    return this.lastChangeReason;
  }

  /** Clear the last change reason after it has been reported. */
  clearChangeReason(): void {
    this.lastChangeReason = undefined;
  }
}
