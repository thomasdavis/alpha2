/**
 * CUSUM (Cumulative Sum) change-point detector.
 * Inspired by symbiogenesis/monitor.py:GelationMonitor — written from scratch.
 *
 * Uses one-sided upper Page's test:
 *   baseline: first baselineWindow steps establish mean μ and std σ
 *   deviation(t) = (signal(t) - μ) / σ
 *   S(t) = max(0, S(t-1) + deviation(t))
 *   alert when S(t) > sensitivity
 */

export class CusumMonitor {
  readonly name: string;
  private readonly sensitivity: number;
  private readonly baselineWindow: number;
  private baselineValues: number[] = [];
  private mu = 0;
  private sigma = 1;
  private baselineReady = false;
  private S = 0;
  private _alerted = false;

  constructor(name: string, sensitivity: number, baselineWindow: number) {
    this.name = name;
    this.sensitivity = sensitivity;
    this.baselineWindow = baselineWindow;
  }

  /** Feed a new signal value. Returns the current CUSUM statistic S(t). */
  update(value: number): number {
    if (!this.baselineReady) {
      this.baselineValues.push(value);
      if (this.baselineValues.length >= this.baselineWindow) {
        this.computeBaseline();
        this.baselineReady = true;
      }
      return 0;
    }

    const deviation = this.sigma > 0 ? (value - this.mu) / this.sigma : 0;
    this.S = Math.max(0, this.S + deviation);
    this._alerted = this.S > this.sensitivity;
    return this.S;
  }

  /** Whether the current CUSUM statistic exceeds the sensitivity threshold. */
  get alerted(): boolean {
    return this._alerted;
  }

  /** Current CUSUM statistic value. */
  get value(): number {
    return this.S;
  }

  /** Reset the CUSUM accumulator (e.g., after responding to an alert). */
  reset(): void {
    this.S = 0;
    this._alerted = false;
  }

  private computeBaseline(): void {
    const n = this.baselineValues.length;
    let sum = 0;
    for (let i = 0; i < n; i++) sum += this.baselineValues[i];
    this.mu = sum / n;

    let varSum = 0;
    for (let i = 0; i < n; i++) {
      const d = this.baselineValues[i] - this.mu;
      varSum += d * d;
    }
    this.sigma = Math.sqrt(varSum / n);
    // Prevent division by zero for constant signals
    if (this.sigma < 1e-10) this.sigma = 1;
  }
}
