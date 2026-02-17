/**
 * Seeded PRNG (xorshift128+) for reproducibility.
 */
import type { Rng } from "./interfaces.js";

export class SeededRng implements Rng {
  private _s0: number;
  private _s1: number;
  private _seed: number;
  private _hasSpare = false;
  private _spare = 0;

  constructor(seed = 42) {
    this._seed = seed;
    this._s0 = seed;
    this._s1 = seed ^ 0xdeadbeef;
    // Warm up
    for (let i = 0; i < 20; i++) this.next();
  }

  seed(s: number): void {
    this._seed = s;
    this._s0 = s;
    this._s1 = s ^ 0xdeadbeef;
    this._hasSpare = false;
    for (let i = 0; i < 20; i++) this.next();
  }

  state(): number {
    return this._seed;
  }

  setState(s: number): void {
    this.seed(s);
  }

  /** Returns a number in [0, 1). */
  next(): number {
    let s1 = this._s0;
    const s0 = this._s1;
    this._s0 = s0;
    s1 ^= s1 << 23;
    s1 ^= s1 >>> 17;
    s1 ^= s0;
    s1 ^= s0 >>> 26;
    this._s1 = s1;
    // Map to [0, 1)
    return ((this._s0 + this._s1) >>> 0) / 0x100000000;
  }

  /** Box-Muller transform for Gaussian samples. */
  nextGauss(): number {
    if (this._hasSpare) {
      this._hasSpare = false;
      return this._spare;
    }
    let u: number, v: number, s: number;
    do {
      u = this.next() * 2 - 1;
      v = this.next() * 2 - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const mul = Math.sqrt(-2.0 * Math.log(s) / s);
    this._spare = v * mul;
    this._hasSpare = true;
    return u * mul;
  }
}
