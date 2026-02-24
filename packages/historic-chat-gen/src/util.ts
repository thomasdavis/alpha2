/**
 * Utilities: ID generation, timestamps, seeded RNG.
 */
import { randomBytes } from "node:crypto";

/** Generate a random hex ID. */
export function genId(bytes = 8): string {
  return randomBytes(bytes).toString("hex");
}

/** ISO timestamp. */
export function now(): string {
  return new Date().toISOString();
}

/**
 * Seeded PRNG (xoshiro128**).
 * Deterministic shuffle for reproducible assignment generation.
 */
export class SeededRng {
  private s: Uint32Array;

  constructor(seed: number) {
    // SplitMix32 to initialize state from single seed
    this.s = new Uint32Array(4);
    let z = seed | 0;
    for (let i = 0; i < 4; i++) {
      z = (z + 0x9e3779b9) | 0;
      let t = z ^ (z >>> 16);
      t = Math.imul(t, 0x21f0aaad);
      t = t ^ (t >>> 15);
      t = Math.imul(t, 0x735a2d97);
      t = t ^ (t >>> 15);
      this.s[i] = t >>> 0;
    }
  }

  /** Returns a float in [0, 1). */
  next(): number {
    const s = this.s;
    const result = Math.imul(s[1]! * 5, 7) >>> 0;
    const t = s[1]! << 9;
    s[2]! ^= s[0]!;
    s[3]! ^= s[1]!;
    s[1]! ^= s[2]!;
    s[0]! ^= s[3]!;
    s[2]! ^= t;
    s[3]! = (s[3]! << 11) | (s[3]! >>> 21);
    return (result >>> 0) / 0x100000000;
  }

  /** Returns an integer in [min, max). */
  int(min: number, max: number): number {
    return min + Math.floor(this.next() * (max - min));
  }

  /** Fisher-Yates shuffle in place. */
  shuffle<T>(arr: T[]): T[] {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = this.int(0, i + 1);
      const tmp = arr[i]!;
      arr[i] = arr[j]!;
      arr[j] = tmp;
    }
    return arr;
  }
}
