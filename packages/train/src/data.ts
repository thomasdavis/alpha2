/**
 * Minimal data pipeline.
 *
 * Loads text → tokenizes → creates random contiguous windows for training.
 */
import type { Tokenizer, Rng, TensorData } from "@alpha/core";

export interface DataBatch {
  /** Input token ids [B, T] */
  inputs: TensorData;
  /** Target token ids [B, T] */
  targets: TensorData;
}

export class DataLoader {
  private tokens: Int32Array;
  private rng: Rng;
  private batchSize: number;
  private blockSize: number;

  constructor(tokens: Int32Array, rng: Rng, batchSize: number, blockSize: number) {
    this.tokens = tokens;
    this.rng = rng;
    this.batchSize = batchSize;
    this.blockSize = blockSize;
  }

  /** Create a DataLoader from raw text. */
  static fromText(text: string, tokenizer: Tokenizer, rng: Rng, batchSize: number, blockSize: number): DataLoader {
    const tokens = tokenizer.encode(text);
    return new DataLoader(tokens, rng, batchSize, blockSize);
  }

  /** Get a random batch of contiguous windows. */
  nextBatch(): DataBatch {
    const B = this.batchSize;
    const T = this.blockSize;
    const maxStart = this.tokens.length - T - 1;

    const inputs = new Int32Array(B * T);
    const targets = new Int32Array(B * T);

    for (let b = 0; b < B; b++) {
      const start = Math.floor(this.rng.next() * maxStart);
      for (let t = 0; t < T; t++) {
        inputs[b * T + t] = this.tokens[start + t];
        targets[b * T + t] = this.tokens[start + t + 1];
      }
    }

    return {
      inputs: { shape: [B, T], dtype: "i32", data: inputs },
      targets: { shape: [B, T], dtype: "i32", data: targets },
    };
  }

  get length(): number {
    return this.tokens.length;
  }
}

/** Load text from a file path. */
export async function loadText(path: string): Promise<string> {
  const fs = await import("node:fs/promises");
  return fs.readFile(path, "utf-8");
}

/** Split text into train/val by ratio. */
export function splitText(text: string, trainRatio = 0.9): { train: string; val: string } {
  const splitIdx = Math.floor(text.length * trainRatio);
  return {
    train: text.slice(0, splitIdx),
    val: text.slice(splitIdx),
  };
}
