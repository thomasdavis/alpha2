import { describe, expect, it } from "vitest";
import { SeededRng } from "@alpha/core";
import { DataLoader, SftDataLoader, type SftExample } from "@alpha/train";

function tokenArray(size: number): Int32Array {
  return Int32Array.from({ length: size }, (_, index) => index);
}

function inputIds(loader: DataLoader | SftDataLoader): number[] {
  return Array.from(loader.nextBatch().inputs.data as Int32Array);
}

describe("data-loader resume positioning", () => {
  it.each([false, true])("seekBatches reproduces uninterrupted DataLoader batches (packed=%s)", (packed) => {
    const skipped = 7;
    const uninterrupted = new DataLoader(tokenArray(257), new SeededRng(1234), 3, 8, packed);
    for (let i = 0; i < skipped; i++) uninterrupted.nextBatch();
    const expected = inputIds(uninterrupted);

    const resumed = new DataLoader(tokenArray(257), new SeededRng(1234), 3, 8, packed);
    resumed.seekBatches(skipped);
    expect(inputIds(resumed)).toEqual(expected);
  });

  it("seekBatches reproduces uninterrupted SFT conversation order", () => {
    const examples: SftExample[] = Array.from({ length: 7 }, (_, index) => ({
      tokens: Int32Array.of(100 + index, 200 + index, 300 + index),
      roleMask: Uint8Array.of(0, 1, 1),
    }));
    const skipped = 5;
    const uninterrupted = new SftDataLoader(examples, 2, 4);
    for (let i = 0; i < skipped; i++) uninterrupted.nextBatch();
    const expected = inputIds(uninterrupted);

    const resumed = new SftDataLoader(examples, 2, 4);
    resumed.seekBatches(skipped);
    expect(inputIds(resumed)).toEqual(expected);
  });

  it("rejects invalid seek positions", () => {
    const loader = new DataLoader(tokenArray(32), new SeededRng(1), 1, 4, true);
    expect(() => loader.seekBatches(-1)).toThrow(/non-negative/);
    expect(() => loader.seekBatches(1.5)).toThrow(/non-negative/);
  });
});
