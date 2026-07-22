import { describe, expect, it } from "vitest";
import { SeededRng } from "@alpha/core";
import { DataLoader, ShardedDataLoader, SftDataLoader, type BatchSource, type SftExample } from "@alpha/train";

function tokenArray(size: number): Int32Array {
  return Int32Array.from({ length: size }, (_, index) => index);
}

function inputIds(loader: BatchSource): number[] {
  return Array.from(loader.nextBatch().inputs.data as Int32Array);
}

function batchIds(loader: BatchSource): { inputs: number[]; targets: number[] } {
  const batch = loader.nextBatch();
  return {
    inputs: Array.from(batch.inputs.data as Int32Array),
    targets: Array.from(batch.targets.data as Int32Array),
  };
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

  it.each([false, true])("ShardedDataLoader is batch-identical to logical concatenation (packed=%s)", (packed) => {
    const all = tokenArray(47);
    const shards = [all.slice(0, 13), all.slice(13, 20), all.slice(20, 31), all.slice(31)];
    const contiguous = new DataLoader(all, new SeededRng(991), 3, 8, packed);
    const sharded = new ShardedDataLoader(shards, new SeededRng(991), 3, 8, packed);
    expect(sharded.length).toBe(all.length);
    expect(sharded.stepsPerEpoch).toBe(contiguous.stepsPerEpoch);
    for (let batch = 0; batch < 12; batch++) expect(batchIds(sharded)).toEqual(batchIds(contiguous));
  });

  it.each([false, true])("seekBatches reproduces uninterrupted sharded batches (packed=%s)", (packed) => {
    const all = tokenArray(89);
    const shards = [all.slice(0, 11), all.slice(11, 53), all.slice(53)];
    const skipped = 17;
    const uninterrupted = new ShardedDataLoader(shards, new SeededRng(1234), 4, 7, packed);
    for (let batch = 0; batch < skipped; batch++) uninterrupted.nextBatch();
    const expected = batchIds(uninterrupted);
    const resumed = new ShardedDataLoader(shards, new SeededRng(1234), 4, 7, packed);
    resumed.seekBatches(skipped);
    expect(batchIds(resumed)).toEqual(expected);
  });

  it("rejects invalid seek positions", () => {
    const loader = new DataLoader(tokenArray(32), new SeededRng(1), 1, 4, true);
    expect(() => loader.seekBatches(-1)).toThrow(/non-negative/);
    expect(() => loader.seekBatches(1.5)).toThrow(/non-negative/);
  });

  it("rejects empty sharded inputs", () => {
    expect(() => new ShardedDataLoader([], new SeededRng(1), 1, 4, true)).toThrow(/non-empty/);
    expect(() => new ShardedDataLoader([new Int32Array(0)], new SeededRng(1), 1, 4, true)).toThrow(/non-empty/);
  });
});
