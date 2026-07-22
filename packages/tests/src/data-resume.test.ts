import { describe, expect, it } from "vitest";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Effect } from "effect";
import { CpuRefBackend } from "@alpha/tensor";
import { SeededRng, type ModelConfig, type Tokenizer } from "@alpha/core";
import {
  DataLoader, ShardedDataLoader, SftDataLoader,
  loadPretrainShardManifest, verifyPretrainShardManifest,
  train, AdamW, type BatchSource, type SftExample,
} from "@alpha/train";

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

  it("loads a structured shard manifest and resolves relative paths", async () => {
    const dir = await mkdtemp(join(tmpdir(), "alpha-shards-"));
    try {
      await writeFile(join(dir, "a.txt"), "alpha");
      await writeFile(join(dir, "b.txt"), "beta");
      const manifestPath = join(dir, "manifest.json");
      await writeFile(manifestPath, JSON.stringify({
        schema: "alpha-pretrain-shards-v1",
        shards: [
          { path: "a.txt", sha256: createHash("sha256").update("alpha").digest("hex") },
          { path: "b.txt", sha256: createHash("sha256").update("beta").digest("hex") },
        ],
      }));
      const loaded = await loadPretrainShardManifest(manifestPath);
      expect(loaded.paths).toEqual([join(dir, "a.txt"), join(dir, "b.txt")]);
      expect(loaded.manifest.shards).toHaveLength(2);
      expect(await verifyPretrainShardManifest(loaded.manifest, loaded.paths)).toEqual([
        { path: join(dir, "a.txt"), bytes: 5, sha256: createHash("sha256").update("alpha").digest("hex") },
        { path: join(dir, "b.txt"), bytes: 4, sha256: createHash("sha256").update("beta").digest("hex") },
      ]);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it("rejects malformed or duplicate shard manifests", async () => {
    const dir = await mkdtemp(join(tmpdir(), "alpha-shards-invalid-"));
    try {
      await writeFile(join(dir, "a.txt"), "alpha");
      const manifestPath = join(dir, "manifest.json");
      await writeFile(manifestPath, JSON.stringify({
        schema: "alpha-pretrain-shards-v1",
        shards: [
          { path: "a.txt", sha256: "a".repeat(64) },
          { path: "a.txt", sha256: "a".repeat(64) },
        ],
      }));
      await expect(loadPretrainShardManifest(manifestPath)).rejects.toThrow(/duplicate shard/);
      await writeFile(manifestPath, JSON.stringify({
        schema: "alpha-pretrain-shards-v1",
        shards: [
          { path: "a.txt", sha256: "not-a-hash" },
          { path: "missing.txt", sha256: "b".repeat(64) },
        ],
      }));
      await expect(loadPretrainShardManifest(manifestPath)).rejects.toThrow(/invalid shard entry/);
      await writeFile(manifestPath, JSON.stringify({
        schema: "alpha-pretrain-shards-v1",
        shards: [
          { path: "a.txt", sha256: "0".repeat(64) },
          { path: "a.txt.copy", sha256: "1".repeat(64) },
        ],
      }));
      await writeFile(join(dir, "a.txt.copy"), "alpha");
      const wrongHash = await loadPretrainShardManifest(manifestPath);
      await expect(verifyPretrainShardManifest(wrongHash.manifest, wrongHash.paths)).rejects.toThrow(/SHA-256/);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it("trainer records and consumes multiple pretraining shards", async () => {
    const dir = await mkdtemp(join(tmpdir(), "alpha-sharded-trainer-"));
    try {
      const first = join(dir, "first.txt");
      const second = join(dir, "second.txt");
      await writeFile(first, Array.from({ length: 20 }, (_, index) => `alpha-${index}`).join("\n") + "\n");
      await writeFile(second, Array.from({ length: 20 }, (_, index) => `beta-${index}`).join("\n") + "\n");
      const tokenizer: Tokenizer = {
        name: "test-byte",
        vocabSize: 256,
        encode: (text) => Int32Array.from(Buffer.from(text, "utf8")),
        decode: (tokens) => Buffer.from(Array.from(tokens)).toString("utf8"),
        build: () => Effect.succeed({ type: "test-byte", vocabSize: 256, vocab: [] }),
      };
      const backend = new CpuRefBackend();
      const modelConfig: ModelConfig = {
        vocabSize: 256, blockSize: 8, nLayer: 1, nEmbd: 8, nHead: 2, dropout: 0, ffnActivation: "gelu",
      };
      const runDir = join(dir, "run");
      await train({
        backend,
        tokenizer,
        optimizer: new AdamW(backend, { lr: 1e-3, beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0 }),
        rng: new SeededRng(7),
        modelConfig,
        trainConfig: {
          iters: 1, batchSize: 2, lr: 1e-3, lrMin: 1e-4, warmupIters: 0,
          beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0, gradClip: 1,
          evalInterval: 10, checkpointInterval: 10, evalIters: 1, seed: 7, backend: "cpu_ref",
          tokenizer: "test-byte", optimizer: "adamw", logLevel: "info", logEvery: 1,
          trace: false, gradAccumSteps: 1, sampleInterval: 0, spikeThreshold: 0,
          embGradScale: 1, syncEvery: 0, gcEvery: 0, packed: true, symbio: false, symbioConfig: null,
        },
        dataPath: first,
        dataPaths: [first, second],
        runDir,
      });
      const config = JSON.parse(await readFile(join(runDir, "config.json"), "utf8")) as {
        dataPaths: string[];
        dataStats: { trainTokens: number; valTokens: number; shards: unknown[] };
      };
      expect(config.dataPaths).toEqual([first, second]);
      expect(config.dataStats.shards).toHaveLength(2);
      expect(config.dataStats.trainTokens).toBeGreaterThan(0);
      expect(config.dataStats.valTokens).toBeGreaterThan(0);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
