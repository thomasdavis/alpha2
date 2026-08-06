import { describe, expect, it, vi } from "vitest";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Effect } from "effect";
import { CpuRefBackend } from "@alpha/tensor";
import { SeededRng, type CheckpointState, type ModelConfig, type Tokenizer } from "@alpha/core";
import {
  DataLoader, ShardedDataLoader, SftDataLoader,
  loadOrCacheTokens, loadPretrainShardManifest, verifyPretrainShardManifest,
  train, validateCheckpointModelCompatibility, AdamW, FileCheckpoint, releaseCheckpointSnapshotBuffers,
  type BatchSource, type SftExample,
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

describe("checkpoint snapshot lifecycle", () => {
  it("releases cloned optimizer buffers without clearing live parameters", () => {
    const parameterData = Float32Array.of(1, 2);
    const optimizerBuffers = new Map([
      ["weight.m", { shape: [2], dtype: "f32" as const, data: Float32Array.of(3, 4) }],
      ["weight.v", { shape: [2], dtype: "f32" as const, data: Float32Array.of(5, 6) }],
    ]);
    const state: CheckpointState = {
      modelConfig: {
        vocabSize: 2, blockSize: 1, nLayer: 1, nEmbd: 2, nHead: 1,
        dropout: 0, ffnActivation: "gelu",
      },
      params: { weight: { shape: [2], data: parameterData as unknown as number[] } },
      optimizerState: { step: 1, buffers: optimizerBuffers },
      rngState: 1,
      configHash: "test",
      step: 1,
    };

    expect(releaseCheckpointSnapshotBuffers(state)).toBe(2);
    expect(optimizerBuffers.size).toBe(0);
    expect(state.params.weight.data).toBe(parameterData);
  });
});

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

  it("shuffled SFT visits every conversation once per epoch and resumes exactly", () => {
    const examples: SftExample[] = Array.from({ length: 8 }, (_, index) => ({
      tokens: Int32Array.of(100 + index, 200 + index),
      roleMask: Uint8Array.of(0, 1),
    }));
    const options = { shuffleSeed: 20260731 };
    const epoch = new SftDataLoader(examples, 2, 2, options);
    const seen: number[] = [];
    for (let batch = 0; batch < 4; batch++) {
      const inputs = epoch.nextBatch().inputs.data as Int32Array;
      seen.push(inputs[0], inputs[2]);
    }
    expect([...seen].sort((a, b) => a - b)).toEqual(examples.map((ex) => ex.tokens[0]));

    const skipped = 11;
    const uninterrupted = new SftDataLoader(examples, 3, 2, options);
    for (let i = 0; i < skipped; i++) uninterrupted.nextBatch();
    const expected = batchIds(uninterrupted);
    const resumed = new SftDataLoader(examples, 3, 2, options);
    resumed.seekBatches(skipped);
    expect(batchIds(resumed)).toEqual(expected);
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
    const previousGc = (globalThis as any).gc;
    const checkpointGc = vi.fn();
    (globalThis as any).gc = checkpointGc;
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

      const sourceCheckpoint = join(runDir, "checkpoint-1.json");
      const initializedRunDir = join(dir, "initialized-run");
      const initializedBackend = new CpuRefBackend();
      const events: string[] = [];
      let checkpointPayload: Record<string, unknown> | undefined;
      await train({
        backend: initializedBackend,
        tokenizer,
        optimizer: new AdamW(initializedBackend, { lr: 0, beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0 }),
        rng: new SeededRng(99),
        modelConfig,
        trainConfig: {
          iters: 1, batchSize: 2, lr: 0, lrMin: 0, warmupIters: 0,
          beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0, gradClip: 1,
          evalInterval: 10, checkpointInterval: 10, evalIters: 1, seed: 99, backend: "cpu_ref",
          tokenizer: "test-byte", optimizer: "adamw", logLevel: "info", logEvery: 1,
          trace: false, gradAccumSteps: 1, sampleInterval: 0, spikeThreshold: 0,
          embGradScale: 1, syncEvery: 0, gcEvery: 0, packed: true, symbio: false, symbioConfig: null,
        },
        dataPath: first,
        dataPaths: [first, second],
        initCheckpointPath: sourceCheckpoint,
        runDir: initializedRunDir,
        onEvent: (event) => {
          events.push(event.kind);
          if (event.kind === "checkpoint_saved" && event.payload) checkpointPayload = event.payload;
        },
      });
      expect(events).toContain("run_initialized_from_checkpoint");
      // Each one-step run reaches both terminal evaluation and checkpointing;
      // both paths explicitly reclaim host buffers before continuing.
      expect(checkpointGc).toHaveBeenCalledTimes(4);
      expect(checkpointPayload).toMatchObject({
        hostGcRan: true,
        hostRssBeforeGcMB: expect.any(Number),
        hostRssAfterGcMB: expect.any(Number),
        hostExternalBeforeGcMB: expect.any(Number),
        hostExternalAfterGcMB: expect.any(Number),
        hostArrayBuffersBeforeGcMB: expect.any(Number),
        hostArrayBuffersAfterGcMB: expect.any(Number),
        optimizerBuffersReleased: expect.any(Number),
      });
      expect(Number(checkpointPayload?.optimizerBuffersReleased)).toBeGreaterThan(0);
      const checkpoint = new FileCheckpoint();
      const sourceState = await Effect.runPromise(checkpoint.load(sourceCheckpoint));
      const initializedState = await Effect.runPromise(checkpoint.load(join(initializedRunDir, "checkpoint-1.json")));
      expect(Object.keys(initializedState.params)).toEqual(Object.keys(sourceState.params));
      for (const name of Object.keys(sourceState.params)) {
        expect(Array.from(initializedState.params[name].data)).toEqual(Array.from(sourceState.params[name].data));
      }
      const initializedConfig = JSON.parse(await readFile(join(initializedRunDir, "config.json"), "utf8")) as {
        initCheckpointPath: string;
      };
      expect(initializedConfig.initCheckpointPath).toBe(sourceCheckpoint);

      const initializedCheckpoint = join(initializedRunDir, "checkpoint-1.json");
      const resumedBackend = new CpuRefBackend();
      await train({
        backend: resumedBackend,
        tokenizer,
        optimizer: new AdamW(resumedBackend, { lr: 0, beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0 }),
        rng: new SeededRng(123),
        modelConfig,
        trainConfig: {
          iters: 2, batchSize: 2, lr: 0, lrMin: 0, warmupIters: 0,
          beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0, gradClip: 1,
          evalInterval: 10, checkpointInterval: 10, evalIters: 1, seed: 99, backend: "cpu_ref",
          tokenizer: "test-byte", optimizer: "adamw", logLevel: "info", logEvery: 1,
          trace: false, gradAccumSteps: 1, sampleInterval: 0, spikeThreshold: 0,
          embGradScale: 1, syncEvery: 0, gcEvery: 0, packed: true, symbio: false, symbioConfig: null,
        },
        dataPath: first,
        dataPaths: [first, second],
        resumePath: initializedCheckpoint,
        runDir: initializedRunDir,
      });
      const resumedConfig = JSON.parse(await readFile(join(initializedRunDir, "config.json"), "utf8")) as {
        initCheckpointPath: string;
        resumePath: string;
      };
      expect(resumedConfig.initCheckpointPath).toBe(sourceCheckpoint);
      expect(resumedConfig.resumePath).toBe(initializedCheckpoint);

      const configWithoutOrigin = { ...resumedConfig } as Partial<typeof resumedConfig>;
      delete configWithoutOrigin.initCheckpointPath;
      await writeFile(join(initializedRunDir, "config.json"), JSON.stringify(configWithoutOrigin), "utf8");
      const missingOriginBackend = new CpuRefBackend();
      await expect(train({
        backend: missingOriginBackend,
        tokenizer,
        optimizer: new AdamW(missingOriginBackend, { lr: 0, beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0 }),
        rng: new SeededRng(456),
        modelConfig,
        trainConfig: {
          iters: 3, batchSize: 2, lr: 0, lrMin: 0, warmupIters: 0,
          beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0, gradClip: 1,
          evalInterval: 10, checkpointInterval: 10, evalIters: 1, seed: 99, backend: "cpu_ref",
          tokenizer: "test-byte", optimizer: "adamw", logLevel: "info", logEvery: 1,
          trace: false, gradAccumSteps: 1, sampleInterval: 0, spikeThreshold: 0,
          embGradScale: 1, syncEvery: 0, gcEvery: 0, packed: false, symbio: false, symbioConfig: null,
        },
        dataPath: first,
        resumePath: join(initializedRunDir, "checkpoint-2.json"),
        runDir: initializedRunDir,
        sft: true,
      })).rejects.toThrow("SFT resume requires the existing run config to preserve initCheckpointPath provenance");
    } finally {
      if (previousGc === undefined) delete (globalThis as any).gc;
      else (globalThis as any).gc = previousGc;
      await rm(dir, { recursive: true, force: true });
    }
  });

  it("keeps an explicit validation file wholly outside all pretraining shards", async () => {
    const dir = await mkdtemp(join(tmpdir(), "alpha-sharded-heldout-"));
    try {
      const first = join(dir, "first.txt");
      const second = join(dir, "second.txt");
      const validation = join(dir, "validation.txt");
      await writeFile(first, "alpha one\nalpha two\nalpha three\n");
      await writeFile(second, "beta one\nbeta two\nbeta three\n");
      await writeFile(validation, "heldout red\nheldout blue\nheldout green\n");
      const tokenizer: Tokenizer = {
        name: "test-byte",
        vocabSize: 256,
        encode: (text) => Int32Array.from(Buffer.from(text, "utf8")),
        decode: (tokens) => Buffer.from(Array.from(tokens)).toString("utf8"),
        build: () => Effect.succeed({ type: "test-byte", vocabSize: 256, vocab: [] }),
      };
      const backend = new CpuRefBackend();
      const runDir = join(dir, "run");
      await train({
        backend,
        tokenizer,
        optimizer: new AdamW(backend, { lr: 1e-3, beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0 }),
        rng: new SeededRng(17),
        modelConfig: {
          vocabSize: 256, blockSize: 8, nLayer: 1, nEmbd: 8, nHead: 2, dropout: 0, ffnActivation: "gelu",
        },
        trainConfig: {
          iters: 1, batchSize: 2, lr: 1e-3, lrMin: 1e-4, warmupIters: 0,
          beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0, gradClip: 1,
          evalInterval: 1, checkpointInterval: 1, evalIters: 1, seed: 17, backend: "cpu_ref",
          tokenizer: "test-byte", optimizer: "adamw", logLevel: "info", logEvery: 1,
          trace: false, gradAccumSteps: 1, sampleInterval: 0, spikeThreshold: 0,
          embGradScale: 1, syncEvery: 0, gcEvery: 0, packed: true, symbio: false, symbioConfig: null,
        },
        dataPath: first,
        dataPaths: [first, second],
        valDataPath: validation,
        runDir,
      });
      const config = JSON.parse(await readFile(join(runDir, "config.json"), "utf8")) as {
        dataStats: {
          trainTokens: number;
          valTokens: number;
          shards: { path: string; trainTokens: number; valTokens: number }[];
          validation: { path: string; valTokens: number; whollyHeldOut: boolean };
        };
      };
      expect(config.dataStats.shards.map((shard) => shard.path)).toEqual([first, second]);
      expect(config.dataStats.shards.every((shard) => shard.trainTokens > 0 && shard.valTokens === 0)).toBe(true);
      expect(config.dataStats.validation).toEqual({
        path: validation,
        valTokens: config.dataStats.valTokens,
        whollyHeldOut: true,
      });
      expect(config.dataStats.valTokens).toBeGreaterThan(8);
      const terminalMetric = JSON.parse((await readFile(join(runDir, "metrics.jsonl"), "utf8")).trim());
      expect(terminalMetric.step).toBe(1);
      expect(terminalMetric.valLoss).toEqual(expect.any(Number));
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it("isolates token caches by exact tokenizer-artifact identity", async () => {
    const dir = await mkdtemp(join(tmpdir(), "alpha-token-cache-"));
    try {
      const dataPath = join(dir, "data.txt");
      await writeFile(dataPath, "artifact-sensitive tokens\n");
      const tokenizer = (offset: number, fail = false): Tokenizer => ({
        name: "same-name",
        vocabSize: 512,
        encode: (text) => {
          if (fail) throw new Error("encode should not run on a cache hit");
          return Int32Array.from(Buffer.from(text, "utf8"), (value) => value + offset);
        },
        decode: () => "",
        build: () => Effect.succeed({ type: "same-name", vocabSize: 512, vocab: [] }),
      });
      const firstIdentity = "a".repeat(64);
      const secondIdentity = "b".repeat(64);
      const first = await loadOrCacheTokens(dataPath, tokenizer(0), undefined, firstIdentity);
      const second = await loadOrCacheTokens(dataPath, tokenizer(1), undefined, secondIdentity);
      expect(Array.from(second)).not.toEqual(Array.from(first));
      const cachedFirst = await loadOrCacheTokens(dataPath, tokenizer(0, true), undefined, firstIdentity);
      expect(Array.from(cachedFirst)).toEqual(Array.from(first));
      const cacheFiles = (await readdir(dir)).filter((name) => name.endsWith(".tokens"));
      expect(cacheFiles).toHaveLength(2);
      expect(cacheFiles.some((name) => name.includes("-" + "a".repeat(24)))).toBe(true);
      expect(cacheFiles.some((name) => name.includes("-" + "b".repeat(24)))).toBe(true);
      const firstCachePath = join(dir, cacheFiles.find((name) => name.includes("-" + "a".repeat(24)))!);
      await writeFile(firstCachePath, Buffer.alloc(19));
      const recovered = await loadOrCacheTokens(dataPath, tokenizer(0), undefined, firstIdentity);
      expect(Array.from(recovered)).toEqual(Array.from(first));
      expect((await readdir(dir)).some((name) => name.includes(".tokens.tmp-"))).toBe(false);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it.each([
    ["normType", "rmsnorm"],
    ["posEnc", "rope"],
    ["ropeTheta", 500000],
    ["tieEmbeddings", true],
    ["softCap", 0],
  ] as const)("rejects checkpoint architecture mismatch in %s", (key, value) => {
    const checkpoint: ModelConfig = {
      vocabSize: 256, blockSize: 8, nLayer: 2, nEmbd: 16, nHead: 2, dropout: 0,
      ffnActivation: "gelu", ffnDim: 64, normType: "layernorm", posEnc: "learned",
      ropeTheta: 10000, tieEmbeddings: false, softCap: 30,
    };
    const active = { ...checkpoint, [key]: value } as ModelConfig;
    expect(() => validateCheckpointModelCompatibility("fixture.ckpt", checkpoint, active)).toThrow(new RegExp(key));
  });
});
