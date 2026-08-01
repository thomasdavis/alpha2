import { afterEach, describe, expect, it } from "vitest";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { Effect } from "effect";
import { CpuRefBackend } from "@alpha/tensor";
import { SeededRng, type ModelConfig, type Tokenizer, type TokenizerArtifacts, type TrainConfig } from "@alpha/core";
import { collectParamEntries } from "@alpha/model";
import {
  AdamW, train,
  CHAT_USER_TOKEN, CHAT_ASSISTANT_TOKEN, CHAT_EOT_TOKEN,
  type StepMetrics,
} from "@alpha/train";

const specials: [string, number][] = [
  [CHAT_USER_TOKEN, 0],
  [CHAT_ASSISTANT_TOKEN, 1],
  [CHAT_EOT_TOKEN, 2],
];

class TinyChatTokenizer implements Tokenizer {
  readonly name = "rcr-ul-test-chat";
  readonly vocabSize = 259;
  encode(text: string): Int32Array {
    const out: number[] = [];
    let index = 0;
    while (index < text.length) {
      const special = specials.find(([marker]) => text.startsWith(marker, index));
      if (special) {
        out.push(special[1]);
        index += special[0].length;
        continue;
      }
      let end = index;
      while (end < text.length && !specials.some(([marker]) => text.startsWith(marker, end))) end++;
      for (const byte of Buffer.from(text.slice(index, end), "utf8")) out.push(byte + 3);
      index = end;
    }
    return Int32Array.from(out);
  }
  decode(tokens: ArrayLike<number>): string {
    return Array.from(tokens).join(",");
  }
  build(_input: string): Effect.Effect<TokenizerArtifacts, never> {
    return Effect.succeed({ type: this.name, vocabSize: this.vocabSize, vocab: [] });
  }
}

const tempDirs: string[] = [];
const savedEnv = new Map<string, string | undefined>();
const controlledEnv = [
  "ALPHA_SFT_VAL_FRACTION",
  "ALPHA_SFT_SHUFFLE",
  "ALPHA_SFT_BALANCE_CONVERSATIONS",
  "ALPHA_SFT_START_TOKENS",
  "ALPHA_SFT_START_WEIGHT",
  "ALPHA_SFT_END_WEIGHT",
];

afterEach(async () => {
  for (const name of controlledEnv) {
    const value = savedEnv.get(name);
    if (value === undefined) delete process.env[name];
    else process.env[name] = value;
  }
  savedEnv.clear();
  await Promise.all(tempDirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })));
});

describe("RCR-UL matched-control trainer", () => {
  it("a zero-weight negative branch leaves trained parameters byte-exactly unchanged", async () => {
    for (const name of controlledEnv) savedEnv.set(name, process.env[name]);
    process.env.ALPHA_SFT_VAL_FRACTION = "0";
    process.env.ALPHA_SFT_SHUFFLE = "0";
    process.env.ALPHA_SFT_BALANCE_CONVERSATIONS = "0";
    process.env.ALPHA_SFT_START_TOKENS = "0";
    process.env.ALPHA_SFT_START_WEIGHT = "1";
    process.env.ALPHA_SFT_END_WEIGHT = "1";

    const dir = await mkdtemp(join(tmpdir(), "alpha-rcr-ul-trainer-"));
    tempDirs.push(dir);
    const tokenizer = new TinyChatTokenizer();
    const positiveRows = [
      `${CHAT_USER_TOKEN}hi${CHAT_ASSISTANT_TOKEN}hello${CHAT_EOT_TOKEN}`,
      `${CHAT_USER_TOKEN}two plus two${CHAT_ASSISTANT_TOKEN}four${CHAT_EOT_TOKEN}`,
    ];
    const positivePath = join(dir, "positive.txt");
    await writeFile(positivePath, `${positiveRows.join("\n")}\n`);

    const negativeTexts = [
      `${CHAT_USER_TOKEN}hi${CHAT_ASSISTANT_TOKEN}hello hello hello${CHAT_EOT_TOKEN}`,
      `${CHAT_USER_TOKEN}two plus two${CHAT_ASSISTANT_TOKEN}four four four${CHAT_EOT_TOKEN}`,
    ];
    const negativeRecords = negativeTexts.map((text, index) => {
      const tokenIds = Array.from(tokenizer.encode(text));
      return {
        schema: "alpha-rcr-ul-example-v1",
        stable_id: `negative-${index}`,
        positive_conversation_sha256: createHash("sha256").update(positiveRows[index], "utf8").digest("hex"),
        token_ids: tokenIds,
        penalty_target_positions: [tokenIds.length - 2],
      };
    });
    const negativePath = join(dir, "negative.jsonl");
    await writeFile(negativePath, `${negativeRecords.map((record) => JSON.stringify(record)).join("\n")}\n`);

    const modelConfig: ModelConfig = {
      vocabSize: tokenizer.vocabSize,
      blockSize: 32,
      nLayer: 1,
      nEmbd: 8,
      nHead: 2,
      dropout: 0,
      ffnActivation: "gelu",
    };
    const trainConfig: TrainConfig = {
      iters: 2, batchSize: 1, lr: 1e-3, lrMin: 1e-4, warmupIters: 0,
      beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0, gradClip: 0,
      evalInterval: 1000, evalIters: 1, seed: 2026, backend: "cpu_ref",
      tokenizer: tokenizer.name, optimizer: "adamw", logLevel: "info", logEvery: 1000,
      trace: false, gradAccumSteps: 1, sampleInterval: 0, spikeThreshold: 0,
      embGradScale: 1, syncEvery: 0, gcEvery: 0, packed: false, symbio: false, symbioConfig: null,
    };

    const run = async (name: string, rcrWeight?: number) => {
      const backend = new CpuRefBackend();
      const optimizer = new AdamW(backend, { lr: 1e-3, beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0 });
      const metrics: StepMetrics[] = [];
      const result = await train({
        backend,
        tokenizer,
        optimizer,
        rng: new SeededRng(2026),
        modelConfig,
        trainConfig,
        dataPath: positivePath,
        runDir: join(dir, name),
        sft: true,
        ...(rcrWeight !== undefined ? { rcrUl: { dataPath: negativePath, weight: rcrWeight, epsilon: 1e-6 } } : {}),
        onStep: (step) => metrics.push(step),
      });
      const params = collectParamEntries(result.params).map(([paramName, variable]) => ({
        name: paramName,
        data: Float32Array.from(variable.data.data as Float32Array),
      }));
      return { params, metrics };
    };

    const positiveOnly = await run("positive-only");
    const matchedControl = await run("matched-control", 0);
    const weightedCandidate = await run("weighted-candidate", 0.5);
    expect(matchedControl.metrics).toHaveLength(2);
    for (const metric of matchedControl.metrics) {
      expect(metric.ul_weight).toBe(0);
      expect(Number.isFinite(metric.positive_ce_loss)).toBe(true);
      expect(Number.isFinite(metric.negative_ul_loss)).toBe(true);
      expect(metric.negative_penalty_position_mass).toBe(1);
      expect(metric.negative_examples_with_penalty).toBe(1);
      expect(metric.negative_first_penalty_target_position).toBeGreaterThan(0);
      expect(metric.negative_last_penalty_target_position).toBe(metric.negative_first_penalty_target_position);
      expect(metric.negative_mean_bad_token_probability).toBeGreaterThanOrEqual(0);
      expect(metric.negative_mean_bad_token_probability).toBeLessThanOrEqual(1);
      expect(metric.negative_max_bad_token_probability).toBeGreaterThanOrEqual(metric.negative_mean_bad_token_probability!);
      expect(metric.grad_norm_before_clip).toBe(metric.gradNorm);
      expect(metric.grad_norm_after_clip).toBe(metric.gradNorm);
      expect(metric.nan_count).toBe(0);
      expect(metric.inf_count).toBe(0);
      expect(metric.timing_positive_fwd_ms).toBeGreaterThanOrEqual(0);
      expect(metric.timing_positive_bwd_ms).toBeGreaterThanOrEqual(0);
      expect(metric.timing_negative_fwd_ms).toBeGreaterThanOrEqual(0);
      expect(metric.timing_negative_bwd_ms).toBeGreaterThanOrEqual(0);
    }
    expect(matchedControl.params.map((param) => param.name)).toEqual(positiveOnly.params.map((param) => param.name));
    for (let index = 0; index < positiveOnly.params.length; index++) {
      expect(Array.from(matchedControl.params[index].data), positiveOnly.params[index].name)
        .toEqual(Array.from(positiveOnly.params[index].data));
    }
    for (const metric of weightedCandidate.metrics) {
      expect(metric.ul_weight).toBe(0.5);
      expect(metric.loss).toBeCloseTo(metric.positive_ce_loss! + 0.5 * metric.negative_ul_loss!, 7);
    }
    const changed = weightedCandidate.params.some((param, paramIndex) =>
      param.data.some((value, valueIndex) => value !== positiveOnly.params[paramIndex].data[valueIndex]),
    );
    expect(changed).toBe(true);

    // The paired positive/negative cursors and optimizer state must resume to
    // the exact same terminal parameters as an uninterrupted run.
    const baseDir = join(dir, "resume-base");
    const baseBackend = new CpuRefBackend();
    await train({
      backend: baseBackend,
      tokenizer,
      optimizer: new AdamW(baseBackend, { lr: 0, beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0 }),
      rng: new SeededRng(2026),
      modelConfig,
      trainConfig: { ...trainConfig, iters: 1, lr: 0, lrMin: 0, checkpointInterval: 1 },
      dataPath: positivePath,
      runDir: baseDir,
      sft: true,
    });
    const initCheckpointPath = join(baseDir, "checkpoint-1.json");
    const runSegment = async (
      runDir: string,
      iters: number,
      resumePath?: string,
      rcrUl = { dataPath: negativePath, weight: 0.5, epsilon: 1e-6 },
    ) => {
      const backend = new CpuRefBackend();
      return train({
        backend,
        tokenizer,
        optimizer: new AdamW(backend, { lr: 1e-3, beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0 }),
        rng: new SeededRng(2026),
        modelConfig,
        trainConfig: {
          ...trainConfig,
          iters,
          lr: 1e-3,
          lrMin: 1e-3,
          checkpointInterval: 1,
        },
        dataPath: positivePath,
        runDir,
        ...(resumePath ? { resumePath } : { initCheckpointPath }),
        sft: true,
        rcrUl,
      });
    };
    const uninterrupted = await runSegment(join(dir, "uninterrupted"), 2);
    const segmentedDir = join(dir, "segmented");
    await runSegment(segmentedDir, 1);
    const resumed = await runSegment(segmentedDir, 2, join(segmentedDir, "checkpoint-1.json"));
    const uninterruptedParams = new Map(collectParamEntries(uninterrupted.params));
    for (const [name, variable] of collectParamEntries(resumed.params)) {
      expect(Array.from(variable.data.data as Float32Array), name)
        .toEqual(Array.from(uninterruptedParams.get(name)!.data.data as Float32Array));
    }
    await expect(runSegment(
      segmentedDir,
      2,
      join(segmentedDir, "checkpoint-1.json"),
      { dataPath: negativePath, weight: 0.25, epsilon: 1e-6 },
    )).rejects.toThrow("RCR-UL resume contract mismatch");
  }, 30_000);
});
