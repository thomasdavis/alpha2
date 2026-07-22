/** Deterministic greedy D3 evaluation over Alpha's frozen chat + QA suites. */

import { createReadStream } from "node:fs";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { join } from "node:path";
import { Effect } from "effect";
import { parseKV, requireArg, intArg } from "../parse.js";
import {
  FileCheckpoint,
  answerIsContained,
  answerTokenF1,
  formatFrozenChatPrompt,
  fourGramRepeatRate,
  normalizedAnswer,
  type FrozenChatMessage,
} from "@alpha/train";
import { tokenizerFromArtifacts } from "@alpha/tokenizers";
import { SeededRng } from "@alpha/core";
import {
  prepareInferenceModel,
  resetCache,
  prefill,
  decodeStep,
  sampleFromLogits,
} from "@alpha/inference";

interface ChatCase {
  readonly id: string;
  readonly source: string;
  readonly messages: readonly FrozenChatMessage[];
  readonly reference: string;
}

interface QaCase {
  readonly id: string;
  readonly question: string;
  readonly answer: string;
  readonly title: string;
  readonly field: string;
  readonly url: string;
}

async function readJsonl<T>(path: string): Promise<T[]> {
  return (await readFile(path, "utf-8"))
    .split("\n")
    .filter((line) => line.trim().length > 0)
    .map((line, index) => {
      try {
        return JSON.parse(line) as T;
      } catch (error) {
        throw new Error(`invalid JSONL at ${path}:${index + 1}`, { cause: error });
      }
    });
}

async function sha256File(path: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk as Buffer);
  return hash.digest("hex");
}

async function atomicWrite(path: string, content: string): Promise<void> {
  const tmp = `${path}.tmp`;
  await writeFile(tmp, content, "utf-8");
  await rename(tmp, path);
}

function mean(values: readonly number[]): number {
  return values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
}

export async function evalFrozenCmd(args: string[]): Promise<void> {
  const kv = parseKV(args);
  const checkpointPath = requireArg(kv, "checkpoint", "path to checkpoint");
  const chatPath = requireArg(kv, "chat", "frozen chat-prompts.jsonl");
  const qaPath = requireArg(kv, "qa", "frozen closed-book-qa.jsonl");
  const outDir = requireArg(kv, "out", "output directory");
  const chatMaxTokens = intArg(kv, "maxTokens", 128);
  const qaMaxTokens = intArg(kv, "qaMaxTokens", 64);
  const chatLimit = intArg(kv, "chatLimit", 0);
  const qaLimit = intArg(kv, "qaLimit", 0);
  if (chatMaxTokens < 1 || qaMaxTokens < 1) throw new Error("generation token limits must be positive");

  const [allChatCases, allQaCases] = await Promise.all([
    readJsonl<ChatCase>(chatPath),
    readJsonl<QaCase>(qaPath),
  ]);
  const chatCases = chatLimit > 0 ? allChatCases.slice(0, chatLimit) : allChatCases;
  const qaCases = qaLimit > 0 ? allQaCases.slice(0, qaLimit) : allQaCases;
  const checkpoint = new FileCheckpoint();
  const state = await Effect.runPromise(checkpoint.load(checkpointPath));
  if (!state.tokenizerArtifacts) throw new Error("checkpoint has no tokenizer artifacts");

  const tokenizer = tokenizerFromArtifacts(state.tokenizerArtifacts);
  const model = prepareInferenceModel(state.modelConfig, state.params);
  const eosIds = Array.from(tokenizer.encode("<|end_of_text|>"));
  const userIds = Array.from(tokenizer.encode("<|user|>"));
  if (eosIds.length !== 1 || userIds.length !== 1) {
    throw new Error("chat control tokens must each encode to one atomic token");
  }
  const eosId = eosIds[0];
  const userId = userIds[0];
  const rng = new SeededRng(0);

  function generate(prompt: string, maxTokens: number) {
    resetCache(model);
    const promptIds = Array.from(tokenizer.encode(prompt));
    if (promptIds.length === 0) throw new Error("encoded prompt is empty");
    if (promptIds.length >= state.modelConfig.blockSize) {
      return {
        promptTokens: promptIds.length,
        generatedIds: [] as number[],
        text: "",
        eosTerminated: false,
        hitBlockLimit: true,
      };
    }
    let logits = prefill(model, new Int32Array(promptIds));
    const generatedIds: number[] = [];
    let pos = promptIds.length;
    let eosTerminated = false;
    for (let i = 0; i < maxTokens && pos < state.modelConfig.blockSize; i++) {
      const next = sampleFromLogits(model, logits, 0, 0, rng, 1);
      generatedIds.push(next);
      if (next === eosId) {
        eosTerminated = true;
        break;
      }
      logits = decodeStep(model, next, pos);
      pos++;
    }
    const contentIds = eosTerminated ? generatedIds.slice(0, -1) : generatedIds;
    return {
      promptTokens: promptIds.length,
      generatedIds,
      text: tokenizer.decode(contentIds),
      eosTerminated,
      hitBlockLimit: pos >= state.modelConfig.blockSize && !eosTerminated,
    };
  }

  console.log(`Frozen eval: ${chatCases.length} chat + ${qaCases.length} QA cases`);
  console.log(`Checkpoint: step=${state.step}, ${state.modelConfig.nLayer}L/${state.modelConfig.nEmbd}D`);
  const startedAt = new Date().toISOString();
  const wallStart = performance.now();
  const chatResults: Array<Record<string, unknown>> = [];
  for (let i = 0; i < chatCases.length; i++) {
    const test = chatCases[i];
    const generated = generate(formatFrozenChatPrompt(test.messages), chatMaxTokens);
    const contentIds = generated.eosTerminated
      ? generated.generatedIds.slice(0, -1)
      : generated.generatedIds;
    const roleLeak = contentIds.includes(userId);
    const repeatRate = fourGramRepeatRate(contentIds);
    const nonempty = generated.text.trim().length > 0;
    chatResults.push({
      id: test.id,
      source: test.source,
      ...generated,
      roleLeak,
      nonempty,
      fourGramRepeatRate: repeatRate,
      degenerateLoop: repeatRate >= 0.2,
      structuralPass: generated.eosTerminated && !roleLeak && nonempty,
    });
    if ((i + 1) % 10 === 0) console.log(`  chat ${i + 1}/${chatCases.length}`);
  }

  const qaResults: Array<Record<string, unknown>> = [];
  for (let i = 0; i < qaCases.length; i++) {
    const test = qaCases[i];
    const generated = generate(`<|user|> ${test.question} <|assistant|> `, qaMaxTokens);
    const normalizedPrediction = normalizedAnswer(generated.text);
    const normalizedExpected = normalizedAnswer(test.answer);
    qaResults.push({
      id: test.id,
      title: test.title,
      field: test.field,
      url: test.url,
      expected: test.answer,
      ...generated,
      normalizedPrediction,
      normalizedExpected,
      exactMatch: normalizedPrediction === normalizedExpected,
      answerContained: answerIsContained(generated.text, test.answer),
      tokenF1: answerTokenF1(generated.text, test.answer),
    });
    if ((i + 1) % 20 === 0) console.log(`  qa ${i + 1}/${qaCases.length}`);
  }

  const chatRepeatRates = chatResults.map((row) => row.fourGramRepeatRate as number);
  const qaF1 = qaResults.map((row) => row.tokenF1 as number);
  const summary = {
    schema: "alpha-frozen-eval-results-v1",
    startedAt,
    completedAt: new Date().toISOString(),
    wallSeconds: (performance.now() - wallStart) / 1000,
    deterministicGreedy: true,
    checkpoint: {
      path: checkpointPath,
      sha256: await sha256File(checkpointPath),
      step: state.step,
      modelConfig: state.modelConfig,
    },
    inputs: {
      chat: { path: chatPath, sha256: await sha256File(chatPath), rows: chatCases.length },
      qa: { path: qaPath, sha256: await sha256File(qaPath), rows: qaCases.length },
    },
    generation: { chatMaxTokens, qaMaxTokens },
    chat: {
      total: chatResults.length,
      structuralPass: chatResults.filter((row) => row.structuralPass).length,
      eosTerminated: chatResults.filter((row) => row.eosTerminated).length,
      roleLeaks: chatResults.filter((row) => row.roleLeak).length,
      nonempty: chatResults.filter((row) => row.nonempty).length,
      degenerateLoops: chatResults.filter((row) => row.degenerateLoop).length,
      meanFourGramRepeatRate: mean(chatRepeatRates),
      maxFourGramRepeatRate: Math.max(0, ...chatRepeatRates),
    },
    closedBookQa: {
      total: qaResults.length,
      exactMatch: qaResults.filter((row) => row.exactMatch).length,
      answerContained: qaResults.filter((row) => row.answerContained).length,
      meanTokenF1: mean(qaF1),
    },
  };

  await mkdir(outDir, { recursive: true });
  await Promise.all([
    atomicWrite(join(outDir, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`),
    atomicWrite(join(outDir, "chat-results.jsonl"), `${chatResults.map((row) => JSON.stringify(row)).join("\n")}\n`),
    atomicWrite(join(outDir, "qa-results.jsonl"), `${qaResults.map((row) => JSON.stringify(row)).join("\n")}\n`),
  ]);
  console.log(JSON.stringify(summary, null, 2));
}
