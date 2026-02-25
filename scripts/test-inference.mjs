#!/usr/bin/env node
/**
 * Quick inference test — loads a checkpoint and generates text.
 * Usage: node scripts/test-inference.mjs <checkpoint-path> [prompt]
 */
import { readFileSync } from "node:fs";

const ckptPath = process.argv[2] || "runs/20260222_103922/checkpoint-200.json";
const prompt = process.argv[3] || "The ";

console.log(`Loading checkpoint: ${ckptPath}`);
const t0 = Date.now();
const raw = readFileSync(ckptPath);
console.log(`  Read ${(raw.length / 1024 / 1024).toFixed(1)}MB in ${Date.now() - t0}ms`);

// Parse binary checkpoint
let offset = 4; // skip ALPH magic
const headerLen = raw.readUInt32LE(offset); offset += 4;
const header = JSON.parse(raw.subarray(offset, offset + headerLen).toString("utf-8"));
offset += headerLen;

console.log(`  Model: ${header.modelConfig.nEmbd}d ${header.modelConfig.nHead}h ${header.modelConfig.nLayer}L`);
console.log(`  Vocab: ${header.modelConfig.vocabSize}, Block: ${header.modelConfig.blockSize}`);

// Extract param tensors
const params = {};
for (const t of header.tensors) {
  const byteLen = t.elements * 4;
  if (t.name.startsWith("p.")) {
    params[t.name.slice(2)] = {
      shape: t.shape,
      data: new Float32Array(raw.buffer.slice(raw.byteOffset + offset, raw.byteOffset + offset + byteLen)),
    };
  }
  offset += byteLen;
}

// Build model using the alpha packages
const { CpuRefBackend } = await import("@alpha/tensor");
const { initGPT, collectParams, countParams, gptForward } = await import("@alpha/model");
const { Tape, Variable } = await import("@alpha/autograd");
const { SeededRng } = await import("@alpha/core");
const { BpeTokenizer } = await import("@alpha/tokenizers");

const backend = new CpuRefBackend();
const config = header.modelConfig;
const rng = new SeededRng(42);

console.log("Initializing model...");
const gptParams = initGPT(config, backend, rng);

// Restore params from checkpoint
const paramMap = collectParams(gptParams);
for (const [name, variable] of paramMap) {
  const saved = params[name];
  if (!saved) continue;
  const arr = variable.data.data;
  for (let i = 0; i < arr.length; i++) {
    arr[i] = saved.data[i];
  }
}

const paramCount = countParams(gptParams);
console.log(`  ${(paramCount / 1e6).toFixed(2)}M params loaded in ${Date.now() - t0}ms`);

// Load tokenizer
const tokenizer = new BpeTokenizer();
tokenizer.loadArtifacts(header.tokenizerArtifacts);

// Encode prompt
const promptTokens = tokenizer.encode(prompt);
console.log(`\nPrompt: "${prompt}" → ${promptTokens.length} tokens: [${Array.from(promptTokens).slice(0, 10)}]`);

// Generate tokens
const maxTokens = 50;
const temperature = 0.8;
const topk = 40;
const { blockSize, vocabSize } = config;

const tokens = new Int32Array(Math.min(promptTokens.length + maxTokens, blockSize));
tokens.set(promptTokens);
let currentLen = promptTokens.length;

let output = prompt;
console.log(`\nGenerating ${maxTokens} tokens...`);

for (let i = 0; i < maxTokens && currentLen < blockSize; i++) {
  const tokenStart = performance.now();

  const ctxStart = Math.max(0, currentLen - blockSize);
  const ctxLen = currentLen - ctxStart;
  const ctx = tokens.slice(ctxStart, ctxStart + ctxLen);

  const inputData = { shape: [1, ctxLen], dtype: "i32", data: new Int32Array(ctx) };
  const tape = new Tape();
  const { logits } = gptForward(config, gptParams, backend, tape, inputData);

  // Sample from logits
  const lastLogits = new Float32Array(vocabSize);
  const logitsArr = logits.data.data;
  const logitOffset = (ctxLen - 1) * vocabSize;
  for (let v = 0; v < vocabSize; v++) {
    lastLogits[v] = logitsArr[logitOffset + v] / temperature;
  }

  // Top-k filtering
  if (topk > 0 && topk < vocabSize) {
    const indexed = Array.from(lastLogits).map((val, idx) => ({ val, idx }));
    indexed.sort((a, b) => b.val - a.val);
    const threshold = indexed[topk - 1].val;
    for (let v = 0; v < vocabSize; v++) {
      if (lastLogits[v] < threshold) lastLogits[v] = -Infinity;
    }
  }

  // Softmax
  let maxVal = -Infinity;
  for (let v = 0; v < vocabSize; v++) {
    if (lastLogits[v] > maxVal) maxVal = lastLogits[v];
  }
  let sumExp = 0;
  const probs = new Float32Array(vocabSize);
  for (let v = 0; v < vocabSize; v++) {
    probs[v] = Math.exp(lastLogits[v] - maxVal);
    sumExp += probs[v];
  }
  for (let v = 0; v < vocabSize; v++) {
    probs[v] /= sumExp;
  }

  // Sample
  const r = Math.random();
  let cumsum = 0;
  let nextToken = 0;
  for (let v = 0; v < vocabSize; v++) {
    cumsum += probs[v];
    if (r < cumsum) { nextToken = v; break; }
  }

  tokens[currentLen] = nextToken;
  currentLen++;

  const decoded = tokenizer.decode(new Int32Array([nextToken]));
  output += decoded;

  const tokenTime = (performance.now() - tokenStart).toFixed(0);
  if (i < 5 || i % 10 === 0) {
    console.log(`  token ${i + 1}: ${JSON.stringify(decoded)} (${tokenTime}ms)`);
  }
}

console.log(`\n${"=".repeat(60)}`);
console.log(`Output: ${output}`);
console.log(`${"=".repeat(60)}`);
console.log(`Total time: ${((Date.now() - t0) / 1000).toFixed(1)}s`);
