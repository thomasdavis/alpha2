#!/usr/bin/env npx tsx
/** Fail-closed verification for the canonical one-epoch flagship SFT inputs. */

import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { open, readFile, stat } from "node:fs/promises";
import { isDeepStrictEqual } from "node:util";

interface Cli {
  data: string;
  manifest: string;
  lengthAudit: string;
  maskAudit: string;
  tokenizer: string;
  baseCheckpoint: string;
  expectedBaseStep: number;
}

interface FileDigest {
  path: string;
  bytes: number;
  sha256: string;
}

interface CheckpointHeader {
  modelConfig?: Record<string, unknown>;
  step?: number;
  tokenizerArtifacts?: Record<string, unknown>;
  tensors?: { name?: string; shape?: number[]; elements?: number }[];
}

interface ParsedCheckpointHeader {
  header: CheckpointHeader;
  headerLength: number;
}

const EXPECTED_ROWS = 511_428;
const EXPECTED_TRAIN_ROWS = 485_150;
const EXPECTED_VAL_ROWS = 26_278;
const EXPECTED_PARAMS = 57_688_576;
const BATCH_SIZE = 16;
const BLOCK_SIZE = 1024;
const SEED = 42;
const VAL_FRACTION = 0.05;

function parseArgs(): Cli {
  const raw: Record<string, string> = {};
  for (let index = 2; index < process.argv.length; index++) {
    const arg = process.argv[index];
    if (!arg.startsWith("--")) throw new Error(`unexpected argument: ${arg}`);
    const value = process.argv[++index];
    if (!value || value.startsWith("--")) throw new Error(`missing value for ${arg}`);
    raw[arg.slice(2)] = value;
  }
  for (const key of ["data", "manifest", "lengthAudit", "maskAudit", "tokenizer", "baseCheckpoint", "expectedBaseStep"]) {
    if (!raw[key]) throw new Error(`missing --${key}`);
  }
  const expectedBaseStep = Number(raw.expectedBaseStep);
  if (!Number.isSafeInteger(expectedBaseStep) || expectedBaseStep < 1) {
    throw new Error(`invalid --expectedBaseStep: ${raw.expectedBaseStep}`);
  }
  return { ...raw, expectedBaseStep } as Cli;
}

async function digestFile(path: string, countLines = false): Promise<FileDigest & { lines?: number }> {
  const hash = createHash("sha256");
  let bytes = 0;
  let lines = 0;
  let finalByte: number | null = null;
  for await (const rawChunk of createReadStream(path)) {
    const chunk = rawChunk as Buffer;
    hash.update(chunk);
    bytes += chunk.length;
    if (countLines) {
      for (const byte of chunk) if (byte === 0x0a) lines++;
      if (chunk.length > 0) finalByte = chunk[chunk.length - 1];
    }
  }
  if (countLines && bytes > 0 && finalByte !== 0x0a) lines++;
  return { path, bytes, sha256: hash.digest("hex"), ...(countLines ? { lines } : {}) };
}

function requireEqual(actual: unknown, expected: unknown, label: string): void {
  if (!isDeepStrictEqual(actual, expected)) {
    throw new Error(`${label}: ${JSON.stringify(actual)} != ${JSON.stringify(expected)}`);
  }
}

function fnv1a32(input: string): number {
  let hash = 0x811c9dc5;
  for (let index = 0; index < input.length; index++) {
    hash ^= input.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

function deriveSplit(rows: number): { train: number; val: number } {
  let val = 0;
  for (let index = 0; index < rows; index++) {
    if (fnv1a32(`${SEED}:${index}`) / 0x1_0000_0000 < VAL_FRACTION) val++;
  }
  return { train: rows - val, val };
}

async function readCheckpointHeader(path: string): Promise<ParsedCheckpointHeader> {
  const handle = await open(path, "r");
  try {
    const prefix = Buffer.alloc(8);
    const prefixRead = await handle.read(prefix, 0, prefix.length, 0);
    if (prefixRead.bytesRead !== prefix.length || prefix.subarray(0, 4).toString("ascii") !== "ALPH") {
      throw new Error("base checkpoint is not an Alpha binary checkpoint");
    }
    const headerLength = prefix.readUInt32LE(4);
    if (headerLength < 2 || headerLength > 64 * 1024 * 1024) {
      throw new Error(`base checkpoint header length is invalid: ${headerLength}`);
    }
    const headerBytes = Buffer.alloc(headerLength);
    const headerRead = await handle.read(headerBytes, 0, headerLength, 8);
    if (headerRead.bytesRead !== headerLength) throw new Error("base checkpoint header is truncated");
    return {
      header: JSON.parse(headerBytes.toString("utf8")) as CheckpointHeader,
      headerLength,
    };
  } finally {
    await handle.close();
  }
}

async function inspectCheckpointPayload(
  path: string,
  parsed: ParsedCheckpointHeader,
  fileBytes: number,
): Promise<{ finiteParameterElements: number; nonzeroParameterElements: number }> {
  const tensors = parsed.header.tensors;
  if (!Array.isArray(tensors) || tensors.length === 0) throw new Error("base checkpoint has no tensor table");
  let expectedBytes = 8 + parsed.headerLength;
  for (const tensor of tensors) {
    if (!Number.isSafeInteger(tensor.elements) || Number(tensor.elements) < 0) {
      throw new Error(`base checkpoint tensor ${String(tensor.name)} has invalid element count`);
    }
    const elements = Number(tensor.elements);
    const shape = tensor.shape;
    if (!Array.isArray(shape) || shape.some((dim) => !Number.isSafeInteger(dim) || dim < 0)) {
      throw new Error(`base checkpoint tensor ${String(tensor.name)} has invalid shape`);
    }
    const shapeElements = shape.reduce((product, dim) => product * dim, 1);
    requireEqual(shapeElements, elements, `base checkpoint tensor ${String(tensor.name)} shape`);
    expectedBytes += elements * 4;
  }
  requireEqual(fileBytes, expectedBytes, "base checkpoint payload byte count");

  const handle = await open(path, "r");
  let offset = 8 + parsed.headerLength;
  let finiteParameterElements = 0;
  let nonzeroParameterElements = 0;
  const chunk = Buffer.allocUnsafe(8 * 1024 * 1024);
  try {
    for (const tensor of tensors) {
      let remaining = Number(tensor.elements) * 4;
      if (!tensor.name?.startsWith("p.")) {
        offset += remaining;
        continue;
      }
      while (remaining > 0) {
        const wanted = Math.min(chunk.length, remaining);
        let filled = 0;
        while (filled < wanted) {
          const result = await handle.read(chunk, filled, wanted - filled, offset + filled);
          if (result.bytesRead === 0) throw new Error(`base checkpoint tensor ${tensor.name} is truncated`);
          filled += result.bytesRead;
        }
        for (let byteOffset = 0; byteOffset < wanted; byteOffset += 4) {
          const bits = chunk.readUInt32LE(byteOffset);
          if ((bits & 0x7f80_0000) === 0x7f80_0000) {
            throw new Error(`base checkpoint tensor ${tensor.name} contains a non-finite parameter`);
          }
          finiteParameterElements++;
          if ((bits & 0x7fff_ffff) !== 0) nonzeroParameterElements++;
        }
        offset += wanted;
        remaining -= wanted;
      }
    }
  } finally {
    await handle.close();
  }
  if (nonzeroParameterElements < finiteParameterElements / 2) {
    throw new Error(
      `base checkpoint parameter payload is implausibly sparse: ${nonzeroParameterElements}/${finiteParameterElements} nonzero`,
    );
  }
  return { finiteParameterElements, nonzeroParameterElements };
}

async function main(): Promise<void> {
  const cli = parseArgs();
  const [manifestText, lengthText, maskText, tokenizerText, corpus, manifestFile, lengthFile, maskFile, tokenizerFile, baseFile] =
    await Promise.all([
      readFile(cli.manifest, "utf8"),
      readFile(cli.lengthAudit, "utf8"),
      readFile(cli.maskAudit, "utf8"),
      readFile(cli.tokenizer, "utf8"),
      digestFile(cli.data, true),
      digestFile(cli.manifest),
      digestFile(cli.lengthAudit),
      digestFile(cli.maskAudit),
      digestFile(cli.tokenizer),
      digestFile(cli.baseCheckpoint),
    ]);
  const manifest = JSON.parse(manifestText) as any;
  const lengthAudit = JSON.parse(lengthText) as any;
  const maskAudit = JSON.parse(maskText) as any;
  const tokenizer = JSON.parse(tokenizerText) as any;

  requireEqual(manifest.schema, "alpha-sft-corpus-v2", "SFT manifest schema");
  requireEqual(manifest.total, EXPECTED_ROWS, "SFT manifest rows");
  requireEqual(corpus.lines, EXPECTED_ROWS, "SFT corpus line count");
  requireEqual(manifest.output?.bytes, corpus.bytes, "SFT corpus bytes");
  requireEqual(manifest.output?.sha256, corpus.sha256, "SFT corpus SHA-256");
  requireEqual(manifest.max_tokens, BLOCK_SIZE, "SFT manifest token bound");
  const sourceCount = Object.values(manifest.counts ?? {}).reduce((sum: number, value) => sum + Number(value), 0);
  requireEqual(sourceCount, EXPECTED_ROWS, "SFT manifest source-count total");
  let nextLine = 1;
  for (const span of manifest.source_spans ?? []) {
    requireEqual(span.start_line, nextLine, `SFT source span ${String(span.source)} start`);
    if (!Number.isSafeInteger(span.end_line) || span.end_line < span.start_line) {
      throw new Error(`SFT source span ${String(span.source)} has an invalid end`);
    }
    nextLine = span.end_line + 1;
  }
  requireEqual(nextLine, EXPECTED_ROWS + 1, "SFT source spans coverage");

  requireEqual(lengthAudit.schema, "alpha-sft-length-audit-v1", "length-audit schema");
  requireEqual(lengthAudit.result, "PASS", "length-audit result");
  requireEqual(lengthAudit.corpus?.rows, EXPECTED_ROWS, "length-audit rows");
  requireEqual(lengthAudit.corpus?.sha256, corpus.sha256, "length-audit corpus SHA-256");
  requireEqual(lengthAudit.token_bound, BLOCK_SIZE, "length-audit token bound");
  requireEqual(lengthAudit.rows_over_bound, 0, "length-audit over-bound rows");
  if (lengthAudit.overall?.max > BLOCK_SIZE) throw new Error(`length-audit maximum exceeds ${BLOCK_SIZE}`);

  requireEqual(maskAudit.schema, "alpha-sft-mask-audit-v1", "mask-audit schema");
  requireEqual(maskAudit.result, "PASS", "mask-audit result");
  requireEqual(maskAudit.corpus?.rows, EXPECTED_ROWS, "mask-audit rows");
  requireEqual(maskAudit.corpus?.sha256, corpus.sha256, "mask-audit corpus SHA-256");
  requireEqual(maskAudit.selection?.block_size, BLOCK_SIZE, "mask-audit block size");
  requireEqual(maskAudit.mask_checks?.rows_over_block_size, 0, "mask-audit over-bound rows");
  for (const key of ["assistant_only_state_machine", "role_markers_atomic", "final_eot_supervised"]) {
    requireEqual(maskAudit.mask_checks?.[key], "PASS", `mask-audit ${key}`);
  }

  requireEqual(tokenizer.type, "byte_bpe", "tokenizer type");
  requireEqual(tokenizer.vocabSize, 12_288, "tokenizer vocab size");
  requireEqual(tokenizer.specialTokens, ["<|user|>", "<|assistant|>", "<|end_of_text|>"], "tokenizer chat specials");

  const split = deriveSplit(EXPECTED_ROWS);
  requireEqual(split, { train: EXPECTED_TRAIN_ROWS, val: EXPECTED_VAL_ROWS }, "deterministic SFT split");
  const steps = Math.ceil(split.train / BATCH_SIZE);
  requireEqual(steps, 30_322, "one-epoch SFT steps");

  const baseStat = await stat(cli.baseCheckpoint);
  if (baseStat.size < 650 * 1024 * 1024 || baseStat.size > 750 * 1024 * 1024) {
    throw new Error(`base checkpoint size ${baseStat.size} is outside the full flagship+AdamW envelope`);
  }
  const parsedHeader = await readCheckpointHeader(cli.baseCheckpoint);
  const header = parsedHeader.header;
  requireEqual(header.step, cli.expectedBaseStep, "base checkpoint step");
  const expectedModel = {
    vocabSize: 12_288,
    blockSize: BLOCK_SIZE,
    nLayer: 16,
    nEmbd: 512,
    nHead: 8,
    dropout: 0,
    ffnActivation: "swiglu",
    ffnDim: 1408,
    normType: "rmsnorm",
    posEnc: "rope",
    ropeTheta: 10_000,
    tieEmbeddings: true,
  };
  requireEqual(header.modelConfig, expectedModel, "base checkpoint architecture");
  requireEqual(header.tokenizerArtifacts, tokenizer, "base checkpoint tokenizer artifacts");
  if (!Array.isArray(header.tensors) || header.tensors.length === 0) throw new Error("base checkpoint has no tensor table");
  const parameterTensors = header.tensors.filter((tensor) => tensor.name?.startsWith("p."));
  const parameterNames = parameterTensors.map((tensor) => tensor.name);
  requireEqual(new Set(parameterNames).size, parameterNames.length, "base checkpoint parameter tensor names");
  const parameterElements = parameterTensors.reduce((sum, tensor) => sum + Number(tensor.elements), 0);
  requireEqual(parameterElements, EXPECTED_PARAMS, "base checkpoint parameter count");
  const payload = await inspectCheckpointPayload(cli.baseCheckpoint, parsedHeader, baseStat.size);
  requireEqual(payload.finiteParameterElements, EXPECTED_PARAMS, "base checkpoint finite parameter count");

  const result = {
    schema: "alpha-flagship-sft-input-verification-v1",
    result: "PASS",
    corpus,
    manifest: manifestFile,
    length_audit: lengthFile,
    mask_audit: maskFile,
    tokenizer: tokenizerFile,
    base_checkpoint: {
      ...baseFile,
      step: header.step,
      parameter_tensors: parameterTensors.length,
      parameter_elements: parameterElements,
      finite_parameter_elements: payload.finiteParameterElements,
      nonzero_parameter_elements: payload.nonzeroParameterElements,
      model_config: header.modelConfig,
    },
    split: {
      seed: SEED,
      validation_fraction: VAL_FRACTION,
      train_conversations: split.train,
      validation_conversations: split.val,
      batch_size: BATCH_SIZE,
      one_epoch_steps: steps,
      padded_conversations: steps * BATCH_SIZE,
      padded_tokens: steps * BATCH_SIZE * BLOCK_SIZE,
    },
  };
  process.stdout.write(JSON.stringify(result, null, 2) + "\n");
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
