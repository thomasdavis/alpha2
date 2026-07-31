#!/usr/bin/env npx tsx
/**
 * Stream-audit a built SFT corpus against its manifest and the exact tokenizer
 * used for training. Every Nth row plus both ends of every source span are
 * tokenized. The audit independently checks the assistant-only state machine
 * and the next-token loss targets produced by buildSftExample.
 *
 * Usage:
 *   npx tsx scripts/verify_sft_masks.ts \
 *     --data /path/sft-v2.txt \
 *     --manifest /path/sft-v2.txt.manifest.json \
 *     --tokenizer /path/bpe-byte-12k.json \
 *     --out /path/mask-audit.json [--every 500]
 */

import { createReadStream } from "node:fs";
import { readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import * as readline from "node:readline";
import { buildSftExample, resolveChatSpecialIds } from "@alpha/train";
import { tokenizerFromArtifacts } from "@alpha/tokenizers";
import type { TokenizerArtifacts } from "@alpha/core";

interface SourceSpan {
  source: string;
  start_line: number;
  end_line: number;
}

interface CorpusManifest {
  schema?: string;
  total?: number;
  output?: { sha256: string; bytes: number };
  source_spans?: SourceSpan[];
  rows?: Record<string, number>;
  outputs?: Record<string, { sha256: string; bytes: number }>;
}

function args(): Record<string, string> {
  const out: Record<string, string> = {};
  for (let i = 2; i < process.argv.length; i++) {
    const arg = process.argv[i];
    if (!arg.startsWith("--")) throw new Error(`unexpected argument: ${arg}`);
    const key = arg.slice(2);
    const value = process.argv[++i];
    if (!value || value.startsWith("--")) throw new Error(`missing value for --${key}`);
    out[key] = value;
  }
  return out;
}

function count(haystack: string, needle: string): number {
  let total = 0;
  for (let at = 0; (at = haystack.indexOf(needle, at)) >= 0; at += needle.length) total++;
  return total;
}

async function main(): Promise<void> {
  const cli = args();
  for (const required of ["data", "manifest", "tokenizer", "out"]) {
    if (!cli[required]) throw new Error(`required: --${required}`);
  }
  const every = Number(cli.every ?? "500");
  if (!Number.isSafeInteger(every) || every < 1) throw new Error("--every must be a positive integer");
  const blockSize = Number(cli.block ?? "1024");
  if (!Number.isSafeInteger(blockSize) || blockSize < 2) throw new Error("--block must be an integer >= 2");

  const manifest = JSON.parse(await readFile(cli.manifest, "utf8")) as CorpusManifest;
  const split = cli.split;
  const total = split ? manifest.rows?.[split] : manifest.total;
  const output = split ? manifest.outputs?.[split] : manifest.output;
  if (!Number.isSafeInteger(total) || (total ?? 0) < 1 || !output) {
    throw new Error(split
      ? `manifest has no valid rows/outputs entry for split ${split}`
      : "manifest has no valid total/output entry");
  }
  const sourceSpans = Array.isArray(manifest.source_spans) && manifest.source_spans.length > 0
    ? manifest.source_spans
    : [{ source: split ? `interleaved:${split}` : "interleaved", start_line: 1, end_line: total! }];
  const artifacts = JSON.parse(await readFile(cli.tokenizer, "utf8")) as TokenizerArtifacts;
  const tokenizer = tokenizerFromArtifacts(artifacts);
  const ids = resolveChatSpecialIds(tokenizer);

  const selected = new Set<number>([1, total!]);
  for (let line = every; line <= total!; line += every) selected.add(line);
  for (const span of sourceSpans) {
    selected.add(span.start_line);
    selected.add(span.end_line);
  }

  const hash = createHash("sha256");
  const input = createReadStream(cli.data, { encoding: "utf8" });
  const reader = readline.createInterface({ input, crlfDelay: Infinity });
  let lineNumber = 0;
  let bytes = 0;
  let sampled = 0;
  let sampledTokens = 0;
  let supervisedTargets = 0;
  let minTokens = Number.POSITIVE_INFINITY;
  let maxTokens = 0;
  const sampleLengths: number[] = [];

  for await (const line of reader) {
    lineNumber++;
    const encodedLine = Buffer.from(line + "\n", "utf8");
    bytes += encodedLine.length;
    hash.update(encodedLine);
    if (!selected.has(lineNumber)) continue;

    const example = buildSftExample(line, tokenizer, ids);
    if (example.tokens.length !== example.roleMask.length) {
      throw new Error(`line ${lineNumber}: token/mask length mismatch`);
    }
    const rawCounts = {
      user: count(line, "<|user|>"),
      assistant: count(line, "<|assistant|>"),
      eot: count(line, "<|end_of_text|>"),
    };
    const tokenCounts = { user: 0, assistant: 0, eot: 0 };
    let inAssistant = false;
    let lineSupervised = 0;
    for (let i = 0; i < example.tokens.length; i++) {
      const token = example.tokens[i];
      let expected: 0 | 1;
      if (token === ids.assistantId) {
        tokenCounts.assistant++;
        expected = 0;
        inAssistant = true;
      } else if (token === ids.userId) {
        tokenCounts.user++;
        expected = 0;
        inAssistant = false;
      } else if (token === ids.eotId) {
        tokenCounts.eot++;
        expected = inAssistant ? 1 : 0;
        inAssistant = false;
      } else {
        expected = inAssistant ? 1 : 0;
      }
      if (example.roleMask[i] !== expected) {
        throw new Error(`line ${lineNumber}, token ${i}: mask=${example.roleMask[i]}, expected=${expected}`);
      }
      if (i > 0 && example.roleMask[i] === 1) lineSupervised++;
    }
    if (JSON.stringify(rawCounts) !== JSON.stringify(tokenCounts)) {
      throw new Error(
        `line ${lineNumber}: marker counts differ raw=${JSON.stringify(rawCounts)} tokenized=${JSON.stringify(tokenCounts)}`,
      );
    }
    if (rawCounts.eot !== 1 || tokenCounts.eot !== 1) throw new Error(`line ${lineNumber}: expected one EOT`);
    if (example.roleMask[example.roleMask.length - 1] !== 1) {
      throw new Error(`line ${lineNumber}: final EOT is not an assistant-supervised target`);
    }
    if (lineSupervised === 0) throw new Error(`line ${lineNumber}: no supervised next-token targets`);

    sampled++;
    sampledTokens += example.tokens.length;
    supervisedTargets += lineSupervised;
    sampleLengths.push(example.tokens.length);
    minTokens = Math.min(minTokens, example.tokens.length);
    maxTokens = Math.max(maxTokens, example.tokens.length);
  }

  const sha256 = hash.digest("hex");
  if (lineNumber !== total) throw new Error(`row count ${lineNumber} != manifest ${total}`);
  if (bytes !== output.bytes) throw new Error(`byte count ${bytes} != manifest ${output.bytes}`);
  if (sha256 !== output.sha256) throw new Error(`sha256 ${sha256} != manifest ${output.sha256}`);
  if (sampled !== selected.size) throw new Error(`sample count ${sampled} != selected ${selected.size}`);
  sampleLengths.sort((a, b) => a - b);
  const percentile = (p: number) => sampleLengths[Math.min(sampleLengths.length - 1, Math.floor(p * sampleLengths.length))];
  const rowsOverBlock = sampleLengths.filter((length) => length > blockSize).length;

  const report = {
    schema: "alpha-sft-mask-audit-v1",
    result: "PASS",
    corpus: { path: cli.data, rows: lineNumber, bytes, sha256 },
    tokenizer: {
      path: cli.tokenizer,
      type: artifacts.type,
      vocab_size: tokenizer.vocabSize,
      special_ids: ids,
    },
    selection: {
      every,
      block_size: blockSize,
      source_spans: sourceSpans,
      rows_sampled: sampled,
    },
    mask_checks: {
      sampled_tokens: sampledTokens,
      supervised_next_token_targets: supervisedTargets,
      min_tokens_per_sample: minTokens,
      max_tokens_per_sample: maxTokens,
      p50_tokens_per_sample: percentile(0.50),
      p95_tokens_per_sample: percentile(0.95),
      p99_tokens_per_sample: percentile(0.99),
      rows_over_block_size: rowsOverBlock,
      rows_over_block_size_fraction: rowsOverBlock / sampled,
      assistant_only_state_machine: "PASS",
      role_markers_atomic: "PASS",
      final_eot_supervised: "PASS",
    },
  };
  await writeFile(cli.out, JSON.stringify(report, null, 2) + "\n", { encoding: "utf8", flag: "wx" });
  console.log(JSON.stringify(report, null, 2));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
