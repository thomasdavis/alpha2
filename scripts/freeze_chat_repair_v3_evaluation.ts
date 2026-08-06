#!/usr/bin/env npx tsx
/** Freeze the exact development-only evaluation contract for chat repair v3. */

import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import { Effect } from "effect";
import { FileCheckpoint, formatFrozenChatPrompt } from "@alpha/train";

interface FrozenMessage {
  role: "user" | "assistant";
  content: string;
}

interface PromptRow {
  id: string;
  source: string;
  messages: FrozenMessage[];
  reference: string;
  prompt_tokens?: number;
  prompt?: string;
  prompt_sha256?: string;
}

interface FileRecord {
  path: string;
  sha256: string;
  bytes: number;
  rows?: number;
}

function parseArgs(): Record<string, string> {
  const result: Record<string, string> = {};
  for (let index = 2; index < process.argv.length; index += 2) {
    const key = process.argv[index];
    const value = process.argv[index + 1];
    if (!key?.startsWith("--") || !value || value.startsWith("--")) {
      throw new Error(`expected --key value, received ${key ?? ""} ${value ?? ""}`.trim());
    }
    result[key.slice(2)] = value;
  }
  return result;
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function parseJsonl<T>(text: string, label: string): T[] {
  return text.split("\n").filter((line) => line.trim().length > 0).map((line, index) => {
    try {
      return JSON.parse(line) as T;
    } catch (error) {
      throw new Error(`${label}:${index + 1} is invalid JSON`, { cause: error });
    }
  });
}

function sha256Text(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

async function sha256File(path: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk as Buffer);
  return hash.digest("hex");
}

async function fileRecord(path: string, rows?: number): Promise<FileRecord> {
  const { stat } = await import("node:fs/promises");
  const metadata = await stat(path);
  return { path: resolve(path), sha256: await sha256File(path), bytes: metadata.size, ...(rows === undefined ? {} : { rows }) };
}

async function atomicWrite(path: string, content: string): Promise<void> {
  const temporary = `${path}.tmp`;
  await writeFile(temporary, content, { encoding: "utf8", flag: "wx" });
  await rename(temporary, path);
}

function normalizedPrompt(messages: readonly FrozenMessage[]): string {
  return formatFrozenChatPrompt(messages).normalize("NFKC").replace(/\s+/gu, " ").trim().toLocaleLowerCase("en-US");
}

async function main(): Promise<void> {
  const args = parseArgs();
  for (const key of ["freeze", "v2-prompts", "v2-analysis", "initial-checkpoint", "out-dir"] as const) {
    if (!args[key]) throw new Error(`required: --${key}`);
  }
  const { execFileSync } = await import("node:child_process");
  assert(execFileSync("git", ["status", "--porcelain"], { encoding: "utf8" }).trim().length === 0,
    "evaluation freeze requires a clean committed worktree");
  const freezePath = resolve(args.freeze);
  const v2PromptsPath = resolve(args["v2-prompts"]);
  const v2AnalysisPath = resolve(args["v2-analysis"]);
  const initialCheckpointPath = resolve(args["initial-checkpoint"]);
  const outDir = resolve(args["out-dir"]);
  const [freezeText, v2PromptText, analysisText] = await Promise.all([
    readFile(freezePath, "utf8"),
    readFile(v2PromptsPath, "utf8"),
    readFile(v2AnalysisPath, "utf8"),
  ]);
  const freeze = JSON.parse(freezeText) as any;
  const analysis = JSON.parse(analysisText) as any;
  const v2Prompts = parseJsonl<PromptRow>(v2PromptText, v2PromptsPath);

  assert(freeze.schema === "alpha-chat-repair-v3-freeze-v1", "unexpected v3 freeze schema");
  assert(freeze.status === "rollout-candidates-and-development-frozen; no-rollouts-generated; final-sealed", "v3 freeze status drift");
  assert(freeze.contract?.block_size === 512, "v3 freeze block size must be 512");
  assert(freeze.contract?.generation_reserve === 128, "v3 generation reserve must be 128");
  assert(freeze.counts?.development_selected === 96, "v3 selector must contain 96 rows");
  assert(freeze.contract?.panel_count === 24, "v3 panel must contain 24 rows");

  const selectorPath = resolve(freeze.outputs?.development_selector?.path ?? "");
  const panelPath = resolve(freeze.outputs?.development_panel?.path ?? "");
  const [selectorText, panelText] = await Promise.all([readFile(selectorPath, "utf8"), readFile(panelPath, "utf8")]);
  const selector = parseJsonl<PromptRow>(selectorText, selectorPath);
  const panel = parseJsonl<PromptRow>(panelText, panelPath);
  assert(selector.length === 96, `v3 selector row count is ${selector.length}, expected 96`);
  assert(panel.length === 24, `v3 panel row count is ${panel.length}, expected 24`);
  assert(sha256Text(selectorText) === freeze.outputs.development_selector.sha256, "v3 selector hash differs from freeze");
  assert(sha256Text(panelText) === freeze.outputs.development_panel.sha256, "v3 panel hash differs from freeze");
  const selectorIds = new Set(selector.map((row) => row.id));
  assert(selectorIds.size === selector.length, "v3 selector IDs are not unique");
  assert(panel.every((row) => selectorIds.has(row.id)), "v3 panel is not a subset of the selector");
  for (const [index, row] of selector.entries()) {
    const rendered = formatFrozenChatPrompt(row.messages);
    assert(row.prompt === rendered, `v3 selector row ${index + 1} prompt rendering drift`);
    assert(row.prompt_sha256 === sha256Text(rendered), `v3 selector row ${index + 1} prompt hash drift`);
  }

  assert(analysis.schema === "alpha-chat-repair-transition-analysis-v1", "unexpected v2 transition-analysis schema");
  assert(analysis.sharedPromptCount === 69, "v2 regression population must contain 69 prompts");
  assert(Array.isArray(analysis.promptTransitions) && analysis.promptTransitions.length === 69, "v2 transition rows must contain 69 prompts");
  const v2PromptHash = sha256Text(v2PromptText);
  assert(analysis.inputs?.suite?.sha256 === v2PromptHash, "v2 transition analysis is bound to a different prompt suite");
  assert(analysis.inputs?.suite?.rows === v2Prompts.length, "v2 prompt count differs from transition analysis");
  const eligibleIds = new Set<string>(analysis.promptTransitions.map((row: any) => row.id));
  assert(eligibleIds.size === 69, "v2 eligible IDs are not unique");
  const regressionPrompts = v2Prompts.filter((row) => eligibleIds.has(row.id));
  assert(regressionPrompts.length === 69, `only ${regressionPrompts.length}/69 v2 regression prompts were resolved`);
  assert(regressionPrompts.every((row) => row.messages.length > 0 && row.messages.at(-1)?.role === "user"), "v2 regression contains invalid dialogue history");

  const freshNormalized = new Set(selector.map((row) => normalizedPrompt(row.messages)));
  const regressionOverlap = regressionPrompts.filter((row) => freshNormalized.has(normalizedPrompt(row.messages)));
  assert(regressionOverlap.length === 0, "fresh selector and v2 regression contain a normalized-prompt overlap");

  const checkpoint = await Effect.runPromise(new FileCheckpoint().load(initialCheckpointPath));
  assert(checkpoint.step === 1200, `initial checkpoint step is ${checkpoint.step}, expected 1200`);
  assert(checkpoint.modelConfig.blockSize === 512, "initial checkpoint does not have native 512-token context");
  assert(checkpoint.modelConfig.nLayer === 16 && checkpoint.modelConfig.nEmbd === 512 && checkpoint.modelConfig.nHead === 8,
    "initial checkpoint architecture differs from the frozen experiment");
  const initialRecord = await fileRecord(initialCheckpointPath);
  assert(initialRecord.sha256 === "399f776b49acc0c8834ff8a7f2390454e2c5f2d833a264e3f83ff546e973cfec",
    `initial checkpoint hash mismatch: ${initialRecord.sha256}`);

  await mkdir(outDir, { recursive: false });
  const regressionPath = join(outDir, "v2-eligible69-prompts.jsonl");
  const regressionText = `${regressionPrompts.map((row) => JSON.stringify(row)).join("\n")}\n`;
  await atomicWrite(regressionPath, regressionText);
  const [freezeRecord, selectorRecord, panelRecord, regressionRecord, v2SuiteRecord, v2AnalysisRecord] = await Promise.all([
    fileRecord(freezePath),
    fileRecord(selectorPath, selector.length),
    fileRecord(panelPath, panel.length),
    fileRecord(regressionPath, regressionPrompts.length),
    fileRecord(v2PromptsPath, v2Prompts.length),
    fileRecord(v2AnalysisPath),
  ]);
  const sourceCommit = execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim();
  const sourceCommitUtc = execFileSync("git", ["show", "-s", "--format=%cI", sourceCommit], { encoding: "utf8" }).trim();
  const contract = {
    schema: "alpha-chat-repair-v3-evaluation-contract-v1",
    status: "development-only-frozen; sealed-finals-excluded",
    source_commit: sourceCommit,
    source_commit_utc: sourceCommitUtc,
    inputs: {
      freeze_manifest: freezeRecord,
      v2_full_suite: v2SuiteRecord,
      v2_transition_analysis: v2AnalysisRecord,
      initial_checkpoint: { ...initialRecord, step: checkpoint.step },
    },
    suites: {
      fresh96: selectorRecord,
      qualitative24: panelRecord,
      regression69: { ...regressionRecord, path: basename(regressionPath) },
    },
    generation: {
      deterministic_greedy: true,
      max_new_tokens: 128,
      temperature: 0,
      top_k: 0,
      context_tokens: 512,
      terminal_control_token: "<|end_of_text|>",
      role_markers_are_not_injected_after_prompt_rendering: true,
      source_reference_is_never_model_visible: true,
    },
    candidate_contract: {
      arms: ["C0", "U1"],
      declared_steps: [50, 100, 200, 400],
      checkpoint_filename: "checkpoint-{step}.json",
      checkpoint_and_run_contract_must_share_directory: true,
      run_contract_schema: "alpha-chat-repair-contract-v3",
      initial_checkpoint_sha256: initialRecord.sha256,
      required_model_config: { blockSize: 512, nLayer: 16, nEmbd: 512, nHead: 8, vocabSize: 12288 },
    },
    evaluation: {
      machine_metrics: ["structuralPass", "nonempty", "eosTerminated", "roleLeak", "degenerateLoop", "fourGramRepeatRate"],
      panel_requires_human_comparison: true,
      bge_is_supporting_diagnostic_only: true,
      loss_cannot_select: true,
      sealed_final_allowed: false,
    },
    exclusions: {
      v2_sealed_final_sha256: "8b71ab5f8843b14a8bbe56a473ea9cd0672b873024632c023abbe4935e48eb1d",
      older_frozen_final_sha256: "6c463debaaf4f59452bc5e88ce85ca81f64c8a9e91974822609b5ac0883f7121",
      files_materialized_or_opened_by_this_freeze: [],
    },
    notes: {
      regression_order: "original v2 development-suite order restricted to the exact eligible-69 ID set",
      contract_filename: basename(join(outDir, "evaluation-contract.json")),
    },
  };
  const contractPath = join(outDir, "evaluation-contract.json");
  await atomicWrite(contractPath, `${JSON.stringify(contract, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify({
    out_dir: outDir,
    contract: await fileRecord(contractPath),
    regression69: regressionRecord,
    fresh96: selectorRecord,
    panel24: panelRecord,
  }, null, 2)}\n`);
}

await main();
