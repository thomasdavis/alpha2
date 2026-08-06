#!/usr/bin/env npx tsx

/** Compile a complete raw rollout ledger into the immutable RCR-UL cohort. */

import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { mkdir, readFile, rename, stat, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { createInterface } from "node:readline";
import { fourGramRepeatRate, repeatedFourGramCompletionPositions } from "@alpha/train";

type Candidate = {
  schema: "alpha-rcr-ul-rollout-candidate-v1";
  stable_id: string;
  source: string;
  positive_conversation_sha256: string;
  prompt_sha256: string;
  prompt_token_ids: number[];
};

type RawRollout = {
  schema: "alpha-rcr-ul-raw-rollout-v1";
  stable_id: string;
  source: string;
  positive_conversation_sha256: string;
  prompt_sha256: string;
  checkpoint_sha256: string;
  prompt_token_ids: number[];
  generated_token_ids: number[];
  content_token_ids: number[];
  token_audit: Array<{ token_id: number }>;
  text: string;
  stop_reason: "learned_eos" | "role_boundary" | "max_tokens" | "block_limit";
  stop_token_id: number | null;
  eos_terminated: boolean;
  four_gram_repeat_rate: number;
  degenerate_loop: boolean;
  output_sha256: string;
};

type FileIdentity = { path: string; sha256: string; rows?: number };

const cli = new Map<string, string>();
for (const raw of process.argv.slice(2)) {
  const match = raw.match(/^--([^=]+)=(.*)$/s);
  if (!match) throw new Error(`expected --key=value, received ${raw}`);
  cli.set(match[1], match[2]);
}

function required(name: string): string {
  const value = cli.get(name);
  if (!value) throw new Error(`missing --${name}=...`);
  return resolve(value);
}

function sha256Text(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

async function sha256File(path: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk as Buffer);
  return hash.digest("hex");
}

async function fileIdentity(path: string, rows?: number): Promise<FileIdentity> {
  return { path, sha256: await sha256File(path), ...(rows === undefined ? {} : { rows }) };
}

async function readJsonl<T>(path: string): Promise<T[]> {
  const rows: T[] = [];
  const lines = createInterface({ input: createReadStream(path), crlfDelay: Infinity });
  let lineNumber = 0;
  for await (const line of lines) {
    lineNumber++;
    if (!line.trim()) continue;
    try {
      rows.push(JSON.parse(line) as T);
    } catch (error) {
      throw new Error(`${path}:${lineNumber} is not valid JSON`, { cause: error });
    }
  }
  return rows;
}

async function atomicWrite(path: string, content: string): Promise<void> {
  const temporary = `${path}.tmp`;
  await writeFile(temporary, content, "utf8");
  await rename(temporary, path);
}

function equalNumbers(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

function quantiles(values: readonly number[]): Record<string, number | null> {
  if (values.length === 0) return { min: null, q25: null, median: null, q75: null, max: null };
  const ordered = [...values].sort((a, b) => a - b);
  const at = (fraction: number) => ordered[Math.round((ordered.length - 1) * fraction)];
  return { min: ordered[0], q25: at(0.25), median: at(0.5), q75: at(0.75), max: ordered.at(-1)! };
}

const candidatesPath = required("candidates");
const positivesPath = required("positive-cohort");
const freezeManifestPath = required("freeze-manifest");
const rawRolloutsPath = required("raw-rollouts");
const rolloutManifestPath = required("rollout-manifest");
const outDir = required("out-dir");
const parityReportPath = cli.get("parity-report") ? resolve(cli.get("parity-report")!) : null;
const expectedCheckpointSha256 = cli.get("expected-checkpoint-sha256");
if (!expectedCheckpointSha256 || !/^[0-9a-f]{64}$/.test(expectedCheckpointSha256)) {
  throw new Error("--expected-checkpoint-sha256 must be a lowercase SHA-256");
}

await mkdir(outDir, { recursive: true });
const negativePath = resolve(outDir, "negative-cohort.jsonl");
const auditPath = resolve(outDir, "mask-audit.jsonl");
const manifestPath = resolve(outDir, "rcr-ul-manifest.json");
for (const path of [negativePath, auditPath, manifestPath]) {
  try {
    await stat(path);
    throw new Error(`refusing to overwrite existing compiled artifact: ${path}`);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }
}

const [candidates, rawRows, freezeManifest, rolloutManifest] = await Promise.all([
  readJsonl<Candidate>(candidatesPath),
  readJsonl<RawRollout>(rawRolloutsPath),
  readFile(freezeManifestPath, "utf8").then((value) => JSON.parse(value) as any),
  readFile(rolloutManifestPath, "utf8").then((value) => JSON.parse(value) as any),
]);
let parityReport: any = null;
let parityReportIdentity: FileIdentity | null = null;
if (rolloutManifest.generation?.native_parity_required_before_admission === true) {
  if (!parityReportPath) {
    throw new Error("accelerated rollout manifest requires --parity-report=... before mask compilation");
  }
  parityReportIdentity = await fileIdentity(parityReportPath);
  parityReport = JSON.parse(await readFile(parityReportPath, "utf8"));
  if (parityReport.schema !== "alpha-rcr-ul-rollout-parity-v1" || parityReport.status !== "PASS") {
    throw new Error("accelerated rollout parity report is not a PASS");
  }
  if (parityReport.checkpoint_sha256 !== expectedCheckpointSha256 ||
      parityReport.hf_model_sha256 !== rolloutManifest.export?.model_sha256) {
    throw new Error("accelerated rollout parity report checkpoint/export mismatch");
  }
  if (!Number.isSafeInteger(parityReport.native?.rows) || parityReport.native.rows < 24 ||
      parityReport.native.rows !== parityReport.accelerated?.rows) {
    throw new Error("accelerated rollout parity population must contain at least 24 matched rows");
  }
  const acceleratedParityPath = resolve(parityReport.accelerated.path);
  if (await sha256File(acceleratedParityPath) !== parityReport.accelerated.sha256) {
    throw new Error("accelerated rollout parity source hash mismatch");
  }
  const parityRows = await readJsonl<RawRollout>(acceleratedParityPath);
  if (parityRows.length !== parityReport.accelerated.rows) {
    throw new Error("accelerated rollout parity source row count mismatch");
  }
  for (let index = 0; index < parityRows.length; index++) {
    const parityRow = parityRows[index];
    const fullRow = rawRows[index];
    if (parityRow.stable_id !== fullRow?.stable_id ||
        !equalNumbers(parityRow.generated_token_ids, fullRow.generated_token_ids) ||
        parityRow.stop_reason !== fullRow.stop_reason ||
        parityRow.output_sha256 !== fullRow.output_sha256) {
      throw new Error(`accelerated full rollout differs from parity trajectory at row ${index + 1}`);
    }
  }
} else if (parityReportPath) {
  throw new Error("--parity-report was supplied for a rollout manifest that does not declare accelerated parity");
}
const positiveLines = (await readFile(positivesPath, "utf8")).split(/\r?\n/).filter((line) => line.length > 0);
if (freezeManifest.schema !== "alpha-chat-repair-v3-freeze-v1") throw new Error("unexpected freeze manifest schema");
if (rolloutManifest.schema !== "alpha-rcr-ul-rollout-manifest-v1" || rolloutManifest.status !== "complete") {
  throw new Error("raw rollout manifest is not complete");
}
if (candidates.length === 0 || candidates.length !== rawRows.length || candidates.length !== positiveLines.length) {
  throw new Error(
    `cohort row mismatch: candidates=${candidates.length} raw=${rawRows.length} positives=${positiveLines.length}`,
  );
}

const [candidateIdentity, positiveIdentity, rawIdentity] = await Promise.all([
  fileIdentity(candidatesPath, candidates.length),
  fileIdentity(positivesPath, positiveLines.length),
  fileIdentity(rawRolloutsPath, rawRows.length),
]);
if (freezeManifest.outputs?.rollout_candidates?.sha256 !== candidateIdentity.sha256 ||
    freezeManifest.outputs?.positive_cohort?.sha256 !== positiveIdentity.sha256) {
  throw new Error("freeze manifest no longer matches candidate or positive cohort bytes");
}
if (rolloutManifest.candidates?.sha256 !== candidateIdentity.sha256 ||
    rolloutManifest.output?.sha256 !== rawIdentity.sha256 ||
    rolloutManifest.checkpoint?.sha256 !== expectedCheckpointSha256) {
  throw new Error("rollout manifest input/output/checkpoint identity mismatch");
}
const eosId = rolloutManifest.checkpoint?.control_token_ids?.eos;
if (!Number.isSafeInteger(eosId) || eosId < 0) throw new Error("rollout manifest lacks an atomic EOS token identity");

const negativeRows: string[] = [];
const auditRows: string[] = [];
const penaltyPositions: number[] = [];
const onsetPositions: number[] = [];
const badTokenCounts = new Map<number, number>();
const eligibleBySource = new Map<string, number>();
let eligibleRows = 0;

for (let index = 0; index < candidates.length; index++) {
  const candidate = candidates[index];
  const raw = rawRows[index];
  const label = `row ${index + 1}`;
  if (candidate.schema !== "alpha-rcr-ul-rollout-candidate-v1" || raw.schema !== "alpha-rcr-ul-raw-rollout-v1") {
    throw new Error(`${label} has unexpected schema`);
  }
  if (candidate.stable_id !== raw.stable_id || candidate.source !== raw.source ||
      candidate.prompt_sha256 !== raw.prompt_sha256 ||
      candidate.positive_conversation_sha256 !== raw.positive_conversation_sha256) {
    throw new Error(`${label} candidate/raw identity mismatch`);
  }
  const positiveLine = positiveLines[index];
  if (positiveLine !== positiveLine.trim()) throw new Error(`${label} positive conversation has boundary whitespace`);
  if (sha256Text(positiveLine) !== candidate.positive_conversation_sha256) {
    throw new Error(`${label} positive conversation hash mismatch`);
  }
  if (!equalNumbers(candidate.prompt_token_ids, raw.prompt_token_ids)) throw new Error(`${label} prompt tokens drifted`);
  if (raw.checkpoint_sha256 !== expectedCheckpointSha256) throw new Error(`${label} checkpoint drifted`);
  if (raw.token_audit.length !== raw.generated_token_ids.length ||
      raw.token_audit.some((audit, tokenIndex) => audit.token_id !== raw.generated_token_ids[tokenIndex])) {
    throw new Error(`${label} generated-token audit mismatch`);
  }
  const expectedGenerated = raw.stop_reason === "learned_eos" || raw.stop_reason === "role_boundary"
    ? [...raw.content_token_ids, raw.stop_token_id!]
    : [...raw.content_token_ids];
  if (!equalNumbers(expectedGenerated, raw.generated_token_ids)) throw new Error(`${label} stop-token contract mismatch`);
  if (raw.stop_reason === "learned_eos" && raw.stop_token_id !== eosId) throw new Error(`${label} learned-EOS identity mismatch`);
  if (raw.stop_reason !== "learned_eos" && raw.eos_terminated) throw new Error(`${label} EOS termination flag mismatch`);
  const repeatRate = fourGramRepeatRate(raw.content_token_ids);
  if (Math.abs(repeatRate - raw.four_gram_repeat_rate) > 1e-12 || raw.degenerate_loop !== (repeatRate >= 0.2)) {
    throw new Error(`${label} loop classification mismatch`);
  }
  if (sha256Text(JSON.stringify({
    prompt_token_ids: raw.prompt_token_ids,
    generated_token_ids: raw.generated_token_ids,
  })) !== raw.output_sha256) throw new Error(`${label} output hash mismatch`);

  const repeatedContentPositions = repeatedFourGramCompletionPositions(raw.content_token_ids);
  const eligible = raw.degenerate_loop && repeatedContentPositions.length > 0;
  const tokenIds = [
    ...raw.prompt_token_ids,
    ...raw.content_token_ids,
    ...(raw.stop_reason === "learned_eos" ? [eosId] : []),
  ];
  const absolutePenaltyPositions = eligible
    ? repeatedContentPositions.map((position) => raw.prompt_token_ids.length + position)
    : [];
  if (absolutePenaltyPositions.some((position) => tokenIds[position] === eosId)) {
    throw new Error(`${label} attempts to penalize EOS`);
  }
  if (eligible) {
    eligibleRows++;
    eligibleBySource.set(candidate.source, (eligibleBySource.get(candidate.source) ?? 0) + 1);
    onsetPositions.push(repeatedContentPositions[0]);
    for (const position of absolutePenaltyPositions) {
      penaltyPositions.push(position);
      const token = tokenIds[position];
      badTokenCounts.set(token, (badTokenCounts.get(token) ?? 0) + 1);
    }
  }
  negativeRows.push(JSON.stringify({
    schema: "alpha-rcr-ul-example-v1",
    stable_id: candidate.stable_id,
    positive_conversation_sha256: candidate.positive_conversation_sha256,
    token_ids: tokenIds,
    penalty_target_positions: absolutePenaltyPositions,
  }));
  auditRows.push(JSON.stringify({
    schema: "alpha-rcr-ul-mask-audit-v1",
    stable_id: candidate.stable_id,
    source: candidate.source,
    raw_output_sha256: raw.output_sha256,
    stop_reason: raw.stop_reason,
    prompt_tokens: raw.prompt_token_ids.length,
    content_tokens: raw.content_token_ids.length,
    four_gram_repeat_rate: repeatRate,
    degenerate_loop: raw.degenerate_loop,
    eligible,
    repeated_completion_content_positions: repeatedContentPositions,
    penalty_target_positions: absolutePenaltyPositions,
  }));
}

const negativeText = `${negativeRows.join("\n")}\n`;
const auditText = `${auditRows.join("\n")}\n`;
await atomicWrite(negativePath, negativeText);
await atomicWrite(auditPath, auditText);
const negativeIdentity = await fileIdentity(negativePath, negativeRows.length);
const auditIdentity = await fileIdentity(auditPath, auditRows.length);
const topBadTokens = [...badTokenCounts.entries()]
  .sort((a, b) => b[1] - a[1] || a[0] - b[0])
  .slice(0, 100)
  .map(([token_id, count]) => ({ token_id, count }));
const manifest = {
  schema: "alpha-rcr-ul-cohort-manifest-v1",
  status: "complete-and-immutable",
  created_at: new Date().toISOString(),
  inputs: {
    freeze_manifest: await fileIdentity(freezeManifestPath),
    rollout_manifest: await fileIdentity(rolloutManifestPath),
    candidates: candidateIdentity,
    positive_cohort: positiveIdentity,
    raw_rollouts: rawIdentity,
    checkpoint_sha256: expectedCheckpointSha256,
    ...(parityReportIdentity ? { accelerated_parity_report: parityReportIdentity } : {}),
  },
  outputs: {
    negative_cohort: negativeIdentity,
    mask_audit: auditIdentity,
  },
  rule: {
    ngram_size: 4,
    degenerate_loop_threshold: 0.2,
    penalty: "every generated non-EOS token completing a previously seen token 4-gram, only when the full content repeat rate is >= 0.2",
    model_judge: false,
    semantic_blacklist: false,
  },
  summary: {
    rows: candidates.length,
    eligible_negative_rows: eligibleRows,
    ineligible_zero_mask_rows: candidates.length - eligibleRows,
    total_penalty_positions: penaltyPositions.length,
    eligible_by_source: Object.fromEntries([...eligibleBySource.entries()].sort(([a], [b]) => a.localeCompare(b))),
    onset_content_position_distribution: quantiles(onsetPositions),
    absolute_penalty_position_distribution: quantiles(penaltyPositions),
    top_bad_token_ids: topBadTokens,
  },
};
await atomicWrite(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
console.log(JSON.stringify(manifest, null, 2));
