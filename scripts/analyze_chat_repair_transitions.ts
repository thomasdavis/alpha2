#!/usr/bin/env npx tsx

import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { createInterface } from "node:readline";

type Message = { role: string; content: string };

type SuiteItem = {
  id: string;
  source: string;
  prompt_tokens: number;
  messages: Message[];
  reference?: string;
};

type ResultRow = {
  id: string;
  source: string;
  promptTokens: number;
  generatedIds: number[];
  text: string;
  eosTerminated: boolean;
  roleLeak: boolean;
  nonempty: boolean;
  fourGramRepeatRate: number;
  degenerateLoop: boolean;
  structuralPass: boolean;
};

type Run = {
  label: string;
  path: string;
  sha256: string;
  rows: Map<string, ResultRow>;
};

type Transition = {
  newLoops: number;
  fixedLoops: number;
  lostStructural: number;
  gainedStructural: number;
  fixedEmpty: number;
  newEmpty: number;
  lostEos: number;
  gainedEos: number;
};

type PhraseStat = {
  phrase: string;
  generatedRows: number;
  generatedOccurrences: number;
  supervisedTargetRows: number;
  supervisedOccurrences: number;
};

const args = new Map<string, string[]>();
for (const raw of process.argv.slice(2)) {
  const match = raw.match(/^--([^=]+)=(.*)$/s);
  if (!match) throw new Error(`Expected --key=value, received ${raw}`);
  const values = args.get(match[1]) ?? [];
  values.push(match[2]);
  args.set(match[1], values);
}

function required(name: string): string {
  const value = args.get(name)?.at(-1);
  if (!value) throw new Error(`Missing --${name}=...`);
  return resolve(value);
}

function numberArg(name: string, fallback: number): number {
  const value = args.get(name)?.at(-1);
  if (value === undefined) return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new Error(`Invalid --${name}=${value}`);
  return parsed;
}

const suitePath = required("suite");
const trainPath = required("train");
const outJson = required("out-json");
const outMarkdown = required("out-markdown");
const eligibleAuditPath = args.get("eligible-audit")?.at(-1)
  ? resolve(args.get("eligible-audit")!.at(-1)!)
  : null;
const ngramSize = numberArg("ngram", 4);
const echoThreshold = numberArg("echo-threshold", 0.6);
const targetRepeatThreshold = numberArg("target-repeat-threshold", 0.2);
const generatedAt = args.get("generated-at")?.at(-1) ?? new Date().toISOString();
const runArgs = args.get("run") ?? [];

if (runArgs.length < 2) throw new Error("Provide at least two --run=label:path arguments");
if (!Number.isInteger(ngramSize) || ngramSize < 2) throw new Error("--ngram must be an integer >= 2");
if (Number.isNaN(Date.parse(generatedAt))) throw new Error(`Invalid --generated-at=${generatedAt}`);

async function sha256File(path: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk as Buffer);
  return hash.digest("hex");
}

async function readJsonl<T>(path: string): Promise<T[]> {
  const rows: T[] = [];
  const lines = createInterface({ input: createReadStream(path), crlfDelay: Infinity });
  for await (const line of lines) {
    if (!line.trim()) continue;
    rows.push(JSON.parse(line) as T);
  }
  return rows;
}

function words(text: string): string[] {
  return text.toLocaleLowerCase("en-US").match(/[\p{L}\p{N}]+(?:['’-][\p{L}\p{N}]+)*/gu) ?? [];
}

function normalized(text: string): string {
  return words(text).join(" ");
}

function quantile(values: number[], q: number): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = (sorted.length - 1) * q;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower];
  return sorted[lower] * (upper - index) + sorted[upper] * (index - lower);
}

function mean(values: number[]): number | null {
  return values.length === 0 ? null : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function jaccard(leftText: string, rightText: string): number {
  const left = new Set(words(leftText));
  const right = new Set(words(rightText));
  if (left.size === 0 && right.size === 0) return 1;
  const intersection = [...left].filter((token) => right.has(token)).length;
  return intersection / (left.size + right.size - intersection);
}

function ngrams<T>(tokens: T[], n: number): string[] {
  const values: string[] = [];
  for (let index = 0; index <= tokens.length - n; index += 1) {
    values.push(tokens.slice(index, index + n).join("\u001f"));
  }
  return values;
}

function repeatProfile<T>(tokens: T[], n: number): {
  total: number;
  repeated: number;
  rate: number;
  firstRepeatOnset: number | null;
  dominant: string | null;
  dominantCount: number;
} {
  const grams = ngrams(tokens, n);
  const seen = new Map<string, number>();
  let repeated = 0;
  let firstRepeatOnset: number | null = null;
  let dominant: string | null = null;
  let dominantCount = 0;
  grams.forEach((gram, index) => {
    const count = (seen.get(gram) ?? 0) + 1;
    seen.set(gram, count);
    if (count > 1) {
      repeated += 1;
      firstRepeatOnset ??= index;
    }
    if (count > dominantCount) {
      dominant = gram;
      dominantCount = count;
    }
  });
  return {
    total: grams.length,
    repeated,
    rate: grams.length === 0 ? 0 : repeated / grams.length,
    firstRepeatOnset,
    dominant,
    dominantCount,
  };
}

function summarize(rows: ResultRow[], suite: Map<string, SuiteItem>) {
  const repeatOnsets = rows
    .filter((row) => row.degenerateLoop)
    .map((row) => repeatProfile(row.generatedIds, ngramSize).firstRepeatOnset)
    .filter((value): value is number => value !== null);
  const echo = rows.map((row) => {
    const item = suite.get(row.id);
    const lastUser = [...(item?.messages ?? [])].reverse().find((message) => message.role === "user")?.content ?? "";
    return jaccard(lastUser, row.text);
  });
  return {
    total: rows.length,
    structural: rows.filter((row) => row.structuralPass).length,
    nonempty: rows.filter((row) => row.nonempty).length,
    eos: rows.filter((row) => row.eosTerminated).length,
    roleLeaks: rows.filter((row) => row.roleLeak).length,
    loops: rows.filter((row) => row.degenerateLoop).length,
    meanFourGramRepeatRate: mean(rows.map((row) => row.fourGramRepeatRate)),
    generatedTokenMedian: quantile(rows.map((row) => row.generatedIds.length), 0.5),
    loopOnsetTokenQ25: quantile(repeatOnsets, 0.25),
    loopOnsetTokenMedian: quantile(repeatOnsets, 0.5),
    loopOnsetTokenQ75: quantile(repeatOnsets, 0.75),
    lastUserOutputJaccardMean: mean(echo),
    highEcho: echo.filter((value) => value >= echoThreshold).length,
  };
}

function summarizeBySource(rows: ResultRow[], suite: Map<string, SuiteItem>) {
  const grouped = new Map<string, ResultRow[]>();
  for (const row of rows) {
    const source = suite.get(row.id)?.source ?? row.source;
    const values = grouped.get(source) ?? [];
    values.push(row);
    grouped.set(source, values);
  }
  return Object.fromEntries([...grouped].sort().map(([source, values]) => [source, summarize(values, suite)]));
}

function transition(baseline: ResultRow[], candidate: ResultRow[]): Transition {
  const base = new Map(baseline.map((row) => [row.id, row]));
  const result: Transition = {
    newLoops: 0,
    fixedLoops: 0,
    lostStructural: 0,
    gainedStructural: 0,
    fixedEmpty: 0,
    newEmpty: 0,
    lostEos: 0,
    gainedEos: 0,
  };
  for (const current of candidate) {
    const prior = base.get(current.id);
    if (!prior) continue;
    if (!prior.degenerateLoop && current.degenerateLoop) result.newLoops += 1;
    if (prior.degenerateLoop && !current.degenerateLoop) result.fixedLoops += 1;
    if (prior.structuralPass && !current.structuralPass) result.lostStructural += 1;
    if (!prior.structuralPass && current.structuralPass) result.gainedStructural += 1;
    if (!prior.nonempty && current.nonempty) result.fixedEmpty += 1;
    if (prior.nonempty && !current.nonempty) result.newEmpty += 1;
    if (prior.eosTerminated && !current.eosTerminated) result.lostEos += 1;
    if (!prior.eosTerminated && current.eosTerminated) result.gainedEos += 1;
  }
  return result;
}

function parseConversation(line: string): Message[] {
  const marker = /<\|([^|]+)\|>/g;
  const matches = [...line.matchAll(marker)];
  const messages: Message[] = [];
  for (let index = 0; index < matches.length; index += 1) {
    const role = matches[index][1];
    if (role === "end_of_text") continue;
    const start = (matches[index].index ?? 0) + matches[index][0].length;
    const end = matches[index + 1]?.index ?? line.length;
    messages.push({ role, content: line.slice(start, end).trim() });
  }
  return messages;
}

async function scanTraining(
  path: string,
  phrases: Set<string>,
): Promise<{
  conversations: number;
  assistantTargets: number;
  exactTargets: Set<string>;
  targetWordLength: { q25: number | null; median: number | null; q75: number | null };
  targetsAboveRepeatThreshold: number;
  targetRepeatRateMean: number | null;
  targetLastUserJaccardMean: number | null;
  targetHighEcho: number;
  topStarts: Array<{ start: string; count: number }>;
  phraseRows: Map<string, number>;
  phraseOccurrences: Map<string, number>;
}> {
  const exactTargets = new Set<string>();
  const lengths: number[] = [];
  const repeatRates: number[] = [];
  const echoes: number[] = [];
  const starts = new Map<string, number>();
  const phraseRows = new Map([...phrases].map((phrase) => [phrase, 0]));
  const phraseOccurrences = new Map([...phrases].map((phrase) => [phrase, 0]));
  let conversations = 0;
  let assistantTargets = 0;
  let targetsAboveRepeatThreshold = 0;
  let targetHighEcho = 0;
  const lines = createInterface({ input: createReadStream(path), crlfDelay: Infinity });
  for await (const line of lines) {
    if (!line.trim()) continue;
    conversations += 1;
    const messages = parseConversation(line);
    let lastUser = "";
    for (const message of messages) {
      if (message.role === "user") {
        lastUser = message.content;
        continue;
      }
      if (message.role !== "assistant") continue;
      assistantTargets += 1;
      const targetWords = words(message.content);
      const target = targetWords.join(" ");
      exactTargets.add(target);
      lengths.push(targetWords.length);
      const start = targetWords.slice(0, ngramSize).join(" ");
      if (start) starts.set(start, (starts.get(start) ?? 0) + 1);
      const profile = repeatProfile(targetWords, ngramSize);
      repeatRates.push(profile.rate);
      if (profile.rate >= targetRepeatThreshold) targetsAboveRepeatThreshold += 1;
      const echo = jaccard(lastUser, message.content);
      echoes.push(echo);
      if (echo >= echoThreshold) targetHighEcho += 1;
      const grams = ngrams(targetWords, ngramSize).map((gram) => gram.replaceAll("\u001f", " "));
      const local = new Map<string, number>();
      for (const phrase of grams) {
        if (!phrases.has(phrase)) continue;
        local.set(phrase, (local.get(phrase) ?? 0) + 1);
      }
      for (const [phrase, count] of local) {
        phraseRows.set(phrase, (phraseRows.get(phrase) ?? 0) + 1);
        phraseOccurrences.set(phrase, (phraseOccurrences.get(phrase) ?? 0) + count);
      }
    }
  }
  return {
    conversations,
    assistantTargets,
    exactTargets,
    targetWordLength: { q25: quantile(lengths, 0.25), median: quantile(lengths, 0.5), q75: quantile(lengths, 0.75) },
    targetsAboveRepeatThreshold,
    targetRepeatRateMean: mean(repeatRates),
    targetLastUserJaccardMean: mean(echoes),
    targetHighEcho,
    topStarts: [...starts.entries()]
      .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
      .slice(0, 20)
      .map(([start, count]) => ({ start, count })),
    phraseRows,
    phraseOccurrences,
  };
}

const suiteRows = await readJsonl<SuiteItem>(suitePath);
const suite = new Map(suiteRows.map((item) => [item.id, item]));
const runs: Run[] = [];
for (const value of runArgs) {
  const separator = value.indexOf(":");
  if (separator <= 0) throw new Error(`Invalid --run=${value}; expected label:path`);
  const label = value.slice(0, separator);
  const path = resolve(value.slice(separator + 1));
  const rows = await readJsonl<ResultRow>(path);
  runs.push({ label, path, sha256: await sha256File(path), rows: new Map(rows.map((row) => [row.id, row])) });
}

let eligibleIds: Set<string> | null = null;
if (eligibleAuditPath) {
  const audit = JSON.parse(await readFile(eligibleAuditPath, "utf8")) as {
    rows?: Array<{ id?: string; generationEligible?: boolean }>;
  };
  if (!Array.isArray(audit.rows)) throw new Error(`Eligible audit has no rows array: ${eligibleAuditPath}`);
  eligibleIds = new Set(audit.rows
    .filter((row) => row.generationEligible === true && typeof row.id === "string")
    .map((row) => row.id!));
  if (eligibleIds.size === 0) throw new Error(`Eligible audit selected zero IDs: ${eligibleAuditPath}`);
}
const sharedIds = [...suite.keys()]
  .filter((id) => runs.every((run) => run.rows.has(id)))
  .filter((id) => eligibleIds === null || eligibleIds.has(id))
  .sort();
const baseline = runs[0];
const runSummaries = Object.fromEntries(runs.map((run) => {
  const allRows = [...run.rows.values()];
  const sharedRows = sharedIds.map((id) => run.rows.get(id)!);
  return [run.label, {
    path: run.path,
    sha256: run.sha256,
    full: summarize(allRows, suite),
    shared: summarize(sharedRows, suite),
    sharedBySource: summarizeBySource(sharedRows, suite),
    versusBaseline: run === baseline ? null : transition(sharedIds.map((id) => baseline.rows.get(id)!), sharedRows),
  }];
}));

const generatedPhraseCounts = new Map<string, { rows: Set<string>; occurrences: number }>();
for (const run of runs) {
  for (const row of run.rows.values()) {
    if (!row.degenerateLoop) continue;
    const profile = repeatProfile(words(row.text), ngramSize);
    if (!profile.dominant || profile.dominantCount < 2) continue;
    const phrase = profile.dominant.replaceAll("\u001f", " ");
    const value = generatedPhraseCounts.get(phrase) ?? { rows: new Set<string>(), occurrences: 0 };
    value.rows.add(`${run.label}:${row.id}`);
    value.occurrences += profile.dominantCount;
    generatedPhraseCounts.set(phrase, value);
  }
}

const training = await scanTraining(trainPath, new Set(generatedPhraseCounts.keys()));
const phraseStats: PhraseStat[] = [...generatedPhraseCounts]
  .map(([phrase, generated]) => ({
    phrase,
    generatedRows: generated.rows.size,
    generatedOccurrences: generated.occurrences,
    supervisedTargetRows: training.phraseRows.get(phrase) ?? 0,
    supervisedOccurrences: training.phraseOccurrences.get(phrase) ?? 0,
  }))
  .sort((left, right) => right.generatedRows - left.generatedRows || right.generatedOccurrences - left.generatedOccurrences);

const promptTransitions = sharedIds.map((id) => {
  const item = suite.get(id)!;
  const lastUser = [...item.messages].reverse().find((message) => message.role === "user")?.content ?? "";
  const outputs = Object.fromEntries(runs.map((run) => {
    const row = run.rows.get(id)!;
    const profile = repeatProfile(row.generatedIds, ngramSize);
    return [run.label, {
      structural: row.structuralPass,
      nonempty: row.nonempty,
      eos: row.eosTerminated,
      loop: row.degenerateLoop,
      repeatRate: row.fourGramRepeatRate,
      generatedTokens: row.generatedIds.length,
      firstRepeatOnset: profile.firstRepeatOnset,
      lastUserOutputJaccard: jaccard(lastUser, row.text),
      exactSupervisedTarget: training.exactTargets.has(normalized(row.text)),
      text: row.text,
    }];
  }));
  return { id, source: item.source, promptTokens: item.prompt_tokens, lastUser, outputs };
});

const v2Runs = runs.slice(1);
const persistentNewLoopIds = sharedIds.filter((id) =>
  !baseline.rows.get(id)!.degenerateLoop && v2Runs.every((run) => run.rows.get(id)!.degenerateLoop));
const everNewLoopIds = sharedIds.filter((id) =>
  !baseline.rows.get(id)!.degenerateLoop && v2Runs.some((run) => run.rows.get(id)!.degenerateLoop));
const alwaysFixedBaselineLoopIds = sharedIds.filter((id) =>
  baseline.rows.get(id)!.degenerateLoop && v2Runs.every((run) => !run.rows.get(id)!.degenerateLoop));

const report = {
  schema: "alpha-chat-repair-transition-analysis-v1",
  generatedAt,
  configuration: { ngramSize, echoThreshold, targetRepeatThreshold },
  inputs: {
    suite: { path: suitePath, sha256: await sha256File(suitePath), rows: suiteRows.length },
    train: { path: trainPath, sha256: await sha256File(trainPath) },
    eligibleAudit: eligibleAuditPath
      ? { path: eligibleAuditPath, sha256: await sha256File(eligibleAuditPath), selectedIds: eligibleIds!.size }
      : null,
    runs: runs.map(({ label, path, sha256, rows }) => ({ label, path, sha256, rows: rows.size })),
  },
  sharedPromptCount: sharedIds.length,
  runSummaries,
  crossRun: {
    persistentNewLoopIds,
    everNewLoopIds,
    alwaysFixedBaselineLoopIds,
  },
  training: {
    conversations: training.conversations,
    assistantTargets: training.assistantTargets,
    targetWordLength: training.targetWordLength,
    targetRepeatThreshold,
    targetsAboveRepeatThreshold: training.targetsAboveRepeatThreshold,
    targetRepeatRateMean: training.targetRepeatRateMean,
    echoThreshold,
    targetLastUserJaccardMean: training.targetLastUserJaccardMean,
    targetHighEcho: training.targetHighEcho,
    exactDistinctTargets: training.exactTargets.size,
    topStarts: training.topStarts,
  },
  phraseStats,
  promptTransitions,
};

const md: string[] = [];
md.push("# Alpha chat repair transition analysis", "");
md.push(`Generated: ${report.generatedAt}`, "");
md.push(`Shared prompts across ${runs.length} runs: **${sharedIds.length}**.`, "");
md.push("## Shared-prompt summary", "");
md.push("| Run | Structural | Nonempty | EOS | Loops | Mean repeat | Median loop onset | High echo |", "|---|---:|---:|---:|---:|---:|---:|---:|");
for (const run of runs) {
  const summary = runSummaries[run.label].shared;
  md.push(`| ${run.label} | ${summary.structural}/${summary.total} | ${summary.nonempty}/${summary.total} | ${summary.eos}/${summary.total} | ${summary.loops} | ${summary.meanFourGramRepeatRate?.toFixed(4)} | ${summary.loopOnsetTokenMedian ?? "n/a"} | ${summary.highEcho} |`);
}
md.push("", "## Transition from baseline", "");
md.push("| Run | New loops | Fixed loops | Lost structural | Gained structural | Fixed empty | Lost EOS | Gained EOS |", "|---|---:|---:|---:|---:|---:|---:|---:|");
for (const run of runs.slice(1)) {
  const value = runSummaries[run.label].versusBaseline!;
  md.push(`| ${run.label} | ${value.newLoops} | ${value.fixedLoops} | ${value.lostStructural} | ${value.gainedStructural} | ${value.fixedEmpty} | ${value.lostEos} | ${value.gainedEos} |`);
}
md.push("", "## Cross-run invariants", "");
md.push(`- New baseline-clean prompts that looped in at least one v2 run: ${everNewLoopIds.length}.`);
md.push(`- New baseline-clean prompts that looped in every v2 run: ${persistentNewLoopIds.length}.`);
md.push(`- Baseline loops fixed by every v2 run: ${alwaysFixedBaselineLoopIds.length}.`);
md.push("", "## Supervised-target audit", "");
md.push(`- Conversations: ${training.conversations}; assistant targets: ${training.assistantTargets}.`);
md.push(`- Target word lengths q25/median/q75: ${training.targetWordLength.q25}/${training.targetWordLength.median}/${training.targetWordLength.q75}.`);
md.push(`- Targets at or above word-${ngramSize} repeat rate ${targetRepeatThreshold}: ${training.targetsAboveRepeatThreshold}/${training.assistantTargets}.`);
md.push(`- Targets at or above last-user echo Jaccard ${echoThreshold}: ${training.targetHighEcho}/${training.assistantTargets}.`);
md.push("", "## Dominant generated loop phrases versus supervision", "");
md.push("| Phrase | Generated rows | Generated occurrences | Target rows | Target occurrences |", "|---|---:|---:|---:|---:|");
for (const phrase of phraseStats.slice(0, 40)) {
  md.push(`| ${phrase.phrase.replaceAll("|", "\\|")} | ${phrase.generatedRows} | ${phrase.generatedOccurrences} | ${phrase.supervisedTargetRows} | ${phrase.supervisedOccurrences} |`);
}
md.push("", "The JSON sibling contains every shared prompt, exact output, transition flag, loop onset, echo score, and input hash.", "");

await mkdir(dirname(outJson), { recursive: true });
await mkdir(dirname(outMarkdown), { recursive: true });
await writeFile(outJson, `${JSON.stringify(report, null, 2)}\n`, "utf8");
await writeFile(outMarkdown, `${md.join("\n")}\n`, "utf8");

console.log(JSON.stringify({
  schema: report.schema,
  sharedPrompts: sharedIds.length,
  runs: runs.length,
  trainingTargets: training.assistantTargets,
  phraseStats: phraseStats.length,
  outJson,
  outMarkdown,
}));
