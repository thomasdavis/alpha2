#!/usr/bin/env npx tsx
/** Prepare a deterministic, arm-blinded qualitative review packet for admitted v3 checkpoints. */

import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";

interface PanelRow {
  id: string;
  messages: Array<{ role: string; content: string }>;
  prompt_sha256: string;
}

interface ResultRow {
  id: string;
  text: string;
  outputSha256: string;
}

interface CohortInput {
  step: number;
  initialPath: string;
  controlPath: string;
  candidatePath: string;
}

function parseArgs(): Record<string, string> {
  const args: Record<string, string> = {};
  for (let index = 2; index < process.argv.length; index += 2) {
    const key = process.argv[index];
    const value = process.argv[index + 1];
    if (!key?.startsWith("--") || !value || value.startsWith("--")) {
      throw new Error(`expected --key value, received ${key ?? ""} ${value ?? ""}`.trim());
    }
    args[key.slice(2)] = value;
  }
  return args;
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function sha256(text: string): string {
  return createHash("sha256").update(text).digest("hex");
}

function parseJsonl<T>(text: string, label: string): T[] {
  return text.split("\n").filter((line) => line.trim()).map((line, index) => {
    try {
      return JSON.parse(line) as T;
    } catch (error) {
      throw new Error(`${label}:${index + 1} invalid JSON`, { cause: error });
    }
  });
}

function orderedByHash<T>(values: readonly T[], seed: string, key: (value: T) => string): T[] {
  return [...values].sort((left, right) =>
    sha256(`${seed}\u0000${key(left)}`).localeCompare(sha256(`${seed}\u0000${key(right)}`))
  );
}

async function loadResults(path: string, panelIds: ReadonlySet<string>): Promise<{
  path: string;
  sha256: string;
  rows: Map<string, ResultRow>;
}> {
  const absolute = resolve(path);
  const text = await readFile(absolute, "utf8");
  const rows = parseJsonl<ResultRow>(text, absolute);
  const byId = new Map(rows.map((row) => [row.id, row]));
  assert(byId.size === rows.length, `${absolute} has duplicate IDs`);
  for (const id of panelIds) assert(byId.has(id), `${absolute} is missing panel ID ${id}`);
  return { path: absolute, sha256: sha256(text), rows: byId };
}

function renderConversation(messages: readonly { role: string; content: string }[]): string {
  return messages.map((message) => `**${message.role === "assistant" ? "Assistant" : "User"}:** ${message.content}`).join("\n\n");
}

async function main(): Promise<void> {
  const args = parseArgs();
  for (const key of [
    "panel", "initial-results", "control-50", "candidate-50", "control-400", "candidate-400", "out-dir", "seed",
  ] as const) {
    if (!args[key]) throw new Error(`required: --${key}`);
  }

  const panelPath = resolve(args.panel);
  const panelText = await readFile(panelPath, "utf8");
  const panel = parseJsonl<PanelRow>(panelText, panelPath);
  assert(panel.length === 24, `expected 24 frozen panel rows, received ${panel.length}`);
  const panelIds = new Set(panel.map((row) => row.id));
  assert(panelIds.size === panel.length, "frozen panel contains duplicate IDs");

  const cohortInputs: CohortInput[] = [
    {
      step: 50,
      initialPath: args["initial-results"],
      controlPath: args["control-50"],
      candidatePath: args["candidate-50"],
    },
    {
      step: 400,
      initialPath: args["initial-results"],
      controlPath: args["control-400"],
      candidatePath: args["candidate-400"],
    },
  ];
  const cohortLabels = ["X", "Y"];
  const shuffledCohorts = orderedByHash(cohortInputs, args.seed, (row) => `cohort-${row.step}`);
  const packetItems: any[] = [];
  const sealedItems: any[] = [];
  const inputs: any[] = [];

  for (let cohortIndex = 0; cohortIndex < shuffledCohorts.length; cohortIndex++) {
    const cohort = shuffledCohorts[cohortIndex]!;
    const cohortLabel = cohortLabels[cohortIndex]!;
    const [initial, control, candidate] = await Promise.all([
      loadResults(cohort.initialPath, panelIds),
      loadResults(cohort.controlPath, panelIds),
      loadResults(cohort.candidatePath, panelIds),
    ]);
    inputs.push({
      cohort: cohortLabel,
      hiddenStep: cohort.step,
      initial: { path: initial.path, sha256: initial.sha256 },
      control: { path: control.path, sha256: control.sha256 },
      candidate: { path: candidate.path, sha256: candidate.sha256 },
    });

    for (const panelRow of orderedByHash(panel, `${args.seed}:${cohortLabel}`, (row) => row.id)) {
      const roles = [
        { role: "initial", row: initial.rows.get(panelRow.id)! },
        { role: "control", row: control.rows.get(panelRow.id)! },
        { role: "candidate", row: candidate.rows.get(panelRow.id)! },
      ];
      const shuffled = orderedByHash(roles, `${args.seed}:${cohortLabel}:${panelRow.id}`, (row) => row.role);
      const labels = ["A", "B", "C"];
      const itemId = `${cohortLabel}-${panelRow.id}`;
      packetItems.push({
        itemId,
        cohort: cohortLabel,
        promptId: panelRow.id,
        promptSha256: panelRow.prompt_sha256,
        messages: panelRow.messages,
        responses: shuffled.map((entry, index) => ({ label: labels[index], text: entry.row.text })),
        review: {
          A_vs_B: null,
          A_vs_C: null,
          B_vs_C: null,
          wouldContinue: { A: null, B: null, C: null },
          notes: "",
        },
      });
      sealedItems.push({
        itemId,
        cohort: cohortLabel,
        hiddenStep: cohort.step,
        labels: Object.fromEntries(shuffled.map((entry, index) => [labels[index], {
          role: entry.role,
          outputSha256: entry.row.outputSha256,
        }])),
      });
    }
  }

  const packet = {
    schema: "alpha-chat-repair-v3-blind-review-packet-v1",
    status: "blinded; human-review-pending; sealed-final-untouched",
    packetSeedSha256: sha256(args.seed),
    rubric: {
      pairwiseValues: ["A", "B", "C", "tie"],
      instruction: "For each pair, choose the response you prefer or tie. Judge directness, coherence, completeness without padding, natural stopping, absence of parroting/template failure, and whether you would continue the conversation.",
      noReferenceAnswersShown: true,
    },
    items: packetItems,
  };
  const sealedKey = {
    schema: "alpha-chat-repair-v3-blind-review-key-v1",
    status: "SEALED_UNTIL_REVIEW_COMPLETE",
    panel: { path: panelPath, sha256: sha256(panelText), rows: panel.length },
    inputs,
    items: sealedItems,
  };

  const outDir = resolve(args["out-dir"]);
  await mkdir(outDir, { recursive: false });
  const packetText = `${JSON.stringify(packet, null, 2)}\n`;
  const keyText = `${JSON.stringify(sealedKey, null, 2)}\n`;
  const markdown = [
    "# Alpha chat repair v3 — blinded qualitative review",
    "",
    "Judge only the visible conversation and responses. Do not inspect `sealed-key.json` until every pairwise field is complete.",
    "For each pair write `A`, `B`, `C`, or `tie`, selecting only between the two labels named by that row.",
    "",
    ...packetItems.flatMap((item) => [
      `## ${item.itemId}`,
      "",
      renderConversation(item.messages),
      "",
      ...item.responses.flatMap((response: any) => [`### Response ${response.label}`, "", response.text || "*[empty]*", ""]),
      "| Judgment | Choice |",
      "|---|---|",
      "| A vs B |  |",
      "| A vs C |  |",
      "| B vs C |  |",
      "| Would continue with A? |  |",
      "| Would continue with B? |  |",
      "| Would continue with C? |  |",
      "",
      "Notes:",
      "",
    ]),
  ].join("\n");
  const reviewTemplate = {
    schema: "alpha-chat-repair-v3-blind-review-response-v1",
    packetSha256: sha256(packetText),
    reviewer: { kind: "human", identifier: null },
    completedUtc: null,
    items: packetItems.map((item) => ({ itemId: item.itemId, ...item.review })),
  };
  const reviewText = `${JSON.stringify(reviewTemplate, null, 2)}\n`;
  await Promise.all([
    writeFile(join(outDir, "review-packet.json"), packetText, { encoding: "utf8", flag: "wx" }),
    writeFile(join(outDir, "review-form.md"), `${markdown}\n`, { encoding: "utf8", flag: "wx" }),
    writeFile(join(outDir, "review-response.json"), reviewText, { encoding: "utf8", flag: "wx" }),
    writeFile(join(outDir, "sealed-key.json"), keyText, { encoding: "utf8", flag: "wx", mode: 0o600 }),
  ]);
  const manifest = {
    schema: "alpha-chat-repair-v3-blind-review-manifest-v1",
    status: "prepared; human-review-pending; key-sealed",
    files: {
      packet: { name: "review-packet.json", sha256: sha256(packetText) },
      form: { name: "review-form.md", sha256: sha256(`${markdown}\n`) },
      response: { name: "review-response.json", sha256: sha256(reviewText) },
      sealedKey: { name: "sealed-key.json", sha256: sha256(keyText) },
    },
    counts: { cohorts: 2, panelRowsPerCohort: panel.length, blindedTriads: packetItems.length },
    sealedFinal: { executed: false, inspected: false },
  };
  await writeFile(join(outDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
  process.stdout.write(`${JSON.stringify({ outDir, counts: manifest.counts, packetSha256: manifest.files.packet.sha256 }, null, 2)}\n`);
}

await main();
