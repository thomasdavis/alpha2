#!/usr/bin/env npx tsx
/** Build a reference-blinded comparison packet for several checkpoint outputs. */

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";

interface PanelRow {
  id: string;
  source: string;
  messages: Array<{ role: "user" | "assistant"; content: string }>;
}

interface ResultRow {
  id: string;
  text: string;
}

interface CandidateArg {
  blindLabel: string;
  actualLabel: string;
  path: string;
}

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function parseJsonl<T>(text: string, label: string): T[] {
  return text.split("\n").filter(Boolean).map((line, index) => {
    try {
      return JSON.parse(line) as T;
    } catch (error) {
      throw new Error(`${label}:${index + 1} is invalid JSON`, { cause: error });
    }
  });
}

function parseCandidate(value: string): CandidateArg {
  const first = value.indexOf("=");
  const second = value.indexOf("=", first + 1);
  if (first < 1 || second <= first + 1 || second === value.length - 1) {
    throw new Error(`candidate must be BLIND_LABEL=ACTUAL_LABEL=RESULTS_PATH: ${value}`);
  }
  return {
    blindLabel: value.slice(0, first),
    actualLabel: value.slice(first + 1, second),
    path: value.slice(second + 1),
  };
}

function parseArgs(): { panel: string; candidates: CandidateArg[]; packet: string; manifest: string } {
  let panel = "";
  let packet = "";
  let manifest = "";
  const candidates: CandidateArg[] = [];
  for (let index = 2; index < process.argv.length; index++) {
    const arg = process.argv[index];
    const value = process.argv[++index];
    if (!value || value.startsWith("--")) throw new Error(`missing value for ${arg}`);
    if (arg === "--panel") panel = value;
    else if (arg === "--candidate") candidates.push(parseCandidate(value));
    else if (arg === "--packet") packet = value;
    else if (arg === "--manifest") manifest = value;
    else throw new Error(`unexpected argument: ${arg}`);
  }
  if (!panel || !packet || !manifest || candidates.length < 2) {
    throw new Error("required: --panel, two or more --candidate values, --packet, and --manifest");
  }
  const blindLabels = new Set(candidates.map((candidate) => candidate.blindLabel));
  const actualLabels = new Set(candidates.map((candidate) => candidate.actualLabel));
  if (blindLabels.size !== candidates.length || actualLabels.size !== candidates.length) {
    throw new Error("candidate blind and actual labels must each be unique");
  }
  return { panel, candidates, packet, manifest };
}

async function main(): Promise<void> {
  const cli = parseArgs();
  const panelText = await readFile(cli.panel, "utf8");
  const panel = parseJsonl<PanelRow>(panelText, cli.panel);
  const panelIds = new Set(panel.map((row) => row.id));
  if (panelIds.size !== panel.length) throw new Error("panel IDs are not unique");
  for (const row of panel) {
    if (!row.id || !row.source || !Array.isArray(row.messages) || row.messages.length === 0) {
      throw new Error(`invalid panel row: ${row.id}`);
    }
  }

  const candidateData = await Promise.all(cli.candidates.map(async (candidate) => {
    const text = await readFile(candidate.path, "utf8");
    const rows = parseJsonl<ResultRow>(text, candidate.path);
    const byId = new Map(rows.map((row) => [row.id, row]));
    if (byId.size !== rows.length) throw new Error(`duplicate result ID in ${candidate.path}`);
    for (const id of panelIds) {
      if (!byId.has(id)) throw new Error(`candidate ${candidate.actualLabel} has no result for ${id}`);
    }
    return { candidate, text, rows, byId };
  }));

  const packet = {
    schema: "alpha-chat-checkpoint-comparison-packet-v1",
    status: "REFERENCE_BLINDED",
    scope: "Development-only comparison of conversational helpfulness and correctness; held-out source responses omitted",
    rubric: {
      PASS: "Direct, intelligible, relevant, coherent, and materially useful for the latest user turn. Minor omissions are allowed.",
      BORDERLINE: "Understandable and relevant, but substantially incomplete, awkward, circular, or weakly useful.",
      FAIL: "Wrong or nonresponsive in a consequential way, gibberish, contradiction, empty output, instruction failure, or degeneration.",
      selection: "Prefer the candidate that most often gives a natural, correctly contingent, useful response. Do not reward length, jargon, or confident invention.",
    },
    reference_blinded: true,
    candidate_labels: cli.candidates.map((candidate) => candidate.blindLabel),
    cases: panel.map((row, index) => ({
      index: index + 1,
      id: row.id,
      source: row.source,
      messages: row.messages,
      candidates: Object.fromEntries(candidateData.map(({ candidate, byId }) => [
        candidate.blindLabel,
        byId.get(row.id)?.text ?? "",
      ])),
    })),
  };
  const packetText = `${JSON.stringify(packet, null, 2)}\n`;
  await writeFile(cli.packet, packetText, { encoding: "utf8", flag: "wx" });

  const manifest = {
    schema: "alpha-chat-checkpoint-comparison-manifest-v1",
    status: "READY",
    reference_blinded_packet: {
      path: cli.packet,
      sha256: sha256(packetText),
      cases: panel.length,
      candidates: cli.candidates.length,
    },
    panel: { path: cli.panel, sha256: sha256(panelText), rows: panel.length },
    mapping: candidateData.map(({ candidate, text, rows }) => ({
      blind_label: candidate.blindLabel,
      actual_label: candidate.actualLabel,
      results: { path: candidate.path, sha256: sha256(text), rows: rows.length },
    })),
  };
  await writeFile(cli.manifest, `${JSON.stringify(manifest, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
  console.log(`comparison packet=PASS cases=${panel.length} candidates=${cli.candidates.length} out=${cli.packet}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
