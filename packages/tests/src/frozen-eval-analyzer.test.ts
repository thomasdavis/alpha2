import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { afterEach, describe, expect, it } from "vitest";

const execFileAsync = promisify(execFile);
const repoRoot = fileURLToPath(new URL("../../..", import.meta.url));
const analyzer = join(repoRoot, "scripts/analyze_frozen_eval_pair.ts");
const temporaryDirectories: string[] = [];

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

async function writeEvalRun(dir: string, checkpointStep: number, chatInputSha: string, qaInputSha: string): Promise<void> {
  await mkdir(dir, { recursive: true });
  const chat = Array.from({ length: 100 }, (_, index) => ({
    id: `chat-${index}`,
    generatedIds: [42, 2],
    text: "hello",
    eosTerminated: true,
    roleLeak: false,
    nonempty: true,
    fourGramRepeatRate: 0,
    degenerateLoop: false,
    structuralPass: true,
  }));
  const qa = Array.from({ length: 200 }, (_, index) => ({
    id: `qa-${index}`,
    expected: "answer",
    text: "answer",
    normalizedPrediction: "answer",
    normalizedExpected: "answer",
    exactMatch: true,
    answerContained: true,
    tokenF1: 1,
  }));
  const chatText = chat.map((row) => JSON.stringify(row)).join("\n") + "\n";
  const qaText = qa.map((row) => JSON.stringify(row)).join("\n") + "\n";
  const summary = {
    schema: "alpha-frozen-eval-results-v2",
    checkpoint: {
      path: join(dir, `checkpoint-${checkpointStep}.json`),
      sha256: (checkpointStep === 61_036 ? "a" : "b").repeat(64),
      step: checkpointStep,
      modelConfig: { nLayer: 16, nEmbd: 512 },
    },
    inputs: {
      chat: { path: "/frozen/chat.jsonl", sha256: chatInputSha, rows: 100 },
      qa: { path: "/frozen/qa.jsonl", sha256: qaInputSha, rows: 200 },
    },
    outputs: {
      chat: { filename: "chat-results.jsonl", sha256: sha256(chatText), rows: 100 },
      qa: { filename: "qa-results.jsonl", sha256: sha256(qaText), rows: 200 },
    },
    generation: { chatMaxTokens: 128, qaMaxTokens: 64, eosId: 2, userId: 1 },
    chat: {
      total: 100,
      structuralPass: 100,
      eosTerminated: 100,
      roleLeaks: 0,
      nonempty: 100,
      degenerateLoops: 0,
      meanFourGramRepeatRate: 0,
      maxFourGramRepeatRate: 0,
    },
    closedBookQa: { total: 200, exactMatch: 200, answerContained: 200, meanTokenF1: 1 },
  };
  await Promise.all([
    writeFile(join(dir, "summary.json"), JSON.stringify(summary, null, 2) + "\n"),
    writeFile(join(dir, "chat-results.jsonl"), chatText),
    writeFile(join(dir, "qa-results.jsonl"), qaText),
  ]);
}

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map((dir) => rm(dir, { recursive: true, force: true })));
});

describe("frozen-eval pair analyzer", () => {
  it("binds both runs to the final manifest and rejects a substituted frozen input", async () => {
    const dir = await mkdtemp(join(tmpdir(), "alpha-frozen-analyzer-"));
    temporaryDirectories.push(dir);
    const baseDir = join(dir, "base");
    const chatDir = join(dir, "chat");
    const manifestPath = join(dir, "MANIFEST.json");
    const outPath = join(dir, "analysis.json");
    const chatInputSha = sha256("canonical frozen chat");
    const qaInputSha = sha256("canonical frozen QA");
    await Promise.all([
      writeEvalRun(baseDir, 61_036, chatInputSha, qaInputSha),
      writeEvalRun(chatDir, 30_322, chatInputSha, qaInputSha),
    ]);
    const manifest = {
      schema: "alpha-frozen-eval-v1",
      status: "final",
      final: {
        chat: { rows: 100, sha256: chatInputSha },
        closed_book_qa: { rows: 200, sha256: qaInputSha },
      },
    };
    await writeFile(manifestPath, JSON.stringify(manifest, null, 2) + "\n");

    await execFileAsync("npx", ["tsx", analyzer, "--base", baseDir, "--chat", chatDir,
      "--manifest", manifestPath, "--out", outPath], { cwd: repoRoot });
    const report = JSON.parse(await readFile(outPath, "utf8"));
    expect(report.result).toBe("PASS");
    expect(report.inputs_match).toBe(true);
    expect(report.frozen_manifest).toMatchObject({ path: manifestPath, schema: "alpha-frozen-eval-v1", status: "final" });

    const substitutedManifest = join(dir, "MANIFEST-substituted.json");
    await writeFile(substitutedManifest, JSON.stringify({
      ...manifest,
      final: { ...manifest.final, chat: { rows: 100, sha256: "0".repeat(64) } },
    }, null, 2) + "\n");
    await expect(execFileAsync("npx", ["tsx", analyzer, "--base", baseDir, "--chat", chatDir,
      "--manifest", substitutedManifest, "--out", join(dir, "rejected.json")], { cwd: repoRoot }))
      .rejects.toMatchObject({ stderr: expect.stringContaining("chat input frozen-manifest SHA-256") });
  });
});
