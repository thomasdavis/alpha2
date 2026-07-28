import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { afterEach, describe, expect, it } from "vitest";

const execFileAsync = promisify(execFile);
const repoRoot = fileURLToPath(new URL("../../..", import.meta.url));
const preparer = join(repoRoot, "scripts/prepare_frozen_chat_semantic_review.ts");
const finalizer = join(repoRoot, "scripts/finalize_frozen_chat_semantic_review.ts");
const temporaryDirectories: string[] = [];

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map((dir) => rm(dir, { recursive: true, force: true })));
});

describe("frozen chat semantic review packet", () => {
  it("binds all 100 final cases and blinds held-out references", async () => {
    const dir = await mkdtemp(join(tmpdir(), "alpha-semantic-review-"));
    temporaryDirectories.push(dir);
    const promptsPath = join(dir, "chat-prompts.jsonl");
    const resultsPath = join(dir, "chat-results.jsonl");
    const summaryPath = join(dir, "summary.json");
    const manifestPath = join(dir, "MANIFEST.json");
    const outPath = join(dir, "review.json");
    const prompts = Array.from({ length: 100 }, (_, index) => ({
      id: `chat-${index}`,
      source: "synthetic",
      messages: [{ role: "user", content: `Question ${index}?` }],
      reference: `SECRET_REFERENCE_${index}`,
      prompt_tokens: 4,
    }));
    const results = Array.from({ length: 100 }, (_, index) => ({
      id: `chat-${index}`,
      text: `Answer ${index}.`,
      eosTerminated: true,
      roleLeak: false,
      nonempty: true,
      fourGramRepeatRate: 0,
      degenerateLoop: false,
      structuralPass: true,
    }));
    const promptText = prompts.map((row) => JSON.stringify(row)).join("\n") + "\n";
    const resultText = results.map((row) => JSON.stringify(row)).join("\n") + "\n";
    const checkpointSha = "c".repeat(64);
    await Promise.all([
      writeFile(promptsPath, promptText),
      writeFile(resultsPath, resultText),
      writeFile(summaryPath, JSON.stringify({
        schema: "alpha-frozen-eval-results-v2",
        checkpoint: { step: 30_322, sha256: checkpointSha },
        inputs: { chat: { rows: 100, sha256: sha256(promptText) } },
        outputs: { chat: { filename: "chat-results.jsonl", rows: 100, sha256: sha256(resultText) } },
      }) + "\n"),
      writeFile(manifestPath, JSON.stringify({
        schema: "alpha-frozen-eval-v1",
        status: "final",
        final: { chat: { rows: 100, sha256: sha256(promptText) } },
      }) + "\n"),
    ]);

    await execFileAsync("npx", ["tsx", preparer,
      "--prompts", promptsPath,
      "--results", resultsPath,
      "--summary", summaryPath,
      "--manifest", manifestPath,
      "--out", outPath,
    ], { cwd: repoRoot });
    const packetText = await readFile(outPath, "utf8");
    const packet = JSON.parse(packetText);
    expect(packet).toMatchObject({
      schema: "alpha-frozen-chat-semantic-review-packet-v1",
      status: "PENDING_HUMAN_REVIEW",
      reference_blinded: true,
      provenance: { checkpoint: { step: 30_322, sha256: checkpointSha } },
    });
    expect(packet.cases).toHaveLength(100);
    expect(packet.cases[99]).toMatchObject({
      index: 100,
      id: "chat-99",
      model_response: "Answer 99.",
      human_verdict: "PENDING",
    });
    expect(packetText).not.toContain("SECRET_REFERENCE_");

    packet.status = "COMPLETE";
    packet.reviewer = "Synthetic human reviewer";
    packet.reviewed_utc = "2026-07-28T00:00:00Z";
    packet.overall_rationale = "The synthetic suite is intelligible and relevant under the predeclared rubric.";
    packet.cases.forEach((row: Record<string, unknown>, index: number) => {
      row.human_verdict = index < 80 ? "PASS" : "BORDERLINE";
      row.human_rationale = index < 80 ? "Direct and intelligible." : "Understandable but substantially incomplete.";
    });
    await writeFile(outPath, JSON.stringify(packet, null, 2) + "\n");
    const reportPath = join(dir, "semantic-report.json");
    await execFileAsync("npx", ["tsx", finalizer, "--review", outPath, "--out", reportPath], { cwd: repoRoot });
    const report = JSON.parse(await readFile(reportPath, "utf8"));
    expect(report).toMatchObject({
      schema: "alpha-frozen-chat-semantic-review-v1",
      result: "PASS",
      reference_blinded: true,
      counts: { total: 100, PASS: 80, BORDERLINE: 20, FAIL: 0 },
    });

    const failedReview = structuredClone(packet);
    failedReview.cases[79].human_verdict = "FAIL";
    failedReview.cases[79].human_rationale = "Synthetic gibberish failure.";
    const failedReviewPath = join(dir, "failed-review.json");
    const failedReportPath = join(dir, "failed-report.json");
    await writeFile(failedReviewPath, JSON.stringify(failedReview, null, 2) + "\n");
    await expect(execFileAsync("npx", ["tsx", finalizer,
      "--review", failedReviewPath, "--out", failedReportPath,
    ], { cwd: repoRoot })).rejects.toBeDefined();
    expect(JSON.parse(await readFile(failedReportPath, "utf8"))).toMatchObject({
      result: "FAIL",
      counts: { PASS: 79, BORDERLINE: 20, FAIL: 1 },
      fail_ids: ["chat-79"],
    });
  }, 120_000);

  it("rejects result rows whose case order differs from the frozen prompts", async () => {
    const dir = await mkdtemp(join(tmpdir(), "alpha-semantic-review-order-"));
    temporaryDirectories.push(dir);
    const promptsPath = join(dir, "chat-prompts.jsonl");
    const resultsPath = join(dir, "chat-results.jsonl");
    const summaryPath = join(dir, "summary.json");
    const manifestPath = join(dir, "MANIFEST.json");
    const prompts = Array.from({ length: 100 }, (_, index) => ({
      id: `chat-${index}`,
      source: "synthetic",
      messages: [{ role: "user", content: `Question ${index}?` }],
      reference: `Reference ${index}`,
      prompt_tokens: 4,
    }));
    const results = Array.from({ length: 100 }, (_, index) => ({
      id: `chat-${(index + 1) % 100}`,
      text: `Answer ${index}.`,
      eosTerminated: true,
      roleLeak: false,
      nonempty: true,
      fourGramRepeatRate: 0,
      degenerateLoop: false,
      structuralPass: true,
    }));
    const promptText = prompts.map((row) => JSON.stringify(row)).join("\n") + "\n";
    const resultText = results.map((row) => JSON.stringify(row)).join("\n") + "\n";
    await Promise.all([
      writeFile(promptsPath, promptText),
      writeFile(resultsPath, resultText),
      writeFile(summaryPath, JSON.stringify({
        schema: "alpha-frozen-eval-results-v2",
        checkpoint: { step: 30_322, sha256: "d".repeat(64) },
        inputs: { chat: { rows: 100, sha256: sha256(promptText) } },
        outputs: { chat: { filename: "chat-results.jsonl", rows: 100, sha256: sha256(resultText) } },
      }) + "\n"),
      writeFile(manifestPath, JSON.stringify({
        schema: "alpha-frozen-eval-v1",
        status: "final",
        final: { chat: { rows: 100, sha256: sha256(promptText) } },
      }) + "\n"),
    ]);

    await expect(execFileAsync("npx", ["tsx", preparer,
      "--prompts", promptsPath,
      "--results", resultsPath,
      "--summary", summaryPath,
      "--manifest", manifestPath,
      "--out", join(dir, "rejected.json"),
    ], { cwd: repoRoot })).rejects.toMatchObject({ stderr: expect.stringContaining("case order") });
  }, 120_000);
});
