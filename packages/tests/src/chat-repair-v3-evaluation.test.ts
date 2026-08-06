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
const analyzer = join(repoRoot, "scripts/analyze_chat_repair_v3_pair.ts");
const temporaryDirectories: string[] = [];

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function makeRows(count: number, loops: number) {
  return Array.from({ length: count }, (_, index) => ({
    schema: "alpha-chat-repair-v3-eval-row-v1",
    id: `case-${index.toString().padStart(3, "0")}`,
    source: index % 2 === 0 ? "a" : "b",
    nonempty: true,
    eosTerminated: true,
    roleLeak: false,
    structuralPass: true,
    degenerateLoop: index < loops,
    fourGramRepeatRate: index < loops ? 0.3 : 0,
  }));
}

async function writeEvaluation(
  root: string,
  arm: "I0" | "C0" | "U1",
  step: number,
  contractHash: string,
  freshLoops: number,
): Promise<string> {
  await Promise.all([mkdir(join(root, "fresh96"), { recursive: true }), mkdir(join(root, "regression69"), { recursive: true })]);
  const freshText = `${makeRows(96, freshLoops).map((row) => JSON.stringify(row)).join("\n")}\n`;
  const regressionText = `${makeRows(69, Math.min(freshLoops, 6)).map((row) => JSON.stringify(row)).join("\n")}\n`;
  await Promise.all([
    writeFile(join(root, "fresh96/chat-results.jsonl"), freshText),
    writeFile(join(root, "regression69/chat-results.jsonl"), regressionText),
  ]);
  const manifest = {
    schema: "alpha-chat-repair-v3-checkpoint-evaluation-v1",
    status: "machine-development-complete; human-panel-pending; sealed-final-untouched",
    identity: {
      arm,
      checkpoint: { step, sha256: arm.toLocaleLowerCase().padEnd(64, "0") },
      evaluationContract: { sha256: contractHash },
      evaluatorCommit: "a".repeat(40),
      trainingSourceCommit: arm === "I0" ? null : "b".repeat(40),
    },
    suites: {
      fresh96: { input: { sha256: "1".repeat(64) } },
      regression69: { input: { sha256: "2".repeat(64) } },
    },
    sealedFinal: { executed: false, inspected: false },
    artifacts: [
      { path: "fresh96/chat-results.jsonl", sha256: sha256(freshText) },
      { path: "regression69/chat-results.jsonl", sha256: sha256(regressionText) },
    ],
  };
  const path = join(root, "evaluation-manifest.json");
  await writeFile(path, `${JSON.stringify(manifest, null, 2)}\n`);
  return path;
}

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map((dir) => rm(dir, { recursive: true, force: true })));
});

describe("chat-repair-v3 paired development analyzer", () => {
  it("admits a mechanical loop improvement but remains human-review pending and rejects artifact drift", async () => {
    const root = await mkdtemp(join(tmpdir(), "alpha-v3-eval-"));
    temporaryDirectories.push(root);
    const contractPath = join(root, "evaluation-contract.json");
    const contract = {
      schema: "alpha-chat-repair-v3-evaluation-contract-v1",
      suites: {
        fresh96: { sha256: "1".repeat(64) },
        regression69: { sha256: "2".repeat(64) },
      },
    };
    const contractText = `${JSON.stringify(contract, null, 2)}\n`;
    await writeFile(contractPath, contractText);
    const contractHash = sha256(contractText);
    const [initial, control, unlikelihood] = await Promise.all([
      writeEvaluation(join(root, "I0"), "I0", 1200, contractHash, 8),
      writeEvaluation(join(root, "C0"), "C0", 50, contractHash, 10),
      writeEvaluation(join(root, "U1"), "U1", 50, contractHash, 5),
    ]);
    const out = join(root, "analysis.json");
    await execFileAsync("npx", ["tsx", analyzer,
      "--initial", initial, "--control", control, "--unlikelihood", unlikelihood,
      "--evaluation-contract", contractPath, "--out", out,
    ], { cwd: repoRoot });
    const report = JSON.parse(await readFile(out, "utf8"));
    expect(report).toMatchObject({
      result: "MECHANICAL_PASS_HUMAN_PENDING",
      selection: { candidateSelected: false, lossUsed: false },
      sealedFinal: { executed: false, inspected: false },
      qualitative24: { status: "PENDING_BLINDED_HUMAN_COMPARISON" },
    });
    expect(report.fresh96.primary.fixedLoopIds).toHaveLength(5);
    expect(report.fresh96.primary.newLoopIds).toHaveLength(0);

    await writeFile(join(root, "U1/fresh96/chat-results.jsonl"), "corrupted\n");
    await expect(execFileAsync("npx", ["tsx", analyzer,
      "--initial", initial, "--control", control, "--unlikelihood", unlikelihood,
      "--evaluation-contract", contractPath, "--out", join(root, "rejected.json"),
    ], { cwd: repoRoot })).rejects.toMatchObject({ stderr: expect.stringContaining("artifact hash drift") });
  }, 120_000);
});
