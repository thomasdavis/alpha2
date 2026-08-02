#!/usr/bin/env npx tsx
/** Immutable free-generation evaluation for the public baseline or one v4/v5 checkpoint. */

import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { createReadStream, createWriteStream } from "node:fs";
import {
  mkdir,
  readFile,
  readdir,
  rename,
  stat,
  writeFile,
} from "node:fs/promises";
import { dirname, join, relative, resolve } from "node:path";
import { Effect } from "effect";
import { FileCheckpoint } from "@alpha/train";

interface FileEvidence {
  readonly path: string;
  readonly bytes: number;
  readonly sha256: string;
  readonly rows?: number;
}

interface Freeze {
  readonly schema: string;
  readonly status: string;
  readonly inputs: { readonly corpus_manifest: FileEvidence };
  readonly visible_development: {
    readonly selector: FileEvidence;
    readonly panel: FileEvidence;
    readonly regression: FileEvidence;
    readonly releaseProbes: FileEvidence;
  };
  readonly sealed_final: FileEvidence & { readonly execution_policy: string };
  readonly selection: {
    readonly validation_loss_can_select: boolean;
    readonly public_baseline_required: boolean;
  };
}

function parseArgs(): { values: Record<string, string>; flags: Set<string> } {
  const values: Record<string, string> = {};
  const flags = new Set<string>();
  for (let index = 2; index < process.argv.length; index += 1) {
    const key = process.argv[index];
    if (!key?.startsWith("--"))
      throw new Error(`unexpected argument: ${String(key)}`);
    const next = process.argv[index + 1];
    if (!next || next.startsWith("--")) flags.add(key.slice(2));
    else {
      values[key.slice(2)] = next;
      index += 1;
    }
  }
  return { values, flags };
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

async function sha256File(path: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path))
    hash.update(chunk as Buffer);
  return hash.digest("hex");
}

async function verifyFile(
  path: string,
  expected: FileEvidence,
  label: string,
): Promise<void> {
  const metadata = await stat(path);
  assert(
    metadata.isFile() && metadata.size === expected.bytes,
    `${label} byte count drift`,
  );
  assert(
    (await sha256File(path)) === expected.sha256,
    `${label} SHA-256 drift`,
  );
}

async function atomicJson(path: string, value: unknown): Promise<void> {
  const temporary = `${path}.tmp`;
  await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, {
    encoding: "utf8",
    flag: "wx",
  });
  await rename(temporary, path);
}

async function git(...args: string[]): Promise<string> {
  const { execFile } = await import("node:child_process");
  const { promisify } = await import("node:util");
  return (
    await promisify(execFile)("git", args, { encoding: "utf8" })
  ).stdout.trim();
}

async function runLogged(
  command: string,
  args: readonly string[],
  logPath: string,
  cwd: string,
): Promise<void> {
  await new Promise<void>((accept, reject) => {
    const log = createWriteStream(logPath, { flags: "a" });
    log.write(
      `${new Date().toISOString()} command=${JSON.stringify([command, ...args])}\n`,
    );
    const child = spawn(command, args, {
      cwd,
      env: process.env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    child.stdout.on("data", (chunk) => {
      process.stdout.write(chunk);
      log.write(chunk);
    });
    child.stderr.on("data", (chunk) => {
      process.stderr.write(chunk);
      log.write(chunk);
    });
    child.on("error", reject);
    child.on("close", (code, signal) => {
      log.end();
      code === 0
        ? accept()
        : reject(
            new Error(
              `${command} exited code=${String(code)} signal=${String(signal)}; see ${logPath}`,
            ),
          );
    });
  });
}

async function listFiles(root: string, directory = root): Promise<string[]> {
  const found: string[] = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) found.push(...(await listFiles(root, path)));
    else if (
      entry.isFile() &&
      !entry.name.endsWith(".tmp") &&
      entry.name !== "evaluation-manifest.json"
    )
      found.push(path);
  }
  return found.sort((left, right) =>
    relative(root, left).localeCompare(relative(root, right)),
  );
}

async function main(): Promise<void> {
  const { values: args, flags } = parseArgs();
  for (const required of [
    "checkpoint",
    "evaluation-freeze",
    "out-dir",
  ] as const)
    if (!args[required]) throw new Error(`required: --${required}`);
  const repo = resolve(args.repo ?? process.cwd());
  const checkpointPath = resolve(args.checkpoint);
  const freezePath = resolve(args["evaluation-freeze"]);
  const outDir = resolve(args["out-dir"]);
  const python = resolve(
    args.python ?? "/mnt/donto-data/alpha-corpora/.venv/bin/python",
  );
  const cli = join(repo, "apps/cli/dist/main.js");
  const resume = flags.has("resume");
  const allowCpuSmoke = flags.has("allow-cpu-smoke");
  const batchSize = Number(args["batch-size"] ?? "32");
  assert(
    Number.isSafeInteger(batchSize) && batchSize > 0,
    "invalid batch size",
  );
  await Promise.all([stat(cli), stat(python)]);

  const [freezeText, freezeHash, checkpointHash] = await Promise.all([
    readFile(freezePath, "utf8"),
    sha256File(freezePath),
    sha256File(checkpointPath),
  ]);
  const freeze = JSON.parse(freezeText) as Freeze;
  assert(
    freeze.schema === "alpha-chat-semantic-repair-v4-evaluation-freeze-v1",
    "unexpected evaluation freeze schema",
  );
  assert(
    freeze.status === "development-visible; inherited-final-sealed-unexecuted",
    "evaluation freeze is not selection-safe",
  );
  assert(
    freeze.selection.validation_loss_can_select === false &&
      freeze.selection.public_baseline_required === true,
    "selection policy drift",
  );
  const suiteEntries = Object.entries(freeze.visible_development) as Array<
    [keyof Freeze["visible_development"], FileEvidence]
  >;
  await Promise.all([
    ...suiteEntries.map(([label, contract]) =>
      verifyFile(contract.path, contract, label),
    ),
    verifyFile(
      freeze.sealed_final.path,
      freeze.sealed_final,
      "sealed-final identity",
    ),
  ]);
  assert(
    !suiteEntries.some(
      ([, contract]) => contract.sha256 === freeze.sealed_final.sha256,
    ),
    "sealed final was passed as a visible suite",
  );

  const checkpoint = await Effect.runPromise(
    new FileCheckpoint().load(checkpointPath),
  );
  for (const [key, expected] of Object.entries({
    blockSize: 512,
    nLayer: 16,
    nEmbd: 512,
    nHead: 8,
    vocabSize: 12288,
  }))
    assert(
      (checkpoint.modelConfig as Record<string, unknown>)[key] === expected,
      `checkpoint modelConfig.${key} drift`,
    );

  const publicSha =
    "399f776b49acc0c8834ff8a7f2390454e2c5f2d833a264e3f83ff546e973cfec";
  const cleanBaseSha =
    "08e14fa9604bf1b46ebcd5df37933c84d2496c1d05d9e4b32ebad98792cc6049";
  let label = checkpointHash === publicSha ? "I0" : "CANDIDATE";
  let runContract: Record<string, any> | null = null;
  let runContractPath: string | null = null;
  let runContractHash: string | null = null;
  if (label === "I0") {
    assert(
      !args["run-contract"],
      "public baseline must not receive a run contract",
    );
  } else {
    assert(args["run-contract"], "semantic-repair candidate requires --run-contract");
    runContractPath = resolve(args["run-contract"]);
    runContractHash = await sha256File(runContractPath);
    runContract = JSON.parse(await readFile(runContractPath, "utf8")) as Record<
      string,
      any
    >;
    assert(
      [
        "alpha-chat-semantic-repair-contract-v4",
        "alpha-chat-semantic-repair-contract-v5",
      ].includes(runContract.schema),
      "unexpected candidate run contract",
    );
    label = runContract.schema.endsWith("v5") ? "V5" : "V4";
    assert(
      runContract.eligibleForCheckpointSelection === true,
      "candidate is selection-ineligible",
    );
    assert(runContract.sourceTreeDirty === false, "candidate source was dirty");
    const expectedInitialization = label === "V5" ? cleanBaseSha : publicSha;
    assert(
      runContract.initializedFrom?.sha256 === expectedInitialization,
      "candidate initialization drift",
    );
    assert(
      runContract.inputs?.evaluationFreeze?.sha256 === freezeHash,
      "candidate used another evaluation freeze",
    );
    assert(
      runContract.selection?.sealedFinalRemainsClosedUntilSelection === true,
      "candidate contract permits premature sealed-final access",
    );
    assert(
      runContract.selection?.checkpoints?.includes(checkpoint.step),
      `checkpoint step ${checkpoint.step} was not declared selectable`,
    );
    assert(
      dirname(runContractPath) === dirname(checkpointPath),
      "candidate checkpoint and run contract must share a directory",
    );
  }

  assert(
    (await git("status", "--porcelain")).length === 0,
    "evaluation requires a clean worktree",
  );
  const identity = {
    schema: "alpha-chat-semantic-repair-evaluation-state-v1",
    status: "running",
    label,
    checkpoint: {
      path: checkpointPath,
      sha256: checkpointHash,
      step: checkpoint.step,
    },
    freeze: { path: freezePath, sha256: freezeHash },
    runContract: runContractPath
      ? { path: runContractPath, sha256: runContractHash }
      : null,
    evaluatorCommit: await git("rev-parse", "HEAD"),
  };
  if (flags.has("preflight-only")) {
    process.stdout.write(
      `${JSON.stringify({ result: "PASS", mode: "preflight-only", identity }, null, 2)}\n`,
    );
    return;
  }
  if (!resume) await mkdir(outDir, { recursive: false });
  else await stat(outDir);
  const statePath = join(outDir, "evaluation-state-started.json");
  if (resume)
    assert(
      JSON.stringify(JSON.parse(await readFile(statePath, "utf8"))) ===
        JSON.stringify(identity),
      "resume identity drift",
    );
  else await atomicJson(statePath, identity);

  const exportDir = join(outDir, "hf-export");
  if (!(await stat(join(exportDir, "model.safetensors")).catch(() => null)))
    await runLogged(
      process.execPath,
      [
        cli,
        "export-hf",
        `--checkpoint=${checkpointPath}`,
        `--out=${exportDir}`,
      ],
      join(outDir, "01-export-hf.log"),
      repo,
    );
  const modelHash = await sha256File(join(exportDir, "model.safetensors"));
  const logitsPath = join(outDir, "alpha-logits-hello.json");
  if (!(await stat(logitsPath).catch(() => null)))
    await runLogged(
      process.execPath,
      [
        cli,
        "logits",
        `--checkpoint=${checkpointPath}`,
        "--prompt=Hello",
        "--json",
        `--out=${logitsPath}`,
      ],
      join(outDir, "02-alpha-logits.log"),
      repo,
    );
  const parityPath = join(outDir, "hf-export-parity.json");
  if (!(await stat(parityPath).catch(() => null)))
    await runLogged(
      python,
      [
        join(repo, "scripts/verify_hf_export.py"),
        `--export-dir=${exportDir}`,
        `--alpha-logits=${logitsPath}`,
        "--tol=0.001",
        `--json-out=${parityPath}`,
      ],
      join(outDir, "03-hf-export-parity.log"),
      repo,
    );
  const parity = JSON.parse(await readFile(parityPath, "utf8"));
  assert(parity.status === "PASS", "HF export parity failed");
  assert(parity.export?.model_sha256 === modelHash, "parity model hash drift");

  const generatedSuites = suiteEntries.filter(([name]) => name !== "panel");
  for (const [name, suite] of generatedSuites) {
    assert(suite.rows && suite.rows > 0, `${name} row count missing`);
    const suiteDir = join(outDir, name);
    const summaryPath = join(suiteDir, "summary.json");
    if (!(await stat(summaryPath).catch(() => null))) {
      const generationArgs = [
        join(repo, "scripts/generate_chat_repair_v3_eval_hf.py"),
        `--export-dir=${exportDir}`,
        `--native-checkpoint=${checkpointPath}`,
        `--expected-checkpoint-sha256=${checkpointHash}`,
        `--checkpoint-step=${checkpoint.step}`,
        `--prompts=${suite.path}`,
        `--expected-prompts-sha256=${suite.sha256}`,
        `--expected-rows=${suite.rows}`,
        `--out-dir=${suiteDir}`,
        `--batch-size=${batchSize}`,
        "--max-tokens=128",
      ];
      if (await stat(join(suiteDir, "chat-results.jsonl")).catch(() => null))
        generationArgs.push("--resume");
      if (allowCpuSmoke) generationArgs.push("--allow-cpu-smoke");
      await runLogged(
        python,
        generationArgs,
        join(outDir, `04-${name}.log`),
        repo,
      );
    }
    const auditPath = join(suiteDir, "stratified-audit.json");
    if (!(await stat(auditPath).catch(() => null)))
      await runLogged(
        "npx",
        [
          "tsx",
          join(repo, "scripts/audit_frozen_chat_failures.ts"),
          "--prompts",
          suite.path,
          "--results",
          join(suiteDir, "chat-results.jsonl"),
          "--summary",
          summaryPath,
          "--out",
          auditPath,
        ],
        join(outDir, `05-${name}-audit.log`),
        repo,
      );
  }

  const panelRender = join(outDir, "qualitative-panel.md");
  if (!(await stat(panelRender).catch(() => null)))
    await runLogged(
      "npx",
      [
        "tsx",
        join(repo, "scripts/render_chat_repair_panel.ts"),
        "--panel",
        freeze.visible_development.panel.path,
        "--results",
        join(outDir, "selector/chat-results.jsonl"),
        "--out",
        panelRender,
        "--title",
        `Alpha semantic repair — ${label} step ${checkpoint.step}`,
      ],
      join(outDir, "06-render-panel.log"),
      repo,
    );

  const metrics = Object.fromEntries(
    await Promise.all(
      generatedSuites.map(async ([name]) => [
        name,
        JSON.parse(await readFile(join(outDir, name, "summary.json"), "utf8"))
          .chat,
      ]),
    ),
  );
  const artifacts = await Promise.all(
    (await listFiles(outDir)).map(async (path) => ({
      path: relative(outDir, path),
      sha256: await sha256File(path),
      bytes: (await stat(path)).size,
    })),
  );
  const manifest = {
    schema: "alpha-chat-semantic-repair-checkpoint-evaluation-v1",
    status:
      "machine-development-complete; semantic-human-review-pending; sealed-final-untouched",
    completedUtc: new Date().toISOString(),
    identity,
    export: { modelSha256: modelHash, parityStatus: "PASS" },
    suites: metrics,
    qualitativePanel: {
      path: relative(outDir, panelRender),
      humanVerdict: "PENDING",
    },
    selection: { automaticDecision: "NOT_COMPUTED", lossUsed: false },
    sealedFinal: { executed: false, inspected: false },
    artifacts,
  };
  await atomicJson(join(outDir, "evaluation-manifest.json"), manifest);
  process.stdout.write(
    `${JSON.stringify({ result: "PASS", outDir, label, step: checkpoint.step, metrics }, null, 2)}\n`,
  );
}

await main();
