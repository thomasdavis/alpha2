#!/usr/bin/env npx tsx
/** Execute one immutable development-only chat-repair-v3 checkpoint evaluation. */

import { createHash } from "node:crypto";
import { createReadStream, createWriteStream } from "node:fs";
import { mkdir, readFile, readdir, rename, stat, writeFile } from "node:fs/promises";
import { basename, dirname, join, relative, resolve } from "node:path";
import { spawn } from "node:child_process";
import { Effect } from "effect";
import { FileCheckpoint } from "@alpha/train";

interface FileContract {
  path: string;
  sha256: string;
  bytes: number;
  rows?: number;
}

interface EvaluationContract {
  schema: string;
  status: string;
  suites: { fresh96: FileContract; qualitative24: FileContract; regression69: FileContract };
  inputs: { freeze_manifest: FileContract; initial_checkpoint: FileContract & { step: number } };
  generation: { max_new_tokens: number; context_tokens: number; source_reference_is_never_model_visible: boolean };
  candidate_contract: {
    arms: string[];
    declared_steps: number[];
    run_contract_schema: string;
    initial_checkpoint_sha256: string;
    required_model_config: Record<string, number>;
  };
  evaluation: { sealed_final_allowed: boolean; loss_cannot_select: boolean };
  exclusions: { v2_sealed_final_sha256: string; older_frozen_final_sha256: string };
}

function parseArgs(): { values: Record<string, string>; flags: Set<string> } {
  const values: Record<string, string> = {};
  const flags = new Set<string>();
  for (let index = 2; index < process.argv.length; index++) {
    const key = process.argv[index];
    if (!key?.startsWith("--")) throw new Error(`unexpected argument: ${key ?? ""}`);
    const name = key.slice(2);
    const next = process.argv[index + 1];
    if (!next || next.startsWith("--")) flags.add(name);
    else {
      values[name] = next;
      index++;
    }
  }
  return { values, flags };
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

async function sha256File(path: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk as Buffer);
  return hash.digest("hex");
}

async function assertFile(path: string, expected: FileContract, label: string): Promise<void> {
  const metadata = await stat(path);
  assert(metadata.isFile() && metadata.size > 0, `${label} is missing or empty: ${path}`);
  assert(metadata.size === expected.bytes, `${label} byte count drift: ${metadata.size} != ${expected.bytes}`);
  const hash = await sha256File(path);
  assert(hash === expected.sha256, `${label} SHA-256 drift: ${hash} != ${expected.sha256}`);
}

async function atomicJson(path: string, value: unknown): Promise<void> {
  const temporary = `${path}.tmp`;
  await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
  await rename(temporary, path);
}

async function git(...args: string[]): Promise<string> {
  const { execFile } = await import("node:child_process");
  const { promisify } = await import("node:util");
  return (await promisify(execFile)("git", args, { encoding: "utf8" })).stdout.trim();
}

async function runLogged(command: string, args: string[], logPath: string, cwd: string): Promise<void> {
  await new Promise<void>((resolvePromise, reject) => {
    const log = createWriteStream(logPath, { flags: "a" });
    log.write(`${new Date().toISOString()} command=${JSON.stringify([command, ...args])}\n`);
    const child = spawn(command, args, { cwd, env: process.env, stdio: ["ignore", "pipe", "pipe"] });
    child.stdout.on("data", (chunk) => {
      process.stdout.write(chunk);
      log.write(chunk);
    });
    child.stderr.on("data", (chunk) => {
      process.stderr.write(chunk);
      log.write(chunk);
    });
    child.on("error", (error) => {
      log.end();
      reject(error);
    });
    child.on("close", (code, signal) => {
      log.end();
      if (code === 0) resolvePromise();
      else reject(new Error(`${command} exited code=${code ?? "null"} signal=${signal ?? "none"}; see ${logPath}`));
    });
  });
}

async function listFiles(root: string, directory = root): Promise<string[]> {
  const found: string[] = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) found.push(...await listFiles(root, path));
    else if (entry.isFile() && !entry.name.endsWith(".tmp") && entry.name !== "evaluation-manifest.json") found.push(path);
  }
  return found.sort((left, right) => relative(root, left).localeCompare(relative(root, right)));
}

async function main(): Promise<void> {
  const { values: args, flags } = parseArgs();
  for (const key of ["arm", "checkpoint", "evaluation-contract", "freeze-manifest", "fresh-prompts", "panel", "regression-prompts", "out-dir"] as const) {
    if (!args[key]) throw new Error(`required: --${key}`);
  }
  const repo = resolve(args.repo ?? process.cwd());
  const checkpointPath = resolve(args.checkpoint);
  const evaluationContractPath = resolve(args["evaluation-contract"]);
  const freezeManifestPath = resolve(args["freeze-manifest"]);
  const freshPath = resolve(args["fresh-prompts"]);
  const panelPath = resolve(args.panel);
  const regressionPath = resolve(args["regression-prompts"]);
  const outDir = resolve(args["out-dir"]);
  const python = resolve(args.python ?? "/mnt/donto-data/alpha-corpora/.venv/bin/python");
  const cli = join(repo, "apps/cli/dist/main.js");
  const resume = flags.has("resume");
  const allowCpuSmoke = flags.has("allow-cpu-smoke");
  const batchSize = Number(args["batch-size"] ?? "32");
  assert(Number.isSafeInteger(batchSize) && batchSize > 0, "batch-size must be a positive integer");
  await stat(cli);
  await stat(python);

  const [contractText, checkpointHash] = await Promise.all([
    readFile(evaluationContractPath, "utf8"),
    sha256File(checkpointPath),
  ]);
  const contract = JSON.parse(contractText) as EvaluationContract;
  assert(contract.schema === "alpha-chat-repair-v3-evaluation-contract-v1", "unexpected evaluation contract schema");
  assert(contract.status === "development-only-frozen; sealed-finals-excluded", "evaluation contract is not frozen");
  assert(contract.generation.max_new_tokens === 128 && contract.generation.context_tokens === 512, "generation contract drift");
  assert(contract.generation.source_reference_is_never_model_visible === true, "reference-blinding contract is absent");
  assert(contract.evaluation.sealed_final_allowed === false && contract.evaluation.loss_cannot_select === true,
    "development/firewall contract drift");
  await Promise.all([
    assertFile(freshPath, contract.suites.fresh96, "fresh96 suite"),
    assertFile(panelPath, contract.suites.qualitative24, "qualitative24 panel"),
    assertFile(regressionPath, contract.suites.regression69, "regression69 suite"),
    assertFile(freezeManifestPath, contract.inputs.freeze_manifest, "v3 freeze manifest"),
  ]);
  for (const suite of [contract.suites.fresh96, contract.suites.qualitative24, contract.suites.regression69]) {
    assert(![contract.exclusions.v2_sealed_final_sha256, contract.exclusions.older_frozen_final_sha256].includes(suite.sha256),
      "a sealed or previously frozen final was passed as a development suite");
  }

  const checkpoint = await Effect.runPromise(new FileCheckpoint().load(checkpointPath));
  for (const [key, expected] of Object.entries(contract.candidate_contract.required_model_config)) {
    assert((checkpoint.modelConfig as any)[key] === expected, `checkpoint modelConfig.${key} drift`);
  }
  const arm = args.arm;
  let runContractPath: string | null = null;
  let runContractHash: string | null = null;
  let runContract: any = null;
  if (arm === "I0") {
    assert(checkpointHash === contract.candidate_contract.initial_checkpoint_sha256, "I0 is not the immutable initial checkpoint");
    assert(checkpoint.step === contract.inputs.initial_checkpoint.step, "I0 checkpoint step drift");
    assert(!args["run-contract"], "I0 must not be supplied a candidate run contract");
  } else {
    assert(contract.candidate_contract.arms.includes(arm), `arm ${arm} is not declared`);
    assert(contract.candidate_contract.declared_steps.includes(checkpoint.step), `checkpoint step ${checkpoint.step} is undeclared`);
    assert(basename(checkpointPath) === `checkpoint-${checkpoint.step}.json`, "checkpoint filename does not bind its step");
    assert(args["run-contract"], `${arm} requires --run-contract`);
    runContractPath = resolve(args["run-contract"]);
    assert(dirname(runContractPath) === dirname(checkpointPath), "checkpoint and run contract must share a directory");
    runContractHash = await sha256File(runContractPath);
    runContract = JSON.parse(await readFile(runContractPath, "utf8"));
    assert(runContract.schema === contract.candidate_contract.run_contract_schema, "unexpected candidate run-contract schema");
    assert(runContract.arm === arm, "candidate run-contract arm mismatch");
    assert(runContract.initializedFrom?.sha256 === contract.candidate_contract.initial_checkpoint_sha256, "candidate initialization drift");
    assert(runContract.inputs?.freezeManifest?.sha256 === contract.inputs.freeze_manifest.sha256, "candidate freeze-manifest drift");
    assert(runContract.selection?.developmentSelector?.sha256 === contract.suites.fresh96.sha256, "candidate selector drift");
    assert(runContract.selection?.qualitativePanel?.sha256 === contract.suites.qualitative24.sha256, "candidate panel drift");
    assert(runContract.selection?.priorV2Regression === "eligible-69 only", "candidate v2 regression contract drift");
    assert(runContract.selection?.sealedFinalRemainsClosed === true, "candidate run contract permits sealed-final access");
  }

  const dirty = await git("status", "--porcelain");
  assert(dirty.length === 0, "checkpoint evaluation requires a clean committed worktree");
  const evaluatorCommit = await git("rev-parse", "HEAD");
  const contractHash = await sha256File(evaluationContractPath);
  if (!resume) await mkdir(outDir, { recursive: false });
  else await stat(outDir);
  const statePath = join(outDir, "evaluation-state-started.json");
  const identity = {
    schema: "alpha-chat-repair-v3-evaluation-state-v1",
    status: "running",
    arm,
    checkpoint: { path: checkpointPath, sha256: checkpointHash, step: checkpoint.step },
    evaluationContract: { path: evaluationContractPath, sha256: contractHash },
    runContract: runContractPath ? { path: runContractPath, sha256: runContractHash } : null,
    evaluatorCommit,
  };
  if (resume) {
    const existing = JSON.parse(await readFile(statePath, "utf8"));
    assert(JSON.stringify(existing) === JSON.stringify(identity),
      "resume evaluation identity differs from existing state");
  } else {
    await atomicJson(statePath, identity);
  }

  const exportDir = join(outDir, "hf-export");
  if (!(await stat(join(exportDir, "model.safetensors")).catch(() => null))) {
    await runLogged(process.execPath, [cli, "export-hf", `--checkpoint=${checkpointPath}`, `--out=${exportDir}`],
      join(outDir, "01-export-hf.log"), repo);
  }
  const modelHash = await sha256File(join(exportDir, "model.safetensors"));
  const alphaLogits = join(outDir, "alpha-logits-hello.json");
  if (!(await stat(alphaLogits).catch(() => null))) {
    await runLogged(process.execPath, [cli, "logits", `--checkpoint=${checkpointPath}`, "--prompt=Hello", "--json", `--out=${alphaLogits}`],
      join(outDir, "02-alpha-logits.log"), repo);
  }
  const parityPath = join(outDir, "hf-export-parity.json");
  if (!(await stat(parityPath).catch(() => null))) {
    await runLogged(python, [
      join(repo, "scripts/verify_hf_export.py"), `--export-dir=${exportDir}`, `--alpha-logits=${alphaLogits}`,
      "--tol=0.001", `--json-out=${parityPath}`,
    ], join(outDir, "03-hf-export-parity.log"), repo);
  }
  const parity = JSON.parse(await readFile(parityPath, "utf8"));
  assert(parity.status === "PASS", "HF export parity did not pass");
  assert(parity.export?.model_sha256 === modelHash, "HF parity report model hash drift");

  const suites = [
    { label: "fresh96", path: freshPath, contract: contract.suites.fresh96 },
    { label: "regression69", path: regressionPath, contract: contract.suites.regression69 },
  ];
  for (const suite of suites) {
    const suiteDir = join(outDir, suite.label);
    const summaryPath = join(suiteDir, "summary.json");
    if (!(await stat(summaryPath).catch(() => null))) {
      const generationArgs = [
        join(repo, "scripts/generate_chat_repair_v3_eval_hf.py"),
        `--export-dir=${exportDir}`, `--native-checkpoint=${checkpointPath}`,
        `--expected-checkpoint-sha256=${checkpointHash}`, `--checkpoint-step=${checkpoint.step}`,
        `--prompts=${suite.path}`, `--expected-prompts-sha256=${suite.contract.sha256}`,
        `--expected-rows=${suite.contract.rows}`, `--out-dir=${suiteDir}`,
        `--batch-size=${batchSize}`, `--max-tokens=${contract.generation.max_new_tokens}`,
      ];
      if (await stat(join(suiteDir, "chat-results.jsonl")).catch(() => null)) generationArgs.push("--resume");
      if (allowCpuSmoke) generationArgs.push("--allow-cpu-smoke");
      await runLogged(python, generationArgs, join(outDir, `04-${suite.label}.log`), repo);
    }
    const auditPath = join(suiteDir, "stratified-audit.json");
    if (!(await stat(auditPath).catch(() => null))) {
      await runLogged("npx", ["tsx", join(repo, "scripts/audit_frozen_chat_failures.ts"),
        "--prompts", suite.path, "--results", join(suiteDir, "chat-results.jsonl"),
        "--summary", summaryPath, "--out", auditPath], join(outDir, `05-${suite.label}-audit.log`), repo);
    }
  }

  const panelRender = join(outDir, "qualitative-panel.md");
  if (!(await stat(panelRender).catch(() => null))) {
    await runLogged("npx", ["tsx", join(repo, "scripts/render_chat_repair_panel.ts"),
      "--panel", panelPath, "--results", join(outDir, "fresh96/chat-results.jsonl"),
      "--out", panelRender, "--title", `Alpha chat repair v3 — ${arm} step ${checkpoint.step} frozen qualitative panel`],
    join(outDir, "06-render-panel.log"), repo);
  }

  const [freshSummary, regressionSummary] = await Promise.all([
    readFile(join(outDir, "fresh96/summary.json"), "utf8").then(JSON.parse),
    readFile(join(outDir, "regression69/summary.json"), "utf8").then(JSON.parse),
  ]);
  assert(freshSummary.checkpoint.sha256 === checkpointHash && regressionSummary.checkpoint.sha256 === checkpointHash,
    "evaluation summary checkpoint identity drift");
  assert(freshSummary.export.modelSha256 === modelHash && regressionSummary.export.modelSha256 === modelHash,
    "evaluation summary export identity drift");
  const artifactEntries = await Promise.all((await listFiles(outDir)).map(async (path) => ({
    path: relative(outDir, path), sha256: await sha256File(path), bytes: (await stat(path)).size,
  })));
  const manifest = {
    schema: "alpha-chat-repair-v3-checkpoint-evaluation-v1",
    status: "machine-development-complete; human-panel-pending; sealed-final-untouched",
    completedUtc: new Date().toISOString(),
    identity,
    export: { modelSha256: modelHash, parity: { path: relative(outDir, parityPath), sha256: await sha256File(parityPath), status: "PASS" } },
    suites: {
      fresh96: { input: contract.suites.fresh96, metrics: freshSummary.chat },
      regression69: { input: contract.suites.regression69, metrics: regressionSummary.chat },
      qualitative24: { input: contract.suites.qualitative24, render: relative(outDir, panelRender), humanVerdict: "PENDING" },
    },
    selection: { automaticDecision: "NOT_COMPUTED_IN_SINGLE-CHECKPOINT_RUN", lossUsed: false, bgeUsed: false },
    sealedFinal: { executed: false, inspected: false },
    artifacts: artifactEntries,
  };
  await atomicJson(join(outDir, "evaluation-manifest.json"), manifest);
  process.stdout.write(`${JSON.stringify({ outDir, arm, step: checkpoint.step, fresh96: freshSummary.chat, regression69: regressionSummary.chat }, null, 2)}\n`);
}

await main();
