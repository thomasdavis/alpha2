#!/usr/bin/env npx tsx
/** Compare I0, C0, and U1 on the immutable v3 development-only suites. */

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";

interface ResultRow {
  id: string;
  source: string;
  nonempty: boolean;
  eosTerminated: boolean;
  roleLeak: boolean;
  structuralPass: boolean;
  degenerateLoop: boolean;
  fourGramRepeatRate: number;
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

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
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

async function loadEvaluation(path: string, expectedArm: string): Promise<{
  path: string;
  text: string;
  manifest: any;
  root: string;
  fresh: ResultRow[];
  regression: ResultRow[];
}> {
  const absolute = resolve(path);
  const text = await readFile(absolute, "utf8");
  const manifest = JSON.parse(text);
  assert(manifest.schema === "alpha-chat-repair-v3-checkpoint-evaluation-v1", `${expectedArm} manifest schema drift`);
  assert(manifest.status === "machine-development-complete; human-panel-pending; sealed-final-untouched", `${expectedArm} evaluation is incomplete`);
  assert(manifest.identity?.arm === expectedArm, `${expectedArm} manifest arm drift`);
  assert(manifest.sealedFinal?.executed === false && manifest.sealedFinal?.inspected === false, `${expectedArm} touched a sealed final`);
  const root = dirname(absolute);
  const freshPath = join(root, "fresh96/chat-results.jsonl");
  const regressionPath = join(root, "regression69/chat-results.jsonl");
  const [freshText, regressionText] = await Promise.all([readFile(freshPath, "utf8"), readFile(regressionPath, "utf8")]);
  const artifacts = new Map<string, string>((manifest.artifacts ?? []).map((row: any) => [row.path, row.sha256]));
  assert(artifacts.get("fresh96/chat-results.jsonl") === sha256(freshText), `${expectedArm} fresh result artifact hash drift`);
  assert(artifacts.get("regression69/chat-results.jsonl") === sha256(regressionText), `${expectedArm} regression result artifact hash drift`);
  const fresh = parseJsonl<ResultRow>(freshText, freshPath);
  const regression = parseJsonl<ResultRow>(regressionText, regressionPath);
  assert(fresh.length === 96 && regression.length === 69, `${expectedArm} evaluation row counts drift`);
  return { path: absolute, text, manifest, root, fresh, regression };
}

function compareOrder(label: string, left: readonly ResultRow[], right: readonly ResultRow[]): void {
  assert(left.length === right.length, `${label} row counts differ`);
  for (let index = 0; index < left.length; index++) {
    assert(left[index].id === right[index].id && left[index].source === right[index].source,
      `${label} order differs at row ${index + 1}`);
  }
}

function metrics(rows: readonly ResultRow[]) {
  return {
    total: rows.length,
    structuralPass: rows.filter((row) => row.structuralPass).length,
    nonempty: rows.filter((row) => row.nonempty).length,
    eosTerminated: rows.filter((row) => row.eosTerminated).length,
    roleLeaks: rows.filter((row) => row.roleLeak).length,
    degenerateLoops: rows.filter((row) => row.degenerateLoop).length,
    meanFourGramRepeatRate: rows.reduce((sum, row) => sum + row.fourGramRepeatRate, 0) / Math.max(1, rows.length),
  };
}

function transitionIds(left: readonly ResultRow[], right: readonly ResultRow[], predicate: (a: ResultRow, b: ResultRow) => boolean): string[] {
  return left.filter((row, index) => predicate(row, right[index])).map((row) => row.id);
}

async function main(): Promise<void> {
  const args = parseArgs();
  for (const key of ["initial", "control", "unlikelihood", "evaluation-contract", "out"] as const) {
    if (!args[key]) throw new Error(`required: --${key}`);
  }
  const [initial, control, unlikelihood, contractText] = await Promise.all([
    loadEvaluation(args.initial, "I0"),
    loadEvaluation(args.control, "C0"),
    loadEvaluation(args.unlikelihood, "U1"),
    readFile(resolve(args["evaluation-contract"]), "utf8"),
  ]);
  const contractHash = sha256(contractText);
  const contract = JSON.parse(contractText);
  assert(contract.schema === "alpha-chat-repair-v3-evaluation-contract-v1", "evaluation contract schema drift");
  for (const run of [initial, control, unlikelihood]) {
    assert(run.manifest.identity.evaluationContract.sha256 === contractHash, `${run.manifest.identity.arm} evaluation-contract hash drift`);
    assert(run.manifest.identity.evaluatorCommit === initial.manifest.identity.evaluatorCommit, "evaluator commits differ across arms");
    assert(run.manifest.suites.fresh96.input.sha256 === contract.suites.fresh96.sha256, "fresh96 suite drift");
    assert(run.manifest.suites.regression69.input.sha256 === contract.suites.regression69.sha256, "regression69 suite drift");
  }
  assert(control.manifest.identity.checkpoint.step === unlikelihood.manifest.identity.checkpoint.step, "C0/U1 checkpoint steps differ");
  assert(typeof control.manifest.identity.trainingSourceCommit === "string" &&
    control.manifest.identity.trainingSourceCommit === unlikelihood.manifest.identity.trainingSourceCommit,
    "C0/U1 training source commits differ");
  compareOrder("I0/C0 fresh96", initial.fresh, control.fresh);
  compareOrder("I0/U1 fresh96", initial.fresh, unlikelihood.fresh);
  compareOrder("I0/C0 regression69", initial.regression, control.regression);
  compareOrder("I0/U1 regression69", initial.regression, unlikelihood.regression);

  const i0 = metrics(initial.fresh);
  const c0 = metrics(control.fresh);
  const u1 = metrics(unlikelihood.fresh);
  const fixedLoops = transitionIds(control.fresh, unlikelihood.fresh, (left, right) => left.degenerateLoop && !right.degenerateLoop);
  const newLoops = transitionIds(control.fresh, unlikelihood.fresh, (left, right) => !left.degenerateLoop && right.degenerateLoop);
  const newLoopsCleanI0C0 = control.fresh.filter((row, index) => !initial.fresh[index].degenerateLoop && !row.degenerateLoop && unlikelihood.fresh[index].degenerateLoop).map((row) => row.id);
  const lostCommonNonempty = control.fresh.filter((row, index) => initial.fresh[index].nonempty && row.nonempty && !unlikelihood.fresh[index].nonempty).map((row) => row.id);
  const reduction = c0.degenerateLoops === 0 ? null : (c0.degenerateLoops - u1.degenerateLoops) / c0.degenerateLoops;
  const underpowered = c0.degenerateLoops === 0 || Math.ceil(c0.degenerateLoops * 0.3) < 3;
  const primaryChecks = {
    c0_loop_population_supports_three_case_reduction: !underpowered,
    at_least_30_percent_fewer_loops: reduction !== null && reduction >= 0.3,
    fixed_at_least_twice_new: fixedLoops.length >= 2 * newLoops.length,
    at_most_two_new_loops_clean_in_i0_and_c0: newLoopsCleanI0C0.length <= 2,
    mean_repeat_lower_than_c0: u1.meanFourGramRepeatRate < c0.meanFourGramRepeatRate,
    mean_repeat_lower_than_i0: u1.meanFourGramRepeatRate < i0.meanFourGramRepeatRate,
  };
  const structuralChecks = {
    nonempty_at_least_95: u1.nonempty >= 95,
    loses_no_common_nonempty_response: lostCommonNonempty.length === 0,
    structural_pass_not_lower_than_c0: u1.structuralPass >= c0.structuralPass,
    eos_no_more_than_one_below_c0: u1.eosTerminated >= c0.eosTerminated - 1,
    zero_role_leaks: u1.roleLeaks === 0,
  };
  const primaryPass = Object.values(primaryChecks).every(Boolean);
  const structuralPass = Object.values(structuralChecks).every(Boolean);
  const machineResult = underpowered ? "INCONCLUSIVE_UNDERPOWERED" : !primaryPass || !structuralPass ? "FAIL" : "MECHANICAL_PASS_HUMAN_PENDING";
  const report = {
    schema: "alpha-chat-repair-v3-paired-development-analysis-v1",
    status: "development-only; sealed-final-untouched",
    generatedUtc: new Date().toISOString(),
    result: machineResult,
    contract: { path: resolve(args["evaluation-contract"]), sha256: contractHash },
    evaluatorCommit: initial.manifest.identity.evaluatorCommit,
    checkpointStep: control.manifest.identity.checkpoint.step,
    inputs: {
      I0: { path: initial.path, sha256: sha256(initial.text), checkpoint: initial.manifest.identity.checkpoint },
      C0: { path: control.path, sha256: sha256(control.text), checkpoint: control.manifest.identity.checkpoint },
      U1: { path: unlikelihood.path, sha256: sha256(unlikelihood.text), checkpoint: unlikelihood.manifest.identity.checkpoint },
    },
    fresh96: {
      metrics: { I0: i0, C0: c0, U1: u1 },
      primary: {
        c0Loops: c0.degenerateLoops,
        u1Loops: u1.degenerateLoops,
        relativeReduction: reduction,
        fixedLoopIds: fixedLoops,
        newLoopIds: newLoops,
        newLoopIdsCleanInI0AndC0: newLoopsCleanI0C0,
        checks: primaryChecks,
      },
      preservation: { lostCommonNonemptyIds: lostCommonNonempty, checks: structuralChecks },
    },
    regression69: {
      metrics: { I0: metrics(initial.regression), C0: metrics(control.regression), U1: metrics(unlikelihood.regression) },
      decision: "REQUIRES_PAIRED_QUALITATIVE_AND_MULTI-METRIC_REVIEW; no scalar regression score is predeclared",
    },
    qualitative24: {
      status: "PENDING_BLINDED_HUMAN_COMPARISON",
      requirement: "U1 must have more wins than losses against both C0 and I0",
    },
    bge: { status: "NOT_RUN", role: "supporting regression alarm only; never a selector" },
    selection: {
      candidateSelected: false,
      reason: machineResult === "MECHANICAL_PASS_HUMAN_PENDING"
        ? "mechanical development gate passed but human panel and operational evidence remain pending"
        : "mechanical development gate did not admit U1",
      lossUsed: false,
    },
    sealedFinal: { executed: false, inspected: false },
  };
  await writeFile(resolve(args.out), `${JSON.stringify(report, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
  process.stdout.write(`${JSON.stringify({ result: machineResult, primaryChecks, structuralChecks }, null, 2)}\n`);
  if (machineResult === "FAIL") process.exitCode = 1;
}

await main();
