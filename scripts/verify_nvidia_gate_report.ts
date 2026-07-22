#!/usr/bin/env npx tsx
/** Fail closed unless the complete 46-test Helios NVIDIA gate actually executed and passed. */

import { createHash } from "node:crypto";
import { readFile, rename, writeFile } from "node:fs/promises";

function parseArgs(): Record<string, string> {
  const values: Record<string, string> = {};
  for (let index = 2; index < process.argv.length; index++) {
    const arg = process.argv[index];
    if (!arg.startsWith("--")) throw new Error(`unexpected argument: ${arg}`);
    const value = process.argv[++index];
    if (!value || value.startsWith("--")) throw new Error(`missing value for ${arg}`);
    values[arg.slice(2)] = value;
  }
  return values;
}

async function main(): Promise<void> {
  const cli = parseArgs();
  if (!cli.report || !cli.device || !cli.sourceCommit || !cli.out) {
    throw new Error("required: --report, --device, --sourceCommit, and --out");
  }
  if (!/^[0-9a-f]{40}$/.test(cli.sourceCommit)) throw new Error("source commit must be a full lowercase git SHA");
  const [reportText, deviceText] = await Promise.all([
    readFile(cli.report, "utf8"),
    readFile(cli.device, "utf8"),
  ]);
  const report = JSON.parse(reportText) as any;
  const device = JSON.parse(deviceText) as any;
  if (device.vendorId !== 0x10de || typeof device.deviceName !== "string" || !device.deviceName.trim()) {
    throw new Error(`gate device is not NVIDIA: ${JSON.stringify(device)}`);
  }
  const expected = {
    numFailedTestSuites: 0,
    numPendingTestSuites: 0,
    numTotalTests: 46,
    numPassedTests: 46,
    numFailedTests: 0,
    numPendingTests: 0,
    numTodoTests: 0,
    success: true,
  };
  for (const [key, value] of Object.entries(expected)) {
    if (report[key] !== value) throw new Error(`Vitest ${key}: ${String(report[key])} != ${String(value)}`);
  }
  if (!Array.isArray(report.testResults) || report.testResults.length !== 2) {
    throw new Error("Vitest report must contain exactly two test files");
  }
  if (report.testResults.some((result: any) => result.status !== "passed")) {
    throw new Error("one or more NVIDIA gate test files did not execute and pass");
  }
  const expectedFiles = ["gpu-perf.test.ts", "parity-helios.test.ts"];
  const actualFiles = report.testResults.map((result: any) => String(result.name).split("/").at(-1)).sort();
  if (actualFiles.join("\n") !== expectedFiles.join("\n")) {
    throw new Error(`unexpected NVIDIA gate files: ${actualFiles.join(", ")}`);
  }
  const assertions = report.testResults.flatMap((result: any) => result.assertionResults ?? []);
  if (assertions.length !== 46) throw new Error(`assertion rows ${assertions.length} != 46`);
  const nonPassed = assertions.filter((assertion: any) => assertion.status !== "passed");
  if (nonPassed.length > 0) {
    throw new Error(`${nonPassed.length} NVIDIA gate assertions did not execute and pass`);
  }
  const uniqueNames = new Set(assertions.map((assertion: any) => assertion.fullName));
  if (uniqueNames.size !== assertions.length) throw new Error("NVIDIA gate assertion names are not unique");

  const summary = {
    schema: "alpha-nvidia-gate-proof-v1",
    result: "PASS",
    source_commit: cli.sourceCommit,
    completed_utc: new Date().toISOString(),
    device,
    vitest: {
      report_path: cli.report,
      report_sha256: createHash("sha256").update(reportText).digest("hex"),
      files: actualFiles,
      test_suites: 2,
      tests_executed: 46,
      passed: 46,
      failed: 0,
      skipped: 0,
      todo: 0,
    },
  };
  const tmp = `${cli.out}.tmp`;
  await writeFile(tmp, `${JSON.stringify(summary, null, 2)}\n`, { encoding: "utf8", flag: "wx" });
  await rename(tmp, cli.out);
  console.log(JSON.stringify(summary, null, 2));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
