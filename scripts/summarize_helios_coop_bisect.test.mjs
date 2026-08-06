import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";

function writeRow(root, name, lines, exitCode = 0) {
  const rowRoot = join(root, name);
  mkdirSync(rowRoot, { recursive: true });
  writeFileSync(join(rowRoot, "console.log"), `${lines.join("\n")}\n`);
  writeFileSync(join(rowRoot, "exit-code.txt"), `${exitCode}\n`);
  writeFileSync(join(rowRoot, "CONTROLLED-ENVIRONMENT.txt"), "HELIOS_DISABLE_COOP_MAT=0\n");
}

test("compares finite and non-finite cooperative rows with the FP32 trajectory", () => {
  const root = mkdtempSync(join(tmpdir(), "helios-coop-bisect-"));
  writeRow(root, "baseline_fp32", [
    "step 1/2 | loss=9.0 | lr=1e-3 | grad_norm=10.0 | 100ms/it | 1000 tok/s",
    "step 2/2 | loss=8.0 | lr=1e-3 | grad_norm=9.0 | 90ms/it | 1100 tok/s",
    "coop_shapes: []",
  ]);
  writeRow(root, "cooperative_all", [
    "step 1/2 | loss=9.1 | lr=1e-3 | grad_norm=NaN | 80ms/it | 1200 tok/s",
    "coop_shapes: [{\"key\":\"tb:2x3x4:b1\",\"M\":2,\"N\":3,\"K\":4}]",
  ], 1);

  const command = spawnSync(process.execPath, [
    "scripts/summarize_helios_coop_bisect.mjs",
    "--root", root,
  ], { cwd: process.cwd(), encoding: "utf8" });
  assert.equal(command.status, 0, command.stderr);

  const report = JSON.parse(readFileSync(join(root, "summary.json"), "utf8"));
  const baseline = report.rows.find((row) => row.name === "baseline_fp32");
  const candidate = report.rows.find((row) => row.name === "cooperative_all");
  assert.equal(baseline.comparison.finiteTrajectory, true);
  assert.equal(baseline.comparison.medianTokensPerSecond, 1050);
  assert.equal(candidate.comparison.finiteTrajectory, false);
  assert.ok(Math.abs(candidate.comparison.maxAbsoluteLossDifference - 0.1) < 1e-12);
  assert.equal(candidate.comparison.maxAbsoluteGradNormDifference, null);
  assert.equal(candidate.coopShapes.length, 1);
});
