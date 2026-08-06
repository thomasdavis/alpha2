import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { summarizeAB } from "./summarize_helios_ab_runs.mjs";

test("summarizes two arms after warmup and checks trajectory parity", () => {
  const root = join(tmpdir(), `helios-ab-${process.pid}-${Date.now()}`);
  const specs = [];
  for (const [label, times] of [["control", [10, 20, 20]], ["candidate", [10, 10, 10]]]) {
    const runDir = join(root, label);
    mkdirSync(runDir, { recursive: true });
    writeFileSync(join(runDir, "metrics.jsonl"), times.map((elapsed_ms, index) => JSON.stringify({
      step: index + 1,
      loss: 3 - index,
      gradNorm: 2 - index / 2,
      elapsed_ms,
      tokens_per_sec: 1000 / elapsed_ms,
    })).join("\n") + "\n");
    specs.push({ label, runDir });
  }
  const summary = summarizeAB(specs, 1);
  assert.equal(summary.arms.control.samples, 2);
  assert.equal(summary.arms.candidate.mean, 100);
  assert.equal(summary.comparison.meanTokensPerSecondGainPct, 100);
  assert.equal(summary.trajectoryParity.exactAtRecordedPrecision, true);
});
