import assert from "node:assert/strict";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";

test("aggregates weighted V8 self-time by source frame", async () => {
  const dir = await mkdtemp(path.join(os.tmpdir(), "alpha-cpu-profile-"));
  const profilePath = path.join(dir, "fixture.cpuprofile");
  await writeFile(profilePath, JSON.stringify({
    nodes: [
      { id: 1, callFrame: { functionName: "hot", url: "file:///hot.js", lineNumber: 9, columnNumber: 1 } },
      { id: 2, callFrame: { functionName: "cold", url: "file:///cold.js", lineNumber: 3, columnNumber: 0 } },
    ],
    samples: [1, 2, 1],
    timeDeltas: [1000, 500, 2000],
  }));
  const script = path.resolve("scripts/summarize_node_cpu_profile.mjs");
  const result = spawnSync(process.execPath, [script, "--format", "json", profilePath], { encoding: "utf8" });
  assert.equal(result.status, 0, result.stderr);
  const summary = JSON.parse(result.stdout);
  assert.equal(summary.totalSampledUs, 3500);
  assert.equal(summary.entries[0].functionName, "hot");
  assert.equal(summary.entries[0].selfUs, 3000);
  assert.equal(summary.entries[0].line, 10);
  assert.equal(summary.entries[1].selfUs, 500);
  assert.equal(typeof (await readFile(profilePath)), "object");
});
