import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";

const script = new URL("./summarize_helios_graph_trace.mjs", import.meta.url).pathname;

function run(rows) {
  const directory = mkdtempSync(join(tmpdir(), "helios-graph-trace-"));
  const trace = join(directory, "trace.jsonl");
  writeFileSync(trace, `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`);
  try {
    const result = spawnSync(process.execPath, [script, trace, "--json"], { encoding: "utf8" });
    assert.equal(result.status, 0, result.stderr);
    return JSON.parse(result.stdout);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

const baseEvents = [
  { event: "op", order: 0, kind: "matmul", kernel: "matmul_R42", bufferCount: 3 },
  { event: "flush", order: 1, operationCount: 1, withWait: true },
];

test("identical traces are reported as exact topology stability", () => {
  const result = run([
    { step: 1, graphSignature: "aaaa", events: baseEvents },
    { step: 2, graphSignature: "aaaa", events: baseEvents },
  ]);
  assert.equal(result.exactTopologyStable, true);
  assert.equal(result.comparisons[0].firstDifferenceIndex, null);
  assert.equal(result.uniqueSignatures.length, 1);
});

test("first changed operation is localized with common suffix", () => {
  const changed = [
    { ...baseEvents[0], kernel: "matmul_transposed_R42C" },
    baseEvents[1],
  ];
  const result = run([
    { step: 1, graphSignature: "aaaa", events: baseEvents },
    { step: 2, graphSignature: "bbbb", events: changed },
  ]);
  assert.equal(result.exactTopologyStable, false);
  assert.equal(result.comparisons[0].firstDifferenceIndex, 0);
  assert.equal(result.comparisons[0].commonSuffixEvents, 1);
  assert.equal(result.comparisons[0].candidateEvent.kernel, "matmul_transposed_R42C");
});
