import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";

const script = new URL("./analyze_helios_buffer_lifetimes.mjs", import.meta.url).pathname;

function analyze(rows, args = []) {
  const directory = mkdtempSync(join(tmpdir(), "helios-lifetime-"));
  const trace = join(directory, "trace.jsonl");
  writeFileSync(trace, `${rows.map((row) => JSON.stringify({ traceSchemaVersion: 2, ...row })).join("\n")}\n`);
  try {
    const result = spawnSync(process.execPath, [script, trace, ...args, "--json"], { encoding: "utf8" });
    assert.equal(result.status, 0, result.stderr);
    return JSON.parse(result.stdout);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

function op(order, bufferIds, bufferBytes, writeMask, kernel = `k${order}`, kind = "matmul") {
  return {
    event: "op", order, kind, kernel,
    bufferCount: bufferIds.length, bufferIds, bufferBytes, writeMask,
    groups: [1, 1, 1], pushSize: 0, shape: [1], elementCount: null,
  };
}

test("keeps a partially filled output in one static slot across an in-place continuation", () => {
  const result = analyze([{
    step: 1,
    events: [
      op(0, [0, 1, 2], [256, 64, 768], 0b100, "produce-q"),
      op(1, [3, 2], [64, 768], 0b10, "fill-kv", "inplace"),
      op(2, [2, 4], [768, 128], 0b10, "consume"),
    ],
  }], ["--emit-plan"]);
  const row = result.analyses[0];
  const groupedAssignments = row.staticSlotPlan.assignments.filter(
    ({ logicalBytes }) => logicalBytes === 768,
  );
  assert.equal(groupedAssignments.length, 1);
  assert.equal(groupedAssignments[0].producerOperation, 0);
  assert.equal(groupedAssignments[0].lastUse, 2);
});

function hostRead(order, operationCount, bufferId, bufferBytes) {
  return { event: "host_read", order, operationCount, bufferId, bufferBytes };
}

test("versions a reused physical buffer and computes conservative peak liveness", () => {
  const events = [
    op(0, [0, 1, 2], [100, 100, 64], 4),
    op(1, [2, 1, 3], [64, 100, 32], 4),
    op(2, [3, 2], [32, 64], 2),
  ];
  const result = analyze([
    { step: 1, graphSignature: "a", events },
    { step: 2, graphSignature: "a", events },
  ]);
  const row = result.analyses[0];
  assert.equal(result.planStable, true);
  assert.equal(row.externalValues, 2);
  assert.equal(row.transientValues, 3);
  assert.equal(row.totalTransientBytesCreated, 160);
  assert.equal(row.peakLiveTransientBytes, 96);
  assert.equal(row.greedyArenaBytes, 512);
  assert.equal(row.physicalBuffersObserved, 4);
  assert.equal(row.staticBufferSlots, 2);
  assert.equal(row.staticBufferSlotBytes, 512);
  assert.equal(row.staticBufferSlotReuseVsCreated, 160 / 512);
});

test("emits a deterministic producer-to-slot plan and reuses only non-overlapping values", () => {
  const result = analyze([{
    step: 1,
    events: [
      op(0, [0, 1, 2], [256, 256, 256], 4, "first"),
      op(1, [2, 1, 3], [256, 256, 256], 4, "middle"),
      op(2, [3, 1, 4], [256, 256, 256], 4, "last"),
    ],
  }], ["--emit-plan"]);
  const row = result.analyses[0];
  assert.equal(row.staticBufferSlots, 2);
  assert.equal(row.staticBufferSlotBytes, 512);
  assert.deepEqual(
    row.staticSlotPlan.assignments.map(({ producerOperation, slotId }) => ({ producerOperation, slotId })),
    [
      { producerOperation: 0, slotId: 0 },
      { producerOperation: 1, slotId: 1 },
      { producerOperation: 2, slotId: 0 },
    ],
  );
  assert.equal(row.staticSlotPlan.slots[0].assignmentCount, 2);
});

test("keeps a produced value live through a host readback boundary", () => {
  const result = analyze([{
    step: 1,
    events: [
      op(0, [0, 1, 2], [256, 256, 256], 4, "first"),
      hostRead(1, 1, 2, 256),
      op(2, [0, 1, 3], [256, 256, 256], 4, "second"),
    ],
  }], ["--emit-plan"]);
  const row = result.analyses[0];
  assert.equal(row.hostReadEvents, 1);
  assert.equal(row.hostReadValues, 1);
  assert.equal(row.staticBufferSlots, 2);
  assert.notEqual(
    row.staticSlotPlan.assignments[0].slotId,
    row.staticSlotPlan.assignments[1].slotId,
  );
});

test("rejects legacy traces without buffer identities", () => {
  const directory = mkdtempSync(join(tmpdir(), "helios-lifetime-legacy-"));
  const trace = join(directory, "trace.jsonl");
  writeFileSync(trace, `${JSON.stringify({
    step: 1,
    events: [{ event: "op", order: 0, kind: "matmul", kernel: "x", bufferCount: 1, writeMask: 1 }],
  })}\n`);
  try {
    const result = spawnSync(process.execPath, [script, trace, "--json"], { encoding: "utf8" });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /lacks bufferIds\/bufferBytes/);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test("excludes allocation warm-up before checking plan stability", () => {
  const warmup = [op(0, [0, 1, 2], [100, 100, 128], 4)];
  const steady = [op(0, [0, 1, 2], [100, 100, 64], 4)];
  const result = analyze([
    { step: 1, events: warmup },
    { step: 2, events: steady },
    { step: 3, events: steady },
  ], ["--skip-first", "1"]);
  assert.equal(result.capturedSteps, 3);
  assert.equal(result.skippedLeadingSteps, 1);
  assert.equal(result.steps, 2);
  assert.equal(result.planStable, true);
});
