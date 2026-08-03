#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

if (process.argv.length !== 6 || process.argv[3] !== "--max-operation") {
  console.error("usage: slice_helios_static_slot_plan.mjs INPUT.json --max-operation N OUTPUT.json");
  process.exit(2);
}

const inputPath = resolve(process.argv[2]);
const maxOperation = Number.parseInt(process.argv[4], 10);
const outputPath = resolve(process.argv[5]);
if (!Number.isInteger(maxOperation) || maxOperation < 0) {
  throw new Error("--max-operation requires a non-negative integer");
}

const sourceBytes = readFileSync(inputPath);
const source = JSON.parse(sourceBytes.toString("utf8"));
if (source.schemaVersion !== 2 || source.planStable !== true || !Array.isArray(source.analyses)) {
  throw new Error("input must be a stable schemaVersion=2 static-slot plan");
}

for (const [analysisIndex, analysis] of source.analyses.entries()) {
  const plan = analysis.staticSlotPlan;
  if (!plan || !Array.isArray(plan.slots) || !Array.isArray(plan.assignments)) {
    throw new Error(`analysis ${analysisIndex} has no executable static-slot plan`);
  }
  const assignments = plan.assignments.filter((row) => row.producerOperation <= maxOperation);
  const usedOldSlotIds = [...new Set(assignments.map((row) => row.slotId))].sort((a, b) => a - b);
  const newSlotIdByOld = new Map(usedOldSlotIds.map((slotId, index) => [slotId, index]));
  const assignmentCountByOld = new Map();
  for (const assignment of assignments) {
    assignmentCountByOld.set(assignment.slotId, (assignmentCountByOld.get(assignment.slotId) ?? 0) + 1);
  }
  const slots = usedOldSlotIds.map((oldSlotId, slotId) => {
    const sourceSlot = plan.slots[oldSlotId];
    if (!sourceSlot || sourceSlot.slotId !== oldSlotId) {
      throw new Error(`analysis ${analysisIndex} has a non-dense source slot table`);
    }
    return {
      slotId,
      allocationBytes: sourceSlot.allocationBytes,
      assignmentCount: assignmentCountByOld.get(oldSlotId),
    };
  });
  const remappedAssignments = assignments.map((row) => ({
    ...row,
    slotId: newSlotIdByOld.get(row.slotId),
  }));
  const fingerprintRows = remappedAssignments.map((row) => [
    row.producerOperation,
    row.producerPosition,
    row.slotId,
    row.slotBytes,
    row.lastUse,
  ]);
  analysis.staticSlotFingerprint = createHash("sha256")
    .update(JSON.stringify(fingerprintRows))
    .digest("hex");
  analysis.staticSlotPlan = { slots, assignments: remappedAssignments };
  analysis.staticBufferSlots = slots.length;
  analysis.staticBufferSlotBytes = slots.reduce((sum, slot) => sum + slot.allocationBytes, 0);
  analysis.slicedAssignmentCount = remappedAssignments.length;
  analysis.sliceMaxProducerOperation = maxOperation;
}

source.derivedPlan = {
  kind: "producer_operation_prefix",
  maxProducerOperation: maxOperation,
  sourcePath: inputPath,
  sourceSha256: createHash("sha256").update(sourceBytes).digest("hex"),
};
writeFileSync(outputPath, `${JSON.stringify(source, null, 2)}\n`);

const first = source.analyses[0];
console.log(JSON.stringify({
  outputPath,
  maxProducerOperation: maxOperation,
  assignments: first.staticSlotPlan.assignments.length,
  slots: first.staticSlotPlan.slots.length,
  bytes: first.staticBufferSlotBytes,
  fingerprint: first.staticSlotFingerprint,
}));
