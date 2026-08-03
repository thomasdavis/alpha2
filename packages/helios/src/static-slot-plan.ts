import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

export interface StaticSlotSpec {
  slotId: number;
  allocationBytes: number;
  assignmentCount: number;
}

export interface StaticSlotAssignment {
  producerOperation: number;
  producerKind: string;
  producerKernel: string;
  producerPosition: number;
  start: number;
  lastUse: number;
  logicalBytes: number;
  slotId: number;
  slotBytes: number;
}

export interface StaticSlotPlan {
  sourcePath: string;
  sourceSha256: string;
  planFingerprint: string;
  staticSlotFingerprint: string;
  operationCount: number;
  assignmentCount: number;
  totalSlotBytes: number;
  slots: StaticSlotSpec[];
  assignmentsByOperation: Map<number, StaticSlotAssignment[]>;
}

interface AnalyzerEnvelope {
  schemaVersion?: unknown;
  planStable?: unknown;
  analyses?: unknown;
}

function integer(value: unknown, label: string, minimum = 0): number {
  if (!Number.isSafeInteger(value) || Number(value) < minimum) {
    throw new Error(`${label} must be a safe integer >= ${minimum}`);
  }
  return Number(value);
}

function text(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0) throw new Error(`${label} must be a non-empty string`);
  return value;
}

/** Load one representative, stable plan emitted by analyze_helios_buffer_lifetimes.mjs. */
export function loadStaticSlotPlan(path: string): StaticSlotPlan {
  const raw = readFileSync(path);
  const sourceSha256 = createHash("sha256").update(raw).digest("hex");
  const envelope = JSON.parse(raw.toString("utf8")) as AnalyzerEnvelope;
  if (envelope.schemaVersion !== 1) throw new Error(`unsupported static-slot plan schema: ${String(envelope.schemaVersion)}`);
  if (envelope.planStable !== true) throw new Error("static-slot plan must come from a stable warmup-excluded trace");
  if (!Array.isArray(envelope.analyses) || envelope.analyses.length === 0) {
    throw new Error("static-slot plan has no analyses");
  }

  const analyses = envelope.analyses as Record<string, unknown>[];
  const representative = analyses[0];
  const planFingerprint = text(representative.planFingerprint, "planFingerprint");
  const staticSlotFingerprint = text(representative.staticSlotFingerprint, "staticSlotFingerprint");
  for (const [index, analysis] of analyses.entries()) {
    if (analysis.planFingerprint !== planFingerprint || analysis.staticSlotFingerprint !== staticSlotFingerprint) {
      throw new Error(`analysis ${index} does not match the representative lifetime and slot plan`);
    }
  }

  const plan = representative.staticSlotPlan as Record<string, unknown> | undefined;
  if (!plan || !Array.isArray(plan.slots) || !Array.isArray(plan.assignments)) {
    throw new Error("static-slot plan was not emitted; rerun the analyzer with --emit-plan");
  }

  const slots = (plan.slots as Record<string, unknown>[]).map((slot, index): StaticSlotSpec => {
    const slotId = integer(slot.slotId, `slots[${index}].slotId`);
    if (slotId !== index) throw new Error(`slots must be dense and ordered; expected ${index}, got ${slotId}`);
    return {
      slotId,
      allocationBytes: integer(slot.allocationBytes, `slots[${index}].allocationBytes`, 1),
      assignmentCount: integer(slot.assignmentCount, `slots[${index}].assignmentCount`, 1),
    };
  });
  const assignmentsByOperation = new Map<number, StaticSlotAssignment[]>();
  const assignmentKeys = new Set<string>();
  const assignmentCounts = new Uint32Array(slots.length);
  const assignments = (plan.assignments as Record<string, unknown>[]).map((assignment, index): StaticSlotAssignment => {
    const parsed = {
      producerOperation: integer(assignment.producerOperation, `assignments[${index}].producerOperation`),
      producerKind: text(assignment.producerKind, `assignments[${index}].producerKind`),
      producerKernel: text(assignment.producerKernel, `assignments[${index}].producerKernel`),
      producerPosition: integer(assignment.producerPosition, `assignments[${index}].producerPosition`),
      start: integer(assignment.start, `assignments[${index}].start`),
      lastUse: integer(assignment.lastUse, `assignments[${index}].lastUse`),
      logicalBytes: integer(assignment.logicalBytes, `assignments[${index}].logicalBytes`, 1),
      slotId: integer(assignment.slotId, `assignments[${index}].slotId`),
      slotBytes: integer(assignment.slotBytes, `assignments[${index}].slotBytes`, 1),
    };
    if (parsed.start !== parsed.producerOperation || parsed.lastUse < parsed.start) {
      throw new Error(`assignments[${index}] has an invalid inclusive lifetime`);
    }
    const slot = slots[parsed.slotId];
    if (!slot || slot.allocationBytes !== parsed.slotBytes || parsed.logicalBytes > slot.allocationBytes) {
      throw new Error(`assignments[${index}] is incompatible with slot ${parsed.slotId}`);
    }
    const key = `${parsed.producerOperation}:${parsed.producerPosition}`;
    if (assignmentKeys.has(key)) throw new Error(`duplicate producer assignment ${key}`);
    assignmentKeys.add(key);
    assignmentCounts[parsed.slotId]++;
    const rows = assignmentsByOperation.get(parsed.producerOperation) ?? [];
    rows.push(parsed);
    assignmentsByOperation.set(parsed.producerOperation, rows);
    return parsed;
  });
  for (const rows of assignmentsByOperation.values()) rows.sort((a, b) => a.producerPosition - b.producerPosition);
  for (const slot of slots) {
    if (assignmentCounts[slot.slotId] !== slot.assignmentCount) {
      throw new Error(
        `slot ${slot.slotId} declares ${slot.assignmentCount} assignments but has ${assignmentCounts[slot.slotId]}`,
      );
    }
  }

  return {
    sourcePath: path,
    sourceSha256,
    planFingerprint,
    staticSlotFingerprint,
    operationCount: integer(representative.operations, "operations", 1),
    assignmentCount: assignments.length,
    totalSlotBytes: slots.reduce((sum, slot) => sum + slot.allocationBytes, 0),
    slots,
    assignmentsByOperation,
  };
}
