import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

export type CoopMatmulLayout = "nn" | "tb" | "ta";

export interface CoopF16x3MatmulDescriptor {
  layout: CoopMatmulLayout;
  M: number;
  N: number;
  K: number;
  batchSize: number;
  aShape: number[];
  bShape: number[];
  aDtype: string;
  bDtype: string;
}

export interface CoopF16x3CalibrationEntry {
  ordinal: number;
  descriptor: CoopF16x3MatmulDescriptor;
  maxAbsA: number;
  maxAbsB: number;
}

export interface CoopF16x3CalibrationRecord {
  schemaVersion: 1;
  kind: "helios-coop-f16x3-calibration";
  checkpointFingerprint: string;
  createdAt: string;
  graphFingerprint: string;
  entries: CoopF16x3CalibrationEntry[];
}

export interface CoopF16x3BalancePlanEntry {
  ordinal: number;
  descriptor: CoopF16x3MatmulDescriptor;
  exponent: number;
  samples: number;
  observedMaxAbsA: number;
  observedMaxAbsB: number;
  observedExponentMin: number;
  observedExponentMax: number;
}

export interface CoopF16x3BalancePlan {
  schemaVersion: 1;
  kind: "helios-coop-f16x3-balance-plan";
  checkpointFingerprint: string;
  graphFingerprint: string;
  planFingerprint: string;
  sourceCalibrationFingerprint: string;
  createdAt: string;
  safeFp16Max: number;
  entries: CoopF16x3BalancePlanEntry[];
  sourcePath?: string;
  sourceSha256?: string;
}

export interface CompileCoopF16x3BalancePlanOptions {
  graphFingerprint?: string;
  safeFp16Max?: number;
  createdAt?: string;
}

const MAX_BALANCE_EXPONENT = 120;
const DEFAULT_SAFE_FP16_MAX = 32_752;
const SHA256_RE = /^[0-9a-f]{64}$/;

function sha256(data: string | Uint8Array): string {
  return createHash("sha256").update(data).digest("hex");
}

function integer(value: unknown, label: string, minimum = 0): number {
  if (!Number.isSafeInteger(value) || Number(value) < minimum) {
    throw new Error(`${label} must be a safe integer >= ${minimum}`);
  }
  return Number(value);
}

function finiteNonNegative(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new Error(`${label} must be a finite number >= 0`);
  }
  return value;
}

function text(value: unknown, label: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${label} must be a non-empty string`);
  }
  return value;
}

function fingerprint(value: unknown, label: string): string {
  const result = text(value, label).toLowerCase();
  if (!SHA256_RE.test(result)) throw new Error(`${label} must be a lowercase SHA-256 fingerprint`);
  return result;
}

function shape(value: unknown, label: string): number[] {
  if (!Array.isArray(value) || value.length < 2) throw new Error(`${label} must be an array with at least two dimensions`);
  return value.map((dimension, index) => integer(dimension, `${label}[${index}]`, 1));
}

export function parseCoopF16x3Descriptor(value: unknown, label = "descriptor"): CoopF16x3MatmulDescriptor {
  if (!value || typeof value !== "object") throw new Error(`${label} must be an object`);
  const raw = value as Record<string, unknown>;
  const layout = text(raw.layout, `${label}.layout`);
  if (layout !== "nn" && layout !== "tb" && layout !== "ta") {
    throw new Error(`${label}.layout must be nn, tb, or ta`);
  }
  return {
    layout,
    M: integer(raw.M, `${label}.M`, 1),
    N: integer(raw.N, `${label}.N`, 1),
    K: integer(raw.K, `${label}.K`, 1),
    batchSize: integer(raw.batchSize, `${label}.batchSize`, 1),
    aShape: shape(raw.aShape, `${label}.aShape`),
    bShape: shape(raw.bShape, `${label}.bShape`),
    aDtype: text(raw.aDtype, `${label}.aDtype`),
    bDtype: text(raw.bDtype, `${label}.bDtype`),
  };
}

export function canonicalCoopF16x3Descriptor(descriptor: CoopF16x3MatmulDescriptor): string {
  return [
    descriptor.layout,
    `${descriptor.M}x${descriptor.N}x${descriptor.K}`,
    `batch=${descriptor.batchSize}`,
    `a=${descriptor.aDtype}:${descriptor.aShape.join("x")}`,
    `b=${descriptor.bDtype}:${descriptor.bShape.join("x")}`,
  ].join("|");
}

export function coopF16x3GraphFingerprint(descriptors: readonly CoopF16x3MatmulDescriptor[]): string {
  return sha256(descriptors.map((descriptor, ordinal) => `${ordinal}:${canonicalCoopF16x3Descriptor(descriptor)}`).join("\n"));
}

function planFingerprint(plan: Omit<CoopF16x3BalancePlan, "planFingerprint" | "sourcePath" | "sourceSha256">): string {
  return sha256(JSON.stringify(plan));
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle];
}

function preferredExponent(samplesA: number[], samplesB: number[]): number {
  const logA = samplesA.filter((value) => value > 0).map(Math.log2);
  const logB = samplesB.filter((value) => value > 0).map(Math.log2);
  if (logA.length === 0 || logB.length === 0) return 0;
  return Math.round((median(logB) - median(logA)) / 2);
}

function safeExponentInterval(maxAbsA: number, maxAbsB: number, safeFp16Max: number): [number, number] {
  const minimum = maxAbsB === 0
    ? -MAX_BALANCE_EXPONENT
    : Math.ceil(Math.log2(maxAbsB / safeFp16Max));
  const maximum = maxAbsA === 0
    ? MAX_BALANCE_EXPONENT
    : Math.floor(Math.log2(safeFp16Max / maxAbsA));
  return [
    Math.max(-MAX_BALANCE_EXPONENT, minimum),
    Math.min(MAX_BALANCE_EXPONENT, maximum),
  ];
}

function parseCalibrationRecord(value: unknown, label: string): CoopF16x3CalibrationRecord {
  if (!value || typeof value !== "object") throw new Error(`${label} must be an object`);
  const raw = value as Record<string, unknown>;
  if (raw.schemaVersion !== 1 || raw.kind !== "helios-coop-f16x3-calibration") {
    throw new Error(`${label} is not a Helios FP16x3 calibration record v1`);
  }
  if (!Array.isArray(raw.entries) || raw.entries.length === 0) throw new Error(`${label}.entries must be non-empty`);
  const entries = raw.entries.map((entry, index): CoopF16x3CalibrationEntry => {
    if (!entry || typeof entry !== "object") throw new Error(`${label}.entries[${index}] must be an object`);
    const item = entry as Record<string, unknown>;
    const ordinal = integer(item.ordinal, `${label}.entries[${index}].ordinal`);
    if (ordinal !== index) throw new Error(`${label}.entries must be dense and ordered; expected ${index}, got ${ordinal}`);
    return {
      ordinal,
      descriptor: parseCoopF16x3Descriptor(item.descriptor, `${label}.entries[${index}].descriptor`),
      maxAbsA: finiteNonNegative(item.maxAbsA, `${label}.entries[${index}].maxAbsA`),
      maxAbsB: finiteNonNegative(item.maxAbsB, `${label}.entries[${index}].maxAbsB`),
    };
  });
  const graphFingerprint = fingerprint(raw.graphFingerprint, `${label}.graphFingerprint`);
  const calculated = coopF16x3GraphFingerprint(entries.map((entry) => entry.descriptor));
  if (graphFingerprint !== calculated) throw new Error(`${label}.graphFingerprint does not match its ordered descriptors`);
  return {
    schemaVersion: 1,
    kind: "helios-coop-f16x3-calibration",
    checkpointFingerprint: fingerprint(raw.checkpointFingerprint, `${label}.checkpointFingerprint`),
    createdAt: text(raw.createdAt, `${label}.createdAt`),
    graphFingerprint,
    entries,
  };
}

export function parseCoopF16x3CalibrationJsonl(textValue: string): CoopF16x3CalibrationRecord[] {
  const records: CoopF16x3CalibrationRecord[] = [];
  for (const [index, line] of textValue.split(/\r?\n/).entries()) {
    if (line.trim().length === 0) continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(line);
    } catch (error) {
      throw new Error(`calibration line ${index + 1} is invalid JSON: ${error instanceof Error ? error.message : String(error)}`);
    }
    records.push(parseCalibrationRecord(parsed, `calibration line ${index + 1}`));
  }
  if (records.length === 0) throw new Error("calibration JSONL has no records");
  return records;
}

export function compileCoopF16x3BalancePlan(
  records: readonly CoopF16x3CalibrationRecord[],
  options: CompileCoopF16x3BalancePlanOptions = {},
): CoopF16x3BalancePlan {
  if (records.length === 0) throw new Error("cannot compile a balance plan without calibration records");
  const checkpointFingerprint = records[0].checkpointFingerprint;
  for (const [index, record] of records.entries()) {
    if (record.checkpointFingerprint !== checkpointFingerprint) {
      throw new Error(`calibration record ${index} belongs to a different checkpoint`);
    }
  }
  const graphFingerprints = [...new Set(records.map((record) => record.graphFingerprint))].sort();
  const selectedGraph = options.graphFingerprint?.toLowerCase() ?? (graphFingerprints.length === 1 ? graphFingerprints[0] : "");
  if (!selectedGraph) {
    throw new Error(`calibration contains ${graphFingerprints.length} graph variants; choose one graph fingerprint explicitly`);
  }
  if (!graphFingerprints.includes(selectedGraph)) throw new Error(`requested graph fingerprint ${selectedGraph} is absent from calibration`);
  const selected = records.filter((record) => record.graphFingerprint === selectedGraph);
  const representative = selected[0];
  const safeFp16Max = options.safeFp16Max ?? DEFAULT_SAFE_FP16_MAX;
  if (!Number.isFinite(safeFp16Max) || safeFp16Max <= 0 || safeFp16Max > 65_504) {
    throw new Error(`safeFp16Max must be finite and in (0, 65504], got ${safeFp16Max}`);
  }

  for (const [recordIndex, record] of selected.entries()) {
    if (record.entries.length !== representative.entries.length) {
      throw new Error(`selected calibration record ${recordIndex} has a different operation count`);
    }
    for (let ordinal = 0; ordinal < record.entries.length; ordinal++) {
      if (canonicalCoopF16x3Descriptor(record.entries[ordinal].descriptor) !==
          canonicalCoopF16x3Descriptor(representative.entries[ordinal].descriptor)) {
        throw new Error(`selected calibration record ${recordIndex} drifted at operation ${ordinal}`);
      }
    }
  }

  const entries = representative.entries.map((representativeEntry, ordinal): CoopF16x3BalancePlanEntry => {
    const samplesA = selected.map((record) => record.entries[ordinal].maxAbsA);
    const samplesB = selected.map((record) => record.entries[ordinal].maxAbsB);
    const observedMaxAbsA = Math.max(...samplesA);
    const observedMaxAbsB = Math.max(...samplesB);
    const [observedExponentMin, observedExponentMax] = safeExponentInterval(
      observedMaxAbsA,
      observedMaxAbsB,
      safeFp16Max,
    );
    if (observedExponentMin > observedExponentMax) {
      throw new Error(
        `operation ${ordinal} cannot fit both operands under safeFp16Max=${safeFp16Max}; ` +
          `allowed exponent interval is empty (${observedExponentMin}..${observedExponentMax})`,
      );
    }
    const desired = preferredExponent(samplesA, samplesB);
    const exponent = Math.max(observedExponentMin, Math.min(observedExponentMax, desired));
    return {
      ordinal,
      descriptor: representativeEntry.descriptor,
      exponent,
      samples: selected.length,
      observedMaxAbsA,
      observedMaxAbsB,
      observedExponentMin,
      observedExponentMax,
    };
  });

  const sourceCalibrationFingerprint = sha256(JSON.stringify(selected));
  const withoutFingerprint = {
    schemaVersion: 1 as const,
    kind: "helios-coop-f16x3-balance-plan" as const,
    checkpointFingerprint,
    graphFingerprint: selectedGraph,
    sourceCalibrationFingerprint,
    createdAt: options.createdAt ?? new Date().toISOString(),
    safeFp16Max,
    entries,
  };
  return {
    ...withoutFingerprint,
    planFingerprint: planFingerprint(withoutFingerprint),
  };
}

export function loadCoopF16x3BalancePlan(path: string): CoopF16x3BalancePlan {
  const rawBytes = readFileSync(path);
  const raw = JSON.parse(rawBytes.toString("utf8")) as Record<string, unknown>;
  if (raw.schemaVersion !== 1 || raw.kind !== "helios-coop-f16x3-balance-plan") {
    throw new Error("unsupported Helios FP16x3 balance-plan schema");
  }
  if (!Array.isArray(raw.entries) || raw.entries.length === 0) throw new Error("balance plan has no entries");
  const entries = raw.entries.map((entry, index): CoopF16x3BalancePlanEntry => {
    if (!entry || typeof entry !== "object") throw new Error(`entries[${index}] must be an object`);
    const item = entry as Record<string, unknown>;
    const ordinal = integer(item.ordinal, `entries[${index}].ordinal`);
    if (ordinal !== index) throw new Error(`entries must be dense and ordered; expected ${index}, got ${ordinal}`);
    const exponent = Number(item.exponent);
    if (!Number.isInteger(exponent) || Math.abs(exponent) > MAX_BALANCE_EXPONENT) {
      throw new Error(`entries[${index}].exponent must be an integer in [-120, 120]`);
    }
    const observedExponentMin = Number(item.observedExponentMin);
    const observedExponentMax = Number(item.observedExponentMax);
    if (!Number.isInteger(observedExponentMin) || !Number.isInteger(observedExponentMax) ||
        observedExponentMin > exponent || exponent > observedExponentMax) {
      throw new Error(`entries[${index}] exponent falls outside its observed safe interval`);
    }
    return {
      ordinal,
      descriptor: parseCoopF16x3Descriptor(item.descriptor, `entries[${index}].descriptor`),
      exponent,
      samples: integer(item.samples, `entries[${index}].samples`, 1),
      observedMaxAbsA: finiteNonNegative(item.observedMaxAbsA, `entries[${index}].observedMaxAbsA`),
      observedMaxAbsB: finiteNonNegative(item.observedMaxAbsB, `entries[${index}].observedMaxAbsB`),
      observedExponentMin,
      observedExponentMax,
    };
  });
  const graphFingerprint = fingerprint(raw.graphFingerprint, "graphFingerprint");
  if (graphFingerprint !== coopF16x3GraphFingerprint(entries.map((entry) => entry.descriptor))) {
    throw new Error("balance-plan graphFingerprint does not match its ordered descriptors");
  }
  const safeFp16Max = finiteNonNegative(raw.safeFp16Max, "safeFp16Max");
  if (safeFp16Max <= 0 || safeFp16Max > 65_504) {
    throw new Error(`safeFp16Max must be in (0, 65504], got ${safeFp16Max}`);
  }
  const withoutFingerprint = {
    schemaVersion: 1 as const,
    kind: "helios-coop-f16x3-balance-plan" as const,
    checkpointFingerprint: fingerprint(raw.checkpointFingerprint, "checkpointFingerprint"),
    graphFingerprint,
    sourceCalibrationFingerprint: fingerprint(raw.sourceCalibrationFingerprint, "sourceCalibrationFingerprint"),
    createdAt: text(raw.createdAt, "createdAt"),
    safeFp16Max,
    entries,
  };
  const declaredPlanFingerprint = fingerprint(raw.planFingerprint, "planFingerprint");
  if (declaredPlanFingerprint !== planFingerprint(withoutFingerprint)) {
    throw new Error("balance-plan fingerprint does not match its contents");
  }
  return {
    ...withoutFingerprint,
    planFingerprint: declaredPlanFingerprint,
    sourcePath: path,
    sourceSha256: sha256(rawBytes),
  };
}

/** Sequential fail-closed matcher for one stable training graph. */
export class CoopF16x3BalancePlanRuntime {
  private ordinal = 0;
  private descriptors: CoopF16x3MatmulDescriptor[] = [];

  constructor(readonly plan: CoopF16x3BalancePlan, checkpointFingerprint: string) {
    const actual = fingerprint(checkpointFingerprint, "runtime checkpoint fingerprint");
    if (actual !== plan.checkpointFingerprint) {
      throw new Error(`FP16x3 balance plan belongs to checkpoint ${plan.checkpointFingerprint}, not ${actual}`);
    }
  }

  beginStep(): void {
    this.ordinal = 0;
    this.descriptors = [];
  }

  exponentFor(descriptor: CoopF16x3MatmulDescriptor): number {
    const expected = this.plan.entries[this.ordinal];
    if (!expected) throw new Error(`FP16x3 balance plan has no operation ${this.ordinal}`);
    const actualKey = canonicalCoopF16x3Descriptor(descriptor);
    const expectedKey = canonicalCoopF16x3Descriptor(expected.descriptor);
    if (actualKey !== expectedKey) {
      throw new Error(`FP16x3 balance-plan drift at operation ${this.ordinal}: expected ${expectedKey}, got ${actualKey}`);
    }
    this.descriptors.push(descriptor);
    this.ordinal++;
    return expected.exponent;
  }

  finishStep(): void {
    if (this.ordinal !== this.plan.entries.length) {
      throw new Error(`FP16x3 balance-plan graph ended after ${this.ordinal} operations; expected ${this.plan.entries.length}`);
    }
    const actual = coopF16x3GraphFingerprint(this.descriptors);
    if (actual !== this.plan.graphFingerprint) {
      throw new Error(`FP16x3 balance-plan graph fingerprint drifted: expected ${this.plan.graphFingerprint}, got ${actual}`);
    }
  }
}
