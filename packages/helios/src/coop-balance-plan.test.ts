import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  CoopF16x3BalancePlanRuntime,
  canonicalCoopF16x3Descriptor,
  compileCoopF16x3BalancePlan,
  coopF16x3GraphFingerprint,
  loadCoopF16x3BalancePlan,
  parseCoopF16x3CalibrationJsonl,
  type CoopF16x3CalibrationRecord,
  type CoopF16x3MatmulDescriptor,
} from "./coop-balance-plan.js";

const directories: string[] = [];
const checkpointFingerprint = "a".repeat(64);

afterEach(() => {
  while (directories.length > 0) rmSync(directories.pop()!, { recursive: true, force: true });
});

function descriptor(layout: "nn" | "tb" | "ta" = "nn"): CoopF16x3MatmulDescriptor {
  return {
    layout,
    M: 1024,
    N: 1728,
    K: 640,
    batchSize: 1,
    aShape: [1024, 640],
    bShape: layout === "tb" ? [1728, 640] : [640, 1728],
    aDtype: "f32",
    bDtype: "f32",
  };
}

function record(
  descriptors: CoopF16x3MatmulDescriptor[],
  ranges: Array<[number, number]>,
): CoopF16x3CalibrationRecord {
  return {
    schemaVersion: 1,
    kind: "helios-coop-f16x3-calibration",
    checkpointFingerprint,
    createdAt: "2026-08-04T00:00:00.000Z",
    graphFingerprint: coopF16x3GraphFingerprint(descriptors),
    entries: descriptors.map((item, ordinal) => ({
      ordinal,
      descriptor: item,
      maxAbsA: ranges[ordinal][0],
      maxAbsB: ranges[ordinal][1],
    })),
  };
}

function writePlan(value: unknown): string {
  const directory = mkdtempSync(join(tmpdir(), "helios-coop-balance-"));
  directories.push(directory);
  const path = join(directory, "plan.json");
  writeFileSync(path, JSON.stringify(value));
  return path;
}

describe("cooperative FP16x3 calibration and balance plans", () => {
  it("fingerprints layout and ordered shape semantics, not object identity", () => {
    const nn = descriptor("nn");
    expect(canonicalCoopF16x3Descriptor(nn)).toContain("nn|1024x1728x640");
    expect(coopF16x3GraphFingerprint([nn])).toBe(coopF16x3GraphFingerprint([{ ...nn, aShape: [...nn.aShape] }]));
    expect(coopF16x3GraphFingerprint([nn])).not.toBe(coopF16x3GraphFingerprint([descriptor("tb")]));
  });

  it("derives opposite reciprocal exponents from opposite operand ranges", () => {
    const descriptors = [descriptor("nn"), descriptor("ta")];
    const plan = compileCoopF16x3BalancePlan([
      record(descriptors, [[1, 2 ** -10], [2 ** -10, 1]]),
      record(descriptors, [[2, 2 ** -9], [2 ** -9, 2]]),
    ], { createdAt: "2026-08-04T00:00:00.000Z" });
    expect(plan.entries[0].exponent).toBe(-5);
    expect(plan.entries[1].exponent).toBe(5);
    expect(plan.entries[0].samples).toBe(2);
    expect(plan.graphFingerprint).toBe(coopF16x3GraphFingerprint(descriptors));
  });

  it("clamps the preferred exponent to the observed FP16 safety interval", () => {
    const plan = compileCoopF16x3BalancePlan([
      record([descriptor()], [[2 ** 20, 2 ** -20]]),
    ], { safeFp16Max: 2 ** 14, createdAt: "2026-08-04T00:00:00.000Z" });
    expect(plan.entries[0].observedExponentMax).toBe(-6);
    expect(plan.entries[0].exponent).toBe(-20);
    expect(plan.entries[0].exponent).toBeGreaterThanOrEqual(plan.entries[0].observedExponentMin);
  });

  it("rejects a range pair that cannot both fit in FP16", () => {
    expect(() => compileCoopF16x3BalancePlan([
      record([descriptor()], [[2 ** 30, 2 ** 30]]),
    ], { safeFp16Max: 2 ** 14 })).toThrow(/allowed exponent interval is empty/);
  });

  it("requires an explicit graph choice when calibration contains variants", () => {
    const nn = record([descriptor("nn")], [[1, 1]]);
    const tb = record([descriptor("tb")], [[1, 1]]);
    expect(() => compileCoopF16x3BalancePlan([nn, tb])).toThrow(/choose one graph fingerprint/);
  });

  it("round-trips a plan and detects content tampering", () => {
    const plan = compileCoopF16x3BalancePlan([
      record([descriptor()], [[1, 2 ** -10]]),
    ], { createdAt: "2026-08-04T00:00:00.000Z" });
    expect(loadCoopF16x3BalancePlan(writePlan(plan)).planFingerprint).toBe(plan.planFingerprint);
    const tampered = structuredClone(plan);
    tampered.entries[0].exponent++;
    expect(() => loadCoopF16x3BalancePlan(writePlan(tampered))).toThrow(/fingerprint does not match/);
  });

  it("matches every operation and checkpoint before a step is accepted", () => {
    const descriptors = [descriptor("nn"), descriptor("ta")];
    const plan = compileCoopF16x3BalancePlan([
      record(descriptors, [[1, 2 ** -10], [2 ** -10, 1]]),
    ], { createdAt: "2026-08-04T00:00:00.000Z" });
    expect(() => new CoopF16x3BalancePlanRuntime(plan, "b".repeat(64))).toThrow(/belongs to checkpoint/);
    const runtime = new CoopF16x3BalancePlanRuntime(plan, checkpointFingerprint);
    runtime.beginStep();
    expect(runtime.exponentFor(descriptors[0])).toBe(-5);
    expect(runtime.exponentFor(descriptors[1])).toBe(5);
    expect(() => runtime.finishStep()).not.toThrow();
    runtime.beginStep();
    expect(() => runtime.exponentFor(descriptors[1])).toThrow(/drift at operation 0/);
  });

  it("parses append-only calibration JSONL and verifies graph fingerprints", () => {
    const value = record([descriptor()], [[1, 0.25]]);
    expect(parseCoopF16x3CalibrationJsonl(`${JSON.stringify(value)}\n`).length).toBe(1);
    const corrupted = { ...value, graphFingerprint: "0".repeat(64) };
    expect(() => parseCoopF16x3CalibrationJsonl(JSON.stringify(corrupted))).toThrow(/does not match/);
  });
});
