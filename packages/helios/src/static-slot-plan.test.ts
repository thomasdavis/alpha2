import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { loadStaticSlotPlan } from "./static-slot-plan.js";

const directories: string[] = [];

afterEach(() => {
  while (directories.length > 0) rmSync(directories.pop()!, { recursive: true, force: true });
});

function planFile(overrides: Record<string, unknown> = {}): string {
  const directory = mkdtempSync(join(tmpdir(), "helios-static-plan-"));
  directories.push(directory);
  const path = join(directory, "plan.json");
  const analysis = {
    operations: 2,
    planFingerprint: "logical",
    staticSlotFingerprint: "physical",
    staticSlotPlan: {
      slots: [{ slotId: 0, allocationBytes: 256, assignmentCount: 2 }],
      assignments: [
        { producerOperation: 0, producerKind: "matmul", producerKernel: "a", producerPosition: 2, start: 0, lastUse: 0, logicalBytes: 128, slotId: 0, slotBytes: 256 },
        { producerOperation: 1, producerKind: "binary", producerKernel: "b", producerPosition: 2, start: 1, lastUse: 1, logicalBytes: 256, slotId: 0, slotBytes: 256 },
      ],
    },
    ...overrides,
  };
  writeFileSync(path, JSON.stringify({ schemaVersion: 1, planStable: true, analyses: [analysis, analysis] }));
  return path;
}

describe("loadStaticSlotPlan", () => {
  it("loads a stable dense plan and indexes assignments by operation", () => {
    const result = loadStaticSlotPlan(planFile());
    expect(result.operationCount).toBe(2);
    expect(result.assignmentCount).toBe(2);
    expect(result.totalSlotBytes).toBe(256);
    expect(result.assignmentsByOperation.get(1)?.[0].producerKernel).toBe("b");
    expect(result.sourceSha256).toMatch(/^[0-9a-f]{64}$/);
  });

  it("rejects a producer assignment larger than its slot", () => {
    const path = planFile({
      staticSlotPlan: {
        slots: [{ slotId: 0, allocationBytes: 256, assignmentCount: 1 }],
        assignments: [{ producerOperation: 0, producerKind: "matmul", producerKernel: "a", producerPosition: 2, start: 0, lastUse: 0, logicalBytes: 512, slotId: 0, slotBytes: 256 }],
      },
    });
    expect(() => loadStaticSlotPlan(path)).toThrow(/incompatible with slot/);
  });
});
