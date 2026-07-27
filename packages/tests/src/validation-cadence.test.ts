import { describe, expect, it } from "vitest";
import { repairTerminalValidationMetric, shouldEvaluateStep } from "@alpha/train";

describe("validation cadence", () => {
  it("always evaluates the terminal step even off cadence", () => {
    expect(shouldEvaluateStep(61_000, 500, 61_036)).toBe(true);
    expect(shouldEvaluateStep(61_035, 500, 61_036)).toBe(false);
    expect(shouldEvaluateStep(61_036, 500, 61_036)).toBe(true);
  });

  it("does not evaluate when the interval is disabled", () => {
    expect(shouldEvaluateStep(100, 0, 100)).toBe(false);
  });

  it("repairs only the final JSONL row", () => {
    const prefix = '{"step":1,"loss":2}\n';
    const terminal = '{"step":2,"loss":1}\n';
    const repaired = repairTerminalValidationMetric(prefix + terminal, 2, 0.75);
    expect(repaired).toBe(`${prefix}{"step":2,"loss":1,"valLoss":0.75}\n`);
  });

  it("fails closed on an existing terminal validation or non-sequential stream", () => {
    expect(() => repairTerminalValidationMetric('{"step":1,"valLoss":1}\n', 1, 0.5))
      .toThrow("already has validation loss");
    expect(() => repairTerminalValidationMetric('{"step":2}\n', 1, 0.5))
      .toThrow("expected step 1");
  });
});
