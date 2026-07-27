import { describe, expect, it } from "vitest";
import { shouldEvaluateStep } from "@alpha/train";

describe("validation cadence", () => {
  it("always evaluates the terminal step even off cadence", () => {
    expect(shouldEvaluateStep(61_000, 500, 61_036)).toBe(true);
    expect(shouldEvaluateStep(61_035, 500, 61_036)).toBe(false);
    expect(shouldEvaluateStep(61_036, 500, 61_036)).toBe(true);
  });

  it("does not evaluate when the interval is disabled", () => {
    expect(shouldEvaluateStep(100, 0, 100)).toBe(false);
  });
});
