import { describe, expect, it } from "vitest";
import { shouldUseCoopBackwardAtStep } from "@alpha/train";

describe("cooperative backward exact sentinel cadence", () => {
  it("keeps backward exact when the experimental path is disabled", () => {
    expect(shouldUseCoopBackwardAtStep(false, 0, 1)).toBe(false);
    expect(shouldUseCoopBackwardAtStep(false, 8, 2)).toBe(false);
  });

  it("uses cooperative backward on every step when no sentinel cadence is set", () => {
    expect(shouldUseCoopBackwardAtStep(true, 0, 1)).toBe(true);
    expect(shouldUseCoopBackwardAtStep(true, 0, 100)).toBe(true);
  });

  it("starts exact and repeats the exact sentinel at the declared cadence", () => {
    const schedule = Array.from({ length: 17 }, (_, index) =>
      shouldUseCoopBackwardAtStep(true, 8, index + 1));
    expect(schedule.map((useCoop, index) => useCoop ? null : index + 1))
      .toEqual([1, null, null, null, null, null, null, null, 9, null, null, null, null, null, null, null, 17]);
  });

  it("supports an all-exact backward control", () => {
    expect(shouldUseCoopBackwardAtStep(true, 1, 1)).toBe(false);
    expect(shouldUseCoopBackwardAtStep(true, 1, 20)).toBe(false);
  });

  it("rejects invalid step indices when a cadence is active", () => {
    expect(() => shouldUseCoopBackwardAtStep(true, 8, 0)).toThrow("positive integer step");
  });
});
