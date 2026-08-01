import { describe, expect, it } from "vitest";
import {
  evaluateSamePositionEstimatorOracle,
} from "./estimator.js";

describe("same-position Jacobian estimator", () => {
  it("cancels causal cross-position terms and recovers the mean diagonal blocks", () => {
    const result = evaluateSamePositionEstimatorOracle();
    expect(result.maximumAbsoluteError).toBeLessThan(0.015);
  });
});
