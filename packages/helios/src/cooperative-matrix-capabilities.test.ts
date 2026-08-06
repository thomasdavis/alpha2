import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import {
  VulkanComponentType,
  VulkanScope,
  analyzeCooperativeMatrixCapabilities,
  canonicalizeCooperativeMatrixProperties,
  type CooperativeMatrixProperty,
} from "./cooperative-matrix-capabilities.js";

function tuple(overrides: Partial<CooperativeMatrixProperty> = {}): CooperativeMatrixProperty {
  return {
    MSize: 16,
    NSize: 8,
    KSize: 16,
    AType: VulkanComponentType.Float16,
    BType: VulkanComponentType.Float16,
    CType: VulkanComponentType.Float32,
    ResultType: VulkanComponentType.Float32,
    saturatingAccumulation: false,
    scope: VulkanScope.Subgroup,
    ...overrides,
  };
}

describe("cooperative matrix capability analysis", () => {
  it("keeps the hand-written native VkStructureType aligned with the Khronos ABI", () => {
    const here = dirname(fileURLToPath(import.meta.url));
    const nativeSource = readFileSync(join(here, "..", "native", "helios_vk.c"), "utf8");
    expect(nativeSource).toMatch(
      /#define VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR 1000506001\b/,
    );
    expect(nativeSource).not.toMatch(
      /#define VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR 1000506002\b/,
    );
  });

  it("does not mislabel FP16 inputs with FP32 accumulation as a TF32 candidate", () => {
    const result = analyzeCooperativeMatrixCapabilities([tuple()]);
    expect(result.fp16InputFp32AccumulatorCandidates).toHaveLength(1);
    expect(result.float32InputFp32AccumulatorCandidates).toHaveLength(0);
    expect(result.float32InputExperimentEligible).toBe(false);
    expect(result.tf32Status).toBe("unproven");
  });

  it("admits a float32-input tuple only as an unproven physical experiment", () => {
    const result = analyzeCooperativeMatrixCapabilities([
      tuple(),
      tuple({
        AType: VulkanComponentType.Float32,
        BType: VulkanComponentType.Float32,
      }),
    ]);
    expect(result.float32InputFp32AccumulatorCandidates).toHaveLength(1);
    expect(result.float32InputExperimentEligible).toBe(true);
    expect(result.tf32Status).toBe("unproven");
    expect(result.tf32ConfirmationRequirements).toContain("physical_timing");
  });

  it("rejects saturating and non-subgroup float32 tuples", () => {
    const result = analyzeCooperativeMatrixCapabilities([
      tuple({ AType: 1, BType: 1, saturatingAccumulation: true }),
      tuple({ AType: 1, BType: 1, scope: 2 }),
    ]);
    expect(result.float32InputExperimentEligible).toBe(false);
  });

  it("canonicalizes driver ordering without discarding unknown component types", () => {
    const unknown = tuple({ AType: 99, BType: 101, MSize: 32 });
    const canonical = canonicalizeCooperativeMatrixProperties([unknown, tuple()]);
    expect(canonical.map((property) => property.AType)).toEqual([0, 99]);
    expect(canonical[1]).toMatchObject({ AType: 99, BType: 101, MSize: 32 });
  });

  it("fails closed on malformed native capability values", () => {
    expect(() => canonicalizeCooperativeMatrixProperties([
      tuple({ KSize: Number.NaN }),
    ])).toThrow(/KSize/);
  });
});
