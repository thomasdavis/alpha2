/**
 * Driver-reported VK_KHR_cooperative_matrix capability analysis.
 *
 * This module deliberately distinguishes a float32-input cooperative tuple
 * from a proven TF32 fast path.  The tuple only licenses a physical shader
 * experiment; parity and timing on the target device must still establish the
 * arithmetic mode and value.
 */

export const VulkanComponentType = {
  Float16: 0,
  Float32: 1,
} as const;

export const VulkanScope = {
  Subgroup: 3,
} as const;

export interface CooperativeMatrixProperty {
  MSize: number;
  NSize: number;
  KSize: number;
  AType: number;
  BType: number;
  CType: number;
  ResultType: number;
  saturatingAccumulation: boolean;
  scope: number;
}

export interface CooperativeMatrixCapabilityAnalysis {
  propertyCount: number;
  fp16InputFp32AccumulatorCandidates: CooperativeMatrixProperty[];
  float32InputFp32AccumulatorCandidates: CooperativeMatrixProperty[];
  float32InputExperimentEligible: boolean;
  tf32Status: "unproven";
  tf32ConfirmationRequirements: readonly [
    "driver_float32_input_tuple",
    "shader_pipeline_creation",
    "numerical_parity",
    "physical_timing",
  ];
}

function finiteNonnegativeInteger(value: number, name: string): number {
  if (!Number.isFinite(value) || value < 0 || !Number.isInteger(value)) {
    throw new Error(`Invalid cooperative-matrix ${name}: ${String(value)}`);
  }
  return value;
}

export function canonicalizeCooperativeMatrixProperties(
  properties: readonly CooperativeMatrixProperty[],
): CooperativeMatrixProperty[] {
  const normalized = properties.map((property) => ({
    MSize: finiteNonnegativeInteger(property.MSize, "MSize"),
    NSize: finiteNonnegativeInteger(property.NSize, "NSize"),
    KSize: finiteNonnegativeInteger(property.KSize, "KSize"),
    AType: finiteNonnegativeInteger(property.AType, "AType"),
    BType: finiteNonnegativeInteger(property.BType, "BType"),
    CType: finiteNonnegativeInteger(property.CType, "CType"),
    ResultType: finiteNonnegativeInteger(property.ResultType, "ResultType"),
    saturatingAccumulation: Boolean(property.saturatingAccumulation),
    scope: finiteNonnegativeInteger(property.scope, "scope"),
  }));

  normalized.sort((a, b) =>
    a.scope - b.scope ||
    a.AType - b.AType ||
    a.BType - b.BType ||
    a.CType - b.CType ||
    a.ResultType - b.ResultType ||
    a.MSize - b.MSize ||
    a.NSize - b.NSize ||
    a.KSize - b.KSize ||
    Number(a.saturatingAccumulation) - Number(b.saturatingAccumulation)
  );
  return normalized;
}

function matchingTuples(
  properties: readonly CooperativeMatrixProperty[],
  aType: number,
  bType: number,
): CooperativeMatrixProperty[] {
  return properties.filter((property) =>
    property.scope === VulkanScope.Subgroup &&
    property.AType === aType &&
    property.BType === bType &&
    property.CType === VulkanComponentType.Float32 &&
    property.ResultType === VulkanComponentType.Float32 &&
    !property.saturatingAccumulation
  );
}

export function analyzeCooperativeMatrixCapabilities(
  properties: readonly CooperativeMatrixProperty[],
): CooperativeMatrixCapabilityAnalysis {
  const canonical = canonicalizeCooperativeMatrixProperties(properties);
  const float32Candidates = matchingTuples(
    canonical,
    VulkanComponentType.Float32,
    VulkanComponentType.Float32,
  );
  return {
    propertyCount: canonical.length,
    fp16InputFp32AccumulatorCandidates: matchingTuples(
      canonical,
      VulkanComponentType.Float16,
      VulkanComponentType.Float16,
    ),
    float32InputFp32AccumulatorCandidates: float32Candidates,
    float32InputExperimentEligible: float32Candidates.length > 0,
    tf32Status: "unproven",
    tf32ConfirmationRequirements: [
      "driver_float32_input_tuple",
      "shader_pipeline_creation",
      "numerical_parity",
      "physical_timing",
    ],
  };
}
