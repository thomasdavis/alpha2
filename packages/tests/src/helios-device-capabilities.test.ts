import { describe, expect, it } from "vitest";
import { assessHeliosTrainingDevice, gpuVendorName } from "@alpha/train";

const subgroupArithmetic = {
  subgroupSupportedStages: 0x00000020,
  subgroupSupportedOperations: 0x00000004,
};

describe("Helios capability-based device admission", () => {
  it("admits AMD RDNA wave32 without a vendor exception", () => {
    const result = assessHeliosTrainingDevice({
      deviceName: "AMD Radeon RX 7900 XTX",
      vendorId: 0x1002,
      deviceType: 2,
      subgroupSize: 32,
      ...subgroupArithmetic,
    });
    expect(result).toEqual({ supported: true, mode: "portable", reasons: [] });
    expect(gpuVendorName(0x1002)).toBe("AMD");
  });

  it("admits NVIDIA by the same capabilities rather than its vendor id", () => {
    const result = assessHeliosTrainingDevice({
      deviceName: "NVIDIA GeForce RTX 4090",
      vendorId: 0x10de,
      deviceType: 2,
      subgroupSize: 32,
      ...subgroupArithmetic,
    });
    expect(result.supported).toBe(true);
  });

  it("rejects a software Vulkan ICD precisely", () => {
    const result = assessHeliosTrainingDevice({
      deviceName: "llvmpipe",
      vendorId: 0x10005,
      deviceType: 4,
      subgroupSize: 8,
      subgroupSupportedStages: 0x20,
      subgroupSupportedOperations: 0,
    });
    expect(result.supported).toBe(false);
    expect(result.reasons.join(" ")).toContain("not an integrated or discrete Vulkan GPU");
    expect(result.reasons.join(" ")).toContain("subgroup arithmetic is unavailable");
  });

  it("reports the current wave64 kernel-layout blocker without blaming AMD", () => {
    const result = assessHeliosTrainingDevice({
      deviceName: "AMD Instinct",
      vendorId: 0x1002,
      deviceType: 2,
      subgroupSize: 64,
      ...subgroupArithmetic,
    });
    expect(result.supported).toBe(false);
    expect(result.reasons).toEqual([
      "native subgroup size 64 is not yet supported by the current 32-lane kernel layouts",
    ]);
  });
});
