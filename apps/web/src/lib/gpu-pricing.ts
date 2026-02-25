/** GPU pricing table for cost estimation. */

interface GpuRate {
  pattern: string;
  rate: number; // $/hr
  provider: string;
}

const GPU_RATES: GpuRate[] = [
  { pattern: "H100", rate: 3.40, provider: "GCP" },
  { pattern: "A100-SXM4-80GB", rate: 1.10, provider: "GCP" },
  { pattern: "A100", rate: 1.10, provider: "GCP" },
  { pattern: "L4", rate: 0.50, provider: "GCP" },
  { pattern: "T4", rate: 0.35, provider: "GCP" },
  { pattern: "RTX 4090", rate: 0.69, provider: "GCP" },
  { pattern: "RTX 3090", rate: 0.44, provider: "GCP" },
];

export interface CostEstimate {
  cost: number;
  rate: number;
  hours: number;
  provider: string;
}

export function estimateCost(
  gpuName: string,
  createdAt: string,
  updatedAt: string,
): CostEstimate | null {
  const match = GPU_RATES.find((r) => gpuName.includes(r.pattern));
  if (!match) return null;

  const start = new Date(createdAt).getTime();
  const end = new Date(updatedAt).getTime();
  const hours = Math.max(0, (end - start) / (1000 * 60 * 60));

  return {
    cost: hours * match.rate,
    rate: match.rate,
    hours,
    provider: match.provider,
  };
}
