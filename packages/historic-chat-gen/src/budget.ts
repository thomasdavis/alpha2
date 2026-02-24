/**
 * Budget calculation and enforcement.
 *
 * GPT-4.1-mini pricing:
 *   Input:  $0.40 / 1M tokens
 *   Output: $1.60 / 1M tokens
 */

const INPUT_COST_PER_TOKEN = 0.40 / 1_000_000;
const OUTPUT_COST_PER_TOKEN = 1.60 / 1_000_000;

/** Estimated tokens per conversation (for planning). */
export const EST_INPUT_TOKENS = 600;
export const EST_OUTPUT_TOKENS = 800;

export function calculateCost(inputTokens: number, outputTokens: number): number {
  return inputTokens * INPUT_COST_PER_TOKEN + outputTokens * OUTPUT_COST_PER_TOKEN;
}

export function estimateCostPerConversation(): number {
  return calculateCost(EST_INPUT_TOKENS, EST_OUTPUT_TOKENS);
}

export function estimateTotalCost(count: number): number {
  return count * estimateCostPerConversation();
}

export function canAffordBatch(
  budgetLimit: number,
  currentCost: number,
  batchSize: number,
  avgCostPerConversation?: number
): boolean {
  const perConv = avgCostPerConversation ?? estimateCostPerConversation();
  const batchEstimate = batchSize * perConv;
  const remaining = budgetLimit - currentCost;
  return remaining >= batchEstimate;
}

export function isBudgetExhausted(budgetLimit: number, currentCost: number): boolean {
  return (budgetLimit - currentCost) < 0.01;
}

export function formatCost(cost: number): string {
  return `$${cost.toFixed(4)}`;
}

export function budgetSummary(budgetLimit: number, currentCost: number): string {
  const remaining = budgetLimit - currentCost;
  const pct = budgetLimit > 0 ? ((currentCost / budgetLimit) * 100).toFixed(1) : "0.0";
  return `${formatCost(currentCost)} / ${formatCost(budgetLimit)} (${pct}% used, ${formatCost(remaining)} remaining)`;
}
