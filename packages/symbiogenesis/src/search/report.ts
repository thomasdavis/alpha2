/**
 * Artifact generation: summary JSON, candidates JSONL, markdown report.
 */
import type { SearchCandidate } from "./candidates.js";
import type { SymbioConfig } from "../config/schema.js";
import { nameGraph } from "../activation/index.js";

export interface SearchSummary {
  winner: SearchCandidate | null;
  totalCandidates: number;
  generations: number;
  config: SymbioConfig;
  rankedCandidates: SearchCandidate[];
}

/** Generate JSON summary of the search. */
export function generateSummary(
  candidates: SearchCandidate[],
  config: SymbioConfig,
): SearchSummary {
  const ranked = candidates
    .filter(c => c.steps > 0)
    .sort((a, b) => a.bestValLoss - b.bestValLoss);

  return {
    winner: ranked.length > 0 ? ranked[0] : null,
    totalCandidates: candidates.length,
    generations: Math.max(0, ...candidates.map(c => c.generation)),
    config,
    rankedCandidates: ranked,
  };
}

/** Generate JSONL lines for each candidate. */
export function generateCandidatesJSONL(candidates: SearchCandidate[]): string {
  return candidates.map(c => JSON.stringify(c)).join("\n") + "\n";
}

/** Generate a markdown report of the search results. */
export function generateReport(summary: SearchSummary): string {
  const lines: string[] = [];
  lines.push("# Symbiogenesis FFN Activation Search Report\n");
  lines.push(`**Generations:** ${summary.generations}`);
  lines.push(`**Total Candidates:** ${summary.totalCandidates}`);
  lines.push(`**Strategy:** ${summary.config.searchStrategy}`);
  lines.push(`**Ranked By:** ${summary.config.rankBy}\n`);

  if (summary.winner) {
    lines.push("## Winner\n");
    const winnerFormula = summary.winner.activationGraph
      ? nameGraph(summary.winner.activationGraph)
      : summary.winner.activation;
    lines.push(`- **Activation:** ${winnerFormula}`);
    lines.push(`- **Best Val Loss:** ${summary.winner.bestValLoss.toFixed(4)}`);
    lines.push(`- **Fitness Score:** ${summary.winner.fitnessScore.toFixed(4)}`);
    lines.push(`- **Steps Trained:** ${summary.winner.steps}`);
    lines.push(`- **Generation:** ${summary.winner.generation}\n`);
  }

  lines.push("## All Candidates\n");
  lines.push("| Rank | Name | Activation | Gen | Parent | Val Loss | Fitness | Steps |");
  lines.push("|------|------|------------|-----|--------|----------|---------|-------|");

  for (let i = 0; i < summary.rankedCandidates.length; i++) {
    const c = summary.rankedCandidates[i];
    const formula = c.activationGraph ? nameGraph(c.activationGraph) : c.activation;
    const shortFormula = formula.length > 40 ? formula.slice(0, 37) + "..." : formula;
    lines.push(
      `| ${i + 1} | ${c.name} | ${shortFormula} | ${c.generation} | ${c.parentName ?? "-"} | ${c.bestValLoss === Infinity ? "N/A" : c.bestValLoss.toFixed(4)} | ${c.fitnessScore === -Infinity ? "N/A" : c.fitnessScore.toFixed(4)} | ${c.steps} |`,
    );
  }

  return lines.join("\n") + "\n";
}
