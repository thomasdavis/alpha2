/**
 * Fitness ranking with configurable weights and selection strategies.
 * Inspired by symbiogenesis/training.py — written from scratch.
 */
import type { SearchCandidate } from "./candidates.js";
import type { SymbioConfig } from "../config/schema.js";

/** Rank candidates by the configured metric. Lower rank = better. */
export function rankCandidates(
  candidates: SearchCandidate[],
  config: SymbioConfig,
): SearchCandidate[] {
  const alive = candidates.filter(c => c.alive);
  if (config.rankBy === "fitness") {
    return alive.sort((a, b) => b.fitnessScore - a.fitnessScore);
  }
  // Default: rank by valLoss (lower is better)
  return alive.sort((a, b) => a.bestValLoss - b.bestValLoss);
}

/** Select top-k candidates for the next generation. */
export function selectTopK(
  ranked: SearchCandidate[],
  k: number,
): SearchCandidate[] {
  return ranked.slice(0, k);
}

/** Tournament selection: pick k random candidates, return the best. */
export function tournamentSelect(
  candidates: SearchCandidate[],
  tournamentK: number,
  config: SymbioConfig,
): SearchCandidate {
  const pool: SearchCandidate[] = [];
  for (let i = 0; i < tournamentK; i++) {
    const idx = Math.floor(Math.random() * candidates.length);
    pool.push(candidates[idx]);
  }
  const ranked = rankCandidates(pool, config);
  return ranked[0];
}

/**
 * Select parents for the next generation based on the configured strategy.
 * Returns the selected parent candidates.
 */
export function selectParents(
  candidates: SearchCandidate[],
  numParents: number,
  config: SymbioConfig,
): SearchCandidate[] {
  const ranked = rankCandidates(candidates, config);

  if (config.selectionStrategy === "tournament") {
    const parents: SearchCandidate[] = [];
    for (let i = 0; i < numParents; i++) {
      parents.push(tournamentSelect(ranked, config.tournamentK, config));
    }
    return parents;
  }

  // Default: topk
  return selectTopK(ranked, numParents);
}
