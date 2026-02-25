/**
 * FFN activation search orchestrator.
 * Evolutionary search over activation functions with fitness ranking.
 * Inspired by symbiogenesis/population.py — written from scratch.
 */
import type { SymbioConfig } from "../config/schema.js";
import {
  type SearchCandidate,
  createCandidate,
  generateInitialPopulation,
  mutateCandidate,
} from "./candidates.js";
import { selectParents, rankCandidates } from "./ranking.js";
import { generateSummary, generateCandidatesJSONL, generateReport, type SearchSummary } from "./report.js";
import { computeArchitectureDiversity } from "../metrics/population.js";

export interface SearchState {
  generation: number;
  candidateIndex: number;
  stepInCandidate: number;
  population: SearchCandidate[];
  allCandidates: SearchCandidate[];
  done: boolean;
  winner: SearchCandidate | null;
}

export class SearchOrchestrator {
  private readonly config: SymbioConfig;
  private state: SearchState;

  constructor(config: SymbioConfig) {
    this.config = config;
    const initial = generateInitialPopulation(config.activationPool, config.populationSize);
    this.state = {
      generation: 0,
      candidateIndex: 0,
      stepInCandidate: 0,
      population: initial,
      allCandidates: [...initial],
      done: false,
      winner: null,
    };
  }

  /** Get the current candidate being evaluated. */
  get currentCandidate(): SearchCandidate | null {
    if (this.state.done) return null;
    return this.state.population[this.state.candidateIndex] ?? null;
  }

  /** Get the current generation number. */
  get generation(): number {
    return this.state.generation;
  }

  /** Whether the search is complete. */
  get isDone(): boolean {
    return this.state.done;
  }

  /** Get architecture diversity for current population. */
  get architectureDiversity(): number {
    return computeArchitectureDiversity(
      this.state.population.map(c => c.activation),
    );
  }

  /**
   * Record a training step for the current candidate.
   * Returns true if the candidate's evaluation is complete (should switch to next).
   */
  recordStep(loss: number, valLoss?: number, fitnessScore?: number): boolean {
    const candidate = this.currentCandidate;
    if (!candidate) return false;

    candidate.steps++;
    this.state.stepInCandidate++;

    if (loss < candidate.bestLoss) candidate.bestLoss = loss;
    if (valLoss !== undefined && valLoss < candidate.bestValLoss) candidate.bestValLoss = valLoss;
    if (fitnessScore !== undefined && fitnessScore > candidate.fitnessScore) {
      candidate.fitnessScore = fitnessScore;
    }

    return this.state.stepInCandidate >= this.config.stepsPerCandidate;
  }

  /**
   * Advance to the next candidate or generation.
   * Returns the activation for the next candidate, or null if search is done.
   */
  advance(): string | null {
    this.state.stepInCandidate = 0;
    this.state.candidateIndex++;

    if (this.state.candidateIndex >= this.state.population.length) {
      // Current generation complete — evolve
      return this.evolveGeneration();
    }

    return this.currentCandidate?.activation ?? null;
  }

  /** Evolve to the next generation. Returns next activation or null if done. */
  private evolveGeneration(): string | null {
    this.state.generation++;

    if (this.state.generation >= this.config.generations) {
      this.state.done = true;
      const ranked = rankCandidates(this.state.allCandidates, this.config);
      this.state.winner = ranked.length > 0 ? ranked[0] : null;
      return null;
    }

    // Select parents
    const numParents = Math.ceil(this.config.populationSize / 2);
    const parents = selectParents(this.state.population, numParents, this.config);

    // Generate new population: keep parents + mutate to fill
    const newPop: SearchCandidate[] = [];

    // Carry forward top parents (elite)
    for (const parent of parents) {
      parent.alive = false; // Mark old generation as done
    }

    // Generate children via mutation — track parent lineage for tree visualization
    for (let i = 0; i < this.config.populationSize; i++) {
      const parentIdx = i % parents.length;
      const parent = parents[parentIdx];
      if (Math.random() < this.config.mutationRate) {
        const child = mutateCandidate(parent, this.config.activationPool, this.state.generation, i);
        newPop.push(child);
        this.state.allCandidates.push(child);
      } else {
        // Clone parent's activation for new generation
        const child = createCandidate(parent.activation, this.state.generation, parent.id, parent.name, parent.lineage, i);
        newPop.push(child);
        this.state.allCandidates.push(child);
      }
    }

    this.state.population = newPop;
    this.state.candidateIndex = 0;
    this.state.stepInCandidate = 0;

    return this.currentCandidate?.activation ?? null;
  }

  /** Generate final search summary. */
  getSummary(): SearchSummary {
    return generateSummary(this.state.allCandidates, this.config);
  }

  /** Generate JSONL of all candidates. */
  getCandidatesJSONL(): string {
    return generateCandidatesJSONL(this.state.allCandidates);
  }

  /** Generate markdown report. */
  getReport(): string {
    return generateReport(this.getSummary());
  }

  /** Get the winner (after search is done). */
  getWinner(): SearchCandidate | null {
    return this.state.winner;
  }
}
