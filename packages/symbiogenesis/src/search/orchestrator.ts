/**
 * FFN activation search orchestrator.
 * Supports both fixed-activation pool and composed-activation graph evolution.
 */
import type { SymbioConfig } from "../config/schema.js";
import type { ActivationNode, BasisOp, MutationConfig } from "../activation/index.js";
import { nameGraph, serializeGraph, BASIS_POOL } from "../activation/index.js";
import {
  type SearchCandidate,
  createCandidate,
  generateInitialPopulation,
  generateComposedPopulation,
  mutateCandidate,
  mutateComposedCandidate,
  cloneComposedCandidate,
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

export interface SearchAdaptiveControls {
  populationSize?: number;
  mutationRate?: number;
}

export class SearchOrchestrator {
  private readonly config: SymbioConfig;
  private readonly composed: boolean;
  private readonly mutCfg: MutationConfig;
  private adaptivePopulationSize: number;
  private adaptiveMutationRate: number;
  private state: SearchState;

  constructor(config: SymbioConfig, resumeActivationGraph?: ActivationNode) {
    this.config = config;
    this.composed = config.searchMode === "composed-activation-search";

    this.mutCfg = {
      maxDepth: config.maxGraphDepth ?? 4,
      maxNodes: config.maxGraphNodes ?? 10,
      basisPool: (config.basisPool ?? BASIS_POOL) as BasisOp[],
    };

    let initial: SearchCandidate[];
    if (resumeActivationGraph && this.composed) {
      // Seed population with the resumed activation as the first candidate,
      // then fill remaining slots with mutations of it for diversity.
      const seed = createCandidate(
        "composed", 0, null, null, [], 0,
        resumeActivationGraph, "resume",
      );
      initial = [seed];
      for (let i = 1; i < config.populationSize; i++) {
        const child = mutateComposedCandidate(seed, 0, i, this.mutCfg);
        initial.push(child);
      }
      console.log(`  [symbio] resumed with activation: ${nameGraph(resumeActivationGraph)} + ${config.populationSize - 1} mutations`);
    } else if (this.composed) {
      initial = generateComposedPopulation(this.mutCfg.basisPool, config.populationSize);
    } else {
      initial = generateInitialPopulation(config.activationPool, config.populationSize);
    }
    this.adaptivePopulationSize = config.populationSize;
    this.adaptiveMutationRate = config.mutationRate;

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

  /** Whether this is a composed-activation search. */
  get isComposed(): boolean {
    return this.composed;
  }

  /** Get architecture diversity for current population. */
  get architectureDiversity(): number {
    if (this.composed) {
      // For composed mode, use graph name diversity
      const names = this.state.population.map(c =>
        c.activationGraph ? nameGraph(c.activationGraph) : c.activation,
      );
      const unique = new Set(names).size;
      return unique / Math.max(1, names.length);
    }
    return computeArchitectureDiversity(
      this.state.population.map(c => c.activation),
    );
  }

  /** Current effective population size (may be adapted at runtime). */
  get effectivePopulationSize(): number {
    return this.adaptivePopulationSize;
  }

  /** Current effective mutation rate (may be adapted at runtime). */
  get effectiveMutationRate(): number {
    return this.adaptiveMutationRate;
  }

  /** Apply runtime adaptation controls from the trainer. */
  setAdaptiveControls(ctrl: SearchAdaptiveControls): void {
    if (ctrl.populationSize != null && Number.isFinite(ctrl.populationSize)) {
      this.adaptivePopulationSize = Math.max(2, Math.round(ctrl.populationSize));
    }
    if (ctrl.mutationRate != null && Number.isFinite(ctrl.mutationRate)) {
      this.adaptiveMutationRate = Math.max(0, Math.min(1, ctrl.mutationRate));
    }
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
    const targetPopulationSize = Math.max(2, this.adaptivePopulationSize);
    const mutationRate = this.adaptiveMutationRate;
    const numParents = Math.ceil(targetPopulationSize / 2);
    const parents = selectParents(this.state.population, numParents, this.config);

    // Mark old generation as done
    for (const parent of parents) {
      parent.alive = false;
    }

    const newPop: SearchCandidate[] = [];

    if (this.composed) {
      // Composed mode: structural graph mutations
      for (let i = 0; i < targetPopulationSize; i++) {
        const parentIdx = i % parents.length;
        const parent = parents[parentIdx];
        if (Math.random() < mutationRate) {
          const child = mutateComposedCandidate(parent, this.state.generation, i, this.mutCfg);
          newPop.push(child);
          this.state.allCandidates.push(child);
        } else {
          const child = cloneComposedCandidate(parent, this.state.generation, i);
          newPop.push(child);
          this.state.allCandidates.push(child);
        }
      }
    } else {
      // Fixed-activation mode: swap between pool
      for (let i = 0; i < targetPopulationSize; i++) {
        const parentIdx = i % parents.length;
        const parent = parents[parentIdx];
        if (Math.random() < mutationRate) {
          const child = mutateCandidate(parent, this.config.activationPool, this.state.generation, i);
          newPop.push(child);
          this.state.allCandidates.push(child);
        } else {
          const child = createCandidate(parent.activation, this.state.generation, parent.id, parent.name, parent.lineage, i);
          newPop.push(child);
          this.state.allCandidates.push(child);
        }
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
