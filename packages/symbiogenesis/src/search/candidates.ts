/**
 * Candidate generation and lifecycle for FFN activation search.
 * Supports both fixed activation pool and composed activation graphs.
 */

import type { ActivationNode, BasisOp, MutationConfig } from "../activation/index.js";
import { basisGraph, nameGraph, cloneGraph, mutateActivationGraph, crossoverGraphs, BASIS_POOL } from "../activation/index.js";

export interface SearchCandidate {
  readonly id: string;
  readonly name: string;
  readonly activation: string;
  readonly generation: number;
  readonly parentId: string | null;
  readonly parentName: string | null;
  readonly lineage: readonly string[];
  /** Activation expression tree (composed mode). Null for fixed-activation mode. */
  readonly activationGraph: ActivationNode | null;
  /** Which mutation created this candidate (composed mode). */
  readonly mutationApplied: string | null;
  bestLoss: number;
  bestValLoss: number;
  fitnessScore: number;
  steps: number;
  alive: boolean;
}

let candidateCounter = 0;

// Greek letters for generation naming
const GREEK = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa", "Lambda", "Mu"];
const ACTIVATION_SHORT: Record<string, string> = {
  gelu: "G", silu: "S", relu: "R", swiglu: "Sw", universal: "U", kan_spline: "K",
};

/** Generate a descriptive name for a candidate based on lineage and activation. */
function generateName(activation: string, generation: number, parentName: string | null, childIndex: number): string {
  const prefix = ACTIVATION_SHORT[activation] ?? activation.charAt(0).toUpperCase();
  if (generation === 0) {
    const greekIdx = childIndex % GREEK.length;
    return `${prefix}-${GREEK[greekIdx]}`;
  }
  // Children: inherit parent name + mutation/clone suffix
  if (parentName) {
    return `${prefix}-${parentName.split("-").slice(1).join("-")}.${generation}`;
  }
  return `${prefix}-Gen${generation}-${childIndex}`;
}

/** Generate a name for composed-mode candidates using the graph formula. */
function generateComposedName(graph: ActivationNode, generation: number, parentName: string | null, childIndex: number): string {
  const formula = nameGraph(graph);
  // Truncate long formulas
  const short = formula.length > 30 ? formula.slice(0, 27) + "..." : formula;
  if (generation === 0) {
    const greekIdx = childIndex % GREEK.length;
    return `${short}-${GREEK[greekIdx]}`;
  }
  if (parentName) {
    // Extract lineage suffix from parent name (everything after the first dash-letter)
    const parts = parentName.split("-");
    const lineagePart = parts.length > 1 ? parts[parts.length - 1] : `g${generation}`;
    return `${short}-${lineagePart}.${generation}`;
  }
  return `${short}-g${generation}.${childIndex}`;
}

/** Create a new search candidate. */
export function createCandidate(
  activation: string,
  generation: number,
  parentId: string | null = null,
  parentName: string | null = null,
  parentLineage: readonly string[] = [],
  childIndex = 0,
  activationGraph: ActivationNode | null = null,
  mutationApplied: string | null = null,
): SearchCandidate {
  const counter = ++candidateCounter;
  const name = activationGraph
    ? generateComposedName(activationGraph, generation, parentName, childIndex)
    : generateName(activation, generation, parentName, childIndex);
  const id = `gen${generation}_${activation}_${counter}`;
  return {
    id,
    name,
    activation,
    generation,
    parentId,
    parentName,
    lineage: [...parentLineage, id],
    activationGraph,
    mutationApplied,
    bestLoss: Infinity,
    bestValLoss: Infinity,
    fitnessScore: -Infinity,
    steps: 0,
    alive: true,
  };
}

/**
 * Generate initial population from the activation pool.
 * Distributes evenly across available activations, cycling if populationSize > pool size.
 */
export function generateInitialPopulation(
  pool: readonly string[],
  populationSize: number,
): SearchCandidate[] {
  const candidates: SearchCandidate[] = [];
  for (let i = 0; i < populationSize; i++) {
    const activation = pool[i % pool.length];
    candidates.push(createCandidate(activation, 0, null, null, [], i));
  }
  return candidates;
}

/**
 * Generate initial population for composed-activation mode.
 * Each candidate starts with a pure basis function graph.
 */
export function generateComposedPopulation(
  basisPool: readonly BasisOp[],
  populationSize: number,
): SearchCandidate[] {
  const candidates: SearchCandidate[] = [];
  for (let i = 0; i < populationSize; i++) {
    const basis = basisPool[i % basisPool.length];
    const graph = basisGraph(basis);
    // Use "composed" as the activation type — the graph defines the actual function
    candidates.push(createCandidate("composed", 0, null, null, [], i, graph, null));
  }
  return candidates;
}

/**
 * Mutate a candidate: randomly swap its activation to another from the pool.
 * Maintains parent lineage for tree visualization.
 */
export function mutateCandidate(
  parent: SearchCandidate,
  pool: readonly string[],
  generation: number,
  childIndex = 0,
): SearchCandidate {
  const otherActivations = pool.filter(a => a !== parent.activation);
  if (otherActivations.length === 0) {
    return createCandidate(parent.activation, generation, parent.id, parent.name, parent.lineage, childIndex);
  }
  const newActivation = otherActivations[Math.floor(Math.random() * otherActivations.length)];
  return createCandidate(newActivation, generation, parent.id, parent.name, parent.lineage, childIndex);
}

/**
 * Mutate a candidate's activation graph structurally.
 * Creates a new graph by applying random mutations (inject_residual, inject_gate, add_term, etc.)
 */
export function mutateComposedCandidate(
  parent: SearchCandidate,
  generation: number,
  childIndex: number,
  mutCfg: MutationConfig,
): SearchCandidate {
  if (!parent.activationGraph) {
    throw new Error("mutateComposedCandidate called on candidate without activation graph");
  }
  const rng = { next: () => Math.random() };
  const { graph, mutationApplied } = mutateActivationGraph(parent.activationGraph, rng, mutCfg);
  return createCandidate(
    "composed", generation, parent.id, parent.name, parent.lineage,
    childIndex, graph, mutationApplied,
  );
}

/**
 * Clone a candidate's activation graph for the next generation (no structural change).
 */
export function cloneComposedCandidate(
  parent: SearchCandidate,
  generation: number,
  childIndex: number,
): SearchCandidate {
  if (!parent.activationGraph) {
    throw new Error("cloneComposedCandidate called on candidate without activation graph");
  }
  return createCandidate(
    "composed", generation, parent.id, parent.name, parent.lineage,
    childIndex, cloneGraph(parent.activationGraph), "clone",
  );
}
