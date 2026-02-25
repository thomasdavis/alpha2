/**
 * Candidate generation and lifecycle for FFN activation search.
 * Inspired by symbiogenesis/population.py — written from scratch.
 */

export interface SearchCandidate {
  readonly id: string;
  readonly name: string;
  readonly activation: string;
  readonly generation: number;
  readonly parentId: string | null;
  readonly parentName: string | null;
  readonly lineage: readonly string[];
  bestLoss: number;
  bestValLoss: number;
  fitnessScore: number;
  steps: number;
  alive: boolean;
}

let candidateCounter = 0;

// Greek letters for generation naming: α, β, γ, δ, ε, ζ, η, θ, ι, κ, λ, μ
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

/** Create a new search candidate. */
export function createCandidate(
  activation: string,
  generation: number,
  parentId: string | null = null,
  parentName: string | null = null,
  parentLineage: readonly string[] = [],
  childIndex = 0,
): SearchCandidate {
  const counter = ++candidateCounter;
  const name = generateName(activation, generation, parentName, childIndex);
  const id = `gen${generation}_${activation}_${counter}`;
  return {
    id,
    name,
    activation,
    generation,
    parentId,
    parentName,
    lineage: [...parentLineage, id],
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
