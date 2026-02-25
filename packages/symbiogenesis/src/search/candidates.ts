/**
 * Candidate generation and lifecycle for FFN activation search.
 * Inspired by symbiogenesis/population.py — written from scratch.
 */

export interface SearchCandidate {
  readonly id: string;
  readonly activation: string;
  readonly generation: number;
  bestLoss: number;
  bestValLoss: number;
  fitnessScore: number;
  steps: number;
  alive: boolean;
}

let candidateCounter = 0;

/** Create a new search candidate. */
export function createCandidate(activation: string, generation: number): SearchCandidate {
  const id = `gen${generation}_${activation}_${++candidateCounter}`;
  return {
    id,
    activation,
    generation,
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
    candidates.push(createCandidate(activation, 0));
  }
  return candidates;
}

/**
 * Mutate a candidate: randomly swap its activation to another from the pool.
 */
export function mutateCandidate(
  parent: SearchCandidate,
  pool: readonly string[],
  generation: number,
): SearchCandidate {
  const otherActivations = pool.filter(a => a !== parent.activation);
  if (otherActivations.length === 0) {
    return createCandidate(parent.activation, generation);
  }
  const newActivation = otherActivations[Math.floor(Math.random() * otherActivations.length)];
  return createCandidate(newActivation, generation);
}
