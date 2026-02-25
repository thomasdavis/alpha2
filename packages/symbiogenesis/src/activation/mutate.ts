/**
 * Mutation operators for activation expression trees.
 *
 * Each operator takes a graph and returns a structurally modified copy.
 * Graphs are constrained to maxDepth and maxNodes after each mutation.
 */

import type { ActivationNode, BasisOp } from "./graph.js";
import { basisGraph, cloneGraph, simplifyGraph, graphDepth, nodeCount, BASIS_POOL } from "./graph.js";

export interface MutationConfig {
  maxDepth: number;       // default 4
  maxNodes: number;       // default 10
  basisPool: readonly BasisOp[];  // which atoms to use
}

const DEFAULT_CONFIG: MutationConfig = {
  maxDepth: 4,
  maxNodes: 10,
  basisPool: BASIS_POOL,
};

// ── RNG helper ─────────────────────────────────────────────

interface SimpleRng {
  next(): number;  // 0-1
}

function pick<T>(arr: readonly T[], rng: SimpleRng): T {
  return arr[Math.floor(rng.next() * arr.length)];
}

function randRange(lo: number, hi: number, rng: SimpleRng): number {
  return lo + rng.next() * (hi - lo);
}

// ── Mutation operators ─────────────────────────────────────

/**
 * inject_residual: f(x) → α·f(x) + (1-α)·x
 * Adds a skip connection — the most stabilizing mutation.
 */
function injectResidual(node: ActivationNode, rng: SimpleRng): ActivationNode {
  const alpha = randRange(0.6, 0.9, rng);
  return {
    type: "add",
    left: { type: "scale", child: cloneGraph(node), factor: round(alpha) },
    right: { type: "scale", child: basisGraph("identity"), factor: round(1 - alpha) },
  };
}

/**
 * inject_gate: f(x) → f(x) × g(x)
 * Adds a gating mechanism — can discover SwiGLU-like patterns.
 */
function injectGate(node: ActivationNode, rng: SimpleRng, cfg: MutationConfig): ActivationNode {
  const gate = pick(cfg.basisPool.filter(b => b !== "identity" && b !== "square"), rng);
  return { type: "mul", left: cloneGraph(node), right: basisGraph(gate) };
}

/**
 * add_term: f(x) → f(x) + α·g(x)
 * Adds a new basis function with small weight.
 */
function addTerm(node: ActivationNode, rng: SimpleRng, cfg: MutationConfig): ActivationNode {
  const basis = pick(cfg.basisPool, rng);
  const weight = randRange(0.05, 0.3, rng);
  return {
    type: "add",
    left: cloneGraph(node),
    right: { type: "scale", child: basisGraph(basis), factor: round(weight) },
  };
}

/**
 * swap_basis: Replace a random leaf with a different atom.
 */
function swapBasis(node: ActivationNode, rng: SimpleRng, cfg: MutationConfig): ActivationNode {
  const clone = cloneGraph(node);
  const leaves = collectLeafPaths(clone);
  if (leaves.length === 0) return clone;
  const path = pick(leaves, rng);
  const leaf = getNode(clone, path) as { type: "basis"; op: BasisOp };
  const others = cfg.basisPool.filter(b => b !== leaf.op);
  if (others.length === 0) return clone;
  setLeaf(clone, path, basisGraph(pick(others, rng)));
  return clone;
}

/**
 * perturb_scale: Nudge a scale factor by ±20%.
 */
function perturbScale(node: ActivationNode, rng: SimpleRng): ActivationNode {
  const clone = cloneGraph(node);
  const scales = collectScalePaths(clone);
  if (scales.length === 0) return clone;
  const path = pick(scales, rng);
  const scaleNode = getNode(clone, path) as { type: "scale"; factor: number; child: ActivationNode };
  const delta = randRange(-0.2, 0.2, rng);
  scaleNode.factor = round(Math.max(0.01, scaleNode.factor + delta));
  return clone;
}

/**
 * prune: Remove a branch with small scale (< 0.05), replacing with simpler form.
 */
function pruneNode(node: ActivationNode, _rng: SimpleRng): ActivationNode {
  return simplifyGraph(cloneGraph(node));
}

// ── Main mutation entry point ──────────────────────────────

const MUTATIONS = [injectResidual, injectGate, addTerm, swapBasis, perturbScale, pruneNode];
const MUTATION_NAMES = ["inject_residual", "inject_gate", "add_term", "swap_basis", "perturb_scale", "prune"];

export interface MutationResult {
  graph: ActivationNode;
  mutationApplied: string;
}

export function mutateActivationGraph(
  node: ActivationNode,
  rng: SimpleRng,
  cfg: MutationConfig = DEFAULT_CONFIG,
): MutationResult {
  // Try up to 5 times to produce a valid mutation
  for (let attempt = 0; attempt < 5; attempt++) {
    const idx = Math.floor(rng.next() * MUTATIONS.length);
    const mutFn = MUTATIONS[idx];
    let result = mutFn(node, rng, cfg);
    result = simplifyGraph(result);

    // Enforce constraints
    if (graphDepth(result) <= cfg.maxDepth && nodeCount(result) <= cfg.maxNodes) {
      return { graph: result, mutationApplied: MUTATION_NAMES[idx] };
    }
    // If too complex, try prune instead
    result = simplifyGraph(result);
    if (graphDepth(result) <= cfg.maxDepth && nodeCount(result) <= cfg.maxNodes) {
      return { graph: result, mutationApplied: MUTATION_NAMES[idx] + "+prune" };
    }
  }

  // Fallback: swap basis (always valid, same complexity)
  return { graph: swapBasis(node, rng, cfg), mutationApplied: "swap_basis_fallback" };
}

/**
 * Crossover: take the left subtree from parent A, right subtree from parent B.
 */
export function crossoverGraphs(
  a: ActivationNode,
  b: ActivationNode,
  rng: SimpleRng,
  cfg: MutationConfig = DEFAULT_CONFIG,
): ActivationNode {
  const alpha = randRange(0.3, 0.7, rng);
  let result: ActivationNode = {
    type: "add",
    left: { type: "scale", child: cloneGraph(a), factor: round(alpha) },
    right: { type: "scale", child: cloneGraph(b), factor: round(1 - alpha) },
  };
  result = simplifyGraph(result);
  // Enforce constraints — if too big, just pick one parent
  if (graphDepth(result) > cfg.maxDepth || nodeCount(result) > cfg.maxNodes) {
    return rng.next() > 0.5 ? cloneGraph(a) : cloneGraph(b);
  }
  return result;
}

// ── Path-based tree traversal helpers ──────────────────────

type Path = ("left" | "right" | "child")[];

function collectLeafPaths(node: ActivationNode, path: Path = []): Path[] {
  if (node.type === "basis") return [path];
  if (node.type === "scale") return collectLeafPaths(node.child, [...path, "child"]);
  if (node.type === "add" || node.type === "mul") {
    return [
      ...collectLeafPaths(node.left, [...path, "left"]),
      ...collectLeafPaths(node.right, [...path, "right"]),
    ];
  }
  return [];
}

function collectScalePaths(node: ActivationNode, path: Path = []): Path[] {
  const result: Path[] = [];
  if (node.type === "scale") {
    result.push(path);
    result.push(...collectScalePaths(node.child, [...path, "child"]));
  } else if (node.type === "add" || node.type === "mul") {
    result.push(...collectScalePaths(node.left, [...path, "left"]));
    result.push(...collectScalePaths(node.right, [...path, "right"]));
  }
  return result;
}

function getNode(root: ActivationNode, path: Path): ActivationNode {
  let node = root;
  for (const step of path) {
    if (step === "child" && node.type === "scale") node = node.child;
    else if (step === "left" && (node.type === "add" || node.type === "mul")) node = node.left;
    else if (step === "right" && (node.type === "add" || node.type === "mul")) node = node.right;
    else break;
  }
  return node;
}

function setLeaf(root: ActivationNode, path: Path, replacement: ActivationNode): void {
  if (path.length === 0) return; // can't replace root this way
  let node = root;
  for (let i = 0; i < path.length - 1; i++) {
    const step = path[i];
    if (step === "child" && node.type === "scale") node = node.child;
    else if (step === "left" && (node.type === "add" || node.type === "mul")) node = node.left;
    else if (step === "right" && (node.type === "add" || node.type === "mul")) node = node.right;
    else return;
  }
  const last = path[path.length - 1];
  if (last === "child" && node.type === "scale") (node as any).child = replacement;
  else if (last === "left" && (node.type === "add" || node.type === "mul")) (node as any).left = replacement;
  else if (last === "right" && (node.type === "add" || node.type === "mul")) (node as any).right = replacement;
}

function round(n: number): number {
  return Math.round(n * 100) / 100;
}
