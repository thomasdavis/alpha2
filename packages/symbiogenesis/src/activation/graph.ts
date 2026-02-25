/**
 * Activation expression tree — compositional activation functions.
 *
 * Every activation is an AST of ops. Fixed activations like "gelu" are
 * single-node trees. Mutations compose them into novel structures.
 * Evaluated at runtime using autograd ops — backprop works automatically.
 */

// ── Types ──────────────────────────────────────────────────

export type BasisOp = "silu" | "relu" | "gelu" | "identity" | "square";

export type ActivationNode =
  | { type: "basis"; op: BasisOp }
  | { type: "scale"; child: ActivationNode; factor: number }
  | { type: "add"; left: ActivationNode; right: ActivationNode }
  | { type: "mul"; left: ActivationNode; right: ActivationNode };

// ── Presets ────────────────────────────────────────────────

export const BASIS_POOL: readonly BasisOp[] = ["silu", "relu", "gelu", "identity", "square"];

export function basisGraph(op: BasisOp): ActivationNode {
  return { type: "basis", op };
}

// ── Naming ─────────────────────────────────────────────────

/** Generate a human-readable compositional name for the activation. */
export function nameGraph(node: ActivationNode): string {
  switch (node.type) {
    case "basis":
      return node.op === "identity" ? "id" : node.op === "square" ? "sq" : node.op;
    case "scale": {
      const f = node.factor;
      const child = nameGraph(node.child);
      if (Math.abs(f - 1) < 0.001) return child;
      if (Math.abs(f) < 0.001) return "0";
      const fStr = Number.isInteger(f) ? String(f) : f.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
      return `${fStr}·${wrapIfComplex(child, node.child)}`;
    }
    case "add": {
      const l = nameGraph(node.left);
      const r = nameGraph(node.right);
      return `${l}+${r}`;
    }
    case "mul": {
      const l = nameGraph(node.left);
      const r = nameGraph(node.right);
      return `${wrapIfComplex(l, node.left)}×${wrapIfComplex(r, node.right)}`;
    }
  }
}

function wrapIfComplex(name: string, node: ActivationNode): string {
  if (node.type === "add") return `(${name})`;
  return name;
}

// ── Metrics ────────────────────────────────────────────────

export function nodeCount(node: ActivationNode): number {
  switch (node.type) {
    case "basis": return 1;
    case "scale": return 1 + nodeCount(node.child);
    case "add": return 1 + nodeCount(node.left) + nodeCount(node.right);
    case "mul": return 1 + nodeCount(node.left) + nodeCount(node.right);
  }
}

export function graphDepth(node: ActivationNode): number {
  switch (node.type) {
    case "basis": return 1;
    case "scale": return 1 + graphDepth(node.child);
    case "add": return 1 + Math.max(graphDepth(node.left), graphDepth(node.right));
    case "mul": return 1 + Math.max(graphDepth(node.left), graphDepth(node.right));
  }
}

// ── Serialization ──────────────────────────────────────────

export function serializeGraph(node: ActivationNode): string {
  return JSON.stringify(node);
}

export function deserializeGraph(json: string): ActivationNode {
  return JSON.parse(json) as ActivationNode;
}

// ── Simplification ─────────────────────────────────────────

/** Simplify a graph: remove identity scales, prune zero branches. */
export function simplifyGraph(node: ActivationNode): ActivationNode {
  switch (node.type) {
    case "basis":
      return node;

    case "scale": {
      const child = simplifyGraph(node.child);
      // scale(x, 1) → x
      if (Math.abs(node.factor - 1) < 0.001) return child;
      // scale(x, 0) → zero (identity * 0 is useless, but keep as scale)
      if (Math.abs(node.factor) < 0.001) return { type: "scale", child: { type: "basis", op: "identity" }, factor: 0 };
      // scale(scale(x, a), b) → scale(x, a*b)
      if (child.type === "scale") return simplifyGraph({ type: "scale", child: child.child, factor: node.factor * child.factor });
      return { type: "scale", child, factor: node.factor };
    }

    case "add": {
      const left = simplifyGraph(node.left);
      const right = simplifyGraph(node.right);
      // add(x, scale(identity, 0)) → x
      if (isZero(right)) return left;
      if (isZero(left)) return right;
      return { type: "add", left, right };
    }

    case "mul": {
      const left = simplifyGraph(node.left);
      const right = simplifyGraph(node.right);
      // mul(x, identity) → x
      if (isIdentity(right)) return left;
      if (isIdentity(left)) return right;
      return { type: "mul", left, right };
    }
  }
}

function isZero(node: ActivationNode): boolean {
  return node.type === "scale" && Math.abs(node.factor) < 0.001;
}

function isIdentity(node: ActivationNode): boolean {
  return node.type === "basis" && node.op === "identity";
}

// ── Clone ──────────────────────────────────────────────────

export function cloneGraph(node: ActivationNode): ActivationNode {
  return deserializeGraph(serializeGraph(node));
}

// ── Collect all leaves ─────────────────────────────────────

export function collectBases(node: ActivationNode): BasisOp[] {
  switch (node.type) {
    case "basis": return [node.op];
    case "scale": return collectBases(node.child);
    case "add": return [...collectBases(node.left), ...collectBases(node.right)];
    case "mul": return [...collectBases(node.left), ...collectBases(node.right)];
  }
}
