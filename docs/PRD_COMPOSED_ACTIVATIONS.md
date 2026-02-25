# PRD: Composed Activation Evolution

## Problem

Current symbiogenesis swaps between 6 fixed activations (gelu, silu, relu, swiglu, universal, kan_spline). Mutation = pick a different one from the pool. This is selection, not evolution — the search space is 6 discrete points.

Real evolution requires **structural mutation**: existing activations mutate into new compositions that don't exist in any textbook. By the end of 50k steps, we should see activations like `(0.7·silu×gelu)+0.3·id` — invented by the system, not designed by a human.

## Design

### Activation Graph

Every activation is an expression tree of ops:

```
Atoms:    silu, relu, gelu, identity, square
Compose:  add(a, b), mul(a, b), scale(a, factor)
```

Fixed activations are trivial graphs: `{ type: "basis", op: "gelu" }`.
After mutation: `{ type: "add", left: { type: "scale", child: { op: "gelu" }, factor: 0.8 }, right: { op: "identity" } }`.

Evaluated at runtime using existing autograd ops — backprop works automatically through any graph.

### Mutations

| Mutation | What it does | Example |
|----------|-------------|---------|
| inject_residual | `f(x)` → `α·f(x) + (1-α)·x` | `gelu` → `0.8·gelu + 0.2·id` |
| inject_gate | `f(x)` → `f(x) × g(x)` | `relu` → `relu × silu` |
| add_term | `f(x)` → `f(x) + α·g(x)` | `silu` → `silu + 0.15·sq` |
| swap_basis | Replace a leaf with different atom | `gelu` → `silu` |
| perturb_scale | Nudge a scale factor ±20% | `0.8·gelu` → `0.72·gelu` |
| prune | Remove a near-zero branch | `silu + 0.01·sq` → `silu` |
| crossover | Graft subtree from another parent | Take gating from parent A, core from parent B |

Each mutation is constrained: max depth 4, max 10 nodes. Graphs auto-simplify after mutation.

### Compositional Names

Names are the formula itself:
- `gelu` → `gelu`
- After inject_residual → `0.8·gelu+0.2·id`
- After inject_gate → `(0.8·gelu+0.2·id)×silu`
- Candidate: `(0.8·gelu+0.2·id)×silu ← gen3 via gelu-Alpha`

### Initial Population

Gen-0 starts with one pure basis per candidate: `gelu`, `silu`, `relu`, `id`, `silu`, `gelu` (for pop=6).
These are the "Adam and Eve" of each lineage. Every subsequent generation mutates structurally.

### Integration

1. **Model forward**: new `"composed"` activation type evaluates the graph using autograd ops
2. **Candidate switching**: orchestrator passes graph to model config on each swap
3. **Metrics**: `symbio_activation_graph` (serialized JSON) logged per step through remote metrics
4. **Dashboard**: graph name shows in candidate tables, switch log, tree chart

### What This Enables

- **Novel activations**: the system invents `(silu×gelu)+0.3·id` because it works better than any fixed activation
- **Per-dataset optimization**: novels data might prefer different activation shapes than chat data
- **Interpretable evolution**: you can trace the lineage of any activation back to its gen-0 ancestor
- **Open-ended search**: the space is infinite (bounded trees), not 6 fixed points

## Implementation

### New Files
- `packages/symbiogenesis/src/activation/graph.ts` — types, evaluate, name, serialize, simplify
- `packages/symbiogenesis/src/activation/mutate.ts` — mutation operators
- `packages/symbiogenesis/src/activation/index.ts` — re-exports

### Modified Files
- `packages/model/src/gpt.ts` — "composed" activation evaluator in forward pass
- `packages/symbiogenesis/src/search/candidates.ts` — graph on candidate, graph-aware naming
- `packages/symbiogenesis/src/search/orchestrator.ts` — graph-aware advance/mutation
- `packages/symbiogenesis/src/config/schema.ts` — composed config options
- `packages/train/src/trainer.ts` — pass graph, emit graph in metrics
- `packages/db/src/{schema,types,metrics}.ts` — new column for graph JSON

### Config

```json
{
  "searchMode": "composed-activation-search",
  "basisPool": ["silu", "relu", "gelu", "identity", "square"],
  "maxGraphDepth": 4,
  "maxGraphNodes": 10,
  "mutationRate": 0.8,
  "populationSize": 6,
  "stepsPerCandidate": 20,
  "generations": 416
}
```
