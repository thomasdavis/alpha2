# Alpha GPU Operation Universe

Generated on 6 August 2026.

This scaffold contains **2,644 canonical operation stubs** across the eight project-owned layers of the Alpha stack. It is designed to give Codex a very broad vocabulary without presenting every dtype/layout/algebra combination as a separate hand-maintained API.

## Contents

- `catalog/operation-registry.json` — canonical machine-readable registry.
- `catalog/operation-registry.schema.json` — JSON Schema.
- `catalog/operation-registry.yaml` — human-editable YAML mirror.
- `catalog/CATALOG.md` — complete human-readable list.
- `catalog/coverage.csv` — easy filtering and spreadsheet import.
- `packages/*/src/generated/*.ts` — one exported, throwing TypeScript stub per operation.
- `packages/common/src/types.ts` — shared request/result contracts.
- `docs/MATRIX_MULTIPLICATION_UNIVERSE.md` — expanded GEMM design grammar.
- `docs/LOWERING_GUIDE.md` — examples through all stack layers.
- `docs/CODEX_USAGE.md` — rules for coding agents.
- `tools/validate_registry.py` — schema and uniqueness validation.
- `tools/regenerate_stubs.py` — regenerate TypeScript from the canonical JSON registry.
- `tools/add_operation.py` — append one canonical operation safely.
- `docs/RESEARCH_REPORT.md` — design rationale, technical research and implementation priorities.

## Validate

```bash
python3 tools/validate_registry.py
npm run typecheck
```

## Important limitation

“All possible operations” is not a finite set: arbitrary user-defined functions, semirings, layouts, and fused programs create an unbounded space. The correct exhaustive object is an **operation grammar**. This repository supplies a large canonical dictionary plus variant dimensions and generic IR primitives so the space stays extensible.
