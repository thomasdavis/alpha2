#!/usr/bin/env python3
"""Regenerate TypeScript stubs from catalog/operation-registry.json."""
from __future__ import annotations
import json
import shutil
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parents[1]
registry = json.loads((root/'catalog/operation-registry.json').read_text())
by = defaultdict(list)
for op in registry['operations']:
    by[(op['layer'],op['family'])].append(op)
for layer in {o['layer'] for o in registry['operations']}:
    generated = root/'packages'/layer/'src'/'generated'
    if generated.exists(): shutil.rmtree(generated)
    generated.mkdir(parents=True)
for (layer,family), operations in sorted(by.items()):
    request_types = sorted({o['requestType'] for o in operations})
    lines = [
        '/* AUTO-GENERATED. Do not hand-edit; edit operation-registry.json. */',
        'import { defineStub } from "../../../common/src/types";',
        'import type { ' + ', '.join(request_types) + ' } from "../../../common/src/types";',
        '',
    ]
    for o in sorted(operations,key=lambda x:x['name'].lower()):
        lines += [
            '/**', f" * {o['id']}", f" * {o['summary']}",
            f" * Status: {o['status']}; target: {', '.join(o['target'])}; differentiability: {o['differentiability']}.",
            ' */', f"export const {o['exportName']} = defineStub<{o['requestType']}>(\"{o['id']}\");", ''
        ]
    (root/'packages'/layer/'src'/'generated'/f'{family}.ts').write_text('\n'.join(lines))
for layer in sorted({o['layer'] for o in registry['operations']}):
    families = sorted({o['family'] for o in registry['operations'] if o['layer']==layer})
    (root/'packages'/layer/'src'/'index.ts').write_text(''.join(f'export * from "./generated/{f}";\n' for f in families))
print(f"regenerated {len(registry['operations'])} stubs")
