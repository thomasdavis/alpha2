#!/usr/bin/env python3
import json
from pathlib import Path
import jsonschema
root=Path(__file__).resolve().parents[1]
registry=json.loads((root/'catalog/operation-registry.json').read_text())
schema=json.loads((root/'catalog/operation-registry.schema.json').read_text())
jsonschema.validate(registry,schema)
ids=[o['id'] for o in registry['operations']]
exports=[(o['layer'],o['exportName']) for o in registry['operations']]
assert len(ids)==len(set(ids)), 'duplicate operation id'
assert len(exports)==len(set(exports)), 'duplicate exported name within layer'
print(f"validated {len(ids)} operations")
