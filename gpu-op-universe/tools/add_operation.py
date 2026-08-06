#!/usr/bin/env python3
"""Safely append one canonical operation to the registry, then regenerate stubs.

Example:
  python tools/add_operation.py alpha gemm myNewGemm MatmulRequest research \
    "One-line semantics, not merely an implementation nickname."
"""
from __future__ import annotations
import json, re, subprocess, sys
from pathlib import Path
if len(sys.argv) < 7:
    raise SystemExit(__doc__)
layer,family,name,request_type,status,summary = sys.argv[1:7]
root=Path(__file__).resolve().parents[1]
p=root/'catalog/operation-registry.json'
r=json.loads(p.read_text())
def kebab(s):
    s=re.sub(r'([a-z0-9])([A-Z])',r'\1-\2',s)
    return re.sub(r'[^A-Za-z0-9]+','-',s).strip('-').lower()
def camel(s):
    parts=[x for x in re.split(r'[^A-Za-z0-9]+',s) if x]
    return parts[0][0].lower()+parts[0][1:]+''.join(x[0].upper()+x[1:] for x in parts[1:])
op_id=f'{layer}.{family}.{kebab(name)}'
if any(o['id']==op_id for o in r['operations']): raise SystemExit(f'already exists: {op_id}')
entry={'id':op_id,'layer':layer,'family':family,'name':name,'exportName':camel(f'{family}_{name}'),
       'requestType':request_type,'summary':summary,'status':status,'target':['architecture-agnostic'],
       'algebra':[],'differentiability':'not-applicable','fusionTags':[],'loweringHints':[],
       'sourceTags':[],'notes':'Added manually; attach sources and a validation plan.'}
r['operations'].append(entry); r['operations'].sort(key=lambda o:(o['layer'],o['family'],o['name'].lower()))
p.write_text(json.dumps(r,indent=2)+'\n')
subprocess.check_call([sys.executable,str(root/'tools/regenerate_stubs.py')])
subprocess.check_call([sys.executable,str(root/'tools/validate_registry.py')])
print(op_id)
