#!/usr/bin/env python3
"""
alphaperf — the comprehensive performance-tracking database for the alpha2
native/Vulkan 30k-tokens/s program.

WHY A DATABASE AND NOT A LOG. The program is months of "measure, change one
thing, re-measure", and the same instruments have MISLED this stack repeatedly
(a drained profiler that overstates by a drain per call, a probe that timed host
enqueue, a leak census that erased live buffers). A finding is only worth acting
on if it can be re-checked against the reading it replaced, and a flat commit log
cannot answer "what did layerNorm cost three commits ago" or "which levers were
measured and spent". This can.

WHAT IT HOLDS, one table per KIND of measurement so a query never has to guess:
  gate     — goal-gate / bench-shape end-to-end tok/s, per backend per commit
  kernel   — isolated single-kernel microbenchmarks (us/call, GB/s vs a control)
  gemm     — GEMM rate probes by shape and layout (TFLOP/s)
  commit   — the session's commits, with the before/after tok/s each moved
  finding  — measured findings and REFUTATIONS, so a dead lever stays dead
  isa      — the ISA coverage register snapshot (encoded / captured / missing)

Every row carries the git commit it was taken at and a timestamp, because a
number without a commit is a number about an unknown kernel.

Usage:
  tools/alphaperf.py init                       create the DB and schema
  tools/alphaperf.py gate <commit> <backend> <tok_s> [--median-ms .. --loss ..]
  tools/alphaperf.py kernel <commit> <name> <shape> <us> <gbs> [--pct .. --note ..]
  tools/alphaperf.py gemm <commit> <name> <m> <n> <k> <layout> <batch> <tflops> <us>
  tools/alphaperf.py commit <hash> <title> [--before N --after N --note ..]
  tools/alphaperf.py finding <category> <summary> [--value .. --status .. --note ..]
  tools/alphaperf.py isa <mnemonic> <state> <blocks>
  tools/alphaperf.py show [gate|kernel|gemm|commit|finding|isa]   dump a table
  tools/alphaperf.py latest                     the newest gate per backend
  tools/alphaperf.py sql "<query>"              arbitrary read-only query

The DB lives beside the repo at alphaperf.db (gitignored — it is a workbench,
not a source, and it is rebuilt by re-running the backfill). Measurements are
taken ON the pod (the GPU) and recorded HERE on the box (the git history), so
this is written from the box after reading a pod run.
"""
import sqlite3
import sys
import os
import json
import argparse

DB = os.environ.get("ALPHAPERF_DB",
                    os.path.join(os.path.dirname(__file__), "..", "alphaperf.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS gate (
  id        INTEGER PRIMARY KEY,
  ts        TEXT DEFAULT (datetime('now')),
  commit_id TEXT NOT NULL,          -- git short hash the number was taken at
  backend   TEXT NOT NULL,          -- native | vulkan
  tok_s     REAL NOT NULL,
  layers    INTEGER DEFAULT 18,
  embd      INTEGER DEFAULT 640,
  heads     INTEGER DEFAULT 10,
  vocab     INTEGER DEFAULT 12288,
  seq       INTEGER DEFAULT 64,
  batch     INTEGER,
  median_ms REAL,
  gpu_ms    REAL,
  host_ms   REAL,
  loss      REAL,
  held_gb   REAL,
  note      TEXT
);
CREATE TABLE IF NOT EXISTS kernel (
  id        INTEGER PRIMARY KEY,
  ts        TEXT DEFAULT (datetime('now')),
  commit_id TEXT NOT NULL,
  name      TEXT NOT NULL,          -- layerNorm, rmsNorm, softmax, ...
  shape     TEXT NOT NULL,          -- "1536x640", "15360x64"
  us_call   REAL NOT NULL,
  gbs       REAL,
  pct_ctrl  REAL,                   -- % of the elementwise control's bandwidth
  note      TEXT
);
CREATE TABLE IF NOT EXISTS gemm (
  id        INTEGER PRIMARY KEY,
  ts        TEXT DEFAULT (datetime('now')),
  commit_id TEXT NOT NULL,
  name      TEXT NOT NULL,          -- "qkv B^T", "attn qk nt", "mlp fc dW"
  m INTEGER, n INTEGER, k INTEGER,
  layout    TEXT,                   -- nn | nt | ta
  batch     INTEGER DEFAULT 1,
  tflops    REAL NOT NULL,
  us_call   REAL,
  blocks    INTEGER,
  note      TEXT
);
CREATE TABLE IF NOT EXISTS commit_log (
  hash        TEXT PRIMARY KEY,
  ts          TEXT DEFAULT (datetime('now')),
  title       TEXT NOT NULL,
  tok_s_before REAL,
  tok_s_after  REAL,
  note        TEXT
);
CREATE TABLE IF NOT EXISTS finding (
  id        INTEGER PRIMARY KEY,
  ts        TEXT DEFAULT (datetime('now')),
  category  TEXT NOT NULL,          -- gemm | reduction | isa | attention | ...
  summary   TEXT NOT NULL,
  value     TEXT,                   -- the measurement, as text ("2.1ms", "4%")
  status    TEXT DEFAULT 'confirmed', -- confirmed | refuted | todo | inprogress
  note      TEXT
);
CREATE TABLE IF NOT EXISTS isa (
  mnemonic  TEXT PRIMARY KEY,
  ts        TEXT DEFAULT (datetime('now')),
  state     TEXT NOT NULL,          -- encoded | captured | missing
  blocks    TEXT
);

/*
 * operation — the GPU-operation universe (gpu-op-universe/), given the axis it
 * lacks: IMPLEMENTATION state. The registry ships a DESIGN-maturity status
 * (research/speculative/standard); what a performance program needs to know is
 * how far each operation has come toward a fast, measured kernel. Imported from
 * the registry, then advanced as work lands. This is how the dataset EVOLVES:
 * every op we implement moves along impl and gains a measurement and a code ref.
 *
 *   impl_status ladder (each strictly stronger than the last):
 *     stub       named in the registry, no code
 *     captured   the ISA/bits are known (a .cu capture), nothing emits it
 *     encoded    an emitter exists and passes a ptxas/bit test
 *     tested     correct on hardware against a reference
 *     measured   its cost is in the kernel/gemm tables at a known commit
 *     optimized  measured AND at/near its roofline, no known lever left
 */
CREATE TABLE IF NOT EXISTS operation (
  id           TEXT PRIMARY KEY,    -- layer.family.kebab
  layer        TEXT,
  family       TEXT,
  name         TEXT,
  export_name  TEXT,
  request_type TEXT,
  summary      TEXT,
  design_status TEXT,               -- research | speculative | standard
  target       TEXT,                -- JSON array, as the registry carries it
  algebra      TEXT,                -- JSON array
  differentiability TEXT,
  fusion_tags  TEXT,                -- JSON array
  lowering_hints TEXT,              -- JSON array
  source_tags  TEXT,                -- JSON array
  reg_notes    TEXT,                -- the registry's own notes field
  -- ---- the implementation axis the registry lacks (this DB owns it) ----
  impl_status  TEXT DEFAULT 'stub', -- stub|captured|encoded|tested|measured|optimized
  tflops       REAL,                -- best measured rate, when it has one
  roofline     REAL,                -- the rate that would make it 'optimized'
  code_ref     TEXT,                -- file:symbol where it lives
  commit_id    TEXT,                -- commit at the current impl_status
  note         TEXT,                -- our progress note
  updated      TEXT DEFAULT (datetime('now'))
);

/*
 * experiment — one turn of the autoresearch loop, recorded whether it won or
 * lost. This is the loop's memory: a hypothesis, the lever it tests, the paired
 * before/after, and a verdict. A REFUTED experiment is as valuable as a
 * confirmed one — it is what stops the same dead lever being tried again, which
 * is the failure mode this whole program keeps hitting.
 *
 *   verdict: confirmed | refuted | inconclusive | inprogress
 */
CREATE TABLE IF NOT EXISTS experiment (
  id        INTEGER PRIMARY KEY,
  ts        TEXT DEFAULT (datetime('now')),
  hypothesis TEXT NOT NULL,         -- what we expected and why
  op_id     TEXT,                   -- the operation it touches, if one
  lever     TEXT,                   -- the mechanism (cp.async, warp-shuffle, ...)
  before_v  REAL,                   -- the reading it started from
  after_v   REAL,                   -- the reading it produced
  unit      TEXT DEFAULT 'tok/s',   -- tok/s | TFLOP/s | us | GB/s
  verdict   TEXT DEFAULT 'inprogress',
  commit_id TEXT,
  note      TEXT
);
"""


def conn():
    c = sqlite3.connect(DB)
    c.execute("PRAGMA journal_mode=WAL")
    return c


def cmd_init(_):
    c = conn()
    c.executescript(SCHEMA)
    c.commit()
    print(f"alphaperf: schema ready at {os.path.realpath(DB)}")


def cmd_gate(a):
    c = conn(); c.executescript(SCHEMA)
    c.execute(
        "INSERT INTO gate(commit_id,backend,tok_s,batch,median_ms,gpu_ms,host_ms,loss,held_gb,note)"
        " VALUES(?,?,?,?,?,?,?,?,?,?)",
        (a.commit_id, a.backend, a.tok_s, a.batch, a.median_ms, a.gpu_ms,
         a.host_ms, a.loss, a.held_gb, a.note))
    c.commit()
    print(f"gate: {a.backend} {a.tok_s} tok/s @ {a.commit_id}")


def cmd_kernel(a):
    c = conn(); c.executescript(SCHEMA)
    c.execute(
        "INSERT INTO kernel(commit_id,name,shape,us_call,gbs,pct_ctrl,note)"
        " VALUES(?,?,?,?,?,?,?)",
        (a.commit_id, a.name, a.shape, a.us, a.gbs, a.pct, a.note))
    c.commit()
    print(f"kernel: {a.name} {a.shape} {a.us}us @ {a.commit_id}")


def cmd_gemm(a):
    c = conn(); c.executescript(SCHEMA)
    c.execute(
        "INSERT INTO gemm(commit_id,name,m,n,k,layout,batch,tflops,us_call,blocks,note)"
        " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (a.commit_id, a.name, a.m, a.n, a.k, a.layout, a.batch, a.tflops,
         a.us, a.blocks, a.note))
    c.commit()
    print(f"gemm: {a.name} {a.layout} {a.tflops} TFLOP/s @ {a.commit_id}")


def cmd_commit(a):
    c = conn(); c.executescript(SCHEMA)
    c.execute(
        "INSERT OR REPLACE INTO commit_log(hash,title,tok_s_before,tok_s_after,note)"
        " VALUES(?,?,?,?,?)",
        (a.hash, a.title, a.before, a.after, a.note))
    c.commit()
    print(f"commit: {a.hash} {a.title}")


def cmd_finding(a):
    c = conn(); c.executescript(SCHEMA)
    c.execute(
        "INSERT INTO finding(category,summary,value,status,note) VALUES(?,?,?,?,?)",
        (a.category, a.summary, a.value, a.status, a.note))
    c.commit()
    print(f"finding [{a.status}] {a.category}: {a.summary}")


def cmd_isa(a):
    c = conn(); c.executescript(SCHEMA)
    c.execute("INSERT OR REPLACE INTO isa(mnemonic,state,blocks) VALUES(?,?,?)",
              (a.mnemonic, a.state, a.blocks))
    c.commit()
    print(f"isa: {a.mnemonic} = {a.state}")


def cmd_show(a):
    c = conn(); c.executescript(SCHEMA)
    table = a.table or "gate"
    real = "commit_log" if table == "commit" else table
    rows = c.execute(f"SELECT * FROM {real} ORDER BY 1").fetchall()
    cols = [d[0] for d in c.execute(f"SELECT * FROM {real} LIMIT 0").description]
    print("\t".join(cols))
    for r in rows:
        print("\t".join("" if v is None else str(v) for v in r))


def cmd_latest(_):
    c = conn(); c.executescript(SCHEMA)
    for b in ("native", "vulkan"):
        r = c.execute(
            "SELECT tok_s,commit_id,ts,loss FROM gate WHERE backend=? ORDER BY id DESC LIMIT 1",
            (b,)).fetchone()
        if r:
            print(f"  {b:8s} {r[0]:>8.0f} tok/s  @ {r[1]}  ({r[2]})  loss {r[3]}")
        else:
            print(f"  {b:8s} (no rows)")


def cmd_sql(a):
    c = conn(); c.executescript(SCHEMA)
    cur = c.execute(a.query)
    if cur.description:
        print("\t".join(d[0] for d in cur.description))
        for r in cur.fetchall():
            print("\t".join("" if v is None else str(v) for v in r))


# ---- the autoresearch loop -------------------------------------------------

IMPL_LADDER = ["stub", "captured", "encoded", "tested", "measured", "optimized"]


def cmd_op_import(a):
    """Seed the operation registry INTO this DB — the DB is then its home.

    The universe JSON is only the initial vocabulary; after this import the
    `operation` table is the single source of truth for both what an operation
    is AND how far it is built. Idempotent on id: the registry-owned columns
    refresh, but our implementation columns (impl_status, tflops, code_ref, ...)
    are PRESERVED, so re-seeding a newer dump never loses progress."""
    c = conn(); c.executescript(SCHEMA)
    reg = json.load(open(a.registry))
    ops = reg["operations"] if isinstance(reg, dict) else reg
    j = lambda v: json.dumps(v) if v else None
    added = updated = 0
    for o in ops:
        vals = (o["layer"], o["family"], o["name"], o.get("exportName"),
                o.get("requestType"), o.get("summary"), o.get("status"),
                j(o.get("target")), j(o.get("algebra")), o.get("differentiability"),
                j(o.get("fusionTags")), j(o.get("loweringHints")), j(o.get("sourceTags")),
                o.get("notes"))
        if c.execute("SELECT id FROM operation WHERE id=?", (o["id"],)).fetchone():
            c.execute("UPDATE operation SET layer=?,family=?,name=?,export_name=?,"
                      "request_type=?,summary=?,design_status=?,target=?,algebra=?,"
                      "differentiability=?,fusion_tags=?,lowering_hints=?,source_tags=?,"
                      "reg_notes=? WHERE id=?", vals + (o["id"],))
            updated += 1
        else:
            c.execute("INSERT INTO operation(id,layer,family,name,export_name,"
                      "request_type,summary,design_status,target,algebra,"
                      "differentiability,fusion_tags,lowering_hints,source_tags,reg_notes)"
                      " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (o["id"],) + vals)
            added += 1
    c.commit()
    print(f"operations: {added} imported, {updated} refreshed ({len(ops)} total)."
          f" The DB now owns the registry.")


def cmd_op(a):
    """Advance an operation's implementation state, with its measurement + code."""
    c = conn(); c.executescript(SCHEMA)
    if a.impl_status not in IMPL_LADDER:
        raise SystemExit(f"impl_status must be one of {IMPL_LADDER}")
    ex = c.execute("SELECT id FROM operation WHERE id=?", (a.op_id,)).fetchone()
    if not ex:
        # allow ad-hoc ops (our own primitives) not in the registry. Parse the
        # id as layer.family.name so they still slot into the universe by layer.
        parts = a.op_id.split(".")
        layer = parts[0] if parts else a.op_id
        family = parts[1] if len(parts) >= 3 else ""
        name = parts[-1]
        c.execute("INSERT INTO operation(id,layer,family,name,design_status) VALUES(?,?,?,?,?)",
                  (a.op_id, layer, family, name, "own"))
    c.execute(
        "UPDATE operation SET impl_status=?, tflops=COALESCE(?,tflops), "
        "roofline=COALESCE(?,roofline), code_ref=COALESCE(?,code_ref), "
        "commit_id=COALESCE(?,commit_id), note=COALESCE(?,note), updated=datetime('now') WHERE id=?",
        (a.impl_status, a.tflops, a.roofline, a.ref, a.commit_id, a.note, a.op_id))
    c.commit()
    print(f"op: {a.op_id} -> {a.impl_status}"
          + (f" @ {a.tflops} TFLOP/s" if a.tflops else ""))


def cmd_experiment(a):
    c = conn(); c.executescript(SCHEMA)
    delta = None
    if a.before is not None and a.after is not None and a.before:
        delta = 100.0 * (a.after - a.before) / a.before
    c.execute(
        "INSERT INTO experiment(hypothesis,op_id,lever,before_v,after_v,unit,verdict,commit_id,note)"
        " VALUES(?,?,?,?,?,?,?,?,?)",
        (a.hypothesis, a.op_id, a.lever, a.before, a.after, a.unit, a.verdict, a.commit_id, a.note))
    c.commit()
    d = f" ({delta:+.1f}%)" if delta is not None else ""
    print(f"experiment [{a.verdict}]: {a.hypothesis}{d}")


def cmd_roadmap(a):
    """The implementation frontier: how far the universe has been built."""
    c = conn(); c.executescript(SCHEMA)
    total = c.execute("SELECT count(*) FROM operation").fetchone()[0]
    if not total:
        print("no operations imported — run: alphaperf.py op-import <registry.json>")
        return
    print(f"operation universe — {total} operations\n")
    print("  impl_status     count   (the implementation frontier)")
    for s in IMPL_LADDER:
        n = c.execute("SELECT count(*) FROM operation WHERE impl_status=?", (s,)).fetchone()[0]
        bar = "#" * min(40, n)
        print(f"  {s:12s} {n:7d}   {bar}")
    print("\n  the operations that ARE built (past stub), newest first:")
    for r in c.execute("SELECT id,impl_status,tflops,commit_id FROM operation "
                       "WHERE impl_status!='stub' ORDER BY updated DESC LIMIT 20"):
        t = f"{r[2]:.1f} TFLOP/s" if r[2] else ""
        print(f"    {r[1]:9s} {r[0]:52s} {t:14s} {r[3] or ''}")


def cmd_loop(a):
    """The autoresearch state: where the number is, and what to try next."""
    c = conn(); c.executescript(SCHEMA)
    print("=== alphaperf — autoresearch state ===\n")
    print("current throughput:")
    cmd_latest(a)
    print("\nlast experiments (the loop's memory):")
    for r in c.execute("SELECT verdict,hypothesis,before_v,after_v,unit FROM experiment "
                       "ORDER BY id DESC LIMIT 6"):
        d = ""
        if r[2] and r[3]:
            d = f"  {r[2]:.0f}->{r[3]:.0f} {r[4]}"
        print(f"  [{r[0]:11s}] {r[1][:72]}{d}")
    print("\nopen levers (findings still todo/inprogress):")
    for r in c.execute("SELECT category,summary,value FROM finding "
                       "WHERE status IN ('todo','inprogress') ORDER BY id DESC LIMIT 8"):
        print(f"  [{r[0]:10s}] {r[1][:70]}" + (f"  ({r[2]})" if r[2] else ""))
    print("\nREFUTED levers (do not retry):")
    for r in c.execute("SELECT summary FROM finding WHERE status='refuted' ORDER BY id DESC LIMIT 8"):
        print(f"  x {r[0][:80]}")


def main():
    p = argparse.ArgumentParser(description="alphaperf tracking DB")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init").set_defaults(fn=cmd_init)

    g = sub.add_parser("gate"); g.set_defaults(fn=cmd_gate)
    g.add_argument("commit_id"); g.add_argument("backend"); g.add_argument("tok_s", type=float)
    g.add_argument("--batch", type=int); g.add_argument("--median-ms", dest="median_ms", type=float)
    g.add_argument("--gpu-ms", dest="gpu_ms", type=float); g.add_argument("--host-ms", dest="host_ms", type=float)
    g.add_argument("--loss", type=float); g.add_argument("--held-gb", dest="held_gb", type=float)
    g.add_argument("--note", default="")

    k = sub.add_parser("kernel"); k.set_defaults(fn=cmd_kernel)
    k.add_argument("commit_id"); k.add_argument("name"); k.add_argument("shape")
    k.add_argument("us", type=float); k.add_argument("gbs", type=float)
    k.add_argument("--pct", type=float); k.add_argument("--note", default="")

    gm = sub.add_parser("gemm"); gm.set_defaults(fn=cmd_gemm)
    gm.add_argument("commit_id"); gm.add_argument("name")
    gm.add_argument("m", type=int); gm.add_argument("n", type=int); gm.add_argument("k", type=int)
    gm.add_argument("layout"); gm.add_argument("batch", type=int)
    gm.add_argument("tflops", type=float); gm.add_argument("us", type=float)
    gm.add_argument("--blocks", type=int); gm.add_argument("--note", default="")

    cm = sub.add_parser("commit"); cm.set_defaults(fn=cmd_commit)
    cm.add_argument("hash"); cm.add_argument("title")
    cm.add_argument("--before", type=float); cm.add_argument("--after", type=float)
    cm.add_argument("--note", default="")

    fd = sub.add_parser("finding"); fd.set_defaults(fn=cmd_finding)
    fd.add_argument("category"); fd.add_argument("summary")
    fd.add_argument("--value", default=""); fd.add_argument("--status", default="confirmed")
    fd.add_argument("--note", default="")

    ia = sub.add_parser("isa"); ia.set_defaults(fn=cmd_isa)
    ia.add_argument("mnemonic"); ia.add_argument("state"); ia.add_argument("blocks")

    sh = sub.add_parser("show"); sh.set_defaults(fn=cmd_show)
    sh.add_argument("table", nargs="?")

    sub.add_parser("latest").set_defaults(fn=cmd_latest)

    sq = sub.add_parser("sql"); sq.set_defaults(fn=cmd_sql)
    sq.add_argument("query")

    oi = sub.add_parser("op-import"); oi.set_defaults(fn=cmd_op_import)
    oi.add_argument("registry")

    op = sub.add_parser("op"); op.set_defaults(fn=cmd_op)
    op.add_argument("op_id"); op.add_argument("impl_status")
    op.add_argument("--tflops", type=float); op.add_argument("--roofline", type=float)
    op.add_argument("--ref"); op.add_argument("--commit", dest="commit_id"); op.add_argument("--note")

    xp = sub.add_parser("experiment"); xp.set_defaults(fn=cmd_experiment)
    xp.add_argument("hypothesis")
    xp.add_argument("--op", dest="op_id"); xp.add_argument("--lever")
    xp.add_argument("--before", type=float); xp.add_argument("--after", type=float)
    xp.add_argument("--unit", default="tok/s"); xp.add_argument("--verdict", default="inprogress")
    xp.add_argument("--commit", dest="commit_id"); xp.add_argument("--note")

    sub.add_parser("roadmap").set_defaults(fn=cmd_roadmap)
    sub.add_parser("loop").set_defaults(fn=cmd_loop)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
