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

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
