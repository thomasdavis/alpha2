# Execution 02 — Public Alpha Corpus explorer

- **Date:** 2026-07-30
- **Contract:** [PRD-11](PRD-11-PUBLIC-CORPUS-EXPLORER.md)
- **Operator direction:** publish every ledger table and view read-only at `alpha.donto.org/corpus` while
  retaining the existing Alpha dashboard
- **Training/GPU activity:** none

## 1. Outcome

The Alpha web application now has a first-class public `/corpus` route backed directly by the canonical SQLite
scientific ledger. It discovers the live schema instead of enumerating research relations in front-end code.
Every current table and view is publicly navigable, with bounded rows, schema, indexes, declared lineage, and
full-value inspection.

The implementation retains the distinctions on which the synthetic-curriculum program depends: generated,
rejected, reviewed, public-ready, released, and training-exposed material are not collapsed into a single
count or status.

## 2. Implemented components

### 2.1 Read-only browser library

`packages/corpus/src/browser.ts` adds a narrow `CorpusReader`:

- Node `DatabaseSync` opened with `readOnly: true`;
- `query_only`, `trusted_schema`, and bounded busy-timeout pragmas;
- live table/view discovery;
- live column, index, and foreign-key discovery;
- schema-version-aware metadata caching;
- exact identifier resolution and quoting;
- bound and escaped search values;
- allowlisted sorting;
- clamped pagination;
- safe blob and bigint serialization; and
- no mutation or arbitrary-SQL method.

The package exports this reader through its existing public module.

### 2.2 Server route

`apps/web/src/app/corpus/page.tsx` is a dynamic Node route. It selects the requested relation and view from the
live schema, obtains a bounded row page, derives available lifecycle counts, and sends inert serializable data
to the explorer component.

`apps/web/src/lib/corpus.ts` owns the process-local read-only reader and binds it to `CORPUS_DB_PATH`, defaulting
to the canonical Alpha Corpus ledger when running on this host.

### 2.3 Explorer interface

`apps/web/src/app/corpus/corpus-explorer.tsx` implements:

- persistent desktop and collapsible mobile relation navigation;
- table/view grouping and schema-name search;
- lifecycle-stage counts and links;
- relation-local filtering;
- page-size selection, sort links, and pagination;
- rows, schema, and lineage tabs;
- native modal cell inspection with complete row context and copy;
- light/dark Alpha styling;
- explicit read-only wording; and
- keyboard and responsive behavior.

Navigation was added to the existing sidebar. Loading and error boundaries preserve the route's product shape
during slow or failed reads.

## 3. Independent ledger census

At the acceptance pass the canonical ledger contained:

| Measure | Count |
|---|---:|
| Tables | 106 |
| Views | 4 |
| Total public relations | 110 |
| Candidates | 48 |
| Reviews | 0 |
| Release members | 0 |
| Training exposures | 0 |

The empty later stages are intentionally visible. The 48 calibration candidates remain quarantined; this
release does not adjudicate or promote them.

The complete Alpha Corpus artifact tree measured 4.8 MiB, far below the operator's 15 GiB soft pause for new
corpus generation. No generation occurred during this execution.

## 4. Automated verification

### 4.1 Corpus package

Nine tests passed:

- six existing migration, immutability, artifact, validation, provenance, and resumable-storage tests; and
- three new browser tests covering complete discovery, schema search, pagination, filtering, sorting, full
  values, lineage, identifier validation, and injection resistance.

### 4.2 Type and production build

- Alpha web TypeScript check: pass;
- full filtered monorepo build: 12/12 packages successful; and
- Next production output includes `/corpus` as a dynamic server-rendered route.

The build retained two non-blocking pre-existing warnings: the Next middleware naming deprecation and broad
file tracing from the existing `server-state` import path. Node also marks the built-in SQLite module
experimental. None created a runtime error in the explorer path.

### 4.3 Security checks

- direct mutation through a read-only connection failed with `attempt to write a readonly database`;
- identifier-shaped injection inputs did not execute and the 48 candidate rows remained intact;
- only identifiers resolved from current schema metadata can reach query construction; and
- the browser exposes no mutation control or generic query endpoint;
- the public proxy returns 404 for every legacy `/api/*`, `/v1/*`, and `/chat/completions` route; and
- the public proxy returns 405 for non-GET/HEAD requests, including `POST /corpus`.

## 5. Browser acceptance

The production standalone build was tested with a real Chromium runtime.

### Desktop

- 1440x900;
- light theme inspected;
- dark theme inspected;
- relation search returned name- and column-matched relations;
- row filtering reduced candidate rows to the expected matching stratum;
- full-value dialog opened with relation, column, value, identity, and complete row;
- schema view exposed columns, indexes, and SQL definition;
- `model_call` lineage showed eight outbound and five inbound declared references; and
- no browser page errors or Next error overlay appeared.

### Mobile

- 390x844;
- no document-level horizontal overflow;
- lifecycle and wide table scrolling remained locally contained;
- relation navigation collapsed into an accessible disclosure;
- full-value inspection became a full-width reading surface; and
- Escape closed the native dialog.

### Measured local production profile

- TTFB: 44.6 ms;
- FCP: 136 ms;
- LCP: 552 ms; and
- CLS: 0.

These figures describe the warm local production route on this host; they are not a public-network SLA.

### Measured public production profile

The same Chromium profiler against the live Cloudflare URL measured:

- TTFB: 87.3 ms;
- FCP: 172 ms;
- LCP: 340 ms; and
- CLS: 0.

The live browser repeated the desktop and mobile interaction checks, including relation filtering and the
full-width mobile cell inspector. The public route had no document overflow, Next error overlay, or page error.

## 6. Deployment record

The production gate passed on 2026-07-30.

- Web implementation revision: `569eb2c` (`feat(corpus): add public read-only ledger explorer`)
- Service/Caddy contract revision: `5a30549`
- Public write/API hardening revision: `7331b90`
- Immutable release directory:
  `/home/ajax/alpha2-web-releases/5a305495b329d87af1362ac09148470899c14552`
- Current release pointer: `/home/ajax/alpha2-web-current`
- Service: `alpha-corpus-web.service`, enabled and active as unprivileged user `ajax`
- Service unit hash: `60316fc902d74430922ad9b7c1688704e9f822be3963694ea1cda1fa09d71f40`
- Loopback: `127.0.0.1:3104`, HTTP 200
- Caddy origin proof with local resolution: HTTPS 200 in 143 ms
- Cloudflare DNS: proxied `A alpha.donto.org -> 15.235.185.42`
- Independent DNS resolution: both `1.1.1.1` and `8.8.8.8` returned the Cloudflare edge addresses
- Public URL: `https://alpha.donto.org/corpus`, HTTP/2 200 through Cloudflare and Caddy
- Public HTML proof: `Alpha Corpus`, `Public · read only`, and 110 discovered relations
- Post-release service state: active/running, zero restarts, 95.3 MB current and 146.8 MB peak memory

Four post-release public journeys returned HTTP 200: the default corpus, `model_call` lineage, candidate schema,
and a filtered rejected-candidate page. Warm response times ranged from 97 ms to 1.50 seconds; the slowest was
the lineage path that derives all inbound foreign keys.

The standalone build logs one pre-existing dashboard warning because no Turso application database is attached
to this public read-only service. The corpus uses its separately declared SQLite path and is healthy. Caddy
blocks every historical application API at this hostname, so the missing dashboard database cannot become a
public mutation fallback.

## 7. Operational runbook

### Inspect service

```bash
systemctl status alpha-corpus-web.service --no-pager -l
journalctl -u alpha-corpus-web.service -n 100 --no-pager
curl -fsS http://127.0.0.1:3104/corpus >/dev/null
```

### Verify the ledger boundary

```bash
systemctl show alpha-corpus-web.service -p Environment
sqlite3 /mnt/donto-data/donto-resources/research/alpha2-corpus/alpha-corpus.sqlite \
  "SELECT count(*) FROM candidate;"
```

### Verify public routing

```bash
curl -fsSI https://alpha.donto.org/corpus
```

### Roll back

1. point the service's release path to the prior immutable release;
2. run `systemctl daemon-reload` if the unit changed;
3. restart the scoped service;
4. prove loopback and Caddy routing; and
5. preserve the rejected release and its logs for diagnosis.

The SQLite ledger is not migrated or copied as part of a web rollback.

## 8. Boundaries retained

- No Alpha training or RunPod work was authorized or performed.
- No new synthetic generation was performed.
- No candidate was human-approved, released, or exposed to training.
- No Donto substrate data was mutated.
- No global disk cleanup was performed.
- The public browser reads exact scientific state; it does not manufacture a friendlier demonstration state.
