# PRD-11 — Public Alpha Corpus explorer

- **Status:** implemented candidate; production release proof is recorded in Execution 02
- **Route:** `https://alpha.donto.org/corpus`
- **Audience:** the operator, collaborating research agents, dataset reviewers, and public researchers
- **Data authority:** the canonical Alpha Corpus SQLite ledger on the mounted research drive
- **Access policy:** public and read-only

## 1. Product outcome

Alpha Corpus is not merely a file used by a trainer. It is the scientific record of how synthetic
conversational material was proposed, generated, rejected, reviewed, released, rendered, and eventually shown
to a model. The public explorer must let a visitor inspect that record without installing the repository,
downloading SQLite, learning the schema in advance, or trusting a summary statistic.

The route succeeds when a visitor can:

1. discover every current table and view;
2. understand which relation is being inspected and whether it is a table or view;
3. search relation names and live column names;
4. page, filter, and sort rows without loading an unbounded result set;
5. inspect the complete untruncated value and its containing row;
6. see schema, indexes, and inbound or outbound foreign-key lineage;
7. distinguish candidates, reviews, public-ready material, release membership, and actual training exposure;
8. use the same surface by keyboard, touch, narrow screen, light theme, or dark theme; and
9. encounter no mutation control or server mutation path.

The explorer is a public research instrument. It must not turn generated material into an implied dataset
release, structural validity into human approval, or release membership into proof that a row was used for
training.

## 2. Scope

### 2.1 Included

- all non-internal SQLite tables and views discovered from the live schema;
- table/view type and logical row counts;
- column names, declared types, nullability, defaults, and primary-key position;
- declared indexes and indexed columns;
- foreign keys in both directions;
- bounded row search across the selected relation;
- allowlisted column sorting and bounded pagination;
- full values, including readable JSON, serialized big integers, and identified base64 blobs;
- row identity derived from declared primary-key columns when available;
- ledger modification time and explicit read-only status;
- a lifecycle strip linking the canonical scientific stages when those relations exist; and
- automatic discovery after a compatible ledger schema change, without a front-end table registry.

### 2.2 Excluded

- inserts, updates, deletes, retractions, adjudication, or any other mutation;
- arbitrary SQL supplied by the browser;
- hidden administration actions;
- treating blob paths as authorization to publish unlisted filesystem content;
- downloadable training exports whose release and licensing gates have not passed;
- authentication-dependent differences in which scientific tables are visible; and
- claims that unreviewed calibration candidates are accepted training data.

## 3. Information architecture

The `/corpus` route has four stable regions.

### 3.1 Research header

The header names the surface, states that it is public and read-only, explains the ledger boundary, and shows:

- discovered relation count;
- table/view split; and
- ledger modification time.

The wording must remain candid. The presence of rows is evidence of recorded work, not evidence of quality.

### 3.2 Scientific lifecycle strip

When present in the live schema, the explorer counts and links:

`candidate -> review -> public_training_candidate -> release_member -> training_exposure`

These are separate relations because they answer separate questions:

- Was something generated and retained, including as a rejection?
- Was it reviewed?
- Does it satisfy the current public-training eligibility view?
- Was it included in an immutable release?
- Was the rendered material actually exposed to a training run?

The lifecycle strip is an explanatory shortcut, not a replacement for the complete schema. Its links disappear
naturally if a named relation does not exist.

### 3.3 Relation navigator

On wide screens a persistent navigator groups tables and views and exposes each relation's column count. On
narrow screens the same navigator lives in a collapsed disclosure above the selected relation. Search covers:

- relation name;
- relation type; and
- live column names.

The complete relation list remains available; search is filtering, not authorization.

### 3.4 Relation workbench

Every relation exposes three views:

- **Rows:** filter, page, sort, and open complete cell values.
- **Schema:** inspect columns, indexes, and the SQLite definition.
- **Lineage:** follow declared inbound and outbound foreign keys.

The selected relation and view are URL-addressable. Reviewers can therefore share a stable link to the relevant
schema or lineage surface even though the underlying row order remains an explicit query choice.

## 4. Dynamic schema contract

The browser package reads `sqlite_schema` and safe PRAGMA table-valued functions. It must not maintain a
handwritten inventory of the ledger's tables or columns.

The metadata cache is keyed by SQLite's `schema_version`. On a schema change the next request rebuilds the
relation, column, index, and lineage inventory. This gives the public surface two important properties:

1. compatible new tables and views appear automatically; and
2. a user-supplied identifier cannot become SQL merely because it resembles a table or column name.

Every relation and sort column must first resolve through the current schema. Only the resolved identifier is
quoted into a query. Search text remains a bound value. Wildcard characters are escaped before the bounded
`LIKE` filter is constructed.

## 5. Query behavior

### 5.1 Pagination

- default: 25 rows;
- offered sizes: 10, 25, 50, and 100;
- server-enforced maximum: 100;
- invalid or excessive page numbers are clamped to a valid page; and
- each result reports matching rows, current page, and page count.

### 5.2 Filtering

The selected relation may be searched across all of its columns. The filter is intentionally simple and
transparent: it is a bounded inspection aid, not semantic retrieval. Empty filters return the complete
relation within pagination.

### 5.3 Sorting

Sorting is available only for discovered columns. Direction is either ascending or descending. An invalid
column falls back to a safe deterministic order rather than entering the SQL statement.

### 5.4 Full-value inspection

Table cells are deliberately compact in the grid. Activating a cell opens an inspector containing:

- relation and column name;
- declared value kind;
- the complete value without ellipsis;
- pretty-printed JSON when the value is valid JSON;
- a copy action;
- declared primary-key identity; and
- every value in the containing row.

The inspector uses the native modal-dialog model, moves focus to its close action, closes on Escape, and becomes
a full-width reading surface on small screens.

## 6. Read-only security model

Public access is intentionally anonymous because the corpus itself is meant to be inspectable. The security
boundary is therefore the absence of mutation authority, not an authentication wall.

Required controls:

1. open SQLite with the runtime's read-only flag;
2. set `PRAGMA query_only = ON` after connection;
3. set `PRAGMA trusted_schema = OFF`;
4. expose no generic execute/query endpoint;
5. resolve every relation and column through live schema metadata;
6. bind all browser-supplied values;
7. cap page size and normalize page numbers;
8. serialize values into inert React text, never raw HTML;
9. publish no filesystem endpoint for artifact paths; and
10. run the service as the unprivileged `ajax` user behind the existing reverse proxy.

The UI repeats `read only` because public legibility is part of the security contract, but labels are never
treated as enforcement.

## 7. Visual and interaction system

The explorer inherits the Alpha dashboard instead of creating an unrelated database-administration product.

- existing light and dark tokens;
- restrained blue only for selection, navigation, and focus;
- green reserved for the factual `Public · read only` state;
- dense, legible rows rather than decorative cards;
- monospace for identifiers and stored values;
- prose typography for explanations and action labels;
- visible keyboard focus and 44px-class primary controls;
- no status communicated only through color;
- no global horizontal page scrolling; and
- horizontal scrolling contained to lifecycle or scientific-table regions.

The design source of truth is root `PRODUCT.md`. The explorer's composition follows the selected
instrument-panel direction: relation navigator, row workbench, and dedicated cell inspector.

## 8. Accessibility contract

- semantic heading order and landmark labels;
- labelled search, filter, page-size, sort, view, and pagination controls;
- table caption describing current relation and page;
- keyboard-accessible cells and relation links;
- native dialog semantics and Escape close;
- light/dark muted-text contrast at WCAG AA for normal text;
- no document-level overflow at 390 CSS pixels;
- mobile schema disclosure usable without hover;
- visible focus in both themes; and
- readable no-results and error states.

## 9. Reliability and performance

The ledger is small today but the interface must not assume it remains small.

- schema inventory is cached until SQLite `schema_version` changes;
- relation rows are never selected without a limit;
- count and page queries are separate and bounded;
- only one selected relation's detail is rendered;
- the service opens the database locally rather than copying research data into a web bundle;
- schema additions require no redeploy when they use supported SQLite relations; and
- an unavailable or empty ledger produces an explicit state rather than a fabricated sample.

Node's built-in SQLite module is experimental in the deployed Node line. This is an operational watch item,
not a hidden fallback: the route has direct regression tests, and a future stable adapter can replace the
reader without changing the public contract.

## 10. Deployment contract

- build the repository's Next.js application in standalone mode;
- package the standalone server and static assets into an immutable release directory;
- record the git revision beside the release;
- run `alpha-corpus-web.service` on loopback port 3104 as `ajax`;
- pass the canonical ledger only through `CORPUS_DB_PATH`;
- proxy `alpha.donto.org` through Caddy;
- use the established Cloudflare-to-origin TLS pattern;
- validate Caddy before reload;
- verify origin routing with `--resolve` before relying on public DNS; and
- retain the previous release path for rollback.

The deployment publishes the existing Alpha dashboard as well as `/corpus`. The corpus is a first-class route,
not a separate visual microsite.

## 11. Acceptance gates

### G11.1 — complete discovery

- every live non-internal table and view appears;
- table and view counts match independent SQLite census; and
- a newly created compatible relation appears after schema-version change without code modification.

### G11.2 — bounded inspection

- row filter, allowlisted sort, 10/25/50/100 pagination, and page clamping pass;
- full values round-trip in the inspector; and
- blobs and large integer values serialize deterministically.

### G11.3 — lineage

- indexes and both foreign-key directions are visible; and
- a known relation with inbound and outbound references shows both.

### G11.4 — public read-only boundary

- a direct write on the same connection fails;
- injection-shaped relation and column identifiers are rejected;
- no mutation UI or mutation route exists; and
- the canonical candidate table remains intact after adversarial queries.

### G11.5 — responsive accessibility

- desktop light and dark surfaces receive visual inspection;
- 390x844 receives visual inspection;
- schema navigation and cell inspection work at the narrow viewport;
- document overflow is false; and
- browser console and page errors are empty.

### G11.6 — production proof

- service starts and remains active;
- loopback route returns the live ledger;
- Caddy route returns the live ledger;
- public `https://alpha.donto.org/corpus` returns 200;
- public browser interaction succeeds; and
- deployed revision matches the pushed repository revision.

## 12. Honest limitations

- Search is lexical and relation-local; it is not Donto alignment or semantic retrieval.
- Row count is not a quality measure.
- Zero reviews, releases, or exposures are legitimate visible states.
- SQLite declarations cannot express every semantic dependency in the research program; lineage shows declared
  relational structure, not every conceptual relationship.
- Public visibility does not itself grant a dataset license beyond the repository and release metadata.
- The current calibration is quarantined pending human conceptual adjudication.

## 13. Future extensions that require separate decisions

- release-manifest downloads after release/licensing gates pass;
- stable row permalinks for composite-key records;
- artifact previews subject to explicit public-blob policy;
- schema diagrams derived from the same live relation inventory;
- saved public queries without user tracking;
- diff views between immutable releases; and
- Donto-powered semantic discovery layered above, never substituted for, the exact ledger.
