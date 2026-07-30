import assert from "node:assert/strict";
import { afterEach, test } from "node:test";
import { DatabaseSync } from "node:sqlite";
import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { CorpusReader } from "./browser.js";

const temporaryHomes: string[] = [];

function fixture(): string {
  const home = mkdtempSync(join(tmpdir(), "alpha-corpus-browser-test-"));
  temporaryHomes.push(home);
  const path = join(home, "fixture.sqlite");
  const database = new DatabaseSync(path);
  database.exec(`
    PRAGMA foreign_keys = ON;
    CREATE TABLE source (
      id TEXT PRIMARY KEY,
      title TEXT NOT NULL,
      created_at TEXT NOT NULL
    ) STRICT;
    CREATE TABLE candidate (
      id TEXT PRIMARY KEY,
      source_id TEXT NOT NULL REFERENCES source(id),
      status TEXT NOT NULL,
      metadata TEXT NOT NULL,
      created_at TEXT NOT NULL
    ) STRICT;
    CREATE VIEW candidate_current AS
      SELECT id, source_id, status, metadata, created_at FROM candidate;
    INSERT INTO source VALUES ('src-1', 'A source', '2026-07-30T00:00:00Z');
    INSERT INTO candidate VALUES
      ('cand-1', 'src-1', 'quarantine', '{"reason":"unreviewed"}', '2026-07-30T01:00:00Z'),
      ('cand-2', 'src-1', 'accepted', '{"reason":"human review"}', '2026-07-30T02:00:00Z');
  `);
  database.close();
  return path;
}

afterEach(() => {
  while (temporaryHomes.length > 0) {
    rmSync(temporaryHomes.pop()!, { recursive: true, force: true });
  }
});
test("reader discovers every table and view and searches their live schema", () => {
  const reader = new CorpusReader(fixture());
  try {
    assert.deepEqual(reader.safety(), { readOnly: true, queryOnly: true });
    assert.deepEqual(reader.listRelations().map((relation) => relation.name), [
      "candidate",
      "source",
      "candidate_current"
    ]);
    assert.deepEqual(reader.listRelations("status").map((relation) => relation.name), [
      "candidate",
      "candidate_current"
    ]);
    assert.equal(reader.relation("candidate").columns.length, 5);
  } finally {
    reader.close();
  }
});

test("reader paginates, filters, sorts, preserves full values, and reports lineage", () => {
  const reader = new CorpusReader(fixture());
  try {
    const page = reader.page("candidate", {
      page: 1,
      pageSize: 10,
      query: "human review",
      sortColumn: "created_at",
      sortDirection: "desc"
    });
    assert.equal(page.totalRows, 1);
    assert.equal(page.rows[0]?.["id"], "cand-2");
    assert.equal(page.rows[0]?.["metadata"], '{"reason":"human review"}');
    const detail = reader.relation("candidate");
    assert.equal(detail.outbound[0]?.targetRelation, "source");
    assert.equal(reader.relation("source").inbound[0]?.sourceRelation, "candidate");
  } finally {
    reader.close();
  }
});

test("relation and column identifiers must resolve through the live schema", () => {
  const path = fixture();
  const reader = new CorpusReader(path);
  try {
    assert.throws(() => reader.page('candidate"; DROP TABLE source; --'), /Unknown corpus relation/);
    const page = reader.page("candidate", { sortColumn: 'created_at"; DROP TABLE source; --' });
    assert.equal(page.sortColumn, null);
    assert.equal(reader.page("source").totalRows, 1);
  } finally {
    reader.close();
  }
});
