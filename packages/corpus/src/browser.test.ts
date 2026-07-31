import assert from "node:assert/strict";
import { afterEach, test } from "node:test";
import { DatabaseSync } from "node:sqlite";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { CorpusReader } from "./browser.js";
import { closeLedger, createCampaign, openLedger, seedLedger } from "./db.js";
import { canonicalJson, sha256Bytes, stableId } from "./hash.js";
import { emptyHumanReviewResponse } from "./review-contract.js";
import type { CampaignConfig, HumanReviewPacket, JsonValue } from "./types.js";

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

function reviewFixture(): { databasePath: string; blobPath: string; packet: HumanReviewPacket; sha256: string } {
  const databasePath = fixture();
  const home = join(databasePath, "..");
  const packet: HumanReviewPacket = {
    schemaVersion: 1,
    campaignSlug: "fixture-campaign",
    sessionId: "review_session_11111111-2222-4333-8444-555555555555",
    pass: "A",
    reviewerAlias: "fixture-reviewer",
    rubricSlug: "d5-human-adjudication",
    rubricVersion: 1,
    seed: "fixture-seed",
    createdAt: "2026-07-31T00:00:00.000Z",
    instructions: ["Review only model-visible messages."],
    assignments: [{
      assignmentId: "assignment_fixture",
      opaqueItemId: "opaque_fixture",
      candidateContentSha256: "a".repeat(64),
      candidate: {
        kind: "micro_dialogue",
        messages: [
          { role: "user", content: "What changed?" },
          { role: "assistant", content: "Only the current role, not the person." }
        ]
      },
      response: emptyHumanReviewResponse("A")
    }]
  };
  const bytes = JSON.stringify(packet);
  const sha256 = sha256Bytes(bytes);
  const relativePath = join("blobs", "sha256", sha256.slice(0, 2), sha256);
  const blobPath = join(home, relativePath);
  mkdirSync(join(blobPath, ".."), { recursive: true });
  writeFileSync(blobPath, bytes);
  const database = new DatabaseSync(databasePath);
  database.exec(`
    CREATE TABLE blob (
      sha256 TEXT PRIMARY KEY,
      byte_length INTEGER NOT NULL,
      media_type TEXT NOT NULL,
      relative_path TEXT NOT NULL,
      created_at TEXT NOT NULL
    ) STRICT;
    CREATE TABLE export_artifact (
      id TEXT PRIMARY KEY,
      format TEXT NOT NULL,
      blob_sha256 TEXT NOT NULL REFERENCES blob(sha256),
      manifest_json TEXT NOT NULL,
      created_at TEXT NOT NULL
    ) STRICT;
    CREATE TABLE review_assignment (
      id TEXT PRIMARY KEY,
      status TEXT NOT NULL,
      blindness_json TEXT NOT NULL
    ) STRICT;
  `);
  database.prepare("INSERT INTO blob VALUES (?, ?, 'application/json', ?, ?)")
    .run(sha256, Buffer.byteLength(bytes), relativePath, "2026-07-31T00:00:00.000Z");
  database.prepare("INSERT INTO export_artifact VALUES (?, 'human_review_packet_json', ?, ?, ?)")
    .run("export_fixture", sha256, JSON.stringify({ sessionId: packet.sessionId }), "2026-07-31T00:00:01.000Z");
  database.prepare("INSERT INTO review_assignment VALUES (?, 'assigned', ?)")
    .run("assignment_fixture", JSON.stringify({ sessionId: packet.sessionId, pass: "A" }));
  database.close();
  return { databasePath, blobPath, packet, sha256 };
}

async function reviewProgressFixture(): Promise<string> {
  const home = mkdtempSync(join(tmpdir(), "alpha-corpus-browser-progress-test-"));
  temporaryHomes.push(home);
  const ledger = await openLedger(home);
  await seedLedger(ledger);
  const config: CampaignConfig = {
    slug: "progress-campaign",
    purpose: "public review progress fixture",
    workerModel: "gpt-5.4",
    criticModel: "disabled",
    maxGenerationCalls: 0,
    maxReviewCalls: 0,
    itemsPerFamily: 2,
    artifactLimitBytes: 15 * 1024 * 1024 * 1024
  };
  const campaignId = await createCampaign(ledger, config);
  const family = await ledger.client.execute(
    "SELECT id FROM concept_family WHERE slug = 'role-versus-bearer'"
  );
  const familyId = String(family.rows[0]!["id"]);
  const actorId = stableId("actor", "human:fixture-reviewer");
  const candidateIds = ["one", "two"].map((key) => stableId("candidate", `${campaignId}:${familyId}:${key}`));
  const versionIds = candidateIds.map((candidateId) => stableId("candidatev", `${candidateId}:1`));
  const content = canonicalJson({
    itemKey: "progress-fixture",
    kind: "micro_dialogue",
    title: "Fixture",
    primaryLens: "social_ontology",
    secondaryLenses: [],
    transformation: "temporal_shift",
    intendedResponsePolicy: "Answer directly.",
    difficulty: "introductory",
    messages: [
      { role: "user", content: "Did the person change?" },
      { role: "assistant", content: "The role changed; the person persisted." }
    ],
    linguisticPair: null,
    generatorNotes: "fixture"
  } as JsonValue);
  const hidden = canonicalJson({
    requiredCommitments: ["The person persists."],
    prohibitedCommitments: ["The person ceased to exist."],
    preserve: ["Identity"],
    change: ["Role"],
    admissibleAnalyses: ["Role and bearer differ."],
    discriminatingEvidence: []
  } as JsonValue);
  await ledger.client.batch([
    {
      sql: "INSERT INTO actor(id, kind, display_name, created_at) VALUES (?, 'human', 'fixture-reviewer', '2026-07-31T00:00:00Z')",
      args: [actorId]
    },
    ...candidateIds.flatMap((candidateId, index) => [
      {
        sql: `INSERT INTO candidate
              (id, campaign_id, family_id, item_key, kind, status, created_at, updated_at)
              VALUES (?, ?, ?, ?, 'micro_dialogue', ?, '2026-07-31T00:00:00Z', '2026-07-31T00:00:00Z')`,
        args: [candidateId, campaignId, familyId, `progress-${index + 1}`,
          index === 0 ? "structurally_valid" : "structurally_rejected"]
      },
      {
        sql: `INSERT INTO candidate_version
              (id, candidate_id, version, content_json, hidden_contract_json, content_sha256, created_at)
              VALUES (?, ?, 1, ?, ?, ?, '2026-07-31T00:00:00Z')`,
        args: [versionIds[index]!, candidateId, content, hidden, sha256Bytes(content)]
      }
    ]),
    {
      sql: `INSERT INTO review_assignment
            (id, candidate_version_id, reviewer_actor_id, blindness_json, status, created_at, updated_at)
            VALUES ('assignment_progress', ?, ?, '{"pass":"A","sessionId":"fixture"}', 'assigned',
                    '2026-07-31T00:00:00Z', '2026-07-31T00:00:00Z')`,
      args: [versionIds[0]!, actorId]
    }
  ], "write");
  closeLedger(ledger);
  return ledger.paths.database;
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

test("reader lists and verifies the latest public human-review packet", () => {
  const fixture = reviewFixture();
  const reader = new CorpusReader(fixture.databasePath);
  try {
    const packets = reader.listReviewPackets();
    assert.equal(packets.length, 1);
    assert.deepEqual(packets[0], {
      sessionId: fixture.packet.sessionId,
      campaignSlug: "fixture-campaign",
      pass: "A",
      reviewerAlias: "fixture-reviewer",
      assignmentCount: 1,
      assignedCount: 1,
      completedCount: 0,
      createdAt: "2026-07-31T00:00:01.000Z",
      packetSha256: fixture.sha256
    });
    const loaded = reader.reviewPacket(fixture.packet.sessionId);
    assert.equal(loaded?.packet.assignments[0]?.opaqueItemId, "opaque_fixture");
    assert.equal(loaded?.packetSha256, fixture.sha256);
    assert.equal(reader.reviewPacket("../../alpha-corpus.sqlite"), null);
  } finally {
    reader.close();
  }
});

test("reader fails closed when a public review packet blob is modified", () => {
  const fixture = reviewFixture();
  writeFileSync(fixture.blobPath, "tampered");
  const reader = new CorpusReader(fixture.databasePath);
  try {
    assert.throws(() => reader.reviewPacket(fixture.packet.sessionId), /blob hash mismatch/);
  } finally {
    reader.close();
  }
});

test("reader reports the reviewer-scoped D5 pipeline without exposing candidate lineage", async () => {
  const reader = new CorpusReader(await reviewProgressFixture());
  try {
    assert.deepEqual(reader.reviewCampaignProgress("progress-campaign", "fixture-reviewer"), {
      campaignSlug: "progress-campaign",
      reviewerAlias: "fixture-reviewer",
      candidates: 2,
      families: 1,
      structuralRejections: 1,
      passA: { assigned: 1, completed: 0, total: 2 },
      hiddenRepeats: { assigned: 0, completed: 0, total: 2, stabilityRows: 0 },
      passB: { assigned: 0, completed: 0, total: 2 },
      passC: { assigned: 0, completed: 0, total: 1 },
      structuralDispositions: { completed: 0, total: 1 },
      passD: {
        assigned: 0,
        completed: 0,
        total: 1,
        adjudications: 0,
        executionAuthorizations: 0
      }
    });
    assert.equal(reader.reviewCampaignProgress("missing", "fixture-reviewer"), null);
    assert.equal(reader.reviewCampaignProgress("progress-campaign", "missing"), null);
  } finally {
    reader.close();
  }
});
