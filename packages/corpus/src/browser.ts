import { DatabaseSync, type SQLInputValue } from "node:sqlite";
import { readFileSync } from "node:fs";
import { dirname, resolve, sep } from "node:path";
import { sha256Bytes } from "./hash.js";
import {
  HUMAN_REVIEW_RUBRIC_SLUG,
  HUMAN_REVIEW_RUBRIC_VERSION,
  parseHumanReviewPacketText
} from "./review-contract.js";
import { resolveLedgerPaths } from "./storage.js";
import type { HumanReviewPacket, HumanReviewPass } from "./types.js";

export type CorpusRelationKind = "table" | "view";
export type CorpusCellValue = string | number | null;

export interface CorpusColumn {
  position: number;
  name: string;
  type: string;
  notNull: boolean;
  defaultValue: string | null;
  primaryKeyPosition: number;
  hidden: number;
}

export interface CorpusIndex {
  name: string;
  unique: boolean;
  origin: string;
  partial: boolean;
  columns: string[];
}

export interface CorpusForeignKey {
  id: number;
  sequence: number;
  targetRelation: string;
  sourceColumn: string;
  targetColumn: string;
  onUpdate: string;
  onDelete: string;
}

export interface CorpusInboundReference {
  sourceRelation: string;
  sourceColumn: string;
  targetColumn: string;
  onUpdate: string;
  onDelete: string;
}

export interface CorpusRelationSummary {
  name: string;
  kind: CorpusRelationKind;
  sql: string | null;
  columns: CorpusColumn[];
}

export interface CorpusRelationDetail extends CorpusRelationSummary {
  indexes: CorpusIndex[];
  outbound: CorpusForeignKey[];
  inbound: CorpusInboundReference[];
}

export interface CorpusPage {
  relation: CorpusRelationSummary;
  rows: Record<string, CorpusCellValue>[];
  totalRows: number;
  page: number;
  pageSize: number;
  pageCount: number;
  query: string;
  sortColumn: string | null;
  sortDirection: "asc" | "desc";
  hasPreviousPage: boolean;
  hasNextPage: boolean;
}

export interface CorpusPageRequest {
  page?: number;
  pageSize?: number;
  query?: string;
  sortColumn?: string;
  sortDirection?: "asc" | "desc";
}

export interface CorpusReviewPacketSummary {
  sessionId: string;
  campaignSlug: string;
  pass: HumanReviewPass;
  reviewerAlias: string;
  assignmentCount: number;
  assignedCount: number;
  completedCount: number;
  createdAt: string;
  packetSha256: string;
}

export interface CorpusReviewPacket {
  packet: HumanReviewPacket;
  packetSha256: string;
  exportedAt: string;
}

export interface CorpusReviewStageProgress {
  assigned: number;
  completed: number;
  total: number;
}

export interface CorpusReviewCampaignProgress {
  campaignSlug: string;
  reviewerAlias: string;
  candidates: number;
  families: number;
  structuralRejections: number;
  passA: CorpusReviewStageProgress;
  hiddenRepeats: CorpusReviewStageProgress & { stabilityRows: number };
  passB: CorpusReviewStageProgress;
  passC: CorpusReviewStageProgress;
  structuralDispositions: { completed: number; total: number };
  passD: CorpusReviewStageProgress & {
    adjudications: number;
    executionAuthorizations: number;
  };
}

interface SchemaRow {
  name: string;
  type: "table" | "view";
  sql: string | null;
}

interface ColumnRow {
  cid: number;
  name: string;
  type: string;
  not_null: number;
  dflt_value: string | null;
  pk: number;
  hidden: number;
}

interface IndexRow {
  name: string;
  unique: number;
  origin: string;
  partial: number;
}

interface IndexColumnRow {
  name: string | null;
}

interface ForeignKeyRow {
  id: number;
  seq: number;
  table: string;
  from: string;
  to: string;
  on_update: string;
  on_delete: string;
}

interface ReviewPacketRow {
  blob_sha256: string;
  relative_path: string;
  created_at: string;
}

interface ReviewCampaignProgressRow {
  candidate_count: number | bigint;
  family_count: number | bigint;
  structural_rejection_count: number | bigint;
  pass_a_assigned: number | bigint;
  pass_a_completed: number | bigint;
  repeat_assigned: number | bigint;
  repeat_completed: number | bigint;
  repeat_stability_rows: number | bigint;
  pass_b_assigned: number | bigint;
  pass_b_completed: number | bigint;
  pass_c_assigned: number | bigint;
  pass_c_completed: number | bigint;
  structural_dispositions: number | bigint;
  pass_d_assigned: number | bigint;
  pass_d_completed: number | bigint;
  pass_d_adjudications: number | bigint;
  execution_authorizations: number | bigint;
}

const DEFAULT_PAGE_SIZE = 25;
const MAX_PAGE_SIZE = 100;
const MAX_SEARCH_LENGTH = 256;

function quoteIdentifier(identifier: string): string {
  return `"${identifier.replaceAll('"', '""')}"`;
}

function clampInteger(value: number | undefined, fallback: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return fallback;
  return Math.max(min, Math.min(max, Math.floor(value!)));
}

function escapeLike(value: string): string {
  return value.replaceAll("\\", "\\\\").replaceAll("%", "\\%").replaceAll("_", "\\_");
}

function serializableValue(value: unknown): CorpusCellValue {
  if (value == null) return null;
  if (typeof value === "string" || typeof value === "number") return value;
  if (typeof value === "bigint") return value.toString();
  if (value instanceof Uint8Array) return `base64:${Buffer.from(value).toString("base64")}`;
  if (value instanceof ArrayBuffer) return `base64:${Buffer.from(value).toString("base64")}`;
  return String(value);
}

/**
 * A deliberately narrow, read-only interface to the Alpha Corpus ledger.
 *
 * The SQLite file is opened with readOnly=true and PRAGMA query_only=ON. The
 * class exposes no arbitrary-SQL or mutation method. Every interpolated
 * identifier must first resolve through sqlite_schema / table_xinfo, while all
 * user-provided values remain bound parameters.
 */
export class CorpusReader {
  readonly databasePath: string;
  private readonly database: DatabaseSync;
  private cachedSchemaVersion = -1;
  private cachedRelations: CorpusRelationSummary[] = [];
  private readonly cachedIndexes = new Map<string, CorpusIndex[]>();
  private readonly cachedForeignKeys = new Map<string, CorpusForeignKey[]>();

  constructor(databasePath = resolveLedgerPaths().database) {
    this.databasePath = databasePath;
    this.database = new DatabaseSync(databasePath, {
      readOnly: true,
      enableForeignKeyConstraints: true,
      enableDoubleQuotedStringLiterals: false
    });
    this.database.exec("PRAGMA query_only = ON");
    this.database.exec("PRAGMA trusted_schema = OFF");
    this.database.exec("PRAGMA busy_timeout = 2000");
  }

  close(): void {
    this.database.close();
  }

  safety(): { readOnly: true; queryOnly: boolean } {
    const row = this.database.prepare("PRAGMA query_only").get() as { query_only: number };
    return { readOnly: true, queryOnly: row.query_only === 1 };
  }

  listRelations(search = ""): CorpusRelationSummary[] {
    const needle = search.trim().toLocaleLowerCase().slice(0, MAX_SEARCH_LENGTH);
    return this.allRelations()
      .filter((relation) => {
        if (!needle) return true;
        return relation.name.toLocaleLowerCase().includes(needle)
          || relation.kind.includes(needle)
          || relation.columns.some((column) =>
            column.name.toLocaleLowerCase().includes(needle)
            || column.type.toLocaleLowerCase().includes(needle)
          );
      });
  }

  relation(name: string): CorpusRelationDetail {
    const relation = this.requireRelation(name);
    return {
      ...relation,
      indexes: relation.kind === "table" ? this.indexesFor(name) : [],
      outbound: this.foreignKeysFor(name),
      inbound: this.inboundReferencesFor(name)
    };
  }

  page(name: string, request: CorpusPageRequest = {}): CorpusPage {
    const relation = this.requireRelation(name);
    const pageSize = clampInteger(request.pageSize, DEFAULT_PAGE_SIZE, 10, MAX_PAGE_SIZE);
    const requestedPage = clampInteger(request.page, 1, 1, Number.MAX_SAFE_INTEGER);
    const query = (request.query ?? "").trim().slice(0, MAX_SEARCH_LENGTH);
    const requestedSort = request.sortColumn ?? "";
    const sortableColumn = relation.columns.find((column) => column.name === requestedSort)?.name ?? null;
    const sortDirection = request.sortDirection === "desc" ? "desc" : "asc";
    const relationSql = quoteIdentifier(relation.name);

    const where = query && relation.columns.length > 0
      ? `WHERE (${relation.columns.map((column) =>
        `CAST(${quoteIdentifier(column.name)} AS TEXT) LIKE ? ESCAPE '\\' COLLATE NOCASE`
      ).join(" OR ")})`
      : "";
    const searchValue = `%${escapeLike(query)}%`;
    const whereParameters: SQLInputValue[] = query
      ? relation.columns.map(() => searchValue)
      : [];

    const countRow = this.database.prepare(
      `SELECT COUNT(*) AS count FROM ${relationSql} ${where}`
    ).get(...whereParameters) as { count: number | bigint };
    const totalRows = Number(countRow.count);
    const pageCount = Math.max(1, Math.ceil(totalRows / pageSize));
    const page = Math.min(requestedPage, pageCount);
    const offset = (page - 1) * pageSize;
    const orderBy = this.orderClause(relation, sortableColumn, sortDirection);

    const rows = this.database.prepare(
      `SELECT * FROM ${relationSql} ${where} ${orderBy} LIMIT ? OFFSET ?`
    ).all(...whereParameters, pageSize, offset) as unknown as Record<string, unknown>[];

    return {
      relation,
      rows: rows.map((row) => Object.fromEntries(
        Object.entries(row).map(([key, value]) => [key, serializableValue(value)])
      )),
      totalRows,
      page,
      pageSize,
      pageCount,
      query,
      sortColumn: sortableColumn,
      sortDirection,
      hasPreviousPage: page > 1,
      hasNextPage: page < pageCount
    };
  }

  listReviewPackets(): CorpusReviewPacketSummary[] {
    if (!this.allRelations().some((relation) => relation.name === "export_artifact")
      || !this.allRelations().some((relation) => relation.name === "blob")) return [];
    const rows = this.database.prepare(`
      SELECT ea.blob_sha256, b.relative_path, ea.created_at
      FROM export_artifact ea
      JOIN blob b ON b.sha256 = ea.blob_sha256
      WHERE ea.format = 'human_review_packet_json'
      ORDER BY ea.created_at DESC, ea.id DESC
    `).all() as unknown as ReviewPacketRow[];
    const sessions = new Map<string, CorpusReviewPacketSummary>();
    for (const row of rows) {
      const loaded = this.loadReviewPacketRow(row);
      const packet = loaded.packet;
      if (sessions.has(packet.sessionId)) continue;
      const states = this.database.prepare(`
        SELECT status, COUNT(*) AS count
        FROM review_assignment
        WHERE json_extract(blindness_json, '$.sessionId') = ?
        GROUP BY status
      `).all(packet.sessionId) as unknown as Array<{ status: string; count: number | bigint }>;
      const stateCounts = Object.fromEntries(states.map((state) => [state.status, Number(state.count)]));
      sessions.set(packet.sessionId, {
        sessionId: packet.sessionId,
        campaignSlug: packet.campaignSlug,
        pass: packet.pass,
        reviewerAlias: packet.reviewerAlias,
        assignmentCount: packet.assignments.length,
        assignedCount: stateCounts["assigned"] ?? 0,
        completedCount: stateCounts["completed"] ?? 0,
        createdAt: loaded.exportedAt,
        packetSha256: loaded.packetSha256
      });
    }
    return [...sessions.values()];
  }

  reviewCampaignProgress(campaignSlug: string, reviewerAlias: string): CorpusReviewCampaignProgress | null {
    const requiredRelations = [
      "actor",
      "adjudication",
      "campaign_closeout",
      "campaign_closeout_assignment",
      "candidate",
      "candidate_version",
      "corpus_candidate_current",
      "family_synthesis",
      "family_synthesis_assignment",
      "generation_campaign",
      "review_assignment",
      "review_presentation",
      "review_presentation_session",
      "review_repeat_stability",
      "structural_disposition"
    ];
    const available = new Set(this.allRelations().map((relation) => relation.name));
    if (requiredRelations.some((relation) => !available.has(relation))) return null;

    const row = this.database.prepare(`
      WITH scope AS (
        SELECT gc.id AS campaign_id, a.id AS actor_id
        FROM generation_campaign gc
        JOIN actor a ON a.kind = 'human' AND a.display_name = ?
        WHERE gc.slug = ?
      )
      SELECT
        (SELECT COUNT(*) FROM corpus_candidate_current cc
          WHERE cc.campaign_id = scope.campaign_id) AS candidate_count,
        (SELECT COUNT(DISTINCT cc.family_id) FROM corpus_candidate_current cc
          WHERE cc.campaign_id = scope.campaign_id) AS family_count,
        (SELECT COUNT(*) FROM corpus_candidate_current cc
          WHERE cc.campaign_id = scope.campaign_id AND cc.status = 'structurally_rejected')
          AS structural_rejection_count,
        (SELECT COUNT(*) FROM review_assignment ra
          JOIN candidate_version cv ON cv.id = ra.candidate_version_id
          JOIN candidate c ON c.id = cv.candidate_id
          WHERE c.campaign_id = scope.campaign_id AND ra.reviewer_actor_id = scope.actor_id
            AND json_extract(ra.blindness_json, '$.pass') = 'A' AND ra.status = 'assigned')
          AS pass_a_assigned,
        (SELECT COUNT(*) FROM review_assignment ra
          JOIN candidate_version cv ON cv.id = ra.candidate_version_id
          JOIN candidate c ON c.id = cv.candidate_id
          WHERE c.campaign_id = scope.campaign_id AND ra.reviewer_actor_id = scope.actor_id
            AND json_extract(ra.blindness_json, '$.pass') = 'A' AND ra.status = 'completed')
          AS pass_a_completed,
        (SELECT COUNT(*) FROM review_presentation rp
          JOIN review_presentation_session rps ON rps.id = rp.session_id
          WHERE rps.campaign_id = scope.campaign_id AND rps.reviewer_actor_id = scope.actor_id
            AND rp.presentation_kind = 'hidden_repeat' AND rp.status = 'assigned')
          AS repeat_assigned,
        (SELECT COUNT(*) FROM review_presentation rp
          JOIN review_presentation_session rps ON rps.id = rp.session_id
          WHERE rps.campaign_id = scope.campaign_id AND rps.reviewer_actor_id = scope.actor_id
            AND rp.presentation_kind = 'hidden_repeat' AND rp.status = 'completed')
          AS repeat_completed,
        (SELECT COUNT(*) FROM review_repeat_stability rrs
          JOIN review_presentation rp ON rp.id = rrs.presentation_id
          JOIN review_presentation_session rps ON rps.id = rp.session_id
          WHERE rps.campaign_id = scope.campaign_id AND rps.reviewer_actor_id = scope.actor_id)
          AS repeat_stability_rows,
        (SELECT COUNT(*) FROM review_assignment ra
          JOIN candidate_version cv ON cv.id = ra.candidate_version_id
          JOIN candidate c ON c.id = cv.candidate_id
          WHERE c.campaign_id = scope.campaign_id AND ra.reviewer_actor_id = scope.actor_id
            AND json_extract(ra.blindness_json, '$.pass') = 'B' AND ra.status = 'assigned')
          AS pass_b_assigned,
        (SELECT COUNT(*) FROM review_assignment ra
          JOIN candidate_version cv ON cv.id = ra.candidate_version_id
          JOIN candidate c ON c.id = cv.candidate_id
          WHERE c.campaign_id = scope.campaign_id AND ra.reviewer_actor_id = scope.actor_id
            AND json_extract(ra.blindness_json, '$.pass') = 'B' AND ra.status = 'completed')
          AS pass_b_completed,
        (SELECT COUNT(*) FROM family_synthesis_assignment fsa
          WHERE fsa.campaign_id = scope.campaign_id AND fsa.reviewer_actor_id = scope.actor_id
            AND fsa.status = 'assigned') AS pass_c_assigned,
        (SELECT COUNT(*) FROM family_synthesis_assignment fsa
          WHERE fsa.campaign_id = scope.campaign_id AND fsa.reviewer_actor_id = scope.actor_id
            AND fsa.status = 'completed') AS pass_c_completed,
        (SELECT COUNT(*) FROM structural_disposition sd
          JOIN family_synthesis fs ON fs.id = sd.family_synthesis_id
          JOIN family_synthesis_assignment fsa ON fsa.id = fs.assignment_id
          WHERE fsa.campaign_id = scope.campaign_id AND fsa.reviewer_actor_id = scope.actor_id)
          AS structural_dispositions,
        (SELECT COUNT(*) FROM campaign_closeout_assignment cca
          WHERE cca.campaign_id = scope.campaign_id AND cca.adjudicator_actor_id = scope.actor_id
            AND cca.status = 'assigned') AS pass_d_assigned,
        (SELECT COUNT(*) FROM campaign_closeout_assignment cca
          WHERE cca.campaign_id = scope.campaign_id AND cca.adjudicator_actor_id = scope.actor_id
            AND cca.status = 'completed') AS pass_d_completed,
        (SELECT COUNT(*) FROM adjudication a
          JOIN campaign_closeout cc
            ON cc.id = CASE WHEN json_valid(a.rationale)
              THEN json_extract(a.rationale, '$.campaignCloseoutId') END
          WHERE cc.campaign_id = scope.campaign_id AND cc.adjudicator_actor_id = scope.actor_id)
          AS pass_d_adjudications,
        (SELECT COUNT(*) FROM campaign_closeout cc
          WHERE cc.campaign_id = scope.campaign_id AND cc.adjudicator_actor_id = scope.actor_id
            AND cc.execution_authorized <> 0) AS execution_authorizations
      FROM scope
    `).get(reviewerAlias, campaignSlug) as ReviewCampaignProgressRow | undefined;
    if (!row) return null;

    const candidates = Number(row.candidate_count);
    const families = Number(row.family_count);
    const structuralRejections = Number(row.structural_rejection_count);
    return {
      campaignSlug,
      reviewerAlias,
      candidates,
      families,
      structuralRejections,
      passA: {
        assigned: Number(row.pass_a_assigned),
        completed: Number(row.pass_a_completed),
        total: candidates
      },
      hiddenRepeats: {
        assigned: Number(row.repeat_assigned),
        completed: Number(row.repeat_completed),
        total: Math.min(6, candidates),
        stabilityRows: Number(row.repeat_stability_rows)
      },
      passB: {
        assigned: Number(row.pass_b_assigned),
        completed: Number(row.pass_b_completed),
        total: candidates
      },
      passC: {
        assigned: Number(row.pass_c_assigned),
        completed: Number(row.pass_c_completed),
        total: families
      },
      structuralDispositions: {
        completed: Number(row.structural_dispositions),
        total: structuralRejections
      },
      passD: {
        assigned: Number(row.pass_d_assigned),
        completed: Number(row.pass_d_completed),
        total: 1,
        adjudications: Number(row.pass_d_adjudications),
        executionAuthorizations: Number(row.execution_authorizations)
      }
    };
  }

  reviewPacket(sessionId: string): CorpusReviewPacket | null {
    if (!/^review_session_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(sessionId)) {
      return null;
    }
    if (!this.allRelations().some((relation) => relation.name === "export_artifact")
      || !this.allRelations().some((relation) => relation.name === "blob")) return null;
    const row = this.database.prepare(`
      SELECT ea.blob_sha256, b.relative_path, ea.created_at
      FROM export_artifact ea
      JOIN blob b ON b.sha256 = ea.blob_sha256
      WHERE ea.format = 'human_review_packet_json'
        AND json_extract(ea.manifest_json, '$.sessionId') = ?
      ORDER BY ea.created_at DESC, ea.id DESC
      LIMIT 1
    `).get(sessionId) as ReviewPacketRow | undefined;
    return row ? this.loadReviewPacketRow(row) : null;
  }

  private loadReviewPacketRow(row: ReviewPacketRow): CorpusReviewPacket {
    const corpusRoot = resolve(dirname(this.databasePath));
    const absolutePath = resolve(corpusRoot, row.relative_path);
    if (!absolutePath.startsWith(`${corpusRoot}${sep}`)) {
      throw new Error(`Review packet blob escapes corpus root: ${row.relative_path}`);
    }
    const bytes = readFileSync(absolutePath);
    const actualSha256 = sha256Bytes(bytes);
    if (actualSha256 !== row.blob_sha256) {
      throw new Error(`Review packet blob hash mismatch: expected ${row.blob_sha256}, received ${actualSha256}`);
    }
    const packet = parseHumanReviewPacketText(bytes.toString("utf8"));
    if (packet.rubricSlug !== HUMAN_REVIEW_RUBRIC_SLUG
      || packet.rubricVersion < 1
      || packet.rubricVersion > HUMAN_REVIEW_RUBRIC_VERSION) {
      throw new Error(`Unsupported human-review rubric ${packet.rubricSlug} v${packet.rubricVersion}`);
    }
    return { packet, packetSha256: row.blob_sha256, exportedAt: row.created_at };
  }

  private requireRelation(name: string): CorpusRelationSummary {
    const relation = this.allRelations().find((candidate) => candidate.name === name);
    if (!relation) throw new Error(`Unknown corpus relation: ${name}`);
    return relation;
  }

  private allRelations(): CorpusRelationSummary[] {
    const versionRow = this.database.prepare("PRAGMA schema_version").get() as { schema_version: number };
    if (this.cachedSchemaVersion === versionRow.schema_version) return this.cachedRelations;

    const rows = this.database.prepare(`
      SELECT name, type, sql
      FROM sqlite_schema
      WHERE type IN ('table', 'view')
        AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\'
      ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END, name
    `).all() as unknown as SchemaRow[];
    this.cachedRelations = rows.map((row) => ({
      name: row.name,
      kind: row.type,
      sql: row.sql,
      columns: this.columnsFor(row.name)
    }));
    this.cachedSchemaVersion = versionRow.schema_version;
    this.cachedIndexes.clear();
    this.cachedForeignKeys.clear();
    return this.cachedRelations;
  }

  private columnsFor(name: string): CorpusColumn[] {
    const rows = this.database.prepare(`
      SELECT cid, name, type, "notnull" AS not_null, dflt_value, pk, hidden
      FROM pragma_table_xinfo(?)
      ORDER BY cid
    `).all(name) as unknown as ColumnRow[];
    return rows.map((row) => ({
      position: row.cid + 1,
      name: row.name,
      type: row.type || "ANY",
      notNull: row.not_null === 1,
      defaultValue: row.dflt_value,
      primaryKeyPosition: row.pk,
      hidden: row.hidden
    }));
  }

  private indexesFor(name: string): CorpusIndex[] {
    const cached = this.cachedIndexes.get(name);
    if (cached) return cached;
    const rows = this.database.prepare(`
      SELECT name, "unique", origin, partial
      FROM pragma_index_list(?)
      ORDER BY "unique" DESC, name
    `).all(name) as unknown as IndexRow[];
    const indexes = rows.map((row) => ({
      name: row.name,
      unique: row.unique === 1,
      origin: row.origin,
      partial: row.partial === 1,
      columns: (this.database.prepare(`
        SELECT name FROM pragma_index_info(?) ORDER BY seqno
      `).all(row.name) as unknown as IndexColumnRow[])
        .map((column) => column.name)
        .filter((column): column is string => column != null)
    }));
    this.cachedIndexes.set(name, indexes);
    return indexes;
  }

  private foreignKeysFor(name: string): CorpusForeignKey[] {
    const cached = this.cachedForeignKeys.get(name);
    if (cached) return cached;
    const rows = this.database.prepare(`
      SELECT id, seq, "table", "from", "to", on_update, on_delete
      FROM pragma_foreign_key_list(?)
      ORDER BY id, seq
    `).all(name) as unknown as ForeignKeyRow[];
    const foreignKeys = rows.map((row) => ({
      id: row.id,
      sequence: row.seq,
      targetRelation: row.table,
      sourceColumn: row.from,
      targetColumn: row.to,
      onUpdate: row.on_update,
      onDelete: row.on_delete
    }));
    this.cachedForeignKeys.set(name, foreignKeys);
    return foreignKeys;
  }

  private inboundReferencesFor(name: string): CorpusInboundReference[] {
    const inbound: CorpusInboundReference[] = [];
    for (const relation of this.allRelations()) {
      for (const foreignKey of this.foreignKeysFor(relation.name)) {
        if (foreignKey.targetRelation !== name) continue;
        inbound.push({
          sourceRelation: relation.name,
          sourceColumn: foreignKey.sourceColumn,
          targetColumn: foreignKey.targetColumn,
          onUpdate: foreignKey.onUpdate,
          onDelete: foreignKey.onDelete
        });
      }
    }
    return inbound.sort((left, right) =>
      left.sourceRelation.localeCompare(right.sourceRelation)
      || left.sourceColumn.localeCompare(right.sourceColumn)
    );
  }

  private orderClause(
    relation: CorpusRelationSummary,
    requestedColumn: string | null,
    direction: "asc" | "desc"
  ): string {
    if (requestedColumn) {
      return `ORDER BY ${quoteIdentifier(requestedColumn)} ${direction.toUpperCase()}`;
    }
    const primary = relation.columns
      .filter((column) => column.primaryKeyPosition > 0)
      .sort((left, right) => left.primaryKeyPosition - right.primaryKeyPosition);
    if (primary.length > 0) {
      return `ORDER BY ${primary.map((column) => quoteIdentifier(column.name)).join(", ")} ASC`;
    }
    const createdAt = relation.columns.find((column) => column.name === "created_at");
    if (createdAt) return `ORDER BY ${quoteIdentifier(createdAt.name)} DESC`;
    return "";
  }
}
