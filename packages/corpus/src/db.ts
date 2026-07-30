import { createClient, type Client, type InValue } from "@libsql/client";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { canonicalJson, sha256Bytes, sha256Json, stableId } from "./hash.js";
import { migrations, migrationDigest, requiredTables, requiredViews } from "./schema.js";
import { categorySeeds, familySeeds, transformationSeeds } from "./seeds.js";
import {
  directorySize,
  ensureLedgerDirectories,
  resolveLedgerPaths,
  writeContentAddressedBlob
} from "./storage.js";
import type {
  CampaignConfig,
  CandidateStatus,
  CandidateValidation,
  GeneratedItem,
  LedgerPaths,
  ReviewItem,
  StructuredCallResult
} from "./types.js";

export interface Ledger {
  client: Client;
  paths: LedgerPaths;
}

export interface CandidateForReview {
  candidateId: string;
  candidateVersionId: string;
  familySlug: string;
  item: GeneratedItem;
}

export interface LedgerValidationReport {
  integrity: string;
  foreignKeyViolations: number;
  missingTables: string[];
  missingViews: string[];
  missingBlobs: string[];
  corruptBlobs: string[];
  migrationCount: number;
  footprintBytes: number;
}

export interface CampaignStats {
  campaignId: string;
  slug: string;
  status: string;
  tasks: Record<string, number>;
  candidates: Record<string, number>;
  reviews: Record<string, number>;
  modelCalls: number;
  inputTokens: number;
  cachedInputTokens: number;
  outputTokens: number;
  footprintBytes: number;
  artifactLimitBytes: number;
}

function now(): string {
  return new Date().toISOString();
}

function newId(prefix: string): string {
  return `${prefix}_${randomUUID()}`;
}

export async function openLedger(home?: string): Promise<Ledger> {
  const paths = resolveLedgerPaths(home);
  ensureLedgerDirectories(paths);
  const client = createClient({ url: `file:${paths.database}` });
  await client.execute("PRAGMA journal_mode=WAL");
  await client.execute("PRAGMA foreign_keys=ON");
  await client.execute("PRAGMA synchronous=FULL");
  await migrate(client);
  return { client, paths };
}

export function closeLedger(ledger: Ledger): void {
  ledger.client.close();
}

async function migrate(client: Client): Promise<void> {
  await client.execute(`CREATE TABLE IF NOT EXISTS schema_migration (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    applied_at TEXT NOT NULL
  ) STRICT`);

  const rows = await client.execute("SELECT version, name, sha256 FROM schema_migration ORDER BY version");
  const applied = new Map<number, { name: string; sha256: string }>();
  for (const row of rows.rows) {
    applied.set(Number(row["version"]), {
      name: String(row["name"]),
      sha256: String(row["sha256"])
    });
  }

  for (const migration of migrations) {
    const digest = migrationDigest(migration);
    const existing = applied.get(migration.version);
    if (existing) {
      if (existing.name !== migration.name || existing.sha256 !== digest) {
        throw new Error(`Migration ${migration.version} differs from the applied ledger`);
      }
      continue;
    }
    await client.batch(
      [
        ...migration.statements.map((sql) => ({ sql, args: [] })),
        {
          sql: "INSERT INTO schema_migration(version, name, sha256, applied_at) VALUES (?, ?, ?, ?)",
          args: [migration.version, migration.name, digest, now()]
        }
      ],
      "write"
    );
  }
}

export async function putBlob(
  ledger: Ledger,
  bytes: Uint8Array | string,
  mediaType: string
): Promise<string> {
  const buffer = typeof bytes === "string" ? Buffer.from(bytes, "utf8") : Buffer.from(bytes);
  const stored = writeContentAddressedBlob(ledger.paths, buffer);
  const ts = now();
  await ledger.client.batch(
    [
      {
        sql: `INSERT OR IGNORE INTO blob(sha256, byte_length, media_type, relative_path, created_at)
              VALUES (?, ?, ?, ?, ?)`,
        args: [stored.sha256, stored.byteLength, mediaType, stored.relativePath, ts]
      },
      {
        sql: `INSERT OR IGNORE INTO blob_location(id, blob_sha256, location, storage_kind, verified_at, created_at)
              VALUES (?, ?, ?, 'local_content_addressed', ?, ?)`,
        args: [stableId("blobloc", stored.relativePath), stored.sha256, stored.relativePath, ts, ts]
      }
    ],
    "write"
  );
  return stored.sha256;
}

async function assertStoredDigest(
  client: Client,
  table: string,
  whereColumn: string,
  whereValue: string,
  expected: string
): Promise<void> {
  const result = await client.execute({
    sql: `SELECT content_sha256 FROM ${table} WHERE ${whereColumn} = ? LIMIT 1`,
    args: [whereValue]
  });
  if (result.rows.length > 0 && String(result.rows[0]!["content_sha256"]) !== expected) {
    throw new Error(`${table}.${whereColumn}=${whereValue} changed without a new version`);
  }
}

export async function seedLedger(ledger: Ledger): Promise<void> {
  const ts = now();
  const programId = stableId("program", "alpha-synthetic-conversational-intelligence");
  const objective = "Build a chatty conceptually intelligent model and reusable synthetic-curriculum ledger.";
  const objectiveDigest = sha256Bytes(objective);
  const programVersionId = stableId("programv", `${programId}:1`);

  await ledger.client.batch(
    [
      {
        sql: "INSERT OR IGNORE INTO program(id, slug, status, created_at) VALUES (?, ?, 'active', ?)",
        args: [programId, "alpha-synthetic-conversational-intelligence", ts]
      },
      {
        sql: `INSERT OR IGNORE INTO program_version
              (id, program_id, version, objective, authority, content_sha256, created_at)
              VALUES (?, ?, 1, ?, 'operator', ?, ?)`,
        args: [programVersionId, programId, objective, objectiveDigest, ts]
      },
      {
        sql: `INSERT OR IGNORE INTO decision
              (id, program_id, title, decision_text, authority, created_at)
              VALUES (?, ?, ?, ?, 'operator', ?)`,
        args: [
          stableId("decision", "2026-07-30-model-role-split"),
          programId,
          "Initial model role split",
          "Use gpt-5.6-sol for counsel and orchestration reasoning, gpt-5.4 for initial synthetic generation, and keep gpt-5.5 disabled unless a paired task-specific probe justifies it.",
          ts
        ]
      },
      {
        sql: `INSERT OR IGNORE INTO decision
              (id, program_id, title, decision_text, authority, created_at)
              VALUES (?, ?, ?, ?, 'operator', ?)`,
        args: [
          stableId("decision", "2026-07-30-storage-pause"),
          programId,
          "Corpus-owned storage pause",
          "Track only the corpus program's own artifact footprint and pause new work if it exceeds 15 GiB; do not impose a global disk hard rule or delete unrelated data.",
          ts
        ]
      }
    ],
    "write"
  );
  await assertStoredDigest(ledger.client, "program_version", "id", programVersionId, objectiveDigest);

  for (const seed of categorySeeds) {
    const categoryId = stableId("category", seed.slug);
    const versionId = stableId("categoryv", `${seed.slug}:1`);
    const digest = sha256Json({
      preferredName: seed.name,
      definition: seed.definition,
      metaClass: seed.metaClass
    });
    await ledger.client.batch(
      [
        {
          sql: "INSERT OR IGNORE INTO category(id, slug, status, created_at) VALUES (?, ?, 'active', ?)",
          args: [categoryId, seed.slug, ts]
        },
        {
          sql: `INSERT OR IGNORE INTO category_version
                (id, category_id, version, preferred_name, concise_definition, extended_definition,
                 meta_class, authority_kind, content_sha256, created_at)
                VALUES (?, ?, 1, ?, ?, ?, ?, 'program_design', ?, ?)`,
          args: [versionId, categoryId, seed.name, seed.definition, seed.definition, seed.metaClass, digest, ts]
        }
      ],
      "write"
    );
    await assertStoredDigest(ledger.client, "category_version", "id", versionId, digest);
  }

  for (const [slug, definition] of transformationSeeds) {
    await ledger.client.execute({
      sql: "INSERT OR IGNORE INTO transformation(id, slug, definition, created_at) VALUES (?, ?, ?, ?)",
      args: [stableId("transform", slug), slug, definition, ts]
    });
  }

  for (const blueprint of familySeeds) {
    const familyId = stableId("family", blueprint.slug);
    const familyVersionId = stableId("familyv", `${blueprint.slug}:1`);
    const blueprintJson = canonicalJson(blueprint as unknown as import("./types.js").JsonValue);
    const digest = sha256Bytes(blueprintJson);
    const statements: Array<{ sql: string; args: InValue[] }> = [
      {
        sql: `INSERT OR IGNORE INTO concept_family(id, slug, status, split, created_at)
              VALUES (?, ?, 'canary', 'quarantine', ?)`,
        args: [familyId, blueprint.slug, ts]
      },
      {
        sql: `INSERT OR IGNORE INTO family_version
              (id, family_id, version, title, purpose, blueprint_json, content_sha256, authority_kind, created_at)
              VALUES (?, ?, 1, ?, ?, ?, ?, 'program_canary', ?)`,
        args: [familyVersionId, familyId, blueprint.title, blueprint.purpose, blueprintJson, digest, ts]
      }
    ];
    for (const lens of blueprint.primaryLenses) {
      statements.push({
        sql: `INSERT OR IGNORE INTO family_category(family_id, category_id, assignment_kind, created_at)
              VALUES (?, ?, 'design_primary_lens', ?)`,
        args: [familyId, stableId("category", lens), ts]
      });
    }
    for (const [index, question] of blueprint.competencyQuestions.entries()) {
      statements.push({
        sql: `INSERT OR IGNORE INTO competency_question(id, family_id, question_text, created_at)
              VALUES (?, ?, ?, ?)`,
        args: [stableId("cq", `${blueprint.slug}:${index}:${question}`), familyId, question, ts]
      });
    }
    for (const projection of blueprint.projections) {
      statements.push({
        sql: `INSERT OR IGNORE INTO family_projection
              (id, family_id, slug, domain, description, relation, created_at)
              VALUES (?, ?, ?, ?, ?, ?, ?)`,
        args: [
          stableId("projection", `${blueprint.slug}:${projection.slug}`),
          familyId,
          projection.slug,
          projection.domain,
          projection.description,
          projection.relation,
          ts
        ]
      });
    }
    const stateId = stableId("state", `${blueprint.slug}:baseline`);
    statements.push({
      sql: `INSERT OR IGNORE INTO semantic_state(id, family_id, state_key, purpose, created_at)
            VALUES (?, ?, 'baseline', ?, ?)`,
      args: [stateId, familyId, blueprint.purpose, ts]
    });
    for (const [index, proposition] of blueprint.requiredCommitments.entries()) {
      statements.push({
        sql: `INSERT OR IGNORE INTO commitment
              (id, semantic_state_id, holder, proposition, status, scope, depends_on_json, created_at)
              VALUES (?, ?, 'shared_ground', ?, 'required', 'family', '[]', ?)`,
        args: [stableId("commitment", `${blueprint.slug}:required:${index}`), stateId, proposition, ts]
      });
    }
    for (const [index, proposition] of blueprint.prohibitedCommitments.entries()) {
      statements.push({
        sql: `INSERT OR IGNORE INTO commitment
              (id, semantic_state_id, holder, proposition, status, scope, depends_on_json, created_at)
              VALUES (?, ?, 'shared_ground', ?, 'prohibited', 'family', '[]', ?)`,
        args: [stableId("commitment", `${blueprint.slug}:prohibited:${index}`), stateId, proposition, ts]
      });
    }
    for (const [index, hazard] of blueprint.shortcutHazards.entries()) {
      statements.push({
        sql: `INSERT OR IGNORE INTO shortcut_hazard
              (id, family_id, hazard_kind, description, detection_plan, created_at)
              VALUES (?, ?, 'declared', ?, 'Lexical and structural holdout review', ?)`,
        args: [stableId("hazard", `${blueprint.slug}:${index}`), familyId, hazard, ts]
      });
    }
    await ledger.client.batch(statements, "write");
    await assertStoredDigest(ledger.client, "family_version", "id", familyVersionId, digest);
  }

  await ensureProviderAndModels(ledger);
}

async function ensureProviderAndModels(ledger: Ledger): Promise<void> {
  const ts = now();
  await ledger.client.execute({
    sql: "INSERT OR IGNORE INTO provider(id, slug, transport, created_at) VALUES (?, 'openai', 'codex_cli_subscription', ?)",
    args: [stableId("provider", "openai-codex-cli"), ts]
  });
  const profiles = [
    { id: "gpt-5.4", role: "worker", transport: "codex-cli-schema" },
    { id: "gpt-5.5", role: "critic", transport: "codex-cli-schema-disabled-pending-probe" },
    { id: "gpt-5.6-sol", role: "counsel", transport: "codex-agent-counsel" }
  ];
  for (const profile of profiles) {
    const modelId = stableId("model", `openai:${profile.id}`);
    const revisionId = stableId("modelrev", `${profile.id}:2026-07-30:${profile.role}:${profile.transport}`);
    await ledger.client.batch(
      [
        {
          sql: "INSERT OR IGNORE INTO model(id, provider, model_id, created_at) VALUES (?, 'openai', ?, ?)",
          args: [modelId, profile.id, ts]
        },
        {
          sql: `INSERT OR IGNORE INTO model_revision
                (id, model_id, revision, role, transport, cli_version, created_at)
                VALUES (?, ?, '2026-07-30-alias', ?, ?, '0.146.0', ?)`,
          args: [revisionId, modelId, profile.role, profile.transport, ts]
        }
      ],
      "write"
    );
  }
}

export async function createCampaign(ledger: Ledger, config: CampaignConfig): Promise<string> {
  const existing = await ledger.client.execute({
    sql: "SELECT * FROM generation_campaign WHERE slug = ?",
    args: [config.slug]
  });
  if (existing.rows.length > 0) {
    const row = existing.rows[0]!;
    const mismatches = [
      ["purpose", config.purpose],
      ["worker_model", config.workerModel],
      ["critic_model", config.criticModel],
      ["max_review_calls", config.maxReviewCalls],
      ["items_per_family", config.itemsPerFamily],
      ["artifact_limit_bytes", config.artifactLimitBytes]
    ].filter(([column, expected]) => String(row[String(column)]) !== String(expected));
    if (config.maxGenerationCalls > Number(row["max_generation_calls"])) {
      mismatches.push(["max_generation_calls", config.maxGenerationCalls]);
    }
    if (mismatches.length > 0) {
      throw new Error(`Campaign ${config.slug} differs from its frozen contract: ${mismatches.map(([key]) => key).join(", ")}`);
    }
    return String(row["id"]);
  }
  const id = stableId("campaign", config.slug);
  const ts = now();
  await ledger.client.execute({
    sql: `INSERT INTO generation_campaign
          (id, slug, purpose, status, worker_model, critic_model, max_generation_calls,
           max_review_calls, items_per_family, artifact_limit_bytes, created_at, updated_at)
          VALUES (?, ?, ?, 'planned', ?, ?, ?, ?, ?, ?, ?, ?)`,
    args: [
      id,
      config.slug,
      config.purpose,
      config.workerModel,
      config.criticModel,
      config.maxGenerationCalls,
      config.maxReviewCalls,
      config.itemsPerFamily,
      config.artifactLimitBytes,
      ts,
      ts
    ]
  });
  return id;
}

export async function nextTaskAttempt(ledger: Ledger, taskId: string): Promise<number> {
  const result = await ledger.client.execute({
    sql: "SELECT COALESCE(MAX(attempt), 0) + 1 AS next_attempt FROM model_call_attempt WHERE task_id = ?",
    args: [taskId]
  });
  return Number(result.rows[0]!["next_attempt"]);
}

export async function loadRecordedStructuredResponse<T>(
  ledger: Ledger,
  taskId: string
): Promise<{ callId: string; parsed: T } | null> {
  const result = await ledger.client.execute({
    sql: `SELECT mc.id, b.relative_path
          FROM model_call mc
          JOIN blob b ON b.sha256 = mc.response_blob_sha256
          WHERE mc.task_id = ? AND mc.exit_code = 0 AND mc.response_blob_sha256 IS NOT NULL
          ORDER BY mc.completed_at DESC LIMIT 1`,
    args: [taskId]
  });
  if (result.rows.length === 0) return null;
  const row = result.rows[0]!;
  const path = join(ledger.paths.home, String(row["relative_path"]));
  return {
    callId: String(row["id"]),
    parsed: JSON.parse(readFileSync(path, "utf8")) as T
  };
}

export async function setCampaignStatus(ledger: Ledger, campaignId: string, status: string): Promise<void> {
  await ledger.client.execute({
    sql: "UPDATE generation_campaign SET status = ?, updated_at = ? WHERE id = ?",
    args: [status, now(), campaignId]
  });
}

export async function checkCampaignStorage(ledger: Ledger, campaignId: string): Promise<boolean> {
  const row = await ledger.client.execute({
    sql: "SELECT artifact_limit_bytes FROM generation_campaign WHERE id = ?",
    args: [campaignId]
  });
  if (row.rows.length === 0) throw new Error(`Unknown campaign ${campaignId}`);
  const limit = Number(row.rows[0]!["artifact_limit_bytes"]);
  const footprint = directorySize(ledger.paths.home);
  if (footprint <= limit) return true;
  await setCampaignStatus(ledger, campaignId, "paused_storage");
  await appendEvent(ledger, "campaign_paused_storage", "generation_campaign", campaignId, {
    footprintBytes: footprint,
    artifactLimitBytes: limit
  });
  return false;
}

export async function createTask(
  ledger: Ledger,
  campaignId: string,
  familyId: string | null,
  taskKind: string,
  idempotencyKey: string,
  modelAlias: string
): Promise<{ id: string; status: string }> {
  const existing = await ledger.client.execute({
    sql: "SELECT id, status FROM generation_task WHERE idempotency_key = ?",
    args: [idempotencyKey]
  });
  if (existing.rows.length > 0) {
    return { id: String(existing.rows[0]!["id"]), status: String(existing.rows[0]!["status"]) };
  }
  const id = stableId("task", idempotencyKey);
  const ts = now();
  await ledger.client.execute({
    sql: `INSERT INTO generation_task
          (id, campaign_id, family_id, task_kind, idempotency_key, status, model_alias, created_at, updated_at)
          VALUES (?, ?, ?, ?, ?, 'planned', ?, ?, ?)`,
    args: [id, campaignId, familyId, taskKind, idempotencyKey, modelAlias, ts, ts]
  });
  return { id, status: "planned" };
}

export async function setTaskStatus(ledger: Ledger, taskId: string, status: string): Promise<void> {
  await ledger.client.execute({
    sql: "UPDATE generation_task SET status = ?, updated_at = ? WHERE id = ?",
    args: [status, now(), taskId]
  });
}

export async function appendEvent(
  ledger: Ledger,
  eventType: string,
  objectKind: string,
  objectId: string,
  payload: Record<string, unknown>
): Promise<string> {
  const id = newId("event");
  await ledger.client.execute({
    sql: `INSERT INTO event(id, event_type, object_kind, object_id, payload_json, created_at)
          VALUES (?, ?, ?, ?, ?, ?)`,
    args: [id, eventType, objectKind, objectId, canonicalJson(payload as import("./types.js").JsonValue), now()]
  });
  return id;
}

async function resolveModelRevisionId(ledger: Ledger, modelName: string, role: string): Promise<string> {
  const result = await ledger.client.execute({
    sql: `SELECT mr.id
          FROM model_revision mr JOIN model m ON m.id = mr.model_id
          WHERE m.model_id = ? AND mr.role = ?
          ORDER BY mr.created_at DESC LIMIT 1`,
    args: [modelName, role]
  });
  if (result.rows.length === 0) throw new Error(`No model revision registered for ${modelName}/${role}`);
  return String(result.rows[0]!["id"]);
}

export async function recordStructuredCall(
  ledger: Ledger,
  taskId: string,
  modelName: string,
  role: "orchestrator" | "worker" | "critic",
  promptSlug: string,
  prompt: string,
  schemaSlug: string,
  schema: import("./types.js").JsonValue,
  result: StructuredCallResult,
  attempt: number
): Promise<string> {
  const ts = now();
  const promptSha = await putBlob(ledger, prompt, "text/plain; charset=utf-8");
  const schemaText = canonicalJson(schema);
  const schemaSha = await putBlob(ledger, schemaText, "application/schema+json");
  const stdoutSha = await putBlob(ledger, result.stdout, "application/x-ndjson");
  const stderrSha = await putBlob(ledger, result.stderr, "text/plain; charset=utf-8");
  const responseSha = result.lastMessage
    ? await putBlob(ledger, result.lastMessage, "application/json")
    : null;

  const promptTemplateId = stableId("prompt", promptSlug);
  const promptVersionId = stableId("promptv", `${promptSlug}:1:${promptSha}`);
  const toolSchemaId = stableId("schema", `${schemaSlug}:1:${schemaSha}`);
  const modelRevisionId = await resolveModelRevisionId(ledger, modelName, role);
  const callId = newId("call");
  const artifactRows: Array<{ sql: string; args: InValue[] }> = [
    {
      sql: "INSERT OR IGNORE INTO prompt_template(id, slug, created_at) VALUES (?, ?, ?)",
      args: [promptTemplateId, promptSlug, ts]
    },
    {
      sql: `INSERT OR IGNORE INTO prompt_template_version
            (id, prompt_template_id, version, content_blob_sha256, content_sha256, created_at)
            VALUES (?, ?, 1, ?, ?, ?)`,
      args: [promptVersionId, promptTemplateId, promptSha, promptSha, ts]
    },
    {
      sql: `INSERT OR IGNORE INTO tool_schema
            (id, slug, version, schema_blob_sha256, content_sha256, created_at)
            VALUES (?, ?, 1, ?, ?, ?)`,
      args: [toolSchemaId, schemaSlug, schemaSha, schemaSha, ts]
    },
    {
      sql: `INSERT INTO model_call
            (id, task_id, model_revision_id, prompt_template_version_id, tool_schema_id,
             request_blob_sha256, stdout_blob_sha256, stderr_blob_sha256, response_blob_sha256,
             command_json, exit_code, input_tokens, cached_input_tokens, output_tokens, started_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      args: [
        callId,
        taskId,
        modelRevisionId,
        promptVersionId,
        toolSchemaId,
        promptSha,
        stdoutSha,
        stderrSha,
        responseSha,
        JSON.stringify(result.commandArgs),
        result.exitCode,
        result.usage.inputTokens,
        result.usage.cachedInputTokens,
        result.usage.outputTokens,
        result.startedAt,
        result.completedAt
      ]
    },
    {
      sql: `INSERT INTO model_call_message
            (id, model_call_id, ordinal, role, content_blob_sha256, created_at)
            VALUES (?, ?, 0, 'user', ?, ?)`,
      args: [newId("callmsg"), callId, promptSha, ts]
    },
    {
      sql: `INSERT INTO model_call_usage
            (id, model_call_id, input_tokens, cached_input_tokens, output_tokens, monetary_cost, usage_source, created_at)
            VALUES (?, ?, ?, ?, ?, NULL, 'codex_cli_events', ?)`,
      args: [
        newId("usage"),
        callId,
        result.usage.inputTokens,
        result.usage.cachedInputTokens,
        result.usage.outputTokens,
        ts
      ]
    },
    {
      sql: `INSERT INTO model_call_attempt
            (id, task_id, attempt, status, error_text, started_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)`,
      args: [
        newId("attempt"),
        taskId,
        attempt,
        result.exitCode === 0 && responseSha ? "completed" : "failed",
        result.exitCode === 0 ? null : result.stderr.toString("utf8").slice(0, 4000),
        result.startedAt,
        result.completedAt
      ]
    },
    {
      sql: `INSERT INTO routing_decision
            (id, task_id, selected_model_revision_id, rationale, alternatives_json, created_at)
            VALUES (?, ?, ?, ?, '[]', ?)`,
      args: [newId("route"), taskId, modelRevisionId, `Frozen role route: ${role}`, ts]
    }
  ];

  for (const [kind, sha] of [
    ["codex_stdout_jsonl", stdoutSha],
    ["codex_stderr", stderrSha],
    ...(responseSha ? [["structured_last_message", responseSha]] : [])
  ] as Array<[string, string]>) {
    artifactRows.push({
      sql: "INSERT INTO raw_artifact(id, task_id, kind, blob_sha256, created_at) VALUES (?, ?, ?, ?, ?)",
      args: [newId("artifact"), taskId, kind, sha, ts]
    });
  }
  artifactRows.push({
    sql: `INSERT INTO budget_event
          (id, campaign_id, task_id, event_kind, calls_delta, token_delta, monetary_delta, detail, created_at)
          SELECT ?, campaign_id, id, 'call_completed', 1, ?, NULL, ?, ? FROM generation_task WHERE id = ?`,
    args: [
      newId("budget"),
      (result.usage.inputTokens ?? 0) + (result.usage.outputTokens ?? 0),
      `${modelName}/${role}`,
      ts,
      taskId
    ]
  });

  await ledger.client.batch(artifactRows, "write");
  return callId;
}

export async function recordCandidate(
  ledger: Ledger,
  campaignId: string,
  familyId: string,
  callId: string,
  item: GeneratedItem,
  validation: CandidateValidation
): Promise<string> {
  const candidateId = stableId("candidate", `${campaignId}:${familyId}:${item.itemKey}`);
  const candidateVersionId = stableId("candidatev", `${candidateId}:1`);
  const dialogueId = stableId("dialogue", candidateId);
  const dialogueVersionId = stableId("dialoguev", `${dialogueId}:1`);
  const ts = now();
  const content = {
    itemKey: item.itemKey,
    kind: item.kind,
    title: item.title,
    primaryLens: item.primaryLens,
    secondaryLenses: item.secondaryLenses,
    transformation: item.transformation,
    intendedResponsePolicy: item.intendedResponsePolicy,
    difficulty: item.difficulty,
    messages: item.messages,
    linguisticPair: item.linguisticPair,
    generatorNotes: item.generatorNotes
  };
  const contentJson = canonicalJson(content as unknown as import("./types.js").JsonValue);
  const contractJson = canonicalJson(item.hiddenContract as unknown as import("./types.js").JsonValue);
  const digest = sha256Bytes(contentJson);
  const existing = await ledger.client.execute({
    sql: `SELECT content_json, hidden_contract_json FROM candidate_version WHERE id = ?`,
    args: [candidateVersionId]
  });
  if (existing.rows.length > 0) {
    const row = existing.rows[0]!;
    if (String(row["content_json"]) !== contentJson || String(row["hidden_contract_json"]) !== contractJson) {
      throw new Error(`Candidate ${candidateId} changed under the same immutable version`);
    }
    return candidateId;
  }
  const status: CandidateStatus = validation.valid ? "structurally_valid" : "structurally_rejected";
  const statements: Array<{ sql: string; args: InValue[] }> = [
    {
      sql: `INSERT OR IGNORE INTO candidate
            (id, campaign_id, family_id, item_key, kind, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'generated', ?, ?)`,
      args: [candidateId, campaignId, familyId, item.itemKey, item.kind, ts, ts]
    },
    {
      sql: `INSERT OR IGNORE INTO dialogue(id, family_id, status, created_at)
            VALUES (?, ?, ?, ?)`,
      args: [dialogueId, familyId, status, ts]
    },
    {
      sql: `INSERT OR IGNORE INTO dialogue_version
            (id, dialogue_id, version, purpose, content_sha256, created_at)
            VALUES (?, ?, 1, ?, ?, ?)`,
      args: [dialogueVersionId, dialogueId, item.title, digest, ts]
    },
    {
      sql: `INSERT OR IGNORE INTO candidate_version
            (id, candidate_id, version, generation_call_id, dialogue_id, content_json,
             hidden_contract_json, content_sha256, created_at)
            VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?)`,
      args: [candidateVersionId, candidateId, callId, dialogueId, contentJson, contractJson, digest, ts]
    },
    {
      sql: "UPDATE candidate SET status = ?, updated_at = ? WHERE id = ?",
      args: [status, ts, candidateId]
    },
    {
      sql: `INSERT INTO quality_state_transition
            (id, candidate_id, from_status, to_status, reason, authority, created_at)
            VALUES (?, ?, 'generated', ?, 'deterministic_validation', 'validator', ?)`,
      args: [newId("quality"), candidateId, status, ts]
    }
  ];

  for (const [ordinal, message] of item.messages.entries()) {
    const messageId = stableId("message", `${dialogueId}:${ordinal}`);
    const messageVersionId = stableId("messagev", `${messageId}:1`);
    const textSha = await putBlob(ledger, message.content, "text/plain; charset=utf-8");
    statements.push(
      {
        sql: "INSERT OR IGNORE INTO message(id, dialogue_id, ordinal, created_at) VALUES (?, ?, ?, ?)",
        args: [messageId, dialogueId, ordinal, ts]
      },
      {
        sql: `INSERT OR IGNORE INTO message_version
              (id, message_id, version, role, natural_text_blob_sha256, language, created_at)
              VALUES (?, ?, 1, ?, ?, 'en', ?)`,
        args: [messageVersionId, messageId, message.role, textSha, ts]
      }
    );
  }
  for (const finding of validation.findings) {
    statements.push({
      sql: `INSERT INTO candidate_failure(id, task_id, candidate_id, code, detail, created_at)
            SELECT ?, mc.task_id, ?, ?, ?, ? FROM model_call mc WHERE mc.id = ?`,
      args: [stableId("failure", `${candidateId}:${finding.code}:${finding.message}`), candidateId, finding.code, finding.message, ts, callId]
    });
  }
  await ledger.client.batch(statements, "write");
  return candidateId;
}

export async function listFamilies(ledger: Ledger): Promise<Array<{ id: string; slug: string; blueprint: string }>> {
  const result = await ledger.client.execute(`
    SELECT f.id, f.slug, fv.blueprint_json
    FROM concept_family f
    JOIN family_version fv ON fv.family_id = f.id
    WHERE fv.version = (SELECT MAX(fv2.version) FROM family_version fv2 WHERE fv2.family_id = f.id)
    ORDER BY f.slug
  `);
  return result.rows.map((row) => ({
    id: String(row["id"]),
    slug: String(row["slug"]),
    blueprint: String(row["blueprint_json"])
  }));
}

export async function listCandidatesForReview(
  ledger: Ledger,
  campaignId: string,
  limit: number
): Promise<CandidateForReview[]> {
  const result = await ledger.client.execute({
    sql: `SELECT c.id AS candidate_id, cv.id AS candidate_version_id, f.slug AS family_slug, cv.content_json,
                 cv.hidden_contract_json
          FROM candidate c
          JOIN concept_family f ON f.id = c.family_id
          JOIN candidate_version cv ON cv.candidate_id = c.id
          LEFT JOIN review r ON r.candidate_version_id = cv.id
          WHERE c.campaign_id = ? AND c.status = 'structurally_valid' AND r.id IS NULL
          ORDER BY f.slug, c.item_key
          LIMIT ?`,
    args: [campaignId, limit]
  });
  return result.rows.map((row) => {
    const content = JSON.parse(String(row["content_json"])) as Omit<GeneratedItem, "hiddenContract">;
    const hiddenContract = JSON.parse(String(row["hidden_contract_json"])) as GeneratedItem["hiddenContract"];
    return {
      candidateId: String(row["candidate_id"]),
      candidateVersionId: String(row["candidate_version_id"]),
      familySlug: String(row["family_slug"]),
      item: { ...content, hiddenContract }
    };
  });
}

export async function recordReview(
  ledger: Ledger,
  reviewCallId: string,
  reviewerModel: string,
  item: ReviewItem
): Promise<void> {
  const ts = now();
  const reviewerModelRevisionId = await resolveModelRevisionId(ledger, reviewerModel, "critic");
  const reviewId = newId("review");
  const candidate = await ledger.client.execute({
    sql: "SELECT candidate_id FROM candidate_version WHERE id = ?",
    args: [item.candidateId]
  });
  if (candidate.rows.length === 0) throw new Error(`Review references unknown candidate version ${item.candidateId}`);
  const candidateId = String(candidate.rows[0]!["candidate_id"]);
  const nextStatus: CandidateStatus = item.outcome === "accept"
    ? "model_accepted_pending_human"
    : item.outcome === "repair"
      ? "repair_requested"
      : item.outcome === "reject"
        ? "model_rejected"
        : "structurally_valid";
  const previous = await ledger.client.execute({ sql: "SELECT status FROM candidate WHERE id = ?", args: [candidateId] });
  const previousStatus = String(previous.rows[0]!["status"]);
  const statements: Array<{ sql: string; args: InValue[] }> = [
    {
      sql: `INSERT INTO review
            (id, candidate_version_id, reviewer_model_revision_id, review_call_id, outcome, rationale, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)`,
      args: [reviewId, item.candidateId, reviewerModelRevisionId, reviewCallId, item.outcome, item.rationale, ts]
    },
    {
      sql: "UPDATE candidate SET status = ?, updated_at = ? WHERE id = ?",
      args: [nextStatus, ts, candidateId]
    },
    {
      sql: `INSERT INTO quality_state_transition
            (id, candidate_id, from_status, to_status, reason, authority, created_at)
            VALUES (?, ?, ?, ?, 'model_review', 'calibrated_model_pending_human', ?)`,
      args: [newId("quality"), candidateId, previousStatus, nextStatus, ts]
    }
  ];
  for (const [dimension, score] of Object.entries(item.scores)) {
    statements.push({
      sql: `INSERT INTO review_dimension_score(id, review_id, dimension, score, created_at)
            VALUES (?, ?, ?, ?, ?)`,
      args: [newId("score"), reviewId, dimension, score, ts]
    });
  }
  for (const finding of item.findings) {
    statements.push({
      sql: `INSERT INTO review_finding
            (id, review_id, dimension, severity, evidence, recommendation, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)`,
      args: [
        newId("finding"), reviewId, finding.dimension, finding.severity,
        finding.evidence, finding.recommendation, ts
      ]
    });
  }
  if (item.outcome === "repair") {
    statements.push({
      sql: `INSERT INTO repair_request
            (id, candidate_version_id, review_id, requested_change, preserve_json, status, created_at)
            VALUES (?, ?, ?, ?, '[]', 'open', ?)`,
      args: [newId("repair"), item.candidateId, reviewId, item.rationale, ts]
    });
  }
  await ledger.client.batch(statements, "write");
}

export async function campaignStats(ledger: Ledger, campaignSlug: string): Promise<CampaignStats> {
  const campaign = await ledger.client.execute({
    sql: "SELECT * FROM generation_campaign WHERE slug = ?",
    args: [campaignSlug]
  });
  if (campaign.rows.length === 0) throw new Error(`Unknown campaign ${campaignSlug}`);
  const row = campaign.rows[0]!;
  const campaignId = String(row["id"]);
  const grouped = async (table: string, column: string): Promise<Record<string, number>> => {
    const result = await ledger.client.execute({
      sql: `SELECT ${column} AS key, COUNT(*) AS count FROM ${table} WHERE campaign_id = ? GROUP BY ${column}`,
      args: [campaignId]
    });
    return Object.fromEntries(result.rows.map((entry) => [String(entry["key"]), Number(entry["count"])]));
  };
  const tasks = await grouped("generation_task", "status");
  const candidates = await grouped("candidate", "status");
  const reviewsResult = await ledger.client.execute({
    sql: `SELECT r.outcome AS key, COUNT(*) AS count
          FROM review r JOIN candidate_version cv ON cv.id = r.candidate_version_id
          JOIN candidate c ON c.id = cv.candidate_id
          WHERE c.campaign_id = ? GROUP BY r.outcome`,
    args: [campaignId]
  });
  const reviews = Object.fromEntries(reviewsResult.rows.map((entry) => [String(entry["key"]), Number(entry["count"])]));
  const usage = await ledger.client.execute({
    sql: `SELECT COUNT(*) AS calls,
                 COALESCE(SUM(mc.input_tokens), 0) AS input_tokens,
                 COALESCE(SUM(mc.cached_input_tokens), 0) AS cached_input_tokens,
                 COALESCE(SUM(mc.output_tokens), 0) AS output_tokens
          FROM model_call mc JOIN generation_task gt ON gt.id = mc.task_id
          WHERE gt.campaign_id = ?`,
    args: [campaignId]
  });
  return {
    campaignId,
    slug: campaignSlug,
    status: String(row["status"]),
    tasks,
    candidates,
    reviews,
    modelCalls: Number(usage.rows[0]!["calls"]),
    inputTokens: Number(usage.rows[0]!["input_tokens"]),
    cachedInputTokens: Number(usage.rows[0]!["cached_input_tokens"]),
    outputTokens: Number(usage.rows[0]!["output_tokens"]),
    footprintBytes: directorySize(ledger.paths.home),
    artifactLimitBytes: Number(row["artifact_limit_bytes"])
  };
}

export async function validateLedger(ledger: Ledger): Promise<LedgerValidationReport> {
  const integrityResult = await ledger.client.execute("PRAGMA integrity_check");
  const integrity = String(integrityResult.rows[0]?.["integrity_check"] ?? "unknown");
  const foreignKeys = await ledger.client.execute("PRAGMA foreign_key_check");
  const tables = await ledger.client.execute("SELECT name FROM sqlite_master WHERE type = 'table'");
  const tableSet = new Set(tables.rows.map((row) => String(row["name"])));
  const missingTables = requiredTables.filter((table) => !tableSet.has(table));
  const views = await ledger.client.execute("SELECT name FROM sqlite_master WHERE type = 'view'");
  const viewSet = new Set(views.rows.map((row) => String(row["name"])));
  const missingViews = requiredViews.filter((view) => !viewSet.has(view));
  const blobRows = await ledger.client.execute("SELECT sha256, relative_path FROM blob");
  const missingBlobs: string[] = [];
  const corruptBlobs: string[] = [];
  for (const row of blobRows.rows) {
    const digest = String(row["sha256"]);
    const path = join(ledger.paths.home, String(row["relative_path"]));
    if (!existsSync(path)) {
      missingBlobs.push(digest);
      continue;
    }
    if (sha256Bytes(readFileSync(path)) !== digest) corruptBlobs.push(digest);
  }
  const migrationRows = await ledger.client.execute("SELECT COUNT(*) AS count FROM schema_migration");
  return {
    integrity,
    foreignKeyViolations: foreignKeys.rows.length,
    missingTables: [...missingTables],
    missingViews: [...missingViews],
    missingBlobs,
    corruptBlobs,
    migrationCount: Number(migrationRows.rows[0]!["count"]),
    footprintBytes: directorySize(ledger.paths.home)
  };
}
