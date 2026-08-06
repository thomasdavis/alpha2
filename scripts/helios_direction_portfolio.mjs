#!/usr/bin/env node

import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { DatabaseSync } from "node:sqlite";

const DEFAULT_RESEARCH_ROOT =
  "/mnt/donto-data/donto-resources/research/alpha-helios-reimagined";
const DEFAULT_SOURCE = `${DEFAULT_RESEARCH_ROOT}/X17-ONE-HUNDRED-WAYS-TO-REIMAGINE-TRAINING.md`;
const DEFAULT_ATLAS = `${DEFAULT_RESEARCH_ROOT}/X19-ONE-HUNDRED-LENS-ALPHA-RESEARCH-ATLAS.md`;
const DEFAULT_DB = `${DEFAULT_RESEARCH_ROOT}/helios-100-directions.sqlite`;
const DEFAULT_REPORT = `${DEFAULT_RESEARCH_ROOT}/PORTFOLIO-STATUS.md`;

function usage() {
  console.error(`Usage:
  node scripts/helios_direction_portfolio.mjs init [--source PATH] [--atlas PATH] [--db PATH] [--report PATH]
  node scripts/helios_direction_portfolio.mjs report [--db PATH] [--report PATH]
  node scripts/helios_direction_portfolio.mjs status --direction N --to STATUS --reason TEXT [--evidence PATH] [--db PATH] [--report PATH]
  node scripts/helios_direction_portfolio.mjs atlas-status --idea N --to STATUS --reason TEXT [--evidence PATH] [--db PATH] [--report PATH]
  node scripts/helios_direction_portfolio.mjs run --direction N --run-id ID --stage STAGE --status STATUS --artifact PATH --metrics PATH [--revision SHA] [--checkpoint HASH] [--workload HASH] [--hardware TEXT] [--accelerator-seconds N] [--cost N] [--started ISO] [--finished ISO] [--db PATH] [--report PATH]
  node scripts/helios_direction_portfolio.mjs link-atlas-run --idea N --run-id ID --status STATUS [--relation TEXT] [--db PATH] [--report PATH]

Statuses are descriptive research states, not automatic promotion decisions:
  queued, designed, cheap_test_running, cheap_test_complete,
  cpu_checkpoint_complete, gpu_candidate, gpu_test_complete,
  promoted, merged, closed, inconclusive
`);
}

function parseArgs(argv) {
  const command = argv[2];
  const options = new Map();
  for (let index = 3; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) throw new Error(`Unexpected argument: ${token}`);
    const value = argv[index + 1];
    if (value === undefined || value.startsWith("--")) {
      options.set(token.slice(2), true);
    } else {
      options.set(token.slice(2), value);
      index += 1;
    }
  }
  return { command, options };
}

function sha256(text) {
  return createHash("sha256").update(text).digest("hex");
}

function parseDirections(sourceText) {
  const directions = [];
  let familyCode = null;
  let familyName = null;
  for (const line of sourceText.split(/\r?\n/)) {
    const family = line.match(/^### ([A-J])\. (.+)$/);
    if (family) {
      familyCode = family[1];
      familyName = family[2].trim();
      continue;
    }
    const row = line.match(
      /^\|\s*(\d{1,3})\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$/,
    );
    if (!row || !familyCode) continue;
    directions.push({
      id: Number.parseInt(row[1], 10),
      familyCode,
      familyName,
      name: row[2].trim(),
      mechanism: row[3].trim(),
      cheapestTest: row[4].trim(),
      lever: row[5].trim(),
    });
  }

  const ids = directions.map(({ id }) => id);
  const expected = Array.from({ length: 100 }, (_, index) => index + 1);
  if (JSON.stringify(ids) !== JSON.stringify(expected)) {
    throw new Error(
      `Expected exactly the ordered direction IDs 1..100; parsed ${ids.length}: ${ids.join(",")}`,
    );
  }
  return directions;
}

function parseAtlasIdeas(sourceText) {
  const ideas = [];
  for (const line of sourceText.split(/\r?\n/)) {
    const row = line.match(
      /^\|\s*(\d{1,3})\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$/,
    );
    if (!row) continue;
    const sourceIndex = Number.parseInt(row[1], 10);
    if (sourceIndex < 1 || sourceIndex > 100) continue;
    ideas.push({
      sourceIndex,
      domain: row[2].trim(),
      lens: row[3].trim(),
      cheapestTest: row[4].trim(),
      levers: row[5].trim(),
    });
  }
  const ids = ideas.map(({ sourceIndex }) => sourceIndex);
  const expected = Array.from({ length: 100 }, (_, index) => index + 1);
  if (JSON.stringify(ids) !== JSON.stringify(expected)) {
    throw new Error(`Expected exactly the ordered X19 atlas IDs 1..100; parsed ${ids.length}: ${ids.join(",")}`);
  }
  return ideas;
}

function firstStage(direction) {
  if (direction.familyCode === "E") return "trace_simulation_then_3090";
  if (direction.id >= 91 && direction.id <= 94) return "profile_or_schedule_simulation";
  if (direction.id === 95 || direction.id === 96 || direction.id === 98) {
    return "offline_decision_replay";
  }
  return "cpu_proxy_or_offline_replay";
}

function primaryMetric(direction) {
  if (direction.familyCode === "E" || (direction.id >= 91 && direction.id <= 94)) {
    return "exact_step_wall_time_at_parity";
  }
  if ([13, 14, 36, 78, 95, 96, 98].includes(direction.id)) {
    return "decision_compute_or_experiment_cost";
  }
  return "held_out_behavior_per_model_visible_token_and_gpu_second";
}

function schema(db) {
  db.exec(`
    PRAGMA foreign_keys = ON;
    PRAGMA journal_mode = WAL;

    CREATE TABLE IF NOT EXISTS program_meta (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS direction (
      direction_id INTEGER PRIMARY KEY CHECK (direction_id BETWEEN 1 AND 100),
      family_code TEXT NOT NULL,
      family_name TEXT NOT NULL,
      name TEXT NOT NULL,
      source_mechanism TEXT NOT NULL,
      alpha_analogue TEXT NOT NULL,
      lever TEXT NOT NULL,
      source_document TEXT NOT NULL,
      source_sha256 TEXT NOT NULL,
      imported_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS atlas_idea (
      atlas_idea_id TEXT PRIMARY KEY,
      source_index INTEGER NOT NULL CHECK (source_index BETWEEN 1 AND 100),
      domain TEXT NOT NULL,
      lens_and_mechanism TEXT NOT NULL,
      cheapest_faithful_test TEXT NOT NULL,
      levers TEXT NOT NULL,
      source_document TEXT NOT NULL,
      source_sha256 TEXT NOT NULL,
      imported_at TEXT NOT NULL,
      UNIQUE(source_document, source_index)
    );

    CREATE TABLE IF NOT EXISTS atlas_experiment_contract (
      contract_id TEXT PRIMARY KEY,
      atlas_idea_id TEXT NOT NULL REFERENCES atlas_idea(atlas_idea_id),
      contract_version INTEGER NOT NULL,
      hypothesis TEXT NOT NULL,
      null_condition TEXT NOT NULL,
      cheapest_faithful_test TEXT NOT NULL,
      required_control TEXT NOT NULL,
      primary_metric TEXT NOT NULL,
      first_stage TEXT NOT NULL,
      promotion_stage TEXT NOT NULL,
      initial_gpu_ceiling_minutes INTEGER NOT NULL,
      created_at TEXT NOT NULL,
      UNIQUE(atlas_idea_id, contract_version)
    );

    CREATE TABLE IF NOT EXISTS atlas_state (
      atlas_idea_id TEXT PRIMARY KEY REFERENCES atlas_idea(atlas_idea_id),
      status TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      latest_reason TEXT NOT NULL,
      latest_evidence_path TEXT
    );

    CREATE TABLE IF NOT EXISTS atlas_state_event (
      event_id INTEGER PRIMARY KEY AUTOINCREMENT,
      atlas_idea_id TEXT NOT NULL REFERENCES atlas_idea(atlas_idea_id),
      from_status TEXT,
      to_status TEXT NOT NULL,
      reason TEXT NOT NULL,
      evidence_path TEXT,
      created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS experiment_contract (
      contract_id TEXT PRIMARY KEY,
      direction_id INTEGER NOT NULL REFERENCES direction(direction_id),
      contract_version INTEGER NOT NULL,
      hypothesis TEXT NOT NULL,
      null_condition TEXT NOT NULL,
      cheapest_faithful_test TEXT NOT NULL,
      required_control TEXT NOT NULL,
      primary_metric TEXT NOT NULL,
      first_stage TEXT NOT NULL,
      promotion_stage TEXT NOT NULL,
      initial_gpu_ceiling_minutes INTEGER NOT NULL,
      created_at TEXT NOT NULL,
      UNIQUE(direction_id, contract_version)
    );

    CREATE TABLE IF NOT EXISTS direction_state (
      direction_id INTEGER PRIMARY KEY REFERENCES direction(direction_id),
      status TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      latest_reason TEXT NOT NULL,
      latest_evidence_path TEXT
    );

    CREATE TABLE IF NOT EXISTS state_event (
      event_id INTEGER PRIMARY KEY AUTOINCREMENT,
      direction_id INTEGER NOT NULL REFERENCES direction(direction_id),
      from_status TEXT,
      to_status TEXT NOT NULL,
      reason TEXT NOT NULL,
      evidence_path TEXT,
      created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS prior_art (
      prior_art_id INTEGER PRIMARY KEY AUTOINCREMENT,
      direction_id INTEGER NOT NULL REFERENCES direction(direction_id),
      title TEXT NOT NULL,
      canonical_url TEXT NOT NULL,
      relation TEXT NOT NULL,
      checked_at TEXT NOT NULL,
      UNIQUE(direction_id, canonical_url)
    );

    CREATE TABLE IF NOT EXISTS experiment_run (
      run_id TEXT PRIMARY KEY,
      contract_id TEXT NOT NULL REFERENCES experiment_contract(contract_id),
      direction_id INTEGER NOT NULL REFERENCES direction(direction_id),
      stage TEXT NOT NULL,
      status TEXT NOT NULL,
      repository_revision TEXT,
      checkpoint_fingerprint TEXT,
      workload_fingerprint TEXT,
      hardware TEXT,
      accelerator_seconds REAL,
      estimated_cost_usd REAL,
      metrics_json TEXT NOT NULL,
      artifact_path TEXT NOT NULL,
      started_at TEXT NOT NULL,
      finished_at TEXT
    );

    CREATE TABLE IF NOT EXISTS physical_run (
      run_id TEXT PRIMARY KEY,
      stage TEXT NOT NULL,
      status TEXT NOT NULL,
      repository_revision TEXT,
      checkpoint_fingerprint TEXT,
      workload_fingerprint TEXT,
      hardware TEXT,
      accelerator_seconds REAL,
      estimated_cost_usd REAL,
      metrics_json TEXT NOT NULL,
      artifact_path TEXT NOT NULL,
      started_at TEXT NOT NULL,
      finished_at TEXT
    );

    CREATE TABLE IF NOT EXISTS direction_run (
      run_id TEXT NOT NULL REFERENCES physical_run(run_id),
      direction_id INTEGER NOT NULL REFERENCES direction(direction_id),
      contract_id TEXT NOT NULL REFERENCES experiment_contract(contract_id),
      relation TEXT NOT NULL,
      status TEXT NOT NULL,
      linked_at TEXT NOT NULL,
      PRIMARY KEY(run_id, direction_id)
    );

    CREATE TABLE IF NOT EXISTS atlas_idea_run (
      run_id TEXT NOT NULL REFERENCES physical_run(run_id),
      atlas_idea_id TEXT NOT NULL REFERENCES atlas_idea(atlas_idea_id),
      contract_id TEXT NOT NULL REFERENCES atlas_experiment_contract(contract_id),
      relation TEXT NOT NULL,
      status TEXT NOT NULL,
      linked_at TEXT NOT NULL,
      PRIMARY KEY(run_id, atlas_idea_id)
    );

    CREATE TABLE IF NOT EXISTS verdict (
      verdict_id INTEGER PRIMARY KEY AUTOINCREMENT,
      direction_id INTEGER NOT NULL REFERENCES direction(direction_id),
      verdict TEXT NOT NULL,
      rationale TEXT NOT NULL,
      evidence_path TEXT NOT NULL,
      revisit_condition TEXT,
      created_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS experiment_run_direction_idx
      ON experiment_run(direction_id, started_at);
    CREATE INDEX IF NOT EXISTS direction_run_direction_idx
      ON direction_run(direction_id, linked_at);
    CREATE INDEX IF NOT EXISTS atlas_idea_run_idea_idx
      ON atlas_idea_run(atlas_idea_id, linked_at);
    CREATE INDEX IF NOT EXISTS state_event_direction_idx
      ON state_event(direction_id, created_at);
  `);
}

function initialize(dbPath, sourcePath, atlasPath) {
  const sourceText = readFileSync(sourcePath, "utf8");
  const sourceHash = sha256(sourceText);
  const directions = parseDirections(sourceText);
  const atlasText = readFileSync(atlasPath, "utf8");
  const atlasHash = sha256(atlasText);
  const atlasIdeas = parseAtlasIdeas(atlasText);
  const importedAt = new Date().toISOString();
  mkdirSync(dirname(dbPath), { recursive: true });
  const db = new DatabaseSync(dbPath);
  schema(db);

  const insertDirection = db.prepare(`
    INSERT INTO direction (
      direction_id, family_code, family_name, name, source_mechanism,
      alpha_analogue, lever, source_document, source_sha256, imported_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(direction_id) DO UPDATE SET
      family_code = excluded.family_code,
      family_name = excluded.family_name,
      name = excluded.name,
      source_mechanism = excluded.source_mechanism,
      alpha_analogue = excluded.alpha_analogue,
      lever = excluded.lever,
      source_document = excluded.source_document,
      source_sha256 = excluded.source_sha256,
      imported_at = excluded.imported_at
  `);
  const insertContract = db.prepare(`
    INSERT OR IGNORE INTO experiment_contract (
      contract_id, direction_id, contract_version, hypothesis, null_condition,
      cheapest_faithful_test, required_control, primary_metric, first_stage,
      promotion_stage, initial_gpu_ceiling_minutes, created_at
    ) VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  const insertAtlasIdea = db.prepare(`
    INSERT INTO atlas_idea (
      atlas_idea_id, source_index, domain, lens_and_mechanism,
      cheapest_faithful_test, levers, source_document, source_sha256, imported_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(atlas_idea_id) DO UPDATE SET
      source_index = excluded.source_index,
      domain = excluded.domain,
      lens_and_mechanism = excluded.lens_and_mechanism,
      cheapest_faithful_test = excluded.cheapest_faithful_test,
      levers = excluded.levers,
      source_document = excluded.source_document,
      source_sha256 = excluded.source_sha256,
      imported_at = excluded.imported_at
  `);
  const insertAtlasContract = db.prepare(`
    INSERT OR IGNORE INTO atlas_experiment_contract (
      contract_id, atlas_idea_id, contract_version, hypothesis, null_condition,
      cheapest_faithful_test, required_control, primary_metric, first_stage,
      promotion_stage, initial_gpu_ceiling_minutes, created_at
    ) VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  const insertAtlasState = db.prepare(`
    INSERT OR IGNORE INTO atlas_state (
      atlas_idea_id, status, updated_at, latest_reason, latest_evidence_path
    ) VALUES (?, 'queued', ?, 'Imported from X19; faithful experiment not yet claimed.', NULL)
  `);
  const insertAtlasEvent = db.prepare(`
    INSERT INTO atlas_state_event (
      atlas_idea_id, from_status, to_status, reason, evidence_path, created_at
    ) VALUES (?, NULL, 'queued', 'Imported from X19; faithful experiment not yet claimed.', NULL, ?)
  `);
  const insertState = db.prepare(`
    INSERT OR IGNORE INTO direction_state (
      direction_id, status, updated_at, latest_reason, latest_evidence_path
    ) VALUES (?, 'queued', ?, 'Imported from X17; faithful experiment not yet claimed.', NULL)
  `);
  const insertEvent = db.prepare(`
    INSERT INTO state_event (
      direction_id, from_status, to_status, reason, evidence_path, created_at
    ) VALUES (?, NULL, 'queued', 'Imported from X17; faithful experiment not yet claimed.', NULL, ?)
  `);
  const upsertMeta = db.prepare(`
    INSERT INTO program_meta (key, value) VALUES (?, ?)
    ON CONFLICT(key) DO UPDATE SET value = excluded.value
  `);

  db.exec("BEGIN IMMEDIATE");
  try {
    for (const direction of directions) {
      insertDirection.run(
        direction.id,
        direction.familyCode,
        direction.familyName,
        direction.name,
        direction.mechanism,
        direction.cheapestTest,
        direction.lever,
        resolve(sourcePath),
        sourceHash,
        importedAt,
      );
      insertContract.run(
        `D${String(direction.id).padStart(3, "0")}-V1`,
        direction.id,
        `Applying ${direction.name} to Alpha can improve the declared fixed target more efficiently than the matched baseline because ${direction.mechanism}`,
        "At matched model-visible tokens, accelerator time, and target-quality constraints, the mechanism does not beat its baseline or causes a parity/behavior regression.",
        direction.cheapestTest,
        "Matched-cost baseline plus a mechanism-corrupted or mechanism-ablated control whenever one can be constructed.",
        primaryMetric(direction),
        firstStage(direction),
        "Bounded RTX 3090 discriminator before any long matched-loss run.",
        30,
        importedAt,
      );
      const stateResult = insertState.run(direction.id, importedAt);
      if (stateResult.changes === 1) insertEvent.run(direction.id, importedAt);
    }
    for (const idea of atlasIdeas) {
      const atlasIdeaId = `X19-${String(idea.sourceIndex).padStart(3, "0")}`;
      insertAtlasIdea.run(
        atlasIdeaId,
        idea.sourceIndex,
        idea.domain,
        idea.lens,
        idea.cheapestTest,
        idea.levers,
        resolve(atlasPath),
        atlasHash,
        importedAt,
      );
      insertAtlasContract.run(
        `${atlasIdeaId}-V1`,
        atlasIdeaId,
        `Applying ${idea.lens} to Alpha or Helios can improve the declared fixed target more efficiently than the matched baseline.`,
        "At matched tokens, accelerator time, and quality constraints, the mechanism does not beat its baseline or introduces an unacceptable parity or behavior regression.",
        idea.cheapestTest,
        "Matched-cost baseline plus a mechanism-corrupted or mechanism-ablated control wherever one can be constructed.",
        idea.levers.includes("R") || idea.levers.includes("M")
          ? "exact_step_wall_time_at_parity"
          : "held_out_behavior_per_model_visible_token_and_gpu_second",
        "analytical_or_cpu_discriminator_before_gpu",
        "Bounded RTX 3090 discriminator before any long matched-loss run.",
        30,
        importedAt,
      );
      const atlasStateResult = insertAtlasState.run(atlasIdeaId, importedAt);
      if (atlasStateResult.changes === 1) insertAtlasEvent.run(atlasIdeaId, importedAt);
    }
    upsertMeta.run("program", "Alpha Helios 100-direction faithful experiment portfolio");
    upsertMeta.run("source_sha256", sourceHash);
    upsertMeta.run("source_document", resolve(sourcePath));
    upsertMeta.run("direction_count", String(directions.length));
    upsertMeta.run("atlas_source_sha256", atlasHash);
    upsertMeta.run("atlas_source_document", resolve(atlasPath));
    upsertMeta.run("atlas_idea_count", String(atlasIdeas.length));
    upsertMeta.run("last_imported_at", importedAt);
    db.exec("COMMIT");
  } catch (error) {
    db.exec("ROLLBACK");
    throw error;
  } finally {
    db.close();
  }
}

function updateStatus(dbPath, directionId, toStatus, reason, evidencePath) {
  const db = new DatabaseSync(dbPath);
  schema(db);
  const current = db.prepare("SELECT status FROM direction_state WHERE direction_id = ?").get(directionId);
  if (!current) throw new Error(`Direction ${directionId} is not in the portfolio`);
  const now = new Date().toISOString();
  db.exec("BEGIN IMMEDIATE");
  try {
    db.prepare(`
      INSERT INTO state_event (
        direction_id, from_status, to_status, reason, evidence_path, created_at
      ) VALUES (?, ?, ?, ?, ?, ?)
    `).run(directionId, current.status, toStatus, reason, evidencePath ?? null, now);
    db.prepare(`
      UPDATE direction_state
      SET status = ?, updated_at = ?, latest_reason = ?, latest_evidence_path = ?
      WHERE direction_id = ?
    `).run(toStatus, now, reason, evidencePath ?? null, directionId);
    db.exec("COMMIT");
  } catch (error) {
    db.exec("ROLLBACK");
    throw error;
  } finally {
    db.close();
  }
}

function updateAtlasStatus(dbPath, sourceIndex, toStatus, reason, evidencePath) {
  const db = new DatabaseSync(dbPath);
  schema(db);
  const atlasIdeaId = `X19-${String(sourceIndex).padStart(3, "0")}`;
  const current = db.prepare("SELECT status FROM atlas_state WHERE atlas_idea_id = ?").get(atlasIdeaId);
  if (!current) throw new Error(`X19 atlas idea ${sourceIndex} is not in the portfolio`);
  const now = new Date().toISOString();
  db.exec("BEGIN IMMEDIATE");
  try {
    db.prepare(`
      INSERT INTO atlas_state_event (
        atlas_idea_id, from_status, to_status, reason, evidence_path, created_at
      ) VALUES (?, ?, ?, ?, ?, ?)
    `).run(atlasIdeaId, current.status, toStatus, reason, evidencePath ?? null, now);
    db.prepare(`
      UPDATE atlas_state
      SET status = ?, updated_at = ?, latest_reason = ?, latest_evidence_path = ?
      WHERE atlas_idea_id = ?
    `).run(toStatus, now, reason, evidencePath ?? null, atlasIdeaId);
    db.exec("COMMIT");
  } catch (error) {
    db.exec("ROLLBACK");
    throw error;
  } finally {
    db.close();
  }
}

function recordRun(dbPath, options) {
  const directionId = Number.parseInt(String(options.get("direction")), 10);
  const runId = options.get("run-id");
  const stage = options.get("stage");
  const status = options.get("status");
  const artifact = options.get("artifact");
  const metricsPath = options.get("metrics");
  if (!Number.isInteger(directionId) || directionId < 1 || directionId > 100 ||
      !runId || !stage || !status || !artifact || !metricsPath) {
    throw new Error("run requires --direction 1..100, --run-id, --stage, --status, --artifact, and --metrics");
  }
  const metricsText = readFileSync(resolve(String(metricsPath)), "utf8");
  JSON.parse(metricsText);
  const optionalNumber = (name) => {
    if (!options.has(name)) return null;
    const value = Number(options.get(name));
    if (!Number.isFinite(value) || value < 0) throw new Error(`--${name} must be a finite non-negative number`);
    return value;
  };
  const startedAt = String(options.get("started") ?? new Date().toISOString());
  const finishedAt = options.has("finished") ? String(options.get("finished")) : null;
  const relation = String(options.get("relation") ?? "direct_test");
  const db = new DatabaseSync(dbPath);
  schema(db);
  const contractId = `D${String(directionId).padStart(3, "0")}-V1`;
  db.exec("BEGIN IMMEDIATE");
  try {
    db.prepare(`
      INSERT OR IGNORE INTO physical_run (
        run_id, stage, status, repository_revision,
        checkpoint_fingerprint, workload_fingerprint, hardware,
        accelerator_seconds, estimated_cost_usd, metrics_json, artifact_path,
        started_at, finished_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      String(runId),
      String(stage),
      String(status),
      options.has("revision") ? String(options.get("revision")) : null,
      options.has("checkpoint") ? String(options.get("checkpoint")) : null,
      options.has("workload") ? String(options.get("workload")) : null,
      options.has("hardware") ? String(options.get("hardware")) : null,
      optionalNumber("accelerator-seconds"),
      optionalNumber("cost"),
      metricsText,
      resolve(String(artifact)),
      startedAt,
      finishedAt,
    );
    const existing = db.prepare(`
      SELECT stage, status, repository_revision, workload_fingerprint,
             checkpoint_fingerprint, hardware, accelerator_seconds,
             estimated_cost_usd, metrics_json, artifact_path, started_at, finished_at
      FROM physical_run WHERE run_id = ?
    `).get(String(runId));
    if (!existing || existing.stage !== String(stage) ||
        existing.status !== String(status) ||
        existing.repository_revision !== (options.has("revision") ? String(options.get("revision")) : null) ||
        existing.workload_fingerprint !== (options.has("workload") ? String(options.get("workload")) : null) ||
        existing.checkpoint_fingerprint !== (options.has("checkpoint") ? String(options.get("checkpoint")) : null) ||
        existing.hardware !== (options.has("hardware") ? String(options.get("hardware")) : null) ||
        existing.accelerator_seconds !== optionalNumber("accelerator-seconds") ||
        existing.estimated_cost_usd !== optionalNumber("cost") ||
        existing.metrics_json !== metricsText ||
        existing.artifact_path !== resolve(String(artifact)) ||
        existing.started_at !== startedAt ||
        existing.finished_at !== finishedAt) {
      throw new Error(`Physical run ${String(runId)} already exists with different immutable evidence`);
    }
    db.prepare(`
      INSERT INTO direction_run (
        run_id, direction_id, contract_id, relation, status, linked_at
      ) VALUES (?, ?, ?, ?, ?, ?)
    `).run(String(runId), directionId, contractId, relation, String(status), new Date().toISOString());
    db.exec("COMMIT");
  } catch (error) {
    db.exec("ROLLBACK");
    throw error;
  }
  db.close();
}

function linkAtlasRun(dbPath, options) {
  const sourceIndex = Number.parseInt(String(options.get("idea")), 10);
  const runId = options.get("run-id");
  const status = options.get("status");
  const relation = String(options.get("relation") ?? "shared_evidence");
  if (!Number.isInteger(sourceIndex) || sourceIndex < 1 || sourceIndex > 100 || !runId || !status) {
    throw new Error("link-atlas-run requires --idea 1..100, --run-id, and --status");
  }
  const atlasIdeaId = `X19-${String(sourceIndex).padStart(3, "0")}`;
  const contractId = `${atlasIdeaId}-V1`;
  const db = new DatabaseSync(dbPath);
  schema(db);
  if (!db.prepare("SELECT 1 FROM physical_run WHERE run_id = ?").get(String(runId))) {
    db.close();
    throw new Error(`Physical run ${String(runId)} is not recorded`);
  }
  db.prepare(`
    INSERT INTO atlas_idea_run (
      run_id, atlas_idea_id, contract_id, relation, status, linked_at
    ) VALUES (?, ?, ?, ?, ?, ?)
  `).run(String(runId), atlasIdeaId, contractId, relation, String(status), new Date().toISOString());
  db.close();
}

function markdownCell(value) {
  return String(value ?? "").replaceAll("|", "\\|").replaceAll("\n", " ");
}

function writeReport(dbPath, reportPath) {
  const db = new DatabaseSync(dbPath, { readOnly: true });
  const meta = Object.fromEntries(
    db.prepare("SELECT key, value FROM program_meta ORDER BY key").all().map((row) => [row.key, row.value]),
  );
  const counts = db.prepare(`
    SELECT status, COUNT(*) AS count
    FROM direction_state
    GROUP BY status
    ORDER BY status
  `).all();
  const rows = db.prepare(`
    SELECT d.direction_id, d.family_code, d.name, d.source_mechanism,
           d.alpha_analogue, d.lever, c.first_stage, c.primary_metric,
           s.status, s.latest_reason, s.latest_evidence_path
    FROM direction d
    JOIN experiment_contract c ON c.direction_id = d.direction_id AND c.contract_version = 1
    JOIN direction_state s ON s.direction_id = d.direction_id
    ORDER BY d.direction_id
  `).all();
  const runCounts = db.prepare(`
    SELECT COUNT(*) AS runs,
           COUNT(accelerator_seconds) AS runs_with_accelerator_seconds,
           COALESCE(SUM(accelerator_seconds), 0) AS accelerator_seconds,
           COUNT(estimated_cost_usd) AS runs_with_estimated_cost,
           COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost_usd
    FROM physical_run
  `).get();
  const directionsWithRuns = db.prepare(`
    SELECT COUNT(DISTINCT direction_id) AS count FROM direction_run
  `).get().count;
  const atlasIdeasWithRuns = db.prepare(`
    SELECT COUNT(DISTINCT atlas_idea_id) AS count FROM atlas_idea_run
  `).get().count;
  const atlasIdeaCount = db.prepare("SELECT COUNT(*) AS count FROM atlas_idea").get().count;
  const atlasCounts = db.prepare(`
    SELECT status, COUNT(*) AS count
    FROM atlas_state
    GROUP BY status
    ORDER BY status
  `).all();
  db.close();

  const statusSummary = counts.map(({ status, count }) => `${status}: ${count}`).join(" · ");
  const atlasStatusSummary = atlasCounts.map(({ status, count }) => `${status}: ${count}`).join(" · ");
  const lines = [
    "# Helios 100-direction portfolio status",
    "",
    `**Generated:** ${new Date().toISOString()}  `,
    `**Source SHA-256:** \`${meta.source_sha256}\`  `,
    `**Directions:** ${rows.length}  `,
    `**Companion X19 atlas ideas:** ${atlasIdeaCount} (source SHA-256: \`${meta.atlas_source_sha256 ?? "not imported"}\`; state: ${atlasStatusSummary})  `,
    `**State:** ${statusSummary}  `,
    `**Recorded physical runs:** ${runCounts.runs}, linked to ${directionsWithRuns} directions; accelerator time recorded for ${runCounts.runs_with_accelerator_seconds}/${runCounts.runs} runs (${Number(runCounts.accelerator_seconds).toFixed(1)} s total); cost recorded for ${runCounts.runs_with_estimated_cost}/${runCounts.runs} runs ($${Number(runCounts.estimated_cost_usd).toFixed(4)} total).`,
    `**X19 ideas with linked physical evidence:** ${atlasIdeasWithRuns}. Shared links do not duplicate accelerator time or cost.`,
    "",
    "A direction is not counted as attempted merely because it appeared in X17. Its state changes only when an evidence artifact is attached. The first experiment is deliberately cheap; survivors progress to a bounded RTX 3090 discriminator and only then to matched-loss training.",
    "",
    "| # | Family | Direction | Mechanism | Cheapest faithful test | First stage | Primary metric | State | Evidence/reason |",
    "|---:|:---:|---|---|---|---|---|---|---|",
  ];
  for (const row of rows) {
    lines.push(
      `| ${row.direction_id} | ${markdownCell(row.family_code)} | ${markdownCell(row.name)} | ${markdownCell(row.source_mechanism)} | ${markdownCell(row.alpha_analogue)} | ${markdownCell(row.first_stage)} | ${markdownCell(row.primary_metric)} | ${markdownCell(row.status)} | ${markdownCell(row.latest_evidence_path ?? row.latest_reason)} |`,
    );
  }
  lines.push("", "This file is generated from `helios-100-directions.sqlite`; edit state through the portfolio CLI so the event trail remains append-only.", "");
  mkdirSync(dirname(reportPath), { recursive: true });
  writeFileSync(reportPath, lines.join("\n"));
}

try {
  const { command, options } = parseArgs(process.argv);
  const sourcePath = resolve(String(options.get("source") ?? DEFAULT_SOURCE));
  const atlasPath = resolve(String(options.get("atlas") ?? DEFAULT_ATLAS));
  const dbPath = resolve(String(options.get("db") ?? DEFAULT_DB));
  const reportPath = resolve(String(options.get("report") ?? DEFAULT_REPORT));

  if (command === "init") {
    initialize(dbPath, sourcePath, atlasPath);
    writeReport(dbPath, reportPath);
  } else if (command === "report") {
    writeReport(dbPath, reportPath);
  } else if (command === "status") {
    const directionId = Number.parseInt(String(options.get("direction")), 10);
    const toStatus = options.get("to");
    const reason = options.get("reason");
    if (!Number.isInteger(directionId) || directionId < 1 || directionId > 100 || !toStatus || !reason) {
      usage();
      process.exitCode = 2;
    } else {
      updateStatus(
        dbPath,
        directionId,
        String(toStatus),
        String(reason),
        options.has("evidence") ? resolve(String(options.get("evidence"))) : null,
      );
      writeReport(dbPath, reportPath);
    }
  } else if (command === "atlas-status") {
    const sourceIndex = Number.parseInt(String(options.get("idea")), 10);
    const toStatus = options.get("to");
    const reason = options.get("reason");
    if (!Number.isInteger(sourceIndex) || sourceIndex < 1 || sourceIndex > 100 || !toStatus || !reason) {
      usage();
      process.exitCode = 2;
    } else {
      updateAtlasStatus(
        dbPath,
        sourceIndex,
        String(toStatus),
        String(reason),
        options.has("evidence") ? resolve(String(options.get("evidence"))) : null,
      );
      writeReport(dbPath, reportPath);
    }
  } else if (command === "run") {
    recordRun(dbPath, options);
    writeReport(dbPath, reportPath);
  } else if (command === "link-atlas-run") {
    linkAtlasRun(dbPath, options);
    writeReport(dbPath, reportPath);
  } else {
    usage();
    process.exitCode = 2;
  }
} catch (error) {
  console.error(error instanceof Error ? error.stack : String(error));
  process.exitCode = 1;
}
