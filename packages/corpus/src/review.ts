import { randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import type { InValue } from "@libsql/client";
import type { Ledger } from "./db.js";
import { putBlob } from "./db.js";
import { canonicalJson, sha256Bytes, stableId } from "./hash.js";
import { writeAtomic } from "./storage.js";
import type {
  GeneratedItem,
  HumanReviewPacket,
  HumanReviewPacketAssignment,
  HumanReviewPass,
  JsonValue
} from "./types.js";
import {
  HUMAN_REVIEW_MISSING_CLARIFICATION,
  HUMAN_REVIEW_QUESTION_POLICIES,
  HUMAN_REVIEW_RUBRIC_SLUG,
  HUMAN_REVIEW_RUBRIC_VERSION,
  emptyHumanReviewResponse,
  humanReviewDimensions,
  humanReviewOutcomes,
  humanReviewResponseErrors,
  parseHumanReviewPacketText
} from "./review-contract.js";

const RUBRIC_DEFINITION = {
  sourceDocuments: [
    "docs/synthetic-curriculum-prd/PRD-12-D5-HUMAN-ADJUDICATION.md",
    "docs/synthetic-curriculum-prd/APPENDIX-D-D5-REVIEW-INSTRUMENT.md"
  ],
  scoreAnchors: {
    0: "critical failure",
    1: "major failure",
    2: "locally repairable",
    3: "acceptable",
    4: "exemplar"
  },
  passes: {
    A: {
      purpose: "Blind model-visible conversational and intellectual review",
      dimensions: humanReviewDimensions("A").map((dimension) => dimension.key),
      outcomes: humanReviewOutcomes("A").map((outcome) => outcome.value)
    },
    B: {
      purpose: "Contract-aware blueprint and realization review",
      dimensions: humanReviewDimensions("B").map((dimension) => dimension.key),
      outcomes: humanReviewOutcomes("B").map((outcome) => outcome.value)
    }
  },
  questionPolicies: HUMAN_REVIEW_QUESTION_POLICIES.map((choice) => choice.value),
  missingClarification: HUMAN_REVIEW_MISSING_CLARIFICATION.map((choice) => choice.value)
} as unknown as JsonValue;

interface CandidateRow {
  candidateVersionId: string;
  familySlug: string;
  kind: string;
  status: string;
  contentSha256: string;
  item: GeneratedItem;
}

interface AssignmentRow extends CandidateRow {
  assignmentId: string;
  blindness: Record<string, unknown>;
}

export interface PrepareHumanReviewOptions {
  campaignSlug: string;
  reviewerAlias: string;
  pass: HumanReviewPass;
  limit: number;
  seed?: string;
  outputDirectory?: string;
}

export interface PreparedHumanReview {
  packetPath: string;
  markdownPath: string;
  sessionId: string;
  assignmentCount: number;
  resumed: boolean;
  packetSha256: string;
}

export interface HumanReviewStatus {
  campaignSlug: string;
  assignments: Record<string, number>;
  reviews: Record<string, number>;
  humanReviewArtifacts: number;
  candidateStatuses: Record<string, number>;
  releaseMembers: number;
  trainingExposures: number;
}

function now(): string {
  return new Date().toISOString();
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function parseJsonRecord(value: string, label: string): Record<string, unknown> {
  const parsed = JSON.parse(value) as unknown;
  if (!isRecord(parsed)) throw new Error(`${label} is not a JSON object`);
  return parsed;
}

function assertPass(value: string): asserts value is HumanReviewPass {
  if (value !== "A" && value !== "B") throw new Error(`Review pass must be A or B, received ${value}`);
}

function visibleCandidate(pass: HumanReviewPass, row: CandidateRow): JsonValue {
  if (pass === "A") {
    return {
      kind: row.kind,
      messages: row.item.messages as unknown as JsonValue
    };
  }
  return {
    familySlug: row.familySlug,
    structuralStatus: row.status,
    item: row.item as unknown as JsonValue
  };
}

function seededRank(seed: string, candidateVersionId: string): string {
  return sha256Bytes(`${seed}\0${candidateVersionId}`);
}

async function campaignId(ledger: Ledger, slug: string): Promise<string> {
  const result = await ledger.client.execute({
    sql: "SELECT id FROM generation_campaign WHERE slug = ?",
    args: [slug]
  });
  if (result.rows.length === 0) throw new Error(`Unknown campaign ${slug}`);
  return String(result.rows[0]!["id"]);
}

async function ensureHumanActor(ledger: Ledger, alias: string): Promise<string> {
  const clean = alias.trim();
  if (clean.length < 1 || clean.length > 80) throw new Error("Reviewer alias must contain 1-80 characters");
  const id = stableId("actor", `human:${clean}`);
  const ts = now();
  await ledger.client.execute({
    sql: "INSERT OR IGNORE INTO actor(id, kind, display_name, created_at) VALUES (?, 'human', ?, ?)",
    args: [id, clean, ts]
  });
  const stored = await ledger.client.execute({ sql: "SELECT kind, display_name FROM actor WHERE id = ?", args: [id] });
  if (stored.rows.length !== 1 || String(stored.rows[0]!["kind"]) !== "human"
    || String(stored.rows[0]!["display_name"]) !== clean) {
    throw new Error(`Human actor identity collision for ${clean}`);
  }
  return id;
}

async function ensureRubric(ledger: Ledger): Promise<string> {
  const rubricId = stableId("rubric", HUMAN_REVIEW_RUBRIC_SLUG);
  const definitionJson = canonicalJson(RUBRIC_DEFINITION);
  const digest = sha256Bytes(definitionJson);
  const versionId = stableId(
    "rubricv", `${HUMAN_REVIEW_RUBRIC_SLUG}:${HUMAN_REVIEW_RUBRIC_VERSION}:${digest}`
  );
  const ts = now();
  await ledger.client.batch([
    {
      sql: "INSERT OR IGNORE INTO rubric(id, slug, created_at) VALUES (?, ?, ?)",
      args: [rubricId, HUMAN_REVIEW_RUBRIC_SLUG, ts]
    },
    {
      sql: `INSERT OR IGNORE INTO rubric_version
            (id, rubric_id, version, definition_json, content_sha256, created_at)
            VALUES (?, ?, ?, ?, ?, ?)`,
      args: [versionId, rubricId, HUMAN_REVIEW_RUBRIC_VERSION, definitionJson, digest, ts]
    }
  ], "write");
  const stored = await ledger.client.execute({
    sql: "SELECT content_sha256 FROM rubric_version WHERE id = ?",
    args: [versionId]
  });
  if (stored.rows.length !== 1 || String(stored.rows[0]!["content_sha256"]) !== digest) {
    throw new Error("Stored D5 human-review rubric differs from the executable definition");
  }
  return versionId;
}

function candidateFromRow(row: Record<string, unknown>): CandidateRow {
  const content = parseJsonRecord(String(row["content_json"]), "candidate content");
  const hiddenContract = parseJsonRecord(String(row["hidden_contract_json"]), "hidden contract");
  return {
    candidateVersionId: String(row["candidate_version_id"]),
    familySlug: String(row["family_slug"]),
    kind: String(row["kind"]),
    status: String(row["status"]),
    contentSha256: String(row["content_sha256"]),
    item: { ...content, hiddenContract } as unknown as GeneratedItem
  };
}

async function openAssignments(
  ledger: Ledger,
  campaign: string,
  actorId: string,
  rubricVersionId: string,
  pass: HumanReviewPass
): Promise<AssignmentRow[]> {
  const result = await ledger.client.execute({
    sql: `SELECT ra.id AS assignment_id, ra.blindness_json,
                 cc.candidate_version_id, cc.family_slug, cc.kind, cc.status,
                 cc.content_json, cc.hidden_contract_json, cc.content_sha256
          FROM review_assignment ra
          JOIN corpus_candidate_current cc ON cc.candidate_version_id = ra.candidate_version_id
          JOIN generation_campaign gc ON gc.id = cc.campaign_id
          WHERE gc.slug = ? AND ra.reviewer_actor_id = ? AND ra.rubric_version_id = ?
            AND ra.status = 'assigned' AND json_extract(ra.blindness_json, '$.pass') = ?
          ORDER BY CAST(json_extract(ra.blindness_json, '$.presentationIndex') AS INTEGER)`,
    args: [campaign, actorId, rubricVersionId, pass]
  });
  return result.rows.map((raw) => {
    const row = raw as Record<string, unknown>;
    return {
      ...candidateFromRow(row),
      assignmentId: String(row["assignment_id"]),
      blindness: parseJsonRecord(String(row["blindness_json"]), "assignment blindness")
    };
  });
}

async function selectCandidates(
  ledger: Ledger,
  campaign: string,
  actorId: string,
  rubricVersionId: string,
  pass: HumanReviewPass,
  limit: number,
  seed: string
): Promise<CandidateRow[]> {
  const passBRequirement = pass === "B"
    ? `AND EXISTS (
         SELECT 1 FROM review_assignment prior
         WHERE prior.candidate_version_id = cc.candidate_version_id
           AND prior.reviewer_actor_id = ? AND prior.rubric_version_id = ?
           AND prior.status = 'completed' AND json_extract(prior.blindness_json, '$.pass') = 'A'
       )`
    : "";
  const args: InValue[] = [campaign, actorId, rubricVersionId, pass];
  if (pass === "B") args.push(actorId, rubricVersionId);
  const result = await ledger.client.execute({
    sql: `SELECT cc.candidate_version_id, cc.family_slug, cc.kind, cc.status,
                 cc.content_json, cc.hidden_contract_json, cc.content_sha256
          FROM corpus_candidate_current cc
          JOIN generation_campaign gc ON gc.id = cc.campaign_id
          WHERE gc.slug = ?
            AND NOT EXISTS (
              SELECT 1 FROM review_assignment existing
              WHERE existing.candidate_version_id = cc.candidate_version_id
                AND existing.reviewer_actor_id = ? AND existing.rubric_version_id = ?
                AND json_extract(existing.blindness_json, '$.pass') = ?
            )
            ${passBRequirement}`,
    args
  });
  return result.rows
    .map((row) => candidateFromRow(row as Record<string, unknown>))
    .sort((left, right) => seededRank(seed, left.candidateVersionId).localeCompare(seededRank(seed, right.candidateVersionId)))
    .slice(0, limit);
}

function packetInstructions(pass: HumanReviewPass): string[] {
  if (pass === "A") {
    return [
      "Review only the model-visible messages. Do not inspect the public ledger or hidden contract for this item first.",
      "Fill every response field. Use scores 0-4; quote exact evidence for findings.",
      "Judge conceptual plausibility and conversational quality separately. Seal Pass A before preparing Pass B."
    ];
  }
  return [
    "Pass A is sealed. Review the revealed blueprint, hidden contract, metadata, and structural status.",
    "Judge blueprint validity separately from the rendered conversation.",
    "Use scores 0-4; quote exact evidence and select a PRD-04 scientific disposition."
  ];
}

function renderMarkdown(packet: HumanReviewPacket): string {
  const lines = [
    `# Alpha D5 human review — Pass ${packet.pass}`,
    "",
    `- Campaign: \`${packet.campaignSlug}\``,
    `- Session: \`${packet.sessionId}\``,
    `- Reviewer: \`${packet.reviewerAlias}\``,
    `- Rubric: \`${packet.rubricSlug}\` v${packet.rubricVersion}`,
    `- Assignments: ${packet.assignments.length}`,
    "",
    "> This packet records no judgment until its completed JSON form is validated and submitted locally.",
    "",
    "## Instructions",
    "",
    ...packet.instructions.map((instruction) => `- ${instruction}`)
  ];
  for (const [index, assignment] of packet.assignments.entries()) {
    lines.push("", `## ${index + 1}. ${assignment.opaqueItemId}`, "");
    const candidate = assignment.candidate as Record<string, unknown>;
    const item = (candidate["item"] ?? candidate) as Record<string, unknown>;
    const messages = item["messages"];
    if (Array.isArray(messages)) {
      for (const message of messages) {
        if (!isRecord(message)) continue;
        lines.push(`**${String(message["role"])}:** ${String(message["content"])}`, "");
      }
    }
    if (packet.pass === "B") {
      lines.push("<details><summary>Contract-aware fields</summary>", "", "```json",
        JSON.stringify(candidate, null, 2), "```", "", "</details>", "");
    }
    lines.push(`Complete the matching \`response\` object in the JSON packet for \`${assignment.opaqueItemId}\`.`);
  }
  return `${lines.join("\n")}\n`;
}

async function recordPacketExport(
  ledger: Ledger,
  packet: HumanReviewPacket,
  packetPath: string,
  markdownPath: string
): Promise<string> {
  const packetText = canonicalJson(packet as unknown as JsonValue);
  const markdown = renderMarkdown(packet);
  writeAtomic(packetPath, packetText);
  writeAtomic(markdownPath, markdown);
  const packetSha = await putBlob(ledger, packetText, "application/json");
  const markdownSha = await putBlob(ledger, markdown, "text/markdown");
  const ts = now();
  await ledger.client.batch([
    {
      sql: `INSERT OR IGNORE INTO export_artifact
            (id, release_id, cohort_snapshot_id, format, blob_sha256, manifest_json, created_at)
            VALUES (?, NULL, NULL, 'human_review_packet_json', ?, ?, ?)`,
      args: [stableId("export", `review-packet:${packet.sessionId}:json:${packetSha}`), packetSha,
        canonicalJson({ path: packetPath, pass: packet.pass, sessionId: packet.sessionId } as JsonValue), ts]
    },
    {
      sql: `INSERT OR IGNORE INTO export_artifact
            (id, release_id, cohort_snapshot_id, format, blob_sha256, manifest_json, created_at)
            VALUES (?, NULL, NULL, 'human_review_packet_markdown', ?, ?, ?)`,
      args: [stableId("export", `review-packet:${packet.sessionId}:markdown:${markdownSha}`), markdownSha,
        canonicalJson({ path: markdownPath, pass: packet.pass, sessionId: packet.sessionId } as JsonValue), ts]
    }
  ], "write");
  return packetSha;
}

export async function prepareHumanReviewPacket(
  ledger: Ledger,
  options: PrepareHumanReviewOptions
): Promise<PreparedHumanReview> {
  assertPass(options.pass);
  if (!Number.isInteger(options.limit) || options.limit < 1 || options.limit > 48) {
    throw new Error("Human-review packet limit must be an integer from 1 to 48");
  }
  await campaignId(ledger, options.campaignSlug);
  const actorId = await ensureHumanActor(ledger, options.reviewerAlias);
  const rubricVersionId = await ensureRubric(ledger);
  let assignments = await openAssignments(
    ledger, options.campaignSlug, actorId, rubricVersionId, options.pass
  );
  const resumed = assignments.length > 0;
  let sessionId: string;
  let seed: string;
  if (resumed) {
    sessionId = String(assignments[0]!.blindness["sessionId"]);
    seed = String(assignments[0]!.blindness["seed"]);
  } else {
    sessionId = `review_session_${randomUUID()}`;
    seed = options.seed ?? randomUUID();
    const candidates = await selectCandidates(
      ledger, options.campaignSlug, actorId, rubricVersionId, options.pass, options.limit, seed
    );
    if (candidates.length === 0) {
      throw new Error(`No candidates are eligible for Pass ${options.pass}`);
    }
    const ts = now();
    const statements: Array<{ sql: string; args: InValue[] }> = [];
    assignments = candidates.map((candidate, index) => {
      const assignmentId = stableId(
        "assignment", `${candidate.candidateVersionId}:${actorId}:${rubricVersionId}:${options.pass}`
      );
      const blindness = {
        pass: options.pass,
        sessionId,
        seed,
        presentationIndex: index + 1,
        hiddenRepeat: false,
        visibleFields: options.pass === "A" ? ["kind", "messages"] : ["all_candidate_and_contract_fields"],
        hiddenFields: options.pass === "A"
          ? ["identity", "family", "title", "difficulty", "generator_notes", "response_policy", "lenses", "transformation", "hidden_contract", "structural_status", "other_reviews"]
          : ["other_reviews"]
      };
      statements.push({
        sql: `INSERT INTO review_assignment
              (id, candidate_version_id, reviewer_actor_id, rubric_version_id, blindness_json,
               status, created_at, updated_at)
              VALUES (?, ?, ?, ?, ?, 'assigned', ?, ?)`,
        args: [assignmentId, candidate.candidateVersionId, actorId, rubricVersionId,
          canonicalJson(blindness as unknown as JsonValue), ts, ts]
      });
      return { ...candidate, assignmentId, blindness };
    });
    await ledger.client.batch(statements, "write");
  }
  const packetAssignments: HumanReviewPacketAssignment[] = assignments.map((assignment) => ({
    assignmentId: assignment.assignmentId,
    opaqueItemId: stableId("opaque", `${sessionId}:${assignment.candidateVersionId}`).slice(0, 19),
    candidateContentSha256: assignment.contentSha256,
    candidate: visibleCandidate(options.pass, assignment),
    response: emptyHumanReviewResponse(options.pass)
  }));
  const createdAt = now();
  const packet: HumanReviewPacket = {
    schemaVersion: 1,
    campaignSlug: options.campaignSlug,
    sessionId,
    pass: options.pass,
    reviewerAlias: options.reviewerAlias.trim(),
    rubricSlug: HUMAN_REVIEW_RUBRIC_SLUG,
    rubricVersion: HUMAN_REVIEW_RUBRIC_VERSION,
    seed,
    createdAt,
    instructions: packetInstructions(options.pass),
    assignments: packetAssignments
  };
  const directory = resolve(options.outputDirectory
    ?? join(ledger.paths.releases, "review", `${options.campaignSlug}-${options.pass.toLowerCase()}-${sessionId}`));
  const packetPath = join(directory, "review-packet.json");
  const markdownPath = join(directory, "README.md");
  const packetSha256 = await recordPacketExport(ledger, packet, packetPath, markdownPath);
  return {
    packetPath,
    markdownPath,
    sessionId,
    assignmentCount: packet.assignments.length,
    resumed,
    packetSha256
  };
}

export async function submitHumanReviewPacket(
  ledger: Ledger,
  path: string
): Promise<{ submitted: number; pass: HumanReviewPass; sessionId: string; submissionSha256: string }> {
  const submissionBytes = readFileSync(resolve(path));
  const packet = parseHumanReviewPacketText(submissionBytes.toString("utf8"));
  const actorId = await ensureHumanActor(ledger, packet.reviewerAlias);
  const rubricVersionId = await ensureRubric(ledger);
  if (packet.rubricSlug !== HUMAN_REVIEW_RUBRIC_SLUG
    || packet.rubricVersion !== HUMAN_REVIEW_RUBRIC_VERSION) {
    throw new Error("Human-review packet uses an unsupported rubric version");
  }
  if (packet.assignments.length < 1) throw new Error("Human-review packet has no assignments");
  const assignmentIds = new Set<string>();
  const validated: Array<{
    assignment: HumanReviewPacketAssignment;
    candidateVersionId: string;
    candidateId: string;
  }> = [];
  for (const assignment of packet.assignments) {
    if (assignmentIds.has(assignment.assignmentId)) throw new Error(`Duplicate assignment ${assignment.assignmentId}`);
    assignmentIds.add(assignment.assignmentId);
    const responseErrors = humanReviewResponseErrors(packet.pass, assignment.response, assignment.opaqueItemId);
    if (responseErrors.length > 0) throw new Error(responseErrors[0]);
    const stored = await ledger.client.execute({
      sql: `SELECT ra.candidate_version_id, ra.status, ra.blindness_json, cv.content_sha256, cv.candidate_id,
                   ra.reviewer_actor_id, ra.rubric_version_id
            FROM review_assignment ra
            JOIN candidate_version cv ON cv.id = ra.candidate_version_id
            WHERE ra.id = ?`,
      args: [assignment.assignmentId]
    });
    if (stored.rows.length !== 1) throw new Error(`Unknown review assignment ${assignment.assignmentId}`);
    const row = stored.rows[0]!;
    const blindness = parseJsonRecord(String(row["blindness_json"]), "stored blindness");
    if (String(row["status"]) !== "assigned" || String(blindness["pass"]) !== packet.pass
      || String(blindness["sessionId"]) !== packet.sessionId) {
      throw new Error(`Assignment ${assignment.assignmentId} is not open for this pass/session`);
    }
    if (String(row["reviewer_actor_id"]) !== actorId || String(row["rubric_version_id"]) !== rubricVersionId) {
      throw new Error(`Assignment ${assignment.assignmentId} reviewer or rubric mismatch`);
    }
    if (String(row["content_sha256"]) !== assignment.candidateContentSha256) {
      throw new Error(`Assignment ${assignment.assignmentId} candidate version changed`);
    }
    validated.push({
      assignment,
      candidateVersionId: String(row["candidate_version_id"]),
      candidateId: String(row["candidate_id"])
    });
  }
  const submissionSha256 = await putBlob(ledger, submissionBytes, "application/json");
  const ts = now();
  const statements: Array<{ sql: string; args: InValue[] }> = [{
    sql: "INSERT OR IGNORE INTO raw_artifact(id, task_id, kind, blob_sha256, created_at) VALUES (?, NULL, ?, ?, ?)",
    args: [stableId("artifact", `human-review:${packet.sessionId}:${submissionSha256}`),
      `human_review_submission_pass_${packet.pass.toLowerCase()}`, submissionSha256, ts]
  }];
  for (const entry of validated) {
    const response = entry.assignment.response;
    const reviewId = stableId("review", `${entry.assignment.assignmentId}:${submissionSha256}`);
    const rationale = canonicalJson({
      rationale: response.rationale,
      summaryUserAim: response.summaryUserAim,
      summaryAssistantMove: response.summaryAssistantMove,
      questionPolicy: response.questionPolicy,
      missingClarification: response.missingClarification,
      confidence: response.confidence,
      uncertainty: response.uncertainty,
      expertiseNeeded: response.expertiseNeeded,
      pass: packet.pass,
      assignmentId: entry.assignment.assignmentId,
      submissionSha256
    } as unknown as JsonValue);
    statements.push({
      sql: `INSERT INTO review
            (id, candidate_version_id, reviewer_actor_id, outcome, rationale, created_at)
            VALUES (?, ?, ?, ?, ?, ?)`,
      args: [reviewId, entry.candidateVersionId, actorId, response.outcome!, rationale, ts]
    });
    for (const [dimension, score] of Object.entries(response.scores)) {
      statements.push({
        sql: `INSERT INTO review_dimension_score(id, review_id, dimension, score, created_at)
              VALUES (?, ?, ?, ?, ?)`,
        args: [stableId("score", `${reviewId}:${dimension}`), reviewId, dimension, score!, ts]
      });
    }
    statements.push({
      sql: `INSERT INTO review_dimension_score(id, review_id, dimension, score, created_at)
            VALUES (?, ?, 'reviewer_confidence', ?, ?)`,
      args: [stableId("score", `${reviewId}:reviewer_confidence`), reviewId, response.confidence!, ts]
    });
    for (const [index, finding] of response.findings.entries()) {
      statements.push({
        sql: `INSERT INTO review_finding
              (id, review_id, dimension, severity, evidence, recommendation, created_at)
              VALUES (?, ?, ?, ?, ?, ?, ?)`,
        args: [stableId("finding", `${reviewId}:${index}:${finding.dimension}:${finding.evidence}`), reviewId,
          finding.dimension, finding.severity, finding.evidence, finding.recommendation, ts]
      });
    }
    statements.push({
      sql: "UPDATE review_assignment SET status = 'completed', updated_at = ? WHERE id = ? AND status = 'assigned'",
      args: [ts, entry.assignment.assignmentId]
    });
    const eventId = stableId("event", `human-review-submitted:${reviewId}`);
    statements.push({
      sql: `INSERT INTO event(id, event_type, object_kind, object_id, payload_json, created_at)
            VALUES (?, 'human_review_submitted', 'review', ?, ?, ?)`,
      args: [eventId, reviewId, canonicalJson({
        assignmentId: entry.assignment.assignmentId,
        candidateId: entry.candidateId,
        pass: packet.pass,
        sessionId: packet.sessionId,
        submissionSha256
      } as JsonValue), ts]
    });
    statements.push({
      sql: `INSERT INTO event_object(id, event_id, object_kind, object_id, created_at)
            VALUES (?, ?, 'candidate_version', ?, ?)`,
      args: [stableId("eventobj", `${eventId}:${entry.candidateVersionId}`), eventId, entry.candidateVersionId, ts]
    });
  }
  await ledger.client.batch(statements, "write");
  return { submitted: validated.length, pass: packet.pass, sessionId: packet.sessionId, submissionSha256 };
}

export async function humanReviewStatus(ledger: Ledger, campaignSlug: string): Promise<HumanReviewStatus> {
  const id = await campaignId(ledger, campaignSlug);
  const assignments = await ledger.client.execute({
    sql: `SELECT json_extract(ra.blindness_json, '$.pass') || ':' || ra.status AS key, COUNT(*) AS count
          FROM review_assignment ra JOIN candidate_version cv ON cv.id = ra.candidate_version_id
          JOIN candidate c ON c.id = cv.candidate_id WHERE c.campaign_id = ? GROUP BY key`,
    args: [id]
  });
  const reviews = await ledger.client.execute({
    sql: `SELECT (CASE WHEN json_valid(r.rationale) THEN json_extract(r.rationale, '$.pass') ELSE 'legacy' END)
                 || ':' || r.outcome AS key, COUNT(*) AS count
          FROM review r JOIN candidate_version cv ON cv.id = r.candidate_version_id
          JOIN candidate c ON c.id = cv.candidate_id
          WHERE c.campaign_id = ? AND r.reviewer_actor_id IS NOT NULL GROUP BY key`,
    args: [id]
  });
  const candidates = await ledger.client.execute({
    sql: "SELECT status AS key, COUNT(*) AS count FROM candidate WHERE campaign_id = ? GROUP BY status",
    args: [id]
  });
  const artifactCount = await ledger.client.execute(
    "SELECT COUNT(*) AS count FROM raw_artifact WHERE kind LIKE 'human_review_submission_pass_%'"
  );
  const releaseMembers = await ledger.client.execute("SELECT COUNT(*) AS count FROM release_member");
  const trainingExposures = await ledger.client.execute("SELECT COUNT(*) AS count FROM training_exposure");
  const grouped = (rows: Array<Record<string, unknown>>): Record<string, number> => Object.fromEntries(
    rows.map((row) => [String(row["key"]), Number(row["count"])])
  );
  return {
    campaignSlug,
    assignments: grouped(assignments.rows as Array<Record<string, unknown>>),
    reviews: grouped(reviews.rows as Array<Record<string, unknown>>),
    humanReviewArtifacts: Number(artifactCount.rows[0]!["count"]),
    candidateStatuses: grouped(candidates.rows as Array<Record<string, unknown>>),
    releaseMembers: Number(releaseMembers.rows[0]!["count"]),
    trainingExposures: Number(trainingExposures.rows[0]!["count"])
  };
}
