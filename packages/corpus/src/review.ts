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

interface PacketAssignmentRow extends AssignmentRow {
  presentationId?: string;
  opaqueItemId?: string;
  presentationKind?: "primary" | "hidden_repeat";
  sourceReviewId?: string;
}

interface RepeatCandidateRow extends AssignmentRow {
  sourceReviewId: string;
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
  presentations: Record<string, number>;
  repeatStabilityRows: number;
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

export async function ensureHumanActor(ledger: Ledger, alias: string): Promise<string> {
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

export async function ensureHumanReviewRubric(ledger: Ledger): Promise<string> {
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

async function openPresentationAssignments(
  ledger: Ledger,
  campaign: string,
  actorId: string,
  rubricVersionId: string,
  pass: HumanReviewPass
): Promise<PacketAssignmentRow[]> {
  const result = await ledger.client.execute({
    sql: `SELECT rp.id AS presentation_id, rp.opaque_item_id, rp.presentation_kind, rp.source_review_id,
                 rp.review_assignment_id AS assignment_id, ra.blindness_json,
                 cc.candidate_version_id, cc.family_slug, cc.kind, cc.status,
                 cc.content_json, cc.hidden_contract_json, cc.content_sha256,
                 rps.id AS session_id, rps.seed
          FROM review_presentation rp
          JOIN review_presentation_session rps ON rps.id = rp.session_id
          JOIN review_assignment ra ON ra.id = rp.review_assignment_id
          JOIN corpus_candidate_current cc ON cc.candidate_version_id = ra.candidate_version_id
          JOIN generation_campaign gc ON gc.id = cc.campaign_id
          WHERE gc.slug = ? AND rps.reviewer_actor_id = ? AND rps.rubric_version_id = ?
            AND rps.review_pass = ? AND rps.status = 'assigned' AND rp.status = 'assigned'
          ORDER BY rp.ordinal`,
    args: [campaign, actorId, rubricVersionId, pass]
  });
  return result.rows.map((raw) => {
    const row = raw as Record<string, unknown>;
    const blindness = parseJsonRecord(String(row["blindness_json"]), "assignment blindness");
    blindness["sessionId"] = String(row["session_id"]);
    blindness["seed"] = String(row["seed"]);
    return {
      ...candidateFromRow(row),
      assignmentId: String(row["assignment_id"]),
      blindness,
      presentationId: String(row["presentation_id"]),
      opaqueItemId: String(row["opaque_item_id"]),
      presentationKind: String(row["presentation_kind"]) as "primary" | "hidden_repeat",
      sourceReviewId: row["source_review_id"] === null ? undefined : String(row["source_review_id"])
    };
  });
}

async function completedRepeatCount(
  ledger: Ledger,
  campaign: string,
  actorId: string,
  rubricVersionId: string
): Promise<number> {
  const result = await ledger.client.execute({
    sql: `SELECT COUNT(*) AS count
          FROM review_presentation rp
          JOIN review_presentation_session rps ON rps.id = rp.session_id
          JOIN generation_campaign gc ON gc.id = rps.campaign_id
          WHERE gc.slug = ? AND rps.reviewer_actor_id = ? AND rps.rubric_version_id = ?
            AND rps.review_pass = 'A' AND rp.presentation_kind = 'hidden_repeat'`,
    args: [campaign, actorId, rubricVersionId]
  });
  return Number(result.rows[0]!["count"]);
}

async function selectRepeatCandidates(
  ledger: Ledger,
  campaign: string,
  actorId: string,
  rubricVersionId: string,
  limit: number,
  seed: string
): Promise<RepeatCandidateRow[]> {
  if (limit <= 0) return [];
  const result = await ledger.client.execute({
    sql: `SELECT ra.id AS assignment_id, ra.blindness_json, r.id AS source_review_id,
                 cc.candidate_version_id, cc.family_slug, cc.kind, cc.status,
                 cc.content_json, cc.hidden_contract_json, cc.content_sha256
          FROM review_assignment ra
          JOIN review r ON json_extract(r.rationale, '$.assignmentId') = ra.id
          JOIN corpus_candidate_current cc ON cc.candidate_version_id = ra.candidate_version_id
          JOIN generation_campaign gc ON gc.id = cc.campaign_id
          WHERE gc.slug = ? AND ra.reviewer_actor_id = ? AND ra.rubric_version_id = ?
            AND ra.status = 'completed' AND json_extract(ra.blindness_json, '$.pass') = 'A'
            AND json_extract(r.rationale, '$.pass') = 'A'
            AND NOT EXISTS (
              SELECT 1 FROM review_presentation prior
              WHERE prior.review_assignment_id = ra.id AND prior.presentation_kind = 'hidden_repeat'
            )`,
    args: [campaign, actorId, rubricVersionId]
  });
  return result.rows
    .map((raw) => {
      const row = raw as Record<string, unknown>;
      return {
        ...candidateFromRow(row),
        assignmentId: String(row["assignment_id"]),
        blindness: parseJsonRecord(String(row["blindness_json"]), "assignment blindness"),
        sourceReviewId: String(row["source_review_id"])
      };
    })
    .sort((left, right) => seededRank(seed, `${left.candidateVersionId}:repeat`)
      .localeCompare(seededRank(seed, `${right.candidateVersionId}:repeat`)))
    .slice(0, limit);
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

async function assertPassBPrerequisites(
  ledger: Ledger,
  resolvedCampaignId: string,
  campaignSlug: string,
  actorId: string,
  rubricVersionId: string
): Promise<void> {
  const [candidateEvidence, repeatEvidence, openSessions] = await Promise.all([
    ledger.client.execute({
      sql: `SELECT COUNT(*) AS candidate_count,
                   COALESCE(SUM(CASE WHEN EXISTS (
                     SELECT 1
                     FROM review_assignment ra
                     JOIN review r
                       ON r.candidate_version_id = ra.candidate_version_id
                      AND r.reviewer_actor_id = ra.reviewer_actor_id
                     WHERE ra.candidate_version_id = cc.candidate_version_id
                       AND ra.reviewer_actor_id = ? AND ra.rubric_version_id = ?
                       AND ra.status = 'completed'
                       AND json_valid(ra.blindness_json)
                       AND json_extract(ra.blindness_json, '$.pass') = 'A'
                       AND json_valid(r.rationale)
                       AND json_extract(r.rationale, '$.pass') = 'A'
                       AND json_extract(r.rationale, '$.assignmentId') = ra.id
                     GROUP BY ra.id
                     HAVING COUNT(r.id) = 1
                   ) THEN 1 ELSE 0 END), 0) AS pass_a_completed
            FROM corpus_candidate_current cc
            WHERE cc.campaign_id = ?`,
      args: [actorId, rubricVersionId, resolvedCampaignId]
    }),
    ledger.client.execute({
      sql: `SELECT COUNT(*) AS count
            FROM review_repeat_stability rrs
            JOIN review_presentation rp ON rp.id = rrs.presentation_id
            JOIN review_presentation_session rps ON rps.id = rp.session_id
            WHERE rrs.campaign_id = ? AND rps.reviewer_actor_id = ?
              AND rps.rubric_version_id = ? AND rps.review_pass = 'A'
              AND rps.status = 'completed' AND rp.status = 'completed'`,
      args: [resolvedCampaignId, actorId, rubricVersionId]
    }),
    ledger.client.execute({
      sql: `SELECT COUNT(*) AS count FROM review_presentation_session
            WHERE campaign_id = ? AND reviewer_actor_id = ? AND rubric_version_id = ?
              AND review_pass = 'A' AND status = 'assigned'`,
      args: [resolvedCampaignId, actorId, rubricVersionId]
    })
  ]);
  const candidateCount = Number(candidateEvidence.rows[0]!["candidate_count"]);
  const completedPassA = Number(candidateEvidence.rows[0]!["pass_a_completed"]);
  const expectedRepeats = Math.min(6, candidateCount);
  const completedRepeats = Number(repeatEvidence.rows[0]!["count"]);
  const assignedPassASessions = Number(openSessions.rows[0]!["count"]);
  if (candidateCount < 1) throw new Error(`Campaign ${campaignSlug} has no current candidates`);
  if (completedPassA !== candidateCount || completedRepeats !== expectedRepeats || assignedPassASessions !== 0) {
    throw new Error(
      `Pass B is locked until blinded Pass A is sealed for every current candidate and all hidden repeats: `
      + `${completedPassA}/${candidateCount} candidate reviews, ${completedRepeats}/${expectedRepeats} `
      + `repeat-stability rows, ${assignedPassASessions} open first-class Pass A presentation sessions`
    );
  }
}

function packetInstructions(pass: HumanReviewPass): string[] {
  if (pass === "A") {
    return [
      "Review only the model-visible messages. Do not inspect the public ledger or hidden contract for this item first.",
      "Fill every response field. Use scores 0-4; quote exact evidence for findings.",
      "Judge conceptual plausibility and conversational quality separately. Seal Pass A before preparing Pass B.",
      "A later session may contain blinded consistency presentations. Review every item independently; do not inspect presentation lineage."
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
  const resolvedCampaignId = await campaignId(ledger, options.campaignSlug);
  const actorId = await ensureHumanActor(ledger, options.reviewerAlias);
  const rubricVersionId = await ensureHumanReviewRubric(ledger);
  if (options.pass === "B") {
    await assertPassBPrerequisites(
      ledger, resolvedCampaignId, options.campaignSlug, actorId, rubricVersionId
    );
  }
  let assignments: PacketAssignmentRow[] = await openPresentationAssignments(
    ledger, options.campaignSlug, actorId, rubricVersionId, options.pass
  );
  if (assignments.length === 0) {
    assignments = await openAssignments(
      ledger, options.campaignSlug, actorId, rubricVersionId, options.pass
    );
  }
  const resumed = assignments.length > 0;
  let sessionId: string;
  let seed: string;
  if (resumed) {
    sessionId = String(assignments[0]!.blindness["sessionId"]);
    seed = String(assignments[0]!.blindness["seed"]);
  } else {
    sessionId = `review_session_${randomUUID()}`;
    seed = options.seed ?? randomUUID();
    const primaryPool = await selectCandidates(
      ledger, options.campaignSlug, actorId, rubricVersionId, options.pass, options.limit, seed
    );
    let repeats: RepeatCandidateRow[] = [];
    if (options.pass === "A") {
      const repeatCount = await completedRepeatCount(
        ledger, options.campaignSlug, actorId, rubricVersionId
      );
      const repeatRemaining = Math.max(0, 6 - repeatCount);
      const repeatSlots = primaryPool.length > 0
        ? Math.min(2, repeatRemaining, Math.max(0, options.limit - 1))
        : Math.min(repeatRemaining, options.limit);
      repeats = await selectRepeatCandidates(
        ledger, options.campaignSlug, actorId, rubricVersionId, repeatSlots, seed
      );
    }
    const candidates = primaryPool.slice(0, Math.max(0, options.limit - repeats.length));
    if (candidates.length === 0 && repeats.length === 0) {
      throw new Error(`No candidates are eligible for Pass ${options.pass}`);
    }
    const ts = now();
    const statements: Array<{ sql: string; args: InValue[] }> = [];
    const entries: Array<{
      candidate: CandidateRow | RepeatCandidateRow;
      assignmentId: string;
      presentationKind: "primary" | "hidden_repeat";
      sourceReviewId?: string;
    }> = [
      ...candidates.map((candidate) => ({
        candidate,
        assignmentId: stableId(
          "assignment", `${candidate.candidateVersionId}:${actorId}:${rubricVersionId}:${options.pass}`
        ),
        presentationKind: "primary" as const
      })),
      ...repeats.map((candidate) => ({
        candidate,
        assignmentId: candidate.assignmentId,
        presentationKind: "hidden_repeat" as const,
        sourceReviewId: candidate.sourceReviewId
      }))
    ];
    entries.sort((left, right) => seededRank(
      seed, `${left.candidate.candidateVersionId}:${left.presentationKind}:${left.sourceReviewId ?? ""}`
    ).localeCompare(seededRank(
      seed, `${right.candidate.candidateVersionId}:${right.presentationKind}:${right.sourceReviewId ?? ""}`
    )));
    const inputSnapshotSha256 = sha256Bytes(canonicalJson(entries.map((entry) => ({
      assignmentId: entry.assignmentId,
      candidateVersionId: entry.candidate.candidateVersionId,
      candidateContentSha256: entry.candidate.contentSha256,
      presentationKind: entry.presentationKind,
      sourceReviewId: entry.sourceReviewId ?? null
    })) as unknown as JsonValue));
    statements.push({
      sql: `INSERT INTO review_presentation_session
            (id, campaign_id, reviewer_actor_id, rubric_version_id, review_pass, seed,
             input_snapshot_sha256, requested_presentations, repeat_presentations, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'assigned', ?, ?)`,
      args: [sessionId, resolvedCampaignId, actorId, rubricVersionId, options.pass, seed,
        inputSnapshotSha256, entries.length, repeats.length, ts, ts]
    });
    assignments = entries.map((entry, index) => {
      const blindness = {
        pass: options.pass,
        sessionId,
        seed,
        presentationIndex: index + 1,
        hiddenRepeat: entry.presentationKind === "hidden_repeat",
        visibleFields: options.pass === "A" ? ["kind", "messages"] : ["all_candidate_and_contract_fields"],
        hiddenFields: options.pass === "A"
          ? ["identity", "family", "title", "difficulty", "generator_notes", "response_policy", "lenses", "transformation", "hidden_contract", "structural_status", "other_reviews"]
          : ["other_reviews"]
      };
      if (entry.presentationKind === "primary") {
        statements.push({
          sql: `INSERT INTO review_assignment
                (id, candidate_version_id, reviewer_actor_id, rubric_version_id, blindness_json,
                 status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'assigned', ?, ?)`,
          args: [entry.assignmentId, entry.candidate.candidateVersionId, actorId, rubricVersionId,
            canonicalJson(blindness as unknown as JsonValue), ts, ts]
        });
      }
      const presentationId = stableId(
        "presentation", `${sessionId}:${entry.assignmentId}:${entry.presentationKind}`
      );
      const opaqueItemId = stableId(
        "opaque", `${sessionId}:${entry.candidate.candidateVersionId}:${presentationId}`
      ).slice(0, 19);
      statements.push({
        sql: `INSERT INTO review_presentation
              (id, session_id, review_assignment_id, presentation_kind, source_review_id, ordinal,
               opaque_item_id, candidate_content_sha256, status, created_at, updated_at)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'assigned', ?, ?)`,
        args: [presentationId, sessionId, entry.assignmentId, entry.presentationKind,
          entry.sourceReviewId ?? null, index + 1, opaqueItemId, entry.candidate.contentSha256, ts, ts]
      });
      return {
        ...entry.candidate,
        assignmentId: entry.assignmentId,
        blindness,
        presentationId,
        opaqueItemId,
        presentationKind: entry.presentationKind,
        sourceReviewId: entry.sourceReviewId
      };
    });
    await ledger.client.batch(statements, "write");
  }
  const packetAssignments: HumanReviewPacketAssignment[] = assignments.map((assignment) => {
    const packetAssignment: HumanReviewPacketAssignment = {
      assignmentId: assignment.assignmentId,
      opaqueItemId: assignment.opaqueItemId
        ?? stableId("opaque", `${sessionId}:${assignment.candidateVersionId}`).slice(0, 19),
      candidateContentSha256: assignment.contentSha256,
      candidate: visibleCandidate(options.pass, assignment),
      response: emptyHumanReviewResponse(options.pass)
    };
    if (assignment.presentationId !== undefined) packetAssignment.presentationId = assignment.presentationId;
    return packetAssignment;
  });
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
): Promise<{
  submitted: number;
  primaryReviews: number;
  repeatResponses: number;
  pass: HumanReviewPass;
  sessionId: string;
  submissionSha256: string;
}> {
  const submissionBytes = readFileSync(resolve(path));
  const packet = parseHumanReviewPacketText(submissionBytes.toString("utf8"));
  const actorId = await ensureHumanActor(ledger, packet.reviewerAlias);
  const rubricVersionId = await ensureHumanReviewRubric(ledger);
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
    presentationId?: string;
    presentationKind: "legacy_primary" | "primary" | "hidden_repeat";
    sourceReviewId?: string;
  }> = [];
  for (const assignment of packet.assignments) {
    if (assignmentIds.has(assignment.assignmentId)) throw new Error(`Duplicate assignment ${assignment.assignmentId}`);
    assignmentIds.add(assignment.assignmentId);
    const responseErrors = humanReviewResponseErrors(packet.pass, assignment.response, assignment.opaqueItemId);
    if (responseErrors.length > 0) throw new Error(responseErrors[0]);
    const stored = assignment.presentationId === undefined
      ? await ledger.client.execute({
        sql: `SELECT ra.candidate_version_id, ra.status AS assignment_status, ra.blindness_json,
                     cv.content_sha256, cv.candidate_id, ra.reviewer_actor_id, ra.rubric_version_id,
                     NULL AS presentation_id, 'legacy_primary' AS presentation_kind,
                     NULL AS source_review_id, NULL AS presentation_status, NULL AS presentation_session_id
              FROM review_assignment ra
              JOIN candidate_version cv ON cv.id = ra.candidate_version_id
              WHERE ra.id = ?`,
        args: [assignment.assignmentId]
      })
      : await ledger.client.execute({
        sql: `SELECT ra.candidate_version_id, ra.status AS assignment_status, ra.blindness_json,
                     cv.content_sha256, cv.candidate_id, ra.reviewer_actor_id, ra.rubric_version_id,
                     rp.id AS presentation_id, rp.presentation_kind, rp.source_review_id,
                     rp.status AS presentation_status, rps.id AS presentation_session_id,
                     rps.review_pass, rps.status AS session_status
              FROM review_presentation rp
              JOIN review_presentation_session rps ON rps.id = rp.session_id
              JOIN review_assignment ra ON ra.id = rp.review_assignment_id
              JOIN candidate_version cv ON cv.id = ra.candidate_version_id
              WHERE rp.id = ? AND ra.id = ?`,
        args: [assignment.presentationId, assignment.assignmentId]
      });
    if (stored.rows.length !== 1) throw new Error(`Unknown review assignment ${assignment.assignmentId}`);
    const row = stored.rows[0]!;
    const blindness = parseJsonRecord(String(row["blindness_json"]), "stored blindness");
    const presentationKind = String(row["presentation_kind"]) as "legacy_primary" | "primary" | "hidden_repeat";
    if (presentationKind === "legacy_primary") {
      if (String(row["assignment_status"]) !== "assigned" || String(blindness["pass"]) !== packet.pass
        || String(blindness["sessionId"]) !== packet.sessionId) {
        throw new Error(`Assignment ${assignment.assignmentId} is not open for this pass/session`);
      }
    } else {
      if (String(row["presentation_id"]) !== assignment.presentationId
        || String(row["presentation_status"]) !== "assigned"
        || String(row["presentation_session_id"]) !== packet.sessionId
        || String(row["session_status"]) !== "assigned"
        || String(row["review_pass"]) !== packet.pass) {
        throw new Error(`Presentation ${assignment.presentationId} is not open for this pass/session`);
      }
      const expectedAssignmentStatus = presentationKind === "primary" ? "assigned" : "completed";
      if (String(row["assignment_status"]) !== expectedAssignmentStatus) {
        throw new Error(`Presentation ${assignment.presentationId} has an invalid assignment state`);
      }
    }
    if (String(row["reviewer_actor_id"]) !== actorId || String(row["rubric_version_id"]) !== rubricVersionId) {
      throw new Error(`Assignment ${assignment.assignmentId} reviewer or rubric mismatch`);
    }
    if (String(row["content_sha256"]) !== assignment.candidateContentSha256) {
      throw new Error(`Assignment ${assignment.assignmentId} candidate version changed`);
    }
    const entry: {
      assignment: HumanReviewPacketAssignment;
      candidateVersionId: string;
      candidateId: string;
      presentationId?: string;
      presentationKind: "legacy_primary" | "primary" | "hidden_repeat";
      sourceReviewId?: string;
    } = {
      assignment,
      candidateVersionId: String(row["candidate_version_id"]),
      candidateId: String(row["candidate_id"]),
      presentationKind
    };
    if (assignment.presentationId !== undefined) entry.presentationId = assignment.presentationId;
    if (row["source_review_id"] !== null) entry.sourceReviewId = String(row["source_review_id"]);
    validated.push(entry);
  }
  const presentationEntries = validated.filter((entry) => entry.presentationId !== undefined);
  if (presentationEntries.length > 0) {
    if (presentationEntries.length !== validated.length) {
      throw new Error("A review packet cannot mix legacy assignments and first-class presentations");
    }
    const open = await ledger.client.execute({
      sql: `SELECT COUNT(*) AS count FROM review_presentation
            WHERE session_id = ? AND status = 'assigned'`,
      args: [packet.sessionId]
    });
    if (Number(open.rows[0]!["count"]) !== presentationEntries.length) {
      throw new Error("Review packet does not contain every open presentation in its session");
    }
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
    const createsReview = entry.presentationKind !== "hidden_repeat";
    const reviewId = createsReview
      ? stableId("review", `${entry.assignment.assignmentId}:${submissionSha256}`)
      : undefined;
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
      presentationId: entry.presentationId ?? null,
      submissionSha256
    } as unknown as JsonValue);
    if (createsReview && reviewId !== undefined) {
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
          presentationId: entry.presentationId ?? null,
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
    if (entry.presentationId !== undefined) {
      const presentationResponseId = stableId(
        "presentation_response", `${entry.presentationId}:${submissionSha256}`
      );
      statements.push({
        sql: `INSERT INTO review_presentation_response
              (id, presentation_id, reviewer_actor_id, created_review_id, outcome, response_json,
               confidence, submission_blob_sha256, created_at)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        args: [presentationResponseId, entry.presentationId, actorId, reviewId ?? null, response.outcome!,
          canonicalJson(response as unknown as JsonValue), response.confidence!, submissionSha256, ts]
      });
      for (const [dimension, score] of Object.entries(response.scores)) {
        statements.push({
          sql: `INSERT INTO review_presentation_score
                (id, presentation_response_id, dimension, score, created_at)
                VALUES (?, ?, ?, ?, ?)`,
          args: [stableId("presentation_score", `${presentationResponseId}:${dimension}`),
            presentationResponseId, dimension, score!, ts]
        });
      }
      for (const [index, finding] of response.findings.entries()) {
        statements.push({
          sql: `INSERT INTO review_presentation_finding
                (id, presentation_response_id, ordinal, dimension, severity, evidence, recommendation, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
          args: [stableId("presentation_finding", `${presentationResponseId}:${index + 1}`),
            presentationResponseId, index + 1, finding.dimension, finding.severity,
            finding.evidence, finding.recommendation, ts]
        });
      }
      statements.push({
        sql: "UPDATE review_presentation SET status = 'completed', updated_at = ? WHERE id = ? AND status = 'assigned'",
        args: [ts, entry.presentationId]
      });
      if (entry.presentationKind === "hidden_repeat") {
        if (entry.sourceReviewId === undefined) throw new Error("Hidden repeat is missing its source review");
        const repeatEventId = stableId("event", `human-review-repeat-submitted:${presentationResponseId}`);
        statements.push({
          sql: `INSERT INTO event(id, event_type, object_kind, object_id, payload_json, created_at)
                VALUES (?, 'human_review_repeat_submitted', 'review_presentation_response', ?, ?, ?)`,
          args: [repeatEventId, presentationResponseId, canonicalJson({
            presentationId: entry.presentationId,
            sourceReviewId: entry.sourceReviewId,
            candidateId: entry.candidateId,
            sessionId: packet.sessionId,
            submissionSha256
          } as JsonValue), ts]
        });
        statements.push({
          sql: `INSERT INTO event_object(id, event_id, object_kind, object_id, created_at)
                VALUES (?, ?, 'candidate_version', ?, ?)`,
          args: [stableId("eventobj", `${repeatEventId}:${entry.candidateVersionId}`),
            repeatEventId, entry.candidateVersionId, ts]
        });
      }
    }
  }
  if (presentationEntries.length > 0) {
    statements.push({
      sql: `UPDATE review_presentation_session SET status = 'completed', updated_at = ?
            WHERE id = ? AND status = 'assigned'
              AND NOT EXISTS (SELECT 1 FROM review_presentation rp WHERE rp.session_id = ? AND rp.status <> 'completed')`,
      args: [ts, packet.sessionId, packet.sessionId]
    });
  }
  await ledger.client.batch(statements, "write");
  return {
    submitted: validated.length,
    primaryReviews: validated.filter((entry) => entry.presentationKind !== "hidden_repeat").length,
    repeatResponses: validated.filter((entry) => entry.presentationKind === "hidden_repeat").length,
    pass: packet.pass,
    sessionId: packet.sessionId,
    submissionSha256
  };
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
  const presentations = await ledger.client.execute({
    sql: `SELECT rp.presentation_kind || ':' || rp.status AS key, COUNT(*) AS count
          FROM review_presentation rp
          JOIN review_presentation_session rps ON rps.id = rp.session_id
          WHERE rps.campaign_id = ? GROUP BY key`,
    args: [id]
  });
  const repeatStability = await ledger.client.execute({
    sql: "SELECT COUNT(*) AS count FROM review_repeat_stability WHERE campaign_id = ?",
    args: [id]
  });
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
    presentations: grouped(presentations.rows as Array<Record<string, unknown>>),
    repeatStabilityRows: Number(repeatStability.rows[0]!["count"]),
    candidateStatuses: grouped(candidates.rows as Array<Record<string, unknown>>),
    releaseMembers: Number(releaseMembers.rows[0]!["count"]),
    trainingExposures: Number(trainingExposures.rows[0]!["count"])
  };
}
