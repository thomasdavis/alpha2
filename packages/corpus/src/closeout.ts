import type { InValue } from "@libsql/client";
import { randomUUID } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { putBlob, type Ledger } from "./db.js";
import { canonicalJson, sha256Bytes, stableId } from "./hash.js";
import {
  ensureHumanActor,
  ensureHumanReviewRubric,
  requireHumanActor,
  requireHumanReviewRubric
} from "./review.js";
import { requireExportedPacketEnvelope } from "./packet-envelope.js";
import { writeAtomic } from "./storage.js";
import {
  CAMPAIGN_CLOSEOUT_RUBRIC_SLUG,
  CAMPAIGN_CLOSEOUT_RUBRIC_VERSION,
  campaignCloseoutPacketEnvelopeJson,
  campaignCloseoutResponseErrors,
  closeoutContractDefinition,
  emptyCampaignCloseoutResponse,
  parseCampaignCloseoutPacketText,
  type CampaignCloseoutAnalysisEvidence,
  type CampaignCloseoutCandidateEvidence,
  type CampaignCloseoutFamilyEvidence,
  type CampaignCloseoutPacket,
  type CampaignCloseoutRepeatEvidence
} from "./closeout-contract.js";
import type { JsonValue } from "./types.js";

interface CloseoutEvidence {
  campaignId: string;
  candidates: CampaignCloseoutCandidateEvidence[];
  families: CampaignCloseoutFamilyEvidence[];
  repeats: CampaignCloseoutRepeatEvidence[];
  analysis: CampaignCloseoutAnalysisEvidence;
  structurallyRejected: number;
  expectedRepeats: number;
  inputSnapshotSha256: string;
}

export interface PrepareCampaignCloseoutOptions {
  campaignSlug: string;
  adjudicatorAlias: string;
  outputDirectory?: string;
}

export interface PreparedCampaignCloseout {
  packetPath: string;
  markdownPath: string;
  sessionId: string;
  candidateCount: number;
  familyCount: number;
  repeatCount: number;
  packetSha256: string;
  resumed: boolean;
}

export interface SubmittedCampaignCloseout {
  sessionId: string;
  candidateAdjudications: number;
  failureClusters: number;
  recommendedStates: number;
  packetEnvelopeSha256: string;
  submissionSha256: string;
  executionAuthorized: false;
}

export interface CampaignCloseoutStatus {
  campaignSlug: string;
  assignments: Record<string, number>;
  campaignCloseouts: number;
  candidateAdjudications: number;
  failureClusters: number;
  recommendedStates: Record<string, number>;
  releaseMembers: number;
  trainingExposures: number;
  executionAuthorizations: number;
}

function now(): string {
  return new Date().toISOString();
}

function parseJson(text: string, label: string): JsonValue {
  try {
    return JSON.parse(text) as JsonValue;
  } catch {
    throw new Error(`${label} is not valid JSON`);
  }
}

function exactWrite(path: string, bytes: string): void {
  if (existsSync(path)) {
    const existing = readFileSync(path);
    if (sha256Bytes(existing) !== sha256Bytes(bytes)) {
      throw new Error(`Refusing to replace an edited or non-identical closeout packet at ${path}`);
    }
    return;
  }
  writeAtomic(path, bytes);
}

async function campaignId(ledger: Ledger, slug: string): Promise<string> {
  const result = await ledger.client.execute({
    sql: "SELECT id FROM generation_campaign WHERE slug = ?",
    args: [slug]
  });
  if (result.rows.length !== 1) throw new Error(`Unknown campaign ${slug}`);
  return String(result.rows[0]!["id"]);
}

async function ensureCloseoutRubric(ledger: Ledger): Promise<string> {
  const rubricId = stableId("rubric", CAMPAIGN_CLOSEOUT_RUBRIC_SLUG);
  const definitionJson = canonicalJson(closeoutContractDefinition());
  const digest = sha256Bytes(definitionJson);
  const versionId = stableId(
    "rubricv", `${CAMPAIGN_CLOSEOUT_RUBRIC_SLUG}:${CAMPAIGN_CLOSEOUT_RUBRIC_VERSION}:${digest}`
  );
  const ts = now();
  await ledger.client.batch([
    {
      sql: "INSERT OR IGNORE INTO rubric(id, slug, created_at) VALUES (?, ?, ?)",
      args: [rubricId, CAMPAIGN_CLOSEOUT_RUBRIC_SLUG, ts]
    },
    {
      sql: `INSERT OR IGNORE INTO rubric_version
            (id, rubric_id, version, definition_json, content_sha256, created_at)
            VALUES (?, ?, ?, ?, ?, ?)`,
      args: [versionId, rubricId, CAMPAIGN_CLOSEOUT_RUBRIC_VERSION, definitionJson, digest, ts]
    }
  ], "write");
  const stored = await ledger.client.execute({
    sql: "SELECT content_sha256 FROM rubric_version WHERE id = ?",
    args: [versionId]
  });
  if (stored.rows.length !== 1 || String(stored.rows[0]!["content_sha256"]) !== digest) {
    throw new Error("Stored D5 campaign-closeout rubric differs from the executable definition");
  }
  return versionId;
}

async function requireCloseoutRubric(ledger: Ledger): Promise<string> {
  const definitionJson = canonicalJson(closeoutContractDefinition());
  const digest = sha256Bytes(definitionJson);
  const versionId = stableId(
    "rubricv", `${CAMPAIGN_CLOSEOUT_RUBRIC_SLUG}:${CAMPAIGN_CLOSEOUT_RUBRIC_VERSION}:${digest}`
  );
  const stored = await ledger.client.execute({
    sql: "SELECT content_sha256 FROM rubric_version WHERE id = ?",
    args: [versionId]
  });
  if (stored.rows.length !== 1 || String(stored.rows[0]!["content_sha256"]) !== digest) {
    throw new Error("D5 campaign-closeout rubric was not registered by packet preparation");
  }
  return versionId;
}

async function loadReview(
  ledger: Ledger,
  candidateVersionId: string,
  actorId: string,
  candidateRubricVersionId: string,
  pass: "A" | "B"
): Promise<{ id: string; evidence: JsonValue }> {
  const reviews = await ledger.client.execute({
    sql: `SELECT r.id, r.outcome, r.rationale, r.created_at
          FROM review_assignment ra
          JOIN review r ON json_extract(r.rationale, '$.assignmentId') = ra.id
          WHERE ra.candidate_version_id = ? AND ra.reviewer_actor_id = ?
            AND ra.rubric_version_id = ? AND ra.status = 'completed'
            AND json_extract(ra.blindness_json, '$.pass') = ?
            AND json_extract(r.rationale, '$.pass') = ?`,
    args: [candidateVersionId, actorId, candidateRubricVersionId, pass, pass]
  });
  if (reviews.rows.length !== 1) {
    throw new Error(`Candidate ${candidateVersionId} needs exactly one sealed Pass ${pass} review before Pass D`);
  }
  const row = reviews.rows[0]!;
  const reviewId = String(row["id"]);
  const scores = await ledger.client.execute({
    sql: "SELECT dimension, score FROM review_dimension_score WHERE review_id = ? ORDER BY dimension",
    args: [reviewId]
  });
  const dimensionAssessments = await ledger.client.execute({
    sql: `SELECT rde.dimension, rde.assessment_state, rde.evidence, rds.score
            FROM review_dimension_evidence rde
            LEFT JOIN review_dimension_score rds
              ON rds.review_id = rde.review_id AND rds.dimension = rde.dimension
           WHERE rde.review_id = ? ORDER BY rde.dimension`,
    args: [reviewId]
  });
  const findings = await ledger.client.execute({
    sql: `SELECT rf.dimension, rf.severity, rf.evidence, rf.recommendation,
                 rfe.why_it_matters, rfe.preserve
            FROM review_finding rf
            JOIN review_finding_explanation rfe ON rfe.review_finding_id = rf.id
           WHERE rf.review_id = ? ORDER BY rf.id`,
    args: [reviewId]
  });
  return {
    id: reviewId,
    evidence: {
      reviewId,
      pass,
      outcome: String(row["outcome"]),
      rationale: parseJson(String(row["rationale"]), `review ${reviewId} rationale`),
      scores: Object.fromEntries(scores.rows.map((score) => [
        String(score["dimension"]), Number(score["score"])
      ])),
      dimensionAssessments: Object.fromEntries(dimensionAssessments.rows.map((assessment) => [
        String(assessment["dimension"]),
        {
          state: String(assessment["assessment_state"]),
          score: assessment["score"] === null ? null : Number(assessment["score"]),
          evidence: String(assessment["evidence"])
        }
      ])),
      findings: findings.rows.map((finding) => ({
        dimension: String(finding["dimension"]),
        severity: String(finding["severity"]),
        evidence: String(finding["evidence"]),
        recommendation: String(finding["recommendation"]),
        whyItMatters: String(finding["why_it_matters"]),
        preserve: String(finding["preserve"])
      })),
      createdAt: String(row["created_at"])
    } as unknown as JsonValue
  };
}

async function loadCloseoutEvidence(
  ledger: Ledger,
  campaignSlug: string,
  actorId: string,
  candidateRubricVersionId: string
): Promise<CloseoutEvidence> {
  const resolvedCampaignId = await campaignId(ledger, campaignSlug);
  const rows = await ledger.client.execute({
    sql: `SELECT cc.candidate_id, cc.candidate_version_id, cc.family_slug, cc.status, cc.content_sha256,
                 fv.id AS family_version_id
          FROM corpus_candidate_current cc
          JOIN family_version fv ON fv.family_id = cc.family_id
            AND fv.version = (SELECT MAX(latest.version) FROM family_version latest WHERE latest.family_id = cc.family_id)
          WHERE cc.campaign_id = ? ORDER BY cc.family_slug, cc.candidate_id`,
    args: [resolvedCampaignId]
  });
  if (rows.rows.length === 0) throw new Error(`Campaign ${campaignSlug} has no current candidates`);

  // Check the human evidence in the same order it was collected. This makes a
  // failed closeout point at the next real campaign gate instead of whichever
  // downstream table happens to be queried first.
  const passAByCandidate = new Map<string, { id: string; evidence: JsonValue }>();
  for (const row of rows.rows) {
    const candidateVersionId = String(row["candidate_version_id"]);
    passAByCandidate.set(
      candidateVersionId,
      await loadReview(ledger, candidateVersionId, actorId, candidateRubricVersionId, "A")
    );
  }

  const repeatRows = await ledger.client.execute({
    sql: `SELECT rrs.presentation_id, rrs.repeat_response_id, rrs.source_review_id,
                 rrs.candidate_version_id, rrs.outcome_match, rrs.question_policy_match,
                 rrs.missing_clarification_match, rrs.confidence_delta, rrs.dimension_exact_rate,
                 rrs.mean_absolute_score_delta
          FROM review_repeat_stability rrs
          JOIN review_presentation rp ON rp.id = rrs.presentation_id
          JOIN review_presentation_session rps ON rps.id = rp.session_id
          WHERE rrs.campaign_id = ? AND rps.reviewer_actor_id = ?
          ORDER BY rrs.presentation_id`,
    args: [resolvedCampaignId, actorId]
  });
  const expectedRepeats = Math.min(6, rows.rows.length);
  if (repeatRows.rows.length !== expectedRepeats) {
    throw new Error(`Pass D needs ${expectedRepeats} completed hidden-repeat stability rows; found ${repeatRows.rows.length}`);
  }
  const repeats: CampaignCloseoutRepeatEvidence[] = repeatRows.rows.map((row) => ({
    presentationId: String(row["presentation_id"]),
    repeatResponseId: String(row["repeat_response_id"]),
    sourceReviewId: String(row["source_review_id"]),
    candidateVersionId: String(row["candidate_version_id"]),
    outcomeMatch: Number(row["outcome_match"]),
    questionPolicyMatch: Number(row["question_policy_match"]),
    missingClarificationMatch: Number(row["missing_clarification_match"]),
    confidenceDelta: Number(row["confidence_delta"]),
    dimensionExactRate: Number(row["dimension_exact_rate"]),
    meanAbsoluteScoreDelta: Number(row["mean_absolute_score_delta"])
  }));

  const passBByCandidate = new Map<string, { id: string; evidence: JsonValue }>();
  for (const row of rows.rows) {
    const candidateVersionId = String(row["candidate_version_id"]);
    passBByCandidate.set(
      candidateVersionId,
      await loadReview(ledger, candidateVersionId, actorId, candidateRubricVersionId, "B")
    );
  }

  const familyRows = await ledger.client.execute({
    sql: `SELECT fs.id, fs.family_version_id, cf.slug AS family_slug, fs.disposition,
                 fs.central_distinction, fs.coverage_json, fs.diagnosis_json, fs.rationale, fs.confidence
          FROM family_synthesis fs
          JOIN family_synthesis_assignment fsa ON fsa.id = fs.assignment_id
          JOIN family_version fv ON fv.id = fs.family_version_id
          JOIN concept_family cf ON cf.id = fv.family_id
          WHERE fsa.campaign_id = ? AND fsa.reviewer_actor_id = ? AND fsa.status = 'completed'
          ORDER BY cf.slug`,
    args: [resolvedCampaignId, actorId]
  });
  const familyVersions = new Set(rows.rows.map((row) => String(row["family_version_id"])));
  if (familyRows.rows.length !== familyVersions.size) {
    throw new Error(`Pass D needs exactly one completed family synthesis for each of ${familyVersions.size} families`);
  }
  const familyByVersion = new Map<string, CampaignCloseoutFamilyEvidence>();
  for (const row of familyRows.rows) {
    const familyVersionId = String(row["family_version_id"]);
    if (!familyVersions.has(familyVersionId) || familyByVersion.has(familyVersionId)) {
      throw new Error(`Pass D family synthesis set does not match current family ${familyVersionId}`);
    }
    const family: CampaignCloseoutFamilyEvidence = {
      familyVersionId,
      familySlug: String(row["family_slug"]),
      familySynthesisId: String(row["id"]),
      disposition: String(row["disposition"]),
      synthesis: {
        centralDistinction: String(row["central_distinction"]),
        coverage: parseJson(String(row["coverage_json"]), `family ${familyVersionId} coverage`),
        diagnosis: parseJson(String(row["diagnosis_json"]), `family ${familyVersionId} diagnosis`),
        rationale: String(row["rationale"]),
        confidence: Number(row["confidence"])
      } as unknown as JsonValue
    };
    familyByVersion.set(familyVersionId, family);
  }

  const structuralRows = await ledger.client.execute({
    sql: `SELECT sd.id, sd.candidate_version_id, sd.content_utility, sd.validator_finding_correctness,
                 sd.identified_value, sd.semantic_type, sd.remedy, sd.automatic_acceptance_hazard,
                 sd.automatic_rejection_hazard, sd.rationale, sd.confidence
          FROM structural_disposition sd
          JOIN family_synthesis fs ON fs.id = sd.family_synthesis_id
          JOIN family_synthesis_assignment fsa ON fsa.id = fs.assignment_id
          WHERE fsa.campaign_id = ? AND fsa.reviewer_actor_id = ?`,
    args: [resolvedCampaignId, actorId]
  });
  const structuralByCandidate = new Map<string, { id: string; evidence: JsonValue }>();
  for (const row of structuralRows.rows) {
    const candidateVersionId = String(row["candidate_version_id"]);
    if (structuralByCandidate.has(candidateVersionId)) {
      throw new Error(`Candidate ${candidateVersionId} has duplicate structural dispositions`);
    }
    structuralByCandidate.set(candidateVersionId, {
      id: String(row["id"]),
      evidence: {
        contentUtility: String(row["content_utility"]),
        validatorFindingCorrectness: String(row["validator_finding_correctness"]),
        identifiedValue: String(row["identified_value"]),
        semanticType: String(row["semantic_type"]),
        remedy: String(row["remedy"]),
        automaticAcceptanceHazard: String(row["automatic_acceptance_hazard"]),
        automaticRejectionHazard: String(row["automatic_rejection_hazard"]),
        rationale: String(row["rationale"]),
        confidence: Number(row["confidence"])
      } as unknown as JsonValue
    });
  }

  const candidates: CampaignCloseoutCandidateEvidence[] = [];
  let structurallyRejected = 0;
  for (const row of rows.rows) {
    const candidateVersionId = String(row["candidate_version_id"]);
    const passA = passAByCandidate.get(candidateVersionId)!;
    const passB = passBByCandidate.get(candidateVersionId)!;
    const familyVersionId = String(row["family_version_id"]);
    const family = familyByVersion.get(familyVersionId);
    if (family === undefined) throw new Error(`Candidate ${candidateVersionId} lacks a completed family synthesis`);
    const status = String(row["status"]);
    const structural = structuralByCandidate.get(candidateVersionId);
    if (status === "structurally_rejected") {
      structurallyRejected += 1;
      if (structural === undefined) {
        throw new Error(`Structurally rejected candidate ${candidateVersionId} needs a separate disposition before Pass D`);
      }
    } else if (structural !== undefined) {
      throw new Error(`Non-rejected candidate ${candidateVersionId} has an unexpected structural disposition`);
    }
    candidates.push({
      candidateId: String(row["candidate_id"]),
      candidateVersionId,
      familyVersionId,
      familySlug: String(row["family_slug"]),
      status,
      contentSha256: String(row["content_sha256"]),
      passAReviewId: passA.id,
      passBReviewId: passB.id,
      passAReview: passA.evidence,
      passBReview: passB.evidence,
      familySynthesisId: family.familySynthesisId,
      structuralDispositionId: structural?.id ?? null,
      structuralDisposition: structural?.evidence ?? null
    });
  }
  if (structuralByCandidate.size !== structurallyRejected) {
    throw new Error("Structural-disposition population does not match structurally rejected candidates");
  }

  const analysisRows = await ledger.client.execute({
    sql: `SELECT ar.id, ar.input_snapshot_sha256,
                 (SELECT COUNT(*) FROM analysis_metric am WHERE am.analysis_run_id = ar.id) AS metric_count,
                 (SELECT COUNT(*) FROM similarity_edge se WHERE se.analysis_run_id = ar.id) AS edge_count,
                 (SELECT COUNT(*) FROM template_signature ts WHERE ts.analysis_run_id = ar.id) AS signature_count
          FROM analysis_run ar
          LEFT JOIN analysis_run_correction correction ON correction.erroneous_analysis_run_id = ar.id
          WHERE ar.campaign_id = ? AND correction.erroneous_analysis_run_id IS NULL
          ORDER BY ar.completed_at DESC LIMIT 1`,
    args: [resolvedCampaignId]
  });
  if (analysisRows.rows.length !== 1) throw new Error("Pass D needs one current authoritative analysis run");
  const analysisRow = analysisRows.rows[0]!;
  const analysis: CampaignCloseoutAnalysisEvidence = {
    analysisRunId: String(analysisRow["id"]),
    inputSnapshotSha256: String(analysisRow["input_snapshot_sha256"]),
    metricCount: Number(analysisRow["metric_count"]),
    similarityEdgeCount: Number(analysisRow["edge_count"]),
    templateSignatureCount: Number(analysisRow["signature_count"])
  };
  const families = [...familyByVersion.values()].sort((a, b) => a.familySlug.localeCompare(b.familySlug));
  const snapshotPayload = {
    campaignId: resolvedCampaignId,
    candidates,
    families,
    repeats,
    analysis,
    structurallyRejected,
    expectedRepeats
  } as unknown as JsonValue;
  return {
    campaignId: resolvedCampaignId,
    candidates,
    families,
    repeats,
    analysis,
    structurallyRejected,
    expectedRepeats,
    inputSnapshotSha256: sha256Bytes(canonicalJson(snapshotPayload))
  };
}

function renderMarkdown(packet: CampaignCloseoutPacket): string {
  const lines = [
    "# Alpha D5 campaign closeout — Pass D",
    "",
    `- Campaign: \`${packet.campaignSlug}\``,
    `- Session: \`${packet.sessionId}\``,
    `- Adjudicator: \`${packet.adjudicatorAlias}\``,
    `- Candidates: ${packet.population.candidates}`,
    `- Families: ${packet.population.families}`,
    `- Structural dispositions: ${packet.population.structurallyRejected}`,
    `- Repeat stability rows: ${packet.population.completedRepeatPresentations}/${packet.population.expectedRepeatPresentations}`,
    `- Input snapshot: \`${packet.inputSnapshotSha256}\``,
    "",
    "> This packet is non-binding scientific synthesis. It cannot authorize model calls, generation, release, training, GPU use, or Donto mutation.",
    "",
    "Complete every response field in the JSON packet. Candidate dispositions remain calibration evidence and do not change candidate lifecycle state.",
    "",
    "## Frozen family syntheses",
    "",
    ...packet.families.map((family) => `- **${family.familySlug}:** ${family.disposition} (\`${family.familySynthesisId}\`)`),
    "",
    "## Frozen stability evidence",
    "",
    ...packet.repeats.map((repeat) => `- \`${repeat.presentationId}\`: outcome=${repeat.outcomeMatch}, dimension exact=${repeat.dimensionExactRate.toFixed(3)}, mean delta=${repeat.meanAbsoluteScoreDelta.toFixed(3)}`),
    "",
    "## Response worksheet",
    "",
    "Use `response.candidateDispositions`, `failureClusters`, `distributionAssessments`, `recommendedStates`, `known`, `unknown`, `proposedNext`, `disagreements`, and `overallRationale` in the JSON packet."
  ];
  return `${lines.join("\n")}\n`;
}

async function recordPacket(
  ledger: Ledger,
  packet: CampaignCloseoutPacket,
  packetPath: string,
  markdownPath: string
): Promise<string> {
  const packetText = canonicalJson(packet as unknown as JsonValue);
  const markdown = renderMarkdown(packet);
  exactWrite(packetPath, packetText);
  exactWrite(markdownPath, markdown);
  const packetSha = await putBlob(ledger, packetText, "application/json");
  const markdownSha = await putBlob(ledger, markdown, "text/markdown");
  const ts = now();
  await ledger.client.batch([
    {
      sql: `INSERT OR IGNORE INTO export_artifact
            (id, release_id, cohort_snapshot_id, format, blob_sha256, manifest_json, created_at)
            VALUES (?, NULL, NULL, 'campaign_closeout_packet_json', ?, ?, ?)`,
      args: [stableId("export", `campaign-closeout:${packet.sessionId}:json:${packetSha}`), packetSha,
        canonicalJson({ path: packetPath, pass: "D", sessionId: packet.sessionId } as JsonValue), ts]
    },
    {
      sql: `INSERT OR IGNORE INTO export_artifact
            (id, release_id, cohort_snapshot_id, format, blob_sha256, manifest_json, created_at)
            VALUES (?, NULL, NULL, 'campaign_closeout_packet_markdown', ?, ?, ?)`,
      args: [stableId("export", `campaign-closeout:${packet.sessionId}:markdown:${markdownSha}`), markdownSha,
        canonicalJson({ path: markdownPath, pass: "D", sessionId: packet.sessionId } as JsonValue), ts]
    }
  ], "write");
  return packetSha;
}

export async function prepareCampaignCloseoutPacket(
  ledger: Ledger,
  options: PrepareCampaignCloseoutOptions
): Promise<PreparedCampaignCloseout> {
  const actorId = await ensureHumanActor(ledger, options.adjudicatorAlias);
  const candidateRubricVersionId = await ensureHumanReviewRubric(ledger);
  const evidence = await loadCloseoutEvidence(
    ledger,
    options.campaignSlug,
    actorId,
    candidateRubricVersionId
  );
  const rubricVersionId = await ensureCloseoutRubric(ledger);
  const existing = await ledger.client.execute({
    sql: `SELECT id, session_id, input_snapshot_sha256, status, created_at
          FROM campaign_closeout_assignment
          WHERE campaign_id = ? AND adjudicator_actor_id = ? AND rubric_version_id = ?`,
    args: [evidence.campaignId, actorId, rubricVersionId]
  });
  let sessionId: string;
  let packetCreatedAt: string;
  let resumed = false;
  if (existing.rows.length === 1) {
    const row = existing.rows[0]!;
    if (String(row["status"]) !== "assigned") throw new Error("Campaign closeout assignment is already completed");
    if (String(row["input_snapshot_sha256"]) !== evidence.inputSnapshotSha256) {
      throw new Error("Campaign closeout evidence changed after assignment");
    }
    sessionId = String(row["session_id"]);
    packetCreatedAt = String(row["created_at"]);
    resumed = true;
  } else if (existing.rows.length > 1) {
    throw new Error("Campaign has duplicate closeout assignments");
  } else {
    sessionId = `closeout_session_${randomUUID()}`;
    const ts = now();
    packetCreatedAt = ts;
    await ledger.client.execute({
      sql: `INSERT INTO campaign_closeout_assignment
            (id, campaign_id, adjudicator_actor_id, rubric_version_id, session_id,
             input_snapshot_sha256, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 'assigned', ?, ?)`,
      args: [stableId("closeout_assignment", `${evidence.campaignId}:${actorId}:${rubricVersionId}`),
        evidence.campaignId, actorId, rubricVersionId, sessionId, evidence.inputSnapshotSha256, ts, ts]
    });
  }
  const packet: CampaignCloseoutPacket = {
    schemaVersion: 1,
    campaignSlug: options.campaignSlug,
    sessionId,
    adjudicatorAlias: options.adjudicatorAlias.trim(),
    rubricSlug: CAMPAIGN_CLOSEOUT_RUBRIC_SLUG,
    rubricVersion: CAMPAIGN_CLOSEOUT_RUBRIC_VERSION,
    inputSnapshotSha256: evidence.inputSnapshotSha256,
    createdAt: packetCreatedAt,
    population: {
      candidates: evidence.candidates.length,
      families: evidence.families.length,
      structurallyRejected: evidence.structurallyRejected,
      completedRepeatPresentations: evidence.repeats.length,
      expectedRepeatPresentations: evidence.expectedRepeats
    },
    candidates: evidence.candidates,
    families: evidence.families,
    repeats: evidence.repeats,
    analysis: evidence.analysis,
    response: emptyCampaignCloseoutResponse(evidence.candidates)
  };
  const directory = resolve(options.outputDirectory
    ?? join(ledger.paths.releases, "review", `${options.campaignSlug}-d-${sessionId}`));
  const packetPath = join(directory, "campaign-closeout-packet.json");
  const markdownPath = join(directory, "README.md");
  const packetSha256 = await recordPacket(ledger, packet, packetPath, markdownPath);
  return {
    packetPath,
    markdownPath,
    sessionId,
    candidateCount: evidence.candidates.length,
    familyCount: evidence.families.length,
    repeatCount: evidence.repeats.length,
    packetSha256,
    resumed
  };
}

function evidenceSnapshot(packet: CampaignCloseoutPacket, resolvedCampaignId: string): JsonValue {
  return {
    campaignId: resolvedCampaignId,
    candidates: packet.candidates,
    families: packet.families,
    repeats: packet.repeats,
    analysis: packet.analysis,
    structurallyRejected: packet.population.structurallyRejected,
    expectedRepeats: packet.population.expectedRepeatPresentations
  } as unknown as JsonValue;
}

export async function submitCampaignCloseoutPacket(
  ledger: Ledger,
  path: string
): Promise<SubmittedCampaignCloseout> {
  const bytes = readFileSync(resolve(path));
  const packet = parseCampaignCloseoutPacketText(bytes.toString("utf8"));
  if (packet.rubricSlug !== CAMPAIGN_CLOSEOUT_RUBRIC_SLUG
    || packet.rubricVersion !== CAMPAIGN_CLOSEOUT_RUBRIC_VERSION) {
    throw new Error("Campaign-closeout packet uses an unsupported rubric version");
  }
  const actorId = await requireHumanActor(ledger, packet.adjudicatorAlias);
  const candidateRubricVersionId = await requireHumanReviewRubric(ledger);
  const evidence = await loadCloseoutEvidence(
    ledger,
    packet.campaignSlug,
    actorId,
    candidateRubricVersionId
  );
  if (packet.population.candidates !== evidence.candidates.length
    || packet.population.families !== evidence.families.length
    || packet.population.structurallyRejected !== evidence.structurallyRejected
    || packet.population.completedRepeatPresentations !== evidence.repeats.length
    || packet.population.expectedRepeatPresentations !== evidence.expectedRepeats) {
    throw new Error("Campaign-closeout packet population differs from the current frozen ledger evidence");
  }
  const expectedSnapshot = sha256Bytes(canonicalJson(evidenceSnapshot(packet, evidence.campaignId)));
  if (packet.inputSnapshotSha256 !== evidence.inputSnapshotSha256
    || expectedSnapshot !== evidence.inputSnapshotSha256) {
    throw new Error("Campaign-closeout packet evidence differs from the current frozen ledger evidence");
  }
  const rubricVersionId = await requireCloseoutRubric(ledger);
  const assignment = await ledger.client.execute({
    sql: `SELECT id, session_id, input_snapshot_sha256, status
          FROM campaign_closeout_assignment
          WHERE campaign_id = ? AND adjudicator_actor_id = ? AND rubric_version_id = ?`,
    args: [evidence.campaignId, actorId, rubricVersionId]
  });
  if (assignment.rows.length !== 1 || String(assignment.rows[0]!["status"]) !== "assigned"
    || String(assignment.rows[0]!["session_id"]) !== packet.sessionId
    || String(assignment.rows[0]!["input_snapshot_sha256"]) !== packet.inputSnapshotSha256) {
    throw new Error("Campaign-closeout assignment is not open for this evidence/session");
  }
  const errors = campaignCloseoutResponseErrors(packet);
  if (errors.length > 0) throw new Error(errors[0]);

  const packetEnvelopeSha256 = await requireExportedPacketEnvelope(ledger, {
    format: "campaign_closeout_packet_json",
    sessionId: packet.sessionId,
    pass: "D",
    envelopeJson: campaignCloseoutPacketEnvelopeJson(packet)
  }).catch((error: unknown) => {
    if (error instanceof Error && error.message === "Submission immutable envelope does not match an exported packet") {
      throw new Error("Campaign-closeout submission immutable envelope does not match an exported packet");
    }
    throw error;
  });

  const submissionSha256 = await putBlob(ledger, bytes, "application/json");
  const ts = now();
  const assignmentId = String(assignment.rows[0]!["id"]);
  const closeoutId = stableId("campaign_closeout", `${assignmentId}:${submissionSha256}`);
  const statements: Array<{ sql: string; args: InValue[] }> = [
    {
      sql: "INSERT OR IGNORE INTO raw_artifact(id, task_id, kind, blob_sha256, created_at) VALUES (?, NULL, 'campaign_closeout_submission', ?, ?)",
      args: [stableId("artifact", `campaign-closeout:${packet.sessionId}:${submissionSha256}`), submissionSha256, ts]
    },
    {
      sql: `INSERT INTO campaign_closeout
            (id, assignment_id, campaign_id, adjudicator_actor_id, recommendation_summary,
             known_json, unknown_json, proposed_next_json, disagreement_json, overall_rationale,
             confidence, execution_authorized, submission_blob_sha256, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)`,
      args: [closeoutId, assignmentId, evidence.campaignId, actorId,
        packet.response.recommendationSummary,
        canonicalJson(packet.response.known as unknown as JsonValue),
        canonicalJson(packet.response.unknown as unknown as JsonValue),
        canonicalJson(packet.response.proposedNext as unknown as JsonValue),
        canonicalJson({ disagreements: packet.response.disagreements,
          noDisagreementRationale: packet.response.noDisagreementRationale } as unknown as JsonValue),
        packet.response.overallRationale, packet.response.confidence!, submissionSha256, ts]
    }
  ];

  for (const candidateResponse of packet.response.candidateDispositions) {
    const candidate = evidence.candidates.find(
      (entry) => entry.candidateVersionId === candidateResponse.candidateVersionId
    )!;
    const adjudicationId = stableId("adjudication", `${closeoutId}:${candidate.candidateVersionId}`);
    statements.push({
      sql: `INSERT INTO adjudication
            (id, candidate_version_id, authority, outcome, rationale, created_at)
            VALUES (?, ?, ?, ?, ?, ?)`,
      args: [adjudicationId, candidate.candidateVersionId,
        `human:${packet.adjudicatorAlias.trim()}:d5-pass-d`, candidateResponse.outcome!, canonicalJson({
          rationale: candidateResponse.rationale,
          confidence: candidateResponse.confidence,
          uncertainty: candidateResponse.uncertainty,
          repairRequest: candidateResponse.repairRequest,
          preserve: candidateResponse.preserve,
          disagreementDescription: candidateResponse.disagreementDescription,
          campaignCloseoutId: closeoutId,
          executionAuthority: false
        } as unknown as JsonValue), ts]
    });
    const bases = [
      ["review_pass_a", candidate.passAReviewId],
      ["review_pass_b", candidate.passBReviewId],
      ["family_synthesis", candidate.familySynthesisId]
    ];
    if (candidate.structuralDispositionId !== null) {
      bases.push(["structural_disposition", candidate.structuralDispositionId]);
    }
    for (const [basisKind, basisId] of bases) {
      statements.push({
        sql: `INSERT INTO adjudication_basis(id, adjudication_id, basis_kind, basis_id, created_at)
              VALUES (?, ?, ?, ?, ?)`,
        args: [stableId("adjudication_basis", `${adjudicationId}:${basisKind}:${basisId}`),
          adjudicationId, basisKind!, basisId!, ts]
      });
    }
    statements.push({
      sql: `INSERT INTO campaign_closeout_basis
            (id, campaign_closeout_id, basis_kind, basis_id, created_at) VALUES (?, ?, 'adjudication', ?, ?)`,
      args: [stableId("closeout_basis", `${closeoutId}:adjudication:${adjudicationId}`),
        closeoutId, adjudicationId, ts]
    });
    if (candidateResponse.repairRequest.trim().length > 0) {
      statements.push({
        sql: `INSERT INTO repair_request
              (id, candidate_version_id, review_id, requested_change, preserve_json, status, created_at)
              VALUES (?, ?, ?, ?, ?, 'requested', ?)`,
        args: [stableId("repair_request", `${adjudicationId}:${candidateResponse.repairRequest}`),
          candidate.candidateVersionId, candidate.passBReviewId, candidateResponse.repairRequest,
          canonicalJson(candidateResponse.preserve as unknown as JsonValue), ts]
      });
    }
    if (candidateResponse.disagreementDescription.trim().length > 0
      || candidateResponse.outcome === "defer_theory_disagreement") {
      statements.push({
        sql: `INSERT INTO disagreement_case
              (id, candidate_version_id, status, description, review_ids_json, created_at)
              VALUES (?, ?, 'contested', ?, ?, ?)`,
        args: [stableId("disagreement", `${adjudicationId}:${candidateResponse.disagreementDescription}`),
          candidate.candidateVersionId,
          candidateResponse.disagreementDescription || "Theory disagreement preserved by Pass D adjudication.",
          canonicalJson([candidate.passAReviewId, candidate.passBReviewId] as unknown as JsonValue), ts]
      });
    }
  }

  for (const family of evidence.families) {
    statements.push({
      sql: `INSERT INTO campaign_closeout_basis
            (id, campaign_closeout_id, basis_kind, basis_id, created_at)
            VALUES (?, ?, 'family_synthesis', ?, ?)`,
      args: [stableId("closeout_basis", `${closeoutId}:family_synthesis:${family.familySynthesisId}`),
        closeoutId, family.familySynthesisId, ts]
    });
  }
  for (const candidate of evidence.candidates) {
    if (candidate.structuralDispositionId === null) continue;
    statements.push({
      sql: `INSERT INTO campaign_closeout_basis
            (id, campaign_closeout_id, basis_kind, basis_id, created_at)
            VALUES (?, ?, 'structural_disposition', ?, ?)`,
      args: [stableId("closeout_basis", `${closeoutId}:structural_disposition:${candidate.structuralDispositionId}`),
        closeoutId, candidate.structuralDispositionId, ts]
    });
  }
  for (const repeat of evidence.repeats) {
    statements.push({
      sql: `INSERT INTO campaign_closeout_basis
            (id, campaign_closeout_id, basis_kind, basis_id, created_at)
            VALUES (?, ?, 'repeat_response', ?, ?)`,
      args: [stableId("closeout_basis", `${closeoutId}:repeat_response:${repeat.repeatResponseId}`),
        closeoutId, repeat.repeatResponseId, ts]
    });
  }
  statements.push({
    sql: `INSERT INTO campaign_closeout_basis
          (id, campaign_closeout_id, basis_kind, basis_id, created_at)
          VALUES (?, ?, 'analysis_run', ?, ?)`,
    args: [stableId("closeout_basis", `${closeoutId}:analysis_run:${evidence.analysis.analysisRunId}`),
      closeoutId, evidence.analysis.analysisRunId, ts]
  });

  for (const state of packet.response.recommendedStates) {
    statements.push({
      sql: `INSERT INTO campaign_closeout_state
            (id, campaign_closeout_id, state, rationale, created_at) VALUES (?, ?, ?, ?, ?)`,
      args: [stableId("closeout_state", `${closeoutId}:${state.state}`), closeoutId,
        state.state!, state.rationale, ts]
    });
  }
  for (const cluster of packet.response.failureClusters) {
    const clusterId = stableId("failure_cluster", `${closeoutId}:${cluster.clusterKey}`);
    statements.push({
      sql: `INSERT INTO campaign_failure_cluster
            (id, campaign_closeout_id, cluster_key, label, locus, severity, proposed_repair,
             new_calls_needed, rationale, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      args: [clusterId, closeoutId, cluster.clusterKey, cluster.label, cluster.locus!, cluster.severity!,
        cluster.proposedRepair, cluster.newCallsNeeded!, cluster.rationale, ts]
    });
    for (const member of cluster.members) {
      statements.push({
        sql: `INSERT INTO campaign_failure_cluster_member
              (id, failure_cluster_id, member_kind, member_id, created_at) VALUES (?, ?, ?, ?, ?)`,
        args: [stableId("failure_cluster_member", `${clusterId}:${member.memberKind}:${member.memberId}`),
          clusterId, member.memberKind, member.memberId, ts]
      });
    }
  }
  for (const assessment of packet.response.distributionAssessments) {
    statements.push({
      sql: `INSERT INTO campaign_distribution_assessment
            (id, campaign_closeout_id, dimension, assessment, evidence_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)`,
      args: [stableId("distribution_assessment", `${closeoutId}:${assessment.dimension}`),
        closeoutId, assessment.dimension, assessment.assessment,
        canonicalJson(assessment.evidenceIds as unknown as JsonValue), ts]
    });
  }
  statements.push({
    sql: `UPDATE campaign_closeout_assignment SET status = 'completed', updated_at = ?
          WHERE id = ? AND status = 'assigned'`,
    args: [ts, assignmentId]
  });
  const eventId = stableId("event", `campaign-closeout-submitted:${closeoutId}`);
  statements.push({
    sql: `INSERT INTO event(id, event_type, object_kind, object_id, payload_json, created_at)
          VALUES (?, 'campaign_closeout_submitted', 'campaign_closeout', ?, ?, ?)`,
    args: [eventId, closeoutId, canonicalJson({
      campaignId: evidence.campaignId,
      sessionId: packet.sessionId,
      packetEnvelopeSha256,
      submissionSha256,
      executionAuthorized: false
    } as JsonValue), ts]
  });
  await ledger.client.batch(statements, "write");
  return {
    sessionId: packet.sessionId,
    candidateAdjudications: packet.response.candidateDispositions.length,
    failureClusters: packet.response.failureClusters.length,
    recommendedStates: packet.response.recommendedStates.length,
    packetEnvelopeSha256,
    submissionSha256,
    executionAuthorized: false
  };
}

export async function campaignCloseoutStatus(
  ledger: Ledger,
  campaignSlug: string
): Promise<CampaignCloseoutStatus> {
  const id = await campaignId(ledger, campaignSlug);
  const assignments = await ledger.client.execute({
    sql: `SELECT status AS key, COUNT(*) AS count FROM campaign_closeout_assignment
          WHERE campaign_id = ? GROUP BY status`,
    args: [id]
  });
  const closeouts = await ledger.client.execute({
    sql: "SELECT COUNT(*) AS count FROM campaign_closeout WHERE campaign_id = ?",
    args: [id]
  });
  const adjudications = await ledger.client.execute({
    sql: `SELECT COUNT(*) AS count FROM adjudication a
          JOIN candidate_version cv ON cv.id = a.candidate_version_id
          JOIN candidate c ON c.id = cv.candidate_id
          WHERE c.campaign_id = ? AND json_valid(a.rationale)
            AND json_extract(a.rationale, '$.campaignCloseoutId') IS NOT NULL`,
    args: [id]
  });
  const clusters = await ledger.client.execute({
    sql: `SELECT COUNT(*) AS count FROM campaign_failure_cluster cfc
          JOIN campaign_closeout cc ON cc.id = cfc.campaign_closeout_id WHERE cc.campaign_id = ?`,
    args: [id]
  });
  const states = await ledger.client.execute({
    sql: `SELECT ccs.state AS key, COUNT(*) AS count FROM campaign_closeout_state ccs
          JOIN campaign_closeout cc ON cc.id = ccs.campaign_closeout_id
          WHERE cc.campaign_id = ? GROUP BY ccs.state`,
    args: [id]
  });
  const releaseMembers = await ledger.client.execute({
    sql: `SELECT COUNT(*) AS count FROM release_member rm
          JOIN candidate_version cv ON cv.id = rm.candidate_version_id
          JOIN candidate c ON c.id = cv.candidate_id WHERE c.campaign_id = ?`,
    args: [id]
  });
  const trainingExposures = await ledger.client.execute({
    sql: `SELECT COUNT(*) AS count FROM training_exposure te
          JOIN rendered_unit ru ON ru.id = te.rendered_unit_id
          JOIN candidate_version cv ON cv.id = ru.candidate_version_id
          JOIN candidate c ON c.id = cv.candidate_id WHERE c.campaign_id = ?`,
    args: [id]
  });
  const authorizations = await ledger.client.execute({
    sql: "SELECT COUNT(*) AS count FROM campaign_closeout WHERE campaign_id = ? AND execution_authorized <> 0",
    args: [id]
  });
  const grouped = (rows: Array<Record<string, unknown>>): Record<string, number> => Object.fromEntries(
    rows.map((row) => [String(row["key"]), Number(row["count"])])
  );
  return {
    campaignSlug,
    assignments: grouped(assignments.rows as Array<Record<string, unknown>>),
    campaignCloseouts: Number(closeouts.rows[0]!["count"]),
    candidateAdjudications: Number(adjudications.rows[0]!["count"]),
    failureClusters: Number(clusters.rows[0]!["count"]),
    recommendedStates: grouped(states.rows as Array<Record<string, unknown>>),
    releaseMembers: Number(releaseMembers.rows[0]!["count"]),
    trainingExposures: Number(trainingExposures.rows[0]!["count"]),
    executionAuthorizations: Number(authorizations.rows[0]!["count"])
  };
}
