import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import {
  buildSurfaceAnalysisData,
  SURFACE_ANALYSIS_METHOD,
  type SurfaceAnalysisData
} from "./analysis.js";
import { putBlob, type Ledger } from "./db.js";
import { canonicalJson, sha256Bytes, stableId } from "./hash.js";
import { writeAtomic } from "./storage.js";
import type { JsonValue } from "./types.js";

const EVIDENCE_SCOPE = "surface_distribution_only";
const DISCLAIMER = "Deterministic surface and distribution evidence only. A flagged pair is not a semantic duplicate judgment; structural validity is not human acceptance; this run cannot promote candidates, create release membership, or authorize training exposure.";

export interface CalibrationProfileResult {
  analysisRunId: string;
  campaignSlug: string;
  inputSnapshotSha256: string;
  outputBlobSha256: string;
  profilePath: string;
  metricCount: number;
  similarityEdgeCount: number;
  templateSignatureCount: number;
  resumed: boolean;
  warning: string;
}

function exactWrite(path: string, bytes: string): void {
  if (existsSync(path)) {
    const existing = readFileSync(path);
    if (sha256Bytes(existing) !== sha256Bytes(bytes)) {
      throw new Error(`Refusing to replace non-identical analysis artifact at ${path}`);
    }
    return;
  }
  writeAtomic(path, bytes);
}

function methodContract(): JsonValue {
  return {
    slug: SURFACE_ANALYSIS_METHOD.slug,
    version: SURFACE_ANALYSIS_METHOD.version,
    definition: SURFACE_ANALYSIS_METHOD.definition,
    config: SURFACE_ANALYSIS_METHOD.config
  } as unknown as JsonValue;
}

function profileDocument(
  data: SurfaceAnalysisData,
  softwareRevision: string,
  environment: JsonValue
): JsonValue {
  return {
    schemaVersion: 1,
    kind: "deterministic_surface_distribution_profile",
    evidenceScope: EVIDENCE_SCOPE,
    warning: DISCLAIMER,
    method: methodContract(),
    software: { component: "alpha-corpus", revision: softwareRevision, environment },
    campaign: { id: data.campaignId, slug: data.campaignSlug },
    inputSnapshotSha256: data.inputSnapshotSha256,
    inputSnapshot: data.inputSnapshot,
    summary: data.summary,
    metrics: data.metrics,
    similarityEdges: data.similarityEdges,
    templateSignatures: data.templateSignatures
  } as unknown as JsonValue;
}

export async function writeCalibrationProfile(
  ledger: Ledger,
  options: { campaignSlug: string; softwareRevision: string }
): Promise<CalibrationProfileResult> {
  const softwareRevision = options.softwareRevision.trim();
  if (!softwareRevision) throw new Error("softwareRevision must be non-empty");
  const data = await buildSurfaceAnalysisData(ledger, options.campaignSlug);
  const methodJson = canonicalJson(methodContract());
  const methodSha256 = sha256Bytes(methodJson);
  const methodId = stableId(
    "analysis_method",
    `${SURFACE_ANALYSIS_METHOD.slug}:${SURFACE_ANALYSIS_METHOD.version}`
  );
  const environment = {
    node: process.version,
    platform: process.platform,
    architecture: process.arch
  } satisfies JsonValue;
  const environmentJson = canonicalJson(environment);
  const softwareRevisionId = stableId(
    "software_revision",
    `alpha-corpus:${softwareRevision}:${sha256Bytes(environmentJson)}`
  );
  const analysisRunId = stableId(
    "analysis_run",
    `${data.campaignId}:${methodId}:${softwareRevisionId}:${data.inputSnapshotSha256}`
  );
  const document = `${canonicalJson(profileDocument(data, softwareRevision, environment))}\n`;
  const outputBlobSha256 = await putBlob(ledger, document, "application/json");
  const directory = join(
    ledger.paths.releases,
    "analysis",
    `${data.campaignSlug}-${data.inputSnapshotSha256.slice(0, 16)}-${outputBlobSha256.slice(0, 12)}`
  );
  const profilePath = join(directory, "surface-profile.json");
  exactWrite(profilePath, document);

  const existingMethod = await ledger.client.execute({
    sql: "SELECT content_sha256 FROM analysis_method WHERE id = ?",
    args: [methodId]
  });
  if (existingMethod.rows.length > 0
    && String(existingMethod.rows[0]!["content_sha256"]) !== methodSha256) {
    throw new Error(`Analysis method ${methodId} changed without a new version`);
  }
  const existingSoftware = await ledger.client.execute({
    sql: "SELECT revision, environment_json FROM software_revision WHERE id = ?",
    args: [softwareRevisionId]
  });
  if (existingSoftware.rows.length > 0
    && (String(existingSoftware.rows[0]!["revision"]) !== softwareRevision
      || String(existingSoftware.rows[0]!["environment_json"]) !== environmentJson)) {
    throw new Error(`Software revision ${softwareRevisionId} changed under the same identity`);
  }

  const existingRun = await ledger.client.execute({
    sql: `SELECT output_blob_sha256 FROM analysis_run WHERE id = ?`,
    args: [analysisRunId]
  });
  if (existingRun.rows.length > 0) {
    if (String(existingRun.rows[0]!["output_blob_sha256"]) !== outputBlobSha256) {
      throw new Error(`Analysis run ${analysisRunId} changed output without a new identity`);
    }
    return {
      analysisRunId,
      campaignSlug: data.campaignSlug,
      inputSnapshotSha256: data.inputSnapshotSha256,
      outputBlobSha256,
      profilePath,
      metricCount: data.metrics.length,
      similarityEdgeCount: data.similarityEdges.length,
      templateSignatureCount: data.templateSignatures.length,
      resumed: true,
      warning: DISCLAIMER
    };
  }

  const ts = new Date().toISOString();
  const statements: Array<{ sql: string; args: Array<string | number | null> }> = [
    {
      sql: `INSERT OR IGNORE INTO analysis_method
            (id, slug, version, definition, config_json, content_sha256, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)`,
      args: [methodId, SURFACE_ANALYSIS_METHOD.slug, SURFACE_ANALYSIS_METHOD.version,
        SURFACE_ANALYSIS_METHOD.definition, canonicalJson(SURFACE_ANALYSIS_METHOD.config as unknown as JsonValue),
        methodSha256, ts]
    },
    {
      sql: `INSERT OR IGNORE INTO software_revision
            (id, component, revision, build_digest, environment_json, created_at)
            VALUES (?, 'alpha-corpus', ?, NULL, ?, ?)`,
      args: [softwareRevisionId, softwareRevision, environmentJson, ts]
    },
    {
      sql: `INSERT INTO analysis_run
            (id, campaign_id, analysis_method_id, software_revision_id, input_snapshot_sha256,
             output_blob_sha256, status, evidence_scope, disclaimer, started_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, 'completed', ?, ?, ?, ?)`,
      args: [analysisRunId, data.campaignId, methodId, softwareRevisionId, data.inputSnapshotSha256,
        outputBlobSha256, EVIDENCE_SCOPE, DISCLAIMER, ts, ts]
    }
  ];
  for (const metric of data.metrics) {
    statements.push({
      sql: `INSERT INTO analysis_metric
            (id, analysis_run_id, scope_kind, scope_id, metric, value_real, value_text,
             unit, denominator, detail, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      args: [
        stableId("analysis_metric", `${analysisRunId}:${metric.scopeKind}:${metric.scopeId}:${metric.metric}`),
        analysisRunId,
        metric.scopeKind,
        metric.scopeId,
        metric.metric,
        typeof metric.value === "number" ? metric.value : null,
        typeof metric.value === "string" ? metric.value : null,
        metric.unit,
        metric.denominator,
        metric.detail,
        ts
      ]
    });
  }
  for (const edge of data.similarityEdges) {
    statements.push({
      sql: `INSERT INTO similarity_edge
            (id, analysis_run_id, left_candidate_version_id, right_candidate_version_id,
             method, score, review_threshold, classification, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      args: [
        stableId("similarity_edge", `${analysisRunId}:${edge.leftCandidateVersionId}:${edge.rightCandidateVersionId}:${edge.method}`),
        analysisRunId,
        edge.leftCandidateVersionId,
        edge.rightCandidateVersionId,
        edge.method,
        edge.score,
        edge.reviewThreshold,
        edge.classification,
        ts
      ]
    });
  }
  for (const signature of data.templateSignatures) {
    statements.push({
      sql: `INSERT INTO template_signature
            (id, analysis_run_id, scope_kind, scope_id, signature_kind, signature,
             candidate_count, denominator, rate, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      args: [
        stableId("template_signature", `${analysisRunId}:${signature.scopeKind}:${signature.scopeId}:${signature.signatureKind}:${signature.signature}`),
        analysisRunId,
        signature.scopeKind,
        signature.scopeId,
        signature.signatureKind,
        signature.signature,
        signature.candidateCount,
        signature.denominator,
        signature.rate,
        ts
      ]
    });
  }
  statements.push(
    {
      sql: `INSERT INTO export_artifact
            (id, release_id, cohort_snapshot_id, format, blob_sha256, manifest_json, created_at)
            VALUES (?, NULL, NULL, 'surface_distribution_profile_json', ?, ?, ?)`,
      args: [
        stableId("export", `surface-profile:${analysisRunId}:${outputBlobSha256}`),
        outputBlobSha256,
        canonicalJson({
          analysisRunId,
          campaignSlug: data.campaignSlug,
          evidenceScope: EVIDENCE_SCOPE,
          path: profilePath
        }),
        ts
      ]
    },
    {
      sql: `INSERT INTO event(id, event_type, object_kind, object_id, payload_json, created_at)
            VALUES (?, 'surface_analysis_completed', 'analysis_run', ?, ?, ?)`,
      args: [
        stableId("event", `surface-analysis-completed:${analysisRunId}`),
        analysisRunId,
        canonicalJson({
          campaignId: data.campaignId,
          inputSnapshotSha256: data.inputSnapshotSha256,
          outputBlobSha256,
          metricCount: data.metrics.length,
          similarityEdgeCount: data.similarityEdges.length,
          templateSignatureCount: data.templateSignatures.length,
          evidenceScope: EVIDENCE_SCOPE
        }),
        ts
      ]
    }
  );
  await ledger.client.batch(statements, "write");

  const storedMethod = await ledger.client.execute({
    sql: "SELECT content_sha256 FROM analysis_method WHERE id = ?",
    args: [methodId]
  });
  if (String(storedMethod.rows[0]!["content_sha256"]) !== methodSha256) {
    throw new Error(`Analysis method ${methodId} changed without a new version`);
  }
  return {
    analysisRunId,
    campaignSlug: data.campaignSlug,
    inputSnapshotSha256: data.inputSnapshotSha256,
    outputBlobSha256,
    profilePath,
    metricCount: data.metrics.length,
    similarityEdgeCount: data.similarityEdges.length,
    templateSignatureCount: data.templateSignatures.length,
    resumed: false,
    warning: DISCLAIMER
  };
}
