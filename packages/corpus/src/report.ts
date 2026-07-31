import { randomUUID } from "node:crypto";
import { join } from "node:path";
import { canonicalJson } from "./hash.js";
import { campaignStats, putBlob, type Ledger } from "./db.js";
import { writeAtomic } from "./storage.js";
import { analyzeCampaign } from "./analysis.js";
import type { GeneratedItem, JsonValue } from "./types.js";

interface AuditCandidate {
  candidateId: string;
  familySlug: string;
  status: string;
  item: GeneratedItem;
  findings: Array<{ code: string; detail: string }>;
}

export interface AuditPacketResult {
  directory: string;
  jsonPath: string;
  markdownPath: string;
  candidateCount: number;
}

export async function writeAuditPacket(
  ledger: Ledger,
  campaignSlug: string
): Promise<AuditPacketResult> {
  const stats = await campaignStats(ledger, campaignSlug);
  const analysis = await analyzeCampaign(ledger, campaignSlug);
  const rows = await ledger.client.execute({
    sql: `SELECT cc.candidate_id, cc.family_slug, cc.status, cc.content_json,
                 cc.hidden_contract_json
          FROM corpus_candidate_current cc
          WHERE cc.campaign_id = ?
          ORDER BY cc.family_slug, cc.item_key`,
    args: [stats.campaignId]
  });
  const candidates: AuditCandidate[] = [];
  for (const row of rows.rows) {
    const candidateId = String(row["candidate_id"]);
    const content = JSON.parse(String(row["content_json"])) as Omit<GeneratedItem, "hiddenContract">;
    const hiddenContract = JSON.parse(String(row["hidden_contract_json"])) as GeneratedItem["hiddenContract"];
    const failures = await ledger.client.execute({
      sql: "SELECT code, detail FROM candidate_failure WHERE candidate_id = ? ORDER BY created_at",
      args: [candidateId]
    });
    candidates.push({
      candidateId,
      familySlug: String(row["family_slug"]),
      status: String(row["status"]),
      item: { ...content, hiddenContract },
      findings: failures.rows.map((finding) => ({
        code: String(finding["code"]),
        detail: String(finding["detail"])
      }))
    });
  }

  const createdAt = new Date().toISOString();
  const directory = join(ledger.paths.releases, "audit", `${campaignSlug}-${createdAt.replace(/[:.]/g, "-")}`);
  const packet = {
    kind: "human_audit_packet",
    campaignSlug,
    createdAt,
    warning: "Calibration material only. No candidate is approved for training or evaluation without human adjudication.",
    stats,
    analysis,
    candidates
  };
  const json = canonicalJson(packet as unknown as JsonValue);
  const jsonPath = join(directory, "audit-packet.json");
  const markdownPath = join(directory, "README.md");
  writeAtomic(jsonPath, `${json}\n`);

  const byFamily = new Map<string, AuditCandidate[]>();
  for (const candidate of candidates) {
    const group = byFamily.get(candidate.familySlug) ?? [];
    group.push(candidate);
    byFamily.set(candidate.familySlug, group);
  }
  const lines = [
    `# Alpha calibration audit: ${campaignSlug}`,
    "",
    `Created: ${createdAt}`,
    "",
    "> Calibration material only. Nothing in this packet is approved for training or evaluation until human adjudication.",
    "",
    `Candidates: ${candidates.length}; calls: ${stats.modelCalls}; structurally valid: ${stats.candidates["structurally_valid"] ?? 0}; structurally rejected: ${stats.candidates["structurally_rejected"] ?? 0}.`,
    "",
    `Structural yield: ${(analysis.structuralYield * 100).toFixed(1)}%; assistant question-ending rate: ${(analysis.assistantQuestionEndRate * 100).toFixed(1)}%; exact duplicate assistant messages: ${analysis.exactDuplicateAssistantMessages}; near-duplicate pairs (3-word Jaccard >= 0.70): ${analysis.nearDuplicatePairsAbove070}.`,
    "",
    `Assistant response words: mean ${analysis.assistantWords.mean.toFixed(1)}, median ${analysis.assistantWords.median}, p90 ${analysis.assistantWords.p90}, maximum ${analysis.assistantWords.maximum}.`,
    ""
  ];
  for (const [family, group] of byFamily) {
    lines.push(`## ${family}`, "");
    for (const candidate of group) {
      lines.push(`### ${candidate.item.title}`, "", `Status: ${candidate.status}`, "");
      for (const message of candidate.item.messages) {
        lines.push(`**${message.role}:** ${message.content}`, "");
      }
      lines.push(
        `Required commitments: ${candidate.item.hiddenContract.requiredCommitments.join("; ") || "none"}`,
        "",
        `Prohibited commitments: ${candidate.item.hiddenContract.prohibitedCommitments.join("; ") || "none"}`,
        ""
      );
      if (candidate.findings.length > 0) {
        lines.push(`Findings: ${candidate.findings.map((finding) => `${finding.code}: ${finding.detail}`).join("; ")}`, "");
      }
    }
  }
  writeAtomic(markdownPath, `${lines.join("\n")}\n`);
  const jsonSha = await putBlob(ledger, `${json}\n`, "application/json");
  const markdownSha = await putBlob(ledger, `${lines.join("\n")}\n`, "text/markdown; charset=utf-8");
  await ledger.client.batch(
    [
      {
        sql: `INSERT INTO export_artifact
              (id, release_id, cohort_snapshot_id, format, blob_sha256, manifest_json, created_at)
              VALUES (?, NULL, NULL, 'human_audit_json', ?, ?, ?)`,
        args: [`export_${randomUUID()}`, jsonSha, canonicalJson({ campaignSlug, path: jsonPath } as JsonValue), createdAt]
      },
      {
        sql: `INSERT INTO export_artifact
              (id, release_id, cohort_snapshot_id, format, blob_sha256, manifest_json, created_at)
              VALUES (?, NULL, NULL, 'human_audit_markdown', ?, ?, ?)`,
        args: [`export_${randomUUID()}`, markdownSha, canonicalJson({ campaignSlug, path: markdownPath } as JsonValue), createdAt]
      }
    ],
    "write"
  );
  return { directory, jsonPath, markdownPath, candidateCount: candidates.length };
}
