import type { Ledger } from "./db.js";
import { canonicalJson, sha256Bytes } from "./hash.js";
import type { JsonValue, NaturalMessage } from "./types.js";

export interface CalibrationAnalysis {
  candidates: number;
  structurallyValid: number;
  structurallyRejected: number;
  structuralYield: number;
  assistantMessages: number;
  multiTurnCandidates: number;
  assistantWords: { mean: number; median: number; p90: number; maximum: number };
  assistantQuestionEndRate: number;
  exactDuplicateAssistantMessages: number;
  mostCommonOpeningPrefixes: Array<{ prefix: string; count: number }>;
  nearDuplicatePairsAbove070: number;
  maximumPairwiseShingleSimilarity: number;
  kinds: Record<string, number>;
  difficulty: Record<string, number>;
  responsePolicies: Record<string, number>;
}

export interface CampaignAnalysisSample {
  campaignId: string;
  candidateId: string;
  candidateVersionId: string;
  candidateVersion: number;
  candidateContentSha256: string;
  familyId: string;
  familySlug: string;
  status: string;
  kind: string;
  difficulty: string;
  responsePolicy: string;
  messages: NaturalMessage[];
  assistantTexts: string[];
}

export interface SurfaceAnalysisMetric {
  scopeKind: "campaign" | "family";
  scopeId: string;
  metric: string;
  value: number | string;
  unit: string;
  denominator: number | null;
  detail: string;
}

export interface SurfaceSimilarityEdge {
  leftCandidateVersionId: string;
  rightCandidateVersionId: string;
  method: "assistant_word_3gram_jaccard" | "assistant_character_5gram_jaccard";
  score: number;
  reviewThreshold: number;
  classification: "surface_review_candidate" | "not_flagged";
}

export interface SurfaceTemplateSignature {
  scopeKind: "campaign" | "family";
  scopeId: string;
  signatureKind: string;
  signature: string;
  candidateCount: number;
  denominator: number;
  rate: number;
}

export interface SurfaceAnalysisData {
  campaignId: string;
  campaignSlug: string;
  inputSnapshot: JsonValue;
  inputSnapshotSha256: string;
  summary: CalibrationAnalysis;
  metrics: SurfaceAnalysisMetric[];
  similarityEdges: SurfaceSimilarityEdge[];
  templateSignatures: SurfaceTemplateSignature[];
}

export const SURFACE_ANALYSIS_METHOD = {
  slug: "deterministic-surface-distribution-profile",
  version: 1,
  definition: "Deterministic candidate-level surface and distribution measurements over current candidate versions. These measurements nominate items for review; they do not establish semantic duplication, conceptual validity, conversational quality, or training eligibility.",
  config: {
    inputVersionPolicy: "latest_candidate_version_only",
    similarityMethods: [
      { name: "assistant_word_3gram_jaccard", reviewThreshold: 0.7 },
      { name: "assistant_character_5gram_jaccard", reviewThreshold: 0.7 }
    ],
    templateSignatures: {
      source: "normalized_assistant_word_ngrams",
      minimumN: 2,
      maximumN: 6,
      minimumDistinctCandidates: 2,
      campaignLimit: 250,
      familyLimit: 50
    }
  }
} as const;

function normalizedWords(text: string): string[] {
  return text.toLowerCase().replace(/[^\p{L}\p{N}'-]+/gu, " ").trim().split(/\s+/).filter(Boolean);
}

function normalizedCharacters(text: string): string {
  return normalizedWords(text).join(" ");
}

function quantile(values: number[], fraction: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * fraction))]!;
}

function wordShingles(text: string, width = 3): Set<string> {
  const words = normalizedWords(text);
  if (words.length < width) return new Set([words.join(" ")]);
  return new Set(words.slice(0, words.length - width + 1).map((_, index) => words.slice(index, index + width).join(" ")));
}

function characterShingles(text: string, width = 5): Set<string> {
  const characters = normalizedCharacters(text);
  if (characters.length < width) return new Set([characters]);
  return new Set([...Array(characters.length - width + 1).keys()].map((index) => characters.slice(index, index + width)));
}

function jaccard(left: Set<string>, right: Set<string>): number {
  let intersection = 0;
  for (const value of left) if (right.has(value)) intersection++;
  const union = left.size + right.size - intersection;
  return union === 0 ? 1 : intersection / union;
}

function increment(record: Record<string, number>, key: string): void {
  record[key] = (record[key] ?? 0) + 1;
}

function combinedAssistantText(sample: CampaignAnalysisSample): string {
  return sample.assistantTexts.join("\n");
}

export async function loadCampaignAnalysisSamples(
  ledger: Ledger,
  campaignSlug: string
): Promise<CampaignAnalysisSample[]> {
  const result = await ledger.client.execute({
    sql: `SELECT cc.campaign_id, cc.candidate_id, cc.candidate_version_id, cc.version,
                 cc.content_sha256, cc.family_id, cc.family_slug, cc.status, cc.kind, cc.content_json
          FROM corpus_candidate_current cc
          JOIN generation_campaign gc ON gc.id = cc.campaign_id
          WHERE gc.slug = ?
          ORDER BY cc.candidate_id`,
    args: [campaignSlug]
  });
  return result.rows.map((row) => {
    const content = JSON.parse(String(row["content_json"])) as {
      kind: string;
      difficulty: string;
      intendedResponsePolicy: string;
      messages: NaturalMessage[];
    };
    return {
      campaignId: String(row["campaign_id"]),
      candidateId: String(row["candidate_id"]),
      candidateVersionId: String(row["candidate_version_id"]),
      candidateVersion: Number(row["version"]),
      candidateContentSha256: String(row["content_sha256"]),
      familyId: String(row["family_id"]),
      familySlug: String(row["family_slug"]),
      status: String(row["status"]),
      kind: content.kind,
      difficulty: content.difficulty,
      responsePolicy: content.intendedResponsePolicy,
      messages: content.messages,
      assistantTexts: content.messages
        .filter((message) => message.role === "assistant")
        .map((message) => message.content.trim())
    };
  });
}

export function summarizeCampaign(samples: CampaignAnalysisSample[]): CalibrationAnalysis {
  const assistantTexts = samples.flatMap((sample) => sample.assistantTexts);
  const wordCounts = assistantTexts.map((text) => normalizedWords(text).length);
  const kinds: Record<string, number> = {};
  const difficulty: Record<string, number> = {};
  const responsePolicies: Record<string, number> = {};
  for (const sample of samples) {
    increment(kinds, sample.kind);
    increment(difficulty, sample.difficulty);
    increment(responsePolicies, sample.responsePolicy);
  }
  const normalizedCounts = new Map<string, number>();
  const prefixCounts = new Map<string, number>();
  for (const text of assistantTexts) {
    const words = normalizedWords(text);
    const normalized = words.join(" ");
    normalizedCounts.set(normalized, (normalizedCounts.get(normalized) ?? 0) + 1);
    const prefix = words.slice(0, 4).join(" ");
    prefixCounts.set(prefix, (prefixCounts.get(prefix) ?? 0) + 1);
  }
  const sampleShingles = samples.map((sample) => wordShingles(combinedAssistantText(sample)));
  let nearDuplicatePairsAbove070 = 0;
  let maximumPairwiseShingleSimilarity = 0;
  for (let left = 0; left < sampleShingles.length; left++) {
    for (let right = left + 1; right < sampleShingles.length; right++) {
      const similarity = jaccard(sampleShingles[left]!, sampleShingles[right]!);
      maximumPairwiseShingleSimilarity = Math.max(maximumPairwiseShingleSimilarity, similarity);
      if (similarity >= 0.7) nearDuplicatePairsAbove070++;
    }
  }
  const structurallyValid = samples.filter((sample) => sample.status === "structurally_valid").length;
  const structurallyRejected = samples.filter((sample) => sample.status === "structurally_rejected").length;
  return {
    candidates: samples.length,
    structurallyValid,
    structurallyRejected,
    structuralYield: samples.length === 0 ? 0 : structurallyValid / samples.length,
    assistantMessages: assistantTexts.length,
    multiTurnCandidates: samples.filter((sample) => sample.messages.length > 2).length,
    assistantWords: {
      mean: wordCounts.length === 0 ? 0 : wordCounts.reduce((sum, count) => sum + count, 0) / wordCounts.length,
      median: quantile(wordCounts, 0.5),
      p90: quantile(wordCounts, 0.9),
      maximum: Math.max(0, ...wordCounts)
    },
    assistantQuestionEndRate: assistantTexts.length === 0
      ? 0
      : assistantTexts.filter((text) => text.endsWith("?")).length / assistantTexts.length,
    exactDuplicateAssistantMessages: [...normalizedCounts.values()]
      .reduce((sum, count) => sum + Math.max(0, count - 1), 0),
    mostCommonOpeningPrefixes: [...prefixCounts.entries()]
      .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
      .slice(0, 10)
      .map(([prefix, count]) => ({ prefix, count })),
    nearDuplicatePairsAbove070,
    maximumPairwiseShingleSimilarity,
    kinds,
    difficulty,
    responsePolicies
  };
}

function scalarMetrics(
  samples: CampaignAnalysisSample[],
  scopeKind: "campaign" | "family",
  scopeId: string
): SurfaceAnalysisMetric[] {
  const summary = summarizeCampaign(samples);
  const metrics: SurfaceAnalysisMetric[] = [];
  const add = (metric: string, value: number, unit: string, denominator: number | null, detail: string): void => {
    metrics.push({ scopeKind, scopeId, metric, value, unit, denominator, detail });
  };
  add("candidate_count", summary.candidates, "candidates", null, "Current candidate versions in scope.");
  add("structurally_valid_count", summary.structurallyValid, "candidates", summary.candidates, "Structural validation state only; not human acceptance.");
  add("structurally_rejected_count", summary.structurallyRejected, "candidates", summary.candidates, "Structural validation state only; not human rejection.");
  add("structural_yield", summary.structuralYield, "proportion", summary.candidates, "Structurally valid candidates divided by current candidates.");
  add("assistant_message_count", summary.assistantMessages, "messages", null, "Assistant messages in current candidate versions.");
  add("multi_turn_candidate_count", summary.multiTurnCandidates, "candidates", summary.candidates, "Candidates containing more than two messages.");
  add("assistant_words_mean", summary.assistantWords.mean, "words_per_message", summary.assistantMessages, "Mean normalized word count per assistant message.");
  add("assistant_words_median", summary.assistantWords.median, "words_per_message", summary.assistantMessages, "Median normalized word count per assistant message.");
  add("assistant_words_p90", summary.assistantWords.p90, "words_per_message", summary.assistantMessages, "Ninetieth-percentile normalized word count per assistant message.");
  add("assistant_words_maximum", summary.assistantWords.maximum, "words_per_message", summary.assistantMessages, "Maximum normalized word count of an assistant message.");
  add("assistant_question_end_rate", summary.assistantQuestionEndRate, "proportion", summary.assistantMessages, "Assistant messages ending in a question mark.");
  add("exact_duplicate_assistant_message_excess", summary.exactDuplicateAssistantMessages, "messages", summary.assistantMessages, "Normalized assistant messages beyond the first exact occurrence.");
  add("word_3gram_pair_count_at_or_above_0_70", summary.nearDuplicatePairsAbove070, "candidate_pairs", null, "Surface-review candidates only; not a semantic duplicate judgment.");
  add("maximum_word_3gram_jaccard", summary.maximumPairwiseShingleSimilarity, "similarity", null, "Maximum candidate-level normalized assistant word 3-gram Jaccard score.");
  for (const [kind, count] of Object.entries(summary.kinds).sort()) {
    add(`kind.${kind}.count`, count, "candidates", summary.candidates, "Candidate kind distribution.");
  }
  for (const [difficulty, count] of Object.entries(summary.difficulty).sort()) {
    add(`difficulty.${difficulty}.count`, count, "candidates", summary.candidates, "Declared generation difficulty distribution.");
  }
  for (const [policy, count] of Object.entries(summary.responsePolicies).sort()) {
    add(`response_policy.${policy}.count`, count, "candidates", summary.candidates, "Declared intended response-policy distribution.");
  }
  return metrics;
}

function similarityEdges(samples: CampaignAnalysisSample[]): SurfaceSimilarityEdge[] {
  const threshold = 0.7;
  const edges: SurfaceSimilarityEdge[] = [];
  for (let leftIndex = 0; leftIndex < samples.length; leftIndex++) {
    for (let rightIndex = leftIndex + 1; rightIndex < samples.length; rightIndex++) {
      const first = samples[leftIndex]!;
      const second = samples[rightIndex]!;
      const [left, right] = first.candidateVersionId < second.candidateVersionId
        ? [first, second]
        : [second, first];
      const leftText = combinedAssistantText(left);
      const rightText = combinedAssistantText(right);
      const methods = [
        ["assistant_word_3gram_jaccard", jaccard(wordShingles(leftText), wordShingles(rightText))],
        ["assistant_character_5gram_jaccard", jaccard(characterShingles(leftText), characterShingles(rightText))]
      ] as const;
      for (const [method, score] of methods) {
        edges.push({
          leftCandidateVersionId: left.candidateVersionId,
          rightCandidateVersionId: right.candidateVersionId,
          method,
          score,
          reviewThreshold: threshold,
          classification: score >= threshold ? "surface_review_candidate" : "not_flagged"
        });
      }
    }
  }
  return edges.sort((left, right) => left.leftCandidateVersionId.localeCompare(right.leftCandidateVersionId)
    || left.rightCandidateVersionId.localeCompare(right.rightCandidateVersionId)
    || left.method.localeCompare(right.method));
}

function templateSignaturesForScope(
  samples: CampaignAnalysisSample[],
  scopeKind: "campaign" | "family",
  scopeId: string,
  limit: number
): SurfaceTemplateSignature[] {
  if (samples.length === 0) return [];
  const counts = new Map<string, { kind: string; signature: string; count: number }>();
  for (const sample of samples) {
    const words = normalizedWords(combinedAssistantText(sample));
    const seen = new Set<string>();
    for (let width = 2; width <= 6; width++) {
      for (let index = 0; index <= words.length - width; index++) {
        const signature = words.slice(index, index + width).join(" ");
        const key = `${width}\0${signature}`;
        if (seen.has(key)) continue;
        seen.add(key);
        const current = counts.get(key);
        counts.set(key, {
          kind: `assistant_word_${width}gram`,
          signature,
          count: (current?.count ?? 0) + 1
        });
      }
    }
  }
  return [...counts.values()]
    .filter((entry) => entry.count >= 2)
    .sort((left, right) => right.count - left.count
      || Number(right.kind.match(/\d+/)?.[0] ?? 0) - Number(left.kind.match(/\d+/)?.[0] ?? 0)
      || left.signature.localeCompare(right.signature))
    .slice(0, limit)
    .map((entry) => ({
      scopeKind,
      scopeId,
      signatureKind: entry.kind,
      signature: entry.signature,
      candidateCount: entry.count,
      denominator: samples.length,
      rate: entry.count / samples.length
    }));
}

export async function buildSurfaceAnalysisData(
  ledger: Ledger,
  campaignSlug: string
): Promise<SurfaceAnalysisData> {
  const samples = await loadCampaignAnalysisSamples(ledger, campaignSlug);
  if (samples.length === 0) throw new Error(`Campaign ${campaignSlug} has no current candidates`);
  const campaignIds = new Set(samples.map((sample) => sample.campaignId));
  if (campaignIds.size !== 1) throw new Error(`Campaign ${campaignSlug} resolved to multiple campaign IDs`);
  const campaignId = samples[0]!.campaignId;
  const inputSnapshot = {
    schemaVersion: 1,
    campaignId,
    campaignSlug,
    versionPolicy: "latest_candidate_version_only",
    candidates: samples.map((sample) => ({
      candidateId: sample.candidateId,
      candidateVersionId: sample.candidateVersionId,
      candidateVersion: sample.candidateVersion,
      contentSha256: sample.candidateContentSha256,
      familyId: sample.familyId,
      familySlug: sample.familySlug,
      status: sample.status
    }))
  } satisfies JsonValue;
  const families = new Map<string, CampaignAnalysisSample[]>();
  for (const sample of samples) {
    const group = families.get(sample.familyId) ?? [];
    group.push(sample);
    families.set(sample.familyId, group);
  }
  const metrics = scalarMetrics(samples, "campaign", campaignId);
  const signatures = templateSignaturesForScope(samples, "campaign", campaignId, 250);
  for (const [familyId, familySamples] of [...families.entries()].sort(([left], [right]) => left.localeCompare(right))) {
    metrics.push(...scalarMetrics(familySamples, "family", familyId));
    signatures.push(...templateSignaturesForScope(familySamples, "family", familyId, 50));
  }
  return {
    campaignId,
    campaignSlug,
    inputSnapshot,
    inputSnapshotSha256: sha256Bytes(canonicalJson(inputSnapshot)),
    summary: summarizeCampaign(samples),
    metrics,
    similarityEdges: similarityEdges(samples),
    templateSignatures: signatures
  };
}

export async function analyzeCampaign(ledger: Ledger, campaignSlug: string): Promise<CalibrationAnalysis> {
  return summarizeCampaign(await loadCampaignAnalysisSamples(ledger, campaignSlug));
}
