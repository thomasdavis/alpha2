import type { Ledger } from "./db.js";

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

function normalizedWords(text: string): string[] {
  return text.toLowerCase().replace(/[^\p{L}\p{N}'-]+/gu, " ").trim().split(/\s+/).filter(Boolean);
}

function quantile(values: number[], fraction: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * fraction))]!;
}

function shingles(text: string, width = 3): Set<string> {
  const words = normalizedWords(text);
  if (words.length < width) return new Set([words.join(" ")]);
  return new Set(words.slice(0, words.length - width + 1).map((_, index) => words.slice(index, index + width).join(" ")));
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

export async function analyzeCampaign(ledger: Ledger, campaignSlug: string): Promise<CalibrationAnalysis> {
  const result = await ledger.client.execute({
    sql: `SELECT c.status, cv.content_json
          FROM candidate c
          JOIN candidate_version cv ON cv.candidate_id = c.id
          JOIN generation_campaign gc ON gc.id = c.campaign_id
          WHERE gc.slug = ?
          ORDER BY c.id`,
    args: [campaignSlug]
  });
  const assistantTexts: string[] = [];
  const wordCounts: number[] = [];
  const kinds: Record<string, number> = {};
  const difficulty: Record<string, number> = {};
  const responsePolicies: Record<string, number> = {};
  let structurallyValid = 0;
  let structurallyRejected = 0;
  let multiTurnCandidates = 0;
  for (const row of result.rows) {
    const status = String(row["status"]);
    if (status === "structurally_valid") structurallyValid++;
    if (status === "structurally_rejected") structurallyRejected++;
    const content = JSON.parse(String(row["content_json"])) as {
      kind: string;
      difficulty: string;
      intendedResponsePolicy: string;
      messages: Array<{ role: string; content: string }>;
    };
    increment(kinds, content.kind);
    increment(difficulty, content.difficulty);
    increment(responsePolicies, content.intendedResponsePolicy);
    if (content.messages.length > 2) multiTurnCandidates++;
    for (const message of content.messages) {
      if (message.role !== "assistant") continue;
      assistantTexts.push(message.content.trim());
      wordCounts.push(normalizedWords(message.content).length);
    }
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
  const shingleSets = assistantTexts.map((text) => shingles(text));
  let nearDuplicatePairsAbove070 = 0;
  let maximumPairwiseShingleSimilarity = 0;
  for (let left = 0; left < shingleSets.length; left++) {
    for (let right = left + 1; right < shingleSets.length; right++) {
      const similarity = jaccard(shingleSets[left]!, shingleSets[right]!);
      maximumPairwiseShingleSimilarity = Math.max(maximumPairwiseShingleSimilarity, similarity);
      if (similarity >= 0.7) nearDuplicatePairsAbove070++;
    }
  }
  const exactDuplicateAssistantMessages = [...normalizedCounts.values()]
    .reduce((sum, count) => sum + Math.max(0, count - 1), 0);
  return {
    candidates: result.rows.length,
    structurallyValid,
    structurallyRejected,
    structuralYield: result.rows.length === 0 ? 0 : structurallyValid / result.rows.length,
    assistantMessages: assistantTexts.length,
    multiTurnCandidates,
    assistantWords: {
      mean: wordCounts.length === 0 ? 0 : wordCounts.reduce((sum, count) => sum + count, 0) / wordCounts.length,
      median: quantile(wordCounts, 0.5),
      p90: quantile(wordCounts, 0.9),
      maximum: Math.max(0, ...wordCounts)
    },
    assistantQuestionEndRate: assistantTexts.length === 0
      ? 0
      : assistantTexts.filter((text) => text.endsWith("?")).length / assistantTexts.length,
    exactDuplicateAssistantMessages,
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
