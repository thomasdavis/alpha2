/**
 * All TypeScript interfaces for historic-chat-gen.
 */

export interface Figure {
  id: string;
  name: string;
  era: string;
  birth: number;
  death: number | null;
  speechTraits: string;
  worldview: string;
  vocabulary: string;
  bio: string;
}

export interface Topic {
  id: string;
  name: string;
  description: string;
  category: string;
}

export type Tone =
  | "formal_debate"
  | "casual_discussion"
  | "heated_argument"
  | "philosophical_inquiry"
  | "mentorship"
  | "reluctant_agreement"
  | "comedic_misunderstanding";

export const TONES: Tone[] = [
  "formal_debate",
  "casual_discussion",
  "heated_argument",
  "philosophical_inquiry",
  "mentorship",
  "reluctant_agreement",
  "comedic_misunderstanding",
];

export interface ConversationAssignment {
  figureA: Figure;
  figureB: Figure;
  topic: Topic;
  tone: Tone;
  turnCount: number;
}

export interface ConversationTurn {
  speaker: string;
  text: string;
}

export interface GeneratedConversation {
  id: string;
  runId: string;
  batchId: string;
  figureA: string;
  figureB: string;
  topic: string;
  tone: Tone;
  turnCount: number;
  turns: ConversationTurn[];
  inputTokens: number;
  outputTokens: number;
  cost: number;
  createdAt: string;
}

export type RunStatus = "active" | "completed" | "paused" | "failed";
export type BatchStatus = "pending" | "committed" | "rolled_back";

export interface Run {
  id: string;
  targetCount: number;
  budgetLimit: number;
  concurrency: number;
  batchSize: number;
  status: RunStatus;
  completedCount: number;
  failedCount: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  totalCost: number;
  seed: number;
  createdAt: string;
  updatedAt: string;
}

export interface Batch {
  id: string;
  runId: string;
  index: number;
  status: BatchStatus;
  conversationCount: number;
  createdAt: string;
}

export interface GenerateOptions {
  count: number;
  budget: number;
  concurrency: number;
  batchSize: number;
  seed?: number;
}

export type ExportFormat = "jsonl" | "chat";

export interface ExportOptions {
  runId?: string;
  out: string;
  format: ExportFormat;
}

export interface StatsResult {
  runId: string;
  status: RunStatus;
  completed: number;
  failed: number;
  target: number;
  totalCost: number;
  budgetLimit: number;
  budgetRemaining: number;
  avgCostPerConversation: number;
  inputTokens: number;
  outputTokens: number;
  figureDistribution: Record<string, number>;
  topicDistribution: Record<string, number>;
}

export interface PlanResult {
  figureCount: number;
  topicCount: number;
  toneCount: number;
  uniqueCombinations: number;
  requestedCount: number;
  estimatedCost: number;
  budget: number;
  apiKeySet: boolean;
}

export interface OpenAIResponse {
  choices: Array<{
    message: {
      content: string;
    };
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}
