/**
 * @alpha/historic-chat-gen — Synthetic historical dialogue generator
 *
 * Public API exports.
 */
export { chatgenCmd } from "./cli.js";
export { generate, resume, generateAssignments } from "./generator.js";
export { exportTrainingData } from "./export.js";
export { doctor } from "./doctor.js";
export { createDb, closeDb, getDb, getRunStats, getLatestRun } from "./db.js";
export { loadFigures } from "./figures.js";
export { loadTopics } from "./topics.js";
export { hasApiKey } from "./openai.js";
export { estimateTotalCost, estimateCostPerConversation, formatCost } from "./budget.js";
export type {
  Figure, Topic, Tone, ConversationAssignment, ConversationTurn,
  GeneratedConversation, Run, Batch, GenerateOptions, ExportOptions,
  StatsResult, PlanResult,
} from "./types.js";
