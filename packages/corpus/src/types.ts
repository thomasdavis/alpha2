export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };

export type MessageRole = "system" | "user" | "assistant";
export type CandidateKind = "micro_dialogue" | "dialogue" | "linguistic_pair";
export type CandidateStatus =
  | "generated"
  | "structurally_valid"
  | "structurally_rejected"
  | "model_accepted_pending_human"
  | "model_rejected"
  | "repair_requested"
  | "human_accepted"
  | "human_rejected";

export interface NaturalMessage {
  role: MessageRole;
  content: string;
}

export interface LinguisticPair {
  sentenceA: string;
  sentenceB: string;
  contrast: string;
}

export interface HiddenContract {
  requiredCommitments: string[];
  prohibitedCommitments: string[];
  preserve: string[];
  change: string[];
  admissibleAnalyses: string[];
  discriminatingEvidence: string[];
}

export interface GeneratedItem {
  itemKey: string;
  kind: CandidateKind;
  title: string;
  primaryLens: string;
  secondaryLenses: string[];
  transformation: string;
  intendedResponsePolicy: string;
  difficulty: "introductory" | "intermediate" | "advanced";
  messages: NaturalMessage[];
  linguisticPair: LinguisticPair | null;
  hiddenContract: HiddenContract;
  generatorNotes: string;
}

export interface GenerationEnvelope {
  familySlug: string;
  items: GeneratedItem[];
  batchNotes: string;
}

export interface FamilyBlueprint {
  slug: string;
  title: string;
  purpose: string;
  competencyQuestions: string[];
  primaryLenses: string[];
  positiveCases: string[];
  hardNegatives: string[];
  legitimatePlurality: string[];
  projections: Array<{
    slug: string;
    domain: string;
    description: string;
    relation: "true_bridge" | "false_bridge" | "partial_bridge";
  }>;
  requiredCommitments: string[];
  prohibitedCommitments: string[];
  shortcutHazards: string[];
}

export type ReviewOutcome = "accept" | "reject" | "repair" | "needs_human";

export interface ReviewItem {
  candidateId: string;
  outcome: ReviewOutcome;
  scores: {
    conceptualValidity: number;
    conversationalQuality: number;
    linguisticNaturalness: number;
    pedagogicalValue: number;
    pluralityCalibration: number;
  };
  findings: Array<{
    dimension: string;
    severity: "info" | "warning" | "error";
    evidence: string;
    recommendation: string;
  }>;
  rationale: string;
}

export interface ReviewEnvelope {
  reviews: ReviewItem[];
  batchFindings: string[];
}

export type HumanReviewPass = "A" | "B";
export type HumanReviewScore = number | "not_applicable" | "uncertain" | null;

export interface HumanReviewFinding {
  dimension: string;
  severity: "observation" | "minor" | "major" | "critical";
  evidence: string;
  whyItMatters: string;
  recommendation: string;
  preserve: string;
}

export interface PassBBlueprintAssessment {
  criterion: string;
  judgment: string | null;
  rationale: string;
}

export interface PassBRequiredCommitmentAssessment {
  ordinal: number;
  targetText: string;
  expressed: string | null;
  correctness: string | null;
  naturalness: string | null;
  notes: string;
}

export interface PassBProhibitedCommitmentAssessment {
  ordinal: number;
  targetText: string;
  violated: string | null;
  evidence: string;
  severity: string | null;
}

export interface PassBDeltaInstructionAssessment {
  instructionKind: "preserve" | "change";
  ordinal: number;
  targetText: string;
  satisfied: string | null;
  evidence: string;
}

export interface PassBPluralityEvidenceAssessment {
  importantAnalysesRetained: string;
  unsupportedAnalysesIntroduced: string;
  unjustifiedCollapse: string;
  missingEvidenceNamed: string;
  sourceBoundariesPreserved: string;
  clarificationWouldResolve: string | null;
}

export interface PassBMetadataFitAssessment {
  field: string;
  judgment: string | null;
  repair: string;
}

export interface PassBPassComparison {
  changedPassA: string | null;
  contractEffect: string;
  evidenceBasis: string;
}

export interface PassBContractReview {
  blueprint: PassBBlueprintAssessment[];
  requiredCommitments: PassBRequiredCommitmentAssessment[];
  prohibitedCommitments: PassBProhibitedCommitmentAssessment[];
  deltaInstructions: PassBDeltaInstructionAssessment[];
  pluralityEvidence: PassBPluralityEvidenceAssessment;
  metadataFit: PassBMetadataFitAssessment[];
  passComparison: PassBPassComparison;
}

export interface HumanReviewResponse {
  outcome: string | null;
  summaryUserAim: string;
  summaryAssistantMove: string;
  firstSentenceEngagement: string | null;
  answeredBeforeUnnecessaryQuestion: string | null;
  scores: Record<string, HumanReviewScore>;
  dimensionEvidence: Record<string, string>;
  questionPolicy: string | null;
  missingClarification: string | null;
  findings: HumanReviewFinding[];
  rationale: string;
  confidence: number | null;
  uncertainty: string;
  expertiseNeeded: string;
  /** Present only in Pass B. Pass A's immutable packet envelope remains unchanged. */
  passBReview?: PassBContractReview;
}

export interface HumanReviewSessionResponse {
  declaredCompetencies: string[];
  competenceNote: string;
  startedAt: string;
  endedAt: string;
  interruptionStatus: string | null;
  fatigueLevel: string | null;
  conditionsNote: string;
}

export interface HumanReviewPacketAssignment {
  assignmentId: string;
  presentationId?: string;
  opaqueItemId: string;
  candidateContentSha256: string;
  candidate: JsonValue;
  response: HumanReviewResponse;
}

export interface HumanReviewPacket {
  schemaVersion: 1;
  campaignSlug: string;
  sessionId: string;
  pass: HumanReviewPass;
  reviewerAlias: string;
  rubricSlug: string;
  rubricVersion: number;
  seed: string;
  createdAt: string;
  instructions: string[];
  sessionResponse: HumanReviewSessionResponse;
  assignments: HumanReviewPacketAssignment[];
}

export interface ModelAlias {
  alias: string;
  modelId: string;
  role: "counsel" | "orchestrator" | "worker" | "critic";
  provider: string;
  transport: string;
}

export interface CampaignConfig {
  slug: string;
  purpose: string;
  workerModel: string;
  criticModel: string;
  maxGenerationCalls: number;
  maxReviewCalls: number;
  itemsPerFamily: number;
  artifactLimitBytes: number;
}

export interface StructuredCallRequest<TSchema extends JsonValue = JsonValue> {
  taskId: string;
  model: string;
  role: "orchestrator" | "worker" | "critic";
  prompt: string;
  schemaName: string;
  schema: TSchema;
  repoRoot: string;
  callRoot: string;
}

export interface StructuredCallResult {
  startedAt: string;
  completedAt: string;
  exitCode: number;
  stdout: Buffer;
  stderr: Buffer;
  lastMessage: Buffer | null;
  parsed: unknown;
  usage: {
    inputTokens: number | null;
    cachedInputTokens: number | null;
    outputTokens: number | null;
  };
  callDirectory: string;
  commandArgs: string[];
}

export interface LedgerPaths {
  home: string;
  database: string;
  blobs: string;
  calls: string;
  releases: string;
}

export interface ValidationFinding {
  code: string;
  severity: "error" | "warning";
  message: string;
}

export interface CandidateValidation {
  valid: boolean;
  findings: ValidationFinding[];
}
