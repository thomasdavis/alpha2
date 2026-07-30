import type { JsonValue } from "./types.js";

const stringArray = {
  type: "array",
  items: { type: "string", minLength: 1 }
} as const;

const messageSchema = {
  type: "object",
  additionalProperties: false,
  required: ["role", "content"],
  properties: {
    role: { type: "string", enum: ["system", "user", "assistant"] },
    content: { type: "string", minLength: 1 }
  }
} as const;

const hiddenContractSchema = {
  type: "object",
  additionalProperties: false,
  required: [
    "requiredCommitments",
    "prohibitedCommitments",
    "preserve",
    "change",
    "admissibleAnalyses",
    "discriminatingEvidence"
  ],
  properties: {
    requiredCommitments: stringArray,
    prohibitedCommitments: stringArray,
    preserve: stringArray,
    change: stringArray,
    admissibleAnalyses: stringArray,
    discriminatingEvidence: stringArray
  }
} as const;

export const generationEnvelopeSchema = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  type: "object",
  additionalProperties: false,
  required: ["familySlug", "items", "batchNotes"],
  properties: {
    familySlug: { type: "string", minLength: 1 },
    items: {
      type: "array",
      minItems: 1,
      items: {
        type: "object",
        additionalProperties: false,
        required: [
          "itemKey",
          "kind",
          "title",
          "primaryLens",
          "secondaryLenses",
          "transformation",
          "intendedResponsePolicy",
          "difficulty",
          "messages",
          "linguisticPair",
          "hiddenContract",
          "generatorNotes"
        ],
        properties: {
          itemKey: { type: "string", minLength: 1 },
          kind: { type: "string", enum: ["micro_dialogue", "dialogue", "linguistic_pair"] },
          title: { type: "string", minLength: 1 },
          primaryLens: { type: "string", minLength: 1 },
          secondaryLenses: stringArray,
          transformation: { type: "string", minLength: 1 },
          intendedResponsePolicy: { type: "string", minLength: 1 },
          difficulty: { type: "string", enum: ["introductory", "intermediate", "advanced"] },
          messages: { type: "array", minItems: 2, items: messageSchema },
          linguisticPair: {
            anyOf: [
              { type: "null" },
              {
                type: "object",
                additionalProperties: false,
                required: ["sentenceA", "sentenceB", "contrast"],
                properties: {
                  sentenceA: { type: "string", minLength: 1 },
                  sentenceB: { type: "string", minLength: 1 },
                  contrast: { type: "string", minLength: 1 }
                }
              }
            ]
          },
          hiddenContract: hiddenContractSchema,
          generatorNotes: { type: "string" }
        }
      }
    },
    batchNotes: { type: "string" }
  }
} as unknown as JsonValue;

export const reviewEnvelopeSchema = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  type: "object",
  additionalProperties: false,
  required: ["reviews", "batchFindings"],
  properties: {
    reviews: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["candidateId", "outcome", "scores", "findings", "rationale"],
        properties: {
          candidateId: { type: "string", minLength: 1 },
          outcome: { type: "string", enum: ["accept", "reject", "repair", "needs_human"] },
          scores: {
            type: "object",
            additionalProperties: false,
            required: [
              "conceptualValidity",
              "conversationalQuality",
              "linguisticNaturalness",
              "pedagogicalValue",
              "pluralityCalibration"
            ],
            properties: Object.fromEntries(
              [
                "conceptualValidity",
                "conversationalQuality",
                "linguisticNaturalness",
                "pedagogicalValue",
                "pluralityCalibration"
              ].map((key) => [key, { type: "number", minimum: 0, maximum: 5 }])
            )
          },
          findings: {
            type: "array",
            items: {
              type: "object",
              additionalProperties: false,
              required: ["dimension", "severity", "evidence", "recommendation"],
              properties: {
                dimension: { type: "string", minLength: 1 },
                severity: { type: "string", enum: ["info", "warning", "error"] },
                evidence: { type: "string", minLength: 1 },
                recommendation: { type: "string", minLength: 1 }
              }
            }
          },
          rationale: { type: "string", minLength: 1 }
        }
      }
    },
    batchFindings: stringArray
  }
} as unknown as JsonValue;
