import type {
  CandidateValidation,
  GeneratedItem,
  GenerationEnvelope,
  NaturalMessage,
  ValidationFinding
} from "./types.js";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((entry) => typeof entry === "string");
}

function isMessage(value: unknown): value is NaturalMessage {
  return isRecord(value)
    && ["system", "user", "assistant"].includes(String(value["role"]))
    && typeof value["content"] === "string";
}

export function parseGenerationEnvelope(value: unknown): GenerationEnvelope {
  if (!isRecord(value) || typeof value["familySlug"] !== "string" || !Array.isArray(value["items"])) {
    throw new Error("Structured response is not a generation envelope");
  }
  if (typeof value["batchNotes"] !== "string") throw new Error("Generation envelope lacks batchNotes");
  for (const item of value["items"]) {
    if (!isRecord(item)) throw new Error("Generation item is not an object");
    const requiredStrings = [
      "itemKey", "kind", "title", "primaryLens", "transformation",
      "intendedResponsePolicy", "difficulty", "generatorNotes"
    ];
    if (requiredStrings.some((key) => typeof item[key] !== "string")) {
      throw new Error("Generation item has a missing string field");
    }
    if (!isStringArray(item["secondaryLenses"]) || !Array.isArray(item["messages"])
      || !item["messages"].every(isMessage)) {
      throw new Error("Generation item has malformed lenses or messages");
    }
    if (!isRecord(item["hiddenContract"])) throw new Error("Generation item lacks hiddenContract");
    for (const key of [
      "requiredCommitments", "prohibitedCommitments", "preserve", "change",
      "admissibleAnalyses", "discriminatingEvidence"
    ]) {
      if (!isStringArray(item["hiddenContract"][key])) {
        throw new Error(`Generation item hiddenContract.${key} is malformed`);
      }
    }
  }
  return value as unknown as GenerationEnvelope;
}

function add(findings: ValidationFinding[], code: string, message: string, severity: "error" | "warning" = "error"): void {
  findings.push({ code, severity, message });
}

export function validateCandidate(
  item: GeneratedItem,
  expectedFamilySlug: string,
  expectedItemPrefix: string,
  allowedLenses: Set<string>,
  allowedTransformations: Set<string>
): CandidateValidation {
  const findings: ValidationFinding[] = [];
  if (!item.itemKey.startsWith(expectedItemPrefix)) {
    add(findings, "item_key_scope", `itemKey must start with ${expectedItemPrefix}`);
  }
  if (!allowedLenses.has(item.primaryLens)) {
    add(findings, "unknown_primary_lens", `Unknown primary lens ${item.primaryLens}`);
  }
  for (const lens of item.secondaryLenses) {
    if (!allowedLenses.has(lens)) add(findings, "unknown_secondary_lens", `Unknown secondary lens ${lens}`);
  }
  if (!allowedTransformations.has(item.transformation)) {
    add(findings, "unknown_transformation", `Unknown transformation ${item.transformation}`);
  }
  if (item.messages.length < 2) add(findings, "too_few_messages", "At least two natural-language messages are required");
  if (item.messages[0]?.role !== "user") {
    add(findings, "opening_role", "The first model-visible message must be a user turn");
  }
  if (!item.messages.some((message) => message.role === "assistant")) {
    add(findings, "missing_assistant", "At least one assistant turn is required");
  }
  for (const [index, message] of item.messages.entries()) {
    const text = message.content.trim();
    if (text.length === 0) add(findings, "blank_message", `Message ${index} is blank`);
    if (/<\/?(?:assistant|user|system)>/i.test(text) || /\[(?:assistant|user|system)\]/i.test(text)) {
      add(findings, "delimiter_leak", `Message ${index} contains a serialized role delimiter`);
    }
    if (/\b(?:hiddenContract|requiredCommitments|prohibitedCommitments|primaryLens)\b/.test(text)) {
      add(findings, "metadata_leak", `Message ${index} exposes researcher-side metadata`);
    }
  }
  const allContractText = Object.values(item.hiddenContract).flat();
  if (allContractText.some((text) => text.trim().length === 0)) {
    add(findings, "blank_contract", "Hidden contract contains an empty entry");
  }
  const required = new Set(item.hiddenContract.requiredCommitments.map((entry) => entry.trim().toLowerCase()));
  if (item.hiddenContract.prohibitedCommitments.some((entry) => required.has(entry.trim().toLowerCase()))) {
    add(findings, "contract_contradiction", "The same commitment is both required and prohibited");
  }
  if (item.kind === "linguistic_pair" && item.linguisticPair === null) {
    add(findings, "missing_linguistic_pair", "A linguistic_pair item requires its contrast fields");
  }
  if (item.kind !== "linguistic_pair" && item.linguisticPair !== null) {
    add(findings, "unexpected_linguistic_pair", "Dialogue items must leave linguisticPair null", "warning");
  }
  if (item.generatorNotes.includes(expectedFamilySlug)) {
    add(findings, "generator_note_template", "Generator note repeats the family slug; inspect for template leakage", "warning");
  }
  return {
    valid: findings.every((finding) => finding.severity !== "error"),
    findings
  };
}
