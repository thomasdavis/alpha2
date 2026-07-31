import type { JsonValue } from "./types.js";

function sortJson(value: JsonValue): JsonValue {
  if (Array.isArray(value)) return value.map(sortJson);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, child]) => [key, sortJson(child)])
    );
  }
  return value;
}

/** Browser-safe canonicalization for an already response-redacted packet. */
export function canonicalPacketEnvelopeJson(envelope: JsonValue): string {
  return JSON.stringify(sortJson(envelope));
}
