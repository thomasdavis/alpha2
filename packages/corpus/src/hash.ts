import { createHash } from "node:crypto";
import type { JsonValue } from "./types.js";

export function sha256Bytes(value: Uint8Array | string): string {
  return createHash("sha256").update(value).digest("hex");
}

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

export function canonicalJson(value: JsonValue): string {
  return JSON.stringify(sortJson(value));
}

export function sha256Json(value: JsonValue): string {
  return sha256Bytes(canonicalJson(value));
}

export function stableId(namespace: string, value: string): string {
  const digest = sha256Bytes(`${namespace}\0${value}`).slice(0, 32);
  return `${namespace}_${digest}`;
}
