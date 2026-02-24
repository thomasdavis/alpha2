/**
 * Topic registry loader.
 */
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import type { Topic } from "./types.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

let _topics: Topic[] | null = null;

export function loadTopics(): Topic[] {
  if (_topics) return _topics;
  const jsonPath = resolve(__dirname, "..", "data", "topics.json");
  const raw = readFileSync(jsonPath, "utf-8");
  _topics = JSON.parse(raw) as Topic[];
  return _topics;
}
