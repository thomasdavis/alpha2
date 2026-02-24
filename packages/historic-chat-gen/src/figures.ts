/**
 * Figure registry loader.
 */
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import type { Figure } from "./types.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

let _figures: Figure[] | null = null;

export function loadFigures(): Figure[] {
  if (_figures) return _figures;
  // In dist/, go up to package root then into data/
  const jsonPath = resolve(__dirname, "..", "data", "figures.json");
  const raw = readFileSync(jsonPath, "utf-8");
  _figures = JSON.parse(raw) as Figure[];
  return _figures;
}
