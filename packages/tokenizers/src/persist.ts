/**
 * Persistence helpers for tokenizer artifacts.
 *
 * Saves and loads `TokenizerArtifacts` as JSON files using node:fs/promises,
 * with every I/O operation wrapped in `Effect.tryPromise` so callers get
 * typed `TokenizerError` failures instead of raw exceptions.
 */
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import { Effect } from "effect";
import { TokenizerError, type TokenizerArtifacts } from "@alpha/core";

/**
 * Serialise tokenizer artifacts to a JSON file.
 *
 * Creates parent directories if they don't already exist, then writes a
 * pretty-printed JSON blob that is easy to inspect manually.
 *
 * @param path  - Destination file path (should end in `.json`).
 * @param artifacts - The artifacts produced by a tokenizer's `build()` call.
 */
export function saveArtifacts(
  path: string,
  artifacts: TokenizerArtifacts,
): Effect.Effect<void, TokenizerError> {
  return Effect.tryPromise({
    try: async () => {
      const dir = dirname(path);
      await mkdir(dir, { recursive: true });

      const json = JSON.stringify(
        {
          type: artifacts.type,
          vocabSize: artifacts.vocabSize,
          vocab: artifacts.vocab,
          ...(artifacts.merges ? { merges: artifacts.merges } : {}),
        },
        null,
        2,
      );

      await writeFile(path, json, "utf-8");
    },
    catch: (cause) =>
      new TokenizerError({
        message: `Failed to save tokenizer artifacts to "${path}"`,
        cause,
      }),
  });
}

/**
 * Load tokenizer artifacts from a JSON file.
 *
 * Reads the file, parses the JSON, and performs basic validation to make
 * sure the payload looks like a valid `TokenizerArtifacts` object before
 * returning it.
 *
 * @param path - Source file path.
 */
export function loadArtifacts(
  path: string,
): Effect.Effect<TokenizerArtifacts, TokenizerError> {
  return Effect.tryPromise({
    try: async () => {
      const raw = await readFile(path, "utf-8");
      const data = JSON.parse(raw) as Record<string, unknown>;

      // Validate required fields.
      if (typeof data.type !== "string") {
        throw new Error("Missing or invalid 'type' field");
      }
      if (typeof data.vocabSize !== "number") {
        throw new Error("Missing or invalid 'vocabSize' field");
      }
      if (!Array.isArray(data.vocab)) {
        throw new Error("Missing or invalid 'vocab' field");
      }

      const artifacts: TokenizerArtifacts = {
        type: data.type as string,
        vocabSize: data.vocabSize as number,
        vocab: data.vocab as string[],
        ...(Array.isArray(data.merges)
          ? { merges: data.merges as [number, number][] }
          : {}),
      };

      return artifacts;
    },
    catch: (cause) =>
      new TokenizerError({
        message: `Failed to load tokenizer artifacts from "${path}"`,
        cause,
      }),
  });
}
