import { statSync } from "node:fs";
import { CorpusReader, resolveLedgerPaths } from "@alpha/corpus";

type CorpusGlobal = typeof globalThis & {
  __alphaCorpusReader?: CorpusReader;
  __alphaCorpusReaderPath?: string;
};

function configuredPath(): string {
  return process.env.CORPUS_DB_PATH ?? resolveLedgerPaths().database;
}
export function getCorpusReader(): CorpusReader {
  const state = globalThis as CorpusGlobal;
  const path = configuredPath();
  if (state.__alphaCorpusReader && state.__alphaCorpusReaderPath === path) {
    return state.__alphaCorpusReader;
  }
  state.__alphaCorpusReader?.close();
  state.__alphaCorpusReader = new CorpusReader(path);
  state.__alphaCorpusReaderPath = path;
  return state.__alphaCorpusReader;
}

export function corpusDatabaseUpdatedAt(): string {
  return statSync(configuredPath()).mtime.toISOString();
}
