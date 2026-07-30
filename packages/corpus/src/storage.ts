import {
  existsSync,
  lstatSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  renameSync,
  statSync,
  writeFileSync
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import { randomUUID } from "node:crypto";
import { sha256Bytes } from "./hash.js";
import type { LedgerPaths } from "./types.js";

export const DEFAULT_ARTIFACT_LIMIT_BYTES = 15 * 1024 * 1024 * 1024;

export function resolveLedgerPaths(home?: string): LedgerPaths {
  const resolvedHome = resolve(
    home
      ?? process.env["ALPHA_CORPUS_HOME"]
      ?? "/mnt/donto-data/donto-resources/research/alpha2-corpus"
  );
  return {
    home: resolvedHome,
    database: join(resolvedHome, "alpha-corpus.sqlite"),
    blobs: join(resolvedHome, "blobs", "sha256"),
    calls: join(resolvedHome, "calls"),
    releases: join(resolvedHome, "releases")
  };
}

export function ensureLedgerDirectories(paths: LedgerPaths): void {
  mkdirSync(paths.home, { recursive: true });
  mkdirSync(paths.blobs, { recursive: true });
  mkdirSync(paths.calls, { recursive: true });
  mkdirSync(paths.releases, { recursive: true });
}

export function blobRelativePath(sha256: string): string {
  return join("blobs", "sha256", sha256.slice(0, 2), sha256);
}

export function writeContentAddressedBlob(
  paths: LedgerPaths,
  bytes: Uint8Array
): { sha256: string; byteLength: number; relativePath: string; absolutePath: string } {
  const sha256 = sha256Bytes(bytes);
  const relativePath = blobRelativePath(sha256);
  const absolutePath = join(paths.home, relativePath);
  mkdirSync(dirname(absolutePath), { recursive: true });

  if (existsSync(absolutePath)) {
    const existing = readFileSync(absolutePath);
    if (sha256Bytes(existing) !== sha256) {
      throw new Error(`Content-address collision or corruption at ${absolutePath}`);
    }
  } else {
    const temporary = `${absolutePath}.tmp-${process.pid}-${randomUUID()}`;
    writeFileSync(temporary, bytes, { flag: "wx" });
    renameSync(temporary, absolutePath);
  }

  return { sha256, byteLength: bytes.byteLength, relativePath, absolutePath };
}

export function writeAtomic(path: string, bytes: Uint8Array | string): void {
  mkdirSync(dirname(path), { recursive: true });
  const temporary = `${path}.tmp-${process.pid}-${randomUUID()}`;
  writeFileSync(temporary, bytes, { flag: "wx" });
  renameSync(temporary, path);
}

export function directorySize(path: string): number {
  if (!existsSync(path)) return 0;
  const rootStat = lstatSync(path);
  if (!rootStat.isDirectory()) return rootStat.size;

  let total = 0;
  const stack = [path];
  while (stack.length > 0) {
    const current = stack.pop()!;
    for (const entry of readdirSync(current, { withFileTypes: true })) {
      const child = join(current, entry.name);
      if (entry.isSymbolicLink()) {
        total += lstatSync(child).size;
      } else if (entry.isDirectory()) {
        stack.push(child);
      } else if (entry.isFile()) {
        total += statSync(child).size;
      }
    }
  }
  return total;
}

export function formatBytes(bytes: number): string {
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit++;
  }
  return `${value.toFixed(unit === 0 ? 0 : 2)} ${units[unit]}`;
}
