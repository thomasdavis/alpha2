import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import type { GPTParams } from "@alpha/model";
import { collectParamEntries } from "@alpha/model";

export function sha256Bytes(value: string | Uint8Array): string {
  return `sha256:${createHash("sha256").update(value).digest("hex")}`;
}

export function stableJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  const record = value as Record<string, unknown>;
  return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${stableJson(record[key])}`).join(",")}}`;
}

export async function sha256File(path: string): Promise<string> {
  const hash = createHash("sha256");
  await new Promise<void>((resolve, reject) => {
    const stream = createReadStream(path);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("error", reject);
    stream.on("end", resolve);
  });
  return `sha256:${hash.digest("hex")}`;
}

/** Composite over sorted native parameter names, shapes, and exact f32 bytes. */
export function fingerprintWeights(params: GPTParams): string {
  const hash = createHash("sha256");
  for (const [name, variable] of collectParamEntries(params).sort(([a], [b]) => a.localeCompare(b))) {
    hash.update(name);
    hash.update("\0");
    hash.update(variable.data.shape.join(","));
    hash.update("\0");
    const data = variable.data.data as Float32Array;
    hash.update(Buffer.from(data.buffer, data.byteOffset, data.byteLength));
  }
  return `sha256:${hash.digest("hex")}`;
}
