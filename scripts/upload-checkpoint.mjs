#!/usr/bin/env node
/**
 * Upload a checkpoint to Railway, stripping optimizer state.
 * Usage: node scripts/upload-checkpoint.mjs <checkpoint-path> <run-id> <step>
 */
import { readFileSync, readFile } from "node:fs";
import { readFile as readFileAsync } from "node:fs/promises";
import { gzipSync } from "node:zlib";
import { dirname, join } from "node:path";

const REMOTE_URL = process.env.ALPHA_REMOTE_URL || "ALPHA_REMOTE_URL";
const REMOTE_SECRET = process.env.ALPHA_REMOTE_SECRET;
const CHUNK_SIZE = 1536 * 1024; // 1.5MB

if (!REMOTE_SECRET) {
  console.error("ALPHA_REMOTE_SECRET not set");
  process.exit(1);
}

const [ckptPath, runId, stepStr] = process.argv.slice(2);
if (!ckptPath || !runId || !stepStr) {
  console.error("Usage: node upload-checkpoint.mjs <checkpoint-path> <run-id> <step>");
  process.exit(1);
}
const step = parseInt(stepStr, 10);

console.log(`Reading checkpoint: ${ckptPath}`);
const raw = readFileSync(ckptPath);
console.log(`  raw size: ${(raw.length / 1024 / 1024).toFixed(1)}MB`);

// Strip optimizer state from binary checkpoint
let data = raw;
if (raw.length >= 8 && raw[0] === 0x41 && raw[1] === 0x4c && raw[2] === 0x50 && raw[3] === 0x48) {
  let offset = 4;
  const headerLen = raw.readUInt32LE(offset); offset += 4;
  const header = JSON.parse(raw.subarray(offset, offset + headerLen).toString("utf-8"));
  const dataStart = offset + headerLen;

  const paramTensors = [];
  const paramBuffers = [];
  let tensorOffset = dataStart;
  for (const t of header.tensors) {
    const byteLen = t.elements * 4;
    if (t.name.startsWith("p.")) {
      paramTensors.push(t);
      paramBuffers.push(raw.subarray(tensorOffset, tensorOffset + byteLen));
    }
    tensorOffset += byteLen;
  }

  const newHeader = JSON.stringify({ ...header, tensors: paramTensors, optimizerStep: 0 });
  const newHeaderBuf = Buffer.from(newHeader, "utf-8");
  const dataSize = paramBuffers.reduce((acc, b) => acc + b.length, 0);
  const totalSize = 4 + 4 + newHeaderBuf.length + dataSize;
  data = Buffer.alloc(totalSize);
  let pos = 0;
  Buffer.from("ALPH").copy(data, pos); pos += 4;
  data.writeUInt32LE(newHeaderBuf.length, pos); pos += 4;
  newHeaderBuf.copy(data, pos); pos += newHeaderBuf.length;
  for (const buf of paramBuffers) { buf.copy(data, pos); pos += buf.length; }

  console.log(`  stripped optimizer: ${(raw.length / 1024 / 1024).toFixed(1)}MB → ${(data.length / 1024 / 1024).toFixed(1)}MB`);
}

console.log("Compressing...");
const compressed = gzipSync(data);
console.log(`  compressed: ${(compressed.length / 1024 / 1024).toFixed(1)}MB`);

const totalChunks = Math.ceil(compressed.length / CHUNK_SIZE);
const uploadId = `${runId}_${step}_${Date.now()}`;

// Read config.json if exists
const runDir = dirname(ckptPath);
let config;
try {
  config = JSON.parse(readFileSync(join(runDir, "config.json"), "utf-8"));
} catch { config = { runId }; }

let metrics;
try {
  metrics = readFileSync(join(runDir, "metrics.jsonl"), "utf-8");
} catch { metrics = undefined; }

console.log(`Uploading ${totalChunks} chunks to ${REMOTE_URL}...`);
const start = Date.now();

for (let i = 0; i < totalChunks; i++) {
  const chunkStart = i * CHUNK_SIZE;
  const chunkEnd = Math.min(chunkStart + CHUNK_SIZE, compressed.length);
  const chunk = compressed.subarray(chunkStart, chunkEnd);

  const res = await fetch(`${REMOTE_URL}/api/upload/chunk`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${REMOTE_SECRET}`,
      "Content-Type": "application/octet-stream",
      "X-Upload-Id": uploadId,
      "X-Chunk-Index": String(i),
      "X-Total-Chunks": String(totalChunks),
    },
    body: chunk,
  });
  if (!res.ok) {
    console.error(`  chunk ${i} failed: ${res.status} ${await res.text()}`);
    process.exit(1);
  }
  if ((i + 1) % 25 === 0 || i + 1 === totalChunks) {
    const elapsed = (Date.now() - start) / 1000;
    const rate = (i + 1) / elapsed;
    const eta = ((totalChunks - i - 1) / rate).toFixed(0);
    console.log(`  ${i + 1}/${totalChunks} chunks (${rate.toFixed(1)}/s, ETA ${eta}s)`);
  }
}

console.log("Assembling on server...");
const assembleRes = await fetch(`${REMOTE_URL}/api/upload/assemble`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    Authorization: `Bearer ${REMOTE_SECRET}`,
  },
  body: JSON.stringify({
    uploadId,
    name: runId,
    step,
    totalChunks,
    config,
    metrics,
  }),
});

if (!assembleRes.ok) {
  console.error(`  assemble failed: ${assembleRes.status} ${await assembleRes.text()}`);
  process.exit(1);
}

const elapsed = ((Date.now() - start) / 1000).toFixed(1);
console.log(`Done! Uploaded ${runId} step ${step} in ${elapsed}s`);
