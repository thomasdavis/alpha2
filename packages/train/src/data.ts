/**
 * Minimal data pipeline.
 *
 * Loads text → tokenizes → creates random contiguous windows for training.
 */
import type { Tokenizer, Rng, TensorData } from "@alpha/core";

export interface DataBatch {
  /** Input token ids [B, T] */
  inputs: TensorData;
  /** Target token ids [B, T] */
  targets: TensorData;
}

export class DataLoader {
  private tokens: Int32Array;
  private rng: Rng;
  private batchSize: number;
  private blockSize: number;

  constructor(tokens: Int32Array, rng: Rng, batchSize: number, blockSize: number) {
    this.tokens = tokens;
    this.rng = rng;
    this.batchSize = batchSize;
    this.blockSize = blockSize;
  }

  /** Create a DataLoader from raw text. Encodes in chunks for large texts. */
  static fromText(text: string, tokenizer: Tokenizer, rng: Rng, batchSize: number, blockSize: number): DataLoader {
    // For large texts, encode in chunks to avoid exceeding JS array limits
    const CHUNK_CHARS = 5_000_000; // 5M chars per chunk
    if (text.length <= CHUNK_CHARS) {
      return new DataLoader(tokenizer.encode(text), rng, batchSize, blockSize);
    }

    const chunks: Int32Array[] = [];
    let totalLen = 0;
    for (let i = 0; i < text.length; i += CHUNK_CHARS) {
      const chunk = text.slice(i, i + CHUNK_CHARS);
      const encoded = tokenizer.encode(chunk);
      chunks.push(encoded);
      totalLen += encoded.length;
    }

    const tokens = new Int32Array(totalLen);
    let offset = 0;
    for (const chunk of chunks) {
      tokens.set(chunk, offset);
      offset += chunk.length;
    }

    return new DataLoader(tokens, rng, batchSize, blockSize);
  }

  /** Get a random batch of contiguous windows. */
  nextBatch(): DataBatch {
    const B = this.batchSize;
    const T = this.blockSize;
    if (this.tokens.length <= T) {
      throw new RangeError(
        `Token count (${this.tokens.length}) must exceed block size (${T}) — need at least ${T + 1} tokens for input+target windows`
      );
    }
    const maxStart = this.tokens.length - T;

    const inputs = new Int32Array(B * T);
    const targets = new Int32Array(B * T);

    for (let b = 0; b < B; b++) {
      const start = Math.floor(this.rng.next() * maxStart);
      for (let t = 0; t < T; t++) {
        inputs[b * T + t] = this.tokens[start + t];
        targets[b * T + t] = this.tokens[start + t + 1];
      }
    }

    return {
      inputs: { shape: [B, T], dtype: "i32", data: inputs },
      targets: { shape: [B, T], dtype: "i32", data: targets },
    };
  }

  get length(): number {
    return this.tokens.length;
  }
}

/** V8 max string length (~512MB for Latin-1, ~256MB for two-byte). Stay well below. */
const MAX_STRING_BYTES = 200 * 1024 * 1024;

/** Load text from a file path. For files > 200MB, throws — use loadTextSample instead. */
export async function loadText(path: string): Promise<string> {
  const fs = await import("node:fs/promises");
  const stat = await fs.stat(path);
  if (stat.size > MAX_STRING_BYTES) {
    throw new RangeError(
      `File is ${(stat.size / 1024 / 1024).toFixed(0)}MB — too large for a single JS string. ` +
      `Use loadTextSample() for tokenizer building or loadAndTokenize() for data loading.`
    );
  }
  return fs.readFile(path, "utf-8");
}

/**
 * Load a sample of text from a file (for tokenizer vocabulary building).
 * Reads the first `maxBytes` of the file, breaking at the last newline.
 * BPE vocabularies converge well on a 100MB sample — no need for the full corpus.
 */
export async function loadTextSample(path: string, maxBytes = 100 * 1024 * 1024): Promise<string> {
  const fs = await import("node:fs/promises");
  const stat = await fs.stat(path);
  if (stat.size <= maxBytes) {
    return fs.readFile(path, "utf-8");
  }

  const handle = await fs.open(path, "r");
  try {
    const buf = Buffer.alloc(maxBytes);
    await handle.read(buf, 0, maxBytes, 0);
    // Break at last newline to avoid splitting a UTF-8 character or word
    let end = maxBytes;
    while (end > 0 && buf[end - 1] !== 0x0a) end--;
    if (end === 0) end = maxBytes;
    return buf.subarray(0, end).toString("utf-8");
  } finally {
    await handle.close();
  }
}

/**
 * Read a file and tokenize it in chunks, avoiding V8 string length limits.
 * Returns the full token array. Works for files of any size.
 *
 * @param splitByte — if provided, only reads bytes in [0, splitByte) or [splitByte, end).
 *   Use this for train/val splitting by byte position.
 */
export async function loadAndTokenize(
  path: string,
  tokenizer: Tokenizer,
  range?: { startByte: number; endByte: number },
): Promise<Int32Array> {
  const fs = await import("node:fs/promises");
  const stat = await fs.stat(path);

  const startByte = range?.startByte ?? 0;
  const endByte = range?.endByte ?? stat.size;
  const totalBytes = endByte - startByte;
  const CHUNK_BYTES = 10 * 1024 * 1024; // 10MB chunks (keeps BPE heap memory manageable)

  // Small enough to read as a single string
  if (totalBytes <= MAX_STRING_BYTES) {
    const handle = await fs.open(path, "r");
    try {
      const buf = Buffer.alloc(totalBytes);
      await handle.read(buf, 0, totalBytes, startByte);
      const text = buf.toString("utf-8");
      return tokenizer.encode(text);
    } finally {
      await handle.close();
    }
  }

  // Chunked reading for large files
  const handle = await fs.open(path, "r");
  try {
    const chunks: Int32Array[] = [];
    let totalLen = 0;
    let position = startByte;

    while (position < endByte) {
      const readSize = Math.min(CHUNK_BYTES, endByte - position);
      const buf = Buffer.alloc(readSize);
      const { bytesRead } = await handle.read(buf, 0, readSize, position);

      // Find last newline to avoid splitting a UTF-8 character
      let end = bytesRead;
      if (position + bytesRead < endByte) {
        const lastNl = buf.lastIndexOf(0x0a, end - 1);
        if (lastNl > 0) end = lastNl + 1;
      }

      const text = buf.subarray(0, end).toString("utf-8");
      const encoded = tokenizer.encode(text);
      chunks.push(encoded);
      totalLen += encoded.length;
      position += end;
    }

    const tokens = new Int32Array(totalLen);
    let offset = 0;
    for (const chunk of chunks) {
      tokens.set(chunk, offset);
      offset += chunk.length;
    }
    return tokens;
  } finally {
    await handle.close();
  }
}

/**
 * Load tokenized data from cache or tokenize and cache.
 * Cache key: `{dataPath}.{tokenizerName}-{vocabSize}.tokens` (binary Int32Array).
 * The cache file also stores a header with the source file mtime for invalidation.
 */
export async function loadOrCacheTokens(
  dataPath: string,
  tokenizer: Tokenizer,
  range?: { startByte: number; endByte: number },
): Promise<Int32Array> {
  const fs = await import("node:fs/promises");
  const pathMod = await import("node:path");

  const suffix = range ? `.${range.startByte}-${range.endByte}` : "";
  const cacheFile = `${dataPath}.${tokenizer.name}-${tokenizer.vocabSize}${suffix}.tokens`;
  const srcStat = await fs.stat(dataPath);

  // Try loading cache
  try {
    const cacheStat = await fs.stat(cacheFile);
    if (cacheStat.size > 8) {
      const handle = await fs.open(cacheFile, "r");
      try {
        // Header: 8 bytes (Float64 source mtime)
        const headerBuf = Buffer.alloc(8);
        await handle.read(headerBuf, 0, 8, 0);
        const cachedMtime = headerBuf.readDoubleBE(0);
        if (Math.abs(cachedMtime - srcStat.mtimeMs) < 1) {
          const tokenBytes = cacheStat.size - 8;
          const dataBuf = Buffer.alloc(tokenBytes);
          await handle.read(dataBuf, 0, tokenBytes, 8);
          const tokens = new Int32Array(dataBuf.buffer, dataBuf.byteOffset, tokenBytes / 4);
          console.log(`  Loaded ${tokens.length.toLocaleString()} cached tokens from ${pathMod.basename(cacheFile)}`);
          return tokens;
        }
      } finally {
        await handle.close();
      }
    }
  } catch { /* no cache — tokenize */ }

  // Tokenize
  console.log(`  Tokenizing ${pathMod.basename(dataPath)}${suffix}...`);
  const t0 = performance.now();
  const tokens = await loadAndTokenize(dataPath, tokenizer, range);
  const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
  console.log(`  Tokenized: ${tokens.length.toLocaleString()} tokens in ${elapsed}s`);

  // Write cache — stream header + token data to avoid doubling memory with Buffer.concat
  try {
    const header = Buffer.alloc(8);
    header.writeDoubleBE(srcStat.mtimeMs, 0);
    const tokenBuf = Buffer.from(tokens.buffer, tokens.byteOffset, tokens.byteLength);
    const handle = await fs.open(cacheFile, "w");
    try {
      await handle.write(header);
      await handle.write(tokenBuf);
    } finally {
      await handle.close();
    }
    console.log(`  Cached tokens to ${pathMod.basename(cacheFile)} (${(tokenBuf.byteLength / 1024 / 1024).toFixed(0)}MB)`);
  } catch (e) {
    console.warn(`  Failed to cache tokens: ${(e as Error).message}`);
  }

  return tokens;
}

/**
 * Get the byte position for train/val split.
 * Finds the nearest newline to the split point.
 */
export async function getSplitByte(path: string, trainRatio = 0.9): Promise<number> {
  const fs = await import("node:fs/promises");
  const stat = await fs.stat(path);
  const approxSplit = Math.floor(stat.size * trainRatio);

  // Read a small window around the split point to find a newline
  const handle = await fs.open(path, "r");
  try {
    const windowSize = 4096;
    const start = Math.max(0, approxSplit - windowSize);
    const buf = Buffer.alloc(Math.min(windowSize * 2, stat.size - start));
    await handle.read(buf, 0, buf.length, start);
    // Find the closest newline to the approximate split
    const target = approxSplit - start;
    for (let i = target; i < buf.length; i++) {
      if (buf[i] === 0x0a) return start + i + 1;
    }
    for (let i = target - 1; i >= 0; i--) {
      if (buf[i] === 0x0a) return start + i + 1;
    }
    return approxSplit;
  } finally {
    await handle.close();
  }
}

/** Split text into train/val by ratio. */
export function splitText(text: string, trainRatio = 0.9): { train: string; val: string } {
  const splitIdx = Math.floor(text.length * trainRatio);
  return {
    train: text.slice(0, splitIdx),
    val: text.slice(splitIdx),
  };
}
