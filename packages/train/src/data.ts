/**
 * Data pipeline.
 *
 * Two modes:
 * - Random: independent random windows per batch item (original behavior)
 * - Packed: B cursors advance sequentially through the corpus for deterministic
 *   coverage, sequential memory access, and no random number overhead.
 */
import type { Tokenizer, Rng, TensorData } from "@alpha/core";

export interface DataBatch {
  /** Input token ids [B, T] */
  inputs: TensorData;
  /** Target token ids [B, T] */
  targets: TensorData;
  /**
   * Optional per-position loss weights [B, T] f32 (assistant-only SFT). Absent
   * for pretraining (all-ones semantics, zero overhead). When present, a 0 marks
   * a position that must not contribute loss/gradient (user tokens + padding).
   */
  lossMask?: TensorData;
}

/**
 * Common shape of anything the trainer's step loop pulls batches from — the
 * pretraining {@link DataLoader} and the {@link SftDataLoader} both satisfy it.
 */
export interface BatchSource {
  nextBatch(): DataBatch;
  /** Position a freshly constructed loader after `count` already-consumed batches. */
  seekBatches(count: number): void;
  readonly stepsPerEpoch: number;
  readonly length: number;
}

export interface PretrainShardManifest {
  schema: "alpha-pretrain-shards-v1";
  shards: { path: string; sha256: string }[];
}

/** Load and structurally validate a multi-file pretraining manifest. */
export async function loadPretrainShardManifest(
  manifestPath: string,
): Promise<{ manifest: PretrainShardManifest; paths: string[] }> {
  const fs = await import("node:fs/promises");
  const path = await import("node:path");
  const parsed = JSON.parse(await fs.readFile(manifestPath, "utf8")) as Partial<PretrainShardManifest>;
  if (parsed.schema !== "alpha-pretrain-shards-v1" || !Array.isArray(parsed.shards) || parsed.shards.length < 2) {
    throw new Error(`${manifestPath}: expected alpha-pretrain-shards-v1 with at least two shards`);
  }
  const base = path.dirname(path.resolve(manifestPath));
  const paths: string[] = [];
  const seen = new Set<string>();
  for (const [index, shard] of parsed.shards.entries()) {
    if (!shard || typeof shard.path !== "string" || shard.path.length === 0 ||
        typeof shard.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(shard.sha256)) {
      throw new Error(`${manifestPath}: invalid shard entry ${index}`);
    }
    const resolved = path.isAbsolute(shard.path) ? shard.path : path.resolve(base, shard.path);
    if (seen.has(resolved)) throw new Error(`${manifestPath}: duplicate shard path ${resolved}`);
    const fileStat = await fs.stat(resolved);
    if (!fileStat.isFile() || fileStat.size === 0) throw new Error(`${manifestPath}: shard is not a non-empty file: ${resolved}`);
    seen.add(resolved);
    paths.push(resolved);
  }
  return { manifest: parsed as PretrainShardManifest, paths };
}

/** Stream-hash every shard before paid training; returns byte counts for provenance. */
export async function verifyPretrainShardManifest(
  manifest: PretrainShardManifest,
  paths: readonly string[],
): Promise<{ path: string; bytes: number; sha256: string }[]> {
  if (manifest.shards.length !== paths.length) throw new Error("manifest/path count mismatch");
  const { createHash } = await import("node:crypto");
  const { createReadStream } = await import("node:fs");
  const fs = await import("node:fs/promises");
  const verified: { path: string; bytes: number; sha256: string }[] = [];
  for (let index = 0; index < paths.length; index++) {
    const hash = createHash("sha256");
    for await (const chunk of createReadStream(paths[index])) hash.update(chunk);
    const actual = hash.digest("hex");
    const expected = manifest.shards[index].sha256;
    if (actual !== expected) throw new Error(`${paths[index]}: SHA-256 ${actual} != manifest ${expected}`);
    verified.push({ path: paths[index], bytes: (await fs.stat(paths[index])).size, sha256: actual });
  }
  return verified;
}

export class DataLoader implements BatchSource {
  private tokens: Int32Array;
  private rng: Rng;
  private batchSize: number;
  private blockSize: number;
  private packed: boolean;
  /** Packed mode: B cursors spread evenly across the corpus. */
  private cursors?: number[];
  /** Reusable batch buffers to avoid per-step allocations. */
  private batchRing: DataBatch[] = [];
  private batchRingIdx = 0;

  constructor(tokens: Int32Array, rng: Rng, batchSize: number, blockSize: number, packed = false) {
    this.tokens = tokens;
    this.rng = rng;
    this.batchSize = batchSize;
    this.blockSize = blockSize;
    this.packed = packed;

    if (packed) {
      // Spread B cursors evenly so each batch item processes a different
      // non-overlapping stripe of the corpus. Guarantees all tokens are
      // seen once per full pass (epoch).
      const stride = Math.floor(tokens.length / batchSize);
      this.cursors = [];
      for (let b = 0; b < batchSize; b++) {
        this.cursors.push(b * stride);
      }
    }

    // Double-buffer batches so callers that retain a returned batch for a short
    // time won't be clobbered by the next call.
    const elems = batchSize * blockSize;
    for (let i = 0; i < 2; i++) {
      const inputs = new Int32Array(elems);
      const targets = new Int32Array(elems);
      this.batchRing.push({
        inputs: { shape: [batchSize, blockSize], dtype: "i32", data: inputs },
        targets: { shape: [batchSize, blockSize], dtype: "i32", data: targets },
      });
    }
  }

  /** Create a DataLoader from raw text. Encodes in chunks for large texts. */
  static fromText(text: string, tokenizer: Tokenizer, rng: Rng, batchSize: number, blockSize: number, packed = false): DataLoader {
    // For large texts, encode in chunks to avoid exceeding JS array limits
    const CHUNK_CHARS = 5_000_000; // 5M chars per chunk
    if (text.length <= CHUNK_CHARS) {
      return new DataLoader(tokenizer.encode(text), rng, batchSize, blockSize, packed);
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

    return new DataLoader(tokens, rng, batchSize, blockSize, packed);
  }

  /** Get the next batch (dispatches to random or packed mode). */
  nextBatch(): DataBatch {
    return this.packed ? this.nextBatchPacked() : this.nextBatchRandom();
  }

  seekBatches(count: number): void {
    if (!Number.isSafeInteger(count) || count < 0) throw new RangeError(`batch count must be a non-negative integer: ${count}`);
    this.batchRingIdx = count % this.batchRing.length;
    if (this.packed) {
      const stride = Math.floor(this.tokens.length / this.batchSize);
      const offset = (count * this.blockSize) % this.tokens.length;
      for (let b = 0; b < this.batchSize; b++) {
        this.cursors![b] = (b * stride + offset) % this.tokens.length;
      }
      return;
    }
    // Random mode consumes one draw per batch item. The loader owns a fresh,
    // data-only RNG, so replaying draws positions it exactly without touching
    // model initialization or dropout streams.
    for (let i = 0; i < count * this.batchSize; i++) this.rng.next();
  }

  /** Random mode: each batch item gets an independent random window. */
  private nextBatchRandom(): DataBatch {
    const B = this.batchSize;
    const T = this.blockSize;
    if (this.tokens.length <= T) {
      throw new RangeError(
        `Token count (${this.tokens.length}) must exceed block size (${T}) — need at least ${T + 1} tokens for input+target windows`
      );
    }
    const maxStart = this.tokens.length - T;

    const batch = this.batchRing[this.batchRingIdx];
    this.batchRingIdx = (this.batchRingIdx + 1) % this.batchRing.length;
    const inputs = batch.inputs.data as Int32Array;
    const targets = batch.targets.data as Int32Array;

    for (let b = 0; b < B; b++) {
      const start = Math.floor(this.rng.next() * maxStart);
      const dst = b * T;
      inputs.set(this.tokens.subarray(start, start + T), dst);
      targets.set(this.tokens.subarray(start + 1, start + T + 1), dst);
    }

    return {
      inputs: { ...batch.inputs },
      targets: { ...batch.targets },
    };
  }

  /**
   * Packed mode: B cursors advance sequentially through the corpus.
   *
   * Benefits:
   * - Sequential memory access (cache-friendly reads from token array)
   * - Deterministic coverage: every token seen exactly once per epoch
   * - No random number generation overhead per batch
   * - All documents naturally packed contiguously (no wasted padding)
   */
  private nextBatchPacked(): DataBatch {
    const B = this.batchSize;
    const T = this.blockSize;
    const N = this.tokens.length;
    const cursors = this.cursors!;

    const batch = this.batchRing[this.batchRingIdx];
    this.batchRingIdx = (this.batchRingIdx + 1) % this.batchRing.length;
    const inputs = batch.inputs.data as Int32Array;
    const targets = batch.targets.data as Int32Array;

    for (let b = 0; b < B; b++) {
      let pos = cursors[b];
      let remaining = T;
      let dst = b * T;
      while (remaining > 0) {
        const span = Math.min(remaining, N - pos);
        inputs.set(this.tokens.subarray(pos, pos + span), dst);
        if (pos + span < N) {
          targets.set(this.tokens.subarray(pos + 1, pos + span + 1), dst);
        } else {
          if (span > 1) targets.set(this.tokens.subarray(pos + 1, N), dst);
          targets[dst + span - 1] = this.tokens[0];
        }
        dst += span;
        remaining -= span;
        pos += span;
        if (pos === N) pos = 0;
      }
      cursors[b] = pos;
    }

    return {
      inputs: { ...batch.inputs },
      targets: { ...batch.targets },
    };
  }

  /** Steps per epoch (approximate, for logging). */
  get stepsPerEpoch(): number {
    return Math.floor(this.tokens.length / (this.batchSize * this.blockSize));
  }

  get length(): number {
    return this.tokens.length;
  }
}

/**
 * A logical concatenation of independently cached token shards. It preserves
 * DataLoader's packed/random ordering without allocating one giant Int32Array,
 * which would exceed Node's Buffer/TypedArray practical limits for a multi-
 * billion-token corpus.
 */
export class ShardedDataLoader implements BatchSource {
  private readonly shards: readonly Int32Array[];
  /** Cumulative starts plus a final total-length sentinel. */
  private readonly offsets: number[];
  private readonly rng: Rng;
  private readonly batchSize: number;
  private readonly blockSize: number;
  private readonly packed: boolean;
  private readonly totalLength: number;
  private cursors?: number[];
  private readonly batchRing: DataBatch[] = [];
  private batchRingIdx = 0;

  constructor(shards: readonly Int32Array[], rng: Rng, batchSize: number, blockSize: number, packed = false) {
    if (shards.length === 0 || shards.some((shard) => shard.length === 0)) {
      throw new RangeError("ShardedDataLoader needs at least one non-empty token shard");
    }
    this.shards = shards;
    this.rng = rng;
    this.batchSize = batchSize;
    this.blockSize = blockSize;
    this.packed = packed;
    this.offsets = [0];
    for (const shard of shards) this.offsets.push(this.offsets.at(-1)! + shard.length);
    this.totalLength = this.offsets.at(-1)!;
    if (this.totalLength <= blockSize) {
      throw new RangeError(
        `Token count (${this.totalLength}) must exceed block size (${blockSize}) — need at least ${blockSize + 1} tokens`,
      );
    }
    if (packed) {
      const stride = Math.floor(this.totalLength / batchSize);
      this.cursors = Array.from({ length: batchSize }, (_, index) => index * stride);
    }
    const elems = batchSize * blockSize;
    for (let index = 0; index < 2; index++) {
      this.batchRing.push({
        inputs: { shape: [batchSize, blockSize], dtype: "i32", data: new Int32Array(elems) },
        targets: { shape: [batchSize, blockSize], dtype: "i32", data: new Int32Array(elems) },
      });
    }
  }

  nextBatch(): DataBatch {
    return this.packed ? this.nextBatchPacked() : this.nextBatchRandom();
  }

  seekBatches(count: number): void {
    if (!Number.isSafeInteger(count) || count < 0) throw new RangeError(`batch count must be a non-negative integer: ${count}`);
    this.batchRingIdx = count % this.batchRing.length;
    if (this.packed) {
      const stride = Math.floor(this.totalLength / this.batchSize);
      const offset = (count * this.blockSize) % this.totalLength;
      for (let batch = 0; batch < this.batchSize; batch++) {
        this.cursors![batch] = (batch * stride + offset) % this.totalLength;
      }
      return;
    }
    for (let index = 0; index < count * this.batchSize; index++) this.rng.next();
  }

  private locateShard(position: number): number {
    let low = 0;
    let high = this.shards.length - 1;
    while (low <= high) {
      const middle = (low + high) >>> 1;
      if (position < this.offsets[middle]) high = middle - 1;
      else if (position >= this.offsets[middle + 1]) low = middle + 1;
      else return middle;
    }
    throw new RangeError(`logical token position out of range: ${position}`);
  }

  /** Copy a logical range, wrapping from the final shard to the first. */
  private copyRange(destination: Int32Array, destinationOffset: number, start: number, length: number): void {
    let position = start;
    let remaining = length;
    let outputOffset = destinationOffset;
    while (remaining > 0) {
      const shardIndex = this.locateShard(position);
      const shard = this.shards[shardIndex];
      const localPosition = position - this.offsets[shardIndex];
      const span = Math.min(remaining, shard.length - localPosition);
      destination.set(shard.subarray(localPosition, localPosition + span), outputOffset);
      outputOffset += span;
      remaining -= span;
      position += span;
      if (position === this.totalLength) position = 0;
    }
  }

  private nextBatchRandom(): DataBatch {
    const batch = this.batchRing[this.batchRingIdx];
    this.batchRingIdx = (this.batchRingIdx + 1) % this.batchRing.length;
    const inputs = batch.inputs.data as Int32Array;
    const targets = batch.targets.data as Int32Array;
    const maxStart = this.totalLength - this.blockSize;
    for (let item = 0; item < this.batchSize; item++) {
      const start = Math.floor(this.rng.next() * maxStart);
      const destination = item * this.blockSize;
      this.copyRange(inputs, destination, start, this.blockSize);
      this.copyRange(targets, destination, start + 1, this.blockSize);
    }
    return { inputs: { ...batch.inputs }, targets: { ...batch.targets } };
  }

  private nextBatchPacked(): DataBatch {
    const batch = this.batchRing[this.batchRingIdx];
    this.batchRingIdx = (this.batchRingIdx + 1) % this.batchRing.length;
    const inputs = batch.inputs.data as Int32Array;
    const targets = batch.targets.data as Int32Array;
    for (let item = 0; item < this.batchSize; item++) {
      const start = this.cursors![item];
      const destination = item * this.blockSize;
      this.copyRange(inputs, destination, start, this.blockSize);
      this.copyRange(targets, destination, (start + 1) % this.totalLength, this.blockSize);
      this.cursors![item] = (start + this.blockSize) % this.totalLength;
    }
    return { inputs: { ...batch.inputs }, targets: { ...batch.targets } };
  }

  get stepsPerEpoch(): number {
    return Math.floor(this.totalLength / (this.batchSize * this.blockSize));
  }

  get length(): number {
    return this.totalLength;
  }
}

// ── SFT (assistant-only) data pipeline ──────────────────────────────────────

/** The three atomic chat role markers used to render conversations. */
export const CHAT_USER_TOKEN = "<|user|>";
export const CHAT_ASSISTANT_TOKEN = "<|assistant|>";
export const CHAT_EOT_TOKEN = "<|end_of_text|>";

/** Token ids of the chat role markers for a specific tokenizer. */
export interface ChatSpecialIds {
  userId: number;
  assistantId: number;
  eotId: number;
}

/** One tokenized conversation + its per-token assistant-content mask. */
export interface SftExample {
  /** Full conversation token ids. */
  tokens: Int32Array;
  /** roleMask[i] === 1 iff tokens[i] is assistant-generated content (the content
   *  AFTER an <|assistant|> marker up to and including the terminating
   *  <|end_of_text|>). Role markers and user tokens are 0. */
  roleMask: Uint8Array;
  /** Compact [start,end) pairs for ordinary content in each assistant turn.
   *  EOS is excluded. This lets training emphasize the decision to BEGIN an
   *  answer without a second per-token array or accidentally boosting EOS. */
  assistantContentSpans?: Uint32Array;
  /** Token positions of assistant-turn EOS markers. Kept separate from ordinary
   *  content so response initiation and clean stopping can be weighted independently. */
  assistantEndPositions?: Uint32Array;
}

/** Optional SFT sampling and loss-weighting policy. Defaults preserve the
 * historical loader exactly: source order, binary assistant-token mask. */
export interface SftDataLoaderOptions {
  /** Deterministically reshuffle conversations at each epoch when provided. */
  shuffleSeed?: number;
  /** Give every conversation equal total loss weight after truncation. */
  balanceConversations?: boolean;
  /** Number of ordinary content tokens to emphasize at each assistant start. */
  startTokenCount?: number;
  /** Relative weight of emphasized start tokens before row normalization. */
  startTokenMultiplier?: number;
  /** Relative weight of each assistant-turn EOS before row normalization. */
  endTokenMultiplier?: number;
}

/**
 * Resolve the chat role-marker token ids for a tokenizer. Each marker MUST
 * encode to exactly one atomic token (the byte-level / bpe-chat tokenizers
 * reserve them as specials) — otherwise SFT masking is undefined, so we throw a
 * clear error rather than silently mis-mask.
 */
export function resolveChatSpecialIds(tokenizer: Tokenizer): ChatSpecialIds {
  const one = (marker: string): number => {
    const ids = tokenizer.encode(marker);
    if (ids.length !== 1) {
      throw new Error(
        `SFT requires a chat tokenizer whose "${marker}" marker is a single atomic token, ` +
        `but ${tokenizer.name} encodes it to ${ids.length} tokens. Use a bpe-chat / byte-level tokenizer with reserved specials.`,
      );
    }
    return ids[0];
  };
  return { userId: one(CHAT_USER_TOKEN), assistantId: one(CHAT_ASSISTANT_TOKEN), eotId: one(CHAT_EOT_TOKEN) };
}

/**
 * Build an {@link SftExample} from a rendered conversation string.
 *
 * The mask marks ONLY assistant response spans: content after an `<|assistant|>`
 * marker up to and including the next role boundary. The terminating
 * `<|end_of_text|>` of an assistant turn is INCLUDED (mask 1) so the model learns
 * to emit EOS; the `<|assistant|>`/`<|user|>` markers themselves are template
 * scaffolding (mask 0). Because the tokenizer encodes the markers atomically,
 * ordinary words like "assistant" appearing in content do NOT flip the state.
 */
export function buildSftExample(convText: string, tokenizer: Tokenizer, ids: ChatSpecialIds): SftExample {
  const tokens = tokenizer.encode(convText);
  const roleMask = new Uint8Array(tokens.length);
  const assistantContentSpans: number[] = [];
  const assistantEndPositions: number[] = [];
  let inAssistant = false;
  let contentStart = -1;
  const closeContentSpan = (end: number): void => {
    if (contentStart >= 0) assistantContentSpans.push(contentStart, end);
    contentStart = -1;
  };
  for (let i = 0; i < tokens.length; i++) {
    const id = tokens[i];
    if (id === ids.assistantId) {
      closeContentSpan(i);
      roleMask[i] = 0;        // marker itself is prompt scaffolding
      inAssistant = true;     // content that follows is assistant-generated
    } else if (id === ids.userId) {
      closeContentSpan(i);
      roleMask[i] = 0;
      inAssistant = false;
    } else if (id === ids.eotId) {
      closeContentSpan(i);    // EOS remains supervised but is not ordinary content
      roleMask[i] = inAssistant ? 1 : 0; // include the EOS that ends the turn
      if (inAssistant) assistantEndPositions.push(i);
      inAssistant = false;
    } else {
      roleMask[i] = inAssistant ? 1 : 0;
      if (inAssistant && contentStart < 0) contentStart = i;
    }
  }
  closeContentSpan(tokens.length);
  return {
    tokens,
    roleMask,
    assistantContentSpans: Uint32Array.from(assistantContentSpans),
    assistantEndPositions: Uint32Array.from(assistantEndPositions),
  };
}

/** Small deterministic PRNG used only to construct an epoch permutation. */
function shuffleState(seed: number, epoch: number): () => number {
  let state = (seed ^ Math.imul(epoch + 1, 0x9e3779b1)) >>> 0;
  if (state === 0) state = 0x6d2b79f5;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return state >>> 0;
  };
}

function epochPermutation(length: number, seed: number, epoch: number): Int32Array {
  const order = Int32Array.from({ length }, (_, index) => index);
  const next = shuffleState(seed, epoch);
  for (let i = length - 1; i > 0; i--) {
    const j = next() % (i + 1);
    const tmp = order[i];
    order[i] = order[j];
    order[j] = tmp;
  }
  return order;
}

/**
 * SFT batch source: ONE conversation per row (no packing in v1). Each row is
 * pad/truncated to blockSize. For a conversation with L tokens, positions
 * i∈[0,L-2] map to input=tokens[i], target=tokens[i+1], lossMask=roleMask[i+1]
 * (the mask of the PREDICTED token). Positions ≥ L-1 are padding: input/target
 * id 0, lossMask 0 — padding never contributes to the loss. A row whose whole
 * mask is zero (padding-only, or a turn with no assistant content) is legal and
 * produces no NaN because the masked-CE denominator is floored at 1.
 *
 * Deterministic: a monotonically advancing position walks either source order
 * or a seed-derived epoch permutation, so uninterrupted and resumed runs agree.
 */
export class SftDataLoader implements BatchSource {
  private examples: SftExample[];
  private batchSize: number;
  private blockSize: number;
  private position = 0;
  private batchRing: DataBatch[] = [];
  private batchRingIdx = 0;
  private totalTokens: number;
  private options: Required<Omit<SftDataLoaderOptions, "shuffleSeed">> & Pick<SftDataLoaderOptions, "shuffleSeed">;
  private permutationEpoch = -1;
  private permutation: Int32Array | null = null;

  constructor(examples: SftExample[], batchSize: number, blockSize: number, options: SftDataLoaderOptions = {}) {
    if (examples.length === 0) throw new RangeError("SftDataLoader needs at least one conversation");
    const startTokenCount = options.startTokenCount ?? 0;
    const startTokenMultiplier = options.startTokenMultiplier ?? 1;
    const endTokenMultiplier = options.endTokenMultiplier ?? 1;
    if (!Number.isSafeInteger(startTokenCount) || startTokenCount < 0) {
      throw new RangeError(`startTokenCount must be a non-negative integer: ${startTokenCount}`);
    }
    if (!Number.isFinite(startTokenMultiplier) || startTokenMultiplier <= 0) {
      throw new RangeError(`startTokenMultiplier must be finite and positive: ${startTokenMultiplier}`);
    }
    if (!Number.isFinite(endTokenMultiplier) || endTokenMultiplier <= 0) {
      throw new RangeError(`endTokenMultiplier must be finite and positive: ${endTokenMultiplier}`);
    }
    this.examples = examples;
    this.batchSize = batchSize;
    this.blockSize = blockSize;
    this.totalTokens = examples.reduce((acc, e) => acc + e.tokens.length, 0);
    this.options = {
      shuffleSeed: options.shuffleSeed,
      balanceConversations: options.balanceConversations ?? false,
      startTokenCount,
      startTokenMultiplier,
      endTokenMultiplier,
    };

    const elems = batchSize * blockSize;
    for (let i = 0; i < 2; i++) {
      this.batchRing.push({
        inputs: { shape: [batchSize, blockSize], dtype: "i32", data: new Int32Array(elems) },
        targets: { shape: [batchSize, blockSize], dtype: "i32", data: new Int32Array(elems) },
        lossMask: { shape: [batchSize, blockSize], dtype: "f32", data: new Float32Array(elems) },
      });
    }
  }

  nextBatch(): DataBatch {
    const B = this.batchSize;
    const T = this.blockSize;
    const batch = this.batchRing[this.batchRingIdx];
    this.batchRingIdx = (this.batchRingIdx + 1) % this.batchRing.length;
    const inputs = batch.inputs.data as Int32Array;
    const targets = batch.targets.data as Int32Array;
    const mask = batch.lossMask!.data as Float32Array;
    // Fresh zero-fill so short conversations leave clean padding (0 id, 0 mask).
    inputs.fill(0);
    targets.fill(0);
    mask.fill(0);

    for (let b = 0; b < B; b++) {
      const examplePosition = this.position + b;
      const ex = this.examples[this.exampleIndex(examplePosition)];
      const tok = ex.tokens;
      const rm = ex.roleMask;
      const contentSpans = ex.assistantContentSpans;
      const endPositions = ex.assistantEndPositions;
      const L = tok.length;
      const dst = b * T;
      const pairs = Math.min(T, L - 1); // usable (input,target) positions
      let rowWeight = 0;
      let spanOffset = 0;
      let endOffset = 0;
      for (let i = 0; i < pairs; i++) {
        inputs[dst + i] = tok[i];
        targets[dst + i] = tok[i + 1];
        if (rm[i + 1] === 0) continue;
        const targetIndex = i + 1;
        while (contentSpans && spanOffset < contentSpans.length && targetIndex >= contentSpans[spanOffset + 1]) {
          spanOffset += 2;
        }
        const isAnswerStart = !!contentSpans && spanOffset < contentSpans.length &&
          targetIndex >= contentSpans[spanOffset] &&
          targetIndex < Math.min(contentSpans[spanOffset + 1], contentSpans[spanOffset] + this.options.startTokenCount);
        while (endPositions && endOffset < endPositions.length && endPositions[endOffset] < targetIndex) endOffset++;
        const isAnswerEnd = !!endPositions && endOffset < endPositions.length && endPositions[endOffset] === targetIndex;
        const weight = isAnswerStart
          ? this.options.startTokenMultiplier
          : isAnswerEnd
            ? this.options.endTokenMultiplier
            : 1;
        mask[dst + i] = weight; // weight of the PREDICTED token
        rowWeight += weight;
      }
      if (this.options.balanceConversations && rowWeight > 0) {
        for (let i = 0; i < pairs; i++) mask[dst + i] /= rowWeight;
      }
    }
    this.position += B;

    return {
      inputs: { ...batch.inputs },
      targets: { ...batch.targets },
      lossMask: { ...batch.lossMask! },
    };
  }

  seekBatches(count: number): void {
    if (!Number.isSafeInteger(count) || count < 0) throw new RangeError(`batch count must be a non-negative integer: ${count}`);
    const position = count * this.batchSize;
    if (!Number.isSafeInteger(position)) throw new RangeError(`batch position exceeds safe integer range: ${position}`);
    this.position = position;
    this.batchRingIdx = count % this.batchRing.length;
    this.permutationEpoch = -1;
    this.permutation = null;
  }

  private exampleIndex(position: number): number {
    const length = this.examples.length;
    const offset = position % length;
    if (this.options.shuffleSeed === undefined) return offset;
    const epoch = Math.floor(position / length);
    if (this.permutationEpoch !== epoch || this.permutation === null) {
      this.permutation = epochPermutation(length, this.options.shuffleSeed, epoch);
      this.permutationEpoch = epoch;
    }
    return this.permutation[offset];
  }

  get stepsPerEpoch(): number {
    return Math.max(1, Math.ceil(this.examples.length / this.batchSize));
  }

  get length(): number {
    return this.totalTokens;
  }

  get conversationCount(): number {
    return this.examples.length;
  }
}

// ── RCR-UL rollout data pipeline ───────────────────────────────────────────

/** One frozen failed-rollout training trajectory. `penaltyTargetPositions`
 * indexes tokens (not input rows): position p penalizes prediction of tokens[p]
 * from tokens[0..p). Position zero is therefore invalid. */
export interface RcrUlExample {
  stableId: string;
  positiveConversationSha256: string;
  tokens: Int32Array;
  penaltyTargetPositions: Uint32Array;
}

interface RcrUlJsonRecord {
  schema?: unknown;
  stable_id?: unknown;
  positive_conversation_sha256?: unknown;
  token_ids?: unknown;
  penalty_target_positions?: unknown;
}

/**
 * Batch source for rollout-conditioned repetition unlikelihood.
 *
 * The file already fixes the exact token trajectory and mechanical penalty
 * positions. This loader performs no semantic judgment and no truncation: a
 * record longer than blockSize+1 is rejected, preserving the frozen exposure
 * rather than silently changing its context. Its permutation is identical to
 * SftDataLoader when both receive the same length, seed, batch size, and seek.
 */
export class RcrUlDataLoader implements BatchSource {
  private examples: RcrUlExample[];
  private batchSize: number;
  private blockSize: number;
  private shuffleSeed?: number;
  private position = 0;
  private batchRing: DataBatch[] = [];
  private batchRingIdx = 0;
  private permutationEpoch = -1;
  private permutation: Int32Array | null = null;
  private totalTokens: number;
  private totalPenaltyPositions: number;

  constructor(examples: RcrUlExample[], batchSize: number, blockSize: number, shuffleSeed?: number) {
    if (examples.length === 0) throw new RangeError("RcrUlDataLoader needs at least one rollout");
    if (!Number.isSafeInteger(batchSize) || batchSize <= 0) throw new RangeError(`invalid batchSize: ${batchSize}`);
    if (!Number.isSafeInteger(blockSize) || blockSize <= 0) throw new RangeError(`invalid blockSize: ${blockSize}`);
    for (const ex of examples) {
      if (ex.tokens.length > blockSize + 1) {
        throw new RangeError(
          `RCR-UL rollout ${ex.stableId} has ${ex.tokens.length} tokens; maximum exact trajectory is ${blockSize + 1}`,
        );
      }
    }
    this.examples = examples;
    this.batchSize = batchSize;
    this.blockSize = blockSize;
    this.shuffleSeed = shuffleSeed;
    this.totalTokens = examples.reduce((sum, ex) => sum + ex.tokens.length, 0);
    this.totalPenaltyPositions = examples.reduce((sum, ex) => sum + ex.penaltyTargetPositions.length, 0);

    const elems = batchSize * blockSize;
    for (let i = 0; i < 2; i++) {
      this.batchRing.push({
        inputs: { shape: [batchSize, blockSize], dtype: "i32", data: new Int32Array(elems) },
        targets: { shape: [batchSize, blockSize], dtype: "i32", data: new Int32Array(elems) },
        lossMask: { shape: [batchSize, blockSize], dtype: "f32", data: new Float32Array(elems) },
      });
    }
  }

  nextBatch(): DataBatch {
    const B = this.batchSize;
    const T = this.blockSize;
    const batch = this.batchRing[this.batchRingIdx];
    this.batchRingIdx = (this.batchRingIdx + 1) % this.batchRing.length;
    const inputs = batch.inputs.data as Int32Array;
    const targets = batch.targets.data as Int32Array;
    const mask = batch.lossMask!.data as Float32Array;
    inputs.fill(0);
    targets.fill(0);
    mask.fill(0);

    for (let b = 0; b < B; b++) {
      const ex = this.examples[this.exampleIndex(this.position + b)];
      const dst = b * T;
      const pairs = ex.tokens.length - 1;
      for (let i = 0; i < pairs; i++) {
        inputs[dst + i] = ex.tokens[i];
        targets[dst + i] = ex.tokens[i + 1];
      }
      for (const targetPosition of ex.penaltyTargetPositions) {
        mask[dst + targetPosition - 1] = 1;
      }
    }
    this.position += B;
    return {
      inputs: { ...batch.inputs },
      targets: { ...batch.targets },
      lossMask: { ...batch.lossMask! },
    };
  }

  seekBatches(count: number): void {
    if (!Number.isSafeInteger(count) || count < 0) throw new RangeError(`batch count must be a non-negative integer: ${count}`);
    const position = count * this.batchSize;
    if (!Number.isSafeInteger(position)) throw new RangeError(`batch position exceeds safe integer range: ${position}`);
    this.position = position;
    this.batchRingIdx = count % this.batchRing.length;
    this.permutationEpoch = -1;
    this.permutation = null;
  }

  private exampleIndex(position: number): number {
    const length = this.examples.length;
    const offset = position % length;
    if (this.shuffleSeed === undefined) return offset;
    const epoch = Math.floor(position / length);
    if (this.permutationEpoch !== epoch || this.permutation === null) {
      this.permutation = epochPermutation(length, this.shuffleSeed, epoch);
      this.permutationEpoch = epoch;
    }
    return this.permutation[offset];
  }

  get stepsPerEpoch(): number {
    return Math.max(1, Math.ceil(this.examples.length / this.batchSize));
  }

  get length(): number {
    return this.totalTokens;
  }

  get conversationCount(): number {
    return this.examples.length;
  }

  get penaltyPositionCount(): number {
    return this.totalPenaltyPositions;
  }
}

/** Stream and fail-closed validate an immutable RCR-UL JSONL cohort. */
export async function loadRcrUlExamples(path: string): Promise<RcrUlExample[]> {
  const fs = await import("node:fs");
  const readline = await import("node:readline");
  const examples: RcrUlExample[] = [];
  const stableIds = new Set<string>();
  const rl = readline.createInterface({
    input: fs.createReadStream(path, { encoding: "utf-8" }),
    crlfDelay: Infinity,
  });
  let lineNumber = 0;
  for await (const rawLine of rl) {
    lineNumber++;
    const line = rawLine.trim();
    if (line.length === 0) continue;
    let record: RcrUlJsonRecord;
    try {
      record = JSON.parse(line) as RcrUlJsonRecord;
    } catch (error) {
      throw new Error(`RCR-UL ${path}:${lineNumber} is not valid JSON: ${(error as Error).message}`);
    }
    if (record.schema !== "alpha-rcr-ul-example-v1") {
      throw new Error(`RCR-UL ${path}:${lineNumber} has unexpected schema ${String(record.schema)}`);
    }
    if (typeof record.stable_id !== "string" || record.stable_id.length === 0) {
      throw new Error(`RCR-UL ${path}:${lineNumber} has invalid stable_id`);
    }
    if (stableIds.has(record.stable_id)) {
      throw new Error(`RCR-UL ${path}:${lineNumber} duplicates stable_id ${record.stable_id}`);
    }
    if (typeof record.positive_conversation_sha256 !== "string" || !/^[0-9a-f]{64}$/.test(record.positive_conversation_sha256)) {
      throw new Error(`RCR-UL ${path}:${lineNumber} has invalid positive_conversation_sha256`);
    }
    if (!Array.isArray(record.token_ids) || record.token_ids.length < 2 ||
        record.token_ids.some((value) => !Number.isSafeInteger(value) || value < 0)) {
      throw new Error(`RCR-UL ${path}:${lineNumber} has invalid token_ids`);
    }
    if (!Array.isArray(record.penalty_target_positions) ||
        record.penalty_target_positions.some((value) => !Number.isSafeInteger(value) || value <= 0 || value >= (record.token_ids as unknown[]).length)) {
      throw new Error(`RCR-UL ${path}:${lineNumber} has invalid penalty_target_positions`);
    }
    const positions = record.penalty_target_positions as number[];
    for (let i = 1; i < positions.length; i++) {
      if (positions[i] <= positions[i - 1]) {
        throw new Error(`RCR-UL ${path}:${lineNumber} penalty_target_positions must be sorted and unique`);
      }
    }
    stableIds.add(record.stable_id);
    examples.push({
      stableId: record.stable_id,
      positiveConversationSha256: record.positive_conversation_sha256,
      tokens: Int32Array.from(record.token_ids as number[]),
      penaltyTargetPositions: Uint32Array.from(positions),
    });
  }
  if (examples.length === 0) throw new Error(`RCR-UL cohort is empty: ${path}`);
  return examples;
}

/** Hash the exact non-empty rendered conversations consumed by SFT, in file
 * order. RCR-UL uses these identities to fail closed if a rollout trajectory
 * is paired with anything other than the positive conversation from which it
 * was frozen. */
export async function loadSftConversationSha256s(path: string): Promise<string[]> {
  const fs = await import("node:fs");
  const readline = await import("node:readline");
  const crypto = await import("node:crypto");
  const hashes: string[] = [];
  const rl = readline.createInterface({
    input: fs.createReadStream(path, { encoding: "utf-8" }),
    crlfDelay: Infinity,
  });
  for await (const rawLine of rl) {
    const line = rawLine.trim();
    if (line.length === 0) continue;
    hashes.push(crypto.createHash("sha256").update(line, "utf8").digest("hex"));
  }
  return hashes;
}

/**
 * Load conversations from a chat corpus file (one rendered conversation per
 * line) and tokenize each into an {@link SftExample}. Streams line-by-line so
 * arbitrarily large SFT files stay within the JS heap. Blank lines are skipped.
 */
export async function loadSftExamples(path: string, tokenizer: Tokenizer): Promise<SftExample[]> {
  const fs = await import("node:fs");
  const readline = await import("node:readline");
  const ids = resolveChatSpecialIds(tokenizer);
  const examples: SftExample[] = [];
  const rl = readline.createInterface({
    input: fs.createReadStream(path, { encoding: "utf-8" }),
    crlfDelay: Infinity,
  });
  for await (const rawLine of rl) {
    const line = rawLine.trim();
    if (line.length === 0) continue;
    const ex = buildSftExample(line, tokenizer, ids);
    if (ex.tokens.length >= 2) examples.push(ex); // need at least one (input,target) pair
  }
  return examples;
}

/** FNV-1a 32-bit hash of a string (deterministic doc-aware split key). */
function fnv1a32(input: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/**
 * Deterministic doc-aware train/val split over conversations: each conversation
 * is assigned to val iff hash(seed:index) < valFraction. Same conversation →
 * same side across runs (reproducible, no cross-contamination of a doc).
 */
export function splitSftExamples(
  examples: SftExample[],
  valFraction: number,
  seed: number,
): { train: SftExample[]; val: SftExample[] } {
  const train: SftExample[] = [];
  const val: SftExample[] = [];
  for (let i = 0; i < examples.length; i++) {
    const takeVal = valFraction > 0 && (fnv1a32(`${seed}:${i}`) / 0x1_0000_0000) < valFraction;
    (takeVal ? val : train).push(examples[i]);
  }
  return { train, val };
}

/** V8 max string length (~512MB for Latin-1, ~256MB for two-byte). Stay well below. */
const MAX_STRING_BYTES = 30 * 1024 * 1024; // 30MB — BPE tokenizer hits V8 array limits on larger strings

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
 * Cache key includes tokenizer identity when supplied. The 16-byte header stores
 * source mtime + byte size so stale legacy/equally named inputs are invalidated.
 */
export async function loadOrCacheTokens(
  dataPath: string,
  tokenizer: Tokenizer,
  range?: { startByte: number; endByte: number },
  cacheIdentity?: string,
): Promise<Int32Array> {
  const fs = await import("node:fs/promises");
  const pathMod = await import("node:path");

  const suffix = range ? `.${range.startByte}-${range.endByte}` : "";
  const identity = cacheIdentity
    ? `-${cacheIdentity.toLowerCase().replace(/[^a-z0-9]/g, "").slice(0, 24)}`
    : "";
  const cacheFile = `${dataPath}.${tokenizer.name}-${tokenizer.vocabSize}${identity}${suffix}.tokens`;
  const srcStat = await fs.stat(dataPath);

  // Try loading cache
  try {
    const cacheStat = await fs.stat(cacheFile);
    if (cacheStat.size > 16) {
      const handle = await fs.open(cacheFile, "r");
      try {
        // Header: source mtime + byte size as two Float64 values.
        const headerBuf = Buffer.alloc(16);
        const headerRead = await handle.read(headerBuf, 0, 16, 0);
        if (headerRead.bytesRead !== 16) throw new Error("short token-cache header");
        const cachedMtime = headerBuf.readDoubleBE(0);
        const cachedSize = headerBuf.readDoubleBE(8);
        if (Math.abs(cachedMtime - srcStat.mtimeMs) < 1 && cachedSize === srcStat.size) {
          const tokenBytes = cacheStat.size - 16;
          if (tokenBytes % 4 !== 0) throw new Error(`invalid token cache byte length: ${tokenBytes}`);
          const dataBuf = Buffer.alloc(tokenBytes);
          const chunkBytes = 64 * 1024 * 1024;
          for (let offset = 0; offset < tokenBytes;) {
            const requested = Math.min(chunkBytes, tokenBytes - offset);
            const { bytesRead } = await handle.read(dataBuf, offset, requested, 16 + offset);
            if (bytesRead === 0) throw new Error(`unexpected EOF in token cache at byte ${offset}`);
            offset += bytesRead;
          }
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
  const cacheTmp = `${cacheFile}.tmp-${process.pid}-${Date.now()}`;
  try {
    const header = Buffer.alloc(16);
    header.writeDoubleBE(srcStat.mtimeMs, 0);
    header.writeDoubleBE(srcStat.size, 8);
    const tokenBuf = Buffer.from(tokens.buffer, tokens.byteOffset, tokens.byteLength);
    const handle = await fs.open(cacheTmp, "wx");
    try {
      const headerWrite = await handle.write(header, 0, header.length, 0);
      if (headerWrite.bytesWritten !== header.length) throw new Error("short token-cache header write");
      const chunkBytes = 64 * 1024 * 1024;
      for (let offset = 0; offset < tokenBuf.length;) {
        const requested = Math.min(chunkBytes, tokenBuf.length - offset);
        const { bytesWritten } = await handle.write(tokenBuf, offset, requested, header.length + offset);
        if (bytesWritten === 0) throw new Error(`zero-byte token-cache write at byte ${offset}`);
        offset += bytesWritten;
      }
      await handle.sync();
    } finally {
      await handle.close();
    }
    await fs.rename(cacheTmp, cacheFile);
    console.log(`  Cached tokens to ${pathMod.basename(cacheFile)} (${(tokenBuf.byteLength / 1024 / 1024).toFixed(0)}MB)`);
  } catch (e) {
    await fs.unlink(cacheTmp).catch(() => {});
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
