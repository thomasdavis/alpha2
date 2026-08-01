import type { TensorData } from "@alpha/core";
import type { AlphaLensAdapter } from "./adapter.js";
import type { CopiedTensor, CapturedForward } from "./types.js";
import type { SafeTensorValue } from "./safetensors.js";

export interface RankedToken {
  readonly id: number;
  readonly text: string;
  readonly logit: number;
  readonly rank: number;
}

export function applyDenseTransport(source: CopiedTensor, transport: SafeTensorValue): CopiedTensor {
  if (source.shape.length !== 3) throw new Error(`source must be BTD, got [${source.shape}]`);
  const [batch, time, sourceWidth] = source.shape;
  const [targetWidth, transportSourceWidth] = transport.shape;
  if (sourceWidth !== transportSourceWidth) throw new Error(`transport input width ${transportSourceWidth} != source width ${sourceWidth}`);
  const data = new Float32Array(batch * time * targetWidth);
  for (let row = 0; row < batch * time; row++) {
    const sourceBase = row * sourceWidth;
    const outputBase = row * targetWidth;
    for (let output = 0; output < targetWidth; output++) {
      const matrixBase = output * sourceWidth;
      let value = 0;
      for (let input = 0; input < sourceWidth; input++) value += source.data[sourceBase + input] * transport.data[matrixBase + input];
      data[outputBase + output] = value;
    }
  }
  return { shape: [batch, time, targetWidth], dtype: "f32", data };
}

export function siteReadout(
  adapter: AlphaLensAdapter,
  capture: CapturedForward,
  siteId: string,
  transport?: SafeTensorValue,
): CopiedTensor {
  const site = capture.sites.get(siteId);
  if (!site) throw new Error(`site ${siteId} was not captured`);
  const basis: TensorData = transport
    ? applyDenseTransport(adapter.copyTensor(site.data), transport)
    : site.data;
  return adapter.exactFinalDecode(basis);
}

export function tensorReadout(
  adapter: AlphaLensAdapter,
  siteTensor: CopiedTensor,
  transport?: SafeTensorValue,
): CopiedTensor {
  return adapter.exactFinalDecode(transport ? applyDenseTransport(siteTensor, transport) : siteTensor);
}

export function rankLogitRow(
  logits: Float32Array,
  offset: number,
  vocabSize: number,
  adapter: AlphaLensAdapter,
  topK: number,
  pinnedIds: readonly number[] | null,
  filterNonWordTokens: boolean,
): { top: RankedToken[]; pinned?: RankedToken[] } {
  const ids = Array.from({ length: vocabSize }, (_, id) => id);
  ids.sort((a, b) => logits[offset + b] - logits[offset + a] || a - b);
  const eligible = filterNonWordTokens
    ? ids.filter((id) => /[\p{L}\p{N}]/u.test(adapter.decode([id])))
    : ids;
  const top = eligible.slice(0, topK).map((id, index) => ({
    id,
    text: adapter.tokenString(id),
    logit: logits[offset + id],
    rank: index + 1,
  }));
  if (!pinnedIds) return { top };
  const rankById = new Int32Array(vocabSize);
  ids.forEach((id, index) => { rankById[id] = index + 1; });
  return {
    top,
    pinned: pinnedIds.map((id) => ({ id, text: adapter.tokenString(id), logit: logits[offset + id], rank: rankById[id] })),
  };
}

export function greedyToken(logits: Float32Array, offset: number, vocabSize: number): number {
  let best = 0;
  for (let id = 1; id < vocabSize; id++) {
    if (logits[offset + id] > logits[offset + best]) best = id;
  }
  return best;
}
