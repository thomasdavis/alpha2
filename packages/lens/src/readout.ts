import type { TensorData } from "@alpha/core";
import type { AlphaLensAdapter } from "./adapter.js";
import type { CopiedTensor, CapturedForward } from "./types.js";
import type { SafeTensorValue } from "./safetensors.js";

export interface RankedToken {
  readonly id: number;
  readonly text: string;
  readonly bytes_base64?: string;
  readonly logit: number;
  readonly rank: number;
}

export interface AffineCentering {
  readonly sourceMean: SafeTensorValue;
  readonly targetMean: SafeTensorValue;
}

export function applyDenseTransport(
  source: CopiedTensor,
  transport: SafeTensorValue,
  centering?: AffineCentering,
): CopiedTensor {
  if (source.shape.length !== 3) throw new Error(`source must be BTD, got [${source.shape}]`);
  const [batch, time, sourceWidth] = source.shape;
  const [targetWidth, transportSourceWidth] = transport.shape;
  if (sourceWidth !== transportSourceWidth) throw new Error(`transport input width ${transportSourceWidth} != source width ${sourceWidth}`);
  if (centering) {
    if (centering.sourceMean.shape.length !== 1 || centering.sourceMean.shape[0] !== sourceWidth) {
      throw new Error(`source mean shape [${centering.sourceMean.shape}] != [${sourceWidth}]`);
    }
    if (centering.targetMean.shape.length !== 1 || centering.targetMean.shape[0] !== targetWidth) {
      throw new Error(`target mean shape [${centering.targetMean.shape}] != [${targetWidth}]`);
    }
  }
  const data = new Float32Array(batch * time * targetWidth);
  for (let row = 0; row < batch * time; row++) {
    const sourceBase = row * sourceWidth;
    const outputBase = row * targetWidth;
    for (let output = 0; output < targetWidth; output++) {
      const matrixBase = output * sourceWidth;
      let value = centering?.targetMean.data[output] ?? 0;
      for (let input = 0; input < sourceWidth; input++) {
        const centeredSource = source.data[sourceBase + input] - (centering?.sourceMean.data[input] ?? 0);
        value += centeredSource * transport.data[matrixBase + input];
      }
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
  centering?: AffineCentering,
): CopiedTensor {
  const site = capture.sites.get(siteId);
  if (!site) throw new Error(`site ${siteId} was not captured`);
  const basis: TensorData = transport
    ? applyDenseTransport(adapter.copyTensor(site.data), transport, centering)
    : site.data;
  return adapter.exactFinalDecode(basis);
}

export function tensorReadout(
  adapter: AlphaLensAdapter,
  siteTensor: CopiedTensor,
  transport?: SafeTensorValue,
  centering?: AffineCentering,
): CopiedTensor {
  return adapter.exactFinalDecode(transport ? applyDenseTransport(siteTensor, transport, centering) : siteTensor);
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
    ...adapter.tokenDescriptor(id),
    logit: logits[offset + id],
    rank: index + 1,
  }));
  if (!pinnedIds) return { top };
  const rankById = new Int32Array(vocabSize);
  ids.forEach((id, index) => { rankById[id] = index + 1; });
  return {
    top,
    pinned: pinnedIds.map((id) => ({
      ...adapter.tokenDescriptor(id),
      logit: logits[offset + id],
      rank: rankById[id],
    })),
  };
}

export function greedyToken(logits: Float32Array, offset: number, vocabSize: number): number {
  let best = 0;
  for (let id = 1; id < vocabSize; id++) {
    if (logits[offset + id] > logits[offset + best]) best = id;
  }
  return best;
}
