import type { Backend, ModelConfig, Optimizer, TensorData } from "@alpha/core";
import { shapeSize, SeededRng } from "@alpha/core";
import { collectParams, initGPT, type GPTParams } from "@alpha/model";

export interface ParamTransferStats {
  exactCopies: number;
  partialCopies: number;
  initializedFresh: number;
}

export interface OptimizerCarryStats {
  copiedBuffers: number;
  partialBuffers: number;
  freshBuffers: number;
  carriedStep: number;
}

function sourceCandidatesForParamName(name: string): string[] {
  const out = [name];
  if (name.includes(".mlp.fc_gate")) {
    out.push(name.replace(".mlp.fc_gate", ".mlp.fc1"));
  } else if (name.includes(".mlp.fc_up")) {
    out.push(name.replace(".mlp.fc_up", ".mlp.fc_gate"));
    out.push(name.replace(".mlp.fc_up", ".mlp.fc1"));
  } else if (name.includes(".mlp.fc_proj")) {
    out.push(name.replace(".mlp.fc_proj", ".mlp.fc2"));
  } else if (name.includes(".mlp.fc1")) {
    out.push(name.replace(".mlp.fc1", ".mlp.fc_gate"));
    out.push(name.replace(".mlp.fc1", ".mlp.fc_up"));
  } else if (name.includes(".mlp.fc2")) {
    out.push(name.replace(".mlp.fc2", ".mlp.fc_proj"));
  }
  // dedupe while preserving order
  return [...new Set(out)];
}

function copyFlatOverlap(dst: Float32Array, src: ArrayLike<number>): "exact" | "partial" {
  const n = Math.min(dst.length, src.length);
  for (let i = 0; i < n; i++) dst[i] = src[i];
  if (dst.length !== src.length) {
    for (let i = n; i < dst.length; i++) dst[i] = 0;
    return "partial";
  }
  return "exact";
}

function copy2DOverlap(dst: Float32Array, dstShape: readonly number[], src: ArrayLike<number>, srcShape: readonly number[]): "exact" | "partial" {
  const [dr, dc] = dstShape;
  const [sr, sc] = srcShape;
  const r = Math.min(dr, sr);
  const c = Math.min(dc, sc);
  dst.fill(0);
  for (let i = 0; i < r; i++) {
    const doff = i * dc;
    const soff = i * sc;
    for (let j = 0; j < c; j++) {
      dst[doff + j] = src[soff + j];
    }
  }
  return dr === sr && dc === sc ? "exact" : "partial";
}

function copyTensorOverlap(dst: TensorData, src: TensorData): "exact" | "partial" | "none" {
  if (dst.dtype !== "f32" || src.dtype !== "f32") return "none";
  const d = dst.data as Float32Array;
  const s = src.data as Float32Array;
  if (dst.shape.length === 2 && src.shape.length === 2) {
    return copy2DOverlap(d, dst.shape, s, src.shape);
  }
  if (dst.shape.length === 1 && src.shape.length === 1) {
    return copyFlatOverlap(d, s);
  }
  if (shapeSize(dst.shape) > 0 && shapeSize(src.shape) > 0) {
    return copyFlatOverlap(d, s);
  }
  return "none";
}

function chooseSourceParamName(targetName: string, sourceNames: Iterable<string>): string | null {
  const sourceSet = sourceNames instanceof Set ? sourceNames : new Set(sourceNames);
  for (const cand of sourceCandidatesForParamName(targetName)) {
    if (sourceSet.has(cand)) return cand;
  }
  return null;
}

function cloneTensorDataLikeFlat(size: number): TensorData {
  return { shape: [size], dtype: "f32", data: new Float32Array(size) };
}

export function initGPTWithTransferredWeights(
  config: ModelConfig,
  backend: Backend,
  rng: SeededRng,
  previous: GPTParams | null,
): { params: GPTParams; stats: ParamTransferStats } {
  const next = initGPT(config, backend, rng);
  const stats: ParamTransferStats = { exactCopies: 0, partialCopies: 0, initializedFresh: 0 };
  if (!previous) {
    stats.initializedFresh = collectParams(next).size;
    return { params: next, stats };
  }

  const prevMap = collectParams(previous);
  const nextMap = collectParams(next);
  const prevNames = new Set(prevMap.keys());

  for (const [name, dstVar] of nextMap) {
    const srcName = chooseSourceParamName(name, prevNames);
    if (!srcName) {
      stats.initializedFresh++;
      continue;
    }
    const srcVar = prevMap.get(srcName);
    if (!srcVar) {
      stats.initializedFresh++;
      continue;
    }
    const copied = copyTensorOverlap(dstVar.data, srcVar.data);
    if (copied === "exact") stats.exactCopies++;
    else if (copied === "partial") stats.partialCopies++;
    else stats.initializedFresh++;
  }

  return { params: next, stats };
}

export function carryOptimizerStateAcrossSwitch(
  optimizer: Optimizer,
  previous: GPTParams,
  next: GPTParams,
): OptimizerCarryStats {
  const prevState = optimizer.stateDict();
  const prevParamMap = collectParams(previous);
  const nextParamMap = collectParams(next);
  const prevParamNames = new Set(prevParamMap.keys());
  const nextBuffers = new Map<string, TensorData>();

  let copiedBuffers = 0;
  let partialBuffers = 0;
  let freshBuffers = 0;

  for (const [name, dstVar] of nextParamMap) {
    const targetSize = shapeSize(dstVar.data.shape);
    const srcName = chooseSourceParamName(name, prevParamNames);

    for (const suffix of [".m", ".v"] as const) {
      const dstKey = `${name}${suffix}`;
      const dstTd = cloneTensorDataLikeFlat(targetSize);
      if (srcName) {
        const srcKey = `${srcName}${suffix}`;
        const srcTd = prevState.buffers.get(srcKey);
        if (srcTd && srcTd.dtype === "f32") {
          const result = copyFlatOverlap(dstTd.data as Float32Array, srcTd.data as Float32Array);
          if (result === "exact") copiedBuffers++;
          else partialBuffers++;
          nextBuffers.set(dstKey, dstTd);
          continue;
        }
      }
      freshBuffers++;
      nextBuffers.set(dstKey, dstTd);
    }
  }

  optimizer.loadStateDict({ step: prevState.step, buffers: nextBuffers });
  return { copiedBuffers, partialBuffers, freshBuffers, carriedStep: prevState.step };
}

export class ConsensusFusionShadow {
  private shadow = new Map<string, Float32Array>();

  constructor(params?: GPTParams) {
    if (params) this.rebind(params);
  }

  rebind(params: GPTParams): void {
    const nextShadow = new Map<string, Float32Array>();
    const paramsMap = collectParams(params);
    const prevNames = new Set(this.shadow.keys());
    for (const [name, v] of paramsMap) {
      const arr = v.data.data as Float32Array;
      const dst = new Float32Array(arr.length);
      const srcName = chooseSourceParamName(name, prevNames);
      if (srcName) {
        const src = this.shadow.get(srcName)!;
        copyFlatOverlap(dst, src);
      } else {
        dst.set(arr);
      }
      nextShadow.set(name, dst);
    }
    this.shadow = nextShadow;
  }

  step(params: GPTParams, fusionAlpha: number, shadowEma: number): void {
    if (fusionAlpha <= 0 && shadowEma <= 0) return;
    const map = collectParams(params);
    for (const [name, v] of map) {
      if (v.data.dtype !== "f32") continue;
      const p = v.data.data as Float32Array;
      let s = this.shadow.get(name);
      if (!s || s.length !== p.length) {
        s = new Float32Array(p);
        this.shadow.set(name, s);
      }
      for (let i = 0; i < p.length; i++) {
        s[i] = s[i] + shadowEma * (p[i] - s[i]);
        p[i] = p[i] + fusionAlpha * (s[i] - p[i]);
      }
    }
  }
}

