import { mkdir, readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { join } from "node:path";
import type { TensorData } from "@alpha/core";
import { AlphaLensAdapter } from "./adapter.js";
import { loadLensPrompts } from "./prompts.js";
import { readLensSafetensors, writeLensSafetensors, type SafeDtype, type SafeTensorValue } from "./safetensors.js";

export interface LensFitOptions {
  readonly checkpoint: string;
  readonly prompts: string;
  readonly samples: number;
  readonly maxSeqLen: number;
  readonly skipFirst: number;
  readonly dimBatch: number;
  readonly sourceSites?: readonly string[];
  readonly targetSite: "decoder.final.post";
  readonly dtype: "float32" | "float16";
  readonly checkpointEvery: number;
  readonly resume: boolean;
  readonly output: string;
  readonly backend?: string;
  readonly corpusName?: string;
  readonly corpusDatasetId?: string;
  readonly corpusRevision?: string;
  readonly corpusSplit?: string;
  readonly corpusVisibility?: "public" | "synthetic" | "proprietary" | "private";
  readonly onProgress?: (message: string) => void;
}

interface FitStateFile {
  readonly version: 1;
  readonly options_fingerprint: string;
  readonly completed_prompts: number;
  readonly valid_prompts: number;
  readonly valid_even_prompts: number;
  readonly valid_odd_prompts: number;
  readonly token_count: number;
  readonly skipped: readonly { index: number; reason: string }[];
  readonly elapsed_ms: number;
}

export interface LensFitResult {
  readonly adapter: AlphaLensAdapter;
  readonly report: Record<string, unknown>;
  readonly transports: ReadonlyMap<string, SafeTensorValue>;
  readonly splitEven: ReadonlyMap<string, SafeTensorValue>;
  readonly splitOdd: ReadonlyMap<string, SafeTensorValue>;
}

export async function fitJacobianLens(options: LensFitOptions): Promise<LensFitResult> {
  validateFitOptions(options);
  const log = options.onProgress ?? (() => {});
  const adapter = await AlphaLensAdapter.load({ checkpoint: options.checkpoint, backend: options.backend });
  if (options.targetSite !== adapter.description.targetSite.id) throw new Error(`unsupported target site ${options.targetSite}`);
  const sourceSites = options.sourceSites?.length
    ? [...options.sourceSites]
    : adapter.description.sites.map((site) => site.id);
  const known = new Set(adapter.description.sites.map((site) => site.id));
  for (const site of sourceSites) if (!known.has(site)) throw new Error(`unsupported source site ${site}`);

  const loaded = await loadLensPrompts(options.prompts);
  if (loaded.prompts.length < options.samples) {
    throw new Error(`requested ${options.samples} samples but prompt file contains ${loaded.prompts.length}`);
  }
  const width = adapter.description.targetSite.width;
  const sums = makeMatrices(sourceSites, width);
  const evenSums = makeMatrices(sourceSites, width);
  const oddSums = makeMatrices(sourceSites, width);
  const optionsFingerprint = fitOptionsFingerprint(options, sourceSites, loaded.fingerprint, adapter.description.weightsFingerprint);
  let state: FitStateFile = {
    version: 1,
    options_fingerprint: optionsFingerprint,
    completed_prompts: 0,
    valid_prompts: 0,
    valid_even_prompts: 0,
    valid_odd_prompts: 0,
    token_count: 0,
    skipped: [],
    elapsed_ms: 0,
  };
  await mkdir(options.output, { recursive: true });
  if (options.resume) state = await restoreFitState(options.output, optionsFingerprint, sums, evenSums, oddSums);

  for (let promptIndex = state.completed_prompts; promptIndex < options.samples; promptIndex++) {
    const promptStarted = performance.now();
    const prompt = loaded.prompts[promptIndex];
    let ids = adapter.encode(prompt);
    if (ids.length > options.maxSeqLen) ids = ids.slice(0, options.maxSeqLen);
    if (ids.length <= options.skipFirst + 1) {
      state = { ...state, completed_prompts: promptIndex + 1, skipped: [...state.skipped, { index: promptIndex, reason: `only ${ids.length} tokens after truncation` }] };
      continue;
    }

    const capture = adapter.forwardCapture(ids, sourceSites, options.dimBatch);
    try {
      for (let dimensionStart = 0; dimensionStart < width; dimensionStart += options.dimBatch) {
        const activeDimensions = Math.min(options.dimBatch, width - dimensionStart);
        const cotangent = buildCausalCotangent(options.dimBatch, ids.length, width, dimensionStart, activeDimensions, options.skipFirst);
        const retainGraph = dimensionStart + options.dimBatch < width;
        const gradients = adapter.vjp(capture, cotangent, sourceSites, retainGraph);
        for (const siteId of sourceSites) {
          const gradient = gradients.get(siteId)!;
          accumulateRows(
            sums.get(siteId)!.data,
            gradient.data,
            options.dimBatch,
            ids.length,
            width,
            dimensionStart,
            activeDimensions,
            options.skipFirst,
          );
          const split = promptIndex % 2 === 0 ? evenSums : oddSums;
          accumulateRows(
            split.get(siteId)!.data,
            gradient.data,
            options.dimBatch,
            ids.length,
            width,
            dimensionStart,
            activeDimensions,
            options.skipFirst,
          );
        }
      }
    } finally {
      adapter.disposeCapture(capture);
    }

    state = {
      ...state,
      completed_prompts: promptIndex + 1,
      valid_prompts: state.valid_prompts + 1,
      valid_even_prompts: state.valid_even_prompts + (promptIndex % 2 === 0 ? 1 : 0),
      valid_odd_prompts: state.valid_odd_prompts + (promptIndex % 2 === 1 ? 1 : 0),
      token_count: state.token_count + ids.length,
      elapsed_ms: state.elapsed_ms + (performance.now() - promptStarted),
    };
    log(`lens fit ${state.completed_prompts}/${options.samples}: ${ids.length} tokens, ${state.valid_prompts} valid`);
    if (state.completed_prompts % options.checkpointEvery === 0) {
      await saveFitState(options.output, state, sums, evenSums, oddSums);
    }
  }
  if (state.valid_prompts === 0) throw new Error("no valid prompts remained after filtering");
  await saveFitState(options.output, state, sums, evenSums, oddSums);

  const transports = averaged(sums, state.valid_prompts);
  const splitEven = state.valid_even_prompts > 0 ? averaged(evenSums, state.valid_even_prompts) : zeroed(evenSums);
  const splitOdd = state.valid_odd_prompts > 0 ? averaged(oddSums, state.valid_odd_prompts) : zeroed(oddSums);
  const exportDtype: SafeDtype = options.dtype === "float16" ? "F16" : "F32";
  const artifactTensors = new Map<string, SafeTensorValue>();
  sourceSites.forEach((siteId, index) => artifactTensors.set(`transport.${index.toString().padStart(4, "0")}`, transports.get(siteId)!));
  await writeLensSafetensors(join(options.output, "transports.safetensors"), artifactTensors, exportDtype, {
    format: "blah-jacobian-lens",
    orientation: "J[output_dimension,input_dimension]",
    model_weights_fingerprint: adapter.description.weightsFingerprint,
  });
  const splitConvergence = splitMetrics(splitEven, splitOdd, state.valid_even_prompts, state.valid_odd_prompts);
  const report: Record<string, unknown> = {
    format: "blah-jacobian-lens-fit-report",
    version: 1,
    method: "average-input-output-jacobian",
    estimator: "dimension-batched causal VJP; one-hot target-dimension cotangents at all valid target positions; source-position mean; prompt mean",
    model_checkpoint: options.checkpoint,
    checkpoint_sha256: adapter.description.checkpointSha256,
    weights_fingerprint: adapter.description.weightsFingerprint,
    corpus: {
      name: options.corpusName ?? "local-jsonl",
      dataset_identifier: options.corpusDatasetId ?? options.prompts,
      immutable_revision: options.corpusRevision ?? loaded.fingerprint,
      split: options.corpusSplit ?? "fit",
      filtering_rules: `native tokenize; truncate to ${options.maxSeqLen}; require more than skip_first+1 tokens`,
      prompt_count: options.samples,
      valid_prompt_count: state.valid_prompts,
      token_count: state.token_count,
      maximum_sequence_length: options.maxSeqLen,
      corpus_fingerprint: loaded.fingerprint,
      visibility: options.corpusVisibility ?? "synthetic",
    },
    fitting: {
      device: adapter.backend.name,
      dtype: "float32",
      exported_dtype: exportDtype,
      elapsed_seconds: state.elapsed_ms / 1000,
      vjp_batch_size: options.dimBatch,
      reverse_passes_per_prompt: Math.ceil(width / options.dimBatch),
      source_sites: sourceSites,
      target_site: options.targetSite,
      skip_first_positions: options.skipFirst,
      source_position_policy: "positions skip_first through sequence_length-2 inclusive",
      target_position_policy: "same valid positions; causal backward accumulates current-and-future effects",
    },
    skipped_prompts: state.skipped,
    split_half_convergence: splitConvergence,
    created_at: new Date().toISOString(),
  };
  await writeFile(join(options.output, "fit-report.json"), JSON.stringify(report, null, 2) + "\n");
  return { adapter, report, transports, splitEven, splitOdd };
}

function validateFitOptions(options: LensFitOptions): void {
  for (const [name, value] of Object.entries({ samples: options.samples, maxSeqLen: options.maxSeqLen, dimBatch: options.dimBatch, checkpointEvery: options.checkpointEvery })) {
    if (!Number.isInteger(value) || value < 1) throw new Error(`${name} must be a positive integer`);
  }
  if (!Number.isInteger(options.skipFirst) || options.skipFirst < 0) throw new Error("skipFirst must be a nonnegative integer");
}

function makeMatrices(sites: readonly string[], width: number): Map<string, SafeTensorValue> {
  return new Map(sites.map((site) => [site, { shape: [width, width], data: new Float32Array(width * width) }]));
}

function buildCausalCotangent(batch: number, time: number, width: number, start: number, active: number, skip: number): TensorData {
  const data = new Float32Array(batch * time * width);
  for (let row = 0; row < active; row++) {
    const dimension = start + row;
    for (let position = skip; position < time - 1; position++) {
      data[(row * time + position) * width + dimension] = 1;
    }
  }
  return { shape: [batch, time, width], dtype: "f32", data };
}

function accumulateRows(
  matrix: Float32Array,
  gradient: Float32Array,
  batch: number,
  time: number,
  sourceWidth: number,
  dimensionStart: number,
  activeDimensions: number,
  skipFirst: number,
): void {
  const positions = time - 1 - skipFirst;
  for (let batchRow = 0; batchRow < activeDimensions; batchRow++) {
    const outputDimension = dimensionStart + batchRow;
    const matrixBase = outputDimension * sourceWidth;
    for (let position = skipFirst; position < time - 1; position++) {
      const gradientBase = (batchRow * time + position) * sourceWidth;
      for (let column = 0; column < sourceWidth; column++) {
        matrix[matrixBase + column] += gradient[gradientBase + column] / positions;
      }
    }
  }
}

function averaged(sums: ReadonlyMap<string, SafeTensorValue>, count: number): Map<string, SafeTensorValue> {
  if (count <= 0) throw new Error("cannot average zero matrices");
  return new Map([...sums].map(([key, tensor]) => [key, {
    shape: tensor.shape,
    data: Float32Array.from(tensor.data, (value) => value / count),
  }]));
}

function zeroed(sums: ReadonlyMap<string, SafeTensorValue>): Map<string, SafeTensorValue> {
  return new Map([...sums].map(([key, tensor]) => [key, {
    shape: tensor.shape,
    data: new Float32Array(tensor.data.length),
  }]));
}

function fitOptionsFingerprint(options: LensFitOptions, sites: readonly string[], corpus: string, weights: string): string {
  const payload = JSON.stringify({
    checkpoint: options.checkpoint,
    samples: options.samples,
    maxSeqLen: options.maxSeqLen,
    skipFirst: options.skipFirst,
    dimBatch: options.dimBatch,
    targetSite: options.targetSite,
    sites,
    corpus,
    weights,
  });
  return `sha256:${createHash("sha256").update(payload).digest("hex")}`;
}

async function saveFitState(
  output: string,
  state: FitStateFile,
  all: ReadonlyMap<string, SafeTensorValue>,
  even: ReadonlyMap<string, SafeTensorValue>,
  odd: ReadonlyMap<string, SafeTensorValue>,
): Promise<void> {
  const tensors = new Map<string, SafeTensorValue>();
  for (const [key, value] of all) tensors.set(`all.${key}`, value);
  for (const [key, value] of even) tensors.set(`even.${key}`, value);
  for (const [key, value] of odd) tensors.set(`odd.${key}`, value);
  await writeLensSafetensors(join(output, "fit-state.safetensors"), tensors, "F32", { options_fingerprint: state.options_fingerprint });
  await writeFile(join(output, "fit-state.json"), JSON.stringify(state, null, 2) + "\n");
}

async function restoreFitState(
  output: string,
  expectedFingerprint: string,
  all: Map<string, SafeTensorValue>,
  even: Map<string, SafeTensorValue>,
  odd: Map<string, SafeTensorValue>,
): Promise<FitStateFile> {
  const state = JSON.parse(await readFile(join(output, "fit-state.json"), "utf8")) as FitStateFile;
  if (state.version !== 1 || state.options_fingerprint !== expectedFingerprint) throw new Error("resume state does not match fitting options/checkpoint/corpus");
  const stored = await readLensSafetensors(join(output, "fit-state.safetensors"));
  for (const [prefix, destination] of [["all", all], ["even", even], ["odd", odd]] as const) {
    for (const key of destination.keys()) {
      const tensor = stored.tensors.get(`${prefix}.${key}`);
      if (!tensor) throw new Error(`resume state missing ${prefix}.${key}`);
      destination.set(key, tensor);
    }
  }
  return state;
}

function splitMetrics(
  even: ReadonlyMap<string, SafeTensorValue>,
  odd: ReadonlyMap<string, SafeTensorValue>,
  evenCount: number,
  oddCount: number,
): Record<string, unknown> {
  if (evenCount === 0 || oddCount === 0) {
    return {
      split: "even-versus-odd valid fitting prompts",
      status: "insufficient-valid-prompts",
      valid_even_prompts: evenCount,
      valid_odd_prompts: oddCount,
      per_site: {},
      heldout_readout_metrics: "requires at least one valid prompt in each split",
    };
  }
  const perSite: Record<string, unknown> = {};
  for (const [site, a] of even) {
    const b = odd.get(site)!;
    let diffSq = 0;
    let normSq = 0;
    let rowCosine = 0;
    const rows = a.shape[0];
    const cols = a.shape[1];
    for (let row = 0; row < rows; row++) {
      let dot = 0, an = 0, bn = 0;
      for (let col = 0; col < cols; col++) {
        const index = row * cols + col;
        const delta = a.data[index] - b.data[index];
        diffSq += delta * delta;
        normSq += a.data[index] * a.data[index];
        dot += a.data[index] * b.data[index];
        an += a.data[index] * a.data[index];
        bn += b.data[index] * b.data[index];
      }
      rowCosine += an > 0 && bn > 0 ? dot / Math.sqrt(an * bn) : 0;
    }
    perSite[site] = {
      relative_frobenius_difference: Math.sqrt(diffSq) / Math.max(Math.sqrt(normSq), 1e-30),
      mean_row_cosine_similarity: rowCosine / rows,
    };
  }
  return {
    split: "even-versus-odd valid fitting prompts",
    per_site: perSite,
    heldout_readout_metrics: "computed during bundle validation",
  };
}
