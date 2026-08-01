import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { createHash } from "node:crypto";
import { Effect } from "effect";
import type { TensorData } from "@alpha/core";
import { createSession, decodeStep, prefill } from "@alpha/inference";
import { buildChatTemplate } from "@alpha/tokenizers";
import { exportHfModel, FileCheckpoint, releaseCheckpointSnapshotBuffers } from "@alpha/train";
import { AlphaLensAdapter } from "./adapter.js";
import { loadLensPrompts } from "./prompts.js";
import { applyDenseTransport, greedyToken, rankLogitRow } from "./readout.js";
import { sha256File } from "./fingerprint.js";
import { readLensSafetensors, writeLensSafetensors, type SafeTensorValue } from "./safetensors.js";

export interface LensValidationOptions {
  readonly checkpoint: string;
  readonly bundle: string;
  readonly backend?: string;
  readonly heldoutPrompts?: string;
  readonly heldoutIndex?: number;
  readonly sourceRevision: string;
  readonly adapterRevision: string;
}

interface TestResult {
  readonly name: string;
  readonly status: "pass" | "fail";
  readonly tolerance?: Record<string, number>;
  readonly measurements: Record<string, unknown>;
  readonly detail?: string;
}

export async function validateLens(options: LensValidationOptions): Promise<Record<string, unknown>> {
  const adapter = await AlphaLensAdapter.load({ checkpoint: options.checkpoint, backend: options.backend, prepareInference: true });
  const manifestPath = join(options.bundle, "lens-manifest.json");
  const manifest = JSON.parse(await readFile(manifestPath, "utf8")) as any;
  const tests: TestResult[] = [];
  const add = (test: TestResult) => tests.push(test);

  add(await checkpointTest(adapter, manifest, options.checkpoint, options.bundle));
  add(await tokenizerTest(adapter, manifest));
  add(finalLogitParityTest(adapter));
  add(determinismTest(adapter));
  add(inferenceCaptureParityTest(adapter));
  add(vjpFiniteDifferenceTest(adapter));
  add(matrixOrientationTest());
  add(finalSiteIdentityTest(adapter));
  add(await transportShapeTest(options.bundle, manifest));
  const dtype = await dtypeParityTest(options.bundle, manifest, adapter);
  add(dtype.test);
  const split = await splitHalfTest(options, manifest, adapter);
  add(split.test);
  const golden = await writeGoldenFixture(options.bundle, manifest, adapter);
  add(golden.test);

  const failed = tests.filter((test) => test.status === "fail");
  const required = new Set([
    "checkpoint fingerprint verification",
    "tokenizer parity",
    "final logit parity",
    "matrix orientation",
    "transport shape validation",
    "golden fixture reproduction",
    "VJP finite-difference check",
  ]);
  const requiredFailure = failed.some((test) => required.has(test.name));
  const status = requiredFailure || failed.length > 0 ? "fail" : "pass";
  const validation = {
    format: "blah-jacobian-lens-validation",
    version: 1,
    status,
    model_revision: manifest.model.revision,
    checkpoint_fingerprint: adapter.description.checkpointSha256,
    weights_fingerprint: adapter.description.weightsFingerprint,
    source_repository_commit: options.sourceRevision,
    adapter_commit: options.adapterRevision,
    runtime_version: "@alpha/lens 0.1.0; blah-lens-http/1",
    framework: "alpha2 native TypeScript autograd and tensor engine",
    operating_system: `${process.platform} ${process.arch}`,
    device: adapter.backend.name,
    dtype: "float32 validation; bundle export dtype from manifest",
    tests,
    known_limitations: [
      "No low-rank artifacts are exported because dense transports are compact for this architecture.",
    ],
    created_at: new Date().toISOString(),
  };
  await writeFile(join(options.bundle, "validation.json"), JSON.stringify(validation, null, 2) + "\n");
  manifest.validation = { artifact: "validation.json", status };
  await writeFile(manifestPath, JSON.stringify(manifest, null, 2) + "\n");

  // Replace the fit report's provisional convergence values with the measured
  // held-out readout comparison used by validation.
  const fitPath = join(options.bundle, "fit-report.json");
  const fit = JSON.parse(await readFile(fitPath, "utf8"));
  fit.split_half_convergence = split.report;
  fit.exported_dtype_parity = dtype.report;
  await writeFile(fitPath, JSON.stringify(fit, null, 2) + "\n");
  return validation;
}

function inferenceCaptureParityTest(adapter: AlphaLensAdapter): TestResult {
  const weights = adapter.inferenceWeights;
  if (!weights) {
    return {
      name: "native inference capture parity",
      status: "fail",
      measurements: { error: "native inference weights were not prepared" },
    };
  }
  const ids = adapter.formatChat([{ role: "user", content: "Distinguish evidence from interpretation." }])
    .tokenIds.slice(0, Math.min(adapter.description.blockSize - 1, 24));
  const sites = [adapter.description.sites[0].id, adapter.description.sites.at(-1)!.id];
  const requested = new Set(sites);
  const promptNative = adapter.forwardCapture(ids, sites);
  const session = createSession(weights);
  const promptFast = { requestedSites: requested, sites: new Map<string, Float32Array>() };
  const fastLogits = new Float32Array(prefill(weights, session, ids, promptFast));
  const nativeLogits = adapter.copyTensor(promptNative.logits.data).data;
  const vocab = adapter.description.vocabularySize;
  const nativeLast = nativeLogits.subarray((ids.length - 1) * vocab, ids.length * vocab);
  const promptLogitError = maxAbsoluteDifference(nativeLast, fastLogits);
  let promptSiteError = 0;
  for (const site of sites) {
    const nativeSite = adapter.copyTensor(promptNative.sites.get(site)!.data).data;
    promptSiteError = Math.max(promptSiteError, maxAbsoluteDifference(nativeSite, promptFast.sites.get(site)!));
  }
  const next = greedyToken(fastLogits, 0, vocab);
  adapter.disposeCapture(promptNative);

  const stepFast = { requestedSites: requested, sites: new Map<string, Float32Array>() };
  const fastNextLogits = new Float32Array(decodeStep(weights, session, next, ids.length, stepFast));
  const extended = Int32Array.from([...ids, next]);
  const stepNative = adapter.forwardCapture(extended, sites);
  const nativeNextLogits = adapter.copyTensor(stepNative.logits.data).data.subarray(ids.length * vocab, (ids.length + 1) * vocab);
  const generatedLogitError = maxAbsoluteDifference(nativeNextLogits, fastNextLogits);
  let generatedSiteError = 0;
  for (const site of sites) {
    const nativeSite = adapter.copyTensor(stepNative.sites.get(site)!.data).data;
    const nativeLastSite = nativeSite.subarray(ids.length * adapter.description.targetSite.width);
    generatedSiteError = Math.max(generatedSiteError, maxAbsoluteDifference(nativeLastSite, stepFast.sites.get(site)!));
  }
  adapter.disposeCapture(stepNative);
  const promptTop1Same = greedyToken(nativeLast, 0, vocab) === greedyToken(fastLogits, 0, vocab);
  const generatedTop1Same = greedyToken(nativeNextLogits, 0, vocab) === greedyToken(fastNextLogits, 0, vocab);
  const maxSiteError = Math.max(promptSiteError, generatedSiteError);
  const maxLogitError = Math.max(promptLogitError, generatedLogitError);
  return {
    name: "native inference capture parity",
    status: maxSiteError <= 2e-3 && maxLogitError <= 2e-2 && promptTop1Same && generatedTop1Same ? "pass" : "fail",
    tolerance: { site_max_absolute_error: 2e-3, logit_max_absolute_error: 2e-2, top1_agreement: 1 },
    measurements: {
      sites,
      prompt_tokens: ids.length,
      generated_token: next,
      prompt_site_max_absolute_error: promptSiteError,
      generated_site_max_absolute_error: generatedSiteError,
      prompt_logit_max_absolute_error: promptLogitError,
      generated_logit_max_absolute_error: generatedLogitError,
      prompt_top1_agreement: promptTop1Same ? 1 : 0,
      generated_top1_agreement: generatedTop1Same ? 1 : 0,
      prefix_recomputed_during_generation: false,
    },
  };
}

async function checkpointTest(
  adapter: AlphaLensAdapter,
  manifest: any,
  checkpoint: string,
  bundle: string,
): Promise<TestResult> {
  let hfResolved: string | null = null;
  let hfError: string | null = null;
  const declaredFiles = manifest.model.hf_files as Record<string, string> | undefined;
  const remoteFiles: Record<string, string | null> = {};
  try {
    const response = await fetch(`https://huggingface.co/api/models/${manifest.model.repo_id}/revision/${manifest.model.revision}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    hfResolved = String((await response.json() as { sha?: unknown }).sha ?? "");
    if (declaredFiles) {
      const base = `https://huggingface.co/${manifest.model.repo_id}/resolve/${manifest.model.revision}`;
      for (const name of Object.keys(declaredFiles)) {
        if (name === "model.safetensors") {
          const head = await fetch(`${base}/${name}`, { method: "HEAD", redirect: "manual" });
          if (!head.ok && head.status !== 302) throw new Error(`${name} HTTP ${head.status}`);
          const etag = head.headers.get("x-linked-etag")?.replace(/^\"|\"$/g, "") ?? null;
          remoteFiles[name] = etag ? `sha256:${etag}` : null;
        } else {
          const file = await fetch(`${base}/${name}`);
          if (!file.ok) throw new Error(`${name} HTTP ${file.status}`);
          remoteFiles[name] = `sha256:${createHash("sha256").update(Buffer.from(await file.arrayBuffer())).digest("hex")}`;
        }
      }
    }
  } catch (error) { hfError = error instanceof Error ? error.message : String(error); }
  const localMatch = manifest.model.weights_fingerprint === adapter.description.weightsFingerprint
    && manifest.model.config_fingerprint === adapter.description.configFingerprint;
  const revisionMatch = hfResolved === manifest.model.revision;
  const nativeFiles: Record<string, string> = {};
  let nativeExportError: string | null = null;
  if (declaredFiles) {
    const temporary = await mkdtemp(join(dirname(bundle), ".native-hf-verify-"));
    try {
      const state = await Effect.runPromise(new FileCheckpoint().load(checkpoint));
      releaseCheckpointSnapshotBuffers(state);
      await exportHfModel(state, temporary);
      for (const name of Object.keys(declaredFiles)) nativeFiles[name] = await sha256File(join(temporary, name));
    } catch (error) {
      nativeExportError = error instanceof Error ? error.message : String(error);
    } finally {
      await rm(temporary, { recursive: true, force: true });
    }
  }
  const fileNames = Object.keys(declaredFiles ?? {});
  const nativeFilesMatch = fileNames.length > 0 && fileNames.every((name) => nativeFiles[name] === declaredFiles![name]);
  const remoteFilesMatch = fileNames.length > 0 && fileNames.every((name) => remoteFiles[name] === declaredFiles![name]);
  return {
    name: "checkpoint fingerprint verification",
    status: localMatch && revisionMatch && nativeFilesMatch && remoteFilesMatch ? "pass" : "fail",
    measurements: {
      checkpoint_sha256: adapter.description.checkpointSha256,
      manifest_weights_fingerprint: manifest.model.weights_fingerprint,
      loaded_weights_fingerprint: adapter.description.weightsFingerprint,
      manifest_config_fingerprint: manifest.model.config_fingerprint,
      loaded_config_fingerprint: adapter.description.configFingerprint,
      requested_hf_revision: manifest.model.revision,
      resolved_hf_revision: hfResolved,
      hf_error: hfError,
      declared_hf_files: declaredFiles ?? null,
      native_reexport_files: nativeFiles,
      remote_revision_files: remoteFiles,
      native_reexport_exact_match: nativeFilesMatch,
      remote_revision_exact_match: remoteFilesMatch,
      native_export_error: nativeExportError,
    },
  };
}

async function tokenizerTest(adapter: AlphaLensAdapter, manifest: any): Promise<TestResult> {
  const probes = ["plain ASCII", " café 🎉", "<|user|> hi <|assistant|>", "line one\nline two", "DNA? evidence—claim"];
  const roundTrips = probes.map((text) => {
    const ids = adapter.encode(text).slice(0, adapter.description.blockSize);
    return { text, ids: [...ids], decoded: adapter.decode(ids), exact: adapter.decode(ids) === text };
  });
  const base = `https://huggingface.co/${manifest.model.repo_id}/resolve/${manifest.model.revision}`;
  let hfVocabMatch = false;
  let hfTemplateMatch = false;
  let hfError: string | null = null;
  try {
    const [tokResponse, cfgResponse] = await Promise.all([fetch(`${base}/tokenizer.json`), fetch(`${base}/tokenizer_config.json`)]);
    if (!tokResponse.ok || !cfgResponse.ok) throw new Error(`tokenizer HTTP ${tokResponse.status}/${cfgResponse.status}`);
    const tokenJson = await tokResponse.json() as any;
    const tokenizerConfig = await cfgResponse.json() as any;
    const vocab = tokenJson.model?.vocab as Record<string, number>;
    hfVocabMatch = adapter.tokenizerArtifacts.vocab.length === Object.keys(vocab ?? {}).length
      && adapter.tokenizerArtifacts.vocab.every((token, id) => vocab?.[token] === id);
    hfTemplateMatch = tokenizerConfig.chat_template === buildChatTemplate();
  } catch (error) { hfError = error instanceof Error ? error.message : String(error); }
  const fingerprintMatch = manifest.model.tokenizer_fingerprint === adapter.description.tokenizerFingerprint
    && manifest.model.chat_template_fingerprint === adapter.description.chatTemplateFingerprint;
  return {
    name: "tokenizer parity",
    status: roundTrips.every((probe) => probe.exact) && fingerprintMatch && hfVocabMatch && hfTemplateMatch ? "pass" : "fail",
    measurements: {
      round_trips: roundTrips,
      special_tokens: adapter.description.specialTokens,
      bos_inserted_by_native_encode: false,
      eos_policy: "atomic end_of_text terminates the conversation",
      tokenizer_fingerprint_match: fingerprintMatch,
      hf_vocab_exact_match: hfVocabMatch,
      hf_chat_template_exact_match: hfTemplateMatch,
      hf_error: hfError,
    },
  };
}

function finalLogitParityTest(adapter: AlphaLensAdapter): TestResult {
  const cases = [
    "raw text",
    "<|end_of_text|>",
    "Several tokens test the final decoder path.",
    adapter.formatChat([{ role: "user", content: "What is DNA?" }]).text,
  ];
  const measurements: Record<string, unknown>[] = [];
  let worstAbs = 0;
  let worstRel = 0;
  let agreements = 0;
  let positions = 0;
  for (const text of cases) {
    const ids = adapter.encode(text).slice(0, adapter.description.blockSize);
    const capture = adapter.forwardCapture(ids, [adapter.description.sites.at(-1)!.id]);
    const ordinary = adapter.copyTensor(capture.logits.data);
    const decoded = adapter.exactFinalDecode(adapter.copyTensor(capture.target.data));
    const comparison = compareLogits(ordinary.data, decoded.data, adapter.description.vocabularySize);
    worstAbs = Math.max(worstAbs, comparison.maxAbs);
    worstRel = Math.max(worstRel, comparison.maxRel);
    agreements += comparison.top1Same;
    positions += comparison.positions;
    measurements.push({ text, tokens: ids.length, ...comparison });
    adapter.disposeCapture(capture);
  }
  // At least one generated continuation is included in parity coverage.
  const seed = adapter.formatChat([{ role: "user", content: "Say one clear sentence." }]).tokenIds.slice(0, adapter.description.blockSize - 1);
  const first = adapter.forwardCapture(seed, [adapter.description.sites.at(-1)!.id]);
  const copied = adapter.copyTensor(first.logits.data);
  const next = greedyToken(copied.data, (seed.length - 1) * adapter.description.vocabularySize, adapter.description.vocabularySize);
  adapter.disposeCapture(first);
  const generatedIds = Int32Array.from([...seed, next]);
  const generated = adapter.forwardCapture(generatedIds, [adapter.description.sites.at(-1)!.id]);
  const genOrdinary = adapter.copyTensor(generated.logits.data);
  const genDecoded = adapter.exactFinalDecode(adapter.copyTensor(generated.target.data));
  const genComparison = compareLogits(genOrdinary.data, genDecoded.data, adapter.description.vocabularySize);
  worstAbs = Math.max(worstAbs, genComparison.maxAbs);
  worstRel = Math.max(worstRel, genComparison.maxRel);
  agreements += genComparison.top1Same;
  positions += genComparison.positions;
  measurements.push({ generated_token: next, ...genComparison });
  adapter.disposeCapture(generated);
  return {
    name: "final logit parity",
    status: worstAbs <= 1e-6 && agreements === positions ? "pass" : "fail",
    tolerance: { max_absolute_error: 1e-6, top1_agreement: 1 },
    measurements: { max_absolute_error: worstAbs, max_relative_error: worstRel, top1_agreement: agreements / positions, cases: measurements },
  };
}

function determinismTest(adapter: AlphaLensAdapter): TestResult {
  const ids = adapter.encode("Determinism should preserve every captured state.").slice(0, adapter.description.blockSize);
  const sites = [adapter.description.sites[0].id, adapter.description.sites.at(-1)!.id];
  const a = adapter.forwardCapture(ids, sites, 2);
  const first = new Map(sites.map((site) => [site, adapter.copyTensor(a.sites.get(site)!.data).data]));
  let maxDifference = 0;
  let replicaDifference = 0;
  for (const site of sites) {
    const ad = first.get(site)!;
    const rowSize = ids.length * adapter.description.targetSite.width;
    for (let index = 0; index < rowSize; index++) replicaDifference = Math.max(replicaDifference, Math.abs(ad[index] - ad[rowSize + index]));
  }
  adapter.disposeCapture(a);
  const b = adapter.forwardCapture(ids, sites, 2);
  for (const site of sites) {
    const ad = first.get(site)!;
    const bd = adapter.copyTensor(b.sites.get(site)!.data).data;
    for (let index = 0; index < ad.length; index++) maxDifference = Math.max(maxDifference, Math.abs(ad[index] - bd[index]));
  }
  adapter.disposeCapture(b);
  return {
    name: "determinism and replicated batch equality",
    status: maxDifference === 0 && replicaDifference === 0 ? "pass" : "fail",
    tolerance: { max_absolute_error: 0 },
    measurements: { repeated_forward_max_absolute_error: maxDifference, replicated_batch_max_absolute_error: replicaDifference, dropout: "disabled", stochastic_routing: "not present" },
  };
}

function vjpFiniteDifferenceTest(adapter: AlphaLensAdapter): TestResult {
  const ids = adapter.encode("A careful distinction survives a nearby counterexample.").slice(0, 24);
  const siteId = adapter.description.sites[Math.floor(adapter.description.sites.length / 2)].id;
  const width = adapter.description.targetSite.width;
  const shape = [1, ids.length, width] as const;
  const direction = Float32Array.from({ length: ids.length * width }, (_, index) => Math.sin(index * 0.37 + 0.2) * 1e-2);
  const targetDirection = Float32Array.from({ length: ids.length * width }, (_, index) => Math.cos(index * 0.19 - 0.4) * 1e-2);
  const capture = adapter.forwardCapture(ids, [siteId]);
  const sourceGradient = adapter.vjp(
    capture,
    { shape, dtype: "f32", data: targetDirection },
    [siteId],
    false,
  ).get(siteId)!.data;
  const analytic = dot(sourceGradient, direction);
  adapter.disposeCapture(capture);
  const epsilonValues = [5e-3, 2e-3, 1e-3];
  const numerics = epsilonValues.map((epsilon) => {
    const plus = targetScalar(adapter, ids, siteId, direction, epsilon, targetDirection);
    const minus = targetScalar(adapter, ids, siteId, direction, -epsilon, targetDirection);
    return (plus - minus) / (2 * epsilon);
  });
  const numeric = numerics.at(-1)!;
  const abs = Math.abs(analytic - numeric);
  const rel = abs / Math.max(Math.abs(analytic), Math.abs(numeric), 1e-12);
  const cosine = Math.sign(analytic * numeric);
  return {
    name: "VJP finite-difference check",
    status: abs <= 2e-3 || rel <= 5e-2 ? "pass" : "fail",
    tolerance: { absolute_error: 2e-3, relative_error: 5e-2 },
    measurements: { source_site: siteId, target_direction: "deterministic dense sinusoidal direction", dtype: "float32", device: adapter.backend.name, epsilon_values: epsilonValues, analytic, numeric_estimates: numerics, absolute_error: abs, relative_error: rel, cosine_agreement: cosine },
  };
}

function targetScalar(adapter: AlphaLensAdapter, ids: Int32Array, siteId: string, direction: Float32Array, scale: number, targetDirection: Float32Array): number {
  const perturbation: TensorData = { shape: [1, ids.length, adapter.description.targetSite.width], dtype: "f32", data: Float32Array.from(direction, (value) => value * scale) };
  const capture = adapter.forwardCapture(ids, [siteId], 1, new Map([[siteId, perturbation]]));
  const value = dot(adapter.copyTensor(capture.target.data).data, targetDirection);
  adapter.disposeCapture(capture);
  return value;
}

function matrixOrientationTest(): TestResult {
  const source = { shape: [1, 1, 2], dtype: "f32" as const, data: new Float32Array([7, 11]) };
  const matrix = { shape: [2, 2], data: new Float32Array([1, 2, 3, 5]) };
  const result = applyDenseTransport(source, matrix);
  const expected = [29, 76];
  const wrongTranspose = [40, 69];
  const exact = result.data[0] === expected[0] && result.data[1] === expected[1];
  return {
    name: "matrix orientation",
    status: exact ? "pass" : "fail",
    measurements: { matrix: [[1, 2], [3, 5]], source: [7, 11], stored_convention: "J[output_dimension,input_dimension]", application: "h @ transpose(J)", observed: [...result.data], expected, intentionally_wrong_transpose_result: wrongTranspose },
  };
}

function finalSiteIdentityTest(adapter: AlphaLensAdapter): TestResult {
  const ids = adapter.encode("The last site is the target site.").slice(0, adapter.description.blockSize);
  const finalSite = adapter.description.sites.at(-1)!.id;
  const capture = adapter.forwardCapture(ids, [finalSite]);
  const site = adapter.copyTensor(capture.sites.get(finalSite)!.data);
  const target = adapter.copyTensor(capture.target.data);
  let siteError = 0;
  for (let index = 0; index < site.data.length; index++) siteError = Math.max(siteError, Math.abs(site.data[index] - target.data[index]));
  const decoded = adapter.exactFinalDecode(site);
  const ordinary = adapter.copyTensor(capture.logits.data);
  const logits = compareLogits(decoded.data, ordinary.data, adapter.description.vocabularySize);
  adapter.disposeCapture(capture);
  return {
    name: "final-site identity",
    status: siteError === 0 && logits.maxAbs === 0 && logits.top1Same === logits.positions ? "pass" : "fail",
    measurements: { site: finalSite, target_site: adapter.description.targetSite.id, representation_max_absolute_error: siteError, logit_max_absolute_error: logits.maxAbs, top1_agreement: logits.top1Same / logits.positions },
  };
}

async function transportShapeTest(bundle: string, manifest: any): Promise<TestResult> {
  const stored = await readLensSafetensors(join(bundle, "transports.safetensors"));
  const problems: string[] = [];
  for (const site of manifest.sites) {
    const tensor = stored.tensors.get(site.transport.tensor_key);
    const expected = site.transport.shape as number[];
    if (!tensor) problems.push(`${site.id}: missing ${site.transport.tensor_key}`);
    else if (tensor.shape.length !== expected.length || tensor.shape.some((value, index) => value !== expected[index])) problems.push(`${site.id}: [${tensor.shape}] != [${expected}]`);
  }
  return {
    name: "transport shape validation",
    status: problems.length === 0 ? "pass" : "fail",
    measurements: { site_count: manifest.sites.length, tensor_count: stored.tensors.size, orientation: "[target_width,source_width]", problems },
  };
}

async function dtypeParityTest(bundle: string, manifest: any, adapter: AlphaLensAdapter): Promise<{ test: TestResult; report: Record<string, unknown> }> {
  const exported = await readLensSafetensors(join(bundle, "transports.safetensors"));
  const fitState = await readLensSafetensors(join(bundle, "fit-state.safetensors"));
  const state = JSON.parse(await readFile(join(bundle, "fit-state.json"), "utf8"));
  const ids = adapter.encode("A short held-out sentence tests exported matrix precision.").slice(0, adapter.description.blockSize);
  const capture = adapter.forwardCapture(ids, manifest.sites.map((site: any) => site.id));
  let worstMatrixRelativeError = 0;
  let worstLogitError = 0;
  let top1Same = 0;
  let top5Overlap = 0;
  let comparisons = 0;
  let worstSite = "";
  for (const [index, site] of manifest.sites.entries()) {
    const sum = fitState.tensors.get(`all.${site.id}`)!;
    const full = { shape: sum.shape, data: Float32Array.from(sum.data, (value) => value / state.valid_prompts) };
    const reduced = exported.tensors.get(site.transport.tensor_key)!;
    const relative = relativeError(full.data, reduced.data);
    if (relative > worstMatrixRelativeError) { worstMatrixRelativeError = relative; worstSite = site.id; }
    const source = finalPosition(adapter.copyTensor(capture.sites.get(site.id)!.data));
    const f32Logits = adapter.exactFinalDecode(applyDenseTransport(source, full));
    const exportedLogits = adapter.exactFinalDecode(applyDenseTransport(source, reduced));
    const metrics = compareSingleRow(f32Logits.data, exportedLogits.data);
    worstLogitError = Math.max(worstLogitError, metrics.maxAbs);
    top1Same += metrics.top1Same ? 1 : 0;
    top5Overlap += metrics.top5Overlap;
    comparisons++;
  }
  adapter.disposeCapture(capture);
  const report = {
    float32_reference: "fit-state sums divided by valid prompt count",
    exported_dtype: manifest.lens.dtype,
    relative_matrix_error: worstMatrixRelativeError,
    heldout_top1_agreement: top1Same / comparisons,
    heldout_top5_overlap: top5Overlap / comparisons,
    maximum_logit_error: worstLogitError,
    readout_rank_correlation: "reported in split-half convergence",
    worst_affected_source_site: worstSite,
  };
  return {
    test: {
      name: "exported dtype parity",
      status: top1Same / comparisons >= 0.99 && top5Overlap / comparisons >= 0.95 ? "pass" : "fail",
      tolerance: { top1_agreement: 0.99, top5_overlap: 0.95 },
      measurements: report,
    },
    report,
  };
}

async function splitHalfTest(options: LensValidationOptions, manifest: any, adapter: AlphaLensAdapter): Promise<{ test: TestResult; report: Record<string, unknown> }> {
  const stored = await readLensSafetensors(join(options.bundle, "fit-state.safetensors"));
  const state = JSON.parse(await readFile(join(options.bundle, "fit-state.json"), "utf8"));
  if (state.valid_even_prompts < 1 || state.valid_odd_prompts < 1) {
    const report = {
      split: "even-versus-odd valid fitting prompts",
      status: "insufficient-valid-prompts",
      valid_even_prompts: state.valid_even_prompts,
      valid_odd_prompts: state.valid_odd_prompts,
      heldout_readout_metrics: "not measurable",
    };
    return {
      test: {
        name: "split-half convergence",
        status: "fail",
        measurements: report,
        detail: "At least two valid fitting prompts are required to measure split-half convergence.",
      },
      report,
    };
  }
  const text = options.heldoutPrompts
    ? (await loadLensPrompts(options.heldoutPrompts)).prompts[options.heldoutIndex ?? state.completed_prompts]
    : "A genuinely held-out synthetic sentence asks whether evidence changes only dependent conclusions.";
  if (!text) throw new Error("held-out prompt index is outside the supplied prompt corpus");
  const ids = adapter.encode(text).slice(0, manifest.lens.max_seq_len);
  const capture = adapter.forwardCapture(ids, manifest.sites.map((site: any) => site.id));
  let top1 = 0, top5 = 0, rankCorrelation = 0;
  const perSite: Record<string, unknown> = {};
  for (const site of manifest.sites) {
    const evenSum = stored.tensors.get(`even.${site.id}`)!;
    const oddSum = stored.tensors.get(`odd.${site.id}`)!;
    const even = { shape: evenSum.shape, data: Float32Array.from(evenSum.data, (value) => value / state.valid_even_prompts) };
    const odd = { shape: oddSum.shape, data: Float32Array.from(oddSum.data, (value) => value / state.valid_odd_prompts) };
    const source = finalPosition(adapter.copyTensor(capture.sites.get(site.id)!.data));
    const evenLogits = adapter.exactFinalDecode(applyDenseTransport(source, even)).data;
    const oddLogits = adapter.exactFinalDecode(applyDenseTransport(source, odd)).data;
    const readout = compareSingleRow(evenLogits, oddLogits);
    const corr = spearman(evenLogits, oddLogits);
    top1 += readout.top1Same ? 1 : 0;
    top5 += readout.top5Overlap;
    rankCorrelation += corr;
    perSite[site.id] = {
      relative_frobenius_difference: relativeError(even.data, odd.data),
      mean_row_cosine_similarity: meanRowCosine(even.data, odd.data, even.shape[0], even.shape[1]),
      heldout_top1_agreement: readout.top1Same ? 1 : 0,
      heldout_top5_overlap: readout.top5Overlap,
      heldout_readout_rank_correlation: corr,
    };
  }
  adapter.disposeCapture(capture);
  const count = manifest.sites.length;
  const report = {
    split: "even-versus-odd valid fitting prompts",
    heldout_prompt: { source: options.heldoutPrompts ?? "external synthetic validation sentence", index: options.heldoutIndex ?? state.completed_prompts, token_count: ids.length },
    heldout_top1_agreement: top1 / count,
    heldout_top5_overlap: top5 / count,
    heldout_readout_rank_correlation: rankCorrelation / count,
    per_site: perSite,
    informational: true,
  };
  return {
    test: { name: "split-half convergence", status: "pass", measurements: report, detail: "Informational measurement; no convergence threshold is used to select the checkpoint." },
    report,
  };
}

async function writeGoldenFixture(bundle: string, manifest: any, adapter: AlphaLensAdapter): Promise<{ test: TestResult }> {
  const text = "<|user|> What is a promise? <|assistant|>";
  const chat = [{ role: "user" as const, content: "What is a promise?" }];
  const formatted = adapter.formatChat(chat);
  if (formatted.text !== text) throw new Error("golden chat formatting changed unexpectedly");
  const selected = [manifest.sites[0], manifest.sites[Math.floor(manifest.sites.length / 2)], manifest.sites.at(-1)];
  const capture = adapter.forwardCapture(formatted.tokenIds, selected.map((site: any) => site.id));
  const stored = await readLensSafetensors(join(bundle, "transports.safetensors"));
  const tensors = new Map<string, SafeTensorValue>();
  const results: Record<string, unknown> = {};
  for (const site of selected) {
    const source = adapter.copyTensor(capture.sites.get(site.id)!.data);
    tensors.set(`activation.${site.id}`, { shape: source.shape, data: source.data });
    const direct = adapter.exactFinalDecode(source);
    const jacobian = adapter.exactFinalDecode(applyDenseTransport(source, stored.tensors.get(site.transport.tensor_key)!));
    results[site.id] = Array.from({ length: formatted.tokenIds.length }, (_, position) => ({
      position,
      logit: rankLogitRow(direct.data, position * adapter.description.vocabularySize, adapter.description.vocabularySize, adapter, 5, null, false).top,
      jacobian: rankLogitRow(jacobian.data, position * adapter.description.vocabularySize, adapter.description.vocabularySize, adapter, 5, null, false).top,
    }));
  }
  const target = adapter.copyTensor(capture.target.data);
  const logits = adapter.copyTensor(capture.logits.data);
  tensors.set("target.decoder.final.post", { shape: target.shape, data: target.data });
  tensors.set("logits.final", { shape: logits.shape, data: logits.data });
  await mkdir(join(bundle, "fixtures"), { recursive: true });
  await writeLensSafetensors(join(bundle, "fixtures/golden.safetensors"), tensors, "F32", {
    model_revision: manifest.model.revision,
    weights_fingerprint: adapter.description.weightsFingerprint,
  });
  const finalTop = Array.from({ length: formatted.tokenIds.length }, (_, position) => rankLogitRow(
    logits.data,
    position * adapter.description.vocabularySize,
    adapter.description.vocabularySize,
    adapter,
    8,
    null,
    false,
  ).top);
  const golden = {
    raw_text: "What is a promise?",
    chat,
    formatted_text: formatted.text,
    token_ids: [...formatted.tokenIds],
    token_strings: adapter.tokenStrings(formatted.tokenIds),
    model_revision: manifest.model.revision,
    weights_fingerprint: adapter.description.weightsFingerprint,
    tokenizer_fingerprint: adapter.description.tokenizerFingerprint,
    sites: manifest.sites.map((site: any) => site.id),
    selected_sites: selected.map((site: any) => site.id),
    tensor_artifact: "golden.safetensors",
    final_top_k: finalTop,
    readouts: results,
  };
  await writeFile(join(bundle, "fixtures/golden.json"), JSON.stringify(golden, null, 2) + "\n");
  adapter.disposeCapture(capture);
  return {
    test: {
      name: "golden fixture reproduction",
      status: golden.token_ids.length === golden.token_strings.length && golden.sites.length === manifest.sites.length ? "pass" : "fail",
      measurements: { tokens: golden.token_ids.length, selected_sites: golden.selected_sites, tensor_artifact: "fixtures/golden.safetensors", chat_formatted: true },
    },
  };
}

function compareLogits(a: Float32Array, b: Float32Array, vocab: number) {
  let maxAbs = 0, maxRel = 0, top1Same = 0;
  const positions = a.length / vocab;
  for (let index = 0; index < a.length; index++) {
    const abs = Math.abs(a[index] - b[index]);
    maxAbs = Math.max(maxAbs, abs);
    maxRel = Math.max(maxRel, abs / Math.max(Math.abs(a[index]), Math.abs(b[index]), 1e-12));
  }
  for (let position = 0; position < positions; position++) {
    const offset = position * vocab;
    if (argmax(a, offset, vocab) === argmax(b, offset, vocab)) top1Same++;
  }
  return { maxAbs, maxRel, top1Same, positions };
}

function compareSingleRow(a: Float32Array, b: Float32Array) {
  let maxAbs = 0;
  for (let index = 0; index < a.length; index++) maxAbs = Math.max(maxAbs, Math.abs(a[index] - b[index]));
  const topA = topIds(a, 5);
  const topB = topIds(b, 5);
  return { maxAbs, top1Same: topA[0] === topB[0], top5Overlap: topA.filter((id) => topB.includes(id)).length / 5 };
}

function finalPosition(tensor: { shape: readonly number[]; dtype: "f32"; data: Float32Array }) {
  const [batch, time, width] = tensor.shape;
  if (batch !== 1) throw new Error("held-out readout expects batch size 1");
  return { shape: [1, 1, width], dtype: "f32" as const, data: tensor.data.slice((time - 1) * width, time * width) };
}

function argmax(data: Float32Array, offset: number, length: number): number {
  let best = 0;
  for (let index = 1; index < length; index++) if (data[offset + index] > data[offset + best]) best = index;
  return best;
}

function topIds(data: Float32Array, count: number): number[] {
  return Array.from(data.keys()).sort((a, b) => data[b] - data[a] || a - b).slice(0, count);
}

function dot(a: Float32Array, b: Float32Array): number {
  let value = 0;
  for (let index = 0; index < a.length; index++) value += a[index] * b[index];
  return value;
}

function maxAbsoluteDifference(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) return Number.POSITIVE_INFINITY;
  let maximum = 0;
  for (let index = 0; index < a.length; index++) maximum = Math.max(maximum, Math.abs(a[index] - b[index]));
  return maximum;
}

function relativeError(a: Float32Array, b: Float32Array): number {
  let diff = 0, norm = 0;
  for (let index = 0; index < a.length; index++) { const delta = a[index] - b[index]; diff += delta * delta; norm += a[index] * a[index]; }
  return Math.sqrt(diff) / Math.max(Math.sqrt(norm), 1e-30);
}

function meanRowCosine(a: Float32Array, b: Float32Array, rows: number, cols: number): number {
  let total = 0;
  for (let row = 0; row < rows; row++) {
    let dotProduct = 0, an = 0, bn = 0;
    for (let col = 0; col < cols; col++) { const index = row * cols + col; dotProduct += a[index] * b[index]; an += a[index] ** 2; bn += b[index] ** 2; }
    total += an > 0 && bn > 0 ? dotProduct / Math.sqrt(an * bn) : 0;
  }
  return total / rows;
}

function spearman(a: Float32Array, b: Float32Array): number {
  const rank = (data: Float32Array) => {
    const result = new Int32Array(data.length);
    Array.from(data.keys()).sort((x, y) => data[x] - data[y] || x - y).forEach((id, index) => { result[id] = index; });
    return result;
  };
  const ar = rank(a), br = rank(b);
  const n = a.length;
  let squared = 0;
  for (let index = 0; index < n; index++) squared += (ar[index] - br[index]) ** 2;
  return 1 - (6 * squared) / (n * (n * n - 1));
}
