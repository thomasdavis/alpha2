import { execFile } from "node:child_process";
import { join } from "node:path";
import { promisify } from "node:util";
import {
  AlphaLensAdapter,
  fitJacobianLens,
  writeBundleMetadata,
  serveLensRuntime,
  validateLens,
  sha256File,
  type LensFitOptions,
} from "@alpha/lens";
import { boolArg, intArg, parseKV, requireArg, strArg } from "../parse.js";

const execFileAsync = promisify(execFile);

export async function lensCmd(args: string[]): Promise<void> {
  const action = args[0];
  const kv = parseKV(args.slice(1));
  if (action === "describe") return describe(kv);
  if (action === "fit") return fit(kv);
  if (action === "bundle") return bundle(kv);
  if (action === "serve") return serve(kv);
  if (action === "validate") return validate(kv);
  throw new Error("lens command must be one of: describe, fit, bundle, serve, validate");
}

async function validate(kv: Record<string, string>): Promise<void> {
  const sourceRevision = kv["source-revision"] ?? (await execFileAsync("git", ["rev-parse", "HEAD"])).stdout.trim();
  const result = await validateLens({
    checkpoint: requireArg(kv, "checkpoint", "native ALPH checkpoint"),
    bundle: requireArg(kv, "bundle", "dist/blah-lens directory"),
    backend: strArg(kv, "backend", "cpu_ref"),
    heldoutPrompts: kv["heldout-prompts"],
    heldoutIndex: kv["heldout-index"] ? intArg(kv, "heldout-index", 0) : undefined,
    sourceRevision,
    adapterRevision: kv["adapter-revision"] ?? sourceRevision,
  });
  process.stdout.write(JSON.stringify(result, null, 2) + "\n");
  if (result.status !== "pass") process.exitCode = 1;
}

async function serve(kv: Record<string, string>): Promise<void> {
  const host = strArg(kv, "host", "127.0.0.1");
  const port = intArg(kv, "port", 8000);
  await serveLensRuntime({
    checkpoint: requireArg(kv, "checkpoint", "native ALPH checkpoint"),
    bundle: requireArg(kv, "bundle", "dist/blah-lens directory"),
    backend: strArg(kv, "backend", "cpu_ref"),
    host,
    port,
  });
  process.stderr.write(`Alpha BLAH Lens runtime listening at http://${host}:${port}\n`);
}

async function describe(kv: Record<string, string>): Promise<void> {
  const adapter = await AlphaLensAdapter.load({
    checkpoint: requireArg(kv, "checkpoint", "native ALPH checkpoint"),
    backend: strArg(kv, "backend", "cpu_ref"),
  });
  process.stdout.write(JSON.stringify(adapter.describe(), null, 2) + "\n");
}

async function fit(kv: Record<string, string>): Promise<void> {
  const output = requireArg(kv, "output", "bundle output directory");
  const options: LensFitOptions = {
    checkpoint: requireArg(kv, "checkpoint", "native ALPH checkpoint"),
    prompts: requireArg(kv, "prompts", "JSONL or text prompt corpus"),
    samples: intArg(kv, "samples", 100),
    maxSeqLen: intArg(kv, "max-seq-len", 128),
    skipFirst: intArg(kv, "skip-first", 16),
    dimBatch: intArg(kv, "dim-batch", 8),
    estimatorKind: strArg(kv, "estimator-kind", "same_position") as LensFitOptions["estimatorKind"],
    positionProbeSeed: intArg(kv, "position-probe-seed", 42),
    sourceSites: commaList(kv["source-sites"]),
    targetSite: strArg(kv, "target-site", "decoder.final.post") as "decoder.final.post",
    dtype: strArg(kv, "dtype", "float32") as "float32" | "float16",
    checkpointEvery: intArg(kv, "checkpoint-every", 5),
    resume: boolArg(kv, "resume", false),
    output,
    backend: strArg(kv, "backend", "cpu_ref"),
    corpusName: kv["corpus-name"],
    corpusDatasetId: kv["corpus-dataset-id"],
    corpusRevision: kv["corpus-revision"],
    corpusSplit: kv["corpus-split"],
    corpusVisibility: (kv["corpus-visibility"] ?? "synthetic") as LensFitOptions["corpusVisibility"],
    onProgress: (message) => process.stderr.write(message + "\n"),
  };
  const result = await fitJacobianLens(options);
  const identity = await identityFromArgs(kv);
  await writeBundleMetadata(result.adapter, output, identity);
  process.stderr.write(`wrote BLAH Lens bundle to ${output}\n`);
}

async function bundle(kv: Record<string, string>): Promise<void> {
  const output = requireArg(kv, "output", "existing fit output directory");
  const adapter = await AlphaLensAdapter.load({
    checkpoint: requireArg(kv, "checkpoint", "native ALPH checkpoint"),
    backend: strArg(kv, "backend", "cpu_ref"),
  });
  await writeBundleMetadata(adapter, output, await identityFromArgs(kv));
  process.stderr.write(`updated bundle metadata at ${output}\n`);
}

async function identityFromArgs(kv: Record<string, string>) {
  const sourceRevision = kv["source-revision"] ?? (await execFileAsync("git", ["rev-parse", "HEAD"])).stdout.trim();
  const hfExportDir = requireValue(kv["hf-export-dir"] ?? process.env.HF_EXPORT_DIR, "--hf-export-dir or HF_EXPORT_DIR");
  const hfFileNames = [
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
  ];
  const hfFiles = Object.fromEntries(await Promise.all(hfFileNames.map(async (name) => [name, await sha256File(join(hfExportDir, name))])));
  return {
    modelHfRepo: kv["model-hf-repo"] ?? process.env.MODEL_HF_REPO ?? "ajaxdavis/alpha-60m-chat",
    modelRevision: requireValue(kv["model-revision"] ?? process.env.MODEL_REVISION, "--model-revision or MODEL_REVISION"),
    lensHfRepo: kv["lens-hf-repo"] ?? process.env.LENS_HF_REPO ?? "ajaxdavis/alpha-chat-jlens",
    publicRuntimeUrl: kv["public-runtime-url"] ?? process.env.PUBLIC_RUNTIME_URL,
    sourceRevision,
    license: kv["license"] ?? "apache-2.0",
    hfFiles,
  };
}

function commaList(value: string | undefined): string[] | undefined {
  if (!value) return undefined;
  const items = value.split(",").map((item) => item.trim()).filter(Boolean);
  return items.length > 0 ? items : undefined;
}

function requireValue(value: string | undefined, label: string): string {
  if (!value) throw new Error(`Missing required value: ${label}`);
  return value;
}
