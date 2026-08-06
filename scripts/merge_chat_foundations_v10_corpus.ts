#!/usr/bin/env npx tsx

/** Merge independently generated/reviewed v8-style waves around one frozen dev set. */

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { createReadStream } from "node:fs";
import { mkdir, readFile, rename, stat, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, resolve } from "node:path";

type Turn = { readonly role: "user" | "assistant"; readonly content: string };

interface SourceConfig {
  readonly name: string;
  readonly manifest: string;
}

interface Config {
  readonly schema: "alpha-chat-foundations-v10-merge-config-v1";
  readonly seed: string;
  readonly base: SourceConfig;
  readonly waves: readonly SourceConfig[];
  readonly required_train_minimum: number;
  readonly required_dev_minimum: number;
}

interface V8CatalogRow {
  readonly schema: "alpha-chat-foundations-v8-catalog-v1";
  readonly candidate_id: string;
  readonly batch_id: string;
  readonly focus: string;
  readonly status: "train" | "dev" | "rejected";
  readonly rejection_reasons: readonly Record<string, unknown>[];
  readonly conversation_sha256: string;
  readonly tokens: number;
  readonly turns: number;
  readonly normalized_user_turn_sha256: readonly string[];
  readonly review: Record<string, unknown>;
  readonly candidate: {
    readonly candidate_id: string;
    readonly turns: readonly Turn[];
    readonly [key: string]: unknown;
  };
}

interface LoadedSource {
  readonly name: string;
  readonly kind: "base" | "wave";
  readonly manifestPath: string;
  readonly manifest: Record<string, any>;
  readonly trainText: string;
  readonly devText: string;
  readonly catalog: readonly V8CatalogRow[];
}

interface AcceptedRow {
  readonly qualifiedId: string;
  readonly source: LoadedSource;
  readonly row: V8CatalogRow;
  readonly rendered: string;
  readonly normalizedUserTurns: readonly string[];
  readonly mergedStatus: "train" | "dev";
}

const USER = "<|user|>";
const ASSISTANT = "<|assistant|>";
const END = "<|end_of_text|>";

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function parseArgs(argv: readonly string[]): Record<string, string> {
  const result: Record<string, string> = {};
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key?.startsWith("--") || !value || value.startsWith("--"))
      throw new Error(`invalid argument near ${String(key)}`);
    result[key.slice(2)] = value;
    index += 1;
  }
  return result;
}

function sha256(value: string | Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

async function sha256File(path: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path))
    hash.update(chunk as Buffer);
  return hash.digest("hex");
}

async function evidence(
  path: string,
  rows?: number,
): Promise<Record<string, unknown>> {
  const metadata = await stat(path);
  return {
    path,
    bytes: metadata.size,
    sha256: await sha256File(path),
    ...(rows === undefined ? {} : { rows }),
  };
}

async function verifyEvidence(
  item: Record<string, unknown>,
  label: string,
): Promise<void> {
  assert(typeof item.path === "string", `${label}: path missing`);
  assert(
    typeof item.sha256 === "string" && /^[0-9a-f]{64}$/.test(item.sha256),
    `${label}: SHA-256 missing`,
  );
  const actual = await evidence(item.path);
  assert(actual.sha256 === item.sha256, `${label}: SHA-256 drift`);
  if (item.bytes !== undefined)
    assert(actual.bytes === item.bytes, `${label}: byte count drift`);
}

async function atomicWrite(path: string, content: string): Promise<void> {
  const temporary = `${path}.tmp-${process.pid}`;
  await writeFile(temporary, content, { encoding: "utf8", flag: "wx" });
  await rename(temporary, path);
}

function resolveFrom(value: string, configPath: string): string {
  return isAbsolute(value) ? value : resolve(dirname(configPath), value);
}

function normalize(value: string): string {
  return value
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();
}

function render(turns: readonly Turn[], label: string): string {
  assert(
    turns.length === 2 || turns.length === 4,
    `${label}: invalid turn count`,
  );
  for (const [index, turn] of turns.entries()) {
    assert(
      turn.role === (index % 2 === 0 ? "user" : "assistant"),
      `${label}: role order drift`,
    );
    assert(turn.content.trim().length > 0, `${label}: empty turn`);
    for (const marker of [USER, ASSISTANT, END])
      assert(!turn.content.includes(marker), `${label}: delimiter leak`);
  }
  return `${turns
    .map(
      (turn) =>
        `${turn.role === "user" ? USER : ASSISTANT} ${turn.content.trim().replace(/\s+/g, " ")}`,
    )
    .join(" ")} ${END}`;
}

function lines(text: string): readonly string[] {
  return text.split(/\r?\n/).filter(Boolean);
}

function counted(values: readonly string[]): Map<string, number> {
  const result = new Map<string, number>();
  for (const value of values) result.set(value, (result.get(value) ?? 0) + 1);
  return result;
}

function equalCounts(
  left: Map<string, number>,
  right: Map<string, number>,
): boolean {
  if (left.size !== right.size) return false;
  for (const [key, count] of left) if (right.get(key) !== count) return false;
  return true;
}

async function loadSource(
  source: SourceConfig,
  kind: "base" | "wave",
  configPath: string,
): Promise<LoadedSource> {
  assert(
    /^[a-z0-9][a-z0-9-]*$/.test(source.name),
    `invalid source name ${source.name}`,
  );
  const manifestPath = resolveFrom(source.manifest, configPath);
  const manifest = JSON.parse(await readFile(manifestPath, "utf8")) as Record<
    string,
    any
  >;
  assert(
    manifest.schema === "alpha-chat-foundations-v8-corpus-v1",
    `${source.name}: unexpected source corpus`,
  );
  assert(
    manifest.sourceTreeDirty === false,
    `${source.name}: dirty source build`,
  );
  for (const invariant of [
    "allCandidatesReviewedExactlyOnce",
    "onlyIndependentlyAcceptedSyntheticDataTrains",
    "exactNormalizedVisiblePromptExclusion",
    "exactConversationDeduplication",
    "normalizedUserTurnDeduplication",
  ])
    assert(
      manifest.invariants?.[invariant] === true,
      `${source.name}: ${invariant} failed`,
    );
  assert(
    manifest.holdouts?.acceptedCollisions === 0,
    `${source.name}: holdout collision`,
  );
  await verifyEvidence(manifest.outputs.train, `${source.name} train`);
  await verifyEvidence(manifest.outputs.dev, `${source.name} dev`);
  await verifyEvidence(manifest.outputs.catalog, `${source.name} catalog`);
  const trainText = await readFile(manifest.outputs.train.path, "utf8");
  const devText = await readFile(manifest.outputs.dev.path, "utf8");
  const catalog = lines(
    await readFile(manifest.outputs.catalog.path, "utf8"),
  ).map((line, index) => {
    const row = JSON.parse(line) as V8CatalogRow;
    assert(
      row.schema === "alpha-chat-foundations-v8-catalog-v1",
      `${source.name}:${index + 1}: schema drift`,
    );
    assert(
      row.candidate_id === row.candidate?.candidate_id,
      `${source.name}:${index + 1}: candidate identity drift`,
    );
    return row;
  });
  assert(
    catalog.length === manifest.outputs.catalog.rows,
    `${source.name}: catalog rows drift`,
  );
  assert(
    lines(trainText).length === manifest.outputs.train.rows,
    `${source.name}: train rows drift`,
  );
  assert(
    lines(devText).length === manifest.outputs.dev.rows,
    `${source.name}: dev rows drift`,
  );

  const acceptedHashes = {
    train: catalog
      .filter((row) => row.status === "train")
      .map((row) => row.conversation_sha256),
    dev: catalog
      .filter((row) => row.status === "dev")
      .map((row) => row.conversation_sha256),
  };
  assert(
    equalCounts(
      counted(lines(trainText).map(sha256)),
      counted(acceptedHashes.train),
    ),
    `${source.name}: train/catalog population mismatch`,
  );
  assert(
    equalCounts(
      counted(lines(devText).map(sha256)),
      counted(acceptedHashes.dev),
    ),
    `${source.name}: dev/catalog population mismatch`,
  );
  return {
    name: source.name,
    kind,
    manifestPath,
    manifest,
    trainText,
    devText,
    catalog,
  };
}

async function main(): Promise<void> {
  const cli = parseArgs(process.argv.slice(2));
  assert(cli.config && cli.out, "required: --config and --out");
  const repo = resolve(cli.repo ?? process.cwd());
  const configPath = resolve(cli.config);
  const config = JSON.parse(await readFile(configPath, "utf8")) as Config;
  assert(
    config.schema === "alpha-chat-foundations-v10-merge-config-v1",
    "unexpected config schema",
  );
  assert(config.seed.length > 0, "seed missing");
  assert(config.waves.length > 0, "at least one new wave is required");
  assert(
    Number.isSafeInteger(config.required_train_minimum),
    "invalid train minimum",
  );
  assert(
    Number.isSafeInteger(config.required_dev_minimum),
    "invalid dev minimum",
  );
  const names = [config.base.name, ...config.waves.map((wave) => wave.name)];
  assert(new Set(names).size === names.length, "source names are not unique");

  const outputRoot = resolve(cli.out);
  await mkdir(outputRoot, { recursive: false });
  const base = await loadSource(config.base, "base", configPath);
  const waves: LoadedSource[] = [];
  for (const wave of config.waves)
    waves.push(await loadSource(wave, "wave", configPath));
  const sources = [base, ...waves];

  const seenQualifiedIds = new Set<string>();
  const seenConversation = new Map<string, string>();
  const seenUserTurn = new Map<string, string>();
  const accepted: AcceptedRow[] = [];
  const mergedCatalog: Record<string, unknown>[] = [];
  let crossSourceRejections = 0;
  for (const source of sources) {
    for (const row of source.catalog) {
      const qualifiedId = `${source.name}/${row.candidate_id}`;
      assert(
        !seenQualifiedIds.has(qualifiedId),
        `duplicate qualified ID ${qualifiedId}`,
      );
      seenQualifiedIds.add(qualifiedId);
      const rendered = render(row.candidate.turns, qualifiedId);
      assert(
        sha256(rendered) === row.conversation_sha256,
        `${qualifiedId}: rendering drift`,
      );
      const normalizedUserTurns = row.candidate.turns
        .filter((turn) => turn.role === "user")
        .map((turn) => normalize(turn.content));
      assert(
        normalizedUserTurns.map(sha256).join("\0") ===
          row.normalized_user_turn_sha256.join("\0"),
        `${qualifiedId}: normalized user-turn drift`,
      );

      const mergeRejections: Record<string, unknown>[] = [];
      if (row.status !== "rejected") {
        const duplicateConversation = seenConversation.get(
          row.conversation_sha256,
        );
        if (duplicateConversation)
          mergeRejections.push({
            kind: "cross_source_conversation_duplicate",
            duplicate_of: duplicateConversation,
          });
        const duplicateTurns = normalizedUserTurns
          .map((turn) => ({
            turn_sha256: sha256(turn),
            duplicate_of: seenUserTurn.get(turn),
          }))
          .filter((item) => item.duplicate_of !== undefined);
        if (duplicateTurns.length > 0)
          mergeRejections.push({
            kind: "cross_source_user_turn_duplicate",
            duplicates: duplicateTurns,
          });
      }

      let mergedStatus: "train" | "dev" | "rejected" = "rejected";
      if (row.status !== "rejected" && mergeRejections.length === 0) {
        mergedStatus =
          source.kind === "base" && row.status === "dev" ? "dev" : "train";
        accepted.push({
          qualifiedId,
          source,
          row,
          rendered,
          normalizedUserTurns,
          mergedStatus,
        });
        seenConversation.set(row.conversation_sha256, qualifiedId);
        for (const turn of normalizedUserTurns)
          seenUserTurn.set(turn, qualifiedId);
      } else if (row.status !== "rejected") {
        crossSourceRejections += 1;
      }
      mergedCatalog.push({
        schema: "alpha-chat-foundations-v10-catalog-v1",
        qualified_id: qualifiedId,
        source: source.name,
        source_kind: source.kind,
        original_status: row.status,
        status: mergedStatus,
        merge_rejection_reasons: mergeRejections,
        original_rejection_reasons: row.rejection_reasons,
        batch_id: row.batch_id,
        focus: row.focus,
        conversation_sha256: row.conversation_sha256,
        tokens: row.tokens,
        turns: row.turns,
        normalized_user_turn_sha256: row.normalized_user_turn_sha256,
        review: row.review,
        candidate: row.candidate,
      });
    }
  }

  const train = accepted
    .filter((row) => row.mergedStatus === "train")
    .sort((left, right) =>
      sha256(`${config.seed}\0train\0${left.qualifiedId}`).localeCompare(
        sha256(`${config.seed}\0train\0${right.qualifiedId}`),
      ),
    );
  const dev = accepted.filter((row) => row.mergedStatus === "dev");
  assert(
    train.length >= config.required_train_minimum,
    `train population ${train.length} below required minimum`,
  );
  assert(
    dev.length >= config.required_dev_minimum,
    `dev population ${dev.length} below required minimum`,
  );
  assert(
    dev.every((row) => row.source.name === base.name),
    "new-wave row leaked into fixed dev",
  );
  assert(
    equalCounts(
      counted(dev.map((row) => row.rendered)),
      counted(lines(base.devText)),
    ),
    "fixed development population changed",
  );

  const trainPath = resolve(outputRoot, "train.txt");
  const devPath = resolve(outputRoot, "dev.txt");
  const catalogPath = resolve(outputRoot, "catalog.jsonl");
  await atomicWrite(
    trainPath,
    `${train.map((row) => row.rendered).join("\n")}\n`,
  );
  await atomicWrite(devPath, base.devText);
  await atomicWrite(
    catalogPath,
    `${mergedCatalog.map((row) => JSON.stringify(row)).join("\n")}\n`,
  );

  const bySource = Object.fromEntries(
    sources.map((source) => [
      source.name,
      {
        kind: source.kind,
        generated: source.catalog.length,
        originalAccepted: source.catalog.filter(
          (row) => row.status !== "rejected",
        ).length,
        mergedTrain: train.filter((row) => row.source.name === source.name)
          .length,
        mergedDev: dev.filter((row) => row.source.name === source.name).length,
        mergeRejected: mergedCatalog.filter(
          (row) =>
            row.source === source.name &&
            row.original_status !== "rejected" &&
            row.status === "rejected",
        ).length,
      },
    ]),
  );
  const sourceCommit = execFileSync("git", ["rev-parse", "HEAD"], {
    cwd: repo,
    encoding: "utf8",
  }).trim();
  const sourceTreeDirty =
    execFileSync("git", ["status", "--porcelain"], {
      cwd: repo,
      encoding: "utf8",
    }).trim().length > 0;
  const manifest = {
    schema: "alpha-chat-foundations-v10-corpus-v1",
    createdUtc: new Date().toISOString(),
    purpose:
      "increase unique reviewed conversational coverage while retaining the original untouched v8 development population",
    sourceCommit,
    sourceTreeDirty,
    inputs: {
      config: await evidence(configPath),
      sources: await Promise.all(
        sources.map(async (source) => ({
          name: source.name,
          kind: source.kind,
          manifest: await evidence(source.manifestPath),
          train: source.manifest.outputs.train,
          dev: source.manifest.outputs.dev,
          catalog: source.manifest.outputs.catalog,
        })),
      ),
    },
    recipe: {
      seed: config.seed,
      sourcePriority: [base.name, ...waves.map((wave) => wave.name)],
      baseDevelopmentDisposition:
        "retained byte-for-byte as the only development population",
      waveDevelopmentDisposition:
        "promoted to training after independent review",
      duplicateRule:
        "earlier source owns exact conversation hashes and normalized user turns",
      semanticOrTopicStringFilterApplied: false,
    },
    rows: {
      train: train.length,
      dev: dev.length,
      catalog: mergedCatalog.length,
      crossSourceRejections,
      bySource,
    },
    outputs: {
      train: { ...(await evidence(trainPath)), rows: train.length },
      dev: { ...(await evidence(devPath)), rows: dev.length },
      catalog: { ...(await evidence(catalogPath)), rows: mergedCatalog.length },
    },
    invariants: {
      allSourceCandidatesCataloged: true,
      allTrainingRowsIndependentlyReviewed: true,
      originalDevelopmentPopulationByteExact:
        (await sha256File(devPath)) ===
        (await sha256File(base.manifest.outputs.dev.path)),
      newWaveDevelopmentExcluded: true,
      exactConversationDeduplication: true,
      normalizedUserTurnDeduplication: true,
      inheritedExactNormalizedVisiblePromptExclusion: true,
      modelVisibleDelimitersInjectedAtCompileTime: true,
      assistantOnlyMaskAuditRequiredBeforeTraining: true,
      sealedFinalInspected: false,
    },
  };
  await atomicWrite(
    resolve(outputRoot, "manifest.json"),
    `${JSON.stringify(manifest, null, 2)}\n`,
  );
  process.stdout.write(
    `${JSON.stringify({ result: "PASS", rows: manifest.rows })}\n`,
  );
}

main().catch((error: unknown) => {
  process.stderr.write(
    `${error instanceof Error ? (error.stack ?? error.message) : String(error)}\n`,
  );
  process.exitCode = 1;
});
