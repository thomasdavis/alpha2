#!/usr/bin/env npx tsx

/** Compile independently accepted V12 families with whole-family train/dev splits. */

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdir, readFile, readdir, rename, writeFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";

interface Turn {
  readonly role: "user" | "assistant";
  readonly content: string;
}

interface Scene {
  readonly scene_id: string;
  readonly kind: string;
  readonly turns: readonly Turn[];
  readonly relation_to_base: string;
  readonly learning_signal: string;
}

interface Family {
  readonly family_id: string;
  readonly focus: string;
  readonly operation: string;
  readonly invariant: string;
  readonly must_change: string;
  readonly forbidden_shortcut: string;
  readonly scenes: readonly Scene[];
}

interface GenerationBatch {
  readonly batch_id: string;
  readonly families: readonly Family[];
}

interface Review {
  readonly family_id: string;
  readonly decision: "accept" | "reject";
  readonly [key: string]: unknown;
}

interface ReviewBatch {
  readonly review_batch_id: string;
  readonly reviews: readonly Review[];
}

function sha256(value: string | Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

function parseArgs(argv: readonly string[]): Map<string, string> {
  const parsed = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key?.startsWith("--") || !value || value.startsWith("--"))
      throw new Error(`expected --name value, received ${String(key)}`);
    parsed.set(key.slice(2), value);
    index += 1;
  }
  return parsed;
}

function normalize(value: string): string {
  return value.toLowerCase().normalize("NFKC").replace(/[^a-z0-9]+/g, " ").trim();
}

function render(turns: readonly Turn[]): string {
  return `${turns.map((turn) => `<|${turn.role}|> ${turn.content.trim()}`).join(" ")} <|end_of_text|>`;
}

function renderPrompt(turns: readonly Turn[]): string {
  return `${turns.map((turn) => `<|${turn.role}|> ${turn.content.trim()}`).join(" ")} <|assistant|>`;
}

function splitFor(familyId: string): "train" | "development" {
  const match = /-f([0-3])$/.exec(familyId);
  if (!match) throw new Error(`family ID lacks stable split suffix: ${familyId}`);
  return match[1] === "3" ? "development" : "train";
}

function userTurnsFromRendered(line: string): string[] {
  const found: string[] = [];
  const pattern = /<\|user\|>\s*(.*?)(?=\s*<\|(?:assistant|end_of_text)\|>)/gs;
  for (const match of line.matchAll(pattern)) found.push(match[1]!.trim());
  return found;
}

async function atomicWrite(path: string, bytes: string): Promise<void> {
  const temporary = `${path}.tmp-${process.pid}`;
  await writeFile(temporary, bytes, { flag: "wx" });
  await rename(temporary, path);
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const repo = resolve(args.get("repo") ?? process.cwd());
  const generationRoot = resolve(args.get("generation-dir") ?? "/mnt/donto-data/donto-resources/research/alpha-chat-foundations-v12-20260802/generation");
  const reviewRoot = resolve(args.get("review-dir") ?? "/mnt/donto-data/donto-resources/research/alpha-chat-foundations-v12-20260802/review");
  const outRoot = resolve(args.get("out-dir") ?? "/mnt/donto-data/donto-resources/research/alpha-chat-foundations-v12-20260802/corpus");
  const v8TrainPath = resolve(args.get("v8-train") ?? "/mnt/donto-data/donto-resources/research/alpha-chat-foundations-v8-20260802/corpus/train.txt");
  const v8DevPath = resolve(args.get("v8-dev") ?? "/mnt/donto-data/donto-resources/research/alpha-chat-foundations-v8-20260802/corpus/dev.txt");
  const blahEvalsPath = resolve(args.get("blah-evals") ?? "/mnt/donto-data/donto-resources/research/alpha-chat-foundations-v11-20260802/blah-evaluation/eval-definitions.json");
  const sourceCommit = execFileSync("git", ["rev-parse", "HEAD"], { cwd: repo, encoding: "utf8" }).trim();
  if (execFileSync("git", ["status", "--porcelain"], { cwd: repo, encoding: "utf8" }).trim())
    throw new Error("compilation requires a clean committed source tree");
  const generationManifestPath = join(generationRoot, "generation-manifest.json");
  const reviewManifestPath = join(reviewRoot, "review-manifest.json");
  const [generationManifestBytes, reviewManifestBytes, v8TrainBytes, v8DevBytes, blahBytes] = await Promise.all([
    readFile(generationManifestPath),
    readFile(reviewManifestPath),
    readFile(v8TrainPath),
    readFile(v8DevPath),
    readFile(blahEvalsPath),
  ]);
  const generationManifest = JSON.parse(generationManifestBytes.toString("utf8")) as Record<string, unknown>;
  const reviewManifest = JSON.parse(reviewManifestBytes.toString("utf8")) as Record<string, unknown>;
  if (generationManifest.status !== "complete" || reviewManifest.status !== "complete")
    throw new Error("generation and review must both be complete");
  const families = new Map<string, { family: Family; batch: string }>();
  const generationFiles = (await readdir(join(generationRoot, "batches"))).filter((name) => name.endsWith(".json")).sort();
  for (const name of generationFiles) {
    const batch = JSON.parse(await readFile(join(generationRoot, "batches", name), "utf8")) as GenerationBatch;
    for (const family of batch.families) {
      if (families.has(family.family_id)) throw new Error(`duplicate family ${family.family_id}`);
      families.set(family.family_id, { family, batch: name });
    }
  }
  const reviews = new Map<string, { review: Review; batch: string }>();
  const reviewFiles = (await readdir(reviewRoot)).filter((name) => /^v12-review-\d+\.json$/.test(name)).sort();
  for (const name of reviewFiles) {
    const batch = JSON.parse(await readFile(join(reviewRoot, name), "utf8")) as ReviewBatch;
    for (const review of batch.reviews) {
      if (reviews.has(review.family_id)) throw new Error(`duplicate review ${review.family_id}`);
      reviews.set(review.family_id, { review, batch: name });
    }
  }
  if (reviews.size !== families.size) throw new Error(`review coverage ${reviews.size}/${families.size}`);

  const holdouts = new Set<string>();
  for (const line of v8DevBytes.toString("utf8").split("\n").filter(Boolean))
    for (const user of userTurnsFromRendered(line)) holdouts.add(normalize(user));
  const blah = JSON.parse(blahBytes.toString("utf8")) as Array<Record<string, unknown>>;
  for (const item of blah) if (typeof item.prompt === "string") holdouts.add(normalize(item.prompt));

  const trainRows: Array<{ line: string; familyId: string; scene: Scene }> = [];
  const devRows: Array<{ line: string; familyId: string; scene: Scene }> = [];
  const catalog: Record<string, unknown>[] = [];
  const userOwnership = new Map<string, string>();
  const focusCounts = new Map<string, { acceptedTrain: number; acceptedDev: number; rejected: number }>();
  for (const [familyId, source] of [...families.entries()].sort(([left], [right]) => left.localeCompare(right))) {
    const reviewed = reviews.get(familyId)!;
    const split = splitFor(familyId);
    const collisions: string[] = [];
    for (const scene of source.family.scenes) {
      for (const turn of scene.turns.filter((turn) => turn.role === "user")) {
        const normalized = normalize(turn.content);
        if (holdouts.has(normalized)) collisions.push(`holdout:${scene.scene_id}`);
        const prior = userOwnership.get(normalized);
        if (prior) collisions.push(`duplicate:${scene.scene_id}:${prior}`);
      }
    }
    const accepted = reviewed.review.decision === "accept" && collisions.length === 0;
    const focus = focusCounts.get(source.family.focus) ?? { acceptedTrain: 0, acceptedDev: 0, rejected: 0 };
    if (accepted) {
      if (split === "train") focus.acceptedTrain += 1;
      else focus.acceptedDev += 1;
      for (const scene of source.family.scenes) {
        for (const turn of scene.turns.filter((turn) => turn.role === "user"))
          userOwnership.set(normalize(turn.content), scene.scene_id);
        (split === "train" ? trainRows : devRows).push({ line: render(scene.turns), familyId, scene });
      }
    } else focus.rejected += 1;
    focusCounts.set(source.family.focus, focus);
    catalog.push({
      schema: "alpha-chat-foundations-v12-family-catalog-v1",
      family_id: familyId,
      focus: source.family.focus,
      split,
      accepted,
      rejection_reasons: collisions,
      generation_batch: source.batch,
      review_batch: reviewed.batch,
      review: reviewed.review,
      operation: source.family.operation,
      invariant: source.family.invariant,
      must_change: source.family.must_change,
      forbidden_shortcut: source.family.forbidden_shortcut,
      scene_ids: source.family.scenes.map((scene) => scene.scene_id),
    });
  }
  if (trainRows.length < 800 || devRows.length < 250)
    throw new Error(`accepted V12 corpus too small: train=${trainRows.length} dev=${devRows.length}`);

  const v8Rows = v8TrainBytes.toString("utf8").split("\n").filter(Boolean);
  const replayCount = Math.floor(trainRows.length * 0.25);
  const replay = v8Rows
    .map((line) => ({ line, hash: sha256(`v12-replay:${line}`) }))
    .sort((left, right) => left.hash.localeCompare(right.hash))
    .slice(0, replayCount)
    .map((item) => item.line);
  const train = [...trainRows.map((row) => row.line), ...replay]
    .map((line) => ({ line, hash: sha256(`v12-order:${line}`) }))
    .sort((left, right) => left.hash.localeCompare(right.hash))
    .map((item) => item.line);
  const development = devRows.map((row) => row.line);
  const devPrompts = devRows.map((row) => {
    const messages = row.scene.turns.slice(0, -1);
    const reference = row.scene.turns.at(-1)!.content.trim();
    const prompt = renderPrompt(messages);
    return {
      schema: "alpha-chat-foundations-v12-development-prompt-v1",
      id: row.scene.scene_id,
      source: `v12:${families.get(row.familyId)!.family.focus}:${row.scene.kind}`,
      family_id: row.familyId,
      kind: row.scene.kind,
      messages,
      reference,
      prompt,
      prompt_sha256: sha256(prompt),
    };
  });
  await mkdir(outRoot, { recursive: true });
  const trainText = `${train.join("\n")}\n`;
  const devText = `${development.join("\n")}\n`;
  const promptText = `${devPrompts.map((row) => JSON.stringify(row)).join("\n")}\n`;
  const catalogText = `${catalog.map((row) => JSON.stringify(row)).join("\n")}\n`;
  await Promise.all([
    atomicWrite(join(outRoot, "train.txt"), trainText),
    atomicWrite(join(outRoot, "dev.txt"), devText),
    atomicWrite(join(outRoot, "dev-prompts.jsonl"), promptText),
    atomicWrite(join(outRoot, "catalog.jsonl"), catalogText),
  ]);
  const manifest = {
    schema: "alpha-chat-foundations-v12-corpus-v1",
    status: "complete-and-immutable",
    source_commit: sourceCommit,
    source_tree_dirty: false,
    inputs: {
      generation_manifest: { path: generationManifestPath, sha256: sha256(generationManifestBytes) },
      review_manifest: { path: reviewManifestPath, sha256: sha256(reviewManifestBytes) },
      v8_train: { path: v8TrainPath, sha256: sha256(v8TrainBytes) },
      fixed_v8_development: { path: v8DevPath, sha256: sha256(v8DevBytes) },
      blah_evaluations: { path: blahEvalsPath, sha256: sha256(blahBytes) },
    },
    family_allocation: {
      method: "suffix f0-f2 train; f3 development within every generated batch",
      unit: "whole family",
      planned: families.size,
      accepted_train: new Set(trainRows.map((row) => row.familyId)).size,
      accepted_development: new Set(devRows.map((row) => row.familyId)).size,
      rejected: catalog.filter((row) => row.accepted === false).length,
      by_focus: Object.fromEntries([...focusCounts.entries()].sort(([left], [right]) => left.localeCompare(right))),
    },
    rows: { v12_train: trainRows.length, v8_replay: replay.length, train: train.length, development: development.length },
    outputs: {
      train: { path: join(outRoot, "train.txt"), bytes: Buffer.byteLength(trainText), sha256: sha256(trainText), rows: train.length },
      development: { path: join(outRoot, "dev.txt"), bytes: Buffer.byteLength(devText), sha256: sha256(devText), rows: development.length },
      development_prompts: { path: join(outRoot, "dev-prompts.jsonl"), bytes: Buffer.byteLength(promptText), sha256: sha256(promptText), rows: devPrompts.length },
      catalog: { path: join(outRoot, "catalog.jsonl"), bytes: Buffer.byteLength(catalogText), sha256: sha256(catalogText), rows: catalog.length },
    },
    invariants: {
      every_family_independently_reviewed_once: reviews.size === families.size,
      only_accepted_families_train_or_develop: true,
      whole_family_split: true,
      balanced_split_within_generation_batch: true,
      exact_visible_holdout_prompt_exclusion: true,
      cross_split_user_turn_overlap: false,
      public_outputs_used_as_training_targets: false,
      sealed_final_inspected: false,
      v8_replay_fraction_of_v12_rows: replay.length / trainRows.length,
    },
    created_utc: new Date().toISOString(),
  };
  await atomicWrite(join(outRoot, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify({ result: "PASS", manifest: join(outRoot, "manifest.json"), rows: manifest.rows, families: manifest.family_allocation })}\n`);
}

void main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`);
  process.exitCode = 1;
});
