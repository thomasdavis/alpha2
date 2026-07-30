import assert from "node:assert/strict";
import { afterEach, test } from "node:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import {
  checkCampaignStorage,
  closeLedger,
  createCampaign,
  createTask,
  listFamilies,
  loadRecordedStructuredResponse,
  openLedger,
  putBlob,
  recordCandidate,
  recordStructuredCall,
  seedLedger,
  validateLedger,
  type Ledger
} from "./db.js";
import { generationEnvelopeSchema } from "./schemas.js";
import { categorySeeds, transformationSeeds } from "./seeds.js";
import { writeAuditPacket } from "./report.js";
import type { CampaignConfig, GeneratedItem, StructuredCallResult } from "./types.js";
import { validateCandidate } from "./validate.js";

const temporaryHomes: string[] = [];

function temporaryHome(): string {
  const path = mkdtempSync(join(tmpdir(), "alpha-corpus-test-"));
  temporaryHomes.push(path);
  return path;
}

afterEach(() => {
  while (temporaryHomes.length > 0) {
    rmSync(temporaryHomes.pop()!, { recursive: true, force: true });
  }
});

async function seededLedger(): Promise<Ledger> {
  const ledger = await openLedger(temporaryHome());
  await seedLedger(ledger);
  return ledger;
}

const campaignConfig: CampaignConfig = {
  slug: "test-calibration",
  purpose: "test",
  workerModel: "gpt-5.4",
  criticModel: "disabled",
  maxGenerationCalls: 2,
  maxReviewCalls: 0,
  itemsPerFamily: 1,
  artifactLimitBytes: 15 * 1024 * 1024 * 1024
};

const validItem: GeneratedItem = {
  itemKey: "role-versus-bearer-natural-dialogue-01",
  kind: "micro_dialogue",
  title: "After graduation",
  primaryLens: "social_ontology",
  secondaryLenses: ["time"],
  transformation: "temporal_shift",
  intendedResponsePolicy: "answer-and-stop",
  difficulty: "introductory",
  messages: [
    { role: "user", content: "I graduated yesterday. Am I still the same person if I am no longer a student?" },
    { role: "assistant", content: "Yes. Graduating ends a role you held; it does not replace the person who held it." }
  ],
  linguisticPair: null,
  hiddenContract: {
    requiredCommitments: ["The person persists after the student role ends."],
    prohibitedCommitments: ["Graduation ends the person's identity."],
    preserve: ["Person identity."],
    change: ["Current student status."],
    admissibleAnalyses: ["Student is a time-qualified role."],
    discriminatingEvidence: []
  },
  generatorNotes: "Compact temporal role contrast."
};

function successfulCall(): StructuredCallResult {
  const stamp = new Date().toISOString();
  return {
    startedAt: stamp,
    completedAt: stamp,
    exitCode: 0,
    stdout: Buffer.from('{"type":"turn.completed","usage":{"input_tokens":20,"output_tokens":30}}\n'),
    stderr: Buffer.alloc(0),
    lastMessage: Buffer.from('{"familySlug":"role-versus-bearer","items":[],"batchNotes":"test"}'),
    parsed: { familySlug: "role-versus-bearer", items: [], batchNotes: "test" },
    usage: { inputTokens: 20, cachedInputTokens: 0, outputTokens: 30 },
    callDirectory: "/tmp/test-call",
    commandArgs: ["exec", "-m", "gpt-5.4"]
  };
}

test("fresh ledger migrates, seeds idempotently, and remains internally valid", async () => {
  const ledger = await seededLedger();
  try {
    await seedLedger(ledger);
    const report = await validateLedger(ledger);
    assert.equal(report.integrity, "ok");
    assert.equal(report.foreignKeyViolations, 0);
    assert.deepEqual(report.missingTables, []);
    assert.deepEqual(report.missingViews, []);
    assert.deepEqual(report.missingBlobs, []);
    assert.deepEqual(report.corruptBlobs, []);
    assert.equal((await listFamilies(ledger)).length, 6);
    const publicRows = await ledger.client.execute("SELECT COUNT(*) AS count FROM public_training_candidate");
    assert.equal(Number(publicRows.rows[0]!["count"]), 0);
    const categoryCount = await ledger.client.execute("SELECT COUNT(*) AS count FROM category");
    assert.equal(Number(categoryCount.rows[0]!["count"]), categorySeeds.length);
  } finally {
    closeLedger(ledger);
  }
});

test("versioned scientific records reject destructive mutation", async () => {
  const ledger = await seededLedger();
  try {
    await assert.rejects(
      ledger.client.execute("UPDATE program_version SET objective = 'overwritten'"),
      /append-only/
    );
    await assert.rejects(ledger.client.execute("DELETE FROM family_version"), /append-only/);
  } finally {
    closeLedger(ledger);
  }
});

test("content-addressed artifacts round-trip and validate", async () => {
  const ledger = await seededLedger();
  try {
    const digest = await putBlob(ledger, "preserve this exact output", "text/plain");
    const row = await ledger.client.execute({ sql: "SELECT relative_path FROM blob WHERE sha256 = ?", args: [digest] });
    assert.equal(readFileSync(join(ledger.paths.home, String(row.rows[0]!["relative_path"])), "utf8"), "preserve this exact output");
    assert.deepEqual((await validateLedger(ledger)).corruptBlobs, []);
  } finally {
    closeLedger(ledger);
  }
});

test("validator keeps delimiters out of natural messages and records metadata independently", () => {
  const lenses = new Set(categorySeeds.map((seed) => seed.slug));
  const transformations = new Set(transformationSeeds.map(([slug]) => slug));
  assert.equal(
    validateCandidate(validItem, "role-versus-bearer", "role-versus-bearer-natural-dialogue-", lenses, transformations).valid,
    true
  );
  const leaked: GeneratedItem = {
    ...validItem,
    itemKey: "role-versus-bearer-natural-dialogue-02",
    messages: [{ role: "user", content: "<assistant> tell me" }, validItem.messages[1]!]
  };
  const validation = validateCandidate(
    leaked,
    "role-versus-bearer",
    "role-versus-bearer-natural-dialogue-",
    lenses,
    transformations
  );
  assert.equal(validation.valid, false);
  assert.ok(validation.findings.some((finding) => finding.code === "delimiter_leak"));
});

test("model-call provenance, candidates, and human audit packet remain reconstructable", async () => {
  const ledger = await seededLedger();
  try {
    const campaignId = await createCampaign(ledger, campaignConfig);
    await assert.rejects(
      createCampaign(ledger, { ...campaignConfig, workerModel: "gpt-5.5" }),
      /frozen contract/
    );
    const family = (await listFamilies(ledger)).find((entry) => entry.slug === "role-versus-bearer")!;
    const task = await createTask(ledger, campaignId, family.id, "test", "test:call:1", "gpt-5.4");
    const callId = await recordStructuredCall(
      ledger,
      task.id,
      "gpt-5.4",
      "worker",
      "test-prompt",
      "Generate one test item.",
      "generation-envelope",
      generationEnvelopeSchema,
      successfulCall(),
      1
    );
    const recorded = await loadRecordedStructuredResponse<{ familySlug: string }>(ledger, task.id);
    assert.equal(recorded?.callId, callId);
    assert.equal(recorded?.parsed.familySlug, "role-versus-bearer");
    const validation = validateCandidate(
      validItem,
      family.slug,
      "role-versus-bearer-natural-dialogue-",
      new Set(categorySeeds.map((seed) => seed.slug)),
      new Set(transformationSeeds.map(([slug]) => slug))
    );
    await recordCandidate(ledger, campaignId, family.id, callId, validItem, validation);
    await recordCandidate(ledger, campaignId, family.id, callId, validItem, validation);
    const candidateCount = await ledger.client.execute("SELECT COUNT(*) AS count FROM candidate");
    const transitionCount = await ledger.client.execute("SELECT COUNT(*) AS count FROM quality_state_transition");
    assert.equal(Number(candidateCount.rows[0]!["count"]), 1);
    assert.equal(Number(transitionCount.rows[0]!["count"]), 1);
    const packet = await writeAuditPacket(ledger, campaignConfig.slug);
    assert.equal(packet.candidateCount, 1);
    assert.match(readFileSync(packet.markdownPath, "utf8"), /Graduating ends a role/);
    assert.equal((await validateLedger(ledger)).foreignKeyViolations, 0);
  } finally {
    closeLedger(ledger);
  }
});

test("the corpus-owned storage limit causes a resumable campaign pause", async () => {
  const ledger = await seededLedger();
  try {
    const campaignId = await createCampaign(ledger, { ...campaignConfig, slug: "storage-pause", artifactLimitBytes: 1 });
    assert.equal(await checkCampaignStorage(ledger, campaignId), false);
    const row = await ledger.client.execute({ sql: "SELECT status FROM generation_campaign WHERE id = ?", args: [campaignId] });
    assert.equal(String(row.rows[0]!["status"]), "paused_storage");
  } finally {
    closeLedger(ledger);
  }
});
