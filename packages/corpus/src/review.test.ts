import assert from "node:assert/strict";
import { afterEach, test } from "node:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { closeLedger, createCampaign, openLedger, seedLedger, type Ledger } from "./db.js";
import { canonicalJson, sha256Bytes, stableId } from "./hash.js";
import {
  humanReviewStatus,
  prepareHumanReviewPacket,
  submitHumanReviewPacket
} from "./review.js";
import type { CampaignConfig, GeneratedItem, HumanReviewPacket, JsonValue } from "./types.js";

const temporaryHomes: string[] = [];

function temporaryHome(): string {
  const path = mkdtempSync(join(tmpdir(), "alpha-corpus-review-test-"));
  temporaryHomes.push(path);
  return path;
}

afterEach(() => {
  while (temporaryHomes.length > 0) rmSync(temporaryHomes.pop()!, { recursive: true, force: true });
});

const campaignConfig: CampaignConfig = {
  slug: "review-calibration",
  purpose: "human review test",
  workerModel: "gpt-5.4",
  criticModel: "disabled",
  maxGenerationCalls: 0,
  maxReviewCalls: 0,
  itemsPerFamily: 2,
  artifactLimitBytes: 15 * 1024 * 1024 * 1024
};

function item(itemKey: string, answer: string): GeneratedItem {
  return {
    itemKey,
    kind: "micro_dialogue",
    title: "Hidden title",
    primaryLens: "social_ontology",
    secondaryLenses: ["time"],
    transformation: "temporal_shift",
    intendedResponsePolicy: "Answer directly and stop.",
    difficulty: "introductory",
    messages: [
      { role: "user", content: "I graduated yesterday. Am I still the same person?" },
      { role: "assistant", content: answer }
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
    generatorNotes: "Must remain hidden during Pass A."
  };
}

async function seededReviewLedger(): Promise<Ledger> {
  const ledger = await openLedger(temporaryHome());
  await seedLedger(ledger);
  const campaignId = await createCampaign(ledger, campaignConfig);
  const family = await ledger.client.execute({
    sql: "SELECT id FROM concept_family WHERE slug = 'role-versus-bearer'"
  });
  const familyId = String(family.rows[0]!["id"]);
  const candidates = [
    { item: item("review-candidate-1", "Yes. The role changed; the person did not."), status: "structurally_valid" },
    { item: item("review-candidate-2", "No. You became a different person."), status: "structurally_rejected" }
  ];
  for (const entry of candidates) {
    const candidateId = stableId("candidate", `${campaignId}:${familyId}:${entry.item.itemKey}`);
    const versionId = stableId("candidatev", `${candidateId}:1`);
    const { hiddenContract, ...content } = entry.item;
    const contentJson = canonicalJson(content as unknown as JsonValue);
    await ledger.client.batch([
      {
        sql: `INSERT INTO candidate
              (id, campaign_id, family_id, item_key, kind, status, created_at, updated_at)
              VALUES (?, ?, ?, ?, ?, ?, '2026-07-31T00:00:00Z', '2026-07-31T00:00:00Z')`,
        args: [candidateId, campaignId, familyId, entry.item.itemKey, entry.item.kind, entry.status]
      },
      {
        sql: `INSERT INTO candidate_version
              (id, candidate_id, version, content_json, hidden_contract_json, content_sha256, created_at)
              VALUES (?, ?, 1, ?, ?, ?, '2026-07-31T00:00:00Z')`,
        args: [versionId, candidateId, contentJson,
          canonicalJson(hiddenContract as unknown as JsonValue), sha256Bytes(contentJson)]
      }
    ], "write");
  }
  return ledger;
}

function completePacket(packet: HumanReviewPacket): HumanReviewPacket {
  for (const assignment of packet.assignments) {
    assignment.response.outcome = "acceptable_as_rendered";
    assignment.response.summaryUserAim = "The user asks whether identity survives a role change.";
    assignment.response.summaryAssistantMove = "The assistant separates the persistent person from the temporary role.";
    for (const dimension of Object.keys(assignment.response.scores)) assignment.response.scores[dimension] = 3;
    assignment.response.questionPolicy = "not_applicable";
    assignment.response.missingClarification = "no";
    assignment.response.rationale = "The answer directly addresses the identity question and preserves the relevant distinction.";
    assignment.response.confidence = 3;
  }
  return packet;
}

test("Pass A packets blind contracts, include rejected candidates, and resume open assignments", async () => {
  const ledger = await seededReviewLedger();
  try {
    const prepared = await prepareHumanReviewPacket(ledger, {
      campaignSlug: campaignConfig.slug,
      reviewerAlias: "operator-test",
      pass: "A",
      limit: 2,
      seed: "fixed-review-order"
    });
    assert.equal(prepared.assignmentCount, 2);
    assert.equal(prepared.resumed, false);
    const packet = JSON.parse(readFileSync(prepared.packetPath, "utf8")) as HumanReviewPacket;
    assert.equal(packet.assignments.length, 2);
    for (const assignment of packet.assignments) {
      const visible = assignment.candidate as Record<string, unknown>;
      assert.deepEqual(Object.keys(visible).sort(), ["kind", "messages"]);
      assert.equal(JSON.stringify(visible).includes("Hidden title"), false);
      assert.equal(JSON.stringify(visible).includes("requiredCommitments"), false);
      assert.equal(assignment.response.outcome, null);
    }
    const resumed = await prepareHumanReviewPacket(ledger, {
      campaignSlug: campaignConfig.slug,
      reviewerAlias: "operator-test",
      pass: "A",
      limit: 2,
      seed: "ignored-on-resume"
    });
    assert.equal(resumed.resumed, true);
    assert.equal(resumed.sessionId, prepared.sessionId);
    const assignments = await ledger.client.execute("SELECT COUNT(*) AS count FROM review_assignment");
    assert.equal(Number(assignments.rows[0]!["count"]), 2);
  } finally {
    closeLedger(ledger);
  }
});

test("human submission is append-only evidence and never promotes candidate or training state", async () => {
  const ledger = await seededReviewLedger();
  try {
    const prepared = await prepareHumanReviewPacket(ledger, {
      campaignSlug: campaignConfig.slug,
      reviewerAlias: "operator-test",
      pass: "A",
      limit: 2,
      seed: "fixed-review-order"
    });
    const packet = completePacket(JSON.parse(readFileSync(prepared.packetPath, "utf8")) as HumanReviewPacket);
    writeFileSync(prepared.packetPath, `${canonicalJson(packet as unknown as JsonValue)}\n`);
    const result = await submitHumanReviewPacket(ledger, prepared.packetPath);
    assert.equal(result.submitted, 2);
    const status = await humanReviewStatus(ledger, campaignConfig.slug);
    assert.equal(status.assignments["A:completed"], 2);
    assert.equal(status.reviews["A:acceptable_as_rendered"], 2);
    assert.equal(status.humanReviewArtifacts, 1);
    assert.equal(status.candidateStatuses["structurally_valid"], 1);
    assert.equal(status.candidateStatuses["structurally_rejected"], 1);
    assert.equal(status.releaseMembers, 0);
    assert.equal(status.trainingExposures, 0);
    const humanReviews = await ledger.client.execute(
      "SELECT COUNT(*) AS count FROM review WHERE reviewer_actor_id IS NOT NULL AND reviewer_model_revision_id IS NULL"
    );
    assert.equal(Number(humanReviews.rows[0]!["count"]), 2);
    await assert.rejects(submitHumanReviewPacket(ledger, prepared.packetPath), /not open/);

    const passB = await prepareHumanReviewPacket(ledger, {
      campaignSlug: campaignConfig.slug,
      reviewerAlias: "operator-test",
      pass: "B",
      limit: 2,
      seed: "pass-b-order"
    });
    const revealed = JSON.parse(readFileSync(passB.packetPath, "utf8")) as HumanReviewPacket;
    assert.equal(revealed.assignments.length, 2);
    assert.ok(JSON.stringify(revealed.assignments[0]!.candidate).includes("requiredCommitments"));
    assert.ok(JSON.stringify(revealed.assignments[0]!.candidate).includes("structuralStatus"));
  } finally {
    closeLedger(ledger);
  }
});

test("submission rejects a changed candidate version hash", async () => {
  const ledger = await seededReviewLedger();
  try {
    const prepared = await prepareHumanReviewPacket(ledger, {
      campaignSlug: campaignConfig.slug,
      reviewerAlias: "operator-test",
      pass: "A",
      limit: 1,
      seed: "fixed-review-order"
    });
    const packet = completePacket(JSON.parse(readFileSync(prepared.packetPath, "utf8")) as HumanReviewPacket);
    packet.assignments[0]!.candidateContentSha256 = "0".repeat(64);
    writeFileSync(prepared.packetPath, `${canonicalJson(packet as unknown as JsonValue)}\n`);
    await assert.rejects(submitHumanReviewPacket(ledger, prepared.packetPath), /candidate version changed/);
    const reviews = await ledger.client.execute("SELECT COUNT(*) AS count FROM review");
    assert.equal(Number(reviews.rows[0]!["count"]), 0);
  } finally {
    closeLedger(ledger);
  }
});
