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
import { humanReviewPacketMatchesEnvelope } from "./review-contract.js";
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

    await assert.rejects(
      prepareHumanReviewPacket(ledger, {
        campaignSlug: campaignConfig.slug,
        reviewerAlias: "operator-test",
        pass: "B",
        limit: 2,
        seed: "premature-pass-b"
      }),
      /Pass B is locked.*2\/2 candidate reviews, 0\/2 repeat-stability rows/
    );
    const repeatPreparation = await prepareHumanReviewPacket(ledger, {
      campaignSlug: campaignConfig.slug,
      reviewerAlias: "operator-test",
      pass: "A",
      limit: 2,
      seed: "complete-repeat-gate"
    });
    const repeatPacket = completePacket(
      JSON.parse(readFileSync(repeatPreparation.packetPath, "utf8")) as HumanReviewPacket
    );
    assert.equal(JSON.stringify(repeatPacket).includes("hidden_repeat"), false);
    assert.equal(JSON.stringify(repeatPacket).includes("sourceReviewId"), false);
    writeFileSync(repeatPreparation.packetPath, `${canonicalJson(repeatPacket as unknown as JsonValue)}\n`);
    const repeatResult = await submitHumanReviewPacket(ledger, repeatPreparation.packetPath);
    assert.equal(repeatResult.primaryReviews, 0);
    assert.equal(repeatResult.repeatResponses, 2);

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

test("submission accepts response-only changes but rejects altered visible packet envelopes", async () => {
  const ledger = await seededReviewLedger();
  try {
    const prepared = await prepareHumanReviewPacket(ledger, {
      campaignSlug: campaignConfig.slug,
      reviewerAlias: "operator-test",
      pass: "A",
      limit: 1,
      seed: "immutable-envelope"
    });
    const exported = JSON.parse(readFileSync(prepared.packetPath, "utf8")) as HumanReviewPacket;
    const completed = completePacket(JSON.parse(JSON.stringify(exported)) as HumanReviewPacket);
    assert.equal(humanReviewPacketMatchesEnvelope(completed, exported), true);

    const alteredCandidate = JSON.parse(JSON.stringify(completed)) as HumanReviewPacket;
    const visible = alteredCandidate.assignments[0]!.candidate as {
      messages: Array<{ role: string; content: string }>;
    };
    visible.messages[1]!.content = "This is not the assistant response that was exported for review.";
    assert.equal(humanReviewPacketMatchesEnvelope(alteredCandidate, exported), false);
    writeFileSync(prepared.packetPath, `${canonicalJson(alteredCandidate as unknown as JsonValue)}\n`);
    await assert.rejects(
      submitHumanReviewPacket(ledger, prepared.packetPath),
      /immutable envelope does not match an exported packet/
    );

    const alteredPresentationId = JSON.parse(JSON.stringify(completed)) as HumanReviewPacket;
    alteredPresentationId.assignments[0]!.presentationId = "presentation_tampered";
    assert.equal(humanReviewPacketMatchesEnvelope(alteredPresentationId, exported), false);
    writeFileSync(prepared.packetPath, `${canonicalJson(alteredPresentationId as unknown as JsonValue)}\n`);
    await assert.rejects(
      submitHumanReviewPacket(ledger, prepared.packetPath),
      /Unknown review assignment/
    );

    const alteredPresentation = JSON.parse(JSON.stringify(completed)) as HumanReviewPacket;
    alteredPresentation.assignments[0]!.opaqueItemId = "opaque_tampered";
    assert.equal(humanReviewPacketMatchesEnvelope(alteredPresentation, exported), false);
    writeFileSync(prepared.packetPath, `${canonicalJson(alteredPresentation as unknown as JsonValue)}\n`);
    await assert.rejects(
      submitHumanReviewPacket(ledger, prepared.packetPath),
      /immutable envelope does not match an exported packet/
    );

    const evidence = await ledger.client.execute(`
      SELECT
        (SELECT COUNT(*) FROM review) AS reviews,
        (SELECT COUNT(*) FROM review_presentation_response) AS presentation_responses,
        (SELECT COUNT(*) FROM raw_artifact WHERE kind LIKE 'human_review_submission_pass_%') AS submissions
    `);
    assert.equal(Number(evidence.rows[0]!["reviews"]), 0);
    assert.equal(Number(evidence.rows[0]!["presentation_responses"]), 0);
    assert.equal(Number(evidence.rows[0]!["submissions"]), 0);

    writeFileSync(prepared.packetPath, `${canonicalJson(completed as unknown as JsonValue)}\n`);
    const accepted = await submitHumanReviewPacket(ledger, prepared.packetPath);
    assert.equal(accepted.submitted, 1);
    assert.equal(accepted.packetEnvelopeSha256, prepared.packetSha256);
  } finally {
    closeLedger(ledger);
  }
});

test("Pass A hides repeat identity, records stability separately, and does not inflate candidate reviews", async () => {
  const ledger = await seededReviewLedger();
  try {
    const first = await prepareHumanReviewPacket(ledger, {
      campaignSlug: campaignConfig.slug,
      reviewerAlias: "operator-test",
      pass: "A",
      limit: 1,
      seed: "repeat-primary-session"
    });
    const firstPacket = completePacket(
      JSON.parse(readFileSync(first.packetPath, "utf8")) as HumanReviewPacket
    );
    assert.equal(typeof firstPacket.assignments[0]!.presentationId, "string");
    writeFileSync(first.packetPath, `${canonicalJson(firstPacket as unknown as JsonValue)}\n`);
    const firstResult = await submitHumanReviewPacket(ledger, first.packetPath);
    assert.equal(firstResult.primaryReviews, 1);
    assert.equal(firstResult.repeatResponses, 0);
    await assert.rejects(
      prepareHumanReviewPacket(ledger, {
        campaignSlug: campaignConfig.slug,
        reviewerAlias: "operator-test",
        pass: "B",
        limit: 1,
        seed: "individual-candidate-leak-attempt"
      }),
      /Pass B is locked.*1\/2 candidate reviews, 0\/2 repeat-stability rows/
    );
    const prematurePassB = await ledger.client.execute(
      `SELECT COUNT(*) AS count FROM review_assignment
       WHERE json_extract(blindness_json, '$.pass') = 'B'`
    );
    assert.equal(Number(prematurePassB.rows[0]!["count"]), 0);

    const second = await prepareHumanReviewPacket(ledger, {
      campaignSlug: campaignConfig.slug,
      reviewerAlias: "operator-test",
      pass: "A",
      limit: 2,
      seed: "repeat-mixed-session"
    });
    const secondPacket = JSON.parse(readFileSync(second.packetPath, "utf8")) as HumanReviewPacket;
    assert.equal(secondPacket.assignments.length, 2);
    assert.equal(secondPacket.assignments.every((assignment) => typeof assignment.presentationId === "string"), true);
    assert.equal(JSON.stringify(secondPacket).includes("hidden_repeat"), false);
    assert.equal(JSON.stringify(secondPacket).includes("sourceReviewId"), false);

    const presentationsBefore = await ledger.client.execute({
      sql: `SELECT presentation_kind, COUNT(*) AS count FROM review_presentation
            WHERE session_id = ? GROUP BY presentation_kind ORDER BY presentation_kind`,
      args: [second.sessionId]
    });
    assert.deepEqual(
      presentationsBefore.rows.map((row) => [String(row["presentation_kind"]), Number(row["count"])]),
      [["hidden_repeat", 1], ["primary", 1]]
    );

    completePacket(secondPacket);
    writeFileSync(second.packetPath, `${canonicalJson(secondPacket as unknown as JsonValue)}\n`);
    const secondResult = await submitHumanReviewPacket(ledger, second.packetPath);
    assert.equal(secondResult.submitted, 2);
    assert.equal(secondResult.primaryReviews, 1);
    assert.equal(secondResult.repeatResponses, 1);

    const reviewCount = await ledger.client.execute("SELECT COUNT(*) AS count FROM review");
    const assignmentCount = await ledger.client.execute("SELECT COUNT(*) AS count FROM review_assignment");
    const responseCount = await ledger.client.execute("SELECT COUNT(*) AS count FROM review_presentation_response");
    assert.equal(Number(reviewCount.rows[0]!["count"]), 2);
    assert.equal(Number(assignmentCount.rows[0]!["count"]), 2);
    assert.equal(Number(responseCount.rows[0]!["count"]), 3);

    const stability = await ledger.client.execute(
      "SELECT outcome_match, question_policy_match, missing_clarification_match, confidence_delta, dimension_exact_rate, mean_absolute_score_delta FROM review_repeat_stability"
    );
    assert.equal(stability.rows.length, 1);
    assert.equal(Number(stability.rows[0]!["outcome_match"]), 1);
    assert.equal(Number(stability.rows[0]!["question_policy_match"]), 1);
    assert.equal(Number(stability.rows[0]!["missing_clarification_match"]), 1);
    assert.equal(Number(stability.rows[0]!["confidence_delta"]), 0);
    assert.equal(Number(stability.rows[0]!["dimension_exact_rate"]), 1);
    assert.equal(Number(stability.rows[0]!["mean_absolute_score_delta"]), 0);

    const status = await humanReviewStatus(ledger, campaignConfig.slug);
    assert.equal(status.presentations["primary:completed"], 2);
    assert.equal(status.presentations["hidden_repeat:completed"], 1);
    assert.equal(status.repeatStabilityRows, 1);
    assert.equal(status.releaseMembers, 0);
    assert.equal(status.trainingExposures, 0);

    await assert.rejects(
      ledger.client.execute("UPDATE review_presentation_response SET outcome = 'uncertain'"),
      /append-only/
    );
    await assert.rejects(submitHumanReviewPacket(ledger, second.packetPath), /not open/);
  } finally {
    closeLedger(ledger);
  }
});
