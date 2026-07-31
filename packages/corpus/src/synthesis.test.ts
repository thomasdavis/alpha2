import assert from "node:assert/strict";
import { afterEach, test } from "node:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { closeLedger, createCampaign, openLedger, seedLedger, type Ledger } from "./db.js";
import { canonicalJson, sha256Bytes, stableId } from "./hash.js";
import { prepareHumanReviewPacket, submitHumanReviewPacket } from "./review.js";
import {
  familySynthesisStatus,
  prepareFamilySynthesisPacket,
  submitFamilySynthesisPacket
} from "./synthesis.js";
import type { FamilySynthesisPacket } from "./synthesis-contract.js";
import type { CampaignConfig, GeneratedItem, HumanReviewPacket, JsonValue } from "./types.js";

const temporaryHomes: string[] = [];

function temporaryHome(): string {
  const path = mkdtempSync(join(tmpdir(), "alpha-corpus-synthesis-test-"));
  temporaryHomes.push(path);
  return path;
}

afterEach(() => {
  while (temporaryHomes.length > 0) rmSync(temporaryHomes.pop()!, { recursive: true, force: true });
});

const campaignConfig: CampaignConfig = {
  slug: "synthesis-calibration",
  purpose: "family synthesis test",
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
    title: "Role change",
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
    generatorNotes: "Synthesis fixture."
  };
}

async function synthesisLedger(): Promise<Ledger> {
  const ledger = await openLedger(temporaryHome());
  await seedLedger(ledger);
  const campaignId = await createCampaign(ledger, campaignConfig);
  const family = await ledger.client.execute({
    sql: "SELECT id FROM concept_family WHERE slug = 'role-versus-bearer'"
  });
  const familyId = String(family.rows[0]!["id"]);
  const fixtures = [
    {
      item: item("synthesis-candidate-1", "Yes. The role changed; the person did not."),
      status: "structurally_valid"
    },
    {
      item: item("synthesis-candidate-2", "No. You became a different person."),
      status: "structurally_rejected"
    }
  ];
  for (const fixture of fixtures) {
    const candidateId = stableId("candidate", `${campaignId}:${familyId}:${fixture.item.itemKey}`);
    const versionId = stableId("candidatev", `${candidateId}:1`);
    const { hiddenContract, ...content } = fixture.item;
    const contentJson = canonicalJson(content as unknown as JsonValue);
    await ledger.client.batch([
      {
        sql: `INSERT INTO candidate
              (id, campaign_id, family_id, item_key, kind, status, created_at, updated_at)
              VALUES (?, ?, ?, ?, ?, ?, '2026-07-31T00:00:00Z', '2026-07-31T00:00:00Z')`,
        args: [candidateId, campaignId, familyId, fixture.item.itemKey, fixture.item.kind, fixture.status]
      },
      {
        sql: `INSERT INTO candidate_version
              (id, candidate_id, version, content_json, hidden_contract_json, content_sha256, created_at)
              VALUES (?, ?, 1, ?, ?, ?, '2026-07-31T00:00:00Z')`,
        args: [versionId, candidateId, contentJson,
          canonicalJson(hiddenContract as unknown as JsonValue), sha256Bytes(contentJson)]
      }
    ], "write");
    if (fixture.status === "structurally_rejected") {
      await ledger.client.execute({
        sql: `INSERT INTO candidate_failure(id, candidate_id, code, detail, created_at)
              VALUES (?, ?, 'unknown_secondary_lens', 'Unknown secondary lens delayed_reuse', '2026-07-31T00:00:00Z')`,
        args: [stableId("failure", `${candidateId}:unknown_secondary_lens`), candidateId]
      });
    }
  }
  return ledger;
}

function completeHumanPacket(packet: HumanReviewPacket): HumanReviewPacket {
  for (const assignment of packet.assignments) {
    assignment.response.outcome = packet.pass === "A" ? "acceptable_as_rendered" : "accept_as_positive";
    assignment.response.summaryUserAim = "The user asks whether identity survives a role change.";
    assignment.response.summaryAssistantMove = "The assistant distinguishes the persistent bearer from the temporary role.";
    for (const dimension of Object.keys(assignment.response.scores)) assignment.response.scores[dimension] = 3;
    assignment.response.questionPolicy = "not_applicable";
    assignment.response.missingClarification = "no";
    assignment.response.rationale = "The response can be assessed against the role and bearer distinction.";
    assignment.response.confidence = 3;
  }
  return packet;
}

async function completePass(
  ledger: Ledger,
  pass: "A" | "B"
): Promise<void> {
  const prepared = await prepareHumanReviewPacket(ledger, {
    campaignSlug: campaignConfig.slug,
    reviewerAlias: "operator-test",
    pass,
    limit: 2,
    seed: `synthesis-${pass}`
  });
  const packet = completeHumanPacket(
    JSON.parse(readFileSync(prepared.packetPath, "utf8")) as HumanReviewPacket
  );
  writeFileSync(prepared.packetPath, `${canonicalJson(packet as unknown as JsonValue)}\n`);
  await submitHumanReviewPacket(ledger, prepared.packetPath);
}

async function completeRepeats(ledger: Ledger): Promise<void> {
  const prepared = await prepareHumanReviewPacket(ledger, {
    campaignSlug: campaignConfig.slug,
    reviewerAlias: "operator-test",
    pass: "A",
    limit: 2,
    seed: "synthesis-repeat-gate"
  });
  const packet = completeHumanPacket(
    JSON.parse(readFileSync(prepared.packetPath, "utf8")) as HumanReviewPacket
  );
  writeFileSync(prepared.packetPath, `${canonicalJson(packet as unknown as JsonValue)}\n`);
  const result = await submitHumanReviewPacket(ledger, prepared.packetPath);
  assert.equal(result.primaryReviews, 0);
  assert.equal(result.repeatResponses, 2);
}

function completeSynthesisPacket(packet: FamilySynthesisPacket): FamilySynthesisPacket {
  for (const assignment of packet.assignments) {
    const first = assignment.candidates[0]!.candidateVersionId;
    const last = assignment.candidates.at(-1)!.candidateVersionId;
    assignment.response.disposition = "retain_with_local_repairs";
    assignment.response.centralDistinction = "A person can persist while a temporary social role changes.";
    for (const coverage of assignment.response.coverage) coverage.adequacy = "not_applicable";
    assignment.response.strongestCandidateVersionId = first;
    assignment.response.strongestCandidateRationale = "It states the bearer-role contrast directly.";
    assignment.response.weakestCandidateVersionId = last;
    assignment.response.weakestCandidateRationale = "It reverses the required identity judgment.";
    assignment.response.sharedConceptualError = "none observed across both items";
    assignment.response.sharedStyleSignature = "none observed beyond the targeted contrast";
    assignment.response.responsePolicyImbalance = "The fixture is too small to infer a policy distribution.";
    assignment.response.metadataTaxonomyMismatch = "One rejected item uses a non-lens value in lens metadata.";
    assignment.response.highestLeverageBlueprintRepair = "Separate discourse operations from conceptual lenses.";
    assignment.response.negativeCandidateVersionIds = [last];
    assignment.response.uncertaintyOrTheoryDisagreement = "No unresolved theory disagreement in this fixture.";
    assignment.response.rationale = "Retain the family while repairing the rejected realization and metadata.";
    assignment.response.confidence = 3;
    for (const disposition of assignment.structuralDispositions) {
      disposition.contentUtility = "repairable";
      disposition.validatorFindingCorrectness = "yes";
      disposition.identifiedValue = "delayed_reuse";
      disposition.semanticType = "discourse_operation";
      disposition.remedy = "field_split";
      disposition.automaticAcceptanceHazard = "It would hide that the metadata field contains the wrong semantic type.";
      disposition.automaticRejectionHazard = "It would discard a potentially useful negative conversation.";
      disposition.rationale = "Keep the failure while separating content and schema judgments.";
      disposition.confidence = 3;
    }
  }
  return packet;
}

test("Pass C preparation fails closed until every current candidate has sealed A and B reviews", async () => {
  const ledger = await synthesisLedger();
  try {
    await assert.rejects(
      prepareFamilySynthesisPacket(ledger, {
        campaignSlug: campaignConfig.slug,
        reviewerAlias: "operator-test"
      }),
      /exactly one sealed Pass A and Pass B/
    );
    const assignments = await ledger.client.execute("SELECT COUNT(*) AS count FROM family_synthesis_assignment");
    assert.equal(Number(assignments.rows[0]!["count"]), 0);
  } finally {
    closeLedger(ledger);
  }
});

test("Pass C records family and structural evidence append-only without promotion", async () => {
  const ledger = await synthesisLedger();
  try {
    await completePass(ledger, "A");
    await completeRepeats(ledger);
    await completePass(ledger, "B");
    const prepared = await prepareFamilySynthesisPacket(ledger, {
      campaignSlug: campaignConfig.slug,
      reviewerAlias: "operator-test"
    });
    assert.equal(prepared.familyCount, 1);
    assert.equal(prepared.candidateCount, 2);
    assert.equal(prepared.structuralDispositionCount, 1);
    const resumed = await prepareFamilySynthesisPacket(ledger, {
      campaignSlug: campaignConfig.slug,
      reviewerAlias: "operator-test"
    });
    assert.equal(resumed.resumed, true);
    assert.equal(resumed.packetSha256, prepared.packetSha256);

    const blank = JSON.parse(readFileSync(prepared.packetPath, "utf8")) as FamilySynthesisPacket;
    const tampered = structuredClone(blank);
    tampered.assignments[0]!.familyPurpose = "Changed after preparation";
    writeFileSync(prepared.packetPath, `${canonicalJson(tampered as unknown as JsonValue)}\n`);
    await assert.rejects(submitFamilySynthesisPacket(ledger, prepared.packetPath), /Family evidence changed/);
    writeFileSync(prepared.packetPath, `${canonicalJson(blank as unknown as JsonValue)}\n`);
    await assert.rejects(submitFamilySynthesisPacket(ledger, prepared.packetPath), /needs a family disposition/);
    const completed = completeSynthesisPacket(blank);
    writeFileSync(prepared.packetPath, `${canonicalJson(completed as unknown as JsonValue)}\n`);
    const result = await submitFamilySynthesisPacket(ledger, prepared.packetPath);
    assert.equal(result.familySyntheses, 1);
    assert.equal(result.structuralDispositions, 1);

    const counts = await Promise.all([
      ledger.client.execute("SELECT COUNT(*) AS count FROM family_synthesis"),
      ledger.client.execute("SELECT COUNT(*) AS count FROM family_synthesis_basis"),
      ledger.client.execute("SELECT COUNT(*) AS count FROM structural_disposition"),
      ledger.client.execute("SELECT COUNT(*) AS count FROM structural_disposition_basis"),
      ledger.client.execute("SELECT COUNT(*) AS count FROM release_member"),
      ledger.client.execute("SELECT COUNT(*) AS count FROM training_exposure")
    ]);
    assert.deepEqual(counts.map((entry) => Number(entry.rows[0]!["count"])), [1, 4, 1, 3, 0, 0]);
    const status = await familySynthesisStatus(ledger, campaignConfig.slug);
    assert.deepEqual(status.assignments, { completed: 1 });
    assert.equal(status.familySyntheses, 1);
    assert.equal(status.structuralDispositions, 1);
    assert.equal(status.releaseMembers, 0);
    assert.equal(status.trainingExposures, 0);
    await assert.rejects(
      ledger.client.execute("UPDATE family_synthesis SET disposition = 'retain_blueprint'"),
      /append-only/
    );
    await assert.rejects(ledger.client.execute("DELETE FROM structural_disposition"), /append-only/);
    await assert.rejects(submitFamilySynthesisPacket(ledger, prepared.packetPath), /not open/);
  } finally {
    closeLedger(ledger);
  }
});
