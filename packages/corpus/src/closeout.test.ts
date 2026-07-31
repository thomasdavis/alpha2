import assert from "node:assert/strict";
import { afterEach, test } from "node:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import {
  campaignCloseoutStatus,
  prepareCampaignCloseoutPacket,
  submitCampaignCloseoutPacket
} from "./closeout.js";
import {
  campaignCloseoutPacketMatchesEnvelope,
  type CampaignCloseoutPacket
} from "./closeout-contract.js";
import { closeLedger, createCampaign, openLedger, putBlob, seedLedger, type Ledger } from "./db.js";
import { canonicalJson, sha256Bytes, stableId } from "./hash.js";
import { prepareHumanReviewPacket, submitHumanReviewPacket } from "./review.js";
import {
  prepareFamilySynthesisPacket,
  submitFamilySynthesisPacket
} from "./synthesis.js";
import type { FamilySynthesisPacket } from "./synthesis-contract.js";
import type { CampaignConfig, GeneratedItem, HumanReviewPacket, JsonValue } from "./types.js";

const temporaryHomes: string[] = [];

function temporaryHome(): string {
  const path = mkdtempSync(join(tmpdir(), "alpha-corpus-closeout-test-"));
  temporaryHomes.push(path);
  return path;
}

afterEach(() => {
  while (temporaryHomes.length > 0) rmSync(temporaryHomes.pop()!, { recursive: true, force: true });
});

const campaignConfig: CampaignConfig = {
  slug: "closeout-calibration",
  purpose: "campaign closeout test",
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
    generatorNotes: "Closeout fixture."
  };
}

async function closeoutLedger(): Promise<Ledger> {
  const ledger = await openLedger(temporaryHome());
  await seedLedger(ledger);
  const campaignId = await createCampaign(ledger, campaignConfig);
  const family = await ledger.client.execute({
    sql: "SELECT id FROM concept_family WHERE slug = 'role-versus-bearer'"
  });
  const familyId = String(family.rows[0]!["id"]);
  const fixtures = [
    {
      item: item("closeout-candidate-1", "Yes. The role changed; the person did not."),
      status: "structurally_valid"
    },
    {
      item: item("closeout-candidate-2", "No. You became a different person."),
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
    assignment.response.summaryAssistantMove = "The assistant distinguishes a persistent bearer from a temporary role.";
    for (const dimension of Object.keys(assignment.response.scores)) assignment.response.scores[dimension] = 3;
    assignment.response.questionPolicy = "not_applicable";
    assignment.response.missingClarification = "no";
    assignment.response.rationale = "The response can be assessed against the role and bearer distinction.";
    assignment.response.confidence = 3;
  }
  return packet;
}

async function completeHumanPass(ledger: Ledger, pass: "A" | "B"): Promise<void> {
  const prepared = await prepareHumanReviewPacket(ledger, {
    campaignSlug: campaignConfig.slug,
    reviewerAlias: "operator-test",
    pass,
    limit: 2,
    seed: `closeout-${pass}`
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
    seed: "closeout-repeats"
  });
  const packet = completeHumanPacket(
    JSON.parse(readFileSync(prepared.packetPath, "utf8")) as HumanReviewPacket
  );
  assert.equal(packet.assignments.length, 2);
  writeFileSync(prepared.packetPath, `${canonicalJson(packet as unknown as JsonValue)}\n`);
  const submitted = await submitHumanReviewPacket(ledger, prepared.packetPath);
  assert.equal(submitted.primaryReviews, 0);
  assert.equal(submitted.repeatResponses, 2);
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
      disposition.automaticAcceptanceHazard = "It would hide a wrong semantic field type.";
      disposition.automaticRejectionHazard = "It would discard a potentially useful negative.";
      disposition.rationale = "Keep the failure while separating content and schema judgments.";
      disposition.confidence = 3;
    }
  }
  return packet;
}

async function completeFamilySynthesis(ledger: Ledger): Promise<void> {
  const prepared = await prepareFamilySynthesisPacket(ledger, {
    campaignSlug: campaignConfig.slug,
    reviewerAlias: "operator-test"
  });
  const packet = completeSynthesisPacket(
    JSON.parse(readFileSync(prepared.packetPath, "utf8")) as FamilySynthesisPacket
  );
  writeFileSync(prepared.packetPath, `${canonicalJson(packet as unknown as JsonValue)}\n`);
  await submitFamilySynthesisPacket(ledger, prepared.packetPath);
}

async function addAnalysisRun(ledger: Ledger): Promise<void> {
  const campaign = await ledger.client.execute({
    sql: "SELECT id FROM generation_campaign WHERE slug = ?",
    args: [campaignConfig.slug]
  });
  const campaignId = String(campaign.rows[0]!["id"]);
  const methodId = stableId("analysis_method", "closeout-fixture-v1");
  const methodConfig = canonicalJson({ fixture: true } as JsonValue);
  const softwareId = stableId("software", "closeout-fixture-revision");
  const outputSha = await putBlob(ledger, canonicalJson({ fixture: true } as JsonValue), "application/json");
  const runId = stableId("analysis_run", `${campaignId}:${methodId}:${softwareId}:snapshot`);
  await ledger.client.batch([
    {
      sql: `INSERT INTO analysis_method
            (id, slug, version, definition, config_json, content_sha256, created_at)
            VALUES (?, 'closeout-fixture', 1, 'test only', ?, ?, '2026-07-31T00:00:00Z')`,
      args: [methodId, methodConfig, sha256Bytes(methodConfig)]
    },
    {
      sql: `INSERT INTO software_revision
            (id, component, revision, build_digest, environment_json, created_at)
            VALUES (?, 'alpha-corpus-test', 'fixture', NULL, '{}', '2026-07-31T00:00:00Z')`,
      args: [softwareId]
    },
    {
      sql: `INSERT INTO analysis_run
            (id, campaign_id, analysis_method_id, software_revision_id, input_snapshot_sha256,
             output_blob_sha256, status, evidence_scope, disclaimer, started_at, completed_at)
            VALUES (?, ?, ?, ?, 'snapshot', ?, 'completed', 'surface_distribution_only',
              'test fixture; not semantic evidence', '2026-07-31T00:00:00Z', '2026-07-31T00:00:00Z')`,
      args: [runId, campaignId, methodId, softwareId, outputSha]
    }
  ], "write");
}

async function completePrerequisites(ledger: Ledger): Promise<void> {
  await completeHumanPass(ledger, "A");
  await completeRepeats(ledger);
  await completeHumanPass(ledger, "B");
  await completeFamilySynthesis(ledger);
  await addAnalysisRun(ledger);
}

function completeCloseoutPacket(packet: CampaignCloseoutPacket): CampaignCloseoutPacket {
  for (const candidate of packet.response.candidateDispositions) {
    const evidence = packet.candidates.find((entry) => entry.candidateVersionId === candidate.candidateVersionId)!;
    candidate.outcome = evidence.status === "structurally_rejected" ? "accept_as_negative" : "accept_as_positive";
    candidate.rationale = evidence.status === "structurally_rejected"
      ? "Retain the structurally rejected realization as a diagnosed negative."
      : "The candidate is acceptable within this calibration after A, B, and C review.";
    candidate.confidence = 3;
  }
  const rejected = packet.candidates.find((candidate) => candidate.status === "structurally_rejected")!;
  packet.response.failureClusters = [{
    clusterKey: "secondary-lens-field-confusion",
    label: "Discourse operation stored as a conceptual lens",
    locus: "schema",
    severity: "major",
    proposedRepair: "Separate discourse operations from conceptual secondary lenses.",
    newCallsNeeded: "no",
    rationale: "The validator and human structural disposition agree on the field-type mismatch.",
    members: [{ memberKind: "candidate_version", memberId: rejected.candidateVersionId }]
  }];
  for (const assessment of packet.response.distributionAssessments) {
    assessment.assessment = `The controlled fixture provides bounded evidence for ${assessment.dimension}.`;
    assessment.evidenceIds = [packet.candidates[0]!.passAReviewId];
  }
  packet.response.recommendedStates = [{
    state: "D5_REPAIR_REQUIRED",
    rationale: "Repair the schema distinction before another bounded generation decision."
  }];
  packet.response.recommendationSummary = "Preserve the calibration and repair the metadata schema before any new calls.";
  packet.response.known = ["The reviewed fixture preserves the intended role-bearer distinction in one candidate."];
  packet.response.unknown = ["This fixture cannot establish whether synthetic-only training teaches the distinction."];
  packet.response.proposedNext = ["Draft a schema repair proposal without authorizing generation or training."];
  packet.response.disagreements = [];
  packet.response.noDisagreementRationale = "No disagreement was introduced in this controlled test fixture.";
  packet.response.overallRationale = "The evidence supports a non-binding repair recommendation and no execution authority.";
  packet.response.confidence = 3;
  return packet;
}

test("Pass D preparation refuses incomplete human, family, repeat, and analysis evidence", async () => {
  const ledger = await closeoutLedger();
  try {
    await assert.rejects(
      prepareCampaignCloseoutPacket(ledger, {
        campaignSlug: campaignConfig.slug,
        adjudicatorAlias: "operator-test"
      }),
      /sealed Pass A review/
    );
    const assignments = await ledger.client.execute("SELECT COUNT(*) AS count FROM campaign_closeout_assignment");
    assert.equal(Number(assignments.rows[0]!["count"]), 0);
  } finally {
    closeLedger(ledger);
  }
});

test("Pass D records non-binding adjudication evidence without lifecycle or execution authorization", async () => {
  const ledger = await closeoutLedger();
  try {
    await completePrerequisites(ledger);
    const prepared = await prepareCampaignCloseoutPacket(ledger, {
      campaignSlug: campaignConfig.slug,
      adjudicatorAlias: "operator-test"
    });
    assert.equal(prepared.candidateCount, 2);
    assert.equal(prepared.familyCount, 1);
    assert.equal(prepared.repeatCount, 2);
    const resumed = await prepareCampaignCloseoutPacket(ledger, {
      campaignSlug: campaignConfig.slug,
      adjudicatorAlias: "operator-test"
    });
    assert.equal(resumed.resumed, true);
    assert.equal(resumed.packetSha256, prepared.packetSha256);

    const blank = JSON.parse(readFileSync(prepared.packetPath, "utf8")) as CampaignCloseoutPacket;
    const tampered = structuredClone(blank);
    tampered.candidates[0]!.status = "changed";
    writeFileSync(prepared.packetPath, `${canonicalJson(tampered as unknown as JsonValue)}\n`);
    await assert.rejects(submitCampaignCloseoutPacket(ledger, prepared.packetPath), /evidence differs/);
    const populationTampered = structuredClone(blank);
    populationTampered.population.candidates += 1;
    writeFileSync(prepared.packetPath, `${canonicalJson(populationTampered as unknown as JsonValue)}\n`);
    await assert.rejects(submitCampaignCloseoutPacket(ledger, prepared.packetPath), /population differs/);
    writeFileSync(prepared.packetPath, `${canonicalJson(blank as unknown as JsonValue)}\n`);
    await assert.rejects(submitCampaignCloseoutPacket(ledger, prepared.packetPath), /recommendation summary/);

    const completed = completeCloseoutPacket(blank);
    assert.equal(campaignCloseoutPacketMatchesEnvelope(completed, blank), true);
    const envelopeTampered = structuredClone(completed);
    envelopeTampered.createdAt = "2099-01-01T00:00:00.000Z";
    assert.equal(campaignCloseoutPacketMatchesEnvelope(envelopeTampered, blank), false);
    writeFileSync(prepared.packetPath, `${canonicalJson(envelopeTampered as unknown as JsonValue)}\n`);
    await assert.rejects(
      submitCampaignCloseoutPacket(ledger, prepared.packetPath),
      /immutable envelope does not match an exported packet/
    );
    const beforeSubmission = await Promise.all([
      ledger.client.execute("SELECT COUNT(*) AS count FROM campaign_closeout"),
      ledger.client.execute("SELECT COUNT(*) AS count FROM adjudication"),
      ledger.client.execute("SELECT COUNT(*) AS count FROM raw_artifact WHERE kind = 'campaign_closeout_submission'")
    ]);
    assert.deepEqual(beforeSubmission.map((entry) => Number(entry.rows[0]!["count"])), [0, 0, 0]);
    writeFileSync(prepared.packetPath, `${canonicalJson(completed as unknown as JsonValue)}\n`);
    const result = await submitCampaignCloseoutPacket(ledger, prepared.packetPath);
    assert.equal(result.candidateAdjudications, 2);
    assert.equal(result.failureClusters, 1);
    assert.equal(result.recommendedStates, 1);
    assert.equal(result.packetEnvelopeSha256, prepared.packetSha256);
    assert.equal(result.executionAuthorized, false);

    const counts = await Promise.all([
      ledger.client.execute("SELECT COUNT(*) AS count FROM campaign_closeout"),
      ledger.client.execute("SELECT COUNT(*) AS count FROM adjudication"),
      ledger.client.execute("SELECT COUNT(*) AS count FROM adjudication_basis"),
      ledger.client.execute("SELECT COUNT(*) AS count FROM campaign_closeout_state"),
      ledger.client.execute("SELECT COUNT(*) AS count FROM campaign_failure_cluster"),
      ledger.client.execute("SELECT COUNT(*) AS count FROM campaign_failure_cluster_member"),
      ledger.client.execute("SELECT COUNT(*) AS count FROM campaign_distribution_assessment"),
      ledger.client.execute("SELECT COUNT(*) AS count FROM quality_state_transition"),
      ledger.client.execute("SELECT COUNT(*) AS count FROM release_member"),
      ledger.client.execute("SELECT COUNT(*) AS count FROM training_exposure")
    ]);
    assert.deepEqual(counts.map((entry) => Number(entry.rows[0]!["count"])), [1, 2, 7, 1, 1, 1, 8, 0, 0, 0]);
    const statuses = await ledger.client.execute("SELECT status, COUNT(*) AS count FROM candidate GROUP BY status ORDER BY status");
    assert.deepEqual(statuses.rows.map((row) => [String(row["status"]), Number(row["count"])]),
      [["structurally_rejected", 1], ["structurally_valid", 1]]);

    const status = await campaignCloseoutStatus(ledger, campaignConfig.slug);
    assert.deepEqual(status.assignments, { completed: 1 });
    assert.equal(status.campaignCloseouts, 1);
    assert.equal(status.candidateAdjudications, 2);
    assert.equal(status.failureClusters, 1);
    assert.deepEqual(status.recommendedStates, { D5_REPAIR_REQUIRED: 1 });
    assert.equal(status.executionAuthorizations, 0);
    assert.equal(status.releaseMembers, 0);
    assert.equal(status.trainingExposures, 0);

    await assert.rejects(
      ledger.client.execute("UPDATE campaign_closeout SET execution_authorized = 1"),
      /append-only/
    );
    await assert.rejects(ledger.client.execute("DELETE FROM campaign_closeout_state"), /append-only/);
    await assert.rejects(submitCampaignCloseoutPacket(ledger, prepared.packetPath), /not open/);
  } finally {
    closeLedger(ledger);
  }
});
