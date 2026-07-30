import { resolve } from "node:path";
import { callCodexStructured, loadRecoverableStructuredCall } from "./codex.js";
import {
  appendEvent,
  campaignStats,
  checkCampaignStorage,
  createCampaign,
  createTask,
  listFamilies,
  loadRecordedStructuredResponse,
  nextTaskAttempt,
  recordCandidate,
  recordStructuredCall,
  setCampaignStatus,
  setTaskStatus,
  type Ledger
} from "./db.js";
import { generationPrompt, type GenerationRecipe } from "./prompts.js";
import { generationEnvelopeSchema } from "./schemas.js";
import { categorySeeds, transformationSeeds } from "./seeds.js";
import { DEFAULT_ARTIFACT_LIMIT_BYTES } from "./storage.js";
import type { CampaignConfig, FamilyBlueprint, GenerationEnvelope } from "./types.js";
import { parseGenerationEnvelope, validateCandidate } from "./validate.js";

export const CALIBRATION_CAMPAIGN_SLUG = "alpha-calibration-v1";
const recipes: GenerationRecipe[] = ["natural-dialogue", "contrast-and-repair"];

export interface CalibrationOptions {
  execute: boolean;
  model: string;
  itemsPerCall: number;
  familyLimit: number;
  repoRoot: string;
}

export interface CalibrationResult {
  campaignId: string;
  plannedCalls: number;
  executedCalls: number;
  acceptedStructurally: number;
  rejectedStructurally: number;
  skippedCompleted: number;
  paused: boolean;
}

function campaignConfig(options: CalibrationOptions): CampaignConfig {
  return {
    slug: CALIBRATION_CAMPAIGN_SLUG,
    purpose: "Human-auditable canary corpus for the Alpha synthetic curriculum; quarantined from training and evaluation.",
    workerModel: options.model,
    criticModel: "gpt-5.5-disabled-pending-paired-probe",
    maxGenerationCalls: options.familyLimit * recipes.length,
    maxReviewCalls: 0,
    itemsPerFamily: options.itemsPerCall * recipes.length,
    artifactLimitBytes: DEFAULT_ARTIFACT_LIMIT_BYTES
  };
}

export async function generateCalibration(
  ledger: Ledger,
  options: CalibrationOptions
): Promise<CalibrationResult> {
  const families = (await listFamilies(ledger)).slice(0, options.familyLimit);
  const campaignId = await createCampaign(ledger, campaignConfig(options));
  const result: CalibrationResult = {
    campaignId,
    plannedCalls: families.length * recipes.length,
    executedCalls: 0,
    acceptedStructurally: 0,
    rejectedStructurally: 0,
    skippedCompleted: 0,
    paused: false
  };
  await appendEvent(ledger, "calibration_plan_observed", "generation_campaign", campaignId, {
    execute: options.execute,
    familyCount: families.length,
    recipes,
    itemsPerCall: options.itemsPerCall,
    model: options.model
  });
  if (!options.execute) return result;
  await setCampaignStatus(ledger, campaignId, "running");

  const allowedLenses = new Set(categorySeeds.map((seed) => seed.slug));
  const allowedTransformations = new Set(transformationSeeds.map(([slug]) => slug));
  for (const familyRow of families) {
    const blueprint = JSON.parse(familyRow.blueprint) as FamilyBlueprint;
    for (const recipe of recipes) {
      if (!(await checkCampaignStorage(ledger, campaignId))) {
        result.paused = true;
        return result;
      }
      const idempotencyKey = `${CALIBRATION_CAMPAIGN_SLUG}:${familyRow.slug}:${recipe}:v1`;
      const task = await createTask(
        ledger,
        campaignId,
        familyRow.id,
        "generate_calibration",
        idempotencyKey,
        options.model
      );
      if (task.status === "completed") {
        result.skippedCompleted++;
        continue;
      }
      const currentStats = await campaignStats(ledger, CALIBRATION_CAMPAIGN_SLUG);
      if (currentStats.modelCalls >= result.plannedCalls) {
        await setCampaignStatus(ledger, campaignId, "paused_call_budget");
        await appendEvent(ledger, "campaign_paused_call_budget", "generation_campaign", campaignId, {
          calls: currentStats.modelCalls,
          maximum: result.plannedCalls
        });
        result.paused = true;
        return result;
      }

      await setTaskStatus(ledger, task.id, "running");
      const prompt = generationPrompt(blueprint, recipe, options.itemsPerCall);
      const recorded = await loadRecordedStructuredResponse<GenerationEnvelope>(ledger, task.id);
      const recovered = recorded === null
        ? loadRecoverableStructuredCall<GenerationEnvelope>(
            task.id,
            ledger.paths.calls,
            prompt,
            generationEnvelopeSchema
          )
        : null;
      const callResult = recorded === null && recovered === null
        ? await callCodexStructured<GenerationEnvelope>({
          taskId: task.id,
          model: options.model,
          role: "worker",
          prompt,
          schemaName: "alpha-generation-envelope-v1",
          schema: generationEnvelopeSchema,
          repoRoot: resolve(options.repoRoot),
          callRoot: ledger.paths.calls
          })
        : recovered;
      if (recovered) {
        await appendEvent(ledger, "orphan_call_recovered", "generation_task", task.id, {
          callDirectory: recovered.callDirectory
        });
      }
      if (recorded) {
        await appendEvent(ledger, "recorded_call_resumed", "generation_task", task.id, {
          callId: recorded.callId
        });
      }
      const callId = recorded?.callId ?? await recordStructuredCall(
          ledger,
          task.id,
          options.model,
          "worker",
          `calibration-${familyRow.slug}-${recipe}-v1`,
          prompt,
          "alpha-generation-envelope-v1",
          generationEnvelopeSchema,
          callResult!,
          await nextTaskAttempt(ledger, task.id)
        );
      const parsed = recorded?.parsed ?? callResult?.parsed ?? null;
      if (!recorded) result.executedCalls++;
      if (!recorded && (callResult!.exitCode !== 0 || parsed === null)) {
        await setTaskStatus(ledger, task.id, "failed");
        await appendEvent(ledger, "generation_call_failed", "generation_task", task.id, {
          callId,
          exitCode: callResult!.exitCode,
          hasStructuredResponse: parsed !== null
        });
        continue;
      }

      let envelope: GenerationEnvelope;
      try {
        envelope = parseGenerationEnvelope(parsed);
      } catch (error) {
        await setTaskStatus(ledger, task.id, "failed_schema_validation");
        await appendEvent(ledger, "generation_envelope_rejected", "generation_task", task.id, {
          callId,
          error: String(error)
        });
        continue;
      }
      if (envelope.familySlug !== familyRow.slug || envelope.items.length !== options.itemsPerCall) {
        await setTaskStatus(ledger, task.id, "failed_batch_contract");
        await appendEvent(ledger, "generation_batch_contract_failed", "generation_task", task.id, {
          callId,
          expectedFamily: familyRow.slug,
          observedFamily: envelope.familySlug,
          expectedItems: options.itemsPerCall,
          observedItems: envelope.items.length
        });
        continue;
      }
      const seenKeys = new Set<string>();
      for (const item of envelope.items) {
        const validation = validateCandidate(
          item,
          familyRow.slug,
          `${familyRow.slug}-${recipe}-`,
          allowedLenses,
          allowedTransformations
        );
        if (seenKeys.has(item.itemKey)) {
          validation.findings.push({
            code: "duplicate_item_key",
            severity: "error",
            message: `Duplicate itemKey ${item.itemKey} in one response`
          });
          validation.valid = false;
        }
        seenKeys.add(item.itemKey);
        await recordCandidate(ledger, campaignId, familyRow.id, callId, item, validation);
        if (validation.valid) result.acceptedStructurally++;
        else result.rejectedStructurally++;
      }
      await setTaskStatus(ledger, task.id, "completed");
      await appendEvent(ledger, "generation_task_completed", "generation_task", task.id, {
        callId,
        generated: envelope.items.length,
        structurallyValid: result.acceptedStructurally,
        structurallyRejected: result.rejectedStructurally
      });
    }
  }
  await setCampaignStatus(ledger, campaignId, "generated_pending_human_review");
  return result;
}
