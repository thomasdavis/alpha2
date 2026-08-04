#!/usr/bin/env node

import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

function fail(message) {
  console.error(`error: ${message}`);
  process.exit(2);
}

function parseArgs(argv) {
  const args = new Map();
  for (let index = 2; index < argv.length; index += 2) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key?.startsWith('--') || value === undefined) fail('expected --key value arguments');
    args.set(key.slice(2), value);
  }
  return args;
}

function readMetrics(path) {
  return readFileSync(resolve(path), 'utf8')
    .trim()
    .split('\n')
    .filter(Boolean)
    .map((line) => JSON.parse(line))
    .sort((a, b) => a.step - b.step);
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}

function summarize(metrics) {
  const warm = metrics.filter((metric) => metric.step >= 2);
  return {
    steps: metrics.length,
    measured_step_time_ms: metrics.reduce((sum, metric) => sum + metric.elapsed_ms, 0),
    all_step_mean_tokens_per_second: mean(metrics.map((metric) => metric.tokens_per_sec)),
    warm_mean_tokens_per_second: mean(warm.map((metric) => metric.tokens_per_sec)),
    warm_median_tokens_per_second: median(warm.map((metric) => metric.tokens_per_sec)),
    warm_min_tokens_per_second: Math.min(...warm.map((metric) => metric.tokens_per_sec)),
    warm_max_tokens_per_second: Math.max(...warm.map((metric) => metric.tokens_per_sec)),
    evaluations: metrics
      .filter((metric) => Number.isFinite(metric.valLoss))
      .map((metric) => ({ step: metric.step, val_loss: metric.valLoss })),
  };
}

function stepWithinBudget(metrics, budgetMs) {
  let elapsed = 0;
  let step = 0;
  for (const metric of metrics) {
    if (elapsed + metric.elapsed_ms > budgetMs) break;
    elapsed += metric.elapsed_ms;
    step = metric.step;
  }
  return { step, measured_step_time_ms: elapsed };
}

function validationBracket(evaluations, step) {
  const lower = [...evaluations].reverse().find((evaluation) => evaluation.step <= step) ?? null;
  const upper = evaluations.find((evaluation) => evaluation.step >= step) ?? null;
  let interpolated = null;
  if (lower && upper) {
    if (lower.step === upper.step) {
      interpolated = lower.val_loss;
    } else {
      const fraction = (step - lower.step) / (upper.step - lower.step);
      interpolated = lower.val_loss + fraction * (upper.val_loss - lower.val_loss);
    }
  }
  return { lower, upper, interpolated_val_loss: interpolated };
}

const args = parseArgs(process.argv);
const referencePath = args.get('reference');
const candidatePath = args.get('candidate');
const outputPath = args.get('output');
if (!referencePath || !candidatePath || !outputPath) {
  fail('usage: --reference metrics.jsonl --candidate metrics.jsonl --output report.json');
}

const reference = readMetrics(referencePath);
const candidate = readMetrics(candidatePath);
const referenceSummary = summarize(reference);
const candidateSummary = summarize(candidate);
const referenceEvaluations = referenceSummary.evaluations;
const candidateEvaluations = candidateSummary.evaluations;
const commonEvaluationSteps = candidateEvaluations
  .map((evaluation) => evaluation.step)
  .filter((step) => referenceEvaluations.some((evaluation) => evaluation.step === step));

const equalStep = commonEvaluationSteps.map((step) => {
  const referenceEval = referenceEvaluations.find((evaluation) => evaluation.step === step);
  const candidateEval = candidateEvaluations.find((evaluation) => evaluation.step === step);
  return {
    step,
    reference_val_loss: referenceEval.val_loss,
    candidate_val_loss: candidateEval.val_loss,
    candidate_minus_reference_val_loss: candidateEval.val_loss - referenceEval.val_loss,
  };
});

const candidateFinalEvaluation = candidateEvaluations.at(-1) ?? null;
const candidateBudget = candidateSummary.measured_step_time_ms;
const referenceAtCandidateBudget = stepWithinBudget(reference, candidateBudget);
const bracket = validationBracket(referenceEvaluations, referenceAtCandidateBudget.step);

const report = {
  schema: 'alpha-helios-cross-run-quality-frontier-v1',
  created_at: new Date().toISOString(),
  reference: {
    label: args.get('reference-label') ?? 'reference',
    metrics_path: resolve(referencePath),
    ...referenceSummary,
  },
  candidate: {
    label: args.get('candidate-label') ?? 'candidate',
    metrics_path: resolve(candidatePath),
    ...candidateSummary,
  },
  throughput: {
    warm_mean_gain_fraction:
      candidateSummary.warm_mean_tokens_per_second / referenceSummary.warm_mean_tokens_per_second - 1,
    warm_median_gain_fraction:
      candidateSummary.warm_median_tokens_per_second / referenceSummary.warm_median_tokens_per_second - 1,
    measured_step_time_saving_fraction:
      1 - candidateSummary.measured_step_time_ms / referenceSummary.measured_step_time_ms,
  },
  equal_step_validation: equalStep,
  equal_wall_validation: {
    candidate_final_evaluation: candidateFinalEvaluation,
    candidate_measured_step_time_ms: candidateBudget,
    reference_steps_completed_within_candidate_budget: referenceAtCandidateBudget,
    reference_validation_bracket: bracket,
    candidate_minus_interpolated_reference_val_loss:
      candidateFinalEvaluation && bracket.interpolated_val_loss !== null
        ? candidateFinalEvaluation.val_loss - bracket.interpolated_val_loss
        : null,
    interpolation_is_estimate: true,
  },
};

writeFileSync(resolve(outputPath), `${JSON.stringify(report, null, 2)}\n`);
console.log(resolve(outputPath));
