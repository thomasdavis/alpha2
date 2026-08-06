#!/usr/bin/env node

import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';

function fail(message) {
  console.error(`error: ${message}`);
  process.exit(2);
}

function readMetrics(root, row) {
  const path = join(root, row, 'metrics.jsonl');
  return readFileSync(path, 'utf8')
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

function extrema(values) {
  return { min: Math.min(...values), max: Math.max(...values) };
}

function warm(metrics) {
  return metrics.filter((metric) => metric.step >= 2);
}

function pairedComparison(reference, candidate) {
  const refByStep = new Map(warm(reference).map((metric) => [metric.step, metric]));
  const pairs = warm(candidate)
    .filter((metric) => refByStep.has(metric.step))
    .map((candidateMetric) => ({
      step: candidateMetric.step,
      reference: refByStep.get(candidateMetric.step),
      candidate: candidateMetric,
    }));

  if (pairs.length === 0) fail('no paired warm steps');

  const refSpeeds = pairs.map((pair) => pair.reference.tokens_per_sec);
  const candidateSpeeds = pairs.map((pair) => pair.candidate.tokens_per_sec);
  const ratios = pairs.map((pair) => pair.candidate.tokens_per_sec / pair.reference.tokens_per_sec);
  const lossDeltas = pairs.map((pair) => Math.abs(pair.candidate.loss - pair.reference.loss));
  const gradDeltas = pairs.map((pair) => Math.abs(pair.candidate.gradNorm - pair.reference.gradNorm));
  const final = pairs.at(-1);

  const timingFields = [
    'timing_fwd_ms',
    'timing_bwd_ms',
    'timing_grad_norm_ms',
    'timing_grad_clip_ms',
    'timing_optim_ms',
    'timing_flush_ms',
    'timing_data_ms',
    'timing_host_build_ms',
    'timing_gpu_blocking_ms',
    'timing_core_step_ms',
  ];
  const timing = Object.fromEntries(timingFields.map((field) => {
    const referenceMean = mean(pairs.map((pair) => pair.reference[field]));
    const candidateMean = mean(pairs.map((pair) => pair.candidate[field]));
    return [field, {
      reference_mean: referenceMean,
      candidate_mean: candidateMean,
      delta_ms: candidateMean - referenceMean,
      gain_fraction: 1 - candidateMean / referenceMean,
    }];
  }));

  return {
    paired_warm_steps: pairs.length,
    reference: {
      mean_tokens_per_second: mean(refSpeeds),
      median_tokens_per_second: median(refSpeeds),
      ...extrema(refSpeeds),
    },
    candidate: {
      mean_tokens_per_second: mean(candidateSpeeds),
      median_tokens_per_second: median(candidateSpeeds),
      ...extrema(candidateSpeeds),
    },
    ratios: {
      aggregate_mean_gain_fraction: mean(candidateSpeeds) / mean(refSpeeds) - 1,
      aggregate_median_gain_fraction: median(candidateSpeeds) / median(refSpeeds) - 1,
      mean_paired_ratio: mean(ratios),
      median_paired_ratio: median(ratios),
      geometric_mean_paired_ratio: Math.exp(mean(ratios.map(Math.log))),
      minimum_paired_ratio: Math.min(...ratios),
      maximum_paired_ratio: Math.max(...ratios),
      candidate_faster_step_count: ratios.filter((ratio) => ratio > 1).length,
    },
    trajectory: {
      maximum_absolute_loss_difference: Math.max(...lossDeltas),
      maximum_absolute_gradient_norm_difference: Math.max(...gradDeltas),
      final_step: final.step,
      final_reference_loss: final.reference.loss,
      final_candidate_loss: final.candidate.loss,
      final_absolute_loss_difference: Math.abs(final.candidate.loss - final.reference.loss),
      final_reference_gradient_norm: final.reference.gradNorm,
      final_candidate_gradient_norm: final.candidate.gradNorm,
      final_absolute_gradient_norm_difference: Math.abs(
        final.candidate.gradNorm - final.reference.gradNorm,
      ),
    },
    mean_timing: timing,
  };
}

const [rootArg, candidate = 'profitable_four_default', control = 'control_composition'] = process.argv.slice(2);
if (!rootArg) {
  fail('usage: derive_helios_composition_stats.mjs <artifact-root> [candidate-row] [control-row]');
}

const root = resolve(rootArg);
const baseline = 'baseline_fp32';
const baselineMetrics = readMetrics(root, baseline);
const candidateMetrics = readMetrics(root, candidate);
const controlMetricsPath = join(root, control, 'metrics.jsonl');
const controlMetrics = existsSync(controlMetricsPath) ? readMetrics(root, control) : null;

const report = {
  schema: 'alpha-helios-composition-paired-analysis-v1',
  created_at: new Date().toISOString(),
  artifact_root: basename(root),
  rows: { baseline, control, candidate },
  steps: candidateMetrics.length,
  baseline_to_candidate: pairedComparison(baselineMetrics, candidateMetrics),
};
if (controlMetrics) {
  report.baseline_to_control = pairedComparison(baselineMetrics, controlMetrics);
  report.control_to_candidate = pairedComparison(controlMetrics, candidateMetrics);
} else {
  report.rows.control = null;
}

const output = join(root, 'DERIVED-STATS.json');
writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`);
console.log(output);
