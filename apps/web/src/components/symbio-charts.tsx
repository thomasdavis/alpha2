"use client";

export * from "@alpha/ui";

import { type ChartMetric, type ActivationSwitchEvent } from "@alpha/ui";

export function extractActivationSwitchEvents(metrics: ChartMetric[]): ActivationSwitchEvent[] {
  const events: ActivationSwitchEvent[] = [];
  let prevId: string | null = null;
  for (const m of metrics) {
    if (m.symbio_candidate_id && m.symbio_candidate_id !== prevId) {
      events.push({
        step: m.step,
        fromActivation: null,
        toActivation: m.symbio_candidate_id,
        toGeneration: 0,
        toCandidateId: m.symbio_candidate_id,
        lossAtSwitch: m.loss,
      });
      prevId = m.symbio_candidate_id;
    }
  }
  return events;
}
