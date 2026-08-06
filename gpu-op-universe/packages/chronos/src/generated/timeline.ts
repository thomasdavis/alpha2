/* AUTO-GENERATED. Do not hand-edit; edit operation-registry.json. */
import { defineStub } from "../../../common/src/types";
import type { ScheduleOpRequest } from "../../../common/src/types";

/**
 * chronos.timeline.create-timeline
 * Create timeline operation in the timeline family.
 * Status: standard; target: host, sm86; differentiability: not-applicable.
 */
export const timelineCreateTimeline = defineStub<ScheduleOpRequest>("chronos.timeline.create-timeline");

/**
 * chronos.timeline.destroy-timeline
 * Destroy timeline operation in the timeline family.
 * Status: standard; target: host, sm86; differentiability: not-applicable.
 */
export const timelineDestroyTimeline = defineStub<ScheduleOpRequest>("chronos.timeline.destroy-timeline");

/**
 * chronos.timeline.export-timeline
 * Export timeline operation in the timeline family.
 * Status: standard; target: host, sm86; differentiability: not-applicable.
 */
export const timelineExportTimeline = defineStub<ScheduleOpRequest>("chronos.timeline.export-timeline");

/**
 * chronos.timeline.import-timeline
 * Import timeline operation in the timeline family.
 * Status: standard; target: host, sm86; differentiability: not-applicable.
 */
export const timelineImportTimeline = defineStub<ScheduleOpRequest>("chronos.timeline.import-timeline");

/**
 * chronos.timeline.next-timeline-value
 * Next timeline value operation in the timeline family.
 * Status: standard; target: host, sm86; differentiability: not-applicable.
 */
export const timelineNextTimelineValue = defineStub<ScheduleOpRequest>("chronos.timeline.next-timeline-value");

/**
 * chronos.timeline.poll-timeline
 * Poll timeline operation in the timeline family.
 * Status: standard; target: host, sm86; differentiability: not-applicable.
 */
export const timelinePollTimeline = defineStub<ScheduleOpRequest>("chronos.timeline.poll-timeline");

/**
 * chronos.timeline.query-timeline
 * Query timeline operation in the timeline family.
 * Status: standard; target: host, sm86; differentiability: not-applicable.
 */
export const timelineQueryTimeline = defineStub<ScheduleOpRequest>("chronos.timeline.query-timeline");

/**
 * chronos.timeline.reset-timeline
 * Reset timeline operation in the timeline family.
 * Status: standard; target: host, sm86; differentiability: not-applicable.
 */
export const timelineResetTimeline = defineStub<ScheduleOpRequest>("chronos.timeline.reset-timeline");

/**
 * chronos.timeline.signal-timeline-gpu
 * Signal timeline gpu operation in the timeline family.
 * Status: standard; target: host, sm86; differentiability: not-applicable.
 */
export const timelineSignalTimelineGpu = defineStub<ScheduleOpRequest>("chronos.timeline.signal-timeline-gpu");

/**
 * chronos.timeline.signal-timeline-host
 * Signal timeline host operation in the timeline family.
 * Status: standard; target: host, sm86; differentiability: not-applicable.
 */
export const timelineSignalTimelineHost = defineStub<ScheduleOpRequest>("chronos.timeline.signal-timeline-host");

/**
 * chronos.timeline.validate-monotonicity
 * Validate monotonicity operation in the timeline family.
 * Status: standard; target: host, sm86; differentiability: not-applicable.
 */
export const timelineValidateMonotonicity = defineStub<ScheduleOpRequest>("chronos.timeline.validate-monotonicity");

/**
 * chronos.timeline.wait-timeline-gpu
 * Wait timeline gpu operation in the timeline family.
 * Status: standard; target: host, sm86; differentiability: not-applicable.
 */
export const timelineWaitTimelineGpu = defineStub<ScheduleOpRequest>("chronos.timeline.wait-timeline-gpu");

/**
 * chronos.timeline.wait-timeline-host
 * Wait timeline host operation in the timeline family.
 * Status: standard; target: host, sm86; differentiability: not-applicable.
 */
export const timelineWaitTimelineHost = defineStub<ScheduleOpRequest>("chronos.timeline.wait-timeline-host");
