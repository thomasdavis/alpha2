/* AUTO-GENERATED. Do not hand-edit; edit operation-registry.json. */
import { defineStub } from "../../../common/src/types";
import type { ScheduleOpRequest } from "../../../common/src/types";

/**
 * chronos.dependency.add-dependency
 * Add dependency operation in the dependency family.
 * Status: standard; target: architecture-agnostic; differentiability: not-applicable.
 */
export const dependencyAddDependency = defineStub<ScheduleOpRequest>("chronos.dependency.add-dependency");

/**
 * chronos.dependency.barrier-dependency
 * Barrier dependency operation in the dependency family.
 * Status: standard; target: architecture-agnostic; differentiability: not-applicable.
 */
export const dependencyBarrierDependency = defineStub<ScheduleOpRequest>("chronos.dependency.barrier-dependency");

/**
 * chronos.dependency.critical-path
 * Critical path operation in the dependency family.
 * Status: standard; target: architecture-agnostic; differentiability: not-applicable.
 */
export const dependencyCriticalPath = defineStub<ScheduleOpRequest>("chronos.dependency.critical-path");

/**
 * chronos.dependency.cross-device-dependency
 * Cross device dependency operation in the dependency family.
 * Status: standard; target: architecture-agnostic; differentiability: not-applicable.
 */
export const dependencyCrossDeviceDependency = defineStub<ScheduleOpRequest>("chronos.dependency.cross-device-dependency");

/**
 * chronos.dependency.cross-queue-dependency
 * Cross queue dependency operation in the dependency family.
 * Status: standard; target: architecture-agnostic; differentiability: not-applicable.
 */
export const dependencyCrossQueueDependency = defineStub<ScheduleOpRequest>("chronos.dependency.cross-queue-dependency");

/**
 * chronos.dependency.dependency-batch
 * Dependency batch operation in the dependency family.
 * Status: standard; target: architecture-agnostic; differentiability: not-applicable.
 */
export const dependencyDependencyBatch = defineStub<ScheduleOpRequest>("chronos.dependency.dependency-batch");

/**
 * chronos.dependency.dependency-token
 * Dependency token operation in the dependency family.
 * Status: standard; target: architecture-agnostic; differentiability: not-applicable.
 */
export const dependencyDependencyToken = defineStub<ScheduleOpRequest>("chronos.dependency.dependency-token");

/**
 * chronos.dependency.detect-cycle
 * Detect cycle operation in the dependency family.
 * Status: standard; target: architecture-agnostic; differentiability: not-applicable.
 */
export const dependencyDetectCycle = defineStub<ScheduleOpRequest>("chronos.dependency.detect-cycle");

/**
 * chronos.dependency.event-dependency
 * Event dependency operation in the dependency family.
 * Status: standard; target: architecture-agnostic; differentiability: not-applicable.
 */
export const dependencyEventDependency = defineStub<ScheduleOpRequest>("chronos.dependency.event-dependency");

/**
 * chronos.dependency.execution-dependency
 * Execution dependency operation in the dependency family.
 * Status: standard; target: architecture-agnostic; differentiability: not-applicable.
 */
export const dependencyExecutionDependency = defineStub<ScheduleOpRequest>("chronos.dependency.execution-dependency");

/**
 * chronos.dependency.memory-dependency
 * Memory dependency operation in the dependency family.
 * Status: standard; target: architecture-agnostic; differentiability: not-applicable.
 */
export const dependencyMemoryDependency = defineStub<ScheduleOpRequest>("chronos.dependency.memory-dependency");

/**
 * chronos.dependency.read-after-write
 * Read after write operation in the dependency family.
 * Status: standard; target: architecture-agnostic; differentiability: not-applicable.
 */
export const dependencyReadAfterWrite = defineStub<ScheduleOpRequest>("chronos.dependency.read-after-write");

/**
 * chronos.dependency.remove-dependency
 * Remove dependency operation in the dependency family.
 * Status: standard; target: architecture-agnostic; differentiability: not-applicable.
 */
export const dependencyRemoveDependency = defineStub<ScheduleOpRequest>("chronos.dependency.remove-dependency");

/**
 * chronos.dependency.resolve-dependencies
 * Resolve dependencies operation in the dependency family.
 * Status: standard; target: architecture-agnostic; differentiability: not-applicable.
 */
export const dependencyResolveDependencies = defineStub<ScheduleOpRequest>("chronos.dependency.resolve-dependencies");

/**
 * chronos.dependency.topological-order
 * Topological order operation in the dependency family.
 * Status: standard; target: architecture-agnostic; differentiability: not-applicable.
 */
export const dependencyTopologicalOrder = defineStub<ScheduleOpRequest>("chronos.dependency.topological-order");

/**
 * chronos.dependency.write-after-read
 * Write after read operation in the dependency family.
 * Status: standard; target: architecture-agnostic; differentiability: not-applicable.
 */
export const dependencyWriteAfterRead = defineStub<ScheduleOpRequest>("chronos.dependency.write-after-read");

/**
 * chronos.dependency.write-after-write
 * Write after write operation in the dependency family.
 * Status: standard; target: architecture-agnostic; differentiability: not-applicable.
 */
export const dependencyWriteAfterWrite = defineStub<ScheduleOpRequest>("chronos.dependency.write-after-write");
