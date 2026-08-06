/* AUTO-GENERATED. Do not hand-edit; edit operation-registry.json. */
import { defineStub } from "../../../common/src/types";
import type { MemoryOpRequest } from "../../../common/src/types";

/**
 * gaia.integrity.capture-memory-manifest
 * Capture memory manifest operation in the integrity family.
 * Status: standard; target: host; differentiability: not-applicable.
 */
export const integrityCaptureMemoryManifest = defineStub<MemoryOpRequest>("gaia.integrity.capture-memory-manifest");

/**
 * gaia.integrity.checksum-allocation
 * Checksum allocation operation in the integrity family.
 * Status: research; target: host; differentiability: not-applicable.
 */
export const integrityChecksumAllocation = defineStub<MemoryOpRequest>("gaia.integrity.checksum-allocation");

/**
 * gaia.integrity.compare-memory-manifest
 * Compare memory manifest operation in the integrity family.
 * Status: standard; target: host; differentiability: not-applicable.
 */
export const integrityCompareMemoryManifest = defineStub<MemoryOpRequest>("gaia.integrity.compare-memory-manifest");

/**
 * gaia.integrity.detect-double-free
 * Detect double free operation in the integrity family.
 * Status: standard; target: host; differentiability: not-applicable.
 */
export const integrityDetectDoubleFree = defineStub<MemoryOpRequest>("gaia.integrity.detect-double-free");

/**
 * gaia.integrity.detect-leak
 * Detect leak operation in the integrity family.
 * Status: standard; target: host; differentiability: not-applicable.
 */
export const integrityDetectLeak = defineStub<MemoryOpRequest>("gaia.integrity.detect-leak");

/**
 * gaia.integrity.detect-out-of-bounds
 * Detect out of bounds operation in the integrity family.
 * Status: standard; target: host; differentiability: not-applicable.
 */
export const integrityDetectOutOfBounds = defineStub<MemoryOpRequest>("gaia.integrity.detect-out-of-bounds");

/**
 * gaia.integrity.detect-overlap
 * Detect overlap operation in the integrity family.
 * Status: standard; target: host; differentiability: not-applicable.
 */
export const integrityDetectOverlap = defineStub<MemoryOpRequest>("gaia.integrity.detect-overlap");

/**
 * gaia.integrity.detect-stale-mapping
 * Detect stale mapping operation in the integrity family.
 * Status: standard; target: host; differentiability: not-applicable.
 */
export const integrityDetectStaleMapping = defineStub<MemoryOpRequest>("gaia.integrity.detect-stale-mapping");

/**
 * gaia.integrity.detect-use-after-free
 * Detect use after free operation in the integrity family.
 * Status: standard; target: host; differentiability: not-applicable.
 */
export const integrityDetectUseAfterFree = defineStub<MemoryOpRequest>("gaia.integrity.detect-use-after-free");

/**
 * gaia.integrity.guard-allocation
 * Guard allocation operation in the integrity family.
 * Status: standard; target: host; differentiability: not-applicable.
 */
export const integrityGuardAllocation = defineStub<MemoryOpRequest>("gaia.integrity.guard-allocation");

/**
 * gaia.integrity.poison-allocation
 * Poison allocation operation in the integrity family.
 * Status: standard; target: host; differentiability: not-applicable.
 */
export const integrityPoisonAllocation = defineStub<MemoryOpRequest>("gaia.integrity.poison-allocation");

/**
 * gaia.integrity.scrub-sensitive-memory
 * Scrub sensitive memory operation in the integrity family.
 * Status: standard; target: host; differentiability: not-applicable.
 */
export const integrityScrubSensitiveMemory = defineStub<MemoryOpRequest>("gaia.integrity.scrub-sensitive-memory");

/**
 * gaia.integrity.secure-erase
 * Secure erase operation in the integrity family.
 * Status: standard; target: host; differentiability: not-applicable.
 */
export const integritySecureErase = defineStub<MemoryOpRequest>("gaia.integrity.secure-erase");

/**
 * gaia.integrity.unguard-allocation
 * Unguard allocation operation in the integrity family.
 * Status: standard; target: host; differentiability: not-applicable.
 */
export const integrityUnguardAllocation = defineStub<MemoryOpRequest>("gaia.integrity.unguard-allocation");

/**
 * gaia.integrity.unpoison-allocation
 * Unpoison allocation operation in the integrity family.
 * Status: standard; target: host; differentiability: not-applicable.
 */
export const integrityUnpoisonAllocation = defineStub<MemoryOpRequest>("gaia.integrity.unpoison-allocation");

/**
 * gaia.integrity.verify-checksum
 * Verify checksum operation in the integrity family.
 * Status: research; target: host; differentiability: not-applicable.
 */
export const integrityVerifyChecksum = defineStub<MemoryOpRequest>("gaia.integrity.verify-checksum");
