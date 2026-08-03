import assert from "node:assert/strict";
import test from "node:test";

import { parseGpuOpsLine, summarizeGpuOps } from "./summarize_helios_profile.mjs";

const series = "kinds=matmul:2/800.0us kernels=matmul_R42:2/800.0us";

test("older profiler logs remain parseable without host timing", () => {
  const sample = parseGpuOpsLine(
    `[gpu_ops] flushes=5 waited=1 dgc=0 ops_per_flush=2 timestamped=1 ` +
      `batch_gpu_us=900.0 dispatch_gpu_us=800.0 ${series}`,
  );
  assert.equal(sample.hostBuildMs, null);
  assert.equal(sample.gpuBlockingMs, null);
  assert.equal(sample.coreStepMs, null);
});

test("host and blocking measurements survive parsing and averaging", () => {
  const first = parseGpuOpsLine(
    `[gpu_ops] flushes=5 waited=1 dgc=0 ops_per_flush=2 timestamped=1 ` +
      `batch_gpu_us=900.0 dispatch_gpu_us=800.0 host_build_ms=120.0 ` +
      `gpu_blocking_ms=910.0 core_step_ms=1030.0 ${series}`,
  );
  const second = parseGpuOpsLine(
    `[gpu_ops] flushes=5 waited=1 dgc=0 ops_per_flush=2 timestamped=1 ` +
      `batch_gpu_us=920.0 dispatch_gpu_us=820.0 host_build_ms=140.0 ` +
      `gpu_blocking_ms=930.0 core_step_ms=1070.0 ${series}`,
  );
  const summary = summarizeGpuOps([first, second]);
  assert.equal(summary.averages.hostBuildMs, 130);
  assert.equal(summary.averages.gpuBlockingMs, 920);
  assert.equal(summary.averages.coreStepMs, 1050);
});
