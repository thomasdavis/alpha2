import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["src/**/*.test.ts"],
    testTimeout: 30000,
    // Helios owns one process-global Vulkan device, timeline, command rings,
    // allocator, and pipeline cache. Running GPU-bearing files concurrently in
    // one worker pool lets one file destroy or reset that singleton while
    // another file is still replaying a deterministic step. The resulting
    // failures can be only a few ulps and look like kernel nondeterminism.
    // Serialize files; individual tests and all GPU dispatches remain fully
    // parallel internally.
    fileParallelism: false,
  },
});
