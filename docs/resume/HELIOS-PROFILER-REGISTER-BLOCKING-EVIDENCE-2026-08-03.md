# Helios exact profiler and portable register-blocked GEMM evidence

**Date:** 2026-08-03
**Status:** implementation validated on the NVIDIA reference device; physical AMD validation still open
**Scope:** Helios engine work that precedes the full Alpha foundation run
**Model contract used for end-to-end measurements:** 97,098,880 parameters, sequence length 1,024, batch 24, gradient accumulation 1, SwiGLU/RMSNorm/RoPE, byte BPE 12,288, cooperative matrix disabled, Symbiogenesis disabled

## 1. Outcome

Helios now has an exact, opt-in per-dispatch GPU timestamp profiler and a new portable scalar-FP32 register-blocked GEMM family. On the reference RTX 4090, the new kernel reduced the measured GPU time of the three dominant generic GEMM families by approximately 45–50%, reduced total measured dispatch time for the exact one-step training graph by 36.9%, and increased matched steady-state training throughput by 26.1%.

The six-step training trajectory was identical at the precision printed by the trainer:

| Step | Loss | Gradient norm | Reference | Register-blocked |
|---:|---:|---:|---:|---:|
| 1 | 9.5427 | 4.678 | exact match | exact match |
| 2 | 9.2181 | 22.074 | exact match | exact match |
| 3 | 8.3198 | 0.692 | exact match | exact match |
| 4 | 7.8000 | 0.455 | exact match | exact match |
| 5 | 7.5657 | 0.304 | exact match | exact match |
| 6 | 7.5105 | 0.299 | exact match | exact match |

The final one-iteration held-out validation loss was `7.6964` in both runs. The optimizer clipping coefficients and learning-rate trajectory also matched.

This is an engine improvement, not yet a model-quality improvement. It does not justify a Discord model-sample announcement or a new Hugging Face/blah.dev model version by itself.

## 2. Exact device fingerprint

The primary measurements were made on the currently rented RunPod device:

| Field | Value |
|---|---|
| Device | NVIDIA GeForce RTX 4090 |
| Vendor/device IDs | `0x10de` / `0x2684` |
| Driver version | `2559230080` |
| Vulkan API | `4211029` |
| Reported device-local heap | 51,784,974,336 bytes |
| Maximum workgroup invocations | 1,024 |
| Maximum workgroup size | 1,024 × 1,024 × 64 |
| Maximum compute shared memory | 49,152 bytes |
| Native subgroup | 32 lanes |
| Timestamp valid bits | 64 |
| Timestamp period | 1 ns |
| f16 | supported |
| Push descriptors | supported |
| Buffer device address | supported |
| Device-generated commands | supported |

The capability response is now produced by the native Vulkan bridge and is available to TypeScript kernel selection. The new kernel itself requires only 256 workgroup invocations, a 16 × 16 workgroup, 4 KiB of shared memory, scalar FP32 storage/arithmetic, and ordinary Vulkan compute/storage-buffer support. It has no NVIDIA-vendor check, subgroup intrinsic, CUDA dependency, or cooperative-matrix dependency.

## 3. Profiler implementation

### 3.1 What is measured

`HELIOS_PROFILE_GPU_TIMESTAMPS=1` changes a normal Helios graph flush into one semantically identical profiled submission:

1. allocate/reset a separate timestamp-query pool;
2. record a batch start timestamp;
3. record start/end timestamps around every dispatch in the original command buffer;
4. preserve the graph's original barriers, descriptors, push constants, write masks, and dispatch order;
5. record a batch end timestamp;
6. submit the graph exactly once;
7. wait only for diagnostic readback;
8. return batch and per-dispatch times to the existing graph statistics.

Stateful optimizer and in-place operations are never replayed for timing. This matters: timing an optimizer by running it repeatedly would mutate parameters and invalidate the training trajectory.

The trainer can report:

- operation and flush counts;
- timestamped flushes;
- total batch GPU time;
- sum of dispatch intervals;
- GPU time grouped by semantic operation kind;
- GPU time grouped by exact kernel name.

The profiling path is deliberately diagnostic-only because synchronous timestamp readback changes scheduling and wall-clock throughput. Production tokens/s is always measured without it.

### 3.2 Timestamp-period bug found by the profiler

The first profiler smoke run reported a nonsensical duration approximately 16 times too large. The native bridge had read `VkPhysicalDeviceLimits.timestampPeriod` from a hard-coded byte offset. That offset was wrong for the actual ABI.

The bridge now defines the exact Khronos field prefix through `timestampPeriod`, respecting native alignment, and reads the typed field. The reference device reports the correct 1 ns period. A safe three-operation smoke graph now reports approximately 9.25 microseconds for the batch and 7.55 microseconds across its three dispatches, with the expected numerical result.

### 3.3 Profiler smoke contract

`scripts/smoke-helios-dispatch-profiler.mjs` verifies:

- exactly three original graph operations execute;
- exactly one timestamped flush occurs;
- batch and per-dispatch durations are positive;
- every named kernel receives a duration;
- the arithmetic result matches an independent CPU expression;
- the device capability record is emitted with the artifact.

## 4. Bottleneck result on the exact model graph

With the selected recipe and historical generic GEMMs, the exact one-step profile contained 2,431 GPU operations in eight profiled flushes. The sum of dispatch intervals was 3,371,457.1 microseconds. Generic matmul consumed 2,703,852.7 microseconds, or approximately 80.2% of measured dispatch time.

The three dominant kernels were:

| Kernel | Count | GPU time (µs) |
|---|---:|---:|
| `matmul_transposed_T32` | 91 | 1,048,073.2 |
| `matmul_transposed_a_T32` | 91 | 950,950.3 |
| `matmul_T32` | 91 | 665,813.8 |

This superseded the earlier operation-count-only conclusion that backward graph fragmentation should necessarily be optimized first. Counts still matter, but exact time proved that the immediate bottleneck was the generic FP32 GEMM implementation.

## 5. Tile-selection experiment and rejected artifact

### 5.1 Static tile comparison

The old heuristic selected a 32 × 32 workgroup for large output planes, meaning 1,024 invocations computed one output each. Forcing the existing 16 × 16 kernel on the exact graph produced:

| Metric | Historical selected recipe | Forced tile 16 |
|---|---:|---:|
| Dispatch GPU time (µs) | 3,371,457.1 | 3,216,809.1 |
| Matmul GPU time (µs) | 2,703,852.7 | 2,541,982.1 |
| Six-step steady median tokens/s | 3,579.0 | 3,729.5 |

The forced tile-16 path improved steady median throughput by 4.21% and preserved the complete six-step printed trajectory.

### 5.2 Capability- and shape-aware autotuner

Helios gained an opt-in selector that:

- keys decisions by vendor ID, device ID, driver version, kernel family, M/N/K, and batch size;
- permits tile 32 only when Vulkan reports at least 1,024 workgroup invocations and an X dimension of at least 1,024;
- resolves graph-produced inputs before standalone probes;
- uses the exact output region reserved for the real operation, avoiding a transient duplicate allocation for vocabulary-scale projections;
- records raw samples and the selected reason;
- caches the decision for later occurrences in the process;
- never runs inside exact full-graph profiling.

The first autotuner version measured tile 16 first and tile 32 second. On one cold large projection it observed approximately 39.2 ms versus 11.9 ms and incorrectly chose tile 32, even though a prior run had measured the same tile-16 shape near 10 ms. This was a clock-ramp/order artifact, not a kernel property.

That selector was rejected. The corrected estimator:

1. compiles and prewarms every legal candidate;
2. measures the candidates in forward order;
3. measures them again in reverse order;
4. stores both raw samples;
5. compares warm minimum times.

The corrected 97M run selected tile 16 for every observed training shape. It reproduced the exact trajectory. Its very short post-tuning throughput window was noisy and did not beat the earlier fixed tile-16 control, so no additional speed claim is made for autotuning yet. Persistent, fingerprinted tuning-cache publication remains future work.

This failure is retained because it demonstrates why autotuning needs experimental design, not merely timestamp availability.

## 6. New portable 2 × 2 register-blocked kernel

### 6.1 Algorithm

The new non-batched generic GEMM family uses:

- workgroup: 16 × 16 = 256 invocations;
- output tile: 32 × 32;
- reduction tile: 16;
- per-thread outputs: 2 × 2;
- per-thread FP32 accumulators: four;
- shared A tile: 32 × 16 FP32;
- shared B tile: 16 × 32 FP32;
- total shared memory: 4,096 bytes.

Each shared A value is consumed by two output columns and each shared B value by two output rows. The kernel therefore preserves the large output tile while reducing workgroup size and increasing value reuse. Three layout-aware variants avoid physical transposes:

- `matmul_R2` for `A × B`;
- `matmul_transposed_R2` for `A × Bᵀ` with B stored untransposed;
- `matmul_transposed_a_R2` for `Aᵀ × B` with A stored untransposed.

The implementation is selected with `HELIOS_MATMUL_REG2X2=1` while cross-device evidence is accumulated. Unsupported capability or batched cases retain the existing tiled implementation and report the limitation.

### 6.2 Numerical smoke coverage

The smoke test uses deliberately awkward dimensions `M=113`, `N=157`, `K=93`, crossing every 16/32 tile boundary. It checks all three layout variants against independent CPU loops.

On the RTX 4090 the maximum absolute errors were:

| Variant | Maximum absolute error |
|---|---:|
| `A × B` | `5.587935447692871e-8` |
| `A × Bᵀ` | `3.814697265625e-6` |
| `Aᵀ × B` | `1.9371509552001953e-7` |

The same odd-shape test also passes through Mesa llvmpipe. That is useful compiler/layout evidence but is not physical AMD proof.

### 6.3 Exact graph profile

| Metric | Forced tile 16 | Register-blocked 2 × 2 | Change |
|---|---:|---:|---:|
| Total dispatch GPU time (µs) | 3,216,809.1 | 2,030,423.3 | −36.9% |
| Generic matmul GPU time (µs) | 2,541,982.1 | 1,361,243.1 | −46.5% |
| `matmul_transposed*` (µs) | 1,033,492.5 | 537,773.1 | −48.0% |
| `matmul_transposed_a*` (µs) | 814,445.8 | 449,928.8 | −44.8% |
| `matmul*` (µs) | 657,179.9 | 328,978.9 | −49.9% |

Other kernel families retained approximately the expected times, including flash-attention backward DKV at 263,820.3 microseconds and scale at 107,946.6 microseconds. This supports attribution of the gain to GEMM rather than a different graph.

### 6.4 Matched production throughput

Production timing excludes full-graph profiling and all checkpoint writes. The comparison uses the same config hash `fa0ac879`, seed, data cache, validation cache, optimizer, and six data batches.

| Path | Steps 2–5 tokens/s | Median |
|---|---|---:|
| Historical tile heuristic | 3,620, 3,524, 3,557, 3,601 | 3,579.0 |
| Forced tile 16 | 3,692, 3,803, 3,706, 3,753 | 3,729.5 |
| Register-blocked 2 × 2 | 4,512, 4,447, 4,541, 4,514 | 4,513.0 |

The register-blocked path is:

- 26.10% faster than the historical selected recipe;
- 21.01% faster than the stronger forced-tile-16 control.

The first step remains excluded because pipeline construction, allocator warmup, and initial runtime state are not representative of repeated training.

## 7. Validation performed

### Local host

- native addon build: pass;
- monorepo TypeScript build: pass;
- profiler arithmetic smoke: pass;
- autotuner smoke: pass;
- register-blocked odd-shape smoke through llvmpipe: pass;
- focused device-capability tests: 4 pass;
- local GPU-gated suite skips remain expected on the software device.

### RTX 4090 RunPod

- native addon build: pass;
- monorepo TypeScript build: pass;
- exact timestamp profiler smoke: pass;
- autotuner smoke: pass;
- register-blocked odd-shape smoke: pass;
- six-step trajectory: pass;
- register-blocked test selection:
  - 4 test files passed;
  - 105 tests passed;
  - operation gradient checks passed;
  - model gradient checks passed;
  - Helios parity suite passed;
  - 20-step finite/decreasing-loss test passed;
  - 100-step final-loss parity test passed.

No diagnostic checkpoint was written. `ALPHA_DISABLE_CHECKPOINTS=1` is now available for bounded profiling and throughput runs, preventing another multi-hundred-megabyte or gigabyte artifact from being created solely to time a few steps.

## 8. Rejected cooperative-backward experiment

An opt-in attempt to leave cooperative matrices enabled during backward was tested and rejected:

- it exhausted memory before the second step;
- the first step was slower;
- its gradient norm was grossly different from the safe path;
- cast/cache pressure increased sharply.

The safe default remains to pause the current cooperative path during backward. This does not reject cooperative matrix research generally; it rejects the present naive backward route.

## 9. AMD status—precise, not aspirational

Completed:

- vendor-name NVIDIA training gate replaced by capability-based assessment;
- AMD RDNA-like wave32 capability fixture admitted by the same rules as NVIDIA;
- wave64 limitation reported as a 32-lane kernel-layout blocker rather than as “AMD unsupported”;
- workgroup limits and shared-memory limits exposed by the native Vulkan bridge;
- new dominant GEMM path uses no subgroup or vendor-specific operation;
- tile 32 is no longer assumed legal on every discrete GPU;
- odd-shape scalar FP32 kernel compiles and executes through a non-NVIDIA Mesa Vulkan implementation.

Still open:

- no physical AMD Vulkan device has executed the suite;
- no AMD driver/device fingerprint or throughput exists;
- current attention/reduction kernels still contain 32-lane assumptions;
- no wave64 lowering exists for Instinct/CDNA;
- RunPod currently lists only NVIDIA GPU types for this account, so it cannot supply the required physical AMD proof;
- ROCm/HIP is still a planned second lowering, not an implemented backend.

Therefore the honest claim is: **Helios has a substantially more AMD-portable dominant GEMM and capability substrate; full AMD support is not yet proven.**

## 10. Storage and cost discipline

The bounded work produced source changes, small JSON/log/config artifacts, and no new model checkpoints. Two earlier disposable random-initialized one-step checkpoint files were removed only after their diagnostic status was confirmed; their logs and configs were preserved. Foundation token caches and learning-rate pilot checkpoints were untouched.

Local mounted-disk headroom is currently much lower than the older handbook snapshot, so future work must monitor it. This work did not approach the user's 15 GiB pause threshold.

## 11. Next gates

1. Run the ordinary default-path suite once more to prove the experimental kernel did not change the default.
2. Commit and push this evidence and implementation.
3. Obtain physical AMD hardware from a provider that actually exposes Radeon Vulkan or implement the ROCm/HIP lowering for an available Instinct rental.
4. Add the register-blocked family as a third autotuner candidate only after physical cross-device results; do not make it a universal hardcoded default from one NVIDIA device.
5. Profile the new graph again and move to the next time-ranked operations:
   - flash-attention backward DKV;
   - scale/materialized unary graph;
   - column-sum/reduction path;
   - transposes and backward quotienting.
6. Preserve one strong published baseline and one falsifiable Helios-specific hypothesis for every operation family.
7. Freeze the fastest numerically valid one-GPU recipe, recalculate full-run time/cost, then resume Alpha foundation training and conversational post-training.

## 12. Scientific interpretation

The register-blocked algorithm is established GPU practice; the novelty claim is not that Helios invented per-thread output blocking. The contribution here is an inspectable from-scratch SPIR-V implementation integrated into a custom tensor/autograd/training engine, its exact state-safe profiler, capability-based cross-vendor contract, and full trajectory evidence.

The creative research lane begins where fixed templates end: operation-graph quotienting, semantic megakernel boundaries, sensitivity-budgeted precision, temporal memory coloring with recomputation edges, and cross-device kernel evolution. Those ideas must earn their place against this now much stronger baseline. A novel name without an equal-contract control is not a Helios result.
