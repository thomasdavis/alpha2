# Helios Performance Analysis & Optimization Report

## Executive Summary
As of March 2026, Helios (Alpha's custom Vulkan compute engine) is functional on NVIDIA L4 GPUs but trails CUDA/PyTorch performance by approximately 9x. This gap is primarily due to suboptimal "feeding" of the Tensor Cores and high CPU-side dispatch overhead. This report outlines a prioritized roadmap to reach 1.25x CUDA parity.

---

## 1. Kernel-Level Optimizations (Target: 4x-5x Gain)
The L4's Tensor Cores are high-throughput but require extremely efficient data movement to stay saturated.

### 1.1 Refined Double-Buffering & Software Pipelining
*   **Current State:** Experimental double-buffering exists but is performance-neutral.
*   **Action:** Rewrite the software pipeline in `matmul-coop.ts` to strictly overlap global/shared memory loads with `OpCooperativeMatrixMulAddKHR`. Ensure barriers are placed to allow maximum occupancy without stalling the MMA pipe.
*   **Impact:** Essential for hiding the L4's memory latency.

### 1.2 Register Tiling & Occupancy Tuning
*   **Current State:** Mostly 1x1 or small tiling; static workgroup sizes.
*   **Action:** Implement 2x2 and 4x2 register tiling in `matmul-coop.ts`. Each subgroup should compute multiple output tiles to amortize shared memory load costs.
*   **Action:** Add an auto-tuner that benchmarks `regTiles`, `kMulti`, and `wgSize` combinations on startup for the specific L4/driver environment.

### 1.3 Direct-to-Reg Path (L2-Focused)
*   **Current State:** Shared memory is the primary path; direct global loads are currently slower.
*   **Action:** On L4, if data is prepacked (see 3.1), loading directly from global memory into cooperative registers can bypass shared memory bank conflicts and latency.

---

## 2. Runtime & Dispatch Efficiency (Target: 2x Gain)
Reducing the "CPU tax" is critical, especially for transformer blocks where many small dispatches occur.

### 2.1 Expand DGC (Device-Generated Commands)
*   **Current State:** DGC is limited to binary ops.
*   **Action:** Implement DGC for **Matmul and Flash Attention**. This allows recording an entire training step as a single GPU-side sequence, eliminating almost all CPU-side descriptor set management and N-API overhead.

### 2.2 Async Staging & Transfer Overlap
*   **Current State:** Large uploads use blocking copies.
*   **Action:** Fully implement the Async Staging Ring for all buffer sizes. Utilize the L4's dedicated transfer queue to overlap weight/data uploads with active computation.

### 2.3 Strict Fine-Grained Barriers
*   **Current State:** Occasional fallback to global barriers.
*   **Action:** Enforce `writeMask` usage across all `backend.ts` call sites. Transition the compute graph to explicit dependency tracking to emit the narrowest possible `VkBufferMemoryBarrier` calls.

---

## 3. Structural & Architectural Wins

### 3.1 Persistent Weight Prepacking
*   **Action:** Before training starts, transform model weights into the "native" layout preferred by the L4 Tensor Cores (tiled/padded). This eliminates runtime transpose costs and enables the high-speed Direct-to-Reg path.

### 3.2 Epilogue Fusion (GEMM + Activation)
*   **Action:** Fuse Matmul kernels with subsequent Bias-Add and Activation (GELU/ReLU) steps. This keeps data in registers/L2 and eliminates redundant VRAM round-trips.

---

## Conclusion
Helios has a solid foundation with its from-scratch SPIR-V assembler and Vulkan bridge. By moving from a "generic" compute approach to an **L4-optimized pipeline** (DGC, Prepacking, and Fused Kernels), we can bridge the 9x gap and achieve production-grade training performance.
