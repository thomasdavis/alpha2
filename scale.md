# Alpha Scaling Roadmap

Goal: Train state-of-the-art models on 100+ H100s at 90%+ utilization. All custom code — no NCCL, no DeepSpeed, no Megatron, no external ML frameworks. We write everything.

## Current State (Feb 2025)

- Single GPU training (L4 23GB / A100 80GB)
- ~10K tok/s on L4 with 6.8M param model
- Custom Vulkan/SPIR-V compute backend (Helios)
- Custom autograd, tokenizers, training loop — all TypeScript
- Peak VRAM utilization: ~28% on L4 (limited by activation memory, not compute)
- Bottleneck: Vulkan kernel launch overhead (~1ms per dispatch), not raw compute

## Phase 0: Instrumentation + Performance Model (NEXT)

Before optimizing anything, we need to measure everything. "90% utilization" must be a number we can read off a dashboard, not a hope.

### Per-Step Timing Breakdown

Every training step must be decomposed into:

| Component | What to measure |
|-----------|----------------|
| **forward** | Time from first op to loss value |
| **backward** | Time from loss.backward() to all gradients computed |
| **optimizer** | Time for param update (AdamW step) |
| **grad_clip** | Time for norm computation + clipping |
| **data_load** | Time to fetch next batch from disk/memory |
| **host→device** | Time to upload batch tensors to GPU |
| **device→host** | Time to read back loss/metrics from GPU |
| **comm** | Time for gradient synchronization (future: all-reduce, all-gather) |
| **checkpoint** | Time for serialization + disk write |
| **sample_gen** | Time for inference sample generation |
| **idle/bubble** | Any dead time between components |

### Per-Op GPU Profiling

For every Vulkan dispatch:
- Host enqueue time (command buffer recording)
- Device execution time (GPU-side, via timestamp queries)
- Sync wait time (fence/semaphore blocking)
- Memory bandwidth achieved vs theoretical peak
- Kernel launch count per step

### Metrics to Track

| Metric | Formula | Target |
|--------|---------|--------|
| **MFU** (Model FLOPS Utilization) | actual_flops / theoretical_peak_flops | >50% (good), >80% (excellent) |
| **HFU** (Hardware FLOPS Utilization) | actual_flops / (peak_flops × time) | >40% |
| **Tokens/sec/GPU** | batch × seq_len / step_time | maximize |
| **Memory bandwidth utilization** | bytes_moved / (peak_bw × time) | >60% |
| **Kernel launch overhead** | total_dispatch_time - total_compute_time | minimize |
| **Pipeline bubble ratio** | idle_time / total_time | <10% |
| **Communication overhead** | comm_time / step_time | <15% |

### Implementation

1. **Trainer instrumentation** — Wrap each phase (forward, backward, optimizer, etc.) with `performance.now()` timestamps. Report as part of step metrics.

2. **GPU timeline** — Use Vulkan timestamp queries (`vkCmdWriteTimestamp`) in the native addon to measure actual GPU execution time per kernel. Export as Chrome trace format (JSON) for visualization in `chrome://tracing` or Perfetto.

3. **Op-level profiling** — Each backend op (matmul, softmax, layernorm, etc.) records its shape, dispatch time, and memory traffic. Aggregate into per-op statistics.

4. **DB storage** — New tables:
   - `step_timing` — per-step breakdown (forward_ms, backward_ms, optimizer_ms, comm_ms, etc.)
   - `op_profile` — per-op statistics aggregated per step (op_name, avg_ms, count, bytes_moved)
   - `gpu_timeline` — raw timestamp data for timeline visualization

5. **Dashboard** — New page at `/profiling`:
   - Step-time waterfall chart (stacked bar: forward, backward, optimizer, comm, idle)
   - MFU/HFU over time
   - Per-op breakdown (top-10 slowest ops, kernel launch count)
   - Memory bandwidth utilization
   - Timeline viewer (embed Perfetto or custom canvas)

### FLOPS Calculation

For a transformer with L layers, H hidden dim, S sequence length, B batch size, V vocab size:

```
forward_flops_per_token ≈ 2 × params
backward_flops_per_token ≈ 4 × params (2x forward for gradient computation)
total_flops_per_step = 6 × params × B × S

For our 6.8M param model, batch=512, seq=256:
  = 6 × 6.8M × 512 × 256
  = 5.35 TFLOPS per step

L4 peak FP32: 30.3 TFLOPS
At 13.8s/step: achieved = 5.35T / 13.8 = 0.39 TFLOPS = 1.3% MFU

H100 peak FP16: 989 TFLOPS
Target: 50%+ MFU = 495+ TFLOPS
```

This tells us our current Vulkan backend is operating at 1.3% MFU — massive room for improvement even before multi-GPU.

## Phase 1: Single GPU Efficiency

### Activation Checkpointing

Trade compute for memory: recompute forward activations during backward instead of storing them all.

- Mark checkpoint boundaries in the Tape (e.g., every transformer layer)
- During backward, re-run forward from last checkpoint to reconstruct intermediates
- ~2x larger batch sizes per GPU
- ~33% more compute (recompute forward once during backward)

### Kernel Fusion

Current: each op is a separate Vulkan dispatch with full pipeline flush.

Fuse common sequences into single SPIR-V kernels:
- `layernorm → linear → gelu` → single fused kernel
- `softmax → dropout → matmul` → fused attention kernel
- `add → layernorm` → fused residual + norm

Write a simple op-graph compiler that identifies fusible sequences and generates combined SPIR-V.

### Persistent Command Buffers

Currently rebuilding Vulkan command buffers every dispatch. Pre-record command buffers for the static computation graph and replay them, only updating buffer bindings.

### Vulkan Timeline Semaphores

Replace fence-based synchronization with timeline semaphores for overlapped compute/transfer. Pipeline data loading with computation.

## Phase 2: Multi-GPU Single Node (2-8 GPUs)

### Custom Collective Operations

Build from scratch — no NCCL:

```typescript
// packages/comm/src/collectives.ts
interface CollectiveOps {
  // Core collectives — all custom implementations
  allReduce(tensor: TensorData, op: 'sum' | 'avg'): Promise<TensorData>;
  allGather(tensor: TensorData): Promise<TensorData>;
  reduceScatter(tensor: TensorData, op: 'sum'): Promise<TensorData>;
  broadcast(tensor: TensorData, root: number): Promise<TensorData>;
  barrier(): Promise<void>;
}
```

Implementation layers:
1. **Shared memory** — for GPUs on same node via Vulkan external memory / POSIX shm
2. **PCIe peer-to-peer** — direct GPU-to-GPU transfers via Vulkan buffer sharing
3. **NVLink** — if available, use for high-bandwidth GPU-to-GPU (write custom NVLink kernel via Vulkan cooperative groups or native addon)
4. **Ring all-reduce** — custom ring topology for gradient synchronization
5. **Tree all-reduce** — for better latency on larger GPU counts

### Data Parallelism (DP)

- Each GPU runs the full model on different mini-batches
- After backward: all-reduce gradients using our custom ring all-reduce
- Optimizer step is identical on all GPUs (deterministic)
- Scaling: near-linear with good all-reduce implementation

### Tensor Parallelism (TP)

Split large matrices across GPUs:
- Attention: shard across heads (each GPU handles N/P heads)
- FFN: column-parallel first linear, row-parallel second linear
- Requires all-reduce after each TP layer
- Custom SPIR-V kernels for sharded matmul + fused communication

## Phase 3: Multi-Node (8-100+ GPUs)

### Custom Network Transport

Build from scratch:
- **RDMA over InfiniBand** — native addon wrapping `ibverbs` API directly
- Custom message passing protocol (no MPI)
- Zero-copy GPU→NIC→GPU transfers (GPUDirect RDMA via custom kernel)
- Topology-aware routing (detect NVLink/PCIe/IB topology, optimize data placement)

### Pipeline Parallelism (PP)

- Assign transformer layers to different GPUs/nodes
- 1F1B (one-forward-one-backward) micro-batch schedule to minimize bubble
- Custom micro-batch scheduler that maximizes pipeline occupancy
- Async inter-stage communication overlapped with compute

### Sharded Optimizer (ZeRO-style, custom)

- Stage 1: Shard optimizer states (Adam m, v) across GPUs — 4x memory reduction
- Stage 2: Also shard gradients — 8x memory reduction
- Stage 3: Also shard model parameters — near-linear memory scaling
- All custom: each GPU stores its shard, all-gather full params for forward
- Custom communication schedule to overlap gather with compute

### Gradient Compression

Reduce communication volume:
- FP16 gradients (already doing this)
- Custom 1-bit quantization with error feedback
- Top-K sparsification with custom sparse all-reduce
- Low-rank gradient approximation (PowerSGD-style, custom implementation)

## Phase 4: Frontier Scale Features

### Custom CUDA Backend

For H100 peak performance, add a CUDA backend alongside Vulkan:
- Custom CUDA kernels (not cuBLAS/cuDNN — write from scratch)
- Tensor Core matmuls via `mma.sync` PTX instructions
- FP8 support on Hopper architecture
- Warp-level primitives for fused kernels
- Flash Attention implementation (tiled, memory-efficient)

### Sequence Parallelism

For long sequences (8K-128K tokens):
- Ring attention: each GPU processes a chunk, passes KV cache to neighbor
- Custom ring communication for KV cache rotation
- Enables context lengths beyond single-GPU memory

### Mixture of Experts (MoE)

- Route tokens to specialized sub-networks
- Custom all-to-all communication for expert routing
- Load balancing via auxiliary loss (custom implementation)
- Each GPU hosts different experts
- Enables 10x+ parameter count at same compute cost

### Async Distributed Checkpointing

- Non-blocking checkpoint to NVMe/object storage
- Sharded format: each GPU saves its shard in parallel
- Custom checkpoint format that supports resharding (change GPU count between runs)
- Background compression + upload thread

## Scaling Math

### 100 H100 SXM (80GB each)

| Resource | Total |
|----------|-------|
| VRAM | 8 TB |
| FP16 compute | 98.9 PFLOPS |
| FP8 compute | 197.9 PFLOPS |
| NVLink (intra-node, 8 GPUs) | 900 GB/s bidirectional |
| InfiniBand (inter-node) | 400 Gb/s per node |

### Model sizes at scale

| GPUs | Max model size | Strategy |
|------|---------------|----------|
| 1 | ~7B (FP16) | Single GPU + activation ckpt |
| 8 (1 node) | ~70B | DP + TP |
| 64 (8 nodes) | ~500B | DP + TP + PP |
| 100+ | ~1T+ | DP + TP + PP + MoE + ZeRO |

### Target throughput

| Model | GPUs | Target tok/s | MFU |
|-------|------|-------------|-----|
| 7B | 8 | 200K | 60% |
| 70B | 64 | 150K | 55% |
| 175B | 100 | 80K | 50% |

## New Packages Needed

```
packages/comm        — collective operations, topology detection, transport layer
packages/profile     — instrumentation, timing, trace export, MFU calculation
packages/schedule    — distributed scheduling (DP, TP, PP placement)
packages/cuda        — CUDA backend (PTX/SASS generation from TS, tensor core kernels)
```

## Order of Operations

1. **Phase 0**: Instrumentation + performance model (measure before optimizing)
2. **Phase 1a**: Kernel fusion (biggest single-GPU win)
3. **Phase 1b**: Activation checkpointing (unlock larger batches)
4. **Phase 2a**: Custom collectives (shared memory, PCIe P2P)
5. **Phase 2b**: Data parallelism (first multi-GPU training)
6. **Phase 2c**: Tensor parallelism (larger models)
7. **Phase 3a**: Custom RDMA transport (multi-node)
8. **Phase 3b**: Pipeline parallelism + sharded optimizer
9. **Phase 4**: CUDA backend, MoE, sequence parallelism

Every line of code: ours. Every kernel: ours. Every byte over the wire: our protocol.
