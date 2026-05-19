# Alpha GPT-2 Mission Dossier: The Deep Dive

**Status:** Technical Architecture Fully Mapped | **Execution Target:** 72 Hours

This document provides a comprehensive, code-level analysis of the Alpha repository to execute the GPT-2 mission exactly as defined in `GPT2_ANYTHING_IS_POSSIBLE.md`, without requiring *any* source code modifications.

---

## 1. Core Target Architecture & Data

### The GPT-2 Implementation (`gpt.ts`)
Alpha implements the GPT-2 decoder-only transformer via `packages/model/src/gpt.ts`. The implementation supports standard configurations and modern enhancements like SwiGLU.

**Target `ModelConfig` for GPT-2 124M:**
```json
{
  "nLayer": 12,
  "nEmbd": 768,
  "nHead": 12,
  "blockSize": 1024,
  "vocabSize": 64000
}
```
*Note on L4:* If VRAM is too tight with large batches, the mission spec allows falling back to `--block=512`.

### The Tokenizer (`bpe-64k`)
Implemented in `packages/tokenizers/src/bpe.ts`, the `BpeTokenizer` is a greedy byte-pair encoder registered as `bpe-64k`.
*   **Training:** Automatically trains on a 100MB chunk (`loadTextSample`) when a `.tokens` cache is missing.
*   **Caching:** Saves raw binary token arrays (`.tokens` files) alongside the dataset to avoid re-encoding on remote nodes.
*   **Optimization:** It uses an $O(N \log N)$ amortized binary min-heap for extremely fast encode-time merge applications.

---

## 2. The Engine: Helios & DGC

Alpha completely bypasses PyTorch. The engine is a custom Vulkan backend (`packages/helios`) designed for zero-overhead execution.

### Device-Generated Commands (DGC)
The biggest performance leap in Helios is **DGC**, implemented in `packages/helios/native/helios_vk.c`.
Normally, the CPU must record a command for every GPU operation. DGC pushes this responsibility to the GPU.
*   **How it works:** `helios_vk.c` uses Vulkan's `VK_EXT_device_generated_commands`. Instead of descriptor sets, it uses **Buffer Device Address (BDA)** (`add_bda` kernels) to pass raw 64-bit GPU memory pointers directly to the shaders via push constants.
*   **Impact:** Reduces CPU-GPU sync time from milliseconds to microseconds.
*   **Control:** Ensure `--dgc=true` is passed to the CLI to activate this path.

### Memory: The Slab Allocator
To handle the millions of transient tensors in a 124M model backward pass, `helios_vk.c` uses a custom Slab Allocator.
*   **`devicePool`:** Long-lived parameter storage.
*   **`deviceTempPool`:** Short-lived intermediate activations.
*   If you encounter Vulkan allocation crashes on an L4, it is because you have exceeded the max driver allocation limit (~5500 buffers) and the slab allocator failed to consolidate them. **Keep `--device-batch-size=1` on L4 and rely heavily on `--accumSteps` (gradient accumulation).**

---

## 3. High-Performance Flags (The "Knobs")

Since you cannot modify the code, you must orchestrate the execution using these deep environment variables mapped in `packages/helios/src/backend.ts` and `apps/cli`.

| Variable | Purpose for GPT-2 | Hardware Target |
| :--- | :--- | :--- |
| `HELIOS_FLASH_ATTN=1` | Enables Flash Attention kernels (`attention-coop.ts`). Essential for `T=1024`. | All GPUs |
| `HELIOS_COOP_MATMUL=1` | Uses `VK_KHR_cooperative_matrix` for hardware Tensor Cores. | H100 / A100 |
| `HELIOS_WG_SIZE=256` | Sets workgroup size. L4 prefers `128` or `256` due to smaller SMs and register pressure. | L4 |
| `ALPHA_MIXED_PRECISION=1` | Activates `castToF16` in `gpt.ts` to compress activations, doubling batch size. | All GPUs |
| `ALPHA_MAX_PENDING_OPS=2048` | Expands the compute graph capacity before forcing a pipeline flush. | H100 |
| `ALPHA_FORCE_CPU_GRAD_NORM=1` | Bypasses an `f32` precision bug in the GPU reduction kernel for tensors > 16M elements (like the `wte` gradient). | L4 / A100 |

---

## 4. Execution & Orchestration

### Method A: The Fleet System (GCP Instances)
`apps/cli/src/commands/fleet.ts` acts as the orchestrator for remote execution.

**1. Deployment:**
The `deploy` command runs `scripts/bun-compile-safe.sh` to compile the TypeScript into a single standalone Linux binary (`bun-linux-x64-baseline`). It then ships this, alongside the pre-compiled `helios_vk.node` native C addon, to the target instance via SCP.
```bash
npm run fleet:deploy -- <instance> --rebuild-native
```

**2. Training Launch (The 72-Hour Loop):**
```bash
npm run fleet:train -- <instance> \
  --runtime=binary \
  --dgc=true \
  --no-fallback=true \
  --domain=concordance \
  --tokenizer=bpe-64k \
  --layers=12 \
  --dim=768 \
  --heads=12 \
  --block=512 \
  --batch=32 \
  --accumSteps=8 \
  --lr=6e-4
```
*   `--runtime=binary` forces the system to use the deployed Bun binary rather than the remote Node installation.
*   `--accumSteps=8` handles the memory constraints on L4 by keeping the physical micro-batch size low while maintaining the mathematical batch size.

### Method B: The Modal "Fast Path" (H100)
`scripts/modal_train.py` wraps the execution in a Modal app, provisioning an H100. It dynamically mounts the repository, builds the native C addon on the fly, and runs the training.

**Launch Command:**
```bash
./scripts/modal-run.sh data/concordance-v2.txt \
  --backend=helios \
  --tokenizer=bpe-64k \
  --domain=concordance \
  --layers=12 \
  --dim=768 \
  --heads=12 \
  --block=512 \
  --batch=32
```
*   The script automatically handles syncing the dataset to a Modal Volume (`alpha-datasets`) so it is not re-uploaded on resumes.

---

## 5. Resumption and The Final Ship Condition

If an instance is preempted or crashes, Alpha is designed to resume gracefully.
*   `trainer.ts` saves state to `checkpoint-*.json` files.
*   **Resume Command:**
    ```bash
    npm run fleet:resume -- <instance> --runtime=binary
    ```
    This scans the `runs/` directory, extracts the config from the latest checkpoint, and re-initializes the DataLoader and optimizer state exactly where it left off.

**The Bar for Success:**
Do not stop the loop until the model passes the brutal quality bar:
1. Responds coherently to `Hello`.
2. Follows user/assistant formatting.
3. Shows stable validation curves under `--evalInterval`.