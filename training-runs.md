# Training Runs

Active and recent training runs tracked here.

## Active Runs

### Run: super_chat_l4_* (L4 — super_chat.txt)

| Field | Value |
|-------|-------|
| **Instance** | `alpha-train` |
| **Zone** | `us-central1-b` |
| **Machine** | `g2-standard-4` (L4 24GB) |
| **Dataset** | `super_chat.txt` (91MB, 226k conversations) |
| **Model** | 17.4M params — 384d, 8 heads, 8 layers, vocab 4000, block 512 |
| **Training** | 50k iters, batch 20, lr 5e-5, beta2 0.95, warmup 1000, grad clip 1.0 |
| **Backend** | helios |
| **Remote metrics** | Yes — streaming to alpha2-production.up.railway.app |
| **Discord** | Yes |
| **Started** | 2025-02-25 ~14:32 UTC |
| **Status** | Provisioning |
| **Flags** | `--stop-after` (auto-stop instance on completion) |
| **Code version** | Includes all Round 2 improvements (session pool, input validation, greedy decode, helios safety, flash attn dropout fix, streaming checkpoints) |

## Completed Runs

### Run: 20260225_011650 (L4 — historic-chat-v2)

| Field | Value |
|-------|-------|
| **Dataset** | `historic-chat-v2.txt` (34MB, ~13k conversations) |
| **Machine** | `g2-standard-4` (L4 24GB) |
| **Steps** | 4,853 / 50,000 (9.7%) |
| **Loss** | 8.38 → 4.40 |
| **Throughput** | ~4,870 tok/s avg |
| **Wall time** | ~2.84 hours |
| **Stopped** | 2025-02-25 — killed to free GPU slot |
| **Checkpoint** | `runs/checkpoint-l4-historic-chat-v2-step4000.json` (local) |
| **Diagnostic** | `L4_RUN_DIAGNOSTIC_20260225.md` |

## GPU Quota (GCP_PROJECT project)

| GPU | Limit | Notes |
|-----|-------|-------|
| NVIDIA_L4_GPUS | 1 | In use |
| NVIDIA_A100_80GB_GPUS | 0 | Quota increase requested (3 regions) |
| NVIDIA_V100_GPUS | 1 | Available but blocked by GPUS_ALL_REGIONS=1 |
| GPUS_ALL_REGIONS | 1 | Increase to 2 requested |
