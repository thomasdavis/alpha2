# L4 Training Run Diagnostic Report

**Run ID:** `20260225_011650` / `20260225011732_xk5o`
**Date:** 2025-02-25
**Instance:** `alpha-train` (us-central1-b)
**GPU:** NVIDIA L4 24GB (`g2-standard-4`)
**Dataset:** `historic-chat-v2.txt` (34MB, ~13k conversations)
**Domain:** chat

---

## Model Config

| Parameter | Value |
|-----------|-------|
| Vocab size | 4,000 (BPE-4k) |
| Block size | 512 |
| Layers | 8 |
| Embedding dim | 384 |
| Heads | 8 |
| Dropout | 0.1 |
| Total params | ~17.4M |

## Training Config

| Parameter | Value |
|-----------|-------|
| Target iters | 50,000 |
| Completed steps | 4,853 / 50,000 (9.7%) |
| Batch size | 20 |
| Learning rate | 5e-5 (min: 5e-6) |
| Warmup | 1,000 steps |
| Beta1 / Beta2 | 0.9 / 0.95 |
| Weight decay | 0.1 |
| Grad clip | 1.0 |
| Backend | helios (Vulkan) |
| Optimizer | AdamW |

## Loss Curve

```
Step     Loss     LR          GradNorm  Tok/s   GPU%  VRAM(MB)
   1    8.3783   5.0e-06      1.1309    3,901    3%      15
 100    8.0451   9.5e-06      9.2243    5,187   10%   3,399
 500    7.1572   2.75e-05     0.6698    5,209   30%   3,023
1000    5.9530   5.0e-05      0.8458    4,772   38%   3,393
1500    5.4243   5.0e-05      0.9005    4,853    8%   3,125
2000    5.0687   5.0e-05      1.1060    4,738    7%   3,199
2500    4.9091   4.99e-05     1.0386    4,780    7%   3,202
3000    4.8472   4.98e-05     1.0989    4,677    6%   3,580
3500    4.7240   4.97e-05     1.1252    4,760    7%   3,206
4000    4.6689   4.96e-05     1.2611    4,652    9%   3,961
4500    4.6009   4.94e-05     1.3122    4,673   39%   2,834
4853    4.3960   4.93e-05     2.2035    4,657   12%   3,587
```

**Loss reduction:** 8.38 -> 4.40 (47.5% reduction)

## Timing Breakdown (per step, last step)

| Phase | Time | % of Step |
|-------|------|-----------|
| Forward pass | 1,094.5ms | 50.1% |
| Backward pass | 972.9ms | 44.5% |
| Grad norm | 57.1ms | 2.6% |
| Grad clip | 33.0ms | 1.5% |
| Optimizer step | 0.6ms | 0.0% |
| GPU flush | 28.4ms | 1.3% |
| Data loading | 0.1ms | 0.0% |
| **Total** | **~2,184ms** | **100%** |

## Throughput

| Metric | Value |
|--------|-------|
| Average tok/s | 4,870 |
| Min tok/s | 3,901 |
| Max tok/s | 5,249 |
| Tokens per step | 10,240 (batch=20 x block=512) |
| Time per step | ~2.1s |

## Gradient Stats

| Metric | Value |
|--------|-------|
| Avg clip % | 24.0% |
| Avg grad norm | 1.70 |
| Max grad norm | 326.62 (spike) |

## GPU Utilization

| Metric | Value |
|--------|-------|
| GPU | NVIDIA L4 |
| VRAM total | 23,034 MB |
| VRAM used | ~2,834-3,961 MB (12-17%) |
| GPU util | 6-39% (avg low) |
| Power | 56W / 72W cap |
| Temp | 78C |

## Resource Analysis

- **VRAM underutilized:** Only using ~15% of 23GB. Batch size could be increased significantly (to 60-80+) to fill VRAM.
- **GPU compute underutilized:** GPU utilization averaging well under 50%. The model (17.4M params) is small for an L4 GPU. Larger model dims or bigger batches would improve utilization.
- **Forward-dominant:** Forward pass (50%) + backward (45%) = 95% of step time, which is healthy.
- **Grad norm spike:** Max grad norm 326.62 indicates at least one large spike during training, but grad clipping at 1.0 kept it stable.

## Checkpoints

| Checkpoint | Step | File Size |
|------------|------|-----------|
| checkpoint-1000.json | 1,000 | 200MB |
| checkpoint-2000.json | 2,000 | 200MB |
| checkpoint-3000.json | 3,000 | 200MB |
| checkpoint-4000.json | 4,000 | 200MB |

**Last checkpoint saved locally:** `runs/checkpoint-l4-historic-chat-v2-step4000.json`

## Training Duration

| Metric | Value |
|--------|-------|
| Wall time | ~2.84 hours |
| Steps completed | 4,853 |
| Est. time to 50k | ~29.2 hours |
| Est. cost to 50k | ~$20.44 (at $0.70/hr) |

## System Info

- **Driver:** NVIDIA 570.211.01
- **CUDA:** 12.8
- **RAM:** 16GB (1.6GB used)
- **Uptime:** 2h 59m
- **Remote metrics:** Streaming to alpha2-production.up.railway.app

## Conclusion

The run was progressing normally with steady loss decrease. At ~9.7% completion, loss dropped from 8.38 to 4.40. The model is significantly underutilizing the L4 GPU (15% VRAM, <50% compute). For the next run on `super_chat.txt` (91MB, 226k conversations), consider:
- Increasing batch size to better fill GPU memory
- Using a more powerful GPU (A100/V100) for the 2.7x larger dataset
- The 29h estimated completion time makes the L4 expensive relative to an A100 which would be faster
