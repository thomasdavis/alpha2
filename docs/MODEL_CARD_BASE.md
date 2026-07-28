---
language:
  - en
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
  - llama
  - alpha
  - vulkan
  - from-scratch
  - experimental
datasets:
  - HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled
---

# Alpha 60M Base

Alpha 60M Base is a 57,688,576-parameter causal language model trained from scratch with
[Alpha](https://github.com/thomasdavis/alpha2), a TypeScript tensor/autograd stack and hand-written
Vulkan compute backend. Every training step ran through Alpha and Helios on an RTX 3090; PyTorch and
Transformers were used only after training to verify and package the result.

This is the base checkpoint, not an assistant. It can continue English text, but its frozen chat
evaluation is intentionally poor. Use `ajaxdavis/alpha-60m-chat` for the instruction-tuned sibling
after that repository is published.

## Architecture

| Property | Value |
|---|---:|
| Parameters | 57,688,576 |
| Layers | 16 |
| Hidden width | 512 |
| Attention heads / KV heads | 8 / 8 |
| Head width | 64 |
| SwiGLU intermediate width | 1,408 |
| Context length | 1,024 |
| Vocabulary | 12,288-token Alpha byte-level BPE |
| Positions / normalization | RoPE (`theta=10000`) / RMSNorm (`eps=1e-5`) |
| Embeddings | Input/output weights tied |
| Export dtype | float32 |

The export is a standard `LlamaForCausalLM` repository: no custom code and no
`trust_remote_code=True`.

## Use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "ajaxdavis/alpha-60m-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)

inputs = tokenizer("Hello", return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

The tokenizer includes `<|user|>`, `<|assistant|>`, and `<|end_of_text|>` plus a chat template so the
base and chat exports have the same interface. Their presence does not make this base checkpoint an
instruction-following model.

## Training

- Exact Alpha source: `e561f66c7a88a5294e9cb74a4fc3afd6be167d4f`.
- 61,036 steps, batch 16, block 1,024: exactly 1,000,013,824 training tokens.
- AdamW: `lr=1e-3` cosine-decayed to `1e-4`, 610 warmup steps, betas `(0.9, 0.95)`,
  `eps=1e-8`, weight decay `0.1`, gradient clip `1.0`.
- Full float32 training on one NVIDIA RTX 3090 through Helios Vulkan, with BDA and device-generated
  commands enabled, cooperative-matrix kernels disabled, and CPU fallback forbidden.
- Median post-warmup throughput: 3,882.35 tokens/second. The sealed iteration timings total
  71.7355 GPU-hours, or $15.78 at the pod's $0.22/hour rate. That figure excludes evaluation,
  checkpoint, interrupted-pod, storage, and operational overhead; energy was not independently
  metered.
- Terminal train loss: 2.9975; last-100 mean: 3.1011; final-three held-out validation mean: 3.1368.
- All 61,036 metric rows and all 57,688,576 terminal parameter values were finite; allocator
  telemetry reported zero free-range overflow.

The native terminal checkpoint SHA-256 is
`08e14fa9604bf1b46ebcd5df37933c84d2496c1d05d9e4b32ebad98792cc6049`.

## Data

Pretraining used the first three hash-pinned shards of
[HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled](https://huggingface.co/datasets/HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled),
a globally shuffled 50% FinePDFs-Edu / 30% DCLM / 20% FineWeb-Edu mixture. The upstream dataset is
released under ODC-By 1.0 and remains subject to its source and Common Crawl terms. Alpha trained on
one billion tokens from a 1.389-billion-token training split; the remainder was not repeated to reach
the target.

The model weights are released under Apache-2.0. That does not replace or relax the upstream data
licenses and terms.

## Verification and frozen evaluation

The exact exported `model.safetensors` has SHA-256
`d0aa2ccd171a7748a209921bbc8fbaf14a7fcb23a0e9cea050a8c9d47fbccbd9`. Stock Transformers loaded it
as `LlamaForCausalLM` with 57,688,576 parameters. Against Alpha's own float32 CPU reference forward,
the export achieved 2/2 top-1 agreement, exact tokenizer parity, and maximum absolute logit difference
`6.771e-05` (threshold `1e-3`). Both plain-text and message-list `pipeline("text-generation")` cold
loads completed without custom code.

Greedy frozen evaluation used 100 chat prompts at up to 128 generated tokens and 200 closed-book QA
items at up to 64 generated tokens:

| Metric | Base result |
|---|---:|
| Chat structural pass | 0 / 100 |
| EOS termination | 0 / 100 |
| User-role leaks | 0 / 100 |
| Degenerate repetition loops | 99 / 100 |
| Mean / maximum 4-gram repeat rate | 0.81256 / 0.98400 |
| Closed-book exact match | 0 / 200 |
| Answer contained | 1 / 200 |
| Mean token F1 | 0.000238 |

These failures are reported because this checkpoint is a small base model, not because the suite was
optional. The frozen chat and QA inputs were finalized before flagship training and were excluded from
the training corpora by an exact 13-gram overlap audit.

## Limitations

- This is a research artifact at only 58M parameters and one billion pretraining tokens. It has weak
  factual recall, reasoning, long-form stability, and multilingual ability.
- It is not safety-aligned and must not be relied on for medical, legal, financial, security, or other
  consequential advice.
- The upstream web mixture can contain errors, bias, personal information, unsafe content, and
  copyrighted text. The model may reproduce or transform such material.
- Contexts above 1,024 tokens are unsupported even if a tokenizer API reports a larger generic limit.
- Greedy output can loop badly, as the frozen result demonstrates. Apply downstream safeguards if you
  experiment with the base checkpoint.

## Reproducibility

The training contract, manifests, failure-closed analyzers, RunPod bootstrap, NVIDIA parity tests, and
HF exporter are in the [Alpha repository](https://github.com/thomasdavis/alpha2). The base training
source is `e561f66`; the terminal-evaluation/export tooling source is
`c333bf247fbe87b85d01f3d34789b46615dd1034`.
