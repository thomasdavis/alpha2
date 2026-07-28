---
language:
  - en
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model: ajaxdavis/alpha-60m-base
tags:
  - llama
  - alpha
  - vulkan
  - from-scratch
  - conversational
  - experimental
datasets:
  - HuggingFaceTB/smol-smoltalk
  - HuggingFaceTB/smoltalk2
  - OpenAssistant/oasst2
  - allenai/soda
---

<!-- DRAFT ONLY: do not upload until the terminal SFT analyzer, frozen D3 gate, and human semantic review pass. -->

# Alpha 60M Chat

Alpha 60M Chat is the instruction-tuned sibling of
[Alpha 60M Base](https://huggingface.co/ajaxdavis/alpha-60m-base), a 57,688,576-parameter causal
language model trained from scratch with [Alpha](https://github.com/thomasdavis/alpha2). Every
pretraining and supervised-fine-tuning step ran through Alpha's TypeScript tensor/autograd stack and
hand-written Helios Vulkan backend on an RTX 3090. PyTorch and Transformers were used only after
training to verify and package the result.

This is a very small experimental assistant, not a frontier model. Its frozen results and limitations
are reported below without model-based grading.

## Use

```python
from transformers import pipeline

chat = pipeline("text-generation", model="ajaxdavis/alpha-60m-chat")
result = chat(
    [{"role": "user", "content": "Explain why the sky looks blue in two sentences."}],
    max_new_tokens=128,
    do_sample=False,
)
print(result[0]["generated_text"][-1]["content"])
```

The repository is a standard `LlamaForCausalLM` export with `model.safetensors`, `tokenizer.json`, and
a Jinja chat template. It requires no custom code and no `trust_remote_code=True`.

## Architecture

| Property | Value |
|---|---:|
| Parameters | 57,688,576 |
| Layers | 16 |
| Hidden width | 512 |
| Attention heads / KV heads | 8 / 8 |
| SwiGLU intermediate width | 1,408 |
| Context length | 1,024 |
| Vocabulary | 12,288-token Alpha byte-level BPE |
| Positions / normalization | RoPE (`theta=10000`) / RMSNorm (`eps=1e-5`) |
| Embeddings | Input/output weights tied |
| Export dtype | float32 |

## Training

- Initialized from the exact terminal Alpha 60M Base checkpoint, SHA-256 `08e14fa9604bf1b46ebcd5df37933c84d2496c1d05d9e4b32ebad98792cc6049`.
- Exact SFT source: `c333bf247fbe87b85d01f3d34789b46615dd1034`.
- One deterministic assistant-only masked-loss epoch: 30,322 steps, batch 16, block 1,024, and
  496,795,648 padded tokens over 485,150 training and 26,278 held-out conversations.
- AdamW: `lr=3e-4` cosine-decayed to `3e-5`, 303 warmup steps, betas `(0.9, 0.95)`, `eps=1e-8`,
  weight decay `0.1`, and gradient clip `1.0`. The learning rate was selected by a predeclared
  three-way `{1e-4, 3e-4, 1e-3}` pilot sweep on identical data and validation cadence.
- Full float32 training on one NVIDIA RTX 3090 through Helios Vulkan, with BDA and device-generated
  commands enabled, cooperative-matrix kernels disabled, and CPU fallback forbidden.

Terminal throughput, loss, parameter audit, GPU-hours, and cost will be inserted only after the
30,322-step analyzer passes.

## SFT data

The hash-pinned corpus contains 511,428 structurally validated English conversations:

| Source | Conversations | License / terms |
|---|---:|---|
| [Smol-SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) | 450,402 | Apache-2.0 dataset card; constituent-source terms may also apply |
| [SmolTalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) no-think everyday/system splits | 32,897 | Generated SmolTalk2 data is Apache-2.0; upstream-source terms remain applicable |
| [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) | 3,439 | Apache-2.0 |
| [SODA](https://huggingface.co/datasets/allenai/soda) deterministic hash sample | 24,690 | CC-BY-4.0 |

SODA is 4.828% of final rows. Exact duplicate rendered conversations were removed. Conversations over
the 1,024-token limit were shortened only at complete turn-pair boundaries or skipped; no assistant
answer was cut in half. System prompts were folded into the first user turn. Training supervised only
assistant content and its final EOS while masking user and formatting targets.

The frozen 100-prompt chat and 200-item closed-book QA suites were finalized before flagship training.
An exact 13-gram audit excluded overlap with both the pretraining and SFT corpora.

## Frozen evaluation

<!-- Replace this table only from the sealed terminal frozen-eval reports. -->

| Metric | Base | Chat |
|---|---:|---:|
| Chat structural pass | 0 / 100 | PENDING |
| EOS termination | 0 / 100 | PENDING |
| User-role leaks | 0 / 100 | PENDING |
| Degenerate repetition loops | 99 / 100 | PENDING |
| Mean / maximum 4-gram repeat rate | 0.81256 / 0.98400 | PENDING |
| Closed-book exact match | 0 / 200 | PENDING |
| Answer contained | 1 / 200 | PENDING |
| Mean token F1 | 0.000238 | PENDING |

Publication is blocked unless the chat checkpoint achieves at least 95/100 structural passes, zero
degenerate loops, a per-sample 4-gram repeat rate below 0.20, and a separate human review finds the
outputs conversational rather than gibberish. Closed-book QA is descriptive and will be reported as-is.

## Limitations

- At only 58M parameters, one billion pretraining tokens, and one SFT epoch, this model should be
  expected to have weak factual recall, reasoning, multilingual ability, and long-form stability.
- It is not safety-aligned and must not be relied on for medical, legal, financial, security, or other
  consequential advice.
- The web and conversation mixtures can contain errors, bias, personal information, unsafe content,
  and copyrighted text. The model may reproduce or transform such material.
- Contexts above 1,024 tokens are unsupported even if a tokenizer API reports a larger generic limit.
- Greedy frozen evaluation covers format, termination, repetition, and a small QA sanity set; it is not
  a comprehensive capability or safety evaluation.

## Reproducibility

The immutable SFT contract, source and data hashes, LR selector, failure-closed terminal analyzer,
frozen evaluator, NVIDIA parity tests, RunPod bootstrap, and HF exporter are in the
[Alpha repository](https://github.com/thomasdavis/alpha2). Final checkpoint/export hashes and Hub
cold-load evidence will be added after the release gate passes.
