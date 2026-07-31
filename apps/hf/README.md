---
title: Alpha Conversational Repair
emoji: "💬"
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
models:
  - ajaxdavis/alpha-60m-chat
  - ajaxdavis/alpha-60m-training-checkpoints
pinned: false
---

# Alpha conversational repair checkpoint

This Space talks to the selected Alpha corrective checkpoint. The static page calls Alpha's CPU inference
service at `https://donto.org/alpha-60m`; the service loads the exact native checkpoint and never substitutes
another model.

The earlier published checkpoint returned 92 empty responses in the sealed 100-prompt evaluation. The corrective
model now initiates and terminates ordinary replies reliably, but it remains semantically immature and still
fails the predeclared zero-loop quality gate.

> [!CAUTION]
> **Nonempty is not the same as intelligent.** On the final untouched suite, 55/100 replies passed the basic
> structural check, 70/100 were nonempty, 30/100 were empty, and 31/100 met the repetition-loop threshold.
> Closed-book QA was 0/200 exact. The model can be shallow, wrong, repetitive, or roleplay-like. It is a
> research artifact, not a dependable assistant or factual source.

The training intervention used deterministic epoch shuffling, equal conversation weighting, an 8x multiplier on
the first four assistant content tokens, independent 2x EOS weighting, and free-generation checkpoint selection.
A separate serving defect was also fixed: generation now ends exactly at `<|assistant|>` rather than appending a
standalone byte-BPE space token absent from training.

The standard export passed 87/87 Alpha/Transformers top-1 positions over five prompts, 5/5 exact tokenizer
comparisons, and maximum logit difference `5.531e-05`. Full results and limitations are in the
[model card](https://huggingface.co/ajaxdavis/alpha-60m-chat); native optimizer/RNG state and restart evidence are
in the [recovery archive](https://huggingface.co/ajaxdavis/alpha-60m-training-checkpoints).

## API

The backend exposes an OpenAI-compatible endpoint. It supports an optional leading `system` message,
alternating `user`/`assistant` history ending in `user`, greedy or sampled decoding, and SSE streaming. The
context limit is 512 tokens and the server caps each completion at 256 tokens.

```bash
curl -X POST https://donto.org/alpha-60m/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hey, how is your day going?"}],"max_tokens":64,"temperature":0}'
```

```bash
curl https://donto.org/alpha-60m/health
curl https://donto.org/alpha-60m/evidence
curl https://donto.org/alpha-60m/v1/models
```

## Runtime provenance

- Native selected checkpoint SHA-256: `399f776b49acc0c8834ff8a7f2390454e2c5f2d833a264e3f83ff546e973cfec`
- Checkpoint archive revision: `ffc447e8a0f2240d42ceb0abfd18ab5b427d5e60`
- Standard model revision: `ab1c5be13a12c0feb2d5e2c9af89bd5924a0e8b0`
- Training source commit: `57c065e35c7564688726dafb404efaff952d860b`
- Prompt-boundary fix: `cf4ad61`
- Architecture: Llama-form 16L/512D/8H, float32, 12,288-token Alpha byte BPE
- Framework: Alpha tensor, tokenizer, checkpoint, training, and inference packages; no fallback model

The same source can be deployed as a self-contained Docker Space if the Hub account has Docker Space
entitlement. The public static Space exists because the account's free tier does not provide that runtime.
