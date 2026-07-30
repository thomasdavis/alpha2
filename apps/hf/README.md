---
title: Alpha 60M Research Artifact
emoji: "α"
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
models:
  - ajaxdavis/alpha-60m-chat
  - ajaxdavis/alpha-60m-training-checkpoints
pinned: false
---

# Alpha 60M research artifact

This Space serves the exact terminal Alpha 60M SFT checkpoint through Alpha's custom TypeScript CPU
inference engine. It is an inspectable research artifact, not a capable assistant.

> [!CAUTION]
> **The checkpoint failed its predeclared chat-quality gates.** The frozen 100-prompt evaluation
> produced 92 empty responses, six degenerate loops, and two unusable fragments. Closed-book QA scored
> 0/200. The interface exposes empty EOS responses explicitly and never substitutes another model.

The training mechanics and standard export passed: 57,688,576 finite parameters, 30,322/30,322 SFT
steps, and 2/2 Alpha/Transformers top-1 parity with maximum logit difference `7.153e-06`. Full evidence
and limitations are in the [model card](https://huggingface.co/ajaxdavis/alpha-60m-chat); native model,
optimizer, RNG, and restart state are in the
[recovery archive](https://huggingface.co/ajaxdavis/alpha-60m-training-checkpoints).

## API

The Space exposes an OpenAI-compatible endpoint. It supports an optional leading `system` message,
alternating `user`/`assistant` history ending in `user`, greedy or sampled decoding, and SSE streaming.
The context limit is 1,024 tokens and the server caps each completion at 256 tokens.

```bash
curl -X POST https://ajaxdavis-alpha-60m-chat.hf.space/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64,"temperature":0}'
```

```bash
curl https://ajaxdavis-alpha-60m-chat.hf.space/health
curl https://ajaxdavis-alpha-60m-chat.hf.space/evidence
curl https://ajaxdavis-alpha-60m-chat.hf.space/v1/models
```

An empty `choices[0].message.content` with `finish_reason: "stop"` is a genuine measured model result,
not a failed HTTP request. The response also includes `alpha.empty_eos` for non-streaming clients.

## Runtime provenance

- Native terminal checkpoint SHA-256: `6c279d086d8c0679495e38ebec8a473ac23d16bfb3b93516e144712963fecbc8`
- Training source commit: `c333bf247fbe87b85d01f3d34789b46615dd1034`
- Architecture: Llama-form 16L/512D/8H, float32, 12,288-token Alpha byte BPE
- Framework: Alpha tensor, tokenizer, checkpoint, and inference packages; no fallback model
