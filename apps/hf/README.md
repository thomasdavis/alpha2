---
title: Alpha 60M Research Artifact
emoji: "α"
colorFrom: red
colorTo: gray
sdk: static
app_file: index.html
models:
  - ajaxdavis/alpha-60m-chat
  - ajaxdavis/alpha-60m-training-checkpoints
pinned: false
---

# Alpha 60M research artifact

This Space is the public interface for the exact terminal Alpha 60M SFT checkpoint. The static page
calls an Alpha CPU inference service at `https://donto.org/alpha-60m`; the service loads the immutable
native checkpoint archived on Hugging Face and does not call or substitute another model.

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

The backend exposes an OpenAI-compatible endpoint. It supports an optional leading `system` message,
alternating `user`/`assistant` history ending in `user`, greedy or sampled decoding, and SSE streaming.
The context limit is 1,024 tokens and the server caps each completion at 256 tokens.

```bash
curl -X POST https://donto.org/alpha-60m/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64,"temperature":0}'
```

```bash
curl https://donto.org/alpha-60m/health
curl https://donto.org/alpha-60m/evidence
curl https://donto.org/alpha-60m/v1/models
```

An empty `choices[0].message.content` with `finish_reason: "stop"` is a genuine measured model result,
not a failed HTTP request. The response also includes `alpha.empty_eos` for non-streaming clients.

## Runtime provenance

- Native terminal checkpoint SHA-256: `6c279d086d8c0679495e38ebec8a473ac23d16bfb3b93516e144712963fecbc8`
- Checkpoint archive revision: `7198d1a1f094ffe88d06399ea99fecbd78fa8b66`
- Standard model revision: `b481f46924b7a4777a029de1ffb44c06cc925d4c`
- Training source commit: `c333bf247fbe87b85d01f3d34789b46615dd1034`
- Architecture: Llama-form 16L/512D/8H, float32, 12,288-token Alpha byte BPE
- Framework: Alpha tensor, tokenizer, checkpoint, and inference packages; no fallback model

The same source can be deployed as a self-contained Docker Space with `Dockerfile.space` if the Hub
account has Docker Space entitlement. The public static Space exists because Hugging Face currently
requires a paid PRO account for a Docker Space on `cpu-basic`.
