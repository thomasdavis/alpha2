---
title: Alpha v0 Historic
emoji: "\U0001F9E0"
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
models:
  - ajaxdavis/alpha-v0-historic
---

# Alpha v0 Historic

A scratch-built GPT language model trained from zero on historic chat data. Every component — tensors, autograd, model, tokenizers, training loop — is custom TypeScript with no ML framework dependencies.

## API

OpenAI-compatible chat completions:

```bash
curl -X POST https://ajaxdavis-alpha-v0-historic.hf.space/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100}'
```

Streaming:

```bash
curl -X POST https://ajaxdavis-alpha-v0-historic.hf.space/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100, "stream": true}'
```

List models:

```bash
curl https://ajaxdavis-alpha-v0-historic.hf.space/v1/models
```

## Model Details

- **Architecture**: GPT (6 layers, 256 dim, 8 heads)
- **Parameters**: ~1.7M
- **Vocab**: BPE 4k
- **Training**: 5000 steps on A100 80GB via custom Vulkan GPU backend
- **Framework**: Alpha (scratch-built TypeScript)
