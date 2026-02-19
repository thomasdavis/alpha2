# OpenAI-Compatible API: `/v1/chat/completions`

How Alpha exposes its custom-trained GPT models through an OpenAI-compatible API, enabling participation in LLM arenas, benchmarks, and any tool that speaks the OpenAI protocol.

## Overview

Alpha is a from-scratch GPT training system written entirely in TypeScript — no PyTorch, no ONNX, no external ML frameworks. The inference server (`apps/server`) exposes OpenAI-compatible endpoints so these custom models can be used by any client that speaks the OpenAI Chat Completions API.

**Base URL:** `https://alpha.omegaai.dev`

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/models` | List available models |
| POST | `/v1/chat/completions` | Chat completions (streaming + non-streaming) |
| POST | `/chat/completions` | Alias (same handler) |

The web dashboard at `alpha.omegaai.dev` proxies `/v1/*` requests through a Next.js catch-all route handler (`apps/web/src/app/v1/[...path]/route.ts`) to the backend inference server via the `INTERNAL_SERVER_URL` environment variable.

---

## Architecture

```
                                  ┌─────────────────────────────────┐
  Client (arena, curl, SDK)       │  apps/server/src/server.ts      │
  ──────────────────────────►     │                                 │
  POST /v1/chat/completions       │  route() ─► handleChatCompletions()
                                  │    │                            │
                                  │    ▼                            │
                                  │  ensureModel(modelId)           │
                                  │    │  loads checkpoint from     │
                                  │    │  outputs/{modelId}/        │
                                  │    ▼                            │
                                  │  ┌──────────────────────┐      │
                                  │  │  LoadedModel          │      │
                                  │  │  ├─ GPTParams (weights)│     │
                                  │  │  ├─ Tokenizer (BPE/   │     │
                                  │  │  │    Char/Word)       │     │
                                  │  │  ├─ CpuRefBackend     │     │
                                  │  │  └─ ModelConfig        │     │
                                  │  └──────────────────────┘      │
                                  │    │                            │
                                  │    ▼                            │
                                  │  sampleNextToken() loop         │
                                  │    ├─ gptForward() per token    │
                                  │    ├─ temperature scaling       │
                                  │    ├─ top-k filtering           │
                                  │    └─ softmax + multinomial     │
                                  │                                 │
                                  │  ──► SSE stream or JSON response│
                                  └─────────────────────────────────┘
```

### Component chain

1. **`apps/server/src/server.ts`** — HTTP server (raw `node:http`, no Express/Hono for the main server)
2. **`apps/server/src/lib/engine.ts`** — Model registry, checkpoint loading, token sampling
3. **`@alpha/model`** (`packages/model/src/gpt.ts`) — GPT-2 architecture: token/position embeddings → N transformer blocks (LN → causal self-attention → residual → LN → MLP/GELU → residual) → final LN → LM head
4. **`@alpha/autograd`** — Automatic differentiation (used for the forward pass during inference)
5. **`@alpha/tensor`** — CPU tensor backend (`CpuRefBackend`) with typed array operations
6. **`@alpha/tokenizers`** — BPE, character, and word tokenizers with vocab built from training data
7. **`@alpha/core`** — Shared types (`ModelConfig`, `TrainConfig`), RNG, shape utilities

---

## Endpoint: `GET /v1/models`

Lists all models available for inference. Models are discovered by scanning the `outputs/` directory for subdirectories containing `config.json` + `checkpoint-{step}.json` files.

### Response

```json
{
  "object": "list",
  "data": [
    {
      "id": "novels-5hr",
      "object": "model",
      "created": 1739900000,
      "owned_by": "alpha"
    },
    {
      "id": "chords-run",
      "object": "model",
      "created": 1739850000,
      "owned_by": "alpha"
    }
  ]
}
```

### Implementation

**File:** `apps/server/src/server.ts` — `handleOpenAIModels()`

Calls `getRuns()` from the engine, which returns the in-memory `RunInfo[]` array populated at startup by `scanLocalRuns()`. Each run directory becomes a model with the directory name as the model ID.

---

## Endpoint: `POST /v1/chat/completions`

The core endpoint. Accepts the OpenAI Chat Completions request format and returns either a streaming SSE response or a single JSON response.

### Request

```json
{
  "model": "novels-5hr",
  "messages": [
    { "role": "user", "content": "The old lighthouse stood" }
  ],
  "max_tokens": 200,
  "temperature": 0.7,
  "stream": true
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | first available | Model ID (directory name in `outputs/`) |
| `messages` | array | required | Chat messages array |
| `max_tokens` | int | 2048 | Max completion tokens (capped at 2048) |
| `max_completion_tokens` | int | 2048 | Alias for `max_tokens` |
| `temperature` | float | 0.7 | Sampling temperature |
| `stream` | bool | false | Enable SSE streaming |

### Message handling

**Important:** These are base language models (not instruction-tuned), so messages are concatenated into a single text prompt:

```typescript
function messagesToPrompt(messages: Array<{ role: string; content: string }>): string {
  return messages.map((m) => m.content).join("\n");
}
```

All message roles (system, user, assistant) are treated identically — their `content` fields are joined with newlines. The model then continues this text autoregressively. There is no system prompt handling, no chat template, no special tokens for turn boundaries.

### Non-streaming response

```json
{
  "id": "chatcmpl-a1b2c3d4e5f6a1b2c3d4e5f6",
  "object": "chat.completion",
  "created": 1739900000,
  "model": "novels-5hr",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "on the edge of the cliff, its light long extinguished..."
      },
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 200,
    "total_tokens": 205
  }
}
```

### Streaming response

SSE format with `data:` prefixed JSON chunks:

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1739900000,"model":"novels-5hr","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1739900000,"model":"novels-5hr","choices":[{"index":0,"delta":{"content":" on"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1739900000,"model":"novels-5hr","choices":[{"index":0,"delta":{"content":" the"},"finish_reason":null}]}

...

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1739900000,"model":"novels-5hr","choices":[{"index":0,"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":5,"completion_tokens":200,"total_tokens":205}}

data: [DONE]
```

**Streaming implementation detail:** Each token is generated and flushed via `setImmediate(nextChunk)`, yielding the event loop between tokens. This allows:
- The client to receive tokens as they're generated
- The server to handle connection aborts mid-generation (`req.on("close")`)
- Other requests to be served concurrently

### Finish reasons

| Reason | When |
|--------|------|
| `"length"` | `completion_tokens >= max_tokens` |
| `"stop"` | Context window full (`currentLen >= blockSize`) |

There are no stop sequences or EOS tokens — these are base models that generate until hitting the token limit or context window.

---

## Inference pipeline (step by step)

### 1. Model loading — `ensureModel()`

**File:** `apps/server/src/lib/engine.ts`

Models are loaded lazily on first request and cached in module scope. Only one model is loaded at a time (swapped on model change).

```
ensureModel("novels-5hr")
  → Find RunInfo by ID
  → FileCheckpoint.load("outputs/novels-5hr/checkpoint-900.json")
  → Returns CheckpointState: { modelConfig, params, tokenizerArtifacts, rngState, ... }
  → buildModel(state):
      → CpuRefBackend()          — tensor math engine
      → initGPT(config, backend) — allocate parameter tensors
      → restoreParams(params, state.params)  — fill weights from checkpoint
      → BpeTokenizer/CharTokenizer/WordTokenizer from artifacts
  → Cache as `loaded: LoadedModel`
```

**Checkpoint format:** JSON file containing all model weights as flat arrays, tokenizer vocabulary/merges, optimizer state, and RNG state. A typical 1.7M param model checkpoint is ~110MB JSON.

### 2. Tokenization

The prompt text is encoded to integer token IDs using the tokenizer that was trained alongside the model:

- **BPE** (novels domain) — byte-pair encoding with learned merge rules, vocab ~2000
- **Char** (abc domain) — character-level, vocab = unique characters in training data
- **Word** (chords domain) — whitespace-split word tokenizer

The tokenizer artifacts (vocabulary, merges) are stored inside the checkpoint, so each model carries its own tokenizer.

### 3. Token generation loop — `sampleNextToken()`

For each output token:

```
1. Slice context window: tokens[max(0, currentLen - blockSize) .. currentLen]
2. Build input tensor: shape [1, contextLength], dtype i32

3. Forward pass — gptForward(config, params, backend, tape, input):
   a. Token embedding lookup:     [1, ctx] → [1, ctx, nEmbd]
   b. Position embedding add:     + wpe[0..ctx]
   c. For each transformer layer:
      - LayerNorm
      - Causal self-attention (Q/K/V projections, masked dot-product, output projection)
      - Residual connection
      - LayerNorm
      - MLP: Linear → GELU → Linear
      - Residual connection
   d. Final LayerNorm
   e. LM head projection:        [1, ctx, nEmbd] → [1, ctx, vocabSize]
   → Returns logits tensor

4. Extract last position logits: logits[contextLength - 1, :] → Float32Array[vocabSize]

5. Temperature scaling: logits[v] /= temperature

6. Top-k filtering (k=40):
   - Sort logits descending
   - Set everything below the k-th value to -Infinity

7. Softmax:
   - Subtract max for numerical stability
   - exp() and normalize to probability distribution

8. Multinomial sampling:
   - Draw uniform random r from SeededRng
   - Walk cumulative sum until r < cumsum → selected token

9. Append token to sequence, decode single token to text, emit
```

### 4. Word tokenizer spacing

For the word tokenizer (chords domain), a space separator is prepended to each decoded token unless the token is a newline:

```typescript
const sep = tokenizer.name === "word" && raw !== "\n" ? " " : "";
```

BPE and char tokenizers encode/decode space characters directly, so no separator is needed.

---

## Model discovery

At server startup, `initEngine(OUTPUTS_DIR)` scans the filesystem:

```
outputs/
├── novels-5hr/
│   ├── config.json              ← model + train config, domain
│   ├── checkpoint-900.json      ← latest checkpoint (weights + tokenizer)
│   └── metrics.jsonl            ← training loss history
├── chords-run/
│   ├── config.json
│   ├── checkpoint-2000.json
│   └── metrics.jsonl
└── ...
```

**Scanning logic** (`scanLocalRuns`):
1. List subdirectories of `outputs/`
2. For each, check for `config.json` (skip if missing)
3. Find all `checkpoint-{step}.json` files, pick highest step
4. Skip checkpoints < 1KB (corrupt/empty)
5. Read last line of `metrics.jsonl` for `lastLoss`
6. Register as a `RunInfo` with `id = directory name`

Models can also be uploaded at runtime via `POST /api/upload` (authenticated), which writes the checkpoint to disk and rescans. This is how training machines push new models to the server.

---

## Routing

**File:** `apps/server/src/server.ts` — `route()`

```typescript
if (p === "/v1/models") { handleOpenAIModels(req, res); return; }
if (p === "/v1/chat/completions" || p === "/chat/completions") {
  await handleChatCompletions(req, res); return;
}
```

Both `/v1/chat/completions` and `/chat/completions` are accepted to maximize compatibility with different client libraries.

**CORS:** All responses include `Access-Control-Allow-Origin: *` for cross-origin requests.

**Web proxy:** The Next.js frontend at `alpha.omegaai.dev` proxies `/v1/*` to the backend server via a catch-all route handler (`apps/web/src/app/v1/[...path]/route.ts`), so the endpoints work on both the backend port and the public domain.

---

## Usage examples

### curl (non-streaming)

```bash
curl -X POST https://alpha.omegaai.dev/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "novels-5hr",
    "messages": [{"role": "user", "content": "The ship sailed into"}],
    "max_tokens": 100,
    "temperature": 0.8
  }'
```

### curl (streaming)

```bash
curl -X POST https://alpha.omegaai.dev/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "novels-5hr",
    "messages": [{"role": "user", "content": "Chapter One\n\n"}],
    "max_tokens": 200,
    "temperature": 0.7,
    "stream": true
  }'
```

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://alpha.omegaai.dev/v1",
    api_key="not-needed",  # no auth required for inference
)

# List models
models = client.models.list()
for m in models.data:
    print(m.id)

# Chat completion
response = client.chat.completions.create(
    model="novels-5hr",
    messages=[{"role": "user", "content": "Once upon a time"}],
    max_tokens=200,
    temperature=0.8,
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="novels-5hr",
    messages=[{"role": "user", "content": "The castle"}],
    max_tokens=200,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### OpenAI Node.js SDK

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://alpha.omegaai.dev/v1",
  apiKey: "not-needed",
});

const completion = await client.chat.completions.create({
  model: "novels-5hr",
  messages: [{ role: "user", content: "The old lighthouse" }],
  max_tokens: 200,
});
console.log(completion.choices[0].message.content);
```

### Arena integration

To register Alpha in an LLM arena, point it at:

```
API Base URL:  https://alpha.omegaai.dev/v1
API Key:       (any non-empty string, or "not-needed")
Model ID:      (any model from /v1/models, e.g. "novels-5hr")
```

The endpoint follows the OpenAI Chat Completions spec closely enough for most arena frameworks. Key differences from OpenAI:

- **No authentication required** — the `Authorization` header is ignored for inference endpoints
- **Base models only** — no instruction following, no system prompt behavior. Best used as text continuation
- **No stop sequences** — `stop` parameter is ignored
- **No function calling / tools** — not implemented
- **Context window varies by model** — typically 128–256 tokens (set by `blockSize` in training config)
- **Single choice only** — `n` parameter is ignored, always returns one choice
- **Top-k instead of top-p** — `top_p` is ignored; top-k is hardcoded to 40

---

## Internal endpoints (non-OpenAI)

Alpha also has its own inference endpoints used by the web dashboard:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/models` | List models (Alpha format with modelConfig, trainConfig) |
| GET | `/api/inference?model=X&query=Y&steps=N` | SSE token stream (legacy) |
| POST | `/api/chat` | Chat via AI SDK (`streamText` from `ai` package) |
| POST/GET | `/api/generate` | Single-shot generation (Alpha format) |

The AI SDK integration (`AlphaLanguageModel` class in `engine.ts`) implements the `LanguageModelV3` interface, allowing Alpha models to be used with Vercel's `ai` package and `streamText()`.

---

## Deployment

Both the inference server and web dashboard run on **Railway** (project: `REDACTED_PROJECT`).

```bash
# Deploy server (includes /v1/chat/completions endpoint)
railway service alpha2 && railway up

# Deploy web dashboard (proxies /v1/* to server)
railway service alpha-web && railway up
```

Models are stored in the server's filesystem on Railway. New models are uploaded from training machines via `POST /api/upload` (requires `UPLOAD_SECRET` bearer token).

---

## File reference

| File | Role |
|------|------|
| `apps/server/src/server.ts` | HTTP routing, `handleChatCompletions()`, `handleOpenAIModels()` |
| `apps/server/src/lib/engine.ts` | `ensureModel()`, `sampleNextToken()`, `scanLocalRuns()`, `AlphaLanguageModel` |
| `apps/web/src/app/v1/[...path]/route.ts` | Next.js proxy for `/v1/*` to backend |
| `packages/model/src/gpt.ts` | GPT-2 model architecture (`gptForward`, `initGPT`) |
| `packages/autograd/src/` | Automatic differentiation ops used in forward pass |
| `packages/tensor/src/cpu-ref.ts` | CPU tensor backend (matmul, softmax, etc.) |
| `packages/tokenizers/src/` | BPE, char, word tokenizer implementations |
| `packages/train/src/checkpoint.ts` | `FileCheckpoint` — loads/saves checkpoint JSON |
| `packages/core/src/types.ts` | `ModelConfig`, `TrainConfig` type definitions |
| `packages/core/src/rng.ts` | `SeededRng` — deterministic PRNG for sampling |
