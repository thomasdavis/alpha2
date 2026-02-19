# How to Make Your Model OpenAI-Compatible

A practical guide to exposing any language model through an OpenAI-compatible API. Implement these two endpoints and your model will work with the OpenAI SDKs, LLM arenas, LangChain, LiteLLM, Continue.dev, and anything else that speaks the OpenAI protocol.

---

## What you need to implement

Two endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/v1/models` | List your available models |
| POST | `/v1/chat/completions` | Generate text (streaming + non-streaming) |

That's it. These two endpoints cover 95% of what clients expect.

---

## Endpoint 1: `GET /v1/models`

Returns a list of models your server can run.

### Response format

```json
{
  "object": "list",
  "data": [
    {
      "id": "my-model-7b",
      "object": "model",
      "created": 1700000000,
      "owned_by": "your-org"
    },
    {
      "id": "my-model-3b-chat",
      "object": "model",
      "created": 1700000000,
      "owned_by": "your-org"
    }
  ]
}
```

### Required fields per model

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique model identifier. This is what clients pass in the `model` field |
| `object` | string | Always `"model"` |
| `created` | int | Unix timestamp (seconds) |
| `owned_by` | string | Your org/project name |

### Example implementation (Node.js)

```javascript
app.get("/v1/models", (req, res) => {
  res.json({
    object: "list",
    data: getAvailableModels().map(m => ({
      id: m.id,
      object: "model",
      created: Math.floor(m.createdAt / 1000),
      owned_by: "my-org",
    })),
  });
});
```

### Example implementation (Python / FastAPI)

```python
@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": m.id, "object": "model", "created": int(m.created_at), "owned_by": "my-org"}
            for m in get_available_models()
        ]
    }
```

---

## Endpoint 2: `POST /v1/chat/completions`

This is the main endpoint. It accepts a conversation and returns a model response.

### Request format

```json
{
  "model": "my-model-7b",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "What is the capital of France?" }
  ],
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": false,
  "stop": ["\n\n"]
}
```

### Request fields you should handle

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `model` | string | — | Required. Must match an ID from `/v1/models` |
| `messages` | array | — | Required. Array of `{role, content}` objects |
| `max_tokens` | int | 2048 | Also accept `max_completion_tokens` (newer spec) |
| `temperature` | float | 1.0 | 0.0 = deterministic, higher = more random |
| `top_p` | float | 1.0 | Nucleus sampling threshold |
| `stream` | bool | false | If true, return SSE stream |
| `stop` | string or array | null | Stop sequence(s) to end generation |

Fields you can safely ignore initially: `n`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `logprobs`, `top_logprobs`, `tools`, `tool_choice`, `response_format`, `seed`, `user`.

### Message roles

| Role | Purpose |
|------|---------|
| `system` | System prompt / instructions |
| `user` | User message |
| `assistant` | Previous model response (for multi-turn context) |

How you convert messages to your model's input format depends on your model. Common approaches:

```
# Chat/instruct models — use a chat template
<|system|>You are helpful.<|end|>
<|user|>Hello<|end|>
<|assistant|>

# Base models — just concatenate content
You are helpful.
Hello
```

---

## Non-streaming response

When `stream` is `false` (or absent), return a single JSON response.

### Response format

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "my-model-7b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 8,
    "total_tokens": 33
  }
}
```

### Required response fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique completion ID. Use any format, e.g. `"chatcmpl-" + randomHex()` |
| `object` | string | Always `"chat.completion"` |
| `created` | int | Unix timestamp (seconds) |
| `model` | string | Echo back the model ID |
| `choices` | array | Array with one element (unless you support `n`) |
| `choices[0].index` | int | Always `0` |
| `choices[0].message.role` | string | Always `"assistant"` |
| `choices[0].message.content` | string | The generated text |
| `choices[0].finish_reason` | string | `"stop"`, `"length"`, or `"tool_calls"` |
| `usage.prompt_tokens` | int | Number of input tokens |
| `usage.completion_tokens` | int | Number of output tokens |
| `usage.total_tokens` | int | Sum of prompt + completion tokens |

### Finish reasons

| Reason | When to use |
|--------|-------------|
| `"stop"` | Model hit a stop sequence, EOS token, or naturally finished |
| `"length"` | Hit `max_tokens` limit |

### Example implementation (Node.js)

```javascript
app.post("/v1/chat/completions", async (req, res) => {
  const { model, messages, max_tokens = 2048, temperature = 1.0, stream = false } = req.body;

  // Validate model
  if (!modelExists(model)) {
    return res.status(400).json({
      error: { message: `Unknown model: ${model}`, type: "invalid_request_error" }
    });
  }

  // Convert messages to your model's input format
  const prompt = formatPrompt(messages);

  if (stream) {
    return handleStream(req, res, model, prompt, max_tokens, temperature);
  }

  // Generate
  const { text, promptTokens, completionTokens, finishReason } = await generate(
    model, prompt, max_tokens, temperature
  );

  res.json({
    id: "chatcmpl-" + crypto.randomUUID(),
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{
      index: 0,
      message: { role: "assistant", content: text },
      finish_reason: finishReason,
    }],
    usage: {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: promptTokens + completionTokens,
    },
  });
});
```

### Example implementation (Python / FastAPI)

```python
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model_id = body["model"]
    messages = body["messages"]
    max_tokens = min(body.get("max_tokens", body.get("max_completion_tokens", 2048)), 2048)
    temperature = body.get("temperature", 1.0)
    stream = body.get("stream", False)

    if not model_exists(model_id):
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Unknown model: {model_id}", "type": "invalid_request_error"}}
        )

    prompt = format_prompt(messages)

    if stream:
        return StreamingResponse(
            stream_generate(model_id, prompt, max_tokens, temperature),
            media_type="text/event-stream"
        )

    text, prompt_tokens, completion_tokens, finish_reason = generate(
        model_id, prompt, max_tokens, temperature
    )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
```

---

## Streaming response

When `stream` is `true`, return Server-Sent Events (SSE). Each event is a `data:` line containing JSON, followed by two newlines.

### SSE format

The stream has three phases:

**1. Initial chunk** — sends the role:

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1700000000,"model":"my-model-7b","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

```

**2. Content chunks** — one per token (or group of tokens):

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1700000000,"model":"my-model-7b","choices":[{"index":0,"delta":{"content":"The"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1700000000,"model":"my-model-7b","choices":[{"index":0,"delta":{"content":" capital"},"finish_reason":null}]}

```

**3. Final chunk** — sends finish reason and usage, then `[DONE]`:

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1700000000,"model":"my-model-7b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":25,"completion_tokens":8,"total_tokens":33}}

data: [DONE]

```

### Key differences from non-streaming

| Non-streaming | Streaming |
|---------------|-----------|
| `object: "chat.completion"` | `object: "chat.completion.chunk"` |
| `choices[].message` | `choices[].delta` |
| Full content in one response | Incremental content per chunk |

### Required headers

```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
```

### Example implementation (Node.js)

```javascript
function handleStream(req, res, model, prompt, maxTokens, temperature) {
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
  });

  const completionId = "chatcmpl-" + crypto.randomUUID();
  const created = Math.floor(Date.now() / 1000);

  let aborted = false;
  req.on("close", () => { aborted = true; });

  // Send role chunk
  res.write(`data: ${JSON.stringify({
    id: completionId,
    object: "chat.completion.chunk",
    created,
    model,
    choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }],
  })}\n\n`);

  // Your token generator — adapt this to your inference code
  const generator = createTokenGenerator(model, prompt, maxTokens, temperature);
  let completionTokens = 0;

  function sendNext() {
    if (aborted) return;

    const result = generator.next();

    if (result.done) {
      // Final chunk with finish reason
      res.write(`data: ${JSON.stringify({
        id: completionId,
        object: "chat.completion.chunk",
        created,
        model,
        choices: [{ index: 0, delta: {}, finish_reason: result.value.finishReason }],
        usage: {
          prompt_tokens: result.value.promptTokens,
          completion_tokens: completionTokens,
          total_tokens: result.value.promptTokens + completionTokens,
        },
      })}\n\n`);
      res.write("data: [DONE]\n\n");
      res.end();
      return;
    }

    // Content chunk
    completionTokens++;
    res.write(`data: ${JSON.stringify({
      id: completionId,
      object: "chat.completion.chunk",
      created,
      model,
      choices: [{ index: 0, delta: { content: result.value }, finish_reason: null }],
    })}\n\n`);

    // Yield event loop between tokens so the server stays responsive
    setImmediate(sendNext);
  }

  sendNext();
}
```

### Example implementation (Python / FastAPI)

```python
async def stream_generate(model_id, prompt, max_tokens, temperature):
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    # Role chunk
    yield f"data: {json.dumps({
        'id': completion_id, 'object': 'chat.completion.chunk', 'created': created,
        'model': model_id,
        'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}],
    })}\n\n"

    # Generate tokens
    prompt_tokens = count_tokens(prompt)
    completion_tokens = 0
    finish_reason = "length"

    async for token_text in generate_tokens(model_id, prompt, max_tokens, temperature):
        if token_text is None:  # Natural stop
            finish_reason = "stop"
            break
        completion_tokens += 1
        yield f"data: {json.dumps({
            'id': completion_id, 'object': 'chat.completion.chunk', 'created': created,
            'model': model_id,
            'choices': [{'index': 0, 'delta': {'content': token_text}, 'finish_reason': None}],
        })}\n\n"

    # Final chunk
    yield f"data: {json.dumps({
        'id': completion_id, 'object': 'chat.completion.chunk', 'created': created,
        'model': model_id,
        'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish_reason}],
        'usage': {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
        },
    })}\n\n"
    yield "data: [DONE]\n\n"
```

---

## Error handling

Return errors in OpenAI's error format:

```json
{
  "error": {
    "message": "Unknown model: nonexistent-model",
    "type": "invalid_request_error",
    "code": null
  }
}
```

Common errors:

| Status | Type | When |
|--------|------|------|
| 400 | `invalid_request_error` | Bad request (unknown model, missing messages) |
| 401 | `authentication_error` | Invalid API key (if you use auth) |
| 429 | `rate_limit_error` | Too many requests |
| 500 | `server_error` | Internal error during generation |

---

## CORS

If clients will call your API from browsers, add CORS headers:

```javascript
res.setHeader("Access-Control-Allow-Origin", "*");
res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");

// Handle preflight
if (req.method === "OPTIONS") { res.writeHead(204); res.end(); return; }
```

---

## Authentication (optional)

If you want to require API keys, check the `Authorization` header:

```
Authorization: Bearer sk-your-api-key
```

Most clients send this automatically when configured. For arenas and public demos, you can skip auth entirely — just ignore the header.

---

## Testing your implementation

### 1. Quick test with curl

```bash
# List models
curl http://localhost:3000/v1/models

# Non-streaming completion
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"your-model","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'

# Streaming
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"your-model","messages":[{"role":"user","content":"Hello"}],"max_tokens":50,"stream":true}'
```

### 2. Test with OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:3000/v1", api_key="test")

# Non-streaming
r = client.chat.completions.create(
    model="your-model",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=50,
)
print(r.choices[0].message.content)

# Streaming
for chunk in client.chat.completions.create(
    model="your-model",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=50,
    stream=True,
):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 3. Test with OpenAI Node.js SDK

```typescript
import OpenAI from "openai";
const client = new OpenAI({ baseURL: "http://localhost:3000/v1", apiKey: "test" });

const r = await client.chat.completions.create({
  model: "your-model",
  messages: [{ role: "user", content: "Hello" }],
  max_tokens: 50,
});
console.log(r.choices[0].message.content);
```

If all three work, your API is compatible.

---

## Checklist

- [ ] `GET /v1/models` returns `{ object: "list", data: [...] }`
- [ ] `POST /v1/chat/completions` accepts `model`, `messages`, `max_tokens`, `temperature`
- [ ] Non-streaming returns `{ object: "chat.completion", choices: [...], usage: {...} }`
- [ ] Streaming sets `Content-Type: text/event-stream`
- [ ] Streaming sends initial role chunk with `delta: { role: "assistant", content: "" }`
- [ ] Streaming sends content chunks with `delta: { content: "token" }`
- [ ] Streaming sends final chunk with `finish_reason` and `usage`
- [ ] Streaming ends with `data: [DONE]\n\n`
- [ ] `object` is `"chat.completion"` (non-streaming) or `"chat.completion.chunk"` (streaming)
- [ ] Same `id` is used across all chunks in a single stream
- [ ] Error responses use `{ error: { message, type } }` format
- [ ] CORS headers are set if serving browser clients
- [ ] Handles `max_completion_tokens` as alias for `max_tokens`
- [ ] Returns 400 for unknown model IDs

---

## Common gotchas

1. **Streaming needs `\n\n` after each `data:` line** — not just `\n`. Two newlines separate SSE events.

2. **The `[DONE]` message is `data: [DONE]\n\n`** — it's a literal string, not JSON.

3. **Use `delta` in streaming, `message` in non-streaming** — clients will break if you mix these up.

4. **The `id` must be consistent across all chunks** in a single streaming response. Generate it once and reuse.

5. **`finish_reason` is `null` in content chunks** and only set in the final chunk.

6. **`usage` goes in the final chunk only** (streaming) or in the top-level response (non-streaming).

7. **Handle client disconnects during streaming** — listen for `close`/`abort` events and stop generation to avoid wasted compute.

8. **`created` is seconds, not milliseconds** — `Math.floor(Date.now() / 1000)`, not `Date.now()`.

9. **Token counts should be real** — arenas and monitoring tools use these. If you can't count tokens exactly, estimate from your tokenizer.

10. **Some clients send `max_completion_tokens` instead of `max_tokens`** — the OpenAI spec evolved and newer clients use the new name. Accept both.
