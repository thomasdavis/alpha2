const BASE = "https://alpha.omegaai.dev";

function MethodBadge({ method }: { method: "GET" | "POST" }) {
  const styles =
    method === "GET"
      ? "bg-blue-bg text-blue"
      : "bg-green-bg text-green";
  return (
    <span
      className={`inline-block rounded px-1.5 py-0.5 text-[0.7rem] font-bold uppercase ${styles}`}
    >
      {method}
    </span>
  );
}

function Endpoint({ path }: { path: string }) {
  return <code className="ml-1.5 font-mono text-[0.9rem] text-white">{path}</code>;
}

function Required() {
  return <span className="ml-0.5 text-[0.65rem] text-red">required</span>;
}

function SectionHeading({
  children,
  color,
  className,
}: {
  children: React.ReactNode;
  color?: string;
  className?: string;
}) {
  return (
    <h2
      className={`mb-3 mt-10 border-b border-border pb-2 text-[1.1rem] font-semibold text-white ${className ?? ""}`}
      style={color ? { color } : undefined}
    >
      {children}
    </h2>
  );
}

function EndpointHeading({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="mb-2 mt-8 flex items-center gap-1.5 border-b border-border pb-2 text-[1.1rem] font-semibold text-white">
      {children}
    </h2>
  );
}

function SubHeading({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="mb-2 mt-5 text-[0.95rem] font-medium text-text-primary/80">
      {children}
    </h3>
  );
}

function Pre({ children }: { children: React.ReactNode }) {
  return (
    <pre className="mb-4 mt-2 overflow-x-auto rounded-md border border-border bg-surface px-4 py-3 font-mono text-[0.8rem] leading-relaxed text-text-primary">
      {children}
    </pre>
  );
}

function ParamTable({
  headers,
  rows,
}: {
  headers: string[];
  rows: React.ReactNode[][];
}) {
  return (
    <div className="mb-4 mt-2 overflow-x-auto">
      <table className="w-full text-[0.85rem]">
        <thead>
          <tr>
            {headers.map((h) => (
              <th
                key={h}
                className="border-b border-border-2 px-2.5 py-2 text-left font-semibold text-text-secondary"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((cells, i) => (
            <tr key={i}>
              {cells.map((cell, j) => (
                <td
                  key={j}
                  className="border-b border-surface-2 px-2.5 py-2 text-text-primary/70"
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* Syntax highlighting helpers */
function S({ children }: { children: React.ReactNode }) {
  return <span className="text-green">{children}</span>;
}
function N({ children }: { children: React.ReactNode }) {
  return <span className="text-yellow">{children}</span>;
}
function K({ children }: { children: React.ReactNode }) {
  return <span className="text-purple-400">{children}</span>;
}
function C({ children }: { children: React.ReactNode }) {
  return <span className="text-text-muted">{children}</span>;
}

export default function DocsPage() {
  return (
    <div className="mx-auto max-w-[780px]">
      <h1 className="mb-1 text-lg font-bold text-white">Alpha API</h1>
      <p className="mb-2 text-sm leading-relaxed text-text-primary/70">
        Alpha serves small GPT models trained from scratch. All endpoints are
        unauthenticated except <code className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[0.82rem] text-text-primary">/api/upload</code>.
      </p>
      <p className="mb-6 text-sm leading-relaxed text-text-primary/70">
        Base URL:{" "}
        <code className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[0.82rem] text-text-primary">
          {BASE}
        </code>
      </p>

      {/* ── OpenAI-Compatible API ────────────────────────────────── */}

      <SectionHeading color="#4ade80">OpenAI-Compatible API</SectionHeading>
      <p className="mb-4 text-sm leading-relaxed text-text-primary/70">
        Drop-in compatible with vLLM, Ollama, LiteLLM, FastChat, the OpenAI
        Python/JS SDKs, and any OpenAI-compatible client. Use base URL{" "}
        <code className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[0.82rem] text-text-primary">
          {BASE}/v1
        </code>
        .
      </p>

      {/* ── GET /v1/models ───────────────────────────────────────── */}

      <EndpointHeading>
        <MethodBadge method="GET" />
        <Endpoint path="/v1/models" />
      </EndpointHeading>
      <p className="mb-3 text-sm text-text-primary/70">
        List available models in OpenAI format.
      </p>

      <SubHeading>Response</SubHeading>
      <Pre>
{`{`}
{"\n"}  <S>{`"object"`}</S>: <S>{`"list"`}</S>,
{"\n"}  <S>{`"data"`}</S>: [
{"\n"}    {"{ "}<S>{`"id"`}</S>: <S>{`"novels-5hr"`}</S>, <S>{`"object"`}</S>: <S>{`"model"`}</S>, <S>{`"created"`}</S>: <N>1771439022</N>, <S>{`"owned_by"`}</S>: <S>{`"alpha"`}</S>{" }"},
{"\n"}    ...
{"\n"}  ]
{"\n"}{`}`}
      </Pre>

      {/* ── POST /v1/chat/completions ────────────────────────────── */}

      <EndpointHeading>
        <MethodBadge method="POST" />
        <Endpoint path="/v1/chat/completions" />
      </EndpointHeading>
      <p className="mb-3 text-sm text-text-primary/70">
        OpenAI Chat Completions endpoint. Supports both non-streaming and
        streaming (<code className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[0.82rem] text-text-primary">{`"stream": true`}</code>).
        Also available at{" "}
        <code className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[0.82rem] text-text-primary">/chat/completions</code>.
      </p>

      <SubHeading>Request body (JSON)</SubHeading>
      <Pre>
{`{`}
{"\n"}  <S>{`"model"`}</S>: <S>{`"novels-5hr"`}</S>,
{"\n"}  <S>{`"messages"`}</S>: [{"{ "}<S>{`"role"`}</S>: <S>{`"user"`}</S>, <S>{`"content"`}</S>: <S>{`"Once upon a time"`}</S>{" }"}],
{"\n"}  <S>{`"max_tokens"`}</S>: <N>2048</N>,
{"\n"}  <S>{`"temperature"`}</S>: <N>0.7</N>,
{"\n"}  <S>{`"stream"`}</S>: <K>false</K>
{"\n"}{`}`}
      </Pre>

      <ParamTable
        headers={["Field", "Type", "Default", "Description"]}
        rows={[
          [<code>model</code>, "string", "first model", <span>Model ID from <code>/v1/models</code></span>],
          [<span><code>messages</code><Required /></span>, "array", "\u2014", <span>Array of <code>{`{role, content}`}</code> objects</span>],
          [<code>max_tokens</code>, "int", <code>2048</code>, "Max tokens to generate (capped at 2048)"],
          [<code>temperature</code>, "float", <code>0.7</code>, "Sampling temperature"],
          [<code>stream</code>, "bool", <code>false</code>, "Stream response as SSE chunks"],
        ]}
      />

      <SubHeading>Response (non-streaming)</SubHeading>
      <Pre>
{`{`}
{"\n"}  <S>{`"id"`}</S>: <S>{`"chatcmpl-f018175cb9a6..."`}</S>,
{"\n"}  <S>{`"object"`}</S>: <S>{`"chat.completion"`}</S>,
{"\n"}  <S>{`"created"`}</S>: <N>1771443731</N>,
{"\n"}  <S>{`"model"`}</S>: <S>{`"novels-5hr"`}</S>,
{"\n"}  <S>{`"choices"`}</S>: [{"{"}{"\n"}    <S>{`"index"`}</S>: <N>0</N>,
{"\n"}    <S>{`"message"`}</S>: {"{ "}<S>{`"role"`}</S>: <S>{`"assistant"`}</S>, <S>{`"content"`}</S>: <S>{`"generated text..."`}</S>{" }"},
{"\n"}    <S>{`"finish_reason"`}</S>: <S>{`"length"`}</S>
{"\n"}  {"}"}],
{"\n"}  <S>{`"usage"`}</S>: {"{"}{"\n"}    <S>{`"prompt_tokens"`}</S>: <N>7</N>,
{"\n"}    <S>{`"completion_tokens"`}</S>: <N>50</N>,
{"\n"}    <S>{`"total_tokens"`}</S>: <N>57</N>
{"\n"}  {"}"}
{"\n"}{`}`}
      </Pre>

      <SubHeading>
        Response (streaming, <code className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[0.82rem] text-text-primary">{`"stream": true`}</code>)
      </SubHeading>
      <Pre>
<C>{`// Each SSE chunk:`}</C>
{"\n"}data: {"{"}<S>{`"id"`}</S>:<S>{`"chatcmpl-..."`}</S>,<S>{`"object"`}</S>:<S>{`"chat.completion.chunk"`}</S>,<S>{`"choices"`}</S>:[{"{"}<S>{`"delta"`}</S>:{"{"}<S>{`"content"`}</S>:<S>{`"hello"`}</S>{"}"},{`"finish_reason"`}:<K>null</K>{"}"}]{"}"}
{"\n"}
{"\n"}<C>{`// Final chunk includes usage and finish_reason, followed by:`}</C>
{"\n"}data: [DONE]
      </Pre>

      <SubHeading>Example &mdash; curl</SubHeading>
      <Pre>
{`curl -X POST `}<S>{`"${BASE}/v1/chat/completions"`}</S>{` \\`}
{"\n"}{"  "}-H <S>{`"Content-Type: application/json"`}</S>{` \\`}
{"\n"}{"  "}-H <S>{`"Authorization: Bearer any-key"`}</S>{` \\`}
{"\n"}{"  "}-d <S>{`'{\n    "model": "novels-5hr",\n    "messages": [{"role": "user", "content": "Once upon a time"}],\n    "max_tokens": 100,\n    "temperature": 0.7\n  }'`}</S>
      </Pre>

      <SubHeading>Example &mdash; Python (OpenAI SDK)</SubHeading>
      <Pre>
<K>from</K>{` openai `}<K>import</K>{` OpenAI`}
{"\n"}
{"\n"}client = OpenAI(
{"\n"}    base_url=<S>{`"${BASE}/v1"`}</S>,
{"\n"}    api_key=<S>{`"any-key"`}</S>,
{"\n"})
{"\n"}
{"\n"}<C># Non-streaming</C>
{"\n"}response = client.chat.completions.create(
{"\n"}    model=<S>{`"novels-5hr"`}</S>,
{"\n"}    messages=[{"{"}<S>{`"role"`}</S>: <S>{`"user"`}</S>, <S>{`"content"`}</S>: <S>{`"Once upon a time"`}</S>{"}"}],
{"\n"}    max_tokens=<N>100</N>,
{"\n"})
{"\n"}print(response.choices[<N>0</N>].message.content)
{"\n"}
{"\n"}<C># Streaming</C>
{"\n"}stream = client.chat.completions.create(
{"\n"}    model=<S>{`"novels-5hr"`}</S>,
{"\n"}    messages=[{"{"}<S>{`"role"`}</S>: <S>{`"user"`}</S>, <S>{`"content"`}</S>: <S>{`"The knight"`}</S>{"}"}],
{"\n"}    max_tokens=<N>100</N>,
{"\n"}    stream=<K>True</K>,
{"\n"})
{"\n"}<K>for</K> chunk <K>in</K> stream:
{"\n"}    <K>if</K> chunk.choices[<N>0</N>].delta.content:
{"\n"}        print(chunk.choices[<N>0</N>].delta.content, end=<S>{`""`}</S>)
      </Pre>

      <SubHeading>Example &mdash; JavaScript (OpenAI SDK)</SubHeading>
      <Pre>
<K>import</K>{` OpenAI `}<K>from</K> <S>{`"openai"`}</S>;
{"\n"}
{"\n"}<K>const</K>{` client = `}<K>new</K>{` OpenAI({`}
{"\n"}  baseURL: <S>{`"${BASE}/v1"`}</S>,
{"\n"}  apiKey: <S>{`"any-key"`}</S>,
{"\n"}{"}"});
{"\n"}
{"\n"}<K>const</K>{` response = `}<K>await</K>{` client.chat.completions.create({`}
{"\n"}  model: <S>{`"novels-5hr"`}</S>,
{"\n"}  messages: [{"{ "}role: <S>{`"user"`}</S>, content: <S>{`"Once upon a time"`}</S>{" }"}],
{"\n"}  max_tokens: <N>100</N>,
{"\n"}{"}"});
{"\n"}console.log(response.choices[<N>0</N>].message.content);
      </Pre>

      {/* ── Other Endpoints ──────────────────────────────────────── */}

      <SectionHeading className="mt-14">Other Endpoints</SectionHeading>

      {/* ── GET /api/models ──────────────────────────────────────── */}

      <EndpointHeading>
        <MethodBadge method="GET" />
        <Endpoint path="/api/models" />
      </EndpointHeading>
      <p className="mb-3 text-sm text-text-primary/70">
        Returns the list of available models with full training metadata.
      </p>

      <SubHeading>Response</SubHeading>
      <Pre>
[
{"\n"}  {"{"}
{"\n"}    <S>{`"id"`}</S>: <S>{`"novels-5hr"`}</S>,
{"\n"}    <S>{`"name"`}</S>: <S>{`"novels-5hr"`}</S>,
{"\n"}    <S>{`"step"`}</S>: <N>900</N>,
{"\n"}    <S>{`"mtime"`}</S>: <N>1771438445123.4</N>,
{"\n"}    <S>{`"lastLoss"`}</S>: <N>4.123</N>,
{"\n"}    <S>{`"domain"`}</S>: <S>{`"novels"`}</S>,
{"\n"}    <S>{`"modelConfig"`}</S>: {"{ "}<S>{`"vocabSize"`}</S>: <N>2000</N>, <S>{`"blockSize"`}</S>: <N>256</N>, ... {"}"},
{"\n"}    <S>{`"trainConfig"`}</S>: {"{ "}<S>{`"iters"`}</S>: <N>3000</N>, <S>{`"batchSize"`}</S>: <N>4</N>, ... {"}"}
{"\n"}  {"}"}
{"\n"}]
      </Pre>

      {/* ── POST|GET /api/generate ───────────────────────────────── */}

      <EndpointHeading>
        <MethodBadge method="POST" />
        <MethodBadge method="GET" />
        <Endpoint path="/api/generate" />
      </EndpointHeading>
      <p className="mb-3 text-sm text-text-primary/70">
        Non-streaming text generation. All parameters can be passed as query
        string params or in a POST JSON body (query string takes precedence).
      </p>

      <SubHeading>Request body (JSON)</SubHeading>
      <Pre>
{`{`}
{"\n"}  <S>{`"prompt"`}</S>: <S>{`"string"`}</S>,
{"\n"}  <S>{`"max_tokens"`}</S>: <N>2048</N>,
{"\n"}  <S>{`"temperature"`}</S>: <N>0.7</N>,
{"\n"}  <S>{`"model"`}</S>: <S>{`"string (optional, defaults to first model)"`}</S>
{"\n"}{`}`}
      </Pre>

      <ParamTable
        headers={["Field", "Type", "Default", "Description"]}
        rows={[
          [<span><code>prompt</code><Required /></span>, "string", "\u2014", "Input text to complete"],
          [<code>max_tokens</code>, "int", <code>2048</code>, "Max tokens to generate (capped at 2048)"],
          [<code>temperature</code>, "float", <code>0.7</code>, "Sampling temperature"],
          [<code>model</code>, "string", "first model", <span>Model ID from <code>/api/models</code></span>],
        ]}
      />

      <SubHeading>Response</SubHeading>
      <Pre>
{`{`}
{"\n"}  <S>{`"text"`}</S>: <S>{`"generated completion text"`}</S>,
{"\n"}  <S>{`"model"`}</S>: <S>{`"novels-5hr"`}</S>,
{"\n"}  <S>{`"usage"`}</S>: {"{"}
{"\n"}    <S>{`"prompt_tokens"`}</S>: <N>5</N>,
{"\n"}    <S>{`"completion_tokens"`}</S>: <N>100</N>
{"\n"}  {"}"}
{"\n"}{`}`}
      </Pre>

      <SubHeading>Example &mdash; curl</SubHeading>
      <Pre>
{`curl -X POST `}<S>{`"${BASE}/api/generate"`}</S>{` \\`}
{"\n"}{"  "}-H <S>{`"Content-Type: application/json"`}</S>{` \\`}
{"\n"}{"  "}-d <S>{`'{\n    "prompt": "Once upon a time",\n    "max_tokens": 100,\n    "temperature": 0.7\n  }'`}</S>
      </Pre>

      <SubHeading>Example &mdash; JavaScript</SubHeading>
      <Pre>
<K>const</K>{` res = `}<K>await</K>{` fetch(`}<S>{`"/api/generate"`}</S>{`, {`}
{"\n"}  method: <S>{`"POST"`}</S>,
{"\n"}  headers: {"{ "}<S>{`"Content-Type"`}</S>: <S>{`"application/json"`}</S>{" }"},
{"\n"}  body: JSON.stringify({"{"}
{"\n"}    prompt: <S>{`"Once upon a time"`}</S>,
{"\n"}    max_tokens: <N>100</N>,
{"\n"}    temperature: <N>0.7</N>,
{"\n"}  {"}"}),
{"\n"}{"}"});
{"\n"}
{"\n"}<K>const</K>{` data = `}<K>await</K>{` res.json();`}
{"\n"}console.log(data.text);
{"\n"}<C>{`// "there was a kingdom far away..."`}</C>
{"\n"}console.log(data.usage);
{"\n"}<C>{`// { prompt_tokens: 4, completion_tokens: 100 }`}</C>
      </Pre>

      {/* ── GET /api/inference ────────────────────────────────────── */}

      <EndpointHeading>
        <MethodBadge method="GET" />
        <Endpoint path="/api/inference" />
      </EndpointHeading>
      <p className="mb-3 text-sm text-text-primary/70">
        Stream generated tokens via Server-Sent Events. The first event contains
        the echoed prompt; subsequent events are generated tokens. The stream
        ends with a{" "}
        <code className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[0.82rem] text-text-primary">[DONE]</code>{" "}
        sentinel.
      </p>

      <SubHeading>Query parameters</SubHeading>
      <ParamTable
        headers={["Param", "Type", "Default", "Description"]}
        rows={[
          [<code>query</code>, "string", <code>{`""`}</code>, "Input prompt"],
          [<code>model</code>, "string", "first model", <span>Model ID from <code>/api/models</code></span>],
          [<code>steps</code>, "int", <code>200</code>, "Max tokens to generate (capped at 500)"],
          [<code>temp</code>, "float", <code>0.8</code>, "Sampling temperature"],
          [<code>topk</code>, "int", <code>40</code>, "Top-k filtering (0 = disabled)"],
          [<code>top_p</code>, "float", <code>1.0</code>, "Top-p (nucleus) filtering (1.0 = disabled)"],
        ]}
      />

      <SubHeading>Example &mdash; curl</SubHeading>
      <Pre>
{`curl -N `}<S>{`"${BASE}/api/inference?query=The&model=novels-5hr&steps=100&temp=0.8"`}</S>
      </Pre>

      <SubHeading>Example &mdash; JavaScript (EventSource)</SubHeading>
      <Pre>
<K>const</K>{` url = `}<S>{`"/api/inference?query=The&model=novels-5hr&steps=100"`}</S>;
{"\n"}<K>const</K>{` source = `}<K>new</K>{` EventSource(url);`}
{"\n"}
{"\n"}source.onmessage = (e) =&gt; {"{"}
{"\n"}  <K>if</K> (e.data === <S>{`"[DONE]"`}</S>) {"{"} source.close(); <K>return</K>; {"}"}
{"\n"}  <K>const</K> {"{ "}token{"} "} = JSON.parse(e.data);
{"\n"}  process.stdout.write(token);
{"\n"}{"}"};
      </Pre>

      <SubHeading>SSE event format</SubHeading>
      <Pre>
<C>{`// Each event:`}</C>
{"\n"}data: {"{"}<S>{`"token"`}</S>: <S>{`"hello"`}</S>{"}"}
{"\n"}
{"\n"}<C>{`// Final event:`}</C>
{"\n"}data: [DONE]
      </Pre>

      {/* ── POST /api/chat ───────────────────────────────────────── */}

      <EndpointHeading>
        <MethodBadge method="POST" />
        <Endpoint path="/api/chat" />
      </EndpointHeading>
      <p className="mb-3 text-sm text-text-primary/70">
        AI SDK-compatible streaming chat endpoint. Returns a text stream (not
        SSE). Compatible with the AI SDK{" "}
        <code className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[0.82rem] text-text-primary">useChat</code>{" "}
        hook.
      </p>

      <SubHeading>Request body (JSON)</SubHeading>
      <ParamTable
        headers={["Field", "Type", "Default", "Description"]}
        rows={[
          [<span><code>messages</code><Required /></span>, "array", "\u2014", <span>Array of <code>{`{role, content}`}</code> objects</span>],
          [<code>model</code>, "string", "first model", "Model ID"],
          [<code>maxTokens</code>, "int", <code>200</code>, "Max tokens (capped at 500)"],
          [<code>temperature</code>, "float", <code>0.8</code>, "Sampling temperature"],
          [<code>topk</code>, "int", <code>40</code>, "Top-k filtering"],
          [<code>top_p</code>, "float", <code>1.0</code>, "Top-p (nucleus) filtering"],
        ]}
      />

      <SubHeading>Example &mdash; curl (streaming)</SubHeading>
      <Pre>
{`curl -N -X POST `}<S>{`"${BASE}/api/chat"`}</S>{` \\`}
{"\n"}{"  "}-H <S>{`"Content-Type: application/json"`}</S>{` \\`}
{"\n"}{"  "}-d <S>{`'{\n    "model": "novels-5hr",\n    "messages": [{"role": "user", "content": "Once upon a time"}],\n    "maxTokens": 100,\n    "temperature": 0.8\n  }'`}</S>
      </Pre>

      <SubHeading>Example &mdash; JavaScript (fetch, streaming)</SubHeading>
      <Pre>
<K>const</K>{` res = `}<K>await</K>{` fetch(`}<S>{`"/api/chat"`}</S>{`, {`}
{"\n"}  method: <S>{`"POST"`}</S>,
{"\n"}  headers: {"{ "}<S>{`"Content-Type"`}</S>: <S>{`"application/json"`}</S>{" }"},
{"\n"}  body: JSON.stringify({"{"}
{"\n"}    model: <S>{`"novels-5hr"`}</S>,
{"\n"}    messages: [{"{ "}role: <S>{`"user"`}</S>, content: <S>{`"Once upon a time"`}</S>{" }"}],
{"\n"}    maxTokens: <N>100</N>,
{"\n"}  {"}"}),
{"\n"}{"}"});
{"\n"}
{"\n"}<K>const</K>{` reader = res.body.getReader();`}
{"\n"}<K>const</K>{` decoder = `}<K>new</K>{` TextDecoder();`}
{"\n"}<K>while</K> (<K>true</K>) {"{"}
{"\n"}  <K>const</K> {"{ "}done, value{"} "} = <K>await</K> reader.read();
{"\n"}  <K>if</K> (done) <K>break</K>;
{"\n"}  process.stdout.write(decoder.decode(value));
{"\n"}{"}"}
      </Pre>

      <SubHeading>Example &mdash; non-streaming (read full response)</SubHeading>
      <Pre>
<K>const</K>{` res = `}<K>await</K>{` fetch(`}<S>{`"/api/chat"`}</S>{`, {`}
{"\n"}  method: <S>{`"POST"`}</S>,
{"\n"}  headers: {"{ "}<S>{`"Content-Type"`}</S>: <S>{`"application/json"`}</S>{" }"},
{"\n"}  body: JSON.stringify({"{"}
{"\n"}    model: <S>{`"novels-5hr"`}</S>,
{"\n"}    messages: [{"{ "}role: <S>{`"user"`}</S>, content: <S>{`"The knight"`}</S>{" }"}],
{"\n"}    maxTokens: <N>50</N>,
{"\n"}  {"}"}),
{"\n"}{"}"});
{"\n"}
{"\n"}<C>{`// Just await the full text (ignores streaming)`}</C>
{"\n"}<K>const</K>{` text = `}<K>await</K>{` res.text();`}
{"\n"}console.log(text);
      </Pre>

      {/* ── POST /api/upload ─────────────────────────────────────── */}

      <EndpointHeading>
        <MethodBadge method="POST" />
        <Endpoint path="/api/upload" />
      </EndpointHeading>
      <p className="mb-3 text-sm text-text-primary/70">
        Upload a model checkpoint to the server. Requires Bearer token
        authentication via the{" "}
        <code className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[0.82rem] text-text-primary">UPLOAD_SECRET</code>{" "}
        environment variable.
      </p>

      <SubHeading>Headers</SubHeading>
      <ParamTable
        headers={["Header", "Value"]}
        rows={[
          [<code>Authorization</code>, <code>Bearer &lt;UPLOAD_SECRET&gt;</code>],
          [<code>Content-Type</code>, <code>application/json</code>],
          [<code>Content-Encoding</code>, <span><code>gzip</code> (optional, recommended for large checkpoints)</span>],
        ]}
      />

      <SubHeading>Request body (JSON)</SubHeading>
      <ParamTable
        headers={["Field", "Type", "Description"]}
        rows={[
          [<span><code>name</code><Required /></span>, "string", "Run name (becomes the model ID)"],
          [<span><code>config</code><Required /></span>, "object", "Training config (config.json contents)"],
          [<span><code>checkpoint</code><Required /></span>, "object", "Checkpoint data (checkpoint-N.json contents)"],
          [<span><code>step</code><Required /></span>, "int", "Training step number"],
          [<code>metrics</code>, "string", "Contents of metrics.jsonl (newline-delimited JSON)"],
        ]}
      />

      <SubHeading>Response</SubHeading>
      <Pre>
{"{ "}<S>{`"ok"`}</S>: <K>true</K>, <S>{`"name"`}</S>: <S>{`"my-run"`}</S>, <S>{`"step"`}</S>: <N>500</N>{" }"}
      </Pre>

      {/* ── Notes ────────────────────────────────────────────────── */}

      <SectionHeading>Notes</SectionHeading>
      <ul className="mb-10 ml-6 list-disc space-y-2 text-sm leading-relaxed text-text-primary/70">
        <li>
          These are small GPT models (100K&ndash;10M params) trained from scratch
          on specific domains &mdash; they don&apos;t follow instructions or answer
          questions. Treat them as text completers.
        </li>
        <li>
          The <code className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[0.82rem] text-text-primary">domain</code> field
          on each model indicates what kind of text it generates:{" "}
          <code className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[0.82rem] text-text-primary">novels</code>,{" "}
          <code className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[0.82rem] text-text-primary">abc</code> (ABC music
          notation), or{" "}
          <code className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[0.82rem] text-text-primary">chords</code>.
        </li>
        <li>
          The first token event from{" "}
          <code className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[0.82rem] text-text-primary">/api/inference</code> is
          the echoed prompt. All subsequent events are generated tokens.
        </li>
        <li>
          The <code className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[0.82rem] text-text-primary">/api/chat</code>{" "}
          endpoint streams text using the AI SDK data protocol. To consume
          it without streaming, just call{" "}
          <code className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[0.82rem] text-text-primary">await res.text()</code>.
        </li>
        <li>
          Model loading is lazy &mdash; the first request to a model takes a few
          seconds to load the checkpoint into memory. Subsequent requests reuse
          the cached model.
        </li>
      </ul>
    </div>
  );
}
