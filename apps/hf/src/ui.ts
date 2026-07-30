import { MODEL_ID } from "./protocol.js";

export function renderUi(params: number, step: number, apiBase = ""): string {
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Inspect and query the archived Alpha 60M experimental checkpoint.">
  <title>Alpha 60M · research artifact</title>
  <style>
    :root {
      color-scheme: light;
      --ink: oklch(0.20 0.018 248);
      --muted: oklch(0.48 0.018 248);
      --line: oklch(0.89 0.012 248);
      --soft: oklch(0.975 0.008 248);
      --danger: oklch(0.57 0.21 18);
      --danger-soft: oklch(0.96 0.035 18);
      --teal: oklch(0.48 0.095 190);
      --teal-soft: oklch(0.96 0.025 190);
      --white: oklch(1 0 0);
      --shadow: 0 18px 50px oklch(0.20 0.018 248 / 0.07);
    }
    * { box-sizing: border-box; }
    html { background: var(--white); }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--white);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 15px;
      line-height: 1.55;
    }
    button, textarea { font: inherit; }
    a { color: var(--teal); text-underline-offset: 3px; }
    a:hover { text-decoration-thickness: 2px; }
    .shell { min-height: 100vh; }
    header {
      min-height: 68px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
      padding: 14px clamp(20px, 4vw, 64px);
      border-bottom: 1px solid var(--line);
    }
    .brand { display: flex; align-items: center; gap: 11px; font-weight: 720; letter-spacing: -0.02em; }
    .mark {
      width: 34px;
      height: 34px;
      display: grid;
      place-items: center;
      color: var(--white);
      background: var(--ink);
      border-radius: 7px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 19px;
    }
    .status { display: flex; align-items: center; gap: 9px; color: var(--danger); font-size: 13px; font-weight: 690; }
    .status::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--danger); }
    main { max-width: 1240px; margin: 0 auto; padding: clamp(34px, 6vw, 78px) clamp(20px, 4vw, 64px) 72px; }
    .hero { max-width: 850px; margin-bottom: 42px; }
    .eyebrow { margin: 0 0 10px; color: var(--danger); font-size: 12px; font-weight: 760; letter-spacing: 0.12em; text-transform: uppercase; }
    h1 { margin: 0; max-width: 760px; font-size: clamp(39px, 6vw, 72px); line-height: 0.98; letter-spacing: -0.055em; font-weight: 760; }
    .lede { max-width: 710px; margin: 24px 0 0; color: var(--muted); font-size: clamp(17px, 2vw, 20px); }
    .warning { margin-top: 28px; padding: 17px 20px; border-left: 4px solid var(--danger); background: var(--danger-soft); }
    .warning strong { display: block; margin-bottom: 3px; color: var(--danger); }
    .workspace { display: grid; grid-template-columns: minmax(220px, 0.72fr) minmax(0, 1.75fr); border: 1px solid var(--line); box-shadow: var(--shadow); }
    aside { padding: 26px; background: var(--soft); border-right: 1px solid var(--line); }
    h2 { margin: 0; font-size: 18px; letter-spacing: -0.02em; }
    .metric-list { margin: 24px 0 28px; padding: 0; list-style: none; }
    .metric-list li { display: flex; justify-content: space-between; gap: 16px; padding: 10px 0; border-bottom: 1px solid var(--line); color: var(--muted); }
    .metric-list b { color: var(--ink); font-variant-numeric: tabular-nums; }
    .detail { color: var(--muted); font-size: 13px; }
    .detail p { margin: 0 0 11px; }
    .console { min-width: 0; background: var(--white); }
    .console-head { display: flex; align-items: center; justify-content: space-between; gap: 20px; padding: 22px 26px; border-bottom: 1px solid var(--line); }
    .console-head p { margin: 3px 0 0; color: var(--muted); font-size: 13px; }
    .runtime { color: var(--teal); font: 650 12px ui-monospace, SFMono-Regular, Menlo, monospace; }
    .transcript { min-height: 310px; max-height: 500px; overflow-y: auto; padding: 26px; }
    .empty-state { max-width: 510px; padding-top: 54px; color: var(--muted); }
    .empty-state b { display: block; margin-bottom: 7px; color: var(--ink); font-size: 17px; }
    .turn { display: grid; grid-template-columns: 78px minmax(0, 1fr); gap: 18px; padding: 17px 0; border-bottom: 1px solid var(--line); }
    .turn-role { padding-top: 2px; color: var(--muted); font: 680 11px ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: 0.08em; text-transform: uppercase; }
    .turn-body { min-width: 0; white-space: pre-wrap; overflow-wrap: anywhere; }
    .turn.assistant .turn-role { color: var(--teal); }
    .turn.failure .turn-body { color: var(--danger); font-style: italic; }
    .composer { padding: 22px 26px 26px; border-top: 1px solid var(--line); }
    .presets { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 13px; }
    .preset { border: 1px solid var(--line); border-radius: 5px; padding: 7px 10px; color: var(--muted); background: var(--white); cursor: pointer; font-size: 12px; }
    .preset:hover, .preset:focus-visible { color: var(--ink); border-color: var(--teal); outline: none; }
    textarea { width: 100%; min-height: 104px; resize: vertical; padding: 14px 15px; color: var(--ink); background: var(--white); border: 1px solid var(--line); border-radius: 6px; outline: none; }
    textarea:focus { border-color: var(--teal); box-shadow: 0 0 0 3px var(--teal-soft); }
    .actions { display: flex; align-items: center; justify-content: space-between; gap: 18px; margin-top: 13px; }
    .actions small { color: var(--muted); }
    .submit { min-width: 122px; padding: 10px 17px; border: 1px solid var(--danger); border-radius: 6px; color: var(--white); background: var(--danger); cursor: pointer; font-weight: 710; }
    .submit:hover { background: oklch(0.51 0.21 18); }
    .submit:focus-visible { outline: 3px solid var(--danger-soft); outline-offset: 2px; }
    .submit:disabled { cursor: wait; opacity: 0.62; }
    footer { max-width: 1240px; margin: 0 auto; padding: 0 clamp(20px, 4vw, 64px) 44px; display: flex; flex-wrap: wrap; gap: 8px 24px; color: var(--muted); font-size: 13px; }
    @media (max-width: 780px) {
      header { align-items: flex-start; }
      .status { max-width: 150px; text-align: right; }
      .workspace { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .transcript { min-height: 270px; }
      .actions { align-items: flex-start; flex-direction: column; }
      .submit { width: 100%; }
    }
    @media (prefers-reduced-motion: reduce) { *, *::before, *::after { scroll-behavior: auto !important; } }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div class="brand"><span class="mark" aria-hidden="true">α</span><span>Alpha research</span></div>
      <div class="status">Archived · D3 quality gate failed</div>
    </header>
    <main>
      <section class="hero" aria-labelledby="page-title">
        <p class="eyebrow">57.7M parameters · built from scratch</p>
        <h1 id="page-title">A finished experiment, shown without spin.</h1>
        <p class="lede">Alpha 60M completed one billion pretraining tokens and a full SFT epoch on the custom TypeScript/Vulkan stack. The mechanics passed. The resulting checkpoint did not become a useful chat model.</p>
        <div class="warning" role="note"><strong>Measured limitation</strong>In the sealed evaluation, 92 of 100 replies were empty, six looped, and two were unusable fragments. This console returns the checkpoint’s real output—including a blank EOS.</div>
      </section>

      <section class="workspace" aria-label="Alpha checkpoint console">
        <aside>
          <h2>Frozen evidence</h2>
          <ul class="metric-list">
            <li><span>Structural pass</span><b>2 / 100</b></li>
            <li><span>Empty replies</span><b>92 / 100</b></li>
            <li><span>Degenerate loops</span><b>6 / 100</b></li>
            <li><span>Closed-book QA</span><b>0 / 200</b></li>
            <li><span>Export parity</span><b>PASS</b></li>
            <li><span>Terminal step</span><b>${step.toLocaleString("en-US")}</b></li>
          </ul>
          <div class="detail">
            <p><strong>${(params / 1e6).toFixed(2)}M</strong> finite parameters. Native checkpoint, optimizer state, RNG state, metrics, and failure reports are archived.</p>
            <p><a href="https://huggingface.co/ajaxdavis/alpha-60m-chat" target="_blank" rel="noreferrer">Model card</a> · <a href="https://huggingface.co/ajaxdavis/alpha-60m-training-checkpoints" target="_blank" rel="noreferrer">Recovery archive</a></p>
          </div>
        </aside>

        <div class="console">
          <div class="console-head">
            <div><h2>Checkpoint console</h2><p>Greedy decoding · 1,024-token context · no fallback model</p></div>
            <span class="runtime" id="runtime">READY</span>
          </div>
          <div class="transcript" id="transcript" aria-live="polite">
            <div class="empty-state" id="empty-state"><b>Ask the archived model directly.</b>The response may be empty. That is evidence, not a transport error; the interface will mark it explicitly.</div>
          </div>
          <form class="composer" id="composer">
            <div class="presets" aria-label="Example prompts">
              <button class="preset" type="button" data-prompt="Explain why the sky looks blue in two short sentences.">Blue sky</button>
              <button class="preset" type="button" data-prompt="Write a friendly greeting for a new neighbour.">Friendly greeting</button>
              <button class="preset" type="button" data-prompt="What is 17 multiplied by 6? Explain the calculation.">Simple maths</button>
            </div>
            <label for="prompt" class="turn-role">Your prompt</label>
            <textarea id="prompt" name="prompt" maxlength="3000" required placeholder="Type a short prompt…"></textarea>
            <div class="actions">
              <small>Served by Alpha’s own CPU inference engine on the project host. No request is sent to another model.</small>
              <button class="submit" id="submit" type="submit">Run checkpoint</button>
            </div>
          </form>
        </div>
      </section>
    </main>
    <footer><span>${MODEL_ID}</span><span>Apache-2.0 code and weights</span><a href="${apiBase}/v1/models">OpenAI-compatible API</a><a href="${apiBase}/health">Health</a></footer>
  </div>
  <script>
    const form = document.getElementById("composer");
    const prompt = document.getElementById("prompt");
    const submit = document.getElementById("submit");
    const transcript = document.getElementById("transcript");
    const emptyState = document.getElementById("empty-state");
    const runtime = document.getElementById("runtime");

    function addTurn(role, content, failure) {
      if (emptyState) emptyState.remove();
      const turn = document.createElement("div");
      turn.className = "turn " + role + (failure ? " failure" : "");
      const label = document.createElement("div");
      label.className = "turn-role";
      label.textContent = role;
      const body = document.createElement("div");
      body.className = "turn-body";
      body.textContent = content;
      turn.append(label, body);
      transcript.append(turn);
      transcript.scrollTop = transcript.scrollHeight;
    }

    document.querySelectorAll("[data-prompt]").forEach(function (button) {
      button.addEventListener("click", function () {
        prompt.value = button.getAttribute("data-prompt") || "";
        prompt.focus();
      });
    });

    form.addEventListener("submit", async function (event) {
      event.preventDefault();
      const value = prompt.value.trim();
      if (!value) return;
      addTurn("user", value, false);
      prompt.value = "";
      submit.disabled = true;
      runtime.textContent = "RUNNING";
      const started = performance.now();
      try {
        const response = await fetch("${apiBase}/v1/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages: [{ role: "user", content: value }], max_tokens: 96, temperature: 0 })
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error && payload.error.message ? payload.error.message : "request failed");
        const output = payload.choices[0].message.content;
        addTurn("assistant", output.trim() ? output : "[Empty response: the checkpoint emitted EOS immediately]", !output.trim());
        runtime.textContent = Math.round(performance.now() - started) + " MS";
      } catch (error) {
        addTurn("assistant", "[Runtime error: " + (error instanceof Error ? error.message : String(error)) + "]", true);
        runtime.textContent = "ERROR";
      } finally {
        submit.disabled = false;
        prompt.focus();
      }
    });
  </script>
</body>
</html>`;
}
