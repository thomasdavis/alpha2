"use client";

import { useState, useEffect, useRef, useCallback } from "react";

interface ModelConfig {
  nLayer: number;
  nEmbd: number;
  nHead: number;
  vocabSize: number;
  blockSize: number;
}

interface Model {
  id: string;
  name: string;
  step: number;
  domain: string;
  lastLoss: number | null;
  modelConfig: ModelConfig;
}

const domainPrompts: Record<string, string[]> = {
  novels: ["The ", "Once upon a time"],
  chords: ["Em7 G Dsus4", "Am F C G"],
  abc: ["X:1\nM:4/4\nK:G\n|:G"],
};

function defaultPrompt(domain: string): string {
  const prompts = domainPrompts[domain] || domainPrompts.novels;
  return prompts[0];
}

export default function InferencePage() {
  const [models, setModels] = useState<Model[]>([]);
  const [modelId, setModelId] = useState("");
  const [prompt, setPrompt] = useState("The ");
  const [steps, setSteps] = useState(200);
  const [temperature, setTemperature] = useState(0.8);
  const [topk, setTopk] = useState(40);
  const [topP, setTopP] = useState(1.0);
  const [generating, setGenerating] = useState(false);
  const [tokens, setTokens] = useState<{ text: string; isPrompt: boolean }[]>(
    []
  );
  const [status, setStatus] = useState("");

  const sourceRef = useRef<EventSource | null>(null);
  const tokenCountRef = useRef(0);
  const startTimeRef = useRef(0);
  const outputRef = useRef<HTMLDivElement>(null);

  // Fetch models on mount
  useEffect(() => {
    fetch("/api/models")
      .then((r) => r.json())
      .then((data: Model[]) => {
        setModels(data);
        if (data.length > 0) {
          setModelId(data[0].id);
          const domain = data[0].domain || "novels";
          setPrompt(defaultPrompt(domain));
        }
      })
      .catch(() => setStatus("Failed to load models."));
  }, []);

  const selectedModel = models.find((m) => m.id === modelId);
  const domain = selectedModel?.domain || "novels";

  const handleModelChange = useCallback(
    (id: string) => {
      setModelId(id);
      const m = models.find((x) => x.id === id);
      if (m) {
        const d = m.domain || "novels";
        setPrompt(defaultPrompt(d));
      }
    },
    [models]
  );

  const finish = useCallback(() => {
    if (sourceRef.current) {
      sourceRef.current.close();
      sourceRef.current = null;
    }
    setGenerating(false);
  }, []);

  const doStop = useCallback(() => {
    finish();
  }, [finish]);

  const doGenerate = useCallback(() => {
    // Stop any existing stream
    if (sourceRef.current) {
      sourceRef.current.close();
      sourceRef.current = null;
    }

    setTokens([]);
    setStatus("Generating...");
    setGenerating(true);
    tokenCountRef.current = 0;
    startTimeRef.current = performance.now();

    const params = new URLSearchParams({
      query: prompt,
      model: modelId,
      steps: String(steps),
      temp: String(temperature),
      topk: String(topk),
      top_p: String(topP),
    });

    const es = new EventSource(`/api/inference?${params.toString()}`);
    sourceRef.current = es;
    let isFirst = true;

    es.onmessage = (e) => {
      if (e.data === "[DONE]") {
        const elapsed = (performance.now() - startTimeRef.current) / 1000;
        const count = tokenCountRef.current;
        const tps = count / (elapsed || 1);
        setStatus(
          `${count} tokens in ${elapsed.toFixed(1)}s (${tps.toFixed(1)} tok/s)`
        );
        finish();
        return;
      }

      const data = JSON.parse(e.data);
      const token: string = data.token;

      if (isFirst) {
        isFirst = false;
        setTokens((prev) => [...prev, { text: token, isPrompt: true }]);
      } else {
        tokenCountRef.current++;
        setTokens((prev) => [...prev, { text: token, isPrompt: false }]);
      }
    };

    es.onerror = () => {
      setStatus("Connection lost.");
      finish();
    };
  }, [prompt, modelId, steps, temperature, topk, topP, finish]);

  // Auto-scroll output
  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [tokens]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (sourceRef.current) {
        sourceRef.current.close();
      }
    };
  }, []);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !generating) {
        e.preventDefault();
        doGenerate();
      }
    },
    [generating, doGenerate]
  );

  // Model info string
  const modelInfo = selectedModel
    ? (() => {
        const mc = selectedModel.modelConfig;
        const parts = [
          domain,
          `${mc.nLayer}L ${mc.nEmbd}D ${mc.nHead}H`,
          `vocab=${mc.vocabSize}`,
          `ctx=${mc.blockSize}`,
        ];
        if (selectedModel.lastLoss != null) {
          parts.push(`loss=${selectedModel.lastLoss.toFixed(3)}`);
        }
        return parts.join(" | ");
      })()
    : "";

  return (
    <>
      <h1 className="mb-1 text-lg font-bold text-white">Inference</h1>
      <p className="mb-6 text-xs text-text-muted">
        Text generation with streaming output
      </p>

      <div className="mx-auto max-w-3xl space-y-4">
        {/* Model selector */}
        <div className="rounded-lg border border-border bg-surface p-4">
          <label className="mb-1.5 block text-xs font-medium text-text-secondary">
            Model
          </label>
          <select
            value={modelId}
            onChange={(e) => handleModelChange(e.target.value)}
            className="w-full rounded border border-border-2 bg-surface-2 px-3 py-2 text-sm text-text-primary outline-none focus:border-accent"
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} [{m.domain || "novels"}] (step {m.step})
              </option>
            ))}
          </select>
          {modelInfo && (
            <div className="mt-2 text-xs text-text-muted">{modelInfo}</div>
          )}
          {models.length === 0 && !status && (
            <p className="mt-2 text-xs text-text-muted">
              No models loaded yet. Models become available automatically after training runs upload checkpoints.
            </p>
          )}
        </div>

        {/* Prompt + controls */}
        <div className="rounded-lg border border-border bg-surface p-4">
          <label className="mb-1.5 block text-xs font-medium text-text-secondary">
            Prompt
          </label>
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              domain === "chords"
                ? "Enter chord progression..."
                : "Enter prompt..."
            }
            className="mb-4 w-full rounded border border-border-2 bg-surface-2 px-3 py-2 text-sm text-text-primary outline-none focus:border-accent"
          />

          <div className="flex flex-wrap items-end gap-4">
            <div>
              <label className="mb-1 block text-xs text-text-secondary">
                Steps
              </label>
              <input
                type="number"
                value={steps}
                onChange={(e) => setSteps(Number(e.target.value))}
                min={1}
                max={500}
                className="w-20 rounded border border-border-2 bg-surface-2 px-2 py-1.5 text-sm text-text-primary outline-none focus:border-accent"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-text-secondary">
                Temperature
              </label>
              <input
                type="number"
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                min={0.1}
                max={2.0}
                step={0.1}
                className="w-20 rounded border border-border-2 bg-surface-2 px-2 py-1.5 text-sm text-text-primary outline-none focus:border-accent"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-text-secondary">
                Top-k
              </label>
              <input
                type="number"
                value={topk}
                onChange={(e) => setTopk(Number(e.target.value))}
                min={0}
                max={200}
                className="w-20 rounded border border-border-2 bg-surface-2 px-2 py-1.5 text-sm text-text-primary outline-none focus:border-accent"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-text-secondary">
                Top-p
              </label>
              <input
                type="number"
                value={topP}
                onChange={(e) => setTopP(Number(e.target.value))}
                min={0}
                max={1}
                step={0.05}
                className="w-20 rounded border border-border-2 bg-surface-2 px-2 py-1.5 text-sm text-text-primary outline-none focus:border-accent"
              />
            </div>

            <div className="flex gap-2">
              <button
                onClick={doGenerate}
                disabled={generating || !modelId}
                className="rounded bg-accent px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-accent/80 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Generate
              </button>
              <button
                onClick={doStop}
                disabled={!generating}
                className="rounded border border-border-2 px-4 py-1.5 text-sm text-text-secondary transition-colors hover:bg-surface-2 hover:text-text-primary disabled:cursor-not-allowed disabled:opacity-40"
              >
                Stop
              </button>
            </div>
          </div>
        </div>

        {/* Output area */}
        <div
          ref={outputRef}
          className={`min-h-[200px] max-h-[60vh] overflow-y-auto whitespace-pre-wrap break-words rounded-lg border border-border bg-surface p-4 font-mono text-sm leading-relaxed ${
            tokens.length === 0 ? "flex items-center justify-center" : ""
          }`}
        >
          {tokens.length === 0 && !generating ? (
            <span className="text-text-muted">
              Output will appear here...
            </span>
          ) : (
            tokens.map((t, i) => (
              <span
                key={i}
                className={t.isPrompt ? "text-text-muted" : "text-green"}
                style={
                  !t.isPrompt && domain === "chords"
                    ? { letterSpacing: "0.05em", wordSpacing: "0.3em" }
                    : undefined
                }
              >
                {t.text}
              </span>
            ))
          )}
        </div>

        {/* Status line */}
        {status && (
          <div className="text-xs text-text-muted">{status}</div>
        )}
      </div>
    </>
  );
}
