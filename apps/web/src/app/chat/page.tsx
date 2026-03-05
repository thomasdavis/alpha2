"use client";

import { useState, useEffect, useRef, useCallback, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { Button, Card, CardContent } from "@alpha/ui";

// ── Types ──────────────────────────────────────────────────────

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

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  audioUrl?: string;
}

// ── Constants ──────────────────────────────────────────────────

const STORAGE_KEY = "alpha-chat";
const GAMMA_URL = "https://gamma.omegaai.dev/api/generate";

const MUSIC_STYLES = [
  { value: "", label: "Auto-detect style" },
  { value: "acoustic", label: "Acoustic" },
  { value: "folk", label: "Folk" },
  { value: "ballad", label: "Ballad" },
  { value: "piano_ballad", label: "Piano Ballad" },
  { value: "pop", label: "Pop" },
  { value: "rock", label: "Rock" },
  { value: "clean_rock", label: "Clean Rock" },
  { value: "jazz", label: "Jazz" },
  { value: "rnb", label: "R&B" },
  { value: "electronic", label: "Electronic" },
  { value: "ambient", label: "Ambient" },
  { value: "latin", label: "Latin" },
  { value: "strings", label: "Strings" },
];

// ── Helpers ────────────────────────────────────────────────────

function loadSettings(): Record<string, string> | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function saveSettings(data: Record<string, string>) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
}

function getModelInfo(m: Model): string {
  const mc = m.modelConfig;
  const domain = m.domain || "novels";
  const parts = [
    domain,
    `${mc.nLayer}L ${mc.nEmbd}D ${mc.nHead}H`,
    `vocab=${mc.vocabSize}`,
    `ctx=${mc.blockSize}`,
  ];
  if (m.lastLoss != null) parts.push(`loss=${m.lastLoss.toFixed(3)}`);
  return parts.join(" | ");
}

// ── Component ──────────────────────────────────────────────────

export default function ChatPageWrapper() {
  return (
    <Suspense>
      <ChatPage />
    </Suspense>
  );
}

function ChatPage() {
  // Models
  const [models, setModels] = useState<Model[]>([]);
  const [modelId, setModelId] = useState("");

  // Settings
  const [maxTokens, setMaxTokens] = useState(200);
  const [temperature, setTemperature] = useState(0.8);
  const [topk, setTopk] = useState(40);
  const [ctxMsgs, setCtxMsgs] = useState(4);
  const [settingsOpen, setSettingsOpen] = useState(false);

  // Chat
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [generating, setGenerating] = useState(false);
  const [status, setStatus] = useState("");

  // Music modal
  const [musicModalOpen, setMusicModalOpen] = useState(false);
  const [musicChords, setMusicChords] = useState("");
  const [musicMsgIndex, setMusicMsgIndex] = useState(-1);
  const [musicDesc, setMusicDesc] = useState("");
  const [musicStyle, setMusicStyle] = useState("");
  const [musicGenerating, setMusicGenerating] = useState(false);
  const [musicStatus, setMusicStatus] = useState("");
  const [musicBlobUrl, setMusicBlobUrl] = useState<string | null>(null);
  const [musicMeta, setMusicMeta] = useState("");

  // Refs
  const messagesRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const generatingRef = useRef(false);
  const messagesStateRef = useRef<ChatMessage[]>([]);

  const searchParams = useSearchParams();

  // Keep refs in sync
  useEffect(() => {
    generatingRef.current = generating;
  }, [generating]);
  useEffect(() => {
    messagesStateRef.current = messages;
  }, [messages]);

  // ── Persist settings ─────────────────────────────────────────

  const persistSettings = useCallback(
    (overrides?: Partial<{ model: string; maxTokens: number; temp: number; topk: number; ctxMsgs: number }>) => {
      const data: Record<string, string> = {
        model: overrides?.model ?? modelId,
        maxTokens: String(overrides?.maxTokens ?? maxTokens),
        temp: String(overrides?.temp ?? temperature),
        topk: String(overrides?.topk ?? topk),
        ctxMsgs: String(overrides?.ctxMsgs ?? ctxMsgs),
      };
      saveSettings(data);
    },
    [modelId, maxTokens, temperature, topk, ctxMsgs]
  );

  // ── Load models + settings on mount ──────────────────────────

  useEffect(() => {
    const saved = loadSettings();

    if (saved) {
      if (saved.maxTokens) setMaxTokens(Number(saved.maxTokens));
      if (saved.temp) setTemperature(Number(saved.temp));
      if (saved.topk) setTopk(Number(saved.topk));
      if (saved.ctxMsgs) setCtxMsgs(Number(saved.ctxMsgs));
    }

    fetch("/api/models")
      .then((r) => r.json())
      .then((data: Model[]) => {
        setModels(data);
        if (data.length === 0) return;

        const urlModel = searchParams.get("model");
        let selected = data[0].id;

        if (urlModel && data.find((m) => m.id === urlModel)) {
          selected = urlModel;
        } else if (saved?.model && data.find((m) => m.id === saved.model)) {
          selected = saved.model;
        }

        setModelId(selected);

        // Set URL param
        const url = new URL(window.location.href);
        url.searchParams.set("model", selected);
        history.replaceState(null, "", url.toString());
      })
      .catch(() => setStatus("Failed to load models."));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Derived state ────────────────────────────────────────────

  const selectedModel = models.find((m) => m.id === modelId);
  const domain = selectedModel?.domain || "novels";
  const isChords = domain === "chords";
  const modelInfo = selectedModel ? getModelInfo(selectedModel) : "";

  // ── Settings change handlers ─────────────────────────────────

  const handleMaxTokensChange = useCallback(
    (v: number) => {
      setMaxTokens(v);
      persistSettings({ maxTokens: v });
    },
    [persistSettings]
  );

  const handleTemperatureChange = useCallback(
    (v: number) => {
      setTemperature(v);
      persistSettings({ temp: v });
    },
    [persistSettings]
  );

  const handleTopkChange = useCallback(
    (v: number) => {
      setTopk(v);
      persistSettings({ topk: v });
    },
    [persistSettings]
  );

  const handleCtxMsgsChange = useCallback(
    (v: number) => {
      setCtxMsgs(v);
      persistSettings({ ctxMsgs: v });
    },
    [persistSettings]
  );

  const handleModelChange = useCallback(
    (id: string) => {
      setModelId(id);
      persistSettings({ model: id });
      const url = new URL(window.location.href);
      url.searchParams.set("model", id);
      history.replaceState(null, "", url.toString());
    },
    [persistSettings]
  );

  // ── Auto-scroll messages ─────────────────────────────────────

  useEffect(() => {
    if (messagesRef.current) {
      messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
    }
  }, [messages]);

  // ── Auto-resize textarea ─────────────────────────────────────

  const autoResize = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 150) + "px";
  }, []);

  // ── Clear chat ───────────────────────────────────────────────

  const clearChat = useCallback(() => {
    setMessages([]);
    setStatus("");
  }, []);

  // ── Send message ─────────────────────────────────────────────

  const sendMessage = useCallback(async () => {
    const text = input.trim();
    if (!text || generatingRef.current) return;

    const userMsg: ChatMessage = { role: "user", content: text };
    const assistantMsg: ChatMessage = { role: "assistant", content: "" };

    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setGenerating(true);
    setStatus("Generating...");

    const currentMessages = [...messagesStateRef.current, userMsg];
    const apiMessages = currentMessages.slice(-(ctxMsgs));

    const controller = new AbortController();
    abortRef.current = controller;
    const t0 = performance.now();
    let charCount = 0;

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: apiMessages,
          model: modelId,
          maxTokens,
          temperature,
          topk,
        }),
        signal: controller.signal,
      });

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let assistantText = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        assistantText += chunk;
        charCount += chunk.length;

        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            role: "assistant",
            content: assistantText,
          };
          return updated;
        });
      }

      const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
      setStatus(`${charCount} chars in ${elapsed}s`);
    } catch (e: unknown) {
      if (e instanceof Error && e.name !== "AbortError") {
        setStatus(`Error: ${e.message}`);
        setMessages((prev) => {
          if (prev.length > 0 && prev[prev.length - 1].content === "") {
            return prev.slice(0, -1);
          }
          return prev;
        });
      }
    }

    setGenerating(false);
    abortRef.current = null;
  }, [input, ctxMsgs, modelId, maxTokens, temperature, topk]);

  // ── Keyboard handler ─────────────────────────────────────────

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    },
    [sendMessage]
  );

  // ── Cleanup on unmount ───────────────────────────────────────

  useEffect(() => {
    return () => {
      if (abortRef.current) {
        abortRef.current.abort();
      }
    };
  }, []);

  // ── Music modal ──────────────────────────────────────────────

  const openMusicModal = useCallback(
    (chords: string, msgIndex: number) => {
      setMusicChords(chords.trim());
      setMusicMsgIndex(msgIndex);
      setMusicDesc("");
      setMusicStyle("");
      setMusicStatus("");
      setMusicBlobUrl(null);
      setMusicMeta("");
      setMusicGenerating(false);
      setMusicModalOpen(true);
    },
    []
  );

  const closeMusicModal = useCallback(() => {
    setMusicModalOpen(false);
  }, []);

  const resetModal = useCallback(() => {
    setMusicBlobUrl(null);
    setMusicGenerating(false);
    setMusicStatus("");
    setMusicDesc("");
    setMusicMeta("");
  }, []);

  const generateMusic = useCallback(async () => {
    setMusicGenerating(true);
    setMusicStatus("Generating audio... this may take up to 15 seconds.");

    const body: Record<string, string> = { chords: musicChords };
    if (musicDesc.trim()) body.description = musicDesc.trim();
    if (musicStyle) body.style = musicStyle;

    try {
      const t0 = performance.now();
      const res = await fetch(GAMMA_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Generation failed (${res.status}): ${text}`);
      }

      const blob = await res.blob();
      const blobUrl = URL.createObjectURL(blob);

      const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
      const usedStyle = res.headers.get("X-Style") || musicStyle || "auto";
      const usedBpm = res.headers.get("X-BPM") || "?";

      setMusicBlobUrl(blobUrl);
      setMusicMeta(`Style: ${usedStyle} | BPM: ${usedBpm} | Generated in ${elapsed}s`);
      setMusicStatus("");
      setMusicGenerating(false);

      // Attach audio to the message
      if (musicMsgIndex >= 0) {
        setMessages((prev) => {
          const updated = [...prev];
          if (updated[musicMsgIndex]) {
            updated[musicMsgIndex] = {
              ...updated[musicMsgIndex],
              audioUrl: blobUrl,
            };
          }
          return updated;
        });
      }
    } catch (e: unknown) {
      setMusicStatus(`Error: ${e instanceof Error ? e.message : "Unknown error"}`);
      setMusicGenerating(false);
    }
  }, [musicChords, musicDesc, musicStyle, musicMsgIndex]);

  const downloadAudio = useCallback(
    (url: string) => {
      const a = document.createElement("a");
      a.href = url;
      a.download = "alpha-chords.mp3";
      a.click();
    },
    []
  );

  // ── Render ───────────────────────────────────────────────────

  return (
    <div
      className="-mx-6 -mt-6 flex flex-col"
      style={{ height: "calc(100vh - 3.5rem)" }}
    >
      {/* ── Header bar ── */}
      <div className="flex shrink-0 items-center gap-3 border-b border-border bg-surface px-4 py-2.5">
        <h1 className="whitespace-nowrap text-sm font-semibold text-text-primary">
          Alpha Chat
        </h1>
        <select
          value={modelId}
          onChange={(e) => handleModelChange(e.target.value)}
          className="rounded border border-border-2 bg-surface-2 px-2 py-1.5 text-xs text-text-primary outline-none focus:border-accent"
        >
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.name} [{m.domain || "novels"}] (step {m.step})
            </option>
          ))}
        </select>
        <span className="text-[0.7rem] text-text-muted">{modelInfo}</span>
        <span className="flex-1" />
        <Button
          variant="outline"
          size="sm"
          onClick={() => setSettingsOpen((v) => !v)}
        >
          Settings
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={clearChat}
        >
          Clear
        </Button>
      </div>

      {/* ── Settings panel ── */}
      {settingsOpen && (
        <div className="flex shrink-0 flex-wrap items-center gap-4 border-b border-border bg-surface px-4 py-2">
          <div className="flex items-center gap-1.5">
            <label className="text-xs text-text-secondary">Max tokens</label>
            <input
              type="number"
              value={maxTokens}
              onChange={(e) => handleMaxTokensChange(Number(e.target.value))}
              min={1}
              max={500}
              className="w-[4.5rem] rounded border border-border-2 bg-surface-2 px-2 py-1 text-xs text-text-primary outline-none focus:border-accent"
            />
          </div>
          <div className="flex items-center gap-1.5">
            <label className="text-xs text-text-secondary">Temperature</label>
            <input
              type="number"
              value={temperature}
              onChange={(e) => handleTemperatureChange(Number(e.target.value))}
              min={0.1}
              max={2.0}
              step={0.1}
              className="w-[4.5rem] rounded border border-border-2 bg-surface-2 px-2 py-1 text-xs text-text-primary outline-none focus:border-accent"
            />
          </div>
          <div className="flex items-center gap-1.5">
            <label className="text-xs text-text-secondary">Top-k</label>
            <input
              type="number"
              value={topk}
              onChange={(e) => handleTopkChange(Number(e.target.value))}
              min={0}
              max={200}
              className="w-[4.5rem] rounded border border-border-2 bg-surface-2 px-2 py-1 text-xs text-text-primary outline-none focus:border-accent"
            />
          </div>
          <div className="flex items-center gap-1.5">
            <label className="text-xs text-text-secondary">Context msgs</label>
            <input
              type="number"
              value={ctxMsgs}
              onChange={(e) => handleCtxMsgsChange(Number(e.target.value))}
              min={1}
              max={20}
              className="w-[4.5rem] rounded border border-border-2 bg-surface-2 px-2 py-1 text-xs text-text-primary outline-none focus:border-accent"
            />
          </div>
        </div>
      )}

      {/* ── Messages area ── */}
      <div
        ref={messagesRef}
        className="flex flex-1 flex-col overflow-y-auto py-4"
      >
        {messages.length === 0 ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-2 text-text-muted">
            <h2 className="text-lg font-normal text-text-secondary">
              Alpha Chat
            </h2>
            <p className="text-sm">
              Select a model and start typing. These are base language models
              trained on text, not instruction-tuned.
            </p>
            <p className="text-sm">
              They work best as text continuations — type the beginning of a
              sentence or paragraph.
            </p>
            {models.length === 0 && (
              <p className="mt-2 text-xs text-text-muted">
                No models loaded yet. Models become available automatically after training runs upload checkpoints.
              </p>
            )}
          </div>
        ) : (
          messages.map((msg, i) => (
            <div
              key={i}
              className={`mx-auto w-full max-w-[720px] px-5 py-3 ${
                msg.role === "user"
                  ? "mb-2 self-center rounded-xl bg-surface-2/70"
                  : "mb-2 bg-surface"
              }`}
            >
              <div className="mb-1 text-[0.7rem] uppercase tracking-wider text-text-secondary">
                {msg.role}
              </div>
              <div
                className={`whitespace-pre-wrap break-words text-sm leading-relaxed ${
                  msg.role === "assistant"
                    ? "font-mono text-[0.85rem] text-text-primary"
                    : "text-text-primary"
                }`}
              >
                {msg.content}
                {generating &&
                  i === messages.length - 1 &&
                  msg.role === "assistant" && (
                    <span className="animate-blink ml-0.5 inline-block h-[1em] w-0.5 bg-accent align-text-bottom" />
                  )}
              </div>

              {/* Music button for chords domain assistant messages */}
              {isChords &&
                msg.role === "assistant" &&
                msg.content.trim() &&
                !generating && (
                  <Button
                    variant="outline"
                    size="sm"
                    className="mt-2.5 text-yellow border-yellow/30 bg-yellow-bg/50 hover:bg-yellow-bg hover:text-yellow/90 gap-1.5"
                    onClick={() => openMusicModal(msg.content, i)}
                  >
                    <svg
                      viewBox="0 0 24 24"
                      className="h-3.5 w-3.5 fill-current"
                    >
                      <path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55C7.79 13 6 14.79 6 17s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z" />
                    </svg>
                    Make Music
                  </Button>
                )}

              {/* Inline audio player */}
              {msg.audioUrl && (
                <div className="mt-2 flex items-center gap-2.5 rounded-lg border border-border bg-surface-2 px-3 py-2">
                  <audio controls src={msg.audioUrl} className="h-8 flex-1" />
                  <Button
                    variant="primary"
                    size="sm"
                    onClick={() => downloadAudio(msg.audioUrl!)}
                    className="bg-blue hover:bg-blue/90"
                  >
                    MP3
                  </Button>
                </div>
              )}
            </div>
          ))
        )}
      </div>

      {/* ── Input area ── */}
      <div className="shrink-0 border-t border-border bg-surface px-4 py-3">
        <div className="mx-auto flex max-w-[720px] items-end gap-2">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => {
              setInput(e.target.value);
              autoResize();
            }}
            onKeyDown={handleKeyDown}
            rows={1}
            placeholder={
              isChords ? "Enter chord progression..." : "Type a message..."
            }
            className="flex-1 resize-none rounded-lg border border-border-2 bg-surface-2 px-3 py-2.5 text-sm leading-snug text-text-primary outline-none placeholder:text-text-muted focus:border-accent"
            style={{ minHeight: 44, maxHeight: 150 }}
            autoFocus
          />
          <Button
            variant="primary"
            onClick={sendMessage}
            disabled={generating}
            className="h-[44px] w-[44px] shrink-0 p-0 text-lg"
          >
            &#9654;
          </Button>
        </div>
        <div className="mx-auto mt-1 min-h-[1em] max-w-[720px] text-[0.7rem] text-text-muted">
          {status}
        </div>
      </div>

      {/* ── Music generation modal ── */}
      {musicModalOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
          onClick={(e) => {
            if (e.target === e.currentTarget) closeMusicModal();
          }}
        >
          <Card className="relative w-[90%] max-w-[440px] p-6 shadow-2xl">
            <h3 className="mb-1 text-base font-semibold text-text-primary">
              Generate Music
            </h3>

            {/* Chords display */}
            <div className="mb-3 mt-2 max-h-20 overflow-y-auto whitespace-pre-wrap break-words rounded-md border border-border bg-surface-2 px-3 py-2 font-mono text-xs text-yellow">
              {musicChords}
            </div>

            {/* Description */}
            <label className="mb-1 block text-xs text-text-secondary">
              Describe the sound you want
            </label>
            <textarea
              value={musicDesc}
              onChange={(e) => setMusicDesc(e.target.value)}
              placeholder="e.g. smoky jazz bar, walking bass, mellow piano..."
              className="mb-3 min-h-[60px] w-full resize-none rounded-md border border-border-2 bg-surface-2 px-3 py-2 text-sm text-text-primary outline-none placeholder:text-text-muted focus:border-accent"
              autoFocus
            />

            {/* Style selector */}
            <div className="flex gap-2">
              <select
                value={musicStyle}
                onChange={(e) => setMusicStyle(e.target.value)}
                className="flex-1 rounded-md border border-border-2 bg-surface-2 px-2.5 py-2 text-xs text-text-primary outline-none focus:border-accent"
              >
                {MUSIC_STYLES.map((s) => (
                  <option key={s.value} value={s.value}>
                    {s.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Actions */}
            <div className="mt-4 flex justify-end gap-2">
              <Button
                variant="outline"
                onClick={closeMusicModal}
              >
                Cancel
              </Button>
              <Button
                variant="primary"
                onClick={generateMusic}
                disabled={musicGenerating}
                className="bg-yellow hover:bg-yellow/90 text-black"
              >
                Generate
              </Button>
            </div>

            {/* Status */}
            {musicStatus && (
              <div className="mt-2 min-h-[1.2em] text-xs text-text-muted">
                {musicStatus}
              </div>
            )}

            {/* Audio player in modal */}
            {musicBlobUrl && (
              <div className="mt-3 flex flex-col gap-2">
                <audio controls src={musicBlobUrl} className="h-9 w-full" />
                <div className="flex gap-2">
                  <Button
                    variant="primary"
                    size="sm"
                    onClick={() => downloadAudio(musicBlobUrl)}
                    className="bg-blue hover:bg-blue/90 flex-1"
                  >
                    Download MP3
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={resetModal}
                    className="flex-1"
                  >
                    Generate another
                  </Button>
                </div>
                {musicMeta && (
                  <div className="text-[0.7rem] text-text-muted text-center mt-1">
                    {musicMeta}
                  </div>
                )}
              </div>
            )}
          </Card>
        </div>
      )}

    </div>
  );
}
