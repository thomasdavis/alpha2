"use client";

import { useState, useEffect, useMemo } from "react";
import Link from "next/link";
import { Tip } from "@/components/tooltip";
import { tips } from "@/components/tip-data";

// ── Types ───────────────────────────────────────────────────────

interface DbRun {
  id: string;
  domain: string;
  status: string;
  n_layer: number;
  n_embd: number;
  n_head: number;
  vocab_size: number;
  block_size: number;
  total_iters: number;
  batch_size: number;
  lr: number;
  tokenizer: string;
  optimizer: string;
  latest_step: number;
  last_loss: number | null;
  best_val_loss: number | null;
  estimated_params: number | null;
  checkpoint_count: number;
  updated_at: string;
}

interface EngineModel {
  id: string;
}

interface ModelRow {
  id: string;
  domain: string;
  status: string;
  nLayer: number;
  nEmbd: number;
  nHead: number;
  vocabSize: number;
  blockSize: number;
  totalIters: number;
  step: number;
  lastLoss: number | null;
  bestValLoss: number | null;
  estimatedParams: number | null;
  checkpointCount: number;
  inferenceAvailable: boolean;
  updatedAt: string;
}

// ── Helpers ─────────────────────────────────────────────────────

const DOMAIN_META: Record<string, { label: string; color: string; bg: string; icon: string }> = {
  novels:       { label: "Novels",       color: "text-blue",   bg: "bg-blue-bg",   icon: "N" },
  abc:          { label: "ABC",          color: "text-green",  bg: "bg-green-bg",  icon: "A" },
  chords:       { label: "Chords",       color: "text-yellow", bg: "bg-yellow-bg", icon: "C" },
  dumb_finance: { label: "Finance",      color: "text-red",    bg: "bg-red-bg",    icon: "$" },
  chaos:        { label: "Chaos",        color: "text-purple", bg: "bg-purple-bg", icon: "X" },
};

function domainMeta(d: string) {
  return DOMAIN_META[d] || { label: d, color: "text-text-secondary", bg: "bg-surface-2", icon: d[0]?.toUpperCase() || "?" };
}

function fmtParams(n: number | null): string {
  if (n == null) return "-";
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(n);
}

function fmtLoss(loss: number | null): string {
  return loss != null ? loss.toFixed(3) : "-";
}

function lossColor(loss: number | null): string {
  if (loss == null) return "text-text-muted";
  if (loss < 3) return "text-green";
  if (loss < 5) return "text-yellow";
  return "text-red";
}

function progress(step: number, iters: number): number {
  if (!iters || iters <= 0) return 100;
  return Math.min(100, Math.round((step / iters) * 100));
}

type SortKey = "recent" | "loss" | "params" | "step" | "name";

function sortModels(models: ModelRow[], key: SortKey): ModelRow[] {
  const sorted = [...models];
  switch (key) {
    case "recent":
      return sorted.sort((a, b) => (b.updatedAt ?? "").localeCompare(a.updatedAt ?? ""));
    case "loss":
      return sorted.sort((a, b) => (a.lastLoss ?? Infinity) - (b.lastLoss ?? Infinity));
    case "params":
      return sorted.sort((a, b) => (b.estimatedParams ?? 0) - (a.estimatedParams ?? 0));
    case "step":
      return sorted.sort((a, b) => b.step - a.step);
    case "name":
      return sorted.sort((a, b) => a.id.localeCompare(b.id));
  }
}

// ── Component ───────────────────────────────────────────────────

export default function ModelsPage() {
  const [models, setModels] = useState<ModelRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("recent");
  const [filterDomain, setFilterDomain] = useState<string>("all");

  useEffect(() => {
    Promise.all([
      fetch("/api/runs").then((r) => r.ok ? r.json() : []).catch(() => []),
      fetch("/api/models").then((r) => r.ok ? r.json() : []).catch(() => []),
    ]).then(([dbRuns, engineModels]: [DbRun[], EngineModel[]]) => {
      const engineIds = new Set(engineModels.map((m) => m.id));

      const rows: ModelRow[] = dbRuns.map((r) => ({
        id: r.id,
        domain: r.domain || "unknown",
        status: r.status,
        nLayer: r.n_layer,
        nEmbd: r.n_embd,
        nHead: r.n_head,
        vocabSize: r.vocab_size,
        blockSize: r.block_size,
        totalIters: r.total_iters,
        step: r.latest_step,
        lastLoss: r.last_loss,
        bestValLoss: r.best_val_loss,
        estimatedParams: r.estimated_params,
        checkpointCount: r.checkpoint_count,
        inferenceAvailable: engineIds.has(r.id),
        updatedAt: r.updated_at,
      }));

      setModels(rows);
      setLoading(false);
    }).catch(() => {
      setError("Failed to load models.");
      setLoading(false);
    });
  }, []);

  const domains = useMemo(() => {
    const set = new Set(models.map((m) => m.domain));
    return Array.from(set).sort();
  }, [models]);

  const filtered = useMemo(() => {
    let list = models;
    if (filterDomain !== "all") list = list.filter((m) => m.domain === filterDomain);
    return sortModels(list, sortKey);
  }, [models, filterDomain, sortKey]);

  // Aggregate stats
  const stats = useMemo(() => {
    let bestLoss = Infinity, totalSteps = 0, inferenceCount = 0;
    for (const m of models) {
      if (m.lastLoss != null && m.lastLoss < bestLoss) bestLoss = m.lastLoss;
      totalSteps += m.step;
      if (m.inferenceAvailable) inferenceCount++;
    }
    return { bestLoss, totalSteps, count: models.length, domains: domains.length, inferenceCount };
  }, [models, domains]);

  // ── Loading / Error / Empty ─────────────────────────────────

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <div className="flex flex-col items-center gap-3">
          <div className="h-6 w-6 animate-spin rounded-full border-2 border-border-2 border-t-accent" />
          <span className="text-sm text-text-muted">Loading models...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center py-24">
        <p className="text-sm text-red">{error}</p>
      </div>
    );
  }

  if (models.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-24 gap-4">
        <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-surface-2 border border-border">
          <svg width="24" height="24" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-text-muted">
            <path d="M8 1.5L14 5v6l-6 3.5L2 11V5l6-3.5z" />
            <path d="M8 8.5V15" />
            <path d="M2 5l6 3.5L14 5" />
          </svg>
        </div>
        <div className="text-center">
          <p className="text-sm font-medium text-text-secondary">No models yet</p>
          <p className="mt-1 max-w-xs text-xs text-text-muted leading-relaxed">
            Models appear automatically when training runs save checkpoints.
            Start a training run to see models here.
          </p>
        </div>
      </div>
    );
  }

  // ── Main ────────────────────────────────────────────────────

  return (
    <>
      {/* Header */}
      <div className="mb-6 flex items-start justify-between gap-4">
        <div>
          <h1 className="text-lg font-bold text-white">Models</h1>
          <p className="mt-0.5 text-xs text-text-muted">
            {stats.count} model{stats.count !== 1 ? "s" : ""} across {stats.domains} domain{stats.domains !== 1 ? "s" : ""}
            {" "}&middot; {stats.inferenceCount} ready for inference
          </p>
        </div>
      </div>

      {/* Stats row */}
      <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatCard label="Models" value={String(stats.count)} tip={tips.model} />
        <StatCard label="Best Loss" value={stats.bestLoss < Infinity ? stats.bestLoss.toFixed(3) : "-"} accent tip={tips.loss} />
        <StatCard label="Inference Ready" value={String(stats.inferenceCount)} tip={tips.inferenceAvailable} />
        <StatCard label="Total Steps" value={stats.totalSteps.toLocaleString()} tip={tips.step} />
      </div>

      {/* Filters + sort */}
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <div className="flex gap-1 rounded-lg border border-border bg-surface p-0.5">
          <FilterBtn active={filterDomain === "all"} onClick={() => setFilterDomain("all")}>
            All
          </FilterBtn>
          {domains.map((d) => {
            const meta = domainMeta(d);
            return (
              <FilterBtn key={d} active={filterDomain === d} onClick={() => setFilterDomain(d)}>
                {meta.label}
              </FilterBtn>
            );
          })}
        </div>
        <span className="flex-1" />
        <select
          value={sortKey}
          onChange={(e) => setSortKey(e.target.value as SortKey)}
          className="rounded-lg border border-border bg-surface px-2.5 py-1.5 text-xs text-text-secondary outline-none"
        >
          <option value="recent">Most recent</option>
          <option value="loss">Sort by loss</option>
          <option value="params">Sort by params</option>
          <option value="step">Sort by step</option>
          <option value="name">Sort by name</option>
        </select>
      </div>

      {/* Model table */}
      <div className="overflow-hidden rounded-lg border border-border">
        {/* Table header */}
        <div className="hidden sm:grid grid-cols-[1fr_80px_90px_80px_80px_60px] gap-2 bg-surface-2 px-4 py-2 text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted">
          <span>Model <Tip text={tips.model} /></span>
          <span className="text-right">Params <Tip text={tips.params} /></span>
          <span className="text-right">Architecture <Tip text={tips.architecture} /></span>
          <span className="text-right">Step <Tip text={tips.step} /></span>
          <span className="text-right">Loss <Tip text={tips.loss} /></span>
          <span />
        </div>

        {/* Rows */}
        {filtered.map((m) => {
          const pct = progress(m.step, m.totalIters);
          const meta = domainMeta(m.domain);
          const bestLoss = m.bestValLoss ?? m.lastLoss;

          return (
            <div key={m.id} className="border-t border-border first:border-t-0">
              <Link
                href={`/models/${m.id}`}
                className="group grid grid-cols-1 sm:grid-cols-[1fr_80px_90px_80px_80px_60px] items-center gap-2 px-4 py-3 transition-colors hover:bg-surface hover:no-underline"
              >
                {/* Name + domain + status */}
                <div className="flex items-center gap-3 min-w-0">
                  <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-xs font-bold ${meta.bg} ${meta.color}`}>
                    {meta.icon}
                  </div>
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="truncate text-sm font-medium text-white">{m.id}</span>
                      <span className={`shrink-0 rounded px-1.5 py-0.5 text-[0.6rem] font-semibold uppercase tracking-wide ${meta.bg} ${meta.color}`}>
                        {meta.label}
                      </span>
                      {m.inferenceAvailable && (
                        <span className="shrink-0 rounded bg-green-bg px-1.5 py-0.5 text-[0.6rem] font-semibold text-green">
                          LIVE
                        </span>
                      )}
                    </div>
                    <div className="mt-0.5 text-[0.7rem] text-text-muted truncate">
                      {m.nLayer}L-{m.nEmbd}D-{m.nHead}H &middot; vocab {m.vocabSize} &middot; ctx {m.blockSize}
                    </div>
                  </div>
                </div>

                {/* Params */}
                <div className="hidden sm:block text-right text-sm text-text-secondary">
                  {fmtParams(m.estimatedParams)}
                </div>

                {/* Architecture */}
                <div className="hidden sm:block text-right font-mono text-xs text-text-muted">
                  {m.nLayer}L {m.nEmbd}D
                </div>

                {/* Step + progress */}
                <div className="hidden sm:flex flex-col items-end gap-1">
                  <span className="text-sm text-text-secondary">{m.step.toLocaleString()}</span>
                  <div className="h-1 w-full max-w-[60px] rounded-full bg-surface-2 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-accent transition-all"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>

                {/* Loss */}
                <div className={`hidden sm:block text-right text-sm font-mono ${lossColor(bestLoss)}`}>
                  {fmtLoss(bestLoss)}
                </div>

                {/* Arrow indicator */}
                <div className="hidden sm:flex justify-end">
                  <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-text-muted group-hover:text-text-secondary transition-colors">
                    <polyline points="6,4 10,8 6,12" />
                  </svg>
                </div>
              </Link>
            </div>
          );
        })}
      </div>

      {/* Footer note */}
      <p className="mt-4 text-[0.7rem] text-text-muted leading-relaxed">
        All training runs are shown. Models marked <span className="text-green font-semibold">LIVE</span> have
        checkpoints uploaded to the inference server and can be used for chat and text generation.
      </p>
    </>
  );
}

// ── Sub-components ────────────────────────────────────────────────

function StatCard({ label, value, accent, tip }: { label: string; value: string; accent?: boolean; tip?: string }) {
  return (
    <div className="rounded-lg border border-border bg-surface px-4 py-3">
      <div className={`text-lg font-semibold ${accent ? "text-green" : "text-white"}`}>{value}</div>
      <div className="text-[0.65rem] uppercase tracking-wider text-text-muted">
        {label}
        {tip && <Tip text={tip} />}
      </div>
    </div>
  );
}

function FilterBtn({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`rounded-md px-2.5 py-1 text-xs transition-colors ${
        active
          ? "bg-surface-2 font-medium text-white"
          : "text-text-muted hover:text-text-secondary"
      }`}
    >
      {children}
    </button>
  );
}
