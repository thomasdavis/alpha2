"use client";

import { useState, useEffect, useMemo } from "react";
import Link from "next/link";
import { Tip } from "@/components/tooltip";
import { tips } from "@/components/tip-data";
import { StatCard, FilterBtn, Spinner, EmptyState, Card } from "@alpha/ui";

// ── Types ───────────────────────────────────────────────────────

interface DbRun {
  id: string;
  domain: string;
  status: string;
  latest_step: number | null;
  total_iters: number | null;
  estimated_params: number | null;
  model_config: string;
  last_loss: number | null;
  best_val_loss: number | null;
  updated_at: string;
}

interface ModelRow {
  id: string; // run ID
  name: string; // short name
  domain: string;
  step: number;
  totalIters: number;
  estimatedParams: number | null;
  nLayer: number;
  nEmbd: number;
  nHead: number;
  lastLoss: number | null;
  bestValLoss: number | null;
  inferenceAvailable: boolean;
  updatedAt: string;
}

// ── Helpers ─────────────────────────────────────────────────────

function formatParams(n: number | null): string {
  if (n == null) return "-";
  if (n >= 1e6) return (n / 1e6).toFixed(0) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(0) + "K";
  return String(n);
}

function progress(step: number, total: number): number {
  if (!total) return 0;
  return Math.min(100, Math.round((step / total) * 100));
}

type SortKey = "recent" | "loss" | "params" | "step" | "name";

function sortModels(models: ModelRow[], key: SortKey) {
  const m = [...models];
  switch (key) {
    case "recent":
      m.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
      break;
    case "loss":
      m.sort((a, b) => {
        const al = a.bestValLoss ?? a.lastLoss ?? Infinity;
        const bl = b.bestValLoss ?? b.lastLoss ?? Infinity;
        return al - bl;
      });
      break;
    case "params":
      m.sort((a, b) => (b.estimatedParams ?? 0) - (a.estimatedParams ?? 0));
      break;
    case "step":
      m.sort((a, b) => b.step - a.step);
      break;
    case "name":
      m.sort((a, b) => a.name.localeCompare(b.name));
      break;
  }
  return m;
}

const DOMAIN_META: Record<string, { label: string; icon: string; bg: string; color: string }> = {
  novels: { label: "Novels", icon: "N", bg: "bg-blue-bg", color: "text-blue" },
  chords: { label: "Chords", icon: "C", bg: "bg-yellow-bg", color: "text-yellow" },
  abc: { label: "ABC", icon: "A", bg: "bg-green-bg", color: "text-green" },
  dumb_finance: { label: "Finance", icon: "F", bg: "bg-red-bg", color: "text-red" },
  concordance: { label: "Corpus", icon: "Tx", bg: "bg-cyan-950", color: "text-cyan-400" },
};

function domainMeta(d: string) {
  return DOMAIN_META[d] || { label: d, icon: d.charAt(0).toUpperCase(), bg: "bg-surface-2", color: "text-text-primary" };
}

// ── Component ───────────────────────────────────────────────────

export default function ModelsPage() {
  const [models, setModels] = useState<ModelRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [filterDomain, setFilterDomain] = useState<string>("all");
  const [sortKey, setSortKey] = useState<SortKey>("recent");

  useEffect(() => {
    Promise.all([
      fetch("/api/runs?limit=1000").then((r) => r.ok ? r.json() : []),
      fetch("/api/models").then((r) => r.ok ? r.json() : [])
    ]).then(([runData, activeModels]) => {
      const runs = runData as DbRun[];
      const activeIds = activeModels as { id: string }[];
      const engineIds = new Set(activeIds.map(m => m.id));

      const rows: ModelRow[] = runs.map((r) => {
        let nLayer = 0, nEmbd = 0, nHead = 0;
        try {
          const cfg = JSON.parse(r.model_config);
          nLayer = cfg.nLayer || cfg.n_layer || 0;
          nEmbd = cfg.nEmbd || cfg.n_embd || 0;
          nHead = cfg.nHead || cfg.n_head || 0;
        } catch {}

        return {
          id: r.id,
          name: r.id,
          domain: r.domain,
          step: r.latest_step ?? 0,
          totalIters: r.total_iters ?? 0,
          estimatedParams: r.estimated_params,
          nLayer,
          nEmbd,
          nHead,
          lastLoss: r.last_loss,
          bestValLoss: r.best_val_loss,
          inferenceAvailable: engineIds.has(r.id),
          updatedAt: r.updated_at,
        };
      });

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
          <Spinner size="lg" />
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
      <EmptyState 
        title="No models yet" 
        description="Models appear automatically when training runs save checkpoints. Start a training run to see models here."
        icon={
          <svg width="24" height="24" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-text-muted">
            <path d="M8 1.5L14 5v6l-6 3.5L2 11V5l6-3.5z" />
            <path d="M8 8.5V15" />
            <path d="M2 5l6 3.5L14 5" />
          </svg>
        }
      />
    );
  }

  // ── Main ────────────────────────────────────────────────────

  return (
    <>
      {/* Header */}
      <div className="mb-6 flex items-start justify-between gap-4">
        <div>
          <h1 className="text-lg font-bold text-text-primary">Models</h1>
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
      <Card className="overflow-hidden p-0 border-0 bg-transparent shadow-none">
        {/* Table header */}
        <div className="hidden sm:grid grid-cols-[1fr_80px_90px_80px_80px_60px] gap-2 bg-surface-2 px-4 py-2 text-[0.65rem] font-semibold uppercase tracking-wider text-text-muted rounded-t-lg border border-border border-b-0">
          <span>Model <Tip text={tips.model} /></span>
          <span className="text-right">Params <Tip text={tips.params} /></span>
          <span className="text-right">Architecture <Tip text={tips.architecture} /></span>
          <span className="text-right">Step <Tip text={tips.step} /></span>
          <span className="text-right">Loss <Tip text={tips.loss} /></span>
          <span />
        </div>

        {/* Rows */}
        <div className="border border-border rounded-lg sm:rounded-none sm:rounded-b-lg overflow-hidden bg-surface">
        {filtered.map((m) => {
          const pct = progress(m.step, m.totalIters);
          const meta = domainMeta(m.domain);
          const bestLoss = m.bestValLoss ?? m.lastLoss;

          return (
            <div key={m.id} className="border-t border-border first:border-t-0">
              <Link
                href={`/models/${m.id}`}
                className="group grid grid-cols-1 sm:grid-cols-[1fr_80px_90px_80px_80px_60px] items-center gap-2 px-4 py-3 transition-colors hover:bg-surface-2/50 hover:no-underline"
              >
                {/* Name + domain + status */}
                <div className="flex items-center gap-3 min-w-0">
                  <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-xs font-bold ${meta.bg} ${meta.color}`}>
                    {meta.icon}
                  </div>
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="truncate font-semibold text-text-primary group-hover:text-accent transition-colors">
                        {m.name}
                      </span>
                      {m.inferenceAvailable && (
                        <span className="shrink-0 rounded bg-green-bg px-1.5 py-0.5 text-[0.6rem] font-bold uppercase tracking-wider text-green">
                          Live
                        </span>
                      )}
                    </div>
                    <div className="mt-0.5 text-xs text-text-muted">
                      {meta.label}
                    </div>
                  </div>
                </div>

                {/* Params */}
                <div className="hidden sm:block text-right text-sm text-text-secondary">
                  {formatParams(m.estimatedParams)}
                </div>

                {/* Arch */}
                <div className="hidden sm:block text-right text-xs text-text-muted">
                  {m.nLayer}L {m.nEmbd}D
                </div>

                {/* Step + progress */}
                <div className="hidden sm:block text-right">
                  <div className="text-sm font-mono text-text-primary">
                    {(m.step / 1000).toFixed(1)}k
                  </div>
                  <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-surface-2">
                    <div
                      className="h-full rounded-full bg-accent"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>

                {/* Loss */}
                <div className="hidden sm:block text-right font-mono text-sm text-text-primary">
                  {bestLoss != null ? bestLoss.toFixed(4) : "-"}
                </div>

                {/* Arrow */}
                <div className="hidden sm:flex justify-end text-text-muted group-hover:text-accent transition-colors">
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="6,4 10,8 6,12" />
                  </svg>
                </div>
              </Link>
            </div>
          );
        })}
        </div>
      </Card>

      {/* Footer note */}
      <p className="mt-4 text-[0.7rem] text-text-muted leading-relaxed">
        All training runs are shown. Models marked <span className="text-green font-semibold">LIVE</span> have
        checkpoints uploaded to the inference server and can be used for chat and text generation.
      </p>
    </>
  );
}
