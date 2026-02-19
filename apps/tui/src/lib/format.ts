import type { ModelConfig } from "../types.js";

export function estimateParams(mc: ModelConfig): number {
  const E = mc.nEmbd, L = mc.nLayer, V = mc.vocabSize, B = mc.blockSize;
  return V * E + B * E + L * (4 * E * E + 4 * E + 4 * 4 * E * E + 4 * E + 2 * E) + 2 * E + V * E;
}

export function formatParams(n: number): string {
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(0) + "K";
  return n.toString();
}

export function formatLoss(loss: number | undefined): string {
  if (loss == null) return "—";
  return loss.toFixed(4);
}

export function formatNumber(n: number): string {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return n.toFixed(0);
}

export function formatDuration(ms: number): string {
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

export function formatEta(etaMs: number | undefined): string {
  if (etaMs == null) return "—";
  return formatDuration(etaMs);
}

export function formatStep(step: number, total: number): string {
  return `${step}/${total}`;
}

export function formatTokPerSec(tps: number): string {
  if (tps >= 1e6) return (tps / 1e6).toFixed(1) + "M";
  if (tps >= 1e3) return (tps / 1e3).toFixed(1) + "K";
  return tps.toFixed(0);
}

// ── Progress bar ──────────────────────────────────────────────────────────

export function progressBar(current: number, total: number, width: number): { filled: string; empty: string; pct: number } {
  const pct = total > 0 ? Math.min(current / total, 1) : 0;
  const filledW = Math.round(pct * width);
  const emptyW = width - filledW;
  return {
    filled: "━".repeat(filledW),
    empty: "╌".repeat(emptyW),
    pct: Math.round(pct * 100),
  };
}

// ── Relative time ─────────────────────────────────────────────────────────

export function timeAgo(ms: number): string {
  const delta = Date.now() - ms;
  if (delta < 0) return "now";
  const s = Math.floor(delta / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  return `${d}d ago`;
}

// ── Loss quality color ────────────────────────────────────────────────────

export function lossColor(loss: number | undefined): string {
  if (loss == null) return "gray";
  if (loss < 2.5) return "greenBright";
  if (loss < 3.5) return "green";
  if (loss < 4.5) return "yellow";
  if (loss < 5.5) return "red";
  return "redBright";
}

// ── Bytes ─────────────────────────────────────────────────────────────────

export function formatBytes(bytes: number): string {
  if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + " MB";
  if (bytes >= 1e3) return (bytes / 1e3).toFixed(1) + " KB";
  return bytes + " B";
}
