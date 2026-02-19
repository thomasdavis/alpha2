export function formatParams(n: number | null): string {
  if (n == null) return "-";
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(0) + "K";
  return n.toString();
}

export function formatLoss(v: number | null): string {
  return v != null ? v.toFixed(4) : "-";
}

export function formatNumber(n: number): string {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return n.toFixed(0);
}

export function formatBytes(b: number | null): string {
  if (b == null) return "-";
  if (b >= 1e6) return (b / 1e6).toFixed(1) + " MB";
  if (b >= 1e3) return (b / 1e3).toFixed(1) + " KB";
  return b + " B";
}

export function timeAgo(iso: string | null): string {
  if (!iso) return "-";
  const ms = Date.now() - new Date(iso + "Z").getTime();
  if (ms < 0) return "now";
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  return `${d}d ago`;
}

export function pct(step: number, total: number): number {
  if (!total) return 0;
  return Math.min(Math.round((step / total) * 100), 100);
}
