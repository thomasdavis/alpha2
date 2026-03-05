import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function fmtParams(n: number | null): string {
  if (n == null) return "-";
  if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(n);
}

export function fmtLoss(v: number | null): string {
  return v != null ? v.toFixed(4) : "-";
}

export function fmtBytes(b: number | null): string {
  if (b == null) return "-";
  if (b >= 1e9) return (b / 1e9).toFixed(1) + " GB";
  if (b >= 1e6) return (b / 1e6).toFixed(1) + " MB";
  if (b >= 1e3) return (b / 1e3).toFixed(1) + " KB";
  return b + " B";
}

export function fmtDuration(ms: number): string {
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rs = s % 60;
  if (m < 60) return `${m}m ${rs}s`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  return `${h}h ${rm}m`;
}

export function fmtNum(n: number | null | undefined, decimals = 0): string {
  if (n == null) return "0";
  return n.toLocaleString(undefined, { maximumFractionDigits: decimals });
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

export function fmtDate(iso: string | null): string {
  if (!iso) return "-";
  const d = new Date(iso + "Z");
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })
    + " " + d.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
}

export const STATUS_STYLES: Record<string, { badge: string; bar: string; gradient: string; variant: any }> = {
  active: {
    badge: "border-green/20 bg-green-bg text-green",
    bar: "bg-green",
    gradient: "from-green-bg/50",
    variant: "success",
  },
  completed: {
    badge: "border-blue/20 bg-blue-bg text-blue",
    bar: "bg-blue",
    gradient: "from-blue-bg/50",
    variant: "blue",
  },
  stale: {
    badge: "border-yellow/20 bg-yellow-bg text-yellow",
    bar: "bg-yellow",
    gradient: "from-yellow-bg/50",
    variant: "warning",
  },
  failed: {
    badge: "border-red/20 bg-red-bg text-red",
    bar: "bg-red",
    gradient: "from-red-bg/50",
    variant: "danger",
  },
};

export const DOMAIN_STYLES: Record<string, { badge: string; variant: any }> = {
  novels: { badge: "border-blue/20 bg-blue-bg text-blue", variant: "blue" },
  chords: { badge: "border-yellow/20 bg-yellow-bg text-yellow", variant: "warning" },
  abc: { badge: "border-green/20 bg-green-bg text-green", variant: "success" },
  dumb_finance: { badge: "border-red/20 bg-red-bg text-red", variant: "danger" },
  concordance: { badge: "border-cyan-500/20 bg-cyan-950 text-cyan-400", variant: "blue" },
};
