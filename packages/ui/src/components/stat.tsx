import * as React from "react";
import { Tip } from "./tooltip.js";
import { Card } from "./card.js";

export interface StatProps {
  label: string;
  value: string | number;
  sub?: string;
  color?: string;
  tip?: string;
}

export function Stat({ label, value, sub, color, tip }: StatProps) {
  return (
    <div className="rounded-lg border border-border/60 bg-surface-2/80 px-3 py-2.5">
      <div className={`font-mono text-sm font-bold ${color ?? "text-text-primary"}`}>{value}</div>
      <div className="text-[0.6rem] uppercase tracking-wider text-text-muted">
        {label}{tip && <Tip text={tip} />}
      </div>
      {sub && <div className="mt-0.5 text-[0.6rem] text-text-muted">{sub}</div>}
    </div>
  );
}

export interface StatCardProps {
  label: string;
  value: string | number;
  accent?: boolean;
  tip?: string;
  href?: string;
}

export function StatCard({ label, value, accent, tip, href }: StatCardProps) {
  const content = (
    <Card className="px-4 py-3 transition-colors hover:border-border-2">
      <div className={`text-2xl font-bold tracking-tight ${accent ? "text-green" : "text-text-primary"}`}>
        {value}
      </div>
      <div className="mt-1 text-[0.62rem] uppercase font-semibold tracking-widest text-text-muted">
        {label}
        {tip && <Tip text={tip} />}
      </div>
    </Card>
  );

  if (href) {
    return (
      <a href={href} className="hover:no-underline block h-full">
        {content}
      </a>
    );
  }

  return content;
}

