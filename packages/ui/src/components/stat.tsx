import * as React from "react";
import { Tip } from "./tooltip.js";

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
