import * as React from "react";
import { Tip } from "./tooltip.js";

export interface DetailRowProps {
  label: string;
  value: string | number | null | undefined;
  tip?: string;
}

export function DetailRow({ label, value, tip }: DetailRowProps) {
  return (
    <div className="flex justify-between border-b border-border/30 py-1.5 last:border-0">
      <span className="text-[0.7rem] text-text-muted">{label}{tip && <Tip text={tip} />}</span>
      <span className="font-mono text-[0.7rem] text-text-primary">{value ?? "-"}</span>
    </div>
  );
}
