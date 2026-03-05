import * as React from "react";
import { cn } from "../utils.js";

export function ArchBadge({ variant, children }: { variant: "cpu" | "gpu" | "dispatch"; children: React.ReactNode }) {
  const styles = {
    cpu: "bg-blue-bg text-blue",
    gpu: "bg-yellow-bg text-yellow",
    dispatch: "bg-[#2a1a3a] text-purple-400",
  };
  return (
    <span className={cn("inline-block rounded px-1.5 py-0.5 text-[0.62rem] font-semibold uppercase tracking-wide", styles[variant])}>
      {children}
    </span>
  );
}

export function Shape({ children }: { children: React.ReactNode }) {
  return <span className="ml-1 text-[0.65rem] text-text-muted font-mono">{children}</span>;
}

export function ArchCard({
  variant,
  title,
  shape,
  children,
  className,
}: {
  variant: "cpu" | "gpu" | "dispatch";
  title: string;
  shape?: string;
  children: React.ReactNode;
  className?: string;
}) {
  const border = { cpu: "border-l-blue", gpu: "border-l-yellow", dispatch: "border-l-purple-400" };
  return (
    <div className={cn("rounded-md border border-border bg-surface pl-0", className)}>
      <div className={cn("border-l-2 rounded-md px-3 py-2.5", border[variant])}>
        <div className="mb-1 flex items-center gap-2 text-[0.8rem] font-semibold text-text-primary">
          {title}
          {shape && <Shape>{shape}</Shape>}
        </div>
        <div className="text-[0.75rem] leading-relaxed text-text-secondary">{children}</div>
      </div>
    </div>
  );
}

export function ArchSectionHeading({ color, children }: { color?: string; children: React.ReactNode }) {
  return (
    <h2
      className="mb-3 mt-10 border-b border-border pb-2 text-[1.1rem] font-semibold text-text-primary"
      style={color ? { color } : undefined}
    >
      {children}
    </h2>
  );
}

export function PhaseBadge({ variant, children }: { variant: "forward" | "backward" | "optimize"; children: React.ReactNode }) {
  const styles = {
    forward: "bg-green-bg text-green",
    backward: "bg-red-bg text-red",
    optimize: "bg-yellow-bg text-yellow",
  };
  return (
    <span className={cn("inline-block rounded px-2 py-1 text-[0.7rem] font-bold uppercase tracking-wider", styles[variant])}>
      {children}
    </span>
  );
}

export function KernelChip({ name, children }: { name: string; children: React.ReactNode }) {
  return (
    <div className="rounded-md border border-border bg-surface px-3 py-2.5 transition-colors hover:border-border-2">
      <div className="mb-0.5 text-[0.75rem] font-semibold text-yellow">{name}</div>
      <div className="text-[0.65rem] leading-relaxed text-text-secondary">{children}</div>
    </div>
  );
}

export function InfraChip({ name, children }: { name: string; children: React.ReactNode }) {
  return (
    <div className="rounded-md border border-border bg-surface px-3 py-2.5">
      <div className="mb-0.5 text-[0.75rem] font-semibold text-yellow">{name}</div>
      <div className="text-[0.65rem] leading-relaxed text-text-secondary">{children}</div>
    </div>
  );
}

export function Arrow() {
  return (
    <div className="flex justify-center py-1">
      <svg width="16" height="20" viewBox="0 0 16 20" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-text-muted">
        <line x1="8" y1="0" x2="8" y2="16" />
        <polyline points="4,12 8,17 12,12" />
      </svg>
    </div>
  );
}

export function TransferStrip({ direction, children }: { direction: "upload" | "download"; children: React.ReactNode }) {
  const bg = direction === "upload" ? "from-blue-bg/50 to-yellow-bg/50" : "from-yellow-bg/50 to-blue-bg/50";
  return (
    <div className={cn("my-4 flex items-center justify-center gap-3 rounded-md border border-dashed border-border-2 bg-gradient-to-r px-4 py-2.5 text-[0.7rem] text-text-secondary", bg)}>
      {direction === "upload" ? (
        <>
          <ArchBadge variant="cpu">CPU</ArchBadge>
          <span className="flex gap-0.5">
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-yellow" />
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-yellow" style={{ animationDelay: "0.15s" }} />
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-yellow" style={{ animationDelay: "0.3s" }} />
          </span>
          <span>{children}</span>
          <span className="flex gap-0.5">
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-yellow" style={{ animationDelay: "0.45s" }} />
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-yellow" style={{ animationDelay: "0.6s" }} />
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-yellow" style={{ animationDelay: "0.75s" }} />
          </span>
          <ArchBadge variant="gpu">GPU</ArchBadge>
        </>
      ) : (
        <>
          <ArchBadge variant="gpu">GPU</ArchBadge>
          <span className="flex gap-0.5">
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-blue" />
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-blue" style={{ animationDelay: "0.15s" }} />
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-blue" style={{ animationDelay: "0.3s" }} />
          </span>
          <span>{children}</span>
          <span className="flex gap-0.5">
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-blue" style={{ animationDelay: "0.45s" }} />
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-blue" style={{ animationDelay: "0.6s" }} />
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-blue" style={{ animationDelay: "0.75s" }} />
          </span>
          <ArchBadge variant="cpu">CPU</ArchBadge>
        </>
      )}
    </div>
  );
}
