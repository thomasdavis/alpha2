import * as React from "react";
import { cn } from "../utils.js";

export function MethodBadge({ method }: { method: "GET" | "POST" }) {
  const styles =
    method === "GET"
      ? "bg-blue-bg text-blue"
      : "bg-green-bg text-green";
  return (
    <span
      className={`inline-block rounded px-1.5 py-0.5 text-[0.7rem] font-bold uppercase ${styles}`}
    >
      {method}
    </span>
  );
}

export function Endpoint({ path }: { path: string }) {
  return <code className="ml-1.5 font-mono text-[0.9rem] text-text-primary">{path}</code>;
}

export function Required() {
  return <span className="ml-0.5 text-[0.65rem] text-red">required</span>;
}

export function SectionHeading({
  children,
  color,
  className,
}: {
  children: React.ReactNode;
  color?: string;
  className?: string;
}) {
  return (
    <h2
      className={cn("mb-3 mt-10 border-b border-border pb-2 text-[1.1rem] font-semibold text-text-primary", className)}
      style={color ? { color } : undefined}
    >
      {children}
    </h2>
  );
}

export function EndpointHeading({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="mb-2 mt-8 flex items-center gap-1.5 border-b border-border pb-2 text-[1.1rem] font-semibold text-text-primary">
      {children}
    </h2>
  );
}

export function SubHeading({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="mb-2 mt-5 text-[0.95rem] font-medium text-text-primary/80">
      {children}
    </h3>
  );
}

export function Pre({ children }: { children: React.ReactNode }) {
  return (
    <pre className="mb-4 mt-2 overflow-x-auto rounded-md border border-border bg-surface px-4 py-3 font-mono text-[0.8rem] leading-relaxed text-text-primary">
      {children}
    </pre>
  );
}

export function ParamTable({
  headers,
  rows,
}: {
  headers: string[];
  rows: React.ReactNode[][];
}) {
  return (
    <div className="mb-4 mt-2 overflow-x-auto">
      <table className="w-full text-[0.85rem]">
        <thead>
          <tr>
            {headers.map((h) => (
              <th
                key={h}
                className="border-b border-border-2 px-2.5 py-2 text-left font-semibold text-text-secondary"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((cells, i) => (
            <tr key={i}>
              {cells.map((cell, j) => (
                <td
                  key={j}
                  className="border-b border-surface-2 px-2.5 py-2 text-text-primary/70"
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* Syntax highlighting helpers */
export function S({ children }: { children: React.ReactNode }) {
  return <span className="text-green">{children}</span>;
}
export function N({ children }: { children: React.ReactNode }) {
  return <span className="text-yellow">{children}</span>;
}
export function K({ children }: { children: React.ReactNode }) {
  return <span className="text-purple-400">{children}</span>;
}
export function C({ children }: { children: React.ReactNode }) {
  return <span className="text-text-muted">{children}</span>;
}
