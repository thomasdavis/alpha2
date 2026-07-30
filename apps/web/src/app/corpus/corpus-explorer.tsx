"use client";

import Link from "next/link";
import { useMemo, useRef, useState } from "react";
import type {
  CorpusCellValue,
  CorpusPage,
  CorpusRelationDetail,
  CorpusRelationSummary
} from "@alpha/corpus";

export interface CorpusStageCount {
  relation: string;
  label: string;
  description: string;
  count: number;
}

interface CorpusExplorerProps {
  relations: CorpusRelationSummary[];
  detail: CorpusRelationDetail;
  page: CorpusPage;
  view: "rows" | "schema" | "lineage";
  stages: CorpusStageCount[];
  databaseUpdatedAt: string;
  safety: { readOnly: true; queryOnly: boolean };
}

interface CellSelection {
  column: string;
  value: CorpusCellValue;
  row: Record<string, CorpusCellValue>;
}

function formatInteger(value: number): string {
  return new Intl.NumberFormat("en").format(value);
}

function formatTimestamp(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("en", {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: "UTC"
  }).format(date) + " UTC";
}

function compactValue(value: CorpusCellValue): string {
  if (value == null) return "∅";
  const text = String(value);
  return text.length > 72 ? `${text.slice(0, 71)}…` : text;
}

function expandedValue(value: CorpusCellValue): string {
  if (value == null) return "NULL";
  if (typeof value === "number") return String(value);
  try {
    const parsed = JSON.parse(value) as unknown;
    if (parsed && typeof parsed === "object") return JSON.stringify(parsed, null, 2);
  } catch {
    // Ordinary text is already the exact value we want to display.
  }
  return value;
}

function urlFor(values: Record<string, string | number | undefined>): string {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(values)) {
    if (value == null || value === "") continue;
    params.set(key, String(value));
  }
  return `/corpus?${params.toString()}`;
}

function RelationList({
  relations,
  selected,
  filter
}: {
  relations: CorpusRelationSummary[];
  selected: string;
  filter: string;
}) {
  const visible = useMemo(() => {
    const needle = filter.trim().toLocaleLowerCase();
    if (!needle) return relations;
    return relations.filter((relation) =>
      relation.name.toLocaleLowerCase().includes(needle)
      || relation.kind.includes(needle)
      || relation.columns.some((column) =>
        column.name.toLocaleLowerCase().includes(needle)
        || column.type.toLocaleLowerCase().includes(needle)
      )
    );
  }, [filter, relations]);

  const groups = [
    { kind: "table", label: "Tables" },
    { kind: "view", label: "Views" }
  ] as const;

  return (
    <div className="space-y-5">
      {groups.map((group) => {
        const members = visible.filter((relation) => relation.kind === group.kind);
        if (members.length === 0) return null;
        return (
          <section key={group.kind} aria-labelledby={`corpus-${group.kind}-heading`}>
            <div className="mb-1 flex items-center justify-between px-2">
              <h2 id={`corpus-${group.kind}-heading`} className="text-[0.68rem] font-semibold uppercase tracking-[0.1em] text-text-muted">
                {group.label}
              </h2>
              <span className="font-mono text-[0.68rem] tabular-nums text-text-muted">{members.length}</span>
            </div>
            <ul className="space-y-0.5">
              {members.map((relation) => {
                const active = relation.name === selected;
                return (
                  <li key={relation.name}>
                    <Link
                      href={urlFor({ relation: relation.name })}
                      aria-current={active ? "page" : undefined}
                      className={`flex min-h-9 items-center justify-between gap-3 rounded-md px-2 py-1.5 text-xs transition-colors focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent ${
                        active
                          ? "bg-blue-bg font-medium text-blue"
                          : "text-text-secondary hover:bg-surface-2 hover:text-text-primary"
                      }`}
                    >
                      <span className="min-w-0 truncate font-mono" title={relation.name}>{relation.name}</span>
                      <span className="shrink-0 font-mono text-[0.68rem] tabular-nums text-text-muted">
                        {relation.columns.length}
                      </span>
                    </Link>
                  </li>
                );
              })}
            </ul>
          </section>
        );
      })}
      {visible.length === 0 && (
        <p className="px-2 py-8 text-center text-xs leading-relaxed text-text-muted">
          No table, view, column, or SQLite type matches “{filter}”.
        </p>
      )}
    </div>
  );
}

function StageStrip({ stages }: { stages: CorpusStageCount[] }) {
  return (
    <section aria-label="Corpus lifecycle counts" className="overflow-x-auto border-y border-border bg-surface">
      <div className="flex min-w-max divide-x divide-border">
        {stages.map((stage, index) => (
          <Link
            key={stage.relation}
            href={urlFor({ relation: stage.relation })}
            title={stage.description}
            className="group relative flex min-w-32 items-baseline gap-2 px-3 py-2.5 hover:bg-surface-2 focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-accent"
          >
            <span className="font-mono text-sm font-semibold tabular-nums text-text-primary">
              {formatInteger(stage.count)}
            </span>
            <span className="text-[0.68rem] text-text-muted group-hover:text-text-secondary">
              {stage.label}
            </span>
            {index < stages.length - 1 && <span className="sr-only">then</span>}
          </Link>
        ))}
      </div>
    </section>
  );
}

function CellInspector({
  dialogRef,
  selection,
  detail,
  onClose
}: {
  dialogRef: React.RefObject<HTMLDialogElement | null>;
  selection: CellSelection | null;
  detail: CorpusRelationDetail;
  onClose: () => void;
}) {
  const [copyState, setCopyState] = useState("Copy value");
  const primaryColumns = detail.columns
    .filter((column) => column.primaryKeyPosition > 0)
    .sort((left, right) => left.primaryKeyPosition - right.primaryKeyPosition);

  async function copyValue() {
    if (!selection) return;
    await navigator.clipboard.writeText(expandedValue(selection.value));
    setCopyState("Copied");
    window.setTimeout(() => setCopyState("Copy value"), 1600);
  }

  return (
    <dialog
      ref={dialogRef}
      onClose={onClose}
      onClick={(event) => {
        if (event.target === dialogRef.current) dialogRef.current?.close();
      }}
      className="m-0 ml-auto h-dvh max-h-none w-full max-w-[32rem] border-0 border-l border-border bg-surface p-0 text-text-primary shadow-[-4px_0_8px_rgba(0,0,0,0.12)] backdrop:bg-black/30 open:flex open:flex-col"
      aria-labelledby="cell-inspector-title"
    >
      {selection && (
        <>
          <header className="flex items-start justify-between gap-4 border-b border-border px-5 py-4">
            <div className="min-w-0">
              <p className="text-[0.68rem] font-semibold uppercase tracking-[0.1em] text-text-muted">Full cell value</p>
              <h2 id="cell-inspector-title" className="mt-1 truncate font-mono text-base font-semibold">
                {detail.name}.{selection.column}
              </h2>
            </div>
            <button
              type="button"
              onClick={() => dialogRef.current?.close()}
              className="flex h-11 w-11 shrink-0 items-center justify-center rounded-md text-xl text-text-secondary hover:bg-surface-2 hover:text-text-primary focus-visible:outline-2 focus-visible:outline-accent"
              aria-label="Close cell inspector"
            >
              ×
            </button>
          </header>

          <div className="flex-1 overflow-y-auto px-5 py-5">
            <div className="flex flex-wrap items-center gap-2 text-xs text-text-muted">
              <span className="rounded border border-border bg-surface-2 px-2 py-1 font-mono">
                {detail.columns.find((column) => column.name === selection.column)?.type ?? "ANY"}
              </span>
              {selection.value == null && <span>SQLite NULL</span>}
            </div>
            <pre className="mt-4 max-h-[45vh] overflow-auto whitespace-pre-wrap break-words rounded-md bg-surface-2 p-4 font-mono text-xs leading-5 text-text-primary">
              {expandedValue(selection.value)}
            </pre>
            <button
              type="button"
              onClick={copyValue}
              className="mt-3 min-h-11 rounded-md border border-border-2 px-3 py-2 text-xs font-medium text-text-primary hover:bg-surface-2 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
            >
              {copyState}
            </button>

            <section className="mt-7 border-t border-border pt-5">
              <h3 className="text-xs font-semibold text-text-primary">Row identity</h3>
              {primaryColumns.length > 0 ? (
                <dl className="mt-3 space-y-3">
                  {primaryColumns.map((column) => (
                    <div key={column.name}>
                      <dt className="font-mono text-[0.68rem] text-text-muted">{column.name}</dt>
                      <dd className="mt-0.5 break-all font-mono text-xs text-text-primary">
                        {expandedValue(selection.row[column.name] ?? null)}
                      </dd>
                    </div>
                  ))}
                </dl>
              ) : (
                <p className="mt-2 text-xs text-text-muted">This relation does not declare a primary key.</p>
              )}
            </section>

            <section className="mt-7 border-t border-border pt-5">
              <h3 className="text-xs font-semibold text-text-primary">Complete row</h3>
              <dl className="mt-3 divide-y divide-border">
                {detail.columns.map((column) => (
                  <div key={column.name} className="grid gap-1 py-3 sm:grid-cols-[9rem_minmax(0,1fr)]">
                    <dt className="font-mono text-[0.68rem] text-text-muted">{column.name}</dt>
                    <dd className="break-words font-mono text-xs leading-5 text-text-primary">
                      {compactValue(selection.row[column.name] ?? null)}
                    </dd>
                  </div>
                ))}
              </dl>
            </section>
          </div>
        </>
      )}
    </dialog>
  );
}

function RowsView({ page, detail, openCell }: {
  page: CorpusPage;
  detail: CorpusRelationDetail;
  openCell: (column: string, value: CorpusCellValue, row: Record<string, CorpusCellValue>) => void;
}) {
  const base = {
    relation: detail.name,
    view: "rows",
    q: page.query,
    pageSize: page.pageSize
  };

  return (
    <div className="min-w-0">
      <form action="/corpus" method="get" className="flex flex-wrap items-end gap-2 border-b border-border bg-surface px-3 py-3">
        <input type="hidden" name="relation" value={detail.name} />
        <input type="hidden" name="view" value="rows" />
        <label className="min-w-[14rem] flex-1">
          <span className="sr-only">Filter rows in {detail.name}</span>
          <input
            name="q"
            type="search"
            defaultValue={page.query}
            placeholder={`Filter ${detail.name} across ${detail.columns.length} columns`}
            className="min-h-11 w-full rounded-md border border-border-2 bg-bg px-3 text-sm text-text-primary placeholder:text-text-secondary focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20"
          />
        </label>
        <label>
          <span className="sr-only">Rows per page</span>
          <select
            name="pageSize"
            defaultValue={page.pageSize}
            className="min-h-11 rounded-md border border-border-2 bg-bg px-3 text-xs text-text-primary focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20"
          >
            {[10, 25, 50, 100].map((size) => <option key={size} value={size}>{size} rows</option>)}
          </select>
        </label>
        <button
          type="submit"
          className="min-h-11 rounded-md bg-accent px-4 py-2 text-xs font-medium text-white hover:brightness-95 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
        >
          Filter rows
        </button>
        {page.query && (
          <Link
            href={urlFor({ relation: detail.name, view: "rows", pageSize: page.pageSize })}
            className="flex min-h-11 items-center px-2 text-xs text-text-secondary hover:text-text-primary focus-visible:outline-2 focus-visible:outline-accent"
          >
            Clear filter
          </Link>
        )}
      </form>

      {page.rows.length === 0 ? (
        <div className="px-6 py-20 text-center">
          <h2 className="text-sm font-semibold text-text-primary">No rows match this filter</h2>
          <p className="mt-1 text-xs text-text-muted">Clear the filter to return to the complete relation.</p>
        </div>
      ) : (
        <div className="max-h-[62vh] overflow-auto" tabIndex={0} aria-label={`${detail.name} rows, scroll horizontally for more columns`}>
          <table className="w-max min-w-full border-collapse text-left text-xs">
            <caption className="sr-only">
              Page {page.page} of {page.pageCount} from {detail.name}, {page.totalRows} matching rows
            </caption>
            <thead className="sticky top-0 z-20 bg-surface shadow-[0_1px_0_var(--border)]">
              <tr>
                <th className="sticky left-0 z-30 w-12 bg-surface px-3 py-2 font-mono font-normal text-text-muted">#</th>
                {detail.columns.map((column) => {
                  const active = page.sortColumn === column.name;
                  const nextDirection = active && page.sortDirection === "asc" ? "desc" : "asc";
                  return (
                    <th key={column.name} className="min-w-40 border-l border-border px-3 py-2 align-bottom font-normal">
                      <Link
                        href={urlFor({ ...base, sort: column.name, direction: nextDirection, page: 1 })}
                        className="group block focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                        aria-label={`Sort by ${column.name} ${nextDirection}`}
                      >
                        <span className="flex items-center gap-1.5 font-mono font-semibold text-text-primary">
                          {column.name}
                          <span aria-hidden="true" className={active ? "text-accent" : "text-text-muted"}>
                            {active ? (page.sortDirection === "asc" ? "↑" : "↓") : "↕"}
                          </span>
                        </span>
                        <span className="mt-0.5 block font-mono text-[0.68rem] uppercase text-text-muted">
                          {column.type || "ANY"}{column.primaryKeyPosition ? " · pk" : ""}
                        </span>
                      </Link>
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {page.rows.map((row, rowIndex) => (
                <tr key={`${page.page}-${rowIndex}`} className="group hover:bg-surface-2/70">
                  <td className="sticky left-0 z-10 bg-surface px-3 py-2.5 text-right font-mono tabular-nums text-text-muted group-hover:bg-surface-2">
                    {(page.page - 1) * page.pageSize + rowIndex + 1}
                  </td>
                  {detail.columns.map((column) => {
                    const value = row[column.name] ?? null;
                    return (
                      <td key={column.name} className="max-w-[28rem] border-l border-border p-0">
                        <button
                          type="button"
                          onClick={() => openCell(column.name, value, row)}
                          className={`block min-h-10 w-full truncate px-3 py-2.5 text-left font-mono text-[0.72rem] focus-visible:relative focus-visible:z-10 focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-accent ${
                            value == null ? "text-text-muted" : "text-text-secondary hover:text-text-primary"
                          }`}
                          title={value == null ? "NULL" : String(value)}
                        >
                          {compactValue(value)}
                        </button>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <nav aria-label="Row pages" className="flex flex-wrap items-center justify-between gap-3 border-t border-border bg-surface px-3 py-3">
        <p className="text-xs text-text-muted">
          {formatInteger(page.totalRows)} {page.query ? "matching " : ""}rows · page {page.page} of {page.pageCount}
        </p>
        <div className="flex items-center gap-1">
          {page.hasPreviousPage ? (
            <Link href={urlFor({ ...base, sort: page.sortColumn ?? undefined, direction: page.sortDirection, page: page.page - 1 })} className="flex min-h-11 items-center rounded-md border border-border px-3 text-xs text-text-primary hover:bg-surface-2 focus-visible:outline-2 focus-visible:outline-accent">
              Previous
            </Link>
          ) : <span className="flex min-h-11 items-center rounded-md border border-border px-3 text-xs text-text-muted opacity-50">Previous</span>}
          {page.hasNextPage ? (
            <Link href={urlFor({ ...base, sort: page.sortColumn ?? undefined, direction: page.sortDirection, page: page.page + 1 })} className="flex min-h-11 items-center rounded-md border border-border px-3 text-xs text-text-primary hover:bg-surface-2 focus-visible:outline-2 focus-visible:outline-accent">
              Next
            </Link>
          ) : <span className="flex min-h-11 items-center rounded-md border border-border px-3 text-xs text-text-muted opacity-50">Next</span>}
        </div>
      </nav>
    </div>
  );
}

function SchemaView({ detail }: { detail: CorpusRelationDetail }) {
  return (
    <div className="space-y-8 p-4 sm:p-5">
      <section>
        <div className="mb-3 flex items-baseline justify-between gap-4">
          <h2 className="text-sm font-semibold text-text-primary">Columns</h2>
          <span className="font-mono text-xs text-text-muted">{detail.columns.length} fields</span>
        </div>
        <div className="overflow-x-auto rounded-md border border-border">
          <table className="w-full min-w-[44rem] border-collapse text-left text-xs">
            <thead className="bg-surface-2 text-text-muted">
              <tr><th className="px-3 py-2">#</th><th className="px-3 py-2">Column</th><th className="px-3 py-2">SQLite type</th><th className="px-3 py-2">Constraint</th><th className="px-3 py-2">Default</th></tr>
            </thead>
            <tbody className="divide-y divide-border">
              {detail.columns.map((column) => (
                <tr key={column.name}>
                  <td className="px-3 py-2.5 font-mono text-text-muted">{column.position}</td>
                  <td className="px-3 py-2.5 font-mono font-medium text-text-primary">{column.name}</td>
                  <td className="px-3 py-2.5 font-mono text-accent">{column.type}</td>
                  <td className="px-3 py-2.5 text-text-secondary">
                    {column.primaryKeyPosition ? `primary key ${column.primaryKeyPosition}` : column.notNull ? "not null" : "nullable"}
                  </td>
                  <td className="max-w-64 truncate px-3 py-2.5 font-mono text-text-muted">{column.defaultValue ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <div className="mb-3 flex items-baseline justify-between gap-4">
          <h2 className="text-sm font-semibold text-text-primary">Indexes</h2>
          <span className="font-mono text-xs text-text-muted">{detail.indexes.length}</span>
        </div>
        {detail.indexes.length === 0 ? (
          <p className="rounded-md bg-surface-2 px-4 py-6 text-xs text-text-muted">This relation declares no indexes.</p>
        ) : (
          <div className="divide-y divide-border rounded-md border border-border">
            {detail.indexes.map((index) => (
              <div key={index.name} className="grid gap-2 px-4 py-3 sm:grid-cols-[minmax(12rem,1fr)_2fr]">
                <div>
                  <p className="font-mono text-xs font-medium text-text-primary">{index.name}</p>
                  <p className="mt-1 text-[0.68rem] text-text-muted">
                    {index.unique ? "unique" : "non-unique"} · {index.origin}{index.partial ? " · partial" : ""}
                  </p>
                </div>
                <p className="break-words font-mono text-xs text-text-secondary">{index.columns.join(", ") || "expression index"}</p>
              </div>
            ))}
          </div>
        )}
      </section>

      <section>
        <h2 className="mb-3 text-sm font-semibold text-text-primary">Definition</h2>
        <pre className="max-h-80 overflow-auto whitespace-pre-wrap rounded-md bg-surface-2 p-4 font-mono text-xs leading-5 text-text-secondary">
          {detail.sql ?? "SQLite did not retain a SQL definition for this relation."}
        </pre>
      </section>
    </div>
  );
}

function LineageView({ detail }: { detail: CorpusRelationDetail }) {
  return (
    <div className="grid gap-8 p-4 sm:p-5 xl:grid-cols-2">
      <section>
        <div className="mb-3 flex items-baseline justify-between gap-4">
          <h2 className="text-sm font-semibold text-text-primary">Outbound references</h2>
          <span className="font-mono text-xs text-text-muted">{detail.outbound.length}</span>
        </div>
        {detail.outbound.length === 0 ? (
          <p className="rounded-md bg-surface-2 px-4 py-6 text-xs text-text-muted">No declared foreign keys leave this relation.</p>
        ) : (
          <div className="divide-y divide-border rounded-md border border-border">
            {detail.outbound.map((edge) => (
              <div key={`${edge.id}-${edge.sequence}`} className="px-4 py-3">
                <p className="font-mono text-xs text-text-secondary">
                  <span className="text-text-primary">{edge.sourceColumn}</span>
                  <span className="mx-2 text-text-muted">→</span>
                  <Link href={urlFor({ relation: edge.targetRelation, view: "lineage" })} className="text-accent hover:underline">
                    {edge.targetRelation}.{edge.targetColumn}
                  </Link>
                </p>
                <p className="mt-1 text-[0.68rem] text-text-muted">update {edge.onUpdate.toLowerCase()} · delete {edge.onDelete.toLowerCase()}</p>
              </div>
            ))}
          </div>
        )}
      </section>

      <section>
        <div className="mb-3 flex items-baseline justify-between gap-4">
          <h2 className="text-sm font-semibold text-text-primary">Inbound references</h2>
          <span className="font-mono text-xs text-text-muted">{detail.inbound.length}</span>
        </div>
        {detail.inbound.length === 0 ? (
          <p className="rounded-md bg-surface-2 px-4 py-6 text-xs text-text-muted">No declared foreign keys point to this relation.</p>
        ) : (
          <div className="divide-y divide-border rounded-md border border-border">
            {detail.inbound.map((edge) => (
              <div key={`${edge.sourceRelation}-${edge.sourceColumn}-${edge.targetColumn}`} className="px-4 py-3">
                <p className="font-mono text-xs text-text-secondary">
                  <Link href={urlFor({ relation: edge.sourceRelation, view: "lineage" })} className="text-accent hover:underline">
                    {edge.sourceRelation}.{edge.sourceColumn}
                  </Link>
                  <span className="mx-2 text-text-muted">→</span>
                  <span className="text-text-primary">{edge.targetColumn}</span>
                </p>
                <p className="mt-1 text-[0.68rem] text-text-muted">update {edge.onUpdate.toLowerCase()} · delete {edge.onDelete.toLowerCase()}</p>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

export function CorpusExplorer({
  relations,
  detail,
  page,
  view,
  stages,
  databaseUpdatedAt,
  safety
}: CorpusExplorerProps) {
  const [relationFilter, setRelationFilter] = useState("");
  const [selection, setSelection] = useState<CellSelection | null>(null);
  const inspectorRef = useRef<HTMLDialogElement>(null);
  const tables = relations.filter((relation) => relation.kind === "table").length;
  const views = relations.length - tables;

  function openCell(column: string, value: CorpusCellValue, row: Record<string, CorpusCellValue>) {
    setSelection({ column, value, row });
    window.requestAnimationFrame(() => inspectorRef.current?.showModal());
  }

  return (
    <div className="space-y-4">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="text-2xl font-bold tracking-tight text-text-primary">Alpha Corpus</h1>
            <span className="rounded border border-green/30 bg-green-bg px-2 py-1 text-[0.68rem] font-semibold text-green">
              Public · read only
            </span>
          </div>
          <p className="mt-1 max-w-3xl text-sm leading-relaxed text-text-secondary">
            The complete scientific ledger: generated and rejected material, schema, provenance, review,
            release membership, and actual training exposure remain distinct and inspectable.
          </p>
        </div>
        <dl className="flex flex-wrap gap-x-5 gap-y-2 text-xs">
          <div><dt className="text-text-muted">Relations</dt><dd className="mt-0.5 font-mono text-text-primary">{relations.length}</dd></div>
          <div><dt className="text-text-muted">Tables / views</dt><dd className="mt-0.5 font-mono text-text-primary">{tables} / {views}</dd></div>
          <div><dt className="text-text-muted">Ledger updated</dt><dd className="mt-0.5 font-mono text-text-primary">{formatTimestamp(databaseUpdatedAt)}</dd></div>
        </dl>
      </header>

      <StageStrip stages={stages} />

      <details className="rounded-lg border border-border bg-surface lg:hidden">
        <summary className="flex min-h-12 cursor-pointer items-center justify-between px-4 py-3 text-sm font-medium text-text-primary">
          Browse {relations.length} tables and views
        </summary>
        <div className="border-t border-border p-3">
          <label>
            <span className="sr-only">Search tables, views, and columns</span>
            <input
              type="search"
              value={relationFilter}
              onChange={(event) => setRelationFilter(event.target.value)}
              placeholder="Search schema"
              className="mb-4 min-h-11 w-full rounded-md border border-border-2 bg-bg px-3 text-sm text-text-primary placeholder:text-text-secondary focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20"
            />
          </label>
          <RelationList relations={relations} selected={detail.name} filter={relationFilter} />
        </div>
      </details>

      <div className="grid min-w-0 gap-4 lg:grid-cols-[15.5rem_minmax(0,1fr)]">
        <aside className="sticky top-6 hidden max-h-[calc(100vh-3rem)] self-start overflow-hidden rounded-lg border border-border bg-surface lg:flex lg:flex-col">
          <div className="border-b border-border p-3">
            <label>
              <span className="sr-only">Search tables, views, and columns</span>
              <input
                type="search"
                value={relationFilter}
                onChange={(event) => setRelationFilter(event.target.value)}
                placeholder={`Search ${relations.length} relations`}
                className="min-h-11 w-full rounded-md border border-border-2 bg-bg px-3 text-xs text-text-primary placeholder:text-text-secondary focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20"
              />
            </label>
          </div>
          <nav aria-label="Corpus relations" className="flex-1 overflow-y-auto p-2">
            <RelationList relations={relations} selected={detail.name} filter={relationFilter} />
          </nav>
          <div className="border-t border-border px-3 py-2.5 text-[0.68rem] text-text-muted">
            {safety.readOnly && safety.queryOnly ? "SQLite read-only + query-only" : "Read-only connection"}
          </div>
        </aside>

        <main className="min-w-0 overflow-hidden rounded-lg border border-border bg-surface">
          <header className="border-b border-border px-4 py-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <h2 className="break-all font-mono text-lg font-semibold text-text-primary">{detail.name}</h2>
                  <span className="rounded border border-border bg-surface-2 px-2 py-0.5 text-[0.68rem] text-text-secondary">{detail.kind}</span>
                </div>
                <p className="mt-1 text-xs text-text-muted">
                  {detail.columns.length} columns · {formatInteger(page.totalRows)} {page.query ? "matching " : ""}rows · no mutation surface
                </p>
              </div>
              <span className="rounded border border-border px-2 py-1 font-mono text-[0.68rem] text-text-muted">read only</span>
            </div>
          </header>

          <nav aria-label="Relation views" className="flex gap-1 border-b border-border px-3">
            {(["rows", "schema", "lineage"] as const).map((tab) => (
              <Link
                key={tab}
                href={urlFor({ relation: detail.name, view: tab })}
                aria-current={view === tab ? "page" : undefined}
                className={`border-b-2 px-3 py-3 text-xs font-medium capitalize focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-accent ${
                  view === tab ? "border-accent text-accent" : "border-transparent text-text-muted hover:text-text-primary"
                }`}
              >
                {tab}
              </Link>
            ))}
          </nav>

          {view === "rows" && <RowsView page={page} detail={detail} openCell={openCell} />}
          {view === "schema" && <SchemaView detail={detail} />}
          {view === "lineage" && <LineageView detail={detail} />}
        </main>
      </div>

      <CellInspector
        dialogRef={inspectorRef}
        selection={selection}
        detail={detail}
        onClose={() => setSelection(null)}
      />
    </div>
  );
}
