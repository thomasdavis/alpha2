"use client";

import { useEffect, useMemo, useState } from "react";
import type {
  HumanReviewFinding,
  HumanReviewPacket,
  HumanReviewResponse
} from "@alpha/corpus";
import {
  HUMAN_REVIEW_MISSING_CLARIFICATION,
  HUMAN_REVIEW_QUESTION_POLICIES,
  HUMAN_REVIEW_SCORE_ANCHORS,
  humanReviewDimensions,
  humanReviewOutcomes,
  humanReviewResponseErrors,
  parseHumanReviewPacketText
} from "@alpha/corpus/review-contract";

interface ReviewWorkspaceProps {
  sourcePacket: HumanReviewPacket;
  packetSha256: string;
  exportedAt: string;
}

interface VisibleMessage {
  role: string;
  content: string;
}

type SaveState = "loading" | "saved" | "memory-only";

function clonePacket(packet: HumanReviewPacket): HumanReviewPacket {
  return JSON.parse(JSON.stringify(packet)) as HumanReviewPacket;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function visibleMessages(candidate: unknown): VisibleMessage[] {
  if (!isRecord(candidate)) return [];
  const possibleItem = isRecord(candidate["item"]) ? candidate["item"] : candidate;
  const messages = possibleItem["messages"];
  if (!Array.isArray(messages)) return [];
  return messages.flatMap((message) => {
    if (!isRecord(message) || typeof message["role"] !== "string" || typeof message["content"] !== "string") {
      return [];
    }
    return [{ role: message["role"], content: message["content"] }];
  });
}

function packetMatchesSource(draft: HumanReviewPacket, source: HumanReviewPacket): boolean {
  if (draft.sessionId !== source.sessionId || draft.pass !== source.pass
    || draft.rubricSlug !== source.rubricSlug || draft.rubricVersion !== source.rubricVersion
    || draft.assignments.length !== source.assignments.length) return false;
  return draft.assignments.every((assignment, index) => {
    const original = source.assignments[index];
    return original?.assignmentId === assignment.assignmentId
      && original.candidateContentSha256 === assignment.candidateContentSha256;
  });
}

function downloadPacket(packet: HumanReviewPacket, suffix: "draft" | "completed") {
  const bytes = `${JSON.stringify(packet, null, 2)}\n`;
  const url = URL.createObjectURL(new Blob([bytes], { type: "application/json" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${packet.campaignSlug}-${packet.pass.toLowerCase()}-${packet.sessionId}-${suffix}.json`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
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

function SelectField({
  id,
  label,
  value,
  choices,
  onChange,
  required = true
}: {
  id: string;
  label: string;
  value: string | null;
  choices: readonly { value: string; label: string; description: string }[];
  onChange: (value: string | null) => void;
  required?: boolean;
}) {
  const selected = choices.find((choice) => choice.value === value);
  return (
    <div>
      <label htmlFor={id} className="text-xs font-semibold text-text-primary">{label}</label>
      <select
        id={id}
        value={value ?? ""}
        required={required}
        onChange={(event) => onChange(event.target.value || null)}
        className="mt-1.5 min-h-11 w-full rounded-md border border-border-2 bg-bg px-3 text-sm text-text-primary focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20"
      >
        <option value="">Select…</option>
        {choices.map((choice) => <option key={choice.value} value={choice.value}>{choice.label}</option>)}
      </select>
      {selected && <p className="mt-1.5 text-xs leading-5 text-text-muted">{selected.description}</p>}
    </div>
  );
}

function TextAreaField({
  id,
  label,
  value,
  onChange,
  hint,
  rows = 3
}: {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  hint?: string;
  rows?: number;
}) {
  return (
    <div>
      <label htmlFor={id} className="text-xs font-semibold text-text-primary">{label}</label>
      {hint && <p className="mt-1 text-xs leading-5 text-text-muted">{hint}</p>}
      <textarea
        id={id}
        rows={rows}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="mt-1.5 w-full resize-y rounded-md border border-border-2 bg-bg px-3 py-2.5 text-sm leading-6 text-text-primary focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20"
      />
    </div>
  );
}

function ScoreFields({
  packet,
  response,
  updateScore
}: {
  packet: HumanReviewPacket;
  response: HumanReviewResponse;
  updateScore: (dimension: string, score: number) => void;
}) {
  return (
    <div className="divide-y divide-border rounded-lg border border-border">
      {humanReviewDimensions(packet.pass).map((dimension) => (
        <fieldset key={dimension.key} className="grid gap-3 px-4 py-4 lg:grid-cols-[minmax(12rem,1fr)_auto] lg:items-center">
          <legend className="contents">
            <span>
              <span className="block text-sm font-medium text-text-primary">{dimension.label}</span>
              <span className="mt-0.5 block text-xs leading-5 text-text-muted">{dimension.description}</span>
            </span>
          </legend>
          <div className="flex flex-wrap gap-x-3 gap-y-2" aria-label={`${dimension.label} score`}>
            {HUMAN_REVIEW_SCORE_ANCHORS.map((anchor) => (
              <label key={anchor.value} className="flex min-h-11 cursor-pointer items-center gap-1.5 text-xs text-text-secondary">
                <input
                  type="radio"
                  name={`${packet.sessionId}-${dimension.key}`}
                  value={anchor.value}
                  checked={response.scores[dimension.key] === anchor.value}
                  onChange={() => updateScore(dimension.key, anchor.value)}
                  className="h-4 w-4 accent-[var(--accent)] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                />
                <span title={anchor.label}>{anchor.value}</span>
              </label>
            ))}
          </div>
        </fieldset>
      ))}
      <div className="flex flex-wrap gap-x-4 gap-y-1 bg-surface-2 px-4 py-2 text-[0.68rem] text-text-muted">
        {HUMAN_REVIEW_SCORE_ANCHORS.map((anchor) => (
          <span key={anchor.value}><strong className="text-text-secondary">{anchor.value}</strong> {anchor.label.toLocaleLowerCase()}</span>
        ))}
      </div>
    </div>
  );
}

function FindingEditor({
  finding,
  index,
  dimensions,
  onChange,
  onRemove
}: {
  finding: HumanReviewFinding;
  index: number;
  dimensions: readonly { key: string; label: string }[];
  onChange: (finding: HumanReviewFinding) => void;
  onRemove: () => void;
}) {
  return (
    <fieldset className="border-t border-border py-4 first:border-t-0">
      <legend className="sr-only">Finding {index + 1}</legend>
      <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_11rem_auto]">
        <label className="text-xs font-semibold text-text-primary">
          Dimension
          <select
            value={finding.dimension}
            onChange={(event) => onChange({ ...finding, dimension: event.target.value })}
            className="mt-1.5 min-h-11 w-full rounded-md border border-border-2 bg-bg px-3 text-sm font-normal text-text-primary focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20"
          >
            <option value="">Select…</option>
            {dimensions.map((dimension) => <option key={dimension.key} value={dimension.key}>{dimension.label}</option>)}
            <option value="other">Other</option>
          </select>
        </label>
        <label className="text-xs font-semibold text-text-primary">
          Severity
          <select
            value={finding.severity}
            onChange={(event) => onChange({ ...finding, severity: event.target.value as HumanReviewFinding["severity"] })}
            className="mt-1.5 min-h-11 w-full rounded-md border border-border-2 bg-bg px-3 text-sm font-normal text-text-primary focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20"
          >
            {(["observation", "minor", "major", "critical"] as const).map((severity) => (
              <option key={severity} value={severity}>{severity}</option>
            ))}
          </select>
        </label>
        <button
          type="button"
          onClick={onRemove}
          className="min-h-11 self-end rounded-md border border-border-2 px-3 py-2 text-xs font-medium text-text-secondary hover:bg-surface-2 hover:text-red focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
        >
          Remove
        </button>
      </div>
      <div className="mt-3 grid gap-3 md:grid-cols-2">
        <TextAreaField
          id={`finding-${index}-evidence`}
          label="Exact evidence"
          value={finding.evidence}
          onChange={(evidence) => onChange({ ...finding, evidence })}
          hint="Quote or precisely identify the model-visible wording."
          rows={2}
        />
        <TextAreaField
          id={`finding-${index}-recommendation`}
          label="Recommendation"
          value={finding.recommendation}
          onChange={(recommendation) => onChange({ ...finding, recommendation })}
          hint="State the bounded repair or reason to preserve the failure."
          rows={2}
        />
      </div>
    </fieldset>
  );
}

export function ReviewWorkspace({ sourcePacket, packetSha256, exportedAt }: ReviewWorkspaceProps) {
  const [draft, setDraft] = useState<HumanReviewPacket>(() => clonePacket(sourcePacket));
  const [activeIndex, setActiveIndex] = useState(0);
  const [saveState, setSaveState] = useState<SaveState>("loading");
  const [restored, setRestored] = useState(false);
  const storageKey = `alpha-corpus-review:${sourcePacket.sessionId}:${packetSha256}`;

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(storageKey);
      if (stored) {
        const parsed = parseHumanReviewPacketText(stored);
        if (packetMatchesSource(parsed, sourcePacket)) setDraft(parsed);
      }
      setSaveState("saved");
    } catch {
      setSaveState("memory-only");
    } finally {
      setRestored(true);
    }
  }, [sourcePacket, storageKey]);

  useEffect(() => {
    if (!restored || saveState === "memory-only") return;
    try {
      window.localStorage.setItem(storageKey, JSON.stringify(draft));
      setSaveState("saved");
    } catch {
      setSaveState("memory-only");
    }
  }, [draft, restored, saveState, storageKey]);

  const errors = useMemo(
    () => draft.assignments.map((assignment) => humanReviewResponseErrors(draft.pass, assignment.response, assignment.opaqueItemId)),
    [draft]
  );
  const completedCount = errors.filter((itemErrors) => itemErrors.length === 0).length;
  const allComplete = completedCount === draft.assignments.length;
  const activeAssignment = draft.assignments[activeIndex]!;
  const activeErrors = errors[activeIndex] ?? [];
  const messages = visibleMessages(activeAssignment.candidate);
  const dimensions = humanReviewDimensions(draft.pass);

  function updateResponse(update: (response: HumanReviewResponse) => HumanReviewResponse) {
    setDraft((current) => ({
      ...current,
      assignments: current.assignments.map((assignment, index) => index === activeIndex
        ? { ...assignment, response: update(assignment.response) }
        : assignment)
    }));
  }

  function resetLocalDraft() {
    if (!window.confirm("Discard every locally saved response in this review session?")) return;
    window.localStorage.removeItem(storageKey);
    setDraft(clonePacket(sourcePacket));
    setActiveIndex(0);
  }

  return (
    <div className="space-y-5">
      <header className="flex flex-wrap items-start justify-between gap-4 border-b border-border pb-5">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="text-2xl font-bold tracking-tight text-text-primary">D5 review · Pass {draft.pass}</h1>
            <span className="rounded border border-yellow/30 bg-yellow-bg px-2 py-1 text-[0.68rem] font-semibold text-yellow">
              {draft.pass === "A" ? "Blinded" : "Contract aware"}
            </span>
            <span className="rounded border border-border bg-surface-2 px-2 py-1 text-[0.68rem] font-medium text-text-secondary">
              {saveState === "loading" ? "Loading draft" : saveState === "saved" ? "Saved in this browser" : "Memory only"}
            </span>
          </div>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-text-secondary">
            {draft.pass === "A"
              ? "Judge only the conversation shown here. Do not inspect candidate lineage, family labels, structural status, or hidden contracts until this pass has been sealed and imported."
              : "Pass A is sealed. Judge the contract, blueprint, metadata, and realization separately."}
          </p>
          <p className="mt-2 break-all font-mono text-[0.68rem] text-text-muted">
            {draft.sessionId} · packet {packetSha256.slice(0, 12)}… · exported {formatTimestamp(exportedAt)}
          </p>
        </div>
        <div className="text-right">
          <p className="font-mono text-lg font-semibold tabular-nums text-text-primary">{completedCount}/{draft.assignments.length}</p>
          <p className="text-xs text-text-muted">items complete locally</p>
        </div>
      </header>

      {saveState === "memory-only" && (
        <p role="alert" className="rounded-md border border-yellow/40 bg-yellow-bg px-4 py-3 text-sm leading-6 text-yellow">
          Browser storage is unavailable. Your answers exist only in this tab; download drafts frequently.
        </p>
      )}

      <div className="grid min-w-0 gap-5 xl:grid-cols-[15rem_minmax(0,1fr)]">
        <aside className="self-start overflow-hidden rounded-lg border border-border bg-surface xl:sticky xl:top-6 xl:flex xl:max-h-[calc(100vh-3rem)] xl:flex-col">
          <div className="border-b border-border px-4 py-3">
            <p className="text-xs font-semibold text-text-primary">Assignments</p>
            <p className="mt-0.5 text-[0.68rem] text-text-muted">Family and status remain hidden.</p>
          </div>
          <nav aria-label="Review assignments" className="max-h-56 overflow-y-auto p-2 xl:max-h-none xl:flex-1">
            <ol className="grid grid-cols-3 gap-1 sm:grid-cols-6 xl:grid-cols-1">
              {draft.assignments.map((assignment, index) => {
                const complete = errors[index]?.length === 0;
                const active = index === activeIndex;
                return (
                  <li key={assignment.assignmentId}>
                    <button
                      type="button"
                      onClick={() => setActiveIndex(index)}
                      aria-current={active ? "step" : undefined}
                      className={`flex min-h-11 w-full items-center justify-between gap-2 rounded-md px-2.5 py-2 text-left text-xs focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent ${
                        active ? "bg-blue-bg font-semibold text-blue" : "text-text-secondary hover:bg-surface-2 hover:text-text-primary"
                      }`}
                    >
                      <span>{index + 1}</span>
                      <span className={complete ? "text-green" : "text-text-muted"} aria-label={complete ? "complete" : "incomplete"}>
                        {complete ? "●" : "○"}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ol>
          </nav>
        </aside>

        <main className="min-w-0 space-y-8">
          <section aria-labelledby="candidate-heading" className="overflow-hidden rounded-lg border border-border bg-surface">
            <header className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-4 py-3">
              <div>
                <p className="text-[0.68rem] font-semibold uppercase tracking-[0.1em] text-text-muted">Assignment {activeIndex + 1}</p>
                <h2 id="candidate-heading" className="mt-0.5 font-mono text-sm font-semibold text-text-primary">{activeAssignment.opaqueItemId}</h2>
              </div>
              <span className="rounded border border-border bg-surface-2 px-2 py-1 font-mono text-[0.68rem] text-text-muted">
                {isRecord(activeAssignment.candidate) && typeof activeAssignment.candidate["kind"] === "string"
                  ? activeAssignment.candidate["kind"] : "conversation"}
              </span>
            </header>
            <div className="divide-y divide-border">
              {messages.map((message, index) => (
                <article key={`${message.role}-${index}`} className={`grid gap-2 px-4 py-4 sm:grid-cols-[6rem_minmax(0,1fr)] ${message.role === "assistant" ? "bg-surface" : "bg-surface-2/60"}`}>
                  <p className="text-[0.68rem] font-semibold uppercase tracking-[0.08em] text-text-muted">{message.role}</p>
                  <p className="whitespace-pre-wrap text-sm leading-6 text-text-primary">{message.content}</p>
                </article>
              ))}
              {messages.length === 0 && <p className="px-4 py-8 text-sm text-text-muted">No model-visible messages were found.</p>}
            </div>
            {draft.pass === "B" && (
              <details className="border-t border-border px-4 py-3">
                <summary className="cursor-pointer text-xs font-semibold text-text-primary">Contract-aware candidate record</summary>
                <pre className="mt-3 max-h-96 overflow-auto whitespace-pre-wrap rounded-md bg-surface-2 p-3 font-mono text-xs leading-5 text-text-secondary">
                  {JSON.stringify(activeAssignment.candidate, null, 2)}
                </pre>
              </details>
            )}
          </section>

          <form className="space-y-8" onSubmit={(event) => event.preventDefault()}>
            {draft.pass === "A" && (
              <section aria-labelledby="summaries-heading">
                <div className="mb-3">
                  <h2 id="summaries-heading" className="text-base font-semibold text-text-primary">Independent reading</h2>
                  <p className="mt-1 text-xs leading-5 text-text-muted">Describe the exchange before selecting a disposition.</p>
                </div>
                <div className="grid gap-4 md:grid-cols-2">
                  <TextAreaField
                    id="summary-user-aim"
                    label="What is the user trying to do?"
                    value={activeAssignment.response.summaryUserAim}
                    onChange={(summaryUserAim) => updateResponse((response) => ({ ...response, summaryUserAim }))}
                    rows={3}
                  />
                  <TextAreaField
                    id="summary-assistant-move"
                    label="What intellectual or conversational move does the assistant make?"
                    value={activeAssignment.response.summaryAssistantMove}
                    onChange={(summaryAssistantMove) => updateResponse((response) => ({ ...response, summaryAssistantMove }))}
                    rows={3}
                  />
                </div>
              </section>
            )}

            <section aria-labelledby="scores-heading">
              <div className="mb-3">
                <h2 id="scores-heading" className="text-base font-semibold text-text-primary">Dimension scores</h2>
                <p className="mt-1 text-xs leading-5 text-text-muted">Score each dimension independently. A polished style cannot compensate for conceptual failure.</p>
              </div>
              <ScoreFields
                packet={draft}
                response={activeAssignment.response}
                updateScore={(dimension, score) => updateResponse((response) => ({
                  ...response,
                  scores: { ...response.scores, [dimension]: score }
                }))}
              />
            </section>

            <section aria-labelledby="policy-heading">
              <div className="mb-3">
                <h2 id="policy-heading" className="text-base font-semibold text-text-primary">Conversation policy</h2>
                <p className="mt-1 text-xs leading-5 text-text-muted">Distinguish a necessary clarification from an automatic question-shaped ending.</p>
              </div>
              <div className="grid gap-4 md:grid-cols-2">
                <SelectField
                  id="question-policy"
                  label="Follow-up question policy"
                  value={activeAssignment.response.questionPolicy}
                  choices={HUMAN_REVIEW_QUESTION_POLICIES}
                  onChange={(questionPolicy) => updateResponse((response) => ({ ...response, questionPolicy }))}
                />
                <SelectField
                  id="missing-clarification"
                  label="Did the response miss a required clarification?"
                  value={activeAssignment.response.missingClarification}
                  choices={HUMAN_REVIEW_MISSING_CLARIFICATION}
                  onChange={(missingClarification) => updateResponse((response) => ({ ...response, missingClarification }))}
                />
              </div>
            </section>

            <section aria-labelledby="findings-heading">
              <div className="flex flex-wrap items-start justify-between gap-3 border-b border-border pb-3">
                <div>
                  <h2 id="findings-heading" className="text-base font-semibold text-text-primary">Evidence findings</h2>
                  <p className="mt-1 text-xs leading-5 text-text-muted">Optional. Every finding needs exact evidence and a bounded recommendation.</p>
                </div>
                <button
                  type="button"
                  onClick={() => updateResponse((response) => ({
                    ...response,
                    findings: [...response.findings, { dimension: "", severity: "observation", evidence: "", recommendation: "" }]
                  }))}
                  className="min-h-11 rounded-md border border-border-2 px-3 py-2 text-xs font-semibold text-text-primary hover:bg-surface-2 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                >
                  Add finding
                </button>
              </div>
              {activeAssignment.response.findings.length === 0 ? (
                <p className="py-5 text-sm text-text-muted">No specific finding recorded.</p>
              ) : activeAssignment.response.findings.map((finding, index) => (
                <FindingEditor
                  key={index}
                  finding={finding}
                  index={index}
                  dimensions={dimensions}
                  onChange={(changed) => updateResponse((response) => ({
                    ...response,
                    findings: response.findings.map((current, currentIndex) => currentIndex === index ? changed : current)
                  }))}
                  onRemove={() => updateResponse((response) => ({
                    ...response,
                    findings: response.findings.filter((_, currentIndex) => currentIndex !== index)
                  }))}
                />
              ))}
            </section>

            <section aria-labelledby="disposition-heading">
              <div className="mb-3">
                <h2 id="disposition-heading" className="text-base font-semibold text-text-primary">Disposition</h2>
                <p className="mt-1 text-xs leading-5 text-text-muted">This is review evidence, not automatic acceptance into training.</p>
              </div>
              <div className="grid gap-4 md:grid-cols-2">
                <SelectField
                  id="outcome"
                  label="Outcome"
                  value={activeAssignment.response.outcome}
                  choices={humanReviewOutcomes(draft.pass)}
                  onChange={(outcome) => updateResponse((response) => ({ ...response, outcome }))}
                />
                <div>
                  <label htmlFor="confidence" className="text-xs font-semibold text-text-primary">Confidence</label>
                  <select
                    id="confidence"
                    value={activeAssignment.response.confidence ?? ""}
                    onChange={(event) => updateResponse((response) => ({
                      ...response,
                      confidence: event.target.value === "" ? null : Number(event.target.value)
                    }))}
                    className="mt-1.5 min-h-11 w-full rounded-md border border-border-2 bg-bg px-3 text-sm text-text-primary focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20"
                  >
                    <option value="">Select…</option>
                    {HUMAN_REVIEW_SCORE_ANCHORS.map((anchor) => <option key={anchor.value} value={anchor.value}>{anchor.value} · {anchor.label}</option>)}
                  </select>
                </div>
              </div>
              <div className="mt-4 space-y-4">
                <TextAreaField
                  id="rationale"
                  label="Rationale"
                  value={activeAssignment.response.rationale}
                  onChange={(rationale) => updateResponse((response) => ({ ...response, rationale }))}
                  hint="Explain the judgment in terms of this exact exchange."
                  rows={4}
                />
                <div className="grid gap-4 md:grid-cols-2">
                  <TextAreaField
                    id="uncertainty"
                    label="Uncertainty or admissible alternatives"
                    value={activeAssignment.response.uncertainty}
                    onChange={(uncertainty) => updateResponse((response) => ({ ...response, uncertainty }))}
                    rows={2}
                  />
                  <TextAreaField
                    id="expertise"
                    label="Expertise or authority needed"
                    value={activeAssignment.response.expertiseNeeded}
                    onChange={(expertiseNeeded) => updateResponse((response) => ({ ...response, expertiseNeeded }))}
                    rows={2}
                  />
                </div>
              </div>
            </section>

            <section aria-labelledby="validation-heading" className={`rounded-lg border px-4 py-4 ${activeErrors.length === 0 ? "border-green/30 bg-green-bg" : "border-border bg-surface"}`}>
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h2 id="validation-heading" className={`text-sm font-semibold ${activeErrors.length === 0 ? "text-green" : "text-text-primary"}`}>
                    {activeErrors.length === 0 ? "Assignment complete" : `${activeErrors.length} fields remain`}
                  </h2>
                  <p className={`mt-1 text-xs ${activeErrors.length === 0 ? "text-green" : "text-text-muted"}`}>
                    Validation uses the same rubric contract as the local SQLite importer.
                  </p>
                </div>
                {activeErrors.length > 0 && (
                  <details>
                    <summary className="cursor-pointer text-xs font-semibold text-accent">Show missing fields</summary>
                    <ul className="mt-2 max-w-xl list-disc space-y-1 pl-5 text-xs leading-5 text-text-secondary">
                      {activeErrors.map((error) => <li key={error}>{error}</li>)}
                    </ul>
                  </details>
                )}
              </div>
            </section>

            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border pt-5">
              <button
                type="button"
                disabled={activeIndex === 0}
                onClick={() => setActiveIndex((index) => Math.max(0, index - 1))}
                className="min-h-11 rounded-md border border-border-2 px-4 py-2 text-sm font-medium text-text-primary hover:bg-surface-2 disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
              >
                Previous
              </button>
              <button
                type="button"
                disabled={activeIndex === draft.assignments.length - 1}
                onClick={() => setActiveIndex((index) => Math.min(draft.assignments.length - 1, index + 1))}
                className="min-h-11 rounded-md bg-accent px-4 py-2 text-sm font-semibold text-white hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
              >
                Next assignment
              </button>
            </div>
          </form>
        </main>
      </div>

      <footer className="flex flex-wrap items-center justify-between gap-4 border-t border-border pt-6">
        <div>
          <p className="text-sm font-semibold text-text-primary">Export review evidence</p>
          <p className="mt-1 max-w-2xl text-xs leading-5 text-text-muted">
            Downloaded files contain your responses. Nothing has been submitted to the public server or accepted into a dataset.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={resetLocalDraft}
            className="min-h-11 rounded-md px-3 py-2 text-xs font-medium text-text-muted hover:bg-surface-2 hover:text-red focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          >
            Clear local draft
          </button>
          <button
            type="button"
            onClick={() => downloadPacket(draft, "draft")}
            className="min-h-11 rounded-md border border-border-2 px-3 py-2 text-xs font-semibold text-text-primary hover:bg-surface-2 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          >
            Download draft
          </button>
          <button
            type="button"
            disabled={!allComplete}
            onClick={() => downloadPacket(draft, "completed")}
            className="min-h-11 rounded-md bg-accent px-3 py-2 text-xs font-semibold text-white hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          >
            Download completed packet
          </button>
        </div>
      </footer>
    </div>
  );
}
