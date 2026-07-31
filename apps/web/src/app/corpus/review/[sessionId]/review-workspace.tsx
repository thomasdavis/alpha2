"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type {
  HumanReviewFinding,
  HumanReviewPacket,
  HumanReviewResponse,
  HumanReviewScore,
  HumanReviewSessionResponse
} from "@alpha/corpus";
import {
  HUMAN_REVIEW_ANSWERED_BEFORE_UNNECESSARY_QUESTION,
  HUMAN_REVIEW_COMPETENCIES,
  HUMAN_REVIEW_FATIGUE_LEVELS,
  HUMAN_REVIEW_FIRST_SENTENCE_ENGAGEMENT,
  HUMAN_REVIEW_INTERRUPTION_STATUSES,
  HUMAN_REVIEW_MISSING_CLARIFICATION,
  HUMAN_REVIEW_QUESTION_POLICIES,
  HUMAN_REVIEW_SCORE_ANCHORS,
  humanReviewDimensions,
  humanReviewOutcomes,
  humanReviewPacketMatchesEnvelope,
  humanReviewResponseErrors,
  humanReviewSessionResponseErrors,
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

function firstIncompleteIndex(packet: HumanReviewPacket): number {
  const index = packet.assignments.findIndex(
    (assignment) => humanReviewResponseErrors(packet.pass, assignment.response, assignment.opaqueItemId).length > 0
  );
  return index < 0 ? 0 : index;
}

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
  updateScore,
  updateEvidence
}: {
  packet: HumanReviewPacket;
  response: HumanReviewResponse;
  updateScore: (dimension: string, score: Exclude<HumanReviewScore, null>) => void;
  updateEvidence: (dimension: string, evidence: string) => void;
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
                  aria-label={`${dimension.label}: ${anchor.value}, ${anchor.label}`}
                  checked={response.scores[dimension.key] === anchor.value}
                  onChange={() => updateScore(dimension.key, anchor.value)}
                  className="h-4 w-4 accent-[var(--accent)] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                />
                <span title={anchor.label}>{anchor.value}</span>
              </label>
            ))}
            {(["not_applicable", "uncertain"] as const).map((state) => (
              <label key={state} className="flex min-h-11 cursor-pointer items-center gap-1.5 text-xs text-text-secondary">
                <input
                  type="radio"
                  name={`${packet.sessionId}-${dimension.key}`}
                  value={state}
                  aria-label={`${dimension.label}: ${state === "not_applicable" ? "Not applicable" : "Uncertain"}`}
                  checked={response.scores[dimension.key] === state}
                  onChange={() => updateScore(dimension.key, state)}
                  className="h-4 w-4 accent-[var(--accent)] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                />
                <span>{state === "not_applicable" ? "N/A" : "?"}</span>
              </label>
            ))}
          </div>
          <div className="lg:col-span-2">
            <TextAreaField
              id={`dimension-evidence-${dimension.key}`}
              label={`Evidence for ${dimension.label}`}
              value={response.dimensionEvidence[dimension.key] ?? ""}
              onChange={(evidence) => updateEvidence(dimension.key, evidence)}
              hint="Give one concise sentence grounded in the model-visible exchange."
              rows={2}
            />
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
          id={`finding-${index}-why-it-matters`}
          label="Why it matters"
          value={finding.whyItMatters}
          onChange={(whyItMatters) => onChange({ ...finding, whyItMatters })}
          hint="Explain the conceptual, conversational, or scientific consequence."
          rows={2}
        />
        <TextAreaField
          id={`finding-${index}-recommendation`}
          label="Smallest plausible repair"
          value={finding.recommendation}
          onChange={(recommendation) => onChange({ ...finding, recommendation })}
          hint="State the narrowest change that would address the finding."
          rows={2}
        />
        <TextAreaField
          id={`finding-${index}-preserve`}
          label="What must be preserved"
          value={finding.preserve}
          onChange={(preserve) => onChange({ ...finding, preserve })}
          hint="Name the useful behavior, distinction, or evidence the repair must not erase."
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
  const [restoreNotice, setRestoreNotice] = useState<string | null>(null);
  const candidateHeadingRef = useRef<HTMLHeadingElement>(null);
  const storageKey = `alpha-corpus-review:${sourcePacket.sessionId}:${packetSha256}`;
  const positionStorageKey = `${storageKey}:active-item`;

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(storageKey);
      if (stored) {
        try {
          const parsed = parseHumanReviewPacketText(stored);
          if (humanReviewPacketMatchesEnvelope(parsed, sourcePacket)) {
            setDraft(parsed);
            const savedOpaqueItemId = window.localStorage.getItem(positionStorageKey);
            const savedIndex = savedOpaqueItemId === null
              ? -1
              : parsed.assignments.findIndex((assignment) => assignment.opaqueItemId === savedOpaqueItemId);
            setActiveIndex(savedIndex >= 0
              ? savedIndex
              : firstIncompleteIndex(parsed));
          } else {
            window.localStorage.removeItem(storageKey);
            window.localStorage.removeItem(positionStorageKey);
            setRestoreNotice("An incompatible local draft was discarded. The verified exported packet remains unchanged.");
          }
        } catch {
          window.localStorage.removeItem(storageKey);
          window.localStorage.removeItem(positionStorageKey);
          setRestoreNotice("An unreadable local draft was discarded. The verified exported packet remains unchanged.");
        }
      }
      setSaveState("saved");
    } catch {
      setSaveState("memory-only");
    } finally {
      setRestored(true);
    }
  }, [positionStorageKey, sourcePacket, storageKey]);

  useEffect(() => {
    if (!restored || draft.sessionResponse.startedAt.length > 0) return;
    setDraft((current) => ({
      ...current,
      sessionResponse: { ...current.sessionResponse, startedAt: new Date().toISOString() }
    }));
  }, [draft.sessionResponse.startedAt, restored]);

  useEffect(() => {
    if (!restored || saveState === "memory-only") return;
    try {
      window.localStorage.setItem(storageKey, JSON.stringify(draft));
      setSaveState("saved");
    } catch {
      setSaveState("memory-only");
    }
  }, [draft, restored, saveState, storageKey]);

  useEffect(() => {
    if (!restored || saveState === "memory-only") return;
    try {
      const activeOpaqueItemId = draft.assignments[activeIndex]?.opaqueItemId;
      if (activeOpaqueItemId) window.localStorage.setItem(positionStorageKey, activeOpaqueItemId);
    } catch {
      setSaveState("memory-only");
    }
  }, [activeIndex, draft.assignments, positionStorageKey, restored, saveState]);

  const errors = useMemo(
    () => draft.assignments.map((assignment) => humanReviewResponseErrors(draft.pass, assignment.response, assignment.opaqueItemId)),
    [draft]
  );
  const completedCount = errors.filter((itemErrors) => itemErrors.length === 0).length;
  const allAssignmentsComplete = completedCount === draft.assignments.length;
  const sessionErrors = humanReviewSessionResponseErrors(draft.sessionResponse);
  const readyToFinish = allAssignmentsComplete && sessionErrors.length === 0;
  const activeAssignment = draft.assignments[activeIndex]!;
  const activeErrors = errors[activeIndex] ?? [];
  const messages = visibleMessages(activeAssignment.candidate);
  const dimensions = humanReviewDimensions(draft.pass);
  const firstIncomplete = errors.findIndex((itemErrors) => itemErrors.length > 0);
  const nextIncomplete = errors.findIndex((itemErrors, index) => index > activeIndex && itemErrors.length > 0);
  const wrappedIncomplete = nextIncomplete >= 0
    ? nextIncomplete
    : errors.findIndex((itemErrors, index) => index < activeIndex && itemErrors.length > 0);

  function updateResponse(update: (response: HumanReviewResponse) => HumanReviewResponse) {
    setDraft((current) => ({
      ...current,
      assignments: current.assignments.map((assignment, index) => index === activeIndex
        ? { ...assignment, response: update(assignment.response) }
        : assignment)
    }));
  }

  function updateSessionResponse(update: (response: HumanReviewSessionResponse) => HumanReviewSessionResponse) {
    setDraft((current) => ({ ...current, sessionResponse: update(current.sessionResponse) }));
  }

  function finishAndDownload() {
    const completed = clonePacket(draft);
    completed.sessionResponse.endedAt = new Date().toISOString();
    if (humanReviewSessionResponseErrors(completed.sessionResponse, { requireEndedAt: true }).length > 0) return;
    setDraft(completed);
    downloadPacket(completed, "completed");
  }

  function toggleCompetence(competence: string, selected: boolean) {
    updateSessionResponse((response) => ({
      ...response,
      declaredCompetencies: selected
        ? [...response.declaredCompetencies, competence]
        : response.declaredCompetencies.filter((value) => value !== competence)
    }));
  }

  function resetLocalDraft() {
    if (!window.confirm("Discard every locally saved response in this review session?")) return;
    window.localStorage.removeItem(storageKey);
    window.localStorage.removeItem(positionStorageKey);
    setDraft(clonePacket(sourcePacket));
    setActiveIndex(0);
    setRestoreNotice(null);
  }

  function goToAssignment(index: number) {
    setActiveIndex(index);
    window.requestAnimationFrame(() => {
      candidateHeadingRef.current?.focus({ preventScroll: true });
      candidateHeadingRef.current?.scrollIntoView({ block: "start" });
    });
  }

  return (
    <div className="space-y-5">
      <header className="flex flex-wrap items-start justify-between gap-4 border-b border-border pb-5">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="text-2xl font-bold tracking-tight text-text-primary">D5 review · Pass {draft.pass}</h1>
            <span className="rounded border border-yellow/30 bg-yellow-bg px-2 py-1 text-[0.68rem] font-semibold text-text-primary">
              {draft.pass === "A" ? "Blinded" : "Contract aware"}
            </span>
            <span aria-live="polite" className="rounded border border-border bg-surface-2 px-2 py-1 text-[0.68rem] font-medium text-text-secondary">
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
        <div className="min-w-28 text-right" aria-live="polite">
          <p className="font-mono text-lg font-semibold tabular-nums text-text-primary">{completedCount}/{draft.assignments.length}</p>
          <p className="text-xs text-text-muted">items complete locally</p>
          <progress
            className="mt-2 h-1.5 w-full accent-[var(--accent)]"
            value={completedCount}
            max={draft.assignments.length}
            aria-label={`${completedCount} of ${draft.assignments.length} assignments complete locally`}
          />
        </div>
      </header>

      {restoreNotice && (
        <p role="status" className="rounded-md border border-border bg-surface-2 px-4 py-3 text-sm leading-6 text-text-primary">
          {restoreNotice}
        </p>
      )}

      {saveState === "memory-only" && (
        <p role="alert" className="rounded-md border border-yellow/40 bg-yellow-bg px-4 py-3 text-sm leading-6 text-text-primary">
          Browser storage is unavailable. Your answers exist only in this tab; download drafts frequently.
        </p>
      )}

      <section aria-labelledby="session-declaration-heading" className="overflow-hidden rounded-lg border border-border bg-surface">
        <header className="border-b border-border px-4 py-3">
          <h2 id="session-declaration-heading" className="text-sm font-semibold text-text-primary">Reviewer and session declaration</h2>
          <p className="mt-1 text-xs leading-5 text-text-muted">
            Record the competence and conditions under which these judgments were made. This is public provenance, not a credential certification.
          </p>
        </header>
        <div className="space-y-5 px-4 py-4">
          <fieldset>
            <legend className="text-xs font-semibold text-text-primary">Relevant competence</legend>
            <p className="mt-1 text-xs leading-5 text-text-muted">Select every area you can responsibly judge in this session.</p>
            <div className="mt-2 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {HUMAN_REVIEW_COMPETENCIES.map((choice) => (
                <label key={choice.value} className="flex min-h-11 cursor-pointer items-start gap-2 rounded-md border border-border px-3 py-2 text-xs text-text-primary hover:bg-surface-2">
                  <input
                    type="checkbox"
                    checked={draft.sessionResponse.declaredCompetencies.includes(choice.value)}
                    onChange={(event) => toggleCompetence(choice.value, event.target.checked)}
                    className="mt-0.5 h-4 w-4 accent-[var(--accent)] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                  />
                  <span><strong className="font-semibold">{choice.label}</strong><span className="mt-0.5 block leading-5 text-text-muted">{choice.description}</span></span>
                </label>
              ))}
            </div>
          </fieldset>

          <TextAreaField
            id="session-competence-note"
            label="Competence scope or limitations"
            value={draft.sessionResponse.competenceNote}
            onChange={(competenceNote) => updateSessionResponse((response) => ({ ...response, competenceNote }))}
            hint="Required when Other is selected; otherwise use it to state meaningful limits."
            rows={2}
          />

          <div className="grid gap-4 md:grid-cols-2">
            <SelectField
              id="session-interruption-status"
              label="Interruption status"
              value={draft.sessionResponse.interruptionStatus}
              choices={HUMAN_REVIEW_INTERRUPTION_STATUSES}
              onChange={(interruptionStatus) => updateSessionResponse((response) => ({ ...response, interruptionStatus }))}
            />
            <SelectField
              id="session-fatigue-level"
              label="Fatigue level"
              value={draft.sessionResponse.fatigueLevel}
              choices={HUMAN_REVIEW_FATIGUE_LEVELS}
              onChange={(fatigueLevel) => updateSessionResponse((response) => ({ ...response, fatigueLevel }))}
            />
          </div>

          <TextAreaField
            id="session-conditions-note"
            label="Interruption, fatigue, or review-condition note"
            value={draft.sessionResponse.conditionsNote}
            onChange={(conditionsNote) => updateSessionResponse((response) => ({ ...response, conditionsNote }))}
            hint="Optional. Record anything that may affect interpretation of this session."
            rows={2}
          />

          <div className="flex flex-wrap items-center justify-between gap-3 rounded-md bg-surface-2 px-3 py-2 text-xs">
            <span className="text-text-muted">Session started locally</span>
            <span className="font-mono text-text-secondary">{draft.sessionResponse.startedAt ? formatTimestamp(draft.sessionResponse.startedAt) : "Not recorded"}</span>
          </div>

          {sessionErrors.length > 0 && (
            <details>
              <summary className="cursor-pointer text-xs font-semibold text-accent">{sessionErrors.length} session declaration fields remain</summary>
              <ul className="mt-2 list-disc space-y-1 pl-5 text-xs leading-5 text-text-secondary">
                {sessionErrors.map((error) => <li key={error}>{error}</li>)}
              </ul>
            </details>
          )}
        </div>
      </section>

      <div className="grid min-w-0 gap-5 xl:grid-cols-[15rem_minmax(0,1fr)]">
        <section aria-labelledby="assignment-nav-heading" className="self-start overflow-hidden rounded-lg border border-border bg-surface xl:sticky xl:top-6 xl:flex xl:max-h-[calc(100vh-3rem)] xl:flex-col">
          <div className="flex flex-wrap items-start justify-between gap-2 border-b border-border px-4 py-3 xl:block">
            <div>
              <h2 id="assignment-nav-heading" className="text-xs font-semibold text-text-primary">Assignments</h2>
              <p className="mt-0.5 text-[0.68rem] text-text-muted">Family and status remain hidden.</p>
            </div>
            {firstIncomplete >= 0 && firstIncomplete !== activeIndex && (
              <button
                type="button"
                onClick={() => goToAssignment(firstIncomplete)}
                className="min-h-11 rounded-md border border-border-2 px-2.5 py-2 text-xs font-semibold text-text-primary hover:bg-surface-2 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent xl:mt-3 xl:w-full"
              >
                Resume first incomplete
              </button>
            )}
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
                      onClick={() => goToAssignment(index)}
                      aria-current={active ? "step" : undefined}
                      className={`flex min-h-11 w-full items-center justify-between gap-2 rounded-md px-2.5 py-2 text-left text-xs focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent ${
                        active ? "bg-blue-bg font-semibold text-text-primary" : "text-text-secondary hover:bg-surface-2 hover:text-text-primary"
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
        </section>

        <div className="min-w-0 space-y-8">
          <section aria-labelledby="candidate-heading" className="overflow-hidden rounded-lg border border-border bg-surface">
            <header className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-4 py-3">
              <div>
                <p className="text-[0.68rem] font-semibold uppercase tracking-[0.1em] text-text-muted">Assignment {activeIndex + 1}</p>
                <h2
                  ref={candidateHeadingRef}
                  id="candidate-heading"
                  tabIndex={-1}
                  className="mt-0.5 break-all font-mono text-sm font-semibold text-text-primary focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                >
                  {activeAssignment.opaqueItemId}
                </h2>
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
                  <SelectField
                    id="first-sentence-engagement"
                    label="Did the first assistant sentence directly engage the user?"
                    value={activeAssignment.response.firstSentenceEngagement}
                    choices={HUMAN_REVIEW_FIRST_SENTENCE_ENGAGEMENT}
                    onChange={(firstSentenceEngagement) => updateResponse((response) => ({
                      ...response,
                      firstSentenceEngagement
                    }))}
                  />
                  <SelectField
                    id="answered-before-unnecessary-question"
                    label="Did the assistant answer before asking anything unnecessary?"
                    value={activeAssignment.response.answeredBeforeUnnecessaryQuestion}
                    choices={HUMAN_REVIEW_ANSWERED_BEFORE_UNNECESSARY_QUESTION}
                    onChange={(answeredBeforeUnnecessaryQuestion) => updateResponse((response) => ({
                      ...response,
                      answeredBeforeUnnecessaryQuestion
                    }))}
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
                updateEvidence={(dimension, evidence) => updateResponse((response) => ({
                  ...response,
                  dimensionEvidence: { ...response.dimensionEvidence, [dimension]: evidence }
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
                    findings: [...response.findings, {
                      dimension: "",
                      severity: "observation",
                      evidence: "",
                      whyItMatters: "",
                      recommendation: "",
                      preserve: ""
                    }]
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
                    <h2 id="validation-heading" className="text-sm font-semibold text-text-primary">
                      {activeErrors.length === 0 ? "Assignment complete" : `${activeErrors.length} fields remain`}
                    </h2>
                    <p className={`mt-1 text-xs ${activeErrors.length === 0 ? "text-text-primary" : "text-text-muted"}`}>
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
                onClick={() => goToAssignment(Math.max(0, activeIndex - 1))}
                className="min-h-11 rounded-md border border-border-2 px-4 py-2 text-sm font-medium text-text-primary hover:bg-surface-2 disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
              >
                Previous
              </button>
              <div className="flex flex-wrap justify-end gap-2">
                {wrappedIncomplete >= 0 && wrappedIncomplete !== activeIndex && (
                  <button
                    type="button"
                    onClick={() => goToAssignment(wrappedIncomplete)}
                    className="min-h-11 rounded-md border border-border-2 px-4 py-2 text-sm font-medium text-text-primary hover:bg-surface-2 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                  >
                    Next incomplete
                  </button>
                )}
                <button
                  type="button"
                  disabled={activeIndex === draft.assignments.length - 1}
                  onClick={() => goToAssignment(Math.min(draft.assignments.length - 1, activeIndex + 1))}
                  className="min-h-11 rounded-md bg-accent px-4 py-2 text-sm font-semibold text-bg hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                >
                  Next assignment
                </button>
              </div>
            </div>
          </form>
        </div>
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
            disabled={!readyToFinish}
            onClick={finishAndDownload}
            className="min-h-11 rounded-md bg-accent px-3 py-2 text-xs font-semibold text-bg hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          >
            Download completed packet
          </button>
        </div>
      </footer>
    </div>
  );
}
