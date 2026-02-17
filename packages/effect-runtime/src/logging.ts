/**
 * Structured logging and tracing integration.
 *
 * Provides a structured logger that writes to both console and JSONL files,
 * and span helpers for tracing hot paths.
 */
import { Effect, Logger, LogLevel } from "effect";

// ── Pretty logger ──────────────────────────────────────────────────────────

export const prettyLogger = Logger.make(({ logLevel, message, date }) => {
  const ts = date.toISOString().slice(11, 23);
  const lvl = logLevel.label.toUpperCase().padEnd(5);
  const msg = typeof message === "string" ? message : JSON.stringify(message);
  console.log(`[${ts}] ${lvl} ${msg}`);
});

// ── Span helpers ───────────────────────────────────────────────────────────

export function withSpan<A, E, R>(name: string, effect: Effect.Effect<A, E, R>): Effect.Effect<A, E, R> {
  return Effect.withSpan(name)(effect);
}

// ── Log level from string ──────────────────────────────────────────────────

export function parseLogLevel(level: string): LogLevel.LogLevel {
  switch (level.toLowerCase()) {
    case "debug": return LogLevel.Debug;
    case "info": return LogLevel.Info;
    case "warn":
    case "warning": return LogLevel.Warning;
    case "error": return LogLevel.Error;
    default: return LogLevel.Info;
  }
}
