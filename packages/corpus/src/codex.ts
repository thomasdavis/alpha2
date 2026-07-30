import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { canonicalJson } from "./hash.js";
import { writeAtomic } from "./storage.js";
import type { JsonValue, StructuredCallRequest, StructuredCallResult } from "./types.js";

interface CapturedProcess {
  exitCode: number;
  stdout: Buffer;
  stderr: Buffer;
}

function runProcess(
  executable: string,
  args: string[],
  cwd: string,
  stdin: string,
  timeoutMs: number
): Promise<CapturedProcess> {
  return new Promise((resolve, reject) => {
    const child = spawn(executable, args, {
      cwd,
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env, NO_COLOR: "1" }
    });
    const stdout: Buffer[] = [];
    const stderr: Buffer[] = [];
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGTERM");
      setTimeout(() => child.kill("SIGKILL"), 5_000).unref();
    }, timeoutMs);
    child.stdout.on("data", (chunk: Buffer) => stdout.push(Buffer.from(chunk)));
    child.stderr.on("data", (chunk: Buffer) => stderr.push(Buffer.from(chunk)));
    child.on("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      const capturedStderr = Buffer.concat(stderr);
      resolve({
        exitCode: timedOut ? 124 : (code ?? 1),
        stdout: Buffer.concat(stdout),
        stderr: timedOut
          ? Buffer.concat([capturedStderr, Buffer.from("\nalpha-corpus: call timed out\n")])
          : capturedStderr
      });
    });
    child.stdin.end(stdin);
  });
}

function readTokenNumber(record: Record<string, unknown>, names: string[]): number | null {
  for (const name of names) {
    const value = record[name];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
}

function collectUsage(value: unknown): { inputTokens: number | null; cachedInputTokens: number | null; outputTokens: number | null } | null {
  if (typeof value !== "object" || value === null) return null;
  if (Array.isArray(value)) {
    for (let index = value.length - 1; index >= 0; index--) {
      const found = collectUsage(value[index]);
      if (found) return found;
    }
    return null;
  }
  const record = value as Record<string, unknown>;
  const inputTokens = readTokenNumber(record, ["input_tokens", "inputTokens"]);
  const cachedInputTokens = readTokenNumber(record, ["cached_input_tokens", "cachedInputTokens"]);
  const outputTokens = readTokenNumber(record, ["output_tokens", "outputTokens"]);
  if (inputTokens !== null || cachedInputTokens !== null || outputTokens !== null) {
    return { inputTokens, cachedInputTokens, outputTokens };
  }
  for (const nested of Object.values(record).reverse()) {
    const found = collectUsage(nested);
    if (found) return found;
  }
  return null;
}

export function parseCodexUsage(stdout: Buffer): StructuredCallResult["usage"] {
  let latest: StructuredCallResult["usage"] = {
    inputTokens: null,
    cachedInputTokens: null,
    outputTokens: null
  };
  for (const line of stdout.toString("utf8").split(/\r?\n/)) {
    if (line.trim().length === 0) continue;
    try {
      const found = collectUsage(JSON.parse(line));
      if (found) latest = found;
    } catch {
      // The Codex event stream is expected to be JSONL. Preserve malformed lines raw;
      // usage is observability metadata and never model-visible corpus content.
    }
  }
  return latest;
}

export function loadRecoverableStructuredCall<T>(
  taskId: string,
  callRoot: string,
  expectedPrompt: string,
  expectedSchema: JsonValue
): (StructuredCallResult & { parsed: T | null }) | null {
  if (!existsSync(callRoot)) return null;
  const expectedSchemaText = canonicalJson(expectedSchema);
  const candidates = readdirSync(callRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && entry.name.startsWith(`${taskId}-`))
    .map((entry) => join(callRoot, entry.name))
    .sort((left, right) => statSync(right).mtimeMs - statSync(left).mtimeMs);
  for (const callDirectory of candidates) {
    const promptPath = join(callDirectory, "prompt.txt");
    const schemaPath = join(callDirectory, "output-schema.json");
    const stdoutPath = join(callDirectory, "stdout.jsonl");
    const stderrPath = join(callDirectory, "stderr.txt");
    const lastMessagePath = join(callDirectory, "last-message.json");
    const commandPath = join(callDirectory, "command.json");
    if (![promptPath, schemaPath, stdoutPath, stderrPath, lastMessagePath, commandPath].every(existsSync)) continue;
    if (readFileSync(promptPath, "utf8") !== expectedPrompt) continue;
    if (readFileSync(schemaPath, "utf8") !== expectedSchemaText) continue;
    const stdout = readFileSync(stdoutPath);
    const stderr = readFileSync(stderrPath);
    const lastMessage = readFileSync(lastMessagePath);
    let parsed: T | null = null;
    let commandArgs: string[] = [];
    try {
      parsed = JSON.parse(lastMessage.toString("utf8")) as T;
      const command = JSON.parse(readFileSync(commandPath, "utf8")) as string[];
      commandArgs = command[0] === "codex" ? command.slice(1) : command;
    } catch {
      continue;
    }
    return {
      startedAt: statSync(promptPath).mtime.toISOString(),
      completedAt: statSync(lastMessagePath).mtime.toISOString(),
      exitCode: 0,
      stdout,
      stderr,
      lastMessage,
      parsed,
      usage: parseCodexUsage(stdout),
      callDirectory,
      commandArgs
    };
  }
  return null;
}

export async function callCodexStructured<T>(
  request: StructuredCallRequest,
  timeoutMs = 240_000
): Promise<StructuredCallResult & { parsed: T | null }> {
  const startedAt = new Date().toISOString();
  const callDirectory = join(request.callRoot, `${request.taskId}-${Date.now()}-${randomUUID()}`);
  mkdirSync(callDirectory, { recursive: true });
  const schemaPath = join(callDirectory, "output-schema.json");
  const promptPath = join(callDirectory, "prompt.txt");
  const lastMessagePath = join(callDirectory, "last-message.json");
  writeAtomic(schemaPath, canonicalJson(request.schema as JsonValue));
  writeAtomic(promptPath, request.prompt);

  const args = [
    "exec",
    "--ephemeral",
    "--ignore-user-config",
    "--ignore-rules",
    "--skip-git-repo-check",
    "-s", "read-only",
    "-C", request.repoRoot,
    "-m", request.model,
    "-c", "model_reasoning_effort=medium",
    "--output-schema", schemaPath,
    "--json",
    "-o", lastMessagePath,
    "-"
  ];
  writeAtomic(join(callDirectory, "command.json"), canonicalJson(["codex", ...args] as JsonValue));
  const captured = await runProcess("codex", args, request.repoRoot, request.prompt, timeoutMs);
  writeAtomic(join(callDirectory, "stdout.jsonl"), captured.stdout);
  writeAtomic(join(callDirectory, "stderr.txt"), captured.stderr);
  const lastMessage = existsSync(lastMessagePath) ? readFileSync(lastMessagePath) : null;
  let parsed: T | null = null;
  if (captured.exitCode === 0 && lastMessage) {
    try {
      // This is schema-constrained Codex output, not JSON scraped from free-form text.
      parsed = JSON.parse(lastMessage.toString("utf8")) as T;
    } catch (error) {
      captured.stderr = Buffer.concat([
        captured.stderr,
        Buffer.from(`\nalpha-corpus: schema response parse failed: ${String(error)}\n`)
      ]);
    }
  }
  return {
    startedAt,
    completedAt: new Date().toISOString(),
    exitCode: captured.exitCode,
    stdout: captured.stdout,
    stderr: captured.stderr,
    lastMessage,
    parsed,
    usage: parseCodexUsage(captured.stdout),
    callDirectory,
    commandArgs: args
  };
}
