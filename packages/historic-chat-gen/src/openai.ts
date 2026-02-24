/**
 * Raw fetch OpenAI client with retry + exponential backoff.
 * Zero dependencies — uses native fetch().
 */
import type { OpenAIResponse } from "./types.js";

const OPENAI_API_URL = "https://api.openai.com/v1/chat/completions";
const MAX_RETRIES = 3;
const BASE_DELAY_MS = 1000;

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface CompletionOptions {
  model?: string;
  messages: ChatMessage[];
  temperature?: number;
  maxTokens?: number;
}

export interface CompletionResult {
  content: string;
  inputTokens: number;
  outputTokens: number;
}

function getApiKey(): string {
  const key = process.env["OPENAI_API_KEY"];
  if (!key) throw new Error("OPENAI_API_KEY not set — add it to .env.local");
  return key;
}

export function hasApiKey(): boolean {
  return !!process.env["OPENAI_API_KEY"];
}

export async function chatCompletion(opts: CompletionOptions): Promise<CompletionResult> {
  const apiKey = getApiKey();
  const model = opts.model ?? "gpt-4.1-mini";

  const body = {
    model,
    messages: opts.messages,
    temperature: opts.temperature ?? 0.9,
    max_tokens: opts.maxTokens ?? 2048,
    response_format: { type: "json_object" },
  };

  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    if (attempt > 0) {
      const delay = BASE_DELAY_MS * Math.pow(2, attempt - 1) + Math.random() * 500;
      await sleep(delay);
    }

    try {
      const res = await fetch(OPENAI_API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${apiKey}`,
        },
        body: JSON.stringify(body),
      });

      if (res.status === 429 || res.status >= 500) {
        lastError = new Error(`OpenAI API error: ${res.status} ${res.statusText}`);
        continue;
      }

      if (!res.ok) {
        const errorBody = await res.text();
        throw new Error(`OpenAI API error ${res.status}: ${errorBody}`);
      }

      const json = (await res.json()) as OpenAIResponse;
      const choice = json.choices[0];
      if (!choice) throw new Error("No choices in OpenAI response");

      return {
        content: choice.message.content,
        inputTokens: json.usage.prompt_tokens,
        outputTokens: json.usage.completion_tokens,
      };
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
      if (attempt === MAX_RETRIES) break;
      // Only retry on network errors or rate limits
      if (err instanceof Error && !err.message.includes("API error")) {
        continue;
      }
    }
  }

  throw lastError ?? new Error("Failed after retries");
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
