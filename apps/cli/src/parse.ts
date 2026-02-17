/**
 * Simple arg parsing helpers.
 * Supports --key=value and --flag syntax.
 */

export function parseKV(args: string[]): Record<string, string> {
  const result: Record<string, string> = {};
  for (const arg of args) {
    if (arg.startsWith("--")) {
      const eqIdx = arg.indexOf("=");
      if (eqIdx > 0) {
        result[arg.slice(2, eqIdx)] = arg.slice(eqIdx + 1);
      } else {
        result[arg.slice(2)] = "true";
      }
    }
  }
  return result;
}

export function requireArg(kv: Record<string, string>, key: string, label?: string): string {
  const val = kv[key];
  if (!val) {
    throw new Error(`Missing required argument: --${key}${label ? ` (${label})` : ""}`);
  }
  return val;
}

export function intArg(kv: Record<string, string>, key: string, defaultVal: number): number {
  const val = kv[key];
  return val ? parseInt(val, 10) : defaultVal;
}

export function floatArg(kv: Record<string, string>, key: string, defaultVal: number): number {
  const val = kv[key];
  return val ? parseFloat(val) : defaultVal;
}

export function strArg(kv: Record<string, string>, key: string, defaultVal: string): string {
  return kv[key] ?? defaultVal;
}

export function boolArg(kv: Record<string, string>, key: string, defaultVal: boolean): boolean {
  const val = kv[key];
  if (!val) return defaultVal;
  return val === "true" || val === "1";
}

/** Load a JSON config file and merge with CLI overrides. */
export async function loadConfig(kv: Record<string, string>): Promise<Record<string, string>> {
  const configPath = kv["config"];
  if (!configPath) return kv;
  const fs = await import("node:fs/promises");
  const raw = await fs.readFile(configPath, "utf-8");
  const config = JSON.parse(raw);
  // CLI overrides take precedence
  return { ...config, ...kv };
}
