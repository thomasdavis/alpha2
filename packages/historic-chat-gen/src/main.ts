#!/usr/bin/env node
/**
 * Standalone CLI entry point for historic-chat-gen.
 * Loads .env.local from cwd, then dispatches to chatgenCmd.
 */
import { readFileSync } from "node:fs";

// Load .env.local (no dotenv — zero deps philosophy)
try {
  const envContent = readFileSync(".env.local", "utf8");
  for (const line of envContent.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq < 0) continue;
    const key = trimmed.slice(0, eq).trim();
    const val = trimmed.slice(eq + 1).trim().replace(/^["']|["']$/g, "");
    if (!process.env[key]) process.env[key] = val;
  }
} catch { /* .env.local is optional */ }

import { chatgenCmd } from "./cli.js";

chatgenCmd(process.argv.slice(2)).catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
