#!/usr/bin/env npx tsx
/**
 * Validate and fix chat training data.
 *
 * Usage:
 *   npx tsx scripts/validate-chat-data.ts data/chat_combined.txt          # validate only
 *   npx tsx scripts/validate-chat-data.ts data/chat_combined.txt --fix    # write fixed version
 *   npx tsx scripts/validate-chat-data.ts data/chat_combined.txt --fix --out data/chat_clean.txt
 *
 * Expected format (one conversation per line):
 *   <|user|> msg <|assistant|> resp [<|user|> msg <|assistant|> resp ...] <|end_of_text|>
 *
 * Rules:
 *   1. Every line must start with <|user|>
 *   2. Turns must strictly alternate: user → assistant → user → assistant → ...
 *   3. Every line must end with <|end_of_text|>
 *   4. Last turn must be an <|assistant|> turn (model learns to respond)
 *   5. Must have at least 1 user turn and 1 assistant turn
 *   6. No empty turns (e.g. <|user|> <|assistant|>)
 */

import * as fs from "node:fs";
import * as path from "node:path";

const USER = "<|user|>";
const ASST = "<|assistant|>";
const EOT = "<|end_of_text|>";

interface Issue {
  line: number;
  type: string;
  detail: string;
  fixable: boolean;
}

interface ParsedConvo {
  turns: { role: "user" | "assistant"; text: string }[];
  hasEot: boolean;
  raw: string;
}

function parseConversation(line: string): ParsedConvo {
  const hasEot = line.endsWith(EOT);
  let content = hasEot ? line.slice(0, -EOT.length) : line;

  // Split by role tokens
  const parts = content.split(/(<\|user\|>|<\|assistant\|>)/);
  const turns: { role: "user" | "assistant"; text: string }[] = [];

  let currentRole: "user" | "assistant" | null = null;
  for (const part of parts) {
    if (part === USER) {
      currentRole = "user";
    } else if (part === ASST) {
      currentRole = "assistant";
    } else if (currentRole && part.trim()) {
      turns.push({ role: currentRole, text: part.trim() });
      currentRole = null; // consumed
    } else if (currentRole && !part.trim()) {
      // Empty turn — record it anyway so we can detect it
      turns.push({ role: currentRole, text: "" });
      currentRole = null;
    }
  }

  return { turns, hasEot, raw: line };
}

function validateLine(line: string, lineNum: number): Issue[] {
  const issues: Issue[] = [];

  if (!line.trim()) {
    issues.push({ line: lineNum, type: "empty", detail: "Empty line", fixable: true });
    return issues;
  }

  const convo = parseConversation(line);

  // Rule 1: Must start with <|user|>
  if (!line.startsWith(USER)) {
    if (line.startsWith(ASST)) {
      issues.push({
        line: lineNum,
        type: "assistant_first",
        detail: "Conversation starts with <|assistant|> instead of <|user|>",
        fixable: true,
      });
    } else {
      issues.push({
        line: lineNum,
        type: "no_role_start",
        detail: `Line doesn't start with a role token: "${line.slice(0, 60)}..."`,
        fixable: false,
      });
    }
  }

  // Rule 2: Alternating turns
  for (let i = 1; i < convo.turns.length; i++) {
    if (convo.turns[i].role === convo.turns[i - 1].role) {
      issues.push({
        line: lineNum,
        type: "consecutive_same_role",
        detail: `Consecutive ${convo.turns[i].role} turns at position ${i} and ${i - 1}`,
        fixable: false,
      });
      break;
    }
  }

  // Rule 3: Must end with <|end_of_text|>
  if (!convo.hasEot) {
    issues.push({
      line: lineNum,
      type: "missing_eot",
      detail: "Missing <|end_of_text|> at end of line",
      fixable: true,
    });
  }

  // Rule 4: Last turn should be assistant
  if (convo.turns.length > 0 && convo.turns[convo.turns.length - 1].role === "user") {
    issues.push({
      line: lineNum,
      type: "ends_with_user",
      detail: "Conversation ends with a user turn (no assistant response)",
      fixable: true,
    });
  }

  // Rule 5: At least 1 user + 1 assistant turn
  const userTurns = convo.turns.filter((t) => t.role === "user").length;
  const asstTurns = convo.turns.filter((t) => t.role === "assistant").length;
  if (userTurns === 0 || asstTurns === 0) {
    issues.push({
      line: lineNum,
      type: "single_role",
      detail: `Only has ${userTurns} user and ${asstTurns} assistant turns`,
      fixable: false,
    });
  }

  // Rule 6: No empty turns
  for (let i = 0; i < convo.turns.length; i++) {
    if (!convo.turns[i].text) {
      issues.push({
        line: lineNum,
        type: "empty_turn",
        detail: `Empty ${convo.turns[i].role} turn at position ${i}`,
        fixable: true,
      });
    }
  }

  return issues;
}

function fixLine(line: string): string | null {
  if (!line.trim()) return null; // drop empty lines

  const convo = parseConversation(line);

  let turns = convo.turns.filter((t) => t.text); // remove empty turns

  // If starts with assistant, trim leading assistant turns until we hit a user turn
  while (turns.length > 0 && turns[0].role === "assistant") {
    turns = turns.slice(1);
  }

  // If ends with user, trim trailing user turn
  while (turns.length > 0 && turns[turns.length - 1].role === "user") {
    turns.pop();
  }

  // Need at least user + assistant
  if (turns.length < 2) return null;

  // Verify alternation — if broken, drop the line
  for (let i = 1; i < turns.length; i++) {
    if (turns[i].role === turns[i - 1].role) return null;
  }

  // Verify starts with user
  if (turns[0].role !== "user") return null;

  // Rebuild the line
  const parts = turns.map((t) => `<|${t.role}|> ${t.text}`);
  return parts.join(" ") + EOT;
}

// --- Main ---

const args = process.argv.slice(2);
const inputFile = args.find((a) => !a.startsWith("-"));
const doFix = args.includes("--fix");
const outIdx = args.indexOf("--out");
const outFile = outIdx >= 0 ? args[outIdx + 1] : undefined;

if (!inputFile) {
  console.error("Usage: npx tsx scripts/validate-chat-data.ts <file> [--fix] [--out <output>]");
  process.exit(1);
}

const content = fs.readFileSync(inputFile, "utf-8");
const lines = content.split("\n").filter((l) => l.length > 0);

console.log(`Validating ${lines.length.toLocaleString()} conversations from ${path.basename(inputFile)}...\n`);

const allIssues: Issue[] = [];
const issueCounts: Record<string, number> = {};

for (let i = 0; i < lines.length; i++) {
  const issues = validateLine(lines[i], i + 1);
  for (const issue of issues) {
    allIssues.push(issue);
    issueCounts[issue.type] = (issueCounts[issue.type] || 0) + 1;
  }
}

// Summary
const linesWithIssues = new Set(allIssues.map((i) => i.line)).size;
const cleanLines = lines.length - linesWithIssues;

console.log("=== Validation Summary ===\n");
console.log(`Total conversations: ${lines.length.toLocaleString()}`);
console.log(`Clean (no issues):   ${cleanLines.toLocaleString()} (${((cleanLines / lines.length) * 100).toFixed(1)}%)`);
console.log(`With issues:         ${linesWithIssues.toLocaleString()} (${((linesWithIssues / lines.length) * 100).toFixed(1)}%)\n`);

console.log("Issue breakdown:");
const sortedIssues = Object.entries(issueCounts).sort((a, b) => b[1] - a[1]);
for (const [type, count] of sortedIssues) {
  const fixable = allIssues.find((i) => i.type === type)?.fixable ? "fixable" : "NOT fixable";
  console.log(`  ${type.padEnd(25)} ${count.toLocaleString().padStart(8)}  (${fixable})`);
}

// Show samples
console.log("\nSample issues (first 3 of each type):");
const shown: Record<string, number> = {};
for (const issue of allIssues) {
  shown[issue.type] = (shown[issue.type] || 0) + 1;
  if (shown[issue.type] <= 3) {
    console.log(`  Line ${issue.line}: [${issue.type}] ${issue.detail}`);
  }
}

if (doFix) {
  console.log("\n=== Fixing... ===\n");
  const fixed: string[] = [];
  let dropped = 0;

  for (const line of lines) {
    const result = fixLine(line);
    if (result) {
      fixed.push(result);
    } else {
      dropped++;
    }
  }

  const output = outFile || inputFile.replace(/\.txt$/, "_clean.txt");
  fs.writeFileSync(output, fixed.join("\n") + "\n");

  console.log(`Input:   ${lines.length.toLocaleString()} conversations`);
  console.log(`Output:  ${fixed.length.toLocaleString()} conversations`);
  console.log(`Dropped: ${dropped.toLocaleString()} (unfixable or too short after trimming)`);
  console.log(`Written to: ${output}`);

  // Validate the output
  console.log("\n=== Re-validating fixed output ===\n");
  const fixedContent = fs.readFileSync(output, "utf-8");
  const fixedLines = fixedContent.split("\n").filter((l) => l.length > 0);
  let fixedIssueCount = 0;
  for (let i = 0; i < fixedLines.length; i++) {
    const issues = validateLine(fixedLines[i], i + 1);
    fixedIssueCount += issues.length;
    if (issues.length > 0 && fixedIssueCount <= 5) {
      for (const issue of issues) {
        console.log(`  Line ${issue.line}: [${issue.type}] ${issue.detail}`);
      }
    }
  }
  console.log(`Clean conversations: ${fixedLines.length.toLocaleString()} with ${fixedIssueCount} remaining issues`);
} else {
  console.log("\nRun with --fix to generate a cleaned version.");
}
