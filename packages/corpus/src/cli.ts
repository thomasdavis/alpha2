import { resolve } from "node:path";
import {
  campaignStats,
  closeLedger,
  openLedger,
  seedLedger,
  validateLedger
} from "./db.js";
import { CALIBRATION_CAMPAIGN_SLUG, generateCalibration } from "./generate.js";
import { writeAuditPacket } from "./report.js";
import { analyzeCampaign } from "./analysis.js";
import { formatBytes } from "./storage.js";

function option(args: string[], name: string, fallback?: string): string | undefined {
  const index = args.indexOf(name);
  return index === -1 ? fallback : args[index + 1];
}

function positiveInteger(value: string | undefined, fallback: number, label: string): number {
  if (value === undefined) return fallback;
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) throw new Error(`${label} must be a positive integer`);
  return parsed;
}

function print(value: unknown): void {
  process.stdout.write(`${JSON.stringify(value, null, 2)}\n`);
}

function help(): void {
  process.stdout.write(`alpha-corpus commands:
  init [--home PATH]
  validate [--home PATH]
  plan [--home PATH] [--families N] [--items-per-call N]
  generate [--execute] [--home PATH] [--families N] [--items-per-call N] [--model gpt-5.4]
  status [--home PATH] [--campaign alpha-calibration-v1]
  analyze [--home PATH] [--campaign alpha-calibration-v1]
  audit [--home PATH] [--campaign alpha-calibration-v1]

generate is a dry-run unless --execute is supplied. New generation pauses when this
project's own ledger and artifact tree exceeds 15 GiB. It never trains Alpha.
`);
}

export async function runCli(args = process.argv.slice(2)): Promise<void> {
  const command = args[0] ?? "help";
  if (["help", "--help", "-h"].includes(command)) {
    help();
    return;
  }
  const home = option(args, "--home");
  const ledger = await openLedger(home);
  try {
    await seedLedger(ledger);
    if (command === "init" || command === "validate") {
      const report = await validateLedger(ledger);
      print({ ...report, footprint: formatBytes(report.footprintBytes), home: ledger.paths.home });
      if (report.integrity !== "ok" || report.foreignKeyViolations > 0
        || report.missingTables.length > 0 || report.missingViews.length > 0 || report.missingBlobs.length > 0
        || report.corruptBlobs.length > 0) {
        process.exitCode = 1;
      }
      return;
    }
    if (command === "plan" || command === "generate") {
      const result = await generateCalibration(ledger, {
        execute: command === "generate" && args.includes("--execute"),
        model: option(args, "--model", "gpt-5.4")!,
        itemsPerCall: positiveInteger(option(args, "--items-per-call"), 4, "items-per-call"),
        familyLimit: positiveInteger(option(args, "--families"), 6, "families"),
        repoRoot: resolve(option(args, "--repo-root", process.cwd())!)
      });
      print(result);
      return;
    }
    if (command === "status") {
      print(await campaignStats(ledger, option(args, "--campaign", CALIBRATION_CAMPAIGN_SLUG)!));
      return;
    }
    if (command === "analyze") {
      print(await analyzeCampaign(ledger, option(args, "--campaign", CALIBRATION_CAMPAIGN_SLUG)!));
      return;
    }
    if (command === "audit") {
      print(await writeAuditPacket(ledger, option(args, "--campaign", CALIBRATION_CAMPAIGN_SLUG)!));
      return;
    }
    throw new Error(`Unknown command ${command}`);
  } finally {
    closeLedger(ledger);
  }
}
