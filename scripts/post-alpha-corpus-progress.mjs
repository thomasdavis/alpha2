#!/usr/bin/env node
import { execFileSync } from "node:child_process";
import { existsSync, lstatSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";

const repo = "/mnt/donto-data/workspace/alpha2";
const corpusHome = process.env.ALPHA_CORPUS_HOME
  ?? "/mnt/donto-data/donto-resources/research/alpha2-corpus";
const webhook = process.env.ALPHA_DISCORD_WEBHOOK_URL;
const dryRun = process.env.ALPHA_DISCORD_DRY_RUN === "1";
if (!webhook && !dryRun) throw new Error("ALPHA_DISCORD_WEBHOOK_URL is not configured");

function command(binary, args) {
  return execFileSync(binary, args, { cwd: repo, encoding: "utf8" }).trim();
}

function sqliteRows(sql) {
  const output = execFileSync(
    "/usr/bin/sqlite3",
    ["-readonly", "-json", join(corpusHome, "alpha-corpus.sqlite"), sql],
    { encoding: "utf8" }
  ).trim();
  return output.length === 0 ? [] : JSON.parse(output);
}

function directorySize(path) {
  if (!existsSync(path)) return 0;
  if (!lstatSync(path).isDirectory()) return statSync(path).size;
  let total = 0;
  const stack = [path];
  while (stack.length > 0) {
    const current = stack.pop();
    for (const entry of readdirSync(current, { withFileTypes: true })) {
      const child = join(current, entry.name);
      if (entry.isSymbolicLink()) total += lstatSync(child).size;
      else if (entry.isDirectory()) stack.push(child);
      else if (entry.isFile()) total += statSync(child).size;
    }
  }
  return total;
}

function mib(bytes) {
  return `${(bytes / 1024 / 1024).toFixed(2)} MiB`;
}

const progressRows = sqliteRows(
  "SELECT * FROM campaign_progress WHERE slug = 'alpha-calibration-v1'"
);
if (progressRows.length === 0) throw new Error("alpha-calibration-v1 is absent from the ledger");
const progress = progressRows[0];
const usageRows = sqliteRows(`
  SELECT COALESCE(SUM(input_tokens), 0) AS input_tokens,
         COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens,
         COALESCE(SUM(output_tokens), 0) AS output_tokens
  FROM model_call_usage
`);
const integrityRows = sqliteRows("PRAGMA integrity_check");
const reviewRows = sqliteRows(`
  SELECT COALESCE(SUM(CASE WHEN json_extract(blindness_json, '$.pass') = 'A' AND status = 'assigned' THEN 1 ELSE 0 END), 0) AS pass_a_assigned,
         COALESCE(SUM(CASE WHEN json_extract(blindness_json, '$.pass') = 'A' AND status = 'completed' THEN 1 ELSE 0 END), 0) AS pass_a_completed,
         COALESCE(SUM(CASE WHEN json_extract(blindness_json, '$.pass') = 'B' AND status = 'assigned' THEN 1 ELSE 0 END), 0) AS pass_b_assigned,
         COALESCE(SUM(CASE WHEN json_extract(blindness_json, '$.pass') = 'B' AND status = 'completed' THEN 1 ELSE 0 END), 0) AS pass_b_completed
  FROM review_assignment
`);
const humanReviewRows = sqliteRows("SELECT COUNT(*) AS count FROM review WHERE reviewer_actor_id IS NOT NULL");
const synthesisRows = sqliteRows(`
  SELECT COALESCE(SUM(CASE WHEN status = 'assigned' THEN 1 ELSE 0 END), 0) AS pass_c_assigned,
         COALESCE(SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END), 0) AS pass_c_completed
  FROM family_synthesis_assignment
  WHERE campaign_id = (SELECT id FROM generation_campaign WHERE slug = 'alpha-calibration-v1')
`);
const familySynthesisRows = sqliteRows(`
  SELECT COUNT(*) AS family_syntheses
  FROM family_synthesis fs
  JOIN family_synthesis_assignment fsa ON fsa.id = fs.assignment_id
  WHERE fsa.campaign_id = (SELECT id FROM generation_campaign WHERE slug = 'alpha-calibration-v1')
`);
const structuralDispositionRows = sqliteRows(`
  SELECT COUNT(*) AS structural_dispositions
  FROM structural_disposition sd
  JOIN family_synthesis fs ON fs.id = sd.family_synthesis_id
  JOIN family_synthesis_assignment fsa ON fsa.id = fs.assignment_id
  WHERE fsa.campaign_id = (SELECT id FROM generation_campaign WHERE slug = 'alpha-calibration-v1')
`);
const analysisRows = sqliteRows(`
  SELECT ar.id AS analysis_run_id,
         (SELECT COUNT(*) FROM analysis_metric am WHERE am.analysis_run_id = ar.id) AS metric_count,
         (SELECT COUNT(*) FROM similarity_edge se WHERE se.analysis_run_id = ar.id) AS similarity_edge_count,
         (SELECT COUNT(*) FROM template_signature ts WHERE ts.analysis_run_id = ar.id) AS template_signature_count
  FROM analysis_run ar
  LEFT JOIN analysis_run_correction correction ON correction.erroneous_analysis_run_id = ar.id
  WHERE ar.campaign_id = (SELECT id FROM generation_campaign WHERE slug = 'alpha-calibration-v1')
    AND correction.erroneous_analysis_run_id IS NULL
  ORDER BY ar.completed_at DESC
  LIMIT 1
`);

const commit = command("git", ["log", "-1", "--format=%h %s"]);
const dirtyFiles = command("git", ["status", "--porcelain"])
  .split(/\r?\n/)
  .filter(Boolean).length;
let generationActive = false;
try {
  generationActive = command("pgrep", ["-af", "packages/corpus/dist/main.js generate"])
    .split(/\r?\n/)
    .filter(Boolean).length > 0;
} catch {
  generationActive = false;
}
const footprint = directorySize(corpusHome);
const usage = usageRows[0];
const reviewProgress = reviewRows[0];
const humanReviews = Number(humanReviewRows[0]?.["count"] ?? 0);
const synthesisProgress = synthesisRows[0];
const passCAssigned = Number(synthesisProgress?.["pass_c_assigned"] ?? 0);
const passCCompleted = Number(synthesisProgress?.["pass_c_completed"] ?? 0);
const familySyntheses = Number(familySynthesisRows[0]?.["family_syntheses"] ?? 0);
const structuralDispositions = Number(structuralDispositionRows[0]?.["structural_dispositions"] ?? 0);
const analysis = analysisRows[0];
const humanAccepted = Number(progress["human_accepted"]);
const passACompleted = Number(reviewProgress?.["pass_a_completed"] ?? 0);
const passBAssigned = Number(reviewProgress?.["pass_b_assigned"] ?? 0);
const candidateCount = Number(progress["candidates"]);
const passBCompleted = Number(reviewProgress?.["pass_b_completed"] ?? 0);
const nextGate = passACompleted < candidateCount
  ? "complete blinded Pass A human review; hidden contracts remain sealed"
  : passBCompleted < candidateCount && passBAssigned === 0
    ? "prepare contract-aware Pass B from sealed Pass A evidence"
    : passBCompleted < candidateCount
      ? "complete contract-aware Pass B; Pass C remains fail-closed"
      : passCCompleted === 0
        ? "prepare and complete Pass C family synthesis plus each structural-rejection disposition"
        : "complete operator adjudication and campaign synthesis before any generation decision";

const content = [
  `**Alpha Corpus progress — ${new Date().toISOString()}**`,
  "Goal: a chatty model specialized in language, ontology, philosophy, evidence, intent, and knowledge structure; synthetic curriculum construction is a principal half of the project.",
  `Ledger: integrity=${String(integrityRows[0]?.["integrity_check"] ?? "unknown")}; campaign=${String(progress["status"])}; tasks=${Number(progress["completed_tasks"])}/${Number(progress["task_count"])}; calls=${Number(progress["model_calls"])}.`,
  `Calibration: ${Number(progress["candidates"])} candidates; ${Number(progress["structurally_valid"])} structurally valid; ${Number(progress["structurally_rejected"])} retained rejections; ${humanAccepted} human accepted. Structural validity is not training approval.`,
  `Human review: Pass A ${passACompleted}/${Number(progress["candidates"])} completed (${Number(reviewProgress?.["pass_a_assigned"] ?? 0)} assigned); Pass B ${Number(reviewProgress?.["pass_b_completed"] ?? 0)} completed (${passBAssigned} assigned); ${humanReviews} append-only human review records.`,
  `Family synthesis: Pass C ${passCCompleted} completed (${passCAssigned} assigned); ${familySyntheses} family synthesis records; ${structuralDispositions}/6 retained structural rejections dispositioned. Pass C cannot open until every current candidate has one sealed human Pass A and Pass B review.`,
  analysis
    ? `Deterministic surface evidence: ${Number(analysis["metric_count"])} scoped metrics; ${Number(analysis["similarity_edge_count"])} pair/method edges; ${Number(analysis["template_signature_count"])} dynamic signatures. This is not semantic or human approval.`
    : "Deterministic surface evidence: no current analysis run recorded.",
  `Models: GPT-5.6-sol counsel; GPT-5.4 worker; GPT-5.5 not used. Tokens: ${Number(usage["input_tokens"]).toLocaleString()} input (${Number(usage["cached_input_tokens"]).toLocaleString()} cached), ${Number(usage["output_tokens"]).toLocaleString()} output.`,
  `Repository: ${commit}; ${dirtyFiles} pending file(s). Generation active: ${generationActive ? "yes" : "no"}. Training/GPU active: no.`,
  `Project-owned artifacts: ${mib(footprint)} of the 15 GiB soft-pause threshold.`,
  `Current next gate: ${nextGate}.`
].join("\n");

if (dryRun) {
  process.stdout.write(JSON.stringify({ posted: false, dryRun: true, content }, null, 2) + "\n");
} else {
  const response = await fetch(webhook, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      username: "Alpha Corpus",
      content,
      allowed_mentions: { parse: [] }
    })
  });
  if (!response.ok) {
    throw new Error(`Discord progress post failed with HTTP ${response.status}`);
  }
  process.stdout.write(JSON.stringify({ posted: true, status: response.status, bytes: content.length }) + "\n");
}
