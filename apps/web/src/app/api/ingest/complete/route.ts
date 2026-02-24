import { getDb, updateRunProgress } from "@alpha/db";
import { checkAuth, broadcastLive, jsonResponse } from "@/lib/server-state";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  const authErr = checkAuth(request);
  if (authErr) return authErr;

  const { runId, finalStep } = await request.json();
  try {
    const client = getDb();
    await updateRunProgress(client, runId, { latest_step: finalStep, status: "completed" });
  } catch (e) {
    console.warn("Ingest complete DB error:", (e as Error).message);
  }
  broadcastLive("run_complete", { runId, finalStep });
  return jsonResponse({ ok: true });
}
