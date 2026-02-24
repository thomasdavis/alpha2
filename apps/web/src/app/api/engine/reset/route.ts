import { resetEngine, initEngine, getRuns } from "@/lib/engine";
import { OUTPUTS_DIR, checkAuth, invalidateModelsCache, jsonResponse } from "@/lib/server-state";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  const authErr = checkAuth(request);
  if (authErr) return authErr;

  resetEngine();
  await initEngine(OUTPUTS_DIR);
  invalidateModelsCache();

  const runCount = getRuns().length;
  console.log(`Engine reset: rescanned ${runCount} run(s)`);
  return jsonResponse({ ok: true, runs: runCount });
}
