/**
 * Lazy initialization: engine + DB sync on first request.
 */
import { initEngine } from "./engine";
import { createDb, getDb, syncFromDisk } from "@alpha/db";
import { OUTPUTS_DIR } from "./server-state";

let _initialized = false;

export async function ensureInit(): Promise<void> {
  if (_initialized) return;
  _initialized = true;

  await initEngine(OUTPUTS_DIR);

  try {
    await createDb();
    const syncResult = await syncFromDisk(getDb(), OUTPUTS_DIR);
    console.log(`DB synced: ${syncResult.runsUpserted} runs, ${syncResult.metricsInserted} new metrics`);
  } catch (e) {
    console.warn("DB init failed (dashboard will be unavailable):", (e as Error).message);
  }
}
