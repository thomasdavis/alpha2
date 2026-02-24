/**
 * Lazy initialization: engine + DB sync on first request.
 */
import { initEngine } from "./engine";
import { getClient } from "./db";
import { syncFromDisk } from "@alpha/db";
import { OUTPUTS_DIR } from "./server-state";

let _initialized = false;

export async function ensureInit(): Promise<void> {
  if (_initialized) return;
  _initialized = true;

  await initEngine(OUTPUTS_DIR);

  try {
    const client = await getClient();
    const syncResult = await syncFromDisk(client, OUTPUTS_DIR);
    console.log(`DB synced: ${syncResult.runsUpserted} runs, ${syncResult.metricsInserted} new metrics`);
  } catch (e) {
    console.warn("DB init failed (dashboard will be unavailable):", (e as Error).message);
  }
}
