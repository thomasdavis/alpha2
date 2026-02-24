import { getDb, syncFromDisk } from "@alpha/db";
import { OUTPUTS_DIR, jsonResponse } from "@/lib/server-state";

export const dynamic = "force-dynamic";

export async function POST() {
  const client = getDb();
  const result = await syncFromDisk(client, OUTPUTS_DIR);
  return jsonResponse(result);
}
