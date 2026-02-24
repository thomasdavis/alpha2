import { getClient } from "@/lib/db";
import { syncFromDisk } from "@alpha/db";
import { OUTPUTS_DIR, jsonResponse } from "@/lib/server-state";

export const dynamic = "force-dynamic";

export async function POST() {
  const client = await getClient();
  const result = await syncFromDisk(client, OUTPUTS_DIR);
  return jsonResponse(result);
}
