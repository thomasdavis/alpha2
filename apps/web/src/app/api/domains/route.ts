import { getDb, listDomains } from "@alpha/db";
import { jsonResponse } from "@/lib/server-state";

export const dynamic = "force-dynamic";

export async function GET() {
  const client = getDb();
  const domains = await listDomains(client);
  return jsonResponse(domains);
}
