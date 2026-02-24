import { getClient } from "@/lib/db";
import { listDomains } from "@alpha/db";
import { jsonResponse } from "@/lib/server-state";

export const dynamic = "force-dynamic";

export async function GET() {
  const client = await getClient();
  const domains = await listDomains(client);
  return jsonResponse(domains);
}
