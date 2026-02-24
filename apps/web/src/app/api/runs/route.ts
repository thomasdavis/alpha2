import { NextRequest } from "next/server";
import { getDb, listRuns } from "@alpha/db";
import { jsonResponse } from "@/lib/server-state";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  const client = getDb();
  const status = request.nextUrl.searchParams.get("status") as any;
  const domain = request.nextUrl.searchParams.get("domain") ?? undefined;
  const runs = await listRuns(client, { status: status || undefined, domain });
  return jsonResponse(runs);
}
