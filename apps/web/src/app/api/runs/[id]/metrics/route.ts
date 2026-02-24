import { NextRequest } from "next/server";
import { getDb, getRecentMetrics, getMetrics } from "@alpha/db";
import { jsonResponse } from "@/lib/server-state";

export const dynamic = "force-dynamic";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const client = getDb();
  const last = parseInt(request.nextUrl.searchParams.get("last") ?? "0", 10);
  const metrics = last > 0
    ? await getRecentMetrics(client, id, last)
    : await getMetrics(client, id);
  return jsonResponse(metrics);
}
