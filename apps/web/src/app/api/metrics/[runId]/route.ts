import { NextRequest, NextResponse } from "next/server";
import { getClient } from "@/lib/db";
import { getRecentMetrics, getMetrics } from "@alpha/db";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ runId: string }> }
) {
  const { runId } = await params;
  const client = getClient();
  const last = parseInt(
    request.nextUrl.searchParams.get("last") ?? "0",
    10
  );

  const metrics =
    last > 0
      ? await getRecentMetrics(client, runId, last)
      : await getMetrics(client, runId);

  return NextResponse.json(metrics);
}
