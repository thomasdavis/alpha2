import { getClient } from "@/lib/db";
import { getSamples } from "@alpha/db";
import { jsonResponse } from "@/lib/server-state";

export const dynamic = "force-dynamic";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const client = await getClient();
  const samples = await getSamples(client, id);
  return jsonResponse(samples);
}
