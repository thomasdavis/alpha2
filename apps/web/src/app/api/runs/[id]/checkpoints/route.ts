import { getDb, listCheckpoints } from "@alpha/db";
import { jsonResponse } from "@/lib/server-state";

export const dynamic = "force-dynamic";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const client = getDb();
  const checkpoints = await listCheckpoints(client, id);
  return jsonResponse(checkpoints);
}
