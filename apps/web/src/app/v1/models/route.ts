import { getRuns } from "@/lib/engine";
import { jsonResponse } from "@/lib/server-state";

export const dynamic = "force-dynamic";

export function GET() {
  const runs = getRuns();
  return jsonResponse({
    object: "list",
    data: runs.map((r) => ({
      id: r.id,
      object: "model",
      created: Math.floor(r.mtime / 1000),
      owned_by: "alpha",
    })),
  });
}
