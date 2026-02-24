import { BUILD_INFO, jsonResponse } from "@/lib/server-state";

export const dynamic = "force-dynamic";

export function GET() {
  return jsonResponse(BUILD_INFO);
}
