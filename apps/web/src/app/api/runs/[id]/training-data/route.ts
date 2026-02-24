import * as fs from "node:fs";
import * as path from "node:path";
import { OUTPUTS_DIR, jsonResponse } from "@/lib/server-state";

export const dynamic = "force-dynamic";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const filePath = path.join(OUTPUTS_DIR, id, "training-data.txt");
  if (!fs.existsSync(filePath)) {
    return jsonResponse({ error: "No training data available" }, 404);
  }
  const text = fs.readFileSync(filePath, "utf-8");
  const bytes = Buffer.byteLength(text, "utf-8");
  const lines = text.split("\n").length;
  return jsonResponse({ text, bytes, lines });
}
