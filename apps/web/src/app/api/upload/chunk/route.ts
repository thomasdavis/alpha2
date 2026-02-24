import { checkAuth, pendingChunks, jsonResponse } from "@/lib/server-state";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  const authErr = checkAuth(request);
  if (authErr) return authErr;

  const uploadId = request.headers.get("x-upload-id");
  const chunkIndex = parseInt(request.headers.get("x-chunk-index") ?? "", 10);
  const totalChunks = parseInt(request.headers.get("x-total-chunks") ?? "", 10);

  if (!uploadId || isNaN(chunkIndex) || isNaN(totalChunks)) {
    return jsonResponse({ error: "Missing X-Upload-Id, X-Chunk-Index, or X-Total-Chunks" }, 400);
  }

  const raw = await request.arrayBuffer();
  const body = Buffer.from(raw);

  if (!pendingChunks.has(uploadId)) {
    pendingChunks.set(uploadId, { chunks: new Map(), total: totalChunks, receivedAt: Date.now() });
  }
  const entry = pendingChunks.get(uploadId)!;
  entry.chunks.set(chunkIndex, body);
  entry.receivedAt = Date.now();

  return jsonResponse({ ok: true, chunk: chunkIndex, received: entry.chunks.size, total: totalChunks });
}
