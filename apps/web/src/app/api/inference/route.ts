import { NextRequest } from "next/server";
import { getRuns, ensureModel, generateTokens } from "@/lib/engine";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  const runs = getRuns();
  const query = request.nextUrl.searchParams.get("query") ?? "";
  const modelId = request.nextUrl.searchParams.get("model") ?? runs[0]?.id;
  const steps = Math.min(parseInt(request.nextUrl.searchParams.get("steps") ?? "200", 10), 500);
  const temperature = parseFloat(request.nextUrl.searchParams.get("temp") ?? "0.8");
  const topk = parseInt(request.nextUrl.searchParams.get("topk") ?? "40", 10);

  if (!modelId || !runs.find((r) => r.id === modelId || r.config?.runId === modelId)) {
    return Response.json({ error: "Unknown model" }, { status: 400 });
  }

  const model = await ensureModel(modelId);
  const gen = generateTokens(model, query, steps, temperature, topk);

  const stream = new ReadableStream({
    start(controller) {
      const encoder = new TextEncoder();

      function nextToken() {
        if (request.signal.aborted) return;
        const result = gen.next();
        if (result.done) {
          controller.enqueue(encoder.encode("data: [DONE]\n\n"));
          controller.close();
          return;
        }
        controller.enqueue(encoder.encode(`data: ${JSON.stringify({ token: result.value })}\n\n`));
        setImmediate(nextToken);
      }
      nextToken();
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
