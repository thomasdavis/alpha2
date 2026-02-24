import { getClient } from "@/lib/db";
import { listRuns, getRecentMetrics } from "@alpha/db";
import { liveClients } from "@/lib/server-state";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    async start(controller) {
      // Send snapshot of active runs
      try {
        const client = await getClient();
        const activeRuns = await listRuns(client, { status: "active" });
        const snapshot: Array<Record<string, unknown>> = [];
        for (const run of activeRuns) {
          const metrics = await getRecentMetrics(client, run.id, 200);
          snapshot.push({ ...run, metrics });
        }
        controller.enqueue(encoder.encode(`event: snapshot\ndata: ${JSON.stringify(snapshot)}\n\n`));
      } catch (e) {
        console.warn("Training live snapshot error:", (e as Error).message);
        controller.enqueue(encoder.encode(`event: snapshot\ndata: []\n\n`));
      }

      liveClients.add(controller);

      // Heartbeat every 30s
      const heartbeat = setInterval(() => {
        try { controller.enqueue(encoder.encode(": heartbeat\n\n")); } catch { /* client gone */ }
      }, 30_000);

      // Cleanup on disconnect
      request.signal.addEventListener("abort", () => {
        liveClients.delete(controller);
        clearInterval(heartbeat);
      });
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
