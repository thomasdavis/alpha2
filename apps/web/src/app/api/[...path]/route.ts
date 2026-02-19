import { NextRequest } from "next/server";

const SERVER_URL =
  process.env.INTERNAL_SERVER_URL || "http://localhost:3001";

async function proxy(request: NextRequest, segments: string[]) {
  const target = new URL(`/api/${segments.join("/")}`, SERVER_URL);
  target.search = request.nextUrl.search;

  const headers = new Headers(request.headers);
  headers.delete("host");

  const init: RequestInit = {
    method: request.method,
    headers,
    signal: AbortSignal.timeout(30_000),
  };

  if (request.method !== "GET" && request.method !== "HEAD") {
    init.body = request.body;
    // @ts-expect-error -- duplex required for streaming request bodies
    init.duplex = "half";
  }

  try {
    const upstream = await fetch(target, init);

    return new Response(upstream.body, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers: upstream.headers,
    });
  } catch (e) {
    const message = e instanceof Error ? e.message : "Upstream request failed";
    const status = message.includes("timed out") ? 504 : 502;
    return new Response(
      JSON.stringify({ error: message }),
      { status, headers: { "Content-Type": "application/json" } },
    );
  }
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params;
  return proxy(request, path);
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params;
  return proxy(request, path);
}
