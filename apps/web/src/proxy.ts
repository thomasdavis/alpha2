import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

export function proxy(request: NextRequest) {
  if (request.nextUrl.pathname === "/corpus" || request.nextUrl.pathname.startsWith("/corpus/")) {
    if (request.method !== "GET" && request.method !== "HEAD") {
      return new NextResponse("Alpha Corpus is a read-only public surface.\n", {
        status: 405,
        headers: { Allow: "GET, HEAD" }
      });
    }
    return NextResponse.next();
  }

  // Handle CORS preflight for API and OpenAI-compat routes.
  if (request.method === "OPTIONS") {
    return new NextResponse(null, {
      status: 204,
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization, Content-Encoding"
      }
    });
  }

  const response = NextResponse.next();
  response.headers.set("Access-Control-Allow-Origin", "*");
  response.headers.set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
  response.headers.set("Access-Control-Allow-Headers", "Content-Type, Authorization, Content-Encoding");
  return response;
}

export const config = {
  matcher: ["/api/:path*", "/v1/:path*", "/chat/completions", "/corpus/:path*"]
};
