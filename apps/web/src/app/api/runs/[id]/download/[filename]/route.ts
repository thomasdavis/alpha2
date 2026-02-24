import * as fs from "node:fs";
import * as path from "node:path";
import { createReadStream } from "node:fs";
import { createGzip } from "node:zlib";
import { OUTPUTS_DIR, jsonResponse } from "@/lib/server-state";

export const dynamic = "force-dynamic";

export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string; filename: string }> },
) {
  const { id, filename } = await params;

  // Path traversal protection
  if (filename.includes("..") || filename.includes("/") || filename.includes("\\")) {
    return jsonResponse({ error: "Invalid filename" }, 400);
  }
  if (id.includes("..") || id.includes("/") || id.includes("\\")) {
    return jsonResponse({ error: "Invalid run id" }, 400);
  }

  const filePath = path.join(OUTPUTS_DIR, id, filename);

  if (!fs.existsSync(filePath)) {
    return jsonResponse({ error: "File not found" }, 404);
  }

  const stat = fs.statSync(filePath);

  // Check Accept-Encoding for gzip support
  const acceptEncoding = request.headers.get("accept-encoding") ?? "";
  const supportsGzip = acceptEncoding.includes("gzip");

  if (supportsGzip) {
    // Stream file through gzip into a web ReadableStream
    const gzip = createGzip({ level: 6 });
    const fileStream = createReadStream(filePath);
    fileStream.pipe(gzip);

    const webStream = new ReadableStream({
      start(controller) {
        gzip.on("data", (chunk: Buffer) => controller.enqueue(new Uint8Array(chunk)));
        gzip.on("end", () => controller.close());
        gzip.on("error", (err) => controller.error(err));
      },
      cancel() {
        fileStream.destroy();
        gzip.destroy();
      },
    });

    return new Response(webStream, {
      headers: {
        "Content-Type": "application/octet-stream",
        "Content-Disposition": `attachment; filename="${filename}"`,
        "Content-Encoding": "gzip",
        "Cache-Control": "no-cache",
      },
    });
  }

  // No gzip — stream raw
  const fileStream = createReadStream(filePath);
  const webStream = new ReadableStream({
    start(controller) {
      fileStream.on("data", (chunk) => controller.enqueue(new Uint8Array(chunk as Buffer)));
      fileStream.on("end", () => controller.close());
      fileStream.on("error", (err) => controller.error(err));
    },
    cancel() {
      fileStream.destroy();
    },
  });

  return new Response(webStream, {
    headers: {
      "Content-Type": "application/octet-stream",
      "Content-Disposition": `attachment; filename="${filename}"`,
      "Content-Length": String(stat.size),
      "Cache-Control": "no-cache",
    },
  });
}
