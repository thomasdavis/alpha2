/**
 * Next.js instrumentation hook — runs once at server startup.
 */
export async function register() {
  if (process.env.NEXT_RUNTIME === "nodejs") {
    const { ensureInit } = await import("./lib/init");
    await ensureInit();
  }
}
