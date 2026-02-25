/**
 * Simple config hashing for reproducibility tracking.
 * Uses a basic FNV-1a hash — no crypto needed.
 */

export function hashConfig(config: Record<string, unknown>): string {
  const json = JSON.stringify(config, Object.keys(config).sort());
  let hash = 0x811c9dc5;
  for (let i = 0; i < json.length; i++) {
    hash ^= json.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

export function runId(tag?: string): string {
  const now = new Date();
  const ts = now.toISOString().replace(/[-:T]/g, "").slice(0, 14);
  const rand = Math.random().toString(36).slice(2, 6);
  if (tag) return `${tag}_${ts}_${rand}`;
  return `${ts}_${rand}`;
}
