import { execFile } from "node:child_process";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { promisify } from "node:util";
import { describe, expect, it } from "vitest";

const execFileAsync = promisify(execFile);
const repoRoot = fileURLToPath(new URL("../../..", import.meta.url));
const protocolUrl = pathToFileURL(join(repoRoot, "apps/hf/src/protocol.ts")).href;

describe("HF chat runtime protocol", () => {
  it("resolves all boundaries and fails closed on non-atomic or aliased markers", async () => {
    const program = `
      import { ASSISTANT_TOKEN, END_TOKEN, resolveChatStopTokenIds, USER_TOKEN } from ${JSON.stringify(protocolUrl)};
      const vocabulary = new Map([[USER_TOKEN, 256], [ASSISTANT_TOKEN, 257], [END_TOKEN, 258]]);
      const stops = resolveChatStopTokenIds((token) => vocabulary.has(token) ? [vocabulary.get(token)] : []);
      const error = (callback) => { try { callback(); return null; } catch (caught) { return String(caught.message); } };
      const nonAtomic = error(() => resolveChatStopTokenIds((token) => token === END_TOKEN ? [258] : token === USER_TOKEN ? [256, 0] : [257]));
      const aliased = error(() => resolveChatStopTokenIds((token) => token === END_TOKEN ? [258] : [256]));
      process.stdout.write(JSON.stringify({ eos: stops.eos, user: stops.user, assistant: stops.assistant, all: [...stops.all], nonAtomic, aliased }));
    `;
    const { stdout } = await execFileAsync("npx", ["tsx", "--eval", program], { cwd: repoRoot });
    expect(JSON.parse(stdout)).toEqual({
      eos: 258,
      user: 256,
      assistant: 257,
      all: [258, 256, 257],
      nonAtomic: expect.stringContaining("is not an atomic tokenizer token"),
      aliased: expect.stringContaining("must have distinct token IDs"),
    });
  }, 30_000);
});
