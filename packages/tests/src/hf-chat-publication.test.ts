import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { afterEach, describe, expect, it } from "vitest";

const execFileAsync = promisify(execFile);
const repoRoot = fileURLToPath(new URL("../../..", import.meta.url));
const publisher = join(repoRoot, "scripts/publish_hf_chat.py");
const temporaryPaths: string[] = [];

function sha256(value: string | Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

afterEach(async () => {
  await Promise.all(temporaryPaths.splice(0).map((entry) => rm(entry, { recursive: true, force: true })));
});

describe("HF chat publication preflight", () => {
  it("requires every release gate and exact zero-custom-code export before publication", async () => {
    const dir = await mkdtemp(join(tmpdir(), "alpha-hf-chat-publish-"));
    const outDir = await mkdtemp("/mnt/donto-data/alpha-runs/hf-chat-publication-test-");
    temporaryPaths.push(dir, outDir);
    const exportDir = join(dir, "export");
    await mkdir(exportDir);
    const weights = Buffer.from("synthetic safetensors");
    const checkpointSha = "a".repeat(64);
    const files: Record<string, string | Buffer> = {
      "model.safetensors": weights,
      "config.json": JSON.stringify({
        architectures: ["LlamaForCausalLM"], model_type: "llama", vocab_size: 12_288,
        hidden_size: 512, num_hidden_layers: 16, num_attention_heads: 8, tie_word_embeddings: true,
      }) + "\n",
      "generation_config.json": "{}\n",
      "tokenizer.json": "{}\n",
      "tokenizer_config.json": "{}\n",
      "chat_template.jinja": "{{ message }}\n",
    };
    await Promise.all(Object.entries(files).map(([name, value]) => writeFile(join(exportDir, name), value)));
    const terminalPath = join(dir, "terminal.json");
    const sftPath = join(dir, "sft.json");
    const pairPath = join(dir, "pair.json");
    const semanticPath = join(dir, "semantic.json");
    const parityPath = join(dir, "parity.log");
    const cardPath = join(dir, "README.md");
    await Promise.all([
      writeFile(terminalPath, JSON.stringify({
        schema: "alpha-sft-terminal-finalizer-v1", result: "PASS",
        source_commit: "c333bf247fbe87b85d01f3d34789b46615dd1034",
        checkpoint: { sha256: checkpointSha }, machine_d3: { result: "PASS" },
      }) + "\n"),
      writeFile(sftPath, JSON.stringify({
        schema: "alpha-flagship-sft-analysis-v1", result: "PASS",
        source_commit: "c333bf247fbe87b85d01f3d34789b46615dd1034", rows: 30_322,
        checkpoint: { sha256: checkpointSha, parameter_elements: 57_688_576, finite_parameter_elements: 57_688_576 },
      }) + "\n"),
      writeFile(pairPath, JSON.stringify({
        schema: "alpha-frozen-eval-pair-analysis-v1", result: "PASS", inputs_match: true,
        chat: { checkpoint: { sha256: checkpointSha } },
      }) + "\n"),
      writeFile(semanticPath, JSON.stringify({
        schema: "alpha-frozen-chat-semantic-review-v1", result: "PASS", reference_blinded: true,
        counts: { total: 100, PASS: 80, BORDERLINE: 20, FAIL: 0 },
        provenance: { checkpoint: { sha256: checkpointSha } },
      }) + "\n"),
      writeFile(parityPath, "RESULT               : PASS\n"),
      writeFile(cardPath, `---\nlicense: apache-2.0\nlibrary_name: transformers\n---\n# Alpha 60M Chat\n${checkpointSha}\n${sha256(weights)}\n`),
    ]);
    const outPath = join(outDir, "preflight.json");
    const args = [publisher,
      "--export-dir", exportDir,
      "--model-card", cardPath,
      "--terminal-status", terminalPath,
      "--sft-analysis", sftPath,
      "--pair-analysis", pairPath,
      "--semantic-review", semanticPath,
      "--parity-log", parityPath,
      "--repo", "ajaxdavis/alpha-60m-chat",
      "--out", outPath,
    ];
    await execFileAsync("python3", args, { cwd: repoRoot });
    const report = JSON.parse(await readFile(outPath, "utf8"));
    expect(report).toMatchObject({
      schema: "alpha-hf-chat-publication-v1", result: "PASS", mode: "preflight",
      repo: "ajaxdavis/alpha-60m-chat", checkpoint_sha256: checkpointSha,
      weights_sha256: sha256(weights), parameter_elements: 57_688_576,
    });

    const fakeHubDir = join(dir, "fake-hub");
    await mkdir(fakeHubDir);
    await writeFile(join(fakeHubDir, "huggingface_hub.py"), `
from pathlib import Path

REVISION = "${"b".repeat(40)}"
EXPECTED = {
    "README.md", "model.safetensors", "config.json", "generation_config.json",
    "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
}

class Commit:
    oid = REVISION

class Sibling:
    def __init__(self, filename):
        self.rfilename = filename

class Info:
    sha = REVISION
    private = False
    siblings = [Sibling(name) for name in sorted(EXPECTED | {".gitattributes"})]

class HfApi:
    def __init__(self, token=None):
        self.token = token

    def whoami(self):
        return {"name": "ajaxdavis"}

    def create_repo(self, *, repo_id, repo_type, private, exist_ok):
        assert repo_id == "ajaxdavis/alpha-60m-chat"
        assert repo_type == "model" and private is False and exist_ok is True

    def upload_folder(self, *, repo_id, repo_type, folder_path, commit_message):
        assert repo_id == "ajaxdavis/alpha-60m-chat" and repo_type == "model"
        assert {entry.name for entry in Path(folder_path).iterdir()} == EXPECTED
        assert commit_message
        return Commit()

    def update_repo_settings(self, *, repo_id, repo_type, private, gated):
        assert repo_id == "ajaxdavis/alpha-60m-chat" and repo_type == "model"
        assert private is False and gated is False

    def model_info(self, repo_id, *, revision, token):
        assert self.token is False and token is False
        assert repo_id == "ajaxdavis/alpha-60m-chat" and revision == REVISION
        return Info()
`);
    const publishOut = join(outDir, "published.json");
    const publishArgs = [...args.slice(0, -1), publishOut, "--publish"];
    await execFileAsync("python3", publishArgs, {
      cwd: repoRoot,
      env: { ...process.env, PYTHONPATH: fakeHubDir },
    });
    const published = JSON.parse(await readFile(publishOut, "utf8"));
    expect(published).toMatchObject({
      schema: "alpha-hf-chat-publication-v1", result: "PASS", mode: "publish",
      hub: { revision: "b".repeat(40), public: true, anonymous: true },
    });

    const rejectedOut = join(outDir, "rejected.json");
    await writeFile(pairPath, JSON.stringify({
      schema: "alpha-frozen-eval-pair-analysis-v1", result: "FAIL", inputs_match: true,
      chat: { checkpoint: { sha256: checkpointSha } },
    }) + "\n");
    const rejectedArgs = [...args.slice(0, -1), rejectedOut];
    await expect(execFileAsync("python3", rejectedArgs, { cwd: repoRoot }))
      .rejects.toMatchObject({ stderr: expect.stringContaining("machine D3 pair result") });
    await expect(readFile(rejectedOut)).rejects.toMatchObject({ code: "ENOENT" });

    await Promise.all([
      writeFile(terminalPath, JSON.stringify({
        schema: "alpha-sft-terminal-finalizer-v1", result: "PASS",
        source_commit: "c333bf247fbe87b85d01f3d34789b46615dd1034",
        checkpoint: { sha256: checkpointSha }, machine_d3: { result: "FAIL" },
      }) + "\n"),
      writeFile(semanticPath, JSON.stringify({
        schema: "alpha-frozen-chat-semantic-review-v1", result: "FAIL", reference_blinded: true,
        counts: { total: 100, PASS: 0, BORDERLINE: 0, FAIL: 100 },
        provenance: { checkpoint: { sha256: checkpointSha } },
      }) + "\n"),
      writeFile(cardPath, `---\nlicense: apache-2.0\nlibrary_name: transformers\n---\n# Alpha 60M Chat\nTHIS CHECKPOINT FAILED THE PREDECLARED CHAT-QUALITY GATES\n${checkpointSha}\n${sha256(weights)}\n`),
    ]);
    const experimentalOut = join(outDir, "experimental.json");
    const experimentalArgs = [
      ...args.slice(0, -1), experimentalOut, "--experimental-failed-release",
    ];
    await execFileAsync("python3", experimentalArgs, { cwd: repoRoot });
    const experimental = JSON.parse(await readFile(experimentalOut, "utf8"));
    expect(experimental).toMatchObject({
      schema: "alpha-hf-chat-publication-v1", result: "PASS", mode: "preflight",
      release_classification: "EXPERIMENTAL_FAILED_CHAT_CANDIDATE", quality_gate_result: "FAIL",
    });
  }, 120_000);
});
