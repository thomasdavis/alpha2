/**
 * hf-export — unit tests for the ALPH→HuggingFace LlamaForCausalLM exporter.
 *
 * Covers the parts that don't need a GPU or Python:
 *   - safetensors container round-trips (8-byte LE header len + JSON + LE f32).
 *   - ALPH params map to EXACT Llama state-dict names; the fused wqkv [3E,E]
 *     splits into q/k/v [E,E] with the correct (contiguous, no-permute) bytes.
 *   - tied embeddings ⇒ NO lm_head tensor + tie_word_embeddings:true.
 *   - config.json llama fields (rms_norm_eps, rope_theta, head_dim, bos/eos).
 *   - llamaFormViolations rejects a GPT-2-style config.
 *   - exportHfModel writes the full repo (weights + config + tokenizer files).
 *
 * The Alpha-forward == transformers-forward golden-token equivalence is proven
 * separately (needs Python) by scripts/verify_hf_export.py in the e2e pipeline.
 */
import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { mkdtempSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Effect } from "effect";
import { CpuRefBackend } from "@alpha/tensor";
import { SeededRng, type ModelConfig, type CheckpointState, type TokenizerArtifacts } from "@alpha/core";
import { initGPT, collectParamEntries } from "@alpha/model";
import { ByteBpeTokenizer } from "@alpha/tokenizers";
import {
  writeSafetensors,
  checkpointToLlamaStateDict,
  buildLlamaConfig,
  buildGenerationConfig,
  exportHfModel,
  llamaFormViolations,
  llamaIntermediateSize,
  resolveBosEosId,
  type SafeTensor,
} from "@alpha/train";

const B = new CpuRefBackend();

function llamaConfig(vocabSize: number): ModelConfig {
  return {
    vocabSize,
    blockSize: 16,
    nLayer: 2,
    nEmbd: 32,
    nHead: 4, // headDim = 8 (even)
    dropout: 0,
    ffnActivation: "swiglu",
    normType: "rmsnorm",
    posEnc: "rope",
    ropeTheta: 10000,
    tieEmbeddings: true,
  };
}

/** Build a real byte-level BPE artifact (small vocab) so tokenizer export works. */
async function tinyByteBpeArtifacts(vocab = 300): Promise<TokenizerArtifacts> {
  const tok = new ByteBpeTokenizer(vocab);
  const corpus = "the quick brown fox jumps over the lazy dog. ".repeat(200) +
    "hello world, this is a tiny corpus for byte bpe. ".repeat(200);
  return Effect.runPromise(tok.build(corpus));
}

/** Assemble a CheckpointState around freshly-initialised params (no optimizer). */
function makeState(config: ModelConfig, artifacts: TokenizerArtifacts, tied = true): CheckpointState {
  const rng = new SeededRng(1234);
  const cfg = tied ? config : { ...config, tieEmbeddings: false };
  const params = initGPT(cfg, B, rng);
  const rec: Record<string, { shape: number[]; data: number[] }> = {};
  for (const [name, v] of collectParamEntries(params)) {
    rec[name] = { shape: [...v.data.shape], data: v.data.data as unknown as number[] };
  }
  return {
    modelConfig: cfg,
    params: rec,
    optimizerState: { step: 0, buffers: new Map() },
    tokenizerArtifacts: artifacts,
    rngState: 1234,
    configHash: "test",
    step: 42,
  };
}

/** Parse a safetensors file into {metadata, tensors:{name→{dtype,shape,f32}}}. */
function parseSafetensors(buf: Buffer): {
  metadata: Record<string, string>;
  tensors: Record<string, { dtype: string; shape: number[]; f32: Float32Array }>;
} {
  const headerLen = Number(buf.readBigUInt64LE(0));
  const header = JSON.parse(buf.subarray(8, 8 + headerLen).toString("utf-8"));
  const dataStart = 8 + headerLen;
  const metadata = header.__metadata__ ?? {};
  const tensors: Record<string, { dtype: string; shape: number[]; f32: Float32Array }> = {};
  for (const [name, spec] of Object.entries(header) as [string, any][]) {
    if (name === "__metadata__") continue;
    const [begin, end] = spec.data_offsets;
    const slice = buf.buffer.slice(buf.byteOffset + dataStart + begin, buf.byteOffset + dataStart + end);
    tensors[name] = { dtype: spec.dtype, shape: spec.shape, f32: new Float32Array(slice) };
  }
  return { metadata, tensors };
}

describe("safetensors writer", () => {
  it("round-trips f32 tensors with 8-byte-aligned header and pt metadata", async () => {
    const dir = mkdtempSync(join(tmpdir(), "alpha-st-"));
    const path = join(dir, "t.safetensors");
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    const b = new Float32Array([-0.5, 0.25, 100.0]);
    const tensors: SafeTensor[] = [
      { name: "alpha", shape: [2, 3], data: a },
      { name: "beta", shape: [3], data: b },
    ];
    await writeSafetensors(path, tensors);

    const buf = readFileSync(path);
    const headerLen = Number(buf.readBigUInt64LE(0));
    expect(headerLen % 8).toBe(0); // 8-byte aligned

    const parsed = parseSafetensors(buf);
    expect(parsed.metadata.format).toBe("pt");
    expect(parsed.tensors.alpha.dtype).toBe("F32");
    expect(parsed.tensors.alpha.shape).toEqual([2, 3]);
    expect(Array.from(parsed.tensors.alpha.f32)).toEqual([1, 2, 3, 4, 5, 6]);
    expect(Array.from(parsed.tensors.beta.f32)).toEqual([-0.5, 0.25, 100.0]);
  });

  it("writes correct bytes for subarray (view) tensors", async () => {
    const dir = mkdtempSync(join(tmpdir(), "alpha-st-"));
    const path = join(dir, "v.safetensors");
    const backing = new Float32Array([10, 11, 12, 13, 14, 15]);
    const view = backing.subarray(2, 5); // [12,13,14]
    await writeSafetensors(path, [{ name: "v", shape: [3], data: view }]);
    const parsed = parseSafetensors(readFileSync(path));
    expect(Array.from(parsed.tensors.v.f32)).toEqual([12, 13, 14]);
  });
});

describe("llamaFormViolations", () => {
  it("accepts a Llama-form config", () => {
    expect(llamaFormViolations(llamaConfig(300))).toEqual([]);
  });
  it("rejects GPT-2-style (layernorm/learned/gelu)", () => {
    const gpt2: ModelConfig = {
      vocabSize: 256, blockSize: 16, nLayer: 2, nEmbd: 32, nHead: 4, dropout: 0,
      ffnActivation: "gelu",
    };
    const v = llamaFormViolations(gpt2);
    expect(v.length).toBeGreaterThanOrEqual(3);
    expect(v.join(" ")).toContain("rmsnorm");
    expect(v.join(" ")).toContain("rope");
    expect(v.join(" ")).toContain("swiglu");
  });
  it("rejects an odd head_dim", () => {
    const cfg = { ...llamaConfig(300), nHead: 1, nEmbd: 30 }; // headDim 30 even, but change to odd
    const odd = { ...cfg, nEmbd: 32, nHead: 32 }; // headDim 1 (odd)
    expect(llamaFormViolations(odd).join(" ")).toContain("even");
  });
});

describe("checkpointToLlamaStateDict", () => {
  it("maps to exact Llama names, splits wqkv, omits lm_head when tied", async () => {
    const artifacts = await tinyByteBpeArtifacts();
    const cfg = llamaConfig(artifacts.vocabSize);
    const state = makeState(cfg, artifacts, /*tied*/ true);
    const tensors = checkpointToLlamaStateDict(state);
    const byName = new Map(tensors.map((t) => [t.name, t]));

    const E = cfg.nEmbd;
    // Global.
    expect(byName.has("model.embed_tokens.weight")).toBe(true);
    expect(byName.get("model.embed_tokens.weight")!.shape).toEqual([cfg.vocabSize, E]);
    expect(byName.has("model.norm.weight")).toBe(true);
    // Tied ⇒ no lm_head.
    expect(byName.has("lm_head.weight")).toBe(false);

    for (let i = 0; i < cfg.nLayer; i++) {
      for (const proj of ["q_proj", "k_proj", "v_proj", "o_proj"]) {
        const t = byName.get(`model.layers.${i}.self_attn.${proj}.weight`)!;
        expect(t, `${proj} present`).toBeTruthy();
        expect(t.shape).toEqual([E, E]);
      }
      for (const proj of ["gate_proj", "up_proj", "down_proj"]) {
        expect(byName.has(`model.layers.${i}.mlp.${proj}.weight`), `${proj} present`).toBe(true);
      }
      expect(byName.has(`model.layers.${i}.input_layernorm.weight`)).toBe(true);
      expect(byName.has(`model.layers.${i}.post_attention_layernorm.weight`)).toBe(true);
    }

    // q/k/v are the three contiguous [E,E] blocks of wqkv, in order.
    const wqkv = state.params["layer.0.attn.wqkv"].data as unknown as Float32Array;
    const eE = E * E;
    const q = byName.get("model.layers.0.self_attn.q_proj.weight")!.data;
    const k = byName.get("model.layers.0.self_attn.k_proj.weight")!.data;
    const vv = byName.get("model.layers.0.self_attn.v_proj.weight")!.data;
    expect(Array.from(q)).toEqual(Array.from(wqkv.subarray(0, eE)));
    expect(Array.from(k)).toEqual(Array.from(wqkv.subarray(eE, 2 * eE)));
    expect(Array.from(vv)).toEqual(Array.from(wqkv.subarray(2 * eE, 3 * eE)));
  });

  it("emits lm_head + tie_word_embeddings:false when untied", async () => {
    const artifacts = await tinyByteBpeArtifacts();
    const cfg = llamaConfig(artifacts.vocabSize);
    const state = makeState(cfg, artifacts, /*tied*/ false);
    const names = new Set(checkpointToLlamaStateDict(state).map((t) => t.name));
    expect(names.has("lm_head.weight")).toBe(true);
    expect(buildLlamaConfig(state).tie_word_embeddings).toBe(false);
  });
});

describe("buildLlamaConfig / generation_config", () => {
  it("produces the expected llama fields", async () => {
    const artifacts = await tinyByteBpeArtifacts();
    const cfg = llamaConfig(artifacts.vocabSize);
    const state = makeState(cfg, artifacts, true);
    const c = buildLlamaConfig(state) as Record<string, any>;
    expect(c.architectures).toEqual(["LlamaForCausalLM"]);
    expect(c.model_type).toBe("llama");
    expect(c.hidden_size).toBe(32);
    expect(c.num_attention_heads).toBe(4);
    expect(c.num_key_value_heads).toBe(4);
    expect(c.head_dim).toBe(8);
    expect(c.hidden_act).toBe("silu");
    expect(c.rms_norm_eps).toBe(1e-5);
    expect(c.rope_theta).toBe(10000);
    expect(c.tie_word_embeddings).toBe(true);
    expect(c.attention_bias).toBe(false);
    expect(c.mlp_bias).toBe(false);
    expect(c.intermediate_size).toBe(llamaIntermediateSize(cfg));
    const eos = resolveBosEosId(artifacts);
    expect(c.eos_token_id).toBe(eos);
    expect(c.bos_token_id).toBe(eos);

    const g = buildGenerationConfig(state) as Record<string, any>;
    expect(g.do_sample).toBe(false);
    expect(g.eos_token_id).toBe(eos);
    expect(g.pad_token_id).toBe(eos);
  });

  it("bos/eos resolves to the <|end_of_text|> id", async () => {
    const artifacts = await tinyByteBpeArtifacts();
    const eot = artifacts.vocab.indexOf("<|end_of_text|>");
    expect(eot).toBeGreaterThanOrEqual(0);
    expect(resolveBosEosId(artifacts)).toBe(eot);
  });
});

describe("exportHfModel (full repo)", () => {
  it("writes weights + config + tokenizer files, all names present", async () => {
    const artifacts = await tinyByteBpeArtifacts();
    const cfg = llamaConfig(artifacts.vocabSize);
    const state = makeState(cfg, artifacts, true);
    const dir = mkdtempSync(join(tmpdir(), "alpha-hf-"));
    const written = await exportHfModel(state, dir);

    for (const f of ["model.safetensors", "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]) {
      expect(existsSync(join(dir, f)), `${f} written`).toBe(true);
    }
    expect(written.length).toBeGreaterThanOrEqual(6);

    // Re-parse the safetensors: every expected tensor name is present, F32.
    const parsed = parseSafetensors(readFileSync(join(dir, "model.safetensors")));
    expect(parsed.tensors["model.embed_tokens.weight"].dtype).toBe("F32");
    expect(parsed.tensors["model.layers.1.self_attn.q_proj.weight"].shape).toEqual([cfg.nEmbd, cfg.nEmbd]);
    expect(parsed.tensors["lm_head.weight"]).toBeUndefined(); // tied

    const configJson = JSON.parse(readFileSync(join(dir, "config.json"), "utf-8"));
    expect(configJson.architectures).toEqual(["LlamaForCausalLM"]);
  });

  it("throws on a non-Llama checkpoint", async () => {
    const artifacts = await tinyByteBpeArtifacts();
    const gpt2Cfg: ModelConfig = { ...llamaConfig(artifacts.vocabSize), normType: "layernorm", posEnc: "learned", ffnActivation: "gelu", tieEmbeddings: false };
    const state = makeState(gpt2Cfg, artifacts, false);
    const dir = mkdtempSync(join(tmpdir(), "alpha-hf-"));
    await expect(exportHfModel(state, dir)).rejects.toThrow(/not exportable as LlamaForCausalLM/);
  });
});
