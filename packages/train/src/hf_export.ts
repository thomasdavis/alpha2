/**
 * HuggingFace export: turn an ALPH checkpoint of an **alpha_llama-form** model
 * (RMSNorm + RoPE + SwiGLU + tied embeddings) into a standard, zero-custom-code
 * `LlamaForCausalLM` repo that `transformers` can load on any machine:
 *
 *   model.safetensors        — weights, EXACT Llama state-dict names, F32.
 *   config.json              — architectures:["LlamaForCausalLM"], llama fields.
 *   generation_config.json   — greedy defaults, bos/eos ids.
 *   tokenizer.json           — (from @alpha/tokenizers, Task B) HF fast tokenizer.
 *   tokenizer_config.json    — PreTrainedTokenizerFast wrapper + chat template.
 *   chat_template.jinja
 *
 * ── Weight-orientation verification (the load-bearing correctness argument) ──
 * Alpha stores every linear weight as `[out_features, in_features]` and applies
 * it with `matmulTransposed(x, W)` which computes `x @ W^T` (see
 * packages/autograd/src/ops.ts::matmulTransposed — "B is stored as [N, K] but
 * used as [K, N]", i.e. Wᵀ). That is EXACTLY PyTorch `nn.Linear`'s convention:
 * `weight.shape == [out, in]`, `y = x @ weightᵀ`. Therefore **no transpose is
 * applied on export** — Alpha's `[out, in]` buffers are already the exact bytes
 * `transformers` expects for `q_proj/k_proj/v_proj/o_proj/gate/up/down_proj` and
 * `embed_tokens`.
 *
 * ── RoPE convention (why q/k need NO permutation) ──
 * The HF Llama conversion script famously *permutes* q_proj/k_proj because Meta's
 * original checkpoints use the interleaved (adjacent-pair) rotary layout while HF
 * uses `rotate_half` (split the head dim in half). Alpha's `rope` op
 * (packages/autograd/src/ops.ts) implements the HF `rotate_half` convention
 * NATIVELY — comment there: "the HF-Llama `rotate_half` convention is used
 * EXACTLY (this matters for safetensors export compatibility)" — with identical
 * frequencies `inv_freq_i = θ^(-2i/D)` and identical `[num_heads, head_dim]`
 * reshape ordering. So the fused `wqkv` splits straight into q/k/v with **no
 * permutation** (validated end-to-end by the golden-token test).
 */
import type { CheckpointState, ModelConfig, TokenizerArtifacts } from "@alpha/core";
import { writeHfTokenizer } from "@alpha/tokenizers";

// ── safetensors writer ──────────────────────────────────────────────────────

/** One named F32 tensor destined for the safetensors file. */
export interface SafeTensor {
  readonly name: string;
  readonly shape: readonly number[];
  /** Row-major C-contiguous f32 data (may be a subarray view of a larger buffer). */
  readonly data: Float32Array;
}

/** Coerce a checkpoint tensor payload (Float32Array from binary, number[] from
 *  legacy JSON) into a Float32Array without copying when already f32. */
function asF32(d: Float32Array | number[]): Float32Array {
  return d instanceof Float32Array ? d : new Float32Array(d);
}

/**
 * Write the safetensors container.
 *
 * Format (https://github.com/huggingface/safetensors):
 *   [ 8 bytes  ] little-endian u64 = JSON header length
 *   [ N bytes  ] UTF-8 JSON header, space-padded to an 8-byte boundary
 *   [ remainder] concatenated raw tensor bytes, in header order
 *
 * The header is `{ "__metadata__": {...}, "<name>": {dtype, shape, data_offsets}, ... }`
 * where `data_offsets` are `[begin, end)` byte ranges into the data blob. We emit
 * `dtype:"F32"` and native little-endian bytes (x86/ARM = LE; safetensors is LE).
 */
export async function writeSafetensors(
  filePath: string,
  tensors: readonly SafeTensor[],
  metadata: Record<string, string> = { format: "pt" },
): Promise<void> {
  const header: Record<string, unknown> = { __metadata__: metadata };
  let offset = 0;
  for (const t of tensors) {
    const byteLen = t.data.byteLength;
    header[t.name] = {
      dtype: "F32",
      shape: [...t.shape],
      data_offsets: [offset, offset + byteLen],
    };
    offset += byteLen;
  }

  let headerJson = JSON.stringify(header);
  // Pad the header with spaces so the tensor data starts 8-byte aligned (the
  // standard safetensors serializer does the same — improves mmap loads).
  const enc = new TextEncoder();
  let headerBytes = enc.encode(headerJson);
  const pad = (8 - (headerBytes.length % 8)) % 8;
  if (pad > 0) {
    headerJson += " ".repeat(pad);
    headerBytes = enc.encode(headerJson);
  }

  const fs = await import("node:fs/promises");
  const pathMod = await import("node:path");
  await fs.mkdir(pathMod.dirname(filePath), { recursive: true });

  const handle = await fs.open(filePath, "w");
  try {
    const lenBuf = Buffer.alloc(8);
    lenBuf.writeBigUInt64LE(BigInt(headerBytes.length), 0);
    await handle.write(lenBuf);
    await handle.write(Buffer.from(headerBytes));
    for (const t of tensors) {
      // Buffer.from(view.buffer, byteOffset, byteLength) writes exactly this
      // tensor's slice — correct even when `data` is a subarray of wqkv.
      await handle.write(Buffer.from(t.data.buffer, t.data.byteOffset, t.data.byteLength));
    }
  } finally {
    await handle.close();
  }
}

// ── llama-form validation ───────────────────────────────────────────────────

/** Return the list of reasons a config is NOT exportable as `LlamaForCausalLM`
 *  (empty ⇒ ok). Used by both the exporter and its tests. */
export function llamaFormViolations(cfg: ModelConfig): string[] {
  const errs: string[] = [];
  if ((cfg.normType ?? "layernorm") !== "rmsnorm") {
    errs.push(`normType must be "rmsnorm" (got "${cfg.normType ?? "layernorm"}")`);
  }
  if ((cfg.posEnc ?? "learned") !== "rope") {
    errs.push(`posEnc must be "rope" (got "${cfg.posEnc ?? "learned"}")`);
  }
  if ((cfg.ffnActivation ?? "gelu") !== "swiglu") {
    errs.push(`ffnActivation must be "swiglu" (got "${cfg.ffnActivation ?? "gelu"}") — Llama uses SwiGLU`);
  }
  if (cfg.softCap !== undefined && cfg.softCap > 0) {
    errs.push(`softCap must be OFF for Llama export (got ${cfg.softCap}; no Llama equivalent)`);
  }
  if (cfg.nEmbd % cfg.nHead !== 0) {
    errs.push(`nEmbd (${cfg.nEmbd}) must be divisible by nHead (${cfg.nHead})`);
  }
  if ((cfg.nEmbd / cfg.nHead) % 2 !== 0) {
    errs.push(`head_dim (${cfg.nEmbd / cfg.nHead}) must be even for RoPE`);
  }
  return errs;
}

/** SwiGLU intermediate size, mirroring initGPT's default when ffnDim is unset. */
export function llamaIntermediateSize(cfg: ModelConfig): number {
  if (cfg.ffnDim && cfg.ffnDim > 0) return cfg.ffnDim;
  return (cfg.ffnActivation ?? "gelu") === "swiglu"
    ? Math.ceil((8 / 3) * cfg.nEmbd / 64) * 64
    : 4 * cfg.nEmbd;
}

/** bos/eos ids = the `<|end_of_text|>` token id (matches tokenizer_config, which
 *  sets bos_token = eos_token = `<|end_of_text|>`). Falls back to the last id. */
export function resolveBosEosId(artifacts?: TokenizerArtifacts): number {
  if (!artifacts) return 0;
  const idx = artifacts.vocab.indexOf("<|end_of_text|>");
  return idx >= 0 ? idx : Math.max(0, artifacts.vocabSize - 1);
}

// ── ALPH params → Llama state dict ──────────────────────────────────────────

/**
 * Map ALPH checkpoint params to the EXACT Llama `state_dict` names, splitting the
 * fused `wqkv` `[3E, E]` into `q/k/v_proj` `[E, E]` each. No transposes (see the
 * module header for the `matmulTransposed`-convention proof). No `lm_head` when
 * embeddings are tied (config sets `tie_word_embeddings: true` and transformers
 * reuses `embed_tokens`).
 */
export function checkpointToLlamaStateDict(state: CheckpointState): SafeTensor[] {
  const cfg = state.modelConfig;
  const E = cfg.nEmbd;
  const L = cfg.nLayer;
  const params = state.params as Record<string, { shape: number[]; data: Float32Array | number[] }>;

  const get = (name: string): { shape: number[]; data: Float32Array } => {
    const p = params[name];
    if (!p) throw new Error(`checkpoint is missing required param "${name}" for Llama export`);
    return { shape: [...p.shape], data: asF32(p.data) };
  };

  const tensors: SafeTensor[] = [];

  // Token embeddings [vocab, hidden] → model.embed_tokens.weight (identical layout).
  const wte = get("wte");
  tensors.push({ name: "model.embed_tokens.weight", shape: wte.shape, data: wte.data });

  // Final RMSNorm weight → model.norm.weight.
  const lnF = get("lnF.weight");
  tensors.push({ name: "model.norm.weight", shape: lnF.shape, data: lnF.data });

  for (let i = 0; i < L; i++) {
    // Norms.
    const ln1 = get(`layer.${i}.ln1.weight`);
    tensors.push({ name: `model.layers.${i}.input_layernorm.weight`, shape: ln1.shape, data: ln1.data });
    const ln2 = get(`layer.${i}.ln2.weight`);
    tensors.push({ name: `model.layers.${i}.post_attention_layernorm.weight`, shape: ln2.shape, data: ln2.data });

    // Attention: split fused wqkv [3E, E] → q/k/v [E, E].
    // Row-major [3E, E]: rows [0,E)=q, [E,2E)=k, [2E,3E)=v (matches sliceQkv,
    // which slices the projection OUTPUT columns in the same q,k,v order). Each
    // is a contiguous E*E slice → a zero-copy subarray view.
    const wqkv = get(`layer.${i}.attn.wqkv`);
    if (wqkv.shape.length !== 2 || wqkv.shape[0] !== 3 * E || wqkv.shape[1] !== E) {
      throw new Error(`layer.${i}.attn.wqkv shape ${JSON.stringify(wqkv.shape)} != [${3 * E}, ${E}]`);
    }
    const eE = E * E;
    tensors.push({ name: `model.layers.${i}.self_attn.q_proj.weight`, shape: [E, E], data: wqkv.data.subarray(0, eE) });
    tensors.push({ name: `model.layers.${i}.self_attn.k_proj.weight`, shape: [E, E], data: wqkv.data.subarray(eE, 2 * eE) });
    tensors.push({ name: `model.layers.${i}.self_attn.v_proj.weight`, shape: [E, E], data: wqkv.data.subarray(2 * eE, 3 * eE) });

    const wo = get(`layer.${i}.attn.wo`);
    tensors.push({ name: `model.layers.${i}.self_attn.o_proj.weight`, shape: wo.shape, data: wo.data });

    // SwiGLU MLP: fc_gate→gate_proj, fc_up→up_proj, fc_proj→down_proj.
    const gate = get(`layer.${i}.mlp.fc_gate`);
    tensors.push({ name: `model.layers.${i}.mlp.gate_proj.weight`, shape: gate.shape, data: gate.data });
    const up = get(`layer.${i}.mlp.fc_up`);
    tensors.push({ name: `model.layers.${i}.mlp.up_proj.weight`, shape: up.shape, data: up.data });
    const down = get(`layer.${i}.mlp.fc_proj`);
    tensors.push({ name: `model.layers.${i}.mlp.down_proj.weight`, shape: down.shape, data: down.data });
  }

  // Untied models keep an explicit lm_head; tied models omit it (the #1 silent-
  // garbage pitfall is writing lm_head AND tie_word_embeddings:true).
  if (params["lmHead"]) {
    const lm = get("lmHead");
    tensors.push({ name: "lm_head.weight", shape: lm.shape, data: lm.data });
  }

  return tensors;
}

// ── config.json / generation_config.json ────────────────────────────────────

/** Build the `LlamaForCausalLM` config.json object. */
export function buildLlamaConfig(state: CheckpointState): Record<string, unknown> {
  const cfg = state.modelConfig;
  const H = cfg.nHead;
  const headDim = cfg.nEmbd / H;
  const tied = !(state.params as Record<string, unknown>)["lmHead"];
  const eosId = resolveBosEosId(state.tokenizerArtifacts);
  return {
    architectures: ["LlamaForCausalLM"],
    model_type: "llama",
    vocab_size: cfg.vocabSize,
    hidden_size: cfg.nEmbd,
    intermediate_size: llamaIntermediateSize(cfg),
    num_hidden_layers: cfg.nLayer,
    num_attention_heads: H,
    // MHA: kv heads == attention heads (GOAL: GQA skipped).
    num_key_value_heads: H,
    head_dim: headDim,
    hidden_act: "silu",
    max_position_embeddings: cfg.blockSize,
    // Must match the rmsNorm op's eps (gpt.ts applyNorm passes 1e-5 everywhere).
    rms_norm_eps: 1e-5,
    rope_theta: cfg.ropeTheta ?? 10000,
    rope_scaling: null,
    tie_word_embeddings: tied,
    attention_bias: false,
    attention_dropout: 0.0,
    mlp_bias: false,
    initializer_range: 0.02,
    torch_dtype: "float32",
    use_cache: true,
    pretraining_tp: 1,
    bos_token_id: eosId,
    eos_token_id: eosId,
  };
}

/** Build generation_config.json (greedy defaults; the D3 eval decodes greedily). */
export function buildGenerationConfig(state: CheckpointState): Record<string, unknown> {
  const eosId = resolveBosEosId(state.tokenizerArtifacts);
  return {
    bos_token_id: eosId,
    eos_token_id: eosId,
    pad_token_id: eosId,
    do_sample: false,
    max_new_tokens: 256,
  };
}

// ── orchestration ───────────────────────────────────────────────────────────

/**
 * Export a full HF model repo from an in-memory checkpoint state to `outDir`.
 * Validates the checkpoint is Llama-form first (throws with all reasons).
 * Returns the list of written file paths.
 */
export async function exportHfModel(state: CheckpointState, outDir: string): Promise<string[]> {
  const violations = llamaFormViolations(state.modelConfig);
  if (violations.length > 0) {
    throw new Error(
      `checkpoint is not exportable as LlamaForCausalLM:\n  - ${violations.join("\n  - ")}`,
    );
  }

  const fs = await import("node:fs/promises");
  const pathMod = await import("node:path");
  await fs.mkdir(outDir, { recursive: true });
  const written: string[] = [];

  // 1) Weights.
  const tensors = checkpointToLlamaStateDict(state);
  const stPath = pathMod.join(outDir, "model.safetensors");
  await writeSafetensors(stPath, tensors);
  written.push(stPath);

  // 2) config.json + generation_config.json.
  const cfgPath = pathMod.join(outDir, "config.json");
  await fs.writeFile(cfgPath, JSON.stringify(buildLlamaConfig(state), null, 2) + "\n", "utf-8");
  written.push(cfgPath);
  const gcPath = pathMod.join(outDir, "generation_config.json");
  await fs.writeFile(gcPath, JSON.stringify(buildGenerationConfig(state), null, 2) + "\n", "utf-8");
  written.push(gcPath);

  // 3) Tokenizer files (Task B's HF exporter). Requires byte-level BPE artifacts.
  const artifacts = state.tokenizerArtifacts;
  if (!artifacts) {
    throw new Error("checkpoint has no tokenizerArtifacts — cannot emit tokenizer.json");
  }
  if (artifacts.type !== "byte_bpe" || artifacts.byteVocab !== true) {
    throw new Error(
      `HF export requires byte-level BPE artifacts (type:"byte_bpe"); got "${artifacts.type}". ` +
        `Train with --tokenizer=bpe-byte-12k (or bpe-byte-4k).`,
    );
  }
  const tokFiles = await writeHfTokenizer(outDir, artifacts);
  written.push(...tokFiles);

  return written;
}
