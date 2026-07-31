/**
 * sft-masking — assistant-only SFT loss masking (GOAL.md Stage 4).
 *
 * Three concerns, on cpu_ref only (no GPU needed):
 *   1. buildSftExample / SftDataLoader produce the RIGHT per-position mask:
 *      user tokens + role markers + padding are 0, assistant content (incl. the
 *      terminating <|end_of_text|>) is 1, exact span boundaries hold across
 *      multi-turn conversations, and the word "assistant" appearing IN content
 *      (not as the atomic marker) does NOT flip the state.
 *   2. Masked-out positions get EXACTLY-zero embedding-table gradient — a token
 *      that appears only at masked-out (user/padding) positions with no later
 *      live position receives no learning signal at all.
 *   3. Trainer smoke: 5 SFT steps on a tiny fixture file — loss finite and
 *      decreasing, no NaN even when a batch contains all-padding rows.
 */
import { describe, it, expect, afterEach } from "vitest";
import { Effect } from "effect";
import { CpuRefBackend } from "@alpha/tensor";
import { SeededRng, type ModelConfig, type TensorData, type Tokenizer, type TokenizerArtifacts } from "@alpha/core";
import { Tape } from "@alpha/autograd";
import { initGPT, gptForward, collectParamEntries } from "@alpha/model";
import {
  SftDataLoader, buildSftExample, resolveChatSpecialIds, loadSftExamples, splitSftExamples,
  train, AdamW,
  CHAT_USER_TOKEN, CHAT_ASSISTANT_TOKEN, CHAT_EOT_TOKEN,
} from "@alpha/train";

// ── A deterministic chat tokenizer: 3 atomic specials + 256 byte tokens. ─────
// ids: 0=<|user|>, 1=<|assistant|>, 2=<|end_of_text|>, byte b → b+3. vocab 259.
const SPECIALS: [string, number][] = [
  [CHAT_USER_TOKEN, 0],
  [CHAT_ASSISTANT_TOKEN, 1],
  [CHAT_EOT_TOKEN, 2],
];

class MockChatTokenizer implements Tokenizer {
  readonly name = "mock-chat";
  readonly vocabSize = 259;
  encode(text: string): Int32Array {
    const out: number[] = [];
    let i = 0;
    while (i < text.length) {
      const sp = SPECIALS.find(([s]) => text.startsWith(s, i));
      if (sp) { out.push(sp[1]); i += sp[0].length; continue; }
      let j = i;
      while (j < text.length && !SPECIALS.some(([s]) => text.startsWith(s, j))) j++;
      const bytes = Buffer.from(text.slice(i, j), "utf-8");
      for (const b of bytes) out.push(b + 3);
      i = j;
    }
    return Int32Array.from(out);
  }
  decode(tokens: ArrayLike<number>): string {
    const bytes: number[] = [];
    let out = "";
    const flush = (): void => { if (bytes.length) { out += Buffer.from(bytes).toString("utf-8"); bytes.length = 0; } };
    for (let i = 0; i < tokens.length; i++) {
      const id = tokens[i];
      const sp = SPECIALS.find(([, s]) => s === id);
      if (sp) { flush(); out += sp[0]; } else { bytes.push(id - 3); }
    }
    flush();
    return out;
  }
  build(_input: string): Effect.Effect<TokenizerArtifacts, never> {
    return Effect.succeed({ type: this.name, vocabSize: this.vocabSize, vocab: [] });
  }
}

const tok = new MockChatTokenizer();
const ids = resolveChatSpecialIds(tok);
const byte = (ch: string): number => ch.charCodeAt(0) + 3;

// ── 1. Mask construction ─────────────────────────────────────────────────────

describe("SFT: role-mask construction (buildSftExample)", () => {
  it("single turn: markers/user 0, assistant content + terminating EOS 1", () => {
    const ex = buildSftExample("<|user|>hi<|assistant|>ok<|end_of_text|>", tok, ids);
    // tokens: [user, h, i, assist, o, k, eot]
    expect(Array.from(ex.tokens)).toEqual([ids.userId, byte("h"), byte("i"), ids.assistantId, byte("o"), byte("k"), ids.eotId]);
    expect(Array.from(ex.roleMask)).toEqual([0, 0, 0, 0, 1, 1, 1]);
  });

  it("multi-turn: each assistant span (incl. its EOS) is 1, everything else 0", () => {
    const ex = buildSftExample("<|user|>a<|assistant|>b<|end_of_text|><|user|>c<|assistant|>d<|end_of_text|>", tok, ids);
    // tokens: [user,a,assist,b,eot,user,c,assist,d,eot]
    expect(Array.from(ex.roleMask)).toEqual([0, 0, 0, 1, 1, 0, 0, 0, 1, 1]);
  });

  it("specials-in-content: the WORD 'assistant' in a user turn stays masked 0", () => {
    // "assistant" here is ordinary bytes, NOT the atomic <|assistant|> marker.
    const ex = buildSftExample("<|user|>assistant here<|assistant|>hi<|end_of_text|>", tok, ids);
    // Only ONE role marker (the real <|assistant|>) exists → exactly one asst span.
    const asstMarkerPositions = Array.from(ex.tokens).filter((t) => t === ids.assistantId).length;
    expect(asstMarkerPositions).toBe(1);
    // Everything up to and including the marker is user context (mask 0); only
    // "hi" + eot are assistant content.
    const markerIdx = Array.from(ex.tokens).indexOf(ids.assistantId);
    for (let i = 0; i <= markerIdx; i++) expect(ex.roleMask[i]).toBe(0);
    for (let i = markerIdx + 1; i < ex.tokens.length; i++) expect(ex.roleMask[i]).toBe(1);
  });

  it("empty assistant turn: only the terminating EOS is masked in", () => {
    const ex = buildSftExample("<|user|>q<|assistant|><|end_of_text|>", tok, ids);
    // tokens: [user, q, assist, eot] → the eot right after the marker is the
    // (empty) assistant turn's EOS, still masked 1.
    expect(Array.from(ex.roleMask)).toEqual([0, 0, 0, 1]);
    expect(Array.from(ex.assistantContentSpans!)).toEqual([]);
  });

  it("records compact content spans per turn but never includes EOS", () => {
    const ex = buildSftExample("<|user|>a<|assistant|>bc<|end_of_text|><|user|>d<|assistant|>e<|end_of_text|>", tok, ids);
    expect(Array.from(ex.assistantContentSpans!)).toEqual([3, 5, 9, 10]);
  });
});

// ── 1b. Batch layout: mask aligns with the PREDICTED token ───────────────────

describe("SFT: SftDataLoader batch layout + mask alignment", () => {
  it("mask[i] equals the role of the target token; padding is 0/0", () => {
    const ex = buildSftExample("<|user|>hi<|assistant|>ok<|end_of_text|>", tok, ids); // len 7
    const T = 10;
    const loader = new SftDataLoader([ex], /*batch*/ 1, /*block*/ T);
    const batch = loader.nextBatch();
    const inputs = batch.inputs.data as Int32Array;
    const targets = batch.targets.data as Int32Array;
    const mask = batch.lossMask!.data as Float32Array;

    // pairs = min(T, L-1) = 6
    for (let i = 0; i < 6; i++) {
      expect(inputs[i]).toBe(ex.tokens[i]);
      expect(targets[i]).toBe(ex.tokens[i + 1]);
      expect(mask[i]).toBe(ex.roleMask[i + 1]); // weight of the PREDICTED token
    }
    // Expected mask = roleMask[1..6] = [0,0,0,1,1,1]
    expect(Array.from(mask.slice(0, 6))).toEqual([0, 0, 0, 1, 1, 1]);
    // Padding tail: id 0, mask 0.
    for (let i = 6; i < T; i++) { expect(inputs[i]).toBe(0); expect(targets[i]).toBe(0); expect(mask[i]).toBe(0); }

    // Decode the masked-in predicted tokens → the assistant content 'o','k' + EOS.
    const maskedInTargets = Array.from(targets.slice(0, 6)).filter((_, i) => mask[i] === 1);
    expect(tok.decode(maskedInTargets.filter((t) => t !== ids.eotId))).toBe("ok");
  });

  it("truncation: a conversation longer than blockSize fills every position", () => {
    const ex = buildSftExample("<|user|>abcd<|assistant|>efgh<|end_of_text|>", tok, ids); // len 11
    const T = 4;
    const loader = new SftDataLoader([ex], 1, T);
    const b = loader.nextBatch();
    const mask = b.lossMask!.data as Float32Array;
    for (let i = 0; i < T; i++) expect(mask[i]).toBe(ex.roleMask[i + 1]); // first T pairs
  });

  it("balances rows and emphasizes answer starts without boosting EOS", () => {
    const short = buildSftExample("<|user|>q<|assistant|>ab<|end_of_text|>", tok, ids);
    const long = buildSftExample("<|user|>q<|assistant|>abcdef<|end_of_text|>", tok, ids);
    const loader = new SftDataLoader([short, long], 2, 12, {
      balanceConversations: true,
      startTokenCount: 2,
      startTokenMultiplier: 4,
    });
    const mask = loader.nextBatch().lossMask!.data as Float32Array;
    const rows = [mask.slice(0, 12), mask.slice(12, 24)];
    for (const row of rows) expect(Array.from(row).reduce((sum, weight) => sum + weight, 0)).toBeCloseTo(1, 6);
    // Short row raw weights are [4,4,1] for a,b,EOS.
    const live = Array.from(rows[0]).filter((weight) => weight > 0);
    expect(live[0]).toBeCloseTo(4 / 9, 6);
    expect(live[1]).toBeCloseTo(4 / 9, 6);
    expect(live[2]).toBeCloseTo(1 / 9, 6);
  });
});

// ── 2. Exactly-zero embedding gradient for masked-out-only tokens ────────────

describe("SFT: masked-out positions → exactly-zero embedding gradient", () => {
  const B = new CpuRefBackend();
  const config: ModelConfig = {
    // Untied embeddings on purpose: isolates the INPUT-embedding path so a
    // token used only at a masked-out position (with no later live position)
    // gets a bit-exactly-zero wte gradient. (A tied lmHead would couple every
    // vocab row to the softmax at live positions.)
    vocabSize: 20, blockSize: 8, nLayer: 2, nEmbd: 16, nHead: 2, dropout: 0, ffnActivation: "gelu",
    tieEmbeddings: false,
  };

  it("a token appearing only at the final (masked-out) input position gets zero grad", () => {
    const params = initGPT(config, B, new SeededRng(20260722));
    const T = config.blockSize;
    const U = 19; // sentinel token: appears ONLY at the last input position
    const inputs = Int32Array.from([5, 6, 7, 8, 9, 10, 11, U]);
    const targets = Int32Array.from([6, 7, 8, 9, 10, 11, U, 12]);
    // Assistant span = positions 1,2 (masked in). Positions 6,7 (where U is
    // target / input) are masked out, and no live position follows them.
    const mask = Float32Array.from([0, 1, 1, 0, 0, 0, 0, 0]);

    const tape = new Tape();
    const res = gptForward(
      config, params, B, tape,
      { shape: [1, T], dtype: "i32", data: inputs },
      { shape: [1, T], dtype: "i32", data: targets },
      false, false, false, undefined, undefined,
      { shape: [1, T], dtype: "f32", data: mask },
    );
    expect(Number.isFinite((res.loss!.data.data as Float32Array)[0])).toBe(true);
    tape.backward(res.loss!, B);

    const wteGrad = params.wte.grad!.data as Float32Array;
    const nEmbd = config.nEmbd;
    // Row U: bit-exactly zero (no input-path signal, untied → no output coupling).
    for (let d = 0; d < nEmbd; d++) expect(wteGrad[U * nEmbd + d]).toBe(0);
    // Sanity: token 6 IS an input at masked-in position 1 → non-zero row.
    let nz6 = 0;
    for (let d = 0; d < nEmbd; d++) if (wteGrad[6 * nEmbd + d] !== 0) nz6++;
    expect(nz6).toBeGreaterThan(0);
  });
});

// ── 3. Trainer smoke: 5 SFT steps on a tiny fixture ──────────────────────────

describe("SFT: trainer smoke (5 steps, cpu_ref)", () => {
  const tmpDirs: string[] = [];
  const prevValFrac = process.env.ALPHA_SFT_VAL_FRACTION;
  afterEach(async () => {
    if (prevValFrac === undefined) delete process.env.ALPHA_SFT_VAL_FRACTION;
    else process.env.ALPHA_SFT_VAL_FRACTION = prevValFrac;
    const fs = await import("node:fs/promises");
    for (const d of tmpDirs) await fs.rm(d, { recursive: true, force: true }).catch(() => {});
  });

  it("loss finite + decreasing over 5 steps; no NaN with padded rows", async () => {
    const fs = await import("node:fs/promises");
    const os = await import("node:os");
    const path = await import("node:path");
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), "alpha-sft-"));
    tmpDirs.push(dir);

    // A tiny multi-turn chat corpus, one conversation per line. Short + varied
    // lengths so some rows are heavily padded (exercises the all-zero-mask path).
    const corpus = [
      "<|user|>hi<|assistant|>hello there<|end_of_text|>",
      "<|user|>what is two plus two<|assistant|>four<|end_of_text|>",
      "<|user|>name a color<|assistant|>blue<|end_of_text|><|user|>another<|assistant|>red<|end_of_text|>",
      "<|user|>say ok<|assistant|>ok<|end_of_text|>",
    ].join("\n") + "\n";
    const dataPath = path.join(dir, "chat.txt");
    await fs.writeFile(dataPath, corpus);

    process.env.ALPHA_SFT_VAL_FRACTION = "0"; // deterministic: no val split for the smoke

    const backend = new CpuRefBackend();
    const modelConfig: ModelConfig = {
      vocabSize: tok.vocabSize, blockSize: 24, nLayer: 2, nEmbd: 32, nHead: 2, dropout: 0, ffnActivation: "gelu",
    };
    const optimizer = new AdamW(backend, { lr: 5e-3, beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0.0 });
    const rng = new SeededRng(1234);
    const losses: number[] = [];

    await train({
      backend,
      tokenizer: tok,
      optimizer,
      rng,
      modelConfig,
      trainConfig: {
        iters: 5, batchSize: 2, lr: 5e-3, lrMin: 5e-4, warmupIters: 0,
        beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0.0, gradClip: 1.0,
        evalInterval: 1000, evalIters: 1, seed: 1234, backend: "cpu_ref",
        tokenizer: "mock-chat", optimizer: "adamw", logLevel: "info", logEvery: 1,
        trace: false, gradAccumSteps: 1, sampleInterval: 0, spikeThreshold: 0,
        embGradScale: 1.0, syncEvery: 0, gcEvery: 0, packed: false, symbio: false, symbioConfig: null,
      },
      dataPath,
      runDir: path.join(dir, "run"),
      sft: true,
      onStep: (m) => losses.push(m.loss),
    });

    expect(losses.length).toBe(5);
    for (const l of losses) expect(Number.isFinite(l), `loss non-finite: ${l}`).toBe(true);
    // Decreasing-ish: the last loss must be below the first (tiny corpus, it
    // memorizes quickly). Masked loss over a memorizable set drops fast.
    expect(losses[losses.length - 1]).toBeLessThan(losses[0]);
  });
});

// ── loadSftExamples + split (file round-trip) ────────────────────────────────

describe("SFT: loadSftExamples + splitSftExamples", () => {
  const tmp: string[] = [];
  afterEach(async () => {
    const fs = await import("node:fs/promises");
    for (const d of tmp) await fs.rm(d, { recursive: true, force: true }).catch(() => {});
  });

  it("streams one example per non-blank line; blank lines skipped", async () => {
    const fs = await import("node:fs/promises");
    const os = await import("node:os");
    const path = await import("node:path");
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), "alpha-sft-load-"));
    tmp.push(dir);
    const p = path.join(dir, "c.txt");
    await fs.writeFile(p,
      "<|user|>a<|assistant|>b<|end_of_text|>\n\n<|user|>c<|assistant|>d<|end_of_text|>\n");
    const examples = await loadSftExamples(p, tok);
    expect(examples.length).toBe(2);
    expect(Array.from(examples[0].roleMask)).toEqual([0, 0, 0, 1, 1]); // [user,a,assist,b,eot]

    const { train: tr, val } = splitSftExamples(examples, 0.5, 7);
    expect(tr.length + val.length).toBe(2); // doc-aware, deterministic
  });
});
