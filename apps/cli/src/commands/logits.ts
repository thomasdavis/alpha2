/**
 * Command: alpha logits
 *
 * Dump next-token logits + top-1 for a set of prompts, computed via Alpha's own
 * cpu_ref autograd forward pass (NOT the fast inference engine). This is the
 * Alpha side of the G3 golden-token test (scripts/verify_hf_export.py): the
 * Python side loads the exported safetensors in `transformers`, feeds the SAME
 * token ids, and compares logits position-by-position.
 *
 * Usage:
 *   alpha logits --checkpoint=runs/.../checkpoint-30.json --prompt-file=prompts.txt --json --out=alpha_logits.json
 *   alpha logits --checkpoint=... --prompt="Hello world"        # human summary
 *
 * --prompt-file: one prompt per line (blank lines skipped), OR a JSON array of strings.
 * --json:        emit the full logits JSON (to --out=<path> if given, else stdout).
 */
import { Effect } from "effect";
import { parseKV, requireArg, strArg, boolArg } from "../parse.js";
import { resolveBackend, resolveRng } from "../resolve.js";
import { FileCheckpoint, restoreParams } from "@alpha/train";
import { initGPT, gptForward } from "@alpha/model";
import { Tape } from "@alpha/autograd";
import { tokenizerFromArtifacts } from "@alpha/tokenizers";
import type { TensorData } from "@alpha/core";

interface PromptLogits {
  prompt: string;
  tokens: number[];
  /** [T][V] next-token logits, one row per input position. */
  logits: number[][];
  /** [T] argmax over each position's logits. */
  top1: number[];
}

export async function logitsCmd(args: string[]): Promise<void> {
  const kv = parseKV(args);
  const checkpointPath = requireArg(kv, "checkpoint", "path to ALPH checkpoint");
  const promptFile = kv["prompt-file"];
  const inlinePrompt = kv["prompt"];
  const asJson = boolArg(kv, "json", false);
  const outPath = kv["out"];
  const backendName = strArg(kv, "backend", "cpu_ref");

  // Diagnostics go to stderr so --json stdout stays clean/parseable.
  const log = (msg: string) => process.stderr.write(msg + "\n");

  let prompts: string[];
  if (promptFile) {
    const fs = await import("node:fs/promises");
    const raw = (await fs.readFile(promptFile, "utf-8")).trim();
    prompts = raw.startsWith("[")
      ? (JSON.parse(raw) as string[])
      : raw.split("\n").map((l) => l.replace(/\r$/, "")).filter((l) => l.length > 0);
  } else if (inlinePrompt !== undefined) {
    prompts = [inlinePrompt];
  } else {
    throw new Error("provide --prompt-file=<path> or --prompt=<text>");
  }

  const state = await Effect.runPromise(new FileCheckpoint().load(checkpointPath));
  if (!state.tokenizerArtifacts) {
    throw new Error("checkpoint has no tokenizer artifacts");
  }

  const backend = resolveBackend(backendName);
  const rng = resolveRng(state.rngState ?? 42);
  const tokenizer = tokenizerFromArtifacts(state.tokenizerArtifacts);
  const modelConfig = state.modelConfig;
  const V = modelConfig.vocabSize;

  const params = initGPT(modelConfig, backend, rng as never);
  restoreParams(params, state.params);

  log(`checkpoint step ${state.step} | ${modelConfig.nLayer}L ${modelConfig.nEmbd}D ${modelConfig.nHead}H vocab=${V} backend=${backend.name}`);

  const results: PromptLogits[] = [];
  for (const prompt of prompts) {
    const ids = Array.from(tokenizer.encode(prompt));
    const T = ids.length;
    if (T === 0) {
      log(`skip empty prompt: ${JSON.stringify(prompt)}`);
      results.push({ prompt, tokens: [], logits: [], top1: [] });
      continue;
    }
    const tokens: TensorData = { shape: [1, T], dtype: "i32", data: new Int32Array(ids) };
    const tape = new Tape();
    const { logits } = gptForward(modelConfig, params, backend, tape, tokens, undefined, false);
    // cpu_ref returns a plain Float32Array; logits shape [1, T, V].
    const arr = logits.data.data as Float32Array;
    const perPos: number[][] = [];
    const top1: number[] = [];
    for (let t = 0; t < T; t++) {
      const base = t * V;
      const row = new Array<number>(V);
      let best = 0;
      let bestVal = -Infinity;
      for (let j = 0; j < V; j++) {
        const val = arr[base + j];
        row[j] = val;
        if (val > bestVal) {
          bestVal = val;
          best = j;
        }
      }
      perPos.push(row);
      top1.push(best);
    }
    results.push({ prompt, tokens: ids, logits: perPos, top1 });
    log(`prompt ${JSON.stringify(prompt.slice(0, 48))} → ${T} tokens, top1=[${top1.join(",")}]`);
  }

  if (asJson) {
    const payload = {
      config: {
        vocabSize: V,
        nLayer: modelConfig.nLayer,
        nEmbd: modelConfig.nEmbd,
        nHead: modelConfig.nHead,
        blockSize: modelConfig.blockSize,
        ropeTheta: modelConfig.ropeTheta ?? 10000,
        normType: modelConfig.normType ?? "layernorm",
        posEnc: modelConfig.posEnc ?? "learned",
        tieEmbeddings: !(state.params as Record<string, unknown>)["lmHead"],
      },
      prompts: results,
    };
    const json = JSON.stringify(payload);
    if (outPath) {
      const fs = await import("node:fs/promises");
      await fs.writeFile(outPath, json, "utf-8");
      log(`wrote logits JSON → ${outPath} (${(json.length / 1024).toFixed(0)} KiB)`);
    } else {
      process.stdout.write(json);
    }
  } else {
    for (const r of results) {
      console.log(`prompt: ${JSON.stringify(r.prompt)}  tokens=${r.tokens.length}  top1=[${r.top1.join(",")}]`);
    }
  }
}
