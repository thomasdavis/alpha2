/**
 * Command: alpha export-hf
 *
 * Convert an ALPH checkpoint of an alpha_llama-form model (RMSNorm + RoPE +
 * SwiGLU + tied embeddings) into a standard, zero-custom-code HuggingFace
 * `LlamaForCausalLM` repo: model.safetensors + config.json +
 * generation_config.json + tokenizer.json + tokenizer_config.json +
 * chat_template.jinja.
 *
 * Usage:
 *   alpha export-hf --checkpoint=runs/.../checkpoint-1000.json --out=hf/alpha-60m-base
 */
import { Effect } from "effect";
import { parseKV, requireArg } from "../parse.js";
import { FileCheckpoint, exportHfModel, llamaFormViolations } from "@alpha/train";

export async function exportHfCmd(args: string[]): Promise<void> {
  const kv = parseKV(args);
  const checkpointPath = requireArg(kv, "checkpoint", "path to ALPH checkpoint");
  const outDir = requireArg(kv, "out", "output directory for the HF model repo");

  const state = await Effect.runPromise(new FileCheckpoint().load(checkpointPath));

  const violations = llamaFormViolations(state.modelConfig);
  if (violations.length > 0) {
    console.error(`Error: checkpoint at ${checkpointPath} is not Llama-form:`);
    for (const v of violations) console.error(`  - ${v}`);
    console.error(
      `\nTrain a Llama-form model with:\n` +
        `  alpha train ... --activation=swiglu --normType=rmsnorm --posEnc=rope --tieEmbeddings=true --tokenizer=bpe-byte-12k`,
    );
    process.exit(1);
  }

  const cfg = state.modelConfig;
  console.log(`Exporting HF LlamaForCausalLM from step ${state.step}`);
  console.log(
    `Model: ${cfg.nLayer}L ${cfg.nEmbd}D ${cfg.nHead}H headDim=${cfg.nEmbd / cfg.nHead} ` +
      `vocab=${cfg.vocabSize} block=${cfg.blockSize} rope_theta=${cfg.ropeTheta ?? 10000} ` +
      `tied=${!(state.params as Record<string, unknown>)["lmHead"]}`,
  );

  const written = await exportHfModel(state, outDir);
  console.log(`Wrote ${written.length} files to ${outDir}:`);
  for (const p of written) console.log(`  ${p}`);
  console.log(
    `\nLoad it anywhere with zero custom code:\n` +
      `  from transformers import pipeline\n` +
      `  pipe = pipeline("text-generation", "${outDir}")`,
  );
}
