import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { AlphaLensAdapter } from "./adapter.js";

export interface BundleIdentity {
  readonly modelHfRepo: string;
  readonly modelRevision: string;
  readonly lensHfRepo: string;
  readonly publicRuntimeUrl?: string;
  readonly sourceRevision: string;
  readonly license: string;
  readonly hfFiles: Readonly<Record<string, string>>;
}

export async function writeBundleMetadata(
  adapter: AlphaLensAdapter,
  output: string,
  identity: BundleIdentity,
): Promise<Record<string, unknown>> {
  if (!/^[0-9a-f]{40}$/.test(identity.modelRevision)) throw new Error("modelRevision must be an immutable 40-character Hugging Face commit SHA");
  if (!/^[0-9a-f]{40}$/.test(identity.sourceRevision)) throw new Error("sourceRevision must be an immutable 40-character git commit SHA");
  const report = JSON.parse(await readFile(join(output, "fit-report.json"), "utf8")) as {
    corpus: { valid_prompt_count: number; maximum_sequence_length: number };
    fitting: { exported_dtype: string; source_sites: string[]; skip_first_positions: number; target_position_policy: string };
  };
  const description = adapter.describe();
  const sourceSites = report.fitting.source_sites;
  const siteById = new Map(description.sites.map((site) => [site.id, site]));
  const runtime = identity.publicRuntimeUrl?.replace(/\/+$/, "") || null;
  const manifest = {
    format: "blah-jacobian-lens",
    format_version: 1,
    platform: { consumer: "evals.blah.dev" },
    model: {
      repo_id: identity.modelHfRepo,
      revision: identity.modelRevision,
      checkpoint: `${description.checkpointSha256} (native step ${description.checkpointStep})`,
      framework: "alpha2",
      architecture: description.architecture,
      task: "causal-language-model",
      weights_fingerprint: description.weightsFingerprint,
      config_fingerprint: description.configFingerprint,
      tokenizer_fingerprint: description.tokenizerFingerprint,
      chat_template_fingerprint: description.chatTemplateFingerprint,
      hf_files: identity.hfFiles,
    },
    execution: runtime ? {
      mode: "remote_http",
      protocol: "blah-lens-http/1",
      entrypoint: "alpha lens serve",
      public_runtime_url: runtime,
      requires_remote_code: false,
      requires_auth: false,
    } : {
      mode: "precomputed_only",
      protocol: null,
      entrypoint: null,
      public_runtime_url: null,
      requires_remote_code: false,
      requires_auth: false,
    },
    sequence: {
      causal: true,
      activation_layout: "BTD",
      position_mapping: "token",
      grid_compatible: true,
      bos_policy: "No BOS is inserted by native encode; exported HF config aliases BOS to end_of_text but the chat template does not prepend it.",
      eos_policy: "The atomic <|end_of_text|> token terminates a full conversation; generation stops when it is emitted.",
    },
    target_site: {
      id: description.targetSite.id,
      display_name: description.targetSite.displayName,
      width: description.targetSite.width,
      capture_semantics: description.targetSite.captureSemantics,
    },
    sites: sourceSites.map((siteId, index) => {
      const site = siteById.get(siteId);
      if (!site) throw new Error(`fit report references unknown site ${siteId}`);
      return {
        id: site.id,
        display_name: site.displayName,
        order: site.order,
        layout: site.layout,
        width: site.width,
        capture_semantics: site.captureSemantics,
        token_aligned: site.tokenAligned,
        position_mapping: site.positionMapping,
        logit_lens_supported: site.logitLensSupported,
        transport: {
          representation: "dense",
          tensor_key: `transport.${index.toString().padStart(4, "0")}`,
          shape: [description.targetSite.width, site.width],
        },
        parent_stage: site.parentStage,
        component: site.component,
      };
    }),
    lens: {
      method: "average-input-output-jacobian",
      estimator: "Dimension-batched native causal VJP; valid target cotangents are summed over positions and source gradients are averaged over valid source positions and prompts.",
      artifact: "transports.safetensors",
      dtype: report.fitting.exported_dtype === "F16" ? "float16" : "float32",
      n_prompts: report.corpus.valid_prompt_count,
      max_seq_len: report.corpus.maximum_sequence_length,
      skip_first_positions: report.fitting.skip_first_positions,
      target_position_policy: report.fitting.target_position_policy,
    },
    validation: { artifact: "validation.json", status: "partial" },
  };
  await writeFile(join(output, "lens-manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
  await writeFile(join(output, "README.md"), buildReadme(manifest, identity, report));
  await writeFile(join(output, "CAPABILITY_REPORT.md"), buildCapabilityReport(runtime !== null));
  const enriched = {
    ...report,
    model_hf_repo: identity.modelHfRepo,
    model_hf_revision: identity.modelRevision,
    lens_hf_repo: identity.lensHfRepo,
    native_source_revision: identity.sourceRevision,
    hf_files: identity.hfFiles,
  };
  await writeFile(join(output, "fit-report.json"), JSON.stringify(enriched, null, 2) + "\n");
  return manifest;
}

function buildReadme(manifest: Record<string, any>, identity: BundleIdentity, report: Record<string, any>): string {
  return `---
library_name: blah-jlens
base_model: ${identity.modelHfRepo}
license: ${identity.license}
tags:
  - interpretability
  - jacobian-lens
  - blah-evals
  - evals.blah.dev
---

# Alpha BLAH Jacobian Lens

This is a **blah-jacobian-lens v1** artifact consumed by [evals.blah.dev](https://evals.blah.dev). It is fitted to exactly [${identity.modelHfRepo}](https://huggingface.co/${identity.modelHfRepo}/tree/${identity.modelRevision}) at immutable revision \`${identity.modelRevision}\` and will not be applied to a checkpoint with a different native weight fingerprint.

## Architecture and sites

Alpha is a custom TypeScript tensor/autograd model with a native Helios Vulkan backend. The adapter does not replace that implementation. It captures one token-aligned post-residual representation after each complete decoder block. The target is the final post-block representation immediately before the exact final RMSNorm and tied token-embedding projection. Every published site has the target width and basis, so ordinary Logit Lens decoding is valid.

The bundle stores ${manifest.sites.length} dense matrices as \`J[output_dimension, input_dimension]\`. Application uses \`h @ transpose(J)\`, then Alpha's exact final decoding path. No low-rank approximation is used.

## Fit

- Corpus: ${report.corpus.name} (${report.corpus.dataset_identifier}, revision ${report.corpus.immutable_revision})
- Visibility: ${report.corpus.visibility}
- Prompts / tokens: ${report.corpus.valid_prompt_count} / ${report.corpus.token_count}
- Maximum sequence length: ${report.corpus.maximum_sequence_length}
- Leading positions excluded: ${manifest.lens.skip_first_positions}
- Native fitting backend: ${report.fitting.device}
- Fitting dtype: float32; artifact dtype: ${manifest.lens.dtype}
- Hugging Face model.safetensors SHA-256: ${identity.hfFiles["model.safetensors"]}
- Estimator: ${manifest.lens.estimator}

See \`fit-report.json\` and \`validation.json\` for convergence, parity, and finite-difference measurements.

## Runtime

Execution mode: \`${manifest.execution.mode}\`. ${manifest.execution.public_runtime_url ? `The public \`blah-lens-http/1\` runtime is ${manifest.execution.public_runtime_url}.` : "Live analysis is unavailable until the native runtime is deployed; precomputed fixtures remain inspectable."}

The runtime uses the exact native tokenizer and chat template, preserves exact token IDs and unprettified vocabulary items (including byte-surrogate spelling), checks the checkpoint fingerprint before loading transports, and returns top-k readouts rather than full vocabulary logits. Full completion text is decoded only after the complete token sequence is assembled.

## Reproduce

From the Alpha source revision recorded in \`fit-report.json\`:

\`\`\`bash
alpha lens fit --checkpoint=<native-checkpoint> --prompts=<public-jsonl> --samples=${report.corpus.prompt_count} --max-seq-len=${report.corpus.maximum_sequence_length} --skip-first=${manifest.lens.skip_first_positions} --dim-batch=${report.fitting.vjp_batch_size} --dtype=${manifest.lens.dtype} --checkpoint-every=5 --output=dist/blah-lens
\`\`\`

The model and this artifact use the ${identity.license} license. Fitting prompt provenance and visibility are recorded in \`fit-report.json\`; private prompt text is not included.
`;
}

function buildCapabilityReport(remote: boolean): string {
  const runtime = remote ? "supported" : "partially supported — native server implemented but no public URL recorded";
  return `# BLAH Lens Capability Report

| Capability | Status | Technical note |
|---|---|---|
| Exact tokenization | supported | Native byte-level BPE artifacts embedded in the checkpoint. |
| Exact chat formatting | supported | System messages fold into the first user turn; roles alternate; generation opens exactly at the assistant marker. |
| Arbitrary prompt analysis | supported | Native forward capture accepts raw text, chat, or exact token IDs. |
| Live token generation | supported | Native prefill captures prompt positions once; each generated token uses the KV cache and captures only its new position. |
| Internal site capture | supported | Ordered post-block BTD tensors. |
| Token-position mapping | supported | One captured position per tokenizer position. |
| Logit Lens | supported | All sites share the final target width/basis and use exact final decoding. |
| Jacobian Lens fitting | supported | Dimension-batched native VJPs with resumable prompt-level accumulation. |
| Jacobian Lens application | supported | Dense row-convention transports and exact final decoding. |
| Dense export | supported | Safetensors only. |
| Low-rank export | unsupported | Dense storage is small; no approximation is justified. |
| Remote HTTP serving | ${runtime} | Implements blah-lens-http/1. |
| Exact input-token replay | supported | Supplied IDs bypass tokenization. |
| Pinned-token rank analysis | supported | Runtime returns requested IDs' logits and ranks. |
| Steering | partially supported | Additive post-block intervention seam exists; HTTP steering is not in protocol v1. |
| Suppression | partially supported | Possible through the intervention seam; no public endpoint yet. |
| Ablation | partially supported | Possible through the intervention seam; no public endpoint yet. |
| Concept swapping | unsupported | No validated concept representation or swap operator has been published. |
`;
}
