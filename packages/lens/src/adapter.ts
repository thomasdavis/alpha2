import { Effect } from "effect";
import type { Backend, TensorData, Tokenizer, TokenizerArtifacts } from "@alpha/core";
import { SeededRng } from "@alpha/core";
import { backendRegistry } from "@alpha/tensor";
import { heliosRegistry } from "@alpha/helios";
import { Tape, Variable } from "@alpha/autograd";
import {
  blockPostSiteId,
  collectParamEntries,
  exactFinalDecode,
  gptForward,
  initGPT,
  type GPTParams,
} from "@alpha/model";
import { buildChatTemplate, bytesToUnicode, tokenizerFromArtifacts } from "@alpha/tokenizers";
import { FileCheckpoint, releaseCheckpointSnapshotBuffers, restoreParams } from "@alpha/train";
import { prepareInferenceWeights, type InferenceWeights } from "@alpha/inference";
import { formatAlphaChat } from "./chat.js";
import { fingerprintWeights, sha256Bytes, sha256File, stableJson } from "./fingerprint.js";
import type {
  CapturedForward,
  ChatMessage,
  CopiedTensor,
  LensSiteDescription,
  NativeModelDescription,
} from "./types.js";

for (const name of heliosRegistry.list()) {
  if (!backendRegistry.list().includes(name)) backendRegistry.register(name, () => heliosRegistry.get(name));
}

export interface LoadAdapterOptions {
  readonly checkpoint: string;
  readonly backend?: string;
  readonly prepareInference?: boolean;
}

export class AlphaLensAdapter {
  readonly backend: Backend;
  readonly params: GPTParams;
  readonly tokenizer: Tokenizer;
  readonly tokenizerArtifacts: TokenizerArtifacts;
  readonly description: NativeModelDescription;
  readonly inferenceWeights?: InferenceWeights;
  private readonly releaseTensor?: (tensor: TensorData) => void;
  private readonly byteCharToByte?: ReadonlyMap<string, number>;
  private readonly specialTokenIds: ReadonlySet<number>;

  private constructor(
    backend: Backend,
    params: GPTParams,
    tokenizer: Tokenizer,
    tokenizerArtifacts: TokenizerArtifacts,
    description: NativeModelDescription,
    inferenceWeights?: InferenceWeights,
  ) {
    this.backend = backend;
    this.params = params;
    this.tokenizer = tokenizer;
    this.tokenizerArtifacts = tokenizerArtifacts;
    this.description = description;
    this.inferenceWeights = inferenceWeights;
    this.byteCharToByte = tokenizerArtifacts.byteVocab
      ? bytesToUnicode().charToByte
      : undefined;
    const specialStrings = new Set(tokenizerArtifacts.specialTokens ?? []);
    this.specialTokenIds = new Set(
      tokenizerArtifacts.vocab.flatMap((token, id) => specialStrings.has(token) ? [id] : []),
    );
    const candidate = backend as Backend & { releaseGpuTensor?: (tensor: TensorData) => void };
    this.releaseTensor = typeof candidate.releaseGpuTensor === "function"
      ? candidate.releaseGpuTensor.bind(candidate)
      : undefined;
  }

  static async load(options: LoadAdapterOptions): Promise<AlphaLensAdapter> {
    const state = await Effect.runPromise(new FileCheckpoint().load(options.checkpoint));
    if (!state.tokenizerArtifacts) throw new Error("checkpoint has no embedded tokenizer artifacts");
    releaseCheckpointSnapshotBuffers(state);

    const backendName = options.backend ?? "cpu_ref";
    // Alpha's selected checkpoint was trained and certified through Helios'
    // full-f32 matmul path.  The optional cooperative-matrix path performs
    // f16-input products and is a materially different numerical runtime for
    // this checkpoint: on the real model it changes activations, logits, and
    // VJPs far beyond lens tolerances.  Lens fitting must therefore select the
    // same exact backend mode before the Vulkan device is initialized.
    if (backendName === "helios") process.env.HELIOS_DISABLE_COOP_MAT = "1";
    const backend = backendRegistry.get(backendName);
    const params = initGPT(state.modelConfig, backend, new SeededRng(state.rngState ?? 42));
    restoreParams(params, state.params);
    for (const [, variable] of collectParamEntries(params)) variable.setRequiresGrad(false);
    const inferenceWeights = options.prepareInference
      ? prepareInferenceWeights(state.modelConfig, state.params)
      : undefined;

    const tokenizerArtifacts = state.tokenizerArtifacts;
    const tokenizer = tokenizerFromArtifacts(tokenizerArtifacts);
    const sites: LensSiteDescription[] = Array.from({ length: state.modelConfig.nLayer }, (_, index) => ({
      id: blockPostSiteId(index),
      displayName: `Decoder block ${index + 1} post-residual`,
      order: index,
      width: state.modelConfig.nEmbd,
      layout: "BTD",
      captureSemantics: "Residual representation after the complete decoder block, including attention and MLP residual additions, before the next block or final normalization.",
      tokenAligned: true,
      positionMapping: "token",
      logitLensSupported: true,
      parentStage: `decoder.block.${index.toString().padStart(3, "0")}`,
      component: "decoder-block-post-residual",
    }));
    const chatTemplate = buildChatTemplate();
    const description: NativeModelDescription = {
      framework: "alpha2",
      architecture: "Alpha Llama-form causal decoder (RoPE, RMSNorm, SwiGLU, tied token embeddings)",
      modelConfig: state.modelConfig,
      checkpointPath: options.checkpoint,
      checkpointStep: state.step,
      checkpointSha256: await sha256File(options.checkpoint),
      weightsFingerprint: fingerprintWeights(params),
      configFingerprint: sha256Bytes(stableJson(state.modelConfig)),
      tokenizerFingerprint: sha256Bytes(stableJson(tokenizerArtifacts)),
      chatTemplateFingerprint: sha256Bytes(chatTemplate),
      targetSite: {
        id: "decoder.final.post",
        displayName: "Final decoder post-residual",
        width: state.modelConfig.nEmbd,
        captureSemantics: "Final decoder block post-residual representation immediately before Alpha's exact final RMSNorm and tied output projection.",
      },
      sites,
      vocabularySize: state.modelConfig.vocabSize,
      blockSize: state.modelConfig.blockSize,
      specialTokens: [...(tokenizerArtifacts.specialTokens ?? [])],
    };

    // Drop the checkpoint's duplicate host-side parameter payload after the
    // native model owns restored tensors. This matters for long-lived runtimes.
    for (const value of Object.values(state.params)) value.data = [];
    return new AlphaLensAdapter(backend, params, tokenizer, tokenizerArtifacts, description, inferenceWeights);
  }

  describe(): NativeModelDescription {
    return this.description;
  }

  encode(text: string): Int32Array {
    return this.tokenizer.encode(text);
  }

  decode(tokenIds: ArrayLike<number>): string {
    return this.tokenizer.decode(tokenIds);
  }

  /** Exact, unprettified vocabulary item used by the native tokenizer. */
  tokenString(tokenId: number): string {
    if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= this.tokenizerArtifacts.vocab.length) {
      throw new RangeError(`token ID ${tokenId} is outside the native vocabulary`);
    }
    return this.tokenizerArtifacts.vocab[tokenId];
  }

  tokenStrings(tokenIds: ArrayLike<number>): string[] {
    return Array.from(tokenIds, (id) => this.tokenString(id));
  }

  /**
   * Authoritative raw bytes for byte-BPE vocabulary items. The visible token
   * string is the tokenizer's printable GPT-2 surrogate spelling; it cannot
   * faithfully represent an isolated non-UTF-8 byte in JSON. Special tokens
   * are literal UTF-8 strings and therefore need no byte side channel.
   */
  tokenBytesBase64(tokenId: number): string | undefined {
    const token = this.tokenString(tokenId);
    if (!this.byteCharToByte || this.specialTokenIds.has(tokenId)) return undefined;
    const bytes: number[] = [];
    for (const character of token) {
      const byte = this.byteCharToByte.get(character);
      if (byte === undefined) {
        throw new Error(`byte-BPE token ${tokenId} contains a non-byte surrogate character`);
      }
      bytes.push(byte);
    }
    return Buffer.from(bytes).toString("base64");
  }

  tokenDescriptor(tokenId: number): { id: number; text: string; bytes_base64?: string } {
    const bytes = this.tokenBytesBase64(tokenId);
    return {
      id: tokenId,
      text: this.tokenString(tokenId),
      ...(bytes === undefined ? {} : { bytes_base64: bytes }),
    };
  }

  formatChat(messages: readonly ChatMessage[]): { text: string; tokenIds: Int32Array } {
    const text = formatAlphaChat(messages);
    return { text, tokenIds: this.encode(text) };
  }

  forwardCapture(
    tokenIds: ArrayLike<number>,
    requestedSites: readonly string[] = this.description.sites.map((site) => site.id),
    batchSize = 1,
    sitePerturbations?: ReadonlyMap<string, TensorData>,
  ): CapturedForward {
    if (!Number.isInteger(batchSize) || batchSize < 1) throw new Error("batchSize must be a positive integer");
    const ids = Int32Array.from(tokenIds);
    if (ids.length === 0) throw new Error("forward_capture requires at least one token");
    if (ids.length > this.description.blockSize) {
      throw new Error(`sequence length ${ids.length} exceeds block size ${this.description.blockSize}`);
    }
    const known = new Set(this.description.sites.map((site) => site.id));
    for (const site of requestedSites) if (!known.has(site)) throw new Error(`unsupported site: ${site}`);

    const repeated = new Int32Array(batchSize * ids.length);
    for (let batch = 0; batch < batchSize; batch++) repeated.set(ids, batch * ids.length);
    const input: TensorData = { shape: [batchSize, ids.length], dtype: "i32", data: repeated };
    const tape = new Tape();
    const result = gptForward(
      this.description.modelConfig,
      this.params,
      this.backend,
      tape,
      input,
      undefined,
      false,
      false,
      false,
      undefined,
      this.releaseTensor,
      undefined,
      { kind: "cross_entropy" },
      { requestedSites: new Set(requestedSites), captureTarget: true, sitePerturbations },
    );
    if (!result.sites || !result.target) throw new Error("native forward did not return requested captures");
    return {
      tokenIds: ids,
      batchSize,
      sequenceLength: ids.length,
      tape,
      input,
      sites: result.sites,
      target: result.target,
      logits: result.logits,
    };
  }

  copyTensor(tensor: TensorData): CopiedTensor {
    const source = tensor.data as ArrayLike<number>;
    const data = new Float32Array(source.length);
    for (let i = 0; i < source.length; i++) data[i] = source[i];
    return { shape: [...tensor.shape], dtype: "f32", data };
  }

  exactFinalDecode(targetBasis: TensorData): CopiedTensor {
    const tape = new Tape();
    const input = new Variable(targetBasis, false);
    const logits = exactFinalDecode(
      this.description.modelConfig,
      this.params,
      this.backend,
      tape,
      input,
    );
    const copied = this.copyTensor(logits.data);
    tape.clear(this.releaseTensor, logits);
    if (this.releaseTensor) this.releaseTensor(logits.data);
    return copied;
  }

  /** Native vector-Jacobian product for every requested captured source. */
  vjp(
    capture: CapturedForward,
    cotangent: TensorData,
    sourceSiteIds: readonly string[],
    retainGraph: boolean,
  ): ReadonlyMap<string, CopiedTensor> {
    if (!sameShape(cotangent.shape, capture.target.data.shape)) {
      throw new Error(`cotangent shape [${cotangent.shape}] does not match target [${capture.target.data.shape}]`);
    }
    const byVariable = new Map<number, string>();
    for (const siteId of sourceSiteIds) {
      const variable = capture.sites.get(siteId);
      if (!variable) throw new Error(`site ${siteId} was not captured`);
      byVariable.set(variable.id, siteId);
    }
    const copied = new Map<string, CopiedTensor>();
    capture.tape.backward(capture.target, this.backend, this.releaseTensor, cotangent, {
      retainGraph,
      onGradient: (variable, gradient) => {
        const siteId = byVariable.get(variable.id);
        if (siteId) copied.set(siteId, this.copyTensor(gradient));
      },
    });
    for (const siteId of sourceSiteIds) {
      if (!copied.has(siteId)) throw new Error(`VJP did not encounter captured site ${siteId}`);
    }
    return copied;
  }

  disposeCapture(capture: CapturedForward): void {
    capture.tape.clear(this.releaseTensor);
    if (this.releaseTensor) this.releaseTensor(capture.input);
  }
}

function sameShape(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}
