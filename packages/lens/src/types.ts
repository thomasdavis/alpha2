import type { ModelConfig, TensorData } from "@alpha/core";
import type { Variable, Tape } from "@alpha/autograd";

export interface LensSiteDescription {
  readonly id: string;
  readonly displayName: string;
  readonly order: number;
  readonly width: number;
  readonly layout: "BTD";
  readonly captureSemantics: string;
  readonly tokenAligned: true;
  readonly positionMapping: "token";
  readonly logitLensSupported: boolean;
  readonly parentStage: string;
  readonly component: string;
}

export interface NativeModelDescription {
  readonly framework: "alpha2";
  readonly architecture: string;
  readonly modelConfig: ModelConfig;
  readonly checkpointPath: string;
  readonly checkpointStep: number;
  readonly checkpointSha256: string;
  readonly weightsFingerprint: string;
  readonly configFingerprint: string;
  readonly tokenizerFingerprint: string;
  readonly chatTemplateFingerprint: string;
  readonly targetSite: {
    readonly id: "decoder.final.post";
    readonly displayName: string;
    readonly width: number;
    readonly captureSemantics: string;
  };
  readonly sites: readonly LensSiteDescription[];
  readonly vocabularySize: number;
  readonly blockSize: number;
  readonly specialTokens: readonly string[];
}

export interface CapturedForward {
  readonly tokenIds: Int32Array;
  readonly batchSize: number;
  readonly sequenceLength: number;
  readonly tape: Tape;
  readonly input: TensorData;
  readonly sites: ReadonlyMap<string, Variable>;
  readonly target: Variable;
  readonly logits: Variable;
}

export interface CopiedTensor {
  readonly shape: readonly number[];
  readonly dtype: "f32";
  readonly data: Float32Array;
}

export interface ChatMessage {
  readonly role: "system" | "user" | "assistant";
  readonly content: string;
}
