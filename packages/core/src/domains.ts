/**
 * Model domain definitions.
 *
 * A domain describes a type of content the model is trained on (novels,
 * chord progressions, etc.) along with appropriate defaults for tokenizer,
 * model architecture, and training hyperparameters.
 */
import type { ModelConfig, TrainConfig } from "./types.js";

export interface DomainConfig {
  readonly id: string;
  readonly displayName: string;
  readonly tokenizer: string;
  readonly samplePrompts: string[];
  readonly modelDefaults: Partial<ModelConfig>;
  readonly trainDefaults: Partial<TrainConfig>;
}

export const domains: ReadonlyMap<string, DomainConfig> = new Map<string, DomainConfig>([
  [
    "novels",
    {
      id: "novels",
      displayName: "Novels",
      tokenizer: "bpe",
      samplePrompts: ["The ", "Once upon a time", "He walked into"],
      modelDefaults: {
        blockSize: 128,
        nLayer: 6,
        nEmbd: 128,
        nHead: 8,
      },
      trainDefaults: {
        tokenizer: "bpe",
      },
    },
  ],
  [
    "chords",
    {
      id: "chords",
      displayName: "Chord Progressions",
      tokenizer: "word",
      samplePrompts: ["Em7 G Dsus4", "Am F C G", "D A Bm G"],
      modelDefaults: {
        blockSize: 64,
        nLayer: 3,
        nEmbd: 64,
        nHead: 4,
      },
      trainDefaults: {
        tokenizer: "word",
      },
    },
  ],
  [
    "abc",
    {
      id: "abc",
      displayName: "ABC Notation",
      tokenizer: "char",
      samplePrompts: ["X:1\nM:4/4\nK:G\n|:G", "X:1\nM:6/8\nK:D\nA|:d", "X:1\nM:4/4\nK:Ador\nE"],
      modelDefaults: {
        blockSize: 256,
        nLayer: 4,
        nEmbd: 128,
        nHead: 4,
      },
      trainDefaults: {
        tokenizer: "char",
      },
    },
  ],
  [
    "dumb_finance",
    {
      id: "dumb_finance",
      displayName: "Dumb Finance",
      tokenizer: "char",
      samplePrompts: [
        "1708300800000|AAPL|BID|",
        "1708300800000|TSLA|ASK|",
        "1708300800000|NVDA|TRADE|",
      ],
      modelDefaults: {
        blockSize: 256,
        nLayer: 6,
        nEmbd: 128,
        nHead: 8,
      },
      trainDefaults: {
        tokenizer: "char",
      },
    },
  ],
  [
    "chaos",
    {
      id: "chaos",
      displayName: "Chaos",
      tokenizer: "bpe-4k",
      samplePrompts: ["The word ", "Once there was a ", "She noticed the "],
      modelDefaults: {
        blockSize: 128,
        nLayer: 6,
        nEmbd: 128,
        nHead: 8,
      },
      trainDefaults: {
        tokenizer: "bpe-4k",
        lr: 5e-4,
      },
    },
  ],
  [
    "concordance",
    {
      id: "concordance",
      displayName: "Concordance",
      tokenizer: "bpe-64k",
      samplePrompts: [
        "The ",
        "When in the Course of human events",
        "It was a dark and stormy",
      ],
      modelDefaults: {
        blockSize: 1024,
        nLayer: 12,
        nEmbd: 768,
        nHead: 12,
      },
      trainDefaults: {
        tokenizer: "bpe-64k",
        lr: 6e-4,
        batchSize: 4,
        gradClip: 1.0,
      },
    },
  ],
  [
    "chat",
    {
      id: "chat",
      displayName: "Chat",
      tokenizer: "bpe-4k",
      samplePrompts: [
        "<|user|> Hello, how are you? <|assistant|>",
        "<|user|> What do you like to do for fun? <|assistant|>",
        "<|user|> Tell me about yourself. <|assistant|>",
      ],
      modelDefaults: {
        blockSize: 256,
        nLayer: 6,
        nEmbd: 256,
        nHead: 8,
      },
      trainDefaults: {
        tokenizer: "bpe-4k",
        lr: 3e-4,
        batchSize: 16,
        gradClip: 1.0,
      },
    },
  ],
]);

/** Look up a domain by id. Returns undefined if not found. */
export function getDomain(id: string): DomainConfig | undefined {
  return domains.get(id);
}
