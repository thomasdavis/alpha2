/**
 * HuggingFace `tokenizer.json` exporter for byte-level BPE artifacts.
 *
 * Emits a standard `tokenizers`-format fast tokenizer so the trained model can
 * be loaded on any machine with `transformers` and ZERO custom code:
 *
 *   - `tokenizer.json`        — model (BPE) + ByteLevel pre_tokenizer/decoder
 *                               + added_tokens for the specials.
 *   - `tokenizer_config.json` — PreTrainedTokenizerFast wrapper (bos/eos,
 *                               added_tokens_decoder, chat_template).
 *   - `chat_template.jinja`   — the chat template (also embedded in the config)
 *                               with `{% generation %}` markers.
 *
 * The pre_tokenizer is `ByteLevel(add_prefix_space=false, use_regex=true)`,
 * whose built-in split regex is exactly {@link GPT2_SPLIT_PATTERN} — the same
 * pattern {@link ByteBpeTokenizer.encode} applies — so the exported tokenizer
 * reproduces our ids byte-for-byte (proven by the round-trip test + the Python
 * `tokenizers` cross-check in scripts/verify_tokenizer_export.py).
 */
import { writeFile, mkdir } from "node:fs/promises";
import { join } from "node:path";
import type { TokenizerArtifacts } from "@alpha/core";
import { CHAT_SPECIAL_TOKENS } from "./byte_bpe.js";

export interface HfTokenizerBundle {
  tokenizerJson: unknown;
  tokenizerConfig: unknown;
  chatTemplate: string;
}

/**
 * Chat template that exactly matches scripts/build_sft_corpus.py:
 *
 *   <|user|> ... <|assistant|> ... <|user|> ... <|assistant|> ... <|end_of_text|>
 *
 * EOS terminates the conversation, not each historical assistant turn. A
 * leading system message is folded into the first user turn the same way as
 * the SFT builder. `{% generation %}` marks assistant content for HF clients.
 */
export function buildChatTemplate(): string {
  return (
    "{% for message in messages %}" +
    "{% if message['role'] == 'system' %}" +
    "{{ '<|user|> [Instructions: ' + message['content'] + ']\\n\\n' }}" +
    "{% elif message['role'] == 'user' %}" +
    "{% if loop.index0 > 0 and messages[loop.index0 - 1]['role'] == 'system' %}" +
    "{{ message['content'] + ' ' }}" +
    "{% else %}" +
    "{{ '<|user|> ' + message['content'] + ' ' }}" +
    "{% endif %}" +
    "{% elif message['role'] == 'assistant' %}" +
    "{{ '<|assistant|> ' }}" +
    "{% generation %}" +
    "{{ message['content'] }}" +
    "{% endgeneration %}" +
    "{% if loop.last %}{{ ' <|end_of_text|>' }}{% else %}{{ ' ' }}{% endif %}" +
    "{% endif %}" +
    "{% endfor %}" +
    "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}"
  );
}

function addedTokenEntry(id: number, content: string) {
  return {
    id,
    content,
    single_word: false,
    lstrip: false,
    rstrip: false,
    normalized: false,
    special: true,
  };
}

/**
 * Build the HF tokenizer bundle from byte-BPE artifacts (schema v2).
 * Throws if the artifacts are not `type:"byte_bpe"` / `byteVocab:true`.
 */
export function exportHfTokenizer(artifacts: TokenizerArtifacts): HfTokenizerBundle {
  if (artifacts.type !== "byte_bpe" || artifacts.byteVocab !== true) {
    throw new Error(
      `exportHfTokenizer requires byte-level BPE artifacts (type:"byte_bpe", byteVocab:true); got type:"${artifacts.type}"`,
    );
  }
  const vocabArr = artifacts.vocab;
  const specials = artifacts.specialTokens ?? [...CHAT_SPECIAL_TOKENS];

  // vocab map: token string -> id. Ascending id so the lowest id wins on the
  // (astronomically unlikely) event that a merge string equals a byte/special.
  const vocab: Record<string, number> = {};
  for (let id = 0; id < vocabArr.length; id++) {
    const tok = vocabArr[id];
    if (!(tok in vocab)) vocab[tok] = id;
  }

  // merges as "left right" surrogate-string pairs, in learned rank order.
  const merges: string[] = [];
  for (const [l, r] of artifacts.merges ?? []) {
    merges.push(`${vocabArr[l]} ${vocabArr[r]}`);
  }

  // added_tokens: the specials, with their actual ids from the vocab.
  const addedTokens = specials
    .filter((t) => t in vocab)
    .map((t) => addedTokenEntry(vocab[t], t));

  const tokenizerJson = {
    version: "1.0",
    truncation: null,
    padding: null,
    added_tokens: addedTokens,
    normalizer: null,
    pre_tokenizer: {
      type: "ByteLevel",
      add_prefix_space: false,
      trim_offsets: true,
      use_regex: true,
    },
    post_processor: {
      type: "ByteLevel",
      add_prefix_space: true,
      trim_offsets: false,
      use_regex: true,
    },
    decoder: {
      type: "ByteLevel",
      add_prefix_space: true,
      trim_offsets: true,
      use_regex: true,
    },
    model: {
      type: "BPE",
      dropout: null,
      unk_token: null,
      continuing_subword_prefix: null,
      end_of_word_suffix: null,
      fuse_unk: false,
      byte_fallback: false,
      ignore_merges: false,
      vocab,
      merges,
    },
  };

  const chatTemplate = buildChatTemplate();
  const eos = specials.includes("<|end_of_text|>") ? "<|end_of_text|>" : specials[specials.length - 1];

  const addedTokensDecoder: Record<string, unknown> = {};
  for (const t of specials) {
    if (!(t in vocab)) continue;
    addedTokensDecoder[String(vocab[t])] = {
      content: t,
      lstrip: false,
      normalized: false,
      rstrip: false,
      single_word: false,
      special: true,
    };
  }

  const tokenizerConfig = {
    add_bos_token: false,
    add_eos_token: false,
    added_tokens_decoder: addedTokensDecoder,
    additional_special_tokens: specials,
    bos_token: eos,
    eos_token: eos,
    unk_token: null,
    pad_token: eos,
    clean_up_tokenization_spaces: false,
    model_max_length: 1_000_000_000_000,
    tokenizer_class: "PreTrainedTokenizerFast",
    chat_template: chatTemplate,
  };

  return { tokenizerJson, tokenizerConfig, chatTemplate };
}

/** Export the bundle to `dir` as the three standard files. */
export async function writeHfTokenizer(dir: string, artifacts: TokenizerArtifacts): Promise<string[]> {
  const bundle = exportHfTokenizer(artifacts);
  await mkdir(dir, { recursive: true });
  const files: [string, string][] = [
    ["tokenizer.json", JSON.stringify(bundle.tokenizerJson, null, 2)],
    ["tokenizer_config.json", JSON.stringify(bundle.tokenizerConfig, null, 2)],
    ["chat_template.jinja", bundle.chatTemplate],
  ];
  const written: string[] = [];
  for (const [name, content] of files) {
    const path = join(dir, name);
    await writeFile(path, content, "utf-8");
    written.push(path);
  }
  return written;
}
