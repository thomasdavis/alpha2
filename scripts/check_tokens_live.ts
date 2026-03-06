
import { loadAndTokenize } from "../packages/train/src/data.js";
import { tokenizerRegistry } from "../packages/tokenizers/src/index.js";
import { readFileSync } from "node:fs";

async function main() {
  const tokenizerArtifacts = JSON.parse(readFileSync("/home/ajax/alpha-repo/runs/tokenizer-artifacts-super-chat-bpe4k-v3.json", "utf8"));
  const tokenizer = tokenizerRegistry.get("bpe-chat-4k");
  if (tokenizer.loadArtifacts) {
    tokenizer.loadArtifacts(tokenizerArtifacts);
  }

  const dataPath = "/home/ajax/alpha-repo/data/super_chat.txt";
  const vocabSize = 4096;

  console.log(`Tokenizing ${dataPath}...`);
  const tokens = await loadAndTokenize(dataPath, tokenizer);

  console.log(`Total tokens: ${tokens.length}`);
  let min = 1000000, max = -1000000;
  let oobCount = 0;
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i];
    if (t < min) min = t;
    if (t > max) max = t;
    if (t < 0 || t >= vocabSize) oobCount++;
  }

  console.log(`Min token: ${min}`);
  console.log(`Max token: ${max}`);
  console.log(`Out-of-bounds count: ${oobCount}`);

  if (oobCount > 0) {
    console.error("FAILED: Dataset contains out-of-bounds tokens!");
  } else {
    console.log("SUCCESS: All tokens are within vocabulary range.");
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
