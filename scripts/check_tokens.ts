
import { readFileSync } from "node:fs";
import { join } from "node:path";

async function main() {
  const dataPath = "/home/ajax/alpha-repo/data/super_chat.txt";
  const tokenizerName = "bpe-chat-4k";
  const vocabSize = 4096;
  const cacheFile = `${dataPath}.${tokenizerName}-${vocabSize}.tokens`;

  console.log(`Checking token cache: ${cacheFile}`);
  const buf = readFileSync(cacheFile);
  // Skip 8-byte header (mtime)
  const tokens = new Int32Array(buf.buffer, buf.byteOffset + 8, (buf.byteLength - 8) / 4);

  console.log(`Total tokens: ${tokens.length}`);
  let min = Infinity, max = -Infinity;
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
