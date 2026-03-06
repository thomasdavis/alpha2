
import { initDevice } from "../packages/helios/src/device.js";
import { HeliosBackend } from "../packages/helios/src/backend.js";
import { initGPT, gptForward, collectParamEntries } from "../packages/model/src/gpt.js";
import { SeededRng, type ModelConfig } from "../packages/core/src/index.js";
import { Tape, DropoutRng } from "../packages/autograd/src/index.js";
import { AdamW } from "../packages/train/src/optimizers.js";
import { DataLoader, loadOrCacheTokens } from "../packages/train/src/data.js";
import { tokenizerRegistry } from "../packages/tokenizers/src/index.js";
import { readFileSync } from "node:fs";

async function main() {
  await initDevice();
  const backend = new HeliosBackend();
  const rng = new SeededRng(42);
  const dropoutRng = new DropoutRng(43);

  const tokenizerArtifacts = JSON.parse(readFileSync("/home/ajax/alpha-repo/runs/tokenizer-artifacts-super-chat-bpe4k-v3.json", "utf8"));
  const tokenizer = tokenizerRegistry.get("bpe-chat-4k");
  tokenizer.loadArtifacts(tokenizerArtifacts);

  const config: ModelConfig = {
    vocabSize: 4096,
    blockSize: 256,
    nLayer: 6,
    nEmbd: 192,
    nHead: 6,
    ffnActivation: "silu",
    ffnDim: 512,
    dropout: 0.1,
  };

  const params = initGPT(config, backend, rng);
  const paramEntries = collectParamEntries(params);
  const paramDataMap = new Map(paramEntries.map(([name, variable]) => [name, variable.data]));
  const optimizer = new AdamW(backend, { lr: 1e-4 });

  const tokensArr = await loadOrCacheTokens("/home/ajax/alpha-repo/data/super_chat.txt", tokenizer);
  const loader = new DataLoader(tokensArr, rng, 10, 256, true);

  let lossScale = 65536.0;

  console.log(`Running 5 steps with LossScale=${lossScale}...`);
  for (let step = 1; step <= 5; step++) {
    const batch = loader.nextBatch();
    const tape = new Tape();
    
    // Pass mixedPrecision=true
    const { loss } = gptForward(config, params, backend, tape, batch.inputs, batch.targets, true, false, true, dropoutRng);
    
    if (loss) {
      const lossVal = (loss.data.data as Float32Array)[0];
      console.log(`Step ${step} | Loss: ${lossVal}`);
      
      if (!Number.isFinite(lossVal)) {
        console.error(`FAILED: Loss is non-finite at step ${step}`);
        process.exit(1);
      }

      // Backward with LossScale
      const scaledGrad = backend.full(loss.data.shape, lossScale, loss.data.dtype);
      tape.backward(loss, backend, (td) => backend.releaseGpuTensor(td), scaledGrad);
      backend.releaseGpuTensor(scaledGrad);
      
      const gradMap = new Map();
      for (const [name, v] of paramEntries) {
        if (v.grad) gradMap.set(name, v.grad);
      }
      
      // Step with inverse lossScale
      optimizer.step(paramDataMap, gradMap, 1.0 / lossScale);
      
      for (const [, v] of paramEntries) {
        if (v.grad) backend.releaseGpuTensor(v.grad);
        v.grad = null;
      }
      tape.clear((td) => backend.releaseGpuTensor(td));
    }
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
