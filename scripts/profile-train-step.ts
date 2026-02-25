/**
 * Profile a single training step: forward, backward, grad norm, optimizer
 */
import { Variable, Tape } from "@alpha/autograd";
import { shapeSize, SeededRng } from "@alpha/core";
import type { Backend, TensorData } from "@alpha/core";
import { initGPT, gptForward, collectParams } from "@alpha/model";
import { HeliosBackend } from "@alpha/helios";

async function main() {
  const helios = new HeliosBackend();
  const B = helios as Backend;

  const config = {
    vocabSize: 2000,
    blockSize: 64,
    nLayer: 4,
    nEmbd: 128,
    nHead: 4,
  };

  const batchSize = 4;
  const T = config.blockSize;

  const params = initGPT(config as any, B, new SeededRng(1337));

  // Pre-create optimizer state
  const paramMap = collectParams(params);
  const mState = new Map<string, TensorData>();
  const vState = new Map<string, TensorData>();
  for (const [name, variable] of paramMap) {
    mState.set(name, B.zeros(variable.data.shape));
    vState.set(name, B.zeros(variable.data.shape));
  }

  const lr = 3e-4, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, wd = 0.01;

  // Warmup
  console.log("Warming up...");
  for (let i = 0; i < 2; i++) {
    const tokens: TensorData = {
      shape: [batchSize, T], dtype: "i32",
      data: Int32Array.from({ length: batchSize * T }, () => Math.floor(Math.random() * config.vocabSize)),
    };
    const targets: TensorData = {
      shape: [batchSize, T], dtype: "i32",
      data: Int32Array.from({ length: batchSize * T }, () => Math.floor(Math.random() * config.vocabSize)),
    };
    const tape = new Tape();
    const { loss } = gptForward(config as any, params, B, tape, tokens, targets);
    tape.backward(loss!, B);
    for (const [, variable] of paramMap) variable.grad = null;
    tape.clear();
    if ("flush" in B) (B as any).flush();
  }

  console.log("\nProfiling 5 steps...\n");

  for (let step = 0; step < 5; step++) {
    const tokens: TensorData = {
      shape: [batchSize, T], dtype: "i32",
      data: Int32Array.from({ length: batchSize * T }, () => Math.floor(Math.random() * config.vocabSize)),
    };
    const targets: TensorData = {
      shape: [batchSize, T], dtype: "i32",
      data: Int32Array.from({ length: batchSize * T }, () => Math.floor(Math.random() * config.vocabSize)),
    };

    const bc1 = 1 - Math.pow(beta1, step + 1);
    const bc2 = 1 - Math.pow(beta2, step + 1);

    const t0 = performance.now();

    // Forward
    const tape = new Tape();
    const { loss } = gptForward(config as any, params, B, tape, tokens, targets);
    const t1 = performance.now();

    // Force loss readback (triggers GPU flush for forward)
    const lossVal = (loss!.data.data as Float32Array)[0];
    const t2 = performance.now();

    // Backward
    tape.backward(loss!, B);
    const t3 = performance.now();

    // Grad norm computation (GPU ops)
    const pMap = collectParams(params);
    const sqNormParts: TensorData[] = [];
    for (const [, variable] of pMap) {
      if (variable.grad) {
        const g = variable.grad;
        const g2 = B.mul(g, g);
        sqNormParts.push(B.sum(g2));
      }
    }
    const t4 = performance.now();

    // Grad norm readback
    let gradNormSq = 0;
    for (const part of sqNormParts) {
      gradNormSq += (part.data as Float32Array)[0];
    }
    const gradNorm = Math.sqrt(gradNormSq);
    const t5 = performance.now();

    // AdamW step
    for (const [name, variable] of pMap) {
      if (variable.grad && B.adamwStep) {
        const m = mState.get(name)!;
        const v = vState.get(name)!;
        B.adamwStep(variable.data, variable.grad, m, v, lr, beta1, beta2, eps, wd, bc1, bc2);
      }
    }
    const t6 = performance.now();

    // Flush + wait
    if ("flush" in B) (B as any).flush();
    const t7 = performance.now();

    // Clear
    for (const [, variable] of pMap) variable.grad = null;
    tape.clear();
    const t8 = performance.now();

    console.log(`step ${step + 1}: loss=${lossVal.toFixed(4)}, grad_norm=${gradNorm.toFixed(4)}, total=${(t8-t0).toFixed(1)}ms`);
    console.log(`  forward_record:  ${(t1-t0).toFixed(1)}ms`);
    console.log(`  forward_flush:   ${(t2-t1).toFixed(1)}ms`);
    console.log(`  backward:        ${(t3-t2).toFixed(1)}ms`);
    console.log(`  gradnorm_record: ${(t4-t3).toFixed(1)}ms`);
    console.log(`  gradnorm_read:   ${(t5-t4).toFixed(1)}ms`);
    console.log(`  adamw:           ${(t6-t5).toFixed(1)}ms`);
    console.log(`  flush:           ${(t7-t6).toFixed(1)}ms`);
    console.log(`  clear:           ${(t8-t7).toFixed(1)}ms`);
    console.log();
  }
}

main().catch(console.error);
