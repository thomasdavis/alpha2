/**
 * Optimizers: AdamW and SGD.
 *
 * Inspired by microgpt.py's Adam implementation but generalized to work
 * with named parameter tensors.
 */
import type { Backend, TensorData, Optimizer, OptimizerState } from "@alpha/core";
import { Registry, shapeSize } from "@alpha/core";

// ── AdamW ──────────────────────────────────────────────────────────────────

export interface AdamWConfig {
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
  noDecayNames?: Set<string>;
}

export class AdamW implements Optimizer {
  readonly name = "adamw";
  private _step = 0;
  private _beta1Pow = 1;
  private _beta2Pow = 1;
  private _m = new Map<string, Float32Array>();
  private _v = new Map<string, Float32Array>();
  private _mTd = new Map<string, TensorData>(); // TensorData wrappers for GPU residence
  private _vTd = new Map<string, TensorData>();
  private config: AdamWConfig;
  private backend: Backend;
  private noDecayNames: Set<string>;

  constructor(backend: Backend, config: Partial<AdamWConfig> = {}) {
    this.backend = backend;
    this.noDecayNames = config.noDecayNames ?? new Set();
    this.config = {
      lr: config.lr ?? 3e-4,
      beta1: config.beta1 ?? 0.9,
      beta2: config.beta2 ?? 0.999,
      eps: config.eps ?? 1e-8,
      weightDecay: config.weightDecay ?? 0.01,
    };
  }

  stepParamEntries(entries: readonly [string, { data: TensorData; grad: TensorData | null }][], gradScale = 1.0): void {
    const { lr, beta1, beta2, eps, weightDecay } = this.config;
    const { bc1, bc2 } = this.nextBiasCorrections(beta1, beta2);

    for (const [name, variable] of entries) {
      const param = variable.data;
      const grad = variable.grad;
      if (!grad) continue;
      this.stepTensor(name, param, grad, lr, beta1, beta2, eps, weightDecay, bc1, bc2, gradScale);
    }
  }

  step(params: Map<string, TensorData>, grads: Map<string, TensorData>, gradScale = 1.0): void {
    const { lr, beta1, beta2, eps, weightDecay } = this.config;
    const { bc1, bc2 } = this.nextBiasCorrections(beta1, beta2);

    for (const [name, param] of params) {
      const grad = grads.get(name);
      if (!grad) continue;
      this.stepTensor(name, param, grad, lr, beta1, beta2, eps, weightDecay, bc1, bc2, gradScale);
    }
  }

  private nextBiasCorrections(beta1: number, beta2: number): { bc1: number; bc2: number } {
    this._step++;
    this._beta1Pow *= beta1;
    this._beta2Pow *= beta2;
    return {
      bc1: 1 - this._beta1Pow,
      bc2: 1 - this._beta2Pow,
    };
  }

  private stepTensor(
    name: string,
    param: TensorData,
    grad: TensorData,
    lr: number,
    beta1: number,
    beta2: number,
    eps: number,
    weightDecay: number,
    bc1: number,
    bc2: number,
    gradScale: number,
  ): void {
    const size = shapeSize(param.shape);

    // Lazy init moment buffers
    if (!this._m.has(name)) {
      this._m.set(name, new Float32Array(size));
      this._v.set(name, new Float32Array(size));
    }

    const wd = this.noDecayNames.has(name) ? 0 : weightDecay;

    // Try GPU path if backend supports it
    if (this.backend.adamwStep) {
      const mTd = this._mTd.get(name) ?? { shape: param.shape, dtype: "f32" as const, data: this._m.get(name)! };
      const vTd = this._vTd.get(name) ?? { shape: param.shape, dtype: "f32" as const, data: this._v.get(name)! };
      if (!this._mTd.has(name)) { this._mTd.set(name, mTd); this._vTd.set(name, vTd); }
      this.backend.adamwStep(param, grad, mTd, vTd, lr, beta1, beta2, eps, wd, bc1, bc2, gradScale);
      return;
    }

    // CPU path
    const pData = param.data as Float32Array;
    const gData = grad.data as Float32Array;
    const m = this._m.get(name)!;
    const v = this._v.get(name)!;

    for (let i = 0; i < size; i++) {
      const g = gData[i] * gradScale;
      if (wd > 0) pData[i] -= lr * wd * pData[i];
      m[i] = beta1 * m[i] + (1 - beta1) * g;
      v[i] = beta2 * v[i] + (1 - beta2) * g * g;
      const mHat = m[i] / bc1;
      const vHat = v[i] / bc2;
      pData[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
    }
  }

  stateDict(): OptimizerState {
    const buffers = new Map<string, TensorData>();
    for (const [name, m] of this._m) {
      const v = this._v.get(name)!;
      buffers.set(`${name}.m`, { shape: [m.length], dtype: "f32", data: new Float32Array(m) });
      buffers.set(`${name}.v`, { shape: [v.length], dtype: "f32", data: new Float32Array(v) });
    }
    return { step: this._step, buffers };
  }

  loadStateDict(state: OptimizerState): void {
    this._step = state.step;
    this._beta1Pow = Math.pow(this.config.beta1, this._step);
    this._beta2Pow = Math.pow(this.config.beta2, this._step);
    this._m.clear();
    this._v.clear();
    this._mTd.clear();
    this._vTd.clear();
    for (const [key, td] of state.buffers) {
      if (key.endsWith(".m")) {
        this._m.set(key.slice(0, -2), new Float32Array(td.data));
      } else if (key.endsWith(".v")) {
        this._v.set(key.slice(0, -2), new Float32Array(td.data));
      }
    }
  }

  setLr(lr: number): void {
    this.config.lr = lr;
  }
}

// ── SGD ────────────────────────────────────────────────────────────────────

export class SGD implements Optimizer {
  readonly name = "sgd";
  private _step = 0;
  private lr: number;

  constructor(_backend: Backend, lr = 0.01) {
    this.lr = lr;
  }

  stepParamEntries(entries: readonly [string, { data: TensorData; grad: TensorData | null }][], gradScale = 1.0): void {
    this._step++;
    for (const [, variable] of entries) {
      const grad = variable.grad;
      if (!grad) continue;
      const pData = variable.data.data as Float32Array;
      const gData = grad.data as Float32Array;
      for (let i = 0; i < pData.length; i++) {
        pData[i] -= this.lr * (gData[i] * gradScale);
      }
    }
  }

  step(params: Map<string, TensorData>, grads: Map<string, TensorData>, gradScale = 1.0): void {
    this._step++;
    for (const [name, param] of params) {
      const grad = grads.get(name);
      if (!grad) continue;
      const pData = param.data as Float32Array;
      const gData = grad.data as Float32Array;
      for (let i = 0; i < pData.length; i++) {
        pData[i] -= this.lr * (gData[i] * gradScale);
      }
    }
  }

  stateDict(): OptimizerState {
    return { step: this._step, buffers: new Map() };
  }

  loadStateDict(state: OptimizerState): void {
    this._step = state.step;
  }

  setLr(lr: number): void {
    this.lr = lr;
  }
}

// ── Registry ───────────────────────────────────────────────────────────────

export function createOptimizerRegistry(backend: Backend) {
  const registry = new Registry<Optimizer>("optimizer");
  registry.register("adamw", () => new AdamW(backend));
  registry.register("sgd", () => new SGD(backend));
  return registry;
}
