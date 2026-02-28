/**
 * Optimizers: AdamW, Lion, Adafactor, and SGD.
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
  private _entryCacheRef: readonly [string, { data: TensorData; grad: TensorData | null }][] | null = null;
  private _entrySlots: {
    useWeightDecay: boolean;
    m: Float32Array;
    v: Float32Array;
    mTd: TensorData;
    vTd: TensorData;
  }[] = [];
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
    if (this._entryCacheRef !== entries) this.rebuildEntryCache(entries);
    const slots = this._entrySlots;

    if (this.backend.adamwStep) {
      for (let i = 0; i < entries.length; i++) {
        const variable = entries[i][1];
        const grad = variable.grad;
        if (!grad) continue;
        const slot = slots[i];
        const wd = slot.useWeightDecay ? weightDecay : 0;
        this.backend.adamwStep(variable.data, grad, slot.mTd, slot.vTd, lr, beta1, beta2, eps, wd, bc1, bc2, gradScale);
      }
      return;
    }

    for (let i = 0; i < entries.length; i++) {
      const variable = entries[i][1];
      const grad = variable.grad;
      if (!grad) continue;
      const slot = slots[i];
      const wd = slot.useWeightDecay ? weightDecay : 0;

      const pData = variable.data.data as Float32Array;
      const gData = grad.data as Float32Array;
      const m = slot.m;
      const v = slot.v;
      const size = pData.length;
      for (let j = 0; j < size; j++) {
        const g = gData[j] * gradScale;
        if (wd > 0) pData[j] -= lr * wd * pData[j];
        m[j] = beta1 * m[j] + (1 - beta1) * g;
        v[j] = beta2 * v[j] + (1 - beta2) * g * g;
        const mHat = m[j] / bc1;
        const vHat = v[j] / bc2;
        pData[j] -= lr * mHat / (Math.sqrt(vHat) + eps);
      }
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

  private rebuildEntryCache(entries: readonly [string, { data: TensorData; grad: TensorData | null }][]): void {
    this._entryCacheRef = entries;
    this._entrySlots = new Array(entries.length);
    for (let i = 0; i < entries.length; i++) {
      const [name, variable] = entries[i];
      const param = variable.data;
      const size = shapeSize(param.shape);
      let m = this._m.get(name);
      let v = this._v.get(name);
      if (!m) {
        m = new Float32Array(size);
        v = new Float32Array(size);
        this._m.set(name, m);
        this._v.set(name, v);
      } else if (!v) {
        v = new Float32Array(size);
        this._v.set(name, v);
      }
      let mTd = this._mTd.get(name);
      let vTd = this._vTd.get(name);
      if (!mTd || !vTd) {
        mTd = { shape: param.shape, dtype: "f32", data: m };
        vTd = { shape: param.shape, dtype: "f32", data: v };
        this._mTd.set(name, mTd);
        this._vTd.set(name, vTd);
      }
      this._entrySlots[i] = {
        useWeightDecay: !this.noDecayNames.has(name),
        m,
        v,
        mTd,
        vTd,
      };
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
    this._entryCacheRef = null;
    this._entrySlots = [];
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

// ── Lion ───────────────────────────────────────────────────────────────────

export interface LionConfig {
  lr: number;
  beta1: number;
  beta2: number;
  weightDecay: number;
  noDecayNames?: Set<string>;
}

export class Lion implements Optimizer {
  readonly name = "lion";
  private _step = 0;
  private _m = new Map<string, Float32Array>();
  private config: LionConfig;
  private noDecayNames: Set<string>;

  constructor(_backend: Backend, config: Partial<LionConfig> = {}) {
    this.noDecayNames = config.noDecayNames ?? new Set();
    this.config = {
      lr: config.lr ?? 1e-4,
      beta1: config.beta1 ?? 0.9,
      beta2: config.beta2 ?? 0.99,
      weightDecay: config.weightDecay ?? 0.01,
    };
  }

  stepParamEntries(entries: readonly [string, { data: TensorData; grad: TensorData | null }][], gradScale = 1.0): void {
    this._step++;
    const { lr, beta1, beta2, weightDecay } = this.config;

    for (const [name, variable] of entries) {
      const grad = variable.grad;
      if (!grad) continue;
      this.stepTensor(name, variable.data, grad, lr, beta1, beta2, weightDecay, gradScale);
    }
  }

  step(params: Map<string, TensorData>, grads: Map<string, TensorData>, gradScale = 1.0): void {
    this._step++;
    const { lr, beta1, beta2, weightDecay } = this.config;

    for (const [name, param] of params) {
      const grad = grads.get(name);
      if (!grad) continue;
      this.stepTensor(name, param, grad, lr, beta1, beta2, weightDecay, gradScale);
    }
  }

  private stepTensor(
    name: string,
    param: TensorData,
    grad: TensorData,
    lr: number,
    beta1: number,
    beta2: number,
    weightDecay: number,
    gradScale: number,
  ): void {
    const size = shapeSize(param.shape);
    if (!this._m.has(name)) {
      this._m.set(name, new Float32Array(size));
    }

    const m = this._m.get(name)!;
    const pData = param.data as Float32Array;
    const gData = grad.data as Float32Array;
    const wd = this.noDecayNames.has(name) ? 0 : weightDecay;
    const decayMul = wd > 0 ? (1 - lr * wd) : 1;

    for (let i = 0; i < size; i++) {
      const g = gData[i] * gradScale;
      if (wd > 0) pData[i] *= decayMul;
      const update = beta1 * m[i] + (1 - beta1) * g;
      pData[i] -= lr * Math.sign(update);
      m[i] = beta2 * m[i] + (1 - beta2) * g;
    }
  }

  stateDict(): OptimizerState {
    const buffers = new Map<string, TensorData>();
    for (const [name, m] of this._m) {
      buffers.set(`${name}.m`, { shape: [m.length], dtype: "f32", data: new Float32Array(m) });
    }
    return { step: this._step, buffers };
  }

  loadStateDict(state: OptimizerState): void {
    this._step = state.step;
    this._m.clear();
    for (const [key, td] of state.buffers) {
      if (key.endsWith(".m")) {
        this._m.set(key.slice(0, -2), new Float32Array(td.data));
      }
    }
  }

  setLr(lr: number): void {
    this.config.lr = lr;
  }
}

// ── Adafactor ──────────────────────────────────────────────────────────────

export interface AdafactorConfig {
  lr: number;
  beta2: number;
  eps: number;
  clipThreshold: number;
  weightDecay: number;
  noDecayNames?: Set<string>;
}

export class Adafactor implements Optimizer {
  readonly name = "adafactor";
  private _step = 0;
  // 2D factored second-moment state
  private _vr = new Map<string, Float32Array>();
  private _vc = new Map<string, Float32Array>();
  // fallback unfactored state for non-2D params
  private _v = new Map<string, Float32Array>();
  private config: AdafactorConfig;
  private noDecayNames: Set<string>;

  constructor(_backend: Backend, config: Partial<AdafactorConfig> = {}) {
    this.noDecayNames = config.noDecayNames ?? new Set();
    this.config = {
      lr: config.lr ?? 1e-3,
      beta2: config.beta2 ?? 0.999,
      eps: config.eps ?? 1e-30,
      clipThreshold: config.clipThreshold ?? 1.0,
      weightDecay: config.weightDecay ?? 0.0,
    };
  }

  stepParamEntries(entries: readonly [string, { data: TensorData; grad: TensorData | null }][], gradScale = 1.0): void {
    this._step++;
    const { lr, beta2, eps, clipThreshold, weightDecay } = this.config;
    for (const [name, variable] of entries) {
      const grad = variable.grad;
      if (!grad) continue;
      this.stepTensor(name, variable.data, grad, lr, beta2, eps, clipThreshold, weightDecay, gradScale);
    }
  }

  step(params: Map<string, TensorData>, grads: Map<string, TensorData>, gradScale = 1.0): void {
    this._step++;
    const { lr, beta2, eps, clipThreshold, weightDecay } = this.config;
    for (const [name, param] of params) {
      const grad = grads.get(name);
      if (!grad) continue;
      this.stepTensor(name, param, grad, lr, beta2, eps, clipThreshold, weightDecay, gradScale);
    }
  }

  private stepTensor(
    name: string,
    param: TensorData,
    grad: TensorData,
    lr: number,
    beta2: number,
    eps: number,
    clipThreshold: number,
    weightDecay: number,
    gradScale: number,
  ): void {
    const pData = param.data as Float32Array;
    const gData = grad.data as Float32Array;
    const shape = param.shape;
    const size = pData.length;

    const wd = this.noDecayNames.has(name) ? 0 : weightDecay;
    const decayMul = wd > 0 ? (1 - lr * wd) : 1;

    if (shape.length === 2) {
      const rows = shape[0];
      const cols = shape[1];
      let vr = this._vr.get(name);
      let vc = this._vc.get(name);
      if (!vr || vr.length !== rows) {
        vr = new Float32Array(rows);
        this._vr.set(name, vr);
      }
      if (!vc || vc.length !== cols) {
        vc = new Float32Array(cols);
        this._vc.set(name, vc);
      }

      const oneMinusBeta2 = 1 - beta2;
      const invCols = 1 / cols;
      const invRows = 1 / rows;

      for (let r = 0; r < rows; r++) {
        let rowMean = 0;
        const rowOff = r * cols;
        for (let c = 0; c < cols; c++) {
          const g = gData[rowOff + c] * gradScale;
          rowMean += g * g;
        }
        rowMean *= invCols;
        vr[r] = beta2 * vr[r] + oneMinusBeta2 * rowMean + eps;
      }
      for (let c = 0; c < cols; c++) {
        let colMean = 0;
        for (let r = 0; r < rows; r++) {
          const g = gData[r * cols + c] * gradScale;
          colMean += g * g;
        }
        colMean *= invRows;
        vc[c] = beta2 * vc[c] + oneMinusBeta2 * colMean + eps;
      }

      let vrMean = 0;
      for (let r = 0; r < rows; r++) vrMean += vr[r];
      vrMean = Math.max(vrMean * invRows, eps);
      const invVrMean = 1 / vrMean;

      // Compute RMS of update for clipping.
      let updateSq = 0;
      for (let r = 0; r < rows; r++) {
        const rowOff = r * cols;
        const rScale = Math.sqrt(vr[r] * invVrMean);
        for (let c = 0; c < cols; c++) {
          const denom = Math.max(eps, rScale * Math.sqrt(vc[c]));
          const g = gData[rowOff + c] * gradScale;
          const upd = g / denom;
          updateSq += upd * upd;
        }
      }
      const rms = Math.sqrt(updateSq / Math.max(1, size));
      const clipDen = clipThreshold > 0 ? Math.max(1, rms / clipThreshold) : 1;
      const stepScale = lr / clipDen;

      for (let r = 0; r < rows; r++) {
        const rowOff = r * cols;
        const rScale = Math.sqrt(vr[r] * invVrMean);
        for (let c = 0; c < cols; c++) {
          const idx = rowOff + c;
          const denom = Math.max(eps, rScale * Math.sqrt(vc[c]));
          const g = gData[idx] * gradScale;
          if (wd > 0) pData[idx] *= decayMul;
          pData[idx] -= stepScale * (g / denom);
        }
      }
      return;
    }

    // Non-2D fallback: unfactored second moment.
    let v = this._v.get(name);
    if (!v || v.length !== size) {
      v = new Float32Array(size);
      this._v.set(name, v);
    }
    const oneMinusBeta2 = 1 - beta2;

    let updateSq = 0;
    for (let i = 0; i < size; i++) {
      const g = gData[i] * gradScale;
      v[i] = beta2 * v[i] + oneMinusBeta2 * (g * g) + eps;
      const upd = g / Math.sqrt(v[i]);
      updateSq += upd * upd;
    }
    const rms = Math.sqrt(updateSq / Math.max(1, size));
    const clipDen = clipThreshold > 0 ? Math.max(1, rms / clipThreshold) : 1;
    const stepScale = lr / clipDen;

    for (let i = 0; i < size; i++) {
      const g = gData[i] * gradScale;
      if (wd > 0) pData[i] *= decayMul;
      pData[i] -= stepScale * (g / Math.sqrt(v[i]));
    }
  }

  stateDict(): OptimizerState {
    const buffers = new Map<string, TensorData>();
    for (const [name, vr] of this._vr) {
      buffers.set(`${name}.vr`, { shape: [vr.length], dtype: "f32", data: new Float32Array(vr) });
    }
    for (const [name, vc] of this._vc) {
      buffers.set(`${name}.vc`, { shape: [vc.length], dtype: "f32", data: new Float32Array(vc) });
    }
    for (const [name, v] of this._v) {
      buffers.set(`${name}.v`, { shape: [v.length], dtype: "f32", data: new Float32Array(v) });
    }
    return { step: this._step, buffers };
  }

  loadStateDict(state: OptimizerState): void {
    this._step = state.step;
    this._vr.clear();
    this._vc.clear();
    this._v.clear();
    for (const [key, td] of state.buffers) {
      if (key.endsWith(".vr")) {
        this._vr.set(key.slice(0, -3), new Float32Array(td.data));
      } else if (key.endsWith(".vc")) {
        this._vc.set(key.slice(0, -3), new Float32Array(td.data));
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
  registry.register("lion", () => new Lion(backend));
  registry.register("adafactor", () => new Adafactor(backend));
  registry.register("sgd", () => new SGD(backend));
  return registry;
}
