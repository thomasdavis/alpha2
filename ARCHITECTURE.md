# Alpha: A GPT Training System Built From Scratch

A comprehensive technical reference for the Alpha project — a complete GPT training pipeline written entirely in TypeScript with zero ML framework dependencies. Every component, from tensor operations and automatic differentiation to GPU compute kernels and SPIR-V assembly, is hand-written.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Core Types & Configuration](#2-core-types--configuration)
3. [Tensor Backend (CPU)](#3-tensor-backend-cpu)
4. [Automatic Differentiation](#4-automatic-differentiation)
5. [GPT Model Architecture](#5-gpt-model-architecture)
6. [Tokenizers](#6-tokenizers)
7. [Training Pipeline](#7-training-pipeline)
8. [Helios: GPU Compute via Vulkan](#8-helios-gpu-compute-via-vulkan)
9. [Data Generation](#9-data-generation)
10. [Database & Persistence](#10-database--persistence)
11. [Server & Inference](#11-server--inference)
12. [CLI](#12-cli)
13. [End-to-End Data Flow](#13-end-to-end-data-flow)

---

## 1. System Overview

### Philosophy

Alpha is built on four principles:

- **Pure TypeScript** — no PyTorch, no TensorFlow, no ONNX. The entire stack runs on Node.js.
- **Zero dependencies** — core functionality has no npm dependencies. GPU access is a hand-written C addon with a hand-written SPIR-V assembler.
- **Understand everything** — no black boxes. Every matrix multiply, every gradient, every GPU kernel is written and understood.
- **Super configurable** — every layer is pluggable via registries and interfaces.

### Monorepo Structure

```
packages/core         Types, configs, domains, RNG, registries
packages/tensor       CPU tensor backend (reference implementation)
packages/autograd     Tape-based reverse-mode automatic differentiation
packages/model        GPT-2-style transformer (init, forward, params)
packages/tokenizers   BPE, character, and word tokenizers
packages/train        Training loop, optimizers, checkpointing, data loading
packages/helios       GPU compute backend (Vulkan C addon + SPIR-V from TypeScript)
packages/datagen      Corpus generation from Wikipedia, Gutenberg, Wiktionary
packages/db           Turso/libsql database layer for run tracking
apps/server           Inference server with OpenAI-compatible API
apps/cli              CLI for training, sampling, evaluation, benchmarking
apps/tui              Terminal dashboard (Ink/React)
```

### Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | TypeScript (ESM throughout) |
| Runtime | Node.js |
| Build | npm workspaces + Turbo |
| GPU | Vulkan 1.2 via N-API C addon |
| Shaders | SPIR-V 1.3 assembled from TypeScript |
| Database | Turso (libsql) |
| Server | Native Node.js `http` module |
| Deploy | Railway |

---

## 2. Core Types & Configuration

**Package:** `@alpha/core`

The core package defines every type, interface, and contract used across the system. It has no compute logic — only definitions.

### 2.1 Data Types

```typescript
type Dtype = "f32" | "f64" | "i32";
```

- `f32` — 32-bit float (`Float32Array`), 4 bytes. Default for all tensors.
- `f64` — 64-bit float (`Float64Array`), 8 bytes.
- `i32` — 32-bit integer (`Int32Array`), 4 bytes. Used for token indices.

Helper functions:
- `dtypeBytes(d)` — returns byte size (4 or 8)
- `dtypeArray(d)` — returns the TypedArray constructor
- `shapeSize(shape)` — product of all dimensions (total element count)
- `shapeStrides(shape)` — row-major strides for flat indexing

### 2.2 ModelConfig

Defines the transformer architecture:

```typescript
interface ModelConfig {
  vocabSize: number;    // Vocabulary size for embeddings
  blockSize: number;    // Context/sequence length (max tokens per forward pass)
  nLayer: number;       // Number of transformer blocks
  nEmbd: number;        // Embedding/hidden dimension
  nHead: number;        // Number of attention heads
  dropout: number;      // Dropout rate (0.0 = disabled)
}
```

**Defaults:** `vocabSize=256, blockSize=256, nLayer=6, nEmbd=256, nHead=8, dropout=0.0`

### 2.3 TrainConfig

Controls the training loop:

```typescript
interface TrainConfig {
  iters: number;          // Total training iterations
  batchSize: number;      // Batch size
  lr: number;             // Learning rate
  beta1: number;          // Adam momentum coefficient
  beta2: number;          // Adam second-moment coefficient
  eps: number;            // Adam epsilon for numerical stability
  weightDecay: number;    // L2 regularization coefficient
  gradClip: number;       // Maximum gradient norm
  evalInterval: number;   // Steps between validation runs
  evalIters: number;      // Number of validation batches per eval
  seed: number;           // PRNG seed for reproducibility
  backend: string;        // Compute backend ("cpu_ref" or "helios")
  tokenizer: string;      // Tokenizer type ("bpe", "char", "word")
  optimizer: string;      // Optimizer ("adamw" or "sgd")
  logLevel: string;       // "debug" | "info" | "warn" | "error"
  trace: boolean;         // Enable execution tracing
}
```

**Defaults:** `iters=1000, batchSize=64, lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8, weightDecay=0.01, gradClip=1.0, evalInterval=100, evalIters=10, seed=42`

### 2.4 SampleConfig

Controls text generation:

```typescript
interface SampleConfig {
  steps: number;        // Number of tokens to generate
  temperature: number;  // Softmax temperature (higher = more random)
  topk: number;         // Top-K filtering (0 = disabled)
}
```

**Defaults:** `steps=200, temperature=0.8, topk=40`

### 2.5 Backend Interface

The pluggable compute backend — 30+ methods covering creation, math, neural network ops, reshaping, and comparison:

**Creation:** `zeros`, `ones`, `full`, `randn`, `fromArray`

**Binary math:** `add`, `sub`, `mul`, `div` (all with broadcasting)

**Matrix:** `matmul` (batched N-D support)

**Reductions:** `sum`, `mean` (with axis and keepdims)

**Unary:** `neg`, `exp`, `log`, `sqrt`, `pow`, `scale`

**Activations:** `gelu`, `relu`, `silu`

**Neural network:** `embedding`, `layerNorm`, `softmax`, `logSoftmax`, `crossEntropy`

**Reshape/slice:** `reshape`, `transpose`, `slice`, `cat`

**Selection:** `argmax`, `topk`, `gather`, `clone`

**Masking:** `causalMask`, `maskedFill`

**Optional GPU-accelerated:** `geluBackward`, `reluBackward`, `layerNormBackward`, `broadcast`, `adamwStep`

### 2.6 Tokenizer Interface

```typescript
interface Tokenizer {
  name: string;
  build(input: string): Effect<TokenizerArtifacts, TokenizerError>;
  encode(text: string): Int32Array;
  decode(tokens: ArrayLike<number>): string;
  vocabSize: number;
}

interface TokenizerArtifacts {
  type: string;                           // "bpe", "char", "word"
  vocabSize: number;
  vocab: readonly string[];               // Token ID -> string
  merges?: readonly [number, number][];   // BPE merge pairs
}
```

### 2.7 Optimizer Interface

```typescript
interface Optimizer {
  name: string;
  step(params: Map<string, TensorData>, grads: Map<string, TensorData>): void;
  stateDict(): OptimizerState;
  loadStateDict(state: OptimizerState): void;
}

interface OptimizerState {
  step: number;
  buffers: Map<string, TensorData>;  // Momentum/second-moment states
}
```

### 2.8 Domains

Six predefined training domains with recommended configurations:

| Domain | Tokenizer | blockSize | nEmbd | nLayer | nHead | Use Case |
|--------|-----------|-----------|-------|--------|-------|----------|
| `novels` | bpe | 128 | 128 | 6 | 8 | Prose fiction |
| `chords` | word | 64 | 64 | 3 | 4 | Guitar chord progressions |
| `abc` | char | 256 | 128 | 4 | 4 | ABC music notation |
| `dumb_finance` | char | 256 | 128 | 6 | 8 | Financial tick data |
| `chaos` | bpe-4k | 128 | 128 | 6 | 8 | Mixed text (lr=5e-4) |
| `concordance` | bpe | 256 | 256 | 6 | 8 | Dense vocabulary coverage |

### 2.9 RNG

**xorshift128+** implementation with Box-Muller transform for Gaussian sampling:

- `next()` — uniform [0, 1) via xorshift128+ (shifts: 23, 17, 26 bits)
- `nextGauss()` — N(0, 1) via Box-Muller with spare caching
- `seed(s)` — reinitialize with 20-iteration warmup
- Deterministic: same seed produces identical sequences

### 2.10 Registry

Generic factory registry for pluggable implementations:

```typescript
class Registry<T> {
  register(name: string, factory: () => T): void;
  get(name: string): T;    // Instantiates and returns
  has(name: string): boolean;
  list(): string[];
}
```

Used for backends, tokenizers, and optimizers. Error messages include available options.

### 2.11 Config Hashing

- `hashConfig(config)` — FNV-1a 32-bit hash of sorted JSON, returns 8-char hex string
- `runId()` — `YYYYMMDDHHMMSS_xxxx` timestamp + 4-char random suffix

---

## 3. Tensor Backend (CPU)

**Package:** `@alpha/tensor`

The `CpuRefBackend` is the reference implementation of the `Backend` interface. It prioritizes **correctness over speed** — every operation uses straightforward loops over typed arrays. It serves as the ground truth for testing GPU backends.

### 3.1 Data Representation

Tensors are plain objects, not classes:

```typescript
interface TensorData {
  shape: Shape;           // e.g., [2, 3, 4]
  dtype: Dtype;           // "f32" | "f64" | "i32"
  data: TypedArray;       // Flat, contiguous, row-major (C-order)
}
```

All storage is **flat and contiguous**. A shape `[2, 3, 4]` stores 24 elements in row-major order with strides `[12, 4, 1]`. Element `[i, j, k]` maps to flat index `i*12 + j*4 + k`.

### 3.2 Broadcasting

Broadcasting follows NumPy rules:

1. Pad shorter shape with leading 1s
2. For each dimension: if sizes differ, one must be 1
3. Result dimension is the maximum
4. Broadcast dimensions use stride 0 (no data duplication)

```
Shape A: [3, 1, 4]  +  Shape B: [1, 5, 4]  =  Result: [3, 5, 4]
```

All binary operations (`add`, `sub`, `mul`, `div`) broadcast automatically. Dtype promotion selects the widest type: `f64 > f32 > i32`.

### 3.3 Matrix Multiplication

```
[..., M, K] @ [..., K, N] -> [..., M, N]
```

- Supports batched N-D tensors
- Batch dimensions broadcast (stride 0 for size-1 dims)
- Core: triple-nested loop `M x N x K` with dot product accumulation
- Complexity: `O(batch * M * N * K)`

### 3.4 Reductions

`sum(a, axis?, keepdims?)` and `mean(a, axis?, keepdims?)`:

- No axis: reduce all elements to scalar
- With axis: reduce along that dimension
- `keepdims=true`: collapsed dimension becomes size 1
- Mean divides by axis size after summing

### 3.5 Neural Network Operations

**Embedding:** `embedding(weight, indices)` — gathers rows from `[vocabSize, dim]` weight table using integer indices. Output shape: `[...indices.shape, dim]`.

**LayerNorm:** `layerNorm(x, weight, bias, eps)` — normalizes over last dimension:
```
y[j] = weight[j] * (x[j] - mean) / sqrt(variance + eps) + bias[j]
```

**Activations:**
- GELU: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))` (tanh approximation)
- ReLU: `max(0, x)`
- SiLU: `x / (1 + exp(-x))`

**Softmax:** Numerically stable — subtracts max before exp:
```
softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))
```

**Cross-entropy:** Combines log-softmax with negative log-likelihood:
```
loss = -(1/N) * sum(logSoftmax(logits)[i, targets[i]])
```

**Causal mask:** Lower-triangular `[T, T]` matrix with 0 on/below diagonal, `-Infinity` above:
```
[  0   -inf  -inf  -inf ]
[  0     0   -inf  -inf ]
[  0     0     0   -inf ]
[  0     0     0     0   ]
```

### 3.6 Design Invariants

- **Row-major (C-order)** layout throughout
- **Contiguous storage** — no strided views
- **Eager evaluation** — all operations execute immediately
- **No in-place mutation** — every operation allocates a new TensorData
- **No SIMD, no threading** — pure single-threaded JavaScript
- **Deterministic** — seeded RNG (`SeededRng(42)`)

---

## 4. Automatic Differentiation

**Package:** `@alpha/autograd`

A **tape-based reverse-mode autodiff engine**. Every differentiable operation records itself on a tape during the forward pass. The backward pass walks the tape in reverse to compute gradients via the chain rule.

### 4.1 Variable

```typescript
class Variable {
  id: number;                // Unique auto-incrementing ID
  data: TensorData;          // Forward pass result
  grad: TensorData | null;   // Gradient (set during backward)
  requiresGrad: boolean;     // Whether to compute gradients
}
```

Model parameters are Variables with `requiresGrad=true`. Intermediate activations are also Variables (for gradient routing) but only parameter Variables retain their gradients after backward.

### 4.2 Tape

```typescript
interface TapeEntry {
  output: Variable;
  inputs: Variable[];
  backward(outGrad: TensorData, backend: Backend): TensorData[];
}

class Tape {
  record(entry: TapeEntry): void;
  backward(loss: Variable, backend: Backend, releaseTensor?: callback): void;
  clear(releaseTensor?: callback): void;
  size: number;
}
```

### 4.3 Backward Pass Algorithm

```
1. loss.grad = ones(loss.shape)     // Seed with 1

2. For each entry in REVERSE order:
   a. Skip if entry.output.grad is null
   b. inputGrads = entry.backward(entry.output.grad, backend)
   c. For each input variable:
      - If input.requiresGrad:
        - If input.grad exists: accumulate (add)
        - Else: clone the gradient
   d. Release intermediate GPU buffers (if releaseTensor provided)
```

**Gradient accumulation** handles variables used multiple times (e.g., `x` in `x * x` receives gradients from both paths).

### 4.4 GPU Memory Management

The `releaseTensor` callback is **critical** for GPU training. Without it, hundreds of intermediate GPU buffers accumulate and cause OOM:

1. Old accumulated gradients are freed after adding new ones
2. Computed input gradients are freed after cloning/accumulating
3. Forward activation data is freed immediately after its backward closure runs
4. Output gradients are freed after full consumption

This reduces peak GPU memory from O(tape_size) to O(current_entry).

### 4.5 Differentiable Operations

Each operation records itself on the tape with a closure that computes input gradients from the output gradient:

#### Arithmetic

| Operation | Forward | Backward (dL/da, dL/db) |
|-----------|---------|-------------------------|
| `add(a, b)` | `a + b` | `g`, `g` (with broadcast reduction) |
| `sub(a, b)` | `a - b` | `g`, `-g` |
| `mul(a, b)` | `a * b` | `g * b`, `g * a` |
| `div(a, b)` | `a / b` | `g / b`, `-g * a / b^2` |
| `scale(a, s)` | `a * s` | `g * s` |
| `neg(a)` | `-a` | `-g` |

#### Matrix

| Operation | Forward | Backward |
|-----------|---------|----------|
| `matmul(A, B)` | `A @ B` | `dA = g @ B^T`, `dB = A^T @ g` |

#### Reductions

| Operation | Forward | Backward |
|-----------|---------|----------|
| `sum(a, axis)` | Sum along axis | Broadcast gradient back to input shape |
| `mean(a, axis)` | Mean along axis | Broadcast and scale by `1/n` |

#### Element-wise Nonlinearities

| Operation | Forward | Backward |
|-----------|---------|----------|
| `exp(a)` | `exp(x)` | `g * exp(x)` (reuses forward output) |
| `log(a)` | `log(x)` | `g / x` |
| `sqrt(a)` | `sqrt(x)` | `g / (2 * sqrt(x))` |
| `relu(a)` | `max(0, x)` | `g * (x > 0 ? 1 : 0)` |
| `gelu(a)` | GELU approx | Analytical derivative of tanh approximation |

#### Neural Network

| Operation | Forward | Backward |
|-----------|---------|----------|
| `embedding(W, idx)` | Gather rows | Scatter-add gradients back to weight table |
| `layerNorm(x, w, b)` | Normalize + affine | dX via chain rule through mean/var, dW = sum(g * x_hat), dB = sum(g) |
| `softmax(a)` | Softmax | `s * (g - sum(g * s))` (Jacobian-vector product) |
| `crossEntropy(logits, targets)` | -log(softmax[target]) | `(softmax - one_hot) / N` |

#### Shape Operations

| Operation | Forward | Backward |
|-----------|---------|----------|
| `reshape(a, shape)` | Reinterpret shape | Reshape gradient back to original |
| `transpose(a, d0, d1)` | Swap dimensions | Transpose gradient (self-inverse) |

### 4.6 Broadcast Gradient Reduction

When a tensor is broadcast during the forward pass, gradients must be reduced back:

```typescript
function reduceBroadcast(backend, grad, targetShape):
  // Remove leading dims that were added
  while grad.ndim > targetShape.ndim:
    grad = sum(grad, axis=0)
  // Collapse broadcast dims (size 1) back
  for each dim where targetShape[i] == 1 && grad.shape[i] != 1:
    grad = sum(grad, axis=i, keepdims=true)
```

### 4.7 GPU-Optimized Backward Paths

When a GPU backend is available, several operations use fused backward kernels instead of the generic CPU path:

- `relu` — `backend.reluBackward(input, gradOutput)` instead of building a mask
- `gelu` — `backend.geluBackward(input, gradOutput)` instead of element-wise derivative
- `layerNorm` — `backend.layerNormBackward(x, weight, gradOutput, eps)` returning `{dx, dw, db}` in one kernel

---

## 5. GPT Model Architecture

**Package:** `@alpha/model`

A **GPT-2-style decoder-only transformer** with pre-LayerNorm (the modern variant used in GPT-3+).

### 5.1 Architecture Diagram

```
Input tokens [B, T]
     |
     v
Token Embedding [vocabSize, nEmbd]  +  Position Embedding [blockSize, nEmbd]
     |
     v
x = tokEmb + posEmb  [B, T, nEmbd]
     |
     v
+--------------------------------------------------+
| Transformer Block (repeated nLayer times)         |
|                                                    |
|  residual = x                                      |
|  x = LayerNorm(x)                                  |
|  q = x @ Wq    k = x @ Wk    v = x @ Wv           |
|  q,k,v reshaped to [B, nHead, T, headDim]          |
|  scores = (q @ k^T) / sqrt(headDim)                |
|  scores = causalMask(scores)                        |
|  attnWeights = softmax(scores)                      |
|  attnOut = attnWeights @ v                          |
|  attnOut reshaped to [B, T, nEmbd]                  |
|  x = residual + attnOut @ Wo                        |
|                                                    |
|  residual = x                                      |
|  x = LayerNorm(x)                                  |
|  h = GELU(x @ Wfc1)        [B, T, 4*nEmbd]         |
|  x = residual + h @ Wfc2   [B, T, nEmbd]            |
+--------------------------------------------------+
     |
     v
x = LayerNorm(x)          (final layer norm)
     |
     v
logits = x @ WlmHead      [B, T, vocabSize]
     |
     v
loss = crossEntropy(logits, targets)   (scalar)
```

### 5.2 Parameters

```
GPTParams
  wte:    Variable [vocabSize, nEmbd]         Token embeddings
  wpe:    Variable [blockSize, nEmbd]         Position embeddings
  lmHead: Variable [vocabSize, nEmbd]         Language model head
  lnF:    { weight: [nEmbd], bias: [nEmbd] }  Final layer norm

  layers[i]:  (one per transformer block)
    ln1:    { weight: [nEmbd], bias: [nEmbd] }     Pre-attention layer norm
    attn:
      wq:   Variable [nEmbd, nEmbd]                Query projection
      wk:   Variable [nEmbd, nEmbd]                Key projection
      wv:   Variable [nEmbd, nEmbd]                Value projection
      wo:   Variable [nEmbd, nEmbd]                Output projection
    ln2:    { weight: [nEmbd], bias: [nEmbd] }     Pre-MLP layer norm
    mlp:
      fc1:  Variable [4*nEmbd, nEmbd]              MLP expand (4x)
      fc2:  Variable [nEmbd, 4*nEmbd]              MLP project back
```

### 5.3 Weight Initialization

| Parameter | Distribution | Std Dev |
|-----------|-------------|---------|
| Token/position embeddings | Gaussian | 0.02 |
| Attention Q, K, V projections | Gaussian | 0.02 |
| Attention output projection (Wo) | Gaussian | 0.02 / sqrt(2 * nLayer) |
| MLP fc1 | Gaussian | 0.02 |
| MLP fc2 | Gaussian | 0.02 / sqrt(2 * nLayer) |
| LM Head | Gaussian | 0.02 |
| LayerNorm weights | Ones (1.0) | — |
| LayerNorm biases | Zeros (0.0) | — |

The `1/sqrt(2*nLayer)` scaling on residual-merging layers (Wo, fc2) prevents gradient explosion as residuals accumulate across layers.

### 5.4 Forward Pass Detail

**1. Embedding**

```typescript
tokEmb = embedding(wte, tokens)         // [B, T, nEmbd]
posIndices = [0,1,...,T-1] per batch    // [B, T]
posEmb = embedding(wpe, posIndices)     // [B, T, nEmbd]
x = add(tokEmb, posEmb)                // [B, T, nEmbd]
```

**2. Transformer Block (per layer)**

**Pre-LayerNorm Attention:**
```typescript
ln1Out = layerNorm(x, ln1.weight, ln1.bias, eps=1e-5)

// Project to Q, K, V — reshape for batch matmul
q3d = reshape(ln1Out, [B*T, nEmbd])
q = reshape(matmul(q3d, transpose(wq, 0, 1)), [B, T, nEmbd])
k = reshape(matmul(q3d, transpose(wk, 0, 1)), [B, T, nEmbd])
v = reshape(matmul(q3d, transpose(wv, 0, 1)), [B, T, nEmbd])

// Split into heads: [B, T, nEmbd] -> [B, nHead, T, headDim]
headDim = nEmbd / nHead
qH = transpose(reshape(q, [B, T, nHead, headDim]), 1, 2)
kH = transpose(reshape(k, [B, T, nHead, headDim]), 1, 2)
vH = transpose(reshape(v, [B, T, nHead, headDim]), 1, 2)

// Scaled dot-product attention
scores = scale(matmul(qH, transpose(kH, 2, 3)), 1/sqrt(headDim))
maskedScores = maskedFill(scores, causalMask(T), -Infinity)
attnWeights = softmax(maskedScores, axis=-1)
attnOut = matmul(attnWeights, vH)                // [B, nHead, T, headDim]

// Concatenate heads and project
attnConcat = reshape(transpose(attnOut, 1, 2), [B*T, nEmbd])
projected = reshape(matmul(attnConcat, transpose(wo, 0, 1)), [B, T, nEmbd])

x = add(x, projected)  // Residual connection
```

**Pre-LayerNorm MLP:**
```typescript
ln2Out = layerNorm(x, ln2.weight, ln2.bias, eps=1e-5)
flat = reshape(ln2Out, [B*T, nEmbd])

h = gelu(matmul(flat, transpose(fc1, 0, 1)))     // [B*T, 4*nEmbd]
mlpOut = reshape(matmul(h, transpose(fc2, 0, 1)), [B, T, nEmbd])

x = add(x, mlpOut)  // Residual connection
```

**3. Output Head**
```typescript
x = layerNorm(x, lnF.weight, lnF.bias, eps=1e-5)
logits = reshape(matmul(reshape(x, [B*T, nEmbd]), transpose(lmHead, 0, 1)), [B, T, vocabSize])
```

**4. Loss**
```typescript
loss = crossEntropy(reshape(logits, [B*T, vocabSize]), flatten(targets))  // Scalar
```

### 5.5 Parameter Count

For the default config (V=256, B=256, E=256, L=6, H=8):

| Component | Formula | Count |
|-----------|---------|-------|
| Token embeddings (wte) | V * E | 65,536 |
| Position embeddings (wpe) | B * E | 65,536 |
| Per-layer attention (Q,K,V,O) | 4 * E * E | 262,144 |
| Per-layer LN1 + LN2 | 4 * E | 1,024 |
| Per-layer MLP (fc1 + fc2) | 2 * 4 * E * E | 524,288 |
| Per-layer total | — | 787,456 |
| All layers (x6) | — | 4,724,736 |
| Final LN | 2 * E | 512 |
| LM Head | V * E | 65,536 |
| **Total** | — | **~4.9M** |

### 5.6 Parameter Collection

`collectParams(gptParams)` flattens the hierarchical structure into a `Map<string, Variable>`:

```
"wte", "wpe", "lmHead"
"lnF.weight", "lnF.bias"
"layer.0.ln1.weight", "layer.0.ln1.bias"
"layer.0.attn.wq", "layer.0.attn.wk", "layer.0.attn.wv", "layer.0.attn.wo"
"layer.0.ln2.weight", "layer.0.ln2.bias"
"layer.0.mlp.fc1", "layer.0.mlp.fc2"
... (repeated for all layers)
```

This flat map is used for gradient collection, checkpoint serialization, and optimizer state management.

---

## 6. Tokenizers

**Package:** `@alpha/tokenizers`

Three tokenizer implementations, each targeting different use cases.

### 6.1 CharTokenizer

The simplest approach — one token per unique character.

- **Build:** Extract unique characters from input, sort alphabetically, create bidirectional maps
- **Encode:** Map each character to its ID; unknown characters silently skipped
- **Decode:** Concatenate characters by ID
- **Typical vocab size:** ~95-100 (ASCII printable + whitespace)
- **Best for:** Small, character-level domains (ABC notation, financial data)

### 6.2 BpeTokenizer

Byte-Pair Encoding — iteratively merges the most frequent adjacent token pairs.

**Training algorithm:**

```
1. Initialize vocab with unique characters from input
2. Cap training corpus at 500K characters
3. Represent corpus as array of token IDs
4. Repeat (targetVocabSize - baseChars) times:
   a. Count all adjacent token pairs
   b. Find most frequent pair
   c. Create new token = concatenation of pair
   d. Replace all occurrences in corpus
   e. Update pair counts incrementally
   f. Stop early if no pair occurs >= 2 times
```

**Encoding:** Starts with character-level IDs, then replays every learned merge in training order (deterministic, greedy).

**Pair counting optimization:** Instead of rescanning the entire corpus after each merge, pair counts are updated incrementally — decrement old pairs around merge sites, increment new pairs formed by the merged token.

**Configuration:** Default vocab size 2000; `bpe-4k` variant uses 4000.

### 6.3 WordTokenizer

For discrete, space-separated domains (guitar chords, commands).

- **Build:** Split on whitespace, collect unique words, sort; always includes `\n` as explicit token
- **Encode:** Split lines on whitespace, map to IDs, insert `\n` tokens between lines
- **Decode:** Join with spaces, clean up spacing around newlines
- **Best for:** Chord progressions, structured symbolic data

### 6.4 Persistence

Tokenizer artifacts serialize to JSON:

```json
{
  "type": "bpe",
  "vocabSize": 2000,
  "vocab": ["a", "b", "ab", ...],
  "merges": [[0, 1], [2, 1], ...]
}
```

Artifacts are embedded in checkpoints for self-contained model files.

---

## 7. Training Pipeline

**Package:** `@alpha/train`

### 7.1 Training Loop

The `train()` function orchestrates the complete training process:

```
1. Create run directory with unique runId
2. Load and tokenize training data
   - Files > 200MB: chunked tokenization (50MB chunks)
   - Auto-split train/val at 90/10 ratio (at nearest newline)
3. Optionally resume from checkpoint
4. For each step 0..iters:
   a. Compute learning rate (warmup + cosine decay)
   b. Load random batch from DataLoader
   c. Forward pass -> loss
   d. Backward pass -> gradients
   e. Compute gradient norm (on GPU if available)
   f. NaN guard (skip step if gradNorm is non-finite)
   g. Gradient clipping (scale all grads if norm > threshold)
   h. Optimizer step (AdamW or SGD)
   i. Zero gradients and release GPU buffers
   j. Log metrics (loss, lr, gradNorm, tokens/sec, ms/iter)
   k. Every evalInterval: run validation + save checkpoint
5. Save final checkpoint
6. Generate sample text
```

### 7.2 Learning Rate Schedule

**Linear warmup + cosine decay:**

```
warmup_steps = min(100, total_iters / 10)

if step < warmup:
  lr = base_lr * (step + 1) / warmup_steps      // Linear ramp
else:
  t = (step - warmup) / (total_iters - warmup)
  lr = base_lr * 0.5 * (1 + cos(pi * t))         // Cosine anneal to 0.5x
```

Applied dynamically via `optimizer.setLr(lr)` before each step.

### 7.3 AdamW Optimizer

Full implementation of decoupled weight decay Adam:

```
For each parameter p with gradient g:
  p -= lr * weightDecay * p                          // Weight decay (decoupled)
  m = beta1 * m + (1 - beta1) * g                    // First moment estimate
  v = beta2 * v + (1 - beta2) * g^2                  // Second moment estimate
  m_hat = m / (1 - beta1^step)                       // Bias correction
  v_hat = v / (1 - beta2^step)                       // Bias correction
  p -= lr * m_hat / (sqrt(v_hat) + eps)               // Adaptive update
```

**Two execution paths:**
1. **GPU path:** If `backend.adamwStep` exists, all 4 buffers (params, grads, m, v) are updated in a single fused kernel
2. **CPU path:** Element-wise loops over TypedArrays

**State persistence:** Serializes step counter + all m/v buffers for checkpoint resume.

### 7.4 Gradient Clipping

After computing the gradient norm on GPU:

```typescript
// Compute norm on GPU (avoids full gradient readback)
for each parameter grad:
  g_squared = backend.mul(grad, grad)
  sumVal = backend.sum(g_squared)
  totalNorm += readScalar(sumVal)
gradNorm = sqrt(totalNorm)

// Clip if needed
if gradClip > 0 && gradNorm > gradClip:
  clipCoef = gradClip / gradNorm
  for each grad:
    grad = backend.scale(grad, clipCoef)
```

### 7.5 Data Loading

```typescript
class DataLoader {
  constructor(tokens: Int32Array, rng: Rng, batchSize, blockSize)
  nextBatch(): DataBatch { inputs: Int32Array, targets: Int32Array }
}
```

**Batch generation:** For each batch element, randomly sample a start position in the token sequence, extract `blockSize` contiguous tokens as input, and the next `blockSize` tokens (shifted by 1) as targets.

**Large file handling:**
- `loadAndTokenize(path, tokenizer, range?)` — reads in 50MB chunks, tokenizes each chunk, concatenates results
- `getSplitByte(path, ratio=0.9)` — finds nearest newline to the 90% byte offset for clean train/val splitting
- `loadTextSample(path, maxBytes=100MB)` — reads first N bytes for BPE vocabulary building

### 7.6 Checkpointing

**Binary format (v2):**

```
[4 bytes] "ALPH" magic number
[4 bytes] uint32 LE header JSON byte length
[N bytes] UTF-8 header JSON
[remaining] concatenated raw Float32 tensor data
```

The header JSON contains:
- `modelConfig`, `configHash`, `rngState`, `step`
- `tokenizerArtifacts` (vocab, merges)
- `tensors[]` array with name, shape, element count for each tensor

All parameter and optimizer state tensors are stored as contiguous float32 blocks after the header. This is compact and fast to load.

**Legacy JSON format (v1):** Auto-detected if file starts with `{`. Backward compatible.

**Checkpoint state contains everything needed to resume:**
- All model parameters
- All optimizer moment buffers (m, v)
- RNG state
- Tokenizer artifacts
- Step counter
- Config hash

### 7.7 Sampling During Training

```typescript
function sample(config, params, backend, rng, encode, decode, prompt, sampleConfig):
  1. Encode prompt to tokens
  2. For each generation step:
     a. Forward pass on context window (last blockSize tokens)
     b. Extract last position's logits
     c. Scale by temperature: logits[v] /= temperature
     d. Top-k filter: set all but top-k logits to -Infinity
     e. Softmax to get probabilities
     f. Cumulative sum sampling from distribution
     g. Append sampled token
     h. Clear tape (release GPU buffers)
  3. Decode and return generated text
```

### 7.8 Evaluation

```typescript
function evaluate(config, params, backend, tokenizer, rng, dataPath, batchSize, nBatches):
  1. Load validation text, create DataLoader
  2. For nBatches iterations:
     a. Forward pass on batch
     b. Accumulate loss
  3. Return { loss: avgLoss, perplexity: exp(avgLoss), nBatches }
```

### 7.9 Remote Metrics Reporting

Training runs can stream metrics to the Alpha dashboard in real time:

- **`registerRun()`** — POST to `/api/ingest` with type `"run_start"`, registering the run with config and metadata
- **`onStep(metrics)`** — Buffer step metrics, flush periodically (every 5s or when buffer fills)
- **`uploadCheckpoint(info)`** — Fire-and-forget upload of binary checkpoint (gzipped, with base64 metadata headers)
- **`sendSamples(samples)`** — Send generated text samples for visualization
- **`complete(finalStep)`** — Mark run as finished

All network errors are silently ignored — remote reporting never blocks training.

### 7.10 GPU Memory During Training

The training loop performs explicit deterministic cleanup:

```typescript
// After backward pass:
tape.clear(backend.releaseGpuTensor)    // Free all forward/backward intermediates
backend.flush()                         // Submit pending GPU operations
globalThis.gc?.()                       // Hint V8 to collect garbage
```

Memory stats logged every 10 steps: buffer pool entries, output pool entries, deferred releases.

---

## 8. Helios: GPU Compute via Vulkan

**Package:** `@alpha/helios`

Helios is a complete GPU compute backend built from scratch — a Vulkan C addon, a SPIR-V assembler written in TypeScript, and 30+ compute kernels.

### 8.1 Architecture

```
TypeScript (kernels.ts)           TypeScript (spirv.ts)
  Kernel generators          -->    SPIR-V binary assembler
  (matmul, softmax, etc.)          (types, instructions, layout)
          |                                 |
          v                                 v
      Uint32Array (SPIR-V binary)
          |
          v
   C Addon (helios_vk.c)
     dlopen libvulkan.so
     vkCreateShaderModule
     vkCreateComputePipelines
     Dispatch + timeline sync
          |
          v
   TypeScript (backend.ts)
     HeliosBackend implements Backend
     Compute graph + lazy evaluation
     Buffer pool + memory management
```

### 8.2 SPIR-V Assembler (`spirv.ts`)

Generates SPIR-V 1.3 binary directly from TypeScript — no external shader compilers (no glslc, no spirv-tools).

**`SpirVBuilder` class:**
- `id()` — allocate unique SPIR-V IDs
- `typeVoid()`, `typeFloat()`, `typeInt()`, `typeVector()`, `typeArray()` — declare types
- `typeStruct()`, `typePointer()`, `typeRuntimeArray()` — composite types
- `constant()`, `constantF32()`, `constantComposite()` — declare constants
- `variable()` — declare global variables
- `emit()` — encode function body instructions

**Binary layout:** Header (magic, version, generator, bound) followed by sections in SPIR-V specification order: capabilities, extensions, memory model, entry points, execution modes, decorations, types, variables, functions.

**Capabilities:** `Shader`, `Float16`, `StorageBuffer16BitAccess`

**Extensions:** `GLSL.std.450` for math (Exp, Log, Sqrt, Pow, Tanh, FMax, FMin, Clamp, FMA)

### 8.3 Compute Kernels (`kernels.ts`)

All kernels are functions that return `Uint32Array` (compiled SPIR-V binary), cached by (name, workgroupSize).

#### Element-wise Operations

**Binary:** `add`, `sub`, `mul`, `div` — 3 bindings (A, B, C), one thread per element, bounds-checked.

**Unary:** `neg`, `exp`, `log`, `sqrt`, `relu`, `gelu`, `scale`, `silu` — 2 bindings (A, C).

**Vec4 variants:** `add_vec4`, `mul_vec4`, etc. — process 4 floats per thread via `vec4<f32>`, auto-selected when `size % 4 == 0` for 4x throughput.

**f16 variants:** Load f16, compute in f32, store f16. Requires Vulkan 1.2 float16 features.

#### GELU Kernel (Fused)

```
GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

All constants pre-computed in shader. No intermediate buffers — fully fused.

#### Matrix Multiplication (Tiled)

```
Workgroup: 16x16 threads (256 total)
Shared memory: 2 tiles of 16x16 floats (2KB)
Push constants: M, N, K

Algorithm:
  For each tile of K (step = TILE_SIZE):
    Cooperatively load A[row, tile] and B[tile, col] into shared memory
    ControlBarrier (sync workgroup)
    Each thread: accumulate += sharedA[local_row][k] * sharedB[k][local_col]
    ControlBarrier (sync before next tile load)
  Store accumulated result to C[globalRow, globalCol]
```

**Batched variant:** 3D dispatch — `(ceil(N/16), ceil(M/16), batchSize)`.

#### Reduction (Multi-phase)

```
Phase 1: Each workgroup of WG_SIZE threads reduces WG_SIZE elements to 1
          via parallel tree reduction in shared memory.
          Output: one value per workgroup.

Phase 2+: Repeat until single value remains.
```

**Sum-axis variant:** Reduces along a specific dimension using push constants for `totalOutput`, `axisSize`, `innerSize`.

#### Softmax (Row-wise)

Per-row computation: find max (stability), compute exp, sum, normalize. Uses shared memory for within-workgroup parallel reduction.

#### LayerNorm

Per-row: compute mean, variance (shared memory reduction), normalize, scale, shift. One kernel for full operation.

#### AdamW Step (Fused)

In-place update of all 4 buffers (params, grads, m, v) in one kernel:

```
p -= lr * weightDecay * p
m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * g^2
m_hat = m / bc1
v_hat = v / bc2
p -= lr * m_hat / (sqrt(v_hat) + eps)
```

8 push constant floats: size, lr, beta1, beta2, eps, weightDecay, bc1, bc2.

#### Other Kernels

- `transpose` — stride-based 4D dimension swap
- `broadcast` — `B[i] = A[i % srcSize]`
- `column_sum` — reduce rows to column sums (layernorm backward)
- `gelu_backward`, `relu_backward` — fused activation gradients
- `layernorm_backward` — fused dx/dw/db computation

### 8.4 Native C Addon (`helios_vk.c`)

~2095 lines of C interfacing with Vulkan via `dlopen` (no SDK headers required).

#### Initialization

1. `dlopen("libvulkan.so.1")` — load Vulkan dynamically
2. Load ~60 function pointers via `dlsym`
3. Create Vulkan instance and find GPU (prefer discrete over integrated)
4. Probe capabilities: f16 support, async transfer queue, timestamps
5. Create logical device with compute queue (+ transfer queue if available)
6. Allocate persistent resources:
   - Command pool + 3 command buffers (dispatch, transfer, batch)
   - Descriptor pool (2048 storage buffers, 256 descriptor sets)
   - Timeline semaphore (Vulkan 1.2 async sync)
   - Query pool (GPU timestamps for profiling)

#### Buffer Management

- `createBuffer(bytes, hostVisible?)` — allocate GPU buffer
  - Device-local: individual `vkAllocateMemory` per buffer
  - Host-visible: slab allocation (64MB slabs, bump pointer, ref-counted)
- `uploadBuffer(handle, data)` — CPU to GPU transfer
  - Host-visible: direct memcpy
  - Device-local: stage through persistent staging buffer, submit via transfer queue
- `readBuffer(handle)` — GPU to CPU readback
- `destroyBuffer(handle)` — release memory

#### Pipeline Management

`createPipeline(spirvBytes, numBindings, pushConstantSize)`:
1. `vkCreateShaderModule` from SPIR-V
2. `vkCreateDescriptorSetLayout` (N storage buffer bindings)
3. `vkCreatePipelineLayout` (descriptor set + push constant range)
4. `vkCreateComputePipelines`

#### Single Dispatch (Cached)

```c
dispatch(pipeline, buffers[], gX, gY, gZ, pushConstants?):
  if cache hit (same pipeline + buffers + groups + push data):
    reuse cached command buffer
  else:
    reset descriptor pool
    allocate descriptor set, write descriptors
    reset command buffer, record: bind pipeline -> descriptors -> push -> dispatch
    update cache
  submit async via timeline semaphore
  return timeline value
```

The dispatch cache avoids re-recording command buffers for repeated kernel invocations (common in training where shapes are constant).

#### Batch Dispatch (Multiple ops, single submit)

```c
batchBegin():
  wait for previous GPU work
  reset batch command buffer

batchDispatch(pipeline, buffers, groups, push):  // repeated N times
  allocate descriptor set
  record: bind -> descriptors -> push -> dispatch

batchSubmit():
  end command buffer
  single GPU submission for all N ops
  return timeline value
```

Reduces per-op submit overhead (~2us per op in batch vs ~100us per individual submit).

#### Timeline Semaphore (Vulkan 1.2)

Global monotonic counter. Each GPU submission increments and signals the timeline value. CPU can poll (`getCompleted()`) or wait (`waitTimeline(value)`) without busy-spinning.

### 8.5 Tensor Backend (`backend.ts`)

`HeliosBackend` implements the `Backend` interface with GPU acceleration:

#### GPU/CPU Decision

```
Element-wise ops: GPU if size >= 4096 elements
Matmul: GPU if M*N*K >= 100,000 FLOPs
Small tensors: fall back to CPU (launch overhead > compute)
```

#### Workgroup Size Auto-Tuning

At initialization, benchmarks candidates `[64, 128, 256, 512]` on a 256K-element add kernel, selects the fastest.

#### Compute Graph (Lazy Evaluation)

```typescript
class ComputeGraph {
  pending: PendingOp[]        // Queued ops, not yet submitted
  record(op): void            // Add to pending
  flush(): number             // Submit batch, return timeline value
  deferRelease(region): void  // Schedule buffer release after flush
}
```

Operations are recorded, not immediately dispatched. When 64 ops accumulate, or when `.data` is accessed, the graph flushes: all pending ops are submitted as a single batch.

#### Output Buffer Pool (Timeline-aware)

```typescript
acquireOutputRegion(byteSize):
  Check pool for buffer of matching size
  Verify GPU has completed writing to it (via timeline)
  Return reusable buffer (or allocate new one)

releaseOutputRegion(region, submitValue):
  Return to pool with readiness timeline value
  Cannot be reused until GPU reaches that timeline
```

#### Lazy Tensor Readback

```typescript
graphLazyTensor(vk, shape, outputRegion):
  Returns TensorData with lazy .data getter
  First access: flush graph, wait for GPU, read buffer, cache result
  Subsequent access: return cached CPU data
```

This allows the GPU to overlap compute with CPU work — data isn't copied until it's actually needed.

#### GPU Operation Flow

```
JS: backend.matmul(a, b)
  -> ensureGpu(a), ensureGpu(b)       // Upload to GPU if needed
  -> acquireOutputRegion(resultSize)   // Get output buffer
  -> graph.record(PendingOp)           // Queue dispatch
  -> return graphLazyTensor(result)    // Lazy wrapper

... later ...

JS: result.data                        // Triggers readback
  -> graph.flush()                     // Submit all pending as batch
  -> waitTimeline(submitValue)         // Wait for GPU completion
  -> readBuffer(handle)               // Copy GPU -> CPU
  -> cache and return
```

#### FinalizationRegistry Cleanup

When a `TensorData` object is garbage collected, its GPU buffer is scheduled for deferred release (freed after the next graph flush, ensuring the GPU has finished using it).

### 8.6 Summary of GPU Kernels

| Category | Kernels | Notes |
|----------|---------|-------|
| Binary (scalar) | add, sub, mul, div | Bounds-checked, push constant len |
| Binary (vec4) | add_vec4, sub_vec4, etc. | 4x throughput, auto-selected |
| Unary | neg, exp, log, sqrt, relu, gelu, scale, silu | |
| Backward | gelu_backward, relu_backward, layernorm_backward | Fused gradient kernels |
| Matmul | matmul, matmul_batched | 16x16 tiled, shared memory |
| Reduction | sum_reduce, max_reduce, sum_axis | Multi-phase tree |
| Neural | softmax, layernorm | Row-wise, shared memory |
| Optimizer | adamw_step | Fused in-place 4-buffer update |
| Utility | transpose, broadcast, column_sum | |
| f16 | add_f16, sub_f16, ..., sqrt_f16 | Mixed precision |

---

## 9. Data Generation

**Package:** `@alpha/datagen`

Generates training corpora from multiple public text sources.

### 9.1 Sources

| Source | URL | Size | Content |
|--------|-----|------|---------|
| SCOWL | sourceforge.net/projects/wordlist | ~70K words | English word list |
| SimpleWiki | dumps.wikimedia.org/simplewiki | ~330MB bz2 | Simple English Wikipedia |
| Wiktionary | dumps.wikimedia.org/enwiktionary | ~800MB bz2 | English dictionary definitions |
| Gutenberg | gutenberg.org/cache/epub | ~5000 books | Classic literature |
| Wikipedia | dumps.wikimedia.org/enwiki | ~22GB bz2 | Full English Wikipedia |

### 9.2 Corpus Mode

Linear dump of articles from all sources:

1. Build target vocabulary from SCOWL + wiki titles (words matching `[a-zA-Z]{2,30}`)
2. Process SimpleWiki — stream bz2, strip wikitext, track word coverage
3. Process Wiktionary — add definitions for uncovered words
4. Process Gutenberg — strip boilerplate headers/footers
5. Process Wikipedia — only if uncovered words remain (expensive)
6. Output: plain text file with blank lines as document boundaries

### 9.3 Concordance Mode

Extracts dense context windows around target words:

1. Build target word set, initialize per-word context counter
2. For each article, split into paragraphs
3. For each paragraph containing under-covered words:
   - Extract window: trigger paragraph + 1 neighbor on each side
   - Mark paragraphs as emitted (avoid duplication)
   - Increment context count for all target words in window
4. Process sources in order (SimpleWiki -> Wiktionary -> Gutenberg -> EnWiki)
5. Deterministic shuffle with `SeededRng(42)`
6. Write windows separated by blank lines

### 9.4 Text Cleaning

**Wikitext stripping:** Removes comments, templates (`{{...}}`), tables (`{|...|}`), categories, internal links (`[[...]]`), HTML tags, normalizes whitespace.

**Gutenberg stripping:** Removes Project Gutenberg headers/footers (START/END OF markers), transcriber notes, editorial brackets, normalizes line endings.

---

## 10. Database & Persistence

**Package:** `@alpha/db`

Turso (libsql) database for tracking training runs, metrics, and checkpoints.

### 10.1 Schema

```sql
-- Training runs
CREATE TABLE runs (
  id TEXT PRIMARY KEY,
  run_id TEXT, config_hash TEXT, domain TEXT,
  vocab_size INT, block_size INT, n_layer INT, n_embd INT, n_head INT, dropout REAL,
  total_iters INT, batch_size INT, lr REAL, seed INT,
  backend TEXT, tokenizer TEXT, optimizer TEXT,
  model_config TEXT, train_config TEXT,        -- Full JSON blobs
  status TEXT DEFAULT 'active',               -- active | completed | stale | failed
  latest_step INT DEFAULT 0, last_loss REAL, best_val_loss REAL,
  estimated_params INT,
  created_at TEXT, updated_at TEXT, disk_mtime INT
);

-- Per-step metrics (WITHOUT ROWID for compactness)
CREATE TABLE metrics (
  run_id TEXT, step INT, loss REAL, val_loss REAL,
  lr REAL, grad_norm REAL, elapsed_ms REAL,
  tokens_per_sec REAL, ms_per_iter REAL,
  PRIMARY KEY (run_id, step),
  FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
) WITHOUT ROWID;

-- Saved checkpoints (WITHOUT ROWID)
CREATE TABLE checkpoints (
  run_id TEXT, step INT, filename TEXT, file_path TEXT, file_size INT,
  created_at TEXT,
  PRIMARY KEY (run_id, step),
  FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
) WITHOUT ROWID;

-- Domain configurations (seeded from @alpha/core)
CREATE TABLE domains (id TEXT PRIMARY KEY, ...);
```

### 10.2 Disk Sync

`syncFromDisk(outputsDir)` mirrors the filesystem to the database:

1. Scan `outputs/` for directories containing `config.json`
2. For each run: parse config, read `metrics.jsonl`, find checkpoint files
3. Determine status: `completed` (reached total iters), `active` (modified in last minute), `stale` (otherwise)
4. Upsert run record with all metadata
5. Incremental metrics insert: only insert steps > MAX(step) already in DB
6. Upsert checkpoint records

**Idempotent:** All writes use `INSERT OR IGNORE` / `ON CONFLICT`. Safe to re-run.

### 10.3 Key Operations

- `upsertRun()` — insert or update run with denormalized scalar fields + JSON configs
- `listRuns(status?, domain?)` — query `run_summary` view (joins with metric/checkpoint counts)
- `insertMetrics(runId, metrics[])` — batch insert in chunks of 500
- `getRecentMetrics(runId, count)` — last N metrics for dashboard trends
- `upsertCheckpoint()` — track checkpoint files with step and size

---

## 11. Server & Inference

**Package:** `apps/server`

A Node.js HTTP server providing inference, training metrics ingestion, and an OpenAI-compatible API.

### 11.1 Endpoints

#### Inference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/inference` | GET | SSE stream of generated tokens |
| `/api/chat` | POST | AI SDK streaming chat |
| `/api/generate` | GET/POST | Non-streaming text generation |
| `/v1/chat/completions` | POST | OpenAI-compatible chat (streaming + non-streaming) |
| `/v1/models` | GET | OpenAI-compatible model list |

#### Training

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/ingest` | POST | Bearer | Receive training events (run_start, metrics, samples) |
| `/api/upload` | POST | Bearer | Upload binary/JSON checkpoints |
| `/api/training/live` | GET | — | SSE stream of active training (snapshot + live events) |

#### Dashboard

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/models` | GET | List available models (cached 30s) |
| `/api/runs` | GET | List all DB runs (filterable by status/domain) |
| `/api/runs/{id}/metrics` | GET | Metrics for a run |
| `/api/runs/{id}/checkpoints` | GET | Checkpoints for a run |
| `/api/sync` | POST | Trigger filesystem-to-DB sync |

### 11.2 Inference Engine

**Model loading:**
1. Scan `outputs/` for local run directories with `config.json` + checkpoint files
2. Query Turso for remote training results
3. Merge and deduplicate, cache for 30 seconds

**Model building:**
1. Create CPU tensor backend (inference is CPU-only on the server)
2. Initialize GPT with model config from checkpoint
3. Restore all parameters from checkpoint
4. Load tokenizer from embedded artifacts
5. Cache loaded model (reuse across requests)

**Token generation:**
1. Encode prompt to tokens
2. Extract context window (last `blockSize` tokens)
3. Forward pass to get logits
4. Temperature scaling + top-k filtering + softmax sampling
5. Yield each token for streaming

### 11.3 AI SDK Provider

Implements `LanguageModelV3` interface for Vercel AI SDK integration:

- `doGenerate()` — non-streaming completion with usage stats
- `doStream()` — streaming with `ReadableStream<LanguageModelV3StreamPart>` (start, text-delta, finish parts)

### 11.4 Checkpoint Upload

Supports binary uploads with metadata in headers:
- `X-Checkpoint-Name` — run directory name
- `X-Checkpoint-Step` — training step
- `X-Checkpoint-Config` — base64 config JSON
- `X-Checkpoint-Metrics` — base64 metrics JSONL
- `X-Checkpoint-TrainingData` — base64 training data sample (first 50KB)

Checkpoint is stored to `outputs/{name}/checkpoint-{step}.json`, model cache is invalidated.

### 11.5 Live Training Hub

SSE endpoint (`/api/training/live`) that:
1. Sends initial snapshot of all active runs + recent metrics
2. Broadcasts all subsequent ingest events in real time
3. 30-second heartbeat to prevent timeout
4. Connection pool for multiple dashboard clients

---

## 12. CLI

**Package:** `apps/cli`

### 12.1 Commands

| Command | Description |
|---------|-------------|
| `alpha train` | Full training loop |
| `alpha sample` | Generate text from checkpoint |
| `alpha eval` | Evaluate loss/perplexity on validation data |
| `alpha bench` | Benchmark CPU vs GPU vs WebGPU |
| `alpha datagen` | Generate training corpora |
| `alpha tokenizer build` | Build and save tokenizer artifacts |

### 12.2 Train Command

**Key arguments:**
```
--data=PATH          Training text file (required)
--domain=ID          Domain config (applies defaults)
--backend=NAME       cpu_ref or helios
--iters=N            Training iterations
--batch=N            Batch size
--lr=F               Learning rate
--dim=N              Embedding dimension
--layers=N           Transformer layers
--heads=N            Attention heads
--block=N            Context length
--resume=PATH        Resume from checkpoint
```

**Domain system:** When `--domain=concordance` is specified, the domain's `modelDefaults` and `trainDefaults` override the base config (e.g., block=256, dim=256, layers=6, heads=8 for concordance).

**Remote reporting:** If `ALPHA_REMOTE_URL` and `ALPHA_REMOTE_SECRET` are set, metrics stream to the dashboard server in real time.

### 12.3 Bench Command

Comprehensive benchmarking across backends:

- **ops** — Individual operation benchmarks (matmul, softmax, layernorm, GELU) at sizes 64-512
- **gpu** — Element-wise comparison (CPU vs Helios vs WebGPU) at sizes 1K-4M elements
- **train** — Full training iterations comparing backends (tiny model: 2L, 64D, 4H)
- **e2e** — Forward-pass-like sequence benchmark

WebGPU benchmark runs in a child process (WGSL shaders) to isolate Vulkan contexts.

---

## 13. End-to-End Data Flow

### Training

```
1. DATA GENERATION
   datagen → download Wikipedia/Gutenberg/etc → strip markup → extract contexts
   → concordance.txt (~100MB-1GB of clean English text)

2. TOKENIZATION
   CharTokenizer/BpeTokenizer/WordTokenizer.build(text)
   → vocab + merges → TokenizerArtifacts
   → tokenizer.encode(text) → Int32Array of token IDs

3. DATA LOADING
   DataLoader(tokens, rng, batchSize, blockSize)
   → nextBatch() → { inputs: Int32Array[B, T], targets: Int32Array[B, T] }
   (targets = inputs shifted right by 1)

4. FORWARD PASS
   tokens → embedding → position embedding → add
   → N × (LayerNorm → Q,K,V → attention → residual → LayerNorm → MLP → residual)
   → final LayerNorm → LM head → logits [B, T, vocabSize]
   → crossEntropy(logits, targets) → scalar loss
   (all operations recorded on Tape)

5. BACKWARD PASS
   Tape walks in reverse:
   loss.grad = 1
   crossEntropy backward → softmax backward → matmul backward → ...
   → all parameter Variables have .grad set

6. OPTIMIZER STEP
   AdamW: update moments (m, v), bias-correct, apply adaptive learning rate
   GPU path: fused adamw_step kernel updates params, m, v in-place

7. METRICS & CHECKPOINTING
   Log: step, loss, lr, gradNorm, tokens/sec
   Every evalInterval: validate on held-out data, save checkpoint
   Stream to remote dashboard via RemoteReporter

8. GENERATION
   Prompt → encode → forward pass → temperature scaling → top-k filter
   → softmax → sample → decode → repeat for N tokens
```

### Inference

```
1. Server scans outputs/ for checkpoints + queries Turso DB
2. Client requests /api/inference?model=X&query=Y
3. Engine loads checkpoint: init model → restore params → load tokenizer
4. Generate tokens: encode prompt → forward → sample → yield SSE events
5. Client receives streamed text
```

### GPU Training Pipeline

```
1. HeliosBackend.initDevice() — dlopen Vulkan, create device, probe features
2. Workgroup size auto-tuning (benchmark candidates)
3. Model params uploaded to GPU (ensureGpu for each Variable)
4. Each training step:
   a. Forward ops recorded to ComputeGraph (not yet dispatched)
   b. Graph auto-flushes at 64 pending ops (batch submit)
   c. Backward pass records more ops (gradient kernels)
   d. AdamW step: fused 4-buffer GPU kernel
   e. tape.clear(releaseGpuTensor) — free intermediates
   f. Only loss scalar is read back to CPU (lazy readback)
5. Timeline semaphore ensures GPU ordering without per-buffer tracking
6. Buffer pool recycles completed output regions
```

---

## Appendix: Parameter Estimation Formula

Used by the database sync and dashboard:

```
E = nEmbd, L = nLayer, V = vocabSize, B = blockSize

params = V*E          // token embeddings
       + B*E          // position embeddings
       + L * (
           4*E*E      // attention Q,K,V,O projections
         + 4*E        // attention LN weights+biases
         + 4*4*E*E    // MLP fc1 (4x expand) + fc2 (project back)  [sic: should be 2*4*E*E]
         + 4*E        // MLP LN weights+biases
         + 2*E        // LN1+LN2 (already counted above)
         )
       + 2*E          // final layer norm
       + V*E          // language model head
```
