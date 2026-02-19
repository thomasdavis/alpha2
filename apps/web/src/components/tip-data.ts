/** Centralized tooltip text for ML concepts used across the dashboard. */

export const tips = {
  // ── Model architecture ──────────────────────────────────────────
  params:
    "The total number of learnable values (weights) in the model. More params = more capacity to learn patterns, but also slower and needs more data.",
  vocabSize:
    "How many unique tokens (words or word-pieces) the model knows. A bigger vocab means more words are recognized, but the model gets larger.",
  blockSize:
    "The maximum number of tokens the model can look at in one go (its 'context window'). Longer = can handle more text, but uses more memory.",
  nLayer:
    "The number of transformer layers stacked on top of each other. More layers = the model can learn more complex patterns, but training is slower.",
  nEmbd:
    "The size of the internal representation for each token. Think of it as how much 'detail' the model stores about each word. Bigger = richer understanding.",
  nHead:
    "The number of attention heads. Each head learns to focus on different relationships between words (e.g. grammar, meaning, position). More heads = more perspectives.",
  dropout:
    "During training, randomly 'turns off' this fraction of neurons to prevent the model from memorizing the training data. 0 = no dropout, 0.1 = 10% dropped.",
  architecture:
    "The shape of the neural network — how many layers, how wide each layer is, and how many attention heads it has.",

  // ── Training config ──────────────────────────────────────────────
  totalIters:
    "The total number of training steps. Each step processes one batch of data and updates the model's weights. More steps = more learning (up to a point).",
  batchSize:
    "How many text samples are processed together in each training step. Larger batches give more stable updates but use more memory.",
  lr:
    "Learning rate — how big each weight update is. Too high and the model overshoots; too low and it learns painfully slowly. Usually a tiny number like 0.0003.",
  seed:
    "A random number seed for reproducibility. Using the same seed with the same data and config should produce the same model.",
  backend:
    "The compute engine used for training. 'cpu_ref' means a reference CPU implementation — slower but works anywhere.",
  tokenizer:
    "The method used to split text into tokens. BPE (Byte Pair Encoding) learns common subword units. Char splits by character. Word splits by whitespace.",
  optimizer:
    "The algorithm that decides how to update weights. AdamW is the most popular — it adapts the learning rate per-weight and adds weight decay for regularization.",
  weightDecay:
    "Slightly shrinks all weights each step to prevent them from growing too large. Acts as a form of regularization to reduce overfitting.",
  gradClip:
    "Caps the size of weight updates to prevent 'exploding gradients' — a common problem where updates become unreasonably large and destabilize training.",
  beta1:
    "Controls how much the optimizer remembers past gradient directions. Higher = smoother updates. Default 0.9 means 90% old direction + 10% new.",
  beta2:
    "Controls how much the optimizer remembers past gradient magnitudes for per-weight learning rate scaling. Default 0.999.",
  eps:
    "A tiny number added to prevent division by zero in the optimizer. You almost never need to change this.",
  evalInterval:
    "How often (in steps) the model is tested on held-out validation data to check if it's actually learning or just memorizing.",
  evalIters:
    "How many batches of validation data to use when computing validation loss. More = more accurate estimate but slower.",

  // ── Metrics ──────────────────────────────────────────────────────
  loss:
    "How wrong the model's predictions are. Lower is better. The model tries to minimize this number during training. Think of it as an 'error score'.",
  lastLoss:
    "The loss value at the most recent training step. Shows how well the model is doing right now.",
  bestValLoss:
    "The lowest validation loss seen during training. Validation loss is measured on data the model hasn't trained on, so it shows real generalization ability.",
  valLoss:
    "Loss measured on held-out data the model hasn't seen during training. If this goes up while training loss goes down, the model is overfitting (memorizing).",
  lossChart:
    "Shows how loss changes over training. You want to see it going down! The orange line is training loss, blue dots are validation loss.",
  gradNorm:
    "The magnitude of the weight updates. Very large values mean unstable training; very small values mean the model has stopped learning.",
  tokPerSec:
    "How many tokens the model processes per second. Higher = faster training. Depends on hardware, model size, and batch size.",
  msPerIter:
    "Milliseconds per training step. Lower = faster. Includes the forward pass, loss computation, backward pass, and weight update.",
  throughput:
    "How fast training is progressing — measured in tokens processed per second.",

  // ── Status ───────────────────────────────────────────────────────
  statusActive:
    "This training run is currently in progress. Metrics are being streamed live.",
  statusCompleted:
    "This training run finished all its scheduled iterations successfully.",
  statusStale:
    "This training run stopped sending updates. It may have crashed or been interrupted.",
  statusFailed:
    "This training run encountered an error and stopped.",
  inferenceAvailable:
    "This model has been uploaded to the inference server and can be used for chat and text generation right now.",
  inferenceUnavailable:
    "This model's checkpoint isn't loaded in the inference engine. It was trained but can't generate text yet.",

  // ── Runs / Checkpoints ────────────────────────────────────────────
  checkpoint:
    "A saved snapshot of the model's weights at a specific training step. Like a 'save game' — you can resume training or use it for inference.",
  checkpointCount:
    "How many snapshots of the model were saved during training. More checkpoints = more options to pick the best version.",
  trainingData:
    "The raw text data the model learned from. The model tries to predict the next token in this text, and that's how it learns language patterns.",
  step:
    "A single training iteration. Each step: (1) grab a batch of text, (2) have the model predict next tokens, (3) measure error, (4) update weights.",
  progress:
    "How far through training this run is. 100% means all planned iterations are complete.",
  metricCount:
    "How many individual metric measurements have been recorded for this run.",

  // ── Domains ──────────────────────────────────────────────────────
  domain:
    "The type of text this model was trained on. Different domains (novels, music, finance) produce models with different 'personalities'.",
  domainNovels: "Trained on novel/fiction text. Generates creative prose and stories.",
  domainChords: "Trained on chord progressions. Generates musical chord sequences.",
  domainAbc: "Trained on ABC music notation. Generates melodies and tunes in ABC format.",
  domainFinance: "Trained on financial data patterns.",

  // ── General ──────────────────────────────────────────────────────
  model:
    "A neural network that has been trained to predict the next token in a sequence. Given some text, it generates plausible continuations.",
  eta:
    "Estimated time remaining until training completes, based on recent iteration speed and remaining steps.",
  elapsed:
    "Total wall-clock time spent training so far.",
  rawConfig:
    "The exact configuration values used to create this model and training run, stored as JSON.",
} as const;
